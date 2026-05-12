# LoRA Training for HunyuanVideo-Foley — Design

**Date:** 2026-04-10
**Status:** Approved
**Reference:** SelVA LoRA implementation (`feature/lora-timestep-sampling` branch)

## Goal

Add LoRA fine-tuning to HunyuanVideo-Foley for video-to-audio generation. Port the proven SelVA LoRA training pipeline, adapted for Foley's two-stream architecture and DAC neural codec.

## Architecture Context

### Foley Model Structure
- **54 transformer blocks**: 18 TwoStreamCABlock + 36 SingleStreamBlock
- **Hidden dim**: 1536, 12 heads, 128D per head
- **Conditioning**: SigLIP2 (visual) + Synchformer (sync) + CLAP (text)
- **VAE**: DAC neural codec, 128D latent, 48kHz stereo output
- **Diffusion**: Flow matching with velocity prediction, MSE loss

### Key Differences from SelVA
| Aspect | SelVA | Foley |
|--------|-------|-------|
| Model size | ~2 GB | ~10.3 GB |
| VAE | Mel-spectrogram + BigVGAN vocoder | DAC single-stage codec |
| Latent dim | 40D | 128D |
| Block types | Single type (Joint+Fused) | Two types (TwoStreamCA + SingleStream) |
| Feature extractors | CLIP + TextSynchformer + T5 | SigLIP2 + Synchformer + CLAP |
| Audio output | 44.1kHz mono | 48kHz stereo |
| Diffusion | Flow matching (velocity) | Flow matching (velocity) — same |

## Nodes (6 total)

### 1. Foley Feature Extractor

Caches visual + text features for training. One clip per execution, auto-incremented naming.
Audio is **not** processed here — it is handled by the Dataset Saver after audio cleaning.

**Inputs:**
- `HUNYUAN_DEPS` — provides SigLIP2, Synchformer, CLAP models
- `IMAGE` — video frames
- `prompt` — text description
- `frame_rate` — source video FPS (used only if duration=0)
- `duration` — clip duration in seconds (default **8.0s** — Foley generates 8s audio)
- `cache_dir` — output directory for .npz files
- `name` — base filename for auto-increment (e.g., "gunshot" -> gunshot_001.npz)

> **Important:** Always set `duration=8.0` explicitly. The auto-detect (`total_frames / frame_rate`)
> gives wrong results when the video fps doesn't match the `frame_rate` input (e.g., 30fps video
> with frame_rate=25 computes 9.6s instead of 8.0s). This causes misaligned visual features
> that break audio-video sync during training.

**Process:**
1. Extract SigLIP2 features at 8fps, 512x512 -> `[1, 64, 768]` (for 8s)
2. Extract Synchformer features at 25fps, 224x224 -> `[1, 192, 768]` (for 8s)
3. Encode text via CLAP -> `[1, N_text, 768]`
4. Save all to .npz with metadata (prompt, duration, fps)

**Caching:** SHA256 hash-based dedup to skip reprocessing identical inputs.

### 2. Foley LoRA Trainer

Core training node. Blocks the queue during training.

**Inputs:**
- `HUNYUAN_MODEL` — frozen base model
- `HUNYUAN_DEPS` — for DAC encoding and feature reference
- `data_dir` — directory with .npz + audio pairs
- Training hyperparameters (see below)

**Hyperparameters:**

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| target | `all_attn_mlp` | preset selection | See target presets |
| rank | 64 | 8-128 | LoRA rank |
| alpha | 64 | 1-128 | Scaling factor |
| lr | 1e-4 | 5e-5 to 5e-4 | Learning rate |
| steps | 3000 | 500-20000 | Total training iterations |
| batch_size | 8 | 1-32 | Clips per step |
| grad_accum | 1 | 1-16 | Gradient accumulation |
| warmup_steps | 100 | 0-500 | LR warmup |
| save_every | 500 | 100-5000 | Checkpoint interval |
| timestep_mode | logit_normal | uniform/logit_normal/curriculum | Timestep sampling |
| logit_normal_sigma | 1.0 | 0.5-2.0 | Logit-normal spread |
| curriculum_switch | 0.6 | 0.1-0.9 | Curriculum transition point |
| init_mode | standard | standard/pissa | LoRA initialization |
| use_rslora | false | bool | Rank-stabilized scaling |
| lora_dropout | 0.0 | 0.0-0.3 | LoRA path dropout |
| lora_plus_ratio | 1.0 | 1.0/16.0 | B-matrix LR multiplier |
| schedule_type | constant | constant/cosine | LR schedule after warmup |
| latent_mixup_alpha | 0.0 | 0.0-1.0 | Latent mixup strength |
| latent_noise_sigma | 0.0 | 0.0-0.1 | Additive noise on latents |
| precision | bf16 | bf16/fp16/fp32 | Training precision |
| seed | 42 | int | Random seed |
| output_dir | string | path | Checkpoint save location |
| gradient_checkpointing | false | bool | Recompute activations to save VRAM (~3-5 GB, ~25% slower) |
| blocks_to_swap | 0 | 0-54 | Offload N transformer blocks to CPU (prefetch=2) |
| resume_from | optional | path | Resume from checkpoint |

**Target Presets:**

| Preset | Layers per TwoStreamCABlock | Total Linears |
|--------|----------------------------|---------------|
| `audio_attn` | audio_self_attn_qkv, audio_self_proj | 36 |
| `audio_cross` | above + audio_cross_q, audio_cross_proj, text_cross_kv | 90 |
| `all_attn` | above + v_cond_attn_qkv, v_cond_self_proj, v_cond_cross_q, v_cond_cross_proj | 162 |
| `all_attn_mlp` | above + audio_mlp.fc1/fc2, v_cond_mlp.fc1/fc2 | 234 |

SingleStreamBlock Conv1d layers excluded initially (Linear-only targeting).

**Training Loop (Flow Matching):**

**CRITICAL — Sigma convention must match the scheduler:**

The `FlowMatchDiscreteScheduler` uses `x(sigma) = sigma * noise + (1-sigma) * data`,
with sigma going from 1 (noise) to 0 (data) during generation. The training must match:

```
1. Load cached .npz features + DAC-encoded latents
2. Sample timestep t ~ configured distribution (t in [0, 1])
3. Sample noise x0 ~ N(0, I), let x1 = target data
4. Interpolate: xt = t*x0 + (1-t)*x1     (t=1 → noise, t=0 → data)
5. Forward: v_pred = foley_model(xt, t*1000, clip_feat, sync_feat, text_feat)
6. Loss: MSE(v_pred, x0 - x1)            (velocity = noise - data)
7. Backward through LoRA params only
8. AdamW step (beta1=0.9, beta2=0.95) + gradient clipping (max_norm=1.0)
```

> **Note:** The raw MSE loss appears flat (~1.3) throughout training. This is normal
> for flow matching — the loss is dominated by the irreducible stochastic variance
> of the velocity target. The actual learning signal is a tiny fraction of the total.
> Use eval spectrograms and spectral metrics (LSD, MCD, per-band correlation)
> to track training progress, not the raw loss value.

**Eval Sample Generation (CFG required):**

The base model was trained with classifier-free guidance dropout and **requires CFG
at inference** to produce coherent audio. Eval samples use the same approach as the
main inference pipeline:

1. Create unconditional embeddings via `model.get_empty_clip_sequence()` / `get_empty_sync_sequence()` + zero text
2. Double the batch: `torch.cat([uncond, cond])` for all features
3. Run model once, split output: `v_uncond, v_cond = v_pred.chunk(2)`
4. Apply guidance: `v = v_uncond + cfg_scale * (v_cond - v_uncond)` (default cfg_scale=5.0)

Without CFG, the model produces pure noise regardless of training quality.

**Outputs:**
- Checkpoints: `adapter_step00500.pt`, `adapter_final.pt`
- Metadata: `meta.json`
- Loss curve: `loss.png`
- Eval samples + spectrograms: `samples/step_00000.wav` (pre-training baseline), `samples/step_00500.wav`, etc.
- Validation samples (if `eval_npz` set): `samples/val_step_00000.wav`, `samples/val_step_00500.wav`, etc.
- Spectral metrics: `metrics_history.json` (saved incrementally at each checkpoint)
- Returns: `HUNYUAN_MODEL` with LoRA applied

### 3. Foley LoRA Loader

Applies trained adapter for inference.

**Inputs:**
- `HUNYUAN_MODEL` — base model
- `adapter_path` — path to .pt checkpoint
- `strength` — 0.0 to 2.0 (default 1.0)

**Process:**
1. Load checkpoint, extract metadata (rank, alpha, target, etc.)
2. Deep-copy the model (original unaffected)
3. Inject LoRA layers via `apply_lora()`
4. Load weights, scale lora_B by strength
5. Return patched model

**Output:** `HUNYUAN_MODEL` with LoRA active

### 4. Foley LoRA Scheduler

Multi-experiment sweep orchestrator.

**Inputs:**
- `HUNYUAN_MODEL` + `HUNYUAN_DEPS`
- `sweep_json` — path to experiment sweep configuration

**Sweep JSON Format:**
```json
{
  "name": "rank_sweep",
  "data_dir": "dataset/gunshots",
  "output_root": "lora_output/rank_sweep",
  "eval_npz": "/path/to/validation_clip.npz",
  "base": { "rank": 64, "lr": 1e-4, "steps": 3000, "target": "all_attn_mlp" },
  "experiments": [
    {"id": "rank32", "rank": 32},
    {"id": "rank64"},
    {"id": "rank128", "rank": 128},
    {"id": "loraplus", "lora_plus_ratio": 16.0}
  ]
}
```

**Validation Sample (`eval_npz`):**

Optional path to an NPZ file **outside** the training dataset, with a matching audio file
alongside it (same stem, e.g., `clip_016.npz` + `clip_016.flac`). When set:
- Generates `val_step_00000.wav/png` at step 0 (pre-training baseline)
- Generates `val_step_XXXXX.wav/png` at every checkpoint
- Saves `val_reference.png` spectrogram of the ground-truth audio
- Detects overfitting: training eval improves while val eval plateaus or degrades

The validation clip can be any duration — the model handles variable lengths natively.
Using a rejected clip from the same dataset is ideal: same domain, never trained on.

**VRAM Offload Options (per-experiment):**

| Option | JSON key | Default | Effect |
|--------|----------|---------|--------|
| Gradient checkpointing | `gradient_checkpointing` | false | Saves ~3-5 GB VRAM, ~25% slower. Recomputes activations during backward. |
| Block swap | `blocks_to_swap` | 0 | Offloads N of 54 blocks to CPU. Uses prefetch=2 with async transfers. |

These can be set in `base` (applies to all experiments) or per-experiment.

**Features:**
- Loads dataset once, reuses across experiments
- Deep-copies generator per experiment
- Resume: skips completed experiments via `experiment_summary.json`
- Abort current: `skip_current.flag` file
- Records system info (torch, CUDA, GPU, VRAM)

**Output:**
- `experiment_summary.json` — per-experiment config, loss metrics, adapter paths
- `loss_comparison.png` — overlaid smoothed loss curves

### 5. Foley LoRA Evaluator

Compares multiple adapters on the same dataset.

**Inputs:**
- `HUNYUAN_MODEL` + `HUNYUAN_DEPS`
- `eval_json` — evaluation specification

**Eval JSON Format:**
```json
{
  "name": "eval_batch_1",
  "data_dir": "/path/to/features",
  "output_dir": "/path/to/evals",
  "steps": 25,
  "seed": 42,
  "adapters": [
    {"id": "baseline"},
    {"id": "rank64", "path": "/path/to/adapter_final.pt"}
  ]
}
```

**Spectral Metrics:**
- HF energy ratio (>4kHz / total)
- Spectral centroid (Hz)
- Spectral rolloff (85% energy, Hz)
- Spectral flatness (0=tone, 1=noise)
- Temporal variance (dynamic range)
- Log spectral distance vs reference (dB)
- Mel cepstral distortion vs reference

**Output:**
- `eval_summary.json` — per-adapter per-clip metrics
- `metric_comparison.png` — 2x2 bar chart
- Per-adapter WAV files for manual listening

### 6. Foley VAE Roundtrip

Diagnostic: encode audio through DAC, decode back. Reveals codec quality ceiling.

**Inputs:**
- `HUNYUAN_DEPS` — provides DAC model
- `AUDIO` — input audio

**Process:**
1. Resample to 48kHz if needed
2. DAC encode -> 128D latents
3. DAC decode -> reconstructed audio
4. Normalize output level

**Output:** `AUDIO` (reconstructed)

## File Structure

```
ComfyUI-HunyuanVideo-Foley/
  lora/
    __init__.py
    lora.py                  # LoRALinear class, apply_lora(), load_lora()
    train.py                 # Training loop, dataset loading, loss computation
    spectral_metrics.py      # Spectral analysis utilities
  nodes_lora.py              # All 6 node class definitions
  __init__.py                # Updated to import nodes_lora
```

## What Ports Directly from SelVA

| Component | Reuse Level | Adaptation Needed |
|-----------|-------------|-------------------|
| LoRALinear class | Direct copy | None |
| apply_lora() / load_lora() | Direct copy | Change target suffix names |
| Flow matching loss | Direct copy | None (same MSE velocity matching) |
| Timestep sampling (3 modes) | Direct copy | None |
| Checkpoint format | Direct copy | Update meta fields |
| Scheduler orchestration | Direct copy | Swap model types |
| Evaluator spectral metrics | Direct copy | None |
| Loss curve visualization | Direct copy | None |
| Mixup augmentation | Direct copy | Works on any latent space |
| Feature extraction | Rewrite | Different extractors (SigLIP2/Synchformer/CLAP vs CLIP/TextSynchformer/T5) |
| Audio encoding | Rewrite | DAC encode vs mel-spectrogram VAE |
| Dataset loading | Adapt | Different .npz contents |

## VRAM Estimates

**Model breakdown (bf16):**

| Component | Size | Notes |
|-----------|------|-------|
| Base model weights (frozen) | 4.3 GB | 2.3B params × 2 bytes |
| Activations (backprop) | 5-10 GB | Biggest variable, depends on batch/seq |
| LoRA params + gradients + optimizer | ~0.5-1 GB | Small — only LoRA weights are trained |
| Batch data | ~0.5-1 GB | Latents + features |
| **Total (no offload)** | **~18-20 GB** | batch_size=8, rank=128 |

**Offload configurations:**

| Config | VRAM | Speed | Target GPU |
|--------|------|-------|------------|
| No offload, batch 8 | ~18-20 GB | Fastest | 24+ GB (4090, A5000) |
| Gradient checkpointing, batch 8 | ~13-15 GB | ~25% slower | 16 GB (4080, A4000) |
| Grad ckpt + 20 blocks swapped | ~10-12 GB | ~40% slower | 12 GB (3060 12GB) |
| Grad ckpt + 40 blocks swapped, batch 2 | ~8-9 GB | ~60% slower | 10 GB |

High-VRAM systems (48+ GB) need no offloading at all.

## Defaults Summary

Optimized for maximum quality on high-VRAM hardware:

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| target | all_attn_mlp | Maximum adaptation capacity |
| rank | 64 | High capacity for 128D latent space |
| alpha | 64 | 1:1 ratio (HunyuanVideo convention) |
| batch_size | 8 | Good gradient quality without excess |
| precision | bf16 | Full quality, Ampere+ standard |
| timestep_mode | logit_normal | 0.2-0.3 dB lower loss floor |
| lr | 1e-4 | Proven across AudioLDM and SelVA |
| grad_accum | 1 | No need with 96 GB |

## Technical Notes (from first sweep, 2026-04-11)

### DAC Codec Details
- **Encoder rates:** `[2, 4, 8, 8]` → **hop length = 512**
- 8s @ 48kHz = 384,000 samples → 750 latent frames
- Continuous mode: `encode()` returns `DiagonalGaussianDistribution`, call `.sample()`

### ComfyUI Integration
- ComfyUI wraps node execution in `torch.inference_mode()` — must exit with
  `with torch.inference_mode(False), torch.enable_grad():` for training
- Live loss curve preview: `comfy.utils.ProgressBar` + `pbar.update_absolute(step, total, ("JPEG", pil_image, 800))`
- No `torchaudio` — use `soundfile` for I/O and `soxr` for resampling (avoids torchcodec/FFmpeg dependency)

### Training Quality Indicators
- **Raw MSE loss is NOT informative** — appears flat at ~1.3 due to flow matching noise floor
- **Track these instead:**
  - Eval spectrograms (visual comparison to reference)
  - Spectral convergence (normalized Frobenius distance, lower = better)
  - Per-band correlation (higher = better, negative = bad)
  - Mel cepstral distortion (lower = better)
  - CLAP similarity (cosine sim between generated audio and text prompt)

### First Sweep Results (baseline_r64, 49 clips × 2 augmentations = 99 training pairs)

| Metric | Step 500 | Step 1000 | Trend |
|--------|----------|-----------|-------|
| Loss (MSE) | 1.329 | 1.305 | flat (expected) |
| Spectral convergence | 3.21 | 2.15 | -33% |
| MCD | 33.8 | 33.4 | down |
| Per-band correlation | -0.12 | +0.01 | improving |
| HF energy ratio | 0.056 | 0.015 | normalizing |
| Temporal variance | 2.93 | 1.97 | tightening |

Step 0 (base model) eval sounds like generic audio. By step 500, temporal structure
matches reference. By step 1000, spectral characteristics are converging.

### Third Sweep Results (rank128, standard LR, 10k steps, 99 training pairs)

**Experiments:** r128_10k (constant LR), r128_cosine_10k (cosine decay), r128_lr5e5
(half LR), r128_curriculum (curriculum timestep sampling), r128_dropout, r256_10k.

**Key comparison — constant vs cosine vs curriculum at 10k steps:**

| Metric | r128_10k (constant) | r128_cosine_10k | r128_curriculum |
|--------|--------------------|-----------------|-----------------| 
| SC | 1.05 | 1.13 | **0.99** |
| MCD | 18.1 | 18.5 | **17.9** |
| PBC | 0.70 | 0.64 | **0.71** |
| LSD | **37.2 dB** | 37.7 dB | **36.6 dB** |
| Loss | **1.087** | 1.215 | 1.083 |

**Progression — constant LR peaks at 7k then regresses, curriculum keeps improving:**

| Step | SC (constant) | SC (curriculum) | MCD (constant) | MCD (curriculum) |
|------|--------------|----------------|---------------|-----------------|
| 4k | 1.09 | 1.14 | 18.3 | 18.8 |
| 7k | **1.04** | 1.10 | **17.5** | 17.6 |
| 9k | 1.01 | 1.03 | 18.0 | 17.9 |
| 10k | 1.05 ↑ | **0.99** ↓ | 18.1 ↑ | **17.9** ↓ |

**Findings:**

1. **Curriculum is the best config so far** — first to break SC < 1.0, still improving at 10k.
   The late-stage switch to uniform timesteps prevents the mild overfitting that hits constant LR.
2. **Constant LR peaks at ~7k steps** — best SC 1.04, MCD 17.5, PBC 0.72 at step 7k,
   then metrics regress slightly by 10k. Suggests 7-8k is the sweet spot for constant LR.
3. **Cosine decay hurts** — LR decays too fast, model stops learning by step 6k. Wasted
   the last 4k steps at near-zero LR.
4. **Half LR (5e-5) too slow** — at 8k steps it's where 1e-4 was at 4k. Same destination,
   double the compute. No advantage.
5. **LoRA+ (sweep 2) overfits on this dataset** — loss drops below noise floor (0.5),
   sounds mechanical on some inputs, pure noise on others. Standard LR generalizes better.

### Fourth Sweep Results (no augmentation, 47 unique clips, curriculum r128, 15k steps)

Removing augmented duplicates massively improved training metrics:

| Metric | Augmented (99 clips) @ 10k | No-aug (47 clips) @ 10k | No-aug @ 15k |
|--------|---------------------------|------------------------|-------------|
| SC | 0.99 | **0.47** | **0.35** |
| MCD | 17.9 | **6.5** | **6.5** |
| PBC | 0.71 | **0.81** | **0.83** |
| LSD | 36.6 dB | **18.4 dB** | **18.7 dB** |

Still improving at 15k with no sign of plateau. Augmented copies (same video,
slightly different audio) caused the model to learn averaged spectral patterns,
producing mechanical-sounding output. Unique clips train cleaner.

**Key insight:** Data augmentation on small datasets can hurt more than help.
Duplicated videos with varied audio teach "this visual = average of these sounds"
rather than "follow the video cues." More unique clips > more augmentations.

### Fifth Sweep Results — 399 clips, extended to 40k

**Config:** r128, curriculum (switch 0.6), lr=1e-4, batch=8, 399 unique clips from 46 source videos.
Trained to 15k, extended to 25k, then 40k. Portrait + landscape mix (~20-30% portrait).

**Scalars over training:**

| Step | Loss | SC | MCD | LSD | PBC |
|------|------|----|-----|-----|-----|
| 5k | 1.511 | 3.218 | 7.64 | 18.7 | 0.34 |
| 9k | 1.500 | 3.054 | 6.86 | 17.3 | 0.46 |
| 10k | 1.428 | 2.990 | 6.28 | 16.5 | 0.47 |
| 13k | 1.426 | 2.877 | 6.32 | 16.3 | 0.58 |
| 14k | 1.427 | 2.836 | 7.18 | 17.7 | 0.63 |
| 15k | 1.421 | 2.865 | 6.11 | 15.5 | 0.63 |
| 20k | 1.417 | 2.774 | 5.15 | 14.3 | 0.74 |
| 25k | 1.411 | 2.683 | 5.20 | 13.9 | 0.78 |

**Scalars vs perceptual quality diverge after ~14k.** All metrics continued improving
through 25k (SC 2.87→2.68, PBC 0.63→0.78), but perceptual testing on unseen clips
revealed that step 13-14k produced the best-sounding output. Later checkpoints
lost subtle ambient details — faint breath, room tone, quiet wet textures — that
the metrics don't capture because they weight all frequencies equally.

The model over-specializes on dominant spectral features at the cost of low-energy
ambient sounds. This creates technically better metric scores but perceptually
less natural audio.

**Best checkpoint: step 13-14k** — best balance of learned fidelity and preserved
subtlety. Recommended recipe for similar content:

| Parameter | Value |
|-----------|-------|
| Rank | 128 |
| Timestep mode | curriculum (switch 0.6) |
| Learning rate | 1e-4 constant |
| Dataset size | ~400 unique clips |
| Steps | 13-14k |
| Batch size | 8 |

**Key insight:** Scalar metrics (SC, MCD, PBC) are necessary but not sufficient
for checkpoint selection. Always validate top candidates perceptually on unseen clips.
Metrics track dominant spectral fidelity; subtle ambient details require listening.

### Sixth Sweep Results — 836 clips, doggy POV dataset

**Config:** r128, curriculum (switch 0.6), lr=1e-4, batch=8, 836 unique clips from 43 source videos, 15k steps.

**Comparison with 399-clip run at step 1k:**

| Metric | 399 clips @ 1k | 836 clips @ 1k |
|--------|---------------|---------------|
| Loss | 1.521 | **1.447** |
| SC | 3.369 | **1.531** |
| MCD | 9.986 | **9.823** |
| Temp Var | 0.775 | **1.178** |

SC was 2x better at step 1k — the larger dataset accelerates learning dramatically.

**Scalars over training:**

| Step | Loss | SC | MCD | PBC | Temp Var |
|------|------|----|-----|-----|----------|
| 1k | 1.447 | 1.531 | 9.82 | -0.05 | 1.18 |
| 5k | — | — | — | — | — |
| 8k | 1.428 | 1.544 | 10.48 | -0.13 | 1.94 |
| 10k | 1.365 | 1.531 | 9.86 | -0.10 | 1.62 |
| 11k | 1.369 | 1.507 | 9.84 | -0.07 | 1.51 |
| **12k** | **1.379** | **1.496** | **9.57** | **-0.04** | **1.73** |
| 13k | 1.370 | 1.523 | 9.68 | -0.00 | 1.29 |

**Overfitting onset at step 13k:**
- SC regressed (1.496 → 1.523) — first metric degradation
- Temporal variance collapsed to 1.29 — audio becoming flat/uniform
- Perceptual quality degraded: impacts lost natural variation, sound "weird"
- Loss at 1.37 — approaching the noise floor

**Best checkpoint: step 11-12k.** SC bottomed at 12k, temporal variance still
healthy, and audio retained natural dynamics.

**Pattern confirmed across datasets:** The perceptual peak arrives 2-3k steps
after the curriculum transition (switch at 60% = step 9k), consistently around
steps 11-14k for 15k runs regardless of dataset size. Temporal variance is the
best early warning — when it starts dropping, the model is over-regularizing.

**Resume experiment — training from 13k to 20k:**
Resumed from step 13k (where overfitting was first detected) and trained to 20k.
Scalars recovered and improved beyond the original 12k best:

| Step | SC | MCD | PBC | TV |
|------|----|-----|-----|----|
| 12k (original best) | 1.496 | 9.57 | -0.04 | 1.73 |
| 13k (overfit start) | 1.523 | 9.68 | -0.00 | 1.29 |
| 20k (resumed) | 1.421 | 9.59 | 0.29 | 2.29 |

SC hit a new best (1.421), PBC went positive (0.29), temporal variance recovered (2.29).
But perceptually the audio was "a lot more metallic" — same over-specialization, just
reached via a different path.

**Key insight:** Resuming from an earlier checkpoint does not escape the overfitting
trajectory. The model converges to the same over-specialized state regardless of path.
More training steps always push toward dominant spectral features at the expense of
subtle ambient textures. The perceptual sweet spot is fixed at 2-3k steps post-curriculum
transition — extending training beyond that degrades naturalness even when metrics improve.

### Text Prompt Guidelines (CLAP Conditioning)

**CLAP model:** `laion/larger_clap_general` — trained on AudioSet + AudioCaps captions.

**Prompt style:** AudioCaps format — describe the **sound**, not the visual scene.
CLAP encodes audio semantics; visual context comes from SigLIP2/Synchformer features.

**Rules:**
1. **Describe sound characteristics, not visuals** — "wet sucking and slurping" not "woman performing oral"
2. **Use action + texture** — "heavy boots on a wooden floor", "water dripping into a metal bucket"
3. **Add acoustic modifiers** — "rhythmic", "loud", "close", "deep bass", "high-pitched"
4. **Keep prompts consistent** across clips of the same sound type
5. **Avoid negations** — don't use "no background noise", use positive descriptions instead
6. **Be specific** — "a large dog barking loudly" >> "dog"
7. **Stay concise** — 77 token limit in model, shorter is better for CLAP

**Prompt flow:**
```
Text → CLAP tokenizer (max 77 tokens)
     → CLAP encoder → [B, 77, 768] (per-token embeddings, NOT pooled)
     → ConditionProjection (768 → 1536)
     → Cross-attention in 18 TwoStreamCABlocks (conditions both audio + visual streams)
     → CFG-scaled at inference (default cfg_scale=4.5)
```

**Negative prompt:** Default is `"noisy, harsh"`. Can be customized at inference to
de-emphasize specific sounds (e.g., `"breathing, heavy breathing"` to reduce breathing).

**The prompt affects inference more than training.** During training, the model learns
audio patterns from all three conditioning streams (visual, sync, text). At inference,
CFG amplifies the text guidance, so the prompt steers generation. A sound present in
training audio will still be generated even if not mentioned in the prompt — the visual
and sync features carry it.

### Checkpoint Selection — Scalar Analysis Guide

Use this procedure to analyze `metrics_history.json` and select the best checkpoint.

**File:** `<experiment_dir>/metrics_history.json` — array of objects, one per eval step.

**Available metrics (per step):**

| Metric | Key | What it measures | Better |
|--------|-----|-----------------|--------|
| **Loss** | `loss` | Training MSE, flow matching objective | Lower (but not below noise floor) |
| **HF Energy Ratio** | `hf_energy_ratio` | Energy above 4kHz vs total | Closer to reference |
| **Spectral Centroid** | `spectral_centroid_hz` | Frequency "center of mass" | Closer to reference |
| **Spectral Rolloff** | `spectral_rolloff_hz` | Frequency below which 85% of energy lives | Closer to reference |
| **Spectral Flatness** | `spectral_flatness` | How noise-like vs tonal (Wiener entropy) | Closer to reference |
| **Temporal Variance** | `temporal_variance` | Dynamic range — RMS variation over time | Closer to reference |
| **Log Spectral Distance** | `log_spectral_distance_db` | dB-scale spectral envelope error vs ref | Lower |
| **Spectral Convergence** | `spectral_convergence` | Normalized Frobenius distance vs ref | Lower |
| **Mel Cepstral Distortion** | `mel_cepstral_distortion` | Perceptual distance in mel-cepstral space | Lower |
| **Per-Band Correlation** | `per_band_correlation` | Avg correlation across 80 mel bands vs ref | Higher (max 1.0) |

**Analysis prompt (copy-paste for Claude):**

```
Read <experiment_dir>/metrics_history.json and determine the best checkpoint.

Steps:
1. Print a table of all steps with: loss, LSD, SC, MCD, per_band_correlation, temporal_variance
2. Identify the noise floor — loss typically plateaus around 1.3-1.5 for this model.
   Loss dropping significantly below the noise floor indicates overfitting.
3. Find the best checkpoint using this priority:
   a. PRIMARY: lowest spectral_convergence (SC) — overall spectral fidelity
   b. SECONDARY: lowest mel_cepstral_distortion (MCD) — perceptual quality
   c. TIE-BREAKER: highest per_band_correlation (PBC) — temporal tracking accuracy
4. Check for overfitting signs at that checkpoint:
   - Loss dropped well below noise floor (< 1.3)
   - SC/MCD improving but val spectrograms show horizontal banding or metallic artifacts
   - per_band_correlation near 1.0 on training eval (memorization)
   If overfitting is detected, select the last checkpoint BEFORE the overfitting inflection.
5. Report:
   - Best checkpoint step number
   - Key metrics at that step
   - Whether training should continue, stop, or resume from this checkpoint
   - If val metrics exist, compare train vs val generalization gap

Also plot the trend: is the model still improving, plateaued, or degrading?
Curriculum transition happens at 60% of total steps — expect a quality jump around that point
as the model transitions from easy timesteps to uniform sampling.
```

**Overfitting indicators (from sweep experiments):**
- Loss below ~1.3: model is fitting noise, not learning generalizable patterns
- SC improving on train but val spectrograms degrade: memorization
- Horizontal spectral banding in val samples: averaged spectral patterns from small dataset
- per_band_correlation > 0.9 on train: too close to reference, won't generalize
- temporal_variance collapsing: model producing flat/static audio

**Healthy training indicators:**
- Loss stable around 1.4-1.5 (noise floor for this model + dataset)
- SC, MCD, LSD all trending down together
- per_band_correlation trending up but staying below 0.8
- Val and train metrics moving in the same direction (small gap OK)
- temporal_variance close to reference value (dynamic, not flat)

---

## Training Experiments — Blowjob LoRA (April 2026)

### Dataset Iterations

| Version | Clips | Spread | Scoring | Prompts | Notes |
|---------|-------|--------|---------|---------|-------|
| v1 | ~320 | 2-4s | CLAP+SQ | Generic | Original extraction, high overlap (50-75%) |
| v2 | 319 | 2-4s | CLAP+SQ | Generic | First quality filter pass |
| v3 | 299 | 6s | SQ-only (0.40 threshold, top_n=15) | Per-clip (8 texture × 4 speed) | Wider spread (25% overlap), segment-aware round-robin |
| v4 | 299 | 6s | SQ-only (0.40 threshold, top_n=15) | Generic: "blowjob, wet sucking and gagging, rhythmic oral sounds" | Same clips as v3, single CLAP embedding |

**Key dataset findings:**
- 6s spread (v3/v4) improved spectral convergence by ~0.1 vs 2-4s spread (v2): 1.39 vs 1.48
- Per-clip prompts (v3) vs generic prompt (v4) made no measurable difference at 299-clip scale
- v4 (generic prompt) generalizes better to diverse input videos at inference — the model learns audio variation from visual features rather than relying on text conditioning
- Segment-aware round-robin with effective quota expansion ensures all source video segments get representation

### Hyperparameter Sweep Results

All experiments use v4 dataset (299 clips, generic prompt), rank 96, alpha 96, visual_dropout_prob 0.5.

#### LoRA Rank (rank 96 vs 128)

| Config | Loss | SC | MCD | PBC |
|--------|------|----|-----|-----|
| v4 r96 | 1.453 | 1.388 | **8.75** | **0.177** |
| v4 r128 | 1.452 | 1.392 | 9.09 | 0.168 |
| v3 r128 | 1.451 | 1.393 | 9.15 | 0.163 |

**Finding:** Rank 128 is marginally worse — extra capacity isn't productive. Rank 96 is the sweet spot.

#### LR Schedule × Timestep Mode (12k steps)

| Config | Loss | SC | MCD | PBC | Notes |
|--------|------|----|-----|-----|-------|
| **constant/curriculum 0.7** | 1.453 | 1.388 | 8.75 | **0.177** | Baseline, stable post-curriculum climb |
| cosine/curriculum 0.7 | 1.454 | 1.384 | 8.77 | 0.143 | **Flatlined post-9k** — cosine decays LR when curriculum switch needs it most |
| constant/curriculum 0.5 | 1.453 | 1.399 | 8.76 | 0.154 | Too-early switch, model not ready at 6k |
| constant/uniform | 1.453 | **1.379** | **8.50** | 0.155 | Best SC+MCD but per-band oscillates (0.10-0.21) |
| **cosine/uniform** | 1.460 | 1.387 | 8.67 | 0.175 | Most stable, near-best PBC |

**Critical finding — cosine + curriculum is a bad combination:** Curriculum switches to harder uniform timesteps at 70% of training. Cosine has already decayed the LR significantly by that point, starving the model of learning capacity exactly when it needs it. Metrics froze completely at steps 9-12k.

**Critical finding — uniform timesteps are viable without curriculum:** With 299 clips the curriculum's logit_normal phase may over-fit easy timesteps rather than building useful foundations. Uniform reaches competitive loss in half the steps.

#### Optimal Training Length

Uniform timestep runs peak at **step 7k**, then degrade or plateau:

| Run | Best PBC | At step | 12k PBC |
|-----|----------|---------|---------|
| constant/uniform | 0.214 | 6k | 0.155 (crashed) |
| cosine/uniform | **0.195** | 7k | 0.176 (stable) |

Constant LR + uniform oscillates wildly (wavelength ~4k steps). Cosine dampens this, plateauing at 0.175 after 9k.

For curriculum runs, the 9k inflection (curriculum switch) is key — post-curriculum improvement requires constant LR to keep learning.

#### Cosine/Uniform Fine-Tuning (8k steps)

| Config | Best PBC | At step | MCD | Notes |
|--------|----------|---------|-----|-------|
| **5e-5 baseline (10k)** | **0.195** | **7k** | **8.56** | **Best overall** |
| 7e-5 | 0.177 | 6k | 9.00 | Higher LR decays too fast under cosine |
| 1e-4 | 0.191 | 1k | 8.65 | Too aggressive, only reached 1k step |
| 5e-5, warmup 500 | 0.194 | 1k | 8.99 | Wastes training budget on ramp-up |

**Finding:** 5e-5 is the right LR for cosine/uniform. Higher LR doesn't help because cosine decay brings it down too fast.

### Best Configuration

```json
{
  "target": "all_attn_mlp",
  "rank": 96,
  "alpha": 96,
  "lr": 0.00005,
  "steps": 10000,
  "schedule_type": "cosine",
  "timestep_mode": "uniform",
  "visual_dropout_prob": 0.5,
  "warmup_steps": 100,
  "batch_size": 8,
  "seed": 42
}
```

**Best checkpoint:** `v4_cosine_uniform/adapter_step07000.pt`
- Loss: 1.438, SC: 1.386, MCD: 8.56, PBC: 0.195
- Use generic prompt for inference: "blowjob, wet sucking and gagging, rhythmic oral sounds"
- Generalizes well across diverse input videos

### Key Takeaways

1. **Dataset quality > quantity:** 299 well-filtered clips with 6s spread outperform 319 clips with 2-4s spread on spectral convergence
2. **Generic prompt > per-clip labels** at small dataset scale — forces the model to learn audio variation from visual signal, improving generalization
3. **Cosine/uniform is simpler and better** than curriculum for small datasets — fewer hyperparameters (no curriculum_switch to tune), trains in 7k steps vs 12k
4. **Cosine + curriculum is actively harmful** — never combine them
5. **Rank 96 is sufficient** — 128 adds parameters without improving any metric
6. **LR 5e-5 is the sweet spot** — higher LRs decay too fast under cosine schedule
7. **Stop at step 7k** for cosine/uniform — per-band correlation peaks then plateaus or degrades
8. **Next lever is dataset scale** — more performers needed to test generalization beyond single-performer training

---

### v5 Sweep — Training Pipeline Improvements (April 2026)

Research-driven changes applied globally before this sweep:
- **AdamW beta2: 0.95 → 0.999** — better gradient variance tracking for fine-tuning
- **DAC `.sample()` → `.mode()`** — deterministic latent encoding, removes stochastic noise from dataset

New features tested (all optional, default disabled):
- **Min-SNR loss weighting** (`min_snr_gamma`) — downweights easy high-SNR timesteps
- **EMA** (`ema_decay`) — exponential moving average of LoRA weights
- **Noise offset** (`noise_offset`) — channel-uniform noise for dynamic range

All experiments: v4 dataset, rank 96, cosine/uniform, lr 5e-5, 8k steps, visual_dropout_prob 0.5.

#### Results

| Experiment | Best PBC | At step | MCD | LSD | Notes |
|-----------|---------|---------|-----|-----|-------|
| **v5_baseline** | **0.226** | **6k** | 9.07 | 21.57 | **Betas fix alone = +16% PBC over v4 best (0.195)** |
| v5_snr5 (γ=5) | 0.218 | 4k | 9.04 | 21.70 | Hurts PBC — easy timesteps are informative at this scale |
| v5_ema (0.9995) | 0.213 | 1k | 10.14 | 26.48 | Broken — decay too high, EMA barely evolves over 8k steps |
| v5_offset03 | 0.236 | 3k | 8.55 | 21.14 | Best MCD/LSD, but PBC spikes at 3k then crashes to 0.203 |
| v5_offset01 | 0.233 | 3k | 8.54 | 21.19 | Same pattern, more stable decay — PBC settles at 0.214 |

#### Analysis

**AdamW betas was the big win.** Changing beta2 from 0.95 to 0.999 improved PBC from 0.195 → 0.226 (+16%) with no downsides. The higher beta2 gives the optimizer longer memory for gradient variance, which helps fine-tuning where gradients are small and noisy.

**Min-SNR hurts at small dataset scale.** Standard Min-SNR downweights easy (high-SNR) timesteps to focus on harder ones. But with 299 clips, the easy timesteps carry useful structural information — the model needs them to learn fine frequency detail. Loss is lower (1.13 vs 1.37) but that's misleading since the weighting changes the loss scale.

**EMA 0.9995 is too slow for 8k-step training.** With decay 0.9995, the half-life is ~1400 optimizer steps. Eval uses EMA weights, so the eval samples are always ~1400 steps behind live training. All metrics flatlined from step 1k onward — the EMA weights barely moved from initialization. Would need decay ≤ 0.99 at this training scale, but that defeats the purpose of smoothing.

**Noise offset improves spectral fidelity but destabilizes PBC.** Both 0.01 and 0.03 gave better MCD (~8.54 vs 9.07) and LSD (~21.2 vs 21.6) — the channel-uniform noise helps the model learn dynamic range. But PBC peaks early (3k) then declines, suggesting the offset interferes with fine per-band frequency tracking as cosine LR decays.

#### Updated Best Configuration

```json
{
  "target": "all_attn_mlp",
  "rank": 96,
  "alpha": 96,
  "lr": 0.00005,
  "steps": 8000,
  "schedule_type": "cosine",
  "timestep_mode": "uniform",
  "visual_dropout_prob": 0.5,
  "warmup_steps": 100,
  "batch_size": 8,
  "seed": 42,
  "min_snr_gamma": 0.0,
  "ema_decay": 0.0,
  "noise_offset": 0.0
}
```

**Best checkpoint:** `v5_baseline/adapter_step06000.pt`
- Loss: 1.373, SC: 1.407, MCD: 9.07, PBC: 0.226
- +16% PBC over previous best (v4_cosine_uniform step 7k)
- Improvement came entirely from AdamW betas fix and DAC `.mode()`

#### Updated Takeaways

9. **AdamW beta2=0.999 >> 0.95** for LoRA fine-tuning — single biggest quality lever found so far
10. **DAC `.mode()` over `.sample()`** — deterministic encoding removes unnecessary variance from small datasets
11. **Min-SNR is counterproductive** at 299-clip scale — easy timesteps carry useful signal
12. **EMA needs decay ≤ 0.99** for sub-10k training, otherwise weights barely evolve
13. **Noise offset trades PBC for MCD/LSD** — potentially useful if spectral fidelity matters more than per-band tracking

---

### v6 Sweep — New Features & Optimizer (April 2026)

Testing recently added training features (cosine sim loss, channel weighting, timestep clipping) plus Prodigy optimizer and architectural/alpha ablations.

**Dataset:** features_v4 (blowjob AD, 299 clips). **Base config:** r96, alpha=96, lr=5e-5, 8k steps, batch=8, cosine schedule, uniform timestep, curriculum_switch=0.7, visual_dropout=0.5.

**Bugs fixed during this sweep:**
- Prodigy optimizer crashed on per-group lr — stripped lr from param groups (`77e5b40`)
- Failed experiments left GPU memory dirty — added cleanup on exception (`b7d4ca3`)
- DAC reference audio encoding ran on CPU — moved model to GPU for round-trip eval (`b29ab06`)

**Variables tested:**
- `v6_baseline` — v5 best config (control)
- `v6_alpha32` — alpha=32 (effective lr scaling = alpha/rank = 0.33)
- `v6_attn_only` — target=all_attn (no MLP layers)
- `v6_cos01` — cos_sim_weight=0.1 (cosine similarity auxiliary loss)
- `v6_chweight` — channel_loss_weight=true (per-channel loss weighting)
- `v6_prodigy` — optimizer_type=prodigy (adaptive learning rate)
- `v6_tclip` — t_min=0.01, t_max=0.99 (avoids uninformative endpoints)
- `v6_combined` — alpha=32 + cos_sim=0.1 + channel_weight + tclip — not yet started

#### Results

| Experiment | SC | MCD | PBC | TV | Loss | Duration |
|-----------|------|------|-------|------|-------|----------|
| **v6_prodigy** | **1.327** | 8.30 | **0.378** | 2.02 | **1.360** | 82 min |
| v6_alpha32 | 1.389 | **8.21** | 0.251 | 2.16 | 1.401 | 70 min |
| v6_baseline | 1.407 | 8.28 | 0.235 | **2.59** | 1.389 | 70 min |
| v6_cos01 | 1.413 | 8.59 | 0.234 | 2.71 | 1.432 | 70 min |
| v6_attn_only | 1.412 | 12.35 | 0.224 | 1.98 | 1.407 | 66 min |
| v6_tclip | 1.410 | 8.52 | 0.228 | 2.71 | 1.390 | 70 min |
| v6_chweight | 1.390 | **8.03** | 0.211 | 2.12 | 1.488 | 71 min |

#### Prodigy Trajectory

| Step | SC | MCD | PBC | TV |
|------|------|------|-------|------|
| 1k | 1.376 | 8.65 | 0.165 | 1.95 |
| 2k | 1.352 | 7.45 | 0.251 | 1.64 |
| 3k | 1.383 | 8.42 | 0.211 | 1.88 |
| 4k | 1.372 | 8.01 | 0.238 | 1.92 |
| 5k | 1.386 | 9.15 | 0.283 | 2.23 |
| 6k | 1.372 | 8.22 | 0.371 | 2.05 |
| 7k | 1.327 | 8.16 | 0.377 | 2.02 |
| 8k | 1.327 | 8.30 | 0.378 | 2.02 |

PBC surges between steps 5-6k (0.283 → 0.371) and SC drops sharply at 7k. Prodigy's adaptive lr found a productive regime late in training — PBC is still climbing at 8k and has not converged.

#### Analysis

**Prodigy is the clear winner.** Best SC (1.327, -5.7% vs baseline), best PBC (0.378, +61% vs baseline), lowest loss. PBC kept climbing through all 8k steps while baseline plateaued at ~0.235 by step 4k. Prodigy's adaptive learning rate discovers a better optimization trajectory than fixed cosine — it hasn't converged yet at 8k steps, strongly suggesting the run should be extended to 12-15k.

**alpha=32 is a modest win.** Better PBC (0.251 vs 0.235) and best MCD (8.21). Lower effective lr (alpha/rank = 0.33) acts as implicit regularization. TV drops to 2.16 vs baseline 2.59 — temporal dynamics are slightly flatter.

**attn_only is clearly bad.** MCD explodes to 12.35 (49% worse than baseline). The model needs MLP layers for spectral fidelity — attention layers alone cannot reconstruct fine frequency detail.

**Cosine sim loss (0.1) is neutral.** PBC tied with baseline (0.234 vs 0.235), slightly worse SC and MCD. The auxiliary loss adds no useful signal at this weight.

**Timestep clipping is neutral.** PBC 0.228 vs baseline 0.235 — within noise. The t=0 and t=1 endpoints are not wasting training capacity at this scale. Clipping [0.01, 0.99] changes nothing meaningful.

**Channel weighting hurts PBC.** Worst PBC (0.211) despite best raw MCD (8.03). Per-channel weighting over-focuses on dominant channels at the expense of cross-band correlation. Higher loss (1.488) confirms it fights the main MSE objective.

#### Updated Best Configuration

```json
{
  "target": "all_attn_mlp",
  "rank": 96,
  "alpha": 96,
  "lr": 5e-05,
  "steps": 8000,
  "schedule_type": "cosine",
  "timestep_mode": "uniform",
  "visual_dropout_prob": 0.5,
  "warmup_steps": 100,
  "batch_size": 8,
  "seed": 42,
  "optimizer_type": "prodigy",
  "curriculum_switch": 0.7,
  "min_snr_gamma": 0.0,
  "ema_decay": 0.0,
  "noise_offset": 0.0,
  "cos_sim_weight": 0.0,
  "channel_loss_weight": false
}
```

**Best checkpoint:** `v6_prodigy/adapter_step08000.pt` (likely not converged — extend run)
- Loss: 1.360, SC: 1.327, MCD: 8.30, PBC: 0.378
- +61% PBC over v5 best (0.235), +67% over v5_baseline (0.226)

#### Updated Takeaways

14. **Prodigy >> AdamW** for LoRA fine-tuning — adaptive lr finds better trajectory, +61% PBC over fixed cosine schedule
15. **Prodigy needs longer runs** — PBC still climbing at 8k, extend to 12-15k steps
16. **MLP layers are essential** — attn_only MCD is 49% worse; skip MLP ablations going forward
17. **Cosine sim loss and channel weighting are dead ends** at current scale — drop from future sweeps
18. **alpha < rank provides mild regularization** — alpha=32 gives +7% PBC over alpha=96, worth combining with Prodigy

---

### v7 Sweep — Prodigy Exploration (April 2026)

Testing Prodigy variants: resume from v6, rank ablations, d_coef/growth_rate control, noise offset, curriculum timing, warmup, alpha.

**Base config:** r96, alpha=96, Prodigy, 13k steps, batch=8, cosine, curriculum_switch=0.7, visual_dropout=0.5.

**Bugs fixed during this sweep:**
- Cross-sweep resume treated `steps` as total instead of additional — empty training loop (`5f2afaa`)
- History loading searched new exp dir only — now falls back to source checkpoint dir (`5f2afaa`)

#### Prodigy Resume Trajectory (steps 1k–15k)

Resumed from `output_v6/v6_prodigy/adapter_step08000.pt`.

| Step | SC | MCD | PBC | TV | Loss |
|------|------|------|-------|------|-------|
| 1k | 1.376 | 8.65 | 0.165 | 1.95 | 1.373 |
| 2k | 1.352 | 7.45 | 0.251 | 1.64 | 1.360 |
| 3k | 1.383 | 8.42 | 0.211 | 1.88 | 1.349 |
| 4k | 1.372 | 8.01 | 0.238 | 1.92 | 1.351 |
| 5k | 1.386 | 9.15 | 0.283 | 2.23 | 1.360 |
| 6k | 1.372 | 8.22 | 0.371 | 2.05 | 1.346 |
| 7k | 1.327 | 8.16 | 0.377 | 2.02 | 1.337 |
| 8k | 1.327 | 8.30 | 0.378 | 2.02 | 1.360 |
| 9k | 1.407 | 9.73 | 0.171 | 2.23 | 1.364 |
| 10k | 1.325 | 8.24 | 0.327 | 1.90 | 1.352 |
| 11k | 1.330 | 7.63 | 0.359 | 1.99 | 1.339 |
| **12k** | **1.184** | **7.40** | **0.433** | 1.80 | 1.343 |
| 13k | 1.212 | 7.90 | 0.439 | 1.87 | 1.351 |
| 14k | 1.226 | 7.99 | 0.430 | 1.89 | 1.339 |
| 15k | 1.224 | 7.98 | 0.431 | 1.89 | 1.332 |

#### Full v7 Results (best step per experiment)

| Experiment | Best PBC | At step | SC | MCD | TV | Key change |
|---|---|---|---|---|---|---|
| **v7_prodigy_r64** | **0.590** | 12k | 1.212 | 7.38 | 1.73 | **rank 64** |
| v7_prodigy_dcoef05 | 0.518 | 12k | 1.315 | 8.10 | 1.79 | d_coef=0.5 |
| v7_prodigy_offset01 | 0.444 | 12k | 1.337 | 7.90 | 1.98 | noise offset 0.01 |
| v7_prodigy_resume | 0.439 | 13k | 1.212 | 7.90 | 1.87 | continued v6 8k→15k |
| v7_prodigy_r128 | 0.323 | 13k | 1.404 | 8.42 | 2.65 | rank 128 |
| v7_prodigy_alpha32 | 0.267 | 5k | 1.391 | 9.47 | 2.26 | alpha=32 (incomplete) |
| v7_prodigy_warmup1k | 0.236 | 4k | 1.362 | 7.57 | 1.56 | warmup=1000 (incomplete) |
| v7_prodigy_growth102 | 0.205 | 13k | 1.369 | 7.95 | 1.58 | growth_rate=1.02 |

#### Prodigy Learning Rate Analysis

Prodigy's `d` (auto-tuned step size) revealed why the model learns aggressively in the first 1k steps:

| Step | d (r96) | Cosine | Effective LR | vs AdamW 5e-5 |
|------|---------|--------|-------------|---------------|
| 1k | 0.000327 | 0.968 | 3.17e-4 | 6.3x |
| 4k | 0.000327 | 0.510 | 1.67e-4 | 3.3x |
| 7k | 0.001239 | 0.039 | 4.8e-5 | 1.0x |
| 8k | 0.001255 | 0.000 | 0 | 0x |

Prodigy ramps `d` as cosine decays — compensating for the schedule. The early effective lr is 6x the AdamW baseline, causing fast domain adaptation in the first 1k steps with slower refinement after.

Lower rank naturally constrains `d`: r64 started at d=0.000273 (vs 0.000327 at r96), giving a 15% lower initial effective lr. This implicit regularization explains why r64 outperformed r96.

#### Analysis

**r64 is the clear winner.** PBC=0.590 (+151% vs v5, +35% vs resume). Lower rank constrains Prodigy's step size, acting as implicit regularization. The optimizer doesn't overshoot early, and the model reaches a better final state.

**Rank ordering: r64 > r96 > r128.** More capacity hurts with Prodigy — r128 PBC was only 0.323. Prodigy compensates for higher rank by pushing `d` higher, amplifying the overshoot problem.

**d_coef=0.5 helps at r96 but is redundant at r64.** At r96, halving `d` gave a smooth trajectory to PBC=0.518. At r64, the rank itself provides enough regularization (see v8 results below).

**growth_rate=1.02 was too restrictive.** PBC stuck at 0.205 — the cap prevented Prodigy from ever reaching the productive lr regime.

**alpha=32 is counterproductive with Prodigy.** Prodigy compensated for the lower effective scaling by pushing `d` 45% higher, negating the regularization intent.

**warmup=1000 backfired.** Longer warmup gave Prodigy more gradient history, and it estimated an even higher d=0.000851 (2.6x baseline).

**Resume instability at step 9k.** PBC crashed to 0.171 — Prodigy's lr state was disrupted at the resume boundary. Recovered within 2k steps.

**Noise offset slow start, steady climb.** PBC=0.085 at 1k (TV=0.62 — nearly collapsed) but recovered to 0.444 by 12k. The offset disrupts early training but doesn't prevent eventual convergence.

---

### v8 Sweep — r64 Refinement (April 2026)

Testing r64 variants: d_coef, lower rank (r48), curriculum timing, noise offset.

**Base config:** r64, alpha=64, Prodigy, 13k steps, batch=8, cosine, curriculum_switch=0.7, visual_dropout=0.5.

#### Results

| Experiment | Best PBC | At step | SC | MCD | TV | Key change |
|---|---|---|---|---|---|---|
| **v8_r64_baseline** | **0.512** | 13k | **1.189** | **6.95** | 1.82 | control (reproduce v7 r64) |
| v8_r48 | 0.497 | 12k | 1.293 | 7.42 | 1.95 | rank 48 |
| v8_r64_dcoef05 | 0.428 | 12k | 1.289 | 7.82 | 1.78 | d_coef=0.5 |
| **v8_r64_curriculum05** | **0.592** | 12k | 1.254 | **6.81** | 1.69 | **curriculum_switch=0.5** |
| v8_r64_dcoef03 | 0.372 | 11k | 1.360 | 8.16 | 2.15 | d_coef=0.3 |

**v8_r64_offset01** (noise_offset=0.01) — hit all-time best SC (1.187) and MCD (6.26) at step 10k, but TV collapsed to 1.34. Perceptual comparison with curriculum05 confirmed fewer distinct sound elements in the waveform despite better spectral numbers. Noise offset optimizes for dominant spectral features at the expense of ambient textures and temporal variation. SC/MCD improvements are misleading when TV is this low — the output sounds cleaner but flatter.

#### Analysis

**r64 baseline reproduced.** PBC=0.512 (vs 0.590 in v7 — seed/order variance). MCD hit **6.95**, first time under 7. SC=1.189 nearly matches v7.

**d_coef hurts at r64.** Both 0.5 (PBC=0.428) and 0.3 (PBC=0.372) are worse. At r64, Prodigy's natural `d` is already lower — the rank itself is enough regularization. Scaling `d` down further starves the optimizer. d_coef=0.3 plateaued at step 11k.

**r48 is slightly worse.** PBC=0.497, SC=1.293, MCD=7.42 — all behind r64. TV=1.95 (healthier than r64's 1.82) suggests underfitting. r48 has too little capacity.

**curriculum_switch=0.5 ties for best PBC and sets new MCD record.** PBC=0.592 matches v7 r64 (0.590), MCD=6.81 beats baseline (6.95). The earlier transition at 50% gives Prodigy more cosine-schedule steps to refine after the curriculum switch. PBC surge started at step 8k, exactly 1.5k after the transition at step 6.5k — consistent with the 2-3k post-transition peak pattern seen across all sweeps. TV=1.69 is below the 1.80 warning threshold, suggesting it's right at the overfitting edge.

**r64 is the sweet spot.** Going lower (r48) underfits, going higher (r96/r128) overfits with Prodigy's adaptive lr. No Prodigy tuning knobs (d_coef, growth_rate, warmup) improve on plain r64. curriculum_switch=0.5 is the only variant that matches or beats the default.

#### Updated Best Configuration

```json
{
  "target": "all_attn_mlp",
  "rank": 64,
  "alpha": 64,
  "lr": 5e-05,
  "steps": 13000,
  "schedule_type": "cosine",
  "timestep_mode": "uniform",
  "visual_dropout_prob": 0.5,
  "warmup_steps": 100,
  "batch_size": 8,
  "seed": 42,
  "optimizer_type": "prodigy",
  "curriculum_switch": 0.5,
  "min_snr_gamma": 0.0,
  "ema_decay": 0.0,
  "noise_offset": 0.0,
  "cos_sim_weight": 0.0,
  "channel_loss_weight": false
}
```

**Best checkpoint:** `v8_r64_curriculum05/adapter_step12000.pt`
- Loss: 1.345, SC: 1.253, MCD: 6.82, PBC: 0.590
- +151% PBC over v5 best (0.235)
- MCD 6.82 — all-time best, first time below 6.9

#### Updated Takeaways

19. **Prodigy converges at 12k steps** on 299-clip dataset — all metrics plateau or regress after
20. **r64 is the optimal rank for Prodigy** — lower (r48) underfits, higher (r96/r128) allows lr overshoot
21. **Prodigy's `d` is naturally constrained by rank** — d_coef/growth_rate tuning is unnecessary at r64
22. **d_coef helps only at r96+** where Prodigy overshoots — at r64 it starves the optimizer
23. **MCD < 7 achievable** — r64 Prodigy hit 6.93/6.95 at step 12k, first time below 7
24. **TV < 1.80 signals overfitting onset** — consistent across all sweep versions
25. **Resume causes transient Prodigy instability** — recovers within 2k steps but wastes compute vs fresh run
26. **curriculum_switch=0.5 > 0.7** — earlier transition gives Prodigy more cosine steps to refine; PBC surge lands 1.5k steps after switch
27. **Noise offset inflates SC/MCD while killing TV** — offset=0.01 hit best SC (1.187) and MCD (6.26) but TV=1.34; perceptual listening confirms fewer sound elements despite better spectral numbers. TV is the better proxy for perceptual quality than SC/MCD alone

---

### v9 Sweep — Temporal Variance & Timestep Distribution (April 2026)

Goal: improve TV while maintaining the PBC/SC/MCD gains from v8. Testing curriculum timing and logit-normal sigma (timestep sampling distribution width).

**Base config:** r64, alpha=64, Prodigy, 13k steps, curriculum_switch=0.5, logit_normal_sigma=1.0.

#### Results

| Experiment | Best PBC | At step | SC | MCD | TV | Key change |
|---|---|---|---|---|---|---|
| **v9_sigma07** | **0.635** | 13k | **1.067** | **6.06** | 1.51 | **sigma=0.7** |
| v9_curriculum04 | 0.644 | 12k | 1.143 | 6.05 | 1.53 | curriculum=0.4 |
| v9_curriculum06 | 0.588 | 12k | 1.302 | 7.18 | 1.66 | curriculum=0.6 |
| v9_sigma05 | 0.580 | 12k | 1.167 | 6.91 | 1.61 | sigma=0.5 |
| *v8 winner* | *0.592* | *12k* | *1.254* | *6.81* | *1.69* | *reference* |

#### Analysis

**sigma=0.7 is the new best — perceptually confirmed.** SC broke 1.1 for the first time (1.067), MCD=6.06, PBC=0.635 (+170% vs v5). TV=1.51 is lower than v8 but perceptual listening during inference confirmed it sounds the best yet. The narrower timestep distribution focuses training on informative mid-range timesteps, improving all spectral metrics simultaneously.

**sigma=0.5 was too narrow.** PBC=0.580, worse than both sigma=0.7 and the v8 baseline. Over-constraining the timestep distribution hurts — the model needs some extreme timesteps to learn the full noise-to-signal trajectory.

**curriculum=0.4 has the highest raw PBC (0.644)** but TV=1.53 and SC=1.143 are worse than sigma=0.7. Pushing the curriculum switch earlier gives diminishing returns — the model runs out of logit-normal steps too early.

**curriculum=0.6 is the most conservative.** Best TV of the group (1.66) with PBC=0.588. A viable option if TV matters more than peak spectral quality, but perceptually sigma=0.7 sounds better.

**TV vs PBC tradeoff appears fundamental at 299 clips.** Every lever that improves PBC/SC/MCD compresses TV. The TV warning threshold (1.80) from earlier sweeps may have been too conservative — sigma=0.7 at TV=1.51 sounds better than curriculum=0.5 at TV=1.69. Perceptual quality correlates more with the PBC×SC combination than TV alone at this scale.

#### Updated Best Configuration

```json
{
  "target": "all_attn_mlp",
  "rank": 64,
  "alpha": 64,
  "lr": 5e-05,
  "steps": 13000,
  "schedule_type": "cosine",
  "timestep_mode": "uniform",
  "visual_dropout_prob": 0.5,
  "warmup_steps": 100,
  "batch_size": 8,
  "seed": 42,
  "optimizer_type": "prodigy",
  "curriculum_switch": 0.5,
  "logit_normal_sigma": 0.7,
  "min_snr_gamma": 0.0,
  "ema_decay": 0.0,
  "noise_offset": 0.0,
  "cos_sim_weight": 0.0,
  "channel_loss_weight": false
}
```

**Best checkpoint:** `v9_sigma07/adapter_step13000.pt`
- Loss: 1.351, SC: 1.067, MCD: 6.06, PBC: 0.635
- +170% PBC over v5 best (0.235)
- SC first time under 1.1, MCD first time under 6.1
- Perceptually confirmed as best sounding during inference

#### Updated Takeaways

28. **logit_normal_sigma=0.7 > 1.0** — narrower timestep distribution improves all spectral metrics; focuses training on informative mid-range timesteps
29. **sigma=0.5 is too narrow** — over-constraining timesteps hurts PBC; the model needs some extreme timesteps
30. **TV threshold is not a hard perceptual limit** — TV=1.51 sounds better than TV=1.69 when SC/MCD/PBC are all stronger; perceptual quality is a function of the full metric profile, not TV alone
31. **Perceptual quality correlates best with low SC + high PBC** — these capture spectral fidelity and frequency tracking; TV is a secondary indicator at this dataset scale

---

### v10 Sweep — Sigma Fine-Tuning (April 2026)

Goal: bracket sigma=0.7 with 0.6 and 0.8 to find the optimal value, and test whether extending the best run beyond 13k keeps improving.

**Base config:** r64, alpha=64, Prodigy, 13k steps, curriculum_switch=0.5, logit_normal_sigma=0.7.

#### Results

| Experiment | Best PBC | At step | SC | MCD | TV | Key change |
|---|---|---|---|---|---|---|
| **v10_sigma07_16k** | **0.661** | 15k | **1.058** | **6.12** | 1.51 | **sigma=0.7 extended to 16k** |
| v10_sigma08 | 0.642 | 13k | 1.236 | 7.25 | **1.82** | sigma=0.8 |
| v10_sigma08_16k | 0.624 | 16k | 1.268 | 7.40 | 1.47 | sigma=0.8 extended — overfit |
| v10_sigma06 | 0.550 | 13k | 1.310 | 7.55 | 1.39 | sigma=0.6 — worse on everything |

#### Analysis

**sigma=0.7 extended to 16k is the all-time best.** PBC=0.661 (+181% vs v5), SC=1.058, MCD=6.12. Resumed from v9_sigma07 at 13k, PBC kept climbing for another 2k steps. The run was still improving at 15k but plateaued by 16k.

**sigma=0.8 is the best TV/PBC balance.** TV=1.82 is the highest in the top 5, with competitive PBC=0.642. At 13k steps, sigma=0.8 produces richer temporal dynamics — the wider timestep distribution preserves more audio variation. However, extending to 16k caused overfitting: TV collapsed to 1.47 and PBC dropped to 0.624.

**sigma=0.6 is worse than 0.7 on all metrics.** PBC=0.550, SC=1.310, TV=1.39 — the timestep distribution is too narrow, same pattern as sigma=0.5 in v9.

**Extending sigma=0.8 hurts, extending sigma=0.7 helps.** sigma=0.7 has a tighter optimization landscape that benefits from longer training. sigma=0.8's wider distribution means the model is still exploring at 13k — more steps push it into overfitting rather than refinement.

#### Top 5 Checkpoints (Published)

| Rank | Experiment | Step | PBC | SC | MCD | TV |
|------|-----------|------|-------|-------|------|------|
| 1 | v10_sigma07_16k | 15k | 0.661 | 1.058 | 6.12 | 1.51 |
| 2 | v9_curriculum04 | 12k | 0.644 | 1.143 | 6.05 | 1.53 |
| 3 | v10_sigma08 | 13k | 0.642 | 1.236 | 7.25 | 1.82 |
| 4 | v9_sigma07 | 13k | 0.635 | 1.067 | 6.06 | 1.51 |
| 5 | v8_r64_curriculum05 | 12k | 0.592 | 1.254 | 6.81 | 1.69 |

Published as safetensors to [ethanfel/FoleyTune-LoRAs](https://huggingface.co/ethanfel/FoleyTune-LoRAs) on Hugging Face.

#### Final Best Configuration

```json
{
  "target": "all_attn_mlp",
  "rank": 64,
  "alpha": 64,
  "lr": 5e-05,
  "steps": 13000,
  "schedule_type": "cosine",
  "timestep_mode": "uniform",
  "visual_dropout_prob": 0.5,
  "warmup_steps": 100,
  "batch_size": 8,
  "seed": 42,
  "optimizer_type": "prodigy",
  "curriculum_switch": 0.5,
  "logit_normal_sigma": 0.7
}
```

**Best checkpoint:** `v10_sigma07_16k/adapter_step15000.pt`
- PBC: 0.661, SC: 1.058, MCD: 6.12, TV: 1.51
- +181% PBC over v5 best (0.235)
- All-time best SC and PBC

#### Updated Takeaways

32. **sigma=0.7 is optimal** — bracketing with 0.6 and 0.8 confirms 0.7 gives the best spectral quality
33. **sigma=0.8 maximizes TV** (1.82) with competitive PBC — use when temporal dynamics matter more than peak spectral fidelity
34. **Extended training (15k) helps sigma=0.7 but hurts sigma=0.8** — tighter timestep distributions benefit from longer refinement; wider distributions overfit with more steps
35. **12-15k steps is the convergence window** on 299 clips — sigma=0.7 peaked at 15k, sigma=0.8 at 13k
36. **More data is the remaining big lever** — the PBC/TV tradeoff is a dataset ceiling at 299 clips

---

### JAV Intense Sex Sweep — Rank, Optimizer & Prodigy+ (April–May 2026)

New dataset: 394 unique clips, 8s each, from JAV intense sex content. Starting from the best BJ config (r64, σ=0.7, Prodigy, cosine, curriculum_switch=0.5), adapted for the larger and acoustically different dataset.

#### Phase 1: Sigma & Rank

**Base config:** Prodigy, cosine, curriculum_switch=0.5, batch=8, 15k steps, visual_dropout=0.5.

**Sigma sweep (R64):**
- σ=0.8 at 13k was the perceptual best — richer temporal dynamics than σ=0.7 for this dataset's frequency profile (85% energy below 1875Hz median rolloff)
- σ=1.0 diverged at R128 but worked at R64/R96

**Rank sweep:**

| Config | SC | MCD | PBC | Notes |
|---|---|---|---|---|
| R64 σ=0.8 (13k) | 1.104 | 9.08 | 0.265 | Previous best |
| **R96 σ=0.8 (13k)** | **1.100** | **9.27** | **0.357** | **Perceptual winner** |
| R128 σ=0.8 (15k) | 1.070 | 10.11 | 0.286 | OK but not better perceptually |
| R128 σ=1.0 (15k) | 1.206 | 11.76 | 0.207 | Diverged — HF ratio 0.45 |

**Finding:** R96 is the sweet spot for 394 clips — R64 too constrained, R128 diverges (especially with σ=1.0). Contradicts the BJ finding where R64 > R96 — larger datasets benefit from higher rank.

#### Phase 2: Prodigy Refinements (R96 σ=0.8)

Tested: decouple, constant schedule, growth_rate, d_coef, noise_offset.

| Experiment | SC | MCD | PBC | Steps | Notes |
|---|---|---|---|---|---|
| Resume 13k+decouple | 1.096 | 9.05 | 0.387 | 16k | decouple=True was already default — improvement from more steps, not decouple |
| Constant schedule | 1.144 | 10.10 | 0.187 | 8k | Too volatile — uncapped d causes instability. Cancelled |
| Constant+growth_rate=1.02 | 1.160 | 11.83 | 0.152 | 8k | Cap too restrictive. Cancelled |
| **d_coef=0.5** | **1.088** | 9.12 | 0.215 | 15k | **Best SC** — slower LR adaptation. Rock solid plateau from 10k |
| noise_offset=0.02 | 1.113 | 9.36 | 0.286 | 15k | Lower loss but advantage not sustained vs plain cosine |
| noff+dcoef combined | 1.144 | 10.15 | 0.263 | 8k | Not synergistic — oscillating metrics. Cancelled |
| **Prodigy+ (8k)** | **1.122** | **8.82** | 0.204 | 8k | **Smoothest training, lowest MCD/LSD at 8k** |

**Key findings:**
- Constant schedule fails with Prodigy — cosine decay is necessary for stable convergence
- d_coef=0.5 gives best SC but low PBC — useful insight for Phase 3
- Prodigy+ Schedule-Free (prodigy-plus-schedule-free v2.0.1) was the most promising direction — smoothest training, best loss in fewer steps

#### Phase 3: Prodigy+ Schedule-Free (R96 σ=0.8)

Switched to Prodigy+ as the optimizer. PP has built-in schedule-free optimization, d_limiter, and StableAdamW base. Schedule is forced to constant (PP handles scheduling internally). Requires `optimizer.train()`/`optimizer.eval()` calls for weight averaging.

**PP Experiments:**

| Experiment | SC | MCD | PBC | Steps | Notes |
|---|---|---|---|---|---|
| PP+noff=0.02 | 1.047 | 7.35 | 0.496 | 20k | Strong but baseline beats it |
| **PP baseline** | **1.016** | **7.07** | **0.557** | 20k | **Best at 20k — ahead on every metric** |
| PP+d_coef=0.5 | 1.194 | 8.97 | 0.152 | 7k | d_coef double-damps PP's built-in d_limiter. Cancelled |
| **PP baseline ext** | **0.903** | **6.17** | **0.672** | 30k | **All-time best — still improving** |
| PP no curriculum | 1.062 | 7.94 | 0.453 | 19k | Plateaued — curriculum wins (see below) |

**PP Baseline Trajectory (the money run):**

| Step | SC | MCD | LSD | PBC | Loss |
|---|---|---|---|---|---|
| 6k | 1.103 | 8.86 | 19.49 | 0.207 | 1.360 |
| 10k | 1.122 | 9.35 | 20.11 | 0.247 | 1.382 |
| 15k | 1.059 | 7.25 | 16.64 | 0.461 | 1.333 |
| 20k | 1.016 | 7.07 | 16.15 | 0.557 | 1.356 |
| 22k | 0.996 | 6.48 | 15.42 | 0.591 | 1.323 |
| 25k | 0.971 | 6.36 | 15.28 | 0.634 | 1.349 |
| 27k | 0.944 | 6.32 | 15.32 | 0.646 | 1.333 |
| **30k** | **0.903** | **6.17** | **15.24** | **0.672** | 1.359 |

Massive acceleration after step 10k (curriculum switch at 50% of 20k). SC broke below 1.0 at step 21k. MCD below 7.0 at step 16k. Still improving at 30k — extension to 40k queued.

**Curriculum Ablation:**

Tested PP with `curriculum_switch=0.0` (logit-normal σ=0.8 from step 0, no uniform phase).

| Step | No Curriculum SC | Baseline SC | No Curriculum PBC | Baseline PBC |
|---|---|---|---|---|
| 4k | **1.152** | 1.171 | 0.226 | **0.254** |
| 7k | **1.080** | 1.116 | 0.208 | **0.219** |
| 10k | **1.089** | 1.122 | 0.211 | **0.247** |
| 15k | 1.060 | **1.059** | 0.413 | **0.461** |
| 17k | 1.056 | **1.031** | 0.447 | **0.486** |
| 19k | 1.062 | **1.022** | 0.453 | **0.530** |

No curriculum converges SC faster early (logit-normal from step 0 gives harder gradients immediately) but plateaus at SC~1.06, PBC~0.45 by 17-19k. The baseline with curriculum blows through those levels and keeps improving.

**Conclusion:** curriculum_switch=0.5 is essential for PP training. The uniform-first phase builds a stable gradient foundation that PP exploits aggressively when harder logit-normal timesteps arrive. Without the curriculum switch, PP learns the hard stuff from the start but converges to a weaker final result.

**Waveform Analysis — PP+noff 10k vs all other runs:**

PP+noff at 10k had crest factor 25.1dB (vs 19-21dB for all other runs) and peak amplitude 0.803 (vs 0.42-0.50). Most balanced frequency distribution: 51% in 500-1kHz, 16% in 1-2kHz, 20% in 2-4kHz. Other runs over-concentrated energy in a single band (dcoef: 70% in 500-1k, baseline: 66% in 2-4k). The combination of noise_offset + PP produced the first waveforms with natural-sounding dynamic range and transient structure. However, the PP baseline without noise_offset ultimately produced better spectral metrics at longer training — PP handles dynamic range on its own given enough steps.

#### Key Findings (JAV Sweep)

37. **R96 > R64 for 394-clip dataset** — more data supports higher rank. Contradicts BJ finding (299 clips → R64 optimal). The rank sweet spot scales with dataset size
38. **Prodigy+ Schedule-Free >> regular Prodigy** — smoothest training, best final metrics. SC=0.903, MCD=6.17, PBC=0.672 at 30k. Still improving
39. **PP needs no noise_offset** — noise_offset helps early (better crest factor, dynamic range) but PP baseline overtakes it by 20k. PP handles dynamic range internally given enough steps
40. **PP d_coef=0.5 is counterproductive** — PP has a built-in d_limiter; halving d_coef double-damps the LR adaptation. Loss stuck at 1.41
41. **Curriculum switch is essential for PP** — uniform-first phase (curriculum_switch=0.5) builds a stable foundation that accelerates learning when harder timesteps arrive. No-curriculum plateaus at inferior metrics
42. **PP training scales to 30k+ steps** — no convergence at 30k. The schedule-free optimizer maintains adaptive capacity throughout, unlike cosine which forces convergence via LR decay. Extension to 40k queued
43. **Constant schedule fails with Prodigy** — uncapped d growth causes oscillating metrics. Cosine decay is necessary for stable convergence with regular Prodigy
44. **Post-curriculum acceleration is dramatic with PP** — metrics accelerate 2-5k steps after the curriculum switch. SC drops from 1.12 → 0.90 between steps 10k and 30k. PBC climbs from 0.25 → 0.67 in the same window

### Phase 4: The PBC/TV Tradeoff (May 2026)

**Problem discovered:** Prodigy+ produces excellent spectral metrics (SC, PBC, MCD) but kills temporal variance (TV), resulting in spectrally accurate but poorly synced audio. Confirmed on both BJ (299 clips) and JAV (394 clips) datasets.

**The seesaw:** PBC and TV are inversely correlated across all configurations. PP at 30k: PBC=0.672 but TV=0.73 (collapsed from 1.89). Regular Prodigy: TV=1.19 (healthy) but PBC=0.357 (weak). No config achieved both.

| Config | TV | PBC | SC | Sync Quality |
|---|---|---|---|---|
| Regular Prodigy R96 σ=0.8 (13k) | 1.19 | 0.357 | 1.100 | Good |
| PP baseline (20k) | 0.93 | 0.557 | 1.016 | Poor |
| PP baseline ext (30k) | 0.73 | 0.672 | 0.903 | Very poor |
| BJ rank3 Prodigy (13k) | 1.82 | 0.642 | 1.236 | Good |
| BJ PP baseline (13k) | 1.35 | 0.666 | 1.227 | Poor |

**Root cause:** PP's schedule-free adaptive LR finds that temporally smooth audio minimises loss better than temporally dynamic audio. A smooth prediction is "close enough" in MSE, while capturing temporal spikes costs more when timing is slightly off. Regular Prodigy's cosine LR decay acts as implicit regularisation — it forces the model to consolidate learned sync patterns rather than continuing to optimise toward spectral flatness.

**Failed interventions (all on JAV PP R96 σ=0.8, 20k):**

| Approach | Result |
|---|---|
| Lower visual dropout (0.3) | TV=1.81 but PBC stuck at 0.20 — too much visual signal starved audio learning |
| Lower visual dropout (0.15) | PBC=0.40, TV=0.98 — milder PP-like collapse |
| Temporal variance loss v1 (variance comparison, weight=0.1) | Too weak, TV still collapsed. Formula compared against noise-dominated target |
| Temporal diff loss v2 (MSE on torch.diff, weight=0.5) | Loss doubled to 2.7 but TV=0.96. Applied at all noise levels — at high noise, temporal diffs are pure noise, actively harmful |
| Resume PP→Prodigy+cosine + TV loss | TV dropped further (0.93→0.86), PBC flat |
| SNR-gated multi-scale temporal diff (gate σ=0.3, scales 1/4/16, weight=0.3) | TV=0.83 at 20k. Gating was correct fix conceptually but still couldn't counteract PP's optimisation pressure |
| Visual dropout curriculum (0.1→0.5 ramp over 40%) | TV=0.97 at 9k and falling. Worse PBC than baseline |
| All combined | Same trajectory |

**Key insight:** The PBC/TV tradeoff is fundamentally a **data complexity problem**, not a loss/optimizer problem. The BJ dataset (single position, consistent motion-to-sound mapping) achieves both good PBC and TV with regular Prodigy. The JAV dataset (multiple positions, diverse actions) has too many visual-audio patterns for the model to learn tight sync — PP shortcuts to spectrally flat audio, Prodigy can't push PBC high enough.

**Conclusion:** For multi-position datasets, subset by position before training. A tight, consistent motion-to-sound mapping is more important than dataset size. No amount of loss engineering can substitute for data that the model can actually learn to sync.

#### BJ PP Sweep (299 clips, R64 σ=0.8)

Tested PP on the single-position BJ dataset. PP improves PBC over regular Prodigy but TV already degraded at 13k — cancelled before sync collapsed further.

| Experiment | Optimizer | VD | Steps | PBC | TV | SC | MCD | Status |
|---|---|---|---|---|---|---|---|---|
| bj_pp_baseline | PP | 0.5 | 13k | 0.666 | 1.35 | 1.227 | — | Cancelled (TV regressing) |
| BJ rank3 (reference) | Prodigy | 0.5 | 13k | 0.642 | 1.82 | 1.236 | — | Best BJ |

PP got marginally better PBC (+0.02) but lost 0.47 TV. On this dataset Prodigy already achieves strong PBC, so PP offers no benefit.

#### JAV PP Sweep (394 clips, R96 σ=0.8)

| Experiment | Steps | PBC | TV | SC | MCD | LSD |
|---|---|---|---|---|---|---|
| output_8_pp_noff002 (noise offset 0.02) | 20k | 0.496 | 0.96 | 1.047 | 7.35 | 16.6 |
| output_8_pp_baseline | 20k | 0.557 | 0.93 | 1.016 | 7.07 | 16.2 |
| output_8_pp_baseline_ext | 30k | 0.672 | 0.73 | 0.903 | 6.17 | 15.2 |

PBC climbs monotonically with steps, TV drops monotonically. Extended to 30k shows the trajectory clearly — PP never stops smoothing.

#### JAV TV Recovery Sweep (R96 σ=0.8)

Attempted to recover temporal variance via loss-level interventions.

| Experiment | Method | Steps | PBC | TV | SC | MCD |
|---|---|---|---|---|---|---|
| tv_recovery_prodigy | Resume PP@20k → Prodigy+cosine + TV loss (0.1) | +1k | 0.556 | 0.86 | 1.000 | 7.43 |
| tv_pp_tvloss_full | PP + TV loss v1 (variance formula, 0.1) | 3k* | 0.195 | 1.63 | 1.182 | 11.24 |
| tv_pp_tdiff_full | PP + temporal diff v2 (MSE on torch.diff, 0.5) | 19k | 0.246 | 0.96 | 1.084 | 8.07 |

*Cancelled at 3k — variance comparison formula was ineffective against noise-dominated targets. Rewrote to temporal diff MSE for v2.

Resume from PP checkpoint: TV continued dropping. Once PP has smoothed temporal dynamics, even switching to cosine+TV loss can't reverse it.
Temporal diff v2 at 0.5 weight: doubled total loss to ~2.7 but TV still collapsed. Temporal diffs at high noise levels are pure noise — the loss pushed toward smoothness rather than preserving dynamics.

#### JAV Visual Dropout Sweep (R96 σ=0.8, regular Prodigy+cosine)

Tested whether reducing visual dropout forces more sync learning.

| Experiment | VD Prob | Steps | PBC | TV | SC | MCD |
|---|---|---|---|---|---|---|
| output_8_r96_vd030 | 0.30 | 20k | 0.206 | 1.81 | 1.203 | 13.47 |
| output_8_r96_vd015 | 0.15 | 20k | 0.403 | 0.98 | 1.076 | 8.25 |
| Regular Prodigy (reference, vd=0.5) | 0.50 | 13k | 0.357 | 1.19 | 1.100 | — |

VD 0.3: preserves TV but PBC stuck at 0.2 — too much visual signal starved unconditional audio learning.
VD 0.15: better PBC but TV collapsed to 0.98 — low dropout turns into PP-like behaviour where the model leans on visual signal so hard it smooths temporal dynamics anyway.

#### JAV TV v3 Sweep (R96 σ=0.8, PP, SNR-gated losses)

Final attempt: SNR-gated multi-scale temporal diff, visual dropout curriculum, and both combined.

| Experiment | Method | Steps | PBC | TV | SC | MCD |
|---|---|---|---|---|---|---|
| pp_snr_gated_tv | SNR-gated TV (gate=0.3, scales 1/4/16, weight=0.3) | 20k | 0.531 | 0.83 | 1.016 | 7.09 |
| pp_vd_curriculum | VD curriculum (0.1→0.5 ramp over 40% of steps) | 9k* | 0.318 | 0.97 | 1.091 | 9.25 |
| pp_all_three | SNR-gated TV + VD curriculum | — | — | — | — | — |

*Cancelled at 9k, same TV collapse trajectory. All-three skipped — both components failed individually.

SNR gating (only apply temporal loss at t < 0.3 where temporal structure is visible) was conceptually correct but insufficient against PP's optimization pressure. VD curriculum ramped dropout too slowly to matter. Neither approached the BJ reference (PBC=0.642, TV=1.82).

### Reverse Cowgirl Sweep — Single-Position Validation (May 2026)

New single-position dataset: 124 clips, 8s each, reverse cowgirl only. Tests the Phase 4 hypothesis that dataset composition (not optimizer/loss) is the real lever for achieving both PBC and TV. Uses the proven BJ rank3 config (R64, Prodigy, cosine) as baseline.

#### Results

| Experiment | σ | Curriculum | Steps | PBC | TV | SC | MCD | LSD |
|---|---|---|---|---|---|---|---|---|
| sigma08_cur05 | 0.8 | 0.5 (4k) | 8k | 0.549 | 1.52 | 1.048 | 5.43 | 12.8 |
| sigma08_late_cur (resume@4k) | 0.8 | 0.6 | 10k | 0.383 | 1.36 | 1.372 | 6.25 | 14.2 |
| **sigma08_cur06_fresh** | **0.8** | **0.6 (6k)** | **10k** | **0.661** | **1.56** | **0.950** | **4.71** | **12.2** |

#### Analysis

**TV rose during training** — first time observed across any dataset. Went from 0.68 at 1k to 1.57 at 9k. On BJ and JAV, TV either started high and held or dropped monotonically. The tight motion-to-sound mapping (single position, consistent rhythm) lets the model build temporal dynamics rather than averaging them out.

**Curriculum switch timing is critical on small datasets.** The cur=0.5 run (switch at 4k/8k) plateaued immediately after switching to uniform — PBC stalled at 0.43, TV dipped. The cur=0.6 run (switch at 6k/10k) gave 2k more logit-normal steps, and PBC jumped from 0.41→0.62 in the post-switch phase. The extra focused learning time lets Prodigy lock in spectral patterns before the uniform distribution dilutes gradients.

**Resume with different total steps breaks cosine LR.** Resuming from 4k (of 8k) into a 10k run shifted the cosine schedule — LR was higher than the original at the same step. Results were worse across all metrics. For cosine-scheduled training, always run from scratch when changing total steps.

**All-time best SC (0.950) and MCD (4.71).** Both metrics broke previous records held by the BJ rank3. PBC=0.661 matches BJ rank3's 0.642 while TV=1.56 is competitive with BJ's 1.82. Perceptual quality confirmed — the LoRA sounds excellent with tight motion-to-audio sync.

#### Best Configuration

```json
{
  "target": "all_attn_mlp",
  "rank": 64,
  "alpha": 64,
  "lr": 5e-05,
  "steps": 10000,
  "schedule_type": "cosine",
  "timestep_mode": "uniform",
  "visual_dropout_prob": 0.5,
  "warmup_steps": 100,
  "batch_size": 8,
  "seed": 42,
  "optimizer_type": "prodigy",
  "curriculum_switch": 0.6,
  "logit_normal_sigma": 0.8
}
```

**Best checkpoint:** `sigma08_cur06_fresh/adapter_step09000.pt` (PBC=0.660, TV=1.57, SC=1.026)

#### Updated Findings

50. **Single-position datasets confirm the Phase 4 hypothesis** — 124 RCG clips achieved PBC=0.661 + TV=1.56, matching or beating the 299-clip BJ dataset. Dataset coherence > dataset size
51. **TV can rise during training** — only observed on single-position datasets with consistent rhythmic patterns. Multi-position datasets always show flat or declining TV
52. **curriculum_switch=0.6 outperforms 0.5 on small datasets** — the extra logit-normal steps let Prodigy lock in spectral patterns before uniform timesteps dilute gradients. On 124 clips / 10k steps, this was the difference between PBC=0.55 and PBC=0.66
53. **Never resume with different total steps on cosine schedule** — the LR mismatch degrades all metrics. Always run from scratch when changing step count

---

#### Updated Findings (Phase 4)

45. **PP kills temporal variance on all datasets** — TV drops monotonically (1.89→0.73 over 30k). The schedule-free LR removes the cosine decay that implicitly preserved sync
46. **PBC and TV are inversely correlated** — across all configs, optimisers, and loss functions. Fundamental tradeoff in flow matching with visual conditioning
47. **Loss-level interventions cannot fix the PBC/TV tradeoff** — temporal diff loss, SNR gating, multi-scale, visual dropout curriculum all failed. The optimiser's incentive to smooth temporal dynamics is stronger than any auxiliary loss
48. **Dataset composition is the real lever** — single-position datasets (BJ) achieve both good sync and spectral quality. Multi-position datasets (JAV) force the model to choose between temporal fidelity and spectral accuracy
49. **Temporal losses must be noise-level-aware** — ungated temporal diff loss at high noise just adds noise to gradients, pushing toward smoothness. SNR gating (t < 0.3) is conceptually correct but insufficient alone

---

### Missionary Finetune — Small Dataset Transfer Learning (May 2026)

45-clip missionary dataset. Too small for full training — R64 memorises, R32/R16 underfit. Tested finetuning from the RCG 9k checkpoint with a new `freeze_blocks` parameter that freezes early LoRA layers to prevent catastrophic forgetting.

#### Strategy

`freeze_blocks=N` freezes LoRA weights in `triple_blocks.0..N-1`, keeping pretrained representations in early blocks while only training later blocks on the new dataset. Fresh Prodigy optimizer (checkpoint optimizer state is incompatible due to changed param count).

#### Results — From-Scratch Attempts

| Experiment | Rank | VD | Steps | PBC | TV | SC | MCD |
|---|---|---|---|---|---|---|---|
| baseline (R64) | 64 | 0.5 | 5k | 0.641 | 1.23 | 0.654 | 3.84 |
| r16_vd03 | 16 | 0.3 | 5k | 0.336 | 1.34 | 0.733 | 4.97 |
| r32_vd03 | 32 | 0.3 | 5k | 0.109 | 0.98 | 1.295 | 6.32 |

R64 baseline has best metrics but loss dropped steeply without plateau — memorisation. R16 learned something but noisy. R32 didn't learn.

#### Results — Finetune from RCG 9k (Unfrozen)

| Experiment | Steps | PBC | TV | SC | MCD |
|---|---|---|---|---|---|
| from_rcg (no freeze) | 9.5k | 0.162 | 0.94 | 1.289 | 5.46 |
| from_rcg (no freeze) | 13k | 0.340 | 0.70 | 1.246 | 5.03 |

Catastrophic forgetting — PBC collapsed from 0.696 (RCG base) to 0.162 immediately. Never recovered past 0.34.

#### Results — Freeze Block Experiments

| Experiment | Freeze | d_coef | Best PBC | Best TV | Best SC | Notes |
|---|---|---|---|---|---|---|
| freeze12 | 12/18 | 1.0 | 0.652 | 1.77 | 0.521 | Unstable — PBC swings 0.14–0.65 |
| **freeze14** | **14/18** | **1.0** | **0.682** | **1.43** | **0.572** | More stable, best overall balance |
| freeze14_slow | 14/18 | 0.5 | -0.006 | 1.55 | 1.343 | Too conservative — didn't learn |
| freeze14_d08 | 14/18 | 0.8 | 0.644 | 1.24 | 0.617 | Same oscillation as d_coef=1.0 |

**freeze14 at step 12500** (PBC=0.682, TV=1.27, SC=0.572) is the best checkpoint. Fewer trainable blocks (4 vs 6) reduces overfitting capacity and oscillation. However, perceptual quality is noisy — metallic artifacts present. Not production-ready.

#### Analysis

**Freeze blocks prevent catastrophic forgetting but introduce oscillation.** With only 4 trainable blocks on 45 clips, each gradient update has outsized impact. The frozen RCG blocks and trainable blocks compete — metrics swing depending on whether the trainable blocks align with or interfere with the frozen base.

**d_coef doesn't fix the oscillation.** Tried 1.0, 0.8, and 0.5 — the first two oscillate similarly, the third doesn't learn at all. The instability is structural (too few clips for the trainable capacity), not a learning rate issue.

**45 clips is fundamentally insufficient** for position-specific LoRA training, even with transfer learning. The freeze approach partially works (PBC=0.682 matches the RCG base) but produces noisy, metallic audio. A proper missionary dataset with 100+ clips is needed.

#### Updated Findings

54. **freeze_blocks enables transfer learning for small datasets** — freezing 14/18 blocks preserved RCG knowledge while adapting 4 blocks to missionary. Without freezing, finetuning causes catastrophic forgetting (PBC 0.696→0.162)
55. **More frozen blocks = more stability** — freeze14 (4 trainable) was more stable and higher quality than freeze12 (6 trainable). Fewer trainable params = less capacity to overfit on 45 clips
56. **Prodigy d_coef has a narrow useful range for frozen finetuning** — 1.0 and 0.8 oscillate, 0.5 doesn't learn. The instability is structural, not LR-related
57. **45 clips is below minimum viable dataset size** — even with transfer learning and aggressive freezing, the model can't learn stable position-specific patterns. ~100+ clips needed for production quality

---

### Combined Multi-Dataset Training — RCG + Missionary (May 2026)

Combined 124 RCG clips + 46 missionary clips (170 total) into a single training run using the new multi-dataset support (`dataset_json` as list). Each dataset has its own prompt ("reverse cowgirl sex..." vs "missionary sex..."). Dual eval tracks metrics on both RCG and missionary val clips independently.

#### Infrastructure Added

- **Multi-dataset**: `dataset_json` accepts a list of paths; clips are concatenated. Text embeddings zero-padded to max batch length for mixed-prompt batches
- **Multi-eval**: `eval_npz` accepts a list of `{name, path}` objects. Each entry gets its own samples, spectrograms, and prefixed metrics (e.g. `rcg_per_band_correlation`, `missionary_temporal_variance`)
- **DAC round-trip for eval references**: eval reference audio goes through DAC encode→decode before PBC comparison, matching the main reference pipeline

#### Results

| Experiment | Steps | PBC | TV | SC | MCD |
|---|---|---|---|---|---|
| combined_169clip_baseline | 10k | 0.585 | 1.45 | 1.213 | 5.02 |
| combined_170clip_13k | 13k | 0.681 | 1.67 | 1.001 | 5.92 |
| combined_170clip_15k | 15k (best@14k) | 0.672 | 1.69 | 0.895 | 5.40 |

10k was too few — PBC still climbing. 13k and 15k converge to similar PBC (~0.67-0.68). 15k has slightly better SC (0.895 vs 1.001).

#### Analysis

**Both concepts are learned separately** — the model distinguishes between RCG and missionary prompts. Each position produces position-appropriate audio when conditioned on the right text prompt.

**Imbalanced clip counts degrade the minority class.** With 124 RCG vs 46 missionary clips (~73/27 split), the model biases toward RCG. Missionary eval PBC stayed near zero or negative throughout, while the primary PBC (evaluated on a RCG train clip) reached 0.68. For production multi-position LoRAs, datasets should be roughly balanced per prompt.

**Larger mixed datasets need more steps.** 170 clips required 13k steps vs 10k for 124-clip single-position. Roughly proportional to dataset size.

#### Updated Findings

58. **Multi-dataset training works** — different positions with different prompts are learned in a single LoRA. Text conditioning separates the concepts at inference time
59. **Dataset balance matters for multi-prompt LoRAs** — a 73/27 clip imbalance causes the minority prompt to underperform. Balance datasets per prompt for production quality
60. **Multi-eval with DAC round-trip is required for cross-dataset PBC** — raw reference audio vs DAC-decoded output produces systematically negative PBC due to codec artifacts

---

### CLAP Prompt Experiments — RCG (May 2026)

Tested prompt sensitivity by re-extracting the same RCG videos with different CLAP text prompts.

| Prompt | Clips | PBC | Notes |
|---|---|---|---|
| Original: "reverse cowgirl sex, rhythmic skin slapping, wet clapping, heavy breathing, moaning" | 124 | 0.696 | Best overall |
| V3 hybrid: "reverse cowgirl sex, rhythmic slapping and wet clapping impacts, loud moaning, heavy panting" | 128 | 0.652 | Close second |
| Narrow: "reverse cowgirl sex, rhythmic skin slapping, loud female moaning" | 133 | 0.368 | Lost audio breadth |
| CLAP-optimized: "rhythmic slapping and wet clapping impacts, loud moaning and whimpering vocals, heavy panting and gasping" | 125 | 0.487 | Removed domain context |
| CLAP resume from RCG 9k | 125 | 0.231 | RCG associations fight new embedding |

#### Updated Findings

61. **Prompt must describe full audio texture** — narrow prompts ("loud female moaning") tank PBC from 0.70 to 0.37 on the same videos. CLAP conditions the denoiser; incomplete prompts force the model to learn against its conditioning
62. **Domain context helps despite CLAP not "knowing" it** — removing "reverse cowgirl sex" from the prompt degraded results. The term may anchor the embedding in a useful region even without explicit AudioSet coverage
63. **Gender-specific terms are weak in CLAP** — "female moaning" underperforms "moaning" alone. CLAP training de-biased gender terms
64. **Resuming from a different-prompt checkpoint hurts** — the original checkpoint's learned associations fight the new text embedding, producing worse results than training from scratch

---

### Prodigy Optimizer Tuning (May 2026)

#### safeguard_warmup

During LR warmup, Prodigy overestimates its `d` (step size) parameter because small effective LR keeps gradients large. `safeguard_warmup=True` prevents this by excluding the LR factor from the denominator accumulation. Added as configurable `prodigy_safeguard_warmup` parameter (default: true).

**Tradeoff:** Safeguard prevents loss spikes but makes the LR more conservative. On the doggy dataset, safeguard-on and no-warmup produced nearly identical results — the conservatism may slow convergence without improving final quality.

#### Updated Findings

65. **safeguard_warmup prevents loss spikes but doesn't improve final PBC** — on the doggy dataset, safeguard on, safeguard off, and no warmup all converged to similar PBC (~0.30-0.38). The warmup interaction is cosmetic, not structural
66. **noise_offset is a significant improvement** — noise_offset=0.03 improved PBC from 0.40 to 0.54 on the same dataset. Helps the model learn dynamic range differences between clips

---

### Doggystyle POV Training — Multi-Performer (May 2026)

Two datasets tested: `mp4_doggy_features` (5 sources, 135 clips) and `mp4_doggy_clap_features` (20+ sources, 106 clips).

#### First Dataset (5 sources, 135 clips)

| Experiment | VD | PBC | Notes |
|---|---|---|---|
| baseline (vd=0.5) | 0.5 | 0.215 | Bland average — too few sources for high VD |
| vd=0.2 | 0.2 | 0.472 | 2× PBC — low VD critical for few-source datasets |

#### Second Dataset (20+ sources, 106 clips)

| Experiment | Best PBC | @step | Key Change |
|---|---|---|---|
| **noise_vd0** | **0.632** | **7k** | vd=0 + noise_offset 0.03 — best perceptual quality |
| 13k | 0.640 | 13k | Highest PBC but lost clapping sounds — overtrained |
| vd05_13k | 0.604 | 12k | vd=0.5 + 13k |
| noise_offset | 0.539 | 7k | noise_offset 0.03 alone |
| 8k | 0.440 | 7k | Baseline at fewer steps |
| baseline (vd=0.25) | 0.404 | 8k | Reference |
| no_curriculum | 0.371 | 5k | Uniform timesteps worse |
| σ=0.7/cur=0.5 | 0.360 | 7k | Second-best RCG combo didn't transfer |

#### Updated Findings

67. **Visual dropout is counterproductive for multi-performer datasets** — with 20+ sources, performer diversity naturally prevents identity binding. vd=0 allows the model to use every visual frame for sync, producing better audio-visual coupling. vd=0 + noise_offset reached PBC 0.632 vs 0.404 at vd=0.25
68. **noise_offset=0.03 is a reliable improvement** — consistent gains across experiments. Helps the model learn dynamic range and prevents collapse to a bland average
69. **More steps can overtrain and lose audio components** — 13k had the highest PBC (0.640) but perceptually lost clapping sounds. 7-8k with noise_offset preserved full audio texture. PBC alone doesn't capture component loss
70. **Dataset source diversity matters more than clip count** — 106 clips from 20+ sources (PBC 0.632) dramatically outperformed 135 clips from 5 sources (PBC 0.472). Visual diversity > clip volume
71. **Curriculum and sigma settings transfer poorly across datasets** — σ=0.7/cur=0.5 was second-best for RCG but performed worst for doggy. Optimal hyperparameters are dataset-dependent
72. **Eval PBC on a single held-out clip is unreliable** — eval PBC was near zero or negative across all runs regardless of perceptual quality. Perceptual evaluation (listening tests) remains essential

---

### Noise Offset Sweep — vd=0 (May 2026)

Tested noise_offset values {0.0, 0.01, 0.03, 0.05, 0.1} all with vd=0 on the doggy_clap dataset (20+ sources, 106 clips). All runs used 8k steps, σ=0.8, cur=0.6, Prodigy+cosine, safeguard_warmup=True.

| offset | train PBC @7k | train TV @7k | final loss | Perceptual Assessment |
|---|---|---|---|---|
| 0.00 | — | — | 1.254 | Clean audio, good quality, slightly less sync than 0.03 |
| 0.01 | — | — | 1.253 | No clear improvement over 0.00 |
| **0.03** | — | — | **1.256** | **Best sync + full audio texture — selected** |
| 0.05 | 0.585 | 1.317 | 1.253 | Competitive but no clear advantage over 0.03 |
| 0.10 | 0.615 | 1.356 | 1.272 | Moaning diminished — too aggressive, suppresses sustained textures |

Best checkpoint: `doggy_clap_noise_vd0/adapter_step07000.pt` (noise_offset=0.03, vd=0)

#### Updated Findings

73. **noise_offset sweet spot is 0.03 for foley LoRA** — below 0.03 (0.01) shows no benefit; above (0.05) shows no clear advantage; at 0.1 the channel-uniform noise overwhelms sustained spectral content (moaning) while preserving transients (claps). 0.03 is the Goldilocks value
74. **Higher noise_offset increases loss** — offset=0.1 final loss 1.272 vs 0.03 at 1.256. The added noise makes the denoising task harder, which is only beneficial up to a point
75. **noise_offset primarily helps transient sync, not spectral fidelity** — the mechanism (per-sample, per-channel, uniform across time) teaches the model to distinguish dynamic range patterns. Sustained textures (moaning, breathing) don't benefit and degrade at high values
76. **min_snr_gamma=5 doesn't improve perceptual quality** — changes loss scale (~1.0 vs ~1.25) but spectrally and perceptually indistinguishable from unweighted. Mid-range timestep rebalancing isn't a bottleneck for foley
77. **rank 64 is sufficient for 106-clip datasets** — rank 96 (PBC 0.599) and 128 (PBC 0.647) didn't improve over rank 64 (PBC 0.632) perceptually. Extra capacity without proportionally more data doesn't help; the bottleneck is data diversity, not model capacity

#### Recommended Config for Multi-Performer Foley LoRA (20+ sources)

```
target: all_attn_mlp
rank: 64, alpha: 64
optimizer: prodigy (safeguard_warmup: true)
schedule: cosine
steps: 7000-8000
visual_dropout: 0.0
noise_offset: 0.03
logit_normal_sigma: 0.8
curriculum_switch: 0.6
warmup_steps: 100
batch_size: 8
```
