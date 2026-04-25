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
