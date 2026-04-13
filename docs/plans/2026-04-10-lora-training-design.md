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
