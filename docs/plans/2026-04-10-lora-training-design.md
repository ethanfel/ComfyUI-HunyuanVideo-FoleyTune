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

Caches features for training. One clip per execution, auto-incremented naming.

**Inputs:**
- `HUNYUAN_DEPS` — provides SigLIP2, Synchformer, CLAP, DAC models
- `IMAGE` — video frames
- `AUDIO` — paired audio
- `prompt` — text description
- `frame_rate` — source video FPS
- `duration` — clip duration (0 = auto from audio)
- `cache_dir` — output directory for .npz + audio files
- `name` — base filename for auto-increment (e.g., "gunshot" -> gunshot_001.npz)

**Process:**
1. Extract SigLIP2 features at 8fps, 512x512 -> `[1, N_clip, 768]`
2. Extract Synchformer features at 25fps, 224x224 -> `[1, N_sync, 768]`
3. Encode text via CLAP -> `[1, 768]`
4. Encode audio via DAC encoder -> `[1, 128, T]` latents
5. Save all to .npz with metadata (prompt, duration, fps)
6. Copy/save audio file alongside with matching stem

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
```
1. Load cached .npz features + DAC-encoded latents
2. Sample timestep t ~ configured distribution
3. Sample noise x0 ~ N(0, I)
4. Interpolate: xt = (1-t)*x0 + t*x1
5. Forward: v_pred = foley_model(xt, t, clip_feat, sync_feat, text_feat)
6. Loss: MSE(v_pred, x1 - x0)
7. Backward through LoRA params only
8. AdamW step (beta1=0.9, beta2=0.95) + gradient clipping (max_norm=1.0)
```

**Outputs:**
- Checkpoints: `adapter_step00500.pt`, `adapter_final.pt`
- Metadata: `meta.json`
- Loss curves: `loss_raw.png`, `loss_smoothed.png`
- Eval samples: `samples/step_00500.wav`, etc.
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
  "base": { "rank": 64, "lr": 1e-4, "steps": 3000, "target": "all_attn_mlp" },
  "experiments": [
    {"id": "rank32", "rank": 32},
    {"id": "rank64"},
    {"id": "rank128", "rank": 128},
    {"id": "loraplus", "lora_plus_ratio": 16.0}
  ]
}
```

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

## VRAM Estimates (96 GB target)

| Config | VRAM |
|--------|------|
| bf16 model + rank 64 + batch 8 | ~25-30 GB |
| bf16 model + rank 64 + batch 16 | ~40-50 GB |
| bf16 + rank 64 + all_attn_mlp + batch 16 | ~60-70 GB |

All configurations fit comfortably within 96 GB with no need for fp8, gradient checkpointing, or block swap.

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
