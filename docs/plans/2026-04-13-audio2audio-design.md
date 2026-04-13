# Audio-to-Audio Manipulation Design

**Date:** 2026-04-13
**Goal:** Add audio2audio (img2img equivalent) and inpainting to the existing sampler infrastructure.

## Architecture

Two changes following ComfyUI conventions:

1. **Extend `FoleyTuneChunkedSampler`** with optional `init_audio` (AUDIO) + `strength` (FLOAT) inputs. When connected, encodes audio through DAC, noises at strength level, denoises from there. Covers audio2audio, variations, SDEdit, super-resolution — all the same code path.

2. **New `FoleyTuneInpainter` node** with required `init_audio` + time range (`start_seconds`, `end_seconds`). Builds a latent mask, runs denoising with per-step replacement of known regions. Covers inpainting and temporal extension.

## Component 1: `utils.py` — Core Functions

### `encode_audio_to_latents(audio_waveform, dac_model, device)`
- Input: raw waveform tensor [B, 1, samples]
- DAC encode in continuous mode → DiagonalGaussianDistribution
- Return `.mode()` (deterministic) → [B, 128, T]

### `denoise_process_with_generator()` — New Optional Params
- `init_latents=None` — [1, 128, T] from DAC encode
- `strength=1.0` — 0.0=keep original, 1.0=full generation
- `inpaint_mask=None` — [1, 1, T] bool, True=regenerate
- `inpaint_original=None` — [1, 128, T] original latents for known regions

When `init_latents` + `strength < 1.0`:
- Compute begin_step = `num_steps - int(strength * num_steps)`
- Truncate timesteps to start from begin_step
- Noise init_latents: `z = sigma_start * noise + (1 - sigma_start) * init_latents`
- Denoising loop runs on truncated schedule

When `inpaint_mask` provided:
- Same noise vector used for both init and known-region re-noising
- After each scheduler step, replace known regions with properly noised original:
  `latents = where(mask, latents, sigma_next * noise + (1-sigma_next) * original)`
- Soft mask edges (4 frames gradient) to prevent DAC boundary clicks

### `chunked_denoise_process()` — Same New Params
- Passes `init_latents` and `strength` through
- For chunked a2a: slices init_latents per chunk (like features)
- For inpainting: slices mask and original per chunk

## Component 2: `FoleyTuneChunkedSampler` — Extended

New optional inputs:
```
"init_audio": ("AUDIO",)
"strength": ("FLOAT", default=1.0, min=0.0, max=1.0, step=0.05)
```

Logic:
1. If init_audio → encode through DAC → init_latents
2. Pass init_latents + strength to chunked_denoise_process()
3. No init_audio or strength=1.0 → existing behavior unchanged

## Component 3: `FoleyTuneInpainter` — New Node

Inputs:
```
required:
  hunyuan_model: FOLEYTUNE_MODEL
  hunyuan_deps: FOLEYTUNE_DEPS
  features: FOLEYTUNE_FEATURES
  init_audio: AUDIO
  start_seconds: FLOAT (default=0.0, min=0.0, step=0.1)
  end_seconds: FLOAT (default=2.0, min=0.1, step=0.1)
  cfg_scale: FLOAT (default=4.5)
  steps: INT (default=50)
  sampler: ["euler", "heun-2", "midpoint-2", "kutta-4"]
  seed: INT
optional:
  force_offload: BOOLEAN (default=True)
  torch_compile_cfg: FOLEYTUNE_COMPILE_CFG
  block_swap_args: FOLEYTUNE_BLOCKSWAP
```

Output: `("AUDIO",)` — single inpainted result

Logic:
1. Encode init_audio through DAC → original_latents [1, 128, T]
2. Build mask from seconds → latent frames (50fps = 48000/960)
3. Apply 4-frame soft edges on mask boundaries
4. Run denoise_process_with_generator with inpaint_mask + inpaint_original
5. DAC decode → trim → return AUDIO

Temporal extension: if end_seconds > audio duration, pad original_latents with zeros and extend mask.
