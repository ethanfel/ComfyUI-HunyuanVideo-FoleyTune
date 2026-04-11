# Chunked Long-Form V2A Generation — Design

## Motivation

HunyuanVideo-Foley trains on 8-second clips and has a 15-second hardcoded max.
Generating audio for longer videos requires chunking: split the video into
overlapping segments, denoise each, and stitch the results. Research (SaFa,
LoVA, LD-LAudio, Echoes Over Time) shows that naive concatenation produces
audible boundary artifacts — spectral discontinuities, timbral shifts, and
energy jumps. SaFa's binary latent swap during denoising avoids these by
operating in latent space at each diffusion step rather than patching artifacts
after the fact.

This design adds a `FoleyChunkedSampler` node and extends the existing
`FoleyFeatureExtractor` to output in-memory features, decomposing the monolithic
`HunyuanFoleySampler` into a reusable two-node pipeline.

## Architecture

### Modified Node: FoleyFeatureExtractor (nodes_lora.py)

Add a `FOLEY_FEATURES` output alongside the existing `npz_path`. Add a
`negative_prompt` input for unconditional text encoding.

```
Inputs:
  hunyuan_deps:    HUNYUAN_DEPS
  image:           IMAGE
  prompt:          STRING
  negative_prompt: STRING (default "")
  frame_rate:      FLOAT (default 25.0)
  duration:        FLOAT (default 0.0 = auto)
  cache_dir:       STRING (default "")
  name:            STRING (default "clip")

Outputs:
  npz_path:  STRING           (existing — LoRA training)
  features:  FOLEY_FEATURES   (new — dict of tensors for ChunkedSampler)
```

The `FOLEY_FEATURES` dict contains:
```python
{
    "clip_feat":        torch.Tensor,  # [1, T_clip, 768]  — SigLIP2 at 8fps
    "sync_feat":        torch.Tensor,  # [1, T_sync, 768]  — Synchformer
    "text_feat":        torch.Tensor,  # [1, T_text, 768]  — CLAP prompt
    "uncond_text_feat": torch.Tensor,  # [1, T_text, 768]  — CLAP negative prompt
    "duration":         float,
}
```

Feature extraction runs once for the full video. SigLIP2 and Synchformer are
lightweight — even 5 minutes of features fits comfortably in VRAM.

### New Node: FoleyChunkedSampler (nodes.py)

```
Inputs (required):
  hunyuan_model:   HUNYUAN_MODEL
  hunyuan_deps:    HUNYUAN_DEPS (for DAC)
  features:        FOLEY_FEATURES
  cfg_scale:       FLOAT (default 7.0)
  steps:           INT (default 50)
  sampler:         [euler, heun-2, midpoint-2, kutta-4]
  seed:            INT
  chunk_duration:  FLOAT (default 8.0, max 15.0)
  overlap_seconds: FLOAT (default 1.6)
  crossfade_mode:  [safa, latent, waveform] (default safa)
  batch_size:      INT (default 1)

Inputs (optional):
  force_offload:     BOOLEAN
  torch_compile_cfg: TORCH_COMPILE_CFG
  block_swap_args:   BLOCK_SWAP_ARGS

Outputs:
  audio: AUDIO
```

Short clips (duration <= chunk_duration) skip chunking — single pass, no
overhead. The existing `HunyuanFoleySampler` stays untouched for backwards
compatibility.

## Chunking Strategy

### Chunk Boundary Computation

Given total `duration`, `chunk_duration`, and `overlap_seconds`:

```
stride = chunk_duration - overlap_seconds

Chunk 0: [0.0,            chunk_duration]
Chunk 1: [stride,         stride + chunk_duration]
Chunk 2: [2*stride,       2*stride + chunk_duration]
...
Chunk N: [N*stride,       min(N*stride + chunk_duration, duration)]
```

Example: 25s video, 8s chunks, 1.6s overlap (20%):
```
stride = 6.4s
Chunk 0: [0.0,  8.0]   8.0s
Chunk 1: [6.4, 14.4]   8.0s  (overlaps 6.4–8.0 with chunk 0)
Chunk 2: [12.8, 20.8]  8.0s  (overlaps 12.8–14.4 with chunk 1)
Chunk 3: [19.2, 25.0]  5.8s  (last chunk, shorter)
```

The last chunk may be shorter than `chunk_duration`. The model handles variable
latent lengths via RoPE.

### Feature Slicing Per Chunk

For chunk `[t_start, t_end]`:

- **SigLIP2**: `clip_feat[:, int(t_start*8):int(t_end*8), :]` — direct time slice
- **Synchformer**: Map time boundaries to segment indices. Synchformer uses
  16-frame windows with stride 8 at 25fps. Slice to nearest clean segment
  boundary, then `F.interpolate` to match the chunk's latent length (same
  approach the model uses internally)
- **Text**: Same for all chunks (full prompt and negative prompt)

### Latent Dimensions

For chunk duration D seconds:
- Audio latent: `[B, 128, int(D * 50)]` — 50fps latent frame rate
- Overlap in latent frames: `int(overlap_seconds * 50)` — e.g. 1.6s = 80 frames

## Crossfade Modes

Three modes, all producing identical results for single-chunk clips:

### 1. `safa` (default, best quality)

Binary latent swap during denoising, based on SaFa (arXiv:2502.05130).

At each denoising step, for each pair of adjacent chunks:

```python
# Overlap region: e.g. 80 latent frames
# Shifting binary mask — rotates each step to avoid fixed swap pattern
shift = (step_idx * 5) % overlap_len
mask = (torch.arange(overlap_len) + shift) % 2

# Binary swap (not averaging — preserves high-frequency content)
left_overlap  = left_latents[:, :, -overlap_len:]
right_overlap = right_latents[:, :, :overlap_len]
merged = torch.where(mask == 0, left_overlap, right_overlap)
left_latents[:, :, -overlap_len:]  = merged
right_latents[:, :, :overlap_len] = merged
```

Key insight from SaFa: averaging-based methods (MultiDiffusion) cause spectral
aliasing — they suppress high-frequency components in overlap regions. Binary
swap preserves spectral detail.

After all denoising steps, take non-overlap regions from each chunk plus the
final merged overlap. DAC decode the assembled full latent.

### 2. `latent`

Denoise all chunks independently (no interaction during denoising). After
denoising, apply equal-power crossfade on the `[B, 128, T]` latents in the
overlap region:

```python
# Equal-power: sqrt curves, constant energy
t = torch.linspace(0, 1, overlap_len)
w_left  = torch.sqrt(1 - t)
w_right = torch.sqrt(t)
blended = w_left * left_latent + w_right * right_latent
```

Assemble full latent, then single DAC decode pass.

### 3. `waveform`

Denoise and DAC decode each chunk independently. Apply equal-power crossfade
on the final audio waveforms:

```python
# Same sqrt curves, applied to audio samples
overlap_samples = int(overlap_seconds * 48000)
t = torch.linspace(0, 1, overlap_samples)
w_left  = torch.sqrt(1 - t)
w_right = torch.sqrt(t)
blended = w_left * left_audio + w_right * right_audio
```

Simplest mode, but operates after artifacts are already baked in.

## VRAM Profile

Same as current single-clip generation. During chunked denoising:
- Only one chunk's forward pass occupies GPU at a time
- Chunk latents stored on CPU between steps: `[B, 128, ~400]` per chunk (small)
- All chunks share the same model weights (no duplication)

For `safa` mode: sequential forward passes per chunk per step. Total inference
time scales linearly with number of chunks.

## Edge Cases

- **duration <= chunk_duration**: Single pass, no chunking, no overhead
- **Last chunk shorter**: Model handles via RoPE, no padding needed
- **overlap_seconds >= chunk_duration**: Clamp to `chunk_duration * 0.5`, log warning
- **overlap_seconds = 0**: Pure concatenation (no crossfade), useful for comparison

## Data Flow

```
Video frames (IMAGE)
     │
     ▼
FoleyFeatureExtractor (nodes_lora.py, modified)
  ├─ npz_path (STRING) ──► LoRA training pipeline
  └─ features (FOLEY_FEATURES) ──┐
                                  │
     ┌────────────────────────────┘
     ▼
FoleyChunkedSampler (nodes.py, new)
  │
  │  if duration ≤ chunk_duration:
  │    single-pass denoise → DAC decode → done
  │
  │  if duration > chunk_duration:
  │    1. Compute chunk boundaries with overlap
  │    2. For each denoising step t:
  │       ├─ For each chunk: slice features, model forward
  │       ├─ if safa: binary swap overlap frames
  │       └─ scheduler.step() per chunk
  │    3. Stitch:
  │       ├─ safa:     assemble non-overlap + merged overlap → DAC decode
  │       ├─ latent:   equal-power crossfade on latents → DAC decode
  │       └─ waveform: DAC decode per chunk → equal-power crossfade on audio
  │
  └─► AUDIO (stitched waveform)
```

## References

- **SaFa**: Latent Swap Joint Diffusion (arXiv:2502.05130, ICCV 2025)
- **LoVA**: Long-form Video-to-Audio (arXiv:2409.15157, ICASSP 2025)
- **LD-LAudio**: Dual Lightweight Adapters (arXiv:2508.11074)
- **Echoes Over Time / MMHNet**: Length Generalization (arXiv:2602.20981)
- **SoundReactor**: Frame-level AR Generation (arXiv:2510.02110)
