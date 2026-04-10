# LoRA Training for HunyuanVideo-Foley

LoRA lets you teach the model new or partially-known sound classes using a small set of video+audio pairs. Only the low-rank adapter weights are trained — the full 10.3 GB model stays frozen.

---

## Overview

Training is split into three steps:

1. **Dataset preparation** — optimize and clean your audio files using the Foley Dataset Pipeline nodes.
2. **Feature extraction** — extract SigLIP2/Synchformer/CLAP features from your video+audio pairs using the `Foley Feature Extractor` node.
3. **Training** — run the `Foley LoRA Trainer` node.

Visual features (SigLIP2, Synchformer) and text features (CLAP) are pre-computed and cached as `.npz` files, so the feature extraction models are not loaded during training — only the transformer and DAC encoder.

---

## Requirements

Same environment as Foley inference. Additional optional packages for the dataset pipeline:

```
soxr              # High-quality resampling
pyloudnorm        # LUFS loudness normalization
pedalboard         # Audio compression and filtering
soundfile         # Audio I/O
audiomentations   # Pitch/time-stretch augmentation (optional)
matplotlib        # Loss curves and eval charts
```

---

## Architecture Context

Understanding the model helps choose training parameters:

| Component | Details |
|---|---|
| Transformer | 18 TwoStreamCABlock + 36 SingleStreamBlock, hidden_dim=1536, 12 heads |
| VAE | DAC neural codec, 128D continuous latent, 48kHz, single-stage encode/decode |
| Visual features | SigLIP2 (8fps, 512x512) + Synchformer (25fps, 224x224) |
| Text features | CLAP (last_hidden_state, not pooled) |
| Diffusion | Flow matching with velocity prediction, MSE loss |
| Model size | ~10.3 GB (bf16) |

---

## Step 1 — Prepare the dataset

### 1.1 Collect video+audio pairs

Gather paired video and audio clips of the sound you want to train.

| Guideline | Details |
|---|---|
| **Format** | WAV or FLAC preferred. Avoid MP3 — lossy compression degrades training |
| **Aspect ratio** | 16:9 landscape preferred, 1:1 square also fine |
| **Resolution** | ≥ 480p sufficient (downscaled to 512px and 224px internally) |
| **Frame rate** | Any — set via the `frame_rate` input |
| **Duration** | DAC handles variable lengths, but keep clips consistent (e.g. all ~5-8 seconds) |

**Diversity beats quantity.** Ten clips of a sound in different environments train better than fifty clips of the same recording. Vary: distance, room acoustics, intensity, speed.

### 1.2 How many clips do I need?

| Dataset size | Scenario | Expected result |
|---|---|---|
| **5-10 clips** | Quick test / proof of concept | May work if the model partially knows the sound; often underfits |
| **15-30 clips** | Fine-tuning a sound the model gets wrong | Good starting point |
| **30-60 clips** | Teaching a new but acoustically simple sound | Reliable convergence |
| **60-150 clips** | Unusual or complex sounds, strong style shift | Needed for stable generalization |
| **150-300+ clips** | Sounds the model has never encountered | Required to avoid overfitting; consider rank 128 |

### 1.3 Optimize audio quality (recommended)

Chain the Foley Dataset Pipeline nodes in ComfyUI to clean and normalize your audio before feature extraction:

```
Foley Dataset Loader (folder of raw audio)
  → Foley Dataset Resampler (48000 Hz)
  → Foley Dataset LUFS Normalizer (-23 LUFS, -1 dBTP)
  → Foley Dataset Compressor (ratio 2.5:1, mix 0.4)
  → Foley Dataset HF Smoother (cutoff 16000 Hz, blend 0.5)
  → Foley Dataset Inspector (remove clipped/noisy/silent clips)
  → Foley Dataset Saver (output to cleaned folder as 24-bit FLAC)
```

| Node | Purpose | Key defaults |
|---|---|---|
| **Resampler** | Match DAC's native sample rate | 48000 Hz (soxr VHQ quality) |
| **LUFS Normalizer** | Consistent loudness across clips | -23 LUFS (EBU R128), -1 dBTP peak |
| **Compressor** | Reduce dynamic range, preserve transients | Parallel style, 2.5:1 ratio, 0.4 mix |
| **HF Smoother** | Tame extreme high frequencies | 16 kHz cutoff, 0.5 blend |
| **Inspector** | Quality control | Flags clipping, low SNR, codec artifacts, silence |

**DAC vs BigVGAN note:** DAC's neural codec handles high frequencies better than BigVGAN (used in SelVA), so the HF Smoother defaults are less aggressive — 16 kHz cutoff and 0.5 blend vs 12 kHz and 0.7 in SelVA.

### 1.4 Optional: Spectral matching with reference audio

If you want to match your training audio's spectral profile to what DAC reproduces best:

1. Run a few clips through `Foley VAE Roundtrip` to see what DAC's codec preserves.
2. Save those roundtripped clips to a reference directory.
3. Use `Foley Dataset Spectral Matcher` with that reference directory to EQ your training audio toward DAC's preferred distribution.

This is the Foley equivalent of SelVA's hardcoded VAE spectral matching, but reference-based instead of hardcoded.

### 1.5 Optional: Expand small datasets with augmentation

```
Foley Dataset Augmenter
    variants_per_clip: 2-3
    gain_range_db: 3.0
    pitch_range_semitones: 0.5  (requires audiomentations)
    time_stretch_range: 0.1     (requires audiomentations)
    keep_originals: true
```

This turns 20 clips into 60. Keep augmentation subtle — extreme pitch shifts create unrealistic training targets.

---

## Step 2 — Extract features

For each video+audio pair, run the `Foley Feature Extractor` node in ComfyUI:

```
HunyuanDependenciesLoader (vae_128d_48k.pth, synchformer_state_dict.pth)
  → Foley Feature Extractor
      image: video frames (from Load Video node)
      audio: paired audio (cleaned from Step 1)
      prompt: "description of the sound"
      frame_rate: source video FPS
      duration: 0 (auto from audio length)
      cache_dir: /path/to/features/
      name: "gunshot"
```

Queue once per clip. Each execution saves one `.npz` + `.wav` pair with auto-incremented names (`gunshot_001.npz`, `gunshot_002.npz`, etc.).

### What gets cached

Each `.npz` contains:
- `clip_features` — SigLIP2 visual features `[1, N_clip, 768]` at 8fps
- `sync_features` — Synchformer sync features `[1, N_sync, 768]` at 25fps
- `text_embedding` — CLAP text features `[1, seq_len, 768]`
- `prompt` — the text description
- `duration` — clip duration in seconds
- `fps` — source frame rate

A matching `.wav` file (resampled to 48kHz) is saved alongside.

### Prompt guide

The prompt conditions the CLAP text features used during training. Imprecise prompts produce unfocused conditioning, forcing the LoRA to compensate with noise.

| Sound | Weak prompt | Strong prompt |
|---|---|---|
| Dog bark | `dog` | `a large dog barking loudly` |
| Footsteps | `walking` | `heavy boots on a wooden floor` |
| Water | `water` | `water dripping into a metal bucket` |
| Explosion | `explosion` | `a large explosion with deep bass rumble` |

**Rules of thumb:**
- Describe the *sound*, not the visual scene
- Keep prompts consistent across all clips for the same sound class
- Avoid negations (`no background noise`)

### Directory structure

After feature extraction:

```
dataset/my_sound/
    gunshot_001.npz    ← features from Foley Feature Extractor
    gunshot_001.wav    ← paired audio (48kHz)
    gunshot_002.npz
    gunshot_002.wav
    ...
```

---

## Step 3 — Train

Connect the nodes in ComfyUI:

```
HunyuanModelLoader (hunyuanvideo_foley.pth)
HunyuanDependenciesLoader (vae_128d_48k.pth, synchformer_state_dict.pth)
  → Foley LoRA Trainer
      data_dir: /path/to/features/
      output_dir: /path/to/lora_output/
      (hyperparameters — see below)
```

The trainer:
1. Loads all `.npz` features and DAC-encodes the paired audio to latents.
2. Applies LoRA layers to the transformer, freezes base weights.
3. Runs the flow matching training loop.
4. Saves checkpoints, eval audio samples, and loss curves at intervals.
5. Returns the model with LoRA active for immediate inference.

### Hyperparameters

#### Target layers

Controls which transformer layers get LoRA adapters. More layers = more capacity but higher VRAM and overfitting risk.

| Preset | Layers | Total adapted | Use case |
|---|---|---|---|
| `audio_attn` | Audio self-attention QKV + proj | 36 | Minimal adaptation, fine details |
| `audio_cross` | Above + audio cross-attention + text cross-KV | 90 | Good for text-conditioned sounds |
| `all_attn` | Above + visual conditioning attention | 162 | Full attention adaptation |
| `all_attn_mlp` | Above + MLP layers (fc1/fc2) | 234 | Maximum capacity (recommended with 96 GB) |

**Default:** `all_attn_mlp` — with 96 GB VRAM there's no reason to limit capacity.

#### Rank

| Rank | Use case | Trainable params (all_attn_mlp) |
|---|---|---|
| 16 | Fine details on a known sound | ~2.4M |
| 32 | Good balance | ~4.8M |
| 64 | **Recommended** — high capacity for 128D latent space | ~9.6M |
| 128 | Very complex sounds, 150+ clips | ~19.2M |

**Default:** 64. Foley's 128D latent space (vs SelVA's 40D) benefits from higher rank.

#### Learning rate

`1e-4` is the recommended default. If training is unstable (loss spikes in the first 200 steps), try `5e-5`. If convergence is very slow, try `2e-4`.

Warmup (default 100 steps) ramps the LR from 0 to avoid instability at the start.

#### Batch size

| Batch size | VRAM (bf16, rank 64, all_attn_mlp) | Notes |
|---|---|---|
| 1 | ~22 GB | Noisy gradients, slow |
| 4 | ~25 GB | Reasonable starting point |
| 8 | ~30 GB | **Recommended** — stable gradients |
| 16 | ~45 GB | Better convergence on larger datasets |
| 32 | ~70 GB | Best gradient quality — fits on 96 GB |

Higher batch size gives smoother loss curves and faster convergence. With 96 GB, prefer larger batches over more steps.

#### Steps

| Dataset size | Recommended steps |
|---|---|
| 10-20 clips | 2000-4000 |
| 20-50 clips | 4000-8000 |
| 50+ clips | 6000-15000 |

Watch the loss curve — if the smoothed line has been flat for 2000+ steps, training has converged. Adding more clips will let it go lower.

#### Timestep sampling

Controls how training timesteps are sampled. Flow matching trains by interpolating between noise and target at random timesteps.

| Mode | Description | When to use |
|---|---|---|
| `uniform` | All timesteps equally | Safe baseline |
| `logit_normal` | Concentrates near t=0.5 via `sigmoid(N(0, σ))` | **Recommended** — 0.2-0.3 dB lower loss floor |
| `curriculum` | logit_normal first, then uniform | Experimental — may improve fine detail |

**Default:** `logit_normal` with `sigma=1.0`.

The `logit_normal_sigma` parameter controls the distribution width:
- σ=0.5: sharp peak at t=0.5, less coverage of extremes
- σ=1.0: moderate peak, balanced coverage (default)
- σ=2.0: broader, approaches uniform

#### Advanced options

| Parameter | Default | Description |
|---|---|---|
| `alpha` | 64 | LoRA scaling factor. 1:1 ratio with rank is the HunyuanVideo convention |
| `grad_accum` | 1 | Gradient accumulation steps. Use when batch size alone doesn't fit |
| `init_mode` | `standard` | `standard` (Kaiming A, zero B) or `pissa` (SVD-based) |
| `use_rslora` | false | Rank-stabilized scaling: `alpha/sqrt(rank)` instead of `alpha/rank` |
| `lora_dropout` | 0.0 | Dropout on the LoRA path. 0.05-0.1 helps on small datasets (<20 clips) |
| `lora_plus_ratio` | 1.0 | B-matrix LR multiplier. 16.0 enables LoRA+ (faster convergence) |
| `schedule_type` | `constant` | `constant` or `cosine` LR decay after warmup |
| `latent_mixup_alpha` | 0.0 | Beta-distribution latent interpolation for augmentation |
| `latent_noise_sigma` | 0.0 | Additive Gaussian noise on target latents for regularization |
| `precision` | `bf16` | `bf16` (Ampere+), `fp16` (older GPUs), `fp32` (debug only) |

---

## Step 4 — Use the adapter

Connect `Foley LoRA Loader` between the model loader and the sampler:

```
HunyuanModelLoader (hunyuanvideo_foley.pth)
  → Foley LoRA Loader
      adapter_path: /path/to/lora_output/adapter_final.pt
      strength: 1.0
  → HunyuanFoleySampler (normal inference)
```

> **Important:** Wire the LoRA Loader output to the **Sampler**, not the Feature Extractor. The LoRA adapts the transformer which only runs in the Sampler.

| Strength | Effect |
|---|---|
| 0.5-0.7 | Conservative — blends adapter with base model, less noise |
| 1.0 | Full adapter strength (default) |
| >1.0 | Exaggerated effect, may introduce artifacts |

The loader reads rank, alpha, and target from the metadata embedded in the `.pt` file — no need to set them manually. The base model is not modified (deep copy).

---

## Experiment sweeps

Use `Foley LoRA Scheduler` to run multiple experiments from a JSON configuration:

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

The scheduler loads the dataset once, runs each experiment sequentially, and produces a `loss_comparison.png` overlay chart. Completed experiments are skipped on resume. Drop a `skip_current.flag` file in the output root to abort the current experiment and move to the next.

---

## Evaluation

Use `Foley LoRA Evaluator` to compare multiple adapters on the same dataset:

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

Computes spectral metrics for each adapter:
- HF energy ratio (>4kHz / total)
- Spectral centroid (Hz)
- Spectral rolloff (85% energy)
- Spectral flatness (0=tone, 1=noise)
- Temporal variance (dynamic range)
- Log spectral distance vs reference (dB)
- Mel cepstral distortion vs reference

Outputs a `metric_comparison.png` bar chart and per-adapter WAV files for manual listening.

---

## VAE Roundtrip diagnostic

Use `Foley VAE Roundtrip` to check DAC codec quality on your audio:

```
HunyuanDependenciesLoader → Foley VAE Roundtrip (audio input) → Preview Audio
```

This encodes audio through DAC and decodes it back. The output reveals the quality ceiling — your LoRA can never produce audio better than what DAC can reconstruct. If roundtrip quality is poor, the dataset pipeline (HF Smoother, Spectral Matcher) can help bring the audio into DAC's comfort zone.

---

## Output files

```
lora_output/my_sound/
    adapter_step00500.pt      ← step checkpoint (includes optimizer state for resume)
    adapter_step01000.pt
    ...
    adapter_final.pt          ← final adapter with embedded metadata
    meta.json                 ← human-readable training config
    samples/
        step_00500.wav        ← eval audio sample at each checkpoint
        step_01000.wav
    loss_raw.png              ← raw loss curve
    loss_smoothed.png         ← EMA-smoothed loss curve
```

`adapter_final.pt` format:
```python
{
    "state_dict": { "triple_blocks.0.audio_self_attn_qkv.lora_A": ..., ... },
    "meta": {
        "target": "all_attn_mlp",
        "rank": 64,
        "alpha": 64.0,
        "steps": 3000,
        "init_mode": "standard",
        "use_rslora": false,
        ...
    }
}
```

Step checkpoints additionally contain `optimizer` and `scheduler` state for resuming via `resume_from`.

---

## Model selection

Use full-precision `.pth` weights from Tencent for maximum training quality:

| Component | Recommended | Alternative |
|---|---|---|
| **Transformer** | `hunyuanvideo_foley.pth` (Tencent, full precision) | `hunyuanvideo_foley.safetensors` (phazei, also full precision) |
| **VAE (DAC)** | `vae_128d_48k.pth` (Tencent, full precision) | `vae_128d_48k_fp16.safetensors` (phazei, fp16 — loses precision) |
| **Synchformer** | `synchformer_state_dict.pth` (Tencent, full precision) | `synchformer_state_dict_fp16.safetensors` (phazei, fp16) |

The fp16 VAE and Synchformer lose precision that matters during training. With sufficient VRAM, always use the full-precision versions. The fp8 transformer variants are for inference only — do not train with fp8.

---

## VRAM estimates (96 GB target)

All configurations fit comfortably within 96 GB with no need for gradient checkpointing or model offloading.

| Config | Estimated VRAM |
|---|---|
| bf16 + rank 64 + all_attn_mlp + batch 8 | ~30 GB |
| bf16 + rank 64 + all_attn_mlp + batch 16 | ~45 GB |
| bf16 + rank 64 + all_attn_mlp + batch 32 | ~70 GB |
| bf16 + rank 128 + all_attn_mlp + batch 16 | ~55 GB |

---

## Recommended defaults (96 GB)

Optimized for maximum quality on high-VRAM hardware:

| Parameter | Value | Rationale |
|---|---|---|
| target | `all_attn_mlp` | Maximum adaptation capacity (234 layers) |
| rank | 64 | High capacity for 128D latent space |
| alpha | 64 | 1:1 ratio (HunyuanVideo convention) |
| lr | 1e-4 | Proven across AudioLDM and SelVA |
| batch_size | 8 | Good gradient quality without excess |
| timestep_mode | `logit_normal` | 0.2-0.3 dB lower loss floor |
| precision | bf16 | Full quality, Ampere+ standard |
| warmup_steps | 100 | Stable start |
| save_every | 500 | Regular checkpoints + eval samples |

---

## Troubleshooting

**`No .npz files found in ...`**
The `data_dir` path is wrong or no features were extracted yet. Run `Foley Feature Extractor` first.

**`No audio file found for clip.npz`**
Place a `.wav`, `.flac`, `.ogg`, or `.aiff` file with the exact same stem next to the `.npz`.

**The sound is audible but there is white noise on top**
Lower adapter strength to 0.6-0.7 in Foley LoRA Loader. Also try lowering CFG scale in the Sampler. More clips and more steps will reduce this.

**LoRA appears to have no effect**
Make sure the Foley LoRA Loader output is wired to the **Sampler**, not the Feature Extractor.

**Loss does not decrease**
- Increase batch_size for more stable gradients
- Try a higher learning rate (2e-4) or check warmup isn't too long
- Verify audio files are clean and contain the target sound
- Check that `.npz` features were extracted with a relevant prompt

**Loss explodes or NaN**
- Lower the learning rate (5e-5)
- Check audio is not clipped — run through Foley Dataset Inspector first

**Loss plateaus early (above 0.7)**
Dataset is the bottleneck. Add more clips with diverse recordings.

**torch.cat error on batching**
All clips must have the same duration. The trainer enforces this by trimming to the shortest clip's latent length. If clips have very different durations, pad/trim them to a fixed length before feature extraction.

---

## Differences from SelVA LoRA Training

If you're coming from SelVA's LoRA pipeline, these are the key differences:

| Aspect | SelVA | Foley |
|---|---|---|
| Model size | ~2 GB (large_44k) | ~10.3 GB |
| VAE | Mel-spectrogram + BigVGAN vocoder | DAC single-stage neural codec |
| Latent dim | 40D | 128D |
| Sample rate | 44.1 kHz mono | 48 kHz (mono for training) |
| Default rank | 16 | 64 (128D latent needs more capacity) |
| Default target | `attn.qkv` | `all_attn_mlp` (4 presets available) |
| Feature extractors | CLIP + TextSynchformer + T5 | SigLIP2 + Synchformer + CLAP |
| HF Smoother | 12 kHz cutoff, 0.7 blend | 16 kHz cutoff, 0.5 blend (DAC handles HF better) |
| Spectral Matcher | Hardcoded VAE distribution stats | Reference-based (from DAC roundtrip) |
| CLI training | `train_lora.py` | ComfyUI nodes only |
| Timestep default | `uniform` | `logit_normal` |
