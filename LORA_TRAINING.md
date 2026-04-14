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

Use **[8-cut](https://github.com/ethanfel/8-cut)** to cut exactly 8-second clips from source footage. It provides a visual scrubber, sound annotation (label + category), and exports labeled clips with a `dataset.json` manifest ready for the pipeline below.

Alternatively, gather paired video and audio clips manually.

| Guideline | Details |
|---|---|
| **Format** | WAV or FLAC preferred. Avoid MP3 — lossy compression degrades training |
| **Aspect ratio** | 16:9 landscape preferred, but portrait and square also work fine |
| **Resolution** | >= 480p sufficient (downscaled to 512px and 224px internally) |
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
| **150-300+ clips** | Sounds the model has never encountered | Required to avoid overfitting |

> **Quality over quantity, but quantity matters.** A large dataset of diverse, high-quality clips will always outperform a small one. In testing, 399 unique clips produced dramatically better results than 47. Collect as many unique clips as you can.

### 1.3 Optimize audio quality (recommended)

Chain the Foley Dataset Pipeline nodes in ComfyUI to clean and normalize your audio before feature extraction:

```
Foley Dataset Loader (folder of raw audio)
  -> Foley Dataset Resampler (48000 Hz)
  -> Foley Dataset LUFS Normalizer (-23 LUFS, -1 dBTP)
  -> Foley Dataset Compressor (ratio 2.5:1, mix 0.4)
  -> Foley Dataset HF Smoother (cutoff 16000 Hz, blend 0.5)
  -> Foley Dataset Inspector (remove clipped/noisy/silent clips)
  -> Foley Dataset Saver (output to cleaned folder as 24-bit FLAC)
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

### 1.5 Augmentation — use with caution

> **Warning:** Sweep testing showed that augmented duplicates (same video, slightly different audio) can *hurt* training quality. The model learns averaged spectral patterns instead of following video cues, producing mechanical-sounding output.
>
> In one test, removing augmented copies (99 clips -> 47 unique clips) improved spectral convergence from 0.99 to 0.47 and MCD from 17.9 to 6.5. The improvement was dramatic.

**Rule of thumb:** More unique clips is always better than augmenting existing ones. Only use augmentation if you truly cannot collect more source material, and keep it subtle:

```
Foley Dataset Augmenter
    variants_per_clip: 1       (keep it minimal)
    gain_range_db: 2.0
    pitch_range_semitones: 0.3
    time_stretch_range: 0.05
    keep_originals: true
```

---

## Step 2 — Extract features

For each video+audio pair, run the `Foley Feature Extractor` node in ComfyUI:

```
HunyuanDependenciesLoader (vae_128d_48k.pth, synchformer_state_dict.pth)
  -> Foley Feature Extractor
      image: video frames (from Load Video node)
      audio: paired audio (cleaned from Step 1)
      prompt: "description of the sound"
      frame_rate: source video FPS
      duration: 8.0
      cache_dir: /path/to/features/
      name: "gunshot"
```

> **Important:** Always set `duration=8.0` explicitly. The auto-detect (`total_frames / frame_rate`) gives wrong results when the video fps doesn't match the `frame_rate` input (e.g., 30fps video with frame_rate=25 computes 9.6s instead of 8.0s). This causes misaligned visual features that break audio-video sync during training.

Queue once per clip. Each execution saves one `.npz` + `.wav` pair with auto-incremented names (`gunshot_001.npz`, `gunshot_002.npz`, etc.).

### What gets cached

Each `.npz` contains:
- `clip_features` — SigLIP2 visual features `[1, N_clip, 768]` at 8fps
- `sync_features` — Synchformer sync features `[1, N_sync, 768]` at 25fps
- `text_embedding` — CLAP text features `[1, seq_len, 768]`
- `prompt` — the text description
- `duration` — clip duration in seconds
- `fps` — source frame rate

A matching audio file (resampled to 48kHz) is saved alongside.

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
- Use action + texture — "wet sucking and slurping", "heavy boots on wood"
- Add acoustic modifiers — "rhythmic", "loud", "close", "deep bass"
- Keep prompts consistent across all clips for the same sound class
- Avoid negations (`no background noise`) — use positive descriptions instead
- Stay concise — 77 token limit in CLAP, shorter is better

**The prompt affects inference more than training.** During training, the model learns audio patterns from all three conditioning streams (visual, sync, text). At inference, CFG amplifies the text guidance, so the prompt steers generation.

### Validation clip (recommended)

Set aside one clip as a validation sample — ideally a rejected clip from the same domain that you didn't include in the training set. Save its `.npz` path as the `eval_npz` parameter in the trainer. This generates separate validation audio at each checkpoint, letting you detect overfitting (training eval improves while validation plateaus or degrades).

### Directory structure

After feature extraction:

```
dataset/my_sound/
    gunshot_001.npz    <- features from Foley Feature Extractor
    gunshot_001.wav    <- paired audio (48kHz)
    gunshot_002.npz
    gunshot_002.wav
    ...
    val_clip.npz       <- held-out validation clip
    val_clip.wav
```

---

## Step 3 — Train

Connect the nodes in ComfyUI:

```
HunyuanModelLoader (hunyuanvideo_foley.pth)
HunyuanDependenciesLoader (vae_128d_48k.pth, synchformer_state_dict.pth)
  -> Foley LoRA Trainer
      data_dir: /path/to/features/
      output_dir: /path/to/lora_output/
      eval_npz: /path/to/features/val_clip.npz  (optional but recommended)
      (hyperparameters — see below)
```

The trainer:
1. Loads all `.npz` features and DAC-encodes the paired audio to latents.
2. Applies LoRA layers to the transformer, freezes base weights.
3. Runs the flow matching training loop.
4. Saves checkpoints, eval audio samples, spectrograms, and loss curves at intervals.
5. Returns the model with LoRA active for immediate inference.

### Hyperparameters

#### Target layers

Controls which transformer layers get LoRA adapters. More layers = more capacity but higher VRAM and overfitting risk.

| Preset | Layers | Total adapted | Use case |
|---|---|---|---|
| `audio_attn` | Audio self-attention QKV + proj | 36 | Minimal adaptation, fine details |
| `audio_cross` | Above + audio cross-attention + text cross-KV | 90 | Good for text-conditioned sounds |
| `all_attn` | Above + visual conditioning attention | 162 | Full attention adaptation |
| `all_attn_mlp` | Above + MLP layers (fc1/fc2) | 234 | Maximum capacity (recommended) |

**Default:** `all_attn_mlp` — with sufficient VRAM there's no reason to limit capacity.

#### Rank

| Rank | Use case | Trainable params (all_attn_mlp) |
|---|---|---|
| 16 | Fine details on a known sound | ~2.4M |
| 32 | Good balance for small datasets | ~4.8M |
| 64 | Moderate capacity | ~9.6M |
| 128 | **Recommended** — proven best across all sweeps | ~19.2M |

**Default:** 128. Foley's 128D latent space benefits from high rank. Sweep testing showed rank 128 with curriculum timestep sampling consistently outperformed rank 64 across all metrics.

#### Learning rate

`1e-4` constant is the recommended default. If training is unstable (loss spikes in the first 200 steps), try `5e-5`. Half LR (5e-5) reaches the same destination at double the compute — no advantage.

Warmup (default 100 steps) ramps the LR from 0 to avoid instability at the start.

> **Cosine decay hurts.** Sweep testing showed cosine LR decays too fast — the model stops learning by step 6k, wasting the remaining steps at near-zero LR. Use `constant` schedule.

#### Batch size

| Batch size | VRAM (bf16, rank 128, all_attn_mlp) | Notes |
|---|---|---|
| 1 | ~22 GB | Noisy gradients, slow |
| 4 | ~25 GB | Reasonable starting point |
| 8 | ~30 GB | **Recommended** — stable gradients |
| 16 | ~45 GB | Better convergence on larger datasets |
| 32 | ~70 GB | Best gradient quality — fits on 96 GB |

Higher batch size gives smoother loss curves and faster convergence. With 96 GB, prefer larger batches over more steps.

#### Steps

| Dataset size | Recommended steps | Notes |
|---|---|---|
| 10-30 clips | 3000-5000 | |
| 30-60 clips | 5000-10000 | |
| 60-150 clips | 10000-15000 | |
| 150-400 clips | 13000-15000 | Best checkpoint often at 13-14k |

> **Scalar metrics and perceptual quality diverge.** In a 400-clip sweep, all metrics (SC, MCD, PBC) kept improving through 25k steps, but perceptual testing showed step 13-14k produced the best-sounding output. Later checkpoints lost subtle ambient details — faint breath, room tone, quiet textures — that metrics don't capture. Always validate top candidates by listening to the eval samples.

#### Timestep sampling

Controls how training timesteps are sampled. Flow matching trains by interpolating between noise and target at random timesteps.

| Mode | Description | When to use |
|---|---|---|
| `uniform` | All timesteps equally | Safe baseline |
| `logit_normal` | Concentrates near t=0.5 via `sigmoid(N(0, sigma))` | Good default, 0.2-0.3 dB lower loss floor |
| `curriculum` | logit_normal first, then switches to uniform | **Recommended** — best results across all sweeps |

**Default:** `curriculum` with `curriculum_switch=0.6`.

Curriculum sampling starts with logit_normal (focusing on "easy" mid-range timesteps) then switches to uniform at 60% of total steps (exposing the model to all timesteps for fine detail). This prevents the mild overfitting that hits constant-distribution approaches in long training runs.

In sweep testing, curriculum was the first config to break SC < 1.0 and was still improving at 10k steps while constant LR had already peaked and started regressing.

#### Advanced options

| Parameter | Default | Description |
|---|---|---|
| `alpha` | 128 | LoRA scaling factor. 1:1 ratio with rank is the HunyuanVideo convention |
| `grad_accum` | 1 | Gradient accumulation steps. Use when batch size alone doesn't fit |
| `init_mode` | `standard` | `standard` (Kaiming A, zero B) or `pissa` (SVD-based) |
| `use_rslora` | false | Rank-stabilized scaling: `alpha/sqrt(rank)` instead of `alpha/rank` |
| `lora_dropout` | 0.0 | Dropout on the LoRA path. 0.05-0.1 helps on small datasets (<20 clips) |
| `lora_plus_ratio` | 1.0 | B-matrix LR multiplier (see warning below) |
| `schedule_type` | `constant` | LR schedule after warmup. Use `constant` (see warning below) |
| `latent_mixup_alpha` | 0.0 | Beta-distribution latent interpolation for augmentation |
| `latent_noise_sigma` | 0.0 | Additive Gaussian noise on target latents for regularization |
| `precision` | `bf16` | `bf16` (Ampere+), `fp16` (older GPUs), `fp32` (debug only) |
| `gradient_checkpointing` | false | Recompute activations to save VRAM (~3-5 GB, ~25% slower) |
| `blocks_to_swap` | 0 | Offload N transformer blocks to CPU (0-54, uses prefetch=2) |

> **LoRA+ warning:** Setting `lora_plus_ratio=16.0` enables LoRA+ (higher LR for B matrices). Sweep testing showed this overfits on small-to-medium datasets — loss drops below the noise floor (~0.5), producing mechanical sound on some inputs and pure noise on others. Standard LR (ratio 1.0) generalizes better.

> **Cosine schedule warning:** `schedule_type=cosine` decays the learning rate to zero. Testing showed this wastes the final ~40% of training steps at near-zero LR. Use `constant`.

---

## Step 4 — Select the best checkpoint

Training produces checkpoints at regular intervals. Don't just use `adapter_final.pt` — the best checkpoint is often earlier than the final one.

### Reading the metrics

Each checkpoint records spectral metrics in `metrics_history.json`:

| Metric | What it measures | Better |
|---|---|---|
| **Spectral Convergence (SC)** | Normalized spectral distance vs reference | Lower |
| **Mel Cepstral Distortion (MCD)** | Perceptual distance in mel-cepstral space | Lower |
| **Per-Band Correlation (PBC)** | Avg correlation across 80 mel bands vs ref | Higher (max 1.0) |
| **Log Spectral Distance (LSD)** | dB-scale spectral envelope error vs ref | Lower |
| **Loss (MSE)** | Raw training loss | See note below |

**Priority for checkpoint selection:**
1. Lowest SC (overall spectral fidelity)
2. Lowest MCD (perceptual quality)
3. Highest PBC (temporal tracking accuracy)
4. **Then listen to the eval samples** — metrics miss subtle ambient detail

### Understanding the loss curve

The raw MSE loss appears flat (~1.3-1.5) throughout training. This is **normal** for flow matching — the loss is dominated by the irreducible stochastic variance of the velocity target. The actual learning signal is a tiny fraction of the total. Do not use loss alone to judge training progress.

### Overfitting indicators

- Loss dropping well below the noise floor (< 1.3): model is fitting noise
- SC/MCD improving on training eval but validation eval degrades: memorization
- Horizontal spectral banding in eval spectrograms: averaged patterns from small dataset
- PBC > 0.9 on training eval: too close to reference, won't generalize
- Temporal variance collapsing: model producing flat/static audio

### Healthy training indicators

- Loss stable around 1.4-1.5
- SC, MCD, LSD all trending down together
- PBC trending up but staying below 0.85
- Validation and training metrics moving in the same direction
- Temporal variance close to reference value (dynamic, not flat)

---

## Step 5 — Use the adapter

Connect `Foley LoRA Loader` between the model loader and the sampler:

```
HunyuanModelLoader (hunyuanvideo_foley.pth)
  -> Foley LoRA Loader
      adapter_path: /path/to/lora_output/adapter_step14000.pt
      strength: 1.0
  -> HunyuanFoleySampler (normal inference)
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
  "eval_npz": "/path/to/validation_clip.npz",
  "base": { "rank": 128, "lr": 1e-4, "steps": 15000, "target": "all_attn_mlp",
            "timestep_mode": "curriculum", "curriculum_switch": 0.6 },
  "experiments": [
    {"id": "baseline_r128"},
    {"id": "r64", "rank": 64},
    {"id": "r128_20k", "steps": 20000}
  ]
}
```

The scheduler loads the dataset once, runs each experiment sequentially, and produces a `loss_comparison.png` overlay chart. Completed experiments are skipped on resume. Drop a `skip_current.flag` file in the output root to abort the current experiment and move to the next.

**Validation (`eval_npz`):** Optional path to an NPZ file outside the training dataset, with a matching audio file alongside it (same stem, e.g., `val_clip.npz` + `val_clip.wav`). When set, generates validation audio at each checkpoint so you can detect overfitting — training eval improves while validation plateaus or degrades.

### VRAM offload options (per-experiment)

| Option | JSON key | Default | Effect |
|---|---|---|---|
| Gradient checkpointing | `gradient_checkpointing` | false | Saves ~3-5 GB VRAM, ~25% slower |
| Block swap | `blocks_to_swap` | 0 | Offloads N of 54 blocks to CPU (prefetch=2) |

These can be set in `base` (applies to all experiments) or per-experiment overrides.

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
    {"id": "r128_14k", "path": "/path/to/adapter_step14000.pt"}
  ]
}
```

Computes spectral metrics for each adapter and outputs a `metric_comparison.png` bar chart and per-adapter WAV files for manual listening.

---

## VAE Roundtrip diagnostic

Use `Foley VAE Roundtrip` to check DAC codec quality on your audio:

```
HunyuanDependenciesLoader -> Foley VAE Roundtrip (audio input) -> Preview Audio
```

This encodes audio through DAC and decodes it back. The output reveals the quality ceiling — your LoRA can never produce audio better than what DAC can reconstruct. If roundtrip quality is poor, the dataset pipeline (HF Smoother, Spectral Matcher) can help bring the audio into DAC's comfort zone.

---

## Output files

```
lora_output/my_sound/
    adapter_step00500.pt      <- step checkpoint (includes optimizer state for resume)
    adapter_step01000.pt
    ...
    adapter_final.pt          <- final adapter with embedded metadata
    meta.json                 <- human-readable training config
    metrics_history.json      <- spectral metrics at each checkpoint
    loss.png                  <- loss curve
    samples/
        step_00000.wav        <- pre-training baseline
        step_00000.png        <- spectrogram
        step_00500.wav
        step_00500.png
        ...
        val_step_00000.wav    <- validation samples (if eval_npz set)
        val_step_00500.wav
        val_reference.png     <- ground-truth validation spectrogram
```

`adapter_final.pt` format:
```python
{
    "state_dict": { "triple_blocks.0.audio_self_attn_qkv.lora_A": ..., ... },
    "meta": {
        "target": "all_attn_mlp",
        "rank": 128,
        "alpha": 128.0,
        "steps": 15000,
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

## VRAM estimates

| Config | Estimated VRAM | Target GPU |
|---|---|---|
| bf16 + rank 128 + batch 8, no offload | ~18-20 GB | 24+ GB (4090, A5000) |
| + gradient checkpointing | ~13-15 GB | 16 GB (4080, A4000) |
| + grad ckpt + 20 blocks swapped | ~10-12 GB | 12 GB (3060 12GB) |
| + grad ckpt + 40 blocks swapped, batch 2 | ~8-9 GB | 10 GB |
| bf16 + rank 128 + batch 16, no offload | ~45 GB | 48+ GB |
| bf16 + rank 128 + batch 32, no offload | ~70 GB | 96 GB |

High-VRAM systems (48+ GB) need no offloading at all.

### 24 GB GPUs (RTX 4090, A5000)

Use `batch_size: 1` with `grad_accum: 8` for effective batch 8, plus gradient checkpointing and block swapping:

```json
{
  "name": "my_lora",
  "dataset_json": "/path/to/features/dataset.json",
  "output_root": "/path/to/output",
  "base": {
    "target": "all_attn_mlp",
    "rank": 128,
    "alpha": 128,
    "lr": 1e-4,
    "steps": 15000,
    "batch_size": 1,
    "grad_accum": 8,
    "warmup_steps": 100,
    "save_every": 1000,
    "timestep_mode": "curriculum",
    "precision": "bf16",
    "seed": 42,
    "logit_normal_sigma": 1.0,
    "curriculum_switch": 0.6,
    "init_mode": "standard",
    "use_rslora": false,
    "lora_dropout": 0.0,
    "lora_plus_ratio": 1.0,
    "schedule_type": "constant",
    "latent_mixup_alpha": 0.0,
    "latent_noise_sigma": 0.0,
    "gradient_checkpointing": true,
    "blocks_to_swap": 30,
    "resume_from": ""
  },
  "experiments": [
    {"id": "baseline_r128"}
  ]
}
```

Key settings:
- **`gradient_checkpointing: true`** — saves ~3-5 GB VRAM by recomputing activations (~25% slower)
- **`blocks_to_swap: 30`** — offloads 30 of 57 transformer blocks to CPU RAM, frees ~8 GB VRAM
- **`batch_size: 1` + `grad_accum: 8`** — same gradient quality as batch 8, fits in VRAM

If you still run out of VRAM, increase `blocks_to_swap` to 40 or lower `rank` to 64.

---

## Recommended defaults

Optimized based on five rounds of sweep testing:

| Parameter | Value | Rationale |
|---|---|---|
| target | `all_attn_mlp` | Maximum adaptation capacity (234 layers) |
| rank | 128 | Best across all sweeps for 128D latent space |
| alpha | 128 | 1:1 ratio with rank (HunyuanVideo convention) |
| lr | 1e-4 | Proven across AudioLDM and SelVA |
| schedule_type | `constant` | Cosine decays too fast, wastes later steps |
| batch_size | 8 | Good gradient quality without excess |
| timestep_mode | `curriculum` | Best config — first to break SC < 1.0, no late-stage regression |
| curriculum_switch | 0.6 | Transition from logit_normal to uniform at 60% |
| steps | 15000 | Select best checkpoint by metrics + listening |
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
Check that the LoRA Loader output is connected to the Sampler's model input and that adapter strength is high enough (0.8-1.0).

**Loss does not decrease**
This is normal for FoleyTune — the flow matching loss typically stays flat or fluctuates around a value rather than steadily decreasing. Judge training progress by listening to eval samples at checkpoints, not by watching the loss curve.

**Loss explodes or NaN**
- Lower the learning rate (5e-5)
- Check audio is not clipped — run through Foley Dataset Inspector first

**Loss plateaus early (above 0.7)**
Dataset is the bottleneck. Add more clips with diverse recordings.

**Metrics keep improving but audio sounds worse**
Scalar metrics diverge from perceptual quality after extended training. The model over-specializes on dominant spectral features at the cost of low-energy ambient sounds (breath, room tone, quiet textures). Pick the checkpoint that sounds best, not the one with the best numbers.

**torch.cat error on batching**
All clips must have the same duration. The trainer enforces this by trimming to the shortest clip's latent length. If clips have very different durations, pad/trim them to a fixed length before feature extraction.

