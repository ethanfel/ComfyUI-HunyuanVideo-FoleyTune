# LoRA Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add LoRA fine-tuning, evaluation, and sweep orchestration to HunyuanVideo-Foley for video-to-audio generation.

**Architecture:** Port SelVA's proven LoRA pipeline (from `feature/lora-timestep-sampling` branch) to Foley's two-stream transformer + DAC codec. Six new nodes in a separate `nodes_lora.py`, with core logic in `lora/` package. Flow matching loss is identical; main adaptation is feature extractors and layer targeting.

**Tech Stack:** PyTorch, huggingface_hub, ComfyUI node API, DAC neural codec, SigLIP2/Synchformer/CLAP feature extractors.

---

### Task 1: Create `lora/lora.py` — LoRA layer primitives

**Files:**
- Create: `lora/__init__.py`
- Create: `lora/lora.py`

**Step 1: Create the lora package**

```python
# lora/__init__.py
```

Empty init, we'll import from submodules explicitly.

**Step 2: Write `lora/lora.py`**

Port directly from SelVA's `selva_core/model/lora.py`. This is model-agnostic code that works unchanged. Contains:

- `LoRALinear(nn.Module)` — wraps `nn.Linear` with frozen base + trainable low-rank A/B
  - Standard init (Kaiming A, zero B) and PiSSA init (SVD-based)
  - Standard scaling (`alpha/rank`) and rsLoRA scaling (`alpha/sqrt(rank)`)
  - Dropout on LoRA path
- `apply_lora(model, rank, alpha, target_suffixes, dropout, init_mode, use_rslora)` — in-place replacement of matching `nn.Linear` layers
- `get_lora_state_dict(model)` — extract only LoRA params
- `get_lora_and_base_state_dict(model)` — LoRA + base weights (for PiSSA)
- `load_lora(model, state_dict)` — load LoRA weights into already-wrapped model
- `spectral_surgery(model, calibration_fn, n_calibration, policy)` — post-training SVD reweighting

Only change from SelVA: update the default `target_suffixes` to Foley layer names.

```python
# Default targets for Foley's TwoStreamCABlock
FOLEY_TARGET_PRESETS = {
    "audio_attn": (
        "audio_self_attn_qkv",
        "audio_self_proj",
    ),
    "audio_cross": (
        "audio_self_attn_qkv",
        "audio_self_proj",
        "audio_cross_q",
        "audio_cross_proj",
        "text_cross_kv",
    ),
    "all_attn": (
        "audio_self_attn_qkv",
        "audio_self_proj",
        "audio_cross_q",
        "audio_cross_proj",
        "text_cross_kv",
        "v_cond_attn_qkv",
        "v_cond_self_proj",
        "v_cond_cross_q",
        "v_cond_cross_proj",
    ),
    "all_attn_mlp": (
        "audio_self_attn_qkv",
        "audio_self_proj",
        "audio_cross_q",
        "audio_cross_proj",
        "text_cross_kv",
        "v_cond_attn_qkv",
        "v_cond_self_proj",
        "v_cond_cross_q",
        "v_cond_cross_proj",
        "audio_mlp.fc1",
        "audio_mlp.fc2",
        "v_cond_mlp.fc1",
        "v_cond_mlp.fc2",
    ),
}
```

**Step 3: Commit**

```bash
git add lora/
git commit -m "feat: add LoRA layer primitives (LoRALinear, apply/load/save)"
```

---

### Task 2: Create `lora/spectral_metrics.py` — audio analysis utilities

**Files:**
- Create: `lora/spectral_metrics.py`

**Step 1: Write spectral metrics module**

Port from SelVA trainer's `_spectral_metrics` and `_reference_metrics`. These are pure numpy/torch functions with no model dependency.

```python
"""Spectral analysis utilities for evaluating audio generation quality."""

import numpy as np
import torch


def spectral_metrics(wav: np.ndarray, sr: int) -> dict:
    """Compute spectral descriptors for a mono waveform.
    
    Args:
        wav: 1D numpy array of audio samples
        sr: sample rate
    
    Returns:
        dict with keys: hf_energy_ratio, spectral_centroid_hz,
        spectral_rolloff_hz, spectral_flatness, temporal_variance
    """
    # STFT: 2048-point FFT, 512-sample hop, Hann window
    n_fft = 2048
    hop = 512
    window = np.hanning(n_fft)
    
    # Frame the signal
    n_frames = 1 + (len(wav) - n_fft) // hop
    if n_frames < 1:
        return {
            "hf_energy_ratio": 0.0,
            "spectral_centroid_hz": 0.0,
            "spectral_rolloff_hz": 0.0,
            "spectral_flatness": 0.0,
            "temporal_variance": 0.0,
        }
    
    frames = np.stack([wav[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    spec = np.abs(np.fft.rfft(frames, n=n_fft))
    power = spec ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    
    total_energy = power.sum()
    if total_energy < 1e-10:
        return {k: 0.0 for k in ["hf_energy_ratio", "spectral_centroid_hz",
                                   "spectral_rolloff_hz", "spectral_flatness",
                                   "temporal_variance"]}
    
    # HF energy ratio (>4kHz)
    hf_mask = freqs > 4000
    hf_energy_ratio = float(power[:, hf_mask].sum() / total_energy)
    
    # Spectral centroid
    mean_power = power.mean(axis=0)
    centroid = float(np.sum(freqs * mean_power) / (mean_power.sum() + 1e-10))
    
    # Spectral rolloff (85%)
    cumsum = np.cumsum(mean_power)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    
    # Spectral flatness (Wiener entropy)
    mp = mean_power + 1e-10
    geo_mean = np.exp(np.mean(np.log(mp)))
    arith_mean = np.mean(mp)
    flatness = float(geo_mean / (arith_mean + 1e-10))
    
    # Temporal variance
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    mean_rms = frame_rms.mean()
    temporal_var = float(frame_rms.std() / (mean_rms + 1e-10))
    
    return {
        "hf_energy_ratio": hf_energy_ratio,
        "spectral_centroid_hz": centroid,
        "spectral_rolloff_hz": rolloff,
        "spectral_flatness": flatness,
        "temporal_variance": temporal_var,
    }


def reference_metrics(gen_wav: np.ndarray, ref_wav: np.ndarray, sr: int) -> dict:
    """Compute distance metrics between generated and reference audio.
    
    Returns:
        dict with keys: log_spectral_distance_db, mel_cepstral_distortion,
        per_band_correlation
    """
    n_fft = 2048
    hop = 512
    window = np.hanning(n_fft)
    
    min_len = min(len(gen_wav), len(ref_wav))
    gen_wav = gen_wav[:min_len]
    ref_wav = ref_wav[:min_len]
    
    n_frames = 1 + (min_len - n_fft) // hop
    if n_frames < 1:
        return {"log_spectral_distance_db": 0.0, "mel_cepstral_distortion": 0.0,
                "per_band_correlation": 0.0}
    
    gen_frames = np.stack([gen_wav[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    ref_frames = np.stack([ref_wav[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    
    gen_spec = np.abs(np.fft.rfft(gen_frames, n=n_fft)) + 1e-10
    ref_spec = np.abs(np.fft.rfft(ref_frames, n=n_fft)) + 1e-10
    
    # Log spectral distance
    lsd = float(np.sqrt(np.mean((20 * np.log10(gen_spec) - 20 * np.log10(ref_spec)) ** 2)))
    
    # Mel cepstral distortion (simplified: log-mel space L2)
    n_mels = 80
    mel_fmin, mel_fmax = 0, sr // 2
    mel_points = np.linspace(2595 * np.log10(1 + mel_fmin / 700),
                              2595 * np.log10(1 + mel_fmax / 700), n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    
    mel_fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, mid, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        mel_fb[i] = np.where((freqs >= lo) & (freqs <= mid),
                              (freqs - lo) / (mid - lo + 1e-10), 0)
        mel_fb[i] += np.where((freqs > mid) & (freqs <= hi),
                               (hi - freqs) / (hi - mid + 1e-10), 0)
    
    gen_mel = np.log(gen_spec @ mel_fb.T + 1e-10)
    ref_mel = np.log(ref_spec @ mel_fb.T + 1e-10)
    mcd = float(np.sqrt(np.mean((gen_mel - ref_mel) ** 2)) * (10 / np.log(10)))
    
    # Per-band correlation
    cors = []
    for b in range(n_mels):
        g, r = gen_mel[:, b], ref_mel[:, b]
        if g.std() < 1e-8 or r.std() < 1e-8:
            cors.append(0.0)
        else:
            cors.append(float(np.corrcoef(g, r)[0, 1]))
    pbc = float(np.mean(cors))
    
    return {
        "log_spectral_distance_db": lsd,
        "mel_cepstral_distortion": mcd,
        "per_band_correlation": pbc,
    }
```

**Step 2: Commit**

```bash
git add lora/spectral_metrics.py
git commit -m "feat: add spectral metrics for audio quality evaluation"
```

---

### Task 3: Create `lora/train.py` — training loop and dataset loading

**Files:**
- Create: `lora/train.py`

**Step 1: Write the training core**

This contains:
- `prepare_dataset(data_dir, deps, device)` — load .npz + audio, DAC-encode audio to latents
- `sample_timesteps(batch_size, mode, device, dtype, sigma, curriculum_switch, step, total_steps)` — uniform/logit_normal/curriculum sampling
- `train_step(model, batch, timestep_mode, ...)` — single forward/backward pass with flow matching loss
- `generate_eval_sample(model, deps, dataset, cfg, clip_idx, steps, seed, device)` — inference for checkpointing
- `save_loss_curve(losses, path, start_step)` — matplotlib loss visualization
- `save_checkpoint(model, optimizer, scheduler, step, meta, path)` — checkpoint saving

Key adaptations from SelVA:
- DAC encoding: `dac_model.encoder(audio_waveform)` for latents (no separate vocoder)
- Feature loading: .npz contains `clip_features` (SigLIP2), `sync_features` (Synchformer), `text_embedding` (CLAP)
- Forward pass: `model(x=xt, t=t_expand, cond=text_feat, clip_feat=clip_feat, sync_feat=sync_feat)["x"]`
- Loss: `MSE(v_pred, x1 - x0)` where x1=target latent, x0=noise

```python
"""Training loop and dataset loading for Foley LoRA fine-tuning."""

import os
import gc
import json
import time
import copy
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from loguru import logger

from .lora import apply_lora, get_lora_state_dict, load_lora, spectral_surgery, FOLEY_TARGET_PRESETS
from .spectral_metrics import spectral_metrics


# ── Dataset ──────────────────────────────────────────────────────────────────

def prepare_dataset(data_dir: str, dac_model, device, dtype=torch.bfloat16):
    """Load .npz feature caches + audio files, encode audio via DAC.
    
    Returns list of dicts with keys:
        latents: [1, 128, T] DAC-encoded audio latent (target x1)
        clip_features: [1, N_clip, 768] SigLIP2 visual features
        sync_features: [1, N_sync, 768] Synchformer sync features
        text_embedding: [1, D] CLAP text embedding
        prompt: str
        name: str (stem of .npz file)
    """
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    
    audio_exts = (".wav", ".flac", ".ogg", ".aiff", ".aif")
    dataset = []
    
    # Move DAC encoder to device for encoding
    dac_model.to(device)
    
    for npz_path in npz_files:
        stem = npz_path.stem
        # Find matching audio file
        audio_path = None
        for ext in audio_exts:
            candidate = data_dir / f"{stem}{ext}"
            if candidate.exists():
                audio_path = candidate
                break
        if audio_path is None:
            logger.warning(f"No audio file found for {stem}, skipping")
            continue
        
        # Load features from .npz
        data = np.load(str(npz_path), allow_pickle=True)
        clip_features = torch.from_numpy(data["clip_features"]).float()
        sync_features = torch.from_numpy(data["sync_features"]).float()
        text_embedding = torch.from_numpy(data["text_embedding"]).float()
        prompt = str(data.get("prompt", stem))
        duration = float(data.get("duration", 0))
        
        # Load and encode audio
        waveform, sr = torchaudio.load(str(audio_path))
        # Resample to 48kHz if needed
        if sr != 48000:
            waveform = torchaudio.functional.resample(waveform, sr, 48000)
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Pad/trim to match duration from features (or 8 seconds default)
        target_samples = int((duration if duration > 0 else 8.0) * 48000)
        if waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]
        elif waveform.shape[1] < target_samples:
            waveform = F.pad(waveform, (0, target_samples - waveform.shape[1]))
        
        # DAC encode: [1, 1, samples] -> latents
        with torch.no_grad():
            audio_input = waveform.unsqueeze(0).to(device=device, dtype=dtype)
            z, _, _, _, _ = dac_model.encode(audio_input)
            latents = z.cpu().float()  # [1, 128, T]
        
        dataset.append({
            "latents": latents,
            "clip_features": clip_features,
            "sync_features": sync_features,
            "text_embedding": text_embedding,
            "prompt": prompt,
            "name": stem,
        })
    
    # Offload DAC encoder back to CPU
    dac_model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info(f"Prepared dataset: {len(dataset)} clips from {data_dir}")
    return dataset


# ── Timestep sampling ────────────────────────────────────────────────────────

def sample_timesteps(batch_size, mode, device, dtype,
                     sigma=1.0, curriculum_switch=0.6,
                     step=0, start_step=0, total_steps=1000):
    """Sample timesteps t in [0, 1] for flow matching training."""
    if mode == "logit_normal":
        u = torch.randn(batch_size, device=device, dtype=dtype) * sigma
        t = torch.sigmoid(u)
    elif mode == "curriculum":
        switch_step = start_step + int((total_steps - start_step) * curriculum_switch)
        if step <= switch_step:
            u = torch.randn(batch_size, device=device, dtype=dtype) * sigma
            t = torch.sigmoid(u)
        else:
            t = torch.rand(batch_size, device=device, dtype=dtype)
    else:  # uniform
        t = torch.rand(batch_size, device=device, dtype=dtype)
    return t


# ── Loss computation ─────────────────────────────────────────────────────────

def flow_matching_loss(model, x1, t, clip_feat, sync_feat, text_feat, device, dtype):
    """Compute flow matching velocity prediction loss.
    
    Args:
        model: HunyuanVideoFoley model with LoRA applied
        x1: target latents [B, 128, T] 
        t: timesteps [B] in [0, 1]
        clip_feat: SigLIP2 features [B, N_clip, 768]
        sync_feat: Synchformer features [B, N_sync, 768]
        text_feat: CLAP text embedding [B, N_text, D]
        device: torch device
        dtype: compute dtype
    
    Returns:
        loss: scalar MSE loss
    """
    B = x1.shape[0]
    x0 = torch.randn_like(x1)  # noise
    
    # Interpolate: xt = (1-t)*x0 + t*x1
    t_expand = t.view(B, 1, 1)
    xt = (1 - t_expand) * x0 + t_expand * x1
    
    # Target velocity: v = x1 - x0
    v_target = x1 - x0
    
    # Timestep for model: scale to [0, 1000] range
    t_model = (t * 1000).long()
    
    # Forward pass
    xt = xt.to(device=device, dtype=dtype)
    clip_feat = clip_feat.to(device=device, dtype=dtype)
    sync_feat = sync_feat.to(device=device, dtype=dtype)
    text_feat = text_feat.to(device=device, dtype=dtype)
    
    v_pred = model(
        x=xt, t=t_model,
        cond=text_feat,
        clip_feat=clip_feat,
        sync_feat=sync_feat,
    )["x"]
    
    v_target = v_target.to(device=device, dtype=dtype)
    loss = F.mse_loss(v_pred, v_target)
    return loss


# ── Eval sample generation ───────────────────────────────────────────────────

@torch.no_grad()
def generate_eval_sample(model, dac_model, dataset_entry, device, dtype,
                         num_steps=25, seed=42):
    """Generate an audio sample for evaluation during training.
    
    Returns:
        waveform: [1, samples] numpy array
        sample_rate: 48000
    """
    from hunyuanvideo_foley.utils.schedulers import FlowMatchDiscreteScheduler
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    clip_feat = dataset_entry["clip_features"].to(device=device, dtype=dtype)
    sync_feat = dataset_entry["sync_features"].to(device=device, dtype=dtype)
    text_feat = dataset_entry["text_embedding"].to(device=device, dtype=dtype)
    latent_shape = dataset_entry["latents"].shape  # [1, 128, T]
    
    scheduler = FlowMatchDiscreteScheduler(shift=1.0, solver="euler")
    scheduler.set_timesteps(num_steps, device=device)
    
    latents = torch.randn(latent_shape, device=device, dtype=dtype, generator=generator)
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma
    
    model.eval()
    for t in scheduler.timesteps:
        t_expand = t.expand(latents.shape[0]).to(device)
        compute_dtype = dtype
        with torch.autocast(device_type=device.type, dtype=compute_dtype):
            v_pred = model(
                x=latents.to(compute_dtype),
                t=t_expand,
                cond=text_feat,
                clip_feat=clip_feat,
                sync_feat=sync_feat,
            )["x"]
        latents = scheduler.step(v_pred, t, latents)[0]
    
    # Decode via DAC
    dac_model.to(device)
    audio = dac_model.decode(latents)
    dac_model.cpu()
    
    waveform = audio.squeeze(0).cpu().float().numpy()
    
    # Normalize to -27 dBFS
    rms = np.sqrt(np.mean(waveform ** 2))
    target_rms = 10 ** (-27 / 20)
    if rms > 1e-8:
        waveform = waveform * (target_rms / rms)
    waveform = np.clip(waveform, -1.0, 1.0)
    
    return waveform, 48000


# ── Loss curve visualization ─────────────────────────────────────────────────

def save_loss_curve(losses, path, start_step=0, smoothing=0.95):
    """Save raw and smoothed loss curve PNGs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping loss curve")
        return
    
    steps = list(range(start_step, start_step + len(losses)))
    
    # Smoothed (EMA)
    smoothed = []
    s = losses[0]
    for v in losses:
        s = smoothing * s + (1 - smoothing) * v
        smoothed.append(s)
    
    fig, ax = plt.subplots(figsize=(10, 4.75))
    ax.plot(steps, losses, alpha=0.3, color="steelblue", linewidth=0.5, label="raw")
    ax.plot(steps, smoothed, color="steelblue", linewidth=1.5, label="smoothed")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    base = str(path).rsplit(".", 1)[0]
    fig.savefig(f"{base}_raw.png", dpi=150)
    
    fig2, ax2 = plt.subplots(figsize=(10, 4.75))
    ax2.plot(steps, smoothed, color="steelblue", linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss (smoothed)")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{base}_smoothed.png", dpi=150)
    
    plt.close(fig)
    plt.close(fig2)


# ── Checkpoint I/O ───────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, step, meta, path, final=False):
    """Save training checkpoint or final adapter."""
    state = {"state_dict": get_lora_state_dict(model), "meta": meta}
    if not final:
        state["optimizer"] = optimizer.state_dict()
        state["scheduler"] = scheduler.state_dict()
        state["step"] = step
    torch.save(state, path)


def save_meta_json(meta, path):
    """Save human-readable metadata."""
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
```

**Step 2: Commit**

```bash
git add lora/train.py
git commit -m "feat: add training loop, dataset loading, and eval generation"
```

---

### Task 4: Create `nodes_lora.py` — Feature Extractor and VAE Roundtrip nodes

**Files:**
- Create: `nodes_lora.py`

**Step 1: Write the Feature Extractor and VAE Roundtrip nodes**

These are the data preparation nodes. The Feature Extractor caches SigLIP2/Synchformer/CLAP features + paired audio to .npz. The VAE Roundtrip is a diagnostic that encodes/decodes audio through DAC.

```python
"""LoRA training nodes for HunyuanVideo-Foley."""

import os
import sys
import gc
import copy
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from loguru import logger

import folder_paths
import comfy.model_management as mm

from .lora.lora import (
    apply_lora, get_lora_state_dict, load_lora,
    FOLEY_TARGET_PRESETS, LoRALinear,
)
from .lora.train import (
    prepare_dataset, sample_timesteps, flow_matching_loss,
    generate_eval_sample, save_loss_curve, save_checkpoint, save_meta_json,
)
from .lora.spectral_metrics import spectral_metrics, reference_metrics


logger.remove()
logger.add(sys.stdout, level="INFO", format="HunyuanVideo-Foley LoRA: {message}")


# ─── Node 1: Feature Extractor ──────────────────────────────────────────────

class FoleyFeatureExtractor:
    """Extract and cache SigLIP2/Synchformer/CLAP features + audio for LoRA training."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_deps": ("HUNYUAN_DEPS",),
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0, "step": 0.1}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                              "tooltip": "Clip duration in seconds. 0 = auto from audio length."}),
                "cache_dir": ("STRING", {"default": ""}),
                "name": ("STRING", {"default": "clip",
                          "tooltip": "Base name for auto-incremented files (e.g. clip -> clip_001.npz)"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("npz_path",)
    FUNCTION = "extract_features"
    CATEGORY = "audio/HunyuanFoley/LoRA"
    OUTPUT_NODE = True
    
    def extract_features(self, hunyuan_deps, image, audio, prompt, frame_rate, duration,
                         cache_dir, name):
        from hunyuanvideo_foley.utils.feature_utils import (
            encode_video_with_siglip2, encode_video_with_sync, encode_text_feat,
        )
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-increment filename
        idx = 1
        while (cache_dir / f"{name}_{idx:03d}.npz").exists():
            idx += 1
        npz_path = cache_dir / f"{name}_{idx:03d}.npz"
        audio_out_path = cache_dir / f"{name}_{idx:03d}.wav"
        
        # ── Extract visual features ──
        # image is [B, H, W, C] float32 in [0,1] from ComfyUI
        # Convert to [1, T, C, H, W] uint8 for preprocessing
        frames_bhwc = image  # [T, H, W, C]
        frames_tchw = frames_bhwc.permute(0, 3, 1, 2)  # [T, C, H, W]
        frames_uint8 = (frames_tchw * 255).clamp(0, 255).to(torch.uint8)
        
        waveform = audio["waveform"]  # [1, C, L]
        sample_rate = audio["sample_rate"]
        
        # Compute duration from audio if not specified
        if duration <= 0:
            duration = waveform.shape[-1] / sample_rate
        
        # Resample frames to target FPS for each extractor
        total_frames = frames_uint8.shape[0]
        video_duration = total_frames / frame_rate
        
        # SigLIP2: 8fps, 512x512
        siglip2_fps = 8
        n_siglip2_frames = max(1, int(duration * siglip2_fps))
        siglip2_indices = torch.linspace(0, total_frames - 1, n_siglip2_frames).long()
        siglip2_frames = frames_uint8[siglip2_indices]
        
        # Apply SigLIP2 preprocessing
        siglip2_processed = torch.stack([
            hunyuan_deps.siglip2_preprocess(f) for f in siglip2_frames
        ]).unsqueeze(0)  # [1, T, C, H, W]
        
        hunyuan_deps.siglip2_model.to(device)
        clip_features = encode_video_with_siglip2(
            siglip2_processed.to(device), hunyuan_deps
        ).cpu()
        hunyuan_deps.siglip2_model.to(offload_device)
        
        # Synchformer: 25fps, 224x224
        sync_fps = 25
        n_sync_frames = max(16, int(duration * sync_fps))
        sync_indices = torch.linspace(0, total_frames - 1, n_sync_frames).long()
        sync_frames = frames_uint8[sync_indices]
        
        sync_processed = torch.stack([
            hunyuan_deps.syncformer_preprocess(f) for f in sync_frames
        ]).unsqueeze(0)  # [1, T, C, H, W]
        
        hunyuan_deps.syncformer_model.to(device)
        sync_features = encode_video_with_sync(
            sync_processed.to(device), hunyuan_deps
        ).cpu()
        hunyuan_deps.syncformer_model.to(offload_device)
        
        # CLAP text embedding
        hunyuan_deps.clap_model.to(device)
        text_inputs = hunyuan_deps.clap_tokenizer(
            [prompt], padding=True, truncation=True, max_length=100,
            return_tensors="pt"
        ).to(device)
        text_embedding = hunyuan_deps.clap_model(**text_inputs).text_embeds.cpu()
        # Expand to [1, 1, D] for consistency
        if text_embedding.ndim == 2:
            text_embedding = text_embedding.unsqueeze(1)
        hunyuan_deps.clap_model.to(offload_device)
        
        torch.cuda.empty_cache()
        
        # Save .npz
        np.savez(
            str(npz_path),
            clip_features=clip_features.numpy(),
            sync_features=sync_features.numpy(),
            text_embedding=text_embedding.numpy(),
            prompt=prompt,
            duration=duration,
            fps=frame_rate,
        )
        
        # Save audio file
        wav = waveform.squeeze(0)  # [C, L]
        if sample_rate != 48000:
            wav = torchaudio.functional.resample(wav, sample_rate, 48000)
        torchaudio.save(str(audio_out_path), wav, 48000)
        
        logger.info(f"Saved features to {npz_path}")
        logger.info(f"  clip_features: {clip_features.shape}, sync_features: {sync_features.shape}")
        logger.info(f"  text_embedding: {text_embedding.shape}, duration: {duration:.2f}s")
        
        return (str(npz_path),)


# ─── Node 6: VAE Roundtrip ──────────────────────────────────────────────────

class FoleyVAERoundtrip:
    """Encode audio through DAC, decode back. Reveals codec quality ceiling."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_deps": ("HUNYUAN_DEPS",),
                "audio": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "roundtrip"
    CATEGORY = "audio/HunyuanFoley/LoRA"
    
    def roundtrip(self, hunyuan_deps, audio):
        device = mm.get_torch_device()
        dac = hunyuan_deps.dac_model
        
        waveform = audio["waveform"]  # [1, C, L]
        sample_rate = audio["sample_rate"]
        
        # Resample to 48kHz if needed
        if sample_rate != 48000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 48000)
        
        # Convert to mono
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        
        # DAC encode -> decode
        dac.to(device)
        with torch.no_grad():
            audio_in = waveform.to(device=device, dtype=torch.float32)
            z, _, _, _, _ = dac.encode(audio_in)
            reconstructed = dac.decode(z)
        dac.cpu()
        torch.cuda.empty_cache()
        
        # Normalize output
        out = reconstructed.cpu().float()
        rms = torch.sqrt(torch.mean(out ** 2))
        target_rms = 10 ** (-27 / 20)
        if rms > 1e-8:
            out = out * (target_rms / rms)
        out = out.clamp(-1.0, 1.0)
        
        return ({"waveform": out, "sample_rate": 48000},)
```

**Step 2: Commit**

```bash
git add nodes_lora.py
git commit -m "feat: add Feature Extractor and VAE Roundtrip nodes"
```

---

### Task 5: Add LoRA Trainer node to `nodes_lora.py`

**Files:**
- Modify: `nodes_lora.py`

**Step 1: Add the trainer node class**

Append to `nodes_lora.py`. This is the largest node — it orchestrates the full training loop.

Key behavior:
- Loads dataset once via `prepare_dataset()`
- Deep-copies the model, applies LoRA layers
- Sets up AdamW optimizer with optional LoRA+ (B-matrix LR multiplier)
- Runs training loop with configurable timestep sampling
- Saves checkpoints, loss curves, eval samples at intervals
- Returns model with LoRA active

```python
# ─── Node 2: LoRA Trainer ────────────────────────────────────────────────────

class FoleyLoRATrainer:
    """Train a LoRA adapter for HunyuanVideo-Foley via flow matching."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("HUNYUAN_MODEL",),
                "hunyuan_deps": ("HUNYUAN_DEPS",),
                "data_dir": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": ""}),
                "target": (list(FOLEY_TARGET_PRESETS.keys()), {"default": "all_attn_mlp"}),
                "rank": ("INT", {"default": 64, "min": 4, "max": 128, "step": 4}),
                "alpha": ("FLOAT", {"default": 64.0, "min": 1.0, "max": 128.0}),
                "lr": ("FLOAT", {"default": 1e-4, "min": 1e-6, "max": 1e-2, "step": 1e-5}),
                "steps": ("INT", {"default": 3000, "min": 100, "max": 50000}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 64}),
                "grad_accum": ("INT", {"default": 1, "min": 1, "max": 32}),
                "warmup_steps": ("INT", {"default": 100, "min": 0, "max": 2000}),
                "save_every": ("INT", {"default": 500, "min": 50, "max": 10000}),
                "timestep_mode": (["uniform", "logit_normal", "curriculum"], {"default": "logit_normal"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "seed": ("INT", {"default": 42}),
            },
            "optional": {
                "logit_normal_sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0}),
                "curriculum_switch": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 0.9}),
                "init_mode": (["standard", "pissa"], {"default": "standard"}),
                "use_rslora": ("BOOLEAN", {"default": False}),
                "lora_dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.3}),
                "lora_plus_ratio": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 32.0}),
                "schedule_type": (["constant", "cosine"], {"default": "constant"}),
                "latent_mixup_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "latent_noise_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.1}),
                "resume_from": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = ("HUNYUAN_MODEL",)
    FUNCTION = "train"
    CATEGORY = "audio/HunyuanFoley/LoRA"
    OUTPUT_NODE = True
    
    def train(self, hunyuan_model, hunyuan_deps, data_dir, output_dir, target, rank,
              alpha, lr, steps, batch_size, grad_accum, warmup_steps, save_every,
              timestep_mode, precision, seed,
              logit_normal_sigma=1.0, curriculum_switch=0.6,
              init_mode="standard", use_rslora=False, lora_dropout=0.0,
              lora_plus_ratio=1.0, schedule_type="constant",
              latent_mixup_alpha=0.0, latent_noise_sigma=0.0,
              resume_from=""):
        
        import random
        device = mm.get_torch_device()
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map[precision]
        
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        samples_path = output_path / "samples"
        samples_path.mkdir(exist_ok=True)
        
        # ── Prepare dataset ──
        logger.info("Preparing dataset...")
        dataset = prepare_dataset(data_dir, hunyuan_deps.dac_model, device, dtype)
        n_clips = len(dataset)
        logger.info(f"Dataset ready: {n_clips} clips")
        
        # ── Setup model with LoRA ──
        model = copy.deepcopy(hunyuan_model)
        model.to(device=device, dtype=dtype)
        model.train()
        
        target_suffixes = FOLEY_TARGET_PRESETS[target]
        n_wrapped = apply_lora(
            model, rank=rank, alpha=alpha,
            target_suffixes=target_suffixes,
            dropout=lora_dropout, init_mode=init_mode,
            use_rslora=use_rslora,
        )
        logger.info(f"LoRA applied: {n_wrapped} layers wrapped (target={target}, rank={rank})")
        
        # Freeze base, train LoRA only
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)")
        
        # ── Optimizer ──
        if lora_plus_ratio > 1.0:
            # LoRA+: separate LR for B matrices
            a_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_A" in n]
            b_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
            param_groups = [
                {"params": a_params, "lr": lr},
                {"params": b_params, "lr": lr * lora_plus_ratio},
            ]
        else:
            param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
        
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)
        
        # ── LR Scheduler ──
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            if schedule_type == "cosine":
                progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
                return 0.5 * (1 + np.cos(np.pi * progress))
            return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # ── Resume ──
        start_step = 0
        if resume_from and os.path.exists(resume_from):
            ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
            load_lora(model, ckpt["state_dict"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_step = ckpt.get("step", 0)
            logger.info(f"Resumed from step {start_step}: {resume_from}")
            del ckpt
        
        # ── Training loop ──
        meta = {
            "target": target, "rank": rank, "alpha": alpha,
            "lr": lr, "steps": steps, "batch_size": batch_size,
            "timestep_mode": timestep_mode, "logit_normal_sigma": logit_normal_sigma,
            "curriculum_switch": curriculum_switch,
            "init_mode": init_mode, "use_rslora": use_rslora,
            "lora_dropout": lora_dropout, "lora_plus_ratio": lora_plus_ratio,
            "schedule_type": schedule_type,
            "latent_mixup_alpha": latent_mixup_alpha,
            "latent_noise_sigma": latent_noise_sigma,
            "n_clips": n_clips, "precision": precision, "seed": seed,
        }
        
        losses = []
        log_interval = 50
        
        logger.info(f"Starting training: {steps} steps, batch {batch_size}, lr {lr}")
        t_start = time.time()
        
        for step in range(start_step, steps):
            # Check for skip flag
            skip_flag = output_path.parent / "skip_current.flag"
            if skip_flag.exists():
                logger.info(f"Skip flag detected at step {step}, saving and stopping")
                ckpt_path = output_path / f"adapter_cancelled_step{step:05d}.pt"
                save_checkpoint(model, optimizer, scheduler, step, meta, ckpt_path)
                break
            
            model.train()
            
            # Sample batch
            indices = [np.random.randint(0, n_clips) for _ in range(batch_size)]
            batch_latents = torch.cat([dataset[i]["latents"] for i in indices], dim=0).to(device, dtype=dtype)
            batch_clip = torch.cat([dataset[i]["clip_features"] for i in indices], dim=0)
            batch_sync = torch.cat([dataset[i]["sync_features"] for i in indices], dim=0)
            batch_text = torch.cat([dataset[i]["text_embedding"] for i in indices], dim=0)
            
            # Pad features to consistent lengths
            max_clip_len = max(batch_clip.shape[1], 1)
            max_sync_len = max(batch_sync.shape[1], 1)
            # Ensure sync length is multiple of 8 (required by model)
            max_sync_len = ((max_sync_len + 7) // 8) * 8
            batch_clip = F.pad(batch_clip, (0, 0, 0, max(0, max_clip_len - batch_clip.shape[1])))
            batch_sync = F.pad(batch_sync, (0, 0, 0, max(0, max_sync_len - batch_sync.shape[1])))
            
            # Optional latent augmentation
            if latent_mixup_alpha > 0 and batch_size > 1:
                lam = np.random.beta(latent_mixup_alpha, latent_mixup_alpha)
                perm = torch.randperm(batch_size)
                batch_latents = lam * batch_latents + (1 - lam) * batch_latents[perm]
            
            if latent_noise_sigma > 0:
                batch_latents = batch_latents + torch.randn_like(batch_latents) * latent_noise_sigma
            
            # Sample timesteps
            t = sample_timesteps(
                batch_size, timestep_mode, device, dtype,
                sigma=logit_normal_sigma, curriculum_switch=curriculum_switch,
                step=step, start_step=start_step, total_steps=steps,
            )
            
            # Forward + loss
            loss = flow_matching_loss(
                model, batch_latents, t, batch_clip, batch_sync, batch_text, device, dtype,
            )
            loss = loss / grad_accum
            loss.backward()
            
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            losses.append(loss.item() * grad_accum)
            
            # Logging
            if (step + 1) % log_interval == 0:
                avg_loss = np.mean(losses[-log_interval:])
                elapsed = time.time() - t_start
                logger.info(f"Step {step+1}/{steps} | loss: {avg_loss:.4f} | "
                           f"lr: {scheduler.get_last_lr()[0]:.2e} | "
                           f"elapsed: {elapsed:.0f}s")
            
            # Save checkpoint + eval sample
            if (step + 1) % save_every == 0:
                ckpt_path = output_path / f"adapter_step{step+1:05d}.pt"
                save_checkpoint(model, optimizer, scheduler, step + 1, meta, ckpt_path)
                save_loss_curve(losses, output_path / "loss.png", start_step)
                
                # Generate eval sample
                model.eval()
                wav, sr = generate_eval_sample(
                    model, hunyuan_deps.dac_model, dataset[0], device, dtype,
                )
                wav_t = torch.from_numpy(wav)
                if wav_t.ndim == 1:
                    wav_t = wav_t.unsqueeze(0)
                torchaudio.save(str(samples_path / f"step_{step+1:05d}.wav"), wav_t, sr)
                model.train()
        
        # ── Save final ──
        final_path = output_path / "adapter_final.pt"
        meta["steps_completed"] = step + 1 if step >= start_step else start_step
        save_checkpoint(model, optimizer, scheduler, step + 1, meta, final_path, final=True)
        save_meta_json(meta, output_path / "meta.json")
        save_loss_curve(losses, output_path / "loss.png", start_step)
        
        elapsed_total = time.time() - t_start
        logger.info(f"Training complete: {elapsed_total:.0f}s, final loss: {np.mean(losses[-100:]):.4f}")
        logger.info(f"Adapter saved to {final_path}")
        
        # Return model with LoRA active (on CPU for ComfyUI pipeline)
        model.eval()
        model.to(mm.unet_offload_device())
        return (model,)
```

**Step 2: Commit**

```bash
git add nodes_lora.py
git commit -m "feat: add LoRA Trainer node"
```

---

### Task 6: Add LoRA Loader node to `nodes_lora.py`

**Files:**
- Modify: `nodes_lora.py`

**Step 1: Add the loader node**

```python
# ─── Node 3: LoRA Loader ────────────────────────────────────────────────────

class FoleyLoRALoader:
    """Load a trained LoRA adapter into a Foley model for inference."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("HUNYUAN_MODEL",),
                "adapter_path": ("STRING", {"default": ""}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("HUNYUAN_MODEL",)
    FUNCTION = "load_adapter"
    CATEGORY = "audio/HunyuanFoley/LoRA"
    
    def load_adapter(self, hunyuan_model, adapter_path, strength):
        if not adapter_path or not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        
        ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
        
        # Handle both raw state_dict and wrapped checkpoint formats
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            meta = ckpt.get("meta", {})
        else:
            state_dict = ckpt
            meta = {}
        
        rank = meta.get("rank", 16)
        alpha = meta.get("alpha", float(rank))
        target = meta.get("target", "all_attn_mlp")
        init_mode = meta.get("init_mode", "standard")
        use_rslora = meta.get("use_rslora", False)
        lora_dropout = meta.get("lora_dropout", 0.0)
        
        # Get target suffixes
        if isinstance(target, str) and target in FOLEY_TARGET_PRESETS:
            target_suffixes = FOLEY_TARGET_PRESETS[target]
        elif isinstance(target, (list, tuple)):
            target_suffixes = tuple(target)
        else:
            target_suffixes = FOLEY_TARGET_PRESETS["all_attn_mlp"]
        
        # Deep copy model
        model = copy.deepcopy(hunyuan_model)
        
        # Apply LoRA structure (always standard init for loading)
        n_wrapped = apply_lora(
            model, rank=rank, alpha=alpha,
            target_suffixes=target_suffixes,
            dropout=lora_dropout,
            init_mode="standard",  # PiSSA residuals are in the checkpoint
            use_rslora=use_rslora,
        )
        
        # Load weights
        load_lora(model, state_dict)
        
        # Apply strength scaling
        if strength != 1.0:
            for name, module in model.named_modules():
                if isinstance(module, LoRALinear):
                    module.lora_B.data *= strength
        
        model.eval()
        logger.info(f"Loaded LoRA adapter: {n_wrapped} layers, rank={rank}, strength={strength}")
        
        return (model,)
```

**Step 2: Commit**

```bash
git add nodes_lora.py
git commit -m "feat: add LoRA Loader node"
```

---

### Task 7: Add LoRA Scheduler node to `nodes_lora.py`

**Files:**
- Modify: `nodes_lora.py`

**Step 1: Add the scheduler node**

Reads a JSON sweep file, runs experiments sequentially, produces comparison charts.

```python
# ─── Node 4: LoRA Scheduler ─────────────────────────────────────────────────

class FoleyLoRAScheduler:
    """Run multiple LoRA training experiments from a JSON sweep configuration."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("HUNYUAN_MODEL",),
                "hunyuan_deps": ("HUNYUAN_DEPS",),
                "sweep_json": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "run_sweep"
    CATEGORY = "audio/HunyuanFoley/LoRA"
    OUTPUT_NODE = True
    
    # Default training params (mirrors trainer node defaults)
    _PARAM_DEFAULTS = {
        "target": "all_attn_mlp", "rank": 64, "alpha": 64.0,
        "lr": 1e-4, "steps": 3000, "batch_size": 8, "grad_accum": 1,
        "warmup_steps": 100, "save_every": 500,
        "timestep_mode": "logit_normal", "precision": "bf16", "seed": 42,
        "logit_normal_sigma": 1.0, "curriculum_switch": 0.6,
        "init_mode": "standard", "use_rslora": False, "lora_dropout": 0.0,
        "lora_plus_ratio": 1.0, "schedule_type": "constant",
        "latent_mixup_alpha": 0.0, "latent_noise_sigma": 0.0,
    }
    
    def _merge_config(self, base, experiment):
        merged = {**self._PARAM_DEFAULTS, **base}
        for k, v in experiment.items():
            if k not in ("id", "description"):
                merged[k] = v
        return merged
    
    def run_sweep(self, hunyuan_model, hunyuan_deps, sweep_json):
        if not os.path.exists(sweep_json):
            raise FileNotFoundError(f"Sweep JSON not found: {sweep_json}")
        
        with open(sweep_json) as f:
            sweep = json.load(f)
        
        sweep_name = sweep.get("name", "sweep")
        data_dir = sweep["data_dir"]
        output_root = Path(sweep.get("output_root", f"lora_output/{sweep_name}"))
        base_config = sweep.get("base", {})
        experiments = sweep.get("experiments", [])
        
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "experiment_summary.json"
        
        # Load existing summary for resume
        completed_ids = set()
        results = []
        if summary_path.exists():
            with open(summary_path) as f:
                existing = json.load(f)
            results = existing.get("experiments", [])
            completed_ids = {r["id"] for r in results if r.get("status") == "completed"}
        
        # Prepare dataset once
        device = mm.get_torch_device()
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        base_precision = base_config.get("precision", "bf16")
        dtype = dtype_map[base_precision]
        
        logger.info(f"Preparing shared dataset from {data_dir}...")
        dataset = prepare_dataset(data_dir, hunyuan_deps.dac_model, device, dtype)
        
        # Collect loss histories for comparison chart
        all_loss_histories = {}
        
        for exp in experiments:
            exp_id = exp.get("id", f"exp_{len(results)}")
            
            if exp_id in completed_ids:
                logger.info(f"Skipping completed experiment: {exp_id}")
                # Load loss history if available
                exp_dir = output_root / exp_id
                loss_file = exp_dir / "loss_history.json"
                if loss_file.exists():
                    with open(loss_file) as f:
                        all_loss_histories[exp_id] = json.load(f)
                continue
            
            config = self._merge_config(base_config, exp)
            exp_dir = output_root / exp_id
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Starting experiment: {exp_id}")
            logger.info(f"Config: {json.dumps({k: v for k, v in config.items() if k != 'description'}, indent=2)}")
            
            exp_result = {"id": exp_id, "config": config, "status": "running"}
            
            try:
                # Deep copy model for this experiment
                model = copy.deepcopy(hunyuan_model)
                model.to(device=device, dtype=dtype)
                model.train()
                
                target_suffixes = FOLEY_TARGET_PRESETS[config["target"]]
                n_wrapped = apply_lora(
                    model, rank=config["rank"], alpha=config["alpha"],
                    target_suffixes=target_suffixes,
                    dropout=config["lora_dropout"],
                    init_mode=config["init_mode"],
                    use_rslora=config["use_rslora"],
                )
                
                for name, param in model.named_parameters():
                    param.requires_grad = "lora_" in name
                
                # Optimizer
                _lr = config["lr"]
                if config["lora_plus_ratio"] > 1.0:
                    a_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_A" in n]
                    b_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
                    param_groups = [{"params": a_params, "lr": _lr}, {"params": b_params, "lr": _lr * config["lora_plus_ratio"]}]
                else:
                    param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": _lr}]
                
                optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)
                
                def lr_lambda(step):
                    if step < config["warmup_steps"]:
                        return step / max(config["warmup_steps"], 1)
                    if config["schedule_type"] == "cosine":
                        progress = (step - config["warmup_steps"]) / max(config["steps"] - config["warmup_steps"], 1)
                        return 0.5 * (1 + np.cos(np.pi * progress))
                    return 1.0
                
                lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                
                import random
                torch.manual_seed(config["seed"])
                random.seed(config["seed"])
                np.random.seed(config["seed"])
                
                losses = []
                n_clips = len(dataset)
                t_start = time.time()
                
                for step in range(config["steps"]):
                    # Skip flag
                    skip_flag = output_root / "skip_current.flag"
                    if skip_flag.exists():
                        logger.info(f"Skip flag detected for {exp_id} at step {step}")
                        ckpt_path = exp_dir / f"adapter_cancelled_step{step:05d}.pt"
                        meta = {**config, "steps_completed": step}
                        save_checkpoint(model, optimizer, lr_sched, step, meta, ckpt_path)
                        skip_flag.unlink()
                        raise _SkipExperiment(f"Skipped at step {step}")
                    
                    model.train()
                    bs = config["batch_size"]
                    indices = [np.random.randint(0, n_clips) for _ in range(bs)]
                    batch_latents = torch.cat([dataset[i]["latents"] for i in indices]).to(device, dtype=dtype)
                    batch_clip = torch.cat([dataset[i]["clip_features"] for i in indices])
                    batch_sync = torch.cat([dataset[i]["sync_features"] for i in indices])
                    batch_text = torch.cat([dataset[i]["text_embedding"] for i in indices])
                    
                    # Pad sync to multiple of 8
                    sync_len = batch_sync.shape[1]
                    pad_sync = ((sync_len + 7) // 8) * 8 - sync_len
                    if pad_sync > 0:
                        batch_sync = F.pad(batch_sync, (0, 0, 0, pad_sync))
                    
                    t = sample_timesteps(
                        bs, config["timestep_mode"], device, dtype,
                        sigma=config["logit_normal_sigma"],
                        curriculum_switch=config["curriculum_switch"],
                        step=step, total_steps=config["steps"],
                    )
                    
                    loss = flow_matching_loss(model, batch_latents, t, batch_clip, batch_sync, batch_text, device, dtype)
                    loss = loss / config["grad_accum"]
                    loss.backward()
                    
                    if (step + 1) % config["grad_accum"] == 0:
                        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                        optimizer.step()
                        lr_sched.step()
                        optimizer.zero_grad()
                    
                    losses.append(loss.item() * config["grad_accum"])
                    
                    if (step + 1) % config["save_every"] == 0:
                        meta = {**config, "steps_completed": step + 1}
                        ckpt_path = exp_dir / f"adapter_step{step+1:05d}.pt"
                        save_checkpoint(model, optimizer, lr_sched, step + 1, meta, ckpt_path)
                
                # Save final
                meta = {**config, "steps_completed": config["steps"]}
                final_path = exp_dir / "adapter_final.pt"
                save_checkpoint(model, optimizer, lr_sched, config["steps"], meta, final_path, final=True)
                save_loss_curve(losses, exp_dir / "loss.png")
                
                # Save loss history for comparison chart
                with open(exp_dir / "loss_history.json", "w") as f:
                    json.dump(losses, f)
                all_loss_histories[exp_id] = losses
                
                elapsed = time.time() - t_start
                exp_result.update({
                    "status": "completed",
                    "final_loss": float(np.mean(losses[-100:])),
                    "min_loss": float(min(losses)),
                    "adapter_path": str(final_path),
                    "duration_seconds": elapsed,
                })
                
                del model, optimizer, lr_sched
                gc.collect()
                torch.cuda.empty_cache()
                
            except _SkipExperiment as e:
                exp_result["status"] = f"skipped: {e}"
            except Exception as e:
                exp_result["status"] = f"failed: {e}"
                logger.error(f"Experiment {exp_id} failed: {e}")
            
            results.append(exp_result)
            
            # Save summary after each experiment
            summary = {
                "name": sweep_name, "data_dir": data_dir,
                "experiments": results,
                "system": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A",
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                },
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
        
        # Generate comparison chart
        if all_loss_histories:
            _save_comparison_chart(all_loss_histories, output_root / "loss_comparison.png")
        
        logger.info(f"Sweep complete: {len(results)} experiments")
        return ()


class _SkipExperiment(Exception):
    pass


def _save_comparison_chart(histories, path):
    """Overlay smoothed loss curves for all experiments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10.colors
    
    for i, (exp_id, losses) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        # EMA smoothing
        smoothed = []
        s = losses[0]
        for v in losses:
            s = 0.95 * s + 0.05 * v
            smoothed.append(s)
        ax.plot(smoothed, color=color, linewidth=1.5, label=exp_id)
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (smoothed)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
```

**Step 2: Commit**

```bash
git add nodes_lora.py
git commit -m "feat: add LoRA Scheduler node for experiment sweeps"
```

---

### Task 8: Add LoRA Evaluator node to `nodes_lora.py`

**Files:**
- Modify: `nodes_lora.py`

**Step 1: Add the evaluator node**

```python
# ─── Node 5: LoRA Evaluator ─────────────────────────────────────────────────

class FoleyLoRAEvaluator:
    """Compare multiple LoRA adapters by generating audio and computing spectral metrics."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("HUNYUAN_MODEL",),
                "hunyuan_deps": ("HUNYUAN_DEPS",),
                "eval_json": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "evaluate"
    CATEGORY = "audio/HunyuanFoley/LoRA"
    OUTPUT_NODE = True
    
    def evaluate(self, hunyuan_model, hunyuan_deps, eval_json):
        if not os.path.exists(eval_json):
            raise FileNotFoundError(f"Eval JSON not found: {eval_json}")
        
        with open(eval_json) as f:
            spec = json.load(f)
        
        data_dir = spec["data_dir"]
        output_dir = Path(spec.get("output_dir", "lora_eval"))
        num_steps = spec.get("steps", 25)
        seed = spec.get("seed", 42)
        adapters = spec.get("adapters", [])
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        device = mm.get_torch_device()
        dtype = torch.bfloat16
        
        # Prepare dataset
        dataset = prepare_dataset(data_dir, hunyuan_deps.dac_model, device, dtype)
        
        # Compute reference metrics from original audio
        ref_dir = output_dir / "reference"
        ref_dir.mkdir(exist_ok=True)
        ref_metrics_list = []
        
        for entry in dataset:
            # Load original audio for reference
            audio_exts = (".wav", ".flac", ".ogg", ".aiff", ".aif")
            ref_path = None
            for ext in audio_exts:
                candidate = Path(data_dir) / f"{entry['name']}{ext}"
                if candidate.exists():
                    ref_path = candidate
                    break
            if ref_path:
                import torchaudio as ta
                ref_wav, ref_sr = ta.load(str(ref_path))
                ref_wav_np = ref_wav.mean(dim=0).numpy()  # mono
                if ref_sr != 48000:
                    ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, 48000)
                    ref_wav_np = ref_wav.mean(dim=0).numpy()
                ref_m = spectral_metrics(ref_wav_np, 48000)
                ref_metrics_list.append(ref_m)
                # Save reference
                ta.save(str(ref_dir / f"{entry['name']}.wav"), ref_wav.mean(dim=0, keepdim=True), 48000)
        
        ref_avg = {}
        if ref_metrics_list:
            for key in ref_metrics_list[0]:
                ref_avg[key] = float(np.mean([m[key] for m in ref_metrics_list]))
        
        # Evaluate each adapter
        adapter_results = []
        
        for adapter_spec in adapters:
            adapter_id = adapter_spec.get("id", "unknown")
            adapter_path = adapter_spec.get("path", None)
            
            logger.info(f"Evaluating adapter: {adapter_id}")
            adapter_dir = output_dir / adapter_id
            adapter_dir.mkdir(exist_ok=True)
            
            # Load adapter or use baseline
            if adapter_path and os.path.exists(adapter_path):
                ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
                sd = ckpt.get("state_dict", ckpt)
                meta = ckpt.get("meta", {})
                
                model = copy.deepcopy(hunyuan_model)
                rank = meta.get("rank", 16)
                alpha_val = meta.get("alpha", float(rank))
                target = meta.get("target", "all_attn_mlp")
                target_suffixes = FOLEY_TARGET_PRESETS.get(target, FOLEY_TARGET_PRESETS["all_attn_mlp"])
                
                apply_lora(model, rank=rank, alpha=alpha_val, target_suffixes=target_suffixes,
                           init_mode="standard", use_rslora=meta.get("use_rslora", False))
                load_lora(model, sd)
            else:
                model = copy.deepcopy(hunyuan_model)
                meta = {}
            
            model.to(device=device, dtype=dtype)
            model.eval()
            
            clip_metrics_list = []
            clips = []
            
            for ci, entry in enumerate(dataset):
                wav, sr = generate_eval_sample(
                    model, hunyuan_deps.dac_model, entry, device, dtype,
                    num_steps=num_steps, seed=seed,
                )
                wav_mono = wav.squeeze()
                sm = spectral_metrics(wav_mono, sr)
                clip_metrics_list.append(sm)
                
                wav_path = adapter_dir / f"{entry['name']}.wav"
                wav_t = torch.from_numpy(wav)
                if wav_t.ndim == 1:
                    wav_t = wav_t.unsqueeze(0)
                torchaudio.save(str(wav_path), wav_t, sr)
                clips.append({"clip": entry["name"], "wav_path": str(wav_path), "spectral_metrics": sm})
            
            avg_metrics = {}
            if clip_metrics_list:
                for key in clip_metrics_list[0]:
                    avg_metrics[key] = float(np.mean([m[key] for m in clip_metrics_list]))
            
            adapter_results.append({
                "id": adapter_id, "path": adapter_path, "meta": meta,
                "clips": clips, "avg_metrics": avg_metrics, "status": "completed",
            })
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
        
        # Save summary
        summary = {
            "name": spec.get("name", "eval"),
            "data_dir": data_dir, "output_dir": str(output_dir),
            "n_clips": len(dataset), "steps": num_steps, "seed": seed,
            "reference_avg": ref_avg,
            "adapters": adapter_results,
        }
        with open(output_dir / "eval_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Comparison chart
        _save_eval_chart(ref_avg, adapter_results, output_dir / "metric_comparison.png")
        
        logger.info(f"Evaluation complete: {len(adapter_results)} adapters")
        return ()


def _save_eval_chart(ref_avg, adapter_results, path):
    """2x2 bar chart comparing spectral metrics across adapters."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    metrics_to_plot = ["hf_energy_ratio", "spectral_centroid_hz", "spectral_flatness", "temporal_variance"]
    titles = ["HF Energy Ratio (>4kHz)", "Spectral Centroid (Hz)", "Spectral Flatness", "Temporal Variance"]
    
    ids = ["reference"] + [a["id"] for a in adapter_results]
    colors = plt.cm.tab10.colors
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for ax, metric, title in zip(axes.flat, metrics_to_plot, titles):
        values = [ref_avg.get(metric, 0)]
        for a in adapter_results:
            values.append(a["avg_metrics"].get(metric, 0))
        
        bars = ax.barh(ids, values, color=[colors[i % len(colors)] for i in range(len(ids))])
        for bar, val in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f" {val:.4f}", va="center", fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")
    
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
```

**Step 2: Commit**

```bash
git add nodes_lora.py
git commit -m "feat: add LoRA Evaluator node with spectral metrics"
```

---

### Task 9: Register nodes in `__init__.py`

**Files:**
- Modify: `__init__.py`
- Add node mappings to `nodes_lora.py`

**Step 1: Add NODE_CLASS_MAPPINGS to nodes_lora.py**

Append to end of `nodes_lora.py`:

```python
# ─── Node Mappings ───────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "FoleyFeatureExtractor": FoleyFeatureExtractor,
    "FoleyLoRATrainer": FoleyLoRATrainer,
    "FoleyLoRALoader": FoleyLoRALoader,
    "FoleyLoRAScheduler": FoleyLoRAScheduler,
    "FoleyLoRAEvaluator": FoleyLoRAEvaluator,
    "FoleyVAERoundtrip": FoleyVAERoundtrip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoleyFeatureExtractor": "Foley Feature Extractor",
    "FoleyLoRATrainer": "Foley LoRA Trainer",
    "FoleyLoRALoader": "Foley LoRA Loader",
    "FoleyLoRAScheduler": "Foley LoRA Scheduler",
    "FoleyLoRAEvaluator": "Foley LoRA Evaluator",
    "FoleyVAERoundtrip": "Foley VAE Roundtrip",
}
```

**Step 2: Update `__init__.py` to merge both node sets**

```python
# ComfyUI-HunyuanVideoFoley/__init__.py

import sys
import os
import warnings
from pydantic.warnings import UnsupportedFieldAttributeWarning
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .nodes_lora import (
    NODE_CLASS_MAPPINGS as LORA_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as LORA_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **LORA_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **LORA_NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
```

**Step 3: Commit**

```bash
git add __init__.py nodes_lora.py
git commit -m "feat: register LoRA training nodes in ComfyUI"
```

---

### Task 10: Update requirements.txt and README

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md`

**Step 1: Check if matplotlib is already a dependency**

Run: `grep matplotlib requirements.txt`

If not present, add `matplotlib` to requirements.txt (needed for loss curves and eval charts).

**Step 2: Add LoRA training section to README**

Add a section after the existing content covering:
- New nodes overview (6 nodes)
- Quick start for LoRA training (Feature Extractor -> Trainer -> Loader -> Sampler)
- Dataset preparation guide
- Sweep JSON format
- Eval JSON format
- Hyperparameter recommendations by dataset size

**Step 3: Commit**

```bash
git add requirements.txt README.md
git commit -m "docs: add LoRA training documentation and matplotlib dependency"
```

---

### Task 11: Integration testing

**Step 1: Verify imports work**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley && python -c "from lora.lora import LoRALinear, apply_lora, load_lora, FOLEY_TARGET_PRESETS; print('lora.py OK')"` 

Expected: `lora.py OK`

**Step 2: Verify spectral metrics work**

Run: `python -c "from lora.spectral_metrics import spectral_metrics; import numpy as np; m = spectral_metrics(np.random.randn(48000), 48000); print(m)"`

Expected: dict with 5 metric keys

**Step 3: Verify node registration works**

Run: `python -c "from nodes_lora import NODE_CLASS_MAPPINGS; print(list(NODE_CLASS_MAPPINGS.keys()))"`

Expected: List of 6 node names

**Step 4: Verify LoRA injection works on a mock model**

```python
python -c "
import torch.nn as nn
from lora.lora import apply_lora, get_lora_state_dict, load_lora, FOLEY_TARGET_PRESETS

# Mock model with matching layer names
class MockBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_self_attn_qkv = nn.Linear(64, 192)
        self.audio_self_proj = nn.Linear(64, 64)

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.triple_blocks = nn.ModuleList([MockBlock() for _ in range(2)])

model = MockModel()
n = apply_lora(model, rank=4, target_suffixes=FOLEY_TARGET_PRESETS['audio_attn'])
print(f'Wrapped {n} layers')
sd = get_lora_state_dict(model)
print(f'LoRA state dict keys: {len(sd)}')
print('LoRA injection test PASSED')
"
```

Expected: `Wrapped 4 layers`, `LoRA state dict keys: 8`, `LoRA injection test PASSED`

**Step 5: Commit test results / fix any issues**

```bash
git add -A
git commit -m "test: verify LoRA integration"
```
