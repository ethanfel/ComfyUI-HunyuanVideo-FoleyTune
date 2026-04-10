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


# -- Dataset ------------------------------------------------------------------

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
        # NOTE: DAC with continuous=True returns DiagonalGaussianDistribution, not tensor
        with torch.no_grad():
            audio_input = waveform.unsqueeze(0).to(device=device, dtype=dtype)
            z_dist, _, _, _, _ = dac_model.encode(audio_input)
            latents = z_dist.sample().cpu().float()  # [1, 128, T]

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

    # Enforce fixed latent length across all clips (required for batching with torch.cat)
    if dataset:
        latent_lengths = [d["latents"].shape[-1] for d in dataset]
        target_len = min(latent_lengths)  # trim to shortest
        for d in dataset:
            if d["latents"].shape[-1] > target_len:
                d["latents"] = d["latents"][..., :target_len]
            elif d["latents"].shape[-1] < target_len:
                d["latents"] = F.pad(d["latents"], (0, target_len - d["latents"].shape[-1]))
        logger.info(f"Latents normalized to length {target_len}")

    logger.info(f"Prepared dataset: {len(dataset)} clips from {data_dir}")
    return dataset


# -- Timestep sampling -------------------------------------------------------

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


# -- Loss computation --------------------------------------------------------

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

    # Timestep for model: scale to [0, 1000] range (keep as float -- TimestepEmbedder handles floats)
    t_model = t * 1000

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


# -- Eval sample generation --------------------------------------------------

@torch.no_grad()
def generate_eval_sample(model, dac_model, dataset_entry, device, dtype,
                         num_steps=25, seed=42):
    """Generate an audio sample for evaluation during training.

    NOTE: No classifier-free guidance (CFG) is used here. This is intentional --
    eval samples during training are quick quality checks, not final output.
    The evaluator node and inference pipeline use full CFG.

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


# -- Loss curve visualization ------------------------------------------------

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


# -- Checkpoint I/O ----------------------------------------------------------

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
