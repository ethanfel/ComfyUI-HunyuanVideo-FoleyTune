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
import soundfile as sf
import soxr
from loguru import logger

from .lora import apply_lora, get_lora_state_dict, get_lora_and_base_state_dict, load_lora, spectral_surgery, FOLEY_TARGET_PRESETS
from .spectral_metrics import spectral_metrics


# -- Dataset ------------------------------------------------------------------

def prepare_dataset(data_dir: str, dac_model, device, dtype=torch.bfloat16, clip_names=None):
    """Load .npz feature caches + audio files, encode audio via DAC.

    Args:
        data_dir: Directory containing .npz and audio files.
        dac_model: DAC model for audio encoding.
        device: Torch device.
        dtype: Compute dtype.
        clip_names: Optional list of clip stem names to load. When provided,
            only those clips are loaded instead of globbing all .npz files.

    Returns list of dicts with keys:
        latents: [1, 128, T] DAC-encoded audio latent (target x1)
        clip_features: [1, N_clip, 768] SigLIP2 visual features
        sync_features: [1, N_sync, 768] Synchformer sync features
        text_embedding: [1, D] CLAP text embedding
        prompt: str
        name: str (stem of .npz file)
    """
    data_dir = Path(data_dir)
    if clip_names is not None:
        npz_files = [data_dir / f"{name}.npz" for name in clip_names]
        npz_files = [f for f in npz_files if f.exists()]
    else:
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
        wav_np, sr = sf.read(str(audio_path))  # [L] or [L, C]
        if wav_np.ndim == 1:
            wav_np = wav_np[:, None]  # [L, 1]
        if sr != 48000:
            wav_np = soxr.resample(wav_np, sr, 48000, quality="VHQ")
        waveform = torch.from_numpy(wav_np.T).float()  # [C, L]
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Use actual audio length — duration field is from video and may differ
        # Just ensure length is valid for DAC encoding (no pad/trim needed)

        # DAC encode: [1, 1, samples] -> latents
        # NOTE: DAC with continuous=True returns DiagonalGaussianDistribution, not tensor
        with torch.no_grad():
            audio_input = waveform.unsqueeze(0).to(device=device, dtype=torch.float32)
            z_dist, _, _, _, _ = dac_model.encode(audio_input)
            latents = z_dist.mode().cpu().float()  # [1, 128, T] — deterministic posterior mean

        dataset.append({
            "latents": latents,
            "clip_features": clip_features,
            "sync_features": sync_features,
            "text_embedding": text_embedding,
            "prompt": prompt,
            "name": stem,
        })
        logger.info(f"  {stem}: {prompt!r}")

    # Offload DAC encoder back to CPU
    dac_model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    # Enforce fixed sequence lengths across all clips (required for batching with torch.cat)
    if dataset:
        # Normalize latent lengths
        latent_lengths = [d["latents"].shape[-1] for d in dataset]
        target_lat = min(latent_lengths)
        for d in dataset:
            d["latents"] = d["latents"][..., :target_lat]
        logger.info(f"Latents normalized to length {target_lat}")

        # Normalize clip_features (SigLIP2) sequence lengths — dim 1
        clip_lens = [d["clip_features"].shape[1] for d in dataset]
        target_clip = min(clip_lens)
        for d in dataset:
            d["clip_features"] = d["clip_features"][:, :target_clip, :]

        # Normalize sync_features (Synchformer) sequence lengths — dim 1, pad to multiple of 8
        sync_lens = [d["sync_features"].shape[1] for d in dataset]
        target_sync = min(sync_lens)
        target_sync = ((target_sync + 7) // 8) * 8  # model requires multiple of 8
        for d in dataset:
            seq = d["sync_features"].shape[1]
            if seq > target_sync:
                d["sync_features"] = d["sync_features"][:, :target_sync, :]
            elif seq < target_sync:
                d["sync_features"] = F.pad(d["sync_features"], (0, 0, 0, target_sync - seq))

        # Normalize text_embedding (CLAP) sequence lengths — dim 1
        text_lens = [d["text_embedding"].shape[1] for d in dataset]
        target_text = min(text_lens)
        for d in dataset:
            d["text_embedding"] = d["text_embedding"][:, :target_text, :]

        logger.info(f"Features normalized: clip={target_clip}, sync={target_sync}, text={target_text}")

    logger.info(f"Prepared dataset: {len(dataset)} clips from {data_dir}")
    return dataset


def prepare_single_entry(npz_path: str, dac_model, device, dtype=torch.bfloat16):
    """Load a single NPZ + its audio file and DAC-encode it.

    Same format as prepare_dataset entries but for a single file (e.g. validation).
    The audio file must be alongside the NPZ with matching stem.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    stem = npz_path.stem
    parent = npz_path.parent

    audio_path = None
    for ext in (".wav", ".flac", ".ogg", ".aiff", ".aif"):
        candidate = parent / f"{stem}{ext}"
        if candidate.exists():
            audio_path = candidate
            break
    if audio_path is None:
        raise FileNotFoundError(f"No audio file found for {stem} in {parent}")

    data = np.load(str(npz_path), allow_pickle=True)
    clip_features = torch.from_numpy(data["clip_features"]).float()
    sync_features = torch.from_numpy(data["sync_features"]).float()
    text_embedding = torch.from_numpy(data["text_embedding"]).float()
    prompt = str(data.get("prompt", stem))

    wav_np, sr = sf.read(str(audio_path))
    if wav_np.ndim == 1:
        wav_np = wav_np[:, None]
    if sr != 48000:
        wav_np = soxr.resample(wav_np, sr, 48000, quality="VHQ")
    waveform = torch.from_numpy(wav_np.T).float()
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    dac_model.to(device)
    with torch.no_grad():
        audio_input = waveform.unsqueeze(0).to(device=device, dtype=torch.float32)
        z_dist, _, _, _, _ = dac_model.encode(audio_input)
        latents = z_dist.mode().cpu().float()
    dac_model.cpu()
    torch.cuda.empty_cache()

    # Pad sync to multiple of 8
    sync_len = sync_features.shape[1]
    pad_sync = ((sync_len + 7) // 8) * 8 - sync_len
    if pad_sync > 0:
        sync_features = F.pad(sync_features, (0, 0, 0, pad_sync))

    return {
        "latents": latents,
        "clip_features": clip_features,
        "sync_features": sync_features,
        "text_embedding": text_embedding,
        "prompt": prompt,
        "name": stem,
    }


# -- Timestep sampling -------------------------------------------------------

def sample_timesteps(batch_size, mode, device, dtype,
                     sigma=1.0, curriculum_switch=0.6,
                     step=0, start_step=0, total_steps=1000,
                     t_min=0.0, t_max=1.0):
    """Sample timesteps t in [t_min, t_max] for flow matching training."""
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
    if t_min > 0.0 or t_max < 1.0:
        t = t.clamp(min=t_min, max=t_max)
    return t


def min_snr_weight(sigma, gamma=5.0):
    """Compute Min-SNR loss weighting from sigma values.

    For flow matching with linear schedule: SNR = (1-sigma)^2 / sigma^2.
    Weight = clamp(SNR, max=gamma). Emphasizes low-noise timesteps
    where the model learns fine detail.

    Args:
        sigma: [B] tensor of sigma values in [0, 1]
        gamma: clamp ceiling (default 5.0)

    Returns:
        [B] weight tensor, broadcastable to loss shape
    """
    snr = ((1 - sigma) ** 2) / (sigma ** 2 + 1e-8)
    return snr.clamp(max=gamma)


def multi_resolution_spectral_loss(predicted, target, window_sizes=(4, 16, 64), hf_weight=2.0):
    """Multi-resolution STFT loss in DAC latent space.

    Computes L1 magnitude + log-magnitude loss at multiple STFT resolutions,
    with extra weight on high-frequency bins.

    DAC latents are [B, C, T] where C=128 channels at 50fps. STFT is computed
    per-channel along the time axis.

    Args:
        predicted: [B, C, T] predicted latents
        target: [B, C, T] target latents
        window_sizes: STFT n_fft sizes to use
        hf_weight: multiplier for upper-half frequency bins

    Returns:
        scalar loss
    """
    B, C, T = predicted.shape
    # Flatten batch and channel for per-channel STFT: [B*C, T]
    pred_flat = predicted.reshape(B * C, T)
    tgt_flat = target.reshape(B * C, T)

    total = torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype)
    for ws in window_sizes:
        if T < ws:
            continue  # skip if latent sequence too short for this window

        pred_stft = torch.stft(
            pred_flat, n_fft=ws, hop_length=max(ws // 4, 1),
            win_length=ws, return_complex=True,
            window=torch.hann_window(ws, device=predicted.device, dtype=predicted.dtype),
        )
        tgt_stft = torch.stft(
            tgt_flat, n_fft=ws, hop_length=max(ws // 4, 1),
            win_length=ws, return_complex=True,
            window=torch.hann_window(ws, device=target.device, dtype=target.dtype),
        )

        mag_pred = pred_stft.abs()
        mag_tgt = tgt_stft.abs()

        # L1 magnitude loss
        convergence = F.l1_loss(mag_pred, mag_tgt)

        # Log-magnitude loss (perceptual)
        log_loss = F.l1_loss(torch.log1p(mag_pred), torch.log1p(mag_tgt))

        # HF emphasis: weight upper half of frequency bins
        n_bins = mag_pred.shape[-2]
        hf_mask = torch.zeros_like(mag_pred)
        hf_mask[..., n_bins // 2:, :] = hf_weight
        hf_loss = F.l1_loss(mag_pred * hf_mask, mag_tgt * hf_mask)

        total = total + convergence + log_loss + 0.5 * hf_loss

    n_valid = sum(1 for ws in window_sizes if T >= ws)
    if n_valid > 0:
        total = total / n_valid

    return total


# -- Loss computation --------------------------------------------------------

def flow_matching_loss(model, x1, t, clip_feat, sync_feat, text_feat, device, dtype,
                       visual_dropout_prob=0.0, min_snr_gamma=0.0,
                       cos_sim_weight=0.0, channel_weights=None,
                       temporal_variance_weight=0.0):
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
        visual_dropout_prob: per-sample probability of replacing visual features with
            null embeddings during training. Forces text channel to carry audio signal,
            decoupling identity from sound. Use 0.5 for generic-style LoRAs, 0.0 for
            identity-preserving LoRAs.
        min_snr_gamma: Min-SNR loss weighting gamma. When > 0, downweights
            high-noise timesteps where gradients are noisy and uninformative,
            focusing learning on the perceptually critical mid-range. Use 5.0.

    Returns:
        loss: scalar MSE loss
    """
    B = x1.shape[0]

    # Build per-sample visual dropout mask for the model's native drop_visual arg.
    drop_visual = None
    if visual_dropout_prob > 0:
        drop_visual = (torch.rand(B) < visual_dropout_prob).tolist()
        if not any(drop_visual):
            drop_visual = None

    x0 = torch.randn_like(x1)  # noise

    # Scheduler convention: x(sigma) = sigma * noise + (1-sigma) * data
    # sigma = timestep / 1000, so t_model = sigma * 1000
    # At sigma=1 (t_model=1000): pure noise. At sigma=0 (t_model=0): clean data.
    t_expand = t.view(B, 1, 1)
    xt = t_expand * x0 + (1 - t_expand) * x1

    # Target velocity: dx/dsigma = noise - data (matches scheduler Euler step)
    v_target = x0 - x1

    # Timestep for model: scale to [0, 1000] range
    t_model = t * 1000

    # Forward pass
    xt = xt.to(device=device, dtype=dtype)
    clip_feat = clip_feat.to(device=device, dtype=dtype)
    sync_feat = sync_feat.to(device=device, dtype=dtype)
    text_feat = text_feat.to(device=device, dtype=dtype)

    # Ensure sync features are padded to multiple of 8 (model assertion)
    sync_len = sync_feat.shape[1]
    pad_sync = ((sync_len + 7) // 8) * 8 - sync_len
    if pad_sync > 0:
        sync_feat = F.pad(sync_feat, (0, 0, 0, pad_sync))

    v_pred = model(
        x=xt, t=t_model,
        cond=text_feat,
        clip_feat=clip_feat,
        sync_feat=sync_feat,
        drop_visual=drop_visual,
    )["x"]

    v_target = v_target.to(device=device, dtype=dtype)

    mse_unreduced = F.mse_loss(v_pred, v_target, reduction='none')

    if channel_weights is not None:
        mse_unreduced = channel_weights.view(1, -1, 1).to(device=device, dtype=dtype) * mse_unreduced

    if min_snr_gamma > 0:
        # SNR = signal²/noise² = (1-t)²/t² for flow matching xt = t*noise + (1-t)*data
        snr = ((1 - t) / (t + 1e-8)) ** 2
        weight = torch.clamp(snr, max=min_snr_gamma) / (snr + 1e-8)
        loss = (weight.view(B, 1, 1) * mse_unreduced).mean()
    else:
        loss = mse_unreduced.mean()

    if cos_sim_weight > 0:
        cos_loss = 1 - F.cosine_similarity(v_pred, v_target, dim=-1).mean()
        loss = loss + cos_sim_weight * cos_loss

    if temporal_variance_weight > 0:
        # Temporal difference loss: penalise mismatch in temporal transitions.
        # If v_pred is temporally smooth, its first-order differences are near-zero
        # while v_target's are not — this doubles the gradient for missed spikes.
        diff_pred = torch.diff(v_pred, dim=-1)
        diff_target = torch.diff(v_target, dim=-1)
        tv_loss = F.mse_loss(diff_pred, diff_target)
        loss = loss + temporal_variance_weight * tv_loss

    return loss


# -- Eval sample generation --------------------------------------------------

@torch.no_grad()
def generate_eval_sample(model, dac_model, dataset_entry, device, dtype,
                         num_steps=50, seed=42, cfg_scale=5.0):
    """Generate an audio sample for evaluation during training.

    Uses classifier-free guidance (CFG) matching the inference pipeline.

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

    # Ensure sync features are padded to multiple of 8 (model assertion)
    sync_len = sync_feat.shape[1]
    pad_sync = ((sync_len + 7) // 8) * 8 - sync_len
    if pad_sync > 0:
        sync_feat = F.pad(sync_feat, (0, 0, 0, pad_sync))

    # Build unconditional embeddings for CFG
    uncond_clip = model.get_empty_clip_sequence(bs=1, len=clip_feat.shape[1]).to(device=device, dtype=dtype)
    uncond_sync = model.get_empty_sync_sequence(bs=1, len=sync_feat.shape[1]).to(device=device, dtype=dtype)
    uncond_text = torch.zeros_like(text_feat)

    # Precompute doubled-batch features
    cfg_clip = torch.cat([uncond_clip, clip_feat])
    cfg_sync = torch.cat([uncond_sync, sync_feat])
    cfg_text = torch.cat([uncond_text, text_feat])

    scheduler = FlowMatchDiscreteScheduler(shift=1.0, solver="euler")
    scheduler.set_timesteps(num_steps, device=device)

    latents = torch.randn(latent_shape, device=device, dtype=dtype, generator=generator)
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma

    model.eval()
    for t in scheduler.timesteps:
        latent_input = torch.cat([latents, latents])
        t_expand = t.expand(latent_input.shape[0]).to(device)
        compute_dtype = dtype
        with torch.autocast(device_type=device.type, dtype=compute_dtype):
            v_pred = model(
                x=latent_input.to(compute_dtype),
                t=t_expand,
                cond=cfg_text,
                clip_feat=cfg_clip,
                sync_feat=cfg_sync,
            )["x"]
        v_uncond, v_cond = v_pred.chunk(2)
        v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
        latents = scheduler.step(v_guided, t, latents)[0]

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
    if not losses:
        return

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

def save_checkpoint(model, optimizer, scheduler, step, meta, path, final=False,
                    ema_state=None):
    """Save training checkpoint or final adapter."""
    # PiSSA modifies base weights during init — must save them too
    if meta.get("init_mode") == "pissa":
        state = {"state_dict": get_lora_and_base_state_dict(model), "meta": meta}
    else:
        state = {"state_dict": get_lora_state_dict(model), "meta": meta}
    if not final:
        state["optimizer"] = optimizer.state_dict()
        state["scheduler"] = scheduler.state_dict()
        state["step"] = step
        if ema_state is not None:
            state["ema_state"] = {k: v.cpu() for k, v in ema_state.items()}
    torch.save(state, path)


def save_meta_json(meta, path):
    """Save human-readable metadata."""
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
