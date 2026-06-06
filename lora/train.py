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

    # Enforce consistent sequence lengths for batching with torch.cat.
    #   - clip_features (SigLIP2) and sync_features (Synchformer) are FIXED-length by
    #     design (video frames at a fixed fps). A clip with an off length is a malformed
    #     source clip (e.g. cut a frame short). DROP + LOG it rather than truncating the
    #     whole dataset down to one bad clip (which also mismatches across combined dirs
    #     and crashes at batch time). Keep the modal length, drop the outliers.
    #   - latents (audio length) + text_embedding genuinely vary -> truncate to min.
    if dataset:
        from collections import Counter
        clip_mode = Counter(d["clip_features"].shape[1] for d in dataset).most_common(1)[0][0]
        sync_mode = Counter(d["sync_features"].shape[1] for d in dataset).most_common(1)[0][0]
        bad = [d for d in dataset
               if d["clip_features"].shape[1] != clip_mode
               or d["sync_features"].shape[1] != sync_mode]
        if bad:
            for d in bad:
                logger.warning(
                    f"  DROP malformed clip {d['name']}: clip_len={d['clip_features'].shape[1]} "
                    f"sync_len={d['sync_features'].shape[1]} (expected clip={clip_mode}, sync={sync_mode}) "
                    f"-- ignoring")
            bad_names = {d["name"] for d in bad}
            dataset = [d for d in dataset if d["name"] not in bad_names]
            logger.warning(f"Dropped {len(bad)} malformed clip(s) with off-length features; {len(dataset)} remain")

        # sync_features must be a multiple of 8 for the model -- pad the (now-uniform) length up
        target_sync = ((sync_mode + 7) // 8) * 8
        if target_sync != sync_mode:
            for d in dataset:
                seq = d["sync_features"].shape[1]
                if seq < target_sync:
                    d["sync_features"] = F.pad(d["sync_features"], (0, 0, 0, target_sync - seq))
                elif seq > target_sync:
                    d["sync_features"] = d["sync_features"][:, :target_sync, :]

        # latents + text_embedding genuinely vary -> truncate to min
        target_lat = min(d["latents"].shape[-1] for d in dataset)
        for d in dataset:
            d["latents"] = d["latents"][..., :target_lat]
        target_text = min(d["text_embedding"].shape[1] for d in dataset)
        for d in dataset:
            d["text_embedding"] = d["text_embedding"][:, :target_text, :]

        logger.info(f"Features: clip={clip_mode}, sync={target_sync}, text={target_text}, "
                    f"latents={target_lat} ({len(dataset)} clips)")

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
    # cuFFT requires float32 — cast from bf16/fp16 and cast back
    orig_dtype = predicted.dtype
    pred_flat = predicted.reshape(B * C, T).float()
    tgt_flat = target.reshape(B * C, T).float()

    total = torch.tensor(0.0, device=predicted.device)
    for ws in window_sizes:
        if T < ws:
            continue  # skip if latent sequence too short for this window

        _window = torch.hann_window(ws, device=predicted.device)
        pred_stft = torch.stft(
            pred_flat, n_fft=ws, hop_length=max(ws // 4, 1),
            win_length=ws, return_complex=True, window=_window,
        )
        tgt_stft = torch.stft(
            tgt_flat, n_fft=ws, hop_length=max(ws // 4, 1),
            win_length=ws, return_complex=True, window=_window,
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

    return total.to(orig_dtype)


def compute_channel_weights(all_latents, mode):
    """Per-channel MSE weights from dataset latent statistics.

    NOTE: empirically, the DAC-VAE latents for this model are bimodal — ~120 of 128
    channels are near-unit-variance (std ~0.84-1.27) and ~8 are dead (std ~0.013,
    unused codec dims). The real channels are already well-conditioned, so channel
    weighting has little leverage; the dead channels carry no signal and MUST NOT be
    up-weighted (their velocity target is pure noise). HF lives in low-energy content
    distributed across the healthy channels, not in identifiable low-variance channels
    — use the waveform_spectral_loss for HF, not channel weighting.

    Modes:
        "off"      -> None (uniform)
        "variance" -> weight proportional to per-channel variance, clamp [0.5, 2.0]. Legacy.
        "inverse"  -> weight proportional to 1/std among LIVE channels (clamp [0.5, 4.0]),
                      dead channels pinned to 1.0 so their noise is never amplified.

    Args:
        all_latents: [N, 128, T] stacked dataset latents
        mode: one of the above strings

    Returns:
        [128] weight tensor or None
    """
    if mode == "off" or not mode:
        return None
    if mode == "variance":
        ch_var = all_latents.var(dim=(0, 2))
        return (ch_var / ch_var.mean()).clamp(0.5, 2.0)
    if mode == "inverse":
        ch_std = all_latents.std(dim=(0, 2))
        med = ch_std.median()
        dead = ch_std < 0.1 * med  # unused codec dims — predicting their noise is futile
        w = (med / (ch_std + 1e-6)).clamp(0.5, 4.0)
        w[dead] = 1.0
        return w
    raise ValueError(f"Unknown channel_weight_mode: {mode!r}")


def waveform_spectral_loss(pred_wav, tgt_wav, sr=48000, ffts=(512, 1024, 2048),
                           hf_hz=4000.0, hf_weight=3.0, energy_adaptive=True):
    """Multi-resolution STFT loss on DECODED 48kHz waveforms with >4kHz emphasis.

    Unlike multi_resolution_spectral_loss (which runs on DAC latents and is blind
    to audio frequency — the latent time axis is 50fps/25Hz Nyquist), this operates
    on the real waveform after DAC decode, so the STFT bins map to true audio Hz and
    can target the >4kHz band that collapses during training.

    Args:
        pred_wav: [B, samples] predicted waveform (requires grad through DAC decode)
        tgt_wav: [B, samples] target waveform (no grad)
        sr: sample rate (48000)
        ffts: STFT window sizes for multi-resolution analysis
        hf_hz: high-frequency band threshold (matches spectral_metrics hf_energy_ratio)
        hf_weight: multiplier on the HF-band term
        energy_adaptive: if True, weight HF error by 1/sqrt(target_mag) to up-weight
            low-energy time-frequency bins (Flow2GAN Eq.6 style); else flat HF band L1

    Returns:
        scalar loss
    """
    total = pred_wav.new_zeros(())
    for n_fft in ffts:
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=pred_wav.device)
        P = torch.stft(pred_wav, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                       window=win, return_complex=True).abs()
        T = torch.stft(tgt_wav, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                       window=win, return_complex=True).abs()
        # Spectral convergence (Frobenius) + log-magnitude L1
        sc = torch.linalg.norm(T - P) / (torch.linalg.norm(T) + 1e-7)
        logmag = F.l1_loss(torch.log1p(P), torch.log1p(T))
        # HF band (>hf_hz) emphasis — the band that collapses
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / sr).to(pred_wav.device)
        hf = (freqs > hf_hz).float().view(1, -1, 1)
        if energy_adaptive:
            w = hf / torch.sqrt(T + 1e-5)
            hf_term = (((P - T) * w).abs()).mean()
        else:
            hf_term = F.l1_loss(P * hf, T * hf)
        total = total + sc + logmag + hf_weight * hf_term
    return total / len(ffts)


def visual_dropout_curriculum(base_prob, step, start_step, total_steps,
                              vd_curriculum_ratio=0.0):
    """Ramp visual dropout from low (sync-focused) to base_prob (spectral-focused).

    When vd_curriculum_ratio > 0, dropout starts at 10% of base_prob and linearly
    ramps to base_prob at vd_curriculum_ratio * total_steps. After that, stays at
    base_prob. When vd_curriculum_ratio == 0, returns base_prob (disabled).
    """
    if vd_curriculum_ratio <= 0 or base_prob <= 0:
        return base_prob
    progress = (step - start_step) / max(total_steps, 1)
    if progress >= vd_curriculum_ratio:
        return base_prob
    ramp = progress / vd_curriculum_ratio
    return 0.1 * base_prob + ramp * 0.9 * base_prob


# -- Loss computation --------------------------------------------------------

def flow_matching_loss(model, x1, t, clip_feat, sync_feat, text_feat, device, dtype,
                       visual_dropout_prob=0.0, min_snr_gamma=0.0,
                       cos_sim_weight=0.0, channel_weights=None,
                       temporal_variance_weight=0.0, tv_gate_sigma=0.3,
                       tv_scales=(1, 4, 16),
                       spectral_weight=0.0,
                       dac_model=None, wav_spectral_weight=0.0,
                       wav_spectral_crop=64, wav_spectral_adaptive=True,
                       compute_wav_spectral=False,
                       cfm_lambda=0.0):
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

    # Contrastive Flow Matching (ΔFM, arXiv:2506.05350): subtract λ × MSE to a
    # DIFFERENT batch sample's velocity target. Pushes the predicted flow away
    # from other samples' flows, preventing the conditional-mean collapse that
    # over-smooths high-capacity fits. Reduces to plain FM at λ=0.
    if cfm_lambda > 0 and B > 1:
        idx = torch.roll(torch.arange(B, device=v_pred.device), shifts=1)  # idx[i] != i
        mse_neg = F.mse_loss(v_pred, v_target[idx], reduction='none')
        mse_unreduced = mse_unreduced - cfm_lambda * mse_neg

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
        # SNR-gated multi-scale temporal difference loss.
        # Gate: only fire at low noise where temporal structure is visible.
        gate = torch.clamp(1.0 - t / tv_gate_sigma, min=0.0)  # [B], 1 at t=0, 0 at t>=gate_sigma
        tv_loss = torch.tensor(0.0, device=device, dtype=dtype)
        for s in tv_scales:
            if s == 1:
                dp = torch.diff(v_pred, dim=-1)
                dt = torch.diff(v_target, dim=-1)
            else:
                pp = F.avg_pool1d(v_pred, kernel_size=s, stride=s)
                pt = F.avg_pool1d(v_target, kernel_size=s, stride=s)
                dp = torch.diff(pp, dim=-1)
                dt = torch.diff(pt, dim=-1)
            per_sample = F.mse_loss(dp, dt, reduction='none').mean(dim=(1, 2))
            tv_loss = tv_loss + (gate * per_sample).mean()
        tv_loss = tv_loss / len(tv_scales)
        loss = loss + temporal_variance_weight * tv_loss

    if spectral_weight > 0:
        x1_pred = xt - t_expand.to(dtype=dtype) * v_pred
        spec_loss = multi_resolution_spectral_loss(
            x1_pred, x1.to(device=device, dtype=dtype),
        )
        loss = loss + spectral_weight * spec_loss

    # Waveform-domain spectral loss: reconstruct predicted clean latent, decode a
    # cropped window through DAC (differentiable), and penalise >4kHz error on the
    # real 48kHz waveform. Only runs on flagged steps (decode is expensive).
    if compute_wav_spectral and wav_spectral_weight > 0 and dac_model is not None:
        dac_model.to(device=device)  # idempotent; eval/ref decode may have offloaded it
        dac_dtype = next(dac_model.parameters()).dtype
        x1_pred = xt - t_expand.to(dtype=dtype) * v_pred  # [B,128,T]
        x1_tgt = x1.to(device=device, dtype=dtype)
        T_lat = x1_pred.shape[-1]
        crop = min(wav_spectral_crop, T_lat)
        if T_lat > crop:
            s = int(torch.randint(0, T_lat - crop + 1, (1,)).item())
            x1_pred_c = x1_pred[..., s:s + crop]
            x1_tgt_c = x1_tgt[..., s:s + crop]
        else:
            x1_pred_c, x1_tgt_c = x1_pred, x1_tgt
        # Decode in DAC's native dtype (preserve grad), STFT in fp32 (cuFFT needs it)
        pred_wav = dac_model.decode(x1_pred_c.to(dac_dtype)).squeeze(1)  # [B, samples], grad
        with torch.no_grad():
            tgt_wav = dac_model.decode(x1_tgt_c.to(dac_dtype)).squeeze(1)
        wav_loss = waveform_spectral_loss(
            pred_wav.float(), tgt_wav.float(),
            energy_adaptive=wav_spectral_adaptive,
        )
        loss = loss + wav_spectral_weight * wav_loss.to(loss.dtype)

    return loss


# -- Eval sample generation --------------------------------------------------

@torch.no_grad()
def generate_eval_sample(model, dac_model, dataset_entry, device, dtype,
                         num_steps=50, seed=42, cfg_scale=4.5):
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
