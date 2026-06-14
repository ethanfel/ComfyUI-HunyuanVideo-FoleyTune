"""Event-envelope helpers for Foley LoRA conditioning.

The adapter consumes a normalized one-channel curve at the audio latent frame
rate. During training the curve comes from target audio latents; during
inference it can be approximated from visual/sync feature motion.
"""

import torch
import torch.nn.functional as F


def normalize_event_envelope(envelope: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return a per-sample z-scored event envelope with shape [B, 1, T]."""
    if envelope.ndim == 3:
        if envelope.shape[1] == 1:
            envelope = envelope[:, 0, :]
        else:
            envelope = envelope.float().mean(dim=1)
    elif envelope.ndim != 2:
        raise ValueError(f"event envelope must be [B,T] or [B,1,T], got {tuple(envelope.shape)}")

    env = torch.log1p(envelope.float().clamp_min(0.0))
    mean = env.mean(dim=-1, keepdim=True)
    std = env.std(dim=-1, keepdim=True).clamp_min(eps)
    env = ((env - mean) / std).clamp(-3.0, 3.0)
    return env.unsqueeze(1)


def event_envelope_from_latents(latents: torch.Tensor) -> torch.Tensor:
    """Build a target event curve from DAC latents [B, C, T]."""
    raw = latents.float().pow(2).mean(dim=1).sqrt()
    return normalize_event_envelope(raw)


def event_envelope_from_sync(sync_feat: torch.Tensor, target_len: int) -> torch.Tensor:
    """Build an inference-time event proxy from SyncFormer feature motion."""
    if sync_feat.ndim != 3:
        raise ValueError(f"sync features must be [B,T,C], got {tuple(sync_feat.shape)}")
    diff = torch.diff(sync_feat.float(), dim=1, prepend=sync_feat[:, :1, :])
    raw = diff.pow(2).mean(dim=-1).sqrt().unsqueeze(1)
    if raw.shape[-1] != target_len:
        raw = F.interpolate(raw, size=target_len, mode="linear", align_corners=False)
    return normalize_event_envelope(raw)


def zero_event_envelope(batch_size: int, target_len: int, device, dtype) -> torch.Tensor:
    """Return a neutral normalized envelope [B, 1, T]."""
    return torch.zeros(batch_size, 1, target_len, device=device, dtype=dtype)
