import torch

from lora.event import (
    event_envelope_from_latents,
    event_envelope_from_sync,
    normalize_event_envelope,
    zero_event_envelope,
)


def test_event_envelope_from_latents_shape_and_normalization():
    latents = torch.randn(3, 128, 50)
    env = event_envelope_from_latents(latents)
    assert env.shape == (3, 1, 50)
    assert torch.isfinite(env).all()
    assert env.abs().max() <= 3.0


def test_event_envelope_from_sync_resamples_to_target_len():
    sync = torch.randn(2, 16, 8)
    env = event_envelope_from_sync(sync, target_len=40)
    assert env.shape == (2, 1, 40)


def test_zero_event_envelope_is_neutral():
    env = zero_event_envelope(2, 12, device="cpu", dtype=torch.float32)
    assert env.shape == (2, 1, 12)
    assert torch.count_nonzero(env) == 0


def test_normalize_event_envelope_accepts_2d():
    raw = torch.rand(2, 20)
    env = normalize_event_envelope(raw)
    assert env.shape == (2, 1, 20)
