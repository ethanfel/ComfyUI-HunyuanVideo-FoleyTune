import sys
import types

import torch
import torch.nn as nn

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


class _EvalStrengthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_event_strengths = []
        self._event_strength = 0.37

    def get_empty_clip_sequence(self, bs=1, len=1):
        return torch.zeros(bs, len, 768)

    def get_empty_sync_sequence(self, bs=1, len=8):
        return torch.zeros(bs, len, 768)

    def forward(self, x, t, cond, clip_feat, sync_feat,
                event_envelope=None, event_strength=1.0, **kwargs):
        self.seen_event_strengths.append(float(event_strength))
        return {"x": torch.zeros_like(x)}


class _EvalStrengthDac(nn.Module):
    sample_rate = 48000

    def decode(self, latents):
        return torch.zeros(latents.shape[0], 1, latents.shape[-1], device=latents.device)


def test_generate_eval_sample_uses_model_event_strength(monkeypatch):
    class _Scheduler:
        init_noise_sigma = 1.0

        def __init__(self, *args, **kwargs):
            self.timesteps = None

        def set_timesteps(self, num_steps, device):
            self.timesteps = torch.arange(num_steps, device=device)

        def step(self, model_output, t, sample):
            return (sample,)

    fake_schedulers = types.SimpleNamespace(FlowMatchDiscreteScheduler=_Scheduler)
    monkeypatch.setitem(sys.modules, "hunyuanvideo_foley.utils.schedulers", fake_schedulers)

    from lora.train import generate_eval_sample

    model = _EvalStrengthModel()
    dac = _EvalStrengthDac()
    entry = {
        "latents": torch.zeros(1, 128, 4),
        "clip_features": torch.zeros(1, 2, 768),
        "sync_features": torch.zeros(1, 8, 768),
        "text_embedding": torch.zeros(1, 3, 768),
        "event_envelope": torch.zeros(1, 1, 4),
    }

    generate_eval_sample(
        model, dac, entry, torch.device("cpu"), torch.bfloat16,
        num_steps=1, cfg_scale=1.0,
    )

    assert model.seen_event_strengths == [0.37]
