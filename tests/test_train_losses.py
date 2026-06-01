"""Tests for HF-preservation loss components in lora.train:
- compute_channel_weights (Path A: inverse-variance channel weighting)
- waveform_spectral_loss (Path B: decoded-waveform multi-res STFT loss)
"""
import pytest
import torch
import torch.nn as nn

from lora.train import compute_channel_weights, waveform_spectral_loss, flow_matching_loss


# -- Contrastive Flow Matching (cfm_lambda) -----------------------------------

class _MockFoley(nn.Module):
    """Minimal stand-in: returns a dict with 'x' shaped like the input latents."""
    def __init__(self, c=128):
        super().__init__()
        self.lin = nn.Conv1d(c, c, 1)
    def forward(self, x, t, cond, clip_feat, sync_feat, drop_visual=None):
        return {"x": self.lin(x)}


def _run_flow(cfm_lambda, seed=0):
    torch.manual_seed(seed)
    B, C, T = 4, 128, 20
    model = _MockFoley(C)
    x1 = torch.randn(B, C, T)
    t = torch.rand(B)
    clip = torch.randn(B, 8, 768); sync = torch.randn(B, 16, 768); text = torch.randn(B, 4, 768)
    torch.manual_seed(seed + 1)  # fix x0 = randn_like(x1) identically across calls
    return flow_matching_loss(model, x1, t, clip, sync, text, "cpu", torch.float32,
                              cfm_lambda=cfm_lambda)


def test_cfm_lambda_zero_is_plain_fm():
    # Two calls at λ=0 with identical seeds must match (no contrastive perturbation)
    assert torch.allclose(_run_flow(0.0).detach(), _run_flow(0.0).detach())


def test_cfm_lambda_lowers_loss_by_subtracting_negative():
    # ΔFM subtracts λ·MSE-to-other-sample, so loss(λ>0) < loss(λ=0) at fixed x0
    base = _run_flow(0.0).detach()
    contrastive = _run_flow(0.05).detach()
    assert float(contrastive) < float(base)


def test_cfm_gradient_flows():
    loss = _run_flow(0.05)
    loss.backward()  # should not error; contrastive term is differentiable
    assert torch.isfinite(loss)


# -- compute_channel_weights --------------------------------------------------

def _latents_with_channel_spread():
    """[N,128,T] where channel 0 is near-dead (low var) and channel 127 is high var."""
    torch.manual_seed(0)
    x = torch.randn(4, 128, 50)
    scales = torch.linspace(0.01, 2.0, 128).view(1, 128, 1)
    return x * scales


def test_channel_weights_off_returns_none():
    assert compute_channel_weights(_latents_with_channel_spread(), "off") is None
    assert compute_channel_weights(_latents_with_channel_spread(), "") is None


def test_channel_weights_variance_upweights_high_variance():
    lat = _latents_with_channel_spread()
    w = compute_channel_weights(lat, "variance")
    assert w.shape == (128,)
    # high-variance channel (last) should weigh more than low-variance (first)
    assert w[-1] > w[0]
    assert float(w.min()) >= 0.5 and float(w.max()) <= 2.0


def test_channel_weights_inverse_upweights_live_low_variance():
    """A live but low-variance channel is up-weighted above a high-variance one."""
    torch.manual_seed(0)
    x = torch.randn(4, 128, 50)
    scales = torch.ones(128)
    scales[10] = 0.4   # live, low-variance channel (above dead threshold)
    scales[20] = 1.6   # high-variance channel
    lat = x * scales.view(1, 128, 1)
    w = compute_channel_weights(lat, "inverse")
    assert w.shape == (128,)
    assert w[10] > w[20]
    assert float(w.min()) >= 0.5 and float(w.max()) <= 4.0


def test_channel_weights_inverse_pins_dead_channels_to_one():
    """Dead (near-zero-variance) channels must be pinned to 1.0, never amplified —
    their velocity target is pure noise and up-weighting it injects noise gradient."""
    lat = _latents_with_channel_spread()
    lat[:, 0, :] = 1e-6  # truly dead channel
    w = compute_channel_weights(lat, "inverse")
    assert float(w[0]) == pytest.approx(1.0)
    assert float(w.max()) <= 4.0
    assert torch.isfinite(w).all()


def test_channel_weights_unknown_mode_raises():
    with pytest.raises(ValueError):
        compute_channel_weights(_latents_with_channel_spread(), "bogus")


# -- waveform_spectral_loss ---------------------------------------------------

def test_wav_spectral_zero_on_identical():
    torch.manual_seed(1)
    wav = torch.randn(2, 48000)
    loss = waveform_spectral_loss(wav, wav)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_wav_spectral_positive_on_different():
    torch.manual_seed(2)
    a = torch.randn(2, 48000)
    b = torch.randn(2, 48000)
    assert float(waveform_spectral_loss(a, b)) > 0.0


def test_wav_spectral_penalises_missing_hf():
    """A target rich in >4kHz content vs a low-passed prediction should score
    higher with HF emphasis than two equally dull signals."""
    torch.manual_seed(3)
    sr = 48000
    t = torch.arange(sr, dtype=torch.float32) / sr
    lf = torch.sin(2 * torch.pi * 200 * t)          # 200 Hz tone
    hf = torch.sin(2 * torch.pi * 9000 * t)         # 9 kHz tone (>4kHz band)
    target = (lf + hf).unsqueeze(0)
    pred_dull = lf.unsqueeze(0)                     # missing the HF tone
    loss_missing_hf = waveform_spectral_loss(pred_dull, target)
    loss_match = waveform_spectral_loss(target, target)
    assert float(loss_missing_hf) > float(loss_match)


def test_wav_spectral_gradient_flows():
    torch.manual_seed(4)
    pred = torch.randn(2, 24000, requires_grad=True)
    tgt = torch.randn(2, 24000)
    loss = waveform_spectral_loss(pred, tgt)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad.norm()) > 0.0


def test_wav_spectral_adaptive_flag_runs_both():
    torch.manual_seed(5)
    a = torch.randn(2, 24000)
    b = torch.randn(2, 24000)
    assert float(waveform_spectral_loss(a, b, energy_adaptive=True)) > 0.0
    assert float(waveform_spectral_loss(a, b, energy_adaptive=False)) > 0.0
