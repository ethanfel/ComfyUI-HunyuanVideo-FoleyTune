"""Tests for HF-preservation loss components in lora.train:
- compute_channel_weights (Path A: inverse-variance channel weighting)
- waveform_spectral_loss (Path B: decoded-waveform multi-res STFT loss)
"""
import pytest
import torch

from lora.train import compute_channel_weights, waveform_spectral_loss


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


def test_channel_weights_inverse_upweights_low_variance():
    lat = _latents_with_channel_spread()
    w = compute_channel_weights(lat, "inverse")
    assert w.shape == (128,)
    # inverse mode must up-weight the LOW-variance (HF-carrying) channel
    assert w[0] > w[-1]
    assert float(w.min()) >= 0.5 and float(w.max()) <= 4.0


def test_channel_weights_inverse_clamps_dead_channels():
    """A near-zero-variance channel must not blow up past the clamp ceiling."""
    lat = _latents_with_channel_spread()
    lat[:, 0, :] = 1e-6  # truly dead channel
    w = compute_channel_weights(lat, "inverse")
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
