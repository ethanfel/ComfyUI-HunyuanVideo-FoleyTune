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


# -- harmonize_dataset (multi-directory length alignment) ----------------------

def _entry(lat_t, clip_t, sync_t, name="c"):
    return {
        "latents": torch.randn(1, 128, lat_t),
        "clip_features": torch.randn(1, clip_t, 768),
        "sync_features": torch.randn(1, sync_t, 768),
        "text_embedding": torch.randn(1, 12, 768),
        "prompt": "p", "name": name,
    }


def test_harmonize_noop_on_uniform_lengths():
    from lora.train import harmonize_dataset
    ds = [_entry(500, 80, 200, f"c{i}") for i in range(3)]
    out = harmonize_dataset(ds)
    assert all(d["latents"].shape[-1] == 500 for d in out)
    assert all(d["clip_features"].shape[1] == 80 for d in out)
    assert all(d["sync_features"].shape[1] == 200 for d in out)


def test_harmonize_truncates_cross_dir_mismatch():
    from lora.train import harmonize_dataset
    # dir A: 10s clips, dir B: 8s clips — previously crashed torch.cat at batch time
    ds = [_entry(500, 80, 200, "a"), _entry(400, 64, 160, "b")]
    out = harmonize_dataset(ds)
    assert all(d["latents"].shape[-1] == 400 for d in out)
    assert all(d["clip_features"].shape[1] == 64 for d in out)
    assert all(d["sync_features"].shape[1] == 160 for d in out)
    # batch assembly must now work
    torch.cat([d["latents"] for d in out])
    torch.cat([d["clip_features"] for d in out])
    torch.cat([d["sync_features"] for d in out])


def test_harmonize_keeps_sync_multiple_of_8():
    from lora.train import harmonize_dataset
    ds = [_entry(500, 80, 168, "a"), _entry(480, 78, 164, "b")]
    out = harmonize_dataset(ds)
    assert all(d["sync_features"].shape[1] % 8 == 0 for d in out)


# -- reference_metrics gain alignment ------------------------------------------

def test_reference_metrics_gain_invariant():
    import numpy as np
    from lora.spectral_metrics import reference_metrics
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(48000).astype(np.float64) * 0.05
    gen = ref + rng.standard_normal(48000) * 0.005
    m_matched = reference_metrics(gen, ref, 48000)
    # A pure level offset (e.g. -27 dBFS eval norm vs natural-level reference)
    # must not change the spectral-match metrics
    m_scaled = reference_metrics(gen * 0.1, ref, 48000)
    for k in ("log_spectral_distance_db", "spectral_convergence",
              "mel_cepstral_distortion", "per_band_correlation"):
        assert abs(m_matched[k] - m_scaled[k]) < 1e-6, k


# -- sample_timesteps t_range_mode ---------------------------------------------

def test_t_range_clamp_has_boundary_point_masses():
    from lora.train import sample_timesteps
    torch.manual_seed(0)
    t = sample_timesteps(20000, "uniform", "cpu", torch.float32,
                         t_min=0.05, t_max=0.95, t_range_mode="clamp")
    assert float(t.min()) >= 0.05 and float(t.max()) <= 0.95
    # ~5% of uniform draws land below 0.05 and get clipped to exactly 0.05
    assert float((t == 0.05).float().mean()) > 0.03
    assert float((t == 0.95).float().mean()) > 0.03


def test_t_range_rescale_removes_point_masses():
    from lora.train import sample_timesteps
    torch.manual_seed(0)
    t = sample_timesteps(20000, "uniform", "cpu", torch.float32,
                         t_min=0.05, t_max=0.95, t_range_mode="rescale")
    assert float(t.min()) >= 0.05 and float(t.max()) <= 0.95
    assert float((t == 0.05).float().mean()) < 0.001
    assert float((t == 0.95).float().mean()) < 0.001
    # still uniform-ish over the window: mean near the midpoint
    assert abs(float(t.mean()) - 0.5) < 0.01


def test_t_range_default_is_clamp():
    from lora.train import sample_timesteps
    torch.manual_seed(0)
    a = sample_timesteps(1000, "uniform", "cpu", torch.float32, t_min=0.05, t_max=0.95)
    torch.manual_seed(0)
    b = sample_timesteps(1000, "uniform", "cpu", torch.float32,
                         t_min=0.05, t_max=0.95, t_range_mode="clamp")
    assert torch.equal(a, b)
