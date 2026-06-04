"""Tests for the reference-audio conditioning token bridge (utils._append_reference_token).

The helper zero-pads a CLAP audio embedding (512-d shared space) into the model's text
hidden space, scales it to a real-text-token norm * strength, and appends it as one extra
cross-attention key/value token (with a neutral token on the unconditional branch so the
two CFG halves keep matching length).
"""
import sys
import types

import pytest
import torch


def _stub_missing_modules():
    """utils.py (repo root) imports comfy + diffusers at module load; stub them so the
    pure-tensor helper can be imported without a full ComfyUI/diffusers install."""
    if "comfy" not in sys.modules:
        sys.modules["comfy"] = types.ModuleType("comfy")
    if "comfy.utils" not in sys.modules:
        m = types.ModuleType("comfy.utils")
        m.load_torch_file = lambda *a, **k: None

        class ProgressBar:  # minimal stand-in
            def __init__(self, *a, **k):
                pass

            def update(self, *a, **k):
                pass

        m.ProgressBar = ProgressBar
        sys.modules["comfy.utils"] = m
    if "comfy.model_management" not in sys.modules:
        m = types.ModuleType("comfy.model_management")
        m.throw_exception_if_processing_interrupted = lambda *a, **k: None
        sys.modules["comfy.model_management"] = m
    if "diffusers" not in sys.modules:
        sys.modules["diffusers"] = types.ModuleType("diffusers")
    if "diffusers.utils" not in sys.modules:
        sys.modules["diffusers.utils"] = types.ModuleType("diffusers.utils")
    if "diffusers.utils.torch_utils" not in sys.modules:
        m = types.ModuleType("diffusers.utils.torch_utils")
        m.randn_tensor = lambda *a, **k: None
        sys.modules["diffusers.utils.torch_utils"] = m


_stub_missing_modules()

from utils import _append_reference_token  # noqa: E402


def _make_text(B=2, T=6, D=768, n_real=4):
    """[B,T,D] with n_real non-zero (real) tokens followed by zero pad tokens."""
    text = torch.zeros(B, T, D)
    text[:, :n_real, :] = torch.randn(B, n_real, D)
    return text


def test_noop_when_embed_none():
    text = _make_text()
    uncond = _make_text()
    t_out, u_out = _append_reference_token(text, uncond, None, 0.5)
    assert t_out is text and u_out is uncond


def test_noop_when_strength_zero():
    text = _make_text()
    uncond = _make_text()
    t_out, u_out = _append_reference_token(text, uncond, torch.randn(512), 0.0)
    assert t_out is text and u_out is uncond
    t_out, u_out = _append_reference_token(text, uncond, torch.randn(512), -1.0)
    assert t_out is text and u_out is uncond


def test_appends_one_token_each_branch():
    B, T, D = 2, 6, 768
    text = _make_text(B, T, D)
    uncond = _make_text(B, T, D)
    ref = torch.randn(512)
    t_out, u_out = _append_reference_token(text, uncond, ref, 0.2)
    assert t_out.shape == (B, T + 1, D)
    assert u_out.shape == (B, T + 1, D)
    # original tokens untouched
    assert torch.equal(t_out[:, :T, :], text)
    assert torch.equal(u_out[:, :T, :], uncond)


def test_uncond_appended_token_is_zero():
    text = _make_text()
    uncond = _make_text()
    t_out, u_out = _append_reference_token(text, uncond, torch.randn(512), 0.3)
    assert torch.count_nonzero(u_out[:, -1, :]) == 0


def test_appended_token_norm_matches_strength_times_mean_real_norm():
    B, T, D, n_real = 2, 8, 768, 5
    text = _make_text(B, T, D, n_real)
    uncond = _make_text(B, T, D, n_real)
    strength = 0.25
    ref = torch.randn(512)
    t_out, _ = _append_reference_token(text, uncond, ref, strength)

    # expected target norm = mean norm over the REAL (non-pad) tokens * strength
    token_norms = text.norm(dim=-1)
    real = (token_norms > 1e-6).float()
    mean_real_norm = (token_norms * real).sum() / real.sum()
    expected = (mean_real_norm * strength).item()

    appended_norm = t_out[0, -1, :].norm().item()
    assert appended_norm == pytest.approx(expected, rel=1e-4)


def test_only_first_512_dims_populated():
    # The 512-d CLAP embedding is zero-padded into the 768-d text space, so dims [512:768]
    # of the (pre-scaling) token are zero — and scaling preserves that.
    B, T, D = 1, 4, 768
    text = _make_text(B, T, D)
    uncond = _make_text(B, T, D)
    ref = torch.randn(512)
    t_out, _ = _append_reference_token(text, uncond, ref, 0.5)
    appended = t_out[0, -1, :]
    assert torch.count_nonzero(appended[512:]) == 0
    assert torch.count_nonzero(appended[:512]) > 0


def test_strength_scales_linearly():
    text = _make_text()
    uncond = _make_text()
    ref = torch.randn(512)
    t_low, _ = _append_reference_token(text, uncond, ref, 0.1)
    t_high, _ = _append_reference_token(text, uncond, ref, 0.2)
    n_low = t_low[0, -1, :].norm().item()
    n_high = t_high[0, -1, :].norm().item()
    assert n_high == pytest.approx(2.0 * n_low, rel=1e-4)


# -- clap_centroid (multi-clip averaging) -------------------------------------

from hunyuanvideo_foley.utils.feature_utils import clap_centroid  # noqa: E402


def test_centroid_output_is_unit_norm():
    embs = [torch.nn.functional.normalize(torch.randn(1, 512), dim=-1) for _ in range(4)]
    c = clap_centroid(embs)
    assert c.shape == (1, 512)
    assert c.norm().item() == pytest.approx(1.0, abs=1e-5)


def test_centroid_of_identical_is_identity():
    e = torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
    c = clap_centroid([e, e, e])
    assert torch.allclose(c, e, atol=1e-5)


def test_centroid_accepts_tensor_and_list_equivalently():
    embs = [torch.nn.functional.normalize(torch.randn(1, 512), dim=-1) for _ in range(3)]
    from_list = clap_centroid(embs)
    from_tensor = clap_centroid(torch.cat(embs, dim=0))  # [3, 512]
    assert torch.allclose(from_list, from_tensor, atol=1e-6)


def test_centroid_single_embed_normalized():
    e = torch.randn(1, 512) * 3.0  # non-unit input
    c = clap_centroid([e])
    assert c.norm().item() == pytest.approx(1.0, abs=1e-5)
    # direction preserved
    assert torch.allclose(c, torch.nn.functional.normalize(e, dim=-1), atol=1e-5)


# -- learned-projector branch of _append_reference_token ----------------------

def _perturbed_projector(k_tokens=2):
    from lora.audio_ref_projector import AudioRefProjector
    p = AudioRefProjector(in_dim=512, cond_dim=768, hidden=128, k_tokens=k_tokens)
    with torch.no_grad():  # move off the zero-init so it emits non-zero tokens
        for param in p.parameters():
            param.add_(torch.randn_like(param) * 0.02)
    return p


def test_append_with_projector_appends_k_tokens():
    B, T, D, k = 2, 6, 768, 2
    text = _make_text(B, T, D)
    uncond = _make_text(B, T, D)
    ref = torch.nn.functional.normalize(torch.randn(512), dim=-1)
    proj = _perturbed_projector(k_tokens=k)
    t_out, u_out = _append_reference_token(text, uncond, ref, 1.0, projector=proj)
    assert t_out.shape == (B, T + k, D)
    assert u_out.shape == (B, T + k, D)
    assert torch.equal(t_out[:, :T], text)                 # originals preserved
    assert torch.count_nonzero(u_out[:, T:]) == 0          # uncond block neutral
    assert torch.count_nonzero(t_out[:, T:]) > 0           # learned tokens non-zero


def test_append_with_projector_strength_zero_is_noop():
    text = _make_text()
    uncond = _make_text()
    proj = _perturbed_projector()
    t_out, u_out = _append_reference_token(text, uncond, torch.randn(512), 0.0, projector=proj)
    assert t_out is text and u_out is uncond


def test_append_with_projector_strength_scales():
    text = _make_text()
    uncond = _make_text()
    ref = torch.nn.functional.normalize(torch.randn(512), dim=-1)
    proj = _perturbed_projector(k_tokens=1)
    t_lo, _ = _append_reference_token(text, uncond, ref, 0.5, projector=proj)
    t_hi, _ = _append_reference_token(text, uncond, ref, 1.0, projector=proj)
    assert torch.allclose(t_hi[:, -1], 2.0 * t_lo[:, -1], atol=1e-5)
