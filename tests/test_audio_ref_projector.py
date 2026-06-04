"""Tests for the learned reference-audio projector (B2 scaffold)."""
import torch
import pytest

from lora.audio_ref_projector import (
    AudioRefProjector, save_projector, load_projector, append_ref_tokens,
)


def test_output_shape():
    p = AudioRefProjector(in_dim=512, cond_dim=768, k_tokens=1)
    out = p(torch.randn(4, 512))
    assert out.shape == (4, 1, 768)


def test_k_tokens_shape():
    p = AudioRefProjector(in_dim=512, cond_dim=768, k_tokens=3)
    out = p(torch.randn(2, 512))
    assert out.shape == (2, 3, 768)


def test_zero_init_is_noop():
    # Final layer is zero-initialized -> untrained projector emits exactly zero tokens.
    p = AudioRefProjector(in_dim=512, cond_dim=768, k_tokens=2)
    out = p(torch.randn(5, 512))
    assert torch.count_nonzero(out) == 0


def test_accepts_1d_input():
    p = AudioRefProjector()
    out = p(torch.randn(512))
    assert out.shape == (1, 1, 768)


def test_save_load_roundtrip(tmp_path):
    p = AudioRefProjector(in_dim=512, cond_dim=768, hidden=256, k_tokens=2)
    # perturb weights so it isn't all-zero
    with torch.no_grad():
        for param in p.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    x = torch.randn(3, 512)
    ref_out = p(x)

    path = str(tmp_path / "proj.pt")
    save_projector(p, path, meta={"step": 123})
    loaded, meta = load_projector(path)
    assert meta["step"] == 123
    assert loaded.config == p.config
    assert torch.allclose(loaded(x), ref_out, atol=1e-6)


def test_append_ref_tokens_shapes():
    B, T, D, k = 2, 6, 768, 3
    text = torch.randn(B, T, D)
    uncond = torch.randn(B, T, D)
    tokens = torch.randn(B, k, D)
    t_out, u_out = append_ref_tokens(text, uncond, tokens)
    assert t_out.shape == (B, T + k, D)
    assert u_out.shape == (B, T + k, D)
    # original preserved; uncond's appended block is zeros
    assert torch.equal(t_out[:, :T], text)
    assert torch.count_nonzero(u_out[:, T:]) == 0


def test_append_ref_tokens_broadcast():
    B, T, D, k = 4, 5, 768, 1
    text = torch.randn(B, T, D)
    uncond = torch.randn(B, T, D)
    tokens = torch.randn(1, k, D)  # single token broadcast across batch
    t_out, _ = append_ref_tokens(text, uncond, tokens)
    assert t_out.shape == (B, T + k, D)
    # the appended token is identical across the batch
    assert torch.allclose(t_out[0, T:], t_out[1, T:])
