"""Tests for LoRAConv1d — conv-LoRA used to adapt the single-stream blocks'
ChannelLastConv1d / ConvMLP layers (the previously-unreachable back 2/3)."""
import math
import torch
import torch.nn as nn

from lora.lora import (
    LoRAConv1d, apply_lora, get_lora_state_dict, load_lora, FOLEY_TARGET_PRESETS,
    remove_lora,
)


class ChannelLastConv1d(nn.Conv1d):
    """Mirror of the model's channel-last conv ([B,T,C] I/O)."""
    def forward(self, x):
        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)


def test_conv_lora_noop_at_init_channel_last():
    base = ChannelLastConv1d(16, 24, kernel_size=3, padding=1, bias=False)
    lora = LoRAConv1d(base, rank=4, alpha=4)
    x = torch.randn(2, 10, 16)  # [B, T, C]
    out = lora(x)
    assert out.shape == (2, 10, 24)
    torch.testing.assert_close(out, base(x))  # lora_B zero-init => exact no-op


def test_conv_lora_noop_channel_first():
    base = nn.Conv1d(16, 16, kernel_size=3, padding=1)
    lora = LoRAConv1d(base, rank=4, alpha=4)
    assert not lora.channel_last
    x = torch.randn(2, 16, 10)  # [B, C, T]
    torch.testing.assert_close(lora(x), base(x))


def test_conv_lora_output_length_matches_base():
    for stride, pad in [(1, 1), (1, 0), (2, 1)]:
        base = ChannelLastConv1d(8, 8, kernel_size=3, stride=stride, padding=pad)
        lora = LoRAConv1d(base, rank=2, alpha=2)
        x = torch.randn(1, 9, 8)
        assert lora(x).shape == base(x).shape, (stride, pad)


def test_conv_lora_grad_flows_base_frozen():
    base = ChannelLastConv1d(16, 16, kernel_size=3, padding=1)
    lora = LoRAConv1d(base, rank=4, alpha=8)
    nn.init.normal_(lora.lora_B, std=0.1)  # break the no-op so output depends on LoRA
    x = torch.randn(2, 10, 16)
    assert not torch.allclose(lora(x), base(x))
    lora(x).sum().backward()
    assert lora.lora_A.grad is not None and lora.lora_B.grad is not None
    assert base.weight.grad is None  # base must stay frozen


def test_conv_lora_state_dict_roundtrip():
    base = ChannelLastConv1d(8, 8, kernel_size=3, padding=1)
    m = nn.Module(); m.layer = LoRAConv1d(base, rank=2, alpha=2)
    sd = get_lora_state_dict(m)
    assert any("lora_A" in k for k in sd) and any("lora_B" in k for k in sd)
    assert not any("base.weight" in k for k in sd)  # base excluded
    # mutate then reload
    with torch.no_grad():
        m.layer.lora_B.add_(1.0)
    target = nn.Module(); target.layer = LoRAConv1d(ChannelLastConv1d(8, 8, kernel_size=3, padding=1), rank=2, alpha=2)
    load_lora(target, sd)
    torch.testing.assert_close(target.layer.lora_A, m.layer.lora_A)


def test_apply_lora_wraps_conv_by_suffix():
    class Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_qkv = nn.Linear(8, 24)
            self.linear1 = ChannelLastConv1d(8, 8, kernel_size=3, padding=1)
    root = nn.Module(); root.single_blocks = nn.ModuleList([Blk(), Blk()])
    n = apply_lora(root, rank=2, alpha=2, target_suffixes=("linear_qkv", "linear1"))
    assert n == 4  # 2 blocks x (linear_qkv + linear1)
    assert isinstance(root.single_blocks[0].linear1, LoRAConv1d)
    from lora.lora import LoRALinear
    assert isinstance(root.single_blocks[0].linear_qkv, LoRALinear)


def test_remove_lora_unwraps_conv_and_linear():
    class Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_qkv = nn.Linear(8, 24)
            self.linear1 = ChannelLastConv1d(8, 8, kernel_size=3, padding=1)

    root = nn.Module()
    root.block = Blk()
    apply_lora(root, rank=2, alpha=2, target_suffixes=("linear_qkv", "linear1"))
    assert isinstance(root.block.linear1, LoRAConv1d)

    n = remove_lora(root)
    assert n == 2
    assert isinstance(root.block.linear1, ChannelLastConv1d)
    assert isinstance(root.block.linear_qkv, nn.Linear)


def test_all_blocks_conv_preset_has_conv_suffixes():
    p = FOLEY_TARGET_PRESETS["all_blocks_conv"]
    for s in ("linear1", "linear2.w1", "linear2.w2", "linear2.w3"):
        assert s in p


def test_all_attn_mlp_sync_io_preset_has_io_suffixes():
    p = FOLEY_TARGET_PRESETS["all_attn_mlp_sync_io"]
    for s in (
        "audio_embedder.proj",
        "visual_proj.w1",
        "visual_proj.w2",
        "visual_proj.w3",
        "cond_in.linear_1",
        "cond_in.linear_2",
        "final_layer.linear",
        "final_layer.adaLN_modulation.1",
    ):
        assert s in p
