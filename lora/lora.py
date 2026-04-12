"""LoRA layer primitives for HunyuanVideo-Foley.

Provides LoRALinear wrapping, model injection, save/load, and spectral surgery.
Ported from SelVA's lora.py with Foley-specific target presets.
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ── Foley target presets ────────────────────────────────────────────────────

FOLEY_TARGET_PRESETS = {
    "audio_attn": (
        "audio_self_attn_qkv",
        "audio_self_proj",
    ),
    "audio_cross": (
        "audio_self_attn_qkv",
        "audio_self_proj",
        "audio_cross_q",
        "audio_cross_proj",
        "text_cross_kv",
    ),
    "all_attn": (
        "audio_self_attn_qkv",
        "audio_self_proj",
        "audio_cross_q",
        "audio_cross_proj",
        "text_cross_kv",
        "v_cond_attn_qkv",
        "v_cond_self_proj",
        "v_cond_cross_q",
        "v_cond_cross_proj",
    ),
    "all_attn_mlp": (
        "audio_self_attn_qkv",
        "audio_self_proj",
        "audio_cross_q",
        "audio_cross_proj",
        "text_cross_kv",
        "v_cond_attn_qkv",
        "v_cond_self_proj",
        "v_cond_cross_q",
        "v_cond_cross_proj",
        "audio_mlp.fc1",
        "audio_mlp.fc2",
        "v_cond_mlp.fc1",
        "v_cond_mlp.fc2",
    ),
}


# ── LoRA Linear Layer ───────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """nn.Linear wrapper with frozen base weights and trainable low-rank A/B matrices.

    forward(x) = base(x) + dropout(x @ A^T @ B^T) * scaling
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        init_mode: str = "standard",
        use_rslora: bool = False,
    ):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha

        # Support both nn.Linear and FP8WeightWrapper (which lacks in/out_features)
        in_features = getattr(base, 'in_features', base.weight.shape[1])
        out_features = getattr(base, 'out_features', base.weight.shape[0])

        # Scaling factor
        if use_rslora:
            self.scaling = alpha / math.sqrt(rank)
        else:
            self.scaling = alpha / rank

        # Dropout on LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Low-rank matrices — inherit device/dtype from base to avoid mismatch
        # FP8 storage dtypes can't be used for LoRA params — use bf16 instead
        param_dtype = base.weight.dtype
        if param_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            param_dtype = torch.bfloat16
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=base.weight.device, dtype=param_dtype))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank, device=base.weight.device, dtype=param_dtype))

        # Freeze base weights
        for p in self.base.parameters():
            p.requires_grad = False

        # Initialize
        if init_mode == "pissa":
            self._init_pissa()
        else:
            self._init_standard()

    def _init_standard(self):
        """Kaiming uniform on A, zero on B — LoRA starts as identity."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def _init_pissa(self):
        """PiSSA: SVD-based initialization that captures principal directions."""
        W = self.base.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        sqrt_S = torch.sqrt(S[:self.rank])
        self.lora_B.data = (U[:, :self.rank] * sqrt_S.unsqueeze(0)).to(self.base.weight.dtype)
        self.lora_A.data = (Vt[:self.rank] * sqrt_S.unsqueeze(1)).to(self.base.weight.dtype)
        # Subtract the approximation from base so that base + LoRA = original W
        self.base.weight.data -= (self.lora_B.data @ self.lora_A.data).to(self.base.weight.dtype)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling


# ── Model-level operations ──────────────────────────────────────────────────

def apply_lora(
    model: nn.Module,
    rank: int,
    alpha: Optional[float] = None,
    target_suffixes: Sequence[str] = FOLEY_TARGET_PRESETS["all_attn_mlp"],
    dropout: float = 0.0,
    init_mode: str = "standard",
    use_rslora: bool = False,
) -> int:
    """In-place replacement of matching nn.Linear layers with LoRALinear.

    Args:
        model: model to modify
        rank: LoRA rank
        alpha: scaling alpha (defaults to rank)
        target_suffixes: layer name suffixes to wrap
        dropout: dropout rate on LoRA path
        init_mode: "standard" or "pissa"
        use_rslora: use rsLoRA scaling (alpha/sqrt(rank))

    Returns:
        Number of layers wrapped.
    """
    if alpha is None:
        alpha = float(rank)

    n_wrapped = 0
    for name, module in list(model.named_modules()):
        # Accept nn.Linear or FP8WeightWrapper (duck-type to avoid circular import)
        is_linear = isinstance(module, nn.Linear) or (
            hasattr(module, 'kind') and getattr(module, 'kind', None) == 'linear'
            and hasattr(module, 'weight')
        )
        if not is_linear:
            continue
        if not any(name.endswith(suffix) for suffix in target_suffixes):
            continue

        # Navigate to parent and replace
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr_name = parts[0]

        lora_layer = LoRALinear(
            module, rank=rank, alpha=alpha,
            dropout=dropout, init_mode=init_mode, use_rslora=use_rslora,
        )
        setattr(parent, attr_name, lora_layer)
        n_wrapped += 1

    logger.debug(f"LoRA applied: {n_wrapped} layers wrapped")
    return n_wrapped


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract only LoRA parameters (lora_A, lora_B) from model."""
    return {
        k: v.clone() for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def get_lora_and_base_state_dict(model: nn.Module) -> dict:
    """Extract LoRA + base weights (needed for PiSSA where base was modified)."""
    result = {}
    for k, v in model.state_dict().items():
        if "lora_A" in k or "lora_B" in k or "base.weight" in k or "base.bias" in k:
            result[k] = v.clone()
    return result


def load_lora(model: nn.Module, state_dict: dict) -> int:
    """Load LoRA weights into an already-wrapped model.

    Returns:
        Number of parameters loaded.
    """
    model_sd = model.state_dict()
    loaded = 0
    for k, v in state_dict.items():
        if k in model_sd:
            model_sd[k].copy_(v)
            loaded += 1
        else:
            logger.warning(f"LoRA key not found in model: {k}")
    return loaded


def spectral_surgery(
    model: nn.Module,
    calibration_fn=None,
    n_calibration: int = 8,
    policy: str = "reweight",
) -> None:
    """Post-training SVD reweighting of LoRA layers.

    Redistributes singular values between base and LoRA to reduce
    inference overhead or improve merge quality.

    Args:
        model: model with LoRALinear layers
        calibration_fn: optional callable yielding calibration batches
        n_calibration: number of calibration steps
        policy: "reweight" to redistribute SVD, "merge" to fold LoRA into base
    """
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue

        A = module.lora_A.data.float()  # [rank, in]
        B = module.lora_B.data.float()  # [out, rank]
        delta = (B @ A) * module.scaling  # [out, in]

        if policy == "merge":
            # Fold LoRA into base weights
            module.base.weight.data += delta.to(module.base.weight.dtype)
            nn.init.zeros_(module.lora_A)
            nn.init.zeros_(module.lora_B)
        elif policy == "reweight":
            # SVD reweighting: redistribute singular values
            # Divide out scaling so that forward (which re-applies it) preserves the delta
            delta_unscaled = delta / module.scaling
            U, S, Vt = torch.linalg.svd(delta_unscaled, full_matrices=False)
            r = module.rank
            sqrt_S = torch.sqrt(S[:r])
            module.lora_B.data = (U[:, :r] * sqrt_S.unsqueeze(0)).to(module.lora_B.dtype)
            module.lora_A.data = (Vt[:r] * sqrt_S.unsqueeze(1)).to(module.lora_A.dtype)
