"""Automagic optimizer — per-element adaptive LR via sign agreement.

Adapted from ostris/ai-toolkit (MIT License).
https://github.com/ostris/ai-toolkit/blob/main/toolkit/optimizers/automagic.py

Removed: optimum.quanto dependency, parameter swapping feature.
"""

import torch
from torch import Tensor
from typing import Optional


class Auto8bitTensor:
    def __init__(self, data):
        if isinstance(data, dict):
            self.quantized = data["quantized"]
            self.scale = data["scale"]
        else:
            abs_max = data.abs().max().item()
            self.scale = abs_max / 127.0 if abs_max > 0 else 1.0
            self.quantized = (data / self.scale).round().clamp(-127, 127).to(torch.int8)

    def to(self, *args, **kwargs):
        dtype = None
        if args and isinstance(args[0], torch.dtype):
            dtype = args[0]
            args = args[1:]
        elif "dtype" in kwargs:
            dtype = kwargs.pop("dtype")
        result = self.quantized.to(torch.float32) * self.scale
        if dtype is not None:
            return result.to(dtype=dtype, *args, **kwargs)
        return result.to(*args, **kwargs)

    def state_dict(self):
        return {"quantized": self.quantized, "scale": self.scale}


def _copy_stochastic_bf16(target, source):
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))
    result.add_(source.view(dtype=torch.int32))
    result.bitwise_and_(-65536)
    target.copy_(result.view(dtype=torch.float32))


def copy_stochastic(target, source):
    if target.dtype == torch.float32:
        target.copy_(source)
        return
    if target.dtype == torch.bfloat16:
        _copy_stochastic_bf16(target, source)
        return
    target.copy_(source.to(target.dtype))


def _stochastic_grad_accumulation(param):
    if hasattr(param, "_accum_grad"):
        grad_fp32 = param._accum_grad.clone().to(torch.float32)
        grad_fp32.add_(param.grad.to(torch.float32))
        copy_stochastic(param._accum_grad, grad_fp32)
        del grad_fp32, param.grad
    else:
        param._accum_grad = param.grad.clone()
        del param.grad


class Automagic(torch.optim.Optimizer):
    """Per-element adaptive LR optimizer using gradient sign agreement.

    Each parameter element maintains its own learning rate (stored as int8).
    LR increases when gradient sign is consistent, decreases when it flips.
    Uses Adafactor-style row/col factored second moments for 2D+ tensors.
    """

    def __init__(
        self,
        params,
        lr=1e-6,
        min_lr=1e-7,
        max_lr=1e-3,
        lr_bump=1e-6,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        beta2=0.999,
        weight_decay=0.0,
    ):
        if lr > 1e-3:
            lr = 1e-6
        self.lr = lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "beta2": beta2,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

        self._uses_stochastic_rounding = False
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad and param.dtype != torch.float32:
                    self._uses_stochastic_rounding = True
                    param.register_post_accumulate_grad_hook(_stochastic_grad_accumulation)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def _get_lr(self, param_group, param_state):
        return param_state.get("avg_lr", 0.0)

    def _get_group_lr(self, group):
        lrs = [self._get_lr(group, self.state[p]) for p in group["params"]]
        return sum(lrs) / len(lrs) if lrs else self.lr

    def get_avg_learning_rate(self):
        lrs = [self._get_group_lr(g) for g in self.param_groups]
        return sum(lrs) / len(lrs) if lrs else self.lr

    def _initialize_state(self, p):
        state = self.state[p]
        state["step"] = 0
        state["lr_mask"] = Auto8bitTensor(
            torch.full(p.shape, self.lr, device=p.device, dtype=torch.float32)
        )
        state["avg_lr"] = self.lr
        state["last_polarity"] = torch.zeros(p.shape, dtype=torch.bool, device=p.device)
        factored = len(p.shape) >= 2
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(p.shape[:-1], device=p.device, dtype=torch.float32)
            state["exp_avg_sq_col"] = torch.zeros(p.shape[:-2] + p.shape[-1:], device=p.device, dtype=torch.float32)
        else:
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        if self._uses_stochastic_rounding:
            for group in self.param_groups:
                for param in group["params"]:
                    if param.requires_grad and hasattr(param, "_accum_grad"):
                        param.grad = param._accum_grad
                        del param._accum_grad

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad.float() if p.grad.dtype != torch.float32 else p.grad
                if grad.is_sparse:
                    raise RuntimeError("Automagic does not support sparse gradients")

                state = self.state[p]
                factored = len(grad.shape) >= 2

                if len(state) == 0:
                    self._initialize_state(p)

                p_fp32 = p.float() if p.dtype != torch.float32 else p
                state["step"] += 1

                beta2 = group["beta2"]
                eps = group["eps"][0] if isinstance(group["eps"], (tuple, list)) else group["eps"]
                update = (grad ** 2) + eps

                if factored:
                    row = state["exp_avg_sq_row"]
                    col = state["exp_avg_sq_col"]
                    row.mul_(beta2).add_(update.mean(dim=-1), alpha=1.0 - beta2)
                    col.mul_(beta2).add_(update.mean(dim=-2), alpha=1.0 - beta2)
                    update = self._approx_sq_grad(row, col)
                    update.mul_(grad)
                else:
                    state["exp_avg_sq"].mul_(beta2).add_(update, alpha=1.0 - beta2)
                    update = state["exp_avg_sq"].rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                current_polarity = update > 0
                sign_agreement = torch.where(
                    state["last_polarity"] == current_polarity, 1, -1
                )
                state["last_polarity"] = current_polarity.to(torch.bool)

                lr_mask = state["lr_mask"].to(torch.float32)
                new_lr = torch.where(
                    sign_agreement > 0,
                    lr_mask + self.lr_bump,
                    lr_mask - self.lr_bump,
                )
                new_lr.clamp_(min=self.min_lr, max=self.max_lr)

                update.mul_(new_lr)

                state["lr_mask"] = Auto8bitTensor(new_lr)
                state["avg_lr"] = new_lr.mean().item()

                if group["weight_decay"] != 0:
                    p_fp32.add_(p_fp32 * (-group["weight_decay"]) * new_lr)

                p_fp32.add_(-update)

                if p.dtype != torch.float32:
                    copy_stochastic(p, p_fp32)

        return loss

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        new_state = {}
        for k, v in sd["state"].items():
            entry = {sk: sv for sk, sv in v.items() if sk != "lr_mask"}
            if "lr_mask" in v:
                entry["lr_mask"] = v["lr_mask"].state_dict()
            new_state[k] = entry
        sd["state"] = new_state
        return sd

    def load_state_dict(self, state_dict, strict=True):
        has_lr_mask = any(
            "lr_mask" in s for s in state_dict.get("state", {}).values() if isinstance(s, dict)
        )
        if not has_lr_mask:
            return

        clean_sd = {
            "state": {k: {sk: sv for sk, sv in v.items() if sk != "lr_mask"} for k, v in state_dict["state"].items()},
            "param_groups": state_dict["param_groups"],
        }
        super().load_state_dict(clean_sd)

        current_params = [p for g in self.param_groups for p in g["params"] if p.requires_grad]
        saved_ids = list(state_dict["state"].keys())
        for i, param in enumerate(current_params):
            if i >= len(saved_ids):
                break
            saved = state_dict["state"][saved_ids[i]]
            if "lr_mask" not in saved:
                continue
            if param not in self.state:
                self._initialize_state(param)
            self.state[param]["lr_mask"] = Auto8bitTensor(saved["lr_mask"])
