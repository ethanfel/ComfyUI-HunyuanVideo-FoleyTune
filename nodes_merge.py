"""LoRA merge nodes for FoleyTune.

Two nodes:
- FoleyTuneLoRAMerger: manual strategy selection (TIES/SLERP/weighted_average)
- FoleyTuneLoRAAutoTuner: automatic per-block strategy selection with conflict analysis
"""

import os
import re
import copy
import math
import logging

import torch
from safetensors.torch import load_file as load_safetensors

import folder_paths

try:
    from .lora.merge_math import (
        compute_deltas, merge_weighted_average, merge_ties, merge_slerp,
        merge_slerp_n, dare_sparsify, compute_conflict_mask,
        sample_conflict, classify_relationship, compute_auto_strength,
        THRESHOLDS,
    )
except ImportError:
    from lora.merge_math import (
        compute_deltas, merge_weighted_average, merge_ties, merge_slerp,
        merge_slerp_n, dare_sparsify, compute_conflict_mask,
        sample_conflict, classify_relationship, compute_auto_strength,
        THRESHOLDS,
    )

logger = logging.getLogger("FoleyTune")

_BLOCK_RE = re.compile(r"^(triple_blocks\.\d+)\.")


def _load_adapter_checkpoint(path):
    """Load a LoRA checkpoint from .safetensors or .pt format."""
    if path.endswith(".safetensors"):
        state_dict = load_safetensors(path)
        json_path = path.replace(".safetensors", ".json")
        meta = {}
        if os.path.exists(json_path):
            import json
            with open(json_path) as f:
                meta = json.load(f)
        return {"state_dict": state_dict, "meta": meta}
    return torch.load(path, map_location="cpu", weights_only=False)


def _parse_checkpoint(ckpt):
    """Extract state_dict, rank, alpha, use_rslora, prompts from checkpoint."""
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        meta = ckpt.get("meta", {})
        if "ema_state" in ckpt:
            for key, ema_val in ckpt["ema_state"].items():
                if key in state_dict:
                    state_dict[key] = ema_val
    else:
        state_dict = ckpt
        meta = {}
    inferred_rank = None
    for k, v in state_dict.items():
        if "lora_A" in k and v.ndim == 2:
            inferred_rank = v.shape[0]
            break
    rank = meta.get("rank", inferred_rank or 16)
    alpha = meta.get("alpha", float(rank))
    use_rslora = meta.get("use_rslora", False)
    prompts = meta.get("prompts", [])
    return state_dict, rank, alpha, use_rslora, prompts


class FoleyTuneLoRAMerger:
    """Merge 2-4 FoleyTune LoRAs with a user-selected strategy."""

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "hunyuan_model": ("FOLEYTUNE_MODEL",),
                "lora_name_1": (loras,),
                "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_name_2": (loras,),
                "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "merge_strategy": (["ties", "weighted_average", "slerp"], {
                    "default": "ties",
                    "tooltip": "ties: resolve sign conflicts via trim+elect+disjoint. "
                               "weighted_average: simple weighted blend. "
                               "slerp: spherical interpolation (best for 2 LoRAs).",
                }),
            },
            "optional": {
                "lora_name_3": (["None"] + loras,),
                "strength_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_name_4": (["None"] + loras,),
                "strength_4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("FOLEYTUNE_MODEL", "STRING")
    RETURN_NAMES = ("model", "prompts")
    FUNCTION = "merge"
    CATEGORY = "FoleyTune"
    DESCRIPTION = (
        "Merge 2-4 LoRAs using TIES, SLERP, or weighted average. "
        "Applies the merged delta directly to model weights (single deepcopy)."
    )

    def _group_by_block(self, deltas):
        """Group layer names by block prefix (e.g. triple_blocks.0)."""
        groups = {}
        for layer_name, tensor in deltas.items():
            m = _BLOCK_RE.match(layer_name)
            block = m.group(1) if m else "_other"
            groups.setdefault(block, {})[layer_name] = tensor
        return groups

    def _collect_loras(self, lora_name_1, strength_1, lora_name_2, strength_2,
                       lora_name_3=None, strength_3=1.0,
                       lora_name_4=None, strength_4=1.0):
        """Load checkpoints, compute deltas for each LoRA."""
        entries = []
        for name, strength in [(lora_name_1, strength_1), (lora_name_2, strength_2),
                                (lora_name_3, strength_3), (lora_name_4, strength_4)]:
            if name is None or name == "None" or strength == 0.0:
                continue
            path = folder_paths.get_full_path_or_raise("loras", name)
            ckpt = _load_adapter_checkpoint(path)
            sd, rank, alpha, use_rslora, prompts = _parse_checkpoint(ckpt)
            deltas = compute_deltas(sd, rank, alpha, strength, use_rslora)
            entries.append({"name": name, "strength": strength, "deltas": deltas,
                            "prompts": prompts})
        return entries

    def merge(self, hunyuan_model, lora_name_1, strength_1, lora_name_2, strength_2,
              merge_strategy, lora_name_3=None, strength_3=1.0,
              lora_name_4=None, strength_4=1.0):
        """Merge LoRAs and apply merged deltas to a deepcopy of the model."""
        entries = self._collect_loras(lora_name_1, strength_1, lora_name_2, strength_2,
                                      lora_name_3, strength_3, lora_name_4, strength_4)
        if len(entries) < 2:
            raise ValueError("Need at least 2 LoRAs with non-zero strength to merge.")

        n_loras = len(entries)
        if merge_strategy == "slerp" and n_loras > 2:
            logger.warning("SLERP is optimal for 2 LoRAs -- using iterative pairwise SLERP for %d.", n_loras)

        all_layers = set()
        for e in entries:
            all_layers.update(e["deltas"].keys())

        model = copy.deepcopy(hunyuan_model)
        named_modules = dict(model.named_modules())
        n_applied = 0

        for layer in sorted(all_layers):
            layer_deltas = []
            layer_weights = []
            for e in entries:
                if layer in e["deltas"]:
                    layer_deltas.append(e["deltas"][layer])
                    layer_weights.append(e["strength"])

            if len(layer_deltas) == 0:
                continue
            if len(layer_deltas) == 1:
                merged = layer_deltas[0]
            elif merge_strategy == "ties":
                merged = merge_ties(layer_deltas, layer_weights)
            elif merge_strategy == "slerp":
                if len(layer_deltas) == 2:
                    t = layer_weights[1] / (layer_weights[0] + layer_weights[1])
                    merged = merge_slerp(layer_deltas[0], layer_deltas[1], t)
                else:
                    merged = merge_slerp_n(list(zip(layer_deltas, layer_weights)))
            else:
                merged = merge_weighted_average(layer_deltas, layer_weights)

            module = named_modules.get(layer)
            if module is not None and hasattr(module, "weight"):
                module.weight.data += merged.to(module.weight.dtype)
                n_applied += 1

        model.eval()
        all_prompts = []
        for e in entries:
            all_prompts.extend(e["prompts"])
        logger.info("LoRA merge complete: %d layers, %d LoRAs, strategy=%s",
                     n_applied, n_loras, merge_strategy)
        return (model, "\n".join(all_prompts))


class FoleyTuneLoRAAutoTuner:
    """Auto-tuned LoRA merge with per-block conflict analysis."""

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "hunyuan_model": ("FOLEYTUNE_MODEL",),
                "lora_name_1": (loras,),
                "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_name_2": (loras,),
                "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "auto_strength": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Energy-based strength normalization to prevent over/under-saturation.",
                }),
                "sparsification": (["disabled", "dare", "conflict_aware"], {
                    "default": "disabled",
                    "tooltip": "disabled: no sparsification. dare: random dropout+rescale. "
                               "conflict_aware: only sparsify conflicting positions.",
                }),
                "sparsification_density": ("FLOAT", {
                    "default": 0.7, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Fraction of parameters to keep (1.0 = no sparsification).",
                }),
            },
            "optional": {
                "lora_name_3": (["None"] + loras,),
                "strength_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_name_4": (["None"] + loras,),
                "strength_4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("FOLEYTUNE_MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "prompts", "report")
    FUNCTION = "auto_merge"
    CATEGORY = "FoleyTune"
    DESCRIPTION = (
        "Automatically select merge strategy per block based on conflict analysis. "
        "Reports per-block decisions: consensus -> weighted_average, "
        "orthogonal -> slerp, conflicting -> ties."
    )

    def _analyze_block(self, block_deltas_per_lora, weights):
        """Analyze conflict across all layers in a block for all LoRA pairs.

        block_deltas_per_lora: list of dicts {layer_name: tensor} per LoRA
        weights: list of strength values
        Returns: dict with strategy, avg_cos_sim, avg_conflict, excess_conflict
        """
        n_loras = len(block_deltas_per_lora)
        all_layers = set()
        for d in block_deltas_per_lora:
            all_layers.update(d.keys())

        total_overlap = 0
        total_conflict = 0
        total_dot = 0.0
        total_norm_a_sq = 0.0
        total_norm_b_sq = 0.0
        n_pairs = 0

        for i in range(n_loras):
            for j in range(i + 1, n_loras):
                for layer in all_layers:
                    da = block_deltas_per_lora[i].get(layer)
                    db = block_deltas_per_lora[j].get(layer)
                    if da is None or db is None:
                        continue
                    r = sample_conflict(da, db)
                    total_overlap += r["n_overlap"]
                    total_conflict += int(r["conflict_ratio"] * r["n_overlap"])
                    total_dot += r["dot"]
                    total_norm_a_sq += r["norm_a_sq"]
                    total_norm_b_sq += r["norm_b_sq"]
                n_pairs += 1

        conflict_ratio = total_conflict / total_overlap if total_overlap > 0 else 0.0
        denom = math.sqrt(total_norm_a_sq) * math.sqrt(total_norm_b_sq)
        cos_sim = total_dot / denom if denom > 0 else 0.0
        excess_conflict = max(conflict_ratio - 0.5, 0.0) if abs(cos_sim) < 0.25 else conflict_ratio

        strategy = classify_relationship(cos_sim, conflict_ratio, excess_conflict)
        if strategy == "orthogonal" and n_loras == 2:
            strategy = "slerp"
        elif strategy == "orthogonal":
            strategy = "weighted_average"
        elif strategy == "consensus":
            strategy = "weighted_average"

        return {
            "strategy": strategy,
            "avg_cos_sim": cos_sim,
            "avg_conflict": conflict_ratio,
            "excess_conflict": excess_conflict,
        }

    def auto_merge(self, hunyuan_model, lora_name_1, strength_1, lora_name_2, strength_2,
                   auto_strength, sparsification, sparsification_density,
                   lora_name_3=None, strength_3=1.0, lora_name_4=None, strength_4=1.0):
        """Per-block strategy selection, energy accumulation for auto-strength, sparsification."""
        merger = FoleyTuneLoRAMerger()
        entries = merger._collect_loras(lora_name_1, strength_1, lora_name_2, strength_2,
                                        lora_name_3, strength_3, lora_name_4, strength_4)
        if len(entries) < 2:
            raise ValueError("Need at least 2 LoRAs with non-zero strength to merge.")

        n_loras = len(entries)
        weights = [e["strength"] for e in entries]

        # Auto-strength: accumulate energy across all layers
        scale = 1.0
        if auto_strength == "enabled":
            norm_sq = [0.0] * n_loras
            dots = {}
            for i, e in enumerate(entries):
                for layer, delta in e["deltas"].items():
                    norm_sq[i] += delta.float().pow(2).sum().item()
            for i in range(n_loras):
                for j in range(i + 1, n_loras):
                    dot = 0.0
                    common = set(entries[i]["deltas"].keys()) & set(entries[j]["deltas"].keys())
                    for layer in common:
                        dot += (entries[i]["deltas"][layer].float().flatten() *
                                entries[j]["deltas"][layer].float().flatten()).sum().item()
                    dots[(i, j)] = dot
            scale = compute_auto_strength(weights, norm_sq, dots)
            weights = [w * scale for w in weights]
            logger.info("Auto-strength scale: %.4f", scale)

        # Group by block and analyze
        all_layers = set()
        for e in entries:
            all_layers.update(e["deltas"].keys())

        block_groups = {}
        for layer in all_layers:
            m = _BLOCK_RE.match(layer)
            block = m.group(1) if m else "_other"
            block_groups.setdefault(block, set()).add(layer)

        report_lines = [f"FoleyTune LoRA AutoTuner -- {n_loras} LoRAs"]
        report_lines.append("=" * 50)
        for e in entries:
            report_lines.append(f"  {e['name']} (strength={e['strength']:.2f})")
        report_lines.append("")

        model = copy.deepcopy(hunyuan_model)
        named_modules = dict(model.named_modules())
        n_applied = 0
        strategy_counts = {}

        for block in sorted(block_groups.keys()):
            block_layer_names = block_groups[block]
            block_deltas_per_lora = []
            for e in entries:
                bd = {l: e["deltas"][l] for l in block_layer_names if l in e["deltas"]}
                block_deltas_per_lora.append(bd)

            analysis = self._analyze_block(block_deltas_per_lora, weights)
            strategy = analysis["strategy"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

            report_lines.append(
                f"  {block}: {strategy} "
                f"(cos={analysis['avg_cos_sim']:.3f}, "
                f"conflict={analysis['avg_conflict']:.1%}, "
                f"excess={analysis['excess_conflict']:.1%})"
            )

            for layer in sorted(block_layer_names):
                layer_deltas = []
                layer_weights = []
                for idx, e in enumerate(entries):
                    if layer in e["deltas"]:
                        d = e["deltas"][layer]
                        if sparsification == "dare":
                            d = dare_sparsify(d, sparsification_density)
                        elif sparsification == "conflict_aware" and len(entries) >= 2:
                            pairs = [(e2["deltas"].get(layer, torch.zeros_like(d)), weights[k])
                                     for k, e2 in enumerate(entries) if layer in e2["deltas"]]
                            if len(pairs) >= 2:
                                mask = compute_conflict_mask(pairs)
                                d = dare_sparsify(d, sparsification_density) * mask.float() + d * (~mask).float()
                        layer_deltas.append(d)
                        layer_weights.append(weights[idx])

                if len(layer_deltas) == 0:
                    continue
                if len(layer_deltas) == 1:
                    merged = layer_deltas[0]
                elif strategy == "ties" or strategy == "conflicting":
                    merged = merge_ties(layer_deltas, layer_weights)
                elif strategy == "slerp":
                    if len(layer_deltas) == 2:
                        t = layer_weights[1] / (layer_weights[0] + layer_weights[1])
                        merged = merge_slerp(layer_deltas[0], layer_deltas[1], t)
                    else:
                        merged = merge_slerp_n(list(zip(layer_deltas, layer_weights)))
                else:
                    merged = merge_weighted_average(layer_deltas, layer_weights)

                module = named_modules.get(layer)
                if module is not None and hasattr(module, "weight"):
                    module.weight.data += merged.to(module.weight.dtype)
                    n_applied += 1

        model.eval()
        report_lines.append("")
        report_lines.append(f"Strategy summary: {strategy_counts}")
        report_lines.append(f"Applied to {n_applied} layers.")
        if auto_strength == "enabled":
            report_lines.append(f"Auto-strength scale: {scale:.4f}")
        report = "\n".join(report_lines)
        logger.info(report)

        all_prompts = []
        for e in entries:
            all_prompts.extend(e["prompts"])

        return (model, "\n".join(all_prompts), report)


# --- Node registration (for use by __init__.py or standalone) ---

NODE_CLASS_MAPPINGS = {
    "FoleyTuneLoRAMerger": FoleyTuneLoRAMerger,
    "FoleyTuneLoRAAutoTuner": FoleyTuneLoRAAutoTuner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoleyTuneLoRAMerger": "FoleyTune LoRA Merger",
    "FoleyTuneLoRAAutoTuner": "FoleyTune LoRA AutoTuner",
}
