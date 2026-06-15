"""LoRA merge nodes for FoleyTune.

Modular merge suite (mirrors the ComfyUI-LoRA-Optimizer architecture):

- FoleyTuneLoRAStack:      build a LORA_STACK (chainable) for the merge nodes
- FoleyTuneMergeOptions:   pure-data options node (FOLEYTUNE_MERGE_OPTIONS)
- FoleyTuneLoRAMerger:     manual strategy merge (TIES / SLERP / weighted_average)
- FoleyTuneLoRAAutoTuner:  per-block conflict analysis + ranked top-N candidates
- FoleyTuneSaveMergedLoRA: SVD-decompose a merged result into a loadable checkpoint
- FoleyTuneSaveTunerData / FoleyTuneLoadTunerData: persist/replay AutoTuner rankings
- FoleyTuneMergeSelector:  apply a chosen ranked config from TUNER_DATA

Custom types are plain Python objects passed between nodes:
- LORA_STACK             list[{"name", "path", "strength"}]
- FOLEYTUNE_MERGE_OPTIONS dict of merge settings
- LORA_DATA             {"deltas", "rank_hint", "alpha", "use_rslora",
                         "target_suffixes", "prompts", "source_names", "is_conv"}
- TUNER_DATA            ranked per-block configs + analysis (JSON-serializable)
"""

import os
import re
import copy
import json
import math
import logging

import torch
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

import folder_paths

try:
    from .lora.merge_math import (
        compute_deltas, merge_weighted_average, merge_ties, merge_slerp,
        merge_slerp_n, dare_sparsify, compute_conflict_mask,
        sample_conflict, classify_relationship, compute_auto_strength,
        extract_lora_svd, score_config, THRESHOLDS,
    )
except ImportError:
    from lora.merge_math import (
        compute_deltas, merge_weighted_average, merge_ties, merge_slerp,
        merge_slerp_n, dare_sparsify, compute_conflict_mask,
        sample_conflict, classify_relationship, compute_auto_strength,
        extract_lora_svd, score_config, THRESHOLDS,
    )

logger = logging.getLogger("FoleyTune")

_BLOCK_RE = re.compile(r"^((?:triple_blocks|single_blocks)\.\d+)\.")

# Register a `tuner_data` model folder so Save/Load Tuner Data have a home.
# Guarded so importing this module under test mocks (where folder_paths is a
# MagicMock) does not raise.
try:
    _TUNER_DATA_DIR = os.path.join(folder_paths.models_dir, "tuner_data")
    os.makedirs(_TUNER_DATA_DIR, exist_ok=True)
    folder_paths.add_model_folder_path("tuner_data", _TUNER_DATA_DIR)
except Exception:
    pass


# --- Checkpoint loading helpers (unchanged behavior) ------------------------

def _load_adapter_checkpoint(path):
    """Load a LoRA checkpoint from .safetensors or .pt format.

    Schedule-free training checkpoints carry raw train-mode weights in
    `state_dict` (for resume) and the averaged weights in `eval_state_dict`
    — merge with the latter, matching what the eval samples were generated with.
    """
    if path.endswith(".safetensors"):
        state_dict = load_safetensors(path)
        json_path = path.replace(".safetensors", ".json")
        meta = {}
        if os.path.exists(json_path):
            with open(json_path) as f:
                meta = json.load(f)
        return {"state_dict": state_dict, "meta": meta}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "eval_state_dict" in ckpt:
        ckpt = {**ckpt, "state_dict": ckpt["eval_state_dict"]}
        logger.info("Using schedule-free averaged (eval-mode) weights from checkpoint")
    return ckpt


def _parse_checkpoint(ckpt):
    """Extract state_dict, rank, alpha, use_rslora, target, prompts from checkpoint."""
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
        if "lora_A" in k and v.ndim >= 2:
            inferred_rank = v.shape[0]
            break
    rank = meta.get("rank", inferred_rank or 16)
    alpha = meta.get("alpha", float(rank))
    use_rslora = meta.get("use_rslora", False)
    target = meta.get("target", "all_attn_mlp")
    prompts = meta.get("prompts", [])
    return state_dict, rank, alpha, use_rslora, target, prompts


def _resolve_suffixes(target):
    """Resolve a checkpoint 'target' (preset key or explicit list) to a suffix tuple."""
    if isinstance(target, (list, tuple)):
        return tuple(target)
    presets = {}
    try:
        from .lora.lora import FOLEY_TARGET_PRESETS as presets
    except Exception:
        try:
            from lora.lora import FOLEY_TARGET_PRESETS as presets
        except Exception:
            presets = {}
    if isinstance(target, str) and target in presets:
        return tuple(presets[target])
    return ()


# --- Shared merge helpers ----------------------------------------------------

# Default settings used when no FoleyTuneMergeOptions node is connected.
# (merge_strategy and top_n are node-specific inline widgets, not shared options.)
_DEFAULT_OPTIONS = {
    "merge_strategy": "ties",
    "auto_strength": "disabled",
    "auto_strength_floor": -1.0,
    "sparsification": "disabled",
    "sparsification_density": 0.7,
    "dare_dampening": 0.0,
    "ties_density": 0.7,
    "ties_sign_method": "frequency",
}


def _resolve_options(merge_options, **node_defaults):
    """Merge node widget defaults with an optional FOLEYTUNE_MERGE_OPTIONS dict.

    Connected options override node widgets; node widgets override built-ins.
    """
    opts = dict(_DEFAULT_OPTIONS)
    opts.update({k: v for k, v in node_defaults.items() if v is not None})
    if merge_options:
        opts.update({k: v for k, v in merge_options.items() if v is not None})
    return opts


def _collect_loras_from_stack(lora_stack):
    """Load checkpoints from a LORA_STACK and compute per-layer deltas.

    Returns a list of entries:
    {"name", "strength", "deltas", "prompts", "rank", "alpha", "suffixes"}.
    Entries with zero strength are skipped.
    """
    if not lora_stack:
        return []
    entries = []
    for item in lora_stack:
        name = item.get("name")
        strength = float(item.get("strength", 1.0))
        if strength == 0.0:
            continue
        path = item.get("path")
        if not path:
            path = folder_paths.get_full_path_or_raise("loras", name)
        ckpt = _load_adapter_checkpoint(path)
        sd, rank, alpha, use_rslora, target, prompts = _parse_checkpoint(ckpt)
        deltas = compute_deltas(sd, rank, alpha, strength, use_rslora)
        entries.append({
            "name": name, "strength": strength, "deltas": deltas,
            "prompts": prompts, "rank": rank, "alpha": alpha,
            "suffixes": _resolve_suffixes(target),
        })
    return entries


def _build_lora_data(entries, merged_deltas, scale=1.0):
    """Assemble a LORA_DATA dict from merged deltas + source provenance."""
    suffixes = set()
    for e in entries:
        suffixes.update(e.get("suffixes", ()))
    prompts, seen = [], set()
    for e in entries:
        for p in e.get("prompts", []):
            if p not in seen:
                seen.add(p)
                prompts.append(p)
    return {
        "deltas": merged_deltas,
        "rank_hint": sum(int(e["rank"]) for e in entries),
        "alpha": max((float(e["alpha"]) for e in entries), default=1.0),
        "use_rslora": False,
        "target_suffixes": sorted(suffixes),
        "prompts": prompts,
        "source_names": [e["name"] for e in entries],
        "is_conv": {layer: (d.ndim == 3) for layer, d in merged_deltas.items()},
        "auto_strength_scale": scale,
    }


def _group_layers_by_block(all_layers):
    groups = {}
    for layer in all_layers:
        m = _BLOCK_RE.match(layer)
        block = m.group(1) if m else "_other"
        groups.setdefault(block, set()).add(layer)
    return groups


def _sparsify_present(entries, layer, weights, sparsification, density,
                      dare_dampening, conflict_skip):
    """Return (layer_deltas, layer_weights) after optional per-layer sparsification.

    `conflict_skip` is a 1-element mutable list used to count layers where the
    conflict mask covered >40% of positions (orthogonal base-rate noise) and
    conflict-aware sparsification was downgraded to standard DARE.
    """
    present = [(idx, e) for idx, e in enumerate(entries) if layer in e["deltas"]]
    layer_deltas, layer_weights = [], []
    for idx, e in present:
        d = e["deltas"][layer]
        if sparsification == "dare":
            d = dare_sparsify(d, density, dampening=dare_dampening)
        elif sparsification == "conflict_aware" and len(present) >= 2:
            pairs = [(e2["deltas"][layer], weights[k]) for k, e2 in present]
            mask = compute_conflict_mask(pairs)
            if mask.float().mean().item() > 0.40:
                # Base-rate noise from near-orthogonal LoRAs — uncorrelated
                # vectors disagree in sign ~50% of the time. Treat as noise and
                # fall back to standard DARE rather than masking everywhere.
                conflict_skip[0] += 1
                d = dare_sparsify(d, density, dampening=dare_dampening)
            else:
                sd = dare_sparsify(d, density, dampening=dare_dampening)
                d = sd * mask.float() + d * (~mask).float()
        layer_deltas.append(d)
        layer_weights.append(weights[idx])
    return layer_deltas, layer_weights


def _merge_layer(strategy, layer_deltas, layer_weights, ties_density, ties_sign_method):
    """Merge a single layer's deltas using a concrete strategy op."""
    if len(layer_deltas) == 1:
        return layer_deltas[0].float()
    if strategy == "ties":
        return merge_ties(layer_deltas, layer_weights,
                          density=ties_density, sign_method=ties_sign_method)
    if strategy == "slerp":
        if len(layer_deltas) == 2:
            denom = layer_weights[0] + layer_weights[1]
            t = layer_weights[1] / denom if denom != 0 else 0.5
            return merge_slerp(layer_deltas[0], layer_deltas[1], t)
        return merge_slerp_n(list(zip(layer_deltas, layer_weights)))
    return merge_weighted_average(layer_deltas, layer_weights)


def _apply_block_merge(model, entries, weights, strategy_for_block, opts):
    """Merge all layers into a deepcopy-ready model in place, per-block strategy.

    strategy_for_block: callable(block_name) -> concrete op ('ties'|'slerp'|'weighted_average').
    Returns (n_applied, merged_deltas, strategy_counts, conflict_skipped).
    """
    sparsification = opts["sparsification"]
    density = opts["sparsification_density"]
    dare_dampening = opts["dare_dampening"]
    ties_density = opts["ties_density"]
    ties_sign_method = opts["ties_sign_method"]

    named_modules = dict(model.named_modules())
    all_layers = set()
    for e in entries:
        all_layers.update(e["deltas"].keys())
    block_groups = _group_layers_by_block(all_layers)

    merged_deltas = {}
    strategy_counts = {}
    conflict_skip = [0]
    n_applied = 0

    for block in sorted(block_groups.keys()):
        strategy = strategy_for_block(block)
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        for layer in sorted(block_groups[block]):
            layer_deltas, layer_weights = _sparsify_present(
                entries, layer, weights, sparsification, density,
                dare_dampening, conflict_skip)
            if not layer_deltas:
                continue
            merged = _merge_layer(strategy, layer_deltas, layer_weights,
                                  ties_density, ties_sign_method)
            merged_deltas[layer] = merged
            module = named_modules.get(layer)
            if module is not None and hasattr(module, "weight"):
                module.weight.data += merged.to(module.weight.dtype)
                n_applied += 1
    return n_applied, merged_deltas, strategy_counts, conflict_skip[0]


def _compute_auto_scale(entries, floor):
    """Compute the energy-normalization scale for a stack (or 1.0)."""
    n = len(entries)
    norm_sq = [0.0] * n
    for i, e in enumerate(entries):
        for delta in e["deltas"].values():
            norm_sq[i] += delta.float().pow(2).sum().item()
    dots = {}
    for i in range(n):
        for j in range(i + 1, n):
            common = set(entries[i]["deltas"].keys()) & set(entries[j]["deltas"].keys())
            dot = 0.0
            for layer in common:
                dot += (entries[i]["deltas"][layer].float().flatten() *
                        entries[j]["deltas"][layer].float().flatten()).sum().item()
            dots[(i, j)] = dot
    strengths = [e["strength"] for e in entries]
    return compute_auto_strength(strengths, norm_sq, dots, floor=floor)


# --- Node: LoRA Stack --------------------------------------------------------

class FoleyTuneLoRAStack:
    """Build a chainable LORA_STACK for the merge nodes.

    Chain multiple of these to stack 2+ LoRAs; feed the final lora_stack into
    a FoleyTune LoRA Merger / AutoTuner.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_name": (loras, {
                    "tooltip": "A trained FoleyTune LoRA checkpoint (.safetensors/.pt) to add to the stack.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Per-LoRA merge weight. 1.0 = full. Relative ratios between stacked LoRAs are preserved by auto-strength.",
                }),
            },
            "optional": {
                "lora_stack": ("FOLEYTUNE_LORA_STACK", {
                    "tooltip": "Chain the output of another LoRA Stack node here to add more LoRAs. Leave empty for the first LoRA.",
                }),
            },
        }

    RETURN_TYPES = ("FOLEYTUNE_LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "add_to_stack"
    CATEGORY = "FoleyTune"
    DESCRIPTION = ("Append a LoRA (name + strength) to a LORA_STACK. Chain one node per "
                   "LoRA, then feed the final stack into a Merger or AutoTuner (needs 2+).")

    def add_to_stack(self, lora_name, strength, lora_stack=None):
        entries = list(lora_stack) if lora_stack else []
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        entries.append({"name": lora_name, "path": path, "strength": float(strength)})
        return (entries,)


# --- Node: Merge Options -----------------------------------------------------

class FoleyTuneMergeOptions:
    """Pure-data node: the SHARED merge tuning, used by both the Merger and the
    AutoTuner. Connect to their optional `merge_options` input.

    Node-specific controls live on the nodes themselves, not here: the merge
    STRATEGY is on the Merger (the AutoTuner picks a strategy per block), and the
    number of ranked candidates (top_n) is on the AutoTuner.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auto_strength": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Energy-based normalization: scales stacked LoRA strengths down so the "
                               "combined result doesn't oversaturate. Preserves your relative ratios. "
                               "Applies to both the Merger and the AutoTuner.",
                }),
                "auto_strength_floor": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Lower bound on the auto-strength scale. -1 = architecture default "
                               "(near-orthogonal/independent LoRAs are NOT diluted). 0.0-1.0 = explicit "
                               "floor that bounds the reduction regardless of alignment.",
                }),
                "sparsification": (["disabled", "dare", "conflict_aware"], {
                    "default": "disabled",
                    "tooltip": "Drop weights before merging to reduce interference. "
                               "dare = random drop + rescale everywhere. "
                               "conflict_aware = only drop where LoRAs disagree in sign "
                               "(auto-downgrades to dare when >40% of positions conflict — orthogonal noise).",
                }),
                "sparsification_density": ("FLOAT", {
                    "default": 0.7, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Fraction of weights to KEEP when sparsifying (1.0 = keep all = no sparsification).",
                }),
                "dare_dampening": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "DAREx dampening: softens DARE's 1/density rescale to reduce noise "
                               "amplification at low density. 0 = standard DARE. Only affects dare modes.",
                }),
                "ties_density": ("FLOAT", {
                    "default": 0.7, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "TIES trim: fraction of largest-magnitude weights kept per layer. "
                               "Only used when the merge resolves a layer with the TIES strategy.",
                }),
                "ties_sign_method": (["frequency", "total"], {
                    "default": "frequency",
                    "tooltip": "TIES sign election. 'total' = magnitude-weighted (a strong LoRA can win the "
                               "sign). 'frequency' = one vote per LoRA. Only used for TIES layers.",
                }),
            },
        }

    RETURN_TYPES = ("FOLEYTUNE_MERGE_OPTIONS",)
    RETURN_NAMES = ("merge_options",)
    FUNCTION = "build"
    CATEGORY = "FoleyTune"
    DESCRIPTION = ("Shared merge tuning (auto-strength, sparsification, TIES settings) for the "
                   "FoleyTune Merger / AutoTuner. Strategy is on the Merger; top_n is on the AutoTuner.")

    def build(self, auto_strength, auto_strength_floor, sparsification,
              sparsification_density, dare_dampening, ties_density, ties_sign_method):
        return ({
            "auto_strength": auto_strength,
            "auto_strength_floor": auto_strength_floor,
            "sparsification": sparsification,
            "sparsification_density": sparsification_density,
            "dare_dampening": dare_dampening,
            "ties_density": ties_density,
            "ties_sign_method": ties_sign_method,
        },)


# --- Node: LoRA Merger -------------------------------------------------------

class FoleyTuneLoRAMerger:
    """Merge a LORA_STACK with a single user-selected strategy."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("FOLEYTUNE_MODEL", {
                    "tooltip": "Base FoleyTune model to merge the LoRAs into.",
                }),
                "lora_stack": ("FOLEYTUNE_LORA_STACK", {
                    "tooltip": "Stack of 2+ LoRAs from FoleyTune LoRA Stack node(s).",
                }),
                "merge_strategy": (["ties", "weighted_average", "slerp"], {
                    "default": "ties",
                    "tooltip": "How to combine overlapping weights (this whole merge uses ONE strategy). "
                               "ties: trim + sign-elect + disjoint merge (best when LoRAs conflict). "
                               "weighted_average: simple blend (best when compatible). "
                               "slerp: spherical interpolation / Karcher mean for 3+ (magnitude-preserving). "
                               "For automatic per-block strategy selection, use the AutoTuner instead.",
                }),
            },
            "optional": {
                "merge_options": ("FOLEYTUNE_MERGE_OPTIONS", {
                    "tooltip": "Optional shared tuning (auto-strength, sparsification, TIES settings). "
                               "Uses sensible defaults if not connected.",
                }),
            },
        }

    RETURN_TYPES = ("FOLEYTUNE_MODEL", "STRING", "FOLEYTUNE_LORA_DATA")
    RETURN_NAMES = ("model", "prompts", "lora_data")
    FUNCTION = "merge"
    CATEGORY = "FoleyTune"
    DESCRIPTION = (
        "Merge a LoRA stack (2+ LoRAs) using ONE chosen strategy (TIES / SLERP / weighted "
        "average). Applies the merged delta to a deepcopy of the model and emits LORA_DATA "
        "(connect to Save Merged LoRA). For per-block automatic strategy, use the AutoTuner."
    )

    def _group_by_block(self, deltas):
        """Group layer names by block prefix (e.g. triple_blocks.0)."""
        groups = {}
        for layer_name, tensor in deltas.items():
            m = _BLOCK_RE.match(layer_name)
            block = m.group(1) if m else "_other"
            groups.setdefault(block, {})[layer_name] = tensor
        return groups

    def merge(self, hunyuan_model, lora_stack, merge_strategy="ties", merge_options=None):
        opts = _resolve_options(merge_options, merge_strategy=merge_strategy)
        entries = _collect_loras_from_stack(lora_stack)
        if len(entries) < 2:
            raise ValueError("Need at least 2 LoRAs with non-zero strength to merge. "
                             "Use FoleyTune LoRA Loader for a single LoRA.")

        strategy = opts["merge_strategy"]
        n_loras = len(entries)
        if strategy == "slerp" and n_loras > 2:
            logger.info("SLERP with %d LoRAs uses the Karcher (spherical) mean.", n_loras)

        # Auto-strength (shared option) — scale the stack down to avoid oversaturation.
        scale = 1.0
        if opts["auto_strength"] == "enabled":
            floor = opts["auto_strength_floor"]
            scale = _compute_auto_scale(entries, None if floor < 0 else floor)
            logger.info("Auto-strength scale: %.4f", scale)
        weights = [e["strength"] * scale for e in entries]

        model = copy.deepcopy(hunyuan_model)
        n_applied, merged_deltas, _, _ = _apply_block_merge(
            model, entries, weights, lambda _b: strategy, opts)
        model.eval()

        lora_data = _build_lora_data(entries, merged_deltas, scale=scale)
        all_prompts = lora_data["prompts"]
        logger.info("LoRA merge complete: %d layers, %d LoRAs, strategy=%s",
                    n_applied, n_loras, strategy)
        return (model, "\n".join(all_prompts), lora_data)


# --- Node: LoRA AutoTuner ----------------------------------------------------

# Candidate grid for top-N ranking: merge approach x sparsification.
_AUTOTUNE_APPROACHES = ["per_block_adaptive", "ties", "weighted_average", "slerp"]
_AUTOTUNE_SPARSIFICATIONS = ["disabled", "dare", "conflict_aware"]


class FoleyTuneLoRAAutoTuner:
    """Auto-tuned LoRA merge: per-block conflict analysis + ranked top-N configs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("FOLEYTUNE_MODEL", {
                    "tooltip": "Base FoleyTune model to merge the LoRAs into.",
                }),
                "lora_stack": ("FOLEYTUNE_LORA_STACK", {
                    "tooltip": "Stack of 2+ LoRAs from FoleyTune LoRA Stack node(s).",
                }),
                "top_n": ("INT", {
                    "default": 3, "min": 1, "max": 10, "step": 1,
                    "tooltip": "How many ranked candidate configs to keep and emit in tuner_data "
                               "(for the Merge Selector / Save Tuner Data).",
                }),
                "selection": ("INT", {
                    "default": 1, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Which ranked candidate to apply to the OUTPUT model. 1 = top-ranked. "
                               "Re-run with a different value to A/B the alternatives without a separate "
                               "Merge Selector node. Clamped to the number of candidates (top_n).",
                }),
            },
            "optional": {
                "merge_options": ("FOLEYTUNE_MERGE_OPTIONS", {
                    "tooltip": "Optional shared tuning (auto-strength, sparsification, TIES settings). "
                               "The AutoTuner picks the merge STRATEGY itself, per block — strategy is "
                               "not taken from here. Uses sensible defaults if not connected. "
                               "Ignored when tuner_data is connected (replay uses the saved settings).",
                }),
                "tuner_data": ("FOLEYTUNE_TUNER_DATA", {
                    "tooltip": "Connect saved results (Load Tuner Data) or another AutoTuner's tuner_data "
                               "to REPLAY them: analysis + scoring are skipped and the saved ranking is "
                               "applied directly (respecting 'selection'). Re-tunes from scratch if left empty. "
                               "Use the same lora_stack the data was tuned on.",
                }),
            },
        }

    RETURN_TYPES = ("FOLEYTUNE_MODEL", "STRING", "STRING", "FOLEYTUNE_TUNER_DATA", "FOLEYTUNE_LORA_DATA")
    RETURN_NAMES = ("model", "prompts", "report", "tuner_data", "lora_data")
    FUNCTION = "auto_merge"
    CATEGORY = "FoleyTune"
    DESCRIPTION = (
        "Analyze per-block conflict once, score a grid of merge candidates, and "
        "apply the top-ranked one. Emits TUNER_DATA (ranked alternatives for the "
        "Merge Selector) and LORA_DATA (for Save Merged LoRA)."
    )

    def _analyze_block(self, block_deltas_per_lora, weights):
        """Analyze conflict across all layers in a block for all LoRA pairs.

        Returns dict with strategy (concrete op), avg_cos_sim, avg_conflict,
        excess_conflict.
        """
        n_loras = len(block_deltas_per_lora)
        all_layers = set()
        for d in block_deltas_per_lora:
            all_layers.update(d.keys())

        total_overlap = 0
        total_conflict = 0
        total_excess = 0.0
        total_dot = 0.0
        total_norm_a_sq = 0.0
        total_norm_b_sq = 0.0

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
                    total_excess += r["excess_conflict"] * r["n_overlap"]
                    total_dot += r["dot"]
                    total_norm_a_sq += r["norm_a_sq"]
                    total_norm_b_sq += r["norm_b_sq"]

        conflict_ratio = total_conflict / total_overlap if total_overlap > 0 else 0.0
        excess_conflict = total_excess / total_overlap if total_overlap > 0 else 0.0
        denom = math.sqrt(total_norm_a_sq) * math.sqrt(total_norm_b_sq)
        cos_sim = total_dot / denom if denom > 0 else 0.0

        strategy = classify_relationship(cos_sim, conflict_ratio, excess_conflict)
        if strategy == "orthogonal":
            strategy = "slerp" if n_loras == 2 else "weighted_average"
        elif strategy == "consensus":
            strategy = "weighted_average"
        elif strategy == "conflicting":
            strategy = "ties"
        # else "weighted_average" (already a concrete op)

        return {
            "strategy": strategy,
            "avg_cos_sim": cos_sim,
            "avg_conflict": conflict_ratio,
            "excess_conflict": excess_conflict,
        }

    def auto_merge(self, hunyuan_model, lora_stack, top_n=3, selection=1,
                   merge_options=None, tuner_data=None):
        entries = _collect_loras_from_stack(lora_stack)
        if len(entries) < 2:
            raise ValueError("Need at least 2 LoRAs with non-zero strength to merge.")
        n_loras = len(entries)

        if tuner_data and tuner_data.get("top_n"):
            # ---- REPLAY: apply a saved ranking, skip analysis + scoring ----
            replayed = True
            ranked = tuner_data["top_n"]
            scale = float(tuner_data.get("auto_strength_scale", 1.0))
            block_analysis = tuner_data.get("block_decisions", {})
            merge_settings = {
                "sparsification_density": tuner_data.get("sparsification_density", 0.7),
                "dare_dampening": tuner_data.get("dare_dampening", 0.0),
                "ties_density": tuner_data.get("ties_density", 0.7),
                "ties_sign_method": tuner_data.get("ties_sign_method", "frequency"),
            }
            out_tuner_data = dict(tuner_data)
            saved_names = tuner_data.get("source_names")
            if saved_names and saved_names != [e["name"] for e in entries]:
                logger.warning("AutoTuner replay: lora_stack %s differs from tuner_data source %s "
                               "— some block configs may fall back to weighted_average.",
                               [e["name"] for e in entries], saved_names)
        else:
            # ---- FRESH: analyze per block, score a candidate grid, rank ----
            replayed = False
            opts = _resolve_options(merge_options)
            scale = 1.0
            if opts["auto_strength"] == "enabled":
                floor = opts["auto_strength_floor"]
                scale = _compute_auto_scale(entries, None if floor < 0 else floor)
                logger.info("Auto-strength scale: %.4f", scale)
            analysis_weights = [e["strength"] * scale for e in entries]

            all_layers = set()
            for e in entries:
                all_layers.update(e["deltas"].keys())
            block_groups = _group_layers_by_block(all_layers)
            block_analysis = {}
            for block, layer_names in block_groups.items():
                per_lora = [{l: e["deltas"][l] for l in layer_names if l in e["deltas"]}
                            for e in entries]
                block_analysis[block] = self._analyze_block(per_lora, analysis_weights)
            block_metrics = list(block_analysis.values())

            candidates = []
            for approach in _AUTOTUNE_APPROACHES:
                for spars in _AUTOTUNE_SPARSIFICATIONS:
                    s, breakdown = score_config(
                        approach, spars, opts["sparsification_density"], block_metrics)
                    if approach == "per_block_adaptive":
                        config = {b: a["strategy"] for b, a in block_analysis.items()}
                    else:
                        config = {b: approach for b in block_analysis}
                    candidates.append({
                        "approach": approach, "sparsification": spars,
                        "config": config, "score_heuristic": round(s, 6),
                        "score_breakdown": breakdown,
                    })
            candidates.sort(key=lambda c: c["score_heuristic"], reverse=True)
            ranked = candidates[:max(1, int(top_n))]
            for i, c in enumerate(ranked):
                c["rank"] = i + 1

            merge_settings = {
                "sparsification_density": opts["sparsification_density"],
                "dare_dampening": opts["dare_dampening"],
                "ties_density": opts["ties_density"],
                "ties_sign_method": opts["ties_sign_method"],
            }
            out_tuner_data = {
                "algo_version": "foley-merge-1",
                "source_names": [e["name"] for e in entries],
                "strengths": [e["strength"] for e in entries],
                "auto_strength": opts["auto_strength"],
                "auto_strength_floor": opts["auto_strength_floor"],
                "auto_strength_scale": scale,
                "block_decisions": block_analysis,
                "top_n": ranked,
                "prompt": "",
                "description": "",
                **merge_settings,
            }

        # ---- Merge the selected candidate for real (default = top-ranked) ----
        weights = [e["strength"] * scale for e in entries]
        sel_idx = max(1, min(int(selection), len(ranked))) - 1
        winner = ranked[sel_idx]
        win_opts = {"sparsification": winner.get("sparsification", "disabled"), **merge_settings}
        model = copy.deepcopy(hunyuan_model)
        n_applied, merged_deltas, strategy_counts, conflict_skipped = _apply_block_merge(
            model, entries, weights,
            lambda b, cfg=winner.get("config", {}): cfg.get(b, "weighted_average"), win_opts)
        model.eval()

        lora_data = _build_lora_data(entries, merged_deltas, scale=scale)

        # --- Report ---
        mode = "REPLAY (from tuner_data)" if replayed else "tune"
        report_lines = [f"FoleyTune LoRA AutoTuner [{mode}] -- {n_loras} LoRAs", "=" * 50]
        for e in entries:
            report_lines.append(f"  {e['name']} (strength={e['strength']:.2f})")
        if scale != 1.0:
            report_lines.append(f"Auto-strength scale: {scale:.4f}")
        report_lines.append("")
        report_lines.append("Ranked candidates:")
        for c in ranked:
            report_lines.append(
                f"  #{c.get('rank', '?')} {c.get('approach')} / spars={c.get('sparsification')} "
                f"-> score={c.get('score_heuristic', 0):.3f} "
                f"(excess={c.get('score_breakdown', {}).get('avg_excess', 0):.1%}, "
                f"spread={c.get('score_breakdown', {}).get('spread', 0):.1%})")
        report_lines.append("")
        report_lines.append(f"Applied: #{winner.get('rank', sel_idx + 1)} {winner.get('approach')} / "
                            f"spars={winner.get('sparsification')} (selection={sel_idx + 1} of {len(ranked)})")
        report_lines.append(f"Per-block strategies: {strategy_counts}")
        report_lines.append(f"Applied to {n_applied} layers.")
        if conflict_skipped:
            report_lines.append(
                f"Conflict-aware sparsification downgraded to DARE on {conflict_skipped} "
                f"layer(s) (>40% conflict mask = orthogonal base-rate noise).")
        report = "\n".join(report_lines)
        logger.info(report)

        return (model, "\n".join(lora_data["prompts"]), report, out_tuner_data, lora_data)


# --- Node: Merge Selector ----------------------------------------------------

class FoleyTuneMergeSelector:
    """Apply a ranked config from TUNER_DATA without re-running analysis."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("FOLEYTUNE_MODEL", {
                    "tooltip": "Base FoleyTune model to merge into (same base the AutoTuner ran on).",
                }),
                "lora_stack": ("FOLEYTUNE_LORA_STACK", {
                    "tooltip": "The same LoRA stack the AutoTuner ranked (rebuilt with the same Stack nodes).",
                }),
                "tuner_data": ("FOLEYTUNE_TUNER_DATA", {
                    "tooltip": "Ranked configs from a FoleyTune LoRA AutoTuner (or Load Tuner Data).",
                }),
                "selection": ("INT", {
                    "default": 1, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Which ranked config to apply (1 = top-ranked, 2 = next, ...). "
                               "Clamped to the number of available candidates.",
                }),
            },
        }

    RETURN_TYPES = ("FOLEYTUNE_MODEL", "STRING", "STRING", "FOLEYTUNE_LORA_DATA")
    RETURN_NAMES = ("model", "prompts", "report", "lora_data")
    FUNCTION = "select_merge"
    CATEGORY = "FoleyTune"
    DESCRIPTION = "Replay a chosen ranked merge config from AutoTuner TUNER_DATA."

    def select_merge(self, hunyuan_model, lora_stack, tuner_data, selection):
        top_n = tuner_data.get("top_n", []) if tuner_data else []
        if not top_n:
            logger.warning("Merge Selector: TUNER_DATA has no ranked configs — passing model through.")
            return (hunyuan_model, "", "No ranked configs in tuner_data.", None)

        idx = max(1, min(int(selection), len(top_n))) - 1
        chosen = top_n[idx]

        entries = _collect_loras_from_stack(lora_stack)
        if len(entries) < 2:
            raise ValueError("Need at least 2 LoRAs with non-zero strength to merge.")

        scale = float(tuner_data.get("auto_strength_scale", 1.0))
        weights = [e["strength"] * scale for e in entries]

        opts = dict(_DEFAULT_OPTIONS)
        opts.update({
            "sparsification": chosen.get("sparsification", "disabled"),
            "sparsification_density": tuner_data.get("sparsification_density", 0.7),
            "dare_dampening": tuner_data.get("dare_dampening", 0.0),
            "ties_density": tuner_data.get("ties_density", 0.7),
            "ties_sign_method": tuner_data.get("ties_sign_method", "frequency"),
        })

        model = copy.deepcopy(hunyuan_model)
        config = chosen.get("config", {})
        n_applied, merged_deltas, strategy_counts, conflict_skipped = _apply_block_merge(
            model, entries, weights,
            lambda b, cfg=config: cfg.get(b, "weighted_average"), opts)
        model.eval()

        lora_data = _build_lora_data(entries, merged_deltas, scale=scale)
        report = (
            f"FoleyTune Merge Selector -- selection #{idx + 1}/{len(top_n)}\n"
            f"  approach={chosen.get('approach')} spars={chosen.get('sparsification')} "
            f"score={chosen.get('score_heuristic', 0):.3f}\n"
            f"  Per-block strategies: {strategy_counts}\n"
            f"  Applied to {n_applied} layers.")
        if conflict_skipped:
            report += f"\n  Conflict-aware downgraded to DARE on {conflict_skipped} layer(s)."
        logger.info(report)
        return (model, "\n".join(lora_data["prompts"]), report, lora_data)


# --- Node: Save Merged LoRA --------------------------------------------------

class FoleyTuneSaveMergedLoRA:
    """SVD-decompose a merged LORA_DATA into a checkpoint loadable by the LoRA Loader."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_data": ("FOLEYTUNE_LORA_DATA", {
                    "tooltip": "Merged result from a Merger / AutoTuner / Merge Selector.",
                }),
                "filename": ("STRING", {
                    "default": "merged_lora",
                    "tooltip": "Output name under the loras folder (subdirs allowed). The extension is "
                               "added from save_format. A sidecar .json holds the metadata.",
                }),
                "save_rank": ("INT", {
                    "default": 0, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Rank of the saved LoRA. 0 = auto (smallest rank keeping energy_threshold "
                               "of the SVD energy). Non-zero = force this rank. Higher = larger file, closer to the merge.",
                }),
                "save_format": (["safetensors", "pt"], {
                    "default": "safetensors",
                    "tooltip": "safetensors (+ sidecar .json) or a single .pt with embedded meta. Saved in bf16.",
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Optional activation prompt to embed (prepended to the merged source prompts).",
                }),
                "description": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Optional free-text description stored in the checkpoint metadata.",
                }),
                "energy_threshold": ("FLOAT", {
                    "default": 0.99, "min": 0.5, "max": 1.0, "step": 0.01,
                    "tooltip": "Auto-rank only (save_rank=0): fraction of singular-value energy to retain "
                               "per layer. Higher = higher rank = more faithful, larger file.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_lora"
    OUTPUT_NODE = True
    CATEGORY = "FoleyTune"
    DESCRIPTION = (
        "Decompose merged deltas back into low-rank lora_A/lora_B and write a "
        ".safetensors (+ .json) or .pt checkpoint loadable by FoleyTune LoRA Loader."
    )

    def save_lora(self, lora_data, filename, save_rank, save_format,
                  prompt="", description="", energy_threshold=0.99):
        if not lora_data or not lora_data.get("deltas"):
            raise ValueError("Save Merged LoRA: empty lora_data.")

        loras_dir = folder_paths.get_folder_paths("loras")[0]
        rank_mode = "fixed" if save_rank > 0 else "auto"
        rank_cap = save_rank if save_rank > 0 else lora_data.get("rank_hint", 0)

        # Pass 1: SVD-decompose each layer. In auto mode the rank can differ per
        # layer, so we keep the 2D factors and the true rank, then unify below.
        extracted = []  # (module_path, is_conv, shape, down2d [r,in], up2d [out,r], r)
        max_rank = 0
        skipped = []
        for module_path, delta in lora_data["deltas"].items():
            d = delta.float()
            is_conv = d.ndim == 3
            if is_conv:
                c_out, c_in, k = d.shape
                mat = d.reshape(c_out, c_in * k)
            elif d.ndim == 2:
                mat = d
            else:
                skipped.append(module_path)
                continue
            res = extract_lora_svd(mat, rank=rank_cap, energy_threshold=energy_threshold,
                                   rank_mode=rank_mode)
            if res is None:
                continue
            down, up, r = res  # down [r, in], up [out, r] (2D)
            max_rank = max(max_rank, r)
            extracted.append((module_path, is_conv, tuple(d.shape), down, up, r))

        if not extracted:
            raise ValueError("Save Merged LoRA: no decomposable layers (all near-zero/unsupported).")
        if skipped:
            logger.warning("Save Merged LoRA: skipped %d non-2D/3D layers.", len(skipped))

        # Pass 2: pad every layer to a UNIFORM rank (= max). The LoRA Loader wraps
        # all target layers at meta["rank"] and copies weights by key, so per-layer
        # ranks must match; zero-padding the extra rank dims preserves up @ down
        # exactly (the padded singular directions contribute nothing).
        chosen_rank = max_rank
        state_dict = {}
        for module_path, is_conv, shape, down, up, r in extracted:
            in_f, out_f = down.shape[1], up.shape[0]
            if r < chosen_rank:
                down_p = torch.zeros(chosen_rank, in_f, dtype=down.dtype)
                down_p[:r] = down
                up_p = torch.zeros(out_f, chosen_rank, dtype=up.dtype)
                up_p[:, :r] = up
                down, up = down_p, up_p
            if is_conv:
                c_out, c_in, k = shape
                down = down.reshape(chosen_rank, c_in, k)   # [rank, c_in, k]
                up = up.reshape(c_out, chosen_rank, 1)      # [c_out, rank, 1]
            state_dict[f"{module_path}.lora_A"] = down.to(torch.bfloat16).contiguous()
            state_dict[f"{module_path}.lora_B"] = up.to(torch.bfloat16).contiguous()
        meta = {
            "rank": chosen_rank,
            "alpha": float(chosen_rank),     # scaling = alpha/rank = 1.0 -> strength 1.0 reproduces merge
            "use_rslora": False,
            "init_mode": "standard",
            "target": list(lora_data.get("target_suffixes", [])),
            "lora_dropout": 0.0,
            "prompts": lora_data.get("prompts", []),
            "source_loras": lora_data.get("source_names", []),
            "description": description,
            "merge_mode": "foley_merged",
        }
        if prompt and prompt not in meta["prompts"]:
            meta["prompts"] = [prompt] + meta["prompts"]

        # Resolve + sandbox the output path under the loras folder.
        rel = filename
        if not rel.endswith((".safetensors", ".pt")):
            rel = rel + (".safetensors" if save_format == "safetensors" else ".pt")
        out_path = os.path.abspath(os.path.join(loras_dir, rel))
        if os.path.commonpath([os.path.abspath(loras_dir), out_path]) != os.path.abspath(loras_dir):
            raise ValueError("Save Merged LoRA: filename escapes the loras folder.")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if save_format == "safetensors" or out_path.endswith(".safetensors"):
            save_safetensors(state_dict, out_path)
            with open(out_path.replace(".safetensors", ".json"), "w") as f:
                json.dump(meta, f, indent=2, default=repr)
        else:
            torch.save({"state_dict": state_dict, "meta": meta}, out_path)

        logger.info("Saved merged LoRA: %s (%d layers, rank=%d)",
                    out_path, len(state_dict) // 2, chosen_rank)
        return (out_path,)


# --- Nodes: Save / Load Tuner Data ------------------------------------------

class FoleyTuneSaveTunerData:
    """Persist AutoTuner TUNER_DATA to the tuner_data folder as JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tuner_data": ("FOLEYTUNE_TUNER_DATA", {
                    "tooltip": "Ranked configs from a FoleyTune LoRA AutoTuner.",
                }),
                "filename": ("STRING", {
                    "default": "tuner_data",
                    "tooltip": "Output name under the tuner_data folder (subdirs allowed); '.tuner' is appended.",
                }),
                "overwrite": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If off, append _001, _002, ... instead of overwriting an existing file.",
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Optional prompt stored alongside the rankings.",
                }),
                "description": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Optional free-text note stored alongside the rankings.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "FoleyTune"
    DESCRIPTION = "Save AutoTuner TUNER_DATA (ranked configs) for later replay."

    def save(self, tuner_data, filename, overwrite, prompt="", description=""):
        base_dir = folder_paths.get_folder_paths("tuner_data")[0]
        data = dict(tuner_data or {})
        if prompt:
            data["prompt"] = prompt
        if description:
            data["description"] = description

        rel = filename if filename.endswith(".tuner") else filename + ".tuner"
        out_path = os.path.abspath(os.path.join(base_dir, rel))
        if os.path.commonpath([os.path.abspath(base_dir), out_path]) != os.path.abspath(base_dir):
            raise ValueError("Save Tuner Data: filename escapes the tuner_data folder.")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path) and not overwrite:
            stem, ext = os.path.splitext(out_path)
            i = 1
            while os.path.exists(f"{stem}_{i:03d}{ext}"):
                i += 1
            out_path = f"{stem}_{i:03d}{ext}"

        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=repr)
        os.replace(tmp, out_path)
        logger.info("Saved tuner data: %s", out_path)
        return (out_path,)


class FoleyTuneLoadTunerData:
    """Load saved TUNER_DATA from the tuner_data folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tuner_data_file": (folder_paths.get_filename_list("tuner_data"), {
                    "tooltip": "A saved .tuner file (from Save Tuner Data) to feed into the Merge Selector.",
                }),
            },
        }

    RETURN_TYPES = ("FOLEYTUNE_TUNER_DATA", "STRING", "STRING")
    RETURN_NAMES = ("tuner_data", "prompt", "description")
    FUNCTION = "load"
    CATEGORY = "FoleyTune"
    DESCRIPTION = "Load saved AutoTuner TUNER_DATA for the Merge Selector."

    @classmethod
    def IS_CHANGED(cls, tuner_data_file):
        try:
            path = folder_paths.get_full_path_or_raise("tuner_data", tuner_data_file)
            return os.path.getmtime(path)
        except Exception:
            return float("nan")

    def load(self, tuner_data_file):
        path = folder_paths.get_full_path_or_raise("tuner_data", tuner_data_file)
        with open(path) as f:
            data = json.load(f)
        return (data, data.get("prompt", ""), data.get("description", ""))


# --- Node registration -------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "FoleyTuneLoRAStack": FoleyTuneLoRAStack,
    "FoleyTuneMergeOptions": FoleyTuneMergeOptions,
    "FoleyTuneLoRAMerger": FoleyTuneLoRAMerger,
    "FoleyTuneLoRAAutoTuner": FoleyTuneLoRAAutoTuner,
    "FoleyTuneMergeSelector": FoleyTuneMergeSelector,
    "FoleyTuneSaveMergedLoRA": FoleyTuneSaveMergedLoRA,
    "FoleyTuneSaveTunerData": FoleyTuneSaveTunerData,
    "FoleyTuneLoadTunerData": FoleyTuneLoadTunerData,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoleyTuneLoRAStack": "FoleyTune LoRA Stack",
    "FoleyTuneMergeOptions": "FoleyTune Merge Options",
    "FoleyTuneLoRAMerger": "FoleyTune LoRA Merger",
    "FoleyTuneLoRAAutoTuner": "FoleyTune LoRA AutoTuner",
    "FoleyTuneMergeSelector": "FoleyTune Merge Selector",
    "FoleyTuneSaveMergedLoRA": "FoleyTune Save Merged LoRA",
    "FoleyTuneSaveTunerData": "FoleyTune Save Tuner Data",
    "FoleyTuneLoadTunerData": "FoleyTune Load Tuner Data",
}
