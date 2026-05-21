"""Self-contained LoRA merge math for FoleyTune.

Conflict analysis, merge strategies (TIES, SLERP, weighted average),
auto-strength normalization. No dependency on ZImage LoRA Merger.
"""

import math
import torch

# --- Architecture thresholds (DiT preset — Foley is a DiT variant) ---

THRESHOLDS = {
    "consensus_cos_sim_min": 0.5,
    "consensus_conflict_max": 0.15,
    "orthogonal_cos_sim_max": 0.25,
    "orthogonal_conflict_max": 0.60,
    "ties_conflict_threshold": 0.25,
    "magnitude_ratio_total_sign": 2.0,
    "alignment_threshold": 0.1,
    "auto_strength_orthogonal_floor": 0.85,
}


# --- Conflict analysis ---

def sample_conflict(diff_a, diff_b, max_samples=100000):
    """Compute sign conflict ratio and cosine similarity between two deltas.

    Returns dict with keys: n_overlap, conflict_ratio, cos_sim, dot,
    norm_a_sq, norm_b_sq.
    """
    flat_a = diff_a.flatten().float()
    flat_b = diff_b.flatten().float()

    if flat_a.numel() != flat_b.numel():
        return {"n_overlap": 0, "conflict_ratio": 0.0, "cos_sim": 0.0,
                "dot": 0.0, "norm_a_sq": 0.0, "norm_b_sq": 0.0}

    n = flat_a.numel()
    if n > max_samples:
        g = torch.Generator(device=flat_a.device).manual_seed(42)
        indices = torch.randperm(n, device=flat_a.device, generator=g)[:max_samples]
        flat_a = flat_a[indices]
        flat_b = flat_b[indices]

    mask = (flat_a != 0) & (flat_b != 0)
    n_overlap = mask.sum().item()
    if n_overlap == 0:
        return {"n_overlap": 0, "conflict_ratio": 0.0, "cos_sim": 0.0,
                "dot": 0.0, "norm_a_sq": 0.0, "norm_b_sq": 0.0}

    a_ov = flat_a[mask]
    b_ov = flat_b[mask]
    n_conflict = (a_ov.sign() != b_ov.sign()).sum().item()
    dot = (a_ov * b_ov).sum().item()
    norm_a_sq = (a_ov * a_ov).sum().item()
    norm_b_sq = (b_ov * b_ov).sum().item()
    denom = math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq)
    cos_sim = dot / denom if denom > 0 else 0.0

    return {
        "n_overlap": n_overlap,
        "conflict_ratio": n_conflict / n_overlap,
        "cos_sim": cos_sim,
        "dot": dot,
        "norm_a_sq": norm_a_sq,
        "norm_b_sq": norm_b_sq,
    }


def classify_relationship(cos_sim, conflict_ratio, excess_conflict, subspace_overlap=0.0):
    """Classify the relationship between LoRAs as consensus/orthogonal/conflicting."""
    effective_conflict = max(excess_conflict, 0.0)
    if subspace_overlap > 0:
        effective_conflict *= (0.5 + 0.5 * subspace_overlap)

    if (cos_sim > THRESHOLDS["consensus_cos_sim_min"]
            and effective_conflict < THRESHOLDS["consensus_conflict_max"]
            and subspace_overlap >= 0.35):
        return "consensus"

    if (abs(cos_sim) < THRESHOLDS["orthogonal_cos_sim_max"]
            and effective_conflict < THRESHOLDS["orthogonal_conflict_max"]
            and subspace_overlap < 0.35):
        return "orthogonal"

    if effective_conflict > THRESHOLDS["ties_conflict_threshold"]:
        return "conflicting"

    return "weighted_average"
