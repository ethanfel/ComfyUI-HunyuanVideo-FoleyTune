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


# --- Merge strategies ---

def merge_weighted_average(deltas, weights):
    """Weighted average of delta tensors. weights are NOT normalized — the
    caller decides whether the total should be 1.0."""
    result = torch.zeros_like(deltas[0], dtype=torch.float32)
    total_w = sum(abs(w) for w in weights)
    if total_w == 0:
        return result
    for d, w in zip(deltas, weights):
        result.add_(d.float() * (w / total_w))
    return result


def ties_trim(tensor, density):
    """TIES step 1: keep top-k% by absolute magnitude, zero the rest."""
    flat = tensor.flatten()
    n = flat.numel()
    k = max(1, int(n * density))
    if k >= n:
        return tensor.clone()
    _, indices = torch.topk(flat.abs(), k)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[indices] = True
    return (flat * mask).reshape(tensor.shape)


def ties_elect_sign(trimmed_diffs, method="frequency"):
    """TIES step 2: majority sign vote per position."""
    total = torch.zeros_like(trimmed_diffs[0], dtype=torch.float32)
    if method == "total":
        for d in trimmed_diffs:
            total.add_(d.float())
    else:
        for d in trimmed_diffs:
            total.add_(d.sign())
    return torch.where(total >= 0,
                       torch.tensor(1.0, device=total.device),
                       torch.tensor(-1.0, device=total.device))


def ties_disjoint_merge(trimmed_diffs, weights, majority_sign):
    """TIES step 3: average only contributors agreeing with majority sign."""
    result = torch.zeros_like(trimmed_diffs[0], dtype=torch.float32)
    count = torch.zeros_like(result)
    for d, w in zip(trimmed_diffs, weights):
        d_f = d.float()
        agree = (d_f * majority_sign) > 0
        result.add_(torch.where(agree, d_f * w, torch.zeros_like(d_f)))
        count.add_(agree.float())
    count.clamp_(min=1.0)
    return result.div_(count)


def merge_ties(deltas, weights, density=0.7, sign_method="frequency"):
    """Full TIES pipeline: trim -> elect sign -> disjoint merge."""
    trimmed = [ties_trim(d.float(), density) for d in deltas]
    majority = ties_elect_sign(trimmed, method=sign_method)
    return ties_disjoint_merge(trimmed, weights, majority)


def merge_slerp(a, b, t):
    """Spherical linear interpolation between two tensors.
    t=0 returns a, t=1 returns b. Norm-corrected."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    norm_a = a_flat.norm()
    norm_b = b_flat.norm()
    if norm_a < 1e-8 or norm_b < 1e-8:
        return ((1.0 - t) * a_flat + t * b_flat).reshape(a.shape)
    cos_theta = (torch.dot(a_flat, b_flat) / (norm_a * norm_b)).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    if theta.item() < 1e-6:
        return ((1.0 - t) * a_flat + t * b_flat).reshape(a.shape)
    sin_theta = torch.sin(theta)
    coeff_a = torch.sin((1.0 - t) * theta) / sin_theta
    coeff_b = torch.sin(t * theta) / sin_theta
    result = coeff_a * a_flat + coeff_b * b_flat
    target_norm = (1.0 - t) * norm_a.item() + t * norm_b.item()
    current_norm = result.norm().item()
    if current_norm > 1e-8:
        result = result * (target_norm / current_norm)
    return result.reshape(a.shape)


def merge_slerp_n(items):
    """Iterative pairwise SLERP for N diffs. items = [(tensor, weight), ...].
    Strongest LoRA anchors direction."""
    items = [(v.flatten().float(), abs(w)) for v, w in items]
    items.sort(key=lambda x: x[1], reverse=True)
    total_w = sum(w for _, w in items)
    if total_w == 0:
        return torch.zeros_like(items[0][0])
    input_norms = [(v.norm().item(), w) for v, w in items]
    acc_v, acc_w = items[0]
    for k in range(1, len(items)):
        next_v, next_w = items[k]
        frac = next_w / (acc_w + next_w) if (acc_w + next_w) > 0 else 0.5
        norm_acc = acc_v.norm()
        norm_next = next_v.norm()
        denom = norm_acc * norm_next
        if denom > 0:
            cos_theta = (torch.dot(acc_v, next_v) / denom).clamp(-1.0, 1.0)
        else:
            cos_theta = torch.tensor(1.0, device=acc_v.device)
        theta = torch.acos(cos_theta)
        if theta.item() < 1e-6:
            acc_v = (1.0 - frac) * acc_v + frac * next_v
        else:
            sin_theta = torch.sin(theta)
            a = torch.sin((1.0 - frac) * theta) / sin_theta
            b = torch.sin(frac * theta) / sin_theta
            acc_v = a * acc_v + b * next_v
        acc_w += next_w
    target_norm = sum(n * w for n, w in input_norms) / total_w
    current_norm = acc_v.norm().item()
    if current_norm > 1e-8:
        acc_v = acc_v * (target_norm / current_norm)
    return acc_v


# --- Sparsification ---

def dare_sparsify(tensor, density, seed=None, dampening=0.0):
    """DARE: randomly drop parameters and rescale survivors."""
    if density >= 1.0:
        return tensor.clone()
    gen = torch.Generator(device=tensor.device)
    if seed is not None:
        gen.manual_seed(seed)
    mask = torch.bernoulli(
        torch.full(tensor.shape, density, dtype=tensor.dtype, device=tensor.device),
        generator=gen,
    )
    q = density + dampening * (1.0 - density)
    return tensor * mask * (1.0 / q)


def compute_conflict_mask(diffs_with_weights):
    """Boolean mask: True where 2+ diffs have opposing signs."""
    has_pos = torch.zeros_like(diffs_with_weights[0][0], dtype=torch.bool)
    has_neg = torch.zeros_like(has_pos)
    for diff, weight in diffs_with_weights:
        effective = diff if weight >= 0 else -diff
        nonzero = effective != 0
        has_pos |= (nonzero & (effective > 0))
        has_neg |= (nonzero & (effective < 0))
    return has_pos & has_neg
