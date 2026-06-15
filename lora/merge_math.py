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
    # Foley is a single DiT-class architecture. Orthogonal LoRA stacks should
    # NOT be scaled down (full-strength independent updates) — floor at 1.0,
    # matching the upstream video-DiT presets (Wan/LTX/ACE-Step).
    "auto_strength_orthogonal_floor": 1.0,
    # Excess-conflict baseline: only positions whose magnitude exceeds this
    # fraction of the RMS count toward the arccos sign-conflict baseline.
    "noise_floor_ratio": 0.05,
    # Heuristic candidate scoring: density at which DARE is considered ideal.
    "dare_ideal_density": 0.7,
}


# --- Conflict analysis ---

def sample_conflict(diff_a, diff_b, max_samples=100000):
    """Compute sign conflict ratio, cosine similarity and excess conflict.

    Raw sign-conflict ratio is misleading for uncorrelated (orthogonal) LoRAs:
    two random vectors disagree in sign on ~50% of positions purely by chance.
    `excess_conflict` subtracts that statistical baseline. On the strong-magnitude
    subset we compute the (unweighted) sign-mismatch fraction and the cosine
    similarity; the expected mismatch for a bivariate-normal pair with that cosine
    is ``arccos(cos)/pi`` (Sheppard's theorem). The excess is the part of the
    observed mismatch that the alignment does NOT explain — i.e. genuine conflict.

    Returns dict with keys: n_overlap, conflict_ratio, cos_sim, dot, norm_a_sq,
    norm_b_sq, expected_conflict, excess_conflict.
    """
    _empty = {"n_overlap": 0, "conflict_ratio": 0.0, "cos_sim": 0.0,
              "dot": 0.0, "norm_a_sq": 0.0, "norm_b_sq": 0.0,
              "expected_conflict": 0.0, "excess_conflict": 0.0}

    flat_a = diff_a.flatten().float()
    flat_b = diff_b.flatten().float()

    if flat_a.numel() != flat_b.numel():
        return dict(_empty)

    n = flat_a.numel()
    if n > max_samples:
        g = torch.Generator(device=flat_a.device).manual_seed(42)
        indices = torch.randperm(n, device=flat_a.device, generator=g)[:max_samples]
        flat_a = flat_a[indices]
        flat_b = flat_b[indices]

    mask = (flat_a != 0) & (flat_b != 0)
    n_overlap = mask.sum().item()
    if n_overlap == 0:
        return dict(_empty)

    a_ov = flat_a[mask]
    b_ov = flat_b[mask]
    n_conflict = (a_ov.sign() != b_ov.sign()).sum().item()
    dot = (a_ov * b_ov).sum().item()
    norm_a_sq = (a_ov * a_ov).sum().item()
    norm_b_sq = (b_ov * b_ov).sum().item()
    denom = math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq)
    cos_sim = dot / denom if denom > 0 else 0.0

    # --- Excess-conflict baseline on the strong-magnitude subset ---
    a_rms = math.sqrt(norm_a_sq / n_overlap)
    b_rms = math.sqrt(norm_b_sq / n_overlap)
    noise_floor = max(a_rms, b_rms) * THRESHOLDS["noise_floor_ratio"]
    strong = (a_ov.abs() > noise_floor) & (b_ov.abs() > noise_floor)
    if strong.sum().item() == 0:
        a_s, b_s = a_ov, b_ov
    else:
        a_s, b_s = a_ov[strong], b_ov[strong]
    n_strong = a_s.numel()
    strong_mismatch = (a_s.sign() != b_s.sign()).sum().item() / n_strong
    dot_s = (a_s * b_s).sum().item()
    denom_s = math.sqrt((a_s * a_s).sum().item()) * math.sqrt((b_s * b_s).sum().item())
    cos_strong = dot_s / denom_s if denom_s > 0 else 0.0
    cos_strong = max(-1.0, min(1.0, cos_strong))
    expected_conflict = math.acos(cos_strong) / math.pi
    excess_conflict = max(strong_mismatch - expected_conflict, 0.0)

    return {
        "n_overlap": n_overlap,
        "conflict_ratio": n_conflict / n_overlap,
        "cos_sim": cos_sim,
        "dot": dot,
        "norm_a_sq": norm_a_sq,
        "norm_b_sq": norm_b_sq,
        "expected_conflict": expected_conflict,
        "excess_conflict": excess_conflict,
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
    """Weighted Karcher (Fréchet) mean of N diffs on the unit sphere.

    items = [(tensor, weight), ...]. Iterative pairwise SLERP is order-dependent
    and collapses for N>=3; this computes the true spherical mean via tangent-space
    (log-map) weighted averaging + exp-map back, which is order-independent and
    converges in a few iterations. Norm-corrected to the weighted mean of input
    norms, and reshaped back to the input tensor shape.
    """
    orig_shape = items[0][0].shape
    vecs = [v.flatten().float() for v, _ in items]
    weights = [abs(w) for _, w in items]
    total_w = sum(weights)
    if total_w == 0:
        return torch.zeros(orig_shape, dtype=torch.float32, device=vecs[0].device)

    norms = [v.norm().item() for v in vecs]
    target_norm = sum(nrm * w for nrm, w in zip(norms, weights)) / total_w

    # Drop zero-norm rows; unit-normalize the survivors.
    rows, w_keep = [], []
    for v, nrm, w in zip(vecs, norms, weights):
        if nrm > 1e-12:
            rows.append(v / nrm)
            w_keep.append(w)
    if not rows:
        return torch.zeros(orig_shape, dtype=torch.float32, device=vecs[0].device)
    if len(rows) == 1:
        return (rows[0] * target_norm).reshape(orig_shape)

    U = torch.stack(rows, dim=0)                       # [N, D] unit rows
    w_t = torch.tensor(w_keep, dtype=U.dtype, device=U.device)
    w_t = w_t / w_t.sum()

    # Initialize at the normalized weighted chordal mean.
    m = U.t().mv(w_t)
    nm = m.norm()
    m = m / nm if nm.item() > 1e-12 else U[0].clone()

    for _ in range(8):
        cos = (U @ m).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos)
        sin_theta = torch.sin(theta)
        coef = torch.where(theta < 1e-7, torch.zeros_like(theta), w_t * theta / sin_theta)
        tangent = U.t().mv(coef) - (coef * cos).sum() * m
        t_norm = tangent.norm()
        if t_norm.item() < 1e-7:
            break
        m = torch.cos(t_norm) * m + (torch.sin(t_norm) / t_norm) * tangent
        m = m / m.norm().clamp(min=1e-12)

    return (m * target_norm).reshape(orig_shape)


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


# --- Auto-strength ---

def compute_auto_strength(strengths, norm_sq_list, dot_accum, floor=None):
    """Energy-based auto-strength normalization.

    strengths: list of per-LoRA strength values
    norm_sq_list: list of accumulated Frobenius norm^2 per LoRA
    dot_accum: dict {(i,j): accumulated_dot} for pairwise cross-terms
    floor: explicit minimum scale factor. If >= 0 it bounds the reduction
        regardless of alignment. If None (or < 0) the orthogonal floor from
        THRESHOLDS is applied only when the stack is near-orthogonal.

    Returns: scale factor to multiply all strengths by.
    """
    n = len(strengths)
    effective = [abs(strengths[i]) * math.sqrt(max(norm_sq_list[i], 0.0)) for i in range(n)]
    nonzero = [e for e in effective if e > 0]
    if len(nonzero) <= 1:
        return 1.0

    energy_sq = sum((strengths[i] ** 2) * norm_sq_list[i] for i in range(n))
    for (i, j), dot in dot_accum.items():
        energy_sq += 2.0 * strengths[i] * strengths[j] * dot

    energy_sq = max(energy_sq, 0.0)
    current_energy = math.sqrt(energy_sq)
    reference_energy = max(effective)
    scale = min(reference_energy / current_energy, 1.0) if current_energy > 0 else 1.0

    # Explicit user floor bounds the reduction regardless of alignment.
    if floor is not None and floor >= 0:
        return max(scale, floor)

    # Otherwise, only floor near-orthogonal stacks (independent updates that
    # should not be diluted) using the architecture orthogonal floor.
    pairwise_cos = []
    for (i, j), dot in dot_accum.items():
        denom = math.sqrt(max(norm_sq_list[i], 0.0)) * math.sqrt(max(norm_sq_list[j], 0.0))
        if denom > 0:
            pairwise_cos.append(dot / denom)

    if pairwise_cos:
        avg_cos = sum(pairwise_cos) / len(pairwise_cos)
        if abs(avg_cos) <= THRESHOLDS["alignment_threshold"]:
            scale = max(scale, THRESHOLDS["auto_strength_orthogonal_floor"])

    return scale


# --- Delta computation ---

def compute_deltas(state_dict, rank, alpha, strength, use_rslora=False):
    """Compute merged delta (B @ A * scaling * strength) per layer.

    state_dict keys follow the pattern: <layer_name>.base.lora_A / .base.lora_B
    or <layer_name>.lora_A / <layer_name>.lora_B.

    Returns dict {layer_name: delta_tensor}.
    """
    if use_rslora:
        scaling = alpha / math.sqrt(rank)
    else:
        scaling = alpha / rank

    pairs = {}
    for k, v in state_dict.items():
        if "lora_A" in k:
            layer = k.rsplit(".lora_A", 1)[0]
            if layer.endswith(".base"):
                layer = layer[:-5]
            pairs.setdefault(layer, {})["A"] = v
        elif "lora_B" in k:
            layer = k.rsplit(".lora_B", 1)[0]
            if layer.endswith(".base"):
                layer = layer[:-5]
            pairs.setdefault(layer, {})["B"] = v

    deltas = {}
    for layer, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        A = ab["A"].float()
        B = ab["B"].float()
        if A.ndim == 3 or B.ndim == 3:
            if A.ndim != 3 or B.ndim != 3 or B.shape[-1] != 1:
                raise ValueError(f"Unsupported conv LoRA shape for {layer}: A={tuple(A.shape)} B={tuple(B.shape)}")
            deltas[layer] = torch.einsum("or,rik->oik", B.squeeze(-1), A) * scaling * strength
        else:
            deltas[layer] = (B @ A) * scaling * strength

    return deltas


# --- Merged-LoRA extraction (for saving a merge back to a checkpoint) ---

def extract_lora_svd(delta, rank=0, energy_threshold=0.99, rank_mode="auto"):
    """SVD-decompose a 2D weight delta into low-rank LoRA factors.

    Returns ``(lora_down [r, in], lora_up [out, r], r)`` such that
    ``lora_up @ lora_down`` reconstructs ``delta`` (exactly at full rank).
    Returns None for a near-zero delta or a non-2D input (callers reshape conv
    deltas to 2D before calling, then reshape the factors back).

    rank_mode='auto': pick the smallest rank retaining ``energy_threshold`` of the
    singular-value energy (capped at the matrix's max rank, and at ``rank`` if > 0).
    rank_mode='fixed': use ``rank`` (capped at max rank).
    """
    W = delta.float()
    if W.ndim != 2:
        return None
    out_f, in_f = W.shape
    max_rank = min(out_f, in_f)
    if max_rank == 0 or W.abs().max().item() < 1e-12:
        return None
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    except Exception:
        return None

    if rank_mode == "fixed" and rank > 0:
        r = min(rank, max_rank)
    else:
        total = float(S.sum().item())
        if total <= 0:
            return None
        csum = torch.cumsum(S, dim=0) / total
        r = int((csum < energy_threshold).sum().item()) + 1
        r = max(1, min(r, max_rank))
        if rank > 0:
            r = min(r, rank)

    sqrt_S = torch.sqrt(S[:r])
    lora_up = U[:, :r] * sqrt_S.unsqueeze(0)      # [out, r]
    lora_down = Vh[:r] * sqrt_S.unsqueeze(1)      # [r, in]
    return lora_down, lora_up, r


# --- Heuristic candidate scoring (for AutoTuner top-N ranking) ---

def score_config(merge_approach, sparsification, density, block_metrics, thresholds=None):
    """Heuristic quality score in [0, 1] for a candidate merge config.

    Ranks candidates from aggregate per-block analysis metrics WITHOUT running a
    real merge or model inference (analysis is config-independent and computed
    once). Higher = better predicted fit.

    merge_approach: 'per_block_adaptive' | 'ties' | 'weighted_average' | 'slerp'
    sparsification: 'disabled' | 'dare' | 'conflict_aware'
    block_metrics: list of dicts with 'avg_cos_sim', 'avg_conflict', 'excess_conflict'

    Returns (score, breakdown_dict).
    """
    th = thresholds or THRESHOLDS
    if not block_metrics:
        return 0.0, {"mode": 0.0, "spars": 0.0, "align": 0.0}

    n = len(block_metrics)
    avg_cos = sum(b.get("avg_cos_sim", 0.0) for b in block_metrics) / n
    excess_vals = [max(b.get("excess_conflict", 0.0), 0.0) for b in block_metrics]
    avg_excess = sum(excess_vals) / n
    spread = max(excess_vals) - min(excess_vals)

    ties_thresh = th["ties_conflict_threshold"]
    ortho_cos = th["orthogonal_cos_sim_max"]
    consensus_cos = th["consensus_cos_sim_min"]

    # --- Mode fit (0..0.5): does the strategy match the measured regime? ---
    if merge_approach == "per_block_adaptive":
        # Adaptive wins when blocks differ in conflict (heterogeneous stack)
        # or when the stack is cleanly orthogonal.
        if spread > 0.10:
            mode = 0.50
        elif abs(avg_cos) < ortho_cos and avg_excess < ortho_cos:
            mode = 0.42
        else:
            mode = 0.40
    elif merge_approach == "ties":
        if avg_excess > ties_thresh:
            mode = 0.45
        elif avg_excess > ties_thresh * 0.5:
            mode = 0.25
        else:
            mode = 0.10
    elif merge_approach == "slerp":
        mode = 0.40 if (avg_excess < ties_thresh and abs(avg_cos) < consensus_cos) else 0.15
    else:  # weighted_average
        if avg_excess < th["consensus_conflict_max"]:
            mode = 0.40
        elif avg_excess < ties_thresh:
            mode = 0.25
        else:
            mode = 0.10

    # --- Sparsification fit (0..0.3) ---
    if sparsification == "disabled":
        spars = 0.30 if avg_excess < ties_thresh else 0.18
    elif sparsification == "conflict_aware":
        spars = 0.30 if avg_excess > ties_thresh else 0.20
    else:  # dare
        spars = 0.22

    # --- Alignment bonus (0..0.2): reward low residual interference ---
    align = 0.20 * max(0.0, 1.0 - avg_excess / max(ties_thresh, 1e-6))
    align = min(align, 0.20)

    score = max(0.0, min(mode + spars + align, 1.0))
    return score, {"mode": round(mode, 4), "spars": round(spars, 4),
                   "align": round(align, 4), "avg_cos": round(avg_cos, 4),
                   "avg_excess": round(avg_excess, 4), "spread": round(spread, 4)}
