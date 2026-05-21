# FoleyTune LoRA Merge Nodes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two LoRA merge ComfyUI nodes (`FoleyTuneLoRAMerger` and `FoleyTuneLoRAAutoTuner`) to the FoleyTune wrapper, with a self-contained merge math module.

**Architecture:** A new `lora/merge_math.py` (~300 lines) provides the core merge primitives (TIES, SLERP, conflict analysis, auto-strength). A new `nodes_merge.py` defines two ComfyUI nodes that load FoleyTune LoRA checkpoints, compute deltas, group by block prefix, and apply merge strategies. Both nodes operate on `FOLEYTUNE_MODEL` with no dependency on the ZImage LoRA Merger.

**Tech Stack:** Python, PyTorch, ComfyUI node API, unittest.

**Target repo:** `/media/p5/ComfyUI-HunyuanVideo-Foley/` (also installed at `/media/p5/Comfyui/custom_nodes/ComfyUI-HunyuanVideo-FoleyTune/`)

---

## Task 1: Create `lora/merge_math.py` — conflict analysis and classification

**Files:**
- Create: `lora/merge_math.py`
- Create: `tests/test_merge_math.py`

**Step 1: Write the failing tests**

Create `tests/test_merge_math.py`:

```python
"""Tests for FoleyTune LoRA merge math primitives."""

import unittest
import torch


class TestConflictAnalysis(unittest.TestCase):

    def test_sample_conflict_identical_tensors(self):
        from lora.merge_math import sample_conflict
        a = torch.ones(100)
        b = torch.ones(100)
        result = sample_conflict(a, b)
        self.assertGreater(result["n_overlap"], 0)
        self.assertAlmostEqual(result["conflict_ratio"], 0.0)
        self.assertAlmostEqual(result["cos_sim"], 1.0, places=3)

    def test_sample_conflict_opposing_tensors(self):
        from lora.merge_math import sample_conflict
        a = torch.ones(100)
        b = -torch.ones(100)
        result = sample_conflict(a, b)
        self.assertAlmostEqual(result["conflict_ratio"], 1.0)
        self.assertAlmostEqual(result["cos_sim"], -1.0, places=3)

    def test_sample_conflict_orthogonal(self):
        from lora.merge_math import sample_conflict
        a = torch.zeros(200)
        b = torch.zeros(200)
        a[:100] = 1.0
        b[100:] = 1.0
        result = sample_conflict(a, b)
        self.assertEqual(result["n_overlap"], 0)

    def test_sample_conflict_size_mismatch_returns_zero(self):
        from lora.merge_math import sample_conflict
        a = torch.ones(50)
        b = torch.ones(100)
        result = sample_conflict(a, b)
        self.assertEqual(result["n_overlap"], 0)

    def test_classify_consensus(self):
        from lora.merge_math import classify_relationship
        result = classify_relationship(cos_sim=0.7, conflict_ratio=0.05,
                                        excess_conflict=0.02, subspace_overlap=0.5)
        self.assertEqual(result, "consensus")

    def test_classify_orthogonal(self):
        from lora.merge_math import classify_relationship
        result = classify_relationship(cos_sim=0.05, conflict_ratio=0.48,
                                        excess_conflict=0.01, subspace_overlap=0.1)
        self.assertEqual(result, "orthogonal")

    def test_classify_conflicting(self):
        from lora.merge_math import classify_relationship
        result = classify_relationship(cos_sim=0.3, conflict_ratio=0.6,
                                        excess_conflict=0.4, subspace_overlap=0.5)
        self.assertEqual(result, "conflicting")


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run the tests to verify they fail**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley && python -m pytest tests/test_merge_math.py -v`

Expected: ImportError — `lora.merge_math` does not exist.

**Step 3: Implement conflict analysis and classification**

Create `lora/merge_math.py`:

```python
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
```

**Step 4: Run the tests to verify they pass**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley && python -m pytest tests/test_merge_math.py -v`

Expected: 7 tests pass.

**Step 5: Commit**

```bash
cd /media/p5/ComfyUI-HunyuanVideo-Foley
git add lora/merge_math.py tests/test_merge_math.py
git commit -m "feat(merge): add conflict analysis and classification primitives"
```

---

## Task 2: Add merge strategies to `lora/merge_math.py`

**Files:**
- Modify: `lora/merge_math.py`
- Modify: `tests/test_merge_math.py`

**Step 1: Write the failing tests**

Append to `tests/test_merge_math.py`:

```python
class TestMergeStrategies(unittest.TestCase):

    def test_weighted_average_equal_weights(self):
        from lora.merge_math import merge_weighted_average
        a = torch.tensor([2.0, 4.0])
        b = torch.tensor([6.0, 8.0])
        result = merge_weighted_average([a, b], [1.0, 1.0])
        expected = torch.tensor([4.0, 6.0])
        torch.testing.assert_close(result, expected)

    def test_weighted_average_unequal_weights(self):
        from lora.merge_math import merge_weighted_average
        a = torch.tensor([0.0, 10.0])
        b = torch.tensor([10.0, 0.0])
        result = merge_weighted_average([a, b], [0.75, 0.25])
        expected = torch.tensor([2.5, 7.5])
        torch.testing.assert_close(result, expected)

    def test_ties_trims_low_magnitude(self):
        from lora.merge_math import ties_trim
        t = torch.tensor([0.01, 0.5, -0.8, 0.02, -0.9])
        trimmed = ties_trim(t, density=0.6)
        nonzero = (trimmed != 0).sum().item()
        self.assertEqual(nonzero, 3)

    def test_ties_elect_sign(self):
        from lora.merge_math import ties_elect_sign
        a = torch.tensor([1.0, -1.0, 1.0])
        b = torch.tensor([1.0, 1.0, -1.0])
        c = torch.tensor([1.0, -1.0, -1.0])
        majority = ties_elect_sign([a, b, c])
        expected = torch.tensor([1.0, -1.0, -1.0])
        torch.testing.assert_close(majority, expected)

    def test_ties_disjoint_merge(self):
        from lora.merge_math import ties_disjoint_merge
        a = torch.tensor([1.0, -2.0])
        b = torch.tensor([3.0, 1.0])
        majority = torch.tensor([1.0, -1.0])
        result = ties_disjoint_merge([a, b], [1.0, 1.0], majority)
        # Position 0: both positive and majority is +1 → a agrees (1.0), b agrees (3.0) → avg = 2.0
        # Position 1: a is -2.0 (agrees with -1 majority), b is 1.0 (disagrees) → -2.0/1 = -2.0
        self.assertAlmostEqual(result[0].item(), 2.0, places=4)
        self.assertAlmostEqual(result[1].item(), -2.0, places=4)

    def test_merge_ties_full_pipeline(self):
        from lora.merge_math import merge_ties
        a = torch.randn(64, 64)
        b = torch.randn(64, 64)
        result = merge_ties([a, b], [1.0, 1.0], density=0.7)
        self.assertEqual(result.shape, a.shape)

    def test_slerp_midpoint(self):
        from lora.merge_math import merge_slerp
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        result = merge_slerp(a, b, t=0.5)
        self.assertAlmostEqual(result.norm().item(), 0.5, places=3)

    def test_slerp_t0_returns_a(self):
        from lora.merge_math import merge_slerp
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        result = merge_slerp(a, b, t=0.0)
        torch.testing.assert_close(result, a, atol=1e-5, rtol=1e-5)

    def test_slerp_t1_returns_b(self):
        from lora.merge_math import merge_slerp
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        result = merge_slerp(a, b, t=1.0)
        torch.testing.assert_close(result, b, atol=1e-5, rtol=1e-5)

    def test_slerp_parallel_fallback(self):
        from lora.merge_math import merge_slerp
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 4.0, 6.0])
        result = merge_slerp(a, b, t=0.5)
        expected = 0.5 * a + 0.5 * b
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_iterative_slerp_n_diffs(self):
        from lora.merge_math import merge_slerp_n
        a = torch.randn(100)
        b = torch.randn(100)
        c = torch.randn(100)
        result = merge_slerp_n([(a, 1.0), (b, 1.0), (c, 1.0)])
        self.assertEqual(result.shape, a.shape)


class TestDareSparsify(unittest.TestCase):

    def test_dare_density_1_is_identity(self):
        from lora.merge_math import dare_sparsify
        t = torch.randn(100)
        result = dare_sparsify(t, density=1.0)
        torch.testing.assert_close(result, t)

    def test_dare_zeros_some_elements(self):
        from lora.merge_math import dare_sparsify
        t = torch.ones(10000)
        result = dare_sparsify(t, density=0.5, seed=42)
        zero_frac = (result == 0).float().mean().item()
        self.assertAlmostEqual(zero_frac, 0.5, delta=0.05)

    def test_conflict_mask_opposing_signs(self):
        from lora.merge_math import compute_conflict_mask
        a = torch.tensor([1.0, -1.0, 1.0])
        b = torch.tensor([-1.0, -1.0, 1.0])
        mask = compute_conflict_mask([(a, 1.0), (b, 1.0)])
        self.assertTrue(mask[0].item())
        self.assertFalse(mask[1].item())
        self.assertFalse(mask[2].item())
```

**Step 2: Run the tests to verify they fail**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley && python -m pytest tests/test_merge_math.py -v`

Expected: 14 failures — merge functions not yet defined.

**Step 3: Implement merge strategies**

Append to `lora/merge_math.py`:

```python
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
    """Full TIES pipeline: trim → elect sign → disjoint merge."""
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
```

**Step 4: Run the tests to verify they pass**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley && python -m pytest tests/test_merge_math.py -v`

Expected: all 21 tests pass.

**Step 5: Commit**

```bash
cd /media/p5/ComfyUI-HunyuanVideo-Foley
git add lora/merge_math.py tests/test_merge_math.py
git commit -m "feat(merge): add TIES, SLERP, weighted average, and DARE strategies"
```

---

## Task 3: Add auto-strength and delta computation to `lora/merge_math.py`

**Files:**
- Modify: `lora/merge_math.py`
- Modify: `tests/test_merge_math.py`

**Step 1: Write the failing tests**

Append to `tests/test_merge_math.py`:

```python
class TestAutoStrength(unittest.TestCase):

    def test_single_lora_no_adjustment(self):
        from lora.merge_math import compute_auto_strength
        norms = [10.0]
        dots = {}
        strengths = [1.0]
        scale = compute_auto_strength(strengths, norms, dots)
        self.assertAlmostEqual(scale, 1.0)

    def test_two_aligned_loras_scale_down(self):
        from lora.merge_math import compute_auto_strength
        norms = [100.0, 100.0]
        dots = {(0, 1): 80.0}
        strengths = [1.0, 1.0]
        scale = compute_auto_strength(strengths, norms, dots)
        self.assertLess(scale, 1.0)

    def test_two_orthogonal_loras_floor_applied(self):
        from lora.merge_math import compute_auto_strength
        norms = [100.0, 100.0]
        dots = {(0, 1): 0.0}
        strengths = [1.0, 1.0]
        scale = compute_auto_strength(strengths, norms, dots)
        self.assertGreaterEqual(scale, THRESHOLDS["auto_strength_orthogonal_floor"])


class TestComputeDelta(unittest.TestCase):

    def test_compute_deltas_from_state_dict(self):
        from lora.merge_math import compute_deltas
        rank = 4
        sd = {
            "layer.base.lora_A": torch.randn(rank, 64),
            "layer.base.lora_B": torch.randn(128, rank),
        }
        deltas = compute_deltas(sd, rank=rank, alpha=4.0, strength=1.0)
        self.assertIn("layer", deltas)
        self.assertEqual(deltas["layer"].shape, (128, 64))

    def test_compute_deltas_strength_scales(self):
        from lora.merge_math import compute_deltas
        rank = 4
        sd = {
            "layer.base.lora_A": torch.ones(rank, 8),
            "layer.base.lora_B": torch.ones(16, rank),
        }
        d1 = compute_deltas(sd, rank=rank, alpha=4.0, strength=1.0)
        d2 = compute_deltas(sd, rank=rank, alpha=4.0, strength=0.5)
        ratio = d1["layer"].sum().item() / d2["layer"].sum().item()
        self.assertAlmostEqual(ratio, 2.0, places=3)

    def test_compute_deltas_rslora(self):
        from lora.merge_math import compute_deltas
        rank = 16
        sd = {
            "layer.base.lora_A": torch.ones(rank, 8),
            "layer.base.lora_B": torch.ones(16, rank),
        }
        d_normal = compute_deltas(sd, rank=rank, alpha=16.0, strength=1.0, use_rslora=False)
        d_rslora = compute_deltas(sd, rank=rank, alpha=16.0, strength=1.0, use_rslora=True)
        # rslora scaling = alpha/sqrt(rank) vs normal = alpha/rank
        # For rank=16, alpha=16: normal=1.0, rslora=4.0
        ratio = d_rslora["layer"].sum().item() / d_normal["layer"].sum().item()
        self.assertAlmostEqual(ratio, 4.0, places=2)


from lora.merge_math import THRESHOLDS
```

**Step 2: Run the tests to verify they fail**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley && python -m pytest tests/test_merge_math.py::TestAutoStrength tests/test_merge_math.py::TestComputeDelta -v`

Expected: 6 failures — `compute_auto_strength` and `compute_deltas` not defined.

**Step 3: Implement**

Append to `lora/merge_math.py`:

```python
# --- Auto-strength ---

def compute_auto_strength(strengths, norm_sq_list, dot_accum, floor=None):
    """Energy-based auto-strength normalization.

    strengths: list of per-LoRA strength values
    norm_sq_list: list of accumulated Frobenius norm² per LoRA
    dot_accum: dict {(i,j): accumulated_dot} for pairwise cross-terms
    floor: minimum scale factor (None = use orthogonal floor from THRESHOLDS)

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

    pairwise_cos = []
    for (i, j), dot in dot_accum.items():
        denom = math.sqrt(max(norm_sq_list[i], 0.0)) * math.sqrt(max(norm_sq_list[j], 0.0))
        if denom > 0:
            pairwise_cos.append(dot / denom)

    if pairwise_cos:
        avg_cos = sum(pairwise_cos) / len(pairwise_cos)
        if abs(avg_cos) <= THRESHOLDS["alignment_threshold"]:
            ortho_floor = floor if floor is not None else THRESHOLDS["auto_strength_orthogonal_floor"]
            scale = max(scale, ortho_floor)

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
        deltas[layer] = (B @ A) * scaling * strength

    return deltas
```

**Step 4: Run the tests to verify they pass**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley && python -m pytest tests/test_merge_math.py -v`

Expected: all 27 tests pass.

**Step 5: Commit**

```bash
cd /media/p5/ComfyUI-HunyuanVideo-Foley
git add lora/merge_math.py tests/test_merge_math.py
git commit -m "feat(merge): add auto-strength normalization and delta computation"
```

---

## Task 4: Create `nodes_merge.py` — `FoleyTuneLoRAMerger` node

**Files:**
- Create: `nodes_merge.py`
- Create: `tests/test_nodes_merge.py`

**Step 1: Write the failing tests**

Create `tests/test_nodes_merge.py`:

```python
"""Tests for FoleyTune LoRA merge nodes."""

import os
import sys
import unittest
import unittest.mock
import torch

# Ensure the repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFoleyTuneLoRAMerger(unittest.TestCase):

    def test_input_types_required_fields(self):
        # Import with mocked ComfyUI dependencies
        with unittest.mock.patch.dict("sys.modules", {
            "folder_paths": unittest.mock.MagicMock(
                get_filename_list=lambda x: ["lora_a.safetensors", "lora_b.safetensors"],
            ),
            "comfy": unittest.mock.MagicMock(),
            "comfy.model_management": unittest.mock.MagicMock(),
            "comfy.utils": unittest.mock.MagicMock(),
        }):
            from nodes_merge import FoleyTuneLoRAMerger
            inputs = FoleyTuneLoRAMerger.INPUT_TYPES()
            req = inputs["required"]
            self.assertIn("hunyuan_model", req)
            self.assertIn("lora_name_1", req)
            self.assertIn("strength_1", req)
            self.assertIn("lora_name_2", req)
            self.assertIn("strength_2", req)
            self.assertIn("merge_strategy", req)
            opt = inputs.get("optional", {})
            self.assertIn("lora_name_3", opt)
            self.assertIn("lora_name_4", opt)

    def test_input_types_strategy_enum(self):
        with unittest.mock.patch.dict("sys.modules", {
            "folder_paths": unittest.mock.MagicMock(
                get_filename_list=lambda x: [],
            ),
            "comfy": unittest.mock.MagicMock(),
            "comfy.model_management": unittest.mock.MagicMock(),
            "comfy.utils": unittest.mock.MagicMock(),
        }):
            from nodes_merge import FoleyTuneLoRAMerger
            inputs = FoleyTuneLoRAMerger.INPUT_TYPES()
            strategies = inputs["required"]["merge_strategy"][0]
            self.assertIn("weighted_average", strategies)
            self.assertIn("ties", strategies)
            self.assertIn("slerp", strategies)

    def test_return_types(self):
        with unittest.mock.patch.dict("sys.modules", {
            "folder_paths": unittest.mock.MagicMock(get_filename_list=lambda x: []),
            "comfy": unittest.mock.MagicMock(),
            "comfy.model_management": unittest.mock.MagicMock(),
            "comfy.utils": unittest.mock.MagicMock(),
        }):
            from nodes_merge import FoleyTuneLoRAMerger
            self.assertEqual(FoleyTuneLoRAMerger.RETURN_TYPES, ("FOLEYTUNE_MODEL", "STRING"))

    def test_group_deltas_by_block(self):
        with unittest.mock.patch.dict("sys.modules", {
            "folder_paths": unittest.mock.MagicMock(get_filename_list=lambda x: []),
            "comfy": unittest.mock.MagicMock(),
            "comfy.model_management": unittest.mock.MagicMock(),
            "comfy.utils": unittest.mock.MagicMock(),
        }):
            from nodes_merge import FoleyTuneLoRAMerger
            merger = FoleyTuneLoRAMerger()
            deltas = {
                "triple_blocks.0.audio_self_attn_qkv": torch.randn(64, 64),
                "triple_blocks.0.audio_self_proj": torch.randn(64, 64),
                "triple_blocks.1.audio_self_attn_qkv": torch.randn(64, 64),
            }
            groups = merger._group_by_block(deltas)
            self.assertIn("triple_blocks.0", groups)
            self.assertIn("triple_blocks.1", groups)
            self.assertEqual(len(groups["triple_blocks.0"]), 2)
            self.assertEqual(len(groups["triple_blocks.1"]), 1)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run the tests to verify they fail**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley && python -m pytest tests/test_nodes_merge.py -v`

Expected: ImportError — `nodes_merge` does not exist.

**Step 3: Implement `FoleyTuneLoRAMerger`**

Create `nodes_merge.py`:

```python
"""LoRA merge nodes for FoleyTune.

Two nodes:
- FoleyTuneLoRAMerger: manual strategy selection (TIES/SLERP/weighted_average)
- FoleyTuneLoRAAutoTuner: automatic per-block strategy selection with conflict analysis
"""

import os
import re
import copy
import logging

import torch
from safetensors.torch import load_file as load_safetensors

import folder_paths

from .lora.lora import merge_lora_into_weights, FOLEY_TARGET_PRESETS
from .lora.merge_math import (
    compute_deltas, merge_weighted_average, merge_ties, merge_slerp,
    merge_slerp_n, dare_sparsify, compute_conflict_mask,
    sample_conflict, classify_relationship, compute_auto_strength,
    THRESHOLDS,
)

logger = logging.getLogger("FoleyTune")

_BLOCK_RE = re.compile(r"^(triple_blocks\.\d+)\.")


def _load_adapter_checkpoint(path):
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
        groups = {}
        for layer_name, tensor in deltas.items():
            m = _BLOCK_RE.match(layer_name)
            block = m.group(1) if m else "_other"
            groups.setdefault(block, {})[layer_name] = tensor
        return groups

    def _collect_loras(self, lora_name_1, strength_1, lora_name_2, strength_2,
                       lora_name_3=None, strength_3=1.0,
                       lora_name_4=None, strength_4=1.0):
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
        entries = self._collect_loras(lora_name_1, strength_1, lora_name_2, strength_2,
                                      lora_name_3, strength_3, lora_name_4, strength_4)
        if len(entries) < 2:
            raise ValueError("Need at least 2 LoRAs with non-zero strength to merge.")

        n_loras = len(entries)
        if merge_strategy == "slerp" and n_loras > 2:
            logger.warning("SLERP is optimal for 2 LoRAs — using iterative pairwise SLERP for %d.", n_loras)

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
```

**Step 4: Run the tests to verify they pass**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley && python -m pytest tests/test_nodes_merge.py -v`

Expected: 4 tests pass.

**Step 5: Commit**

```bash
cd /media/p5/ComfyUI-HunyuanVideo-Foley
git add nodes_merge.py tests/test_nodes_merge.py
git commit -m "feat(merge): add FoleyTuneLoRAMerger node with TIES/SLERP/weighted_average"
```

---

## Task 5: Add `FoleyTuneLoRAAutoTuner` node

**Files:**
- Modify: `nodes_merge.py`
- Modify: `tests/test_nodes_merge.py`

**Step 1: Write the failing tests**

Append to `tests/test_nodes_merge.py`:

```python
class TestFoleyTuneLoRAAutoTuner(unittest.TestCase):

    def test_input_types_required_fields(self):
        with unittest.mock.patch.dict("sys.modules", {
            "folder_paths": unittest.mock.MagicMock(get_filename_list=lambda x: []),
            "comfy": unittest.mock.MagicMock(),
            "comfy.model_management": unittest.mock.MagicMock(),
            "comfy.utils": unittest.mock.MagicMock(),
        }):
            from nodes_merge import FoleyTuneLoRAAutoTuner
            inputs = FoleyTuneLoRAAutoTuner.INPUT_TYPES()
            req = inputs["required"]
            self.assertIn("hunyuan_model", req)
            self.assertIn("auto_strength", req)
            self.assertIn("sparsification", req)
            self.assertIn("sparsification_density", req)

    def test_return_types_include_report(self):
        with unittest.mock.patch.dict("sys.modules", {
            "folder_paths": unittest.mock.MagicMock(get_filename_list=lambda x: []),
            "comfy": unittest.mock.MagicMock(),
            "comfy.model_management": unittest.mock.MagicMock(),
            "comfy.utils": unittest.mock.MagicMock(),
        }):
            from nodes_merge import FoleyTuneLoRAAutoTuner
            self.assertEqual(FoleyTuneLoRAAutoTuner.RETURN_TYPES,
                             ("FOLEYTUNE_MODEL", "STRING", "STRING"))

    def test_analyze_block_returns_strategy(self):
        with unittest.mock.patch.dict("sys.modules", {
            "folder_paths": unittest.mock.MagicMock(get_filename_list=lambda x: []),
            "comfy": unittest.mock.MagicMock(),
            "comfy.model_management": unittest.mock.MagicMock(),
            "comfy.utils": unittest.mock.MagicMock(),
        }):
            from nodes_merge import FoleyTuneLoRAAutoTuner
            tuner = FoleyTuneLoRAAutoTuner()
            block_deltas_per_lora = [
                {"layer_a": torch.randn(32, 32), "layer_b": torch.randn(32, 32)},
                {"layer_a": torch.randn(32, 32), "layer_b": torch.randn(32, 32)},
            ]
            result = tuner._analyze_block(block_deltas_per_lora, [1.0, 1.0])
            self.assertIn(result["strategy"],
                          ["consensus", "orthogonal", "conflicting", "weighted_average"])
            self.assertIn("avg_cos_sim", result)
            self.assertIn("avg_conflict", result)
```

**Step 2: Run the tests to verify they fail**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley && python -m pytest tests/test_nodes_merge.py::TestFoleyTuneLoRAAutoTuner -v`

Expected: 3 failures — `FoleyTuneLoRAAutoTuner` not defined.

**Step 3: Implement `FoleyTuneLoRAAutoTuner`**

Append to `nodes_merge.py`:

```python
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
        "Reports per-block decisions: consensus → weighted_average, "
        "orthogonal → slerp, conflicting → ties."
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
        merger = FoleyTuneLoRAMerger()
        entries = merger._collect_loras(lora_name_1, strength_1, lora_name_2, strength_2,
                                        lora_name_3, strength_3, lora_name_4, strength_4)
        if len(entries) < 2:
            raise ValueError("Need at least 2 LoRAs with non-zero strength to merge.")

        n_loras = len(entries)
        weights = [e["strength"] for e in entries]

        # Auto-strength: accumulate energy across all layers
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

        report_lines = [f"FoleyTune LoRA AutoTuner — {n_loras} LoRAs"]
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
```

**Step 4: Run the tests to verify they pass**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley && python -m pytest tests/test_nodes_merge.py -v`

Expected: all 7 tests pass.

**Step 5: Commit**

```bash
cd /media/p5/ComfyUI-HunyuanVideo-Foley
git add nodes_merge.py tests/test_nodes_merge.py
git commit -m "feat(merge): add FoleyTuneLoRAAutoTuner node with per-block conflict analysis"
```

---

## Task 6: Register nodes in `__init__.py`

**Files:**
- Modify: `__init__.py`

**Step 1: Check current __init__.py structure**

Read `/media/p5/ComfyUI-HunyuanVideo-Foley/__init__.py` and find where `NODE_CLASS_MAPPINGS` is assembled from sub-modules.

**Step 2: Add merge node registration**

Add the import and mappings for `nodes_merge.py`. The exact edit depends on the current `__init__.py` structure — follow the existing pattern for how `nodes_lora.py` nodes are registered.

Add to `NODE_CLASS_MAPPINGS`:
```python
"FoleyTuneLoRAMerger": FoleyTuneLoRAMerger,
"FoleyTuneLoRAAutoTuner": FoleyTuneLoRAAutoTuner,
```

Add to `NODE_DISPLAY_NAME_MAPPINGS`:
```python
"FoleyTuneLoRAMerger": "FoleyTune LoRA Merger",
"FoleyTuneLoRAAutoTuner": "FoleyTune LoRA AutoTuner",
```

**Step 3: Run tests to verify nothing broke**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley && python -m pytest tests/ -v`

**Step 4: Commit**

```bash
cd /media/p5/ComfyUI-HunyuanVideo-Foley
git add __init__.py
git commit -m "feat(merge): register FoleyTuneLoRAMerger and FoleyTuneLoRAAutoTuner nodes"
```

---

## Task 7: Sync to installed custom node

**Files:**
- No code changes — sync the development repo to the installed location.

**Step 1: Copy changed files**

```bash
cp /media/p5/ComfyUI-HunyuanVideo-Foley/nodes_merge.py /media/p5/Comfyui/custom_nodes/ComfyUI-HunyuanVideo-FoleyTune/nodes_merge.py
cp /media/p5/ComfyUI-HunyuanVideo-Foley/lora/merge_math.py /media/p5/Comfyui/custom_nodes/ComfyUI-HunyuanVideo-FoleyTune/lora/merge_math.py
cp /media/p5/ComfyUI-HunyuanVideo-Foley/__init__.py /media/p5/Comfyui/custom_nodes/ComfyUI-HunyuanVideo-FoleyTune/__init__.py
```

**Step 2: Verify ComfyUI loads the nodes**

Restart ComfyUI and check that `FoleyTune LoRA Merger` and `FoleyTune LoRA AutoTuner` appear in the node list.

---

## Out of scope

- Community cache / HF upload — local-only merge.
- Phase 2 candidate search — single-pass heuristic only.
- CLIP handling — Foley has no CLIP component.
- LoKr/LoHa — FoleyTune only trains standard LoRA.
- SingleStreamBlock targeting — not used by any FoleyTune preset.
