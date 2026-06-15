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
        # Position 0: both positive and majority is +1 -> a agrees (1.0), b agrees (3.0) -> avg = 2.0
        # Position 1: a is -2.0 (agrees with -1 majority), b is 1.0 (disagrees) -> -2.0/1 = -2.0
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
        # Norm-corrected SLERP: target_norm = 0.5*||a|| + 0.5*||b|| = 1.0
        self.assertAlmostEqual(result.norm().item(), 1.0, places=3)

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
        from lora.merge_math import compute_auto_strength, THRESHOLDS
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

    def test_compute_deltas_conv1d(self):
        from lora.merge_math import compute_deltas
        rank = 2
        A = torch.arange(rank * 3 * 3, dtype=torch.float32).reshape(rank, 3, 3)
        B = torch.ones(4, rank, 1)
        sd = {
            "conv.lora_A": A,
            "conv.lora_B": B,
        }
        deltas = compute_deltas(sd, rank=rank, alpha=2.0, strength=0.5)
        expected = torch.einsum("or,rik->oik", B.squeeze(-1), A) * 0.5
        self.assertIn("conv", deltas)
        self.assertEqual(deltas["conv"].shape, (4, 3, 3))
        torch.testing.assert_close(deltas["conv"], expected)


class TestExcessConflict(unittest.TestCase):

    def test_identical_zero_excess(self):
        from lora.merge_math import sample_conflict
        a = torch.randn(500)
        r = sample_conflict(a, a)
        self.assertAlmostEqual(r["excess_conflict"], 0.0, places=4)

    def test_uncorrelated_low_excess_despite_half_conflict(self):
        from lora.merge_math import sample_conflict
        torch.manual_seed(0)
        a = torch.randn(20000)
        b = torch.randn(20000)
        r = sample_conflict(a, b)
        # Raw conflict ~0.5 (random sign disagreement) but the arccos baseline
        # explains it — so EXCESS is near zero.
        self.assertAlmostEqual(r["conflict_ratio"], 0.5, delta=0.05)
        self.assertLess(r["excess_conflict"], 0.1)

    def test_concentrated_conflict_high_excess(self):
        from lora.merge_math import sample_conflict
        # Large-magnitude dim agrees (drives cosine high); two above-noise dims
        # disagree (drive sign-mismatch high) -> genuine excess conflict.
        a = torch.tensor([10.0, 1.0, 1.0]).repeat(200)
        b = torch.tensor([10.0, -1.0, -1.0]).repeat(200)
        r = sample_conflict(a, b)
        self.assertGreater(r["cos_sim"], 0.9)        # magnitude-weighted alignment high
        self.assertGreater(r["excess_conflict"], 0.3)  # but real conflict surfaces


class TestKarcherSlerp(unittest.TestCase):

    def test_slerp_n_returns_input_shape(self):
        from lora.merge_math import merge_slerp_n
        a = torch.randn(16, 8)
        b = torch.randn(16, 8)
        c = torch.randn(16, 8)
        result = merge_slerp_n([(a, 1.0), (b, 1.0), (c, 1.0)])
        self.assertEqual(result.shape, (16, 8))

    def test_slerp_n_permutation_invariant(self):
        from lora.merge_math import merge_slerp_n
        torch.manual_seed(1)
        a, b, c = torch.randn(128), torch.randn(128), torch.randn(128)
        r1 = merge_slerp_n([(a, 1.0), (b, 1.0), (c, 1.0)])
        r2 = merge_slerp_n([(c, 1.0), (a, 1.0), (b, 1.0)])
        torch.testing.assert_close(r1, r2, atol=1e-4, rtol=1e-4)

    def test_slerp_n_identical_directions_parallel(self):
        from lora.merge_math import merge_slerp_n
        v = torch.randn(64)
        result = merge_slerp_n([(v, 1.0), (2.0 * v, 1.0), (3.0 * v, 1.0)])
        cos = torch.dot(result, v) / (result.norm() * v.norm())
        self.assertAlmostEqual(cos.item(), 1.0, places=4)


class TestAutoStrengthFloor(unittest.TestCase):

    def test_explicit_floor_honored_on_aligned_stack(self):
        from lora.merge_math import compute_auto_strength
        norms = [100.0, 100.0]
        dots = {(0, 1): 80.0}  # aligned (cos=0.8) -> no orthogonal floor
        scale = compute_auto_strength([1.0, 1.0], norms, dots, floor=0.9)
        self.assertAlmostEqual(scale, 0.9, places=4)

    def test_explicit_zero_floor_allows_reduction(self):
        from lora.merge_math import compute_auto_strength
        norms = [100.0, 100.0]
        dots = {(0, 1): 80.0}
        scale = compute_auto_strength([1.0, 1.0], norms, dots, floor=0.0)
        self.assertLess(scale, 1.0)

    def test_orthogonal_default_floor_is_one(self):
        from lora.merge_math import compute_auto_strength, THRESHOLDS
        self.assertEqual(THRESHOLDS["auto_strength_orthogonal_floor"], 1.0)
        scale = compute_auto_strength([1.0, 1.0], [100.0, 100.0], {(0, 1): 0.0})
        self.assertAlmostEqual(scale, 1.0, places=4)


class TestExtractLoraSvd(unittest.TestCase):

    def test_linear_full_rank_roundtrip(self):
        from lora.merge_math import extract_lora_svd
        torch.manual_seed(2)
        up0 = torch.randn(16, 4)
        down0 = torch.randn(4, 8)
        delta = up0 @ down0
        down, up, r = extract_lora_svd(delta, rank=4, rank_mode="fixed")
        recon = up @ down
        torch.testing.assert_close(recon, delta, atol=1e-4, rtol=1e-4)
        self.assertEqual(r, 4)

    def test_near_zero_returns_none(self):
        from lora.merge_math import extract_lora_svd
        self.assertIsNone(extract_lora_svd(torch.zeros(8, 8)))

    def test_conv_reshape_roundtrip(self):
        from lora.merge_math import extract_lora_svd
        torch.manual_seed(3)
        c_out, c_in, k = 6, 3, 3
        delta = torch.randn(c_out, c_in, k)
        mat = delta.reshape(c_out, c_in * k)
        down, up, r = extract_lora_svd(mat, rank=min(c_out, c_in * k), rank_mode="fixed")
        down_c = down.reshape(r, c_in, k)
        up_c = up.reshape(c_out, r, 1)
        recon = torch.einsum("or,rik->oik", up_c.squeeze(-1), down_c)
        torch.testing.assert_close(recon, delta, atol=1e-4, rtol=1e-4)


class TestScoreConfig(unittest.TestCase):

    def test_ties_beats_avg_on_high_conflict(self):
        from lora.merge_math import score_config
        high = [{"avg_cos_sim": 0.3, "avg_conflict": 0.6, "excess_conflict": 0.5}]
        ties, _ = score_config("ties", "disabled", 0.7, high)
        wavg, _ = score_config("weighted_average", "disabled", 0.7, high)
        self.assertGreater(ties, wavg)

    def test_avg_beats_ties_on_low_conflict(self):
        from lora.merge_math import score_config
        low = [{"avg_cos_sim": 0.1, "avg_conflict": 0.05, "excess_conflict": 0.02}]
        ties, _ = score_config("ties", "disabled", 0.7, low)
        wavg, _ = score_config("weighted_average", "disabled", 0.7, low)
        self.assertGreater(wavg, ties)

    def test_score_in_unit_range(self):
        from lora.merge_math import score_config
        metrics = [{"avg_cos_sim": 0.2, "avg_conflict": 0.3, "excess_conflict": 0.2}]
        for approach in ("per_block_adaptive", "ties", "weighted_average", "slerp"):
            for spars in ("disabled", "dare", "conflict_aware"):
                s, _ = score_config(approach, spars, 0.7, metrics)
                self.assertGreaterEqual(s, 0.0)
                self.assertLessEqual(s, 1.0)


if __name__ == "__main__":
    unittest.main()
