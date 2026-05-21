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
