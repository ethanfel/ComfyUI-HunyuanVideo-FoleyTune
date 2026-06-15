"""Tests for FoleyTune merged-LoRA save, tuner-data round-trip, and merge apply."""

import os
import sys
import json
import tempfile
import unittest
import unittest.mock

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_mock_folder_paths = unittest.mock.MagicMock(
    get_filename_list=lambda x: [],
    get_full_path_or_raise=lambda folder, name: f"/fake/{folder}/{name}",
)
_comfy_mocks = {
    "folder_paths": _mock_folder_paths,
    "comfy": unittest.mock.MagicMock(),
    "comfy.model_management": unittest.mock.MagicMock(),
    "comfy.utils": unittest.mock.MagicMock(),
}

with unittest.mock.patch.dict("sys.modules", _comfy_mocks):
    import nodes_merge
    from nodes_merge import (
        FoleyTuneSaveMergedLoRA, FoleyTuneSaveTunerData, FoleyTuneLoadTunerData,
        FoleyTuneMergeSelector, FoleyTuneLoRAAutoTuner, _apply_block_merge,
    )
    from lora.merge_math import compute_deltas


class TestSaveMergedLoRA(unittest.TestCase):

    def test_roundtrip_reproduces_delta(self):
        torch.manual_seed(7)
        # Exactly rank-2 delta so SVD reconstructs it well.
        up0 = torch.randn(32, 2)
        down0 = torch.randn(2, 16)
        delta = up0 @ down0
        module_path = "triple_blocks.0.audio_self_proj"
        lora_data = {
            "deltas": {module_path: delta},
            "rank_hint": 8,
            "alpha": 8.0,
            "use_rslora": False,
            "target_suffixes": ["audio_self_proj"],
            "prompts": ["a dog barking"],
            "source_names": ["a.safetensors", "b.safetensors"],
            "is_conv": {module_path: False},
        }
        with tempfile.TemporaryDirectory() as tmp:
            _mock_folder_paths.get_folder_paths = lambda x: [tmp]
            node = FoleyTuneSaveMergedLoRA()
            (out_path,) = node.save_lora(lora_data, tmp, "merged_test", 0, "safetensors")
            self.assertTrue(os.path.exists(out_path))
            self.assertTrue(os.path.exists(out_path.replace(".safetensors", ".json")))

            ckpt = nodes_merge._load_adapter_checkpoint(out_path)
            sd, rank, alpha, use_rslora, target, prompts = nodes_merge._parse_checkpoint(ckpt)
            self.assertEqual(target, ["audio_self_proj"])
            self.assertIn("a dog barking", prompts)
            recon = compute_deltas(sd, rank, alpha, strength=1.0, use_rslora=use_rslora)
            rel_err = (recon[module_path] - delta).norm() / delta.norm()
            self.assertLess(rel_err.item(), 0.05)

    def test_auto_rank_pads_to_uniform_rank(self):
        # Two layers with different intrinsic ranks; auto mode must still emit a
        # single uniform rank so the LoRA Loader's single-rank wrap can load it.
        torch.manual_seed(11)
        low = torch.randn(64, 2) @ torch.randn(2, 32)     # rank-2
        high = torch.randn(64, 12) @ torch.randn(12, 32)  # rank-12
        deltas = {"triple_blocks.0.audio_self_proj": low,
                  "triple_blocks.1.audio_self_proj": high}
        lora_data = {
            "deltas": deltas, "rank_hint": 64, "alpha": 64.0, "use_rslora": False,
            "target_suffixes": ["audio_self_proj"], "prompts": [], "source_names": [],
            "is_conv": {k: False for k in deltas},
        }
        with tempfile.TemporaryDirectory() as tmp:
            _mock_folder_paths.get_folder_paths = lambda x: [tmp]
            (out_path,) = FoleyTuneSaveMergedLoRA().save_lora(lora_data, tmp, "uni", 0, "safetensors")
            from safetensors.torch import load_file
            sd = load_file(out_path)
            a_ranks = {v.shape[0] for k, v in sd.items() if k.endswith(".lora_A")}
            self.assertEqual(len(a_ranks), 1, f"non-uniform lora_A ranks: {a_ranks}")
            # And the merge still reconstructs each layer despite zero-padding.
            ckpt = nodes_merge._load_adapter_checkpoint(out_path)
            s, rank, alpha, ur, tgt, pr = nodes_merge._parse_checkpoint(ckpt)
            recon = compute_deltas(s, rank, alpha, strength=1.0, use_rslora=ur)
            for layer, orig in deltas.items():
                rel = (recon[layer] - orig).norm() / orig.norm()
                self.assertLess(rel.item(), 0.05, f"{layer} rel err {rel.item()}")

    def test_keys_follow_loader_convention(self):
        delta = torch.randn(8, 4)
        lora_data = {
            "deltas": {"triple_blocks.1.audio_self_proj": delta},
            "rank_hint": 4, "alpha": 4.0, "use_rslora": False,
            "target_suffixes": ["audio_self_proj"], "prompts": [],
            "source_names": [], "is_conv": {"triple_blocks.1.audio_self_proj": False},
        }
        with tempfile.TemporaryDirectory() as tmp:
            _mock_folder_paths.get_folder_paths = lambda x: [tmp]
            (out_path,) = FoleyTuneSaveMergedLoRA().save_lora(lora_data, tmp, "k", 0, "safetensors")
            from safetensors.torch import load_file
            sd = load_file(out_path)
            self.assertIn("triple_blocks.1.audio_self_proj.lora_A", sd)
            self.assertIn("triple_blocks.1.audio_self_proj.lora_B", sd)


class TestTunerDataRoundtrip(unittest.TestCase):

    def test_save_then_load(self):
        tuner_data = {
            "algo_version": "foley-merge-1",
            "top_n": [{"rank": 1, "config": {"triple_blocks.0": "ties"},
                       "score_heuristic": 0.8, "approach": "ties",
                       "sparsification": "disabled"}],
            "auto_strength_scale": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            _mock_folder_paths.get_folder_paths = lambda x: [tmp]
            (path,) = FoleyTuneSaveTunerData().save(tuner_data, "td", True,
                                                    prompt="p", description="d")
            self.assertTrue(path.endswith(".tuner"))
            _mock_folder_paths.get_full_path_or_raise = lambda folder, name: path
            (loaded, prompt, desc) = FoleyTuneLoadTunerData().load(os.path.basename(path))
            self.assertEqual(loaded["top_n"][0]["config"], {"triple_blocks.0": "ties"})
            self.assertEqual(prompt, "p")
            self.assertEqual(desc, "d")


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8, bias=False)

    def named_modules_dict(self):
        return dict(self.named_modules())


class TestApplyBlockMerge(unittest.TestCase):

    def test_applies_merged_delta_to_weights(self):
        model = _TinyModel()
        layer = "lin"  # no block prefix -> "_other" block
        before = model.lin.weight.data.clone()
        entries = [
            {"name": "a", "strength": 1.0, "deltas": {layer: torch.ones(8, 8)},
             "rank": 4, "alpha": 4.0, "suffixes": ("lin",)},
            {"name": "b", "strength": 1.0, "deltas": {layer: torch.ones(8, 8) * 3.0},
             "rank": 4, "alpha": 4.0, "suffixes": ("lin",)},
        ]
        opts = dict(nodes_merge._DEFAULT_OPTIONS)
        n_applied, merged_deltas, counts, skipped = _apply_block_merge(
            model, entries, [1.0, 1.0], lambda b: "weighted_average", opts)
        self.assertEqual(n_applied, 1)
        self.assertIn(layer, merged_deltas)
        # weighted_average of all-ones and all-threes (equal weights) -> twos
        torch.testing.assert_close(merged_deltas[layer], torch.full((8, 8), 2.0))
        torch.testing.assert_close(model.lin.weight.data, before + 2.0)


class TestMergeSelectorEdge(unittest.TestCase):

    def test_empty_top_n_passthrough(self):
        model = _TinyModel()
        out_model, prompts, report, lora_data = FoleyTuneMergeSelector().select_merge(
            model, [], {"top_n": []}, 1)
        self.assertIs(out_model, model)
        self.assertIsNone(lora_data)


class _BlockModel(nn.Module):
    def __init__(self):
        super().__init__()
        blk = nn.Module()
        blk.lin = nn.Linear(8, 8, bias=False)
        self.triple_blocks = nn.ModuleList([blk])


class TestAutoTunerReplay(unittest.TestCase):

    def test_replay_skips_analysis_and_applies_saved_config(self):
        model = _BlockModel()
        layer = "triple_blocks.0.lin"
        before = model.triple_blocks[0].lin.weight.data.clone()
        entries = [
            {"name": "a", "strength": 1.0, "deltas": {layer: torch.ones(8, 8)},
             "rank": 4, "alpha": 4.0, "suffixes": ("lin",), "prompts": []},
            {"name": "b", "strength": 1.0, "deltas": {layer: torch.ones(8, 8) * 3.0},
             "rank": 4, "alpha": 4.0, "suffixes": ("lin",), "prompts": []},
        ]
        tuner_data = {
            "top_n": [{"rank": 1, "config": {"triple_blocks.0": "weighted_average"},
                       "approach": "weighted_average", "sparsification": "disabled",
                       "score_heuristic": 0.7, "score_breakdown": {}}],
            "auto_strength_scale": 1.0,
            "source_names": ["a", "b"],
            "sparsification_density": 0.7, "dare_dampening": 0.0,
            "ties_density": 0.7, "ties_sign_method": "frequency",
        }
        with unittest.mock.patch.object(nodes_merge, "_collect_loras_from_stack",
                                        return_value=entries):
            out_model, prompts, report, out_td, lora_data = FoleyTuneLoRAAutoTuner().auto_merge(
                model, [], top_n=3, selection=1, tuner_data=tuner_data)
        # auto_merge deep-copies the input; the merge lands on the returned model.
        out_weight = dict(out_model.named_modules())[layer].weight.data
        # weighted_average of ones and threes -> twos, applied to the weight
        torch.testing.assert_close(out_weight, before + 2.0)
        self.assertIn("REPLAY", report)
        self.assertEqual(out_td["top_n"][0]["config"], {"triple_blocks.0": "weighted_average"})
        self.assertIsNotNone(lora_data)


if __name__ == "__main__":
    unittest.main()
