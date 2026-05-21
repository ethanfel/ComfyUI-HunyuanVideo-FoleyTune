"""Tests for FoleyTune LoRA merge nodes."""

import os
import sys
import unittest
import unittest.mock
import torch

# Ensure the repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock ComfyUI dependencies once at module level to avoid PyO3 reinit issues
# with safetensors when patching sys.modules repeatedly.
_mock_folder_paths = unittest.mock.MagicMock(
    get_filename_list=lambda x: ["lora_a.safetensors", "lora_b.safetensors"],
)
_comfy_mocks = {
    "folder_paths": _mock_folder_paths,
    "comfy": unittest.mock.MagicMock(),
    "comfy.model_management": unittest.mock.MagicMock(),
    "comfy.utils": unittest.mock.MagicMock(),
}

with unittest.mock.patch.dict("sys.modules", _comfy_mocks):
    from nodes_merge import FoleyTuneLoRAMerger, FoleyTuneLoRAAutoTuner


class TestFoleyTuneLoRAMerger(unittest.TestCase):

    def test_input_types_required_fields(self):
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
        inputs = FoleyTuneLoRAMerger.INPUT_TYPES()
        strategies = inputs["required"]["merge_strategy"][0]
        self.assertIn("weighted_average", strategies)
        self.assertIn("ties", strategies)
        self.assertIn("slerp", strategies)

    def test_return_types(self):
        self.assertEqual(FoleyTuneLoRAMerger.RETURN_TYPES, ("FOLEYTUNE_MODEL", "STRING"))

    def test_group_deltas_by_block(self):
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


class TestFoleyTuneLoRAAutoTuner(unittest.TestCase):

    def test_input_types_required_fields(self):
        inputs = FoleyTuneLoRAAutoTuner.INPUT_TYPES()
        req = inputs["required"]
        self.assertIn("hunyuan_model", req)
        self.assertIn("auto_strength", req)
        self.assertIn("sparsification", req)
        self.assertIn("sparsification_density", req)

    def test_return_types_include_report(self):
        self.assertEqual(FoleyTuneLoRAAutoTuner.RETURN_TYPES,
                         ("FOLEYTUNE_MODEL", "STRING", "STRING"))

    def test_analyze_block_returns_strategy(self):
        tuner = FoleyTuneLoRAAutoTuner()
        block_deltas_per_lora = [
            {"layer_a": torch.randn(32, 32), "layer_b": torch.randn(32, 32)},
            {"layer_a": torch.randn(32, 32), "layer_b": torch.randn(32, 32)},
        ]
        result = tuner._analyze_block(block_deltas_per_lora, [1.0, 1.0])
        self.assertIn(result["strategy"],
                      ["slerp", "weighted_average", "conflicting"])
        self.assertIn("avg_cos_sim", result)
        self.assertIn("avg_conflict", result)


if __name__ == "__main__":
    unittest.main()
