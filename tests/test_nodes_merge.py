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
        FoleyTuneLoRAMerger, FoleyTuneLoRAAutoTuner, FoleyTuneLoRAStack,
        FoleyTuneMergeOptions, _collect_loras_from_stack,
    )


class TestFoleyTuneLoRAStack(unittest.TestCase):

    def test_input_types(self):
        inputs = FoleyTuneLoRAStack.INPUT_TYPES()
        self.assertIn("lora_name", inputs["required"])
        self.assertIn("strength", inputs["required"])
        self.assertIn("lora_stack", inputs["optional"])

    def test_return_type(self):
        self.assertEqual(FoleyTuneLoRAStack.RETURN_TYPES, ("LORA_STACK",))

    def test_add_to_stack_chaining(self):
        node = FoleyTuneLoRAStack()
        (stack1,) = node.add_to_stack("lora_a.safetensors", 1.0)
        self.assertEqual(len(stack1), 1)
        self.assertEqual(stack1[0]["name"], "lora_a.safetensors")
        self.assertEqual(stack1[0]["strength"], 1.0)
        self.assertIn("path", stack1[0])
        (stack2,) = node.add_to_stack("lora_b.safetensors", 0.8, lora_stack=stack1)
        self.assertEqual(len(stack2), 2)
        self.assertEqual(stack2[1]["name"], "lora_b.safetensors")
        # original stack is not mutated
        self.assertEqual(len(stack1), 1)


class TestFoleyTuneMergeOptions(unittest.TestCase):

    def test_build_returns_dict(self):
        node = FoleyTuneMergeOptions()
        # Shared tuning only — no merge_strategy (Merger-inline) or top_n (AutoTuner-inline).
        (opts,) = node.build("enabled", -1.0, "dare", 0.7, 0.1, 0.7, "total")
        self.assertEqual(opts["auto_strength"], "enabled")
        self.assertEqual(opts["sparsification"], "dare")
        self.assertEqual(opts["ties_sign_method"], "total")
        self.assertNotIn("merge_strategy", opts)
        self.assertNotIn("top_n", opts)


class TestCollectFromStack(unittest.TestCase):

    def test_skips_zero_strength(self):
        stack = [
            {"name": "a", "path": "/fake/a", "strength": 1.0},
            {"name": "b", "path": "/fake/b", "strength": 0.0},
        ]
        fake_ckpt = {"state_dict": {}, "meta": {"rank": 4, "alpha": 4.0}}
        with unittest.mock.patch.object(nodes_merge, "_load_adapter_checkpoint",
                                        return_value=fake_ckpt), \
             unittest.mock.patch.object(nodes_merge, "compute_deltas",
                                        return_value={"layer": torch.randn(8, 8)}):
            entries = _collect_loras_from_stack(stack)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["name"], "a")


class TestFoleyTuneLoRAMerger(unittest.TestCase):

    def test_input_types_required_fields(self):
        inputs = FoleyTuneLoRAMerger.INPUT_TYPES()
        req = inputs["required"]
        self.assertIn("hunyuan_model", req)
        self.assertIn("lora_stack", req)
        self.assertIn("merge_strategy", req)
        self.assertNotIn("lora_name_1", req)
        self.assertIn("merge_options", inputs.get("optional", {}))

    def test_input_types_strategy_enum(self):
        strategies = FoleyTuneLoRAMerger.INPUT_TYPES()["required"]["merge_strategy"][0]
        self.assertIn("weighted_average", strategies)
        self.assertIn("ties", strategies)
        self.assertIn("slerp", strategies)

    def test_return_types(self):
        self.assertEqual(FoleyTuneLoRAMerger.RETURN_TYPES,
                         ("FOLEYTUNE_MODEL", "STRING", "LORA_DATA"))

    def test_group_deltas_by_block(self):
        merger = FoleyTuneLoRAMerger()
        deltas = {
            "triple_blocks.0.audio_self_attn_qkv": torch.randn(64, 64),
            "triple_blocks.0.audio_self_proj": torch.randn(64, 64),
            "triple_blocks.1.audio_self_attn_qkv": torch.randn(64, 64),
            "single_blocks.5.linear_qkv": torch.randn(32, 32),
        }
        groups = merger._group_by_block(deltas)
        self.assertIn("triple_blocks.0", groups)
        self.assertIn("triple_blocks.1", groups)
        self.assertIn("single_blocks.5", groups)
        self.assertEqual(len(groups["triple_blocks.0"]), 2)
        self.assertEqual(len(groups["triple_blocks.1"]), 1)


class TestFoleyTuneLoRAAutoTuner(unittest.TestCase):

    def test_input_types_required_fields(self):
        inputs = FoleyTuneLoRAAutoTuner.INPUT_TYPES()
        req = inputs["required"]
        self.assertIn("hunyuan_model", req)
        self.assertIn("lora_stack", req)
        self.assertIn("top_n", req)
        # auto_strength/sparsification moved to the shared Merge Options node.
        self.assertNotIn("auto_strength", req)
        self.assertNotIn("sparsification", req)
        self.assertNotIn("lora_name_1", req)
        self.assertIn("merge_options", inputs.get("optional", {}))

    def test_return_types_include_tuner_and_lora_data(self):
        self.assertEqual(
            FoleyTuneLoRAAutoTuner.RETURN_TYPES,
            ("FOLEYTUNE_MODEL", "STRING", "STRING", "TUNER_DATA", "LORA_DATA"))

    def test_analyze_block_returns_strategy(self):
        tuner = FoleyTuneLoRAAutoTuner()
        block_deltas_per_lora = [
            {"layer_a": torch.randn(32, 32), "layer_b": torch.randn(32, 32)},
            {"layer_a": torch.randn(32, 32), "layer_b": torch.randn(32, 32)},
        ]
        result = tuner._analyze_block(block_deltas_per_lora, [1.0, 1.0])
        self.assertIn(result["strategy"], ["slerp", "weighted_average", "ties"])
        self.assertIn("avg_cos_sim", result)
        self.assertIn("avg_conflict", result)
        self.assertIn("excess_conflict", result)


if __name__ == "__main__":
    unittest.main()
