# FoleyTune Node Rename Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename all nodes, custom types, and categories from Hunyuan/Foley to FoleyTune branding, drop the redundant HunyuanFoleySampler, and clean up stale files.

**Architecture:** Mechanical find-and-replace across 4 source files + 2 workflow JSON files. No logic changes. The class names, display names, custom ComfyUI types, and CATEGORY strings all get the FoleyTune prefix. Internal cross-references (e.g., `HunyuanFoleyTorchCompile._apply_torch_compile`) must also be updated.

**Tech Stack:** Python (ComfyUI custom nodes), JSON (example workflows)

---

### Task 1: Rename nodes.py classes, types, categories, and delete HunyuanFoleySampler

**Files:**
- Modify: `nodes.py`

**Step 1: Rename class names**

Apply these renames to class definitions and all internal references:

| Old | New |
|-----|-----|
| `HunyuanModelLoader` | `FoleyTuneModelLoader` |
| `HunyuanDependenciesLoader` | `FoleyTuneDependenciesLoader` |
| `FoleyChunkedSampler` | `FoleyTuneChunkedSampler` |
| `HunyuanFoleyTorchCompile` | `FoleyTuneTorchCompile` |
| `HunyuanBlockSwap` | `FoleyTuneBlockSwap` |
| `SelectAudioFromBatch` | `FoleyTuneSelectAudioFromBatch` |

Important: `HunyuanFoleyTorchCompile` is referenced by class name at lines 408 and 565 — update those too.

**Step 2: Rename custom types (all occurrences in the file)**

| Old | New |
|-----|-----|
| `"HUNYUAN_MODEL"` | `"FOLEYTUNE_MODEL"` |
| `"HUNYUAN_DEPS"` | `"FOLEYTUNE_DEPS"` |
| `"FOLEY_FEATURES"` | `"FOLEYTUNE_FEATURES"` |
| `"TORCH_COMPILE_CFG"` | `"FOLEYTUNE_COMPILE_CFG"` |
| `"BLOCKSWAPARGS"` | `"FOLEYTUNE_BLOCKSWAP"` |

**Step 3: Rename CATEGORY strings**

Replace all `"audio/HunyuanFoley"` and `"audio/utils"` with `"FoleyTune"`.

**Step 4: Delete `HunyuanFoleySampler` class**

Remove the entire `HunyuanFoleySampler` class (lines ~258-477) and its entries from NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.

**Step 5: Update NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS**

Replace the mappings dict (lines 857-874) with:

```python
NODE_CLASS_MAPPINGS = {
    "FoleyTuneModelLoader": FoleyTuneModelLoader,
    "FoleyTuneDependenciesLoader": FoleyTuneDependenciesLoader,
    "FoleyTuneChunkedSampler": FoleyTuneChunkedSampler,
    "FoleyTuneTorchCompile": FoleyTuneTorchCompile,
    "FoleyTuneBlockSwap": FoleyTuneBlockSwap,
    "FoleyTuneSelectAudioFromBatch": FoleyTuneSelectAudioFromBatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FoleyTuneModelLoader": "FoleyTune Model Loader",
    "FoleyTuneDependenciesLoader": "FoleyTune Dependencies Loader",
    "FoleyTuneChunkedSampler": "FoleyTune Chunked Sampler",
    "FoleyTuneTorchCompile": "FoleyTune Torch Compile",
    "FoleyTuneBlockSwap": "FoleyTune BlockSwap Settings",
    "FoleyTuneSelectAudioFromBatch": "FoleyTune Select Audio From Batch",
}
```

**Step 6: Commit**

```bash
git add nodes.py
git commit -m "refactor: rename nodes.py classes/types/categories to FoleyTune, drop HunyuanFoleySampler"
```

---

### Task 2: Rename nodes_lora.py classes, types, and categories

**Files:**
- Modify: `nodes_lora.py`

**Step 1: Rename class names**

| Old | New |
|-----|-----|
| `FoleyFeatureExtractor` | `FoleyTuneFeatureExtractor` |
| `FoleyBatchFeatureExtractor` | `FoleyTuneBatchFeatureExtractor` |
| `FoleyLoRATrainer` | `FoleyTuneLoRATrainer` |
| `FoleyLoRALoader` | `FoleyTuneLoRALoader` |
| `FoleyLoRAScheduler` | `FoleyTuneLoRAScheduler` |
| `FoleyLoRAEvaluator` | `FoleyTuneLoRAEvaluator` |
| `FoleyVAERoundtrip` | `FoleyTuneVAERoundtrip` |

**Step 2: Rename custom types (all occurrences)**

| Old | New |
|-----|-----|
| `"HUNYUAN_MODEL"` | `"FOLEYTUNE_MODEL"` |
| `"HUNYUAN_DEPS"` | `"FOLEYTUNE_DEPS"` |
| `"FOLEY_FEATURES"` | `"FOLEYTUNE_FEATURES"` |

**Step 3: Rename CATEGORY strings**

Replace all `"audio/HunyuanFoley/LoRA"` with `"FoleyTune"`.

**Step 4: Update NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS (lines 1897-1915)**

```python
NODE_CLASS_MAPPINGS = {
    "FoleyTuneFeatureExtractor": FoleyTuneFeatureExtractor,
    "FoleyTuneBatchFeatureExtractor": FoleyTuneBatchFeatureExtractor,
    "FoleyTuneLoRATrainer": FoleyTuneLoRATrainer,
    "FoleyTuneLoRALoader": FoleyTuneLoRALoader,
    "FoleyTuneLoRAScheduler": FoleyTuneLoRAScheduler,
    "FoleyTuneLoRAEvaluator": FoleyTuneLoRAEvaluator,
    "FoleyTuneVAERoundtrip": FoleyTuneVAERoundtrip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoleyTuneFeatureExtractor": "FoleyTune Feature Extractor",
    "FoleyTuneBatchFeatureExtractor": "FoleyTune Batch Feature Extractor",
    "FoleyTuneLoRATrainer": "FoleyTune LoRA Trainer",
    "FoleyTuneLoRALoader": "FoleyTune LoRA Loader",
    "FoleyTuneLoRAScheduler": "FoleyTune LoRA Scheduler",
    "FoleyTuneLoRAEvaluator": "FoleyTune LoRA Evaluator",
    "FoleyTuneVAERoundtrip": "FoleyTune VAE Roundtrip",
}
```

**Step 5: Commit**

```bash
git add nodes_lora.py
git commit -m "refactor: rename nodes_lora.py classes/types/categories to FoleyTune"
```

---

### Task 3: Rename nodes_dataset.py classes, types, and categories

**Files:**
- Modify: `nodes_dataset.py`

**Step 1: Rename the type constant and category constants (line 31-33)**

```python
FOLEYTUNE_AUDIO_DATASET = "FOLEYTUNE_AUDIO_DATASET"
FOLEYTUNE_DS_CATEGORY = "FoleyTune"
FOLEYTUNE_AUDIO_CATEGORY = "FoleyTune"
```

Then replace all references to the old constants throughout the file:
- `FOLEY_AUDIO_DATASET` → `FOLEYTUNE_AUDIO_DATASET`
- `FOLEY_DS_CATEGORY` → `FOLEYTUNE_DS_CATEGORY`
- `FOLEY_AUDIO_CATEGORY` → `FOLEYTUNE_AUDIO_CATEGORY`

**Step 2: Rename class names**

| Old | New |
|-----|-----|
| `FoleyDatasetLoader` | `FoleyTuneDatasetLoader` |
| `FoleyDatasetResampler` | `FoleyTuneDatasetResampler` |
| `FoleyDatasetLUFSNormalizer` | `FoleyTuneDatasetLUFSNormalizer` |
| `FoleyDatasetCompressor` | `FoleyTuneDatasetCompressor` |
| `FoleyDatasetInspector` | `FoleyTuneDatasetInspector` |
| `FoleyDatasetQualityFilter` | `FoleyTuneDatasetQualityFilter` |
| `FoleyVideoQualityFilter` | `FoleyTuneVideoQualityFilter` |
| `FoleyDatasetHfSmoother` | `FoleyTuneDatasetHfSmoother` |
| `FoleyDatasetAugmenter` | `FoleyTuneDatasetAugmenter` |
| `FoleyDatasetSaver` | `FoleyTuneDatasetSaver` |
| `FoleyDatasetItemExtractor` | `FoleyTuneDatasetItemExtractor` |
| `FoleyDatasetSpectralMatcher` | `FoleyTuneDatasetSpectralMatcher` |
| `FoleyHfSmoother` | `FoleyTuneHfSmoother` |
| `FoleyHarmonicExciter` | `FoleyTuneHarmonicExciter` |
| `FoleyOutputNormalizer` | `FoleyTuneOutputNormalizer` |
| `FoleyDatasetBrowser` | `FoleyTuneDatasetBrowser` |

**Step 3: Update NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS (lines 1910-1946)**

All keys and values get the FoleyTune prefix (see design doc for full table).

**Step 4: Commit**

```bash
git add nodes_dataset.py
git commit -m "refactor: rename nodes_dataset.py classes/types/categories to FoleyTune"
```

---

### Task 4: Update example workflows

**Files:**
- Modify: `example_workflows/LoRA_Training_Workflow.json`
- Modify: `example_workflows/HunyuanVideoFoleyExample.json`

**Step 1: Replace all old type and class references in both JSON files**

Apply find-and-replace for all type strings:
- `HUNYUAN_MODEL` → `FOLEYTUNE_MODEL`
- `HUNYUAN_DEPS` → `FOLEYTUNE_DEPS`
- `FOLEY_FEATURES` → `FOLEYTUNE_FEATURES`
- `TORCH_COMPILE_CFG` → `FOLEYTUNE_COMPILE_CFG`
- `BLOCKSWAPARGS` → `FOLEYTUNE_BLOCKSWAP`
- `FOLEY_AUDIO_DATASET` → `FOLEYTUNE_AUDIO_DATASET`

And all node class name references (the `"type"` field in node definitions):
- `HunyuanModelLoader` → `FoleyTuneModelLoader`
- `HunyuanDependenciesLoader` → `FoleyTuneDependenciesLoader`
- `HunyuanFoleySampler` → `FoleyTuneChunkedSampler` (redirect to chunked)
- `FoleyChunkedSampler` → `FoleyTuneChunkedSampler`
- `HunyuanFoleyTorchCompile` → `FoleyTuneTorchCompile`
- `HunyuanBlockSwap` → `FoleyTuneBlockSwap`
- `SelectAudioFromBatch` → `FoleyTuneSelectAudioFromBatch`
- All `Foley*` node names → `FoleyTune*` equivalents

**Step 2: Commit**

```bash
git add example_workflows/
git commit -m "refactor: update example workflows with FoleyTune node/type names"
```

---

### Task 5: Delete node_list.json

**Files:**
- Delete: `node_list.json`

**Step 1: Delete the file**

```bash
git rm node_list.json
```

**Step 2: Commit**

```bash
git commit -m "chore: remove stale node_list.json"
```

---

### Task 6: Verify everything loads

**Step 1: Run a basic import check**

```bash
cd /media/p5/ComfyUI-HunyuanVideo-Foley
python -c "from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS; print(f'{len(NODE_CLASS_MAPPINGS)} nodes in nodes.py'); assert 'FoleyTuneModelLoader' in NODE_CLASS_MAPPINGS; assert 'HunyuanFoleySampler' not in NODE_CLASS_MAPPINGS"
python -c "from nodes_lora import NODE_CLASS_MAPPINGS; print(f'{len(NODE_CLASS_MAPPINGS)} nodes in nodes_lora.py'); assert 'FoleyTuneLoRATrainer' in NODE_CLASS_MAPPINGS"
python -c "from nodes_dataset import NODE_CLASS_MAPPINGS; print(f'{len(NODE_CLASS_MAPPINGS)} nodes in nodes_dataset.py'); assert 'FoleyTuneDatasetLoader' in NODE_CLASS_MAPPINGS"
```

Expected: 6 nodes in nodes.py, 7 in nodes_lora.py, 16 in nodes_dataset.py. No old names present.

**Step 2: Grep for any leftover old names**

```bash
grep -rn "HunyuanModelLoader\|HunyuanDependenciesLoader\|HunyuanFoleySampler\|HunyuanFoleyTorchCompile\|HunyuanBlockSwap\|SelectAudioFromBatch" nodes.py nodes_lora.py nodes_dataset.py __init__.py
grep -rn '"HUNYUAN_MODEL"\|"HUNYUAN_DEPS"\|"FOLEY_FEATURES"\|"TORCH_COMPILE_CFG"\|"BLOCKSWAPARGS"\|"FOLEY_AUDIO_DATASET"' nodes.py nodes_lora.py nodes_dataset.py
grep -rn 'audio/HunyuanFoley' nodes.py nodes_lora.py nodes_dataset.py
```

Expected: No matches.
