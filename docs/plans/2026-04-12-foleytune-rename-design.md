# FoleyTune Node Rename Design

## Goal

Rename all nodes to use the `FoleyTune` brand to avoid conflicts with the
original HunyuanVideo-Foley repo, preparing for an eventual fork split.

## Decisions

- **Prefix**: `FoleyTune` (evokes LoRA fine-tuning, the fork's identity)
- **Category**: All nodes use `"FoleyTune"` (flat, no subcategories)
- **Drop `HunyuanFoleySampler`**: `FoleyTuneChunkedSampler` subsumes it
  (single-pass for clips shorter than `chunk_duration`)
- **Delete `node_list.json`**: Stale, unused by ComfyUI

## Node Renames

### Core Inference (nodes.py) -- 6 nodes

| Old Class                  | New Class                       | New Display Name                     |
|----------------------------|---------------------------------|--------------------------------------|
| HunyuanModelLoader         | FoleyTuneModelLoader            | FoleyTune Model Loader               |
| HunyuanDependenciesLoader  | FoleyTuneDependenciesLoader     | FoleyTune Dependencies Loader        |
| ~~HunyuanFoleySampler~~    | **deleted**                     | --                                   |
| FoleyChunkedSampler        | FoleyTuneChunkedSampler         | FoleyTune Chunked Sampler            |
| HunyuanFoleyTorchCompile   | FoleyTuneTorchCompile           | FoleyTune Torch Compile              |
| HunyuanBlockSwap           | FoleyTuneBlockSwap              | FoleyTune BlockSwap Settings         |
| SelectAudioFromBatch       | FoleyTuneSelectAudioFromBatch   | FoleyTune Select Audio From Batch    |

### LoRA (nodes_lora.py) -- 7 nodes

| Old Class                  | New Class                       | New Display Name                     |
|----------------------------|---------------------------------|--------------------------------------|
| FoleyFeatureExtractor      | FoleyTuneFeatureExtractor       | FoleyTune Feature Extractor          |
| FoleyBatchFeatureExtractor | FoleyTuneBatchFeatureExtractor  | FoleyTune Batch Feature Extractor    |
| FoleyLoRATrainer           | FoleyTuneLoRATrainer            | FoleyTune LoRA Trainer               |
| FoleyLoRALoader            | FoleyTuneLoRALoader             | FoleyTune LoRA Loader                |
| FoleyLoRAScheduler         | FoleyTuneLoRAScheduler          | FoleyTune LoRA Scheduler             |
| FoleyLoRAEvaluator         | FoleyTuneLoRAEvaluator          | FoleyTune LoRA Evaluator             |
| FoleyVAERoundtrip          | FoleyTuneVAERoundtrip           | FoleyTune VAE Roundtrip              |

### Dataset (nodes_dataset.py) -- 16 nodes

| Old Class                    | New Class                          | New Display Name                       |
|------------------------------|------------------------------------|----------------------------------------|
| FoleyDatasetLoader           | FoleyTuneDatasetLoader             | FoleyTune Dataset Loader               |
| FoleyDatasetResampler        | FoleyTuneDatasetResampler          | FoleyTune Dataset Resampler            |
| FoleyDatasetLUFSNormalizer   | FoleyTuneDatasetLUFSNormalizer     | FoleyTune Dataset LUFS Normalizer      |
| FoleyDatasetCompressor       | FoleyTuneDatasetCompressor         | FoleyTune Dataset Compressor           |
| FoleyDatasetInspector        | FoleyTuneDatasetInspector          | FoleyTune Dataset Inspector            |
| FoleyDatasetQualityFilter    | FoleyTuneDatasetQualityFilter      | FoleyTune Dataset Quality Filter       |
| FoleyVideoQualityFilter      | FoleyTuneVideoQualityFilter        | FoleyTune Video Quality Filter         |
| FoleyDatasetHfSmoother       | FoleyTuneDatasetHfSmoother         | FoleyTune Dataset HF Smoother          |
| FoleyDatasetAugmenter        | FoleyTuneDatasetAugmenter          | FoleyTune Dataset Augmenter            |
| FoleyDatasetSaver            | FoleyTuneDatasetSaver              | FoleyTune Dataset Saver                |
| FoleyDatasetItemExtractor    | FoleyTuneDatasetItemExtractor      | FoleyTune Dataset Item Extractor       |
| FoleyDatasetSpectralMatcher  | FoleyTuneDatasetSpectralMatcher    | FoleyTune Dataset Spectral Matcher     |
| FoleyHfSmoother              | FoleyTuneHfSmoother                | FoleyTune HF Smoother                  |
| FoleyHarmonicExciter         | FoleyTuneHarmonicExciter           | FoleyTune Harmonic Exciter             |
| FoleyOutputNormalizer        | FoleyTuneOutputNormalizer          | FoleyTune Output Normalizer            |
| FoleyDatasetBrowser          | FoleyTuneDatasetBrowser            | FoleyTune Dataset Browser              |

## Custom Type Renames

| Old Type           | New Type                |
|--------------------|-------------------------|
| HUNYUAN_MODEL      | FOLEYTUNE_MODEL         |
| HUNYUAN_DEPS       | FOLEYTUNE_DEPS          |
| FOLEY_FEATURES     | FOLEYTUNE_FEATURES      |
| TORCH_COMPILE_CFG  | FOLEYTUNE_COMPILE_CFG   |
| BLOCKSWAPARGS      | FOLEYTUNE_BLOCKSWAP     |

## Scope

- Rename classes, display names, types, and categories in `nodes.py`,
  `nodes_lora.py`, `nodes_dataset.py`, and `__init__.py`
- Delete `HunyuanFoleySampler` class and its mappings
- Delete `node_list.json`
- No changes to internal logic, model code, or `hunyuanvideo_foley/` subpackage
