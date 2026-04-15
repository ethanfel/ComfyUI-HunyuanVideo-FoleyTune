# Voice Tagger Design

**Date:** 2026-04-15
**Status:** Approved

## Goal

Differentiate performers' voices in training datasets by auto-analyzing vocal
characteristics and prepending descriptors to CLAP prompts. Prevents the model
from merging distinct voices into an averaged sound during LoRA training.

## Problem

When training on clips with multiple performers, the same generic prompt
(e.g., "wet sucking sounds with rhythmic moaning") is used for all clips.
The model learns an averaged voice, losing individual vocal characteristics
like pitch, breathiness, and register.

## Architecture

Two nodes:

1. **FoleyTuneVoiceTagger** — in-pipeline node that analyzes audio clips,
   clusters by speaker, generates voice descriptors, and modifies prompts.
   Runs before feature extraction.

2. **FoleyTuneRetagNPZ** — post-hoc node that updates prompt + CLAP text
   embedding in existing NPZ files without re-extracting visual features.
   For fast iteration on voice descriptors.

## Node 1: FoleyTuneVoiceTagger

**Category:** `audio/FoleyTune/Dataset`
**Function:** `tag_voices`
**Return types:** `FOLEYTUNE_AUDIO_DATASET`, `STRING`

### Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| dataset | FOLEYTUNE_AUDIO_DATASET | required | Audio clips from loader |
| num_speakers | INT | 2 | Expected number of distinct voices |
| min_f0_female | FLOAT | 165.0 | F0 threshold (Hz) for male/female split |
| samples_per_source | INT | 3 | Clips to analyze per source video |
| descriptor_mode | COMBO | "auto" | `auto`, `label_only`, or `custom` |
| custom_descriptors | STRING | "" | JSON map of cluster_id → descriptor (for custom mode) |
| tag_position | COMBO | "prepend" | `prepend` or `append` |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| dataset | FOLEYTUNE_AUDIO_DATASET | Dataset with modified `prompt` fields |
| report | STRING | Full tag report: per-source descriptors and tagged prompts |

### Algorithm

1. **Group by source prefix.** Parse item `name` field — strip trailing `_NN`
   suffix to get source ID (e.g., `clip_001_03` → `clip_001`).

2. **Sample evenly per source.** For each source group, pick `samples_per_source`
   items spread evenly across the present (filtered) segments using
   `np.linspace(0, len(group)-1, samples_per_source)`. This ensures coverage
   of different moments regardless of which segments survived quality filtering.

3. **Extract acoustic features.** For each sampled clip, using Parselmouth:
   - Median F0 (fundamental frequency) — pitch register
   - HNR (harmonics-to-noise ratio) — breathiness vs clarity

4. **Cluster sources.**
   - If `num_speakers <= 2`: simple F0 threshold split at `min_f0_female`.
   - If `num_speakers > 2`: use Resemblyzer d-vector embeddings on sampled
     clips, then KMeans(k=num_speakers) on the per-source averaged embeddings.

5. **Generate descriptors** (per `descriptor_mode`):
   - `auto`: Map acoustic features to CLAP-compatible descriptors:
     - F0 > 250 Hz → "high-pitched", 165-250 Hz → "mid-pitched", < 165 Hz → "deep"
     - HNR < 10 dB → "breathy", HNR >= 10 dB → "clear"
     - Gender label from F0 threshold
     - Example output: `"breathy high-pitched female"`
   - `label_only`: Just `"female voice"` / `"male voice"`
   - `custom`: Use user-provided JSON mapping

6. **Tag prompts.** For each item in the dataset, look up its source's
   descriptor and prepend/append to the existing prompt:
   `"breathy high-pitched female, wet sucking sounds with rhythmic moaning"`

7. **Build report string.** List each source with: segment count, sampled
   segments, acoustic features, assigned cluster, generated descriptor,
   and example tagged prompt. Output as STRING for user inspection.

### Dependencies

- **Parselmouth** (pip: `praat-parselmouth`) — F0 and HNR extraction
- **Resemblyzer** (pip: `resemblyzer`) — only needed when `num_speakers > 2`

## Node 2: FoleyTuneRetagNPZ

**Category:** `audio/FoleyTune/Dataset`
**Function:** `retag_npz`
**Return types:** `STRING`

### Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| hunyuan_deps | FOLEYTUNE_DEPS | required | Provides CLAP model |
| npz_dir | STRING | required | Path to folder with existing .npz files |
| tag_map | STRING | required | JSON: source prefix → descriptor string |
| tag_position | COMBO | "prepend" | `prepend` or `append` |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| report | STRING | List of updated files with old → new prompts |

### Algorithm

1. Parse `tag_map` JSON into dict.
2. Glob all `.npz` files in `npz_dir`.
3. For each NPZ:
   a. Load, extract current `prompt`.
   b. Match source prefix from filename to `tag_map`.
   c. Build new prompt (prepend/append descriptor).
   d. Re-encode with CLAP → new `text_embedding`.
   e. Save NPZ with updated `prompt` and `text_embedding`, all other fields unchanged.
4. Return report with old → new prompt mappings.

## CLAP Descriptor Guidelines

Stick to AudioCaps-style vocabulary that CLAP embeds meaningfully:

**Pitch:** "high-pitched", "deep", "low-pitched"
**Breathiness:** "breathy", "airy", "clear"
**Intensity:** "loud", "soft", "gentle"
**Gender:** "female", "male"
**Register:** "falsetto", "chest voice"

Avoid exotic adjectives outside AudioCaps training distribution.

## Pipeline Integration

### Pre-extraction flow (new dataset)
```
DatasetLoader → VoiceTagger → BatchFeatureExtractor → DatasetSaver
```
VoiceTagger modifies prompts before CLAP embedding computation.

### Post-extraction flow (existing NPZ files)
```
VoiceTagger (on audio) → copy report's tag_map → RetagNPZ (on NPZ dir)
```
Only re-computes CLAP text embeddings. Visual features untouched.

## Example

Source: 43 videos, 836 clips, 2 performers (1 female dominant, 1 male sparse)

```
clip_001 (17 segs) → sample [_0, _8, _16] → F0=280Hz HNR=8dB → "breathy high-pitched female"
clip_002 (12 segs) → sample [_0, _6, _11] → F0=140Hz HNR=14dB → "deep male"
clip_003 (20 segs) → sample [_0, _10, _19] → F0=310Hz HNR=12dB → "clear high-pitched female"

clip_001_00: "wet sucking sounds" → "breathy high-pitched female, wet sucking sounds"
clip_001_03: "wet sucking sounds" → "breathy high-pitched female, wet sucking sounds"
clip_002_00: "deep grunting"      → "deep male, deep grunting"
```
