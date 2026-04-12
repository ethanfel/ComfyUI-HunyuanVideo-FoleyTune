# Pipeline Streamline Design

**Goal:** Unify the dataset preparation pipeline so data flows in-memory via FOLEYTUNE_AUDIO_DATASET from video quality filtering through feature extraction, audio optimization, inspection, and final save — eliminating intermediate disk I/O. Auto-generate `dataset.json` for the trainer with a train/val split.

**Architecture:** Thread FOLEYTUNE_AUDIO_DATASET through all pipeline nodes. Each node enriches the item dict (video_path, features, rejection metadata) while leaving keys it doesn't own untouched. Only the final DatasetSaver writes to disk. The VideoQualityFilter picks one random rejected clip as a validation candidate that flows through the pipeline alongside accepted clips.

---

## Data Model

Each item in FOLEYTUNE_AUDIO_DATASET grows as it passes through the pipeline:

```python
# After VideoQualityFilter:
{
    "waveform": tensor [1, C, L],   # native sample rate, extracted from video
    "sample_rate": int,
    "name": str,                     # original video filename stem
    "video_path": str,               # source video path (for feature extraction)
    "val": bool,                     # True for the one randomly picked rejected clip
}

# After BatchFeatureExtractor (adds):
{
    ...,
    "features": {
        "clip_feat": tensor [1, T, 768],        # SigLIP2 @ 8fps
        "sync_feat": tensor [1, T, 768],        # Synchformer @ 25fps
        "text_feat": tensor [1, T, 768],        # CLAP text
        "uncond_text_feat": tensor [1, T, 768], # CLAP negative
        "duration": float,
        "fps": float,
    },
    "prompt": str,
}

# Audio optimization nodes (Resampler, LUFS, Compressor, HfSmoother):
#   Only modify "waveform" and "sample_rate" — all other keys pass through unchanged.

# After DatasetInspector (may add):
{
    ...,
    "rejected": bool,
    "reject_reasons": [...],
}
```

Nodes that don't know about extra keys ignore them — they only touch `waveform`/`sample_rate`/`name`.

---

## Node Changes

### FoleyTuneVideoQualityFilter (nodes_dataset.py, lines 732-1016)

- **RETURN_TYPES:** `("STRING",)` → `(FOLEYTUNE_AUDIO_DATASET, "STRING")`
- Already extracts audio from each video via FFmpeg for quality scoring — keep that waveform in memory as `item["waveform"]` instead of discarding it
- Each accepted clip becomes a dataset item with `video_path` and `name` (original filename stem)
- All rejected clips are dropped **except one random pick**, flagged `"val": True`
- That one val clip flows through the rest of the pipeline alongside accepted clips
- `output_folder` stays as optional for backward compatibility (copy files to disk)
- `skip_rejected` controls whether rejected items appear in output (val pick is always included)

### FoleyTuneBatchFeatureExtractor (nodes_lora.py, lines 478-729)

- **Input:** `video_folder` (STRING) → `dataset` (FOLEYTUNE_AUDIO_DATASET)
- **RETURN_TYPES:** `("STRING",)` → `(FOLEYTUNE_AUDIO_DATASET, "STRING")`
- Reads video frames from each item's `video_path` (same FFmpeg frame extraction)
- Adds `features` dict and `prompt` to each item
- No disk writes — features travel in memory
- Keeps threaded prefetch pattern for SigLIP2/Synchformer passes
- Per-clip `.txt` sidecar prompt override still works (checks `{stem}.txt` next to `video_path`)

### FoleyTuneDatasetInspector (nodes_dataset.py, lines 343-429)

- Sets `item["rejected"]` and `item["reject_reasons"]` metadata on every item
- `skip_rejected` still controls whether rejected items appear in output
- Val-flagged items (`item.get("val")`) pass through regardless of rejection
- If an item already has `reject_reasons` from upstream, inspector appends new reasons

### FoleyTuneDatasetSaver (nodes_dataset.py, lines 1209-1283)

- Writes `.npz` directly from `item["features"]` — no need for `npz_source_dir` param (removed)
- Writes `.flac` + `.npz` pairs for training clips in output_dir root
- Val-flagged clip goes in `val/` subfolder
- Auto-generates `dataset.json` with train/val split
- All clips keep their original names from `item["name"]`

### FoleyTuneLoRATrainer (nodes_lora.py, lines 791-1153)

- New optional input: `dataset_json` (STRING) — path to `dataset.json`
- When provided: loads only `"train"` clips for training dataset, uses `"val"` clip for eval metrics
- When not provided: existing `data_dir` behavior unchanged (scan for .npz pairs)
- Val clip is never trained on — only used for inference at checkpoint intervals

---

## Output Structure

```
output_dir/
  ├── glass_hit_01.flac
  ├── glass_hit_01.npz
  ├── glass_shatter_02.flac
  ├── glass_shatter_02.npz
  ├── ...
  ├── val/
  │   ├── glass_crack_05.flac
  │   └── glass_crack_05.npz
  └── dataset.json
```

```json
{
  "prompt": "glass breaking on tile floor",
  "train": ["glass_hit_01", "glass_shatter_02", "glass_drop_03"],
  "val": "val/glass_crack_05"
}
```

If no rejected clips exist, the `"val"` key is omitted from dataset.json.

---

## Unchanged Nodes

- **DatasetResampler, DatasetLUFSNormalizer, DatasetCompressor, DatasetHfSmoother** — already work with FOLEYTUNE_AUDIO_DATASET, only modify waveform/sample_rate
- **DatasetLoader** — still works as entry point to reload saved checkpoints
- **DatasetQualityFilter** — optional step, unchanged
- **FoleyTuneFeatureExtractor** (single-clip) — inference path, unchanged
- **All inference nodes** (ModelLoader, DependenciesLoader, ChunkedSampler, etc.)

---

## Resulting Workflow

```
VideoQualityFilter → BatchFeatureExtractor → Resampler → LUFS → Compressor → HfSmoother → Inspector → Saver
                                                                                                         ↓
                                                                                                   dataset.json → Trainer
```

All in-memory except the final save. DatasetLoader + DatasetSaver remain as opt-in checkpoints to break the chain at any point.
