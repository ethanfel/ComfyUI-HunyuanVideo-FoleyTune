# Pipeline Streamline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thread FOLEYTUNE_AUDIO_DATASET through the entire pipeline (VideoQualityFilter → BatchFeatureExtractor → audio optimization → Inspector → Saver), eliminating intermediate disk I/O, auto-generating `dataset.json`, and adding val clip support.

**Architecture:** Modify 5 existing nodes to pass data in-memory via FOLEYTUNE_AUDIO_DATASET. VideoQualityFilter outputs dataset items with audio + video_path. BatchFeatureExtractor enriches items with features. DatasetSaver writes everything at the end + generates dataset.json. Trainer accepts dataset_json for train/val split.

**Tech Stack:** Python (ComfyUI custom nodes), PyTorch, FFmpeg, numpy, soundfile

**Design doc:** `docs/plans/2026-04-12-pipeline-streamline-design.md`

---

### Task 1: VideoQualityFilter — output FOLEYTUNE_AUDIO_DATASET

**Files:**
- Modify: `nodes_dataset.py`

**What to change:**

The `FoleyTuneVideoQualityFilter` class (lines 732-1016) currently returns only a STRING report. It needs to also output a `FOLEYTUNE_AUDIO_DATASET` containing the accepted clips + one random rejected clip flagged as val.

**Step 1: Update RETURN_TYPES and RETURN_NAMES**

At lines 805-806, change:

```python
# OLD
RETURN_TYPES = ("STRING",)
RETURN_NAMES = ("report",)

# NEW
RETURN_TYPES = (FOLEYTUNE_AUDIO_DATASET, "STRING")
RETURN_NAMES = ("dataset", "report")
```

**Step 2: Add seed parameter to INPUT_TYPES**

In the `"optional"` dict (around line 761), add:

```python
"seed": ("INT", {
    "default": 42,
    "tooltip": "Seed for random val clip selection from rejected clips.",
}),
```

Also add `seed: int = 42` to the `filter_videos` method signature.

**Step 3: Modify `_process_clip` to keep the waveform**

In `_process_clip()` (line 886), the waveform is currently deleted on line 907 (`del wav`). Change it to keep the waveform and sample rate in the return dict:

```python
# In _process_clip, REMOVE: del wav
# CHANGE the return dict to include wav and sr:
return {
    "path": f, "rel": rel, "duration": duration,
    "bw": bw, "sq": sq, "mono_48k": mono_48k,
    "wav": wav, "sr": sr,
}
```

**Step 4: Build dataset items after quality decisions**

After the Phase 2 loop (after line 1002), build the dataset. Replace the current pass/reject output logic to collect items:

In the Phase 2 loop, instead of just copying files, build lists of accepted and rejected items:

```python
# Before the Phase 2 loop, add:
accepted_items = []
rejected_items = []

# Inside the loop, when clip_passed:
accepted_items.append({
    "waveform": r["wav"],
    "sample_rate": r["sr"],
    "name": r["rel"].stem,
    "video_path": str(r["path"]),
})

# When rejected:
rejected_items.append({
    "waveform": r["wav"],
    "sample_rate": r["sr"],
    "name": r["rel"].stem,
    "video_path": str(r["path"]),
})
```

Keep the existing `do_copy` / `shutil.copy2` logic intact for backward compatibility — it still copies files if `output_folder` is set.

**Step 5: Pick one random rejected clip as val**

After the Phase 2 loop, pick val and build final dataset:

```python
import random as _rng
_rng.seed(seed)

dataset = list(accepted_items)
if rejected_items:
    val_pick = _rng.choice(rejected_items)
    val_pick["val"] = True
    dataset.append(val_pick)
```

**Step 6: Update the return statement**

Change line 1016 from:

```python
return (report,)
```

to:

```python
return (dataset, report)
```

**Step 7: Update DESCRIPTION**

Update the DESCRIPTION string (lines 809-812) to mention the dataset output.

**Step 8: Commit**

```bash
git add nodes_dataset.py
git commit -m "feat: VideoQualityFilter outputs FOLEYTUNE_AUDIO_DATASET with val clip"
```

---

### Task 2: BatchFeatureExtractor — accept FOLEYTUNE_AUDIO_DATASET

**Files:**
- Modify: `nodes_lora.py`

**What to change:**

The `FoleyTuneBatchFeatureExtractor` class (lines 478-729) currently takes `video_folder` + `output_folder` STRING inputs and writes .npz + .wav to disk. It needs to accept a FOLEYTUNE_AUDIO_DATASET, add features to each item, and return the enriched dataset.

**Step 1: Update INPUT_TYPES**

Replace the current required inputs (lines 489-505):

```python
# OLD
"required": {
    "hunyuan_deps": ("FOLEYTUNE_DEPS",),
    "video_folder": ("STRING", {...}),
    "output_folder": ("STRING", {...}),
    "prompt": ("STRING", {...}),
    "negative_prompt": ("STRING", {...}),
},

# NEW
"required": {
    "hunyuan_deps": ("FOLEYTUNE_DEPS",),
    "dataset": (FOLEYTUNE_AUDIO_DATASET,),
    "prompt": ("STRING", {
        "default": "", "multiline": True,
        "tooltip": "Global text prompt. Overridden by per-clip .txt sidecar files.",
    }),
},
```

Note: `FOLEYTUNE_AUDIO_DATASET` is defined in `nodes_dataset.py`. Add an import or define the string constant at the top of `nodes_lora.py`:

```python
FOLEYTUNE_AUDIO_DATASET = "FOLEYTUNE_AUDIO_DATASET"
```

**Step 2: Update RETURN_TYPES and RETURN_NAMES**

```python
# OLD
RETURN_TYPES = ("STRING",)
RETURN_NAMES = ("report",)

# NEW
RETURN_TYPES = (FOLEYTUNE_AUDIO_DATASET, "STRING")
RETURN_NAMES = ("dataset", "report")
```

**Step 3: Rewrite `extract_batch` method signature**

```python
# OLD
def extract_batch(self, hunyuan_deps, video_folder, output_folder,
                  prompt, negative_prompt):

# NEW
def extract_batch(self, hunyuan_deps, dataset, prompt):
```

**Step 4: Rewrite Phase 1 (metadata probing)**

Replace the folder scanning and FFprobe section (lines 523-576). Instead of scanning `video_folder`, iterate the dataset items and probe each item's `video_path`:

```python
folder = None  # no longer scanning a folder

clips = []
lines = ["=== Batch Feature Extraction ===", ""]
seen_names = set()

for item in dataset:
    video_path = Path(item["video_path"])
    try:
        fps, dur = _ffprobe_metadata(video_path)
    except Exception as e:
        lines.append(f"  SKIP  {item['name']}: FFprobe error — {e}")
        continue

    # Per-clip prompt from sidecar .txt, else global
    txt_path = video_path.with_suffix(".txt")
    clip_prompt = txt_path.read_text().strip() if txt_path.exists() else prompt

    clips.append({
        "item": item,          # reference to original dataset item
        "path": video_path,
        "fps": fps,
        "duration": dur,
        "prompt": clip_prompt,
        "name": item["name"],
    })
```

**Step 5: Keep SigLIP2 and Synchformer passes mostly unchanged**

The prefetch functions (`_prefetch_siglip2`, `_prefetch_sync`) read frames from `clips[idx]["path"]` — this already works since we're storing `video_path` as `"path"` in the clips list.

Only change: update `clips[i]['rel']` references in logging to use `clips[i]['name']` since there's no longer a `rel` field.

**Step 6: Replace Pass 3 (save .npz + extract audio) with feature attachment**

Replace the `_save_clip` function and its ThreadPoolExecutor (lines 689-721) with:

```python
# Pass 3: Attach features to dataset items
logger.info("[BatchFeatureExtractor] Attaching features to dataset items...")
output_dataset = []
for i in range(n):
    clip = clips[i]
    text_feat = prompt_cache[clip["prompt"]]
    item = dict(clip["item"])  # shallow copy to avoid mutating input
    item["features"] = {
        "clip_features": clip_feats[i],
        "sync_features": sync_feats[i],
        "text_embedding": text_feat,
        "duration": clip["duration"],
        "fps": clip["fps"],
    }
    item["prompt"] = clip["prompt"]
    output_dataset.append(item)
    lines.append(
        f"  OK    {clip['name']} ({clip['duration']:.1f}s @ "
        f"{clip['fps']:.1f}fps)  clip_feat={clip_feats[i].shape}  "
        f"sync_feat={sync_feats[i].shape}"
    )
del clip_feats, sync_feats

lines.append("")
lines.append(f"Processed {n} clips")

report = "\n".join(lines)
logger.info(f"[BatchFeatureExtractor]\n{report}")
return (output_dataset, report)
```

**Step 7: Update class docstring**

Update the docstring (lines 478-484) to describe the new behavior.

**Step 8: Commit**

```bash
git add nodes_lora.py
git commit -m "feat: BatchFeatureExtractor accepts FOLEYTUNE_AUDIO_DATASET, no disk writes"
```

---

### Task 3: DatasetInspector — add rejection metadata and val passthrough

**Files:**
- Modify: `nodes_dataset.py`

**What to change:**

The `FoleyTuneDatasetInspector` class (lines 343-429) currently either keeps or drops items from the list. It needs to set `rejected`/`reject_reasons` metadata on each item and always pass through val-flagged items.

**Step 1: Modify the inspect method's item handling**

Replace the current marking logic (lines 413-419):

```python
# OLD
if issues:
    flagged.append(name)
    lines.append(f"  FLAGGED  {name} ({duration:.2f}s): {', '.join(issues)}")
    if not skip_rejected:
        clean.append(item)
else:
    clean.append(item)
    lines.append(f"  OK       {name} ({duration:.2f}s)")

# NEW
if issues:
    flagged.append(name)
    lines.append(f"  FLAGGED  {name} ({duration:.2f}s): {', '.join(issues)}")
    # Merge with any upstream rejection reasons
    existing_reasons = item.get("reject_reasons", [])
    item["rejected"] = True
    item["reject_reasons"] = existing_reasons + issues
    # Always keep val-flagged items; otherwise respect skip_rejected
    if item.get("val") or not skip_rejected:
        clean.append(item)
else:
    item.setdefault("rejected", False)
    item.setdefault("reject_reasons", [])
    clean.append(item)
    lines.append(f"  OK       {name} ({duration:.2f}s)")
```

**Step 2: Commit**

```bash
git add nodes_dataset.py
git commit -m "feat: DatasetInspector sets rejection metadata, passes through val items"
```

---

### Task 4: DatasetSaver — write features + dataset.json + val split

**Files:**
- Modify: `nodes_dataset.py`

**What to change:**

The `FoleyTuneDatasetSaver` class (lines 1209-1283) currently writes FLAC files and optionally copies .npz from a source dir. It needs to write .npz from item features, handle val clips, and generate `dataset.json`.

**Step 1: Update INPUT_TYPES**

Replace the current inputs (lines 1213-1228):

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "dataset": (FOLEYTUNE_AUDIO_DATASET,),
            "output_dir": ("STRING", {
                "default": "",
                "tooltip": "Absolute path to output folder. Created if it does not exist.",
            }),
        },
        "optional": {
            "prompt": ("STRING", {
                "default": "",
                "tooltip": "Global prompt written to dataset.json. Per-clip prompts are stored in .npz files.",
            }),
        },
    }
```

Note: `npz_source_dir` is removed.

**Step 2: Rewrite the `save` method**

Replace the entire `save` method (lines 1240-1283):

```python
def save(self, dataset, output_dir: str, prompt: str = ""):
    import json
    import soundfile as sf

    out_path = Path(output_dir.strip())
    out_path.mkdir(parents=True, exist_ok=True)

    train_names = []
    val_name = None
    saved = 0
    features_saved = 0

    for item in dataset:
        name = item["name"]
        wav = item["waveform"][0]  # [C, L]
        sr = item["sample_rate"]
        is_val = item.get("val", False)

        # Determine output directory
        if is_val:
            item_dir = out_path / "val"
            item_dir.mkdir(exist_ok=True)
        else:
            item_dir = out_path

        # Write FLAC
        wav_np = wav.permute(1, 0).float().numpy()  # [L, C]
        if wav_np.shape[1] == 1:
            wav_np = wav_np[:, 0]  # [L] mono
        flac_path = item_dir / f"{name}.flac"
        sf.write(str(flac_path), wav_np, sr, subtype="PCM_24")
        saved += 1

        # Write .npz from features if present
        if "features" in item:
            feats = item["features"]
            npz_path = item_dir / f"{name}.npz"
            save_kwargs = {}
            for key in ("clip_features", "sync_features", "text_embedding"):
                val = feats.get(key)
                if val is not None:
                    save_kwargs[key] = val.float().numpy() if hasattr(val, 'numpy') else val
            if "duration" in feats:
                save_kwargs["duration"] = feats["duration"]
            if "fps" in feats:
                save_kwargs["fps"] = feats["fps"]
            if item.get("prompt"):
                save_kwargs["prompt"] = item["prompt"]
            np.savez(str(npz_path), **save_kwargs)
            features_saved += 1

        # Track names for dataset.json
        if is_val:
            val_name = f"val/{name}"
        else:
            train_names.append(name)

    # Write dataset.json
    ds_json = {"train": train_names}
    if prompt.strip():
        ds_json["prompt"] = prompt.strip()
    if val_name:
        ds_json["val"] = val_name
    json_path = out_path / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(ds_json, f, indent=2)

    lines = [f"[FoleyTuneDatasetSaver] Saved {saved} clips -> {out_path}"]
    lines.append(f"  FLAC: {saved}  NPZ: {features_saved}")
    lines.append(f"  Train: {len(train_names)}  Val: {1 if val_name else 0}")
    lines.append(f"  dataset.json -> {json_path}")

    report = "\n".join(lines)
    print(report, flush=True)
    return (report,)
```

**Step 3: Update DESCRIPTION**

```python
DESCRIPTION = (
    "Save every clip in a FOLEYTUNE_AUDIO_DATASET to output_dir as 24-bit FLAC. "
    "Writes .npz feature files from item data and generates dataset.json with train/val split."
)
```

**Step 4: Commit**

```bash
git add nodes_dataset.py
git commit -m "feat: DatasetSaver writes features + dataset.json with val split"
```

---

### Task 5: LoRATrainer — accept dataset_json

**Files:**
- Modify: `nodes_lora.py`
- Modify: `lora/train.py`

**What to change:**

Add optional `dataset_json` input to the trainer. When provided, `prepare_dataset` loads only the listed train clips, and the val clip is loaded via `prepare_single_entry` for checkpoint evaluation.

**Step 1: Add `dataset_json` to INPUT_TYPES**

In `FoleyTuneLoRATrainer.INPUT_TYPES` (around line 827), add to the `"optional"` dict:

```python
"dataset_json": ("STRING", {
    "default": "",
    "tooltip": "Path to dataset.json. When set, uses its train/val split instead of scanning data_dir for all .npz files.",
}),
```

**Step 2: Update `train` method signature**

Add `dataset_json=""` to both `train` (line 841) and `_train_inner` (line 868) signatures.

In `train()`, pass `dataset_json` through to `_train_inner`.

**Step 3: Modify `prepare_dataset` in `lora/train.py`**

Add an optional `clip_names` parameter to `prepare_dataset` (line 26):

```python
def prepare_dataset(data_dir: str, dac_model, device, dtype=torch.bfloat16, clip_names=None):
```

When `clip_names` is provided, instead of globbing all `*.npz`, only load the listed clips:

```python
# After: data_dir = Path(data_dir)
if clip_names is not None:
    npz_files = [data_dir / f"{name}.npz" for name in clip_names]
    npz_files = [f for f in npz_files if f.exists()]
else:
    npz_files = sorted(data_dir.glob("*.npz"))
```

**Step 4: In `_train_inner`, parse dataset_json and load val clip**

In `_train_inner`, before the `prepare_dataset` call (around line 888), add JSON parsing:

```python
# Parse dataset_json if provided
clip_names = None
val_entry = None
if dataset_json and os.path.exists(dataset_json):
    import json
    with open(dataset_json) as f:
        ds_cfg = json.load(f)
    data_dir = str(Path(dataset_json).parent)  # resolve paths relative to JSON
    clip_names = ds_cfg.get("train")

dataset = prepare_dataset(data_dir, hunyuan_deps.dac_model, device, dtype,
                          clip_names=clip_names)
n_clips = len(dataset)
logger.info(f"Dataset ready: {n_clips} clips")

# Load val clip if specified
if dataset_json and os.path.exists(dataset_json):
    val_key = ds_cfg.get("val")
    if val_key:
        val_npz = Path(data_dir) / f"{val_key}.npz"
        if val_npz.exists():
            val_entry = prepare_single_entry(str(val_npz), hunyuan_deps.dac_model, device, dtype)
            logger.info(f"Val clip loaded: {val_key}")
```

**Step 5: Add val clip eval at checkpoint intervals**

In the checkpoint eval section (around lines 1094-1120), after the existing eval sample generation from `dataset[0]`, add val clip evaluation:

```python
# After existing eval sample code (line ~1120), add:
if val_entry is not None:
    val_wav, val_sr = generate_eval_sample(
        model, hunyuan_deps.dac_model, val_entry, device, dtype,
    )
    val_wav_mono = val_wav.squeeze()
    val_wav_t = torch.from_numpy(val_wav)
    if val_wav_t.ndim == 1:
        val_wav_t = val_wav_t.unsqueeze(0)
    _save_wav(samples_path / f"val_step_{step+1:05d}.wav", val_wav_t, val_sr)
    _save_spectrogram(val_wav_mono, val_sr, samples_path / f"val_step_{step+1:05d}")

    # Val reference metrics
    val_ref_path = None
    for ext in (".flac", ".wav", ".ogg"):
        candidate = Path(data_dir) / f"{val_entry['name']}{ext}"
        if candidate.exists():
            val_ref_path = candidate
            break
    if val_ref_path:
        import soundfile as _sf
        _vraw, _vsr = _sf.read(str(val_ref_path))
        if _vraw.ndim > 1:
            _vraw = _vraw.mean(axis=1)
        if _vsr != 48000:
            import soxr as _soxr
            _vraw = _soxr.resample(_vraw[:, None], _vsr, 48000, quality="VHQ").squeeze(-1)
        vm = reference_metrics(val_wav_mono, _vraw, val_sr)
        step_metrics.update({f"val_{k}": v for k, v in vm.items()})
```

**Step 6: Commit**

```bash
git add nodes_lora.py lora/train.py
git commit -m "feat: LoRATrainer accepts dataset_json with train/val split"
```

---

### Task 6: Update module docstrings

**Files:**
- Modify: `nodes_dataset.py` (lines 1-23, module docstring)
- Modify: `nodes_lora.py` (BatchFeatureExtractor docstring)

**Step 1: Update nodes_dataset.py module docstring**

Update the pipeline diagram at the top of the file (lines 1-23) to reflect the new flow:

```python
"""Foley Audio Dataset Pipeline — chainable in-memory preprocessing nodes.

Typical chain:
  FoleyTuneVideoQualityFilter
      ↓ FOLEYTUNE_AUDIO_DATASET  (+ val clip from rejected)
  FoleyTuneBatchFeatureExtractor
      ↓ FOLEYTUNE_AUDIO_DATASET  (+ features attached)
  FoleyTuneDatasetResampler       (optional)
      ↓ FOLEYTUNE_AUDIO_DATASET
  FoleyTuneDatasetLUFSNormalizer  (optional)
      ↓ FOLEYTUNE_AUDIO_DATASET
  FoleyTuneDatasetCompressor      (optional)
      ↓ FOLEYTUNE_AUDIO_DATASET
  FoleyTuneDatasetHfSmoother      (optional)
      ↓ FOLEYTUNE_AUDIO_DATASET
  FoleyTuneDatasetAugmenter       (optional)
      ↓ FOLEYTUNE_AUDIO_DATASET
  FoleyTuneDatasetInspector       (optional)
      ↓ FOLEYTUNE_AUDIO_DATASET  +  STRING report
  FoleyTuneDatasetSaver
      ↓ STRING report  +  dataset.json  +  val/ subfolder

Alternative entry points:
  FoleyTuneDatasetLoader          → FOLEYTUNE_AUDIO_DATASET (from saved FLAC files)
  FoleyTuneDatasetItemExtractor   → AUDIO (bridges to standard nodes)
"""
```

**Step 2: Commit**

```bash
git add nodes_dataset.py nodes_lora.py
git commit -m "docs: update module docstrings for streamlined pipeline"
```

---

### Task 7: Update example workflow

**Files:**
- Modify: `example_workflows/LoRA_Training_Workflow.json`

**Step 1: Update the workflow JSON**

Update the example workflow to reflect the new pipeline:
- Replace the `FoleyTuneDatasetLoader` entry point with `FoleyTuneVideoQualityFilter` feeding into `FoleyTuneBatchFeatureExtractor`
- Wire `FoleyTuneBatchFeatureExtractor` to accept `FOLEYTUNE_AUDIO_DATASET` input instead of `video_folder`/`output_folder` strings
- Remove the `FoleyTuneDatasetSaver`'s `npz_source_dir` connection
- Update type references where needed

**Step 2: Commit**

```bash
git add example_workflows/
git commit -m "docs: update example workflow for streamlined pipeline"
```

---

### Task 8: Verification

**Step 1: AST syntax check**

```bash
python -c "
import ast, sys
for f in ['nodes.py', 'nodes_lora.py', 'nodes_dataset.py', 'lora/train.py']:
    try:
        ast.parse(open(f).read())
        print(f'  OK  {f}')
    except SyntaxError as e:
        print(f'  FAIL  {f}: {e}')
        sys.exit(1)
print('All files parse OK')
"
```

**Step 2: Grep for consistency**

Check that FOLEYTUNE_AUDIO_DATASET is used consistently:

```bash
grep -n "FOLEYTUNE_AUDIO_DATASET" nodes_dataset.py nodes_lora.py
```

Expected: VideoQualityFilter and BatchFeatureExtractor both reference the type.

Check no stale references to removed params:

```bash
grep -n "npz_source_dir" nodes_dataset.py
grep -n "negative_prompt" nodes_lora.py
grep -n '"video_folder"' nodes_lora.py
grep -n '"output_folder"' nodes_lora.py
```

Expected: All return zero matches.

**Step 3: Verify dataset.json integration**

```bash
grep -n "dataset_json" nodes_lora.py lora/train.py
```

Expected: Matches in trainer INPUT_TYPES, train method, _train_inner, and prepare_dataset.

**Step 4: Check for val handling consistency**

```bash
grep -n '"val"' nodes_dataset.py nodes_lora.py
```

Expected: Matches in VideoQualityFilter (setting val flag), Inspector (checking val flag), DatasetSaver (writing val subdir), and Trainer (loading val clip).
