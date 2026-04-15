# Voice Tagger Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two ComfyUI nodes (VoiceTagger + RetagNPZ) that auto-analyze vocal characteristics, cluster by speaker, and tag prompts with voice descriptors for LoRA training.

**Architecture:** VoiceTagger analyzes audio waveforms from FOLEYTUNE_AUDIO_DATASET, groups clips by source video prefix, samples a few per source for F0/HNR analysis via Parselmouth, clusters into speaker groups, and prepends descriptors to prompts. RetagNPZ updates existing NPZ files with new CLAP embeddings. A one-line fix to BatchFeatureExtractor lets it pick up per-item prompts.

**Tech Stack:** Parselmouth (praat-parselmouth), Resemblyzer (optional, for >2 speakers), numpy, ComfyUI node API.

---

### Task 1: Add Parselmouth dependency

**Files:**
- Modify: `requirements.txt`

**Step 1: Add parselmouth to requirements**

Add to `requirements.txt`:

```
praat-parselmouth
```

**Step 2: Install and verify**

Run: `pip install praat-parselmouth`

Verify:
```bash
python3 -c "import parselmouth; print(parselmouth.__version__)"
```
Expected: version string, no error.

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add praat-parselmouth dependency for voice analysis"
```

---

### Task 2: Voice analysis helpers

**Files:**
- Create: `dataset/voice_analysis.py`

**Step 1: Write the voice analysis module**

```python
"""Voice analysis utilities for speaker clustering and descriptor generation."""

import re
import numpy as np


def extract_voice_features(waveform: np.ndarray, sr: int) -> dict:
    """Extract F0 and HNR from a mono waveform using Parselmouth.

    Args:
        waveform: 1D numpy array of audio samples
        sr: sample rate

    Returns:
        dict with keys: median_f0, mean_hnr
    """
    import parselmouth

    snd = parselmouth.Sound(waveform, sampling_frequency=sr)

    # Pitch (F0) — use default Praat settings
    pitch = snd.to_pitch()
    f0_values = pitch.selected_array["frequency"]
    voiced = f0_values[f0_values > 0]
    median_f0 = float(np.median(voiced)) if len(voiced) > 0 else 0.0

    # Harmonics-to-Noise Ratio
    hnr = snd.to_harmonicity()
    hnr_values = hnr.values[0]
    valid_hnr = hnr_values[hnr_values != -200]  # Praat uses -200 for unvoiced
    mean_hnr = float(np.mean(valid_hnr)) if len(valid_hnr) > 0 else 0.0

    return {"median_f0": median_f0, "mean_hnr": mean_hnr}


def group_by_source(names: list[str]) -> dict[str, list[int]]:
    """Group item indices by source video prefix.

    Strips trailing _NN suffix: clip_001_03 -> clip_001

    Args:
        names: list of clip names

    Returns:
        dict mapping source prefix -> list of item indices
    """
    groups = {}
    for idx, name in enumerate(names):
        prefix = re.sub(r"_\d+$", "", name)
        groups.setdefault(prefix, []).append(idx)
    return groups


def sample_indices(group_size: int, samples_per_source: int) -> list[int]:
    """Pick evenly-spaced indices from a group.

    Args:
        group_size: number of items in the group
        samples_per_source: how many to pick

    Returns:
        list of integer indices
    """
    n = min(samples_per_source, group_size)
    return [int(i) for i in np.linspace(0, group_size - 1, n)]


def generate_descriptor(median_f0: float, mean_hnr: float,
                        min_f0_female: float = 165.0,
                        mode: str = "auto") -> str:
    """Generate a CLAP-compatible voice descriptor string.

    Args:
        median_f0: median fundamental frequency in Hz
        mean_hnr: mean harmonics-to-noise ratio in dB
        min_f0_female: F0 threshold for male/female split
        mode: "auto" for full descriptors, "label_only" for gender only

    Returns:
        descriptor string, e.g. "breathy high-pitched female"
    """
    is_female = median_f0 >= min_f0_female

    if mode == "label_only":
        return "female voice" if is_female else "male voice"

    # Pitch label
    if median_f0 > 250:
        pitch = "high-pitched"
    elif median_f0 >= min_f0_female:
        pitch = "mid-pitched"
    else:
        pitch = "deep"

    # Breathiness label
    breath = "breathy" if mean_hnr < 10 else "clear"

    gender = "female" if is_female else "male"

    return f"{breath} {pitch} {gender}"


def tag_prompt(prompt: str, descriptor: str, position: str = "prepend") -> str:
    """Prepend or append a voice descriptor to a prompt.

    Args:
        prompt: existing prompt string
        descriptor: voice descriptor to add
        position: "prepend" or "append"

    Returns:
        tagged prompt string
    """
    if not descriptor:
        return prompt
    if not prompt:
        return descriptor
    if position == "append":
        return f"{prompt}, {descriptor}"
    return f"{descriptor}, {prompt}"
```

**Step 2: Verify module loads**

Run:
```bash
python3 -c "from dataset.voice_analysis import extract_voice_features, group_by_source, sample_indices, generate_descriptor, tag_prompt; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add dataset/voice_analysis.py
git commit -m "feat: add voice analysis helpers (F0, HNR, grouping, descriptors)"
```

---

### Task 3: FoleyTuneVoiceTagger node

**Files:**
- Modify: `nodes_dataset.py`

**Step 1: Add the VoiceTagger class**

Add after the `FoleyTuneDatasetQualityFilter` class (around line 720) in `nodes_dataset.py`:

```python
class FoleyTuneVoiceTagger:
    """Analyze audio clips, cluster by speaker, and tag prompts with voice descriptors.

    Groups clips by source video prefix, samples a few per source for acoustic
    analysis (F0, HNR), clusters into speaker groups, and prepends/appends
    voice descriptors to each clip's prompt field.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEYTUNE_AUDIO_DATASET,),
                "num_speakers": ("INT", {
                    "default": 2, "min": 1, "max": 10,
                    "tooltip": "Expected number of distinct voices in the dataset.",
                }),
                "samples_per_source": ("INT", {
                    "default": 3, "min": 1, "max": 10,
                    "tooltip": "How many clips to analyze per source video (evenly spaced).",
                }),
            },
            "optional": {
                "min_f0_female": ("FLOAT", {
                    "default": 165.0, "min": 80.0, "max": 400.0, "step": 5.0,
                    "tooltip": "F0 threshold (Hz) for male/female split. Only used when num_speakers <= 2.",
                }),
                "descriptor_mode": (["auto", "label_only", "custom"], {
                    "default": "auto",
                    "tooltip": "auto: F0+HNR descriptors. label_only: gender only. custom: user JSON.",
                }),
                "custom_descriptors": ("STRING", {
                    "default": "",
                    "tooltip": 'JSON: {"clip_001": "breathy soprano", "clip_002": "deep male"}',
                }),
                "tag_position": (["prepend", "append"], {
                    "default": "prepend",
                }),
            },
        }

    RETURN_TYPES = (FOLEYTUNE_AUDIO_DATASET, "STRING")
    RETURN_NAMES = ("dataset", "report")
    FUNCTION = "tag_voices"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
    DESCRIPTION = (
        "Analyze vocal characteristics per source video, cluster by speaker, "
        "and tag prompts with voice descriptors (pitch, breathiness, gender)."
    )

    def tag_voices(self, dataset, num_speakers, samples_per_source,
                   min_f0_female=165.0, descriptor_mode="auto",
                   custom_descriptors="", tag_position="prepend"):
        from dataset.voice_analysis import (
            extract_voice_features, group_by_source, sample_indices,
            generate_descriptor, tag_prompt,
        )

        names = [item["name"] for item in dataset]
        groups = group_by_source(names)

        lines = ["=== Voice Tagger Report ===", ""]
        lines.append(f"Sources: {len(groups)}, Clips: {len(dataset)}, "
                      f"Speakers: {num_speakers}, Mode: {descriptor_mode}")
        lines.append("")

        # --- Analyze samples per source ---
        source_features = {}  # prefix -> {median_f0, mean_hnr}
        for prefix, indices in sorted(groups.items()):
            pick = sample_indices(len(indices), samples_per_source)
            sampled_indices = [indices[i] for i in pick]
            sampled_names = [names[i] for i in sampled_indices]

            f0s, hnrs = [], []
            for idx in sampled_indices:
                item = dataset[idx]
                wav = item["waveform"]
                sr = item["sample_rate"]
                # Convert to mono numpy
                if hasattr(wav, "numpy"):
                    wav_np = wav.squeeze().numpy()
                elif isinstance(wav, np.ndarray):
                    wav_np = wav.squeeze()
                else:
                    wav_np = np.array(wav).squeeze()
                if wav_np.ndim > 1:
                    wav_np = wav_np.mean(axis=0)

                feats = extract_voice_features(wav_np, sr)
                f0s.append(feats["median_f0"])
                hnrs.append(feats["mean_hnr"])

            avg_f0 = float(np.mean([f for f in f0s if f > 0])) if any(f > 0 for f in f0s) else 0.0
            avg_hnr = float(np.mean(hnrs)) if hnrs else 0.0
            source_features[prefix] = {"median_f0": avg_f0, "mean_hnr": avg_hnr}

            lines.append(f"  {prefix} ({len(indices)} segs) — "
                          f"sampled {sampled_names} — "
                          f"F0={avg_f0:.0f}Hz HNR={avg_hnr:.1f}dB")

        lines.append("")

        # --- Generate descriptors ---
        source_descriptors = {}
        if descriptor_mode == "custom" and custom_descriptors.strip():
            import json
            source_descriptors = json.loads(custom_descriptors)
        elif num_speakers <= 2:
            # Simple F0 threshold split
            for prefix, feats in source_features.items():
                source_descriptors[prefix] = generate_descriptor(
                    feats["median_f0"], feats["mean_hnr"],
                    min_f0_female=min_f0_female, mode=descriptor_mode,
                )
        else:
            # Multi-speaker: use Resemblyzer clustering
            from resemblyzer import VoiceEncoder, preprocess_wav
            from sklearn.cluster import KMeans

            encoder = VoiceEncoder()
            source_embeddings = {}
            for prefix, indices in groups.items():
                pick = sample_indices(len(indices), samples_per_source)
                embeds = []
                for i in pick:
                    item = dataset[indices[i]]
                    wav = item["waveform"]
                    sr = item["sample_rate"]
                    if hasattr(wav, "numpy"):
                        wav_np = wav.squeeze().numpy()
                    elif isinstance(wav, np.ndarray):
                        wav_np = wav.squeeze()
                    else:
                        wav_np = np.array(wav).squeeze()
                    if wav_np.ndim > 1:
                        wav_np = wav_np.mean(axis=0)
                    processed = preprocess_wav(wav_np, source_sr=sr)
                    embeds.append(encoder.embed_utterance(processed))
                source_embeddings[prefix] = np.mean(embeds, axis=0)

            prefixes = sorted(source_embeddings.keys())
            X = np.stack([source_embeddings[p] for p in prefixes])
            labels = KMeans(n_clusters=num_speakers, random_state=42).fit_predict(X)

            for prefix, label in zip(prefixes, labels):
                feats = source_features[prefix]
                desc = generate_descriptor(
                    feats["median_f0"], feats["mean_hnr"],
                    min_f0_female=min_f0_female, mode=descriptor_mode,
                )
                source_descriptors[prefix] = desc

        # --- Tag map JSON for RetagNPZ ---
        import json
        lines.append("Tag assignments:")
        for prefix in sorted(source_descriptors.keys()):
            desc = source_descriptors[prefix]
            count = len(groups.get(prefix, []))
            lines.append(f"  {prefix} ({count} clips) → \"{desc}\"")

        lines.append("")
        lines.append("tag_map JSON (for RetagNPZ):")
        lines.append(json.dumps(source_descriptors, indent=2))
        lines.append("")

        # --- Apply tags to dataset ---
        tagged_count = 0
        for item in dataset:
            name = item["name"]
            prefix = re.sub(r"_\d+$", "", name)
            descriptor = source_descriptors.get(prefix, "")
            if descriptor:
                old_prompt = item.get("prompt", "")
                item["prompt"] = tag_prompt(old_prompt, descriptor, tag_position)
                tagged_count += 1

        lines.append(f"Tagged {tagged_count}/{len(dataset)} clips")

        # Show a few examples
        lines.append("")
        lines.append("Examples:")
        for item in dataset[:5]:
            lines.append(f"  {item['name']}: \"{item.get('prompt', '')}\"")

        report = "\n".join(lines)
        return (dataset, report)
```

**Step 2: Add `re` import at top of file if not present**

Check if `import re` exists at the top of `nodes_dataset.py`. If not, add it.

**Step 3: Register the node in NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS**

In `NODE_CLASS_MAPPINGS` dict add:
```python
    "FoleyTuneVoiceTagger": FoleyTuneVoiceTagger,
```

In `NODE_DISPLAY_NAME_MAPPINGS` dict add:
```python
    "FoleyTuneVoiceTagger": "FoleyTune Voice Tagger",
```

**Step 4: Verify node loads**

Run:
```bash
python3 -c "from nodes_dataset import FoleyTuneVoiceTagger; print('INPUT_TYPES:', list(FoleyTuneVoiceTagger.INPUT_TYPES()['required'].keys())); print('OK')"
```
Expected: `INPUT_TYPES: ['dataset', 'num_speakers', 'samples_per_source']` then `OK`

**Step 5: Commit**

```bash
git add nodes_dataset.py
git commit -m "feat: add FoleyTuneVoiceTagger node for speaker-aware prompt tagging"
```

---

### Task 4: FoleyTuneRetagNPZ node

**Files:**
- Modify: `nodes_dataset.py`

**Step 1: Add the RetagNPZ class**

Add after the `FoleyTuneVoiceTagger` class in `nodes_dataset.py`:

```python
class FoleyTuneRetagNPZ:
    """Update prompt and CLAP text embedding in existing NPZ files.

    Re-encodes prompts with voice descriptors without re-extracting
    visual features (SigLIP2, Synchformer). Only updates text_embedding
    and prompt fields.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_deps": ("FOLEYTUNE_DEPS",),
                "npz_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Path to folder with existing .npz files to update.",
                }),
                "tag_map": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "tooltip": 'JSON: {"clip_001": "breathy high-pitched female", ...}. '
                               'Copy from VoiceTagger report output.',
                }),
            },
            "optional": {
                "tag_position": (["prepend", "append"], {"default": "prepend"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "retag_npz"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Update prompt and CLAP text embedding in existing NPZ files. "
        "Only re-computes text embeddings — visual features are untouched."
    )

    def retag_npz(self, hunyuan_deps, npz_dir, tag_map, tag_position="prepend"):
        import json
        import re
        from dataset.voice_analysis import tag_prompt

        npz_dir = Path(npz_dir.strip())
        if not npz_dir.exists():
            raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")

        descriptors = json.loads(tag_map)
        if not descriptors:
            return ("No tag_map provided, nothing to do.",)

        npz_files = sorted(npz_dir.glob("*.npz"))
        if not npz_files:
            return (f"No .npz files found in {npz_dir}",)

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        lines = ["=== RetagNPZ Report ===", ""]
        lines.append(f"Directory: {npz_dir}")
        lines.append(f"NPZ files: {len(npz_files)}")
        lines.append(f"Tag map entries: {len(descriptors)}")
        lines.append("")

        updated = 0
        skipped = 0

        # Load CLAP once
        hunyuan_deps.clap_model.to(device)

        for npz_path in npz_files:
            stem = npz_path.stem
            prefix = re.sub(r"_\d+$", "", stem)

            descriptor = descriptors.get(prefix)
            if not descriptor:
                skipped += 1
                continue

            # Load existing data
            data = dict(np.load(str(npz_path), allow_pickle=True))
            old_prompt = str(data.get("prompt", ""))
            new_prompt = tag_prompt(old_prompt, descriptor, tag_position)

            if new_prompt == old_prompt:
                skipped += 1
                continue

            # Re-encode CLAP text embedding
            text_inputs = hunyuan_deps.clap_tokenizer(
                [new_prompt], padding=True, truncation=True, max_length=100,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                clap_outputs = hunyuan_deps.clap_model(
                    **text_inputs, output_hidden_states=True, return_dict=True
                )
            new_text_embedding = clap_outputs.last_hidden_state.cpu().float().numpy()

            # Save updated NPZ
            data["prompt"] = new_prompt
            data["text_embedding"] = new_text_embedding
            np.savez(str(npz_path.with_suffix("")), **data)

            lines.append(f"  {stem}: \"{old_prompt}\" → \"{new_prompt}\"")
            updated += 1

        hunyuan_deps.clap_model.to(offload_device)

        lines.append("")
        lines.append(f"Updated: {updated}, Skipped: {skipped}")

        report = "\n".join(lines)
        return (report,)
```

**Step 2: Register the node**

In `NODE_CLASS_MAPPINGS` dict add:
```python
    "FoleyTuneRetagNPZ": FoleyTuneRetagNPZ,
```

In `NODE_DISPLAY_NAME_MAPPINGS` dict add:
```python
    "FoleyTuneRetagNPZ": "FoleyTune Retag NPZ",
```

**Step 3: Verify node loads**

Run:
```bash
python3 -c "from nodes_dataset import FoleyTuneRetagNPZ; print('INPUT_TYPES:', list(FoleyTuneRetagNPZ.INPUT_TYPES()['required'].keys())); print('OK')"
```
Expected: `INPUT_TYPES: ['hunyuan_deps', 'npz_dir', 'tag_map']` then `OK`

**Step 4: Commit**

```bash
git add nodes_dataset.py
git commit -m "feat: add FoleyTuneRetagNPZ node for updating CLAP embeddings in existing NPZs"
```

---

### Task 5: Patch BatchFeatureExtractor to respect per-item prompts

**Files:**
- Modify: `nodes_lora.py:526`

**Step 1: Update prompt resolution in BatchFeatureExtractor**

In `nodes_lora.py`, in the `extract_batch` method of `FoleyTuneBatchFeatureExtractor`,
change line 526 from:

```python
            clip_prompt = txt_path.read_text().strip() if txt_path.exists() else prompt
```

to:

```python
            clip_prompt = (txt_path.read_text().strip() if txt_path.exists()
                           else item.get("prompt") or prompt)
```

This adds a fallback chain: sidecar `.txt` > `item["prompt"]` (from VoiceTagger) > global prompt input.

**Step 2: Verify no breakage**

Run:
```bash
python3 -c "from nodes_lora import FoleyTuneBatchFeatureExtractor; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add nodes_lora.py
git commit -m "feat: BatchFeatureExtractor respects per-item prompt from VoiceTagger"
```

---

### Task 6: Add torch import guard to RetagNPZ

**Files:**
- Modify: `nodes_dataset.py`

**Step 1: Verify torch is imported at module level**

Check that `import torch` exists at the top of `nodes_dataset.py`. The RetagNPZ node
uses `torch.no_grad()`. If not present, add it to the existing imports.

**Step 2: Commit if changed**

```bash
git add nodes_dataset.py
git commit -m "fix: ensure torch import for RetagNPZ node"
```

---

### Task 7: End-to-end smoke test

**Step 1: Test the full VoiceTagger pipeline on a small dataset**

```bash
python3 -c "
import numpy as np
from nodes_dataset import FoleyTuneVoiceTagger

# Create fake dataset with 2 'sources', 3 segments each
dataset = []
for src in range(2):
    for seg in range(3):
        sr = 48000
        dur = 2.0  # short for speed
        t = np.linspace(0, dur, int(sr * dur))
        # Source 0: high pitch (300Hz), Source 1: low pitch (120Hz)
        freq = 300 if src == 0 else 120
        wav = np.sin(2 * np.pi * freq * t).astype(np.float32)
        dataset.append({
            'waveform': wav,
            'sample_rate': sr,
            'name': f'clip_{src:03d}_{seg}',
            'prompt': 'test sound',
        })

tagger = FoleyTuneVoiceTagger()
result, report = tagger.tag_voices(dataset, num_speakers=2, samples_per_source=2)
print(report)
print()
for item in result:
    print(f\"{item['name']}: {item['prompt']}\")
"
```

Expected: `clip_000_*` tagged with a female descriptor, `clip_001_*` tagged with a male descriptor.

**Step 2: Commit all remaining changes**

```bash
git add -A
git commit -m "feat: voice tagger pipeline complete — VoiceTagger + RetagNPZ nodes"
```
