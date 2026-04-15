# Dataset Denoiser Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add spectral gating noise reduction as an optional feature of the quality filter. A small `FoleyTuneDenoiserSettings` node provides the config; the quality filter accepts it as an optional input, denoises before scoring, and passes denoised audio downstream.

**Architecture:** Settings node outputs a typed `FOLEYTUNE_DENOISE_SETTINGS` dict. Quality filter gets a new optional `denoise_settings` input. When connected, it groups clips by source, finds quietest segment per source as noise reference, denoises via `noisereduce`, then scores on clean audio. Follows existing patterns in `nodes_dataset.py`.

**Tech Stack:** noisereduce (spectral gating), numpy, torch, ComfyUI node API.

---

### Task 1: Add noisereduce dependency

**Files:**
- Modify: `requirements.txt`

**Step 1: Add noisereduce to requirements**

Add to the `# Audio processing` section in `requirements.txt`, after `pyloudnorm`:

```
noisereduce
```

**Step 2: Install and verify**

Run:
```bash
pip install noisereduce
python3 -c "import noisereduce; print(noisereduce.__version__)"
```
Expected: version string, no error.

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add noisereduce dependency for spectral gating"
```

---

### Task 2: Add FoleyTuneDenoiserSettings node

**Files:**
- Modify: `nodes_dataset.py`

**Step 1: Add the type constant and settings class**

Add the type constant near the top of `nodes_dataset.py`, after the existing type constants (around line 35):

```python
FOLEYTUNE_DENOISE_SETTINGS = "FOLEYTUNE_DENOISE_SETTINGS"
```

Add the settings node class before `FoleyTuneDatasetQualityFilter` (around line 518):

```python
# ─── Node: Denoiser Settings ────────────────────────────────────────────────


class FoleyTuneDenoiserSettings:
    """Configuration node for spectral gating noise reduction.

    Connect to a Quality Filter's denoise_settings input to enable
    denoising before quality scoring. Removes stationary noise like
    AC hum, fans, and room tone.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Noise reduction strength. 0 = off, 1 = full removal. "
                               "0.6-0.8 is good for AC noise without artifacts.",
                }),
            },
            "optional": {
                "stationary": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Assume noise is stationary (AC, fans). "
                               "Disable for varying background noise.",
                }),
                "n_fft": ("INT", {
                    "default": 2048, "min": 512, "max": 8192, "step": 512,
                    "tooltip": "FFT window size. Larger = better frequency resolution, slower.",
                }),
            },
        }

    RETURN_TYPES = (FOLEYTUNE_DENOISE_SETTINGS,)
    RETURN_NAMES = ("denoise_settings",)
    FUNCTION = "get_settings"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
    DESCRIPTION = (
        "Configure spectral gating noise reduction. Connect to a "
        "Quality Filter to denoise clips before scoring."
    )

    def get_settings(self, strength=0.7, stationary=True, n_fft=2048):
        return ({"strength": strength, "stationary": stationary, "n_fft": n_fft},)
```

**Step 2: Register the node**

In `NODE_CLASS_MAPPINGS` dict add:
```python
    "FoleyTuneDenoiserSettings": FoleyTuneDenoiserSettings,
```

In `NODE_DISPLAY_NAME_MAPPINGS` dict add:
```python
    "FoleyTuneDenoiserSettings": "FoleyTune Denoiser Settings",
```

**Step 3: Commit**

```bash
git add nodes_dataset.py
git commit -m "feat: add FoleyTuneDenoiserSettings config node"
```

---

### Task 3: Add denoising to QualityFilter

**Files:**
- Modify: `nodes_dataset.py` (FoleyTuneDatasetQualityFilter class, around line 520)

**Step 1: Add optional denoise_settings input**

In `FoleyTuneDatasetQualityFilter.INPUT_TYPES`, add to the `"optional"` dict:

```python
                "denoise_settings": (FOLEYTUNE_DENOISE_SETTINGS, {
                    "tooltip": "Connect a Denoiser Settings node to denoise clips "
                               "before scoring. Denoised audio flows downstream.",
                }),
```

**Step 2: Add denoise_settings parameter to filter_quality method**

Update the method signature to accept the new optional parameter:

```python
    def filter_quality(self, dataset, min_quality_score: float,
                       skip_rejected: bool, clap_prompt: str = "",
                       min_bandwidth_score: float = 0.0,
                       min_spectral_score: float = 0.2,
                       min_clap_score: float = 0.0,
                       weight_bandwidth: float = 0.4,
                       weight_spectral: float = 0.4,
                       weight_clap: float = 0.2,
                       denoise_settings=None):
```

**Step 3: Add denoising logic before the scoring loop**

After the CLAP setup and weight normalization, before the `passed = []` line,
add the denoising pass. This replaces each item's waveform with the denoised
version so all downstream scoring and output uses clean audio:

```python
        # --- Optional denoising pass ---
        if denoise_settings is not None and denoise_settings["strength"] > 0:
            import re
            import noisereduce as nr
            from voice_analysis import group_by_source

            strength = denoise_settings["strength"]
            stationary = denoise_settings["stationary"]
            n_fft = denoise_settings["n_fft"]

            names = [item["name"] for item in dataset]
            groups = group_by_source(names)

            # Find quietest segment per source as noise reference
            noise_profiles = {}
            for prefix, indices in groups.items():
                best_idx, best_rms = indices[0], float("inf")
                for idx in indices:
                    wav = dataset[idx]["waveform"][0].float().cpu()
                    rms = wav.pow(2).mean().sqrt().item()
                    if rms < best_rms:
                        best_rms = rms
                        best_idx = idx
                quiet_wav = dataset[best_idx]["waveform"][0].float().cpu().numpy()
                noise_profiles[prefix] = (
                    quiet_wav.mean(axis=0) if quiet_wav.ndim > 1 else quiet_wav
                )

            # Denoise each clip
            denoised_dataset = []
            for item in dataset:
                throw_exception_if_processing_interrupted()
                wav = item["waveform"][0].float()
                sr = item["sample_rate"]
                wav_np = wav.cpu().numpy()
                rms_in = np.sqrt(np.mean(wav_np ** 2)).clip(1e-8)

                prefix = re.sub(r"_\d+$", "", item["name"])
                y_noise = noise_profiles.get(prefix)

                denoised = np.stack([
                    nr.reduce_noise(
                        y=wav_np[c], sr=sr, y_noise=y_noise,
                        prop_decrease=strength, stationary=stationary,
                        n_fft=n_fft,
                    )
                    for c in range(wav_np.shape[0])
                ])

                rms_out = np.sqrt(np.mean(denoised ** 2)).clip(1e-8)
                denoised = denoised * (rms_in / rms_out)
                peak = np.abs(denoised).max()
                if peak > 1.0:
                    denoised = denoised / peak

                new_item = dict(item)
                new_item["waveform"] = torch.from_numpy(denoised).unsqueeze(0)
                denoised_dataset.append(new_item)

            dataset = denoised_dataset
            print(f"[QualityFilter] Denoised {len(dataset)} clips  "
                  f"strength={strength}  stationary={stationary}", flush=True)
```

**Step 4: Commit**

```bash
git add nodes_dataset.py
git commit -m "feat: quality filter accepts optional denoise settings"
```

---

### Task 4: Smoke test

**Step 1: Test that denoiser settings flow through quality filter**

```bash
python3 -c "
import sys, types
comfy_mod = types.ModuleType('comfy')
mm_mod = types.ModuleType('comfy.model_management')
mm_mod.throw_exception_if_processing_interrupted = lambda: None
comfy_mod.model_management = mm_mod
sys.modules['comfy'] = comfy_mod
sys.modules['comfy.model_management'] = mm_mod

from nodes_dataset import FoleyTuneDenoiserSettings, FoleyTuneDatasetQualityFilter

# Test settings node
settings_node = FoleyTuneDenoiserSettings()
(settings,) = settings_node.get_settings(strength=0.8, stationary=True, n_fft=2048)
print(f'Settings: {settings}')
assert settings['strength'] == 0.8
assert settings['stationary'] == True
assert settings['n_fft'] == 2048

# Test quality filter accepts it without crashing
import torch, numpy as np
sr = 48000
dur = 2.0
t = np.linspace(0, dur, int(sr * dur))
clean = np.sin(2 * np.pi * 300 * t).astype(np.float32)
noise = 0.05 * np.random.randn(len(t)).astype(np.float32)
noisy = clean + noise
wav = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0)  # [1, 1, L]

dataset = [
    {'waveform': wav, 'sample_rate': sr, 'name': 'clip_000_0'},
    {'waveform': wav.clone(), 'sample_rate': sr, 'name': 'clip_000_1'},
]

qf = FoleyTuneDatasetQualityFilter()
result, report = qf.filter_quality(
    dataset, min_quality_score=0.0, skip_rejected=False,
    denoise_settings=settings,
)
print(report[:500])
print()

# Original should not be mutated
assert dataset[0]['waveform'] is wav, 'Input mutated!'
print('All checks passed')
"
```

Expected: settings dict created, quality filter runs with denoising, original not mutated.
