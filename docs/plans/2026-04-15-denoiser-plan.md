# Dataset Denoiser Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `FoleyTuneDatasetDenoiser` ComfyUI node that removes stationary background noise (AC hum, fans) from training audio clips using spectral gating.

**Architecture:** Single node using `noisereduce` library. Processes each clip's waveform through spectral gating, preserves RMS level, returns denoised dataset. Follows the same `dict(item)` shallow-copy pattern as all other dataset processing nodes.

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

### Task 2: Add FoleyTuneDatasetDenoiser node

**Files:**
- Modify: `nodes_dataset.py` (insert before `FoleyTuneDatasetAugmenter` at line ~1317)

**Step 1: Add the Denoiser class**

Insert before `# --- Node 7: Dataset Augmenter ---` (line ~1317) in `nodes_dataset.py`:

```python
# --- Node: Dataset Denoiser ------------------------------------------------


class FoleyTuneDatasetDenoiser:
    """Remove stationary background noise (AC hum, fans) via spectral gating.

    Uses noisereduce to auto-estimate a noise profile from each clip and
    gate it out. Best for stationary noise. Preserves RMS level.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEYTUNE_AUDIO_DATASET,),
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Noise reduction strength. 0 = off, 1 = full removal. "
                               "0.6-0.8 is good for AC noise without artifacts.",
                }),
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

    RETURN_TYPES = (FOLEYTUNE_AUDIO_DATASET, "STRING")
    RETURN_NAMES = ("dataset", "report")
    FUNCTION = "denoise"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
    DESCRIPTION = (
        "Remove stationary background noise (AC hum, fans) from clips "
        "using spectral gating. Adjustable strength to avoid artifacts."
    )

    def denoise(self, dataset, strength=0.7, stationary=True, n_fft=2048):
        import noisereduce as nr

        out = []
        lines = ["=== Denoiser Report ===", ""]
        lines.append(f"Clips: {len(dataset)}, Strength: {strength}, "
                      f"Stationary: {stationary}, n_fft: {n_fft}")
        lines.append("")

        for item in dataset:
            throw_exception_if_processing_interrupted()
            wav = item["waveform"][0].float()  # [C, L]
            sr = item["sample_rate"]

            wav_np = wav.cpu().numpy()  # [C, L]
            rms_in = np.sqrt(np.mean(wav_np ** 2)).clip(1e-8)

            # Process each channel
            denoised = np.stack([
                nr.reduce_noise(
                    y=wav_np[c], sr=sr,
                    prop_decrease=strength,
                    stationary=stationary,
                    n_fft=n_fft,
                )
                for c in range(wav_np.shape[0])
            ])  # [C, L]

            # Preserve RMS level
            rms_out = np.sqrt(np.mean(denoised ** 2)).clip(1e-8)
            denoised = denoised * (rms_in / rms_out)

            # Peak limit
            peak = np.abs(denoised).max()
            if peak > 1.0:
                denoised = denoised / peak

            wav_out = torch.from_numpy(denoised).unsqueeze(0)  # [1, C, L]
            new_item = dict(item)
            new_item["waveform"] = wav_out
            out.append(new_item)

            reduction_db = 20 * np.log10(rms_out / rms_in + 1e-8)
            lines.append(f"  {item['name']}: noise floor {reduction_db:+.1f}dB")

        lines.append("")
        lines.append(f"Denoised {len(out)} clips")
        report = "\n".join(lines)

        print(f"[FoleyTuneDatasetDenoiser] {len(out)} clips denoised  "
              f"strength={strength}  stationary={stationary}", flush=True)
        return (out, report)
```

**Step 2: Register the node in NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS**

In `NODE_CLASS_MAPPINGS` dict add:
```python
    "FoleyTuneDatasetDenoiser": FoleyTuneDatasetDenoiser,
```

In `NODE_DISPLAY_NAME_MAPPINGS` dict add:
```python
    "FoleyTuneDatasetDenoiser": "FoleyTune Dataset Denoiser",
```

**Step 3: Verify node loads**

Run (with comfy mock if needed):
```bash
python3 -c "
import sys, types
comfy_mod = types.ModuleType('comfy')
mm_mod = types.ModuleType('comfy.model_management')
mm_mod.throw_exception_if_processing_interrupted = lambda: None
comfy_mod.model_management = mm_mod
sys.modules['comfy'] = comfy_mod
sys.modules['comfy.model_management'] = mm_mod

from nodes_dataset import FoleyTuneDatasetDenoiser
print('INPUT_TYPES:', list(FoleyTuneDatasetDenoiser.INPUT_TYPES()['required'].keys()))
print('OK')
"
```
Expected: `INPUT_TYPES: ['dataset']` then `OK`

**Step 4: Commit**

```bash
git add nodes_dataset.py
git commit -m "feat: add FoleyTuneDatasetDenoiser node (spectral gating for AC/fan noise)"
```

---

### Task 3: Smoke test with synthetic data

**Step 1: Test denoiser on a sine wave with added noise**

```bash
python3 -c "
import sys, types
comfy_mod = types.ModuleType('comfy')
mm_mod = types.ModuleType('comfy.model_management')
mm_mod.throw_exception_if_processing_interrupted = lambda: None
comfy_mod.model_management = mm_mod
sys.modules['comfy'] = comfy_mod
sys.modules['comfy.model_management'] = mm_mod

import torch
import numpy as np
from nodes_dataset import FoleyTuneDatasetDenoiser

sr = 48000
dur = 2.0
t = np.linspace(0, dur, int(sr * dur))

# Clean signal: 300Hz tone
clean = np.sin(2 * np.pi * 300 * t).astype(np.float32)
# Add noise: low-level broadband hiss
noise = 0.05 * np.random.randn(len(t)).astype(np.float32)
noisy = clean + noise

wav_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0)  # [1, 1, L]
dataset = [{'waveform': wav_tensor, 'sample_rate': sr, 'name': 'test_clip_0'}]

denoiser = FoleyTuneDatasetDenoiser()
result, report = denoiser.denoise(dataset, strength=0.7, stationary=True)

print(report)

# Check output shape matches input
assert result[0]['waveform'].shape == wav_tensor.shape, 'Shape mismatch!'
# Check original not mutated
assert dataset[0]['waveform'] is wav_tensor, 'Input was mutated!'
print()
print('All checks passed')
"
```

Expected: report with noise reduction stats, shape check passes, no mutation.
