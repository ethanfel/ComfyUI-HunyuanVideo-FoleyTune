# VAE Round-Trip Nodes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two paired ComfyUI nodes (DAC + Woosh) that run an AUDIO clip through a VAE encode→decode round-trip, for A/B comparison of reconstruction ceilings.

**Architecture:** Vendor only the AE subtree of SonyResearch/Woosh into `woosh_ae/`. Reuse the existing DAC-VAE codepath in `hunyuanvideo_foley/`. Implement shared helpers (resample/downmix/pad/trim) in `nodes_vae.py`; implement both nodes there. Auto-download Woosh-AE weights from its GitHub Release on first node run.

**Tech Stack:** Python 3.10+, PyTorch, ComfyUI node API, torchaudio, urllib (stdlib zip download), existing `hunyuanvideo_foley.models.dac_vae.model.dac.DAC`.

**Design doc:** `docs/plans/2026-04-21-vae-roundtrip-design.md`

**Verification strategy:** No pytest infra exists in this repo. Each task ends with a runnable `python -c '...'` smoke check that exercises the unit directly. Final task is a manual ComfyUI UI test documented in the plan.

---

## Task 1: Vendor Woosh-AE source

**Files:**
- Create: `woosh_ae/` (new directory; exact contents determined by Woosh repo inspection — see Step 3)
- Create: `THIRD_PARTY_NOTICES.md` at repo root

**Step 1: Clone Woosh to a scratch directory**

```bash
cd /tmp && rm -rf woosh-upstream && git clone --depth 1 https://github.com/SonyResearch/Woosh.git woosh-upstream
```

Expected: clone succeeds, ~a few hundred MB of source (no weights).

**Step 2: Read Woosh's test_Woosh-AE.py to find the exact import surface**

```bash
cat /tmp/woosh-upstream/test_Woosh-AE.py
```

Record the module paths of `AudioAutoEncoder`, `LoadConfig`, and any `flowmatching_integrate` import. These determine which files to vendor.

**Step 3: Identify the minimal AE module set**

Starting from the imports in `test_Woosh-AE.py`, trace the dependency tree (use `grep -r "^import\|^from" ...`) and list only the Python files reachable from `AudioAutoEncoder` and `LoadConfig`. Exclude:
- Anything under `flowmatching_integrate/` that isn't imported by the AE path
- T2A, V2A, CLAP code
- Gradio demos
- Test data

Write the file list to `/tmp/woosh-ae-filelist.txt` for the next step.

**Step 4: Copy those files into `woosh_ae/`**

```bash
cd /media/p5/ComfyUI-HunyuanVideo-Foley
mkdir -p woosh_ae
# For each path in /tmp/woosh-ae-filelist.txt, cp --parents preserving structure under woosh_ae/
```

Preserve relative directory structure so in-file imports keep working (e.g. if Woosh imports `from .vocos.blocks import X`, keep `vocos/blocks.py` under the same relative path).

Create `woosh_ae/__init__.py` that re-exports:
```python
from .audio_autoencoder import AudioAutoEncoder
from .utils import LoadConfig  # path adjusted to match actual vendored location
```

**Step 5: Fix any absolute imports**

Woosh's code may use `from flowmatching_integrate.X import Y` as absolute imports. Convert these to relative imports within `woosh_ae/` or (if cleaner) add a small shim module. Do NOT add `sys.path` hacks.

Run:
```bash
python -c "from woosh_ae import AudioAutoEncoder, LoadConfig; print('OK')"
```
Expected output: `OK`. If any `ModuleNotFoundError`, fix that import and rerun.

**Step 6: Write THIRD_PARTY_NOTICES.md**

```markdown
# Third-Party Notices

## Woosh Audio Autoencoder (Woosh-AE)

Portions of `woosh_ae/` are vendored from Sony AI's Woosh project
(https://github.com/SonyResearch/Woosh), commit <HASH>.

Licensed under the MIT License. Original copyright © Sony AI.
See https://github.com/SonyResearch/Woosh/blob/main/LICENSE for the
full license text.
```

Fill `<HASH>` with the short SHA from step 1.

**Step 7: Commit**

```bash
cd /media/p5/ComfyUI-HunyuanVideo-Foley
git add woosh_ae/ THIRD_PARTY_NOTICES.md
git commit -m "feat: vendor Woosh-AE audio autoencoder source"
```

---

## Task 2: Weight auto-download helper

**Files:**
- Create: `nodes_vae.py` (top of file — helper only, nodes come later)

**Step 1: Find the Woosh v1.0.0 release asset URL for the AE**

```bash
curl -s https://api.github.com/repos/SonyResearch/Woosh/releases/tags/v1.0.0 | python -c "import json,sys; r=json.load(sys.stdin); [print(a['name'], a['browser_download_url'], a['size']) for a in r['assets']]"
```

Expected: a list of asset names and URLs. Find the one for Woosh-AE (likely `Woosh-AE.zip` or similar). Record:
- Asset filename
- Full download URL
- Expected file size (for integrity check)

**Step 2: Write the downloader**

Create `nodes_vae.py`:

```python
"""VAE round-trip nodes: DAC and Woosh-AE.

A/B tool for comparing reconstruction ceilings. Each node runs an AUDIO
input through an encode→decode cycle of its VAE and returns the output.
"""
import os
import sys
import urllib.request
import zipfile
import shutil
from loguru import logger

import torch
import torchaudio
import folder_paths
import comfy.model_management as mm

# --- Woosh-AE weight auto-download -------------------------------------------

WOOSH_AE_RELEASE_URL = "<FILL FROM TASK 2 STEP 1>"
WOOSH_AE_ZIP_NAME = "<FILL FROM TASK 2 STEP 1>"
WOOSH_AE_EXPECTED_SIZE = <FILL FROM TASK 2 STEP 1>  # bytes


def _woosh_ae_dir() -> str:
    return os.path.join(folder_paths.models_dir, "foley", "Woosh-AE")


def ensure_woosh_ae() -> str:
    """Download + extract Woosh-AE weights if missing. Returns absolute path to the checkpoint folder."""
    target = _woosh_ae_dir()
    if os.path.isdir(target) and os.listdir(target):
        return target

    os.makedirs(os.path.dirname(target), exist_ok=True)
    zip_path = os.path.join(folder_paths.models_dir, "foley", WOOSH_AE_ZIP_NAME)

    logger.info(f"Woosh-AE weights not found. Downloading {WOOSH_AE_ZIP_NAME} (~{WOOSH_AE_EXPECTED_SIZE // (1024*1024)} MB)")
    urllib.request.urlretrieve(WOOSH_AE_RELEASE_URL, zip_path)

    actual_size = os.path.getsize(zip_path)
    if actual_size != WOOSH_AE_EXPECTED_SIZE:
        os.remove(zip_path)
        raise RuntimeError(
            f"Woosh-AE download size mismatch: got {actual_size}, expected {WOOSH_AE_EXPECTED_SIZE}. "
            f"Try deleting {zip_path} and re-running."
        )

    logger.info(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(os.path.dirname(target))
    os.remove(zip_path)

    if not (os.path.isdir(target) and os.listdir(target)):
        raise RuntimeError(f"Extraction succeeded but {target} is empty. Zip layout may have changed.")

    logger.info(f"Woosh-AE ready at {target}")
    return target
```

**Step 3: Smoke-test the downloader without actually downloading**

```bash
python -c "
import sys
sys.path.insert(0, '/media/p5/ComfyUI-HunyuanVideo-Foley')
# Fake folder_paths for isolated import
import types
fp = types.ModuleType('folder_paths')
fp.models_dir = '/tmp'
sys.modules['folder_paths'] = fp
# comfy.model_management too
comfy = types.ModuleType('comfy'); mm = types.ModuleType('comfy.model_management'); comfy.model_management = mm
sys.modules['comfy'] = comfy; sys.modules['comfy.model_management'] = mm
from nodes_vae import ensure_woosh_ae, _woosh_ae_dir
print('helper import OK, target dir:', _woosh_ae_dir())
"
```

Expected: `helper import OK, target dir: /tmp/foley/Woosh-AE`.

**Step 4: Commit**

```bash
git add nodes_vae.py
git commit -m "feat: add Woosh-AE weight auto-downloader"
```

---

## Task 3: Shared audio helpers

**Files:**
- Modify: `nodes_vae.py` (append below downloader)

**Step 1: Write the helpers**

Append to `nodes_vae.py`:

```python
# --- Shared audio helpers ----------------------------------------------------

TARGET_SR = 48000

def _prep_audio(audio: dict) -> tuple[torch.Tensor, int, int]:
    """Normalize ComfyUI AUDIO to [B, 1, L] mono @ 48 kHz. Returns (wave, orig_sr, orig_len)."""
    wave = audio["waveform"]  # [B, C, L]
    sr = int(audio["sample_rate"])
    orig_len = wave.shape[-1]

    if wave.shape[1] > 1:
        logger.warning(f"Stereo input, downmixing to mono")
        wave = wave.mean(dim=1, keepdim=True)

    if sr != TARGET_SR:
        logger.warning(f"Resampling {sr} Hz -> {TARGET_SR} Hz")
        wave = torchaudio.functional.resample(wave, sr, TARGET_SR)

    return wave, sr, orig_len


def _pad_to_min(wave: torch.Tensor, min_len: int) -> tuple[torch.Tensor, int]:
    """Right-pad with silence if shorter than min_len. Returns (padded_wave, pad_applied)."""
    pad = max(0, min_len - wave.shape[-1])
    if pad:
        wave = torch.nn.functional.pad(wave, (0, pad))
    return wave, pad


def _finalize(wave: torch.Tensor, orig_len_at_target_sr: int) -> dict:
    """Trim to original duration (at target SR) and package as ComfyUI AUDIO."""
    wave = wave[..., :orig_len_at_target_sr]
    return {"waveform": wave.contiguous().cpu(), "sample_rate": TARGET_SR}
```

**Step 2: Smoke-test the helpers**

```bash
python -c "
import sys, types
sys.path.insert(0, '/media/p5/ComfyUI-HunyuanVideo-Foley')
fp = types.ModuleType('folder_paths'); fp.models_dir = '/tmp'; sys.modules['folder_paths'] = fp
comfy = types.ModuleType('comfy'); mm = types.ModuleType('comfy.model_management'); comfy.model_management = mm
sys.modules['comfy'] = comfy; sys.modules['comfy.model_management'] = mm

import torch
from nodes_vae import _prep_audio, _pad_to_min, _finalize, TARGET_SR

# stereo 44.1k, 2s clip
audio_in = {'waveform': torch.randn(1, 2, 44100*2), 'sample_rate': 44100}
wave, sr, orig_len = _prep_audio(audio_in)
print('after prep:', wave.shape, 'orig_sr:', sr, 'orig_len:', orig_len)
assert wave.shape[1] == 1, 'should be mono'
assert wave.shape[-1] == 96000, f'expected 96000 samples @ 48k, got {wave.shape[-1]}'

# short clip padding
short = torch.randn(1, 1, 100)
padded, pad = _pad_to_min(short, 2048)
assert padded.shape[-1] == 2048 and pad == 1948

# finalize trims
out = _finalize(torch.randn(1, 1, 96500), 96000)
assert out['waveform'].shape[-1] == 96000
assert out['sample_rate'] == 48000
print('helpers OK')
"
```

Expected ending: `helpers OK`.

**Step 3: Commit**

```bash
git add nodes_vae.py
git commit -m "feat: add shared audio helpers for VAE round-trip nodes"
```

---

## Task 4: Woosh VAE Round-Trip node

**Files:**
- Modify: `nodes_vae.py`

**Step 1: Write the node**

Append to `nodes_vae.py`:

```python
# --- Woosh node --------------------------------------------------------------

_WOOSH_AE_SINGLETON = {"model": None, "device": None, "dtype": None}


def _get_woosh_ae(device: str, dtype_str: str):
    """Lazy-load + cache the Woosh-AE. Reloads if device or dtype changed."""
    import torch
    from woosh_ae import AudioAutoEncoder, LoadConfig

    dtype = {"fp32": torch.float32, "fp16": torch.float16}[dtype_str]
    target_device = torch.device(device)

    cached = _WOOSH_AE_SINGLETON
    if (cached["model"] is not None
            and cached["device"] == target_device
            and cached["dtype"] == dtype):
        return cached["model"]

    ckpt_dir = ensure_woosh_ae()
    logger.info(f"Loading Woosh-AE from {ckpt_dir} on {target_device} ({dtype})")
    ae = AudioAutoEncoder(LoadConfig(ckpt_dir))
    ae = ae.to(target_device).to(dtype).eval()
    ae.requires_grad_(False)

    cached.update(model=ae, device=target_device, dtype=dtype)
    return ae


class FoleyTuneWooshVAERoundTrip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["fp32", "fp16"], {"default": "fp32"}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "round_trip"
    CATEGORY = "FoleyTune/Utils"
    DESCRIPTION = "Encode→decode audio through Woosh-AE (VOCOS, 128-d, 48 kHz). For A/B comparison against DAC-VAE."

    def round_trip(self, audio, device, dtype):
        import torch
        wave, _, orig_len = _prep_audio(audio)
        orig_len_48k = wave.shape[-1]

        # Woosh minimum frame is hop_length (480 samples); pad safely to 4x that.
        wave, _ = _pad_to_min(wave, 1920)

        ae = _get_woosh_ae(device, dtype)
        torch_dtype = {"fp32": torch.float32, "fp16": torch.float16}[dtype]
        x = wave.to(ae.device_any() if hasattr(ae, 'device_any') else next(ae.parameters()).device).to(torch_dtype)

        with torch.no_grad():
            z = ae.forward(x)
            y = ae.inverse(z)

        y = y.float()
        return (_finalize(y, orig_len_48k),)
```

**Step 2: Verify node class contract without running the model**

```bash
python -c "
import sys, types
sys.path.insert(0, '/media/p5/ComfyUI-HunyuanVideo-Foley')
fp = types.ModuleType('folder_paths'); fp.models_dir = '/tmp'; sys.modules['folder_paths'] = fp
comfy = types.ModuleType('comfy'); mm = types.ModuleType('comfy.model_management'); comfy.model_management = mm
sys.modules['comfy'] = comfy; sys.modules['comfy.model_management'] = mm
from nodes_vae import FoleyTuneWooshVAERoundTrip as N
assert N.INPUT_TYPES()['required']['audio'] == ('AUDIO',)
assert N.RETURN_TYPES == ('AUDIO',)
assert N.CATEGORY == 'FoleyTune/Utils'
print('Woosh node contract OK')
"
```

Expected: `Woosh node contract OK`.

**Step 3: Commit**

```bash
git add nodes_vae.py
git commit -m "feat: add Woosh VAE round-trip node"
```

---

## Task 5: DAC VAE Round-Trip node

**Files:**
- Modify: `nodes_vae.py`

**Step 1: Write the node**

Append to `nodes_vae.py`:

```python
# --- DAC node ----------------------------------------------------------------

_DAC_SINGLETON = {"model": None, "device": None}


def _get_dac(device: str):
    """Lazy-load + cache DAC-VAE matching HV-Foley's config."""
    import torch
    from hunyuanvideo_foley.models.dac_vae.model.dac import DAC
    from huggingface_hub import hf_hub_download

    target_device = torch.device(device)
    cached = _DAC_SINGLETON
    if cached["model"] is not None and cached["device"] == target_device:
        return cached["model"]

    # Match the existing auto-download pattern in nodes.py
    weights_path = os.path.join(folder_paths.models_dir, "foley", "vae_128d_48k.pth")
    if not os.path.exists(weights_path):
        logger.info("DAC-VAE weights not found. Downloading from Tencent/HunyuanVideo-Foley")
        weights_path = hf_hub_download(
            repo_id="Tencent/HunyuanVideo-Foley",
            filename="vae_128d_48k.pth",
            local_dir=os.path.join(folder_paths.models_dir, "foley"),
        )

    dac = DAC.load(weights_path).to(target_device).eval()
    dac.requires_grad_(False)

    cached.update(model=dac, device=target_device)
    return dac


class FoleyTuneDACVAERoundTrip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "round_trip"
    CATEGORY = "FoleyTune/Utils"
    DESCRIPTION = "Encode→decode audio through HunyuanVideo-Foley's DAC-VAE (128-d, 48 kHz). For A/B comparison against Woosh-AE."

    def round_trip(self, audio, device):
        import torch
        wave, _, _ = _prep_audio(audio)
        orig_len_48k = wave.shape[-1]

        # DAC stride is 640 samples at 48k; pad to 4x that.
        wave, _ = _pad_to_min(wave, 2560)

        dac = _get_dac(device)
        x = wave.to(next(dac.parameters()).device).float()

        with torch.no_grad():
            z = dac.encode(x)[0]      # DAC.encode returns (z, codes, latents) — [0] is continuous z
            y = dac.decode(z)

        return (_finalize(y, orig_len_48k),)
```

> **Note:** Verify the return tuple of `DAC.encode()` matches the shape used here. Inspect `hunyuanvideo_foley/models/dac_vae/model/dac.py` for the exact return signature. If `encode` returns a single tensor (not a tuple), drop the `[0]`.

**Step 2: Verify the DAC encode return signature**

```bash
grep -n "def encode" /media/p5/ComfyUI-HunyuanVideo-Foley/hunyuanvideo_foley/models/dac_vae/model/dac.py
```

Read the method body. Adjust the `z = dac.encode(x)[0]` line to match the actual return type. Record what it returns in a code comment.

**Step 3: Verify node contract**

```bash
python -c "
import sys, types
sys.path.insert(0, '/media/p5/ComfyUI-HunyuanVideo-Foley')
fp = types.ModuleType('folder_paths'); fp.models_dir = '/tmp'; sys.modules['folder_paths'] = fp
comfy = types.ModuleType('comfy'); mm = types.ModuleType('comfy.model_management'); comfy.model_management = mm
sys.modules['comfy'] = comfy; sys.modules['comfy.model_management'] = mm
from nodes_vae import FoleyTuneDACVAERoundTrip as N
assert N.INPUT_TYPES()['required']['audio'] == ('AUDIO',)
assert N.RETURN_TYPES == ('AUDIO',)
print('DAC node contract OK')
"
```

Expected: `DAC node contract OK`.

**Step 4: Commit**

```bash
git add nodes_vae.py
git commit -m "feat: add DAC VAE round-trip node"
```

---

## Task 6: Register nodes and end-to-end verify

**Files:**
- Modify: `nodes_vae.py` (add NODE_CLASS_MAPPINGS at bottom)
- Modify: `__init__.py`

**Step 1: Add mappings to `nodes_vae.py`**

Append:

```python
# --- Registration ------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "FoleyTuneDACVAERoundTrip": FoleyTuneDACVAERoundTrip,
    "FoleyTuneWooshVAERoundTrip": FoleyTuneWooshVAERoundTrip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoleyTuneDACVAERoundTrip": "FoleyTune DAC VAE Round-Trip",
    "FoleyTuneWooshVAERoundTrip": "FoleyTune Woosh VAE Round-Trip",
}
```

**Step 2: Wire into `__init__.py`**

Edit `__init__.py` to import and merge, mirroring the existing pattern:

```python
from .nodes_vae import (
    NODE_CLASS_MAPPINGS as VAE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as VAE_NODE_DISPLAY_NAME_MAPPINGS,
)
```

And in the merge block:

```python
NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **LORA_NODE_CLASS_MAPPINGS, **DATASET_NODE_CLASS_MAPPINGS, **VAE_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **LORA_NODE_DISPLAY_NAME_MAPPINGS, **DATASET_NODE_DISPLAY_NAME_MAPPINGS, **VAE_NODE_DISPLAY_NAME_MAPPINGS}
```

**Step 3: Verify node discovery**

```bash
cd /media/p5/ComfyUI-HunyuanVideo-Foley
python -c "
import sys
sys.path.insert(0, '.')
# Need to satisfy ComfyUI's modules:
import types
fp = types.ModuleType('folder_paths')
fp.models_dir = '/tmp'
fp.folder_names_and_paths = {}
fp.supported_pt_extensions = {'.pth', '.safetensors'}
fp.get_filename_list = lambda x: []
sys.modules['folder_paths'] = fp
comfy = types.ModuleType('comfy'); mm = types.ModuleType('comfy.model_management'); comfy.model_management = mm
sys.modules['comfy'] = comfy; sys.modules['comfy.model_management'] = mm
comfy_utils = types.ModuleType('comfy.utils'); comfy_utils.load_torch_file = lambda *a, **k: None
sys.modules['comfy.utils'] = comfy_utils

from __init__ import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
assert 'FoleyTuneDACVAERoundTrip' in NODE_CLASS_MAPPINGS
assert 'FoleyTuneWooshVAERoundTrip' in NODE_CLASS_MAPPINGS
assert NODE_DISPLAY_NAME_MAPPINGS['FoleyTuneDACVAERoundTrip'] == 'FoleyTune DAC VAE Round-Trip'
assert NODE_DISPLAY_NAME_MAPPINGS['FoleyTuneWooshVAERoundTrip'] == 'FoleyTune Woosh VAE Round-Trip'
print('registration OK')
"
```

Expected: `registration OK`.

**Step 4: Manual end-to-end test in ComfyUI (user-driven)**

Stop here and ask the user to:
1. Restart ComfyUI so it picks up the new branch
2. Drag a short audio file into a workflow (3–10 s)
3. Wire it into both `FoleyTune DAC VAE Round-Trip` and `FoleyTune Woosh VAE Round-Trip`
4. Hook each output to a PreviewAudio node
5. Queue once (first run will auto-download Woosh-AE — 400–900 MB)
6. Compare the two outputs by ear

Expected user observations:
- Both outputs should be audibly close to input
- Woosh-AE may preserve transients and high-frequency detail better
- Any glitches/clipping suggests a bug in pad/trim or a dtype issue — report back

**Step 5: Commit**

```bash
git add nodes_vae.py __init__.py
git commit -m "feat: register DAC + Woosh VAE round-trip nodes"
```

---

## Out of scope (deferred)

- Latent tensor output socket (waiting on HF dataset project scope)
- Numerical reconstruction metrics node (SI-SDR, MelDist between two AUDIO inputs)
- Integration tests (no pytest infra in repo; smoke checks per task are sufficient for this branch)
- Commercial-use license audit for Woosh-AE weights (CC-BY-NC) — relevant only if a final shipping decision is made to depend on Woosh

## Follow-ups if ceiling clearly favors Woosh-AE

- Evaluate training a Woosh-VFlow-compatible LoRA (separate multi-week effort, separate plan)
- Consider adding Woosh-AE as an alternative encoder in the Feature Extractor for the open HF dataset
