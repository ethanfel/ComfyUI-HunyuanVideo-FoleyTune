# VAE Round-Trip Nodes — Design

**Date:** 2026-04-21
**Branch:** `feature/vae-roundtrip`
**Status:** design approved, pending implementation plan

## Problem

A LoRA trained against a VAE-based diffusion model cannot produce audio of higher fidelity than that VAE's own reconstruction ceiling. To compare HunyuanVideo-Foley's DAC-VAE against Sony Woosh-AE as alternative codecs for future training, we need a way to run the same clip through each VAE's encode→decode cycle and listen to the output.

No published head-to-head exists between DAC-VAE (at HV-Foley's 128-d, 48 kHz configuration) and Woosh-AE. Published numbers (Woosh-AE vs Stable Audio Open VAE) aren't directly useful for deciding whether a Woosh-AE-backed pipeline would be worth training.

## Goal

Two ComfyUI nodes, identical interface, that run a VAE round-trip on an AUDIO input. Drop the same clip into both in one workflow, A/B listen.

Explicitly out of scope: latent saving, metric computation, training infrastructure, V2A/T2A integration.

## Scope

- **FoleyTune DAC VAE Round-Trip** — uses existing `vae_128d_48k.pth` (HV-Foley's DAC).
- **FoleyTune Woosh VAE Round-Trip** — uses vendored Woosh-AE.

Both are pure audio→audio. No additional outputs.

## File layout

```
ComfyUI-HunyuanVideo-Foley/
├── nodes_vae.py                       # new — both round-trip nodes
├── hunyuanvideo_foley/                # existing DAC code, reused
└── woosh_ae/                          # new — vendored from SonyResearch/Woosh
    ├── __init__.py
    ├── audio_autoencoder.py
    ├── vocos/                         # ConvNeXt blocks, iSTFT head
    └── utils.py                       # LoadConfig
```

Only the AE subtree of Woosh is vendored. T2A/V2A/CLAP/flow-matching code is excluded. Upstream license (MIT) headers preserved; a `THIRD_PARTY_NOTICES.md` at repo root records the attribution.

## Weight distribution

- **DAC-VAE**: unchanged. `vae_128d_48k.pth` already auto-downloads from `Tencent/HunyuanVideo-Foley` via the `DOWNLOADABLE_MODELS` registry in `nodes.py:41`.
- **Woosh-AE**: shipped as a zip in Woosh v1.0.0 GitHub Release. On first run of the Woosh node, if `ComfyUI/models/foley/Woosh-AE/` is missing, a helper `ensure_woosh_ae()` downloads the release zip via `urllib`, extracts only the AE subfolder, deletes the zip. Size unknown precisely but ~400–900 MB expected given 221M params.

A checksum verification step runs after extraction. A clear one-time user-facing message logs the download start.

## Node contracts

Both nodes:

| Slot | Type | Notes |
|---|---|---|
| IN: `audio` | AUDIO | Standard ComfyUI format `{"waveform": [B,C,L], "sample_rate": int}` |
| IN: `device` | COMBO (cuda/cpu) | Default cuda |
| OUT: `audio` | AUDIO | Reconstructed, always 48 kHz |

Woosh node additionally exposes `dtype` (fp32/fp16). DAC node does not, because HV-Foley's DAC is fp32-only in the existing codepath.

Both nodes:
- Resample input to 48 kHz if it differs (logged warning).
- Downmix stereo to mono with one-time warning (both VAEs are single-channel).
- Pad short clips to VAE minimum frame size, trim output back to original length.
- Sit in `CATEGORY = "FoleyTune/Utils"`.

## Data flow

**DAC:**
```
audio (AUDIO)
  → resample 48k
  → downmix [B,1,L]
  → DAC.encode() → z
  → DAC.decode() → [B,1,L]
  → AUDIO (48 kHz)
```
Reuses existing `DAC.load(vae_128d_48k.pth)` pattern from `hunyuanvideo_foley/utils/model_utils.py:53-62`.

**Woosh:**
```
audio (AUDIO)
  → resample 48k
  → downmix [B,1,L]
  → ae.forward(x) → z
  → ae.inverse(z) → [B,1,L]
  → AUDIO (48 kHz)
```
Loaded via `AudioAutoEncoder(LoadConfig(<path>))` from vendored module. Cached on a module-level singleton (mirrors `MODEL_MANAGER` pattern from `model_utils.py`), released via `mm.soft_empty_cache()` at workflow end.

## Error handling

| Case | Behavior |
|---|---|
| Woosh weights missing | Auto-download + extract on first run; fail with clear path error if download fails |
| Corrupted Woosh zip | Redownload once; then surface error |
| Sample rate ≠ 48 kHz | Resample, log warning once per node instance |
| Stereo input | Downmix to mono, log warning once |
| Audio shorter than VAE min frame | Pad with silence to min frame, trim output to input length |
| CUDA OOM | Catch, retry on CPU with warning |
| fp16 decode produces NaN | Fall back to fp32 once, log warning |

## Testing plan

- **Manual A/B**: run a 5 s test clip and a 30 s test clip through both nodes; listen side by side. Expect DAC and Woosh-AE outputs to both be perceptually close to input, with Woosh-AE likely showing better transient preservation if papers are right.
- **Idempotence check**: run the same clip twice through the Woosh node; outputs should be bit-identical (no hidden RNG).
- **Shape round-trip**: confirm `input.shape == output.shape` after trim.
- **No unit tests** — both nodes are thin wrappers over upstream-tested code.

## Explicit non-goals

- No latent tensor output (YAGNI; revisit when HF dataset project needs it).
- No CLI/standalone script.
- No LoRA training against Woosh-AE.
- No reconstruction metrics (SI-SDR, MelDist) computed in-node. Manual ear A/B is the primary comparison.
- No V2A or T2A using Woosh backbone.
- No chunking across long audio — both VAEs are fully convolutional, arbitrary length works in one pass. If a user OOMs on a 10-minute clip, they clip it themselves.

## Follow-ups (deferred, not blocking)

- If ear A/B favors Woosh-AE clearly, evaluate training a LoRA against Woosh-AE — but that's a separate multi-week effort (different conditioning, different sampler, possibly different VFlow backbone).
- A "latent save" output socket on the Woosh node could feed the open HF dataset plan.
- Numerical metrics node (SI-SDR, MelDist between two AUDIO inputs) would formalize the A/B.
