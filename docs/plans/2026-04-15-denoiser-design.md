# Dataset Denoiser Design

**Date:** 2026-04-15
**Status:** Approved

## Goal

Remove stationary background noise (AC hum, fans) from training audio clips
before quality scoring and feature extraction, so the LoRA doesn't learn to
reproduce it and noisy clips aren't incorrectly rejected by the quality filter.

## Problem

Many training clips have varying levels of AC/room noise. The model learns
this noise as part of the target audio, baking it into generated output.
Additionally, noise lowers spectral quality scores in the quality filter,
causing otherwise good clips to be rejected. Spectral gating can remove
stationary noise while preserving vocal characteristics.

## Approach

Use `noisereduce` library (spectral gating). A small settings node provides
the denoiser config; the quality filter accepts it as an optional input.
When connected, the filter denoises before scoring and passes denoised audio
downstream. When not connected, the filter works exactly as before.

## Architecture

Two components:

### Node 1: FoleyTuneDenoiserSettings

Small config node — just exposes the 3 knobs and outputs a typed settings
object. No audio processing here.

**Category:** `FoleyTune`
**Function:** `get_settings`
**Return types:** `FOLEYTUNE_DENOISE_SETTINGS`

#### Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| strength | FLOAT | 0.7 | Noise reduction strength (0.0-1.0). Maps to `prop_decrease` |
| stationary | BOOLEAN | True | Assume noise is stationary (AC, fans) |
| n_fft | INT | 2048 | FFT window size. Larger = better freq resolution, slower |

#### Output

Returns a dict: `{"strength": float, "stationary": bool, "n_fft": int}`

### Node 2: FoleyTuneDatasetQualityFilter (modified)

Add optional input `denoise_settings` of type `FOLEYTUNE_DENOISE_SETTINGS`.

When connected, before scoring each clip:

1. Group clips by source prefix (reuse `group_by_source` from voice_analysis).
2. Per source group, find the segment with lowest RMS — the quietest segment
   is the best proxy for "pure noise". Use it as `y_noise` for all segments
   of the same source.
3. Denoise each clip using `noisereduce.reduce_noise(y=wav, sr=sr,
   y_noise=noise_profile, ...)`.
4. Preserve RMS level — match output RMS to input RMS, clip peaks > 1.0.
5. Score on denoised audio.
6. Output denoised waveforms in the passed dataset.

### Pipeline Integration

```
FoleyTuneDenoiserSettings ──────┐
                                ↓ (optional)
DatasetLoader ──→ QualityFilter ──→ VoiceTagger ──→ BatchFeatureExtractor
```

When not connected, quality filter behaves identically to current behavior.

### Dependencies

- `noisereduce` (add to requirements.txt)
