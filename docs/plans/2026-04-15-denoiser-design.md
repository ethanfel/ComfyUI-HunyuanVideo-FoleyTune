# Dataset Denoiser Design

**Date:** 2026-04-15
**Status:** Approved

## Goal

Remove stationary background noise (AC hum, fans) from training audio clips
before feature extraction, so the LoRA doesn't learn to reproduce it.

## Problem

Many training clips have varying levels of AC/room noise. The model learns
this noise as part of the target audio, baking it into generated output.
Spectral gating can remove stationary noise while preserving vocal
characteristics.

## Approach

Use `noisereduce` library (spectral gating). It auto-estimates the noise
profile from each clip and gates it out. Works well on stationary noise.
Lightweight, no ML models needed, adjustable strength.

## Node: FoleyTuneDatasetDenoiser

**Category:** `FoleyTune`
**Function:** `denoise`
**Return types:** `FOLEYTUNE_AUDIO_DATASET`, `STRING`

### Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| dataset | FOLEYTUNE_AUDIO_DATASET | required | Audio clips to denoise |
| strength | FLOAT | 0.7 | Noise reduction strength (0.0-1.0). Maps to `prop_decrease` |
| stationary | BOOLEAN | True | Assume noise is stationary (AC, fans) |
| n_fft | INT | 2048 | FFT window size. Larger = better freq resolution, slower |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| dataset | FOLEYTUNE_AUDIO_DATASET | Denoised dataset |
| report | STRING | Per-clip noise reduction stats |

### Algorithm

1. For each clip, extract mono numpy from `[1, C, L]` waveform using
   `waveform_to_mono_numpy` helper.
2. Call `noisereduce.reduce_noise(y=wav, sr=sr, prop_decrease=strength,
   stationary=stationary, n_fft=n_fft)`.
3. Preserve RMS level — match output RMS to input RMS, clip peaks > 1.0.
4. Shallow-copy item dict, update waveform key.
5. Report: clip name, noise reduction in dB.

### Pipeline Position

After Resampler/LUFS/Compressor, before VoiceTagger or BatchFeatureExtractor:

```
DatasetLoader -> Denoiser -> VoiceTagger -> BatchFeatureExtractor -> DatasetSaver
```

### Dependencies

- `noisereduce` (add to requirements.txt)
