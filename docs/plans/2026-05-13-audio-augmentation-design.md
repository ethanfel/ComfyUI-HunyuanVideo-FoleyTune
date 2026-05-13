# Audio Augmentation Enhancements — Design

**Date:** 2026-05-13
**Status:** Approved

## Goal

Enhance the existing `FoleyTuneDatasetAugmenter` node with speed perturbation and time shift augmentations to expand small foley training datasets without requiring more source material.

## Context

The augmenter already supports gain variation, pitch shift (audiomentations), and time stretch (audiomentations). Two natural augmentation types are missing:

1. **Speed perturbation** — changes both pitch and duration together, like playing audio at a different speed. More natural than phase-vocoder pitch shift for impact/mechanical/foley sounds.
2. **Time shift** — small random offset relative to video features, teaching the model tolerance to slight A/V misalignment.

Both apply at dataset creation time. Augmented clips are saved as separate FLAC files, inheriting the original clip's NPZ features (visual + sync + text conditioning are unchanged). DAC encoding of the augmented audio happens at training time in `prepare_dataset()`.

## Changes

### New parameters on `FoleyTuneDatasetAugmenter`

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `speed_range` | FLOAT | 0.0 | 0.0–0.3 | Random speed change ±fraction. 0.1 = 90%–110% speed. 0 = disabled. |
| `time_shift_ms` | FLOAT | 0.0 | 0.0–200.0 | Random time shift ±ms. 0 = disabled. |

### Speed perturbation implementation

Resample trick using soxr (already a dependency):
```python
speed_factor = rng.uniform(1.0 - speed_range, 1.0 + speed_range)
# Pretend audio was recorded at sr * speed_factor, resample to sr
wav_np = soxr.resample(wav_np, int(sr * speed_factor), sr, quality="VHQ")
```

This naturally shifts both pitch and duration. After resampling, trim or zero-pad to original length to maintain feature alignment.

### Time shift implementation

Zero-padded shift (not circular, to avoid discontinuities):
```python
shift_samples = int(rng.uniform(-time_shift_ms, time_shift_ms) * sr / 1000)
if shift_samples > 0:  # delay: prepend silence, truncate end
    wav = F.pad(wav[:, :, :-shift_samples], (shift_samples, 0))
elif shift_samples < 0:  # advance: truncate start, append silence
    wav = F.pad(wav[:, :, -shift_samples:], (0, -shift_samples))
```

### Augmentation stacking order

Per variant: gain → speed perturbation → time shift → (optional pitch shift) → (optional time stretch) → peak normalize.

Speed perturbation and audiomentations pitch shift are independent controls. Speed perturbation is the recommended choice for foley; pitch shift is better for voice content where formant preservation matters.

## What stays unchanged

- Existing `pitch_range_semitones`, `time_stretch_range`, `gain_range_db`, `keep_originals`, `seed` parameters
- Node position in pipeline chain (after LUFS normalization, before inspector/saver)
- Feature inheritance: augmented clips keep the original's NPZ features
- Peak normalization after all transforms
- Naming convention: `{name}_aug{nn}`
