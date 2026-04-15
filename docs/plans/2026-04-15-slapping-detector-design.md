# Slapping Detector Design

**Date:** 2026-04-15
**Status:** Approved

## Goal

Auto-detect rhythmic skin slapping in audio clips and tag prompts with
"with rhythmic slapping" so the model learns when this sound is present
vs absent. Unlike voice descriptors (per-source), slapping detection is
per-clip since it can start/stop mid-scene.

## Approach

Spectral flux onset detection in the 2-8kHz band (where percussive
transients are loudest, vocals are weaker). Count onsets per second and
measure inter-onset regularity. If rhythmic percussive pattern detected,
tag the clip.

## Integration

Built into the existing `FoleyTuneVoiceTagger` node. Runs per-clip in
the tagging loop (not per-source sampling). Cheap — just FFT + peak
picking, no new dependencies (uses scipy already in requirements).

### New helper: `detect_slapping(waveform, sr, min_onset_rate)`

In `voice_analysis.py`:

1. Bandpass filter 2-8kHz (scipy butterworth)
2. Compute STFT magnitude, then spectral flux (frame-to-frame diff)
3. Peak-pick onsets above adaptive threshold (median + 2*std)
4. Compute onset rate (onsets/sec)
5. If >= 3 onsets: compute inter-onset interval (IOI) std relative to mean
6. Return `{"detected": bool, "onset_rate": float, "regularity": float}`
   - detected = onset_rate >= min_onset_rate AND IOI coefficient of
     variation < 0.5 (reasonably regular)

### New optional inputs on VoiceTagger

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| detect_slapping | BOOLEAN | True | Run percussive content detection per clip |
| min_onset_rate | FLOAT | 2.0 | Minimum onsets/sec to consider as slapping |

### Tagging behavior

In the per-clip tagging loop (after voice descriptor is applied):
- Run `detect_slapping` on each clip
- If detected, append ", with rhythmic slapping" to the prompt
- Report shows per-clip: onset rate, regularity, detected yes/no

### Example output

```
clip_051_00: "breathy warm soprano voice, wet sounds, with rhythmic slapping"
clip_051_01: "breathy warm soprano voice, wet sounds"
clip_051_02: "breathy warm soprano voice, wet sounds, with rhythmic slapping"
```

### Dependencies

None new — uses scipy (already in requirements).
