# FoleyDatasetQualityFilter — Design

## Motivation

Aggressive audio denoising (MelBand-RoFormer) strips spectral information that
generative models need: room tone, transient detail, inter-harmonic energy.
Research papers (FALL-E, HunyuanVideo-Foley, MultiFoley, GenAU) prefer
**quality filtering** over aggressive cleaning — reject bad clips rather than
rescue them.

This node adds research-backed quality scoring to the dataset pipeline,
sitting after `FoleyDatasetInspector` (basic signal checks) and before
`FoleyDatasetSaver`.

## Quality Criteria

Three sub-scores, each normalized to 0–1:

### 1. Effective Bandwidth (weight 0.4)
Detects bandwidth-limited or upsampled clips. Computes spectral rolloff at 85%
energy and maps to 0–1: rolloff ≥ 16 kHz → 1.0, ≤ 4 kHz → 0.0, linear between.
Default minimum: 0.3 (~7.5 kHz). Based on HunyuanVideo-Foley's 32 kHz bandwidth
requirement.

### 2. Spectral Quality (weight 0.4)
Detects over-processed, degraded, or unnaturally clean audio. Average of three
sub-metrics min-max scaled to 0–1:
- **Spectral flatness**: higher = more natural broadband content (not over-cleaned)
- **Temporal variance**: higher = dynamic audio (not static/dead)
- **HF energy ratio**: presence of high-frequency content

Default minimum: 0.2. Uses existing `lora/spectral_metrics.py` computation.

### 3. CLAP Text-Audio Similarity (weight 0.2, optional)
Cosine similarity between CLAP audio embedding and text embedding. Filters
mismatched pairs. Default minimum: 0.1 (per GenAU paper). Skipped when no
prompt is provided; remaining weights renormalize.

## Composite Score

Weighted average of active sub-scores. Clips rejected if:
- Composite score < `min_quality_score` (default 0.3), OR
- Any individual sub-score below its per-criterion minimum

## Node Interface

```
Position: after FoleyDatasetInspector, before FoleyDatasetSaver

Inputs (required):
  dataset:            FOLEY_AUDIO_DATASET
  min_quality_score:  FLOAT  (default 0.3, range 0.0–1.0)
  skip_rejected:      BOOLEAN (default True)

Inputs (optional):
  clap_prompt:        STRING (default "")  — empty = skip CLAP
  min_bandwidth_score: FLOAT (default 0.3)
  min_spectral_score:  FLOAT (default 0.2)
  min_clap_score:      FLOAT (default 0.1)
  weight_bandwidth:    FLOAT (default 0.4)
  weight_spectral:     FLOAT (default 0.4)
  weight_clap:         FLOAT (default 0.2)

Outputs:
  dataset: FOLEY_AUDIO_DATASET (filtered)
  report:  STRING
```

## Report Format

```
=== Quality Filter Report ===
clip_001: BW=0.85 SQ=0.72 CLAP=--   SCORE=0.78 [PASS]
clip_002: BW=0.12 SQ=0.45 CLAP=--   SCORE=0.29 [REJECT: below 0.30]
clip_003: BW=0.91 SQ=0.08 CLAP=--   SCORE=0.49 [REJECT: spectral < 0.20]
---
Passed: 45/47 | Rejected: 2 | Avg score: 0.71
```

## Implementation Notes

- No new dependencies: uses torch STFT for bandwidth, existing
  `spectral_metrics()` for spectral quality, existing CLAP for similarity
- CLAP loaded lazily only when `clap_prompt` is non-empty
- All computation on CPU (dataset clips are already CPU tensors)
- Inserted in `nodes_dataset.py` between Inspector and HfSmoother sections
