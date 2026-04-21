# Top-N Per Folder for VideoQualityFilter — Design

## Problem

`FoleyTuneVideoQualityFilter` currently judges each clip independently against
`min_quality_score` and per-metric hard floors. When source videos contribute
wildly different clip counts (e.g. `vid_005` has 42 clips, `vid_012` has 5),
training distribution skews toward the dense sources. The model overfits to
their acoustics rather than learning the underlying sound.

A second skew is intra-source: a single segment with strong scores can take
every selection slot, losing the diversity provided by other segments cut from
the same source.

## Solution

Add an optional `top_n_per_folder` cap to the filter. Apply quality gates
first, then within each source folder run a **segment-aware round-robin** that
balances picks across segments before going deeper into any one segment.

## Filename structure assumed

`<folder>/<basename>_<segment><n>_<idx>.mp4`

- **Folder** = immediate parent directory under `video_folder` (e.g. `vid_005`)
- **Segment prefix** = filename stem with trailing `_<idx>` stripped
  (e.g. `clip_005_m1_3` → `clip_005_m1`)

This mirrors the prefix logic already used by `voice_analysis.group_by_source`
and the denoiser noise-profile path.

## New input

On `FoleyTuneVideoQualityFilter`, optional:

```
top_n_per_folder: INT, default 0, min 0, max 9999
```

`0` disables the cap (current behavior). `>0` enables segment-aware round-robin.

Lives on the main node — not on `FilterOptions` — because it is orthogonal to
scoring weights and should work with any profile.

## Selection logic (Phase 3 modification)

1. Score every clip and apply hard gates exactly as today
   (`min_quality_score`, `min_bandwidth`, `min_spectral`, `min_clap`,
   `max_negative_clap_score`). Clips that fail any gate are rejected with the
   existing reason strings.
2. If `top_n_per_folder == 0`: keep behavior unchanged.
3. Else, group survivors by their relative parent folder. For each folder:
   a. Group clips by segment prefix.
   b. Sort each segment's clips by composite score descending.
   c. Order segments by their *peak* composite score descending (so a folder
      with one excellent segment + several mediocre ones still favors the
      excellent one when ties occur).
   d. Round-robin pick: take the next-best clip from each segment in order
      until N clips are selected for the folder, or every segment is
      exhausted.
4. Clips not selected in step 3 become rejected with reason
   `"folder quota: kept top {N}"`.

## Edge cases

- Folder with fewer survivors than N → keep all of them.
- Single-segment folder → degenerates to plain top-N within that segment.
- Filename with no recognizable segment pattern → each clip becomes its own
  one-element segment (round-robin reduces to top-N).
- Clips at root level (no subfolder) → grouped under the empty-string folder
  bucket and treated as one logical folder.
- Tie on score → break by filename ascending for determinism.

## Report additions

When `top_n_per_folder > 0`, append a per-folder block to the text report:

```
=== Folder quota (top 8 per folder) ===
  vid_001:  14 scored,  11 above gate, kept  8  (4 segments)
  vid_002:  28 scored,  22 above gate, kept  8  (3 segments)
  vid_005:  42 scored,  38 above gate, kept  8  (5 segments)
  vid_012:   5 scored,   4 above gate, kept  4  (1 segment)
  ...
```

This makes it obvious at a glance whether the cap actually rebalanced the
dataset.

## What does NOT change

- Cache (`.quality_cache.json`) — scoring is unchanged, so cache hits remain
  valid across runs that toggle the new option.
- Denoising and CLAP pipeline — all upstream of the change.
- `FilterOptions` node — untouched.
- `skip_first` — applied earlier; still works.
- Val clip selection — still picks one random clip from the combined rejected
  pool (now including folder-quota rejects), which is fine.

## Implementation footprint

~30 LOC in `FoleyTuneVideoQualityFilter.filter_videos`, mostly inside Phase 3.
No new utility module needed; segment grouping uses an inline regex matching
the existing `voice_analysis.group_by_source` pattern.
