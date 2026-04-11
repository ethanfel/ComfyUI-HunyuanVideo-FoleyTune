# FoleyVideoQualityFilter — Design

## Goal

Add a ComfyUI node that quality-filters **video clips** by analyzing their audio track only, without loading any video frames or image tensors. Optionally copies passing clips to a filtered output folder.

## Node: FoleyVideoQualityFilter

Located in `nodes_dataset.py`, reuses existing `_bandwidth_score` and `_spectral_quality_score` functions.

### Audio Extraction

- FFmpeg subprocess: `-vn -f wav pipe:1` — demuxes audio to WAV at native sample rate/channels, no resampling, no video decoding
- Pipe output loaded via `soundfile.read()` from `io.BytesIO`
- If FFmpeg fails on a file (corrupt, no audio track), log and skip

### Folder Scanning

- Accepts `video_folder` path
- Scans for video extensions: `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.flv`
- Checks 1 level of subfolders (not full recursive)
- Sorts files for deterministic ordering

### Quality Scoring

Same as existing `FoleyDatasetQualityFilter`:
- **Bandwidth score** (0-1): spectral rolloff mapping 4-16 kHz
- **Spectral quality score** (0-1): average of flatness, temporal variance, HF energy
- **CLAP similarity** (0-1, optional): text-audio alignment via CLAP model
- **Composite**: weighted sum with configurable weights, normalized

### File Copy (Optional)

- `output_folder` is optional — empty string means inspect-only
- When set, copies passing video files via `shutil.copy2` (preserves metadata)
- Preserves subfolder structure: `subfolder/clip.mp4` → `output_folder/subfolder/clip.mp4`
- Creates output directories as needed

### Inputs

| Input | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `video_folder` | STRING | yes | `""` | Source folder path |
| `min_quality_score` | FLOAT | yes | `0.3` | Minimum composite score |
| `skip_rejected` | BOOLEAN | yes | `True` | Only copy clean clips |
| `output_folder` | STRING | optional | `""` | Empty = inspect only |
| `clap_prompt` | STRING | optional | `""` | Empty = skip CLAP |
| `min_bandwidth_score` | FLOAT | optional | `0.3` | Bandwidth sub-threshold |
| `min_spectral_score` | FLOAT | optional | `0.2` | Spectral sub-threshold |
| `min_clap_score` | FLOAT | optional | `0.1` | CLAP sub-threshold |
| `weight_bandwidth` | FLOAT | optional | `0.4` | Composite weight |
| `weight_spectral` | FLOAT | optional | `0.4` | Composite weight |
| `weight_clap` | FLOAT | optional | `0.2` | Composite weight |

### Output

- `report` (STRING) — per-clip scores + PASS/REJECT status + summary

### No FOLEY_AUDIO_DATASET Output

Deliberately omitted — the point is to avoid holding waveforms in memory. The node processes one clip at a time, scores it, optionally copies, then discards the audio data.
