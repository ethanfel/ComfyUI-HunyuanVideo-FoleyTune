# FoleyTune Video Loader & Combiner — Design

## Problem

The current inference pipeline routes video through VHS nodes: `VHS_LoadVideo` decodes all frames into a full-resolution `IMAGE` tensor `[T, H, W, 3]` in RAM, which `FoleyTuneFeatureExtractor` then resizes down to 512x512 and 224x224 for feature extraction. For a 30fps 1080p 8-second video this means ~1.8 GB of float32 image data sitting in RAM that's immediately discarded after downscaling. The combiner (`VHS_VideoCombine`) then re-encodes the video stream even though no frames were modified.

## Solution

Three changes:

1. **`FoleyTuneVideoLoader`** — decodes video via ffmpeg directly at target resolutions (512x512 for SigLIP2, 224x224 for Synchformer), extracts visual/sync features in-place without ever allocating the full-res tensor. Outputs `FOLEYTUNE_VIDEO_FEATURES`.

2. **`FoleyTuneVideoLoaderUpload`** — same as above but with a combo dropdown + upload button (like VHS's "Load Video (Upload)") instead of a raw file path input.

3. **`FoleyTuneVideoCombiner`** — takes `FOLEYTUNE_VIDEO_FEATURES` (carries source path) + `AUDIO`, remuxes via `ffmpeg -c:v copy` (no re-encoding), outputs to user-specified path with inline preview.

4. **`FoleyTuneFeatureExtractor` modification** — add optional `video_features` input. When provided, skip image→feature extraction and only run CLAP text encoding.

## Data Types

### `FOLEYTUNE_VIDEO_FEATURES` (dict)

```python
{
    "clip_feat": Tensor,     # [1, T_clip, 768] — SigLIP2 features at 8fps
    "sync_feat": Tensor,     # [1, T_sync, 768] — Synchformer features at 25fps
    "video_path": str,       # original file path (passed through for combiner)
    "duration": float,       # seconds
    "fps": float,            # source video fps
}
```

## Node 1: `FoleyTuneVideoLoader`

**File:** `nodes.py`

### INPUT_TYPES

```python
{
    "required": {
        "hunyuan_deps": ("FOLEYTUNE_DEPS",),
        "video_path": ("STRING", {"default": ""}),
    },
    "optional": {
        "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
        "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1,
                     "tooltip": "0 = full video"}),
    },
}
```

### RETURN_TYPES

```python
RETURN_TYPES = ("FOLEYTUNE_VIDEO_FEATURES", "FLOAT")
RETURN_NAMES = ("video_features", "duration")
```

### Processing

1. Probe video with ffprobe to get fps, duration, resolution.
2. Compute trim window from `start_time` / `duration`.
3. Run ffmpeg subprocess to decode frames:
   - SigLIP2 stream: `-vf scale=512:512` at 8fps, decode in batches, feed to `encode_video_with_siglip2()`.
   - Synchformer stream: `-vf scale=224:224` at 25fps, decode in batches, feed to `encode_video_with_sync()`.
   - Frames are read from ffmpeg's stdout pipe as raw RGB, processed in small batches (e.g. 32 frames), and discarded immediately — never stored as a full tensor.
4. Return `FOLEYTUNE_VIDEO_FEATURES` dict + duration float.

### UI Preview

The node returns UI metadata for inline video preview, using ComfyUI's temp file mechanism. A small JS extension (`web/js/FoleyTuneVideo.js`) adds the preview DOM widget identical to VHS's pattern.

## Node 2: `FoleyTuneVideoLoaderUpload`

Same processing as `FoleyTuneVideoLoader`, but `video_path` is replaced with:

```python
"video": (sorted(video_files_in_input_dir),)
```

A combo dropdown populated from `folder_paths.get_input_directory()`. The JS extension adds an upload button + drag-drop handler that posts to ComfyUI's `/upload/image` endpoint (same as VHS).

Both nodes share a common `_load_and_extract()` method.

## Node 3: `FoleyTuneVideoCombiner`

**File:** `nodes.py`

### INPUT_TYPES

```python
{
    "required": {
        "video_features": ("FOLEYTUNE_VIDEO_FEATURES",),
        "audio": ("AUDIO",),
        "output_path": ("STRING", {"default": ""}),
    },
    "optional": {
        "audio_codec": (["aac", "flac", "pcm_s16le"], {"default": "aac"}),
    },
}
```

### RETURN_TYPES

```python
RETURN_TYPES = ("STRING",)
RETURN_NAMES = ("output_path",)
OUTPUT_NODE = True
```

### Processing

1. Extract `video_path` from `video_features` dict.
2. Write audio waveform to a temp WAV file.
3. Run `ffmpeg -i <source_video> -i <temp_audio.wav> -c:v copy -c:a <codec> -map 0:v:0 -map 1:a:0 <output_path>`.
4. Return output path + UI preview metadata.

### UI Preview

Same JS preview widget as the loader nodes, showing the output video inline.

## Node 4: `FoleyTuneFeatureExtractor` Modification

Add optional input:

```python
"optional": {
    "video_features": ("FOLEYTUNE_VIDEO_FEATURES",),
}
```

When `video_features` is provided:
- Use `clip_feat` and `sync_feat` from it directly.
- Use its `duration` value.
- Skip SigLIP2/Synchformer extraction entirely.
- Still run CLAP text encoding on prompt/negative_prompt.
- Make `image` input optional (move to optional dict).

When `video_features` is not provided, behave exactly as before (requires `image`).

## Frontend: `web/js/FoleyTuneVideo.js`

Single JS file handling both loader variants and the combiner:

1. **Upload widget** (upload variant only): hidden file input + button, posts to `/upload/image`, updates combo dropdown.
2. **Video preview widget**: `<video>` DOM element, sources from ComfyUI's `/view` endpoint for temp files. Updated on combo selection change or after execution.
3. **Registration**: `app.registerExtension()` hooking `beforeRegisterNodeDef` for the three node names.

## Workflow

```
FoleyTuneVideoLoader(Upload) ──→ video_features ──→ FoleyTuneFeatureExtractor (+ prompt) ──→ features ──→ Sampler ──→ AUDIO
                               ↘                                                                                        ↓
                                └── video_features ──────────────────────────────────────→ FoleyTuneVideoCombiner ←──────┘
```

## RAM Savings Estimate

For 8s 1080p@30fps video:
- **Before:** VHS loads 240 frames × 1920×1080×3 × 4 bytes = ~5.9 GB float32
- **After:** Peak = one batch of 32 frames × 512×512×3 × 4 bytes = ~100 MB (SigLIP2 batch), plus final feature tensors ~0.5 MB
- **Savings:** ~98% reduction in peak image RAM
