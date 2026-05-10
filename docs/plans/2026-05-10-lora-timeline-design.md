# LoRA Timeline Visualizer — Design

## Goal

A ComfyUI timeline widget that lets users assign different LoRA adapters to specific time segments of a video, with per-segment strength control and smooth transitions. Enables multi-position foley generation from a single long video.

## Node Architecture

### FoleyTuneLoRATimelineEntry (stacker node)

Configures a single LoRA for use in the timeline. Chainable via `prev_entries` input.

**Inputs:**
- `lora_name` (dropdown from loras folder) OR `lora_path` (string)
- `strength` (FLOAT, 0-2, default 1.0)
- `label` (STRING — display name on timeline)
- `color` (dropdown: red/blue/green/yellow/purple/orange)
- `prev_entries` (optional, LORA_TIMELINE_ENTRIES — for chaining)

**Output:** `entries` (LORA_TIMELINE_ENTRIES) — list of `{path, strength, label, color}`

### FoleyTuneLoRATimeline (visual timeline node)

Displays a video timeline with draggable LoRA segments.

**Inputs:**
- `features` (FOLEYTUNE_FEATURES — provides video path + duration)
- `entries` (LORA_TIMELINE_ENTRIES)

**Outputs:**
- `lora_schedule` (LORA_SCHEDULE)
- `features` (FOLEYTUNE_FEATURES — passthrough)

**Widget:** DOM-based timeline with three layers:
1. Video thumbnail strip (frames at ~2fps, generated server-side via ffmpeg)
2. Time ruler (seconds)
3. Segment track (colored draggable bars)

### FoleyTuneChunkedSampler (existing, modified)

New optional input: `lora_schedule` (LORA_SCHEDULE).

## Data Types

### LORA_TIMELINE_ENTRIES
```python
[
    {"path": str, "strength": float, "label": str, "color": str},
    ...
]
```

### LORA_SCHEDULE
```python
[
    {
        "lora_path": str,
        "start_sec": float,
        "end_sec": float,
        "strength": float,
        "fade_in": float,   # seconds, 0 = hard cut
        "fade_out": float,  # seconds, 0 = hard cut
    },
    ...
]
```
Segments sorted by `start_sec`, non-overlapping. Gaps = base model (no LoRA).

## Timeline Widget (Frontend)

### Layout
- Thumbnail strip: video frames as horizontal filmstrip
- Ruler: time markers in seconds
- Segment track: colored bars for each LoRA assignment

### Interactions
- Click empty space → dropdown to assign a LoRA entry to a new segment
- Drag segment edges → resize (snap to 0.5s)
- Drag segment body → move along timeline
- Click segment → select, show strength slider
- Right-click → delete segment
- No overlapping segments — they push each other

### Strength control
- Default: constant strength from the entry node
- Optional fade-in/fade-out handles at segment edges (linear ramp)

### Thumbnail generation
- Backend endpoint `/foleytune/timeline_thumbnails`
- Takes video path + duration, returns sprite sheet via ffmpeg
- Cached per video

### Serialization
Widget state (segment positions) saved with the workflow. Schedule recomputed from widget state + entries on execution.

## LoRA Hot-Swapping in Sampler

### Strategy
Keep a clean base model state. Before each chunk, check the schedule and swap LoRA if needed.

```
base_state = deep_copy(model.state_dict())

for each chunk:
    target_lora = schedule.get_lora_at(chunk_center_time)
    if target_lora != current_lora:
        model.load_state_dict(base_state)      # restore clean
        if target_lora is not None:
            apply_lora(model, target_lora)       # apply new
        current_lora = target_lora
    denoise_chunk(...)
```

### Chunk-to-LoRA mapping
- Each chunk's center time `(start + end) / 2` determines which LoRA is active
- Chunk center in a gap → base model (no LoRA)
- Chunk boundaries adjusted to prefer alignment with segment boundaries, letting existing crossfade (SaFa/latent/waveform) handle transitions

### Memory
Storing `base_state` doubles model memory. Acceptable on the target GPU (RTX PRO 6000 Blackwell, 48GB). Alternative: store LoRA deltas and subtract/add — fragile, not worth the complexity.

## Data Flow

```
[Entry A] ──┐
[Entry B] ──┤── entries ──→ [Timeline] ──→ lora_schedule ──→ [Chunked Sampler]
[Entry C] ──┘                   ↑                                    ↓
                          features (video)                      [AUDIO out]
                                ↑
                      [Feature Extractor]
```
