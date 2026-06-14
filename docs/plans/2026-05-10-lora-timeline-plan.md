# LoRA Timeline Visualizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a ComfyUI timeline widget that assigns different LoRA adapters to specific time segments of a video, with per-segment strength control and transitions via the existing chunked sampler crossfade.

**Architecture:** Three components — a stacker node (FoleyTuneLoRATimelineEntry) for configuring LoRAs, a timeline node (FoleyTuneLoRATimeline) with a video thumbnail timeline widget, and a modification to the existing FoleyTuneChunkedSampler to accept and apply a LoRA schedule during chunked generation. The sampler stores a clean base model state and hot-swaps LoRAs per chunk.

**Tech Stack:** Python (ComfyUI nodes), JavaScript (LiteGraph DOM widget), ffmpeg (thumbnail sprite generation)

---

### Task 1: FoleyTuneLoRATimelineEntry — Backend Node

**Files:**
- Modify: `nodes_lora.py` (add class after FoleyTuneLoRALoaderPath, ~line 1530)
- Modify: `nodes_lora.py:2589-2611` (NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS)

**Step 1: Write the node class**

```python
class FoleyTuneLoRATimelineEntry:
    """Configure a LoRA for placement on the timeline. Chain multiple entries."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "label": ("STRING", {"default": "LoRA"}),
                "color": (["red", "blue", "green", "yellow", "purple", "orange"],),
            },
            "optional": {
                "prev_entries": ("LORA_TIMELINE_ENTRIES",),
            },
        }

    RETURN_TYPES = ("LORA_TIMELINE_ENTRIES",)
    RETURN_NAMES = ("entries",)
    FUNCTION = "add_entry"
    CATEGORY = "FoleyTune"

    def add_entry(self, lora_name, strength, label, color, prev_entries=None):
        entries = list(prev_entries) if prev_entries else []
        adapter_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        entries.append({
            "path": adapter_path,
            "strength": strength,
            "label": label,
            "color": color,
        })
        return (entries,)
```

**Step 2: Register in NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS**

Add to `nodes_lora.py:2589`:
```python
"FoleyTuneLoRATimelineEntry": FoleyTuneLoRATimelineEntry,
```
Add to `nodes_lora.py:2601`:
```python
"FoleyTuneLoRATimelineEntry": "FoleyTune LoRA Timeline Entry",
```

**Step 3: Test manually**

Launch ComfyUI, verify the node appears in the menu under FoleyTune, chain two entries and confirm output is a list of two dicts.

**Step 4: Commit**

```bash
git add nodes_lora.py
git commit -m "feat: add FoleyTuneLoRATimelineEntry stacker node"
```

---

### Task 2: Thumbnail Sprite API Endpoint

**Files:**
- Modify: `nodes_lora.py` (add API route registration, near top after imports)
- Or create: `api_routes.py` if preferred for separation

**Step 1: Add the sprite generation endpoint**

Register a ComfyUI server route that takes a video path and returns a JPEG sprite sheet of thumbnail frames.

```python
from server import PromptServer
import hashlib

@PromptServer.instance.routes.get("/foleytune/timeline_thumbnails")
async def timeline_thumbnails(request):
    """Generate a sprite sheet of video thumbnails for the timeline widget."""
    import aiohttp.web as web
    video_path = request.query.get("video_path", "")
    if not video_path or not os.path.exists(video_path):
        return web.Response(status=404, text="Video not found")

    # Cache key based on path + mtime
    mtime = os.path.getmtime(video_path)
    cache_key = hashlib.md5(f"{video_path}:{mtime}".encode()).hexdigest()
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "thumbnails")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_key}.jpg")

    if not os.path.exists(cache_path):
        # Generate sprite: 2fps, 160px wide thumbnails, tiled horizontally
        import subprocess
        result = subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vf", "fps=2,scale=160:-1,tile=0x1",
            "-frames:v", "1",
            "-q:v", "5",
            cache_path,
        ], capture_output=True, timeout=30)
        if result.returncode != 0 or not os.path.exists(cache_path):
            return web.Response(status=500, text=f"ffmpeg failed: {result.stderr.decode()}")

    return web.FileResponse(cache_path, headers={"Content-Type": "image/jpeg"})
```

Note: `tile=0x1` tells ffmpeg to tile all frames in one horizontal row. The `0` means "auto-calculate columns to fit all frames". This produces a single-row sprite sheet.

**Step 2: Test with curl**

```bash
curl -o /tmp/sprite.jpg "http://localhost:8188/foleytune/timeline_thumbnails?video_path=/path/to/video.mp4"
```

Verify the JPEG is a horizontal strip of frames.

**Step 3: Commit**

```bash
git add nodes_lora.py
git commit -m "feat: add timeline thumbnail sprite API endpoint"
```

---

### Task 3: FoleyTuneLoRATimeline — Backend Node

**Files:**
- Modify: `nodes_lora.py` (add class after FoleyTuneLoRATimelineEntry)
- Modify: `nodes_lora.py:2589-2611` (NODE_CLASS_MAPPINGS)

**Step 1: Write the node class**

```python
class FoleyTuneLoRATimeline:
    """Visual timeline for placing LoRA adapters on video segments."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "features": ("FOLEYTUNE_FEATURES",),
                "entries": ("LORA_TIMELINE_ENTRIES",),
                "segments_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "hidden": True,
                }),
            },
        }

    RETURN_TYPES = ("LORA_SCHEDULE", "FOLEYTUNE_FEATURES")
    RETURN_NAMES = ("lora_schedule", "features")
    FUNCTION = "build_schedule"
    CATEGORY = "FoleyTune"

    def build_schedule(self, features, entries, segments_json="[]"):
        import json
        segments = json.loads(segments_json)

        schedule = []
        for seg in sorted(segments, key=lambda s: s["start_sec"]):
            entry_idx = seg.get("entry_index", 0)
            if entry_idx < 0 or entry_idx >= len(entries):
                continue
            entry = entries[entry_idx]
            schedule.append({
                "lora_path": entry["path"],
                "start_sec": float(seg["start_sec"]),
                "end_sec": float(seg["end_sec"]),
                "strength": float(seg.get("strength", entry["strength"])),
                "fade_in": float(seg.get("fade_in", 0.0)),
                "fade_out": float(seg.get("fade_out", 0.0)),
            })

        return (schedule, features)
```

The `segments_json` is a hidden widget that the frontend JS writes to. It stores the user's segment placements as JSON.

**Step 2: Register in mappings**

```python
"FoleyTuneLoRATimeline": FoleyTuneLoRATimeline,
# display:
"FoleyTuneLoRATimeline": "FoleyTune LoRA Timeline",
```

**Step 3: Commit**

```bash
git add nodes_lora.py
git commit -m "feat: add FoleyTuneLoRATimeline backend node"
```

---

### Task 4: Timeline Widget — Frontend JS

**Files:**
- Create: `web/js/FoleyTuneTimeline.js`

This is the largest task. The JS file registers a ComfyUI extension that attaches a DOM widget to the FoleyTuneLoRATimeline node.

**Step 1: Write the timeline widget**

```javascript
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const COLORS = {
    red: "#e74c3c", blue: "#3498db", green: "#2ecc71",
    yellow: "#f1c40f", purple: "#9b59b6", orange: "#e67e22",
};

app.registerExtension({
    name: "FoleyTune.LoRATimeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "FoleyTuneLoRATimeline") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const node = this;

            // --- Container ---
            const container = document.createElement("div");
            container.style.cssText = "width:100%;position:relative;user-select:none;";

            // --- Thumbnail strip ---
            const thumbStrip = document.createElement("div");
            thumbStrip.style.cssText =
                "width:100%;height:60px;background:#111;overflow:hidden;position:relative;";
            const thumbImg = document.createElement("img");
            thumbImg.style.cssText = "height:100%;position:absolute;left:0;top:0;";
            thumbStrip.appendChild(thumbImg);
            container.appendChild(thumbStrip);

            // --- Ruler ---
            const ruler = document.createElement("canvas");
            ruler.style.cssText = "width:100%;height:20px;display:block;";
            ruler.height = 20;
            container.appendChild(ruler);

            // --- Segment track ---
            const track = document.createElement("div");
            track.style.cssText =
                "width:100%;height:40px;background:#1a1a2e;position:relative;overflow:hidden;border:1px solid #333;";
            container.appendChild(track);

            // --- State ---
            let duration = 0;
            let entries = [];
            let segments = []; // [{entry_index, start_sec, end_sec, strength}]
            let selectedIdx = -1;
            let dragState = null; // {type: "move"|"resize-left"|"resize-right", idx, startX, origStart, origEnd}

            const segmentsWidget = node.widgets.find(w => w.name === "segments_json");

            function secToX(sec) {
                return (sec / duration) * track.clientWidth;
            }
            function xToSec(x) {
                const sec = (x / track.clientWidth) * duration;
                return Math.round(sec * 2) / 2; // snap to 0.5s
            }

            function renderSegments() {
                track.querySelectorAll(".lora-seg").forEach(el => el.remove());
                segments.forEach((seg, i) => {
                    const el = document.createElement("div");
                    el.className = "lora-seg";
                    const entry = entries[seg.entry_index] || {};
                    const color = COLORS[entry.color] || COLORS.blue;
                    const left = secToX(seg.start_sec);
                    const width = secToX(seg.end_sec) - left;
                    el.style.cssText =
                        `position:absolute;top:2px;bottom:2px;left:${left}px;width:${width}px;` +
                        `background:${color}88;border:2px solid ${color};border-radius:4px;` +
                        `cursor:grab;display:flex;align-items:center;justify-content:center;` +
                        `font-size:11px;color:#fff;text-shadow:0 1px 2px #000;overflow:hidden;white-space:nowrap;`;
                    el.textContent = entry.label || `LoRA ${seg.entry_index}`;
                    if (i === selectedIdx) el.style.outline = "2px solid #fff";

                    // Resize handles
                    const leftHandle = document.createElement("div");
                    leftHandle.style.cssText =
                        "position:absolute;left:0;top:0;bottom:0;width:6px;cursor:ew-resize;";
                    const rightHandle = document.createElement("div");
                    rightHandle.style.cssText =
                        "position:absolute;right:0;top:0;bottom:0;width:6px;cursor:ew-resize;";

                    leftHandle.addEventListener("mousedown", (e) => {
                        e.stopPropagation();
                        dragState = { type: "resize-left", idx: i, startX: e.clientX, origStart: seg.start_sec, origEnd: seg.end_sec };
                    });
                    rightHandle.addEventListener("mousedown", (e) => {
                        e.stopPropagation();
                        dragState = { type: "resize-right", idx: i, startX: e.clientX, origStart: seg.start_sec, origEnd: seg.end_sec };
                    });
                    el.addEventListener("mousedown", (e) => {
                        selectedIdx = i;
                        dragState = { type: "move", idx: i, startX: e.clientX, origStart: seg.start_sec, origEnd: seg.end_sec };
                        renderSegments();
                    });
                    el.addEventListener("contextmenu", (e) => {
                        e.preventDefault();
                        segments.splice(i, 1);
                        selectedIdx = -1;
                        syncWidget();
                        renderSegments();
                    });

                    el.appendChild(leftHandle);
                    el.appendChild(rightHandle);
                    track.appendChild(el);
                });
            }

            function renderRuler() {
                ruler.width = ruler.clientWidth;
                const ctx = ruler.getContext("2d");
                ctx.fillStyle = "#222";
                ctx.fillRect(0, 0, ruler.width, ruler.height);
                ctx.fillStyle = "#aaa";
                ctx.font = "10px monospace";
                const step = duration <= 30 ? 1 : duration <= 120 ? 5 : 10;
                for (let t = 0; t <= duration; t += step) {
                    const x = secToX(t);
                    ctx.fillRect(x, 0, 1, t % (step * 5) === 0 ? 14 : 8);
                    if (t % (step * 2) === 0) ctx.fillText(`${t}s`, x + 2, 18);
                }
            }

            function syncWidget() {
                if (segmentsWidget) {
                    segmentsWidget.value = JSON.stringify(segments);
                }
            }

            // --- Mouse events on track for adding new segments ---
            track.addEventListener("dblclick", (e) => {
                if (!entries.length) return;
                const rect = track.getBoundingClientRect();
                const sec = xToSec(e.clientX - rect.left);
                // Default 4s segment, pick first available entry
                const entryIdx = entries.length === 1 ? 0 :
                    parseInt(prompt(`Entry index (0-${entries.length - 1}):`, "0") || "0", 10);
                if (entryIdx < 0 || entryIdx >= entries.length) return;
                const startSec = Math.max(0, sec - 2);
                const endSec = Math.min(duration, sec + 2);
                segments.push({ entry_index: entryIdx, start_sec: startSec, end_sec: endSec, strength: entries[entryIdx].strength });
                segments.sort((a, b) => a.start_sec - b.start_sec);
                syncWidget();
                renderSegments();
            });

            // --- Global mouse move/up for dragging ---
            document.addEventListener("mousemove", (e) => {
                if (!dragState) return;
                const dx = e.clientX - dragState.startX;
                const dSec = (dx / track.clientWidth) * duration;
                const seg = segments[dragState.idx];

                if (dragState.type === "move") {
                    const len = dragState.origEnd - dragState.origStart;
                    let newStart = Math.round((dragState.origStart + dSec) * 2) / 2;
                    newStart = Math.max(0, Math.min(duration - len, newStart));
                    seg.start_sec = newStart;
                    seg.end_sec = newStart + len;
                } else if (dragState.type === "resize-left") {
                    seg.start_sec = Math.max(0, Math.min(seg.end_sec - 0.5,
                        Math.round((dragState.origStart + dSec) * 2) / 2));
                } else if (dragState.type === "resize-right") {
                    seg.end_sec = Math.max(seg.start_sec + 0.5, Math.min(duration,
                        Math.round((dragState.origEnd + dSec) * 2) / 2));
                }
                syncWidget();
                renderSegments();
            });
            document.addEventListener("mouseup", () => { dragState = null; });

            // --- DOM Widget ---
            const widget = this.addDOMWidget("timeline", "preview", container, {
                serialize: false,
                hideOnZoom: false,
                getValue() { return ""; },
                setValue() {},
            });
            widget.computeSize = function (width) {
                return [width, duration > 0 ? 130 : -4];
            };

            // --- Update from inputs ---
            const origOnExecuted = node.onExecuted;
            node.onExecuted = function (output) {
                origOnExecuted?.apply(this, arguments);

                // Read duration and video path from features
                if (output?.ui?.duration) duration = output.ui.duration;
                if (output?.ui?.video_path) {
                    const url = api.apiURL("/foleytune/timeline_thumbnails?video_path=" +
                        encodeURIComponent(output.ui.video_path));
                    thumbImg.src = url;
                }
                if (output?.ui?.entries) entries = output.ui.entries;

                renderRuler();
                renderSegments();
                node.setSize([node.size[0], node.computeSize([node.size[0], 0])[1]]);
                node?.graph?.setDirtyCanvas(true);
            };
        };
    },
});
```

**Step 2: Update the backend node to pass UI data**

Modify FoleyTuneLoRATimeline.build_schedule to return UI hints:

```python
def build_schedule(self, features, entries, segments_json="[]"):
    # ... (existing schedule building code) ...

    # Pass metadata to frontend widget
    return {
        "ui": {
            "duration": features["duration"],
            "video_path": features.get("video_path", ""),
            "entries": entries,
        },
        "result": (schedule, features),
    }
```

**Step 3: Test visually**

Load the node in ComfyUI. Connect a feature extractor and LoRA entries. Verify:
- Thumbnail strip loads from video
- Ruler shows time markers
- Double-click creates a segment
- Drag to move/resize
- Right-click to delete

**Step 4: Commit**

```bash
git add web/js/FoleyTuneTimeline.js nodes_lora.py
git commit -m "feat: add timeline widget frontend for LoRA segment placement"
```

---

### Task 5: LoRA Hot-Swap in Chunked Sampler

**Files:**
- Modify: `nodes.py:397-563` (FoleyTuneChunkedSampler)
- Modify: `utils.py:539+` (chunked_denoise_process)
- Modify: `lora/lora.py` (add remove_lora function)

**Step 1: Add remove_lora to lora/lora.py**

```python
def remove_lora(model: nn.Module) -> int:
    """Remove all LoRA wrappers, restoring original nn.Linear layers."""
    n_removed = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr_name = parts[0]
        setattr(parent, attr_name, module.base)
        n_removed += 1
    return n_removed
```

**Step 2: Add lora_schedule input to FoleyTuneChunkedSampler**

In `nodes.py`, add to INPUT_TYPES optional:
```python
"lora_schedule": ("LORA_SCHEDULE", {"tooltip": "LoRA timeline schedule — assigns different LoRAs to different time segments."}),
```

Pass it through to generate_audio and then to chunked_denoise_process.

**Step 3: Modify chunked_denoise_process to accept lora_schedule**

Add `lora_schedule=None` parameter. Before the chunk loop, save base state. Inside the chunk loop, before denoising each chunk, resolve which LoRA to apply:

```python
# At top of function, after model setup:
_current_lora_path = None
_base_state = None
if lora_schedule:
    _base_state = copy.deepcopy(model_dict.foley_model.state_dict())

# Inside chunk loop, before denoising:
if lora_schedule and _base_state is not None:
    chunk_center = (t_start + t_end) / 2
    target = None
    for seg in lora_schedule:
        if seg["start_sec"] <= chunk_center < seg["end_sec"]:
            target = seg
            break
    target_path = target["lora_path"] if target else None

    if target_path != _current_lora_path:
        # Restore base weights
        model_dict.foley_model.load_state_dict(_base_state, strict=False)
        remove_lora(model_dict.foley_model)

        if target is not None:
            # Load and apply the target LoRA
            ckpt = _load_adapter_checkpoint(target_path)
            sd = ckpt.get("state_dict", ckpt)
            meta = ckpt.get("meta", {})
            rank = meta.get("rank", 64)
            alpha = meta.get("alpha", float(rank))
            target_suffixes = FOLEY_TARGET_PRESETS.get(meta.get("target", "all_attn_mlp"))
            apply_lora(model_dict.foley_model, rank=rank, alpha=alpha, target_suffixes=target_suffixes)
            load_lora(model_dict.foley_model, sd)
            # Apply strength scaling
            strength = target.get("strength", 1.0)
            if strength != 1.0:
                for n, p in model_dict.foley_model.named_parameters():
                    if "lora_" in n:
                        p.data.mul_(strength)
            logger.info(f"Chunk [{t_start:.1f}-{t_end:.1f}]: LoRA={os.path.basename(target_path)} strength={strength}")
        else:
            logger.info(f"Chunk [{t_start:.1f}-{t_end:.1f}]: base model (no LoRA)")
        _current_lora_path = target_path
```

**Step 4: Test end-to-end**

Create a workflow:
1. Feature Extractor → video
2. Two LoRA Timeline Entries (different LoRAs)
3. LoRA Timeline → place each on different segments
4. Chunked Sampler with lora_schedule connected

Verify:
- Different chunks produce audio matching their assigned LoRA
- Base model chunks (gaps) produce generic audio
- Transitions via crossfade are smooth

**Step 5: Commit**

```bash
git add nodes.py utils.py lora/lora.py
git commit -m "feat: LoRA hot-swap in chunked sampler via lora_schedule"
```

---

### Task 6: Chunk Boundary Alignment

**Files:**
- Modify: `utils.py` (compute_chunk_boundaries or add wrapper)

**Step 1: Add schedule-aware chunk boundary function**

```python
def align_chunks_to_schedule(chunks, lora_schedule, min_chunk_sec=4.0):
    """Adjust chunk boundaries to prefer alignment with LoRA segment boundaries.

    When a segment boundary falls inside a chunk, split or shift the chunk
    so the boundary aligns with a chunk edge. This lets the crossfade
    handle the LoRA transition naturally.
    """
    if not lora_schedule:
        return chunks

    boundaries = set()
    for seg in lora_schedule:
        boundaries.add(seg["start_sec"])
        boundaries.add(seg["end_sec"])

    adjusted = []
    for (cs, ce) in chunks:
        splits = sorted(b for b in boundaries if cs + min_chunk_sec < b < ce - min_chunk_sec)
        if not splits:
            adjusted.append((cs, ce))
        else:
            prev = cs
            for b in splits:
                adjusted.append((prev, b))
                prev = b
            adjusted.append((prev, ce))

    return adjusted
```

**Step 2: Wire into chunked_denoise_process**

Call `align_chunks_to_schedule(chunks, lora_schedule)` after receiving chunks, before the denoising loop.

**Step 3: Test**

Place two LoRA segments with a boundary at 15s on a 30s video. Verify chunks split at 15s.

**Step 4: Commit**

```bash
git add utils.py
git commit -m "feat: align chunk boundaries to LoRA schedule segments"
```

---

### Task 7: Polish and Integration Test

**Files:**
- All modified files

**Step 1: End-to-end workflow test**

Build a complete workflow:
- Video Loader → Feature Extractor
- 2x LoRA Timeline Entry (stacked)
- LoRA Timeline (connect features + entries, place segments)
- Chunked Sampler (connect lora_schedule + features)

Verify:
- Widget renders correctly with thumbnails
- Segments serialize/deserialize on workflow save/load
- Generated audio switches LoRA at segment boundaries
- Crossfade transitions are smooth
- Base model (no LoRA) works for gaps

**Step 2: Edge cases**

- Single LoRA covering entire video (should behave like normal LoRA loading)
- All gaps (no segments — pure base model)
- Very short segments (< chunk duration)
- Overlapping with chunk boundaries

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: LoRA timeline visualizer — complete implementation"
```
