# Video Loader & Combiner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace VHS dependency in the inference pipeline with custom nodes that decode video frames at target resolution via ffmpeg, extract features in-place (never holding full-res tensors), and remux audio without re-encoding video.

**Architecture:** A shared `_extract_video_features()` helper in `nodes.py` runs two ffmpeg decode passes (512x512 @8fps for SigLIP2, 224x224 @25fps for Synchformer), feeding frames in small batches through the existing model encoders. Two loader nodes (path-based and upload-based) call this helper and return `FOLEYTUNE_VIDEO_FEATURES`. A combiner node remuxes via `ffmpeg -c:v copy`. The existing `FoleyTuneFeatureExtractor` in `nodes_lora.py` gets an optional `video_features` input to skip image extraction. A JS frontend file handles upload widget and video preview for all three nodes.

**Tech Stack:** Python (subprocess/ffmpeg, torch), JavaScript (ComfyUI extension API), ffprobe/ffmpeg CLI.

---

### Task 1: Add `_ffmpeg_decode_frames()` helper function

**Files:**
- Modify: `nodes.py` (add after imports, before first class ~line 105)

**Step 1: Write the helper function**

Add a utility function that decodes video frames at a target resolution and fps via ffmpeg subprocess, yielding batches of numpy arrays. This avoids loading the entire video into RAM.

```python
import subprocess
import shutil

_VIDEO_EXTENSIONS = {'webm', 'mp4', 'mkv', 'gif', 'mov', 'avi', 'flv', 'wmv'}

def _ffprobe_video_info(video_path: str) -> dict:
    """Get video duration, fps, width, height via ffprobe."""
    cmd = [
        shutil.which("ffprobe") or "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    import json as _json
    info = _json.loads(result.stdout)
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    fps_parts = vs.get("r_frame_rate", "25/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
    duration = float(info["format"].get("duration", 0))
    return {
        "fps": fps,
        "duration": duration,
        "width": int(vs["width"]),
        "height": int(vs["height"]),
    }


def _ffmpeg_decode_frames(video_path: str, target_size: int, target_fps: float,
                          start_time: float = 0.0, duration: float = 0.0) -> np.ndarray:
    """Decode video frames at target resolution/fps via ffmpeg pipe.
    
    Returns [T, C, H, W] float32 tensor normalized to [-1, 1].
    """
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if start_time > 0:
        cmd += ["-ss", str(start_time)]
    cmd += ["-i", str(video_path)]
    if duration > 0:
        cmd += ["-t", str(duration)]
    cmd += [
        "-vf", f"scale={target_size}:{target_size}:flags=bicubic,fps={target_fps}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {result.stderr.decode()}")
    
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    frame_size = target_size * target_size * 3
    n_frames = len(raw) // frame_size
    if n_frames == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")
    frames = raw[:n_frames * frame_size].reshape(n_frames, target_size, target_size, 3)
    # [T, H, W, C] uint8 -> [T, C, H, W] float32 normalized [-1, 1]
    tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
    tensor = (tensor - 0.5) / 0.5  # normalize to [-1, 1] matching preprocess transforms
    return tensor
```

**Why normalize manually instead of using `deps.siglip2_preprocess` / `deps.syncformer_preprocess`:**
Both preprocessing pipelines do: Resize → ToTensor (scales 0-255→0-1) → Normalize(mean=0.5, std=0.5) → maps to [-1,1]. Since ffmpeg already handles the resize, we just need the normalization step. This avoids per-frame Python-level transform overhead.

**Note on Synchformer:** The syncformer preprocess uses `v2.Resize(224)` + `v2.CenterCrop(224)` (resize shortest edge to 224 then center-crop). Since we're forcing both dimensions to 224 with ffmpeg's `scale=224:224`, this handles non-square videos by stretching rather than cropping. This is fine for feature extraction — the sync model is robust to minor aspect ratio differences. If exact behavior matching is needed later, use `scale=224:-1,crop=224:224` in the ffmpeg filter chain.

**Step 2: Verify ffmpeg is available**

Run: `python -c "import shutil; print(shutil.which('ffmpeg')); print(shutil.which('ffprobe'))"`
Expected: paths to both binaries (already available since VHS depends on them)

**Step 3: Commit**

```bash
git add nodes.py
git commit -m "feat: add ffmpeg frame decode helper for video loader nodes"
```

---

### Task 2: Add `_extract_video_features()` shared extraction logic

**Files:**
- Modify: `nodes.py` (add after `_ffmpeg_decode_frames`, before first class)

**Step 1: Write the shared feature extraction function**

This function is called by both loader variants. It decodes frames and runs SigLIP2 + Synchformer encoding.

```python
def _extract_video_features(video_path: str, hunyuan_deps, start_time: float = 0.0,
                            duration: float = 0.0) -> dict:
    """Decode video and extract SigLIP2 + Synchformer features without full-res tensors."""
    from hunyuanvideo_foley.utils.feature_utils import (
        encode_video_with_siglip2, encode_video_with_sync,
    )
    
    device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    
    info = _ffprobe_video_info(video_path)
    if duration <= 0:
        duration = info["duration"] - start_time
    
    # SigLIP2: decode at 512x512, 8fps
    siglip2_frames = _ffmpeg_decode_frames(video_path, 512, 8.0, start_time, duration)
    siglip2_batch = siglip2_frames.unsqueeze(0)  # [1, T, C, H, W]
    
    hunyuan_deps.siglip2_model.to(device)
    clip_feat = encode_video_with_siglip2(siglip2_batch.to(device), hunyuan_deps).cpu()
    del siglip2_frames, siglip2_batch
    hunyuan_deps.siglip2_model.to(offload_device)
    
    # Synchformer: decode at 224x224, 25fps
    sync_frames = _ffmpeg_decode_frames(video_path, 224, 25.0, start_time, duration)
    sync_batch = sync_frames.unsqueeze(0)  # [1, T, C, H, W]
    
    hunyuan_deps.syncformer_model.to(device)
    sync_feat = encode_video_with_sync(sync_batch.to(device), hunyuan_deps).cpu()
    del sync_frames, sync_batch
    hunyuan_deps.syncformer_model.to(offload_device)
    
    torch.cuda.empty_cache()
    
    logger.info(f"Extracted features: clip={clip_feat.shape}, sync={sync_feat.shape}, "
                f"duration={duration:.2f}s from {video_path}")
    
    return {
        "clip_feat": clip_feat,      # [1, T_clip, 768]
        "sync_feat": sync_feat,      # [1, T_sync, 768]
        "video_path": str(video_path),
        "duration": duration,
        "fps": info["fps"],
    }
```

**Step 2: Commit**

```bash
git add nodes.py
git commit -m "feat: add shared video feature extraction function"
```

---

### Task 3: Add `FoleyTuneVideoLoader` node class (path-based)

**Files:**
- Modify: `nodes.py` (add new class before `NODE_CLASS_MAPPINGS`, ~line 1076)
- Modify: `nodes.py` (add to `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`)

**Step 1: Write the node class**

```python
class FoleyTuneVideoLoader:
    """Load video from file path and extract visual features via ffmpeg."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_deps": ("FOLEYTUNE_DEPS",),
                "video_path": ("STRING", {"default": "", "placeholder": "/path/to/video.mp4"}),
            },
            "optional": {
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.1,
                               "tooltip": "Start time in seconds"}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.1,
                             "tooltip": "Duration in seconds (0 = full video)"}),
            },
        }

    RETURN_TYPES = ("FOLEYTUNE_VIDEO_FEATURES", "FLOAT")
    RETURN_NAMES = ("video_features", "duration")
    FUNCTION = "load_video"
    CATEGORY = "FoleyTune"

    def load_video(self, hunyuan_deps, video_path, start_time=0.0, duration=0.0):
        if not video_path or not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        features = _extract_video_features(video_path, hunyuan_deps, start_time, duration)

        # Copy video to temp dir for inline preview
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        ext = os.path.splitext(video_path)[1] or ".mp4"
        temp_name = f"foleytune_preview_{os.path.basename(video_path)}"
        temp_path = os.path.join(temp_dir, temp_name)
        if not os.path.exists(temp_path):
            import shutil as _shutil
            _shutil.copy2(video_path, temp_path)

        return {"ui": {"gifs": [{"filename": temp_name, "subfolder": "", "type": "temp",
                                  "format": f"video/{ext.lstrip('.')}"}]},
                "result": (features, features["duration"])}

    @classmethod
    def IS_CHANGED(cls, video_path, **kwargs):
        if not video_path or not os.path.isfile(video_path):
            return ""
        return os.path.getmtime(video_path)
```

**Step 2: Register in NODE_CLASS_MAPPINGS**

Add to `NODE_CLASS_MAPPINGS`:
```python
"FoleyTuneVideoLoader": FoleyTuneVideoLoader,
```

Add to `NODE_DISPLAY_NAME_MAPPINGS`:
```python
"FoleyTuneVideoLoader": "FoleyTune Video Loader",
```

**Step 3: Set `OUTPUT_NODE = True`** on the class so the UI return works.

**Step 4: Commit**

```bash
git add nodes.py
git commit -m "feat: add FoleyTuneVideoLoader node (path-based)"
```

---

### Task 4: Add `FoleyTuneVideoLoaderUpload` node class (combo/upload-based)

**Files:**
- Modify: `nodes.py` (add new class after `FoleyTuneVideoLoader`)
- Modify: `nodes.py` (add to mappings)

**Step 1: Write the upload variant node class**

```python
class FoleyTuneVideoLoaderUpload:
    """Load video from ComfyUI input directory with upload support."""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in sorted(os.listdir(input_dir)):
            if os.path.isfile(os.path.join(input_dir, f)):
                ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
                if ext in _VIDEO_EXTENSIONS:
                    files.append(f)
        return {
            "required": {
                "hunyuan_deps": ("FOLEYTUNE_DEPS",),
                "video": (files,),
            },
            "optional": {
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.1}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.1,
                             "tooltip": "Duration in seconds (0 = full video)"}),
            },
        }

    RETURN_TYPES = ("FOLEYTUNE_VIDEO_FEATURES", "FLOAT")
    RETURN_NAMES = ("video_features", "duration")
    FUNCTION = "load_video"
    CATEGORY = "FoleyTune"
    OUTPUT_NODE = True

    def load_video(self, hunyuan_deps, video, start_time=0.0, duration=0.0):
        video_path = folder_paths.get_annotated_filepath(video)
        features = _extract_video_features(video_path, hunyuan_deps, start_time, duration)

        # Copy to temp dir for preview
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        ext = os.path.splitext(video)[1] or ".mp4"
        temp_name = f"foleytune_preview_{video}"
        temp_path = os.path.join(temp_dir, temp_name)
        if not os.path.exists(temp_path):
            import shutil as _shutil
            _shutil.copy2(video_path, temp_path)

        return {"ui": {"gifs": [{"filename": temp_name, "subfolder": "", "type": "temp",
                                  "format": f"video/{ext.lstrip('.')}"}]},
                "result": (features, features["duration"])}

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        image_path = folder_paths.get_annotated_filepath(video)
        return os.path.getmtime(image_path)

    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True
```

**Step 2: Register in mappings**

```python
"FoleyTuneVideoLoaderUpload": FoleyTuneVideoLoaderUpload,
```
```python
"FoleyTuneVideoLoaderUpload": "FoleyTune Video Loader (Upload)",
```

**Step 3: Commit**

```bash
git add nodes.py
git commit -m "feat: add FoleyTuneVideoLoaderUpload node (combo dropdown + upload)"
```

---

### Task 5: Add `FoleyTuneVideoCombiner` node class

**Files:**
- Modify: `nodes.py` (add new class after upload loader)
- Modify: `nodes.py` (add to mappings)

**Step 1: Write the combiner node class**

```python
class FoleyTuneVideoCombiner:
    """Mux generated audio onto source video without re-encoding video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_features": ("FOLEYTUNE_VIDEO_FEATURES",),
                "audio": ("AUDIO",),
                "output_path": ("STRING", {"default": "", "placeholder": "/path/to/output.mp4"}),
            },
            "optional": {
                "audio_codec": (["aac", "flac", "pcm_s16le"], {"default": "aac"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "combine"
    CATEGORY = "FoleyTune"
    OUTPUT_NODE = True

    def combine(self, video_features, audio, output_path, audio_codec="aac"):
        import tempfile
        import soundfile as sf
        
        source_video = video_features["video_path"]
        if not os.path.isfile(source_video):
            raise FileNotFoundError(f"Source video not found: {source_video}")
        
        if not output_path:
            # Default: same directory as source with _foley suffix
            base, ext = os.path.splitext(source_video)
            output_path = f"{base}_foley{ext}"
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Write audio to temp WAV
        waveform = audio["waveform"].squeeze(0).cpu().numpy()  # [channels, samples]
        sample_rate = audio["sample_rate"]
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            sf.write(tmp_wav, waveform.T, sample_rate)
            
            ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source_video),
                "-i", tmp_wav,
                "-c:v", "copy",
                "-c:a", audio_codec,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg mux failed: {result.stderr.decode()}")
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)
        
        logger.info(f"Muxed audio onto video: {output_path}")
        
        # Copy to temp for preview
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        temp_name = f"foleytune_combined_{os.path.basename(output_path)}"
        temp_path = os.path.join(temp_dir, temp_name)
        import shutil as _shutil
        _shutil.copy2(output_path, temp_path)
        
        ext = os.path.splitext(output_path)[1] or ".mp4"
        return {"ui": {"gifs": [{"filename": temp_name, "subfolder": "", "type": "temp",
                                  "format": f"video/{ext.lstrip('.')}"}]},
                "result": (str(output_path),)}
```

**Step 2: Register in mappings**

```python
"FoleyTuneVideoCombiner": FoleyTuneVideoCombiner,
```
```python
"FoleyTuneVideoCombiner": "FoleyTune Video Combiner",
```

**Step 3: Commit**

```bash
git add nodes.py
git commit -m "feat: add FoleyTuneVideoCombiner node (copy-mux, no re-encode)"
```

---

### Task 6: Modify `FoleyTuneFeatureExtractor` to accept `video_features`

**Files:**
- Modify: `nodes_lora.py:260-389` (`FoleyTuneFeatureExtractor` class)

**Step 1: Modify `INPUT_TYPES` — move `image` to optional, add `video_features`**

Change the INPUT_TYPES from:
```python
"required": {
    "hunyuan_deps": ("FOLEYTUNE_DEPS",),
    "image": ("IMAGE",),
    ...
}
```
to:
```python
"required": {
    "hunyuan_deps": ("FOLEYTUNE_DEPS",),
    "prompt": ("STRING", {"default": "", "multiline": True}),
    "negative_prompt": ("STRING", {"default": "", "multiline": True}),
    "cache_dir": ("STRING", {"default": ""}),
    "name": ("STRING", {"default": "clip", ...}),
},
"optional": {
    "image": ("IMAGE",),
    "video_features": ("FOLEYTUNE_VIDEO_FEATURES",),
    "frame_rate": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0, "step": 0.1}),
    "duration": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 3600.0, "step": 0.1, ...}),
},
```

**Step 2: Modify `extract_features` method signature and add early return path**

Update the method to accept `video_features=None` and `image=None` as optional kwargs:

```python
def extract_features(self, hunyuan_deps, prompt, negative_prompt,
                     cache_dir, name, image=None, video_features=None,
                     frame_rate=25.0, duration=8.0):
```

At the top of the method body, add the `video_features` shortcut path:

```python
    if video_features is not None:
        clip_features = video_features["clip_feat"]
        sync_features = video_features["sync_feat"]
        duration = video_features["duration"]
        frame_rate = video_features["fps"]
    elif image is not None:
        # ... existing image extraction code ...
    else:
        raise ValueError("Either 'image' or 'video_features' must be provided")
```

When `video_features` is provided, skip directly to the CLAP text encoding section (lines 341-362), then the npz save and return.

**Step 3: Commit**

```bash
git add nodes_lora.py
git commit -m "feat: FoleyTuneFeatureExtractor accepts video_features as alternate input"
```

---

### Task 7: Add `WEB_DIRECTORY` and create JS frontend file

**Files:**
- Modify: `__init__.py` (add `WEB_DIRECTORY`)
- Create: `web/js/FoleyTuneVideo.js`

**Step 1: Add `WEB_DIRECTORY` to `__init__.py`**

Add after the existing `__all__` line:
```python
WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
```

**Step 2: Create `web/js/FoleyTuneVideo.js`**

This JS file provides:
1. Video preview widget for all three node types (both loaders + combiner)
2. Upload button + drag-drop for the upload variant

The JS follows the VHS pattern closely:

```javascript
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const VIDEO_EXTENSIONS = ["webm", "mp4", "mkv", "gif", "mov", "avi"];

function addVideoPreview(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);

        const node = this;
        const videoEl = document.createElement("video");
        videoEl.controls = true;
        videoEl.loop = true;
        videoEl.muted = true;
        videoEl.style.width = "100%";
        videoEl.style.display = "none";

        const previewWidget = this.addDOMWidget("video_preview", "preview", videoEl, {
            serialize: false,
            hideOnZoom: false,
        });

        node._ftVideoPreview = { videoEl, previewWidget };

        // Update preview after execution
        const onExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            onExecuted?.apply(this, arguments);
            if (output?.gifs?.[0]) {
                const g = output.gifs[0];
                const params = new URLSearchParams({
                    filename: g.filename,
                    type: g.type || "temp",
                    subfolder: g.subfolder || "",
                });
                videoEl.src = api.apiURL("/view?" + params.toString());
                videoEl.style.display = "block";
            }
        };
    };
}

function addUploadWidget(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);

        const node = this;
        const pathWidget = this.widgets.find((w) => w.name === "video");
        if (!pathWidget) return;

        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = VIDEO_EXTENSIONS.map((e) => "video/" + e).join(",");
        fileInput.style.display = "none";
        document.body.appendChild(fileInput);

        fileInput.onchange = async () => {
            if (!fileInput.files.length) return;
            const file = fileInput.files[0];
            const body = new FormData();
            body.append("image", file);
            body.append("overwrite", "true");
            const resp = await api.fetchApi("/upload/image", { method: "POST", body });
            if (resp.ok) {
                const data = await resp.json();
                if (!pathWidget.options.values.includes(data.name)) {
                    pathWidget.options.values.push(data.name);
                }
                pathWidget.value = data.name;
                pathWidget.callback?.(data.name);
            }
        };

        const uploadWidget = this.addWidget("button", "choose video to upload", null, () => {
            fileInput.click();
        });
        uploadWidget.serialize = false;

        // Drag-drop support
        this.onDragOver = (e) => !!e?.dataTransfer?.types?.includes?.("Files");
        this.onDragDrop = async (e) => {
            const file = e?.dataTransfer?.files?.[0];
            if (!file) return false;
            const ext = file.name.split(".").pop()?.toLowerCase();
            if (!VIDEO_EXTENSIONS.includes(ext)) return false;
            const body = new FormData();
            body.append("image", file);
            body.append("overwrite", "true");
            const resp = await api.fetchApi("/upload/image", { method: "POST", body });
            if (resp.ok) {
                const data = await resp.json();
                if (!pathWidget.options.values.includes(data.name)) {
                    pathWidget.options.values.push(data.name);
                }
                pathWidget.value = data.name;
                pathWidget.callback?.(data.name);
            }
            return true;
        };

        // Preview on combo selection change
        const origCallback = pathWidget.callback;
        pathWidget.callback = function (value) {
            origCallback?.apply(this, arguments);
            if (!value) return;
            const preview = node._ftVideoPreview;
            if (preview) {
                const params = new URLSearchParams({
                    filename: value,
                    type: "input",
                    subfolder: "",
                });
                preview.videoEl.src = api.apiURL("/view?" + params.toString());
                preview.videoEl.style.display = "block";
            }
        };
    };
}

app.registerExtension({
    name: "FoleyTune.VideoNodes",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === "FoleyTuneVideoLoader") {
            addVideoPreview(nodeType);
        }
        if (nodeData?.name === "FoleyTuneVideoLoaderUpload") {
            addVideoPreview(nodeType);
            addUploadWidget(nodeType);
        }
        if (nodeData?.name === "FoleyTuneVideoCombiner") {
            addVideoPreview(nodeType);
        }
    },
});
```

**Step 3: Commit**

```bash
git add __init__.py web/js/FoleyTuneVideo.js
git commit -m "feat: add JS frontend for video loader upload widget and preview"
```

---

### Task 8: Manual integration test

**Step 1: Restart ComfyUI and verify nodes appear**

Run: Restart the ComfyUI server.
Expected: Log shows "FoleyTune Video Loader", "FoleyTune Video Loader (Upload)", and "FoleyTune Video Combiner" registered. No import errors.

**Step 2: Test FoleyTuneVideoLoader (path-based)**

1. Add `FoleyTuneVideoLoader` node to workflow
2. Set `video_path` to a test video file
3. Connect `hunyuan_deps` from `FoleyTuneDependenciesLoader`
4. Connect `video_features` output to `FoleyTuneFeatureExtractor`
5. Set a prompt on the feature extractor
6. Run the workflow
7. Verify: features are extracted, no full-res tensor in RAM, preview shows in node

**Step 3: Test FoleyTuneVideoLoaderUpload (combo/upload)**

1. Add `FoleyTuneVideoLoaderUpload` node
2. Upload a video via the upload button
3. Verify: video appears in combo dropdown, preview updates on selection
4. Run workflow end-to-end through sampler
5. Verify: audio output is correct

**Step 4: Test FoleyTuneVideoCombiner**

1. Connect `video_features` from loader to combiner
2. Connect `audio` from sampler to combiner
3. Set an output path
4. Run workflow
5. Verify: output video has original video stream + generated audio, preview shows in node
6. Verify: output video is not re-encoded (compare file size / codec info with source)

**Step 5: Test backward compatibility**

1. Load existing workflow using VHS + `FoleyTuneFeatureExtractor` with `image` input
2. Run workflow
3. Verify: still works identically (no regression)

**Step 6: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration test fixes for video loader/combiner nodes"
```
