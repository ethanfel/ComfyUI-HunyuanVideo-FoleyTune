"""LoRA training nodes for FoleyTune."""

import os
import sys
import gc
import copy
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from loguru import logger

import folder_paths
import comfy.model_management as mm
import comfy.utils

from .lora.lora import (
    apply_lora, get_lora_state_dict, load_lora,
    FOLEY_TARGET_PRESETS, LoRALinear,
)
from .lora.train import (
    prepare_dataset, prepare_single_entry, sample_timesteps, flow_matching_loss,
    generate_eval_sample, save_checkpoint, save_meta_json,
)
from .lora.spectral_metrics import spectral_metrics, reference_metrics, clap_similarity
from PIL import Image, ImageDraw


def _load_adapter_checkpoint(path: str) -> dict:
    """Load a LoRA checkpoint from .safetensors or .pt format."""
    if path.endswith(".safetensors"):
        state_dict = load_safetensors(path)
        json_path = path.replace(".safetensors", ".json")
        meta = {}
        if os.path.exists(json_path):
            with open(json_path) as f:
                meta = json.load(f)
        return {"state_dict": state_dict, "meta": meta}
    return torch.load(path, map_location="cpu", weights_only=False)

FOLEYTUNE_AUDIO_DATASET = "FOLEYTUNE_AUDIO_DATASET"

_SPEC_N_FFT = 2048
_SPEC_HOP = 512
_SPEC_DB_FLOOR = -80.0
_SPEC_LOG_BINS = 256


def _save_spectrogram(wav_np, sr, path):
    """Save a log-frequency dB spectrogram PNG for an eval sample.

    wav_np: 1D numpy array (mono).
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    window = torch.hann_window(_SPEC_N_FFT)
    stft = torch.stft(torch.from_numpy(wav_np), n_fft=_SPEC_N_FFT, hop_length=_SPEC_HOP,
                      window=window, return_complex=True)
    mag = stft.abs().numpy()
    db = 20.0 * np.log10(np.maximum(mag, 1e-8))
    db = np.maximum(db, db.max() + _SPEC_DB_FLOOR).astype(np.float32)

    # Log-frequency resampling
    n_freqs = db.shape[0]
    src_idx = np.logspace(0, np.log10(max(n_freqs - 1, 2)), _SPEC_LOG_BINS)
    lo = np.floor(src_idx).astype(int).clip(0, n_freqs - 2)
    frac = (src_idx - lo)[:, None]
    spec = ((1 - frac) * db[lo] + frac * db[lo + 1]).astype(np.float32)
    spec = spec[::-1]  # low freq at bottom

    # Hz labels
    tgt_hz = [100, 500, 1000, 2000, 4000, 8000, 16000]
    tpos, tlbl = [], []
    for hz in tgt_hz:
        bin_f = hz * _SPEC_N_FFT / sr
        if bin_f < 1 or bin_f >= n_freqs:
            continue
        pos = int(np.searchsorted(src_idx, bin_f))
        tpos.append(_SPEC_LOG_BINS - 1 - min(pos, _SPEC_LOG_BINS - 1))
        tlbl.append(f"{hz // 1000}k" if hz >= 1000 else str(hz))

    vmin = float(np.percentile(spec, 2.0))
    vmax = float(np.percentile(spec, 99.5))

    fig = Figure(figsize=(12, 3), dpi=120, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(spec, aspect="auto", cmap="inferno", origin="upper",
                   vmin=vmin, vmax=vmax, interpolation="antialiased")
    ax.set_yticks(tpos)
    ax.set_yticklabels(tlbl, fontsize=8)
    ax.set_ylabel("Hz", fontsize=9)
    ax.set_xlabel("Time frames", fontsize=9)
    ax.set_title(Path(path).stem, fontsize=9)
    fig.colorbar(im, ax=ax, label="dB", fraction=0.02, pad=0.01)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    canvas.print_figure(str(Path(path).with_suffix(".png")), dpi=120)


def _save_wav(path, wav_tensor, sr):
    """Save audio tensor to WAV using soundfile (avoids torchcodec/FFmpeg dependency)."""
    import soundfile as sf
    # wav_tensor: [C, L] or [1, C, L]
    if wav_tensor.ndim == 3:
        wav_tensor = wav_tensor.squeeze(0)
    wav_np = wav_tensor.float().numpy().T  # [L, C]
    sf.write(str(path), wav_np, sr)


def _smooth_losses(losses, beta=0.9):
    """Exponential moving average smoothing."""
    smoothed, ema = [], None
    for v in losses:
        ema = v if ema is None else beta * ema + (1 - beta) * v
        smoothed.append(ema)
    return smoothed


def _pil_to_tensor(img):
    """Convert a PIL Image to a [1, H, W, 3] float32 IMAGE tensor for ComfyUI."""
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _draw_loss_curve(losses, log_interval=1, start_step=0, smoothed=None):
    """Render a loss curve as a PIL Image."""
    W, H = 800, 380
    pl, pr, pt, pb = 70, 20, 25, 45

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    pw = W - pl - pr
    ph = H - pt - pb

    if len(losses) >= 2:
        lo, hi = min(losses), max(losses)
        if hi == lo:
            hi = lo + 1e-6
        rng = hi - lo

        for i in range(5):
            y = pt + int(i * ph / 4)
            val = hi - i * rng / 4
            draw.line([(pl, y), (W - pr, y)], fill=(220, 220, 220), width=1)
            draw.text((2, y - 7), f"{val:.4f}", fill=(120, 120, 120))

        n = len(losses)
        pts = []
        for i, v in enumerate(losses):
            x = pl + int(i * pw / max(n - 1, 1))
            y = pt + int((1.0 - (v - lo) / rng) * ph)
            pts.append((x, y))
        draw.line(pts, fill=(200, 220, 255), width=1)

        if smoothed is not None and len(smoothed) >= 2:
            spts = []
            for i, v in enumerate(smoothed):
                x = pl + int(i * pw / max(n - 1, 1))
                y = pt + int((1.0 - (v - lo) / rng) * ph)
                spts.append((x, y))
            draw.line(spts, fill=(66, 133, 244), width=2)

        first_step = start_step + log_interval
        last_step = start_step + n * log_interval
        for i in range(5):
            x = pl + int(i * pw / 4)
            step = int(first_step + i * (last_step - first_step) / 4)
            draw.text((x - 12, H - pb + 5), str(step), fill=(120, 120, 120))

    draw.line([(pl, pt), (pl, H - pb)], fill=(40, 40, 40), width=1)
    draw.line([(pl, H - pb), (W - pr, H - pb)], fill=(40, 40, 40), width=1)
    draw.text((pl + 4, 5), "Training Loss", fill=(40, 40, 40))

    return img


_COMPARISON_PALETTE = [
    (66, 133, 244), (234, 67, 53), (52, 168, 83), (251, 188, 5),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (149, 165, 166),
]


def _draw_comparison_curves(experiments_data):
    """Draw all smoothed loss curves on the same axes, one color per experiment."""
    W, H = 900, 420
    pl, pr, pt, pb = 75, 160, 30, 50

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    pw = W - pl - pr
    ph = H - pt - pb

    series = []
    for i, ed in enumerate(experiments_data):
        lh = ed.get("loss_history") or []
        if len(lh) < 2:
            continue
        sm = _smooth_losses(lh)
        series.append({
            "id": ed["id"],
            "smoothed": sm,
            "color": _COMPARISON_PALETTE[i % len(_COMPARISON_PALETTE)],
        })

    if not series:
        draw.text((pl + 10, pt + 10), "No data to plot", fill=(80, 80, 80))
        return img

    all_vals = [v for s in series for v in s["smoothed"]]
    lo, hi = min(all_vals), max(all_vals)
    if hi == lo:
        hi = lo + 1e-6
    rng = hi - lo

    for i in range(5):
        y = pt + int(i * ph / 4)
        val = hi - i * rng / 4
        draw.line([(pl, y), (W - pr, y)], fill=(220, 220, 220), width=1)
        draw.text((2, y - 7), f"{val:.4f}", fill=(100, 100, 100))

    for s in series:
        n = len(s["smoothed"])
        pts = []
        for j, v in enumerate(s["smoothed"]):
            x = pl + int(j * pw / max(n - 1, 1))
            y = pt + int((1.0 - (v - lo) / rng) * ph)
            pts.append((x, y))
        draw.line(pts, fill=s["color"], width=2)

    draw.line([(pl, pt), (pl, H - pb)], fill=(40, 40, 40), width=1)
    draw.line([(pl, H - pb), (W - pr, H - pb)], fill=(40, 40, 40), width=1)
    draw.text((pl + 4, 8), "Loss comparison (smoothed)", fill=(40, 40, 40))

    lx = W - pr + 10
    ly = pt
    for s in series:
        draw.rectangle([(lx, ly + 3), (lx + 14, ly + 13)], fill=s["color"])
        draw.text((lx + 18, ly), s["id"][:20], fill=(40, 40, 40))
        ly += 20

    return img


logger.remove()
logger.add(sys.stdout, level="INFO", format="FoleyTune LoRA: {message}")


# --- Node 1: Feature Extractor ----------------------------------------------

class FoleyTuneFeatureExtractor:
    """Extract and cache SigLIP2/Synchformer/CLAP features + audio for LoRA training."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_deps": ("FOLEYTUNE_DEPS",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0, "step": 0.1}),
                "duration": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 3600.0, "step": 0.1,
                              "tooltip": "Clip duration in seconds. For chunked generation, set to full video length."}),
                "cache_dir": ("STRING", {"default": ""}),
                "name": ("STRING", {"default": "clip",
                          "tooltip": "Base name for auto-incremented files (e.g. clip -> clip_001.npz)"}),
            },
        }

    RETURN_TYPES = ("STRING", "FOLEYTUNE_FEATURES")
    RETURN_NAMES = ("npz_path", "features")
    FUNCTION = "extract_features"
    CATEGORY = "FoleyTune"
    OUTPUT_NODE = True

    def extract_features(self, hunyuan_deps, image, prompt, negative_prompt, frame_rate,
                         duration, cache_dir, name):
        from hunyuanvideo_foley.utils.feature_utils import (
            encode_video_with_siglip2, encode_video_with_sync, encode_text_feat,
        )

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Auto-increment filename
        idx = 1
        while (cache_dir / f"{name}_{idx:03d}.npz").exists():
            idx += 1
        npz_path = cache_dir / f"{name}_{idx:03d}.npz"

        # -- Extract visual features --
        # image is [B, H, W, C] float32 in [0,1] from ComfyUI
        # Convert to [1, T, C, H, W] uint8 for preprocessing
        frames_bhwc = image  # [T, H, W, C]
        frames_tchw = frames_bhwc.permute(0, 3, 1, 2)  # [T, C, H, W]
        frames_uint8 = (frames_tchw * 255).clamp(0, 255).to(torch.uint8)

        # Resample frames to target FPS for each extractor
        total_frames = frames_uint8.shape[0]

        # Compute duration from video frames if not specified
        if duration <= 0:
            duration = total_frames / frame_rate
            logger.warning(f"Auto-detected duration={duration:.2f}s from {total_frames} frames at {frame_rate}fps. "
                           f"Set duration explicitly if this is wrong (e.g. mismatched fps).")

        # SigLIP2: 8fps, 512x512
        siglip2_fps = 8
        n_siglip2_frames = max(1, int(duration * siglip2_fps))
        siglip2_indices = torch.linspace(0, total_frames - 1, n_siglip2_frames).long()
        siglip2_frames = frames_uint8[siglip2_indices]

        # Apply SigLIP2 preprocessing
        siglip2_processed = torch.stack([
            hunyuan_deps.siglip2_preprocess(f) for f in siglip2_frames
        ]).unsqueeze(0)  # [1, T, C, H, W]

        hunyuan_deps.siglip2_model.to(device)
        clip_features = encode_video_with_siglip2(
            siglip2_processed.to(device), hunyuan_deps
        ).cpu()
        hunyuan_deps.siglip2_model.to(offload_device)

        # Synchformer: 25fps, 224x224
        sync_fps = 25
        n_sync_frames = max(16, int(duration * sync_fps))
        sync_indices = torch.linspace(0, total_frames - 1, n_sync_frames).long()
        sync_frames = frames_uint8[sync_indices]

        sync_processed = torch.stack([
            hunyuan_deps.syncformer_preprocess(f) for f in sync_frames
        ]).unsqueeze(0)  # [1, T, C, H, W]

        hunyuan_deps.syncformer_model.to(device)
        sync_features = encode_video_with_sync(
            sync_processed.to(device), hunyuan_deps
        ).cpu()
        hunyuan_deps.syncformer_model.to(offload_device)

        # CLAP text embedding -- must use last_hidden_state [B, seq_len, 768], NOT text_embeds (pooled)
        hunyuan_deps.clap_model.to(device)
        text_inputs = hunyuan_deps.clap_tokenizer(
            [prompt], padding=True, truncation=True, max_length=100,
            return_tensors="pt"
        ).to(device)
        clap_outputs = hunyuan_deps.clap_model(
            **text_inputs, output_hidden_states=True, return_dict=True
        )
        text_embedding = clap_outputs.last_hidden_state.cpu()  # [1, seq_len, 768]

        # Encode negative prompt (unconditional)
        neg_text_inputs = hunyuan_deps.clap_tokenizer(
            [negative_prompt], padding=True, truncation=True, max_length=100,
            return_tensors="pt"
        ).to(device)
        neg_clap_outputs = hunyuan_deps.clap_model(
            **neg_text_inputs, output_hidden_states=True, return_dict=True
        )
        uncond_text_embedding = neg_clap_outputs.last_hidden_state.cpu()  # [1, seq_len, 768]

        hunyuan_deps.clap_model.to(offload_device)

        torch.cuda.empty_cache()

        # Save .npz
        np.savez(
            str(npz_path),
            clip_features=clip_features.float().numpy(),
            sync_features=sync_features.float().numpy(),
            text_embedding=text_embedding.float().numpy(),
            prompt=prompt,
            duration=duration,
            fps=frame_rate,
        )

        logger.info(f"Saved features to {npz_path}")
        logger.info(f"  clip_features: {clip_features.shape}, sync_features: {sync_features.shape}")
        logger.info(f"  text_embedding: {text_embedding.shape}, duration: {duration:.2f}s")

        features = {
            "clip_feat": clip_features,              # [1, T_clip, 768]
            "sync_feat": sync_features,              # [1, T_sync, 768]
            "text_feat": text_embedding,             # [1, T_text, 768]
            "uncond_text_feat": uncond_text_embedding,  # [1, T_text, 768]
            "duration": duration,
        }

        return (str(npz_path), features)


# --- Batch Feature Extraction Helpers ----------------------------------------

_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}


def _scan_video_folder(folder: Path):
    """Scan folder + 1 level of subfolders for video files."""
    files = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in _VIDEO_EXTS:
            files.append(f)
        elif f.is_dir():
            for child in sorted(f.iterdir()):
                if child.is_file() and child.suffix.lower() in _VIDEO_EXTS:
                    files.append(child)
    return files


def _ffprobe_metadata(path: Path):
    """Get video fps and duration via FFprobe. Returns (fps, duration)."""
    import subprocess
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe failed on {path}: {result.stderr.decode()}")
    info = json.loads(result.stdout)
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            r_fps = stream.get("r_frame_rate", "25/1")
            num, den = map(int, r_fps.split("/"))
            fps = num / den
            duration = float(info.get("format", {}).get("duration", 0))
            if duration == 0:
                duration = float(stream.get("duration", 0))
            return fps, duration
    raise RuntimeError(f"No video stream found in {path}")


def _load_video_frames(path: Path):
    """Load video frames as [T, C, H, W] uint8 tensor via FFmpeg."""
    import subprocess
    # Get resolution via FFprobe
    cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-print_format", "json", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    w, h = int(stream["width"]), int(stream["height"])

    # Extract raw RGB24 frames
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed on {path}: {result.stderr.decode()}")

    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    frame_bytes = h * w * 3
    n_frames = len(raw) // frame_bytes
    frames = raw[:n_frames * frame_bytes].reshape(n_frames, h, w, 3)
    return torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)  # [T, C, H, W]


def _extract_audio_wav(video_path: Path, wav_path: Path):
    """Extract audio from video as WAV (native sample rate and channels)."""
    import subprocess
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path),
        "-vn", "-f", "wav",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())


# --- Node: Batch Feature Extractor ------------------------------------------

class FoleyTuneBatchFeatureExtractor:
    """Extract SigLIP2/Synchformer/CLAP features from a FOLEYTUNE_AUDIO_DATASET.

    Reads video frames from each item's video_path. Adds a 'features' dict
    to each item with clip_features, sync_features, text_embedding, duration, fps.
    Per-clip prompts via sidecar .txt files override the global prompt.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_deps": ("FOLEYTUNE_DEPS",),
                "dataset": (FOLEYTUNE_AUDIO_DATASET,),
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Global text prompt. Overridden by per-clip .txt sidecar files unless use_sidecar_prompts is off.",
                }),
                "use_sidecar_prompts": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When enabled, per-clip .txt sidecar files override the global prompt. "
                               "Disable to always use the global prompt (useful when sidecars are only used for clip selection).",
                }),
            },
        }

    RETURN_TYPES = (FOLEYTUNE_AUDIO_DATASET, "STRING")
    RETURN_NAMES = ("dataset", "report")
    FUNCTION = "extract_batch"
    CATEGORY = "FoleyTune"
    OUTPUT_NODE = True

    def extract_batch(self, hunyuan_deps, dataset, prompt, use_sidecar_prompts=True):
        from hunyuanvideo_foley.utils.feature_utils import (
            encode_video_with_siglip2, encode_video_with_sync,
        )
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # --- Phase 1: Probe metadata and resolve prompts ---
        clips = []
        lines = ["=== Batch Feature Extraction ===", ""]

        for item in dataset:
            video_path = Path(item["video_path"])
            try:
                fps, dur = _ffprobe_metadata(video_path)
            except Exception as e:
                lines.append(f"  SKIP  {item['name']}: FFprobe error — {e}")
                continue

            txt_path = video_path.with_suffix(".txt")
            if use_sidecar_prompts and txt_path.exists():
                clip_prompt = txt_path.read_text().strip()
            else:
                clip_prompt = item.get("prompt") or prompt

            clips.append({
                "item": item,
                "path": video_path,
                "fps": fps,
                "duration": dur,
                "prompt": clip_prompt,
                "name": item["name"],
            })

        if not clips:
            raise RuntimeError("No valid video clips in dataset")

        n = len(clips)
        logger.info(f"[BatchFeatureExtractor] {n} clips to process")

        # Pre-encode CLAP prompts (lightweight, do once)
        hunyuan_deps.clap_model.to(device)
        prompt_cache = {}
        for clip in clips:
            p = clip["prompt"]
            if p not in prompt_cache:
                inputs = hunyuan_deps.clap_tokenizer(
                    [p], padding=True, truncation=True, max_length=100,
                    return_tensors="pt"
                ).to(device)
                outputs = hunyuan_deps.clap_model(
                    **inputs, output_hidden_states=True, return_dict=True
                )
                prompt_cache[p] = outputs.last_hidden_state.cpu()
        hunyuan_deps.clap_model.to(offload_device)
        torch.cuda.empty_cache()
        logger.info(f"  {len(prompt_cache)} unique prompt(s) encoded")

        # --- Two-pass extraction with I/O prefetch ---
        from concurrent.futures import ThreadPoolExecutor

        clip_feats = [None] * n
        sync_feats = [None] * n

        def _prefetch_siglip2(idx):
            """Load frames + preprocess for SigLIP2 on background thread."""
            c = clips[idx]
            rgb = _load_video_frames(c["path"])
            total = rgb.shape[0]
            n_frames = max(1, int(c["duration"] * 8))
            indices = torch.linspace(0, total - 1, n_frames).long()
            processed = torch.stack([
                hunyuan_deps.siglip2_preprocess(f) for f in rgb[indices]
            ]).unsqueeze(0)
            del rgb
            return idx, processed

        def _prefetch_sync(idx):
            """Load frames + preprocess for Synchformer on background thread."""
            c = clips[idx]
            rgb = _load_video_frames(c["path"])
            total = rgb.shape[0]
            n_frames = max(16, int(c["duration"] * 25))
            indices = torch.linspace(0, total - 1, n_frames).long()
            processed = torch.stack([
                hunyuan_deps.syncformer_preprocess(f) for f in rgb[indices]
            ]).unsqueeze(0)
            del rgb
            return idx, processed

        # Pass 1: SigLIP2 — load model once, prefetch frames in background
        logger.info("[BatchFeatureExtractor] SigLIP2 pass...")
        hunyuan_deps.siglip2_model.to(device)
        with ThreadPoolExecutor(max_workers=2) as pool:
            # Submit first batch of prefetches
            pending = {}
            prefetch_idx = 0
            max_prefetch = 3  # keep up to 3 clips prefetched ahead
            while prefetch_idx < min(max_prefetch, n):
                pending[prefetch_idx] = pool.submit(_prefetch_siglip2, prefetch_idx)
                prefetch_idx += 1

            for i in range(n):
                mm.throw_exception_if_processing_interrupted()
                # Wait for current clip's prefetch
                idx, processed = pending.pop(i).result()
                # Submit next prefetch while GPU works
                if prefetch_idx < n:
                    pending[prefetch_idx] = pool.submit(_prefetch_siglip2, prefetch_idx)
                    prefetch_idx += 1
                # GPU inference
                clip_feats[i] = encode_video_with_siglip2(
                    processed.to(device), hunyuan_deps
                ).cpu()
                del processed
                logger.info(
                    f"  [{i+1}/{n}] {clips[i]['name']}: "
                    f"clip_feat {clip_feats[i].shape}"
                )
        hunyuan_deps.siglip2_model.to(offload_device)
        torch.cuda.empty_cache()

        # Pass 2: Synchformer — same pattern
        logger.info("[BatchFeatureExtractor] Synchformer pass...")
        hunyuan_deps.syncformer_model.to(device)
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {}
            prefetch_idx = 0
            while prefetch_idx < min(max_prefetch, n):
                pending[prefetch_idx] = pool.submit(_prefetch_sync, prefetch_idx)
                prefetch_idx += 1

            for i in range(n):
                mm.throw_exception_if_processing_interrupted()
                idx, processed = pending.pop(i).result()
                if prefetch_idx < n:
                    pending[prefetch_idx] = pool.submit(_prefetch_sync, prefetch_idx)
                    prefetch_idx += 1
                sync_feats[i] = encode_video_with_sync(
                    processed.to(device), hunyuan_deps
                ).cpu()
                del processed
                logger.info(
                    f"  [{i+1}/{n}] {clips[i]['name']}: "
                    f"sync_feat {sync_feats[i].shape}"
                )
        hunyuan_deps.syncformer_model.to(offload_device)
        torch.cuda.empty_cache()

        # Pass 3: Attach features to dataset items
        logger.info("[BatchFeatureExtractor] Attaching features to dataset items...")
        output_dataset = []
        for i in range(n):
            clip = clips[i]
            text_feat = prompt_cache[clip["prompt"]]
            item = dict(clip["item"])  # shallow copy to avoid mutating input
            item["features"] = {
                "clip_features": clip_feats[i],
                "sync_features": sync_feats[i],
                "text_embedding": text_feat,
                "duration": clip["duration"],
                "fps": clip["fps"],
            }
            item["prompt"] = clip["prompt"]
            output_dataset.append(item)
            lines.append(
                f"  OK    {clip['name']} ({clip['duration']:.1f}s @ "
                f"{clip['fps']:.1f}fps)  clip_feat={clip_feats[i].shape}  "
                f"sync_feat={sync_feats[i].shape}"
            )
        del clip_feats, sync_feats

        lines.append("")
        lines.append(f"Processed {n} clips")

        report = "\n".join(lines)
        logger.info(f"[BatchFeatureExtractor]\n{report}")
        return (output_dataset, report)


# --- Node 6: VAE Roundtrip --------------------------------------------------

class FoleyTuneVAERoundtrip:
    """Encode audio through DAC, decode back. Reveals codec quality ceiling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_deps": ("FOLEYTUNE_DEPS",),
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "roundtrip"
    CATEGORY = "FoleyTune"

    def roundtrip(self, hunyuan_deps, audio):
        device = mm.get_torch_device()
        dac = hunyuan_deps.dac_model

        waveform = audio["waveform"]  # [1, C, L]
        sample_rate = audio["sample_rate"]

        # Resample to 48kHz if needed
        if sample_rate != 48000:
            import soxr
            wav_np = waveform.squeeze(0).float().numpy().T  # [L, C]
            wav_np = soxr.resample(wav_np, sample_rate, 48000, quality="VHQ")
            waveform = torch.from_numpy(wav_np.T).float().unsqueeze(0)  # [1, C, L]

        # Convert to mono
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)

        # DAC encode -> decode
        # NOTE: DAC with continuous=True returns DiagonalGaussianDistribution.
        # Use .mode() (posterior mean) for deterministic, reproducible A/B —
        # .sample() makes every run produce different output, which defeats
        # the purpose of a codec-ceiling diagnostic.
        dac.to(device)
        with torch.no_grad():
            audio_in = waveform.to(device=device, dtype=torch.float32)
            z_dist, _, _, _, _ = dac.encode(audio_in)
            z = z_dist.mode()
            reconstructed = dac.decode(z)
        dac.cpu()
        torch.cuda.empty_cache()

        out = reconstructed.cpu().float()
        if not torch.isfinite(out).all():
            raise RuntimeError(
                "DAC round-trip produced non-finite values (NaN/Inf). "
                "Check input audio for silence/extreme values."
            )
        rms = torch.sqrt(torch.mean(out ** 2))
        target_rms = 10 ** (-27 / 20)
        if rms > 1e-8:
            out = out * (target_rms / rms)
        out = out.clamp(-1.0, 1.0)

        return ({"waveform": out, "sample_rate": 48000},)


# --- Node 2: LoRA Trainer ---------------------------------------------------

class FoleyTuneLoRATrainer:
    """Train a LoRA adapter for FoleyTune via flow matching."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("FOLEYTUNE_MODEL",),
                "hunyuan_deps": ("FOLEYTUNE_DEPS",),
                "data_dir": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": ""}),
                "target": (list(FOLEY_TARGET_PRESETS.keys()), {"default": "all_attn_mlp"}),
                "rank": ("INT", {"default": 128, "min": 4, "max": 128, "step": 4}),
                "alpha": ("FLOAT", {"default": 128.0, "min": 1.0, "max": 128.0}),
                "lr": ("FLOAT", {"default": 1e-4, "min": 1e-6, "max": 1e-2, "step": 1e-5}),
                "steps": ("INT", {"default": 15000, "min": 100, "max": 50000}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 64}),
                "grad_accum": ("INT", {"default": 1, "min": 1, "max": 32}),
                "warmup_steps": ("INT", {"default": 100, "min": 0, "max": 2000}),
                "save_every": ("INT", {"default": 500, "min": 50, "max": 10000}),
                "timestep_mode": (["uniform", "logit_normal", "curriculum"], {"default": "curriculum"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "seed": ("INT", {"default": 42}),
            },
            "optional": {
                "logit_normal_sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0}),
                "curriculum_switch": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 0.9}),
                "init_mode": (["standard", "pissa"], {"default": "standard"}),
                "use_rslora": ("BOOLEAN", {"default": False}),
                "lora_dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.3}),
                "lora_plus_ratio": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 32.0}),
                "schedule_type": (["constant", "cosine"], {"default": "constant"}),
                "latent_mixup_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "latent_noise_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.1}),
                "noise_offset": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.1, "step": 0.005,
                    "tooltip": "Per-sample channel-uniform noise added to latents before flow matching. Improves dynamic range unlike latent_noise_sigma which is per-element.",
                }),
                "min_snr_gamma": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 20.0, "step": 1.0,
                    "tooltip": "Min-SNR loss weighting gamma. Downweights high-noise timesteps where gradients are noisy. 5.0 recommended. 0 = disabled.",
                }),
                "ema_decay": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.9999, "step": 0.0001,
                    "tooltip": "EMA decay for LoRA weights. Smooths training, used at save time. 0.9995 recommended for fine-tuning. 0 = disabled.",
                }),
                "cos_sim_weight": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Weight for cosine similarity auxiliary loss on velocity. Directly targets phase/correlation alignment. 0.1 recommended. 0 = disabled.",
                }),
                "channel_loss_weight": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Weight velocity MSE by per-channel variance from dataset. Upweights perceptually important latent dimensions.",
                }),
                "t_min": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.2, "step": 0.01,
                    "tooltip": "Minimum timestep for sampling. Avoids near-clean timesteps. 0.05 recommended.",
                }),
                "t_max": ("FLOAT", {
                    "default": 1.0, "min": 0.8, "max": 1.0, "step": 0.01,
                    "tooltip": "Maximum timestep for sampling. Avoids near-noise timesteps. 0.95 recommended.",
                }),
                "optimizer_type": (["adamw", "prodigy", "automagic"], {"default": "prodigy"}),
                "prodigy_d_coef": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01,
                    "tooltip": "Prodigy d_coef: scales the learned step size. Lower values (e.g. 0.5) reduce effective lr. Only used with Prodigy optimizer.",
                }),
                "prodigy_growth_rate": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Prodigy growth_rate: max multiplicative increase of d per step. 0 = unlimited (inf). E.g. 1.02 = max 2%% growth per step. Only used with Prodigy optimizer.",
                }),
                "visual_dropout_prob": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": "Per-sample probability of zeroing visual features during training. Forces the text channel to carry audio-character signal, decoupling identity from sound. Use 0.5 for generic-style LoRAs (no performer binding); leave 0.0 for identity-preserving LoRAs.",
                }),
                "gradient_checkpointing": ("BOOLEAN", {"default": False}),
                "resume_from": ("STRING", {"default": ""}),
                "dataset_json": ("STRING", {
                    "default": "",
                    "tooltip": "Path to dataset.json. When set, uses its train/val split instead of scanning data_dir for all .npz files.",
                }),
            },
        }

    RETURN_TYPES = ("FOLEYTUNE_MODEL", "IMAGE")
    RETURN_NAMES = ("model", "loss_curve")
    OUTPUT_TOOLTIPS = (
        "Model with trained LoRA adapter applied.",
        "Training loss curve (smoothed).",
    )
    FUNCTION = "train"
    CATEGORY = "FoleyTune"
    OUTPUT_NODE = True

    def train(self, hunyuan_model, hunyuan_deps, data_dir, output_dir, target, rank,
              alpha, lr, steps, batch_size, grad_accum, warmup_steps, save_every,
              timestep_mode, precision, seed,
              logit_normal_sigma=1.0, curriculum_switch=0.6,
              init_mode="standard", use_rslora=False, lora_dropout=0.0,
              lora_plus_ratio=1.0, schedule_type="constant",
              latent_mixup_alpha=0.0, latent_noise_sigma=0.0,
              noise_offset=0.0, min_snr_gamma=0.0, ema_decay=0.0,
              cos_sim_weight=0.0, channel_loss_weight=False,
              t_min=0.0, t_max=1.0, optimizer_type="adamw",
              visual_dropout_prob=0.0,
              gradient_checkpointing=False,
              resume_from="", dataset_json="",
              prodigy_d_coef=1.0, prodigy_growth_rate=0.0):

        import random
        device = mm.get_torch_device()
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map[precision]

        # Exit ComfyUI's inference_mode so gradients work
        with torch.inference_mode(False), torch.enable_grad():
            return self._train_inner(
            hunyuan_model, hunyuan_deps, data_dir, output_dir, target, rank,
            alpha, lr, steps, batch_size, grad_accum, warmup_steps, save_every,
            timestep_mode, precision, seed, device, dtype,
            logit_normal_sigma, curriculum_switch, init_mode, use_rslora,
            lora_dropout, lora_plus_ratio, schedule_type,
            latent_mixup_alpha, latent_noise_sigma,
            noise_offset, min_snr_gamma, ema_decay,
            cos_sim_weight, channel_loss_weight,
            t_min, t_max, optimizer_type,
            visual_dropout_prob,
            gradient_checkpointing, resume_from,
            dataset_json, prodigy_d_coef, prodigy_growth_rate,
        )

    def _train_inner(self, hunyuan_model, hunyuan_deps, data_dir, output_dir, target, rank,
                     alpha, lr, steps, batch_size, grad_accum, warmup_steps, save_every,
                     timestep_mode, precision, seed, device, dtype,
                     logit_normal_sigma, curriculum_switch, init_mode, use_rslora,
                     lora_dropout, lora_plus_ratio, schedule_type,
                     latent_mixup_alpha, latent_noise_sigma,
                     noise_offset, min_snr_gamma, ema_decay,
                     cos_sim_weight, channel_loss_weight,
                     t_min, t_max, optimizer_type,
                     visual_dropout_prob,
                     gradient_checkpointing, resume_from,
                     dataset_json="",
                     prodigy_d_coef=1.0, prodigy_growth_rate=0.0):
        import random

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        samples_path = output_path / "samples"
        samples_path.mkdir(exist_ok=True)

        # -- Prepare dataset --
        logger.info("Preparing dataset...")

        clip_names = None
        val_entry = None
        ds_cfg = None
        if dataset_json and os.path.exists(dataset_json):
            try:
                with open(dataset_json) as f:
                    ds_cfg = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in dataset file {dataset_json}: {e}") from e
            if not isinstance(ds_cfg.get("train"), list):
                raise ValueError(f"dataset_json must contain a 'train' key with a list of clip names")
            # Resolve paths relative to JSON file location
            data_dir = str(Path(dataset_json).parent)
            clip_names = ds_cfg["train"]

        dataset = prepare_dataset(data_dir, hunyuan_deps.dac_model, device, dtype,
                                  clip_names=clip_names)
        n_clips = len(dataset)
        logger.info(f"Dataset ready: {n_clips} clips")

        if ds_cfg is not None and ds_cfg.get("val"):
            val_npz = Path(data_dir) / f"{ds_cfg['val']}.npz"
            if val_npz.exists():
                val_entry = prepare_single_entry(str(val_npz), hunyuan_deps.dac_model, device, dtype)
                logger.info(f"Val clip loaded: {ds_cfg['val']}")

        # -- Setup model with LoRA --
        model = copy.deepcopy(hunyuan_model)
        model.to(device=device, dtype=dtype)
        model.train()

        # VRAM offload strategies
        if gradient_checkpointing:
            model.gradient_checkpoint = True
            model.gradient_checkpoint_layers = -1  # all layers
            logger.info("Gradient checkpointing enabled for all layers")

        target_suffixes = FOLEY_TARGET_PRESETS[target]
        n_wrapped = apply_lora(
            model, rank=rank, alpha=alpha,
            target_suffixes=target_suffixes,
            dropout=lora_dropout, init_mode=init_mode,
            use_rslora=use_rslora,
        )
        logger.info(f"LoRA applied: {n_wrapped} layers wrapped (target={target}, rank={rank})")

        # Freeze base, train LoRA only
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)")

        # -- Optimizer --
        if lora_plus_ratio > 1.0:
            # LoRA+: separate LR for B matrices
            a_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_A" in n]
            b_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
            param_groups = [
                {"params": a_params, "lr": lr},
                {"params": b_params, "lr": lr * lora_plus_ratio},
            ]
        else:
            param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]

        if optimizer_type == "prodigy":
            from prodigyopt import Prodigy
            for pg in param_groups:
                pg.pop("lr", None)
            _growth = float("inf") if prodigy_growth_rate <= 0 else prodigy_growth_rate
            optimizer = Prodigy(param_groups, lr=1.0, betas=(0.9, 0.999), weight_decay=0.01,
                                d_coef=prodigy_d_coef, growth_rate=_growth)
            logger.info(f"Using Prodigy optimizer (d_coef={prodigy_d_coef}, growth_rate={_growth})")
        elif optimizer_type == "automagic":
            from lora.automagic import Automagic
            if lora_plus_ratio > 1.0:
                logger.warning("lora_plus_ratio has no effect with Automagic (per-element LR)")
            for pg in param_groups:
                pg.pop("lr", None)
            optimizer = Automagic(param_groups, lr=1e-6, min_lr=1e-7, max_lr=1e-3, lr_bump=1e-6)
            logger.info("Using Automagic optimizer")
        else:
            optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=0.01)

        # -- LR Scheduler --
        # scheduler.step() is called once per grad_accum training steps,
        # so scale the internal counter back to training steps
        def lr_lambda(sched_step):
            actual_step = sched_step * grad_accum
            if actual_step < warmup_steps:
                return actual_step / max(warmup_steps, 1)
            if schedule_type == "cosine":
                progress = (actual_step - warmup_steps) / max(steps - warmup_steps, 1)
                return 0.5 * (1 + np.cos(np.pi * progress))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # -- Resume --
        start_step = 0
        _resumed_ema = None
        if resume_from and os.path.exists(resume_from):
            ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
            load_lora(model, ckpt["state_dict"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_step = ckpt.get("step", 0)
            _resumed_ema = ckpt.get("ema_state", None)
            if start_step >= steps:
                steps = start_step + steps
            logger.info(f"Resumed from step {start_step}: {resume_from}")
            del ckpt

        # -- EMA --
        ema_state = None
        if ema_decay > 0:
            if _resumed_ema is not None:
                ema_state = {k: v.to(device) for k, v in _resumed_ema.items()}
                logger.info(f"EMA restored from checkpoint (decay={ema_decay})")
                del _resumed_ema
            else:
                ema_state = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
                logger.info(f"EMA initialized fresh (decay={ema_decay})")

        # -- Training loop --
        meta = {
            "target": target, "rank": rank, "alpha": alpha,
            "lr": lr, "steps": steps, "batch_size": batch_size,
            "timestep_mode": timestep_mode, "logit_normal_sigma": logit_normal_sigma,
            "curriculum_switch": curriculum_switch,
            "init_mode": init_mode, "use_rslora": use_rslora,
            "lora_dropout": lora_dropout, "lora_plus_ratio": lora_plus_ratio,
            "schedule_type": schedule_type,
            "latent_mixup_alpha": latent_mixup_alpha,
            "latent_noise_sigma": latent_noise_sigma,
            "noise_offset": noise_offset,
            "min_snr_gamma": min_snr_gamma,
            "ema_decay": ema_decay,
            "cos_sim_weight": cos_sim_weight,
            "channel_loss_weight": channel_loss_weight,
            "t_min": t_min, "t_max": t_max,
            "optimizer_type": optimizer_type,
            "gradient_checkpointing": gradient_checkpointing,
            "n_clips": n_clips, "precision": precision, "seed": seed,
        }

        # Embed training prompts (unique, sorted by frequency)
        from collections import Counter
        prompt_counts = Counter(d["prompt"] for d in dataset)
        meta["prompts"] = [p for p, _ in prompt_counts.most_common()]

        # Pre-compute per-channel variance weights for channel-weighted loss
        channel_weights = None
        if channel_loss_weight:
            all_latents = torch.cat([d["latents"] for d in dataset], dim=0)
            ch_var = all_latents.var(dim=(0, 2))
            channel_weights = (ch_var / ch_var.mean()).clamp(0.5, 2.0)
            logger.info(f"Channel weights: min={channel_weights.min():.2f} max={channel_weights.max():.2f}")
            del all_latents

        losses = []
        metrics_history = []  # list of {step, loss, ...spectral metrics}
        log_interval = 50
        remaining = steps - start_step
        pbar = comfy.utils.ProgressBar(remaining)

        # Load reference audio for metrics comparison
        ref_entry = dataset[0]
        ref_audio_path = None
        for ext in (".flac", ".wav", ".ogg"):
            candidate = Path(data_dir) / f"{ref_entry['name']}{ext}"
            if candidate.exists():
                ref_audio_path = candidate
                break
        ref_wav_np = None
        if ref_audio_path:
            import soundfile as _sf
            _raw, _sr = _sf.read(str(ref_audio_path))
            if _raw.ndim > 1:
                _raw = _raw.mean(axis=1)
            if _sr != 48000:
                import soxr as _soxr
                _raw = _soxr.resample(_raw[:, None], _sr, 48000, quality="VHQ").squeeze(-1)
            # DAC round-trip: measure model quality, not codec ceiling
            with torch.no_grad():
                hunyuan_deps.dac_model.to(device)
                _ref_t = torch.from_numpy(_raw).float().unsqueeze(0).unsqueeze(0)
                _ref_t = _ref_t.to(device=device, dtype=torch.float32)
                _z, _, _, _, _ = hunyuan_deps.dac_model.encode(_ref_t)
                _ref_dec = hunyuan_deps.dac_model.decode(_z.mode())
                ref_wav_np = _ref_dec.squeeze().cpu().numpy()
                hunyuan_deps.dac_model.cpu()
            _save_spectrogram(ref_wav_np, 48000, samples_path / "reference")

        logger.info(f"Starting training: {steps} steps, batch {batch_size}, lr {lr}")
        t_start = time.time()

        step = start_step  # default in case loop doesn't execute
        for step in range(start_step, steps):
            mm.throw_exception_if_processing_interrupted()
            # Check for skip flag
            skip_flag = output_path.parent / "skip_current.flag"
            if skip_flag.exists():
                logger.info(f"Skip flag detected at step {step}, saving and stopping")
                ckpt_path = output_path / f"adapter_cancelled_step{step:05d}.pt"
                save_checkpoint(model, optimizer, scheduler, step, meta, ckpt_path)
                break

            model.train()

            # Sample batch
            indices = [np.random.randint(0, n_clips) for _ in range(batch_size)]
            batch_latents = torch.cat([dataset[i]["latents"] for i in indices], dim=0).to(device, dtype=dtype)
            batch_clip = torch.cat([dataset[i]["clip_features"] for i in indices], dim=0)
            batch_sync = torch.cat([dataset[i]["sync_features"] for i in indices], dim=0)
            batch_text = torch.cat([dataset[i]["text_embedding"] for i in indices], dim=0)

            # Pad features to consistent lengths
            max_clip_len = max(batch_clip.shape[1], 1)
            max_sync_len = max(batch_sync.shape[1], 1)
            # Ensure sync length is multiple of 8 (required by model)
            max_sync_len = ((max_sync_len + 7) // 8) * 8
            batch_clip = F.pad(batch_clip, (0, 0, 0, max(0, max_clip_len - batch_clip.shape[1])))
            batch_sync = F.pad(batch_sync, (0, 0, 0, max(0, max_sync_len - batch_sync.shape[1])))

            # Optional latent augmentation
            if latent_mixup_alpha > 0 and batch_size > 1:
                lam = np.random.beta(latent_mixup_alpha, latent_mixup_alpha)
                perm = torch.randperm(batch_size)
                batch_latents = lam * batch_latents + (1 - lam) * batch_latents[perm]

            if latent_noise_sigma > 0:
                batch_latents = batch_latents + torch.randn_like(batch_latents) * latent_noise_sigma

            if noise_offset > 0:
                # Channel-uniform noise: same value across all spatial/temporal dims per sample per channel
                offset = torch.randn(batch_latents.shape[0], batch_latents.shape[1], 1, device=device, dtype=dtype) * noise_offset
                batch_latents = batch_latents + offset

            # Sample timesteps
            t = sample_timesteps(
                batch_size, timestep_mode, device, dtype,
                sigma=logit_normal_sigma, curriculum_switch=curriculum_switch,
                step=step, start_step=start_step, total_steps=steps,
                t_min=t_min, t_max=t_max,
            )

            # Forward + loss
            loss = flow_matching_loss(
                model, batch_latents, t, batch_clip, batch_sync, batch_text, device, dtype,
                visual_dropout_prob=visual_dropout_prob,
                min_snr_gamma=min_snr_gamma,
                cos_sim_weight=cos_sim_weight,
                channel_weights=channel_weights,
            )
            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if ema_state is not None:
                    with torch.no_grad():
                        for n, p in model.named_parameters():
                            if p.requires_grad and n in ema_state:
                                ema_state[n].mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            losses.append(loss.item() * grad_accum)

            # Logging + live preview
            if (step + 1) % log_interval == 0:
                avg_loss = np.mean(losses[-log_interval:])
                elapsed = time.time() - t_start
                _lr_str = (f"{optimizer.get_avg_learning_rate():.2e}"
                           if hasattr(optimizer, "get_avg_learning_rate")
                           else f"{scheduler.get_last_lr()[0]:.2e}")
                logger.info(f"Step {step+1}/{steps} | loss: {avg_loss:.4f} | "
                           f"lr: {_lr_str} | elapsed: {elapsed:.0f}s")

                preview_img = _draw_loss_curve(
                    losses, start_step=start_step,
                    smoothed=_smooth_losses(losses),
                )
                pbar.update_absolute(
                    step + 1 - start_step, remaining,
                    ("JPEG", preview_img, 800),
                )

            # Save checkpoint + eval sample
            if (step + 1) % save_every == 0:
                # Save with live weights for optimizer consistency on resume
                ckpt_path = output_path / f"adapter_step{step+1:05d}.pt"
                save_checkpoint(model, optimizer, scheduler, step + 1, meta, ckpt_path,
                                ema_state=ema_state)
                _draw_loss_curve(losses, start_step=start_step, smoothed=_smooth_losses(losses)).save(str(output_path / "loss.png"))

                # Swap in EMA weights for eval (better sample quality)
                if ema_state is not None:
                    _live_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
                    for n, p in model.named_parameters():
                        if p.requires_grad and n in ema_state:
                            p.data.copy_(ema_state[n])

                # Generate eval sample + compute metrics
                model.eval()
                wav, sr = generate_eval_sample(
                    model, hunyuan_deps.dac_model, dataset[0], device, dtype,
                )
                wav_mono = wav.squeeze()
                wav_t = torch.from_numpy(wav)
                if wav_t.ndim == 1:
                    wav_t = wav_t.unsqueeze(0)
                _save_wav(samples_path / f"step_{step+1:05d}.wav", wav_t, sr)
                _save_spectrogram(wav_mono, sr, samples_path / f"step_{step+1:05d}")

                # Spectral metrics
                sm = spectral_metrics(wav_mono, sr)
                step_metrics = {"step": step + 1, "loss": float(np.mean(losses[-log_interval:])), **sm}
                if ref_wav_np is not None:
                    rm = reference_metrics(wav_mono, ref_wav_np, sr)
                    step_metrics.update(rm)
                metrics_history.append(step_metrics)
                with open(output_path / "metrics_history.json", "w") as _mf:
                    json.dump(metrics_history, _mf, indent=2)

                logger.info(f"Step {step+1} metrics: "
                           f"LSD={step_metrics.get('log_spectral_distance_db', 0):.2f}dB  "
                           f"MCD={step_metrics.get('mel_cepstral_distortion', 0):.2f}  "
                           f"HF={step_metrics.get('hf_energy_ratio', 0):.3f}  "
                           f"SC={step_metrics.get('spectral_convergence', 0):.3f}")

                # Generate val sample if a val clip was loaded
                if val_entry is not None:
                    val_wav, val_sr = generate_eval_sample(
                        model, hunyuan_deps.dac_model, val_entry, device, dtype,
                    )
                    val_wav_mono = val_wav.squeeze()
                    val_wav_t = torch.from_numpy(val_wav)
                    if val_wav_t.ndim == 1:
                        val_wav_t = val_wav_t.unsqueeze(0)
                    _save_wav(samples_path / f"val_step_{step+1:05d}.wav", val_wav_t, val_sr)
                    _save_spectrogram(val_wav_mono, val_sr, samples_path / f"val_step_{step+1:05d}")

                # Restore live weights for continued training
                if ema_state is not None:
                    for n, p in model.named_parameters():
                        if p.requires_grad and n in _live_params:
                            p.data.copy_(_live_params[n])

                model.train()

        # Save metrics history
        if metrics_history:
            with open(output_path / "metrics_history.json", "w") as f:
                json.dump(metrics_history, f, indent=2)

        # -- Save final --
        if ema_state is not None:
            for n, p in model.named_parameters():
                if p.requires_grad and n in ema_state:
                    p.data.copy_(ema_state[n])

        final_path = output_path / "adapter_final.pt"
        meta["steps_completed"] = step + 1 if step >= start_step else start_step
        save_checkpoint(model, optimizer, scheduler, step + 1, meta, final_path, final=True)
        save_meta_json(meta, output_path / "meta.json")
        # Draw and save loss curve
        smoothed = _smooth_losses(losses)
        loss_img = _draw_loss_curve(losses, start_step=start_step, smoothed=smoothed)
        loss_img.save(str(output_path / "loss.png"))
        loss_curve_tensor = _pil_to_tensor(loss_img)

        elapsed_total = time.time() - t_start
        logger.info(f"Training complete: {elapsed_total:.0f}s, final loss: {np.mean(losses[-100:]):.4f}")
        logger.info(f"Adapter saved to {final_path}")

        # Save to ComfyUI temp dir for inline node preview
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = f"lora_loss_curve.png"
        loss_img.save(os.path.join(temp_dir, temp_file))

        # Return model with LoRA active (on CPU for ComfyUI pipeline)
        model.eval()
        model.to(mm.unet_offload_device())
        return {"ui": {"images": [{"filename": temp_file, "subfolder": "", "type": "temp"}]},
                "result": (model, loss_curve_tensor)}


# --- Node 3: LoRA Loader ----------------------------------------------------

class FoleyTuneLoRALoader:
    """Load a trained LoRA adapter into a FoleyTune model for inference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("FOLEYTUNE_MODEL",),
                "adapter_path": ("STRING", {"default": ""}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("FOLEYTUNE_MODEL", "STRING")
    RETURN_NAMES = ("model", "prompts")
    FUNCTION = "load_adapter"
    CATEGORY = "FoleyTune"

    def load_adapter(self, hunyuan_model, adapter_path, strength):
        if not adapter_path or not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        ckpt = _load_adapter_checkpoint(adapter_path)

        # Handle both raw state_dict and wrapped checkpoint formats
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            meta = ckpt.get("meta", {})
            # Prefer EMA weights for inference when available
            if "ema_state" in ckpt:
                for key, ema_val in ckpt["ema_state"].items():
                    if key in state_dict:
                        state_dict[key] = ema_val
                logger.info("Using EMA weights from checkpoint for inference")
        else:
            state_dict = ckpt
            meta = {}

        rank = meta.get("rank", 16)
        alpha = meta.get("alpha", float(rank))
        target = meta.get("target", "all_attn_mlp")
        init_mode = meta.get("init_mode", "standard")
        use_rslora = meta.get("use_rslora", False)
        lora_dropout = meta.get("lora_dropout", 0.0)

        # Get target suffixes
        if isinstance(target, str) and target in FOLEY_TARGET_PRESETS:
            target_suffixes = FOLEY_TARGET_PRESETS[target]
        elif isinstance(target, (list, tuple)):
            target_suffixes = tuple(target)
        else:
            target_suffixes = FOLEY_TARGET_PRESETS["all_attn_mlp"]

        # Deep copy model
        model = copy.deepcopy(hunyuan_model)

        # Apply LoRA structure (always standard init for loading)
        n_wrapped = apply_lora(
            model, rank=rank, alpha=alpha,
            target_suffixes=target_suffixes,
            dropout=lora_dropout,
            init_mode="standard",  # PiSSA residuals are in the checkpoint
            use_rslora=use_rslora,
        )

        # Load weights — for PiSSA, state_dict includes modified base weights
        if init_mode == "pissa":
            model.load_state_dict(state_dict, strict=False)
        else:
            load_lora(model, state_dict)

        # Apply strength scaling
        if strength != 1.0:
            for name, module in model.named_modules():
                if isinstance(module, LoRALinear):
                    module.lora_B.data *= strength

        model.eval()
        logger.info(f"Loaded LoRA adapter: {n_wrapped} layers, rank={rank}, strength={strength}")

        prompts = "\n".join(meta.get("prompts", []))
        return (model, prompts)


# --- Node 4: LoRA Scheduler -------------------------------------------------

class FoleyTuneLoRAScheduler:
    """Run multiple LoRA training experiments from a JSON sweep configuration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("FOLEYTUNE_MODEL",),
                "hunyuan_deps": ("FOLEYTUNE_DEPS",),
                "sweep_json": ("STRING", {"default": ""}),
                "run_only": ("STRING", {
                    "default": "all",
                    "tooltip": "Run all experiments or a single one by id (e.g. 'sigma07_cur05').",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("summary_path", "comparison_curves")
    OUTPUT_TOOLTIPS = (
        "Path to experiment_summary.json.",
        "All smoothed loss curves overlaid on the same axes.",
    )
    FUNCTION = "run_sweep"
    CATEGORY = "FoleyTune"
    OUTPUT_NODE = True

    # Default training params — tuned from sweep results (v5–v10)
    _PARAM_DEFAULTS = {
        "target": "all_attn_mlp", "rank": 64, "alpha": 64.0,
        "lr": 5e-5, "steps": 13000, "batch_size": 8, "grad_accum": 1,
        "warmup_steps": 100, "save_every": 1000,
        "timestep_mode": "uniform", "precision": "bf16", "seed": 42,
        "logit_normal_sigma": 0.7, "curriculum_switch": 0.5,
        "init_mode": "standard", "use_rslora": False, "lora_dropout": 0.0,
        "lora_plus_ratio": 1.0, "schedule_type": "cosine",
        "latent_mixup_alpha": 0.0, "latent_noise_sigma": 0.0,
        "noise_offset": 0.0, "min_snr_gamma": 0.0, "ema_decay": 0.0,
        "cos_sim_weight": 0.0, "channel_loss_weight": False,
        "t_min": 0.0, "t_max": 1.0, "optimizer_type": "prodigy",
        "prodigy_d_coef": 1.0, "prodigy_growth_rate": 0.0,
        "visual_dropout_prob": 0.5,
        "gradient_checkpointing": False,
        "resume_from": "",
    }

    _DEFAULT_SWEEP = {
        "name": "sweep",
        "dataset_json": "",
        "output_root": "",
        "base": {},
        "experiments": [
            {
                "id": "sigma07_cur05",
                "description": "Best overall — sigma=0.7, curriculum=0.5 (PBC=0.661)",
                "logit_normal_sigma": 0.7,
                "curriculum_switch": 0.5,
            },
            {
                "id": "sigma08_cur05",
                "description": "Best TV — sigma=0.8, curriculum=0.5 (PBC=0.642, TV=1.82)",
                "logit_normal_sigma": 0.8,
                "curriculum_switch": 0.5,
            },
            {
                "id": "sigma07_cur04",
                "description": "Earlier curriculum — sigma=0.7, curriculum=0.4 (PBC=0.644)",
                "logit_normal_sigma": 0.7,
                "curriculum_switch": 0.4,
            },
            {
                "id": "baseline_cur05",
                "description": "Baseline — default sigma=1.0, curriculum=0.5 (PBC=0.592)",
                "logit_normal_sigma": 1.0,
                "curriculum_switch": 0.5,
            },
        ],
    }

    def _merge_config(self, base, experiment):
        merged = {**self._PARAM_DEFAULTS, **base}
        for k, v in experiment.items():
            if k not in ("id", "description"):
                merged[k] = v
        return merged

    def run_sweep(self, hunyuan_model, hunyuan_deps, sweep_json, run_only="all"):
        if not sweep_json:
            raise ValueError("sweep_json path is required")
        if not os.path.exists(sweep_json):
            template = copy.deepcopy(self._DEFAULT_SWEEP)
            template["output_root"] = str(Path(sweep_json).parent / "output")
            os.makedirs(os.path.dirname(sweep_json), exist_ok=True)
            with open(sweep_json, "w") as f:
                json.dump(template, f, indent=2)
            raise FileNotFoundError(
                f"Sweep JSON not found — wrote default template to: {sweep_json}\n"
                "Edit it with your dataset_json path and output_root, then re-run."
            )

        with open(sweep_json) as f:
            sweep = json.load(f)

        sweep_name = sweep.get("name", "sweep")
        data_dir = sweep.get("data_dir", "")
        dataset_json = sweep.get("dataset_json", "")
        output_root = Path(sweep.get("output_root", f"lora_output/{sweep_name}"))
        base_config = sweep.get("base", {})
        experiments = sweep.get("experiments", [])

        if run_only and run_only.strip().lower() != "all":
            target_id = run_only.strip()
            experiments = [e for e in experiments if e.get("id") == target_id]
            if not experiments:
                all_ids = [e.get("id", "?") for e in sweep.get("experiments", [])]
                raise ValueError(f"Experiment '{target_id}' not found. Available: {all_ids}")

        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "experiment_summary.json"

        # Load existing summary for resume
        completed_ids = set()
        results = []
        if summary_path.exists():
            with open(summary_path) as f:
                existing = json.load(f)
            results = existing.get("experiments", [])
            completed_ids = {r["id"] for r in results if r.get("status") == "completed"}

        # Prepare dataset once — support dataset_json for train/val split
        device = mm.get_torch_device()
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        base_precision = base_config.get("precision", "bf16")
        dtype = dtype_map[base_precision]

        clip_names = None
        ds_cfg = None
        if dataset_json and os.path.exists(dataset_json):
            with open(dataset_json) as f:
                ds_cfg = json.load(f)
            if not isinstance(ds_cfg.get("train"), list):
                raise ValueError("dataset_json must contain a 'train' key with a list of clip names")
            data_dir = str(Path(dataset_json).parent)
            clip_names = ds_cfg["train"]
            logger.info(f"Using dataset_json: {dataset_json} ({len(clip_names)} train clips)")
        elif not data_dir:
            raise ValueError("Sweep JSON must specify either 'data_dir' or 'dataset_json'")

        logger.info(f"Preparing shared dataset from {data_dir}...")
        dataset = prepare_dataset(data_dir, hunyuan_deps.dac_model, device, dtype,
                                  clip_names=clip_names)

        from collections import Counter
        _prompt_counts = Counter(d["prompt"] for d in dataset)
        prompts_list = [p for p, _ in _prompt_counts.most_common()]

        # Load optional validation sample — from dataset_json val or explicit eval_npz
        val_entry = None
        val_ref_wav = None
        if ds_cfg is not None and ds_cfg.get("val"):
            val_npz = Path(data_dir) / f"{ds_cfg['val']}.npz"
            if val_npz.exists():
                val_entry = prepare_single_entry(str(val_npz), hunyuan_deps.dac_model, device, dtype)
                logger.info(f"Val clip loaded from dataset_json: {ds_cfg['val']}")
                # Load reference audio for val spectrogram comparison
                for ext in (".flac", ".wav", ".ogg"):
                    candidate = val_npz.with_suffix(ext)
                    if candidate.exists():
                        import soundfile as _sf
                        _raw, _sr = _sf.read(str(candidate))
                        if _raw.ndim > 1:
                            _raw = _raw.mean(axis=1)
                        if _sr != 48000:
                            import soxr as _soxr
                            _raw = _soxr.resample(_raw[:, None], _sr, 48000, quality="VHQ").squeeze(-1)
                        val_ref_wav = _raw
                        break
        eval_npz = sweep.get("eval_npz") or base_config.get("eval_npz")
        if eval_npz and os.path.exists(eval_npz):
            logger.info(f"Loading validation sample from {eval_npz}...")
            val_entry = prepare_single_entry(eval_npz, hunyuan_deps.dac_model, device, dtype)
            # Load reference audio for val spectrogram
            val_stem = Path(eval_npz).stem
            val_parent = Path(eval_npz).parent
            for ext in (".flac", ".wav", ".ogg"):
                candidate = val_parent / f"{val_stem}{ext}"
                if candidate.exists():
                    import soundfile as _sf
                    _raw, _sr = _sf.read(str(candidate))
                    if _raw.ndim > 1:
                        _raw = _raw.mean(axis=1)
                    if _sr != 48000:
                        import soxr as _soxr
                        _raw = _soxr.resample(_raw[:, None], _sr, 48000, quality="VHQ").squeeze(-1)
                    val_ref_wav = _raw
                    break
            logger.info(f"Validation sample loaded: {val_entry['name']}")
        elif eval_npz:
            logger.warning(f"eval_npz path not found, skipping validation sample: {eval_npz}")

        # Collect loss histories for comparison chart
        all_loss_histories = {}

        for exp in experiments:
            exp_id = exp.get("id", f"exp_{len(results)}")

            if exp_id in completed_ids:
                # Check if new config requests more steps than completed run
                config_check = self._merge_config(base_config, exp)
                prev_result = next(r for r in results if r["id"] == exp_id)
                prev_steps = prev_result.get("config", {}).get("steps", 0)

                if config_check["steps"] > prev_steps:
                    # Auto-resume: find last checkpoint and extend training
                    exp_dir = output_root / exp_id
                    last_ckpt = exp_dir / f"adapter_step{prev_steps:05d}.pt"
                    if not last_ckpt.exists():
                        last_ckpt = exp_dir / "adapter_final.pt"
                    if last_ckpt.exists():
                        exp["resume_from"] = str(last_ckpt)
                    logger.info(f"Extending experiment {exp_id}: {prev_steps} -> {config_check['steps']} steps "
                                f"(resume from {exp.get('resume_from', config_check.get('resume_from', '?'))})")
                    # Remove from completed so it runs again
                    results = [r for r in results if r["id"] != exp_id]
                    completed_ids.discard(exp_id)
                else:
                    logger.info(f"Skipping completed experiment: {exp_id}")
                    exp_dir = output_root / exp_id
                    loss_file = exp_dir / "loss_history.json"
                    if loss_file.exists():
                        with open(loss_file) as f:
                            all_loss_histories[exp_id] = json.load(f)
                    continue

            config = self._merge_config(base_config, exp)
            exp_dir = output_root / exp_id
            exp_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Starting experiment: {exp_id}")
            logger.info(f"Config: {json.dumps({k: v for k, v in config.items() if k != 'description'}, indent=2)}")

            exp_result = {"id": exp_id, "config": config, "status": "running"}

            try:
                with torch.inference_mode(False), torch.enable_grad():
                    # Deep copy model for this experiment
                    model = copy.deepcopy(hunyuan_model)
                    model.to(device=device, dtype=dtype)
                    model.train()

                    # VRAM offload strategies
                    if config.get("gradient_checkpointing", False):
                        model.gradient_checkpoint = True
                        model.gradient_checkpoint_layers = -1
                        logger.info(f"[{exp_id}] Gradient checkpointing enabled")

                    target_suffixes = FOLEY_TARGET_PRESETS[config["target"]]
                    n_wrapped = apply_lora(
                        model, rank=config["rank"], alpha=config["alpha"],
                        target_suffixes=target_suffixes,
                        dropout=config["lora_dropout"],
                        init_mode=config["init_mode"],
                        use_rslora=config["use_rslora"],
                    )

                    for name, param in model.named_parameters():
                        param.requires_grad = "lora_" in name

                    # Optimizer
                    _lr = config["lr"]
                    if config["lora_plus_ratio"] > 1.0:
                        a_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_A" in n]
                        b_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
                        param_groups = [{"params": a_params, "lr": _lr}, {"params": b_params, "lr": _lr * config["lora_plus_ratio"]}]
                    else:
                        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": _lr}]

                    _opt_type = config.get("optimizer_type", "prodigy")
                    if _opt_type == "prodigy":
                        from prodigyopt import Prodigy
                        for pg in param_groups:
                            pg.pop("lr", None)
                        _d_coef = config.get("prodigy_d_coef", 1.0)
                        _growth = config.get("prodigy_growth_rate", 0.0)
                        _growth = float("inf") if _growth <= 0 else _growth
                        optimizer = Prodigy(param_groups, lr=1.0, betas=(0.9, 0.999), weight_decay=0.01,
                                            d_coef=_d_coef, growth_rate=_growth)
                        logger.info(f"[{exp_id}] Using Prodigy optimizer (d_coef={_d_coef}, growth_rate={_growth})")
                    elif _opt_type == "automagic":
                        from lora.automagic import Automagic
                        if config.get("lora_plus_ratio", 1.0) > 1.0:
                            logger.warning(f"[{exp_id}] lora_plus_ratio has no effect with Automagic")
                        for pg in param_groups:
                            pg.pop("lr", None)
                        optimizer = Automagic(param_groups, lr=1e-6, min_lr=1e-7, max_lr=1e-3, lr_bump=1e-6)
                        logger.info(f"[{exp_id}] Using Automagic optimizer")
                    else:
                        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=0.01)

                    _ga = config["grad_accum"]
                    def lr_lambda(sched_step):
                        actual_step = sched_step * _ga
                        if actual_step < config["warmup_steps"]:
                            return actual_step / max(config["warmup_steps"], 1)
                        if config["schedule_type"] == "cosine":
                            progress = (actual_step - config["warmup_steps"]) / max(config["steps"] - config["warmup_steps"], 1)
                            return 0.5 * (1 + np.cos(np.pi * progress))
                        return 1.0

                    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

                    # Resume from checkpoint
                    start_step = 0
                    _resumed_ema = None
                    resume_path = config.get("resume_from", "")
                    if resume_path and os.path.exists(resume_path):
                        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
                        load_lora(model, ckpt["state_dict"])
                        if "optimizer" in ckpt:
                            optimizer.load_state_dict(ckpt["optimizer"])
                        if "scheduler" in ckpt:
                            lr_sched.load_state_dict(ckpt["scheduler"])
                        start_step = ckpt.get("step", 0)
                        _resumed_ema = ckpt.get("ema_state", None)
                        # steps field means additional steps when resuming
                        if start_step >= config["steps"]:
                            config["steps"] = start_step + config["steps"]
                        logger.info(f"[{exp_id}] Resumed from step {start_step}: {resume_path}")
                        del ckpt

                    # EMA
                    _ema_decay = config.get("ema_decay", 0.0)
                    ema_state = None
                    if _ema_decay > 0:
                        if _resumed_ema is not None:
                            ema_state = {k: v.to(device) for k, v in _resumed_ema.items()}
                            logger.info(f"[{exp_id}] EMA restored from checkpoint (decay={_ema_decay})")
                            del _resumed_ema
                        else:
                            ema_state = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
                            logger.info(f"[{exp_id}] EMA initialized fresh (decay={_ema_decay})")

                    # Channel-weighted loss
                    _channel_weights = None
                    if config.get("channel_loss_weight", False):
                        _all_lat = torch.cat([d["latents"] for d in dataset], dim=0)
                        _ch_var = _all_lat.var(dim=(0, 2))
                        _channel_weights = (_ch_var / _ch_var.mean()).clamp(0.5, 2.0)
                        logger.info(f"[{exp_id}] Channel weights: min={_channel_weights.min():.2f} max={_channel_weights.max():.2f}")
                        del _all_lat

                    import random
                    torch.manual_seed(config["seed"])
                    random.seed(config["seed"])
                    np.random.seed(config["seed"])

                    losses = []
                    metrics_history = []

                    # Load existing loss/metrics history when resuming
                    if start_step > 0:
                        # Check local dir first, then source checkpoint dir
                        resume_dir = Path(resume_path).parent if resume_path else exp_dir
                        for search_dir in [exp_dir, resume_dir]:
                            loss_file = search_dir / "loss_history.json"
                            if loss_file.exists():
                                with open(loss_file) as f:
                                    losses = json.load(f)
                                logger.info(f"[{exp_id}] Loaded {len(losses)} loss entries from {search_dir}")
                                break
                        for search_dir in [exp_dir, resume_dir]:
                            metrics_file = search_dir / "metrics_history.json"
                            if metrics_file.exists():
                                with open(metrics_file) as f:
                                    metrics_history = json.load(f)
                                logger.info(f"[{exp_id}] Loaded {len(metrics_history)} metrics entries from {search_dir}")
                                break

                    n_clips = len(dataset)
                    t_start = time.time()
                    pbar_train = comfy.utils.ProgressBar(config["steps"] - start_step)

                    # Load reference audio for metrics (DAC round-trip)
                    ref_entry = dataset[0]
                    ref_wav_np = None
                    for ext in (".flac", ".wav", ".ogg"):
                        candidate = Path(data_dir) / f"{ref_entry['name']}{ext}"
                        if candidate.exists():
                            import soundfile as _sf
                            _raw, _sr = _sf.read(str(candidate))
                            if _raw.ndim > 1:
                                _raw = _raw.mean(axis=1)
                            if _sr != 48000:
                                import soxr as _soxr
                                _raw = _soxr.resample(_raw[:, None], _sr, 48000, quality="VHQ").squeeze(-1)
                            with torch.no_grad():
                                hunyuan_deps.dac_model.to(device)
                                _ref_t = torch.from_numpy(_raw).float().unsqueeze(0).unsqueeze(0)
                                _ref_t = _ref_t.to(device=device, dtype=torch.float32)
                                _z, _, _, _, _ = hunyuan_deps.dac_model.encode(_ref_t)
                                _ref_dec = hunyuan_deps.dac_model.decode(_z.mode())
                                ref_wav_np = _ref_dec.squeeze().cpu().numpy()
                                hunyuan_deps.dac_model.cpu()
                            samples_dir_ref = exp_dir / "samples"
                            samples_dir_ref.mkdir(exist_ok=True)
                            _save_spectrogram(ref_wav_np, 48000, samples_dir_ref / "reference")
                            break

                    # Step-0 eval: generate sample before any training (skip if resuming)
                    samples_dir_0 = exp_dir / "samples"
                    samples_dir_0.mkdir(exist_ok=True)
                    if start_step == 0:
                        model.eval()
                        wav0, sr0 = generate_eval_sample(
                            model, hunyuan_deps.dac_model, dataset[0], device, dtype,
                        )
                        wav0_mono = wav0.squeeze()
                        wav0_t = torch.from_numpy(wav0)
                        if wav0_t.ndim == 1:
                            wav0_t = wav0_t.unsqueeze(0)
                        _save_wav(samples_dir_0 / "step_00000.wav", wav0_t, sr0)
                        _save_spectrogram(wav0_mono, sr0, samples_dir_0 / "step_00000")
                        logger.info(f"[{exp_id}] Step 0 eval sample saved")

                    # Step-0 validation eval (skip if resuming)
                    if start_step == 0 and val_entry is not None:
                        wav0v, sr0v = generate_eval_sample(
                            model, hunyuan_deps.dac_model, val_entry, device, dtype,
                        )
                        wav0v_mono = wav0v.squeeze()
                        wav0v_t = torch.from_numpy(wav0v)
                        if wav0v_t.ndim == 1:
                            wav0v_t = wav0v_t.unsqueeze(0)
                        _save_wav(samples_dir_0 / "val_step_00000.wav", wav0v_t, sr0v)
                        _save_spectrogram(wav0v_mono, sr0v, samples_dir_0 / "val_step_00000")
                        if val_ref_wav is not None:
                            _save_spectrogram(val_ref_wav, 48000, samples_dir_0 / "val_reference")
                        logger.info(f"[{exp_id}] Step 0 val sample saved")

                    model.train()

                    for step in range(start_step, config["steps"]):
                        mm.throw_exception_if_processing_interrupted()
                        # Skip flag
                        skip_flag = output_root / "skip_current.flag"
                        if skip_flag.exists():
                            logger.info(f"Skip flag detected for {exp_id} at step {step}")
                            ckpt_path = exp_dir / f"adapter_cancelled_step{step:05d}.pt"
                            meta = {**config, "steps_completed": step, "prompts": prompts_list}
                            save_checkpoint(model, optimizer, lr_sched, step, meta, ckpt_path)
                            skip_flag.unlink()
                            raise _SkipExperiment(f"Skipped at step {step}")

                        model.train()
                        bs = config["batch_size"]
                        indices = [np.random.randint(0, n_clips) for _ in range(bs)]
                        batch_latents = torch.cat([dataset[i]["latents"] for i in indices]).to(device, dtype=dtype)
                        batch_clip = torch.cat([dataset[i]["clip_features"] for i in indices])
                        batch_sync = torch.cat([dataset[i]["sync_features"] for i in indices])
                        batch_text = torch.cat([dataset[i]["text_embedding"] for i in indices])

                        # Pad sync to multiple of 8
                        sync_len = batch_sync.shape[1]
                        pad_sync = ((sync_len + 7) // 8) * 8 - sync_len
                        if pad_sync > 0:
                            batch_sync = F.pad(batch_sync, (0, 0, 0, pad_sync))

                        _noise_offset = config.get("noise_offset", 0.0)
                        if _noise_offset > 0:
                            offset = torch.randn(batch_latents.shape[0], batch_latents.shape[1], 1, device=device, dtype=dtype) * _noise_offset
                            batch_latents = batch_latents + offset

                        t = sample_timesteps(
                            bs, config["timestep_mode"], device, dtype,
                            sigma=config["logit_normal_sigma"],
                            curriculum_switch=config["curriculum_switch"],
                            step=step, total_steps=config["steps"],
                            t_min=config.get("t_min", 0.0),
                            t_max=config.get("t_max", 1.0),
                        )

                        loss = flow_matching_loss(
                            model, batch_latents, t, batch_clip, batch_sync, batch_text,
                            device, dtype,
                            visual_dropout_prob=config.get("visual_dropout_prob", 0.0),
                            min_snr_gamma=config.get("min_snr_gamma", 0.0),
                            cos_sim_weight=config.get("cos_sim_weight", 0.0),
                            channel_weights=_channel_weights,
                        )
                        loss = loss / config["grad_accum"]
                        loss.backward()

                        if (step + 1) % config["grad_accum"] == 0:
                            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                            optimizer.step()
                            lr_sched.step()
                            optimizer.zero_grad()

                            if ema_state is not None:
                                with torch.no_grad():
                                    for n, p in model.named_parameters():
                                        if p.requires_grad and n in ema_state:
                                            ema_state[n].mul_(_ema_decay).add_(p.data, alpha=1 - _ema_decay)

                        losses.append(loss.item() * config["grad_accum"])

                        if (step + 1) % 50 == 0:
                            avg_loss = np.mean(losses[-50:])
                            elapsed = time.time() - t_start
                            _lr_str = (f"{optimizer.get_avg_learning_rate():.2e}"
                                       if hasattr(optimizer, "get_avg_learning_rate")
                                       else f"{lr_sched.get_last_lr()[0]:.2e}")
                            logger.info(f"[{exp_id}] Step {step+1}/{config['steps']} | "
                                       f"loss: {avg_loss:.4f} | lr: {_lr_str} | "
                                       f"elapsed: {elapsed:.0f}s")

                            preview_img = _draw_loss_curve(
                                losses,
                                smoothed=_smooth_losses(losses),
                            )
                            pbar_train.update_absolute(
                                step + 1 - start_step, config["steps"] - start_step,
                                ("JPEG", preview_img, 800),
                            )

                        if (step + 1) % config["save_every"] == 0:
                            # Save with live weights for optimizer consistency on resume
                            meta = {**config, "steps_completed": step + 1, "prompts": prompts_list}
                            ckpt_path = exp_dir / f"adapter_step{step+1:05d}.pt"
                            save_checkpoint(model, optimizer, lr_sched, step + 1, meta, ckpt_path,
                                            ema_state=ema_state)

                            # Swap in EMA weights for eval
                            if ema_state is not None:
                                _live_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
                                for n, p in model.named_parameters():
                                    if p.requires_grad and n in ema_state:
                                        p.data.copy_(ema_state[n])
                            _draw_loss_curve(losses, smoothed=_smooth_losses(losses)).save(str(exp_dir / "loss.png"))

                            # Generate eval audio sample + compute metrics
                            samples_dir = exp_dir / "samples"
                            samples_dir.mkdir(exist_ok=True)
                            model.eval()
                            wav, sr = generate_eval_sample(
                                model, hunyuan_deps.dac_model, dataset[0], device, dtype,
                            )
                            wav_mono = wav.squeeze()
                            wav_t = torch.from_numpy(wav)
                            if wav_t.ndim == 1:
                                wav_t = wav_t.unsqueeze(0)
                            _save_wav(samples_dir / f"step_{step+1:05d}.wav", wav_t, sr)
                            _save_spectrogram(wav_mono, sr, samples_dir / f"step_{step+1:05d}")

                            sm = spectral_metrics(wav_mono, sr)
                            step_metrics = {"step": step + 1, "loss": float(np.mean(losses[-50:])), **sm}
                            if ref_wav_np is not None:
                                rm = reference_metrics(wav_mono, ref_wav_np, sr)
                                step_metrics.update(rm)
                            metrics_history.append(step_metrics)
                            with open(exp_dir / "metrics_history.json", "w") as _mf:
                                json.dump(metrics_history, _mf, indent=2)

                            # Validation sample (external)
                            if val_entry is not None:
                                wav_v, sr_v = generate_eval_sample(
                                    model, hunyuan_deps.dac_model, val_entry, device, dtype,
                                )
                                wav_v_mono = wav_v.squeeze()
                                wav_v_t = torch.from_numpy(wav_v)
                                if wav_v_t.ndim == 1:
                                    wav_v_t = wav_v_t.unsqueeze(0)
                                _save_wav(samples_dir / f"val_step_{step+1:05d}.wav", wav_v_t, sr_v)
                                _save_spectrogram(wav_v_mono, sr_v, samples_dir / f"val_step_{step+1:05d}")

                            # Restore live (non-EMA) weights for continued training
                            if ema_state is not None:
                                for n, p in model.named_parameters():
                                    if p.requires_grad and n in _live_params:
                                        p.data.copy_(_live_params[n])

                            model.train()

                            logger.info(f"[{exp_id}] Step {step+1}: "
                                       f"loss={step_metrics['loss']:.4f}  "
                                       f"LSD={step_metrics.get('log_spectral_distance_db', 0):.2f}dB  "
                                       f"MCD={step_metrics.get('mel_cepstral_distortion', 0):.2f}  "
                                       f"HF={step_metrics.get('hf_energy_ratio', 0):.3f}  "
                                       f"SC={step_metrics.get('spectral_convergence', 0):.3f}")

                    # Save final (with EMA weights if enabled)
                    if ema_state is not None:
                        for n, p in model.named_parameters():
                            if p.requires_grad and n in ema_state:
                                p.data.copy_(ema_state[n])

                    meta = {**config, "steps_completed": config["steps"], "prompts": prompts_list}
                    final_path = exp_dir / "adapter_final.pt"
                    save_checkpoint(model, optimizer, lr_sched, config["steps"], meta, final_path, final=True)
                    # Draw and save per-experiment loss curve
                    smoothed = _smooth_losses(losses)
                    loss_img = _draw_loss_curve(losses, smoothed=smoothed)
                    loss_img.save(str(exp_dir / "loss.png"))

                    # Save loss + metrics history
                    with open(exp_dir / "loss_history.json", "w") as f:
                        json.dump(losses, f)
                    if metrics_history:
                        with open(exp_dir / "metrics_history.json", "w") as f:
                            json.dump(metrics_history, f, indent=2)
                    all_loss_histories[exp_id] = losses

                    elapsed = time.time() - t_start
                    final_metrics = metrics_history[-1] if metrics_history else {}
                    exp_result.update({
                        "status": "completed",
                        "final_loss": float(np.mean(losses[-100:])) if losses else 0.0,
                        "min_loss": float(min(losses)) if losses else 0.0,
                        "final_metrics": final_metrics,
                        "adapter_path": str(final_path),
                        "duration_seconds": elapsed,
                    })

            except _SkipExperiment as e:
                exp_result["status"] = f"skipped: {e}"
            except Exception as e:
                import traceback
                exp_result["status"] = f"failed: {e}"
                logger.error(f"Experiment {exp_id} failed: {e}\n{traceback.format_exc()}")

            gc.collect()
            torch.cuda.empty_cache()

            results.append(exp_result)

            # Save summary after each experiment
            summary = {
                "name": sweep_name, "data_dir": data_dir,
                "experiments": results,
                "system": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A",
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                },
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

        # Generate comparison chart and save to temp for inline preview
        curve_data = [{"id": eid, "loss_history": lh} for eid, lh in all_loss_histories.items()]
        comparison_img = _draw_comparison_curves(curve_data)
        comparison_img.save(str(output_root / "loss_comparison.png"))
        comparison_tensor = _pil_to_tensor(comparison_img)

        # Save to ComfyUI temp dir for inline node preview
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = f"lora_sweep_comparison_{sweep_name}.png"
        comparison_img.save(os.path.join(temp_dir, temp_file))

        logger.info(f"Sweep complete: {len(results)} experiments")
        return {"ui": {"images": [{"filename": temp_file, "subfolder": "", "type": "temp"}]},
                "result": (str(summary_path), comparison_tensor)}


class _SkipExperiment(Exception):
    pass



# --- Node 5: LoRA Evaluator -------------------------------------------------

class FoleyTuneLoRAEvaluator:
    """Compare multiple LoRA adapters by generating audio and computing spectral metrics."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("FOLEYTUNE_MODEL",),
                "hunyuan_deps": ("FOLEYTUNE_DEPS",),
                "eval_json": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "evaluate"
    CATEGORY = "FoleyTune"
    OUTPUT_NODE = True

    def evaluate(self, hunyuan_model, hunyuan_deps, eval_json):
        if not os.path.exists(eval_json):
            raise FileNotFoundError(f"Eval JSON not found: {eval_json}")

        with open(eval_json) as f:
            spec = json.load(f)

        data_dir = spec.get("data_dir", "")
        dataset_json = spec.get("dataset_json", "")
        output_dir = Path(spec.get("output_dir", "lora_eval"))
        num_steps = spec.get("steps", 25)
        seed = spec.get("seed", 42)
        adapters = spec.get("adapters", [])

        output_dir.mkdir(parents=True, exist_ok=True)

        device = mm.get_torch_device()
        dtype = torch.bfloat16

        # Prepare dataset — support dataset_json for train/val split
        clip_names = None
        if dataset_json and os.path.exists(dataset_json):
            with open(dataset_json) as f:
                ds_cfg = json.load(f)
            if not isinstance(ds_cfg.get("train"), list):
                raise ValueError("dataset_json must contain a 'train' key with a list of clip names")
            data_dir = str(Path(dataset_json).parent)
            clip_names = ds_cfg["train"]
        elif not data_dir:
            raise ValueError("Eval JSON must specify either 'data_dir' or 'dataset_json'")
        dataset = prepare_dataset(data_dir, hunyuan_deps.dac_model, device, dtype,
                                  clip_names=clip_names)

        # Compute reference metrics from original audio
        ref_dir = output_dir / "reference"
        ref_dir.mkdir(exist_ok=True)
        ref_metrics_list = []

        for entry in dataset:
            # Load original audio for reference
            audio_exts = (".wav", ".flac", ".ogg", ".aiff", ".aif")
            ref_path = None
            for ext in audio_exts:
                candidate = Path(data_dir) / f"{entry['name']}{ext}"
                if candidate.exists():
                    ref_path = candidate
                    break
            if ref_path:
                import soundfile as sf_eval
                import soxr as soxr_eval
                raw_np, ref_sr = sf_eval.read(str(ref_path))  # [L] or [L, C]
                if raw_np.ndim == 1:
                    raw_np = raw_np[:, None]
                if ref_sr != 48000:
                    raw_np = soxr_eval.resample(raw_np, ref_sr, 48000, quality="VHQ")
                ref_wav = torch.from_numpy(raw_np.T).float()  # [C, L]
                ref_wav_np = ref_wav.mean(dim=0).numpy()  # mono
                ref_m = spectral_metrics(ref_wav_np, 48000)
                prompt = entry.get("prompt", "")
                if prompt:
                    ref_m["clap_similarity"] = clap_similarity(ref_wav_np, 48000, prompt, device)
                ref_metrics_list.append(ref_m)
                # Save reference
                _save_wav(ref_dir / f"{entry['name']}.wav", ref_wav.mean(dim=0, keepdim=True), 48000)

        ref_avg = {}
        if ref_metrics_list:
            for key in ref_metrics_list[0]:
                ref_avg[key] = float(np.mean([m[key] for m in ref_metrics_list]))

        # Evaluate each adapter
        adapter_results = []

        for adapter_spec in adapters:
            adapter_id = adapter_spec.get("id", "unknown")
            adapter_path = adapter_spec.get("path", None)

            logger.info(f"Evaluating adapter: {adapter_id}")
            adapter_dir = output_dir / adapter_id
            adapter_dir.mkdir(exist_ok=True)

            # Load adapter or use baseline
            if adapter_path and os.path.exists(adapter_path):
                ckpt = _load_adapter_checkpoint(adapter_path)
                sd = ckpt.get("state_dict", ckpt)
                meta = ckpt.get("meta", {})

                model = copy.deepcopy(hunyuan_model)
                rank = meta.get("rank", 16)
                alpha_val = meta.get("alpha", float(rank))
                target = meta.get("target", "all_attn_mlp")
                target_suffixes = FOLEY_TARGET_PRESETS.get(target, FOLEY_TARGET_PRESETS["all_attn_mlp"])

                apply_lora(model, rank=rank, alpha=alpha_val, target_suffixes=target_suffixes,
                           init_mode="standard", use_rslora=meta.get("use_rslora", False))
                load_lora(model, sd)
            else:
                model = copy.deepcopy(hunyuan_model)
                meta = {}

            model.to(device=device, dtype=dtype)
            model.eval()

            clip_metrics_list = []
            clips = []

            for ci, entry in enumerate(dataset):
                wav, sr = generate_eval_sample(
                    model, hunyuan_deps.dac_model, entry, device, dtype,
                    num_steps=num_steps, seed=seed,
                )
                wav_mono = wav.squeeze()
                sm = spectral_metrics(wav_mono, sr)

                # CLAP similarity: does the generated audio match the prompt?
                prompt = entry.get("prompt", "")
                if prompt:
                    cs = clap_similarity(wav_mono, sr, prompt, device)
                    sm["clap_similarity"] = cs

                clip_metrics_list.append(sm)

                wav_path = adapter_dir / f"{entry['name']}.wav"
                wav_t = torch.from_numpy(wav)
                if wav_t.ndim == 1:
                    wav_t = wav_t.unsqueeze(0)
                _save_wav(wav_path, wav_t, sr)
                clips.append({"clip": entry["name"], "wav_path": str(wav_path), "spectral_metrics": sm})

            avg_metrics = {}
            if clip_metrics_list:
                for key in clip_metrics_list[0]:
                    avg_metrics[key] = float(np.mean([m[key] for m in clip_metrics_list]))

            adapter_results.append({
                "id": adapter_id, "path": adapter_path, "meta": meta,
                "clips": clips, "avg_metrics": avg_metrics, "status": "completed",
            })

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # Save summary
        summary = {
            "name": spec.get("name", "eval"),
            "data_dir": data_dir, "output_dir": str(output_dir),
            "n_clips": len(dataset), "steps": num_steps, "seed": seed,
            "reference_avg": ref_avg,
            "adapters": adapter_results,
        }
        with open(output_dir / "eval_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Comparison chart
        _save_eval_chart(ref_avg, adapter_results, output_dir / "metric_comparison.png")

        logger.info(f"Evaluation complete: {len(adapter_results)} adapters")
        return ()


def _save_eval_chart(ref_avg, adapter_results, path):
    """2x2 bar chart comparing spectral metrics across adapters."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    metrics_to_plot = ["hf_energy_ratio", "spectral_centroid_hz", "spectral_flatness", "temporal_variance"]
    titles = ["HF Energy Ratio (>4kHz)", "Spectral Centroid (Hz)", "Spectral Flatness", "Temporal Variance"]

    ids = ["reference"] + [a["id"] for a in adapter_results]
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, metric, title in zip(axes.flat, metrics_to_plot, titles):
        values = [ref_avg.get(metric, 0)]
        for a in adapter_results:
            values.append(a["avg_metrics"].get(metric, 0))

        bars = ax.barh(ids, values, color=[colors[i % len(colors)] for i in range(len(ids))])
        for bar, val in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f" {val:.4f}", va="center", fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


# --- Node 8: Checkpoint Finalizer --------------------------------------------


class FoleyTuneCheckpointFinalizer:
    """Strip optimizer/scheduler state from a training checkpoint.

    Converts an intermediate checkpoint (with resume data) into a final
    adapter file — smaller and faster to load for inference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a training checkpoint .pt file.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_path",)
    FUNCTION = "finalize"
    CATEGORY = "FoleyTune"
    OUTPUT_NODE = True

    def finalize(self, checkpoint_path):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "state_dict" not in ckpt:
            raise ValueError("Not a valid training checkpoint (no state_dict)")

        state_dict = ckpt["state_dict"]
        if "ema_state" in ckpt:
            for key, ema_val in ckpt["ema_state"].items():
                if key in state_dict:
                    state_dict[key] = ema_val

        final = {"state_dict": state_dict, "meta": ckpt.get("meta", {})}

        removed = []
        for key in ("optimizer", "scheduler", "step", "ema_state"):
            if key in ckpt:
                removed.append(key)

        out_path = checkpoint_path.replace(".pt", "_final.pt")
        torch.save(final, out_path)

        size_before = os.path.getsize(checkpoint_path) / (1024 * 1024)
        size_after = os.path.getsize(out_path) / (1024 * 1024)
        logger.info(f"Finalized checkpoint: {size_before:.1f} MB -> {size_after:.1f} MB "
                    f"(removed: {', '.join(removed) or 'nothing'})")

        return (out_path,)


# --- Node Mappings -----------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "FoleyTuneFeatureExtractor": FoleyTuneFeatureExtractor,
    "FoleyTuneBatchFeatureExtractor": FoleyTuneBatchFeatureExtractor,
    "FoleyTuneLoRATrainer": FoleyTuneLoRATrainer,
    "FoleyTuneLoRALoader": FoleyTuneLoRALoader,
    "FoleyTuneLoRAScheduler": FoleyTuneLoRAScheduler,
    "FoleyTuneLoRAEvaluator": FoleyTuneLoRAEvaluator,
    "FoleyTuneVAERoundtrip": FoleyTuneVAERoundtrip,
    "FoleyTuneCheckpointFinalizer": FoleyTuneCheckpointFinalizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoleyTuneFeatureExtractor": "FoleyTune Feature Extractor",
    "FoleyTuneBatchFeatureExtractor": "FoleyTune Batch Feature Extractor",
    "FoleyTuneLoRATrainer": "FoleyTune LoRA Trainer",
    "FoleyTuneLoRALoader": "FoleyTune LoRA Loader",
    "FoleyTuneLoRAScheduler": "FoleyTune LoRA Scheduler",
    "FoleyTuneLoRAEvaluator": "FoleyTune LoRA Evaluator",
    "FoleyTuneVAERoundtrip": "FoleyTune VAE Roundtrip",
    "FoleyTuneCheckpointFinalizer": "FoleyTune Checkpoint Finalizer",
}
