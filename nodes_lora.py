"""LoRA training nodes for FoleyTune."""

import os
import sys
import gc
import copy
import json
import time
import signal
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
    apply_lora, get_lora_state_dict, load_lora, merge_lora_into_weights,
    FOLEY_TARGET_PRESETS, LoRALinear, LoRAConv1d,
)
from .lora.train import (
    prepare_dataset, prepare_single_entry, harmonize_dataset,
    sample_timesteps, flow_matching_loss,
    generate_eval_sample, save_checkpoint, save_meta_json,
    visual_dropout_curriculum, compute_channel_weights,
)
from .lora.spectral_metrics import spectral_metrics, reference_metrics, clap_similarity
from PIL import Image, ImageDraw


def _load_adapter_checkpoint(path: str) -> dict:
    """Load a LoRA checkpoint from .safetensors or .pt format.

    Used by inference paths (loader/evaluator), NOT by training resume.
    Schedule-free checkpoints carry raw train-mode weights in `state_dict`
    (for resume) and the averaged weights in `eval_state_dict` — prefer the
    latter so loading a mid-training checkpoint matches its eval sample.
    """
    if path.endswith(".safetensors"):
        state_dict = load_safetensors(path)
        json_path = path.replace(".safetensors", ".json")
        meta = {}
        if os.path.exists(json_path):
            with open(json_path) as f:
                meta = json.load(f)
        return {"state_dict": state_dict, "meta": meta}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "eval_state_dict" in ckpt:
        ckpt = {**ckpt, "state_dict": ckpt["eval_state_dict"]}
        logger.info("Using schedule-free averaged (eval-mode) weights from checkpoint")
    return ckpt

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


def _draw_loss_curve(losses, log_interval=1, start_step=0, smoothed=None, metrics_history=None):
    """Render a loss curve as a PIL Image, with optional HF energy ratio on right axis."""
    hf_points = []
    if metrics_history:
        hf_points = [(m["step"], m["hf_energy_ratio"]) for m in metrics_history if "hf_energy_ratio" in m]

    W, H = 800, 380
    pr = 70 if hf_points else 20
    pl, pt, pb = 70, 25, 45

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

        if hf_points:
            hf_lo = min(v for _, v in hf_points)
            hf_hi = max(v for _, v in hf_points)
            if hf_hi == hf_lo:
                hf_hi = hf_lo + 0.001
            hf_rng = hf_hi - hf_lo
            step_range = last_step - first_step if last_step > first_step else 1

            for i in range(5):
                y = pt + int(i * ph / 4)
                val = hf_hi - i * hf_rng / 4
                draw.text((W - pr + 5, y - 7), f"{val:.4f}", fill=(230, 126, 34))

            draw.line([(W - pr, pt), (W - pr, H - pb)], fill=(230, 126, 34), width=1)

            hf_px = []
            for s, v in hf_points:
                frac = max(0.0, min(1.0, (s - first_step) / step_range))
                x = pl + int(frac * pw)
                y = pt + int((1.0 - (v - hf_lo) / hf_rng) * ph)
                hf_px.append((x, y))

            if len(hf_px) >= 2:
                draw.line(hf_px, fill=(230, 126, 34), width=2)
            for x, y in hf_px:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(230, 126, 34))

            draw.text((W - pr - 20, 5), "HF", fill=(230, 126, 34))

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
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0, "step": 0.1,
                              "tooltip": "Ignored when video_features is connected (fps comes from the video loader)."}),
                "duration": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 3600.0, "step": 0.1,
                              "tooltip": "Ignored when video_features is connected. Connect the loader's duration output to keep them in sync."}),
                "cache_dir": ("STRING", {"default": "",
                              "tooltip": "Directory for the .npz feature caches. Leave empty to use ComfyUI's temp folder (cleared on restart)."}),
                "name": ("STRING", {"default": "clip",
                          "tooltip": "Base name for auto-incremented files (e.g. clip -> clip_001.npz)"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video_features": ("FOLEYTUNE_VIDEO_FEATURES",),
            },
        }

    RETURN_TYPES = ("STRING", "FOLEYTUNE_FEATURES")
    RETURN_NAMES = ("npz_path", "features")
    FUNCTION = "extract_features"
    CATEGORY = "FoleyTune"
    OUTPUT_NODE = True

    def extract_features(self, hunyuan_deps, prompt, negative_prompt,
                         frame_rate, duration, cache_dir, name,
                         image=None, video_features=None):
        from hunyuanvideo_foley.utils.feature_utils import (
            encode_video_with_siglip2, encode_video_with_sync, encode_text_feat,
        )

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        # An empty cache_dir would resolve to the process CWD (the ComfyUI root),
        # littering it with .npz files. Fall back to ComfyUI's temp directory.
        cache_dir = cache_dir.strip()
        cache_dir = Path(cache_dir) if cache_dir else Path(folder_paths.get_temp_directory())
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Auto-increment filename
        idx = 1
        while (cache_dir / f"{name}_{idx:03d}.npz").exists():
            idx += 1
        npz_path = cache_dir / f"{name}_{idx:03d}.npz"

        if video_features is not None:
            clip_features = video_features["clip_feat"]
            sync_features = video_features["sync_feat"]
            duration = video_features["duration"]
            frame_rate = video_features["fps"]
        elif image is not None:
            # -- Extract visual features --
            # image is [T, H, W, C] float32 in [0,1] from ComfyUI
            # Preprocessing pipelines accept float32 directly (ToDtype is a no-op),
            # so we skip the full-resolution uint8 copy that would double RAM usage.
            frames_tchw = image.permute(0, 3, 1, 2)  # [T, C, H, W] float32, view (no copy)
            total_frames = frames_tchw.shape[0]

            if duration <= 0:
                duration = total_frames / frame_rate
                logger.warning(f"Auto-detected duration={duration:.2f}s from {total_frames} frames at {frame_rate}fps. "
                               f"Set duration explicitly if this is wrong (e.g. mismatched fps).")

            # SigLIP2: 8fps, subsample then preprocess (resize to 512x512)
            siglip2_indices = torch.linspace(0, total_frames - 1, max(1, int(duration * 8))).long()
            siglip2_processed = torch.stack([
                hunyuan_deps.siglip2_preprocess(frames_tchw[i]) for i in siglip2_indices
            ]).unsqueeze(0)

            hunyuan_deps.siglip2_model.to(device)
            clip_features = encode_video_with_siglip2(
                siglip2_processed.to(device), hunyuan_deps
            ).cpu()
            del siglip2_processed
            hunyuan_deps.siglip2_model.to(offload_device)

            # Synchformer: 25fps, subsample then preprocess (resize to 224x224)
            sync_indices = torch.linspace(0, total_frames - 1, max(16, int(duration * 25))).long()
            sync_processed = torch.stack([
                hunyuan_deps.syncformer_preprocess(frames_tchw[i]) for i in sync_indices
            ]).unsqueeze(0)

            hunyuan_deps.syncformer_model.to(device)
            sync_features = encode_video_with_sync(
                sync_processed.to(device), hunyuan_deps
            ).cpu()
            del sync_processed
            hunyuan_deps.syncformer_model.to(offload_device)
        else:
            raise ValueError("Either 'image' or 'video_features' must be provided")

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
        # Carry the source path + fps forward (for the timeline thumbnail strip
        # and frame-based ruler/snapping). Only the video_features path has a
        # real path; the raw-IMAGE path does not. frame_rate holds the resolved
        # fps (set from video_features["fps"] above when connected).
        if video_features is not None and video_features.get("video_path"):
            features["video_path"] = video_features["video_path"]
        features["fps"] = float(frame_rate)

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

class FoleyTuneTrainOptions:
    """Advanced/experimental settings for the FoleyTune LoRA Trainer.

    Connect to the trainer's `train_options` input to enable auxiliary losses,
    augmentation, optimizer extras, intensity sub-options, and the eval negative
    prompt. Every default here is the validated IO recipe (all off / validated
    values), so attaching this node unchanged is a no-op — only override what you
    want to experiment with.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                # ── Auxiliary losses (all off in the IO recipe) ──
                "cos_sim_weight": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Cosine-similarity aux loss on velocity (phase/correlation alignment). 0.1 to try. 0 = off.",
                }),
                "spectral_weight": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.005,
                    "tooltip": "Multi-resolution STFT aux loss on the reconstructed clean sample, 2x HF emphasis. 0.02 to try. 0 = off.",
                }),
                "hf_phase_switch": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "After this fraction of training, drop t_min/t_max clipping to train the low-noise HF regime. 0.6 to try. 0 = off.",
                }),
                "wav_spectral_weight": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Waveform-domain multi-res STFT loss (decodes a crop through DAC, >4kHz error). The only loss that sees true audio HF. 0.1 to try. 0 = off.",
                }),
                "wav_spectral_every": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Compute the waveform spectral loss every N steps (DAC decode is expensive)."}),
                "wav_spectral_crop": ("INT", {"default": 64, "min": 16, "max": 256, "step": 8,
                    "tooltip": "Latent-frame crop decoded for the waveform loss (64 ~ 1.3s). Larger = more HF context + VRAM."}),
                "wav_spectral_adaptive": ("BOOLEAN", {"default": True,
                    "tooltip": "Energy-adaptive HF weighting (up-weight low-energy bins). Off = flat HF-band L1."}),
                "temporal_variance_weight": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "SNR-gated multi-scale temporal-diff loss. 0.3 to try. 0 = off.",
                }),
                "tv_gate_sigma": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.8, "step": 0.05,
                    "tooltip": "Noise threshold for the temporal loss (fires when t < this). 0.3 default."}),
                "min_snr_gamma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 1.0,
                    "tooltip": "Min-SNR loss weighting gamma. 5.0 to try. 0 = off."}),
                "cfm_lambda": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Contrastive Flow Matching (arXiv:2506.05350). 0.05 to try. 0 = plain FM. (A wash at fixed LR in our tests.)"}),
                "channel_weight_mode": (["off", "variance", "inverse"], {"default": "off",
                    "tooltip": "Per-channel MSE weighting. 'inverse' up-weights HF-carrying low-variance channels. 'off' = uniform."}),
                "channel_loss_weight": ("BOOLEAN", {"default": False,
                    "tooltip": "DEPRECATED — use channel_weight_mode."}),
                # ── Augmentation ──
                "latent_mixup_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,
                    "tooltip": "Beta-distribution latent interpolation augmentation. 0 = off."}),
                "latent_noise_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.1,
                    "tooltip": "Additive per-element Gaussian noise on target latents. 0 = off."}),
                "ema_decay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9999, "step": 0.0001,
                    "tooltip": "EMA decay for LoRA weights, applied at save. 0.9995 to try. 0 = off."}),
                "visual_dropout_prob": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": "Per-sample probability of zeroing visual features (decouples identity from sound). 0.5 for generic-style; 0 = identity-preserving."}),
                "vd_curriculum_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Ramp visual dropout from 10%% of base to full over this fraction of training. 0 = off."}),
                # ── Optimizer extras ──
                "prodigy_d_coef": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01,
                    "tooltip": "Prodigy d_coef: scales the learned step size. Prodigy/Prodigy+ only."}),
                "prodigy_growth_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Prodigy growth_rate: max multiplicative d increase per step. 0 = unlimited. Prodigy only."}),
                "prodigy_steps": ("INT", {"default": 0, "min": 0, "max": 50000, "step": 100,
                    "tooltip": "Prodigy+ steps cap for d-adaptation. 0 = off."}),
                "use_cautious": ("BOOLEAN", {"default": False,
                    "tooltip": "Prodigy+ cautious updates (sign-aligned). Reaches quality ~1k steps earlier, brighter. Prodigy+ only."}),
                "use_orthograd": ("BOOLEAN", {"default": False,
                    "tooltip": "Prodigy+ orthograd. Tested WORSE for foley (TV collapses). Prodigy+ only."}),
                # ── LoRA extras ──
                "init_mode": (["standard", "pissa"], {"default": "standard",
                    "tooltip": "standard (Kaiming A, zero B) or pissa (SVD-based)."}),
                "use_rslora": ("BOOLEAN", {"default": False,
                    "tooltip": "Rank-stabilized scaling alpha/sqrt(rank) instead of alpha/rank."}),
                "lora_plus_ratio": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 32.0,
                    "tooltip": "B-matrix LR multiplier (LoRA+). >1 overfits on small/medium foley sets — keep 1.0."}),
                # ── Intensity sub-options (intensity_bias is on the main node) ──
                "intensity_metric": (["energy", "tv"], {"default": "energy",
                    "tooltip": "Intensity metric for the weighted sampler. energy = mean latent energy (raises level + cleans moan); tv = std/mean burstiness (pure dynamics)."}),
                # ── Misc ──
                "eval_negative_prompt": ("STRING", {"default": "noisy, harsh",
                    "tooltip": "Negative prompt for the eval-sample CFG uncond branch (CLAP-encoded), matching production. Empty = legacy zero uncond."}),
                "gradient_checkpointing": ("BOOLEAN", {"default": False,
                    "tooltip": "Recompute activations to save VRAM (~3-5 GB, ~25%% slower)."}),
                "freeze_blocks": ("INT", {"default": 0, "min": 0, "max": 17, "step": 1,
                    "tooltip": "Freeze the first N triple_blocks during finetuning. 0 = train all."}),
            },
        }

    RETURN_TYPES = ("TRAIN_OPTIONS",)
    RETURN_NAMES = ("train_options",)
    FUNCTION = "build"
    CATEGORY = "FoleyTune"
    DESCRIPTION = "Advanced/experimental training settings; plug into the LoRA Trainer's train_options. Defaults = IO recipe (no-op)."

    def build(self, **kwargs):
        return (dict(kwargs),)


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
                "target": (list(FOLEY_TARGET_PRESETS.keys()), {"default": "all_blocks_sync_io"}),
                "rank": ("INT", {"default": 32, "min": 4, "max": 128, "step": 4}),
                "alpha": ("FLOAT", {"default": 32.0, "min": 1.0, "max": 128.0}),
                "lr": ("FLOAT", {"default": 5e-5, "min": 1e-6, "max": 1e-2, "step": 1e-5}),
                "steps": ("INT", {"default": 7000, "min": 100, "max": 50000}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 64}),
                "grad_accum": ("INT", {"default": 1, "min": 1, "max": 32}),
                "warmup_steps": ("INT", {"default": 0, "min": 0, "max": 2000}),
                "save_every": ("INT", {"default": 250, "min": 50, "max": 10000}),
                "timestep_mode": (["uniform", "logit_normal", "curriculum"], {"default": "uniform"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "seed": ("INT", {"default": 42}),
            },
            "optional": {
                # ── The validated IO recipe (what you actually set/tune) ──
                "logit_normal_sigma": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 3.0}),
                "curriculum_switch": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 0.9}),
                "lora_dropout": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.3}),
                "schedule_type": (["constant", "cosine"], {"default": "constant"}),
                "noise_offset": ("FLOAT", {
                    "default": 0.03, "min": 0.0, "max": 0.1, "step": 0.005,
                    "tooltip": "Per-sample channel-uniform noise on latents. Improves dynamic range. 0.03 = validated recipe.",
                }),
                "t_min": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.2, "step": 0.01,
                    "tooltip": "Minimum timestep for sampling (sync clipping). 0.05 = validated.",
                }),
                "t_max": ("FLOAT", {
                    "default": 0.95, "min": 0.8, "max": 1.0, "step": 0.01,
                    "tooltip": "Maximum timestep for sampling. 0.95 = validated.",
                }),
                "t_range_mode": (["clamp", "rescale"], {
                    "default": "rescale",
                    "tooltip": "How to restrict t to [t_min, t_max]. rescale = affine-map (no boundary spikes); clamp = historical clip.",
                }),
                "optimizer_type": (["adamw", "prodigy", "prodigy_plus"], {"default": "prodigy_plus"}),
                "schedulefree_c": ("FLOAT", {
                    "default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Prodigy+ schedule-free averaging window. 20 = the validated sfc20 recipe (less averaging = sharper, runs long without washing). 0 = optimizer default. Only used with prodigy_plus.",
                }),
                "intensity_bias": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Intensity-weighted clip sampling: draw clips ~ energy^alpha, biasing toward energetic clips (cleaner moan + more dynamics). 0.5 = validated. 0 = uniform sampling.",
                }),
                "resume_from": ("STRING", {"default": ""}),
                "dataset_json": ("STRING", {
                    "default": "",
                    "tooltip": "Path to dataset.json (or comma-separated paths for multiple datasets). When set, uses train/val split instead of scanning data_dir for all .npz files.",
                }),
                # Advanced/experimental settings (aux losses, augmentation, optimizer
                # extras) live on the FoleyTune Train Options node — defaults to the IO
                # recipe (all off) when not connected.
                "train_options": ("TRAIN_OPTIONS", {
                    "tooltip": "Optional: attach a FoleyTune Train Options node to enable advanced/experimental settings (aux losses, augmentation, optimizer extras, intensity sub-options, eval negative prompt). Leave unconnected for the validated IO recipe.",
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
              cos_sim_weight=0.0, spectral_weight=0.0, hf_phase_switch=0.0,
              wav_spectral_weight=0.0, wav_spectral_every=8, wav_spectral_crop=64,
              wav_spectral_adaptive=True, channel_weight_mode="off",
              cfm_lambda=0.0,
              channel_loss_weight=False,
              temporal_variance_weight=0.0,
              tv_gate_sigma=0.3, vd_curriculum_ratio=0.0,
              t_min=0.0, t_max=1.0, t_range_mode="clamp", optimizer_type="adamw",
              visual_dropout_prob=0.0,
              gradient_checkpointing=False,
              freeze_blocks=0,
              resume_from="", dataset_json="",
              prodigy_d_coef=1.0, prodigy_growth_rate=0.0,
              eval_negative_prompt="noisy, harsh",
              schedulefree_c=20, intensity_bias=0.5, intensity_metric="energy",
              use_cautious=False, use_orthograd=False, prodigy_steps=0,
              train_options=None):

        # Advanced/experimental settings come from the optional Train Options node;
        # unconnected -> the IO-recipe defaults above (all off / validated values).
        _o = train_options or {}
        init_mode = _o.get("init_mode", init_mode)
        use_rslora = _o.get("use_rslora", use_rslora)
        lora_plus_ratio = _o.get("lora_plus_ratio", lora_plus_ratio)
        latent_mixup_alpha = _o.get("latent_mixup_alpha", latent_mixup_alpha)
        latent_noise_sigma = _o.get("latent_noise_sigma", latent_noise_sigma)
        min_snr_gamma = _o.get("min_snr_gamma", min_snr_gamma)
        ema_decay = _o.get("ema_decay", ema_decay)
        cos_sim_weight = _o.get("cos_sim_weight", cos_sim_weight)
        spectral_weight = _o.get("spectral_weight", spectral_weight)
        hf_phase_switch = _o.get("hf_phase_switch", hf_phase_switch)
        wav_spectral_weight = _o.get("wav_spectral_weight", wav_spectral_weight)
        wav_spectral_every = _o.get("wav_spectral_every", wav_spectral_every)
        wav_spectral_crop = _o.get("wav_spectral_crop", wav_spectral_crop)
        wav_spectral_adaptive = _o.get("wav_spectral_adaptive", wav_spectral_adaptive)
        channel_weight_mode = _o.get("channel_weight_mode", channel_weight_mode)
        cfm_lambda = _o.get("cfm_lambda", cfm_lambda)
        channel_loss_weight = _o.get("channel_loss_weight", channel_loss_weight)
        temporal_variance_weight = _o.get("temporal_variance_weight", temporal_variance_weight)
        tv_gate_sigma = _o.get("tv_gate_sigma", tv_gate_sigma)
        vd_curriculum_ratio = _o.get("vd_curriculum_ratio", vd_curriculum_ratio)
        visual_dropout_prob = _o.get("visual_dropout_prob", visual_dropout_prob)
        prodigy_d_coef = _o.get("prodigy_d_coef", prodigy_d_coef)
        prodigy_growth_rate = _o.get("prodigy_growth_rate", prodigy_growth_rate)
        prodigy_steps = _o.get("prodigy_steps", prodigy_steps)
        use_cautious = _o.get("use_cautious", use_cautious)
        use_orthograd = _o.get("use_orthograd", use_orthograd)
        intensity_metric = _o.get("intensity_metric", intensity_metric)
        eval_negative_prompt = _o.get("eval_negative_prompt", eval_negative_prompt)
        gradient_checkpointing = _o.get("gradient_checkpointing", gradient_checkpointing)
        freeze_blocks = _o.get("freeze_blocks", freeze_blocks)

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
            cos_sim_weight, spectral_weight, hf_phase_switch,
            wav_spectral_weight, wav_spectral_every, wav_spectral_crop,
            wav_spectral_adaptive, channel_weight_mode, cfm_lambda, channel_loss_weight,
            temporal_variance_weight, tv_gate_sigma, vd_curriculum_ratio,
            t_min, t_max, optimizer_type,
            visual_dropout_prob,
            gradient_checkpointing, freeze_blocks, resume_from,
            dataset_json, prodigy_d_coef, prodigy_growth_rate,
            t_range_mode=t_range_mode,
            eval_negative_prompt=eval_negative_prompt,
            schedulefree_c=schedulefree_c, intensity_bias=intensity_bias,
            intensity_metric=intensity_metric, use_cautious=use_cautious,
            use_orthograd=use_orthograd, prodigy_steps=prodigy_steps,
        )

    def _train_inner(self, hunyuan_model, hunyuan_deps, data_dir, output_dir, target, rank,
                     alpha, lr, steps, batch_size, grad_accum, warmup_steps, save_every,
                     timestep_mode, precision, seed, device, dtype,
                     logit_normal_sigma, curriculum_switch, init_mode, use_rslora,
                     lora_dropout, lora_plus_ratio, schedule_type,
                     latent_mixup_alpha, latent_noise_sigma,
                     noise_offset, min_snr_gamma, ema_decay,
                     cos_sim_weight, spectral_weight, hf_phase_switch,
                     wav_spectral_weight, wav_spectral_every, wav_spectral_crop,
                     wav_spectral_adaptive, channel_weight_mode, cfm_lambda, channel_loss_weight,
                     temporal_variance_weight, tv_gate_sigma, vd_curriculum_ratio,
                     t_min, t_max, optimizer_type,
                     visual_dropout_prob,
                     gradient_checkpointing, freeze_blocks, resume_from,
                     dataset_json="",
                     prodigy_d_coef=1.0, prodigy_growth_rate=0.0,
                     t_range_mode="clamp", eval_negative_prompt="",
                     schedulefree_c=20, intensity_bias=0.5, intensity_metric="energy",
                     use_cautious=False, use_orthograd=False, prodigy_steps=0):
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

        val_entry = None
        ds_cfg = None
        dataset_jsons = [p.strip() for p in dataset_json.split(",") if p.strip()] if dataset_json else []
        _missing = [p for p in dataset_jsons if not os.path.exists(p)]
        if _missing:
            # Silently dropping missing paths would train on a partial dataset
            raise FileNotFoundError(f"dataset_json path(s) not found: {_missing}")

        if dataset_jsons:
            dataset = []
            for dj_path in dataset_jsons:
                try:
                    with open(dj_path) as f:
                        dj_cfg = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in dataset file {dj_path}: {e}") from e
                if not isinstance(dj_cfg.get("train"), list):
                    raise ValueError(f"dataset_json must contain a 'train' key: {dj_path}")
                dj_dir = str(Path(dj_path).parent)
                dj_clips = dj_cfg["train"]
                logger.info(f"Loading dataset: {dj_path} ({len(dj_clips)} clips)")
                dataset += prepare_dataset(dj_dir, hunyuan_deps.dac_model, device, dtype,
                                           clip_names=dj_clips)
                if ds_cfg is None:
                    ds_cfg = dj_cfg
                    data_dir = dj_dir
            # Per-dir lengths are uniform but can disagree ACROSS dirs, which
            # would crash torch.cat at batch-assembly time
            dataset = harmonize_dataset(dataset)
        else:
            dataset = prepare_dataset(data_dir, hunyuan_deps.dac_model, device, dtype)

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

        if freeze_blocks > 0:
            n_frozen = 0
            for name, param in model.named_parameters():
                if "lora_" in name:
                    for i in range(freeze_blocks):
                        if f"triple_blocks.{i}." in name:
                            param.requires_grad = False
                            n_frozen += 1
                            break
            logger.info(f"Froze LoRA params in blocks 0..{freeze_blocks - 1} ({n_frozen} tensors)")

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
                                d_coef=prodigy_d_coef, growth_rate=_growth, decouple=True,
                                safeguard_warmup=True, use_bias_correction=True)
            logger.info(f"Using Prodigy optimizer (d_coef={prodigy_d_coef}, growth_rate={_growth}, decouple=True, wd=0.01)")
        elif optimizer_type == "prodigy_plus":
            from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
            for pg in param_groups:
                pg.pop("lr", None)
            optimizer = ProdigyPlusScheduleFree(param_groups, lr=1.0, betas=(0.9, 0.999), weight_decay=0.01,
                                               d_coef=prodigy_d_coef, prodigy_steps=int(prodigy_steps),
                                               use_cautious=bool(use_cautious), schedulefree_c=float(schedulefree_c),
                                               use_orthograd=bool(use_orthograd))
            optimizer.train()
            logger.info(f"Prodigy+ Schedule-Free: d_coef={prodigy_d_coef}, sf_c={schedulefree_c}, "
                        f"cautious={use_cautious}, orthograd={use_orthograd}, prodigy_steps={prodigy_steps}")
            logger.info(f"Using Prodigy+ Schedule-Free (d_coef={prodigy_d_coef}, wd=0.01)")
        else:
            optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=0.01)

        # -- LR Scheduler --
        # scheduler.step() is called once per grad_accum training steps,
        # so scale the internal counter back to training steps
        _sched_type = "constant" if optimizer_type == "prodigy_plus" else schedule_type
        def lr_lambda(sched_step):
            actual_step = sched_step * grad_accum
            if actual_step < warmup_steps:
                return actual_step / max(warmup_steps, 1)
            if _sched_type == "cosine":
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
            _ckpt_opt = ckpt.get("meta", {}).get("optimizer_type", "adamw")
            _opt_match = (_ckpt_opt == optimizer_type)
            if not _opt_match:
                logger.info(f"Optimizer mismatch (ckpt={_ckpt_opt}, current={optimizer_type}) — loading weights only, fresh optimizer")
            if freeze_blocks > 0:
                logger.info(f"freeze_blocks={freeze_blocks} — fresh optimizer (param count changed)")
                _opt_match = False
            if _opt_match and "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if _opt_match and "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_step = ckpt.get("step", 0)
            _resumed_ema = ckpt.get("ema_state", None)
            if start_step >= steps:
                steps = start_step + steps
            logger.info(f"Resumed from step {start_step}: {resume_from}")
            del ckpt
            # Re-seed offset by start_step so the resumed run doesn't replay
            # the exact batch/noise/timestep sequence from step 0
            _rng_seed = (seed + start_step) % (2 ** 31)
            torch.manual_seed(_rng_seed)
            random.seed(_rng_seed)
            np.random.seed(_rng_seed)

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
            "spectral_weight": spectral_weight,
            "hf_phase_switch": hf_phase_switch,
            "wav_spectral_weight": wav_spectral_weight,
            "wav_spectral_every": wav_spectral_every,
            "wav_spectral_crop": wav_spectral_crop,
            "wav_spectral_adaptive": wav_spectral_adaptive,
            "channel_weight_mode": channel_weight_mode,
            "cfm_lambda": cfm_lambda,
            "channel_loss_weight": channel_loss_weight,
            "temporal_variance_weight": temporal_variance_weight,
            "tv_gate_sigma": tv_gate_sigma,
            "vd_curriculum_ratio": vd_curriculum_ratio,
            "t_min": t_min, "t_max": t_max, "t_range_mode": t_range_mode,
            "eval_negative_prompt": eval_negative_prompt,
            "optimizer_type": optimizer_type,
            "gradient_checkpointing": gradient_checkpointing,
            "freeze_blocks": freeze_blocks,
            "schedulefree_c": schedulefree_c, "intensity_bias": intensity_bias,
            "intensity_metric": intensity_metric, "use_cautious": use_cautious,
            "use_orthograd": use_orthograd, "prodigy_steps": prodigy_steps,
            "n_clips": n_clips, "precision": precision, "seed": seed,
        }

        # Embed training prompts (unique, sorted by frequency)
        from collections import Counter
        prompt_counts = Counter(d["prompt"] for d in dataset)
        meta["prompts"] = [p for p, _ in prompt_counts.most_common()]

        # Pre-compute per-channel loss weights. "variance" up-weights LF bulk (legacy);
        # "inverse" up-weights low-variance HF channels to counter spectral bias.
        _cw_mode = channel_weight_mode or ("variance" if channel_loss_weight else "off")
        channel_weights = None
        if _cw_mode != "off":
            all_latents = torch.cat([d["latents"] for d in dataset], dim=0)
            channel_weights = compute_channel_weights(all_latents, _cw_mode)
            logger.info(f"Channel weights ({_cw_mode}): min={channel_weights.min():.2f} max={channel_weights.max():.2f}")
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

        # Production-parity eval CFG: CLAP-encode the negative prompt for the
        # uncond branch instead of the legacy zero embedding (empty = legacy).
        # CLAP weights are inference tensors (created under ComfyUI's inference
        # mode) and cannot forward inside this inference_mode(False) region —
        # re-enter inference mode for the encode, then launder the output
        # through numpy so the embedding is a normal tensor usable in training.
        eval_uncond = None
        if eval_negative_prompt:
            with torch.inference_mode():
                hunyuan_deps.clap_model.to(device)
                _neg_inputs = hunyuan_deps.clap_tokenizer(
                    [eval_negative_prompt], padding=True, truncation=True, max_length=100,
                    return_tensors="pt",
                ).to(device)
                _neg_out = hunyuan_deps.clap_model(
                    **_neg_inputs, output_hidden_states=True, return_dict=True
                )
                _neg_emb = _neg_out.last_hidden_state.float().cpu()
                hunyuan_deps.clap_model.to(mm.unet_offload_device())
            eval_uncond = torch.from_numpy(_neg_emb.numpy().copy())
            logger.info(f"Eval uncond text: CLAP({eval_negative_prompt!r}) {tuple(eval_uncond.shape)}")

        # Waveform spectral loss needs the DAC decoder resident on GPU (frozen) for
        # differentiable decode during training.
        _wav_dac = None
        if wav_spectral_weight > 0:
            # DAC was loaded under ComfyUI inference_mode → its weights are inference
            # tensors that cannot join an autograd graph (and weight_norm recomputes
            # one on every forward, so in-place laundering is insufficient). Make a
            # frozen deepcopy outside inference_mode for the differentiable decode;
            # the shared eval DAC is left untouched.
            with torch.inference_mode(False), torch.no_grad():
                _wav_dac = copy.deepcopy(hunyuan_deps.dac_model).to(device=device).eval()
                for _p in _wav_dac.parameters():
                    _p.requires_grad_(False)
            logger.info(f"Waveform spectral loss ON: weight={wav_spectral_weight}, every={wav_spectral_every}, crop={wav_spectral_crop}")

        logger.info(f"Starting training: {steps} steps, batch {batch_size}, lr {lr}")
        t_start = time.time()

        # Optional intensity-weighted sampling (single-dataset): draw clips with
        # probability ~ intensity^alpha, biasing toward energetic/dynamic clips.
        # intensity from the DAC-latent per-frame energy envelope; 'energy' = mean,
        # 'tv' = std/mean (burstiness). Off (uniform) when intensity_bias <= 0.
        _sample_w = None
        if float(intensity_bias) > 0:
            _scores = []
            for _d in dataset:
                _env = _d["latents"][0].float().pow(2).sum(0).clamp(min=1e-12).sqrt()  # [T]
                _m = _env.mean().clamp(min=1e-8)
                _scores.append(max((_env.std() / _m).item() if intensity_metric == "tv" else _m.item(), 1e-8))
            _w = np.asarray(_scores, dtype=np.float64) ** float(intensity_bias)
            if np.isfinite(_w.sum()) and _w.sum() > 0:
                _sample_w = _w / _w.sum()
                logger.info(f"Intensity sampling: bias={intensity_bias} metric={intensity_metric} "
                            f"(top {_sample_w.max()*n_clips:.2f}x, bottom {_sample_w.min()*n_clips:.2f}x vs uniform)")

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

            # Sample batch (intensity-weighted if enabled, else uniform)
            if _sample_w is not None:
                indices = np.random.choice(n_clips, size=batch_size, p=_sample_w).tolist()
            else:
                indices = [np.random.randint(0, n_clips) for _ in range(batch_size)]
            batch_latents = torch.cat([dataset[i]["latents"] for i in indices], dim=0).to(device, dtype=dtype)
            batch_clip = torch.cat([dataset[i]["clip_features"] for i in indices], dim=0)
            batch_sync = torch.cat([dataset[i]["sync_features"] for i in indices], dim=0)
            _text_items = [dataset[i]["text_embedding"] for i in indices]
            _max_tlen = max(t.shape[1] for t in _text_items)
            batch_text = torch.cat([F.pad(t, (0, 0, 0, _max_tlen - t.shape[1])) for t in _text_items], dim=0)

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

            # Sample timesteps — widen to full range after HF phase switch
            eff_t_min, eff_t_max = t_min, t_max
            if hf_phase_switch > 0:
                progress = (step - start_step) / max(steps - start_step, 1)
                if progress >= hf_phase_switch:
                    eff_t_min, eff_t_max = 0.0, 1.0
            t = sample_timesteps(
                batch_size, timestep_mode, device, dtype,
                sigma=logit_normal_sigma, curriculum_switch=curriculum_switch,
                step=step, start_step=start_step, total_steps=steps,
                t_min=eff_t_min, t_max=eff_t_max, t_range_mode=t_range_mode,
            )

            # Forward + loss
            effective_vd = visual_dropout_curriculum(
                visual_dropout_prob, step, start_step, steps, vd_curriculum_ratio,
            )
            _do_wav = _wav_dac is not None and (step % wav_spectral_every == 0)
            loss = flow_matching_loss(
                model, batch_latents, t, batch_clip, batch_sync, batch_text, device, dtype,
                visual_dropout_prob=effective_vd,
                min_snr_gamma=min_snr_gamma,
                cos_sim_weight=cos_sim_weight,
                channel_weights=channel_weights,
                temporal_variance_weight=temporal_variance_weight,
                tv_gate_sigma=tv_gate_sigma,
                spectral_weight=spectral_weight,
                dac_model=_wav_dac, wav_spectral_weight=wav_spectral_weight,
                wav_spectral_crop=wav_spectral_crop,
                wav_spectral_adaptive=wav_spectral_adaptive,
                compute_wav_spectral=_do_wav,
                cfm_lambda=cfm_lambda,
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
                _lr_display = scheduler.get_last_lr()[0]
                _d_val = optimizer.param_groups[0].get('d')
                if _d_val is not None:
                    _lr_display = _d_val * scheduler.get_last_lr()[0]
                logger.info(f"Step {step+1}/{steps} | loss: {avg_loss:.4f} | "
                           f"lr: {_lr_display:.2e} | "
                           f"elapsed: {elapsed:.0f}s")

                preview_img = _draw_loss_curve(
                    losses, start_step=start_step,
                    smoothed=_smooth_losses(losses),
                    metrics_history=metrics_history,
                )
                pbar.update_absolute(
                    step + 1 - start_step, remaining,
                    ("JPEG", preview_img, 800),
                )

            # Save checkpoint + eval sample
            if (step + 1) % save_every == 0:
                # Schedule-free optimizers hold raw train-mode weights; also
                # capture the averaged eval-mode weights so this checkpoint
                # loads for inference exactly as the eval sample sounds
                _sf_opt = hasattr(optimizer, 'eval') and hasattr(optimizer, 'train')
                _eval_sd = None
                if _sf_opt:
                    optimizer.eval()
                    _eval_sd = get_lora_state_dict(model)
                    optimizer.train()

                # Save with live weights for optimizer consistency on resume
                ckpt_path = output_path / f"adapter_step{step+1:05d}.pt"
                save_checkpoint(model, optimizer, scheduler, step + 1, meta, ckpt_path,
                                ema_state=ema_state, eval_state=_eval_sd)
                _draw_loss_curve(losses, start_step=start_step, smoothed=_smooth_losses(losses), metrics_history=metrics_history).save(str(output_path / "loss.png"))

                # Switch schedule-free optimizer to eval mode (averaged weights)
                if _sf_opt:
                    optimizer.eval()

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
                    uncond_text_feat=eval_uncond,
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
                           f"SC={step_metrics.get('spectral_convergence', 0):.3f}  "
                           f"PBC={step_metrics.get('per_band_correlation', 0):.3f}  "
                           f"SF={step_metrics.get('spectral_flatness', 0):.4f}  "
                           f"TV={step_metrics.get('temporal_variance', 0):.3f}  "
                           f"C={step_metrics.get('spectral_centroid_hz', 0):.0f}Hz  "
                           f"R={step_metrics.get('spectral_rolloff_hz', 0):.0f}Hz")

                # Generate val sample if a val clip was loaded
                if val_entry is not None:
                    val_wav, val_sr = generate_eval_sample(
                        model, hunyuan_deps.dac_model, val_entry, device, dtype,
                        uncond_text_feat=eval_uncond,
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
                if _sf_opt:
                    optimizer.train()

        # Save metrics history
        if metrics_history:
            with open(output_path / "metrics_history.json", "w") as f:
                json.dump(metrics_history, f, indent=2)

        # -- Save final --
        _sf_final = hasattr(optimizer, 'eval') and hasattr(optimizer, 'train')
        if _sf_final:
            optimizer.eval()
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
        loss_img = _draw_loss_curve(losses, start_step=start_step, smoothed=smoothed, metrics_history=metrics_history)
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
    """Load a FoleyTune LoRA from the ComfyUI loras folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("FOLEYTUNE_MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("FOLEYTUNE_MODEL", "STRING")
    RETURN_NAMES = ("model", "prompts")
    FUNCTION = "load_adapter"
    CATEGORY = "FoleyTune"

    def load_adapter(self, hunyuan_model, lora_name, strength):
        adapter_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return FoleyTuneLoRALoaderPath._load(hunyuan_model, adapter_path, strength)


class FoleyTuneLoRALoaderPath:
    """Load a FoleyTune LoRA from an absolute path (for training/development)."""

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
        return self._load(hunyuan_model, adapter_path, strength)

    @staticmethod
    def _load(hunyuan_model, adapter_path, strength):
        # Each LoRA switch builds a fresh deepcopy of the base model. Once ComfyUI has
        # released the previously-built copy, reclaim its memory before allocating the new
        # one so alternating adapters doesn't let stale copies / reserved VRAM pile up.
        gc.collect()
        mm.soft_empty_cache()

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

        # Infer rank from lora_A tensor shapes (first one found)
        inferred_rank = None
        for k, v in state_dict.items():
            if "lora_A" in k and v.ndim >= 2:
                inferred_rank = v.shape[0]
                break
        rank = meta.get("rank", inferred_rank or 16)
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
        model._event_conditioning_enabled = bool(meta.get("event_conditioning", False))
        model._event_strength = float(meta.get("event_strength", 1.0)) * float(strength)

        # Detect if model already has a LoRA applied — if so, merge into weights
        has_existing_lora = any(isinstance(m, (LoRALinear, LoRAConv1d)) for m in model.modules())

        if has_existing_lora:
            n_applied = merge_lora_into_weights(
                model, state_dict,
                rank=rank, alpha=alpha, strength=strength,
                target_suffixes=target_suffixes,
                use_rslora=use_rslora,
            )
            event_state = {k: v for k, v in state_dict.items() if k.startswith("event_adapter.")}
            if event_state:
                model.load_state_dict(event_state, strict=False)
            model.eval()
            logger.info(f"LoRA stacked (merged into weights): {n_applied} layers, "
                        f"rank={rank}, strength={strength}")
        else:
            # First LoRA — wrap layers with LoRALinear
            n_wrapped = apply_lora(
                model, rank=rank, alpha=alpha,
                target_suffixes=target_suffixes,
                dropout=lora_dropout,
                init_mode="standard",
                use_rslora=use_rslora,
            )
            if init_mode == "pissa":
                model.load_state_dict(state_dict, strict=False)
            else:
                load_lora(model, state_dict)
            if strength != 1.0:
                for name, module in model.named_modules():
                    if hasattr(module, "lora_B"):
                        module.lora_B.data *= strength
            model.eval()
            logger.info(f"Loaded LoRA adapter: {n_wrapped} layers, rank={rank}, strength={strength}")

        prompts = "\n".join(meta.get("prompts", []))

        # Keep the freshly built adapter model on the offload device at rest; the sampler
        # moves it to GPU only when needed. Drop the adapter state dict and return any free
        # cached VRAM blocks so switching adapters doesn't accumulate in CPU/VRAM.
        model.to(mm.unet_offload_device())
        del ckpt, state_dict
        gc.collect()
        mm.soft_empty_cache()
        return (model, prompts)


# --- Node: LoRA Timeline Entry (stacker) ------------------------------------

_LORA_NONE = "(none — prompt only)"


class FoleyTuneLoRATimelineEntry:
    """Configure a timeline segment. Chain multiple entries.

    Pick a LoRA, or "(none — prompt only)" for a regular-video segment that runs
    the base model conditioned on this entry's prompt (no LoRA).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": ([_LORA_NONE] + folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "label": ("STRING", {"default": "LoRA"}),
                "color": (["red", "blue", "green", "yellow", "purple", "orange"],),
            },
            "optional": {
                "prev_entries": ("LORA_TIMELINE_ENTRIES",),
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Per-segment CLAP prompt. With a LoRA: conditions that LoRA's "
                               "segment on this prompt. With '(none — prompt only)': the segment "
                               "runs the base model on this prompt (regular-video sound, no LoRA). "
                               "Needs hunyuan_deps on the Timeline node. Describe the full audio "
                               "texture — narrow prompts hurt prompt-following.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LORA_TIMELINE_ENTRIES",)
    RETURN_NAMES = ("entries",)
    FUNCTION = "add_entry"
    CATEGORY = "FoleyTune"

    def add_entry(self, lora_name, strength, label, color, prev_entries=None, prompt="",
                  unique_id=None):
        entries = list(prev_entries) if prev_entries else []
        # "(none)" = prompt-only segment: base model + this prompt, no LoRA.
        adapter_path = (None if lora_name == _LORA_NONE
                        else folder_paths.get_full_path_or_raise("loras", lora_name))
        # Stable per-entry id = this node's graph id (persists across runs and
        # save/load), so segments keep mapping even if the chain is reordered.
        # Fall back to position when unique_id is unavailable.
        entry_id = str(unique_id) if unique_id is not None else f"idx{len(entries)}"
        entries.append({
            "id": entry_id,
            "path": adapter_path,
            "strength": strength,
            "label": label,
            "color": color,
            "prompt": prompt,
        })
        return (entries,)


# --- Node: LoRA Timeline (visual widget) ------------------------------------

def _encode_clap_prompts(hunyuan_deps, prompts):
    """CLAP-encode prompts to {prompt: [1, seq, 768] CPU tensor}.

    Mirrors FoleyTuneFeatureExtractor's text path exactly (last_hidden_state,
    max_length=100) so a per-segment prompt yields the same text_feat the
    global prompt would. Moves CLAP to device once for the whole batch.
    """
    device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    out = {}
    hunyuan_deps.clap_model.to(device)
    try:
        for p in prompts:
            inputs = hunyuan_deps.clap_tokenizer(
                [p], padding=True, truncation=True, max_length=100, return_tensors="pt",
            ).to(device)
            res = hunyuan_deps.clap_model(**inputs, output_hidden_states=True, return_dict=True)
            out[p] = res.last_hidden_state.cpu()  # [1, seq, 768]
    finally:
        hunyuan_deps.clap_model.to(offload_device)
        torch.cuda.empty_cache()
    return out


class FoleyTuneLoRATimeline:
    """Visual timeline for placing LoRA adapters on video segments."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "entries": ("LORA_TIMELINE_ENTRIES",),
                "segments_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                }),
            },
            "optional": {
                # Two ways to feed the Timeline:
                #  A) connect `features` from FoleyTuneFeatureExtractor (legacy), or
                #  B) connect `video_features` (loader) + `hunyuan_deps` and let the
                #     Timeline build features itself from base_prompt/negative_prompt.
                "features": ("FOLEYTUNE_FEATURES", {
                    "tooltip": "Optional. Connect a FoleyTuneFeatureExtractor here, OR leave it "
                               "unconnected and connect video_features + hunyuan_deps to let the "
                               "Timeline build features itself (no FeatureExtractor needed).",
                }),
                "video_features": ("FOLEYTUNE_VIDEO_FEATURES", {
                    "tooltip": "Video loader output (clip/sync/fps/duration/path). Used for the "
                               "thumbnail strip, and — when `features` is not connected — as the "
                               "visual source for self-contained feature building.",
                }),
                "hunyuan_deps": ("FOLEYTUNE_DEPS", {
                    "tooltip": "Deps for CLAP. Needed for per-segment prompts and for the "
                               "self-contained path (encoding base_prompt/negative_prompt).",
                }),
                # NOTE: new widgets are appended AFTER existing ones so saved
                # widgets_values still map positionally. Order below is fixed.
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Manual override for the thumbnail video path. Usually unnecessary if "
                               "video_features is connected. Type a path or connect a STRING.",
                }),
                "crossfade_frames": ("INT", {
                    "default": 0, "min": 0, "max": 120, "step": 1,
                    "tooltip": "(Currently inactive) Segment transitions are hard cuts on the exact "
                               "frame. Short segments are now generated with padded context and "
                               "trimmed to their frames (pad-and-trim), which uses hard cuts; "
                               "crossfade is not combined with it yet.",
                }),
                "base_prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Self-contained path only (features unconnected): the positive prompt "
                               "for regions with no per-segment prompt. Ignored if `features` is "
                               "connected (the extractor's prompt is used instead).",
                }),
                "negative_prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Self-contained path only (features unconnected): the CFG negative "
                               "prompt (global). Ignored if `features` is connected.",
                }),
            },
        }

    RETURN_TYPES = ("LORA_SCHEDULE", "FOLEYTUNE_FEATURES")
    RETURN_NAMES = ("lora_schedule", "features")
    FUNCTION = "build_schedule"
    CATEGORY = "FoleyTune"
    OUTPUT_NODE = True

    def build_schedule(self, entries, segments_json="[]", features=None, crossfade_frames=0,
                       video_features=None, hunyuan_deps=None, video_path="",
                       base_prompt="", negative_prompt=""):
        try:
            segments = json.loads(segments_json) if (segments_json or "").strip() else []
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"LoRA timeline: bad segments_json {segments_json!r} — treating as empty")
            segments = []
        if not isinstance(segments, list):
            segments = []

        # fps (loader's video_features > features > 30 fallback) drives the
        # frame-based ruler and converts crossfade_frames -> seconds.
        resolved_fps = 0.0
        if video_features and video_features.get("fps"):
            resolved_fps = float(video_features["fps"])
        elif features and features.get("fps"):
            resolved_fps = float(features["fps"])
        crossfade_sec = crossfade_frames / (resolved_fps or 30.0)

        # Resolve a segment to its entry by STABLE id first (survives chain
        # reordering), falling back to positional entry_index for older graphs.
        entries_by_id = {e["id"]: e for e in entries if e.get("id")}

        def _entry_for(seg):
            eid = seg.get("entry_id")
            if eid is not None and eid in entries_by_id:
                return entries_by_id[eid]
            ei = seg.get("entry_index", 0)
            return entries[ei] if 0 <= ei < len(entries) else None

        # Distinct non-empty per-segment prompts (each becomes a text_feat the
        # sampler swaps in per chunk). In the self-contained path, also encode
        # base_prompt/negative_prompt to build the features dict.
        used_prompts = set()
        for seg in segments:
            entry = _entry_for(seg)
            if entry:
                p = (entry.get("prompt") or "").strip()
                if p:
                    used_prompts.add(p)

        self_contained = features is None
        to_encode = set(used_prompts)
        if self_contained:
            to_encode.add(base_prompt or "")
            to_encode.add(negative_prompt or "")

        prompt_feats = {}
        if to_encode:
            if hunyuan_deps is not None:
                prompt_feats = _encode_clap_prompts(hunyuan_deps, sorted(to_encode))
                logger.info(f"LoRA timeline: encoded {len(prompt_feats)} prompt(s)")
            elif used_prompts:
                logger.warning("LoRA timeline: entries have prompts but hunyuan_deps is not "
                               "connected — per-segment prompts ignored, using the global prompt")

        # Self-contained path: build the features dict from video_features +
        # CLAP-encoded base/negative prompts (no FeatureExtractor needed).
        if self_contained:
            if video_features is None:
                raise ValueError("FoleyTune LoRA Timeline: connect either `features` (from a "
                                 "FoleyTuneFeatureExtractor) or `video_features` (from the video "
                                 "loader).")
            if hunyuan_deps is None:
                raise ValueError("FoleyTune LoRA Timeline: self-contained mode (no `features`) "
                                 "needs `hunyuan_deps` to CLAP-encode base_prompt/negative_prompt.")
            features = {
                "clip_feat": video_features["clip_feat"],
                "sync_feat": video_features["sync_feat"],
                "text_feat": prompt_feats[base_prompt or ""],
                "uncond_text_feat": prompt_feats[negative_prompt or ""],
                "duration": video_features["duration"],
                "fps": video_features.get("fps", resolved_fps),
                "video_path": video_features.get("video_path", ""),
            }
            logger.info("LoRA timeline: built features from video_features "
                        "(self-contained, no FeatureExtractor)")

        schedule = []
        for seg in sorted(segments, key=lambda s: s["start_sec"]):
            entry = _entry_for(seg)
            if entry is None:
                continue
            seg_out = {
                "lora_path": entry["path"],
                "start_sec": float(seg["start_sec"]),
                "end_sec": float(seg["end_sec"]),
                "strength": float(seg.get("strength", entry["strength"])),
                "fade_in": float(seg.get("fade_in", 0.0)),
                "fade_out": float(seg.get("fade_out", 0.0)),
                "crossfade_sec": crossfade_sec,  # per-schedule; read by the sampler
            }
            entry_prompt = (entry.get("prompt") or "").strip()
            if entry_prompt and entry_prompt in prompt_feats:
                seg_out["text_feat"] = prompt_feats[entry_prompt]
                seg_out["prompt"] = entry_prompt
            schedule.append(seg_out)

        # Resolve the thumbnail video path: explicit override > loader's
        # video_features (carries video_path) > whatever the features dict has.
        resolved_video_path = (
            (video_path or "").strip()
            or (video_features.get("video_path", "") if video_features else "")
            or features.get("video_path", "")
        )

        return {
            "ui": {
                "duration": [features["duration"]],
                "video_path": [resolved_video_path],
                "fps": [resolved_fps],
                "entries": [entries],
            },
            "result": (schedule, features),
        }


# --- API: Timeline Thumbnail Sprite -----------------------------------------

try:
    from server import PromptServer
    import aiohttp.web as web
    import subprocess

    def _probe_duration(path):
        """Return video duration in seconds, or 0.0 if it can't be determined."""
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", path],
                capture_output=True, timeout=15,
            )
            return float(out.stdout.decode().strip())
        except (ValueError, OSError, subprocess.SubprocessError):
            return 0.0

    @PromptServer.instance.routes.get("/foleytune/timeline_thumbnails")
    async def timeline_thumbnails(request):
        """Generate a single-row sprite sheet of video thumbnails for the timeline.

        ffmpeg's `tile` filter needs an explicit column count (there is no
        auto/0 layout), so probe the duration and tile exactly `cols` frames
        sampled uniformly across the clip. The widget stretches this strip to
        the full track width, so cols frames map linearly onto the time axis.
        """
        video_path = request.query.get("video_path", "")
        logger.info(f"[timeline thumbnails] request video_path={video_path!r}")
        if not video_path:
            logger.warning("[timeline thumbnails] empty video_path — features dict "
                           "carried no 'video_path' (check the upstream feature node)")
            return web.Response(status=404, text="No video_path provided")
        if not os.path.exists(video_path):
            logger.warning(f"[timeline thumbnails] path does not exist on this host: {video_path}")
            return web.Response(status=404, text=f"Video not found on server: {video_path}")

        duration = _probe_duration(video_path)
        # ~2 thumbnails/sec, capped so long clips don't make an enormous sprite.
        cols = max(1, min(80, round(duration * 2))) if duration > 0 else 16
        fps = cols / duration if duration > 0 else 2.0
        logger.info(f"[timeline thumbnails] duration={duration:.2f}s cols={cols} fps={fps:.3f}")

        mtime = os.path.getmtime(video_path)
        cache_key = hashlib.md5(f"{video_path}:{mtime}:{cols}:v2".encode()).hexdigest()
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "thumbnails")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{cache_key}.jpg")

        if not os.path.exists(cache_path):
            try:
                result = subprocess.run([
                    "ffmpeg", "-y", "-i", video_path,
                    "-vf", f"fps={fps:.6f},scale=160:-2,tile={cols}x1",
                    "-frames:v", "1", "-update", "1",
                    "-q:v", "5",
                    cache_path,
                ], capture_output=True, timeout=60)
            except FileNotFoundError:
                logger.error("[timeline thumbnails] ffmpeg not found on PATH for the ComfyUI process")
                return web.Response(status=500, text="ffmpeg not found on PATH")
            if result.returncode != 0 or not os.path.exists(cache_path):
                err = result.stderr.decode(errors="replace")
                logger.error(f"[timeline thumbnails] ffmpeg failed (rc={result.returncode}): {err[-500:]}")
                return web.Response(status=500, text=f"ffmpeg failed: {err}")
            logger.info(f"[timeline thumbnails] generated sprite: {cache_path}")

        return web.FileResponse(cache_path, headers={"Content-Type": "image/jpeg"})

    @PromptServer.instance.routes.get("/foleytune/timeline_frame")
    async def timeline_frame(request):
        """Extract a single larger frame at time `t` for the scrub player.

        Lets the user align LoRA-segment boundaries to specific frames. Input
        seek (-ss before -i) is fast; clips are short so it's frame-accurate
        enough for alignment. Cached per (path, mtime, t rounded to 0.1s).
        """
        video_path = request.query.get("video_path", "")
        if not video_path or not os.path.exists(video_path):
            return web.Response(status=404, text="Video not found")
        try:
            t = max(0.0, float(request.query.get("t", "0")))
        except ValueError:
            t = 0.0

        t_key = round(t, 1)
        mtime = os.path.getmtime(video_path)
        cache_key = hashlib.md5(f"{video_path}:{mtime}:{t_key}:frame".encode()).hexdigest()
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "frames")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{cache_key}.jpg")

        if not os.path.exists(cache_path):
            try:
                result = subprocess.run([
                    "ffmpeg", "-y", "-ss", f"{t_key:.2f}", "-i", video_path,
                    "-frames:v", "1", "-update", "1",
                    "-vf", "scale=480:-2",
                    "-q:v", "3",
                    cache_path,
                ], capture_output=True, timeout=30)
            except FileNotFoundError:
                return web.Response(status=500, text="ffmpeg not found on PATH")
            if result.returncode != 0 or not os.path.exists(cache_path):
                err = result.stderr.decode(errors="replace")
                logger.error(f"[timeline frame] ffmpeg failed (rc={result.returncode}): {err[-300:]}")
                return web.Response(status=500, text=f"ffmpeg failed: {err}")

        return web.FileResponse(cache_path, headers={"Content-Type": "image/jpeg"})

except ImportError:
    pass  # Running outside ComfyUI server context


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
        "cos_sim_weight": 0.0, "spectral_weight": 0.0, "hf_phase_switch": 0.0,
        "wav_spectral_weight": 0.0, "wav_spectral_every": 8, "wav_spectral_crop": 64,
        "wav_spectral_adaptive": True, "channel_weight_mode": "off", "cfm_lambda": 0.0, "channel_loss_weight": False,
        "temporal_variance_weight": 0.0, "tv_gate_sigma": 0.3, "vd_curriculum_ratio": 0.0,
        "t_min": 0.0, "t_max": 1.0, "t_range_mode": "clamp", "optimizer_type": "prodigy",
        "prodigy_d_coef": 1.0, "prodigy_growth_rate": 0.0,
        "prodigy_safeguard_warmup": True,
        "prodigy_steps": 0, "use_cautious": False, "schedulefree_c": 0, "use_orthograd": False,
        "visual_dropout_prob": 0.5,
        "gradient_checkpointing": False,
        "freeze_blocks": 0,
        "resume_from": "",
        "eval_negative_prompt": "",
        "intensity_bias": 0.0,
        "intensity_metric": "energy",
        "intensity_per_dataset": True,
        "balance_datasets": False,
        "event_conditioning": False,
        "event_strength": 1.0,
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
        # Surface typos and inference-only options (e.g. blocks_to_swap) that
        # the sweep trainer would otherwise silently ignore.
        unknown = sorted(k for k in merged
                         if k not in self._PARAM_DEFAULTS and k != "eval_npz")
        if unknown:
            logger.warning(f"Sweep config keys not used by the trainer (ignored): {unknown}")
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

        ds_cfg = None
        dataset_jsons = dataset_json if isinstance(dataset_json, list) else [dataset_json] if dataset_json else []
        dataset_jsons = [p for p in dataset_jsons if p]
        _missing = [p for p in dataset_jsons if not os.path.exists(p)]
        if _missing:
            # Silently dropping missing paths would train on a partial dataset
            raise FileNotFoundError(f"dataset_json path(s) not found: {_missing}")

        if dataset_jsons:
            dataset = []
            for _src_idx, dj_path in enumerate(dataset_jsons):
                with open(dj_path) as f:
                    dj_cfg = json.load(f)
                if not isinstance(dj_cfg.get("train"), list):
                    raise ValueError(f"dataset_json must contain a 'train' key: {dj_path}")
                dj_dir = str(Path(dj_path).parent)
                dj_clips = dj_cfg["train"]
                logger.info(f"Loading dataset: {dj_path} ({len(dj_clips)} train clips)")
                _new = prepare_dataset(dj_dir, hunyuan_deps.dac_model, device, dtype,
                                       clip_names=dj_clips)
                for _it in _new:
                    _it["source_idx"] = _src_idx   # tag source dataset (for balance_datasets)
                dataset += _new
                if ds_cfg is None:
                    ds_cfg = dj_cfg
                    data_dir = dj_dir
            # Per-dir lengths are uniform but can disagree ACROSS dirs, which
            # would crash torch.cat mid-training (see performer_b_multipos failure)
            dataset = harmonize_dataset(dataset)
        elif data_dir:
            logger.info(f"Preparing shared dataset from {data_dir}...")
            dataset = prepare_dataset(data_dir, hunyuan_deps.dac_model, device, dtype)
        else:
            raise ValueError("Sweep JSON must specify either 'data_dir' or 'dataset_json'")

        from collections import Counter
        _prompt_counts = Counter(d["prompt"] for d in dataset)
        prompts_list = [p for p, _ in _prompt_counts.most_common()]

        # Load validation / eval samples — supports single path or list of {name, path}
        def _load_ref_audio(npz_path, dac_model=None):
            for ext in (".flac", ".wav", ".ogg"):
                candidate = Path(npz_path).with_suffix(ext)
                if candidate.exists():
                    import soundfile as _sf
                    _raw, _sr = _sf.read(str(candidate))
                    if _raw.ndim > 1:
                        _raw = _raw.mean(axis=1)
                    if _sr != 48000:
                        import soxr as _soxr
                        _raw = _soxr.resample(_raw[:, None], _sr, 48000, quality="VHQ").squeeze(-1)
                    if dac_model is not None:
                        with torch.no_grad():
                            dac_model.to(device)
                            _t = torch.from_numpy(_raw).float().unsqueeze(0).unsqueeze(0).to(device)
                            _z, _, _, _, _ = dac_model.encode(_t)
                            _raw = dac_model.decode(_z.mode()).squeeze().cpu().numpy()
                            dac_model.cpu()
                    return _raw
            return None

        # Production-parity eval CFG: CLAP-encode the negative prompt for the
        # uncond branch (the inference pipeline's convention) instead of the
        # legacy zero embedding. Cached per prompt string across experiments.
        _neg_embed_cache = {}

        def _encode_eval_uncond(neg_prompt):
            if not neg_prompt:
                return None
            if neg_prompt not in _neg_embed_cache:
                # CLAP weights are inference tensors (created under ComfyUI's
                # inference mode) and cannot forward inside the training block's
                # inference_mode(False) region — re-enter inference mode for the
                # encode, then launder the output through numpy so the cached
                # embedding is a normal tensor usable during training.
                with torch.inference_mode():
                    hunyuan_deps.clap_model.to(device)
                    _inputs = hunyuan_deps.clap_tokenizer(
                        [neg_prompt], padding=True, truncation=True, max_length=100,
                        return_tensors="pt",
                    ).to(device)
                    _out = hunyuan_deps.clap_model(
                        **_inputs, output_hidden_states=True, return_dict=True
                    )
                    _emb = _out.last_hidden_state.float().cpu()
                    hunyuan_deps.clap_model.to(mm.unet_offload_device())
                _neg_embed_cache[neg_prompt] = torch.from_numpy(_emb.numpy().copy())
                logger.info(f"Eval uncond text: CLAP({neg_prompt!r}) "
                            f"{tuple(_neg_embed_cache[neg_prompt].shape)}")
            return _neg_embed_cache[neg_prompt]

        eval_entries = []  # list of {"name": str, "entry": dict, "ref_wav": ndarray|None}

        eval_npz = sweep.get("eval_npz") or base_config.get("eval_npz")
        if eval_npz:
            if isinstance(eval_npz, str):
                eval_npz = [{"name": "eval", "path": eval_npz}]
            for ev in eval_npz:
                if isinstance(ev, str):
                    ev = {"name": Path(ev).stem, "path": ev}
                ev_path = ev["path"]
                ev_name = ev.get("name", Path(ev_path).stem)
                if os.path.exists(ev_path):
                    eval_entries.append({
                        "name": ev_name,
                        "entry": prepare_single_entry(ev_path, hunyuan_deps.dac_model, device, dtype),
                        "ref_wav": _load_ref_audio(ev_path, hunyuan_deps.dac_model),
                    })
                    logger.info(f"Eval sample loaded: {ev_name} ({ev_path})")
                else:
                    logger.warning(f"eval_npz path not found, skipping: {ev_path}")

        if eval_entries:
            logger.info(f"Loaded {len(eval_entries)} eval entries: {[e['name'] for e in eval_entries]}")

        # Collect loss histories for comparison chart
        all_loss_histories = {}

        # SIGUSR1 handler: skip current experiment and move to next
        _skip_flag_path = output_root / "skip_current.flag"
        _prev_sigusr1 = None

        def _sigusr1_skip(signum, frame):
            _skip_flag_path.touch()
            logger.info("SIGUSR1 received — skipping to next experiment")

        try:
            _prev_sigusr1 = signal.getsignal(signal.SIGUSR1)
            signal.signal(signal.SIGUSR1, _sigusr1_skip)
            logger.info(
                f"Sweep PID {os.getpid()} | "
                f"Skip to next experiment: kill -USR1 {os.getpid()}  or  "
                f"touch {_skip_flag_path}"
            )
        except ValueError:
            logger.info(
                f"Skip to next experiment: touch {_skip_flag_path}"
            )

        for exp in experiments:
            exp_id = exp.get("id", f"exp_{len(results)}")
            # Reload sweep JSON to pick up on-disk edits between experiments
            try:
                with open(sweep_json) as _sf:
                    _live = json.load(_sf)
                _live_by_id = {e.get("id"): e for e in _live.get("experiments", [])}
                if exp_id in _live_by_id:
                    exp = _live_by_id[exp_id]
                    base_config = _live.get("base", base_config)
                    logger.info(f"Reloaded config for {exp_id} from disk")
            except Exception:
                pass

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
            exp_dtype = dtype_map[config.get("precision", base_precision)]
            exp_dir = output_root / exp_id
            exp_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Starting experiment: {exp_id}")
            logger.info(f"Config: {json.dumps({k: v for k, v in config.items() if k != 'description'}, indent=2)}")

            exp_result = {"id": exp_id, "config": config, "status": "running"}

            try:
                with torch.inference_mode(False), torch.enable_grad():
                    # Deep copy model for this experiment
                    model = copy.deepcopy(hunyuan_model)
                    model.to(device=device, dtype=exp_dtype)
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
                    event_conditioning = bool(config.get("event_conditioning", False))
                    model._event_conditioning_enabled = event_conditioning
                    model._event_strength = float(config.get("event_strength", 1.0))

                    for name, param in model.named_parameters():
                        param.requires_grad = "lora_" in name or (
                            event_conditioning and name.startswith("event_adapter.")
                        )

                    _freeze = config.get("freeze_blocks", 0)
                    if _freeze > 0:
                        n_frozen = 0
                        for name, param in model.named_parameters():
                            if "lora_" in name:
                                for i in range(_freeze):
                                    if f"triple_blocks.{i}." in name:
                                        param.requires_grad = False
                                        n_frozen += 1
                                        break
                        logger.info(f"[{exp_id}] Froze LoRA params in blocks 0..{_freeze - 1} ({n_frozen} tensors)")

                    # Optimizer
                    _lr = config["lr"]
                    if config["lora_plus_ratio"] > 1.0:
                        a_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_A" in n]
                        b_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
                        other_params = [p for n, p in model.named_parameters()
                                        if p.requires_grad and "lora_A" not in n and "lora_B" not in n]
                        param_groups = [{"params": a_params, "lr": _lr}, {"params": b_params, "lr": _lr * config["lora_plus_ratio"]}]
                        if other_params:
                            param_groups.append({"params": other_params, "lr": _lr})
                    else:
                        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": _lr}]

                    _opt_type = config.get("optimizer_type", "adamw")
                    if _opt_type == "prodigy":
                        from prodigyopt import Prodigy
                        for pg in param_groups:
                            pg.pop("lr", None)
                        _d_coef = config.get("prodigy_d_coef", 1.0)
                        _growth = config.get("prodigy_growth_rate", 0.0)
                        _growth = float("inf") if _growth <= 0 else _growth
                        _safeguard = config.get("prodigy_safeguard_warmup", True)
                        optimizer = Prodigy(param_groups, lr=1.0, betas=(0.9, 0.999), weight_decay=0.01,
                                            d_coef=_d_coef, growth_rate=_growth, decouple=True,
                                            safeguard_warmup=_safeguard, use_bias_correction=True)
                        logger.info(f"[{exp_id}] Using Prodigy optimizer (d_coef={_d_coef}, growth_rate={_growth}, safeguard={_safeguard}, decouple=True, wd=0.01)")
                    elif _opt_type == "prodigy_plus":
                        from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
                        for pg in param_groups:
                            pg.pop("lr", None)
                        _d_coef = config.get("prodigy_d_coef", 1.0)
                        _pp_steps = int(config.get("prodigy_steps", 0))      # 0 = adapt d forever; >0 freezes LR after N steps
                        _pp_caut = bool(config.get("use_cautious", False))   # cautious updates (sign-aligned only)
                        _pp_sfc = float(config.get("schedulefree_c", 0))     # schedule-free averaging constant (0 = default)
                        _pp_ortho = bool(config.get("use_orthograd", False)) # orthogonal-gradient regularization
                        optimizer = ProdigyPlusScheduleFree(param_groups, lr=1.0, betas=(0.9, 0.999), weight_decay=0.01,
                                                           d_coef=_d_coef, prodigy_steps=_pp_steps,
                                                           use_cautious=_pp_caut, schedulefree_c=_pp_sfc,
                                                           use_orthograd=_pp_ortho)
                        optimizer.train()
                        logger.info(f"[{exp_id}] Using Prodigy+ Schedule-Free (d_coef={_d_coef}, prodigy_steps={_pp_steps}, cautious={_pp_caut}, sf_c={_pp_sfc}, orthograd={_pp_ortho}, wd=0.01)")
                    else:
                        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=0.01)

                    _ga = config["grad_accum"]
                    _sched_type = "constant" if _opt_type == "prodigy_plus" else config["schedule_type"]
                    def lr_lambda(sched_step):
                        actual_step = sched_step * _ga
                        if actual_step < config["warmup_steps"]:
                            return actual_step / max(config["warmup_steps"], 1)
                        if _sched_type == "cosine":
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
                        _ckpt_opt = ckpt.get("meta", {}).get("optimizer_type", "adamw")
                        _opt_match = (_ckpt_opt == _opt_type)
                        if not _opt_match:
                            logger.info(f"[{exp_id}] Optimizer mismatch (ckpt={_ckpt_opt}, current={_opt_type}) — loading weights only, fresh optimizer")
                        if _freeze > 0:
                            logger.info(f"[{exp_id}] freeze_blocks={_freeze} — fresh optimizer (param count changed)")
                            _opt_match = False
                        if _opt_match and "optimizer" in ckpt:
                            optimizer.load_state_dict(ckpt["optimizer"])
                        if _opt_match and "scheduler" in ckpt:
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

                    # Channel-weighted loss: "variance" (legacy LF bulk) or "inverse" (HF channels)
                    _cw_mode = config.get("channel_weight_mode") or ("variance" if config.get("channel_loss_weight", False) else "off")
                    _channel_weights = None
                    if _cw_mode != "off":
                        _all_lat = torch.cat([d["latents"] for d in dataset], dim=0)
                        _channel_weights = compute_channel_weights(_all_lat, _cw_mode)
                        logger.info(f"[{exp_id}] Channel weights ({_cw_mode}): min={_channel_weights.min():.2f} max={_channel_weights.max():.2f}")
                        del _all_lat

                    # CFG uncond for eval samples: production parity when set,
                    # legacy zero embedding when empty
                    _eval_uncond = _encode_eval_uncond(str(config.get("eval_negative_prompt") or ""))

                    import random
                    # Offset by start_step so resumed/extended runs don't replay
                    # the exact batch/noise/timestep sequence from step 0
                    _rng_seed = (config["seed"] + start_step) % (2 ** 31)
                    torch.manual_seed(_rng_seed)
                    random.seed(_rng_seed)
                    np.random.seed(_rng_seed)

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
                    # Optional sampling biases (combined multiplicatively, off by default):
                    #  - intensity_bias (alpha): lean toward energetic/dynamic clips;
                    #    intensity_metric 'energy' (mean per-frame latent energy) or 'tv'
                    #    (std/mean of the energy envelope = dynamic burstiness).
                    #  - balance_datasets: equal exposure per source dataset_json regardless
                    #    of clip count (up-weights smaller datasets to neutralize imbalance).
                    _intensity_alpha = float(config.get("intensity_bias", 0.0))
                    _balance = bool(config.get("balance_datasets", False))
                    _intensity_per_dataset = bool(config.get("intensity_per_dataset", True))
                    _sample_w = None
                    if _intensity_alpha > 0 or _balance:
                        _w = np.ones(n_clips, dtype=np.float64)
                        if _intensity_alpha > 0:
                            _imetric = config.get("intensity_metric", "energy")
                            _scores = []
                            for _d in dataset:
                                _env = _d["latents"][0].float().pow(2).sum(0).clamp(min=1e-12).sqrt()  # [T]
                                _m = _env.mean().clamp(min=1e-8)
                                _scores.append(max((_env.std() / _m).item() if _imetric == "tv" else _m.item(), 1e-8))
                            _scores = np.asarray(_scores, dtype=np.float64)
                            # Per-dataset normalization (default on): rank each clip against
                            # its OWN source's energy distribution (divide by that source's
                            # mean score) so the bias favors clips energetic FOR THEIR CONTENT,
                            # not just absolutely louder. Without it the absolute-energy metric
                            # tilts a multi-dataset run toward the louder source (e.g. multipos
                            # moans drown the quieter blowjob clips). Mathematically a no-op for
                            # a single dataset (one group; the constant cancels under renorm).
                            if _intensity_per_dataset:
                                _src_arr = np.array([int(_d.get("source_idx", 0)) for _d in dataset])
                                _ngrp = len(np.unique(_src_arr))
                                for _s in np.unique(_src_arr):
                                    _mask = _src_arr == _s
                                    _gmean = _scores[_mask].mean()
                                    if _gmean > 0:
                                        _scores[_mask] = _scores[_mask] / _gmean
                                if _ngrp > 1:
                                    logger.info(f"[{exp_id}] intensity_per_dataset ON: intensity ranked "
                                                f"within each of {_ngrp} sources (no cross-source loudness skew)")
                            _w *= _scores ** _intensity_alpha
                        if _balance:
                            _src = [int(_d.get("source_idx", 0)) for _d in dataset]
                            _counts = {s: _src.count(s) for s in set(_src)}
                            _w *= np.array([1.0 / max(_counts[s], 1) for s in _src], dtype=np.float64)
                            logger.info(f"[{exp_id}] balance_datasets ON: {len(_counts)} sources sizes={_counts} -> equal exposure each")
                        if np.isfinite(_w.sum()) and _w.sum() > 0:
                            _sample_w = _w / _w.sum()
                            logger.info(f"[{exp_id}] Weighted sampling: intensity_bias={_intensity_alpha} "
                                        f"balance_datasets={_balance} (top {_sample_w.max()*n_clips:.2f}x, "
                                        f"bottom {_sample_w.min()*n_clips:.2f}x vs uniform)")
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
                            model, hunyuan_deps.dac_model, dataset[0], device, exp_dtype,
                            uncond_text_feat=_eval_uncond,
                        )
                        wav0_mono = wav0.squeeze()
                        wav0_t = torch.from_numpy(wav0)
                        if wav0_t.ndim == 1:
                            wav0_t = wav0_t.unsqueeze(0)
                        _save_wav(samples_dir_0 / "step_00000.wav", wav0_t, sr0)
                        _save_spectrogram(wav0_mono, sr0, samples_dir_0 / "step_00000")
                        logger.info(f"[{exp_id}] Step 0 eval sample saved")

                    # Step-0 validation eval for all eval entries (skip if resuming)
                    if start_step == 0:
                        for _ev in eval_entries:
                            wav0v, sr0v = generate_eval_sample(
                                model, hunyuan_deps.dac_model, _ev["entry"], device, exp_dtype,
                                uncond_text_feat=_eval_uncond,
                            )
                            wav0v_mono = wav0v.squeeze()
                            wav0v_t = torch.from_numpy(wav0v)
                            if wav0v_t.ndim == 1:
                                wav0v_t = wav0v_t.unsqueeze(0)
                            _ev_tag = _ev["name"]
                            _save_wav(samples_dir_0 / f"{_ev_tag}_step_00000.wav", wav0v_t, sr0v)
                            _save_spectrogram(wav0v_mono, sr0v, samples_dir_0 / f"{_ev_tag}_step_00000")
                            if _ev["ref_wav"] is not None:
                                _save_spectrogram(_ev["ref_wav"], 48000, samples_dir_0 / f"{_ev_tag}_reference")
                            logger.info(f"[{exp_id}] Step 0 {_ev_tag} sample saved")

                    model.train()

                    # Waveform spectral loss: keep DAC resident on GPU (frozen) for
                    # differentiable decode during training.
                    _wav_w = config.get("wav_spectral_weight", 0.0)
                    _wav_every = int(config.get("wav_spectral_every", 8))
                    _wav_crop = int(config.get("wav_spectral_crop", 64))
                    _wav_adaptive = bool(config.get("wav_spectral_adaptive", True))
                    _wav_dac = None
                    if _wav_w > 0:
                        # Frozen deepcopy outside inference_mode so the differentiable
                        # decode can backprop (DAC's inference-tensor weights + weight_norm
                        # recompute can't join autograd in place). Shared eval DAC untouched.
                        with torch.inference_mode(False), torch.no_grad():
                            _wav_dac = copy.deepcopy(hunyuan_deps.dac_model).to(device=device).eval()
                            for _p in _wav_dac.parameters():
                                _p.requires_grad_(False)
                        logger.info(f"[{exp_id}] Waveform spectral loss ON: weight={_wav_w}, every={_wav_every}, crop={_wav_crop}")

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
                        if _sample_w is not None:
                            indices = np.random.choice(n_clips, size=bs, p=_sample_w).tolist()
                        else:
                            indices = [np.random.randint(0, n_clips) for _ in range(bs)]
                        batch_latents = torch.cat([dataset[i]["latents"] for i in indices]).to(device, dtype=exp_dtype)
                        # Event envelope from the CLEAN target latents — computed BEFORE the
                        # mixup/latent-noise/noise_offset augmentations below, which would perturb
                        # the semantic energy contour (noise_offset is a per-channel DC the
                        # conditioning shouldn't see) and diverge from the clean envelope used at
                        # eval/inference.
                        batch_event = None
                        if config.get("event_conditioning", False):
                            from .lora.event import event_envelope_from_latents
                            batch_event = event_envelope_from_latents(batch_latents).to(device, dtype=exp_dtype)
                        batch_clip = torch.cat([dataset[i]["clip_features"] for i in indices])
                        batch_sync = torch.cat([dataset[i]["sync_features"] for i in indices])
                        _text_items = [dataset[i]["text_embedding"] for i in indices]
                        _max_tlen = max(t.shape[1] for t in _text_items)
                        batch_text = torch.cat([F.pad(t, (0, 0, 0, _max_tlen - t.shape[1])) for t in _text_items])

                        # Pad sync to multiple of 8
                        sync_len = batch_sync.shape[1]
                        pad_sync = ((sync_len + 7) // 8) * 8 - sync_len
                        if pad_sync > 0:
                            batch_sync = F.pad(batch_sync, (0, 0, 0, pad_sync))

                        # Optional latent augmentation (parity with FoleyTuneLoRATrainer)
                        _mixup_alpha = config.get("latent_mixup_alpha", 0.0)
                        if _mixup_alpha > 0 and bs > 1:
                            lam = np.random.beta(_mixup_alpha, _mixup_alpha)
                            perm = torch.randperm(bs)
                            batch_latents = lam * batch_latents + (1 - lam) * batch_latents[perm]

                        _lat_noise = config.get("latent_noise_sigma", 0.0)
                        if _lat_noise > 0:
                            batch_latents = batch_latents + torch.randn_like(batch_latents) * _lat_noise

                        _noise_offset = config.get("noise_offset", 0.0)
                        if _noise_offset > 0:
                            offset = torch.randn(batch_latents.shape[0], batch_latents.shape[1], 1, device=device, dtype=exp_dtype) * _noise_offset
                            batch_latents = batch_latents + offset

                        _hf_switch = config.get("hf_phase_switch", 0.0)
                        _eff_t_min = config.get("t_min", 0.0)
                        _eff_t_max = config.get("t_max", 1.0)
                        if _hf_switch > 0:
                            _progress = (step - start_step) / max(config["steps"] - start_step, 1)
                            if _progress >= _hf_switch:
                                _eff_t_min, _eff_t_max = 0.0, 1.0
                        t = sample_timesteps(
                            bs, config["timestep_mode"], device, exp_dtype,
                            sigma=config["logit_normal_sigma"],
                            curriculum_switch=config["curriculum_switch"],
                            step=step, start_step=start_step, total_steps=config["steps"],
                            t_min=_eff_t_min, t_max=_eff_t_max,
                            t_range_mode=config.get("t_range_mode", "clamp"),
                        )

                        effective_vd = visual_dropout_curriculum(
                            config.get("visual_dropout_prob", 0.0),
                            step, start_step, config["steps"],
                            config.get("vd_curriculum_ratio", 0.0),
                        )
                        _do_wav = _wav_dac is not None and (step % _wav_every == 0)
                        loss = flow_matching_loss(
                            model, batch_latents, t, batch_clip, batch_sync, batch_text,
                            device, exp_dtype,
                            visual_dropout_prob=effective_vd,
                            min_snr_gamma=config.get("min_snr_gamma", 0.0),
                            cos_sim_weight=config.get("cos_sim_weight", 0.0),
                            channel_weights=_channel_weights,
                            temporal_variance_weight=config.get("temporal_variance_weight", 0.0),
                            tv_gate_sigma=config.get("tv_gate_sigma", 0.3),
                            spectral_weight=config.get("spectral_weight", 0.0),
                            dac_model=_wav_dac, wav_spectral_weight=_wav_w,
                            wav_spectral_crop=_wav_crop,
                            wav_spectral_adaptive=_wav_adaptive,
                            compute_wav_spectral=_do_wav,
                            cfm_lambda=config.get("cfm_lambda", 0.0),
                            event_envelope=batch_event,
                            event_strength=config.get("event_strength", 1.0),
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
                            _lr_display = lr_sched.get_last_lr()[0]
                            _d_val = optimizer.param_groups[0].get('d')
                            if _d_val is not None:
                                _lr_display = _d_val * lr_sched.get_last_lr()[0]
                            logger.info(f"[{exp_id}] Step {step+1}/{config['steps']} | "
                                       f"loss: {avg_loss:.4f} | lr: {_lr_display:.2e} | "
                                       f"elapsed: {elapsed:.0f}s")

                            preview_img = _draw_loss_curve(
                                losses,
                                smoothed=_smooth_losses(losses),
                                metrics_history=metrics_history,
                            )
                            pbar_train.update_absolute(
                                step + 1 - start_step, config["steps"] - start_step,
                                ("JPEG", preview_img, 800),
                            )

                        if (step + 1) % config["save_every"] == 0:
                            # Schedule-free optimizers hold raw train-mode weights; also
                            # capture the averaged eval-mode weights so this checkpoint
                            # loads for inference exactly as the eval sample sounds
                            _sf_opt = hasattr(optimizer, 'eval') and hasattr(optimizer, 'train')
                            _eval_sd = None
                            if _sf_opt:
                                optimizer.eval()
                                _eval_sd = get_lora_state_dict(model)
                                optimizer.train()

                            # Save with live weights for optimizer consistency on resume
                            meta = {**config, "steps_completed": step + 1, "prompts": prompts_list}
                            ckpt_path = exp_dir / f"adapter_step{step+1:05d}.pt"
                            save_checkpoint(model, optimizer, lr_sched, step + 1, meta, ckpt_path,
                                            ema_state=ema_state, eval_state=_eval_sd)

                            # Switch schedule-free optimizer to eval mode (averaged weights)
                            if _sf_opt:
                                optimizer.eval()

                            # Swap in EMA weights for eval
                            if ema_state is not None:
                                _live_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
                                for n, p in model.named_parameters():
                                    if p.requires_grad and n in ema_state:
                                        p.data.copy_(ema_state[n])
                            _draw_loss_curve(losses, smoothed=_smooth_losses(losses), metrics_history=metrics_history).save(str(exp_dir / "loss.png"))

                            # Generate eval audio sample + compute metrics
                            samples_dir = exp_dir / "samples"
                            samples_dir.mkdir(exist_ok=True)
                            model.eval()
                            wav, sr = generate_eval_sample(
                                model, hunyuan_deps.dac_model, dataset[0], device, exp_dtype,
                                uncond_text_feat=_eval_uncond,
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

                            # Eval samples for all eval entries
                            for _ev in eval_entries:
                                _ev_tag = _ev["name"]
                                wav_v, sr_v = generate_eval_sample(
                                    model, hunyuan_deps.dac_model, _ev["entry"], device, exp_dtype,
                                    uncond_text_feat=_eval_uncond,
                                )
                                wav_v_mono = wav_v.squeeze()
                                wav_v_t = torch.from_numpy(wav_v)
                                if wav_v_t.ndim == 1:
                                    wav_v_t = wav_v_t.unsqueeze(0)
                                _save_wav(samples_dir / f"{_ev_tag}_step_{step+1:05d}.wav", wav_v_t, sr_v)
                                _save_spectrogram(wav_v_mono, sr_v, samples_dir / f"{_ev_tag}_step_{step+1:05d}")
                                _ev_sm = spectral_metrics(wav_v_mono, sr_v)
                                if _ev["ref_wav"] is not None:
                                    _ev_sm.update(reference_metrics(wav_v_mono, _ev["ref_wav"], sr_v))
                                for mk, mv in _ev_sm.items():
                                    step_metrics[f"{_ev_tag}_{mk}"] = mv

                            # Restore live (non-EMA) weights for continued training
                            if ema_state is not None:
                                for n, p in model.named_parameters():
                                    if p.requires_grad and n in _live_params:
                                        p.data.copy_(_live_params[n])

                            model.train()
                            if _sf_opt:
                                optimizer.train()

                            _log_parts = [f"[{exp_id}] Step {step+1}: "
                                       f"loss={step_metrics['loss']:.4f}  "
                                       f"PBC={step_metrics.get('per_band_correlation', 0):.3f}  "
                                       f"HF={step_metrics.get('hf_energy_ratio', 0):.4f}  "
                                       f"TV={step_metrics.get('temporal_variance', 0):.3f}  "
                                       f"SC={step_metrics.get('spectral_convergence', 0):.3f}  "
                                       f"MCD={step_metrics.get('mel_cepstral_distortion', 0):.2f}"]
                            for _ev in eval_entries:
                                _t = _ev["name"]
                                _log_parts.append(
                                    f"  [{_t}] PBC={step_metrics.get(f'{_t}_per_band_correlation', 0):.3f}  "
                                    f"TV={step_metrics.get(f'{_t}_temporal_variance', 0):.3f}  "
                                    f"SC={step_metrics.get(f'{_t}_spectral_convergence', 0):.3f}")
                            logger.info("".join(_log_parts))

                    # Save final (with EMA weights if enabled)
                    _sf_final = hasattr(optimizer, 'eval') and hasattr(optimizer, 'train')
                    if _sf_final:
                        optimizer.eval()
                    if ema_state is not None:
                        for n, p in model.named_parameters():
                            if p.requires_grad and n in ema_state:
                                p.data.copy_(ema_state[n])

                    meta = {**config, "steps_completed": config["steps"], "prompts": prompts_list}
                    final_path = exp_dir / "adapter_final.pt"
                    save_checkpoint(model, optimizer, lr_sched, config["steps"], meta, final_path, final=True)
                    # Draw and save per-experiment loss curve
                    smoothed = _smooth_losses(losses)
                    loss_img = _draw_loss_curve(losses, smoothed=smoothed, metrics_history=metrics_history)
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

            # Keep only the latest result per id (re-run failed experiments
            # previously accumulated duplicate entries in the summary)
            results = [r for r in results if r.get("id") != exp_id]
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

        if _prev_sigusr1 is not None:
            try:
                signal.signal(signal.SIGUSR1, _prev_sigusr1)
            except ValueError:
                pass

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
                inferred_rank = None
                for k, v in sd.items():
                    if "lora_A" in k and v.ndim == 2:
                        inferred_rank = v.shape[0]
                        break
                rank = meta.get("rank", inferred_rank or 16)
                alpha_val = meta.get("alpha", float(rank))
                target = meta.get("target", "all_attn_mlp")
                target_suffixes = FOLEY_TARGET_PRESETS.get(target, FOLEY_TARGET_PRESETS["all_attn_mlp"])

                model._event_conditioning_enabled = bool(meta.get("event_conditioning", False))
                model._event_strength = float(meta.get("event_strength", 1.0))
                apply_lora(model, rank=rank, alpha=alpha_val, target_suffixes=target_suffixes,
                           init_mode="standard", use_rslora=meta.get("use_rslora", False))
                load_lora(model, sd)
            else:
                model = copy.deepcopy(hunyuan_model)
                meta = {}
                model._event_conditioning_enabled = False
                model._event_strength = 1.0

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
        # Schedule-free checkpoints: prefer the averaged eval-mode weights
        # (what the eval samples were generated with) over raw train weights
        if "eval_state_dict" in ckpt:
            state_dict = ckpt["eval_state_dict"]
        if "ema_state" in ckpt:
            for key, ema_val in ckpt["ema_state"].items():
                if key in state_dict:
                    state_dict[key] = ema_val

        final = {"state_dict": state_dict, "meta": ckpt.get("meta", {})}

        removed = []
        for key in ("optimizer", "scheduler", "step", "ema_state", "eval_state_dict"):
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
    "FoleyTuneTrainOptions": FoleyTuneTrainOptions,
    "FoleyTuneLoRALoader": FoleyTuneLoRALoader,
    "FoleyTuneLoRALoaderPath": FoleyTuneLoRALoaderPath,
    "FoleyTuneLoRATimelineEntry": FoleyTuneLoRATimelineEntry,
    "FoleyTuneLoRATimeline": FoleyTuneLoRATimeline,
    "FoleyTuneLoRAScheduler": FoleyTuneLoRAScheduler,
    "FoleyTuneLoRAEvaluator": FoleyTuneLoRAEvaluator,
    "FoleyTuneVAERoundtrip": FoleyTuneVAERoundtrip,
    "FoleyTuneCheckpointFinalizer": FoleyTuneCheckpointFinalizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoleyTuneFeatureExtractor": "FoleyTune Feature Extractor",
    "FoleyTuneBatchFeatureExtractor": "FoleyTune Batch Feature Extractor",
    "FoleyTuneLoRATrainer": "FoleyTune LoRA Trainer",
    "FoleyTuneTrainOptions": "FoleyTune Train Options",
    "FoleyTuneLoRALoader": "FoleyTune LoRA Loader",
    "FoleyTuneLoRALoaderPath": "FoleyTune LoRA Loader (Path)",
    "FoleyTuneLoRATimelineEntry": "FoleyTune LoRA Timeline Entry",
    "FoleyTuneLoRATimeline": "FoleyTune LoRA Timeline",
    "FoleyTuneLoRAScheduler": "FoleyTune LoRA Scheduler",
    "FoleyTuneLoRAEvaluator": "FoleyTune LoRA Evaluator",
    "FoleyTuneVAERoundtrip": "FoleyTune VAE Roundtrip",
    "FoleyTuneCheckpointFinalizer": "FoleyTune Checkpoint Finalizer",
}
