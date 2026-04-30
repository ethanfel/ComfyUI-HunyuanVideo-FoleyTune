import sys
import os
import json
import subprocess
import shutil
import numpy as np
import time
import hashlib
import weakref
import gc
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from inspect import cleandoc
from loguru import logger
from torchvision.transforms import v2
from transformers import AutoTokenizer, AutoModel, ClapTextModelWithProjection
from accelerate import init_empty_weights

from huggingface_hub import hf_hub_download
import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file
import comfy.utils

logger.remove()
logger.add(sys.stdout, level="INFO", format="HunyuanVideo-Foley: {message}")

# --- Add 'foley' models directory to ComfyUI's search paths ---
# This ensures ComfyUI can find models placed in 'ComfyUI/models/foley/'
foley_models_dir = os.path.join(folder_paths.models_dir, "foley")
if "foley" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["foley"] = ([foley_models_dir], folder_paths.supported_pt_extensions)

# --- Auto-download registry: filename -> (HuggingFace repo, remote filename) ---
DOWNLOADABLE_MODELS = {
    # Main model — full precision from Tencent, safetensors/fp8 from phazei
    "hunyuanvideo_foley.pth": ("Tencent/HunyuanVideo-Foley", "hunyuanvideo_foley.pth"),
    "hunyuanvideo_foley.safetensors": ("phazei/HunyuanVideo-Foley", "hunyuanvideo_foley.safetensors"),
    "hunyuanvideo_foley_fp8_e4m3fn.safetensors": ("phazei/HunyuanVideo-Foley", "hunyuanvideo_foley_fp8_e4m3fn.safetensors"),
    "hunyuanvideo_foley_fp8_e5m2.safetensors": ("phazei/HunyuanVideo-Foley", "hunyuanvideo_foley_fp8_e5m2.safetensors"),
    # Synchformer — full precision from Tencent, fp16 from phazei
    "synchformer_state_dict.pth": ("Tencent/HunyuanVideo-Foley", "synchformer_state_dict.pth"),
    "synchformer_state_dict_fp16.safetensors": ("phazei/HunyuanVideo-Foley", "synchformer_state_dict_fp16.safetensors"),
    # VAE (DAC) — full precision from Tencent, fp16 from phazei
    "vae_128d_48k.pth": ("Tencent/HunyuanVideo-Foley", "vae_128d_48k.pth"),
    "vae_128d_48k_fp16.safetensors": ("phazei/HunyuanVideo-Foley", "vae_128d_48k_fp16.safetensors"),
}

def get_foley_models(filter_fn=None):
    """List locally available + known downloadable models, with optional name filter."""
    existing = set(folder_paths.get_filename_list("foley"))
    all_models = sorted(existing | set(DOWNLOADABLE_MODELS.keys()))
    if filter_fn:
        all_models = [m for m in all_models if filter_fn(m)]
    return all_models if all_models else ["(no models found)"]

def ensure_model_downloaded(model_name):
    """Return local path to a foley model, auto-downloading from HuggingFace if needed."""
    local_path = folder_paths.get_full_path("foley", model_name)
    if local_path and os.path.exists(local_path):
        return local_path
    # Also check the directory directly (covers freshly created dirs not yet in cache)
    direct_path = os.path.join(foley_models_dir, model_name)
    if os.path.exists(direct_path):
        return direct_path
    if model_name not in DOWNLOADABLE_MODELS:
        raise FileNotFoundError(
            f"Model '{model_name}' not found in ComfyUI/models/foley/ and is not available for auto-download. "
            f"Downloadable models: {', '.join(DOWNLOADABLE_MODELS.keys())}"
        )
    repo_id, filename = DOWNLOADABLE_MODELS[model_name]
    logger.info(f"Auto-downloading {filename} from {repo_id} ...")
    os.makedirs(foley_models_dir, exist_ok=True)
    downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=foley_models_dir)
    logger.info(f"Download complete: {filename}")
    return downloaded_path

# --- Import the original, unmodified HunyuanVideo-Foley modules ---
# We treat the original code as an external library to keep our node clean.
try:
    from hunyuanvideo_foley.utils.config_utils import load_yaml, AttributeDict
    from hunyuanvideo_foley.utils.schedulers import FlowMatchDiscreteScheduler
    from hunyuanvideo_foley.models.dac_vae.model.dac import DAC
    from hunyuanvideo_foley.models.synchformer import Synchformer
    from hunyuanvideo_foley.models.hifi_foley import HunyuanVideoFoley
    from hunyuanvideo_foley.utils.feature_utils import encode_video_with_siglip2, encode_video_with_sync, encode_text_feat
except ImportError as e:
    logger.error(f"Failed to import HunyuanVideo-Foley modules: {e}")
    logger.error("Please ensure the ComfyUI_HunyuanVideoFoley custom node is installed correctly.")
    raise

# --- Import refactored local utilities (moved out of this file) ---
from .utils import (
    denoise_process_with_generator,
    feature_process_from_tensors,
    compute_chunk_boundaries,
    chunked_denoise_process,
    encode_audio_to_latents,
    _wrap_fp8_inplace,
    _detect_ckpt_fp8,
    _detect_ckpt_major_precision,
    _CudaFactoriesDuringCompile,
    load_dac_any
)

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
    info = json.loads(result.stdout)
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
                          start_time: float = 0.0, duration: float = 0.0,
                          resize_mode: str = "stretch") -> torch.Tensor:
    """Decode video frames at target resolution/fps via ffmpeg pipe.

    Args:
        resize_mode: "stretch" forces both dims to target_size (SigLIP2).
                     "crop" resizes shortest edge to target_size then center-crops (Synchformer).

    Returns [T, C, H, W] float32 tensor normalized to [-1, 1].
    """
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if start_time > 0:
        cmd += ["-ss", str(start_time)]
    cmd += ["-i", str(video_path)]
    if duration > 0:
        cmd += ["-t", str(duration)]

    if resize_mode == "crop":
        vf = (f"scale={target_size}:{target_size}:force_original_aspect_ratio=increase:flags=bicubic,"
              f"crop={target_size}:{target_size},fps={target_fps}")
    else:
        vf = f"scale={target_size}:{target_size}:flags=bicubic,fps={target_fps}"

    cmd += ["-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {result.stderr.decode()}")

    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    frame_size = target_size * target_size * 3
    n_frames = len(raw) // frame_size
    if n_frames == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")
    frames = raw[:n_frames * frame_size].reshape(n_frames, target_size, target_size, 3)
    tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor


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

    # SigLIP2: decode at 512x512 (stretch), 8fps
    siglip2_frames = _ffmpeg_decode_frames(video_path, 512, 8.0, start_time, duration,
                                           resize_mode="stretch")
    siglip2_batch = siglip2_frames.unsqueeze(0)  # [1, T, C, H, W]

    hunyuan_deps.siglip2_model.to(device)
    clip_feat = encode_video_with_siglip2(siglip2_batch.to(device), hunyuan_deps).cpu()
    del siglip2_frames, siglip2_batch
    hunyuan_deps.siglip2_model.to(offload_device)

    # Synchformer: decode at 224x224 (resize shortest edge + center-crop), 25fps
    sync_frames = _ffmpeg_decode_frames(video_path, 224, 25.0, start_time, duration,
                                        resize_mode="crop")
    # Synchformer requires at least 16 frames (segment_size=16).
    # Pad by repeating last frame if video is too short.
    if sync_frames.shape[0] < 16:
        pad = sync_frames[-1:].expand(16 - sync_frames.shape[0], -1, -1, -1)
        sync_frames = torch.cat([sync_frames, pad], dim=0)
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

# -----------------------------------------------------------------------------------
# NODE 1: FoleyTune Model Loader (refactored: pure load)
# -----------------------------------------------------------------------------------
class FoleyTuneModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_foley_models(),),
                "precision": (["auto", "bf16", "fp16", "fp32"], {"default": "bf16", "tooltip": "Compute dtype for non-quantized params and autocast (auto = detect from checkpoint)"}),
                "quantization": (["none", "fp8_e4m3fn", "fp8_e5m2", "auto"], {"default": "auto", "tooltip": "FP8 weight-only storage for Linear layers, saves a few GB VRAM (compute still fp16/bf16)"}),
            },
        }

    RETURN_TYPES = ("FOLEYTUNE_MODEL",)
    FUNCTION = "build_model"
    CATEGORY = "FoleyTune"

    def load_model(self, model_name, precision, quantization):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        # dtype resolved after checkpoint is loaded if precision == 'auto'
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(precision, torch.bfloat16)

        model_path = ensure_model_downloaded(model_name)
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "hunyuanvideo-foley-xxl.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Hunyuan config file not found at {config_path}")
        cfg = load_yaml(config_path)

        # Load weights onto the offload device first to save VRAM
        state_dict = load_torch_file(model_path, device=offload_device)

        # Auto-detect quantization and precision from checkpoint
        detected_fp8 = _detect_ckpt_fp8(state_dict)
        if precision == "auto":
            dtype = _detect_ckpt_major_precision(state_dict)
            logger.info(f"Auto precision selected from checkpoint: {str(dtype)}")

        # Initialize the model structure on the 'meta' device (no memory allocated yet)
        with init_empty_weights():
            foley_model = HunyuanVideoFoley(cfg, dtype=dtype)

        # Materialize the model on the offload device (CPU) to avoid VRAM spikes in the loader
        foley_model.to_empty(device=offload_device)

        # Load the state dict into the properly materialized model
        foley_model.load_state_dict(state_dict, strict=False)

        # Ensure the runtime parameter dtype matches the requested precision
        foley_model.to(dtype=dtype)
        foley_model.eval()

        # Optional FP8 weight-only quantization for Linear layers
        if quantization != "none":
            # Choose quantization mode (auto = honor fp8 tensors if present, else default to e4m3fn)
            if quantization == "auto":
                capability = (torch.cuda.get_device_capability()

                if torch.cuda.is_available() else (0, 0))

                # Ampere/Lovelace (SM < 90): avoid e4m3 path
                if capability[0] < 9:
                    qmode = "fp8_e5m2"
                else:
                    qmode = detected_fp8 if detected_fp8 is not None else "fp8_e4m3fn"
            else:
                qmode = quantization

            counts, saved = _wrap_fp8_inplace(foley_model, quantization=qmode, state_dict=state_dict)
            logger.info(f"FP8 wrap -> linear:{counts['linear']} conv1d:{counts['conv1d']} conv2d:{counts['conv2d']} | saved ~{saved/(1024**3):.2f} GiB")

        logger.info(f"Loaded HunyuanVideoFoley main model: {model_name}")
        
        # The state_dict is now copied into the model, so we no longer need the 10GB dictionary.
        # Explicitly delete it and trigger garbage collection.
        del state_dict
        gc.collect() 
        
        return foley_model

    def build_model(self, model_name, precision, quantization):
        foley_model = self.load_model(model_name, precision, quantization)

        # total_model_size_mb = get_module_size_in_mb(foley_model)
        # triple_blocks_size_mb = get_module_size_in_mb(foley_model.triple_blocks)
        # single_blocks_size_mb = get_module_size_in_mb(foley_model.single_blocks)
        # total_blocks_size_mb = triple_blocks_size_mb + single_blocks_size_mb
        
        # logger.info(f"--- Model Size Report ---")
        # logger.info(f"Total Model Size: {total_model_size_mb:.2f} MB")
        # logger.info(f"  - Triple-Stream Blocks (19x): {triple_blocks_size_mb:.2f} MB")
        # logger.info(f"  - Single-Stream Blocks (38x): {single_blocks_size_mb:.2f} MB")
        # logger.info(f"  - Total Swappable Block Size: {total_blocks_size_mb:.2f} MB")
        # logger.info(f"  - Non-Block Parameters (Embedders, etc.): {total_model_size_mb - total_blocks_size_mb:.2f} MB")
        # logger.info(f"-------------------------")

        return (foley_model,)

# -----------------------------------------------------------------------------------
# NODE 2: FoleyTune Dependencies Loader
# -----------------------------------------------------------------------------------
class FoleyTuneDependenciesLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (get_foley_models(lambda f: "vae" in f),),
                "synchformer_name": (get_foley_models(lambda f: "synch" in f),),
                }
            }

    RETURN_TYPES = ("FOLEYTUNE_DEPS",)
    FUNCTION = "load_dependencies"
    CATEGORY = "FoleyTune"

    def load_dependencies(self, vae_name, synchformer_name):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        deps = {}

        # Load local model files (VAE, Synchformer) — auto-download if missing
        deps['dac_model'] = load_dac_any(ensure_model_downloaded(vae_name), device=offload_device)
        synchformer_sd = load_torch_file(ensure_model_downloaded(synchformer_name), device=offload_device)
        syncformer_model = Synchformer()
        syncformer_model.load_state_dict(synchformer_sd, strict=False)
        deps['syncformer_model'] = syncformer_model.to(offload_device).eval()

        # Define pure tensor-based v2 preprocessing pipelines
        # SigLIP2 pipeline: The input is a (C,H,W) uint8 tensor.
        deps['siglip2_preprocess'] = v2.Compose([
            v2.Resize((512, 512), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.ToDtype(torch.float32, scale=True), # Converts uint8 [0,255] to float [0,1]
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Synchformer pipeline: The input is a (C,H,W) uint8 tensor.
        deps['syncformer_preprocess'] = v2.Compose([
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True), # Converts uint8 [0,255] to float [0,1]
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Load models from Hugging Face
        deps['siglip2_model'] = AutoModel.from_pretrained("google/siglip2-base-patch16-512").to(offload_device).eval()
        deps['clap_tokenizer'] = AutoTokenizer.from_pretrained("laion/larger_clap_general")
        deps['clap_model'] = ClapTextModelWithProjection.from_pretrained("laion/larger_clap_general").to(offload_device).eval()

        deps['device'] = device

        logger.info("Loaded all HunyuanVideoFoley dependencies.")
        return (AttributeDict(deps),)

# (HunyuanFoleySampler removed — FoleyTuneChunkedSampler handles single-shot as a special case)

# -----------------------------------------------------------------------------------
# NODE: Chunked Sampler for Long-Form Generation
# -----------------------------------------------------------------------------------

class FoleyTuneChunkedSampler:
    """Generate audio for long videos by chunking with overlap and crossfade.

    Connects to FoleyTuneFeatureExtractor's FOLEYTUNE_FEATURES output. Splits denoising
    into overlapping chunks and stitches with SaFa binary swap (best quality),
    latent-space crossfade, or waveform crossfade.

    For clips shorter than chunk_duration, runs a single pass with no overhead.
    """
    SAMPLER_NAMES = ["euler", "heun-2", "midpoint-2", "kutta-4"]
    CROSSFADE_MODES = ["safa", "latent", "waveform"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("FOLEYTUNE_MODEL",),
                "hunyuan_deps": ("FOLEYTUNE_DEPS",),
                "features": ("FOLEYTUNE_FEATURES",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "sampler_options": ("FOLEYTUNE_SAMPLER_OPTIONS",),
                "init_audio": ("AUDIO", {"tooltip": "Reference audio for audio2audio. Connect to use img2img-style generation."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                             "tooltip": "1.0=full generation from noise, 0.0=keep original. "
                                        "Uses sigma-based mapping for smooth control across the full range."}),
                "force_offload": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("audio_first", "audio_batch")
    FUNCTION = "generate_audio"
    CATEGORY = "FoleyTune"

    def generate_audio(
        self,
        hunyuan_model,
        hunyuan_deps,
        features,
        seed,
        steps,
        cfg_scale,
        sampler_options=None,
        init_audio=None,
        denoise=1.0,
        force_offload=True,
    ):
        opts = sampler_options or {}
        sampler = opts.get("sampler", "euler")
        batch_size = opts.get("batch_size", 1)
        chunk_duration = opts.get("chunk_duration", 8.0)
        crossfade_mode = opts.get("crossfade_mode", "safa")
        noise_blend = opts.get("noise_blend", 0.0)
        torch_compile_cfg = opts.get("torch_compile_cfg")
        block_swap_args = opts.get("block_swap_args")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        if hasattr(hunyuan_model, "_compilation_progress_counter"):
            hunyuan_model._compilation_progress_counter[0] = 0

        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "configs", "hunyuanvideo-foley-xxl.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")
        hunyuan_cfg = load_yaml(config_path)

        rng = torch.Generator(device="cpu").manual_seed(seed)
        duration = features["duration"]

        # Compute chunk boundaries
        min_overlap = chunk_duration * 0.2
        chunks = compute_chunk_boundaries(duration, chunk_duration, min_overlap)
        logger.info(f"Chunked generation: {len(chunks)} chunks for {duration:.1f}s "
                     f"(chunk={chunk_duration}s, mode={crossfade_mode})")
        for i, (ts, te) in enumerate(chunks):
            logger.info(f"  Chunk {i}: [{ts:.1f}, {te:.1f}]s ({te-ts:.1f}s)")

        # Apply torch.compile if configured
        if torch_compile_cfg is not None and not getattr(hunyuan_model, "_blocks_are_compiled", False):
            try:
                hunyuan_model = FoleyTuneTorchCompile._apply_torch_compile(
                    hunyuan_model, torch_compile_cfg
                )
            except Exception as e:
                logger.error(f"TorchCompile failed: {e}")

        # Place model on device
        if block_swap_args is not None:
            hunyuan_model.block_swap(
                blocks_to_swap=block_swap_args.get("blocks_to_swap", 0),
                use_non_blocking=block_swap_args.get("use_non_blocking", False),
                prefetch_blocks=block_swap_args.get("prefetch_blocks", 0),
                block_swap_debug=block_swap_args.get("block_swap_debug", False),
            )
        else:
            hunyuan_model.to(device)

        # Build model_dict
        model_dict_for_process = AttributeDict(dict(hunyuan_deps))
        model_dict_for_process["foley_model"] = hunyuan_model
        model_dict_for_process["device"] = device

        # Ensure DAC is on GPU
        hunyuan_deps["dac_model"].to(device=device, dtype=torch.float32)

        # Encode init audio to DAC latents if provided
        init_latents = None
        if init_audio is not None and denoise >= 1.0:
            logger.info("Audio2Audio: denoise=1.0 means init_audio is ignored (full generation from noise)")
        if init_audio is not None and denoise < 1.0:
            init_waveform = init_audio["waveform"]
            init_sr = init_audio["sample_rate"]
            # Ensure mono [B, 1, samples]
            if init_waveform.dim() == 2:
                init_waveform = init_waveform.unsqueeze(0)
            if init_waveform.shape[1] > 1:
                init_waveform = init_waveform[:, :1, :]  # take first channel
            if init_sr != 48000:
                init_waveform = torchaudio.functional.resample(init_waveform, init_sr, 48000)
            init_latents = encode_audio_to_latents(init_waveform, hunyuan_deps["dac_model"], device)
            # Ensure init_latents match expected duration
            audio_frame_rate = hunyuan_cfg.model_config.model_kwargs.audio_frame_rate
            expected_frames = int(duration * audio_frame_rate)
            if init_latents.shape[-1] != expected_frames:
                logger.warning(f"Audio2Audio: init_audio latent length {init_latents.shape[-1]} != "
                               f"expected {expected_frames} frames. Padding/trimming to match.")
                if init_latents.shape[-1] < expected_frames:
                    init_latents = F.pad(init_latents, (0, expected_frames - init_latents.shape[-1]))
                else:
                    init_latents = init_latents[:, :, :expected_frames]
            logger.info(f"Audio2Audio: encoded init_audio to latents {init_latents.shape}, denoise={denoise}")

        # Run chunked denoising
        decoded_waveform, sample_rate = chunked_denoise_process(
            features=features,
            chunks=chunks,
            crossfade_mode=crossfade_mode,
            model_dict=model_dict_for_process,
            cfg=hunyuan_cfg,
            guidance_scale=cfg_scale,
            num_inference_steps=steps,
            batch_size=batch_size,
            sampler=sampler,
            generator=rng,
            init_latents=init_latents,
            strength=denoise,
            noise_blend=noise_blend,
        )

        waveform_batch = decoded_waveform.float().cpu()

        if force_offload:
            hunyuan_model.to(offload_device)
            hunyuan_deps["dac_model"].to(offload_device)
            mm.soft_empty_cache()

        first_waveform = waveform_batch[0].unsqueeze(0)
        audio_first = {"waveform": first_waveform, "sample_rate": sample_rate}
        audio_batch = {"waveform": waveform_batch, "sample_rate": sample_rate}

        return (audio_first, audio_batch)

# -----------------------------------------------------------------------------------
# NODE: FoleyTune Model Sampling (shift override)
# -----------------------------------------------------------------------------------

class FoleyTuneModelSampling:
    """Override the flow matching shift parameter on the model.

    Shift controls the sigma schedule: 1.0 = linear (default), >1 biases
    toward higher noise levels (more creative/diverse), <1 biases toward
    lower noise levels (more conservative/faithful).

    Connect between Model Loader and Sampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FOLEYTUNE_MODEL",),
                "shift": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05,
                    "tooltip": "Flow matching shift. 1.0=linear (default). >1=more noise/diversity. <1=less noise/more faithful.",
                }),
            },
        }

    RETURN_TYPES = ("FOLEYTUNE_MODEL",)
    FUNCTION = "set_shift"
    CATEGORY = "FoleyTune"

    def set_shift(self, model, shift):
        model._flow_shift_override = shift
        return (model,)

# -----------------------------------------------------------------------------------
# NODE: FoleyTune Torch Compile (optional accelerator)
# -----------------------------------------------------------------------------------

class FoleyTuneTorchCompile:
    """Torch Compile.
    
    If you change anything like duration, or batch, it'll compile again and takes about 2 minutes on a 3090.
    Saves about 30% of the time.
    """
    DESCRIPTION = cleandoc(__doc__ or "")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (["inductor"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Capture entire graph (stricter); usually keep off"}),
                "mode": (["default", "reduce-overhead", "max-autotune"], {"default": "default"}),
                "dynamic": (["true", "false", "None"], {"default": "false", "tooltip": "Allow shape dynamism; safer when duration/batch vary"}),
                "dynamo_cache_limit": ("INT", {"default": 64, "min": 64, "max": 8192, "step": 64,
                                               "tooltip": "TorchDynamo graph cache size to limit graph explosion"}),
            }
        }

    # Emits a config object to be consumed by the Sampler
    RETURN_TYPES = ("FOLEYTUNE_COMPILE_CFG",)
    FUNCTION = "make_config"
    CATEGORY = "FoleyTune"

    def make_config(self, backend, mode, dynamic, fullgraph, dynamo_cache_limit):
        # Map tri-state string to Python value
        dyn_map = {"true": True, "false": False, "None": None}
        dynamic_val = dyn_map.get(str(dynamic), False)
        cfg = {
            "backend": backend,
            "mode": mode,
            "dynamic": dynamic_val,   # may be True/False/None
            "fullgraph": fullgraph,
            "dynamo_cache_limit": int(dynamo_cache_limit),
        }
        # returning a plain dict is fine for custom types in Comfy
        return (cfg,)
    
    # For reuse from the sampler.
    @staticmethod
    def _apply_torch_compile(model: nn.Module, compile_cfg: dict):
        """
        Applies torch.compile to the computationally heavy blocks of the model
        instead of the entire model. This improves compilation reliability and
        enables dynamic operations like BlockSwap in the main forward pass.
        
        This method also wraps each compiled block's forward pass to provide
        real-time progress feedback in the console during execution.
        Uses a weak reference to prevent memory leaks from reference cycles.
        """
        if hasattr(model, "_blocks_are_compiled") and model._blocks_are_compiled:
            logger.info("Model blocks are already compiled. Skipping setup.")
            return model

        try:
            torch._dynamo.config.cache_size_limit = int(compile_cfg.get("dynamo_cache_limit", 64))
        except Exception:
            pass

        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build.")

        # --- Signature Generation Helper Functions (defined inside to be captured by closure) ---
        def _sig(args, kwargs, block_name=None):
            def _meta(o):
                if torch.is_tensor(o):
                    dev = (o.device.type, o.device.index or 0)
                    return (
                        "T",
                        tuple(o.shape),
                        str(o.dtype),
                        dev,
                        tuple(o.stride()),
                        bool(o.requires_grad),
                        bool(o.is_contiguous()),
                    )
                if isinstance(o, (list, tuple)):
                    return tuple(_meta(x) for x in o)
                if isinstance(o, dict):
                    # sort for determinism
                    return tuple(sorted((k, _meta(v)) for k, v in o.items()))
                return (type(o).__name__,)

            meta = (block_name, _meta(args), _meta(kwargs))
            blob = repr(meta).encode()
            # 64-bit stable hash for set membership
            return int.from_bytes(hashlib.blake2s(blob, digest_size=8).digest(), "little")

        # --- JIT Compilation Progress Tracking Setup ---
        model._compilation_progress_counter = [0]
        model._total_blocks_to_compile = len(model.triple_blocks) + len(model.single_blocks)
        model._seen_compile_signatures = set() # This will store the signatures of compiled functions.

        def _create_logged_forward(original_forward_method, block_name, model_ref_weak): # Takes a weak reference now
            """
            A wrapper function that intercepts every call to a block's forward method
            to update a progress bar in the console for each denoising step.
            """

            def logged_forward(*args, **kwargs):
                model_ref = model_ref_weak() # Dereference the weak reference
                if model_ref is None:
                    # The original model has been garbage collected, just run the original forward
                    return original_forward_method(*args, **kwargs)

                # Calculate the signature of the current inputs.
                current_signature = _sig(args, kwargs, block_name)
                # Check if this specific signature has been compiled before.
                if current_signature not in model_ref._seen_compile_signatures:
                    model_ref._seen_compile_signatures.add(current_signature) # Mark as seen at every end
                    # It's a new signature, so a compilation will happen. Update the progress bar.
                    
                    counter_list = model_ref._compilation_progress_counter
                    total_blocks = model_ref._total_blocks_to_compile

                    # Only show the progress bar for the initial set of compilations.
                    if counter_list[0] < total_blocks:
                    # Increment the global counter every time a block is executed.
                        counter_list[0] += 1

                        # --- ASCII Progress Bar Logic ---
                        # Calculate progress for the current denoising step.
                        progress = counter_list[0] / total_blocks
                        bar_length = 40
                        filled_length = int(bar_length * progress)
                        bar = '█' * filled_length + '─' * (bar_length - filled_length)
                        print(f"\rHunyuanVideo-Foley: JIT Compiling {block_name}... [{bar}] {counter_list[0]}/{total_blocks} ({progress:.0%})", end="", flush=True)
                        if counter_list[0] >= total_blocks:
                            logger.info("\nHunyuanVideo-Foley: JIT Compilation finished.")

                with _CudaFactoriesDuringCompile():
                    return original_forward_method(*args, **kwargs)

            return logged_forward

        # --- Main Compilation Logic ---
        backend   = compile_cfg.get("backend", "inductor")
        mode      = compile_cfg.get("mode", "default")
        dynamic   = compile_cfg.get("dynamic", False)  # may be True/False/None
        fullgraph = compile_cfg.get("fullgraph", False)

        logger.info(f"torch.compile transformer blocks with backend='{backend}', mode='{mode}'...")

        # --- Compile and Wrap Triple-Stream Blocks ---
        logger.info(f"{len(model.triple_blocks)} TwoStreamCABlocks...")
        logger.info(f"{len(model.single_blocks)} SingleStreamBlocks...")

        model_ref_weak = weakref.ref(model) # Prevent memory leak in closure
        
        # --- Compile and Wrap Triple-Stream Blocks ---
        for i, block in enumerate(model.triple_blocks):
            original_block = block._orig_mod if hasattr(block, "_orig_mod") else block
            block_name = f"Triple-Stream Block {i+1}"
            try:
                compiled_block = torch.compile(original_block, backend=backend, mode=mode, dynamic=dynamic, fullgraph=fullgraph)
                compiled_block.forward = _create_logged_forward(compiled_block.forward, block_name, model_ref_weak)
                model.triple_blocks[i] = compiled_block
            except Exception as e:
                logger.error(f"Failed to compile {block_name}. Continuing without compiling. Error: {e}")

        # --- Compile and Wrap Single-Stream Blocks ---
        for i, block in enumerate(model.single_blocks):
            original_block = block._orig_mod if hasattr(block, "_orig_mod") else block
            block_name = f"Single-Stream Block {i+1}"
            try:
                compiled_block = torch.compile(original_block, backend=backend, mode=mode, dynamic=dynamic, fullgraph=fullgraph)
                compiled_block.forward = _create_logged_forward(compiled_block.forward, block_name, model_ref_weak)
                model.single_blocks[i] = compiled_block
            except Exception as e:
                logger.error(f"Failed to compile {block_name}. Continuing without compiling. Error: {e}")
        
        model._blocks_are_compiled = True
        return model

class FoleyTuneBlockSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks_to_swap": ("INT", {"default": 30, "min": 0, "max": 57, "step": 1, "tooltip": "Number of transformer blocks to offload to CPU. The model has 57 blocks in total (19 triple-stream + 38 single-stream)."}),
            },
            "optional": {
                # These are added for future compatibility, mirroring WanVideo's options.
                "use_non_blocking": ("BOOLEAN", {"default": False, "tooltip": "Use non-blocking memory transfer for offloading. Can be faster but reserves more RAM."}),
                "prefetch_blocks": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1, "tooltip": "Number of blocks to prefetch to GPU ahead of time. Hides data transfer latency."}),
                "block_swap_debug": ("BOOLEAN", {"default": False, "tooltip": "Enable debug logging for block swapping performance."}),
            },
        }
    RETURN_TYPES = ("FOLEYTUNE_BLOCKSWAP",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "set_args"
    CATEGORY = "FoleyTune"
    DESCRIPTION = "Settings for block swapping to reduce VRAM by offloading transformer blocks to CPU."

    def set_args(self, **kwargs):
        # This node simply bundles its arguments into a dictionary.
        return (kwargs,)

# -----------------------------------------------------------------------------------
# HELPER NODE: Select Audio From Batch
# -----------------------------------------------------------------------------------
class FoleyTuneSelectAudioFromBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_batch": ("AUDIO", {"tooltip": "An audio object containing a batch of waveforms."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 63, "tooltip": "The 0-based index of the audio to select from the batch."}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "select_audio"
    CATEGORY = "FoleyTune"

    def select_audio(self, audio_batch, index):
        waveform_batch = audio_batch['waveform']
        sample_rate = audio_batch['sample_rate']

        # Check if the index is valid
        if index >= waveform_batch.shape[0]:
            logger.warning(f"Index {index} is out of bounds for audio batch of size {waveform_batch.shape[0]}. Clamping to last item.")
            index = waveform_batch.shape[0] - 1

        # Select the waveform at the specified index and keep a batch dimension of 1
        selected_waveform = waveform_batch[index].unsqueeze(0)

        # Package it into the standard AUDIO dictionary format for other nodes
        audio_output = {"waveform": selected_waveform, "sample_rate": sample_rate}
        return (audio_output,)

# -----------------------------------------------------------------------------------
# NODE: FoleyTune Inpainter — regenerate a time region of existing audio
# -----------------------------------------------------------------------------------

class FoleyTuneInpainter:
    """Regenerate a time region of existing audio while keeping the rest.

    Encodes init_audio through DAC, builds a soft mask from start/end seconds,
    runs denoising with per-step replacement of known regions.
    """

    SAMPLER_NAMES = ["euler", "heun-2", "midpoint-2", "kutta-4"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("FOLEYTUNE_MODEL",),
                "hunyuan_deps": ("FOLEYTUNE_DEPS",),
                "features": ("FOLEYTUNE_FEATURES",),
                "init_audio": ("AUDIO",),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1,
                                   "tooltip": "Start of region to regenerate (seconds)."}),
                "end_seconds": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 60.0, "step": 0.1,
                                 "tooltip": "End of region to regenerate (seconds)."}),
                "cfg_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "steps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1}),
                "sampler": (cls.SAMPLER_NAMES, {"default": "euler"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "fade_frames": ("INT", {"default": 4, "min": 0, "max": 20, "step": 1,
                                 "tooltip": "Soft mask edge width in latent frames (~20ms each). Prevents DAC boundary clicks."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01,
                             "tooltip": "Regeneration strength in masked region. 1.0=fully regenerate, "
                                        "lower values preserve more of the original structure."}),
                "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "torch_compile_cfg": ("FOLEYTUNE_COMPILE_CFG",),
                "block_swap_args": ("FOLEYTUNE_BLOCKSWAP",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "inpaint"
    CATEGORY = "FoleyTune"
    DESCRIPTION = (
        "Regenerate a specific time region of existing audio while keeping the rest intact. "
        "Specify start/end seconds for the region to regenerate. Uses per-step latent replacement "
        "with soft mask edges to prevent boundary artifacts."
    )

    def inpaint(
        self,
        hunyuan_model,
        hunyuan_deps,
        features,
        init_audio,
        start_seconds,
        end_seconds,
        cfg_scale,
        steps,
        sampler,
        seed,
        fade_frames,
        denoise,
        force_offload,
        torch_compile_cfg=None,
        block_swap_args=None,
    ):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "configs", "hunyuanvideo-foley-xxl.yaml")
        hunyuan_cfg = load_yaml(config_path)

        rng = torch.Generator(device="cpu").manual_seed(seed)
        audio_frame_rate = hunyuan_cfg.model_config.model_kwargs.audio_frame_rate
        duration = features["duration"]

        if start_seconds >= end_seconds:
            raise ValueError(f"start_seconds ({start_seconds}) must be less than end_seconds ({end_seconds})")

        # Encode init audio
        init_waveform = init_audio["waveform"]
        init_sr = init_audio["sample_rate"]
        if init_waveform.dim() == 2:
            init_waveform = init_waveform.unsqueeze(0)
        if init_waveform.shape[1] > 1:
            init_waveform = init_waveform[:, :1, :]
        if init_sr != 48000:
            init_waveform = torchaudio.functional.resample(init_waveform, init_sr, 48000)

        # Apply torch.compile if configured
        if torch_compile_cfg is not None and not getattr(hunyuan_model, "_blocks_are_compiled", False):
            try:
                hunyuan_model = FoleyTuneTorchCompile._apply_torch_compile(
                    hunyuan_model, torch_compile_cfg
                )
            except Exception as e:
                logger.error(f"TorchCompile failed: {e}")

        # Place model
        if block_swap_args is not None:
            hunyuan_model.block_swap(
                blocks_to_swap=block_swap_args.get("blocks_to_swap", 0),
                use_non_blocking=block_swap_args.get("use_non_blocking", False),
                prefetch_blocks=block_swap_args.get("prefetch_blocks", 0),
                block_swap_debug=block_swap_args.get("block_swap_debug", False),
            )
        else:
            hunyuan_model.to(device)

        model_dict = AttributeDict(dict(hunyuan_deps))
        model_dict["foley_model"] = hunyuan_model
        model_dict["device"] = device
        hunyuan_deps["dac_model"].to(device=device, dtype=torch.float32)

        # DAC encode
        init_latents = encode_audio_to_latents(init_waveform, hunyuan_deps["dac_model"], device)
        target_dtype = hunyuan_model.dtype
        init_latents = init_latents.to(dtype=target_dtype)

        # Ensure latent length matches what the denoiser will create
        expected_frames = int(duration * audio_frame_rate)
        if init_latents.shape[-1] != expected_frames:
            logger.warning(f"Inpainting: DAC latent length {init_latents.shape[-1]} != "
                           f"expected {expected_frames}. Adjusting.")
            if init_latents.shape[-1] < expected_frames:
                init_latents = F.pad(init_latents, (0, expected_frames - init_latents.shape[-1]))
            else:
                init_latents = init_latents[:, :, :expected_frames]
        T_latent = init_latents.shape[-1]

        # Build inpaint mask [1, 1, T] — 1.0 = regenerate, 0.0 = keep
        frame_start = max(0, int(start_seconds * audio_frame_rate))
        frame_end = min(T_latent, int(end_seconds * audio_frame_rate))
        mask = torch.zeros(1, 1, T_latent, device=device, dtype=target_dtype)
        mask[:, :, frame_start:frame_end] = 1.0

        # Apply soft edges
        if fade_frames > 0:
            # Left edge
            fade_start = max(0, frame_start - fade_frames)
            for i in range(fade_start, frame_start):
                alpha = (i - fade_start + 1) / (fade_frames + 1)
                mask[:, :, i] = alpha
            # Right edge
            fade_end = min(T_latent, frame_end + fade_frames)
            for i in range(frame_end, fade_end):
                alpha = 1.0 - (i - frame_end + 1) / (fade_frames + 1)
                mask[:, :, i] = alpha

        logger.info(f"Inpainting: [{start_seconds:.1f}s, {end_seconds:.1f}s] -> "
                     f"frames [{frame_start}, {frame_end}] / {T_latent}, "
                     f"fade={fade_frames} frames")

        # Guard: model trained on ~8s chunks, inpainting works on full duration
        if duration > 16.0:
            logger.warning(f"Inpainting on {duration:.1f}s audio — quality may degrade beyond ~8s. "
                           "Consider trimming or using chunked generation instead.")

        # Generate noise for inpainting (consistent across steps)
        from diffusers.utils.torch_utils import randn_tensor
        inpaint_noise = randn_tensor(
            init_latents.shape, device=device, dtype=target_dtype, generator=rng
        )

        # Prepare features
        visual = {
            "siglip2_feat": features["clip_feat"].to(device),
            "syncformer_feat": features["sync_feat"].to(device),
        }
        text = {
            "text_feat": features["text_feat"].to(device),
            "uncond_text_feat": features["uncond_text_feat"].to(device),
        }

        # Run denoising with inpainting + optional partial regeneration
        audio, sample_rate = denoise_process_with_generator(
            visual, text, duration, model_dict, hunyuan_cfg,
            cfg_scale, steps, 1, sampler, rng,
            init_latents=init_latents,
            strength=denoise,
            inpaint_mask=mask,
            inpaint_original=init_latents,
            inpaint_noise=inpaint_noise,
        )

        if force_offload:
            hunyuan_model.to(offload_device)
            hunyuan_deps["dac_model"].to(offload_device)
            mm.soft_empty_cache()

        audio_out = {"waveform": audio.float().cpu(), "sample_rate": sample_rate}
        return (audio_out,)

# -----------------------------------------------------------------------------------
# NODE: FoleyTune Feature Blender — mix conditioning from two videos
# -----------------------------------------------------------------------------------

class FoleyTuneFeatureBlender:
    """Blend features from two FOLEYTUNE_FEATURES dicts for conditioning mixing.

    Interpolates CLIP, sync, and text features between two sources.
    Useful for blending visual guidance from different videos.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "features_a": ("FOLEYTUNE_FEATURES",),
                "features_b": ("FOLEYTUNE_FEATURES",),
                "blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                           "tooltip": "0.0=100% A, 1.0=100% B. Interpolates all feature tensors."}),
            },
        }

    RETURN_TYPES = ("FOLEYTUNE_FEATURES",)
    RETURN_NAMES = ("features",)
    FUNCTION = "blend_features"
    CATEGORY = "FoleyTune"
    DESCRIPTION = (
        "Blend conditioning features from two videos. "
        "Useful for mixing visual guidance from different sources."
    )

    def blend_features(self, features_a, features_b, blend):
        result = {}
        for key in ("clip_feat", "sync_feat", "text_feat", "uncond_text_feat"):
            a = features_a[key]
            b = features_b[key]
            min_len = min(a.shape[1], b.shape[1])
            a = a[:, :min_len, :]
            b = b[:, :min_len, :]
            result[key] = (1 - blend) * a + blend * b
        result["duration"] = min(features_a["duration"], features_b["duration"])
        return (result,)

# -----------------------------------------------------------------------------------
# NODE: FoleyTune Style Transfer — latent AdaIN between audio
# -----------------------------------------------------------------------------------

class FoleyTuneStyleTransfer:
    """Transfer audio style (timbre, room tone) from one audio to another via latent AdaIN.

    Encodes both content and style audio through DAC, transfers channel-wise
    mean and std from style to content in latent space, then decodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_audio": ("AUDIO", {"tooltip": "Audio whose structure/timing to keep."}),
                "style_audio": ("AUDIO", {"tooltip": "Audio whose tonal quality/timbre to transfer."}),
                "hunyuan_deps": ("FOLEYTUNE_DEPS",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                              "tooltip": "Style transfer strength. 0.0=no change, 1.0=full style transfer."}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "transfer_style"
    CATEGORY = "FoleyTune"
    DESCRIPTION = (
        "Transfer tonal characteristics from style_audio to content_audio "
        "using Adaptive Instance Normalization (AdaIN) in DAC latent space."
    )

    def transfer_style(self, content_audio, style_audio, hunyuan_deps, strength):
        device = mm.get_torch_device()
        dac = hunyuan_deps["dac_model"]
        dac.to(device=device, dtype=torch.float32)

        def _prep_wav(audio_dict):
            wav = audio_dict["waveform"]
            sr = audio_dict["sample_rate"]
            if wav.dim() == 2:
                wav = wav.unsqueeze(0)
            if wav.shape[1] > 1:
                wav = wav[:, :1, :]
            if sr != 48000:
                wav = torchaudio.functional.resample(wav, sr, 48000)
            return wav

        content_wav = _prep_wav(content_audio)
        style_wav = _prep_wav(style_audio)

        z_content = encode_audio_to_latents(content_wav, dac, device)
        z_style = encode_audio_to_latents(style_wav, dac, device)

        # AdaIN: normalize content, apply style statistics
        content_mean = z_content.mean(dim=-1, keepdim=True)
        content_std = z_content.std(dim=-1, keepdim=True) + 1e-6
        style_mean = z_style.mean(dim=-1, keepdim=True)
        style_std = z_style.std(dim=-1, keepdim=True) + 1e-6

        z_normalized = (z_content - content_mean) / content_std
        z_styled = z_normalized * style_std + style_mean

        # Blend with original based on strength
        z_out = (1 - strength) * z_content + strength * z_styled

        # Decode
        with torch.inference_mode():
            dac_weight = next(dac.parameters())
            audio = dac.decode(z_out.to(device=dac_weight.device, dtype=dac_weight.dtype))

        # DAC always outputs at 48kHz — trim to content length
        content_samples = content_wav.shape[-1]
        audio = audio[:, :, :content_samples]

        # Offload DAC to free VRAM
        offload_device = mm.unet_offload_device()
        dac.to(offload_device)
        mm.soft_empty_cache()

        audio_out = {"waveform": audio.float().cpu(), "sample_rate": 48000}
        return (audio_out,)

class FoleyTuneSamplerOptions:
    """Optional settings for the FoleyTune Chunked Sampler.

    When not connected, the sampler uses sensible defaults (euler, batch=1,
    8s chunks, 1.6s overlap, safa crossfade, full denoise, no noise blend).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": (FoleyTuneChunkedSampler.SAMPLER_NAMES, {"default": "euler"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 6, "step": 1}),
                "chunk_duration": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 15.0, "step": 0.1,
                                    "tooltip": "Duration of each chunk in seconds. 8s matches training length."}),
                "crossfade_mode": (FoleyTuneChunkedSampler.CROSSFADE_MODES, {"default": "safa",
                                    "tooltip": "safa: binary swap during denoising (best). latent: blend before DAC. waveform: blend after DAC."}),
                "noise_blend": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                 "tooltip": "Blend reference audio dynamics into the initial noise. "
                                            "0.0=pure gaussian, 1.0=preserves temporal rhythm from reference. "
                                            "Try 0.3-0.5 to keep timing while regenerating timbre."}),
            },
            "optional": {
                "torch_compile_cfg": ("FOLEYTUNE_COMPILE_CFG",),
                "block_swap_args": ("FOLEYTUNE_BLOCKSWAP",),
            },
        }

    RETURN_TYPES = ("FOLEYTUNE_SAMPLER_OPTIONS",)
    RETURN_NAMES = ("sampler_options",)
    FUNCTION = "build"
    CATEGORY = "FoleyTune"

    def build(self, sampler, batch_size, chunk_duration,
              crossfade_mode, noise_blend,
              torch_compile_cfg=None, block_swap_args=None):
        return ({
            "sampler": sampler,
            "batch_size": batch_size,
            "chunk_duration": chunk_duration,
            "crossfade_mode": crossfade_mode,
            "noise_blend": noise_blend,
            "torch_compile_cfg": torch_compile_cfg,
            "block_swap_args": block_swap_args,
        },)


# -----------------------------------------------------------------------------------
# NODE: FoleyTune Video Loader (path-based)
# -----------------------------------------------------------------------------------

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
    OUTPUT_NODE = True

    def load_video(self, hunyuan_deps, video_path, start_time=0.0, duration=0.0):
        if not video_path or not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        features = _extract_video_features(video_path, hunyuan_deps, start_time, duration)

        # Symlink video to temp dir for inline preview (zero-cost, no copy)
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        ext = os.path.splitext(video_path)[1] or ".mp4"
        temp_name = f"foleytune_preview_{os.path.basename(video_path)}"
        temp_path = os.path.join(temp_dir, temp_name)
        if not os.path.exists(temp_path):
            os.symlink(os.path.abspath(video_path), temp_path)

        return {"ui": {"gifs": [{"filename": temp_name, "subfolder": "", "type": "temp",
                                  "format": f"video/{ext.lstrip('.')}"}]},
                "result": (features, features["duration"])}

    @classmethod
    def IS_CHANGED(cls, video_path, **kwargs):
        if not video_path or not os.path.isfile(video_path):
            return ""
        return os.path.getmtime(video_path)

# -----------------------------------------------------------------------------------
# NODE: FoleyTune Video Loader (Upload / combo)
# -----------------------------------------------------------------------------------

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

        # Symlink to temp dir for preview (zero-cost, no copy)
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        ext = os.path.splitext(video)[1] or ".mp4"
        temp_name = f"foleytune_preview_{video}"
        temp_path = os.path.join(temp_dir, temp_name)
        if not os.path.exists(temp_path):
            os.symlink(os.path.abspath(video_path), temp_path)

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

# -----------------------------------------------------------------------------------
# NODE: FoleyTune Video Combiner — mux audio onto video
# -----------------------------------------------------------------------------------

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
            base, ext = os.path.splitext(source_video)
            output_path = f"{base}_foley{ext}"

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        waveform = audio["waveform"].squeeze(0).cpu().numpy()
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

        # Symlink to temp for preview (zero-cost, no copy)
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        temp_name = f"foleytune_combined_{os.path.basename(output_path)}"
        temp_path = os.path.join(temp_dir, temp_name)
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        os.symlink(os.path.abspath(output_path), temp_path)

        ext = os.path.splitext(output_path)[1] or ".mp4"
        return {"ui": {"gifs": [{"filename": temp_name, "subfolder": "", "type": "temp",
                                  "format": f"video/{ext.lstrip('.')}"}]},
                "result": (str(output_path),)}

# -----------------------------------------------------------------------------------
# NODE MAPPINGS - This is how ComfyUI discovers the nodes.
# -----------------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "FoleyTuneModelLoader": FoleyTuneModelLoader,
    "FoleyTuneDependenciesLoader": FoleyTuneDependenciesLoader,
    "FoleyTuneModelSampling": FoleyTuneModelSampling,
    "FoleyTuneChunkedSampler": FoleyTuneChunkedSampler,
    "FoleyTuneTorchCompile": FoleyTuneTorchCompile,
    "FoleyTuneBlockSwap": FoleyTuneBlockSwap,
    "FoleyTuneSelectAudioFromBatch": FoleyTuneSelectAudioFromBatch,
    "FoleyTuneInpainter": FoleyTuneInpainter,
    "FoleyTuneFeatureBlender": FoleyTuneFeatureBlender,
    "FoleyTuneStyleTransfer": FoleyTuneStyleTransfer,
    "FoleyTuneSamplerOptions": FoleyTuneSamplerOptions,
    "FoleyTuneVideoLoader": FoleyTuneVideoLoader,
    "FoleyTuneVideoLoaderUpload": FoleyTuneVideoLoaderUpload,
    "FoleyTuneVideoCombiner": FoleyTuneVideoCombiner,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FoleyTuneModelLoader": "FoleyTune Model Loader",
    "FoleyTuneDependenciesLoader": "FoleyTune Dependencies Loader",
    "FoleyTuneModelSampling": "FoleyTune Model Sampling",
    "FoleyTuneChunkedSampler": "FoleyTune Chunked Sampler",
    "FoleyTuneTorchCompile": "FoleyTune Torch Compile",
    "FoleyTuneBlockSwap": "FoleyTune BlockSwap Settings",
    "FoleyTuneSelectAudioFromBatch": "FoleyTune Select Audio From Batch",
    "FoleyTuneInpainter": "FoleyTune Inpainter",
    "FoleyTuneFeatureBlender": "FoleyTune Feature Blender",
    "FoleyTuneStyleTransfer": "FoleyTune Style Transfer",
    "FoleyTuneSamplerOptions": "FoleyTune Sampler Options",
    "FoleyTuneVideoLoader": "FoleyTune Video Loader",
    "FoleyTuneVideoLoaderUpload": "FoleyTune Video Loader (Upload)",
    "FoleyTuneVideoCombiner": "FoleyTune Video Combiner",
}
