"""LoRA training nodes for HunyuanVideo-Foley."""

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
import torchaudio
from loguru import logger

import folder_paths
import comfy.model_management as mm

from .lora.lora import (
    apply_lora, get_lora_state_dict, load_lora,
    FOLEY_TARGET_PRESETS, LoRALinear,
)
from .lora.train import (
    prepare_dataset, sample_timesteps, flow_matching_loss,
    generate_eval_sample, save_loss_curve, save_checkpoint, save_meta_json,
)
from .lora.spectral_metrics import spectral_metrics, reference_metrics


logger.remove()
logger.add(sys.stdout, level="INFO", format="HunyuanVideo-Foley LoRA: {message}")


# --- Node 1: Feature Extractor ----------------------------------------------

class FoleyFeatureExtractor:
    """Extract and cache SigLIP2/Synchformer/CLAP features + audio for LoRA training."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_deps": ("HUNYUAN_DEPS",),
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0, "step": 0.1}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                              "tooltip": "Clip duration in seconds. 0 = auto from audio length."}),
                "cache_dir": ("STRING", {"default": ""}),
                "name": ("STRING", {"default": "clip",
                          "tooltip": "Base name for auto-incremented files (e.g. clip -> clip_001.npz)"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("npz_path",)
    FUNCTION = "extract_features"
    CATEGORY = "audio/HunyuanFoley/LoRA"
    OUTPUT_NODE = True

    def extract_features(self, hunyuan_deps, image, audio, prompt, frame_rate, duration,
                         cache_dir, name):
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
        audio_out_path = cache_dir / f"{name}_{idx:03d}.wav"

        # -- Extract visual features --
        # image is [B, H, W, C] float32 in [0,1] from ComfyUI
        # Convert to [1, T, C, H, W] uint8 for preprocessing
        frames_bhwc = image  # [T, H, W, C]
        frames_tchw = frames_bhwc.permute(0, 3, 1, 2)  # [T, C, H, W]
        frames_uint8 = (frames_tchw * 255).clamp(0, 255).to(torch.uint8)

        waveform = audio["waveform"]  # [1, C, L]
        sample_rate = audio["sample_rate"]

        # Compute duration from audio if not specified
        if duration <= 0:
            duration = waveform.shape[-1] / sample_rate

        # Resample frames to target FPS for each extractor
        total_frames = frames_uint8.shape[0]

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
        hunyuan_deps.clap_model.to(offload_device)

        torch.cuda.empty_cache()

        # Save .npz
        np.savez(
            str(npz_path),
            clip_features=clip_features.numpy(),
            sync_features=sync_features.numpy(),
            text_embedding=text_embedding.numpy(),
            prompt=prompt,
            duration=duration,
            fps=frame_rate,
        )

        # Save audio file
        wav = waveform.squeeze(0)  # [C, L]
        if sample_rate != 48000:
            wav = torchaudio.functional.resample(wav, sample_rate, 48000)
        torchaudio.save(str(audio_out_path), wav, 48000)

        logger.info(f"Saved features to {npz_path}")
        logger.info(f"  clip_features: {clip_features.shape}, sync_features: {sync_features.shape}")
        logger.info(f"  text_embedding: {text_embedding.shape}, duration: {duration:.2f}s")

        return (str(npz_path),)


# --- Node 6: VAE Roundtrip --------------------------------------------------

class FoleyVAERoundtrip:
    """Encode audio through DAC, decode back. Reveals codec quality ceiling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_deps": ("HUNYUAN_DEPS",),
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "roundtrip"
    CATEGORY = "audio/HunyuanFoley/LoRA"

    def roundtrip(self, hunyuan_deps, audio):
        device = mm.get_torch_device()
        dac = hunyuan_deps.dac_model

        waveform = audio["waveform"]  # [1, C, L]
        sample_rate = audio["sample_rate"]

        # Resample to 48kHz if needed
        if sample_rate != 48000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 48000)

        # Convert to mono
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)

        # DAC encode -> decode
        # NOTE: DAC with continuous=True returns DiagonalGaussianDistribution, not tensor
        dac.to(device)
        with torch.no_grad():
            audio_in = waveform.to(device=device, dtype=torch.float32)
            z_dist, _, _, _, _ = dac.encode(audio_in)
            z = z_dist.sample()
            reconstructed = dac.decode(z)
        dac.cpu()
        torch.cuda.empty_cache()

        # Normalize output
        out = reconstructed.cpu().float()
        rms = torch.sqrt(torch.mean(out ** 2))
        target_rms = 10 ** (-27 / 20)
        if rms > 1e-8:
            out = out * (target_rms / rms)
        out = out.clamp(-1.0, 1.0)

        return ({"waveform": out, "sample_rate": 48000},)
