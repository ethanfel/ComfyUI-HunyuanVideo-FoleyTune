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


# --- Node 2: LoRA Trainer ---------------------------------------------------

class FoleyLoRATrainer:
    """Train a LoRA adapter for HunyuanVideo-Foley via flow matching."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("HUNYUAN_MODEL",),
                "hunyuan_deps": ("HUNYUAN_DEPS",),
                "data_dir": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": ""}),
                "target": (list(FOLEY_TARGET_PRESETS.keys()), {"default": "all_attn_mlp"}),
                "rank": ("INT", {"default": 64, "min": 4, "max": 128, "step": 4}),
                "alpha": ("FLOAT", {"default": 64.0, "min": 1.0, "max": 128.0}),
                "lr": ("FLOAT", {"default": 1e-4, "min": 1e-6, "max": 1e-2, "step": 1e-5}),
                "steps": ("INT", {"default": 3000, "min": 100, "max": 50000}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 64}),
                "grad_accum": ("INT", {"default": 1, "min": 1, "max": 32}),
                "warmup_steps": ("INT", {"default": 100, "min": 0, "max": 2000}),
                "save_every": ("INT", {"default": 500, "min": 50, "max": 10000}),
                "timestep_mode": (["uniform", "logit_normal", "curriculum"], {"default": "logit_normal"}),
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
                "resume_from": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("HUNYUAN_MODEL",)
    FUNCTION = "train"
    CATEGORY = "audio/HunyuanFoley/LoRA"
    OUTPUT_NODE = True

    def train(self, hunyuan_model, hunyuan_deps, data_dir, output_dir, target, rank,
              alpha, lr, steps, batch_size, grad_accum, warmup_steps, save_every,
              timestep_mode, precision, seed,
              logit_normal_sigma=1.0, curriculum_switch=0.6,
              init_mode="standard", use_rslora=False, lora_dropout=0.0,
              lora_plus_ratio=1.0, schedule_type="constant",
              latent_mixup_alpha=0.0, latent_noise_sigma=0.0,
              resume_from=""):

        import random
        device = mm.get_torch_device()
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map[precision]

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        samples_path = output_path / "samples"
        samples_path.mkdir(exist_ok=True)

        # -- Prepare dataset --
        logger.info("Preparing dataset...")
        dataset = prepare_dataset(data_dir, hunyuan_deps.dac_model, device, dtype)
        n_clips = len(dataset)
        logger.info(f"Dataset ready: {n_clips} clips")

        # -- Setup model with LoRA --
        model = copy.deepcopy(hunyuan_model)
        model.to(device=device, dtype=dtype)
        model.train()

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

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)

        # -- LR Scheduler --
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            if schedule_type == "cosine":
                progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
                return 0.5 * (1 + np.cos(np.pi * progress))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # -- Resume --
        start_step = 0
        if resume_from and os.path.exists(resume_from):
            ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
            load_lora(model, ckpt["state_dict"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_step = ckpt.get("step", 0)
            logger.info(f"Resumed from step {start_step}: {resume_from}")
            del ckpt

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
            "n_clips": n_clips, "precision": precision, "seed": seed,
        }

        losses = []
        log_interval = 50

        logger.info(f"Starting training: {steps} steps, batch {batch_size}, lr {lr}")
        t_start = time.time()

        for step in range(start_step, steps):
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

            # Sample timesteps
            t = sample_timesteps(
                batch_size, timestep_mode, device, dtype,
                sigma=logit_normal_sigma, curriculum_switch=curriculum_switch,
                step=step, start_step=start_step, total_steps=steps,
            )

            # Forward + loss
            loss = flow_matching_loss(
                model, batch_latents, t, batch_clip, batch_sync, batch_text, device, dtype,
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

            losses.append(loss.item() * grad_accum)

            # Logging
            if (step + 1) % log_interval == 0:
                avg_loss = np.mean(losses[-log_interval:])
                elapsed = time.time() - t_start
                logger.info(f"Step {step+1}/{steps} | loss: {avg_loss:.4f} | "
                           f"lr: {scheduler.get_last_lr()[0]:.2e} | "
                           f"elapsed: {elapsed:.0f}s")

            # Save checkpoint + eval sample
            if (step + 1) % save_every == 0:
                ckpt_path = output_path / f"adapter_step{step+1:05d}.pt"
                save_checkpoint(model, optimizer, scheduler, step + 1, meta, ckpt_path)
                save_loss_curve(losses, output_path / "loss.png", start_step)

                # Generate eval sample
                model.eval()
                wav, sr = generate_eval_sample(
                    model, hunyuan_deps.dac_model, dataset[0], device, dtype,
                )
                wav_t = torch.from_numpy(wav)
                if wav_t.ndim == 1:
                    wav_t = wav_t.unsqueeze(0)
                torchaudio.save(str(samples_path / f"step_{step+1:05d}.wav"), wav_t, sr)
                model.train()

        # -- Save final --
        final_path = output_path / "adapter_final.pt"
        meta["steps_completed"] = step + 1 if step >= start_step else start_step
        save_checkpoint(model, optimizer, scheduler, step + 1, meta, final_path, final=True)
        save_meta_json(meta, output_path / "meta.json")
        save_loss_curve(losses, output_path / "loss.png", start_step)

        elapsed_total = time.time() - t_start
        logger.info(f"Training complete: {elapsed_total:.0f}s, final loss: {np.mean(losses[-100:]):.4f}")
        logger.info(f"Adapter saved to {final_path}")

        # Return model with LoRA active (on CPU for ComfyUI pipeline)
        model.eval()
        model.to(mm.unet_offload_device())
        return (model,)
