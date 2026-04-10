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
    generate_eval_sample, save_checkpoint, save_meta_json,
)
from .lora.spectral_metrics import spectral_metrics, reference_metrics
from PIL import Image, ImageDraw


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


def _draw_loss_curve(losses, log_interval=50, start_step=0, smoothed=None):
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
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0, "step": 0.1}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                              "tooltip": "Clip duration in seconds. 0 = auto from audio length."}),
                "cache_dir": ("STRING", {"default": ""}),
                "name": ("STRING", {"default": "clip",
                          "tooltip": "Base name for auto-incremented files (e.g. clip -> clip_001.npz)"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("npz_path",)
    FUNCTION = "extract_features"
    CATEGORY = "audio/HunyuanFoley/LoRA"
    OUTPUT_NODE = True

    def extract_features(self, hunyuan_deps, image, prompt, frame_rate, duration,
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

    RETURN_TYPES = ("HUNYUAN_MODEL", "IMAGE")
    RETURN_NAMES = ("model", "loss_curve")
    OUTPUT_TOOLTIPS = (
        "Model with trained LoRA adapter applied.",
        "Training loss curve (smoothed).",
    )
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

        step = start_step  # default in case loop doesn't execute
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
                _draw_loss_curve(losses, start_step=start_step, smoothed=_smooth_losses(losses)).save(str(output_path / "loss.png"))

                # Generate eval sample
                model.eval()
                wav, sr = generate_eval_sample(
                    model, hunyuan_deps.dac_model, dataset[0], device, dtype,
                )
                wav_t = torch.from_numpy(wav)
                if wav_t.ndim == 1:
                    wav_t = wav_t.unsqueeze(0)
                _save_wav(samples_path / f"step_{step+1:05d}.wav", wav_t, sr)
                model.train()

        # -- Save final --
        final_path = output_path / "adapter_final.pt"
        meta["steps_completed"] = step + 1 if step >= start_step else start_step
        save_checkpoint(model, optimizer, scheduler, step + 1, meta, final_path, final=True)
        save_meta_json(meta, output_path / "meta.json")
        # Draw and save loss curve
        smoothed = _smooth_losses(losses)
        loss_img = _draw_loss_curve(losses, log_interval=50, start_step=start_step, smoothed=smoothed)
        loss_img.save(str(output_path / "loss.png"))
        loss_curve_tensor = _pil_to_tensor(loss_img)

        elapsed_total = time.time() - t_start
        logger.info(f"Training complete: {elapsed_total:.0f}s, final loss: {np.mean(losses[-100:]):.4f}")
        logger.info(f"Adapter saved to {final_path}")

        # Return model with LoRA active (on CPU for ComfyUI pipeline)
        model.eval()
        model.to(mm.unet_offload_device())
        return (model, loss_curve_tensor)


# --- Node 3: LoRA Loader ----------------------------------------------------

class FoleyLoRALoader:
    """Load a trained LoRA adapter into a Foley model for inference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("HUNYUAN_MODEL",),
                "adapter_path": ("STRING", {"default": ""}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("HUNYUAN_MODEL",)
    FUNCTION = "load_adapter"
    CATEGORY = "audio/HunyuanFoley/LoRA"

    def load_adapter(self, hunyuan_model, adapter_path, strength):
        if not adapter_path or not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)

        # Handle both raw state_dict and wrapped checkpoint formats
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            meta = ckpt.get("meta", {})
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

        return (model,)


# --- Node 4: LoRA Scheduler -------------------------------------------------

class FoleyLoRAScheduler:
    """Run multiple LoRA training experiments from a JSON sweep configuration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("HUNYUAN_MODEL",),
                "hunyuan_deps": ("HUNYUAN_DEPS",),
                "sweep_json": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("summary_path", "comparison_curves")
    OUTPUT_TOOLTIPS = (
        "Path to experiment_summary.json.",
        "All smoothed loss curves overlaid on the same axes.",
    )
    FUNCTION = "run_sweep"
    CATEGORY = "audio/HunyuanFoley/LoRA"
    OUTPUT_NODE = True

    # Default training params (mirrors trainer node defaults)
    _PARAM_DEFAULTS = {
        "target": "all_attn_mlp", "rank": 64, "alpha": 64.0,
        "lr": 1e-4, "steps": 3000, "batch_size": 8, "grad_accum": 1,
        "warmup_steps": 100, "save_every": 500,
        "timestep_mode": "logit_normal", "precision": "bf16", "seed": 42,
        "logit_normal_sigma": 1.0, "curriculum_switch": 0.6,
        "init_mode": "standard", "use_rslora": False, "lora_dropout": 0.0,
        "lora_plus_ratio": 1.0, "schedule_type": "constant",
        "latent_mixup_alpha": 0.0, "latent_noise_sigma": 0.0,
    }

    def _merge_config(self, base, experiment):
        merged = {**self._PARAM_DEFAULTS, **base}
        for k, v in experiment.items():
            if k not in ("id", "description"):
                merged[k] = v
        return merged

    def run_sweep(self, hunyuan_model, hunyuan_deps, sweep_json):
        if not os.path.exists(sweep_json):
            raise FileNotFoundError(f"Sweep JSON not found: {sweep_json}")

        with open(sweep_json) as f:
            sweep = json.load(f)

        sweep_name = sweep.get("name", "sweep")
        data_dir = sweep["data_dir"]
        output_root = Path(sweep.get("output_root", f"lora_output/{sweep_name}"))
        base_config = sweep.get("base", {})
        experiments = sweep.get("experiments", [])

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

        # Prepare dataset once
        device = mm.get_torch_device()
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        base_precision = base_config.get("precision", "bf16")
        dtype = dtype_map[base_precision]

        logger.info(f"Preparing shared dataset from {data_dir}...")
        dataset = prepare_dataset(data_dir, hunyuan_deps.dac_model, device, dtype)

        # Collect loss histories for comparison chart
        all_loss_histories = {}

        for exp in experiments:
            exp_id = exp.get("id", f"exp_{len(results)}")

            if exp_id in completed_ids:
                logger.info(f"Skipping completed experiment: {exp_id}")
                # Load loss history if available
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
                # Deep copy model for this experiment
                model = copy.deepcopy(hunyuan_model)
                model.to(device=device, dtype=dtype)
                model.train()

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

                optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)

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

                import random
                torch.manual_seed(config["seed"])
                random.seed(config["seed"])
                np.random.seed(config["seed"])

                losses = []
                n_clips = len(dataset)
                t_start = time.time()

                for step in range(config["steps"]):
                    # Skip flag
                    skip_flag = output_root / "skip_current.flag"
                    if skip_flag.exists():
                        logger.info(f"Skip flag detected for {exp_id} at step {step}")
                        ckpt_path = exp_dir / f"adapter_cancelled_step{step:05d}.pt"
                        meta = {**config, "steps_completed": step}
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

                    t = sample_timesteps(
                        bs, config["timestep_mode"], device, dtype,
                        sigma=config["logit_normal_sigma"],
                        curriculum_switch=config["curriculum_switch"],
                        step=step, total_steps=config["steps"],
                    )

                    loss = flow_matching_loss(model, batch_latents, t, batch_clip, batch_sync, batch_text, device, dtype)
                    loss = loss / config["grad_accum"]
                    loss.backward()

                    if (step + 1) % config["grad_accum"] == 0:
                        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                        optimizer.step()
                        lr_sched.step()
                        optimizer.zero_grad()

                    losses.append(loss.item() * config["grad_accum"])

                    if (step + 1) % config["save_every"] == 0:
                        meta = {**config, "steps_completed": step + 1}
                        ckpt_path = exp_dir / f"adapter_step{step+1:05d}.pt"
                        save_checkpoint(model, optimizer, lr_sched, step + 1, meta, ckpt_path)

                # Save final
                meta = {**config, "steps_completed": config["steps"]}
                final_path = exp_dir / "adapter_final.pt"
                save_checkpoint(model, optimizer, lr_sched, config["steps"], meta, final_path, final=True)
                # Draw and save per-experiment loss curve
                smoothed = _smooth_losses(losses)
                loss_img = _draw_loss_curve(losses, smoothed=smoothed)
                loss_img.save(str(exp_dir / "loss.png"))

                # Save loss history for comparison chart
                with open(exp_dir / "loss_history.json", "w") as f:
                    json.dump(losses, f)
                all_loss_histories[exp_id] = losses

                elapsed = time.time() - t_start
                exp_result.update({
                    "status": "completed",
                    "final_loss": float(np.mean(losses[-100:])),
                    "min_loss": float(min(losses)),
                    "adapter_path": str(final_path),
                    "duration_seconds": elapsed,
                })

                del model, optimizer, lr_sched
                gc.collect()
                torch.cuda.empty_cache()

            except _SkipExperiment as e:
                exp_result["status"] = f"skipped: {e}"
            except Exception as e:
                exp_result["status"] = f"failed: {e}"
                logger.error(f"Experiment {exp_id} failed: {e}")

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

        # Generate comparison chart
        curve_data = [{"id": eid, "loss_history": lh} for eid, lh in all_loss_histories.items()]
        comparison_img = _draw_comparison_curves(curve_data)
        comparison_img.save(str(output_root / "loss_comparison.png"))
        comparison_tensor = _pil_to_tensor(comparison_img)

        logger.info(f"Sweep complete: {len(results)} experiments")
        return (str(summary_path), comparison_tensor)


class _SkipExperiment(Exception):
    pass



# --- Node 5: LoRA Evaluator -------------------------------------------------

class FoleyLoRAEvaluator:
    """Compare multiple LoRA adapters by generating audio and computing spectral metrics."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_model": ("HUNYUAN_MODEL",),
                "hunyuan_deps": ("HUNYUAN_DEPS",),
                "eval_json": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "evaluate"
    CATEGORY = "audio/HunyuanFoley/LoRA"
    OUTPUT_NODE = True

    def evaluate(self, hunyuan_model, hunyuan_deps, eval_json):
        if not os.path.exists(eval_json):
            raise FileNotFoundError(f"Eval JSON not found: {eval_json}")

        with open(eval_json) as f:
            spec = json.load(f)

        data_dir = spec["data_dir"]
        output_dir = Path(spec.get("output_dir", "lora_eval"))
        num_steps = spec.get("steps", 25)
        seed = spec.get("seed", 42)
        adapters = spec.get("adapters", [])

        output_dir.mkdir(parents=True, exist_ok=True)

        device = mm.get_torch_device()
        dtype = torch.bfloat16

        # Prepare dataset
        dataset = prepare_dataset(data_dir, hunyuan_deps.dac_model, device, dtype)

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
                ref_wav, ref_sr = torchaudio.load(str(ref_path))
                if ref_sr != 48000:
                    ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, 48000)
                ref_wav_np = ref_wav.mean(dim=0).numpy()  # mono
                ref_m = spectral_metrics(ref_wav_np, 48000)
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
                ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
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


# --- Node Mappings -----------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "FoleyFeatureExtractor": FoleyFeatureExtractor,
    "FoleyLoRATrainer": FoleyLoRATrainer,
    "FoleyLoRALoader": FoleyLoRALoader,
    "FoleyLoRAScheduler": FoleyLoRAScheduler,
    "FoleyLoRAEvaluator": FoleyLoRAEvaluator,
    "FoleyVAERoundtrip": FoleyVAERoundtrip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoleyFeatureExtractor": "Foley Feature Extractor",
    "FoleyLoRATrainer": "Foley LoRA Trainer",
    "FoleyLoRALoader": "Foley LoRA Loader",
    "FoleyLoRAScheduler": "Foley LoRA Scheduler",
    "FoleyLoRAEvaluator": "Foley LoRA Evaluator",
    "FoleyVAERoundtrip": "Foley VAE Roundtrip",
}
