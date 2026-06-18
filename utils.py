# utils.py
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from diffusers.utils.torch_utils import randn_tensor
from comfy.utils import load_torch_file, ProgressBar
from comfy.model_management import throw_exception_if_processing_interrupted

# --- Optional imports from the original HunyuanVideo-Foley package ---
try:
    from hunyuanvideo_foley.models.dac_vae.model.dac import DAC
    from hunyuanvideo_foley.utils.schedulers import FlowMatchDiscreteScheduler
    from hunyuanvideo_foley.utils.feature_utils import (
        encode_video_with_siglip2,
        encode_video_with_sync,
        encode_text_feat,
    )
except Exception:
    # Defer ImportError until the calling site actually uses these helpers.
    DAC = None
    pass

# -----------------------------------------------------------------------------------
# HELPER FUNCTIONS - ADAPTED FOR COMFYUI WORKFLOW
# These are modified versions of the original library's functions to make them
# compatible with ComfyUI's data flow (e.g., accepting a torch.Generator).
# -----------------------------------------------------------------------------------


def _apply_lora_for_time(model, lora_schedule, time_sec, base_state, ckpt_cache):
    """Restore base weights and apply the LoRA active at time_sec (if any).

    Args:
        ckpt_cache: dict used as a per-generation session cache — caller owns
                    its lifetime and clears it after the generation loop.
    """
    from .lora.lora import apply_lora, load_lora, remove_lora, FOLEY_TARGET_PRESETS

    remove_lora(model)
    # Standard LoRAs never touch base weights — they live in module.base and
    # survive wrap/unwrap — so the (expensive, full-model CPU->GPU) base reload
    # is only needed after a pissa load dirtied them. Skipping it is what makes
    # per-segment hot-swap (many swaps per generation) affordable.
    if ckpt_cache.get("__base_dirty__"):
        model.load_state_dict(base_state, strict=False)
        ckpt_cache["__base_dirty__"] = False

    target = None
    for seg in lora_schedule:
        if seg["start_sec"] <= time_sec < seg["end_sec"]:
            target = seg
            break

    # target is None  -> uncovered gap; target with no lora_path -> prompt-only
    # (regular-video) segment. Both run the base model; the segment's text_feat
    # (if any) is applied separately when the per-chunk text is built.
    if target is None or not target.get("lora_path"):
        model._event_conditioning_enabled = False
        model._event_strength = 1.0
        # DEBUG: this fires once per chunk PER diffusion step (interleaved
        # denoising), so it would flood at INFO. The one-time per-chunk plan is
        # logged in chunked_denoise_process instead.
        logger.debug(f"LoRA swap @ {time_sec:.1f}s: base model (no LoRA)")
        return

    lora_path = target["lora_path"]

    # Session-scoped cache — lives only for this generation. Resolve the
    # checkpoint exactly like FoleyTuneLoRALoaderPair._load so hot-swap and the
    # normal loader treat the same adapter identically (safetensors sidecar
    # .json, schedule-free eval weights, EMA, rank inference, pissa, rsLoRA).
    if lora_path in ckpt_cache:
        state_dict, rank, alpha, target_suffixes, init_mode, use_rslora, event_enabled, event_strength = ckpt_cache[lora_path]
    else:
        from .nodes_lora import _load_adapter_checkpoint
        ckpt = _load_adapter_checkpoint(lora_path)
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            meta = ckpt.get("meta", {})
            if "ema_state" in ckpt:
                for key, ema_val in ckpt["ema_state"].items():
                    if key in state_dict:
                        state_dict[key] = ema_val
                logger.info("Using EMA weights from checkpoint for inference")
        else:
            state_dict = ckpt
            meta = {}

        # Infer rank from lora_A tensor shapes when meta omits it
        inferred_rank = None
        for k, v in state_dict.items():
            if "lora_A" in k and v.ndim >= 2:
                inferred_rank = v.shape[0]
                break
        rank = meta.get("rank", inferred_rank or 16)
        alpha = meta.get("alpha", float(rank))
        init_mode = meta.get("init_mode", "standard")
        use_rslora = meta.get("use_rslora", False)

        meta_target = meta.get("target", "all_attn_mlp")
        if isinstance(meta_target, str) and meta_target in FOLEY_TARGET_PRESETS:
            target_suffixes = FOLEY_TARGET_PRESETS[meta_target]
        elif isinstance(meta_target, (list, tuple)):
            target_suffixes = tuple(meta_target)
        else:
            target_suffixes = FOLEY_TARGET_PRESETS["all_attn_mlp"]

        event_enabled = bool(meta.get("event_conditioning", False))
        event_strength = float(meta.get("event_strength", 1.0))

        ckpt_cache[lora_path] = (state_dict, rank, alpha, target_suffixes, init_mode, use_rslora,
                                 event_enabled, event_strength)

    strength = float(target.get("strength", 1.0))
    apply_lora(model, rank=rank, alpha=alpha, target_suffixes=target_suffixes,
               init_mode="standard", use_rslora=use_rslora)
    model._event_conditioning_enabled = event_enabled
    # The timeline strength slider should scale the whole adapter effect. The
    # event adapter is not a LoRA module, so scale its runtime gate explicitly.
    model._event_strength = event_strength * strength
    if init_mode == "pissa":
        # pissa weights include modified base.weight — loading them dirties the
        # base, so the next swap must restore it from base_state.
        model.load_state_dict(state_dict, strict=False)
        ckpt_cache["__base_dirty__"] = True
    else:
        load_lora(model, state_dict)

    if strength != 1.0:
        # Scale the LoRA delta by `strength` once — multiply only lora_B
        # (scaling both lora_A and lora_B would apply strength quadratically).
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.data.mul_(strength)

    logger.debug(f"LoRA swap @ {time_sec:.1f}s: {os.path.basename(lora_path)} "
                 f"rank={rank} strength={strength}")


def _segment_at_time(lora_schedule, time_sec):
    """Return the schedule segment covering time_sec, or None."""
    if not lora_schedule:
        return None
    for seg in lora_schedule:
        if seg["start_sec"] <= time_sec < seg["end_sec"]:
            return seg
    return None


# DAC kwargs + explicit latent_dim (must be 128 or the decoder mismatches)
# extracted from original pth
_DAC_KWARGS = dict(
    encoder_dim=128,
    encoder_rates=[2, 3, 4, 5, 8],
    latent_dim=128,
    decoder_dim=2048,
    decoder_rates=[8, 5, 4, 3, 2],
    n_codebooks=9,
    codebook_size=1024,
    codebook_dim=8,
    quantizer_dropout=False,
    sample_rate=48000,
    continuous=True,
)

def _tdev(d):  # accept "cpu", "cuda:0", torch.device
    return d if isinstance(d, torch.device) else torch.device(str(d))

def _extract_state(obj):
    # Accept: nn.Module, {"state_dict":..., "metadata":...}, or a flat dict of tensors
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        # plain dict of tensors (e.g., safetensors via comfy)
        # keep only tensor entries
        return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    raise RuntimeError(f"Unsupported checkpoint payload: {type(obj)}")

def load_dac_any(path: str, device="cpu", strict: bool = True):
    """
    Single loader for .pth and .safetensors using the KNOWN, FIXED kwargs.
    No header reads, no inference. We set model.metadata ourselves.
    """
    if DAC is None:
        raise RuntimeError("DAC class import failed")

    dev = _tdev(device)

    # Load payload to CPU (Comfy expects a real torch.device here)
    obj = load_torch_file(path, device=torch.device("cpu"))
    sd = _extract_state(obj)

    # Build exactly the architecture you specified
    model = DAC(**_DAC_KWARGS)
    model.load_state_dict(sd, strict=strict)

    # Put the meta where it goes.
    model.metadata = {
        "kwargs": {**_DAC_KWARGS},
        "converted_from": "vae_128d_48k.pth",
        "format": "pth_or_safetensors",
        "source_path": os.path.basename(path),
    }

    return model.to(dev).eval()

def get_module_size_in_mb(module: nn.Module) -> float:
    """Calculates the total size of a module's parameters in megabytes."""
    total_bytes = 0
    for param in module.parameters():
        total_bytes += param.numel() * param.element_size()
    return total_bytes / (1024 * 1024)


def _caps(model_dict, cfg):
    tokmax = int(getattr(getattr(model_dict, "clap_tokenizer", None), "model_max_length", 10**9) or 10**9)
    posmax = int(getattr(getattr(getattr(model_dict, "clap_model", None), "config", None), "max_position_embeddings", 10**9) or 10**9)
    cfgmax = int(getattr(getattr(cfg, "model_config", None), "model_kwargs", {}).get("text_length", 10**9))
    return min(tokmax, posmax, cfgmax)


def _pad_or_trim_time(x, T_fixed: int):
    # x: [B, T_cur, D] -> [B, T_fixed, D]
    B, T_cur, D = x.shape
    if T_cur == T_fixed:
        return x
    if T_cur > T_fixed:
        return x[:, :T_fixed, :]
    return F.pad(x, (0, 0, 0, T_fixed - T_cur))


def prepare_latents_with_generator(scheduler, batch_size, num_channels_latents, length, dtype, device, generator=None):
    """Creates the initial random noise tensor using a specified torch.Generator for reproducibility."""
    shape = (batch_size, num_channels_latents, int(length))
    # Use the passed generator for reproducible random noise, compatible with 64-bit seeds.
    latents = randn_tensor(shape, device=device, dtype=dtype, generator=generator)
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma
    return latents


def _find_start_step(sigmas, strength):
    """Map a2a strength to the starting step via sigma lookup.

    Returns ``(start_step, sigma_used)`` where ``sigma_used`` is the scheduler's
    *grid* sigma at ``start_step`` — NOT the raw ``strength``. The initial latent
    must be noised to the exact sigma the scheduler assumes on its first step;
    using the raw strength (which lands between grid points) noises the latent
    slightly more than the scheduler expects and causes a sub-step discontinuity
    on the first a2a step. Snapping to the grid sigma removes it.

    Instead of the naive `steps - int(steps * strength)` which maps linearly to
    step indices, this finds the first grid step whose sigma is <= the target.
    With non-linear sigma schedules (sd3 shift, flux shift) that difference is
    significant — linear mapping over- or under-noises by up to ~15%.
    """
    # sigmas has num_steps+1 entries; timesteps has num_steps. Keep at least one
    # denoising step so the slice timesteps[start_step:] is never empty.
    n_steps = len(sigmas) - 1
    start = n_steps - 1
    # sigmas are in decreasing order (1.0 → 0.0): first index with sigma <= target.
    for i in range(len(sigmas)):
        if sigmas[i] <= strength:
            start = i
            break
    start = max(0, min(start, n_steps - 1))
    sigma_used = float(sigmas[start])
    return start, sigma_used


def _blend_reference_noise(gaussian_noise, init_latents, noise_blend):
    """Blend reference audio structure into the initial noise.

    Preserves temporal dynamics (rhythm, timing, envelope) from the reference
    while keeping the noise statistically valid for the diffusion process.
    """
    if noise_blend <= 0:
        return gaussian_noise
    # Normalize reference to unit Gaussian per channel
    ref_mean = init_latents.mean(dim=-1, keepdim=True)
    ref_residual = init_latents - ref_mean
    ref_std = ref_residual.std(dim=-1, keepdim=True).clamp(min=1e-6)
    ref_noise = ref_residual / ref_std
    # Blend and renormalize to preserve unit variance
    blended = (1 - noise_blend) * gaussian_noise + noise_blend * ref_noise
    blend_std = blended.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return blended / blend_std


def encode_audio_to_latents(audio_waveform, dac_model, device):
    """Encode raw audio waveform to DAC latent space.

    Args:
        audio_waveform: [B, 1, samples] tensor at 48kHz
        dac_model: DAC model in continuous mode
        device: target device

    Returns:
        Latent tensor [B, 128, T] (deterministic via distribution mode)
    """
    with torch.no_grad():
        dac_weight = next(dac_model.parameters())
        waveform = audio_waveform.to(device=dac_weight.device, dtype=torch.float32)
        z_dist, _, _, _, _ = dac_model.encode(waveform)
        return z_dist.mode()  # deterministic: returns mean of distribution


# Denoise keeps fast CFG path; we optimize memory elsewhere (ping-pong + precision + no extra repeats)
def denoise_process_with_generator(
    visual_feats,
    text_feats,
    audio_len_in_s,
    model_dict,
    cfg,
    guidance_scale,
    num_inference_steps,
    batch_size,
    sampler,
    generator=None,
    init_latents=None,
    strength=1.0,
    noise_blend=0.0,
    init_noise=None,
    inpaint_mask=None,
    inpaint_original=None,
    inpaint_noise=None,
):
    """
    An adaptation of the original denoise_process that accepts a torch.Generator for seeding,
    a sampler/solver name, and uses a ComfyUI progress bar.
    """
    target_dtype = model_dict.foley_model.dtype
    device = model_dict.device

    shift = getattr(model_dict.foley_model, '_flow_shift_override', cfg.diffusion_config.sample_flow_shift)
    if (hasattr(model_dict.foley_model, '_flow_shift_override')
            and shift != cfg.diffusion_config.sample_flow_shift):
        # The override is a model attribute, so it survives removal of the
        # ModelSampling node — surface it so a stale value can't act silently.
        logger.warning(f"Flow shift override {shift} active (ModelSampling node; sticky on the loaded "
                       f"model — set shift back to {cfg.diffusion_config.sample_flow_shift} or reload "
                       f"the model to clear it)")
    scheduler = FlowMatchDiscreteScheduler(
        shift=shift,
        solver=sampler
    )
    scheduler.set_timesteps(num_inference_steps, device=device)

    if init_latents is not None and strength < 1.0:
        # Audio2Audio: sigma-based strength mapping
        start_step, sigma_target = _find_start_step(scheduler.sigmas, strength)
        timesteps = scheduler.timesteps[start_step:]

        # Build noise — optionally blend reference temporal structure
        if init_noise is not None:
            noise = init_noise.to(device=device, dtype=target_dtype)
        else:
            noise = randn_tensor(
                init_latents.shape, device=device, dtype=target_dtype, generator=generator
            )
            noise = _blend_reference_noise(noise, init_latents.to(device=device, dtype=target_dtype), noise_blend)

        # Flow matching: x_t = sigma * noise + (1 - sigma) * data
        latents = sigma_target * noise + (1 - sigma_target) * init_latents.to(device=device, dtype=target_dtype)
        latents = latents.repeat(batch_size, 1, 1) if latents.shape[0] == 1 else latents
    else:
        timesteps = scheduler.timesteps
        latents = prepare_latents_with_generator(
            scheduler, batch_size=batch_size,
            num_channels_latents=cfg.model_config.model_kwargs.audio_vae_latent_dim,
            length=audio_len_in_s * cfg.model_config.model_kwargs.audio_frame_rate,
            dtype=target_dtype, device=device, generator=generator
        )

    # Precompute CFG-invariant feature tensors once outside the loop to reduce allocator churn
    siglip2_feat_rep = visual_feats['siglip2_feat'].repeat(batch_size, 1, 1)
    syncformer_feat_rep = visual_feats['syncformer_feat'].repeat(batch_size, 1, 1)
    text_feat_rep = text_feats['text_feat'].repeat(batch_size, 1, 1)
    uncond_text_rep = text_feats['uncond_text_feat'].repeat(batch_size, 1, 1)

    # --- TEXT SEQUENCE LENGTH ---

    T_cond = int(text_feat_rep.shape[1])
    T_uncond = int(uncond_text_rep.shape[1])
    cap = _caps(model_dict, cfg)

    if getattr(model_dict.foley_model, "_blocks_are_compiled", False):
        # torch.compile path: pad to a fixed bucket so the text shape stays
        # stable across prompts and compiled graphs are reused.
        # Two-bucket policy: 77 normally, 128 if prompt exceeds 77 (respect hard caps)
        T_fixed = min(77, cap) if T_cond <= 77 else min(128, cap)
        # Cache once per session to avoid flapping if prompts bounce around;
        # stick to the bigger bucket once it's triggered.
        if not hasattr(model_dict.foley_model, "_text_len_fixed"):
            model_dict.foley_model._text_len_fixed = T_fixed
        else:
            model_dict.foley_model._text_len_fixed = max(model_dict.foley_model._text_len_fixed, T_fixed)
        T_use = model_dict.foley_model._text_len_fixed
        logger.info(f"Using T_FIXED bucket: {T_use} (prompt had {T_cond} tokens; cap {cap})")
    else:
        # Eager path: natural token length, capped like the reference pipeline.
        # Training, training-eval and the original pipeline all run text at its
        # natural length; bucketing to 77 fed the attention dozens of zero-pad
        # tokens the model never saw in (LoRA) training (use_attention_mask is
        # False, so padding IS attended).
        T_use = max(T_cond, T_uncond) if guidance_scale > 1.0 else T_cond
        T_use = min(T_use, cap)

    text_feat_rep   = _pad_or_trim_time(text_feat_rep,   T_use)
    uncond_text_rep = _pad_or_trim_time(uncond_text_rep, T_use)

    uncond_siglip2_feat = model_dict.foley_model.get_empty_clip_sequence(bs=batch_size, len=siglip2_feat_rep.shape[1]).to(device)
    uncond_syncformer_feat = model_dict.foley_model.get_empty_sync_sequence(bs=batch_size, len=syncformer_feat_rep.shape[1]).to(device)
    if guidance_scale > 1.0:
        pre_siglip2_input = torch.cat([uncond_siglip2_feat, siglip2_feat_rep])
        pre_sync_input = torch.cat([uncond_syncformer_feat, syncformer_feat_rep])
        pre_text_input = torch.cat([uncond_text_rep, text_feat_rep])
    else:
        pre_siglip2_input = siglip2_feat_rep
        pre_sync_input = syncformer_feat_rep
        pre_text_input = text_feat_rep

    # Cast the CFG-invariant conditioning to the model's compute dtype ONCE.
    # These tensors are constant across denoising steps (only the latent changes),
    # so re-casting them every step — as the old loop did — was pure allocator churn.
    compute_dtype = next(model_dict.foley_model.parameters()).dtype
    pre_siglip2_input = pre_siglip2_input.to(dtype=compute_dtype)
    pre_sync_input = pre_sync_input.to(dtype=compute_dtype)
    pre_text_input = pre_text_input.to(dtype=compute_dtype)
    pre_event_input = None
    if getattr(model_dict.foley_model, "_event_conditioning_enabled", False):
        from .lora.event import event_envelope_from_sync, zero_event_envelope
        cond_event = event_envelope_from_sync(syncformer_feat_rep, target_len=latents.shape[-1]).to(device=device, dtype=compute_dtype)
        if guidance_scale > 1.0:
            uncond_event = zero_event_envelope(batch_size, latents.shape[-1], device=device, dtype=compute_dtype)
            pre_event_input = torch.cat([uncond_event, cond_event])
        else:
            pre_event_input = cond_event
        event_strength = float(getattr(model_dict.foley_model, "_event_strength", 1.0))
    else:
        event_strength = 1.0

    pbar = ProgressBar(len(timesteps))
    with torch.inference_mode():
        for i, t in enumerate(timesteps):
            throw_exception_if_processing_interrupted()
            # Prepare inputs for classifier-free guidance
            latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents

            # ---- ensure timestep lives on the SAME device as latents (avoid CPU in graph) ----
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.long, device=latent_input.device)
            else:
                t = t.to(device=latent_input.device)
            # expand to batch without materializing CPU intermediates
            t_expand = t.expand(latent_input.shape[0]).contiguous()
            # -----------------------------------------------------------------------------

            # CFG-invariant features were cast once above; only the latent changes.
            latent_input = latent_input.to(dtype=compute_dtype)

            # Predict the noise residual
            if compute_dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type=latent_input.device.type, dtype=compute_dtype):
                    noise_pred = model_dict.foley_model(
                        x=latent_input, t=t_expand, cond=pre_text_input,
                        clip_feat=pre_siglip2_input, sync_feat=pre_sync_input,
                        event_envelope=pre_event_input, event_strength=event_strength,
                    )["x"]
            else:
                noise_pred = model_dict.foley_model(
                    x=latent_input, t=t_expand, cond=pre_text_input,
                    clip_feat=pre_siglip2_input, sync_feat=pre_sync_input,
                    event_envelope=pre_event_input, event_strength=event_strength,
                )["x"]

            if guidance_scale > 1.0:
                # CFG combine in fp32 (the reference pipeline upcasts before the
                # combine; the scheduler integrates in fp32 anyway)
                noise_pred_uncond, noise_pred_text = noise_pred.float().chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents)[0]

            # Inpainting: replace known regions with properly noised original.
            # Only apply after complete solver steps — for multi-order solvers
            # (heun-2, kutta-4), skip inner/predictor sub-steps where the scheduler
            # hasn't finished its full update cycle.
            if inpaint_mask is not None and inpaint_original is not None and scheduler.state_in_first_order:
                # After a complete step, _step_index was incremented.
                # sigmas[_step_index] is the sigma the sample has been denoised to.
                if scheduler._step_index < len(scheduler.sigmas):
                    sigma_current = scheduler.sigmas[scheduler._step_index]
                else:
                    sigma_current = 0.0
                original_noised = sigma_current * inpaint_noise + (1 - sigma_current) * inpaint_original
                # mask: 1.0 = regenerate (model output), 0.0 = keep original
                mask_expanded = inpaint_mask.to(device=latents.device, dtype=latents.dtype)
                latents = latents * mask_expanded + original_noised * (1 - mask_expanded)

            pbar.update(1)

    # Decode latents to audio waveform
    # Ensure dtype/device match DAC weights to avoid mismatches
    with torch.inference_mode():
        dac_weight = next(model_dict.dac_model.parameters())
        latents_dec = latents.to(device=dac_weight.device, dtype=dac_weight.dtype)
        audio = model_dict.dac_model.decode(latents_dec)

    # Trim to exact length (DAC output is [B, 1, T])
    audio = audio[:, :, :int(audio_len_in_s * model_dict.dac_model.sample_rate)]
    return audio, model_dict.dac_model.sample_rate


# Keep preprocessing on CPU; move to device just-in-time inside encode functions
def feature_process_from_tensors(frames_8fps, frames_25fps, prompt, neg_prompt, deps, cfg):
    """
    Helper function takes pre-sampled frame tensors and extracts all necessary features.
    """
    visual_features = {}

    # Process SigLIP2 features (Content analysis) at 8 FPS
    processed_8fps = torch.stack([deps.siglip2_preprocess(frame) for frame in frames_8fps])  # CPU tensors
    # Process Synchformer features (Timing/Sync analysis) at 25 FPS
    processed_25fps = torch.stack([deps.syncformer_preprocess(frame) for frame in frames_25fps])  # CPU tensors

    # Move just-in-time to device for encoding to minimize residency
    processed_8fps_dev = processed_8fps.unsqueeze(0).to(deps.device, non_blocking=True)
    visual_features['siglip2_feat'] = encode_video_with_siglip2(processed_8fps_dev, deps)

    processed_25fps_dev = processed_25fps.unsqueeze(0).to(deps.device, non_blocking=True)
    visual_features['syncformer_feat'] = encode_video_with_sync(processed_25fps_dev, deps)

    # Audio length is determined by the duration of the sync stream (25 FPS)
    audio_len_in_s = frames_25fps.shape[0] / 25.0

    # Process Text features for both positive and negative prompts
    prompts = [neg_prompt, prompt]
    text_feat_res, _ = encode_text_feat(prompts, deps)

    text_feats = {'text_feat': text_feat_res[1:], 'uncond_text_feat': text_feat_res[:1]}

    # Free CPU preprocessing tensors proactively (they can be large)
    del processed_8fps, processed_25fps, processed_8fps_dev, processed_25fps_dev

    return visual_features, text_feats, audio_len_in_s


# -----------------------------------------------------------------------------------
# CHUNKED LONG-FORM GENERATION UTILITIES
# -----------------------------------------------------------------------------------

def compute_chunk_boundaries(duration: float, chunk_duration: float, overlap_seconds: float):
    """Compute chunk time boundaries with overlap for long-form generation.

    All chunks are exactly chunk_duration long (except single-chunk clips shorter
    than chunk_duration). This prevents the model from operating on short last
    chunks that are far from its training distribution.

    Returns list of (t_start, t_end) tuples in seconds.
    """
    if overlap_seconds >= chunk_duration:
        logger.warning(f"overlap_seconds ({overlap_seconds}) >= chunk_duration ({chunk_duration}), "
                       f"clamping to {chunk_duration * 0.5}")
        overlap_seconds = chunk_duration * 0.5

    if duration <= chunk_duration:
        return [(0.0, duration)]

    max_stride = chunk_duration - overlap_seconds
    n_chunks = math.ceil((duration - chunk_duration) / max_stride) + 1
    stride = (duration - chunk_duration) / (n_chunks - 1)

    chunks = []
    for i in range(n_chunks):
        t_start = i * stride
        t_end = t_start + chunk_duration
        chunks.append((round(t_start, 6), round(min(t_end, duration), 6)))

    return chunks


_MIN_GEN_SEC = 5.0  # pad short segment chunks to >= this many seconds of context
SAFA_OVERLAP_SEC = 1.5  # overlap pulled in at a SaFa seam (must match nodes_lora packing)


def _schedule_to_chunks(schedule, window, overlap=SAFA_OVERLAP_SEC):
    """Timeline chunks = the zones themselves (positions already set by the UI:
    touching = hard cut, overlapping = SaFa seam). Each zone is one chunk; a zone
    longer than the model window is sub-split into sub-chunks overlapping by
    `overlap` (SaFa-blended within). Returns [(start, end), ...] in order.
    """
    chunks = []
    for s in sorted(schedule, key=lambda z: z["start_sec"]):
        a, b = float(s["start_sec"]), float(s["end_sec"])
        if (b - a) <= window + 1e-6:
            chunks.append((round(a, 6), round(b, 6)))
        else:
            logger.warning(f"LoRA timeline: zone {a:.1f}-{b:.1f}s exceeds the {window:.0f}s "
                           f"window — sub-splitting it into multiple chunks.")
            for cs, ce in compute_chunk_boundaries(b - a, window, min(overlap, window * 0.5)):
                chunks.append((round(a + cs, 6), round(a + ce, 6)))
    return chunks


def align_chunks_to_schedule(chunks, lora_schedule):
    """Build per-segment chunks with pad-and-trim, returning (gen, keep).

    Per-chunk LoRA/text only works if each chunk lies within one segment. The
    default fixed ~chunk_duration windows can all centre on the same long
    segment (only that LoRA applies), so we partition [start,end] at every
    segment edge: each segment AND each no-LoRA gap is its own region (long
    regions sub-split to <= window).

    But the model is trained on ~chunk_duration windows and produces NOISE on
    very short clips (a 1s region degenerates). So we don't generate the bare
    region: each region is the `keep` window (the exact frames we output), and
    we generate a padded `gen` window (>= _MIN_GEN_SEC, borrowing neighbouring
    video as context) — then extract only the keep slice. Keep windows tile
    [start,end] exactly (adjacent, hard cuts on the chosen frames); gen windows
    may overlap (context only — not crossfaded).

    Returns (gen_chunks, keep_windows). keep_windows is None when no schedule.
    """
    if not lora_schedule or not chunks:
        return chunks, None

    total_start = chunks[0][0]
    total_end = chunks[-1][1]
    window = max((ce - cs for cs, ce in chunks), default=total_end - total_start)
    if window <= 0:
        return chunks, None

    # 1. exact keep windows from segment/gap edges (sub-split long regions)
    bset = {round(total_start, 6), round(total_end, 6)}
    for seg in lora_schedule:
        for key in ("start_sec", "end_sec"):
            bset.add(round(max(total_start, min(total_end, float(seg[key]))), 6))
    bounds = sorted(bset)

    keep = []
    for cs, ce in zip(bounds[:-1], bounds[1:]):
        span = ce - cs
        if span < 1e-3:
            continue
        if span <= window + 1e-6:
            keep.append((round(cs, 6), round(ce, 6)))
        else:
            n = math.ceil(span / window)
            step = span / n
            for k in range(n):
                a = cs + k * step
                b = ce if k == n - 1 else cs + (k + 1) * step
                keep.append((round(a, 6), round(b, 6)))

    if not keep:
        return chunks, None

    # 2. gen windows: pad keeps shorter than min_gen with neighbouring context
    span_total = total_end - total_start
    min_gen = min(_MIN_GEN_SEC, window, span_total)
    gen = []
    for ks, ke in keep:
        if (ke - ks) >= min_gen - 1e-6:
            gen.append((ks, ke))
            continue
        pad = min_gen - (ke - ks)
        gs, ge = ks - pad / 2.0, ke + pad / 2.0
        if gs < total_start:
            ge += (total_start - gs)
            gs = total_start
        if ge > total_end:
            gs -= (ge - total_end)
            ge = total_end
        gen.append((round(max(total_start, gs), 6), round(min(total_end, ge), 6)))

    return gen, keep


def slice_features_for_chunk(features: dict, t_start: float, t_end: float):
    """Slice pre-computed features to a specific time window.

    Args:
        features: FOLEYTUNE_FEATURES dict with clip_feat, sync_feat, text_feat, etc.
        t_start: chunk start time in seconds
        t_end: chunk end time in seconds

    Returns:
        Dict with sliced clip_feat and sync_feat, shared text features.
    """
    # SigLIP2: 8fps, direct time slice
    clip_start = int(t_start * 8)
    clip_end = int(t_end * 8)
    clip_feat = features["clip_feat"][:, clip_start:clip_end, :]
    # Ensure at least 1 frame
    if clip_feat.shape[1] == 0:
        clip_feat = features["clip_feat"][:, -1:, :]

    # Synchformer: segment_size=16, step_size=8, at 25fps
    # Segment i covers frames [i*8, i*8+16], i.e. time [i*0.32, i*0.32+0.64]s
    # Each segment produces 8 output tokens
    # For an 8s chunk (200 frames): (200-16)//8+1 = 24 segments = 192 tokens
    # The naive formula seg_end = int(t_end/stride) overcounts because it ignores
    # the segment_size=16 window — use the same formula as encode_video_with_sync.
    sync_fps = 25.0
    seg_size = 16
    step_size = 8
    total_sync_tokens = features["sync_feat"].shape[1]
    frame_start = int(t_start * sync_fps)
    frame_end = int(t_end * sync_fps)
    chunk_frames = frame_end - frame_start
    n_segs = max(1, (chunk_frames - seg_size) // step_size + 1)
    seg_start = max(0, frame_start // step_size)
    seg_end = min(seg_start + n_segs, total_sync_tokens // 8)
    tok_start = seg_start * 8
    tok_end = seg_end * 8
    sync_feat = features["sync_feat"][:, tok_start:tok_end, :]
    if sync_feat.shape[1] == 0:
        sync_feat = features["sync_feat"][:, -8:, :]

    return {
        "clip_feat": clip_feat,
        "sync_feat": sync_feat,
        "text_feat": features["text_feat"],
        "uncond_text_feat": features["uncond_text_feat"],
    }


def safa_binary_swap(left_latents, right_latents, overlap_len, step_idx):
    """SaFa-style binary swap in overlap region during denoising.

    Alternating frames taken from left/right chunk with per-step shift.
    No averaging — preserves high-frequency spectral content.

    Args:
        left_latents: [B, 128, T_left] — left chunk latents (modified in-place)
        right_latents: [B, 128, T_right] — right chunk latents (modified in-place)
        overlap_len: number of overlapping latent frames
        step_idx: current denoising step index (for shift pattern)
    """
    shift = (step_idx * 5) % overlap_len
    mask = ((torch.arange(overlap_len, device=left_latents.device) + shift) % 2).bool()
    # mask shape: [overlap_len] -> broadcast to [1, 1, overlap_len]
    mask = mask.unsqueeze(0).unsqueeze(0)

    left_overlap = left_latents[:, :, -overlap_len:]
    right_overlap = right_latents[:, :, :overlap_len]

    merged = torch.where(mask, right_overlap, left_overlap)
    left_latents[:, :, -overlap_len:] = merged
    right_latents[:, :, :overlap_len] = merged


def equal_power_crossfade(left, right, overlap_len, dim=-1):
    """Equal-power crossfade in overlap region.

    Works on both latents [B, 128, T] and audio [B, 1, T].

    Args:
        left: tensor — left chunk
        right: tensor — right chunk
        overlap_len: number of overlapping frames/samples
        dim: temporal dimension (default -1)

    Returns:
        Stitched tensor with crossfaded overlap.
    """
    # No overlap — pure concatenation
    if overlap_len <= 0:
        return torch.cat([left, right], dim=dim)

    t = torch.linspace(0, 1, overlap_len, device=left.device, dtype=left.dtype)
    # Reshape for broadcasting: add dims for batch and channel
    shape = [1] * left.ndim
    shape[dim] = overlap_len
    t = t.reshape(shape)

    w_right = torch.sqrt(t)
    w_left = torch.sqrt(1.0 - t)

    left_body = left.narrow(dim, 0, left.shape[dim] - overlap_len)
    left_tail = left.narrow(dim, left.shape[dim] - overlap_len, overlap_len)
    right_head = right.narrow(dim, 0, overlap_len)
    right_body = right.narrow(dim, overlap_len, right.shape[dim] - overlap_len)

    blended = w_left * left_tail + w_right * right_head
    return torch.cat([left_body, blended, right_body], dim=dim)


def chunked_denoise_process(
    features,
    chunks,
    crossfade_mode,
    model_dict,
    cfg,
    guidance_scale,
    num_inference_steps,
    batch_size,
    sampler,
    generator=None,
    init_latents=None,
    strength=1.0,
    noise_blend=0.0,
    lora_schedule=None,
):
    """Chunked denoising with overlap stitching for long-form generation.

    Overlap between adjacent chunks is computed from chunk positions, so all
    chunks can be exactly chunk_duration long (the model's training length).

    Args:
        features: FOLEYTUNE_FEATURES dict (full video features)
        chunks: list of (t_start, t_end) tuples from compute_chunk_boundaries
        crossfade_mode: "safa", "latent", or "waveform"
        model_dict: dict with foley_model, dac_model, device
        cfg: model config (loaded YAML)
        guidance_scale: CFG scale
        num_inference_steps: denoising steps
        batch_size: number of variations
        sampler: solver name string
        generator: torch.Generator for reproducibility

    Returns:
        (audio_waveform, sample_rate) tuple
    """
    # Timeline = "zones are chunks". build_schedule has already PACKED the zone
    # positions: adjacent zones touch (hard cut) or overlap by SAFA_OVERLAP_SEC
    # (a per-seam SaFa seam). Each zone is a chunk (a zone longer than the window
    # is sub-split). Per-chunk LoRA/seed/variance/prompt is resolved by chunk
    # centre. SaFa blends only the seams whose overlap > 0; the rest hard-concat.
    # keep_windows stays None so _resolve_center uses chunk centres.
    keep_windows = None
    if lora_schedule:
        _window = max((ce - cs for cs, ce in chunks), default=features["duration"])
        _overlap = float(lora_schedule[0].get("safa_overlap", SAFA_OVERLAP_SEC))
        chunks = _schedule_to_chunks(lora_schedule, _window, _overlap)
        crossfade_mode = "safa"
        _sorted = sorted(lora_schedule, key=lambda z: z["start_sec"])
        _seams = sum(1 for i in range(len(_sorted) - 1)
                     if float(_sorted[i + 1]["start_sec"]) < float(_sorted[i]["end_sec"]) - 1e-6)
        logger.info(f"LoRA timeline: {len(chunks)} chunks from {len(lora_schedule)} zone(s), "
                    f"{_seams} SaFa seam(s). chunks={[(round(a, 2), round(b, 2)) for a, b in chunks]}")

    # LoRA/prompt for a chunk is resolved by its KEEP-window centre (the segment
    # it represents) — the padded gen-window centre can fall in a neighbour.
    def _resolve_center(c_idx, t_start, t_end):
        if keep_windows is not None:
            ks, ke = keep_windows[c_idx]
            return (ks + ke) / 2.0
        return (t_start + t_end) / 2.0

    target_dtype = model_dict.foley_model.dtype
    device = model_dict.device
    audio_frame_rate = cfg.model_config.model_kwargs.audio_frame_rate
    latent_dim = cfg.model_config.model_kwargs.audio_vae_latent_dim
    pair_overlap_frames = [
        int((chunks[i][1] - chunks[i + 1][0]) * audio_frame_rate)
        for i in range(len(chunks) - 1)
    ]
    sample_rate = model_dict.dac_model.sample_rate

    # --- LoRA schedule: save base state for hot-swapping ---
    # Only needed if at least one segment actually carries a LoRA. A prompt-only
    # timeline (all "(none)" entries) just swaps text_feat per chunk — no base
    # save/restore, no LoRA wrap/unwrap.
    _lora_base_state = None
    _current_lora_path = None
    _lora_ckpt_cache = {}
    _base_event_enabled = bool(getattr(model_dict.foley_model, "_event_conditioning_enabled", False))
    _base_event_strength = float(getattr(model_dict.foley_model, "_event_strength", 1.0))
    if lora_schedule and any(seg.get("lora_path") for seg in lora_schedule):
        # Store base state on CPU to avoid doubling GPU memory
        _lora_base_state = {k: v.cpu().clone() for k, v in model_dict.foley_model.state_dict().items()}
        logger.info(f"LoRA timeline: {len(lora_schedule)} segments, base state saved (CPU)")

    def _restore_lora_base():
        nonlocal _lora_base_state
        if _lora_base_state is None:
            return
        import gc
        from .lora.lora import remove_lora
        remove_lora(model_dict.foley_model)
        model_dict.foley_model.load_state_dict(_lora_base_state, strict=False)
        model_dict.foley_model._event_conditioning_enabled = _base_event_enabled
        model_dict.foley_model._event_strength = _base_event_strength
        _lora_base_state = None
        _lora_ckpt_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("LoRA timeline: base model restored, caches cleared")

    # Single chunk — delegate to standard denoise (gen == keep here, no trim).
    if len(chunks) == 1:
        t_start, t_end = chunks[0]
        _center = _resolve_center(0, t_start, t_end)
        # Apply LoRA for single-chunk case
        if lora_schedule and _lora_base_state is not None:
            _apply_lora_for_time(model_dict.foley_model, lora_schedule,
                                 _center, _lora_base_state, _lora_ckpt_cache)
        chunk_feats = slice_features_for_chunk(features, t_start, t_end)
        chunk_dur = t_end - t_start
        visual = {
            "siglip2_feat": chunk_feats["clip_feat"].to(device),
            "syncformer_feat": chunk_feats["sync_feat"].to(device),
        }
        # Per-segment prompt: swap text_feat if the active segment carries one.
        _seg = _segment_at_time(lora_schedule, _center)
        _seg_text = _seg.get("text_feat") if _seg else None
        if _seg_text is not None:
            logger.info(f"Per-segment prompt @ single chunk: {_seg.get('prompt', '')[:60]!r}")
        text = {
            "text_feat": (_seg_text if _seg_text is not None
                          else chunk_feats["text_feat"]).to(device),
            "uncond_text_feat": chunk_feats["uncond_text_feat"].to(device),
        }
        # Slice init_latents for this chunk if provided
        chunk_init = None
        if init_latents is not None:
            frame_start = int(t_start * audio_frame_rate)
            frame_end = int(t_end * audio_frame_rate)
            chunk_init = init_latents[:, :, frame_start:frame_end]
        try:
            return denoise_process_with_generator(
                visual, text, chunk_dur, model_dict, cfg,
                guidance_scale, num_inference_steps, batch_size, sampler, generator,
                init_latents=chunk_init, strength=strength, noise_blend=noise_blend,
            )
        finally:
            _restore_lora_base()

    # --- Multi-chunk: set up per-chunk schedulers and latents ---
    # CRITICAL: each chunk needs its own scheduler instance because
    # FlowMatchDiscreteScheduler.step() increments an internal _step_index.
    chunk_schedulers = []
    shift = getattr(model_dict.foley_model, '_flow_shift_override', cfg.diffusion_config.sample_flow_shift)
    if (hasattr(model_dict.foley_model, '_flow_shift_override')
            and shift != cfg.diffusion_config.sample_flow_shift):
        logger.warning(f"Flow shift override {shift} active (ModelSampling node; sticky on the loaded "
                       f"model — set shift back to {cfg.diffusion_config.sample_flow_shift} or reload "
                       f"the model to clear it)")
    for _ in chunks:
        sched = FlowMatchDiscreteScheduler(
            shift=shift,
            solver=sampler
        )
        sched.set_timesteps(num_inference_steps, device=device)
        chunk_schedulers.append(sched)
    timesteps = chunk_schedulers[0].timesteps  # all identical

    # Prepare per-chunk latents and features
    chunk_latents = []
    chunk_visual_feats = []
    chunk_text_feats = []

    # For a2a: generate one continuous noise tensor so overlap regions share
    # the same noise across adjacent chunks. Without this, each chunk gets
    # independent noise in the overlap, breaking crossfade/SaFa coherence.
    shared_noise = None
    a2a_start_step = None
    a2a_sigma = None
    if init_latents is not None and strength < 1.0:
        full_latent_len = init_latents.shape[-1]
        shared_noise = randn_tensor(
            (1, latent_dim, full_latent_len), device=device, dtype=target_dtype, generator=generator
        )
        shared_noise = _blend_reference_noise(
            shared_noise, init_latents.to(device=device, dtype=target_dtype), noise_blend
        )
        a2a_start_step, a2a_sigma = _find_start_step(chunk_schedulers[0].sigmas, strength)

    for c_idx, (t_start, t_end) in enumerate(chunks):
        chunk_dur = t_end - t_start
        latent_len = int(chunk_dur * audio_frame_rate)

        # Resolve this chunk's timeline segment up front — drives the per-section
        # seed (own noise) and variance (CLAP-embedding perturbation). Resolve by
        # keep-window centre (the gen window may straddle a neighbour).
        _seg = _segment_at_time(lora_schedule, _resolve_center(c_idx, t_start, t_end))
        _seg_seed = int(_seg.get("seed", -1)) if _seg else -1
        # Per-section seed: a fresh generator seeded with the section's seed (+chunk
        # offset so a multi-chunk section doesn't repeat its noise). -1 = inherit the
        # global generator (sequential draws, original behaviour).
        _chunk_gen = generator
        if _seg_seed >= 0:
            _chunk_gen = torch.Generator(device=device).manual_seed(
                (int(_seg_seed) + c_idx) & 0xffffffffffffffff)

        # Per-section variance: a soft/partial re-roll of THIS chunk's initial
        # noise toward a fresh take. Text/CLAP is too weak to move foley
        # (video+sync dominate), so variance acts on the noise — the strong
        # lever — like a fractional seed change. 0 = the seed's take, 1 = fully
        # fresh. Offset the variance seed so the fresh draw is decorrelated from
        # this chunk's base noise (else the blend would be a near no-op).
        _var = min(max(float(_seg.get("variance_strength", 0.0)) if _seg else 0.0, 0.0), 1.0)
        _vseed = ((int(_seg_seed) if _seg_seed >= 0 else generator.initial_seed())
                  + 0x5237 + c_idx) & 0xffffffffffffffff

        if shared_noise is not None:
            frame_start = int(t_start * audio_frame_rate)
            frame_end = int(t_end * audio_frame_rate)
            chunk_init = init_latents[:, :, frame_start:frame_end].to(device=device, dtype=target_dtype)
            chunk_noise = shared_noise[:, :, frame_start:frame_end]

            latent = a2a_sigma * chunk_noise + (1 - a2a_sigma) * chunk_init
            latent = latent.repeat(batch_size, 1, 1) if latent.shape[0] == 1 else latent
        else:
            latent = prepare_latents_with_generator(
                chunk_schedulers[c_idx], batch_size, latent_dim, latent_len,
                target_dtype, device, _chunk_gen
            )
        # Variance = partial re-roll of the initial noise toward a fresh take.
        # Variance-preserving blend keeps the latent's own scale, so loudness
        # and tone are untouched — only the take changes.
        if _var > 0:
            _vg = torch.Generator(device=device).manual_seed(_vseed)
            _fresh = randn_tensor(latent.shape, device=device, dtype=target_dtype, generator=_vg)
            latent = math.sqrt(1.0 - _var * _var) * latent + _var * latent.std() * _fresh
        chunk_latents.append(latent)

        c_feats = slice_features_for_chunk(features, t_start, t_end)
        chunk_visual_feats.append({k: c_feats[k].to(device) for k in ["clip_feat", "sync_feat"]})
        # Per-segment prompt: swap the positive text_feat for the segment's own
        # (encoded by the Timeline node); uncond stays global. _pad_or_trim_time
        # below reconciles any sequence-length difference before the CFG concat.
        # Per-segment prompt only — variance now acts on the noise above (the
        # strong lever), not this CLAP embedding (which barely moves foley).
        _seg_text = _seg.get("text_feat") if _seg else None
        _text = (_seg_text if _seg_text is not None else c_feats["text_feat"])
        chunk_text_feats.append({
            "text_feat": _text.to(device),
            "uncond_text_feat": c_feats["uncond_text_feat"].to(device),
        })

    # Truncate timesteps if using audio2audio
    if a2a_start_step is not None:
        timesteps = timesteps[a2a_start_step:]

    # --- Precompute per-chunk CFG features ---
    # Cast conditioning to the compute dtype once here; it's constant across steps.
    compute_dtype = next(model_dict.foley_model.parameters()).dtype
    # Build envelopes when the base model OR any scheduled LoRA *might* use event
    # conditioning (LoRA metas aren't loaded yet — this is a safe over-estimate).
    # Whether each chunk actually RECEIVES its envelope is gated in the loop by
    # that chunk's hot-swapped _event_conditioning_enabled flag, so a non-event
    # LoRA is never fed an event signal (matches regular sampling).
    _event_enabled = bool(getattr(model_dict.foley_model, "_event_conditioning_enabled", False)) or bool(
        lora_schedule and any(seg.get("lora_path") for seg in lora_schedule)
    )
    if _event_enabled:
        from .lora.event import event_envelope_from_sync, zero_event_envelope
    chunk_cfg_inputs = []
    for i in range(len(chunks)):
        vis = chunk_visual_feats[i]
        txt = chunk_text_feats[i]

        siglip2_rep = vis["clip_feat"].repeat(batch_size, 1, 1)
        sync_rep = vis["sync_feat"].repeat(batch_size, 1, 1)
        text_rep = txt["text_feat"].repeat(batch_size, 1, 1)
        uncond_rep = txt["uncond_text_feat"].repeat(batch_size, 1, 1)

        # Text length: fixed bucket under torch.compile (shape stability),
        # natural length otherwise (matches training — see denoise_process_with_generator)
        T_cond = text_rep.shape[1]
        T_uncond = uncond_rep.shape[1]
        cap = _caps(model_dict, cfg)
        if getattr(model_dict.foley_model, "_blocks_are_compiled", False):
            T_use = min(77, cap) if T_cond <= 77 else min(128, cap)
        else:
            T_use = max(T_cond, T_uncond) if guidance_scale > 1.0 else T_cond
            T_use = min(T_use, cap)
        text_rep = _pad_or_trim_time(text_rep, T_use)
        uncond_rep = _pad_or_trim_time(uncond_rep, T_use)

        uncond_clip = model_dict.foley_model.get_empty_clip_sequence(
            bs=batch_size, len=siglip2_rep.shape[1]
        ).to(device)
        uncond_sync = model_dict.foley_model.get_empty_sync_sequence(
            bs=batch_size, len=sync_rep.shape[1]
        ).to(device)

        if guidance_scale > 1.0:
            cfg_clip = torch.cat([uncond_clip, siglip2_rep])
            cfg_sync = torch.cat([uncond_sync, sync_rep])
            cfg_text = torch.cat([uncond_rep, text_rep])
        else:
            cfg_clip = siglip2_rep
            cfg_sync = sync_rep
            cfg_text = text_rep

        cfg_event = None
        if _event_enabled:
            cond_event = event_envelope_from_sync(
                sync_rep, target_len=chunk_latents[i].shape[-1]
            ).to(device=device, dtype=compute_dtype)
            if guidance_scale > 1.0:
                uncond_event = zero_event_envelope(
                    batch_size, chunk_latents[i].shape[-1], device=device, dtype=compute_dtype
                )
                cfg_event = torch.cat([uncond_event, cond_event])
            else:
                cfg_event = cond_event

        chunk_cfg_inputs.append({
            "clip": cfg_clip.to(dtype=compute_dtype),
            "sync": cfg_sync.to(dtype=compute_dtype),
            "text": cfg_text.to(dtype=compute_dtype),
            "event": cfg_event,
        })

    # --- Precompute per-chunk LoRA assignment (by keep-window centre) ---
    _chunk_lora_targets = [None] * len(chunks)
    if lora_schedule and _lora_base_state is not None:
        for c_idx, (cs, ce) in enumerate(chunks):
            _chunk_lora_targets[c_idx] = _segment_at_time(
                lora_schedule, _resolve_center(c_idx, cs, ce))
        lora_summary = [os.path.basename(s["lora_path"]) if (s and s.get("lora_path")) else "base"
                        for s in _chunk_lora_targets]
        logger.info(f"LoRA timeline per-chunk: {lora_summary}")

    # --- Denoising loop ---
    total_steps = len(timesteps) * len(chunks)
    pbar = ProgressBar(total_steps)

    try:
        with torch.inference_mode():
            for step_idx, t in enumerate(timesteps):
                throw_exception_if_processing_interrupted()
                if not torch.is_tensor(t):
                    t = torch.tensor(t, dtype=torch.long, device=device)
                else:
                    t = t.to(device=device)

                for c_idx in range(len(chunks)):
                    # LoRA hot-swap: apply correct LoRA before processing this chunk
                    if lora_schedule and _lora_base_state is not None:
                        target = _chunk_lora_targets[c_idx]
                        target_path = target.get("lora_path") if target else None
                        if target_path != _current_lora_path:
                            _apply_lora_for_time(model_dict.foley_model, lora_schedule,
                                                 _resolve_center(c_idx, *chunks[c_idx]),
                                                 _lora_base_state, _lora_ckpt_cache)
                            _current_lora_path = target_path

                    latents = chunk_latents[c_idx]
                    cfg_in = chunk_cfg_inputs[c_idx]

                    latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                    t_expand = t.expand(latent_input.shape[0]).contiguous()

                    # cfg_in tensors were cast to compute_dtype at build time; only the latent changes.
                    latent_input = latent_input.to(dtype=compute_dtype)
                    current_event_strength = float(getattr(model_dict.foley_model, "_event_strength", 1.0))
                    # Only feed the event envelope when THIS chunk's active LoRA was
                    # trained with event conditioning (the hot-swap set this flag from
                    # the LoRA's meta). Mirrors regular sampling, which passes None
                    # otherwise — so a non-event LoRA isn't fed an event signal it
                    # never learned, and a single-zone timeline matches plain sampling.
                    _chunk_event = (cfg_in["event"]
                                    if getattr(model_dict.foley_model, "_event_conditioning_enabled", False)
                                    else None)

                    if compute_dtype in (torch.float16, torch.bfloat16):
                        with torch.autocast(device_type=device.type, dtype=compute_dtype):
                            noise_pred = model_dict.foley_model(
                                x=latent_input, t=t_expand, cond=cfg_in["text"],
                                clip_feat=cfg_in["clip"], sync_feat=cfg_in["sync"],
                                event_envelope=_chunk_event,
                                event_strength=current_event_strength,
                            )["x"]
                    else:
                        noise_pred = model_dict.foley_model(
                            x=latent_input, t=t_expand, cond=cfg_in["text"],
                            clip_feat=cfg_in["clip"], sync_feat=cfg_in["sync"],
                            event_envelope=_chunk_event,
                            event_strength=current_event_strength,
                        )["x"]

                    if guidance_scale > 1.0:
                        # CFG combine in fp32 (matches reference; scheduler integrates fp32)
                        uncond_pred, text_pred = noise_pred.float().chunk(2)
                        noise_pred = uncond_pred + guidance_scale * (text_pred - uncond_pred)

                    chunk_latents[c_idx] = chunk_schedulers[c_idx].step(noise_pred, t, latents)[0]
                    pbar.update(1)

                # SaFa swap after all chunks are updated this step.
                # Skip in keep-window (timeline) mode: gen windows overlap with
                # DIFFERENT content (different segments/LoRAs), so swapping latents
                # between them would corrupt — each chunk is independent there.
                if crossfade_mode == "safa" and keep_windows is None:
                    for c_idx in range(len(chunks) - 1):
                        ovl = pair_overlap_frames[c_idx]
                        if ovl > 0:
                            safa_binary_swap(
                                chunk_latents[c_idx], chunk_latents[c_idx + 1],
                                ovl, step_idx
                            )
    finally:
        _restore_lora_base()

    # --- Stitch results ---
    # Keep-window (timeline pad-and-trim): extract each chunk's exact keep slice
    # (the padded context is discarded) and concatenate. Keep windows tile the
    # timeline, so this reconstructs the full duration with hard cuts on the
    # chosen frames — independent of crossfade_mode.
    if keep_windows is not None:
        # Hard-cut tiling (crossfade == 0): extract each chunk's exact keep slice
        # (padded context discarded) and concatenate. Keep windows tile the
        # timeline, so this rebuilds the full duration on the chosen frames.
        # (crossfade > 0 never reaches here — it uses the SaFa overlap path.)
        full_latent = None
        for c_idx in range(len(chunks)):
            gs, _ge = chunks[c_idx]
            ks, ke = keep_windows[c_idx]
            lat = chunk_latents[c_idx]
            sf = max(0, int(round((ks - gs) * audio_frame_rate)))
            ef = min(lat.shape[-1], int(round((ke - gs) * audio_frame_rate)))
            if ef <= sf:
                ef = min(lat.shape[-1], sf + 1)
            if full_latent is None:
                full_latent = lat[:, :, sf:ef]
                continue
            full_latent = torch.cat([full_latent, lat[:, :, sf:ef]], dim=-1)
        with torch.inference_mode():
            dac_weight = next(model_dict.dac_model.parameters())
            latents_dec = full_latent.to(device=dac_weight.device, dtype=dac_weight.dtype)
            audio = model_dict.dac_model.decode(latents_dec)
        audio = audio[:, :, :int(features["duration"] * sample_rate)]
        return audio, sample_rate

    if crossfade_mode == "safa":
        parts = []
        for c_idx in range(len(chunks)):
            lat = chunk_latents[c_idx]
            left_trim = pair_overlap_frames[c_idx - 1] // 2 if c_idx > 0 else 0
            right_ovl = pair_overlap_frames[c_idx] if c_idx < len(chunks) - 1 else 0
            right_trim = right_ovl - right_ovl // 2 if right_ovl > 0 else 0
            s = lat[:, :, left_trim:(lat.shape[-1] - right_trim) if right_trim else lat.shape[-1]]
            parts.append(s)
        full_latent = torch.cat(parts, dim=-1)

        with torch.inference_mode():
            dac_weight = next(model_dict.dac_model.parameters())
            latents_dec = full_latent.to(device=dac_weight.device, dtype=dac_weight.dtype)
            audio = model_dict.dac_model.decode(latents_dec)
        # Place at the first chunk's absolute start (timeline packing may begin
        # >0 and covers less than the full clip after seam overlaps) and pad the
        # remainder with silence to the full duration. Normal mode: start=0,
        # coverage≈duration, so this is just a trim.
        total_samples = int(features["duration"] * sample_rate)
        start_off = max(0, int(round(chunks[0][0] * sample_rate)))
        placed = audio.new_zeros(audio.shape[0], audio.shape[1], total_samples)
        n = min(audio.shape[-1], total_samples - start_off)
        if n > 0:
            placed[..., start_off:start_off + n] = audio[..., :n]
        return placed, sample_rate

    elif crossfade_mode == "latent":
        full_latent = chunk_latents[0]
        for c_idx in range(1, len(chunks)):
            full_latent = equal_power_crossfade(
                full_latent, chunk_latents[c_idx], pair_overlap_frames[c_idx - 1], dim=-1
            )

        with torch.inference_mode():
            dac_weight = next(model_dict.dac_model.parameters())
            latents_dec = full_latent.to(device=dac_weight.device, dtype=dac_weight.dtype)
            audio = model_dict.dac_model.decode(latents_dec)
        total_duration = features["duration"]
        audio = audio[:, :, :int(total_duration * sample_rate)]
        return audio, sample_rate

    else:  # waveform
        with torch.inference_mode():
            dac_weight = next(model_dict.dac_model.parameters())
            chunk_audios = []
            for c_idx in range(len(chunks)):
                lat = chunk_latents[c_idx].to(device=dac_weight.device, dtype=dac_weight.dtype)
                chunk_audios.append(model_dict.dac_model.decode(lat))

        full_audio = chunk_audios[0]
        for c_idx in range(1, len(chunk_audios)):
            ovl_samples = int((chunks[c_idx - 1][1] - chunks[c_idx][0]) * sample_rate)
            full_audio = equal_power_crossfade(
                full_audio, chunk_audios[c_idx], ovl_samples, dim=-1
            )
        total_duration = features["duration"]
        full_audio = full_audio[:, :, :int(total_duration * sample_rate)]
        return full_audio, sample_rate


# -----------------------------------------------------------------------------------
# FP8 WEIGHT-ONLY QUANTIZATION HELPERS (storage in fp8, compute in fp16/bf16)
# -----------------------------------------------------------------------------------
_DENY_SUBSTRINGS = (
    ".bias",            # never quantize biases; they’re tiny and can be precision-sensitive
    ".norm",            # covers LayerNorm/RMSNorm params (e.g., ".norm.weight")
    "q_norm.",          # explicit Q-norms
    "k_norm.",          # explicit K-norms
    "final_layer.",     # keep model output projection high precision
    "visual_proj.",     # keep early visual projection high precision
                        # exclude cross-attn query/proj (both audio & v_cond)
    "audio_cross_q.",
    "v_cond_cross_q.",
    "audio_cross_proj.",
    "v_cond_cross_proj.",
)

# FP8 storage dtypes we support (PyTorch exposes these two).
_FP8_DTYPES = (torch.float8_e5m2, torch.float8_e4m3fn)


class FP8WeightWrapper(nn.Module):
    """
    Minimal unified FP8 storage wrapper for Linear / Conv1d / Conv2d.

    - Stores weights in FP8 (qdtype) as buffers (so they serialize with state_dict).
    - On forward, upcasts weights (and bias if present) to the incoming tensor dtype
      (fp16/bf16/float32) before calling the functional op, so compute stays high precision.
    """
    def __init__(self, mod: nn.Module, qdtype: torch.dtype):
        super().__init__()
        # Identify which op we’re wrapping; needed to pick the correct functional call.
        self.kind = (
            "linear" if isinstance(mod, nn.Linear)
            else "conv1d" if isinstance(mod, nn.Conv1d)
            else "conv2d"
        )
        self.qdtype = qdtype  # target FP8 storage dtype (e5m2 or e4m3fn)

        # Convolution parameters are required to replay the exact conv op at inference.
        if self.kind != "linear":
            self.stride   = mod.stride
            self.padding  = mod.padding
            self.dilation = mod.dilation
            self.groups   = mod.groups

        # Allocate FP8 weight storage (on the same device), then copy from the original module.
        # Using a buffer (not a Parameter) avoids FP8 params flowing through optimizers.
        self.register_buffer(
            "weight",
            mod.weight.detach().to(device=mod.weight.device, dtype=qdtype),
            persistent=True,
        )

        # Keep bias in higher precision (float32) to avoid tiny-scale loss; store as buffer too.
        if mod.bias is None:
            self.bias = None
        else:
            self.register_buffer(
                "bias",
                mod.bias.detach().to(device=mod.bias.device, dtype=torch.float32),
                persistent=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast FP8 storage to the activation's compute dtype (fp16/bf16/fp32)
        w = self.weight.to(dtype=x.dtype)
        b = None if self.bias is None else self.bias.to(dtype=x.dtype)

        if self.kind == "linear":
            return F.linear(x, w, b)

        if self.kind == "conv1d":
            # weight shape: [Cout, Cin_per_group, K], so expected Cin = Cin_per_group * groups
            if x.ndim != 3:
                raise RuntimeError(f"conv1d expects 3D input, got {tuple(x.shape)}")
            expected_Cin = w.shape[1] * self.groups

            # channels-first (N, C, L)
            if x.shape[1] == expected_Cin:
                return F.conv1d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

            # channels-last (N, L, C) → transpose to (N, C, L), conv, then transpose back
            if x.shape[2] == expected_Cin:
                x_t = x.transpose(1, 2)
                y_t = F.conv1d(x_t, w, b, self.stride, self.padding, self.dilation, self.groups)
                return y_t.transpose(1, 2)

            raise RuntimeError(
                f"conv1d channel mismatch: input {tuple(x.shape)}, expected Cin {expected_Cin}"
            )

        # self.kind == "conv2d"
        # weight shape: [Cout, Cin_per_group, kH, kW] → expected Cin = Cin_per_group * groups
        if x.ndim != 4:
            raise RuntimeError(f"conv2d expects 4D input, got {tuple(x.shape)}")
        expected_Cin = w.shape[1] * self.groups

        # channels-first (N, C, H, W)
        if x.shape[1] == expected_Cin:
            return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

        # channels-last (N, H, W, C) → permute to (N, C, H, W), conv, permute back
        if x.shape[3] == expected_Cin:
            x_t = x.permute(0, 3, 1, 2)
            y_t = F.conv2d(x_t, w, b, self.stride, self.padding, self.dilation, self.groups)
            return y_t.permute(0, 2, 3, 1)

        raise RuntimeError(
            f"conv2d channel mismatch: input {tuple(x.shape)}, expected Cin {expected_Cin}"
        )


def _wrap_fp8_inplace(module: nn.Module, quantization: str = "fp8_e4m3fn", state_dict: dict | None = None):
    """
    Walk the module tree and replace Linear/Conv1d/Conv2d with FP8WeightWrapper.

    - Skips any submodule whose qualified name contains a deny substring.
    - If the checkpoint (state_dict) already has FP8 for <name>.weight, those bytes are copied
      verbatim into the wrapper (no re-encoding). Otherwise, the weight is downcast once to FP8.
    - Compute remains in the activation dtype at runtime (the wrapper upcasts on forward).
    - Returns (counts_per_type, saved_bytes).

    Args:
        module:      root nn.Module to transform in place.
        quantization:"fp8_e5m2" or "fp8_e4m3fn" — the FP8 storage dtype to use when downcasting.
        state_dict:  optional checkpoint tensors to source FP8 bytes from (for exact retention).

    Example:
        counts, saved = _wrap_fp8_inplace(foley_model, "fp8_e5m2", state_dict)
    """
    # Choose FP8 storage dtype based on the string; default path is e4m3fn.
    qdtype = torch.float8_e5m2 if quantization == "fp8_e5m2" else torch.float8_e4m3fn

    # Per-type replacement counters; useful for logging coverage.
    counts = {"linear": 0, "conv1d": 0, "conv2d": 0}

    # Total bytes saved (approx) = sum(original_bytes - fp8_bytes) for each replaced weight.
    saved_bytes = 0

    def _recurse(parent: nn.Module, prefix: str = ""):
        nonlocal saved_bytes
        # Iterate over immediate children so we can replace them in place.
        for name, child in list(parent.named_children()):
            # Qualified name (e.g., "triple_blocks.2.audio_mlp.fc1")
            full = f"{prefix}{name}" if prefix else name

            # Respect deny list: skip wrapping and keep descending into its children.
            if any(tok in full for tok in _DENY_SUBSTRINGS):
                _recurse(child, full)
                continue

            # Decide if this child is one of the supported types we wrap.
            kind = (
                "linear" if isinstance(child, nn.Linear)
                else "conv1d" if isinstance(child, nn.Conv1d)
                else "conv2d" if isinstance(child, nn.Conv2d)
                else None
            )

            if kind is None:
                # Not a target type; recurse to search deeper.
                _recurse(child, full)
                continue

            # Compute original weight footprint in bytes for reporting.
            before = child.weight.numel() * child.weight.element_size()

            # Build a wrapper with FP8 storage, seeded from the current module.
            wrapped = FP8WeightWrapper(child, qdtype)

            # Fast path: if the checkpoint already had FP8 for this exact tensor name,
            # copy those bytes (no re-quantization drift); cast only if FP8 variant differs.
            if state_dict is not None:
                w_src = state_dict.get(f"{full}.weight")
                if isinstance(w_src, torch.Tensor) and w_src.dtype in _FP8_DTYPES:
                    with torch.no_grad():
                        wrapped.weight.copy_(w_src if w_src.dtype == qdtype else w_src.to(qdtype))

            # Replace the child with our FP8 wrapper in the parent module.
            setattr(parent, name, wrapped)

            # Update counters and saved-bytes estimate (FP8 is 1 byte per element).
            counts[kind] += 1
            saved_bytes += max(0, before - wrapped.weight.numel() * 1)

    # Kick off the in-place transformation from the provided root.
    _recurse(module)

    # Return how many modules we wrapped per type and the approximate memory saved.
    return counts, saved_bytes


# -----------------------------------------------------------------------------------
# DTYPE / QUANT DETECTION HELPERS
# -----------------------------------------------------------------------------------

def _detect_ckpt_fp8(state_dict):
    """Return 'fp8_e5m2' / 'fp8_e4m3fn' if any tensor in the checkpoint uses that dtype; else None."""
    detected = None
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.float8_e5m2:
                detected = "fp8_e5m2"
                break
            if v.dtype == torch.float8_e4m3fn:
                detected = "fp8_e4m3fn"
                break
    return detected


def _detect_ckpt_major_precision(state_dict):
    """Return torch dtype among {bf16, fp16, fp32} that dominates parameter sizes in the checkpoint."""
    counts = {torch.bfloat16: 0, torch.float16: 0, torch.float32: 0}
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            if v.dtype in counts:
                counts[v.dtype] += v.numel()
    if all(c == 0 for c in counts.values()):
        return torch.bfloat16
    return max(counts, key=counts.get)


# --- HY-FOLEY: during Inductor compile, default tensor factories -> CUDA if unspecified ---
class _CudaFactoriesDuringCompile:
    """
    Scope-limited patch: while active, torch factory calls with no explicit device
    will default to CUDA (if available). This targets Inductor's tiny compile-time
    scratch tensors so it never kicks the CPU codegen path on Windows.
    """
    _NAMES = ("empty", "zeros", "full", "arange", "linspace", "tensor")

    def __enter__(self):
        self.torch = torch
        self.saved = {n: getattr(torch, n) for n in self._NAMES}

        def _wrap(fn):
            def inner(*args, **kwargs):
                # Only add device if missing; no change if caller already set it.
                if "device" not in kwargs and torch.cuda.is_available():
                    kwargs["device"] = "cuda"
                return fn(*args, **kwargs)
            return inner

        for n, fn in self.saved.items():
            setattr(torch, n, _wrap(fn))
        return self

    def __exit__(self, exc_type, exc, tb):
        for n, fn in self.saved.items():
            setattr(self.torch, n, fn)
