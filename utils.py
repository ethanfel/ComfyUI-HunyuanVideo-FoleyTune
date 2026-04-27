# utils.py
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
    """Map a2a strength to the correct starting step via sigma lookup.

    Instead of the naive `steps - int(steps * strength)` which maps linearly
    to step indices, this finds the step whose sigma is closest to the target.
    With non-linear sigma schedules (sd3 shift, flux shift) the difference
    is significant — linear mapping over- or under-noises by up to 15%.
    """
    sigma_target = strength
    # sigmas are in decreasing order (1.0 → 0.0)
    # Find first index where sigma <= target (that's where we start denoising)
    for i in range(len(sigmas)):
        if sigmas[i] <= sigma_target:
            return i, sigma_target
    return len(sigmas) - 1, sigma_target


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

    # --- PAD EMBEDDINGS TOKENZIER ---

    T_cur_len = int(text_feat_rep.shape[1])
    cap   = _caps(model_dict, cfg)

    # Two-bucket policy: 77 normally, 128 if prompt exceeds 77 (respect hard caps)
    if T_cur_len <= 77:
        T_fixed = min(77, cap)
    else:
        T_fixed = min(128, cap)

    # Cache once per session to avoid flapping if prompts bounce around
    if not hasattr(model_dict.foley_model, "_text_len_fixed"):
        model_dict.foley_model._text_len_fixed = T_fixed
    # If you prefer “sticky first bucket,” comment the next line.
    else:
        # stick to bigger bucket if it's triggered
        model_dict.foley_model._text_len_fixed = max(model_dict.foley_model._text_len_fixed, T_fixed)

    T_fixed = model_dict.foley_model._text_len_fixed
    logger.info(f"Using T_FIXED bucket: {T_fixed} (prompt had {T_cur_len} tokens; cap {cap})")

    # Normalize shapes for compile reuse
    text_feat_rep   = _pad_or_trim_time(text_feat_rep,   T_fixed)
    uncond_text_rep = _pad_or_trim_time(uncond_text_rep, T_fixed)

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

            # Use precomputed conditional/unconditional features (no per-step rebuild)
            siglip2_feat_input = pre_siglip2_input
            syncformer_feat_input = pre_sync_input
            text_feat_input = pre_text_input

            # Match inputs to the model's actual compute dtype to avoid matmul dtype mismatches
            compute_dtype = next(model_dict.foley_model.parameters()).dtype
            latent_input = latent_input.to(dtype=compute_dtype)
            siglip2_feat_input = siglip2_feat_input.to(dtype=compute_dtype)
            syncformer_feat_input = syncformer_feat_input.to(dtype=compute_dtype)
            text_feat_input = text_feat_input.to(dtype=compute_dtype)

            # Predict the noise residual
            if compute_dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type=latent_input.device.type, dtype=compute_dtype):
                    noise_pred = model_dict.foley_model(
                        x=latent_input, t=t_expand, cond=text_feat_input,
                        clip_feat=siglip2_feat_input, sync_feat=syncformer_feat_input
                    )["x"]
            else:
                noise_pred = model_dict.foley_model(
                    x=latent_input, t=t_expand, cond=text_feat_input,
                    clip_feat=siglip2_feat_input, sync_feat=syncformer_feat_input
                )["x"]

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
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

    Returns list of (t_start, t_end) tuples in seconds.
    """
    # Clamp overlap to half the chunk duration
    if overlap_seconds >= chunk_duration:
        logger.warning(f"overlap_seconds ({overlap_seconds}) >= chunk_duration ({chunk_duration}), "
                       f"clamping to {chunk_duration * 0.5}")
        overlap_seconds = chunk_duration * 0.5

    # Single chunk — no splitting needed
    if duration <= chunk_duration:
        return [(0.0, duration)]

    stride = chunk_duration - overlap_seconds
    chunks = []
    t_start = 0.0
    while t_start < duration:
        t_end = min(t_start + chunk_duration, duration)
        chunks.append((t_start, t_end))
        if t_end >= duration:
            break
        t_start += stride

    # Merge tiny last chunk into previous to avoid wasting a full forward pass
    # on a chunk that's mostly overlap with little unique content
    if len(chunks) >= 2:
        last_dur = chunks[-1][1] - chunks[-1][0]
        if last_dur < overlap_seconds + 0.5:
            # Extend previous chunk to cover the remainder
            prev_start = chunks[-2][0]
            chunks[-2] = (prev_start, duration)
            chunks.pop()

    return chunks


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
    # Each segment spans 16/25 = 0.64s, stride = 8/25 = 0.32s
    # Segment i covers time [i*0.32, i*0.32 + 0.64] seconds
    # Each segment produces 8 output tokens
    seg_stride_s = 8.0 / 25.0   # 0.32s per segment stride
    total_sync_tokens = features["sync_feat"].shape[1]
    seg_start = max(0, int(t_start / seg_stride_s))
    seg_end = max(seg_start + 1, int(t_end / seg_stride_s))
    # Convert segment indices to token indices (8 tokens per segment)
    tok_start = seg_start * 8
    tok_end = min(seg_end * 8, total_sync_tokens)  # clamp to actual token count
    sync_feat = features["sync_feat"][:, tok_start:tok_end, :]
    # Ensure at least 8 tokens (one segment)
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
    overlap_seconds,
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
):
    """Chunked denoising with overlap stitching for long-form generation.

    Args:
        features: FOLEYTUNE_FEATURES dict (full video features)
        chunks: list of (t_start, t_end) tuples from compute_chunk_boundaries
        overlap_seconds: overlap duration in seconds
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
    target_dtype = model_dict.foley_model.dtype
    device = model_dict.device
    audio_frame_rate = cfg.model_config.model_kwargs.audio_frame_rate
    latent_dim = cfg.model_config.model_kwargs.audio_vae_latent_dim
    overlap_frames = int(overlap_seconds * audio_frame_rate)
    sample_rate = model_dict.dac_model.sample_rate

    # Single chunk — delegate to standard denoise
    if len(chunks) == 1:
        t_start, t_end = chunks[0]
        chunk_feats = slice_features_for_chunk(features, t_start, t_end)
        chunk_dur = t_end - t_start
        visual = {
            "siglip2_feat": chunk_feats["clip_feat"].to(device),
            "syncformer_feat": chunk_feats["sync_feat"].to(device),
        }
        text = {
            "text_feat": chunk_feats["text_feat"].to(device),
            "uncond_text_feat": chunk_feats["uncond_text_feat"].to(device),
        }
        # Slice init_latents for this chunk if provided
        chunk_init = None
        if init_latents is not None:
            frame_start = int(t_start * audio_frame_rate)
            frame_end = int(t_end * audio_frame_rate)
            chunk_init = init_latents[:, :, frame_start:frame_end]
        return denoise_process_with_generator(
            visual, text, chunk_dur, model_dict, cfg,
            guidance_scale, num_inference_steps, batch_size, sampler, generator,
            init_latents=chunk_init, strength=strength, noise_blend=noise_blend,
        )

    # --- Multi-chunk: set up per-chunk schedulers and latents ---
    # CRITICAL: each chunk needs its own scheduler instance because
    # FlowMatchDiscreteScheduler.step() increments an internal _step_index.
    chunk_schedulers = []
    shift = getattr(model_dict.foley_model, '_flow_shift_override', cfg.diffusion_config.sample_flow_shift)
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
                target_dtype, device, generator
            )
        chunk_latents.append(latent)

        c_feats = slice_features_for_chunk(features, t_start, t_end)
        chunk_visual_feats.append({k: c_feats[k].to(device) for k in ["clip_feat", "sync_feat"]})
        chunk_text_feats.append({k: c_feats[k].to(device) for k in ["text_feat", "uncond_text_feat"]})

    # Truncate timesteps if using audio2audio
    if a2a_start_step is not None:
        timesteps = timesteps[a2a_start_step:]

    # --- Precompute per-chunk CFG features ---
    chunk_cfg_inputs = []
    for i in range(len(chunks)):
        vis = chunk_visual_feats[i]
        txt = chunk_text_feats[i]

        siglip2_rep = vis["clip_feat"].repeat(batch_size, 1, 1)
        sync_rep = vis["sync_feat"].repeat(batch_size, 1, 1)
        text_rep = txt["text_feat"].repeat(batch_size, 1, 1)
        uncond_rep = txt["uncond_text_feat"].repeat(batch_size, 1, 1)

        # Pad/trim text to fixed bucket
        T_cur = text_rep.shape[1]
        cap = _caps(model_dict, cfg)
        T_fixed = min(77, cap) if T_cur <= 77 else min(128, cap)
        text_rep = _pad_or_trim_time(text_rep, T_fixed)
        uncond_rep = _pad_or_trim_time(uncond_rep, T_fixed)

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

        chunk_cfg_inputs.append({
            "clip": cfg_clip, "sync": cfg_sync, "text": cfg_text,
        })

    # --- Denoising loop ---
    total_steps = len(timesteps) * len(chunks)
    pbar = ProgressBar(total_steps)

    with torch.inference_mode():
        for step_idx, t in enumerate(timesteps):
            throw_exception_if_processing_interrupted()
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.long, device=device)
            else:
                t = t.to(device=device)

            for c_idx in range(len(chunks)):
                latents = chunk_latents[c_idx]
                cfg_in = chunk_cfg_inputs[c_idx]

                latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                t_expand = t.expand(latent_input.shape[0]).contiguous()

                compute_dtype = next(model_dict.foley_model.parameters()).dtype
                latent_input = latent_input.to(dtype=compute_dtype)

                if compute_dtype in (torch.float16, torch.bfloat16):
                    with torch.autocast(device_type=device.type, dtype=compute_dtype):
                        noise_pred = model_dict.foley_model(
                            x=latent_input, t=t_expand, cond=cfg_in["text"].to(compute_dtype),
                            clip_feat=cfg_in["clip"].to(compute_dtype),
                            sync_feat=cfg_in["sync"].to(compute_dtype),
                        )["x"]
                else:
                    noise_pred = model_dict.foley_model(
                        x=latent_input, t=t_expand, cond=cfg_in["text"],
                        clip_feat=cfg_in["clip"], sync_feat=cfg_in["sync"],
                    )["x"]

                if guidance_scale > 1.0:
                    uncond_pred, text_pred = noise_pred.chunk(2)
                    noise_pred = uncond_pred + guidance_scale * (text_pred - uncond_pred)

                chunk_latents[c_idx] = chunk_schedulers[c_idx].step(noise_pred, t, latents)[0]
                pbar.update(1)

            # SaFa swap after all chunks are updated this step
            if crossfade_mode == "safa" and overlap_frames > 0:
                for c_idx in range(len(chunks) - 1):
                    safa_binary_swap(
                        chunk_latents[c_idx], chunk_latents[c_idx + 1],
                        overlap_frames, step_idx
                    )

    # --- Stitch results ---
    if crossfade_mode == "safa":
        # Assemble: take non-overlap from each chunk, overlap already consistent
        parts = []
        overlap_left = overlap_frames // 2
        overlap_right = overlap_frames - overlap_left  # ceiling half
        for c_idx in range(len(chunks)):
            lat = chunk_latents[c_idx]
            if overlap_frames == 0:
                parts.append(lat)
            elif c_idx == 0:
                parts.append(lat[:, :, :-overlap_right])
            elif c_idx == len(chunks) - 1:
                parts.append(lat[:, :, overlap_left:])
            else:
                parts.append(lat[:, :, overlap_left:-overlap_right])
        full_latent = torch.cat(parts, dim=-1)

        with torch.inference_mode():
            dac_weight = next(model_dict.dac_model.parameters())
            latents_dec = full_latent.to(device=dac_weight.device, dtype=dac_weight.dtype)
            audio = model_dict.dac_model.decode(latents_dec)
        total_duration = features["duration"]
        audio = audio[:, :, :int(total_duration * sample_rate)]
        return audio, sample_rate

    elif crossfade_mode == "latent":
        # Equal-power crossfade on latents, then single DAC decode
        full_latent = chunk_latents[0]
        for c_idx in range(1, len(chunks)):
            full_latent = equal_power_crossfade(
                full_latent, chunk_latents[c_idx], overlap_frames, dim=-1
            )

        with torch.inference_mode():
            dac_weight = next(model_dict.dac_model.parameters())
            latents_dec = full_latent.to(device=dac_weight.device, dtype=dac_weight.dtype)
            audio = model_dict.dac_model.decode(latents_dec)
        total_duration = features["duration"]
        audio = audio[:, :, :int(total_duration * sample_rate)]
        return audio, sample_rate

    else:  # waveform
        # DAC decode each chunk, then equal-power crossfade on audio
        overlap_samples = int(overlap_seconds * sample_rate)
        with torch.inference_mode():
            dac_weight = next(model_dict.dac_model.parameters())
            chunk_audios = []
            for c_idx in range(len(chunks)):
                lat = chunk_latents[c_idx].to(device=dac_weight.device, dtype=dac_weight.dtype)
                chunk_audios.append(model_dict.dac_model.decode(lat))

        full_audio = chunk_audios[0]
        for c_idx in range(1, len(chunk_audios)):
            full_audio = equal_power_crossfade(
                full_audio, chunk_audios[c_idx], overlap_samples, dim=-1
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
