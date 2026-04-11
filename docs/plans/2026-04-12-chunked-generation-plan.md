# Chunked Long-Form V2A Generation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add chunked long-form audio generation with three crossfade modes (SaFa, latent, waveform) via a new `FoleyChunkedSampler` node and extend the existing `FoleyFeatureExtractor` to output in-memory features.

**Architecture:** Two-node pipeline: `FoleyFeatureExtractor` (modified, in `nodes_lora.py`) extracts visual/text features once for the full video and outputs them as a `FOLEY_FEATURES` dict. `FoleyChunkedSampler` (new, in `nodes.py`) takes those features, chunks the denoising loop with overlap, and stitches via the selected crossfade mode.

**Tech Stack:** PyTorch, ComfyUI node API, existing HunyuanVideo-Foley model/DAC/scheduler infrastructure.

**Design doc:** `docs/plans/2026-04-12-chunked-generation-design.md`

---

### Task 1: Extend FoleyFeatureExtractor with negative_prompt and FOLEY_FEATURES output

**Files:**
- Modify: `nodes_lora.py:244-365` (FoleyFeatureExtractor class)

**Step 1: Add `negative_prompt` input and `FOLEY_FEATURES` output type**

In `FoleyFeatureExtractor.INPUT_TYPES`, add `negative_prompt` to the `required` dict:

```python
"negative_prompt": ("STRING", {"default": "", "multiline": True}),
```

Change `RETURN_TYPES` and `RETURN_NAMES` to include the features output:

```python
RETURN_TYPES = ("STRING", "FOLEY_FEATURES")
RETURN_NAMES = ("npz_path", "features")
```

**Step 2: Encode negative prompt and build FOLEY_FEATURES dict**

In the `extract_features` method, add the `negative_prompt` parameter. After the existing CLAP encoding block (line ~344), add unconditional text encoding:

```python
# Encode negative prompt (unconditional)
neg_text_inputs = hunyuan_deps.clap_tokenizer(
    [negative_prompt], padding=True, truncation=True, max_length=100,
    return_tensors="pt"
).to(device)
neg_clap_outputs = hunyuan_deps.clap_model(
    **neg_text_inputs, output_hidden_states=True, return_dict=True
)
uncond_text_embedding = neg_clap_outputs.last_hidden_state.cpu()  # [1, seq_len, 768]
```

Before the return statement, build the features dict:

```python
features = {
    "clip_feat": clip_features,           # [1, T_clip, 768]
    "sync_feat": sync_features,           # [1, T_sync, 768]
    "text_feat": text_embedding,          # [1, T_text, 768]
    "uncond_text_feat": uncond_text_embedding,  # [1, T_text, 768]
    "duration": duration,
}
```

Update the return to include features:

```python
return (str(npz_path), features)
```

**Step 3: Verify node loads in ComfyUI**

Run: `cd /media/p5/ComfyUI && python main.py --quick-test-for-ci 2>&1 | tail -5`

Expected: No import errors for the modified node.

**Step 4: Commit**

```bash
git add nodes_lora.py
git commit -m "feat: extend FoleyFeatureExtractor with negative_prompt and FOLEY_FEATURES output"
```

---

### Task 2: Add chunk boundary computation utility

**Files:**
- Modify: `utils.py` (add helper function after `feature_process_from_tensors`)

**Step 1: Implement `compute_chunk_boundaries`**

Add after the `feature_process_from_tensors` function (after line ~293):

```python
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
    return chunks
```

**Step 2: Commit**

```bash
git add utils.py
git commit -m "feat: add compute_chunk_boundaries utility for chunked generation"
```

---

### Task 3: Add feature slicing utility

**Files:**
- Modify: `utils.py` (add helper function after `compute_chunk_boundaries`)

**Step 1: Implement `slice_features_for_chunk`**

This slices the full-video features to a specific time window. The tricky part is Synchformer's overlapping window structure (16-frame windows, stride 8, at 25fps).

```python
def slice_features_for_chunk(features: dict, t_start: float, t_end: float):
    """Slice pre-computed features to a specific time window.
    
    Args:
        features: FOLEY_FEATURES dict with clip_feat, sync_feat, text_feat, etc.
        t_start: chunk start time in seconds
        t_end: chunk end time in seconds
    
    Returns:
        Dict with sliced clip_feat and sync_feat, shared text features.
    """
    chunk_dur = t_end - t_start

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
    # Map time to segment indices
    seg_stride_s = 8.0 / 25.0   # 0.32s per segment stride
    seg_start = max(0, int(t_start / seg_stride_s))
    seg_end = max(seg_start + 1, int(t_end / seg_stride_s))
    # Convert segment indices to token indices (8 tokens per segment)
    tok_start = seg_start * 8
    tok_end = seg_end * 8
    sync_feat = features["sync_feat"][:, tok_start:tok_end, :]
    # Clamp to available tokens
    if sync_feat.shape[1] == 0:
        sync_feat = features["sync_feat"][:, -8:, :]

    return {
        "clip_feat": clip_feat,
        "sync_feat": sync_feat,
        "text_feat": features["text_feat"],
        "uncond_text_feat": features["uncond_text_feat"],
    }
```

**Step 2: Commit**

```bash
git add utils.py
git commit -m "feat: add slice_features_for_chunk utility with Synchformer segment mapping"
```

---

### Task 4: Add crossfade stitching utilities

**Files:**
- Modify: `utils.py` (add helper functions)

**Step 1: Implement `safa_binary_swap`**

```python
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
```

**Step 2: Implement `equal_power_crossfade`**

```python
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
```

**Step 3: Commit**

```bash
git add utils.py
git commit -m "feat: add safa_binary_swap and equal_power_crossfade utilities"
```

---

### Task 5: Implement chunked denoising loop

**Files:**
- Modify: `utils.py` (add new function after `denoise_process_with_generator`)

**Step 1: Implement `chunked_denoise_process`**

This is the core function. It orchestrates the per-chunk denoising with the selected crossfade mode. For `safa` mode, all chunks are denoised jointly with binary swap at each step. For `latent` and `waveform` modes, chunks are denoised independently then stitched.

```python
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
):
    """Chunked denoising with overlap stitching for long-form generation.
    
    Args:
        features: FOLEY_FEATURES dict (full video features)
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
        visual = {k: chunk_feats[k].to(device) for k in ["clip_feat", "sync_feat"]}
        text = {k: chunk_feats[k].to(device) for k in ["text_feat", "uncond_text_feat"]}
        return denoise_process_with_generator(
            visual, text, chunk_dur, model_dict, cfg,
            guidance_scale, num_inference_steps, batch_size, sampler, generator
        )

    # --- Multi-chunk: set up scheduler and per-chunk latents ---
    scheduler = FlowMatchDiscreteScheduler(
        shift=cfg.diffusion_config.sample_flow_shift,
        solver=sampler
    )
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Prepare per-chunk latents and features
    chunk_latents = []
    chunk_visual_feats = []
    chunk_text_feats = []
    chunk_latent_lens = []

    for t_start, t_end in chunks:
        chunk_dur = t_end - t_start
        latent_len = int(chunk_dur * audio_frame_rate)
        chunk_latent_lens.append(latent_len)

        latent = prepare_latents_with_generator(
            scheduler, batch_size, latent_dim, latent_len,
            target_dtype, device, generator
        )
        chunk_latents.append(latent)

        c_feats = slice_features_for_chunk(features, t_start, t_end)
        chunk_visual_feats.append({k: c_feats[k].to(device) for k in ["clip_feat", "sync_feat"]})
        chunk_text_feats.append({k: c_feats[k].to(device) for k in ["text_feat", "uncond_text_feat"]})

    # --- Precompute per-chunk CFG features (same pattern as denoise_process_with_generator) ---
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

                chunk_latents[c_idx] = scheduler.step(noise_pred, t, latents)[0]
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
        for c_idx in range(len(chunks)):
            lat = chunk_latents[c_idx]
            if c_idx == 0:
                # First chunk: keep everything except right half of overlap
                if len(chunks) > 1:
                    parts.append(lat[:, :, :-(overlap_frames // 2)])
                else:
                    parts.append(lat)
            elif c_idx == len(chunks) - 1:
                # Last chunk: skip left half of overlap
                parts.append(lat[:, :, overlap_frames // 2:])
            else:
                # Middle chunks: skip both halves of overlap
                parts.append(lat[:, :, overlap_frames // 2:-(overlap_frames // 2)])
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
```

**Step 2: Add the import for `FlowMatchDiscreteScheduler` at the top of utils.py**

The import already exists at line 13. Verify it's available:

```python
from hunyuanvideo_foley.utils.schedulers import FlowMatchDiscreteScheduler
```

**Step 3: Commit**

```bash
git add utils.py
git commit -m "feat: implement chunked_denoise_process with safa/latent/waveform modes"
```

---

### Task 6: Implement FoleyChunkedSampler node

**Files:**
- Modify: `nodes.py:474` (insert new node class before `HunyuanFoleyTorchCompile`)

**Step 1: Add imports**

At the import block (line ~89), add the new utilities:

```python
from .utils import (
    denoise_process_with_generator,
    feature_process_from_tensors,
    compute_chunk_boundaries,
    chunked_denoise_process,
    _wrap_fp8_inplace,
    _detect_ckpt_fp8,
    _detect_ckpt_major_precision,
    _CudaFactoriesDuringCompile,
    load_dac_any
)
```

**Step 2: Implement the node class**

Insert before line 474 (`# NODE: Hunyuan Foley Torch Compile`):

```python
# -----------------------------------------------------------------------------------
# NODE: Chunked Sampler for Long-Form Generation
# -----------------------------------------------------------------------------------

class FoleyChunkedSampler:
    """Generate audio for long videos by chunking with overlap and crossfade.
    
    Connects to FoleyFeatureExtractor's FOLEY_FEATURES output. Splits denoising
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
                "hunyuan_model": ("HUNYUAN_MODEL",),
                "hunyuan_deps": ("HUNYUAN_DEPS",),
                "features": ("FOLEY_FEATURES",),
                "cfg_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "steps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1}),
                "sampler": (cls.SAMPLER_NAMES, {"default": "euler"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 6, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "chunk_duration": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 15.0, "step": 0.1,
                                    "tooltip": "Duration of each chunk in seconds. 8s matches training length."}),
                "overlap_seconds": ("FLOAT", {"default": 1.6, "min": 0.0, "max": 5.0, "step": 0.1,
                                     "tooltip": "Overlap between chunks in seconds. 1.6s = 20% of 8s chunk."}),
                "crossfade_mode": (cls.CROSSFADE_MODES, {"default": "safa",
                                    "tooltip": "safa: binary swap during denoising (best). latent: blend before DAC. waveform: blend after DAC."}),
                "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "torch_compile_cfg": ("TORCH_COMPILE_CFG",),
                "block_swap_args": ("BLOCKSWAPARGS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("audio_first", "audio_batch")
    FUNCTION = "generate_audio"
    CATEGORY = "audio/HunyuanFoley"

    def generate_audio(
        self,
        hunyuan_model,
        hunyuan_deps,
        features,
        cfg_scale,
        steps,
        sampler,
        batch_size,
        seed,
        chunk_duration,
        overlap_seconds,
        crossfade_mode,
        force_offload,
        torch_compile_cfg=None,
        block_swap_args=None,
    ):
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
        chunks = compute_chunk_boundaries(duration, chunk_duration, overlap_seconds)
        logger.info(f"Chunked generation: {len(chunks)} chunks for {duration:.1f}s "
                     f"(chunk={chunk_duration}s, overlap={overlap_seconds}s, mode={crossfade_mode})")
        for i, (ts, te) in enumerate(chunks):
            logger.info(f"  Chunk {i}: [{ts:.1f}, {te:.1f}]s ({te-ts:.1f}s)")

        # Apply torch.compile if configured
        if torch_compile_cfg is not None and not getattr(hunyuan_model, "_blocks_are_compiled", False):
            try:
                hunyuan_model = HunyuanFoleyTorchCompile._apply_torch_compile(
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

        # Run chunked denoising
        decoded_waveform, sample_rate = chunked_denoise_process(
            features=features,
            chunks=chunks,
            overlap_seconds=overlap_seconds,
            crossfade_mode=crossfade_mode,
            model_dict=model_dict_for_process,
            cfg=hunyuan_cfg,
            guidance_scale=cfg_scale,
            num_inference_steps=steps,
            batch_size=batch_size,
            sampler=sampler,
            generator=rng,
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
```

**Step 3: Register the node in NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS**

At line ~713:

```python
NODE_CLASS_MAPPINGS = {
    "HunyuanModelLoader": HunyuanModelLoader,
    "HunyuanDependenciesLoader": HunyuanDependenciesLoader,
    "HunyuanFoleySampler": HunyuanFoleySampler,
    "FoleyChunkedSampler": FoleyChunkedSampler,      # NEW
    "HunyuanFoleyTorchCompile": HunyuanFoleyTorchCompile,
    "HunyuanBlockSwap": HunyuanBlockSwap,
    "SelectAudioFromBatch": SelectAudioFromBatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanModelLoader": "Hunyuan-Foley Model Loader",
    "HunyuanDependenciesLoader": "Hunyuan-Foley Dependencies Loader",
    "HunyuanFoleySampler": "Hunyuan-Foley Sampler",
    "FoleyChunkedSampler": "Foley Chunked Sampler",  # NEW
    "HunyuanFoleyTorchCompile": "Hunyuan-Foley Torch Compile",
    "HunyuanBlockSwap": "Hunyuan-Foley BlockSwap Settings",
    "SelectAudioFromBatch": "Select Audio From Batch",
}
```

**Step 4: Verify import chain**

Run: `cd /media/p5/ComfyUI && python -c "from custom_nodes.ComfyUI_HunyuanVideoFoley.nodes import NODE_CLASS_MAPPINGS; print(list(NODE_CLASS_MAPPINGS.keys()))"`

Expected: List includes `FoleyChunkedSampler`.

**Step 5: Commit**

```bash
git add nodes.py
git commit -m "feat: add FoleyChunkedSampler node with safa/latent/waveform crossfade modes"
```

---

### Task 7: Export new utilities from utils.py

**Files:**
- Modify: `nodes.py:89-96` (import block)

**Step 1: Ensure all new utils are imported**

The import block at line 89 should include:

```python
from .utils import (
    denoise_process_with_generator,
    feature_process_from_tensors,
    compute_chunk_boundaries,
    chunked_denoise_process,
    _wrap_fp8_inplace,
    _detect_ckpt_fp8,
    _detect_ckpt_major_precision,
    _CudaFactoriesDuringCompile,
    load_dac_any
)
```

This was already addressed in Task 6. Verify no circular imports.

**Step 2: Commit (if any changes needed)**

```bash
git add nodes.py
git commit -m "fix: ensure chunked generation utils are imported"
```

---

### Task 8: Final integration test and commit

**Step 1: Verify the full import chain loads without errors**

Run: `cd /media/p5/ComfyUI && python -c "
from custom_nodes.ComfyUI_HunyuanVideoFoley.nodes import NODE_CLASS_MAPPINGS
from custom_nodes.ComfyUI_HunyuanVideoFoley.nodes_lora import NODE_CLASS_MAPPINGS as LORA_MAPPINGS
print('nodes.py classes:', list(NODE_CLASS_MAPPINGS.keys()))
print('nodes_lora.py classes:', list(LORA_MAPPINGS.keys()))
print('FoleyChunkedSampler inputs:', list(NODE_CLASS_MAPPINGS['FoleyChunkedSampler'].INPUT_TYPES()['required'].keys()))
"`

Expected output should include `FoleyChunkedSampler` in the node list and show all required inputs.

**Step 2: Verify FoleyFeatureExtractor has updated outputs**

Run: `cd /media/p5/ComfyUI && python -c "
from custom_nodes.ComfyUI_HunyuanVideoFoley.nodes_lora import NODE_CLASS_MAPPINGS
ext = NODE_CLASS_MAPPINGS['FoleyFeatureExtractor']
print('RETURN_TYPES:', ext.RETURN_TYPES)
print('RETURN_NAMES:', ext.RETURN_NAMES)
"`

Expected: `('STRING', 'FOLEY_FEATURES')` and `('npz_path', 'features')`.

**Step 3: Final commit if any fixes were needed**

```bash
git add -u
git commit -m "fix: integration fixes for chunked generation pipeline"
```
