# Audio-to-Audio & Inpainting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add audio2audio (img2img equivalent), inpainting, feature blending, and style transfer nodes to the existing FoleyTune sampler infrastructure.

**Architecture:** Extend the existing `denoise_process_with_generator()` and `chunked_denoise_process()` with optional `init_latents`, `strength`, and inpainting mask params. Add `init_audio` + `strength` optional inputs to `FoleyTuneChunkedSampler`. Create three new nodes: `FoleyTuneInpainter`, `FoleyTuneFeatureBlender`, `FoleyTuneStyleTransfer`.

**Tech Stack:** PyTorch, ComfyUI node system, DAC VAE (continuous mode), flow matching scheduler

---

### Task 1: Add `encode_audio_to_latents()` utility

**Files:**
- Modify: `utils.py` (add after line 122, before `denoise_process_with_generator`)

**Step 1: Add the encode utility function**

Add after the existing `prepare_latents_with_generator` function:

```python
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
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('utils.py').read()); print('ok')"`

**Step 3: Commit**

```bash
git add utils.py
git commit -m "feat: add encode_audio_to_latents utility for DAC encoding"
```

---

### Task 2: Extend `denoise_process_with_generator()` with init_latents + strength

**Files:**
- Modify: `utils.py:125-258` (`denoise_process_with_generator` function)

**Step 1: Add new parameters to the function signature**

Change the signature at line 125 to:

```python
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
):
```

**Step 2: Add init_latents logic after scheduler setup**

Replace lines 148-156 (scheduler setup + latent init) with:

```python
    scheduler.set_timesteps(num_inference_steps, device=device)

    if init_latents is not None and strength < 1.0:
        # Audio2Audio: start denoising from partially noised init_latents
        # Calculate how many steps to skip
        start_step = max(num_inference_steps - int(num_inference_steps * strength), 0)
        timesteps = scheduler.timesteps[start_step:]
        sigma_start = scheduler.sigmas[start_step]

        # Noise the init latents at the starting sigma level
        # Flow matching: x_t = sigma * noise + (1 - sigma) * data
        noise = randn_tensor(
            init_latents.shape, device=device, dtype=target_dtype, generator=generator
        )
        latents = sigma_start * noise + (1 - sigma_start) * init_latents.to(device=device, dtype=target_dtype)
        latents = latents.repeat(batch_size, 1, 1) if latents.shape[0] == 1 else latents
    else:
        timesteps = scheduler.timesteps
        latents = prepare_latents_with_generator(
            scheduler, batch_size=batch_size,
            num_channels_latents=cfg.model_config.model_kwargs.audio_vae_latent_dim,
            length=audio_len_in_s * cfg.model_config.model_kwargs.audio_frame_rate,
            dtype=target_dtype, device=device, generator=generator
        )
```

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('utils.py').read()); print('ok')"`

**Step 4: Commit**

```bash
git add utils.py
git commit -m "feat: add init_latents + strength to denoise_process_with_generator"
```

---

### Task 3: Extend `denoise_process_with_generator()` with inpainting support

**Files:**
- Modify: `utils.py:125-258` (`denoise_process_with_generator` function)

**Step 1: Add inpaint parameters to signature**

Add after `strength=1.0`:

```python
    inpaint_mask=None,
    inpaint_original=None,
    inpaint_noise=None,
```

**Step 2: Add inpaint replacement logic inside the denoising loop**

After the scheduler step line (`latents = scheduler.step(noise_pred, t, latents)[0]`), add:

```python
            # Inpainting: replace known regions with properly noised original
            if inpaint_mask is not None and inpaint_original is not None:
                # Get sigma for the NEXT step (after this step's update)
                if i + 1 < len(timesteps):
                    next_step_idx = scheduler._step_index  # already incremented by step()
                    sigma_next = scheduler.sigmas[next_step_idx]
                else:
                    sigma_next = 0.0  # final step: use clean original
                original_noised = sigma_next * inpaint_noise + (1 - sigma_next) * inpaint_original
                # mask: True = regenerate (model output), False = keep original
                mask_expanded = inpaint_mask.to(device=latents.device, dtype=latents.dtype)
                latents = latents * mask_expanded + original_noised * (1 - mask_expanded)
```

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('utils.py').read()); print('ok')"`

**Step 4: Commit**

```bash
git add utils.py
git commit -m "feat: add inpainting mask support to denoise_process_with_generator"
```

---

### Task 4: Extend `chunked_denoise_process()` with init_latents + strength

**Files:**
- Modify: `utils.py:441-677` (`chunked_denoise_process` function)

**Step 1: Add new parameters to signature**

Add after `generator=None`:

```python
    init_latents=None,
    strength=1.0,
```

**Step 2: Update single-chunk delegation (line 480-496)**

Pass new params to `denoise_process_with_generator`:

```python
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
            init_latents=chunk_init, strength=strength,
        )
```

**Step 3: Update multi-chunk latent initialization (line 516-524)**

Replace the latent preparation inside the per-chunk loop:

```python
    for c_idx, (t_start, t_end) in enumerate(chunks):
        chunk_dur = t_end - t_start
        latent_len = int(chunk_dur * audio_frame_rate)

        if init_latents is not None and strength < 1.0:
            # Slice init_latents for this chunk
            frame_start = int(t_start * audio_frame_rate)
            frame_end = int(t_end * audio_frame_rate)
            chunk_init = init_latents[:, :, frame_start:frame_end].to(device=device, dtype=target_dtype)

            # Calculate start step and sigma
            start_step = max(num_inference_steps - int(num_inference_steps * strength), 0)
            sigma_start = chunk_schedulers[c_idx].sigmas[start_step]

            noise = randn_tensor(
                chunk_init.shape, device=device, dtype=target_dtype, generator=generator
            )
            latent = sigma_start * noise + (1 - sigma_start) * chunk_init
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
```

**Step 4: Update timesteps for strength < 1.0**

After the per-chunk loop, before the denoising loop, adjust `timesteps`:

```python
    # Truncate timesteps if using audio2audio
    if init_latents is not None and strength < 1.0:
        start_step = max(num_inference_steps - int(num_inference_steps * strength), 0)
        timesteps = timesteps[start_step:]
```

**Step 5: Verify syntax**

Run: `python -c "import ast; ast.parse(open('utils.py').read()); print('ok')"`

**Step 6: Commit**

```bash
git add utils.py
git commit -m "feat: add init_latents + strength to chunked_denoise_process"
```

---

### Task 5: Add init_audio + strength to `FoleyTuneChunkedSampler`

**Files:**
- Modify: `nodes.py:261-397` (`FoleyTuneChunkedSampler` class)

**Step 1: Add optional inputs to INPUT_TYPES**

In the `"optional"` dict, add:

```python
            "optional": {
                "torch_compile_cfg": ("FOLEYTUNE_COMPILE_CFG",),
                "block_swap_args": ("FOLEYTUNE_BLOCKSWAP",),
                "init_audio": ("AUDIO", {"tooltip": "Reference audio for audio2audio. Connect to use img2img-style generation."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                              "tooltip": "Denoising strength. 1.0=full generation, 0.0=keep original. Lower values preserve more of init_audio."}),
            }
```

**Step 2: Add parameters to `generate_audio` method signature**

Add after `block_swap_args=None`:

```python
        init_audio=None,
        strength=1.0,
```

**Step 3: Add DAC encode logic before chunked_denoise_process call**

After the DAC device setup (`hunyuan_deps["dac_model"].to(device=device, dtype=torch.float32)`), add:

```python
        # Encode init audio to DAC latents if provided
        init_latents = None
        if init_audio is not None and strength < 1.0:
            from utils import encode_audio_to_latents
            init_waveform = init_audio["waveform"]
            # Ensure mono [B, 1, samples]
            if init_waveform.dim() == 2:
                init_waveform = init_waveform.unsqueeze(0)
            if init_waveform.shape[1] > 1:
                init_waveform = init_waveform[:, :1, :]  # take first channel
            init_latents = encode_audio_to_latents(init_waveform, hunyuan_deps["dac_model"], device)
            logger.info(f"Audio2Audio: encoded init_audio to latents {init_latents.shape}, strength={strength}")
```

**Step 4: Pass init_latents + strength to chunked_denoise_process**

Update the `chunked_denoise_process()` call to include the new params:

```python
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
            init_latents=init_latents,
            strength=strength,
        )
```

**Step 5: Verify syntax**

Run: `python -c "import ast; ast.parse(open('nodes.py').read()); print('ok')"`

**Step 6: Commit**

```bash
git add nodes.py
git commit -m "feat: add init_audio + strength optional inputs to ChunkedSampler"
```

---

### Task 6: Create `FoleyTuneInpainter` node

**Files:**
- Modify: `nodes.py` (add new class before NODE_CLASS_MAPPINGS, add to mappings)

**Step 1: Add the FoleyTuneInpainter class**

Add before `NODE_CLASS_MAPPINGS`:

```python
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

        # Encode init audio
        init_waveform = init_audio["waveform"]
        if init_waveform.dim() == 2:
            init_waveform = init_waveform.unsqueeze(0)
        if init_waveform.shape[1] > 1:
            init_waveform = init_waveform[:, :1, :]

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

        # Run denoising with inpainting
        audio, sample_rate = denoise_process_with_generator(
            visual, text, duration, model_dict, hunyuan_cfg,
            cfg_scale, steps, 1, sampler, rng,
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
```

**Step 2: Add to NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS**

```python
NODE_CLASS_MAPPINGS = {
    ...
    "FoleyTuneInpainter": FoleyTuneInpainter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    ...
    "FoleyTuneInpainter": "FoleyTune Inpainter",
}
```

**Step 3: Add imports at top of nodes.py**

Add import for `encode_audio_to_latents` from utils:

```python
from .utils import (denoise_process_with_generator, chunked_denoise_process,
                     compute_chunk_boundaries, encode_audio_to_latents, ...)
```

(Check existing import pattern and match it.)

**Step 4: Verify syntax**

Run: `python -c "import ast; ast.parse(open('nodes.py').read()); print('ok')"`

**Step 5: Commit**

```bash
git add nodes.py
git commit -m "feat: add FoleyTuneInpainter node for time-region regeneration"
```

---

### Task 7: Create `FoleyTuneFeatureBlender` node

**Files:**
- Modify: `nodes.py` (add new class, register in mappings)

**Step 1: Add the FoleyTuneFeatureBlender class**

```python
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
        # Use shortest common length for each feature type
        result = {}
        for key in ("clip_feat", "sync_feat", "text_feat", "uncond_text_feat"):
            a = features_a[key]
            b = features_b[key]
            min_len = min(a.shape[1], b.shape[1])
            a = a[:, :min_len, :]
            b = b[:, :min_len, :]
            result[key] = (1 - blend) * a + blend * b
        result["duration"] = (1 - blend) * features_a["duration"] + blend * features_b["duration"]
        return (result,)
```

**Step 2: Register in mappings**

Add to both `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`:
- `"FoleyTuneFeatureBlender": FoleyTuneFeatureBlender`
- `"FoleyTuneFeatureBlender": "FoleyTune Feature Blender"`

**Step 3: Verify syntax and commit**

```bash
python -c "import ast; ast.parse(open('nodes.py').read()); print('ok')"
git add nodes.py
git commit -m "feat: add FoleyTuneFeatureBlender node for conditioning mixing"
```

---

### Task 8: Create `FoleyTuneStyleTransfer` node

**Files:**
- Modify: `nodes.py` (add new class, register in mappings)

**Step 1: Add the FoleyTuneStyleTransfer class**

```python
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
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
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

        # Encode both
        content_wav = content_audio["waveform"]
        style_wav = style_audio["waveform"]
        if content_wav.dim() == 2:
            content_wav = content_wav.unsqueeze(0)
        if style_wav.dim() == 2:
            style_wav = style_wav.unsqueeze(0)
        if content_wav.shape[1] > 1:
            content_wav = content_wav[:, :1, :]
        if style_wav.shape[1] > 1:
            style_wav = style_wav[:, :1, :]

        z_content = encode_audio_to_latents(content_wav, dac, device)
        z_style = encode_audio_to_latents(style_wav, dac, device)

        # AdaIN: normalize content, apply style statistics
        # Channel-wise (dim=128) statistics over time (dim=-1)
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

        sample_rate = content_audio["sample_rate"]
        # Trim to content length
        content_samples = content_wav.shape[-1]
        audio = audio[:, :, :content_samples]

        audio_out = {"waveform": audio.float().cpu(), "sample_rate": sample_rate}
        return (audio_out,)
```

**Step 2: Register in mappings**

Add to both `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`:
- `"FoleyTuneStyleTransfer": FoleyTuneStyleTransfer`
- `"FoleyTuneStyleTransfer": "FoleyTune Style Transfer"`

**Step 3: Verify syntax and commit**

```bash
python -c "import ast; ast.parse(open('nodes.py').read()); print('ok')"
git add nodes.py
git commit -m "feat: add FoleyTuneStyleTransfer node for latent AdaIN style transfer"
```

---

### Task 9: Final integration — verify imports and registration

**Files:**
- Modify: `nodes.py` (verify all imports work)
- Modify: `__init__.py` (verify node registration)

**Step 1: Verify all imports are correct**

Check that `nodes.py` imports `encode_audio_to_latents` from utils. Check the existing import pattern:

```python
# At top of nodes.py, ensure these are imported:
from .utils import (
    ...,
    encode_audio_to_latents,
    denoise_process_with_generator,
    ...
)
```

**Step 2: Verify `__init__.py` picks up the new nodes**

Check that `__init__.py` imports from `nodes.py` and that `NODE_CLASS_MAPPINGS` propagates.

**Step 3: Full syntax check on all modified files**

```bash
python -c "import ast; ast.parse(open('utils.py').read()); print('utils ok')"
python -c "import ast; ast.parse(open('nodes.py').read()); print('nodes ok')"
```

**Step 4: Commit and push**

```bash
git add utils.py nodes.py
git commit -m "feat: complete audio2audio, inpainting, feature blending, style transfer"
git push
```
