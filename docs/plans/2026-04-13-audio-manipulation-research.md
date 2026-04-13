# Audio Manipulation Techniques for HunyuanVideo-Foley

> Research document covering audio2audio, inpainting, denoising LoRA, and other
> latent-space manipulation techniques applicable to the flow matching architecture.

**Date:** 2026-04-13

---

## Architecture Context

### Flow Matching Scheduler
- Linear interpolation: `x(sigma) = sigma * noise + (1 - sigma) * data`
- Sigma goes from 1.0 (pure noise) to 0.0 (clean data) during inference
- Model predicts **velocity** `v = dx/dsigma = noise - data`
- Euler step: `x_next = x + v * (sigma_next - sigma)`
- Supports euler, heun-2, midpoint-2, kutta-4 solvers
- Timestep fed to model = `sigma * 1000` (range [0, 1000])
- **Already has img2img support** via `set_begin_index()` / `index_for_timestep()`

### DAC Codec
- Continuous mode (VAE): encoder outputs 128-dim latent, sampled via `DiagonalGaussianDistribution`
- Encoder rates: [2, 3, 4, 5, 8] => `hop_length = 960`
- At 48kHz: `audio_frame_rate = 50 frames/sec` (1 second = 50 latent frames x 128 channels)
- Decoder is deterministic; encoder has stochasticity via VAE sampling
- `.sample()` for stochastic encode, `.mode()` for deterministic

### Model Forward
- Input `x` is audio latent [B, 128, T], embedded via `PatchEmbed1D`
- Conditioned on: timestep, text (CLAP), visual (SigLIP2 CLIP), sync (Synchformer)
- 18 triple-stream blocks + 36 single-stream blocks

---

## Technique 1: Audio2Audio (img2img Equivalent)

**Priority: #1 — foundation for techniques 3, 4, 5, 6, 7, 10**

### Mechanism
Instead of starting at sigma=1 (pure noise), start at an intermediate sigma:

1. Encode existing audio through DAC → `z_data` [B, 128, T]
2. Sample noise `z_noise` of same shape
3. Pick starting sigma `s` (strength: 0.0=no change, 1.0=full generation)
4. Compute `z_start = s * z_noise + (1 - s) * z_data`
5. Denoise from sigma=s to sigma=0, skipping the first `(1-s)*N` steps

### Flow Matching vs DDPM
In DDPM, noise addition uses a nonlinear schedule. In flow matching, it's a simple linear
interpolation. The velocity field is trained for the full trajectory — starting at any
intermediate sigma is well-defined because the model sees the sigma value via timestep
embedding and adjusts accordingly.

### Code Changes (inference only)

In `denoise_process_with_generator()` in `utils.py`:

```python
def denoise_process_with_generator(
    ...,
    init_audio_latents=None,  # [1, 128, T] from DAC encode
    strength=1.0,             # 0.0=no change, 1.0=full generation
):
    scheduler.set_timesteps(num_inference_steps, device=device)

    if init_audio_latents is not None and strength < 1.0:
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = num_inference_steps - init_timestep
        timesteps = scheduler.timesteps[t_start:]
        sigma_start = scheduler.sigmas[t_start]

        noise = randn_tensor(init_audio_latents.shape, device=device,
                            dtype=target_dtype, generator=generator)
        latents = sigma_start * noise + (1 - sigma_start) * init_audio_latents
        latents = latents.repeat(batch_size, 1, 1)
    else:
        timesteps = scheduler.timesteps
        latents = prepare_latents_with_generator(...)
```

DAC encode utility:

```python
def encode_audio_to_latents(audio_waveform, dac_model, device):
    """Encode audio waveform [1, 1, samples] to DAC latents [1, 128, T]."""
    with torch.no_grad():
        z_dist, _, _, _, _ = dac_model.encode(audio_waveform.to(device, dtype=torch.float32))
        return z_dist.sample()  # or .mode() for deterministic
```

### Applications
- Regenerating audio that's close but not right
- Iterative refinement with tweaked prompts
- Foundation for all other techniques below

---

## Technique 2: Audio Inpainting

**Priority: #2 — enables temporal extension**

### Mechanism
Regenerate specific time segments while keeping the rest. At each denoising step,
replace "known" regions with the properly noised original at the current sigma level.

For each step at sigma `s`:
1. Model predicts velocity for the entire latent
2. Scheduler computes `x_{s-1}` for the whole tensor
3. Known regions: overwrite with `s_next * noise + (1 - s_next) * x_original`
4. Unknown regions: keep the model's prediction

### Code Changes (inference only)

```python
def denoise_with_inpainting(
    ...,
    original_latents,    # [1, 128, T] from DAC encode
    inpaint_mask,        # [1, 1, T] boolean, True = regenerate
):
    noise = randn_tensor(original_latents.shape, ...)

    for i, t in enumerate(timesteps):
        # normal model forward + CFG
        latents = scheduler.step(noise_pred, t, latents)[0]

        # re-noise original for the NEXT sigma level
        sigma_next = scheduler.sigmas[scheduler.step_index]
        original_noised = sigma_next * noise + (1 - sigma_next) * original_latents

        # replace known regions
        latents = torch.where(inpaint_mask, latents, original_noised)
```

### DAC Boundary Considerations
- Each latent frame = 960/48000 = 20ms of audio (~50fps resolution)
- DAC decoder has receptive field overlap — abrupt transitions cause clicks
- Use soft masks (gradient over a few frames) or crossfade decoded results at boundaries
- Use the same noise vector for both noised original and initial noise for coherence

### Applications
- Fixing a specific bad sound in an otherwise good generation
- Replacing segments where sync is off
- Removing unwanted artifacts in specific time windows

---

## Technique 3: SDEdit-Style Editing

**Priority: #3 — free once audio2audio exists**

### Mechanism
Audio2audio with a **different** text prompt. The structure/rhythm of the original is
preserved while content changes to match the new description.

1. Encode original audio → DAC latents
2. Audio2audio with new text prompt
3. Sync + CLIP features still from the video (anchors temporal structure)
4. Only text conditioning changes

### Strength Selection
- 0.2–0.4: Subtle — same sounds, different texture/quality
- 0.4–0.6: Noticeable — similar structure, different content
- 0.7–0.9: Major — only gross temporal structure preserved

### Code Changes
None beyond audio2audio (Technique 1). Just different prompt at inference.

---

## Technique 4: Denoising LoRA (Cleanup/Enhancement)

**Priority: #10 — most complex, needs training**

### Concept
Train a LoRA to steer the model toward clean audio, enabling noise removal and
quality enhancement.

### Approach A: SDEdit with Clean-Audio LoRA (recommended start)
1. Train LoRA on clean audio with standard flow matching loss (no code changes)
2. At inference: encode noisy audio → audio2audio with LoRA at strength 0.4–0.6
3. LoRA "knows" clean audio characteristics and steers denoising toward them

**Pros:** Uses existing training pipeline. No training code changes.
**Cons:** Indirect — relies on the LoRA's learned prior, not explicit noise mapping.

### Approach B: Noise-Conditioned Training (more principled)
Modify `flow_matching_loss()` to sometimes use degraded audio as the start point:

```python
def flow_matching_loss_denoising(model, x_clean, x_noisy, t, ...):
    B = x_clean.shape[0]
    x0_random = torch.randn_like(x_clean)

    # 50% of time, use noisy audio as start instead of random noise
    use_noisy = torch.rand(B, device=device) < 0.5
    x0 = torch.where(use_noisy.view(B, 1, 1), x_noisy, x0_random)

    # When using noisy start, bias t toward lower values (partial denoising)
    t_noisy = torch.rand(B, device=device) * 0.6   # [0, 0.6]
    t_random = torch.rand(B, device=device)          # [0, 1]
    t = torch.where(use_noisy, t_noisy, t_random)

    t_expand = t.view(B, 1, 1)
    xt = t_expand * x0 + (1 - t_expand) * x_clean
    v_target = x0 - x_clean
    # ... standard loss computation ...
```

**Pros:** Explicitly teaches the model to map degraded → clean.
**Cons:** Requires training code modifications + paired noisy/clean dataset.

### Approach C: Paired Flow Matching (most principled)
Train directly on noisy→clean pairs where the ODE maps degraded audio to clean:
- `x_0 = x_noisy`, `x_1 = x_clean`
- Velocity target = `x_noisy - x_clean`

**Problem:** Breaks standard flow matching assumption (x_0 should be Gaussian).
LoRA would only work for cleanup, not normal generation.

### Dataset Preparation for Approaches B/C
1. Take clean training clips (existing dataset)
2. Add synthetic degradation: background noise, reverb, compression artifacts, room tone
3. Encode both clean and noisy versions through DAC
4. Train with paired latents

---

## Technique 5: Style Transfer

**Priority: #9 — experimental**

### Approach A: Latent AdaIN
```python
z_styled = (z_gen - z_gen.mean(dim=-1, keepdim=True)) / z_gen.std(dim=-1, keepdim=True)
z_styled = z_styled * z_style.std(dim=-1, keepdim=True) + z_style.mean(dim=-1, keepdim=True)
```
Transfer global statistics (timbre, room tone) from one audio to another in latent space.

### Approach B: Early-Step Injection
During high-sigma denoising steps, blend style latent into current latent.
Style seeds the generation; video conditioning takes over for detail.

### Approach C: Prompt Engineering
Use text description of desired style. No code changes. Limited but trivial.

---

## Technique 6: Audio Super-Resolution

**Priority: #5 — free once audio2audio exists**

### Mechanism
1. Upsample low-quality audio to 48kHz
2. Encode through DAC (latents capture limited spectral content)
3. Audio2audio at medium strength (0.4–0.7)
4. Model hallucintates missing high-frequency content guided by video + prompt

### DAC Bottleneck
DAC captures content up to the Nyquist of the input. Low-quality input → limited
latent representation → diffusion model fills in what's missing based on learned
distribution of clean 48kHz audio.

---

## Technique 7: Variation Generation

**Priority: #4 — free once audio2audio exists**

### Mechanism
Audio2audio at very low strength (0.1–0.3) with different seeds.

At strength 0.1: latent = `0.1 * noise + 0.9 * original` — minimal but perceptible
changes in timbre, micro-timing, spectral detail.

Batch generation with different seeds produces multiple variations to pick from.

---

## Technique 8: Temporal Extension (Outpainting)

**Priority: #6 — needs inpainting**

### Mechanism
Extend audio forward or backward in time using inpainting.

**Forward:**
1. Encode existing audio → `z_existing` [1, 128, T_existing]
2. Create noise for extension → `z_extend` [1, 128, T_new]
3. Concatenate: `z_full = cat([z_existing, z_extend], dim=-1)`
4. Inpainting mask: known for [0:T_existing], unknown for [T_existing:]
5. Denoise with full-length video features

Can leverage existing `chunked_denoise_process()` infrastructure — generate
extension as a new chunk with overlap into existing audio.

---

## Technique 9: Conditioning Mixing

**Priority: #8 — easy but niche**

### Approach A: Latent Blending
```python
z_mixed = alpha * z_audio1 + (1 - alpha) * z_audio2
```

### Approach B: Feature Blending
Blend CLIP/sync features from different videos before passing to denoiser.
Already works with existing code — just supply blended features.

### Approach C: Dual-Conditioned Denoising
Run model twice per step with different conditioning, blend velocity predictions:
```python
v_pred = alpha * v_pred_A + (1 - alpha) * v_pred_B
```
Doubles compute cost.

---

## Technique 10: Audio Resynchronization

**Priority: #7 — needs Synchformer integration**

### Approach A: Segment Re-denoising
1. Identify desync'd segments using Synchformer's `compute_desync_score.py`
2. Audio2audio at moderate strength (0.3–0.5) on those segments
3. Sync features guide model to better alignment

### Approach B: Latent Time Warping
Apply sub-frame temporal shifts in latent space via `F.grid_sample`.
Fast but crude — only works for very small timing adjustments.

### Approach C: Hybrid Inpainting
Use Synchformer desync scores as per-frame confidence mask.
Re-denoise low-confidence regions via inpainting (Technique 2).

---

## Implementation Priority

| Priority | Technique | Effort | Training? | Dependencies |
|----------|-----------|--------|-----------|--------------|
| 1 | Audio2Audio | Low | No | None — build first |
| 2 | Inpainting | Low | No | None |
| 3 | SDEdit Editing | Free | No | Technique 1 |
| 4 | Variation Generation | Free | No | Technique 1 |
| 5 | Super-Resolution | Free | No | Technique 1 |
| 6 | Temporal Extension | Medium | No | Technique 2 |
| 7 | Resync | Medium | No | Technique 1 + Synchformer |
| 8 | Conditioning Mixing | Low | No | None |
| 9 | Style Transfer | Medium | No | Experimental |
| 10 | Denoising LoRA | High | Yes | Training code changes |

---

## Key Files to Modify

- **`utils.py`** — Core: add DAC encode utility, modify `denoise_process_with_generator()` for init latents / strength / inpainting mask
- **`nodes.py`** — New ComfyUI nodes: Audio2Audio, Inpainting, etc.
- **`scheduling_flow_match_discrete.py`** — Already has `set_begin_index()` for img2img — no changes needed
- **`lora/train.py`** — Only for Technique 4 (denoising LoRA): modify `flow_matching_loss()`
- **`dac.py`** — No changes needed; understand `encode()` returns `DiagonalGaussianDistribution`

---

## References

- [SDEdit: Image Synthesis and Editing with Stochastic Differential Equations](https://sde-image-editing.github.io/)
- [AUDIT: Audio Editing by Following Instructions with Latent Diffusion Models](https://arxiv.org/abs/2304.00830)
- [Flux img2img pipeline (HuggingFace Diffusers)](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_img2img.py)
- [Audio Super-Resolution with Latent Bridge Models](https://arxiv.org/html/2509.17609v1)
- [AudioLDM: Text-to-Audio Generation with Latent Diffusion Models](https://arxiv.org/abs/2301.12503)
- [Stable Audio Open](https://arxiv.org/html/2407.14358v2)
- [Audio Inpainting using Discrete Diffusion Model](https://arxiv.org/html/2507.08333v1)
- [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/html/2403.03206v1)
- [SemanticAudio: Audio Generation and Editing in Semantic Space](https://arxiv.org/pdf/2601.21402)
