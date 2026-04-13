# Denoising & Upscaling LoRA Design

## Goal

Train two specialized LoRA adapters:

1. **Denoising LoRA** (priority) — teaches the model what clean audio sounds like so it naturally produces cleaner output when used with audio2audio. Fixes mic noise, background hum, room tone, DAC codec artifacts, and high-frequency harshness.
2. **Upscaling LoRA** — richer frequency content, sharper transients, better detail and separation.

Both use audio2audio inference (denoise 0.2-0.5) to apply as a cleanup/enhancement pass on already-generated audio. Can be stacked via existing LoRA loader chaining.

---

## Denoising LoRA

### Dataset

- **Source:** 100-200 high-quality clips from BBC Sound Effects, Freesound (CC0 studio recordings), or similar libraries.
- **Requirements:** Genuinely clean — no room tone, no mic noise, no compression artifacts, no reverb.
- **Processing:** Existing dataset pipeline (resample 48kHz -> LUFS normalize -> inspect -> quality filter) with strict thresholds. Quality filter silence_threshold and anomaly_threshold set high to reject anything marginal.

### Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| **LoRA targets** | `audio_attn` (36 layers) | Denoising is about audio fidelity, not video-audio alignment |
| **Rank** | 64 | Narrower task needs less capacity |
| **Min-SNR gamma** | 5 | Weight loss toward low-noise timesteps where fine detail is learned |
| **Timestep max** | 0.5 | Skip high-noise timesteps entirely — model already knows structural generation |
| **Spectral loss** | Yes, weight 0.1 | Penalize frequency-domain error, especially high-frequency |
| **Spectral HF weight** | 2.0 | Extra penalty on high-frequency bands (>8kHz in latent space) |
| **LR** | 1e-4 | Same as default |
| **Steps** | 15000 | Same as default |

### Min-SNR Weighting

Signal-to-noise ratio for flow matching with linear schedule:

```
sigma = t  (shift=1.0, linear)
SNR = (1 - sigma)^2 / sigma^2
weight = clamp(SNR, max=gamma)
```

At low sigma (clean-ish samples), SNR is high and gets clamped to gamma. At high sigma (noisy), SNR is low and the loss contribution is reduced. Net effect: the model spends most of its learning capacity on the fine-detail cleanup range.

### Spectral Loss

Multi-resolution STFT loss computed in DAC latent space:

```python
def spectral_loss(predicted, target, window_sizes=[4, 16, 64]):
    """STFT loss across multiple resolutions in latent space.
    
    DAC latents are [B, C, T] where C=128 channels at 50fps.
    We treat channels as the "frequency" dimension and compute
    STFT along the time axis at multiple window sizes.
    """
    total = 0
    for ws in window_sizes:
        pred_stft = torch.stft(predicted, n_fft=ws, return_complex=True)
        tgt_stft = torch.stft(target, n_fft=ws, return_complex=True)
        
        mag_pred = pred_stft.abs()
        mag_tgt = tgt_stft.abs()
        
        # Convergence loss (L1 on magnitude)
        convergence = F.l1_loss(mag_pred, mag_tgt)
        
        # Log magnitude loss (perceptual)
        log_pred = torch.log1p(mag_pred)
        log_tgt = torch.log1p(mag_tgt)
        log_loss = F.l1_loss(log_pred, log_tgt)
        
        # HF emphasis: weight upper half of frequency bins 2x
        n_bins = mag_pred.shape[-2]
        hf_mask = torch.ones_like(mag_pred)
        hf_mask[..., n_bins // 2:, :] = 2.0
        hf_loss = F.l1_loss(mag_pred * hf_mask, mag_tgt * hf_mask)
        
        total += convergence + log_loss + 0.5 * hf_loss
    
    return total / len(window_sizes)
```

### Inference

Load denoising LoRA -> audio2audio with denoise 0.2-0.4. Feed generated audio back through the sampler. Light pass cleans artifacts without destroying structure.

---

## Upscaling LoRA

### Dataset

- **Source:** 100-200 spectrally rich recordings — studio music, close-mic foley, high-quality field recordings.
- **Requirements:** Strong high-frequency content (cymbals, breath sounds, paper rustling, rain, sizzle). Full bandwidth up to 24kHz.
- **Processing:** Same pipeline, but quality filter should also check for spectral richness (reject dull/muffled recordings).

### Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| **LoRA targets** | `audio_attn` (36 layers) | Same as denoising |
| **Rank** | 128 | Broader task — adding detail across full spectrum |
| **Min-SNR gamma** | 5 | Same weighting strategy |
| **Timestep max** | 0.7 | Broader range — upscaling needs medium-noise detail too |
| **Spectral loss** | Yes, weight 0.1 | Multi-resolution, HF-weighted |
| **Spectral HF weight** | 2.0 | Same HF emphasis |
| **LR** | 1e-4 | Same as default |
| **Steps** | 15000 | Same as default |

### Inference

Load upscaling LoRA -> audio2audio with denoise 0.3-0.5. Slightly higher denoise than cleanup to allow the model to add detail.

---

## LoRA Stacking

Both adapters apply simultaneously via existing `FoleyTuneLoRALoader` — chain two loader nodes with different adapters and strengths. No code changes needed; it's already weighted addition on the same linear layers.

Recommended stacking: denoising at 0.8 strength + upscaling at 0.5 strength. Denoising first in the chain (applied to base weights), upscaling second (applied on top).

---

## Training Code Changes

All changes are additive — existing training behavior is unchanged when new flags are off.

### New Functions (in `lora/train.py`)

1. **`min_snr_weight(sigma, gamma=5.0)`** — Compute per-sample loss weight from sigma values
2. **`multi_resolution_spectral_loss(predicted, target, window_sizes, hf_weight)`** — STFT loss in latent space

### New Config Parameters (on `FoleyTuneLoRATrainer` node)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_min_snr` | BOOLEAN | False | Enable Min-SNR loss weighting |
| `min_snr_gamma` | FLOAT | 5.0 | Min-SNR clamp value |
| `use_spectral_loss` | BOOLEAN | False | Enable multi-resolution spectral loss |
| `spectral_loss_weight` | FLOAT | 0.1 | Weight of spectral loss vs MSE |
| `spectral_hf_weight` | FLOAT | 2.0 | Extra weight on high-frequency bins |
| `timestep_max` | FLOAT | 1.0 | Cap timestep sampling range (1.0 = no cap) |

### Modified Flow

```
flow_matching_loss():
    # existing: sample timesteps, compute velocity target, predict, MSE
    
    # new: apply timestep_max cap
    if timestep_max < 1.0:
        t = t * timestep_max
    
    # existing: velocity prediction and MSE loss
    mse = F.mse_loss(predicted, target, reduction='none')
    
    # new: min-SNR weighting
    if use_min_snr:
        weights = min_snr_weight(sigma, gamma=min_snr_gamma)
        mse = mse * weights
    
    loss = mse.mean()
    
    # new: spectral loss
    if use_spectral_loss:
        spec_loss = multi_resolution_spectral_loss(predicted, target)
        loss = loss + spectral_loss_weight * spec_loss
    
    return loss
```

---

## Summary

| | Denoising LoRA | Upscaling LoRA |
|---|---|---|
| **Dataset** | Clean studio recordings | Spectrally rich recordings |
| **Targets** | audio_attn (36 layers) | audio_attn (36 layers) |
| **Rank** | 64 | 128 |
| **Timestep max** | 0.5 | 0.7 |
| **Spectral loss** | Yes (0.1 weight, 2x HF) | Yes (0.1 weight, 2x HF) |
| **Min-SNR** | gamma=5 | gamma=5 |
| **Inference denoise** | 0.2-0.4 | 0.3-0.5 |
| **Stackable** | Yes, strength 0.8 | Yes, strength 0.5 |
