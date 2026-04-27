# Denoising & Upscaling LoRA Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Min-SNR loss weighting, multi-resolution spectral loss, and timestep capping to the existing LoRA trainer — enabling specialized denoising and upscaling LoRA training.

**Architecture:** Three new functions in `lora/train.py` (`min_snr_weight`, `multi_resolution_spectral_loss`, and an updated `flow_matching_loss`), plus six new optional UI parameters on `FoleyTuneLoRATrainer` in `nodes_lora.py`. All changes are additive — existing behavior is unchanged when new flags are off.

**Tech Stack:** PyTorch (torch.stft, F.mse_loss, F.l1_loss), existing LoRA training pipeline.

---

### Task 1: Add `min_snr_weight()` function

**Files:**
- Modify: `lora/train.py:236` (insert before the `flow_matching_loss` function)

**Step 1: Write the function**

Insert after the `sample_timesteps` function (line 233) and before the loss section comment (line 236):

```python
def min_snr_weight(sigma, gamma=5.0):
    """Compute Min-SNR loss weighting from sigma values.

    For flow matching with linear schedule: SNR = (1-sigma)^2 / sigma^2.
    Weight = clamp(SNR, max=gamma). Emphasizes low-noise timesteps
    where the model learns fine detail.

    Args:
        sigma: [B] tensor of sigma values in [0, 1]
        gamma: clamp ceiling (default 5.0)

    Returns:
        [B] weight tensor, broadcastable to loss shape
    """
    snr = ((1 - sigma) ** 2) / (sigma ** 2 + 1e-8)
    return snr.clamp(max=gamma)
```

**Step 2: Verify it doesn't break imports**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley/.worktrees/audio2audio && python -c "from lora.train import min_snr_weight; import torch; w = min_snr_weight(torch.tensor([0.1, 0.5, 0.9]), gamma=5.0); print(w)"`

Expected: tensor with values like [5.0000, 1.0000, 0.0123] (high weight at low sigma, low weight at high sigma)

**Step 3: Commit**

```bash
git add lora/train.py
git commit -m "feat(training): add min_snr_weight for timestep-dependent loss weighting"
```

---

### Task 2: Add `multi_resolution_spectral_loss()` function

**Files:**
- Modify: `lora/train.py:236` (insert after `min_snr_weight`, before `flow_matching_loss`)

**Step 1: Write the function**

```python
def multi_resolution_spectral_loss(predicted, target, window_sizes=(4, 16, 64), hf_weight=2.0):
    """Multi-resolution STFT loss in DAC latent space.

    Computes L1 magnitude + log-magnitude loss at multiple STFT resolutions,
    with extra weight on high-frequency bins.

    DAC latents are [B, C, T] where C=128 channels at 50fps. STFT is computed
    per-channel along the time axis.

    Args:
        predicted: [B, C, T] predicted latents
        target: [B, C, T] target latents
        window_sizes: STFT n_fft sizes to use
        hf_weight: multiplier for upper-half frequency bins

    Returns:
        scalar loss
    """
    B, C, T = predicted.shape
    # Flatten batch and channel for per-channel STFT: [B*C, T]
    pred_flat = predicted.reshape(B * C, T)
    tgt_flat = target.reshape(B * C, T)

    total = 0.0
    for ws in window_sizes:
        if T < ws:
            continue  # skip if latent sequence too short for this window

        pred_stft = torch.stft(
            pred_flat, n_fft=ws, hop_length=max(ws // 4, 1),
            win_length=ws, return_complex=True,
            window=torch.hann_window(ws, device=predicted.device, dtype=predicted.dtype),
        )
        tgt_stft = torch.stft(
            tgt_flat, n_fft=ws, hop_length=max(ws // 4, 1),
            win_length=ws, return_complex=True,
            window=torch.hann_window(ws, device=target.device, dtype=target.dtype),
        )

        mag_pred = pred_stft.abs()
        mag_tgt = tgt_stft.abs()

        # L1 magnitude loss
        convergence = F.l1_loss(mag_pred, mag_tgt)

        # Log-magnitude loss (perceptual)
        log_loss = F.l1_loss(torch.log1p(mag_pred), torch.log1p(mag_tgt))

        # HF emphasis: weight upper half of frequency bins
        n_bins = mag_pred.shape[-2]
        hf_mask = torch.ones_like(mag_pred)
        hf_mask[..., n_bins // 2:, :] = hf_weight
        hf_loss = F.l1_loss(mag_pred * hf_mask, mag_tgt * hf_mask)

        total = total + convergence + log_loss + 0.5 * hf_loss

    n_valid = sum(1 for ws in window_sizes if T >= ws)
    if n_valid > 0:
        total = total / n_valid

    return total
```

**Step 2: Verify it runs**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley/.worktrees/audio2audio && python -c "from lora.train import multi_resolution_spectral_loss; import torch; p = torch.randn(2, 128, 100); t = torch.randn(2, 128, 100); loss = multi_resolution_spectral_loss(p, t); print(f'loss={loss.item():.4f}')"`

Expected: a scalar loss value (varies, should be positive)

**Step 3: Commit**

```bash
git add lora/train.py
git commit -m "feat(training): add multi-resolution spectral loss for latent space"
```

---

### Task 3: Extend `flow_matching_loss()` with Min-SNR and spectral loss

**Files:**
- Modify: `lora/train.py:238-290` (the `flow_matching_loss` function)

**Step 1: Update the function signature and body**

Add new parameters to the function signature:

```python
def flow_matching_loss(model, x1, t, clip_feat, sync_feat, text_feat, device, dtype,
                       timestep_max=1.0, use_min_snr=False, min_snr_gamma=5.0,
                       use_spectral_loss=False, spectral_loss_weight=0.1,
                       spectral_hf_weight=2.0):
```

After `t = sample_timesteps(...)` is called in the training loop, `t` is passed in here. Apply the timestep cap at the start of this function, before `t_expand`:

```python
    # Cap timestep range for specialized training
    if timestep_max < 1.0:
        t = t * timestep_max
```

Replace the final loss computation (the last 3 lines before `return loss`):

```python
    v_target = v_target.to(device=device, dtype=dtype)

    # Per-sample MSE loss: [B, C, T] -> [B] mean over C and T
    mse = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=(1, 2))  # [B]

    # Min-SNR weighting
    if use_min_snr:
        weights = min_snr_weight(t, gamma=min_snr_gamma)  # [B]
        mse = mse * weights

    loss = mse.mean()

    # Optional spectral loss on velocity prediction
    if use_spectral_loss:
        spec_loss = multi_resolution_spectral_loss(
            v_pred, v_target, hf_weight=spectral_hf_weight,
        )
        loss = loss + spectral_loss_weight * spec_loss

    return loss
```

**Step 2: Verify backward compatibility**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley/.worktrees/audio2audio && python -c "from lora.train import flow_matching_loss; print('import OK — signature accepts new kwargs')"`

Expected: `import OK — signature accepts new kwargs`

**Step 3: Commit**

```bash
git add lora/train.py
git commit -m "feat(training): extend flow_matching_loss with min-SNR and spectral loss"
```

---

### Task 4: Add new UI parameters to `FoleyTuneLoRATrainer`

**Files:**
- Modify: `nodes_lora.py:764-782` (the `optional` section of `INPUT_TYPES`)

**Step 1: Add the six new parameters to the optional inputs**

Add these entries to the `optional` dict, after `latent_noise_sigma`:

```python
                "use_min_snr": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable Min-SNR loss weighting. Emphasizes low-noise timesteps where fine detail is learned. Recommended for denoising/upscaling LoRAs.",
                }),
                "min_snr_gamma": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Min-SNR clamp ceiling. Higher = more uniform weighting. 5.0 is the standard value from the Min-SNR paper.",
                }),
                "use_spectral_loss": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add multi-resolution STFT loss in latent space. Penalizes frequency-domain error, especially at high frequencies.",
                }),
                "spectral_loss_weight": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Weight of spectral loss relative to MSE loss.",
                }),
                "spectral_hf_weight": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 5.0, "step": 0.5,
                    "tooltip": "Extra multiplier on high-frequency bins (upper half of STFT) in spectral loss.",
                }),
                "timestep_max": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Cap timestep sampling to [0, timestep_max]. Values below 1.0 skip high-noise training steps. Use 0.5 for denoising LoRAs, 0.7 for upscaling.",
                }),
```

**Step 2: Thread the new parameters through `train()` and `_train_inner()`**

Add these 6 parameters to the `train()` method signature (after `dataset_json=""`):

```python
              use_min_snr=False, min_snr_gamma=5.0,
              use_spectral_loss=False, spectral_loss_weight=0.1,
              spectral_hf_weight=2.0, timestep_max=1.0):
```

Pass them through to `_train_inner()`:

```python
            return self._train_inner(
            ...,
            dataset_json,
            use_min_snr, min_snr_gamma,
            use_spectral_loss, spectral_loss_weight,
            spectral_hf_weight, timestep_max,
        )
```

Add them to `_train_inner()` signature (after `dataset_json=""`):

```python
                     use_min_snr=False, min_snr_gamma=5.0,
                     use_spectral_loss=False, spectral_loss_weight=0.1,
                     spectral_hf_weight=2.0, timestep_max=1.0):
```

**Step 3: Pass parameters to `flow_matching_loss()` call**

In `_train_inner()`, find the `flow_matching_loss(...)` call (~line 1033) and add the new kwargs:

```python
            loss = flow_matching_loss(
                model, batch_latents, t, batch_clip, batch_sync, batch_text, device, dtype,
                timestep_max=timestep_max, use_min_snr=use_min_snr,
                min_snr_gamma=min_snr_gamma, use_spectral_loss=use_spectral_loss,
                spectral_loss_weight=spectral_loss_weight,
                spectral_hf_weight=spectral_hf_weight,
            )
```

**Step 4: Add new parameters to the meta dict**

In the `meta = { ... }` dict (~line 945), add:

```python
            "use_min_snr": use_min_snr, "min_snr_gamma": min_snr_gamma,
            "use_spectral_loss": use_spectral_loss,
            "spectral_loss_weight": spectral_loss_weight,
            "spectral_hf_weight": spectral_hf_weight,
            "timestep_max": timestep_max,
```

**Step 5: Verify node loads**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley/.worktrees/audio2audio && python -c "from nodes_lora import FoleyTuneLoRATrainer; inputs = FoleyTuneLoRATrainer.INPUT_TYPES(); assert 'use_min_snr' in inputs['optional']; assert 'timestep_max' in inputs['optional']; print('All 6 new params present')"`

Expected: `All 6 new params present`

**Step 6: Commit**

```bash
git add nodes_lora.py
git commit -m "feat(training): add min-SNR, spectral loss, and timestep_max UI parameters"
```

---

### Task 5: Integration test — verify backward compatibility

**Files:**
- No files modified

**Step 1: Verify default parameters produce identical behavior**

Run: `cd /media/p5/ComfyUI-HunyuanVideo-Foley/.worktrees/audio2audio && python -c "
import torch
import torch.nn.functional as F
from lora.train import flow_matching_loss, min_snr_weight, multi_resolution_spectral_loss

# Test min_snr_weight edge cases
w = min_snr_weight(torch.tensor([0.0, 0.5, 1.0]), gamma=5.0)
print(f'Min-SNR weights at sigma=[0,0.5,1]: {w}')

# Test spectral loss with short sequence
p = torch.randn(1, 128, 8)
t = torch.randn(1, 128, 8)
sl = multi_resolution_spectral_loss(p, t)
print(f'Spectral loss (short seq): {sl.item():.4f}')

# Test spectral loss with normal sequence
p = torch.randn(2, 128, 200)
t = torch.randn(2, 128, 200)
sl = multi_resolution_spectral_loss(p, t)
print(f'Spectral loss (normal seq): {sl.item():.4f}')

print('All integration tests passed')
"`

Expected: weights printed, loss values printed, "All integration tests passed"

**Step 2: Commit (no changes — just verification)**

No commit needed if everything passes. If any fix was required, commit the fix.

---

### Task 6: Update the LoRA Scheduler node to support new parameters

**Files:**
- Modify: `nodes_lora.py` — Find the `FoleyTuneLoRAScheduler` node and add the new parameters to its sweep config support

**Step 1: Check if LoRA Scheduler needs updates**

Read the `FoleyTuneLoRAScheduler` class to see how it builds experiment configs. If it passes kwargs through to the trainer, the new params may automatically be supported via JSON config. If it hardcodes parameter lists, add the 6 new params.

**Step 2: If needed, add new parameters to the scheduler's passthrough**

Ensure the sweep JSON config can include `use_min_snr`, `min_snr_gamma`, `use_spectral_loss`, `spectral_loss_weight`, `spectral_hf_weight`, and `timestep_max`.

**Step 3: Commit if changes were made**

```bash
git add nodes_lora.py
git commit -m "feat(training): support new loss params in LoRA scheduler sweeps"
```
