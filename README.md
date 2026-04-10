# 🎵 ComfyUI – HunyuanVideo‑Foley Video to Audio

A tidy set of nodes for **Tencent HunyuanVideo‑Foley** that runs on modest GPUs and scales up nicely.

## ⚡ Optimized Models Available

**Pre-converted safetensors models with fp16 and fp8 variants are available for faster loading and reduced VRAM usage.** The fp8 models enable operation under 8GB VRAM, and with block swap, you can run under 4GB VRAM. [See model files section](#-where-to-put-the-model-files) for download links and file details.

![Workflow Diagram](./example_workflows/HunyuanVideoFoleyExample.png)

## 🚀 Node overview (start here)

* **Hunyuan‑Foley Model Loader** – loads the main model. Two simple knobs:

  * **Precision**: runtime math quality (bf16/fp16/fp32).
  * **FP8 Quantization** (weight‑only): lowers VRAM usage < 12GB. *Turn this on if you're GPU‑poor.*
* **Hunyuan‑Foley Dependencies Loader** – loads DAC‑VAE, SigLIP2, Synchformer, and CLAP.
* **Hunyuan‑Foley Sampler** – makes the audio. Images are **optional** (works great as **Text→Audio**). Supports **negative prompt** and **batching**.
* **Hunyuan‑Foley Torch Compile** (optional) – uses `torch.compile` for speed. First run compiles; repeats are **\~30% faster**.
* **Hunyuan‑Foley BlockSwap Settings** (optional) – enables under 4GB VRAM operation by offloading transformer blocks to CPU.

## ⚡ Quick start

1. Drop **Model Loader → Dependencies Loader → (optional) Torch Compile → Sampler**.
2. For **Text→Audio**, leave the image input empty. For **Video→Audio**, connect an image sequence and set `frame_rate`.
3. Tweak **Prompt** and **Negative Prompt**. Leave sampler on **Euler**, `CFG≈4.5`, `Steps≈50`.
4. Press **Queue** and preview the audio.

## 📁 Where to put the model files


**Optimized safetensors files available at:**
[https://huggingface.co/phazei/HunyuanVideo-Foley](https://huggingface.co/phazei/HunyuanVideo-Foley) (converted safetensors with fp16 and fp8 variants)

I couldn't tell any difference between the quality with fp8 and fp16, so I'd suggest the 8.  For those on a 3090 and lower, torch compile will only work with the e5m2 file.

_Be sure to **set quantization on the loader node to auto or fp8** if using an fp8 model or it will be upcast to fp16 in memory_

**Converted safetensors files:**
```
hunyuanvideo_foley.safetensors             # ~10.3 GB  main model (fp16)
hunyuanvideo_foley_fp8_e4m3fn.safetensors  # ~5.34 GB  main model (fp8)
hunyuanvideo_foley_fp8_e5m2.safetensors    # ~5.34 GB  main model (fp8)
synchformer_state_dict_fp16.safetensors    # ~475 MB   sync encoder (fp16)
vae_128d_48k_fp16.safetensors              # ~743 MB   DAC‑VAE (fp16)
```

Place them in **`ComfyUI/models/foley/`**:

**Original files:**
Download from Hugging Face:
[https://huggingface.co/tencent/HunyuanVideo-Foley/tree/main](https://huggingface.co/tencent/HunyuanVideo-Foley/tree/main) (original PyTorch files)


```
hunyuanvideo_foley.pth         # ~10.3 GB  main model
synchformer_state_dict.pth     # ~0.95 GB  sync encoder
vae_128d_48k.pth               # ~1.49 GB  DAC‑VAE
```

> Tested with **PyTorch 2.7 and 2.8**.

## ⚙️ The Model Loader dropdowns

* **Precision** = how carefully the math runs. `bf16`/`fp16` are fast and standard; `fp32` is heaviest. Pick `bf16` (default) or `fp16` on 30‑series GPUs if you prefer.
* **FP8 Quantization** = store big Linear weights in **FP8** to save memory. Compute still runs in `Precision`, so sound quality holds.  (Must be selected for fp8 safetensors)

  * **`auto`** tries to match the checkpoint or uses a safe default.
  * Expect **less VRAM**, not more speed.

## 💾 Memory & speed at a glance

* Typical 5s / 50 steps on a 24 GB card:

  * Baseline: \~10–12 GB
  * With ping‑pong offloading (built‑in): \~9–10 GB
  * **With FP8 quant**: subtract another **\~4+ GB** (**under 8GB VRAM**)
  * **With Block Swap**: **under 4GB VRAM** It's slower the higher the swap, up to 60s for 5s, but it'll fit!
  * **Torch Compile**: after the first compile, runs are **\~30% faster**
* **Under‑12 GB recipe:** set **FP8 Quant** on, keep **batch\_size=1**, steps ≤ **50**. That's it.

## 🔄 Batching

* `batch_size` generates multiple variations at once. VRAM scales roughly with batch size.
* Use **Select Audio From Batch** to pick the clip you like.

## 💡 Tips & fixes

* If you OOM, drop `batch_size`, reduce `steps`, or enable **force\_offload** in the sampler.

## 🎛️ LoRA Training

Six new nodes for fine-tuning HunyuanVideo-Foley with LoRA adapters.

### New Nodes

* **Foley Feature Extractor** – Caches SigLIP2/Synchformer/CLAP features + audio to `.npz` files for training.
* **Foley LoRA Trainer** – Trains a LoRA adapter via flow matching. Supports logit-normal/curriculum timestep sampling, LoRA+, PiSSA init, rsLoRA, gradient accumulation, and periodic eval samples.
* **Foley LoRA Loader** – Loads a trained adapter into the model for inference, with adjustable strength.
* **Foley LoRA Scheduler** – Runs multiple training experiments from a JSON sweep config. Produces comparison charts and supports resume.
* **Foley LoRA Evaluator** – Generates audio from multiple adapters and computes spectral metrics (HF energy, centroid, flatness, temporal variance) for comparison.
* **Foley VAE Roundtrip** – Diagnostic node that encodes/decodes audio through DAC to reveal the codec quality ceiling.

### Quick Start

1. **Prepare data:** Use **Foley Feature Extractor** to cache features for each training clip (video frames + audio + text prompt). Output goes to a directory of `.npz` + `.wav` files.
2. **Train:** Connect **Model Loader → Dependencies Loader → Foley LoRA Trainer**. Point `data_dir` at your cached features and set `output_dir`. Start with defaults (`rank=64`, `lr=1e-4`, `steps=3000`).
3. **Inference:** Use **Foley LoRA Loader** to load `adapter_final.pt` into the model, then connect to the standard **Sampler**.

### Dataset Preparation

Each training clip needs a matching `.npz` and audio file with the same stem:
```
my_dataset/
  clip_001.npz    # features from Feature Extractor
  clip_001.wav    # paired audio
  clip_002.npz
  clip_002.wav
```

All clips should have the same duration (set at extraction time) to avoid batching issues.

### Sweep JSON Format

```json
{
  "name": "my_sweep",
  "data_dir": "/path/to/dataset",
  "output_root": "/path/to/output",
  "base": {"steps": 2000, "rank": 64},
  "experiments": [
    {"id": "lr_1e4", "lr": 1e-4},
    {"id": "lr_5e5", "lr": 5e-5},
    {"id": "rank32", "rank": 32, "alpha": 32.0}
  ]
}
```

### Eval JSON Format

```json
{
  "name": "compare_adapters",
  "data_dir": "/path/to/dataset",
  "output_dir": "/path/to/eval_output",
  "steps": 25,
  "adapters": [
    {"id": "baseline", "path": null},
    {"id": "lr_1e4", "path": "/path/to/lr_1e4/adapter_final.pt"},
    {"id": "rank32", "path": "/path/to/rank32/adapter_final.pt"}
  ]
}
```

### Hyperparameter Recommendations

| Dataset Size | Rank | Steps | LR | Target |
|---|---|---|---|---|
| 1-5 clips | 16-32 | 1000-2000 | 5e-5 | audio_attn |
| 5-20 clips | 32-64 | 2000-4000 | 1e-4 | audio_cross |
| 20+ clips | 64-128 | 3000-6000 | 1e-4 | all_attn_mlp |

## 🙏 Credits

* Model & weights: **Tencent HunyuanVideo‑Foley**.
* ComfyUI and community for the scaffolding.
* This repo adds VRAM‑friendly loading, **FP8** weight‑only option, **block swap** for ultra-low VRAM, and an optional **torch.compile** speed path.

