"""Foley Audio Dataset Pipeline — chainable in-memory preprocessing nodes.

Typical chain:
  FoleyDatasetLoader
      ↓ FOLEY_AUDIO_DATASET
  FoleyDatasetResampler       (optional)
      ↓ FOLEY_AUDIO_DATASET
  FoleyDatasetLUFSNormalizer  (optional)
      ↓ FOLEY_AUDIO_DATASET
  FoleyDatasetCompressor      (optional)
      ↓ FOLEY_AUDIO_DATASET
  FoleyDatasetHfSmoother      (optional)
      ↓ FOLEY_AUDIO_DATASET
  FoleyDatasetAugmenter       (optional)
      ↓ FOLEY_AUDIO_DATASET
  FoleyDatasetInspector       (optional)
      ↓ FOLEY_AUDIO_DATASET  +  STRING report
  FoleyDatasetSaver           (optional)
      ↓ STRING report
  FoleyDatasetItemExtractor   → AUDIO (bridges to standard nodes)
"""

from pathlib import Path

import numpy as np
import torch
import torchaudio

FOLEY_AUDIO_DATASET = "FOLEY_AUDIO_DATASET"
FOLEY_DS_CATEGORY = "audio/HunyuanFoley/Dataset"
FOLEY_AUDIO_CATEGORY = "audio/HunyuanFoley/Audio"

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aac", ".m4a", ".aiff", ".aif"}
_SOUNDFILE_EXTS = {".wav", ".flac", ".ogg"}


def _load_audio(path: Path):
    """Load audio file. Uses soundfile for WAV/FLAC/OGG to avoid FFmpeg issues."""
    if path.suffix.lower() in _SOUNDFILE_EXTS:
        import soundfile as sf
        wav_np, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [L, C]
        wav = torch.from_numpy(wav_np).T.unsqueeze(0)  # [1, C, L]
    else:
        wav, sr = torchaudio.load(str(path))  # [C, L]
        wav = wav.unsqueeze(0).float()  # [1, C, L]
    return wav, sr


# ─── Node 1: Dataset Loader ──────────────────────────────────────────────────

class FoleyDatasetLoader:
    """Load all audio files in a folder into an in-memory FOLEY_AUDIO_DATASET."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to folder containing audio files. Searched recursively.",
                }),
            }
        }

    RETURN_TYPES = (FOLEY_AUDIO_DATASET,)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "load"
    CATEGORY = FOLEY_DS_CATEGORY
    DESCRIPTION = "Load all audio files from a folder into memory as a FOLEY_AUDIO_DATASET."

    def load(self, folder: str):
        folder = Path(folder.strip())
        if not folder.exists():
            raise FileNotFoundError(f"[FoleyDatasetLoader] Folder not found: {folder}")

        files = [f for f in folder.rglob("*") if f.suffix.lower() in _AUDIO_EXTS]
        if not files:
            raise RuntimeError(f"[FoleyDatasetLoader] No audio files found in {folder}")

        dataset = []
        for f in sorted(files):
            try:
                wav, sr = _load_audio(f)
                dataset.append({"waveform": wav, "sample_rate": sr, "name": f.stem})
            except Exception as e:
                print(f"[FoleyDatasetLoader] Skipping {f.name}: {e}", flush=True)

        print(f"[FoleyDatasetLoader] Loaded {len(dataset)} clips from {folder}", flush=True)
        return (dataset,)


# ─── Node 2: Dataset Resampler ───────────────────────────────────────────────

class FoleyDatasetResampler:
    """Resample all clips in a dataset to a target sample rate using soxr VHQ."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEY_AUDIO_DATASET,),
                "target_sr": ("INT", {
                    "default": 48000, "min": 8000, "max": 192000,
                    "tooltip": "Target sample rate. 48000 for Foley (DAC codec).",
                }),
            }
        }

    RETURN_TYPES = (FOLEY_AUDIO_DATASET,)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "resample"
    CATEGORY = FOLEY_DS_CATEGORY
    DESCRIPTION = "Resample all clips to target_sr using soxr VHQ. Skips clips already at target rate."

    def resample(self, dataset, target_sr: int):
        import soxr

        out = []
        changed = 0
        for item in dataset:
            sr = item["sample_rate"]
            if sr == target_sr:
                out.append(item)
                continue

            wav = item["waveform"][0]  # [C, L]
            wav_np = wav.permute(1, 0).double().numpy()  # [L, C]
            wav_rs = soxr.resample(wav_np, sr, target_sr, quality="VHQ")
            wav_t = torch.from_numpy(wav_rs).float().permute(1, 0).unsqueeze(0)  # [1, C, L]
            new_item = dict(item)  # preserve origin_name and any extra keys
            new_item["waveform"] = wav_t
            new_item["sample_rate"] = target_sr
            out.append(new_item)
            changed += 1

        print(f"[FoleyDatasetResampler] {changed}/{len(dataset)} clips resampled -> {target_sr} Hz", flush=True)
        return (out,)


# ─── Node 3: Dataset LUFS Normalizer ─────────────────────────────────────────

class FoleyDatasetLUFSNormalizer:
    """Normalize each clip to a target integrated LUFS level + true peak limit."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEY_AUDIO_DATASET,),
                "target_lufs": ("FLOAT", {
                    "default": -23.0, "min": -40.0, "max": -6.0, "step": 0.5,
                    "tooltip": "Target integrated loudness in LUFS. -23 is EBU R128 standard.",
                }),
                "true_peak_dbtp": ("FLOAT", {
                    "default": -1.0, "min": -6.0, "max": 0.0, "step": 0.5,
                    "tooltip": "True peak ceiling in dBTP. Applied after LUFS gain.",
                }),
            }
        }

    RETURN_TYPES = (FOLEY_AUDIO_DATASET,)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "normalize"
    CATEGORY = FOLEY_DS_CATEGORY
    DESCRIPTION = (
        "Normalize each clip to target_lufs (BS.1770-4) then apply a true peak ceiling. "
        "Skips clips that are too short for LUFS measurement (< 0.4 s)."
    )

    def normalize(self, dataset, target_lufs: float, true_peak_dbtp: float):
        import pyloudnorm as pyln

        tp_linear = 10.0 ** (true_peak_dbtp / 20.0)
        out = []
        skipped = 0

        for item in dataset:
            wav = item["waveform"][0]  # [C, L]
            sr = item["sample_rate"]

            wav_np = wav.permute(1, 0).double().numpy()  # [L, C]
            if wav_np.shape[1] == 1:
                wav_np = wav_np[:, 0]  # [L] mono

            meter = pyln.Meter(sr)
            try:
                loudness = meter.integrated_loudness(wav_np)
            except Exception:
                skipped += 1
                out.append(item)
                continue

            if not np.isfinite(loudness):
                skipped += 1
                out.append(item)
                continue

            gain_db = target_lufs - loudness
            gain_linear = 10.0 ** (gain_db / 20.0)

            wav_norm = wav * gain_linear

            peak = wav_norm.abs().max().item()
            if peak > tp_linear:
                wav_norm = wav_norm * (tp_linear / peak)

            new_item = dict(item)  # preserve origin_name and any extra keys
            new_item["waveform"] = wav_norm.unsqueeze(0)
            out.append(new_item)

        print(
            f"[FoleyDatasetLUFSNormalizer] {len(dataset) - skipped}/{len(dataset)} clips normalized  "
            f"target={target_lufs} LUFS  TP={true_peak_dbtp} dBTP  skipped={skipped}",
            flush=True,
        )
        return (out,)
