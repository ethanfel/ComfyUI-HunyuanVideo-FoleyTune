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


# ─── Node 4: Dataset Compressor ──────────────────────────────────────────────

class FoleyDatasetCompressor:
    """Apply mild parallel compression to reduce within-clip dynamic range.

    Uses pedalboard.Compressor. Parallel (New York) style:
    blends compressed signal with dry so transients are preserved while
    the dynamic range is gently tightened. Apply after LUFS normalization.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEY_AUDIO_DATASET,),
                "threshold_db": ("FLOAT", {
                    "default": -18.0, "min": -40.0, "max": -6.0, "step": 1.0,
                    "tooltip": "Compression kicks in above this level. -18 dB is safe after LUFS normalization.",
                }),
                "ratio": ("FLOAT", {
                    "default": 2.5, "min": 1.5, "max": 4.0, "step": 0.5,
                    "tooltip": "Compression ratio. 2:1-3:1 is mild; stay below 4:1 to avoid pumping.",
                }),
                "attack_ms": ("FLOAT", {
                    "default": 10.0, "min": 1.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Attack time in ms. Slower attack preserves transients.",
                }),
                "release_ms": ("FLOAT", {
                    "default": 100.0, "min": 20.0, "max": 500.0, "step": 10.0,
                    "tooltip": "Release time in ms.",
                }),
                "mix": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Parallel blend: 0.0 = dry only, 1.0 = fully compressed. 0.3-0.5 is typical.",
                }),
            }
        }

    RETURN_TYPES = (FOLEY_AUDIO_DATASET,)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "compress"
    CATEGORY = FOLEY_DS_CATEGORY
    DESCRIPTION = (
        "Mild parallel compression to reduce within-clip dynamic range. "
        "Blends compressed signal with dry at the given mix ratio."
    )

    def compress(self, dataset, threshold_db: float, ratio: float,
                 attack_ms: float, release_ms: float, mix: float):
        from pedalboard import Compressor, Pedalboard

        board = Pedalboard([Compressor(
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )])

        out = []
        for item in dataset:
            wav = item["waveform"][0]  # [C, L]
            sr = item["sample_rate"]

            wav_np = wav.float().numpy()  # [C, L]
            compressed = board(wav_np, sr)  # [C, L]
            mixed = (1.0 - mix) * wav_np + mix * compressed
            wav_out = torch.from_numpy(mixed).unsqueeze(0)  # [1, C, L]
            new_item = dict(item)  # preserve origin_name and any extra keys
            new_item["waveform"] = wav_out
            out.append(new_item)

        print(
            f"[FoleyDatasetCompressor] {len(out)} clips compressed  "
            f"thr={threshold_db}dB  ratio={ratio}:1  mix={mix:.0%}",
            flush=True,
        )
        return (out,)


# ─── Node 5: Dataset Inspector ───────────────────────────────────────────────

def _check_hf_shelf(wav: torch.Tensor, sr: int) -> bool:
    """Return True if clip looks codec-compressed (hard HF shelf above 15 kHz)."""
    if sr < 32000:
        return False

    n_fft = 2048
    hop = 512
    mono = wav[0].mean(0)  # [L]
    window = torch.hann_window(n_fft, device=mono.device)
    stft = torch.stft(mono, n_fft, hop, n_fft, window, return_complex=True)
    mag_sq = stft.abs().pow(2).mean(-1)  # [n_freqs]

    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1, device=mono.device)
    band_lo = (freqs >= 1000) & (freqs < 5000)
    band_hi = (freqs >= 15000) & (freqs < 20000)

    if band_hi.sum() == 0:
        return False

    energy_lo = mag_sq[band_lo].mean().clamp(min=1e-12)
    energy_hi = mag_sq[band_hi].mean().clamp(min=1e-12)
    ratio_db = 10.0 * torch.log10(energy_lo / energy_hi).item()
    return ratio_db > 40.0


def _estimate_snr(wav: torch.Tensor) -> float:
    """Rough SNR estimate: ratio of 95th-percentile frame RMS to 5th-percentile."""
    mono = wav[0].mean(0)  # [L]
    if mono.shape[0] < 2048:
        return 60.0
    frames = mono.unfold(0, 2048, 512)  # [N, 2048]
    rms = frames.pow(2).mean(-1).sqrt()  # [N]
    p95 = torch.quantile(rms, 0.95).item()
    p05 = torch.quantile(rms, 0.05).clamp(min=1e-8).item()
    return 20.0 * np.log10(p95 / p05 + 1e-8)


class FoleyDatasetInspector:
    """Analyze each clip for quality issues and optionally filter out flagged clips."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEY_AUDIO_DATASET,),
                "skip_rejected": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If True, flagged clips are removed from output. "
                               "If False, all clips pass through but report still lists issues.",
                }),
                "min_snr_db": ("FLOAT", {
                    "default": 15.0, "min": 0.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Clips with estimated SNR below this value are flagged.",
                }),
                "check_codec_artifacts": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Flag clips with a hard HF shelf above 15 kHz (MP3/codec artifact).",
                }),
                "max_silence_fraction": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Flag clips where more than this fraction of frames are near-silent.",
                }),
            }
        }

    RETURN_TYPES = (FOLEY_AUDIO_DATASET, "STRING")
    RETURN_NAMES = ("dataset", "report")
    FUNCTION = "inspect"
    CATEGORY = FOLEY_DS_CATEGORY
    DESCRIPTION = (
        "Analyze each clip for clipping, low SNR, codec artifacts, and silence. "
        "Outputs a filtered FOLEY_AUDIO_DATASET and a text report."
    )

    def inspect(self, dataset, skip_rejected: bool, min_snr_db: float,
                check_codec_artifacts: bool, max_silence_fraction: float = 0.5):
        clean = []
        flagged = []
        lines = ["Foley Dataset Inspector Report", "=" * 40]

        for item in dataset:
            wav = item["waveform"]
            sr = item["sample_rate"]
            name = item["name"]
            issues = []

            peak = wav.abs().max().item()
            if peak > 0.99:
                issues.append(f"clipping (peak={peak:.3f})")

            snr = _estimate_snr(wav)
            if snr < min_snr_db:
                issues.append(f"low SNR ({snr:.1f} dB < {min_snr_db} dB)")

            if check_codec_artifacts and _check_hf_shelf(wav, sr):
                issues.append("codec artifact (HF shelf > 15 kHz)")

            if max_silence_fraction > 0:
                mono = wav[0].mean(0)
                if mono.shape[0] >= 2048:
                    frames = mono.unfold(0, 2048, 512)
                    rms = frames.pow(2).mean(-1).sqrt()
                    silent_frac = (rms < 1e-3).float().mean().item()
                    if silent_frac > max_silence_fraction:
                        issues.append(f"mostly silent ({silent_frac:.0%} < -60 dBFS)")

            if issues:
                flagged.append(name)
                lines.append(f"  FLAGGED  {name}: {', '.join(issues)}")
                if not skip_rejected:
                    clean.append(item)
            else:
                clean.append(item)
                lines.append(f"  OK       {name}")

        lines.append("=" * 40)
        lines.append(
            f"Total: {len(dataset)}  Clean: {len(clean)}  Flagged: {len(flagged)}"
            + (" (removed)" if skip_rejected else " (kept)")
        )
        report = "\n".join(lines)
        print(f"[FoleyDatasetInspector]\n{report}", flush=True)
        return (clean, report)


# ─── Node 6: Dataset HF Smoother (batch) ─────────────────────────────────────

class FoleyDatasetHfSmoother:
    """Apply soft high-frequency attenuation to every clip in a dataset.

    Blends a low-pass filtered copy with the original. Default cutoff is 16 kHz
    (vs 12 kHz in SelVA) because DAC's neural codec handles HF content better
    than BigVGAN's mel-spectrogram vocoder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEY_AUDIO_DATASET,),
                "cutoff_hz": ("FLOAT", {
                    "default": 16000.0, "min": 2000.0, "max": 22000.0, "step": 500.0,
                    "tooltip": "Low-pass cutoff. 16 kHz is gentle for DAC; lower = more aggressive.",
                }),
                "blend": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = original, 1 = fully filtered. 0.5 is a good starting point for DAC.",
                }),
            }
        }

    RETURN_TYPES = (FOLEY_AUDIO_DATASET,)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "process"
    CATEGORY = FOLEY_DS_CATEGORY
    DESCRIPTION = (
        "Soft HF attenuation for every clip. Blends a low-pass filtered copy "
        "with the original. Less aggressive defaults than SelVA because DAC "
        "handles high frequencies better than BigVGAN."
    )

    def process(self, dataset, cutoff_hz: float, blend: float):
        import torchaudio.functional as AF

        out = []
        for item in dataset:
            wav = item["waveform"].float()  # [1, C, L]
            sr = item["sample_rate"]

            filtered = AF.lowpass_biquad(wav, sr, cutoff_hz)
            result = blend * filtered + (1.0 - blend) * wav

            # Preserve RMS level
            rms_in = wav.pow(2).mean().sqrt().clamp(min=1e-8)
            rms_out = result.pow(2).mean().sqrt().clamp(min=1e-8)
            result = result * (rms_in / rms_out)

            peak = result.abs().max()
            if peak > 1.0:
                result = result / peak

            new_item = dict(item)
            new_item["waveform"] = result
            out.append(new_item)

        print(f"[FoleyDatasetHfSmoother] {len(out)} clips processed  "
              f"cutoff={cutoff_hz:.0f}Hz  blend={blend:.2f}", flush=True)
        return (out,)


# ─── Node 7: Dataset Augmenter ───────────────────────────────────────────────

class FoleyDatasetAugmenter:
    """Create augmented variants of each clip to expand a small dataset.

    Supports gain variation (always available) and optionally pitch shift
    and time stretch via audiomentations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEY_AUDIO_DATASET,),
                "variants_per_clip": ("INT", {
                    "default": 2, "min": 1, "max": 20,
                    "tooltip": "Number of augmented copies per original clip.",
                }),
                "gain_range_db": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Random gain +/-dB applied to each variant.",
                }),
                "seed": ("INT", {"default": 42}),
            },
            "optional": {
                "pitch_range_semitones": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 4.0, "step": 0.25,
                    "tooltip": "Random pitch shift +/-semitones. Requires audiomentations. 0 = disabled.",
                }),
                "time_stretch_range": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.3, "step": 0.05,
                    "tooltip": "Random time stretch +/-fraction (0.1 = 90%-110% speed). "
                               "Requires audiomentations. 0 = disabled.",
                }),
                "keep_originals": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include the original unaugmented clips in the output.",
                }),
            },
        }

    RETURN_TYPES = (FOLEY_AUDIO_DATASET,)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "augment"
    CATEGORY = FOLEY_DS_CATEGORY
    DESCRIPTION = (
        "Create augmented variants of each clip (gain, pitch, time stretch) "
        "to expand small training datasets."
    )

    def augment(self, dataset, variants_per_clip: int, gain_range_db: float,
                seed: int, pitch_range_semitones: float = 0.0,
                time_stretch_range: float = 0.0, keep_originals: bool = True):
        rng = np.random.RandomState(seed)

        use_am = False
        am_compose = None
        needs_am = pitch_range_semitones > 0 or time_stretch_range > 0
        if needs_am:
            try:
                import audiomentations as am
                transforms = []
                if pitch_range_semitones > 0:
                    transforms.append(am.PitchShift(
                        min_semitones=-pitch_range_semitones,
                        max_semitones=pitch_range_semitones,
                        p=0.5,
                    ))
                if time_stretch_range > 0:
                    transforms.append(am.TimeStretch(
                        min_rate=1.0 - time_stretch_range,
                        max_rate=1.0 + time_stretch_range,
                        leave_length_unchanged=True,
                        p=0.5,
                    ))
                am_compose = am.Compose(transforms)
                use_am = True
            except ImportError:
                print("[FoleyDatasetAugmenter] audiomentations not installed — "
                      "pitch_shift and time_stretch disabled. "
                      "Install: pip install audiomentations", flush=True)

        out = []
        if keep_originals:
            out.extend(dataset)

        for item in dataset:
            wav = item["waveform"]  # [1, C, L]
            sr = item["sample_rate"]
            name = item["name"]

            for v in range(variants_per_clip):
                gain_db = rng.uniform(-gain_range_db, gain_range_db) if gain_range_db > 0 else 0.0
                gain_lin = 10.0 ** (gain_db / 20.0)
                wav_aug = wav * gain_lin

                if use_am and am_compose is not None:
                    wav_np = wav_aug[0].numpy()  # [C, L]
                    if wav_np.shape[0] == 1:
                        wav_np = wav_np[0]  # [L] mono
                    wav_np = am_compose(samples=wav_np, sample_rate=sr)
                    if wav_np.ndim == 1:
                        wav_np = wav_np[np.newaxis, :]
                    wav_aug = torch.from_numpy(wav_np).unsqueeze(0)  # [1, C, L]

                peak = wav_aug.abs().max()
                if peak > 1.0:
                    wav_aug = wav_aug / peak

                out.append({
                    "waveform": wav_aug,
                    "sample_rate": sr,
                    "name": f"{name}_aug{v:02d}",
                    "origin_name": name,
                })

        print(f"[FoleyDatasetAugmenter] {len(dataset)} originals -> {len(out)} total clips  "
              f"gain=+/-{gain_range_db:.1f}dB"
              + (f"  pitch=+/-{pitch_range_semitones:.1f}st" if pitch_range_semitones > 0 else "")
              + (f"  stretch=+/-{time_stretch_range:.0%}" if time_stretch_range > 0 else ""),
              flush=True)
        return (out,)


# ─── Node 8: Dataset Saver ───────────────────────────────────────────────────

class FoleyDatasetSaver:
    """Save all clips in a FOLEY_AUDIO_DATASET to disk as FLAC files."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEY_AUDIO_DATASET,),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to output folder. Created if it does not exist.",
                }),
            },
            "optional": {
                "npz_source_dir": ("STRING", {
                    "default": "",
                    "tooltip": "If set, copies {name}.npz from this folder alongside each saved FLAC.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = FOLEY_DS_CATEGORY
    DESCRIPTION = (
        "Save every clip in a FOLEY_AUDIO_DATASET to output_dir as 24-bit FLAC. "
        "Optionally copies matching .npz feature files."
    )

    def save(self, dataset, output_dir: str, npz_source_dir: str = ""):
        import shutil
        import soundfile as sf

        out_path = Path(output_dir.strip())
        out_path.mkdir(parents=True, exist_ok=True)

        npz_src = Path(npz_source_dir.strip()) if npz_source_dir.strip() else None

        saved = 0
        npz_copied = 0
        npz_missing = []

        for item in dataset:
            name = item["name"]
            wav = item["waveform"][0]  # [C, L]
            sr = item["sample_rate"]

            wav_np = wav.permute(1, 0).float().numpy()  # [L, C]
            if wav_np.shape[1] == 1:
                wav_np = wav_np[:, 0]  # [L] mono

            flac_path = out_path / f"{name}.flac"
            sf.write(str(flac_path), wav_np, sr, subtype="PCM_24")
            saved += 1

            if npz_src is not None:
                lookup = item.get("origin_name", name)
                npz_path = npz_src / f"{lookup}.npz"
                if npz_path.exists():
                    shutil.copy2(str(npz_path), str(out_path / f"{name}.npz"))
                    npz_copied += 1
                else:
                    npz_missing.append(name)

        lines = [f"[FoleyDatasetSaver] Saved {saved} clips -> {out_path}"]
        if npz_src is not None:
            lines.append(f"  NPZ copied: {npz_copied}  missing: {len(npz_missing)}")
            for n in npz_missing:
                lines.append(f"    MISSING NPZ: {n}")

        report = "\n".join(lines)
        print(report, flush=True)
        return (report,)


# ─── Node 9: Dataset Item Extractor ──────────────────────────────────────────

class FoleyDatasetItemExtractor:
    """Extract a single AUDIO item from a FOLEY_AUDIO_DATASET by index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEY_AUDIO_DATASET,),
                "index": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "0-based index. Wraps around if index >= dataset length.",
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "INT")
    RETURN_NAMES = ("audio", "name", "total")
    FUNCTION = "extract"
    CATEGORY = FOLEY_DS_CATEGORY
    DESCRIPTION = (
        "Extract one clip from a FOLEY_AUDIO_DATASET by index. "
        "Returns standard AUDIO (compatible with all audio nodes), "
        "the clip name, and the total dataset length."
    )

    def extract(self, dataset, index: int):
        if not dataset:
            raise RuntimeError("[FoleyDatasetItemExtractor] Dataset is empty.")
        idx = index % len(dataset)
        item = dataset[idx]
        audio = {"waveform": item["waveform"], "sample_rate": item["sample_rate"]}
        print(
            f"[FoleyDatasetItemExtractor] [{idx}/{len(dataset)-1}] {item['name']}  "
            f"sr={item['sample_rate']}  shape={tuple(item['waveform'].shape)}",
            flush=True,
        )
        return (audio, item["name"], len(dataset))
