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

        if not dataset:
            raise RuntimeError(f"[FoleyDatasetLoader] All {len(files)} files failed to load from {folder}")

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

                new_item = dict(item)
                new_item["waveform"] = wav_aug
                new_item["name"] = f"{name}_aug{v:02d}"
                new_item["origin_name"] = name
                out.append(new_item)

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


# ─── Mel filterbank utility ──────────────────────────────────────────────────

def _mel_filterbank(sr: int, n_fft: int, n_mels: int,
                    fmin: float, fmax: float) -> torch.Tensor:
    """Returns mel filterbank matrix [n_mels, n_fft//2+1]."""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)
    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_freqs)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, mid, hi = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        up = (fft_freqs - lo) / (mid - lo + 1e-12)
        down = (hi - fft_freqs) / (hi - mid + 1e-12)
        fb[m - 1] = np.maximum(0.0, np.minimum(up, down))
    return torch.from_numpy(fb)


# ─── Node 10: Dataset Spectral Matcher (reference-based) ─────────────────────

class FoleyDatasetSpectralMatcher:
    """Adaptive per-band EQ toward a reference audio distribution.

    Unlike SelVA's hardcoded VAE stats, this computes target mel-band means from
    a reference directory of audio files. Use DAC roundtrip outputs as reference
    to match what the codec reproduces best.

    Process:
    1. Compute mean log-mel profile across all reference files
    2. For each clip: compute its log-mel profile, derive per-band gain difference
    3. Apply smooth frequency-domain correction (multiplicative in linear space)
    4. Preserve original RMS level
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEY_AUDIO_DATASET,),
                "reference_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Path to folder of reference audio files. Use DAC roundtrip outputs "
                               "for best results. The spectral profile of these files becomes the target.",
                }),
                "strength": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = no correction, 1 = full match to reference distribution.",
                }),
                "max_gain_db": ("FLOAT", {
                    "default": 12.0, "min": 1.0, "max": 30.0, "step": 1.0,
                    "tooltip": "Clamps per-band gain to +/-dB. Prevents extreme boosts.",
                }),
            },
            "optional": {
                "n_mels": ("INT", {
                    "default": 128, "min": 40, "max": 256,
                    "tooltip": "Number of mel bands for analysis. 128 is good for 48kHz.",
                }),
            },
        }

    RETURN_TYPES = (FOLEY_AUDIO_DATASET,)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "process"
    CATEGORY = FOLEY_DS_CATEGORY
    DESCRIPTION = (
        "Adaptive per-band EQ toward a reference audio distribution. "
        "Computes target mel-band means from a reference directory, then applies "
        "smooth frequency-domain correction to each clip in the dataset. "
        "Use DAC roundtrip outputs as reference for best LoRA training quality."
    )

    def _compute_reference_profile(self, ref_dir: Path, sr_tgt: int,
                                   n_fft: int, hop: int, n_mels: int) -> torch.Tensor:
        """Compute mean log-mel profile across all reference audio files."""
        ref_files = [f for f in ref_dir.rglob("*") if f.suffix.lower() in _AUDIO_EXTS]
        if not ref_files:
            raise FileNotFoundError(
                f"[FoleyDatasetSpectralMatcher] No audio files found in reference dir: {ref_dir}"
            )

        import torchaudio.functional as AF

        fb = _mel_filterbank(sr_tgt, n_fft, n_mels, 0, sr_tgt // 2)
        window = torch.hann_window(n_fft)
        all_means = []

        for f in sorted(ref_files):
            try:
                wav, sr = _load_audio(f)
                mono = wav[0].mean(0).float()  # [L]
                if sr != sr_tgt:
                    mono = AF.resample(mono.unsqueeze(0), sr, sr_tgt).squeeze(0)

                stft = torch.stft(mono, n_fft, hop_length=hop, win_length=n_fft,
                                  window=window, center=True, return_complex=True)
                mag = stft.abs()  # [n_freqs, T]
                mel_mag = torch.matmul(fb, mag).clamp(min=1e-5)
                mel_log = torch.log(mel_mag)
                all_means.append(mel_log.mean(dim=-1))  # [n_mels]
            except Exception as e:
                print(f"[FoleyDatasetSpectralMatcher] Skipping ref {f.name}: {e}", flush=True)

        if not all_means:
            raise RuntimeError("[FoleyDatasetSpectralMatcher] Could not process any reference files")

        target_mean = torch.stack(all_means).mean(dim=0)  # [n_mels]
        print(f"[FoleyDatasetSpectralMatcher] Computed reference profile from "
              f"{len(all_means)} files", flush=True)
        return target_mean

    def process(self, dataset, reference_dir: str, strength: float,
                max_gain_db: float, n_mels: int = 128):
        import torchaudio.functional as AF

        ref_dir = Path(reference_dir.strip())
        if not ref_dir.exists():
            raise FileNotFoundError(f"Reference dir not found: {ref_dir}")

        # Use 48kHz as analysis SR (Foley native)
        sr_tgt = 48000
        n_fft = 2048
        hop = 512

        target_mean = self._compute_reference_profile(ref_dir, sr_tgt, n_fft, hop, n_mels)
        fb = _mel_filterbank(sr_tgt, n_fft, n_mels, 0, sr_tgt // 2)
        window = torch.hann_window(n_fft)
        max_log = max_gain_db / 8.6859  # ln scale: 20 * log10(e) ~ 8.686

        out = []
        for item in dataset:
            wav = item["waveform"].float()  # [1, C, L]
            sr_in = item["sample_rate"]

            mono = wav[0].mean(0)  # [L]
            rms_original = mono.pow(2).mean().sqrt().clamp(min=1e-8)
            if sr_in != sr_tgt:
                mono = AF.resample(mono.unsqueeze(0), sr_in, sr_tgt).squeeze(0)

            stft = torch.stft(mono, n_fft, hop_length=hop, win_length=n_fft,
                              window=window, center=True, return_complex=True)
            mag = stft.abs()

            mel_mag = torch.matmul(fb, mag).clamp(min=1e-5)
            mel_log = torch.log(mel_mag)
            current_mean = mel_log.mean(dim=-1)  # [n_mels]

            # Per-mel-band gain (log space)
            mel_gain = (target_mean - current_mean) * strength
            mel_gain = mel_gain.clamp(-max_log, max_log)

            # Map mel gains -> STFT frequency bins
            fb_sum = fb.sum(0).clamp(min=1e-8)
            freq_gain = (mel_gain @ fb) / fb_sum
            linear_gain = torch.exp(freq_gain)

            # Apply gain and reconstruct
            stft_out = stft * linear_gain.unsqueeze(-1)
            wav_out = torch.istft(stft_out, n_fft, hop_length=hop, win_length=n_fft,
                                  window=window, center=True, length=mono.shape[0])

            if sr_in != sr_tgt:
                wav_out = AF.resample(wav_out.unsqueeze(0), sr_tgt, sr_in).squeeze(0)

            # Preserve RMS (use original pre-resample RMS as reference)
            rms_out = wav_out.pow(2).mean().sqrt().clamp(min=1e-8)
            wav_out = wav_out * (rms_original / rms_out)

            peak = wav_out.abs().max()
            if peak > 1.0:
                wav_out = wav_out / peak

            result = wav_out.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
            if wav.shape[1] > 1:
                print(f"[FoleyDatasetSpectralMatcher] Warning: stereo clip '{item['name']}' "
                      f"collapsed to dual-mono (spectral matching operates on mono downmix)", flush=True)
                result = result.expand(-1, wav.shape[1], -1).clone()

            new_item = dict(item)
            new_item["waveform"] = result
            out.append(new_item)

        print(f"[FoleyDatasetSpectralMatcher] {len(out)} clips processed  "
              f"strength={strength}  n_mels={n_mels}  ref={ref_dir.name}", flush=True)
        return (out,)


# ─── Single-audio pre/post-processor nodes ───────────────────────────────────

# ─── Node 11: HF Smoother (single audio) ─────────────────────────────────────

class FoleyHfSmoother:
    """Soft high-frequency attenuation for a single audio clip.

    Blends a low-pass filtered copy with the original. Default cutoff is 16 kHz
    for DAC's 48kHz codec (vs 12 kHz in SelVA for BigVGAN at 44.1kHz).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "cutoff_hz": ("FLOAT", {
                    "default": 16000.0, "min": 2000.0, "max": 22000.0, "step": 500.0,
                    "tooltip": "Low-pass cutoff. 16 kHz is gentle for DAC; lower = more aggressive.",
                }),
                "blend": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = original, 1 = fully filtered.",
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"
    CATEGORY = FOLEY_AUDIO_CATEGORY
    DESCRIPTION = (
        "Blends a low-pass filtered version of the audio with the original to gently "
        "attenuate high-frequency content. Use before feature extraction for training."
    )

    def process(self, audio, cutoff_hz: float, blend: float):
        import torchaudio.functional as AF

        waveform = audio["waveform"].float()  # [1, C, L]
        sr = audio["sample_rate"]

        filtered = AF.lowpass_biquad(waveform, sr, cutoff_hz)
        out = blend * filtered + (1.0 - blend) * waveform

        rms_in = waveform.pow(2).mean().sqrt().clamp(min=1e-8)
        rms_out = out.pow(2).mean().sqrt().clamp(min=1e-8)
        out = out * (rms_in / rms_out)

        peak = out.abs().max()
        if peak > 1.0:
            out = out / peak

        print(f"[FoleyHfSmoother] cutoff={cutoff_hz:.0f} Hz  blend={blend:.2f}  "
              f"rms={rms_in:.4f}->{out.pow(2).mean().sqrt():.4f}", flush=True)

        return ({"waveform": out, "sample_rate": sr},)


# ─── Node 12: Harmonic Exciter ───────────────────────────────────────────────

class FoleyHarmonicExciter:
    """Multi-band harmonic exciter for post-generation enhancement.

    Isolates high-frequency content above a cutoff, applies tanh saturation
    to generate harmonics, then mixes back with the dry signal.

    NOTE: DAC's neural codec has better HF fidelity than BigVGAN, so this
    is less critical for Foley than SelVA. Use subtly (mix 0.05-0.15) or
    skip entirely if DAC output sounds good already.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "cutoff_hz": ("FLOAT", {
                    "default": 4000.0, "min": 500.0, "max": 16000.0, "step": 100.0,
                    "tooltip": "Highpass cutoff. Only content above this is excited. "
                               "4000 Hz for DAC (vs 3000 Hz for BigVGAN).",
                }),
                "drive": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Saturation drive. Higher = more harmonics. 2-3 is subtle.",
                }),
                "mix": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Wet/dry blend. 0.05-0.15 is subtle for DAC. "
                               "Lower than SelVA default since DAC needs less HF compensation.",
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "excite"
    CATEGORY = FOLEY_AUDIO_CATEGORY
    DESCRIPTION = (
        "Multi-band harmonic exciter. Applies tanh saturation to the HF band "
        "to restore harmonics. Less aggressive defaults than SelVA — DAC's neural "
        "codec preserves HF better than BigVGAN."
    )

    def excite(self, audio, cutoff_hz: float, drive: float, mix: float):
        from pedalboard import Pedalboard, HighpassFilter

        wav = audio["waveform"][0]  # [C, T]
        sr = audio["sample_rate"]

        wav_np = wav.float().numpy()  # [C, T]

        board = Pedalboard([HighpassFilter(cutoff_frequency_hz=cutoff_hz)])
        hf = board(wav_np, sr)  # [C, T]

        excited = np.tanh(hf * drive) / max(drive, 1.0)
        mixed = wav_np + mix * excited
        mixed = np.tanh(mixed)

        wav_out = torch.from_numpy(mixed).unsqueeze(0)  # [1, C, T]
        print(f"[FoleyHarmonicExciter] cutoff={cutoff_hz}Hz  drive={drive}  mix={mix:.0%}", flush=True)
        return ({"waveform": wav_out, "sample_rate": sr},)


# ─── Node 13: Output Normalizer ──────────────────────────────────────────────

class FoleyOutputNormalizer:
    """Normalize generated audio to a target LUFS level with true peak limiting.

    Apply as the final node before saving. Uses pyloudnorm (BS.1770-4).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_lufs": ("FLOAT", {
                    "default": -14.0, "min": -40.0, "max": -6.0, "step": 0.5,
                    "tooltip": "Target integrated loudness in LUFS. "
                               "-14 for streaming (Spotify/YouTube), -9 to -7 for masters.",
                }),
                "true_peak_dbtp": ("FLOAT", {
                    "default": -1.0, "min": -6.0, "max": 0.0, "step": 0.5,
                    "tooltip": "True peak ceiling in dBTP applied after LUFS gain.",
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "normalize"
    CATEGORY = FOLEY_AUDIO_CATEGORY
    DESCRIPTION = (
        "Normalize output audio to a target LUFS level (BS.1770-4) with true peak limiting. "
        "Apply as the last node before saving."
    )

    def normalize(self, audio, target_lufs: float, true_peak_dbtp: float):
        import pyloudnorm as pyln

        wav = audio["waveform"][0]  # [C, T]
        sr = audio["sample_rate"]

        tp_linear = 10.0 ** (true_peak_dbtp / 20.0)

        wav_np = wav.permute(1, 0).double().numpy()  # [T, C]
        if wav_np.shape[1] == 1:
            wav_np = wav_np[:, 0]  # [T] mono

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav_np)

        if not np.isfinite(loudness):
            print("[FoleyOutputNormalizer] Could not measure loudness — passing through.", flush=True)
            return (audio,)

        gain_db = target_lufs - loudness
        gain_linear = 10.0 ** (gain_db / 20.0)

        wav_out = wav * gain_linear

        peak = wav_out.abs().max().item()
        if peak > tp_linear:
            wav_out = wav_out * (tp_linear / peak)

        print(
            f"[FoleyOutputNormalizer] {loudness:.1f} LUFS -> {target_lufs} LUFS  "
            f"gain={gain_db:+.1f}dB  TP={true_peak_dbtp}dBTP",
            flush=True,
        )
        return ({"waveform": wav_out.unsqueeze(0), "sample_rate": sr},)


# ─── Node 14: Dataset Browser ───────────────────────────────────────────────

class FoleyDatasetBrowser:
    """Browse a dataset.json file entry by entry using an integer index.

    Accepts three JSON formats:

    1. SelVA format (array of objects with base path, no extension):
       [{"path": "/clips/clip_001", "label": "description"}, ...]

    2. Foley format (array of objects with explicit paths):
       [{"video_path": "/clips/clip_001.mp4", "prompt": "description"}, ...]

    3. Compact format (shared prompt + directories + clip names):
       {
         "prompt": "description",
         "clips_dir": "/path/to/frames",
         "audio_dir": "/path/to/cleaned",
         "features_dir": "/path/to/features",
         "clips": ["clip_001", "clip_002"]
       }
       Only clips_dir is required. audio_dir defaults to clips_dir,
       features_dir defaults to clips_dir/features.
       Audio extension auto-detected (.flac then .wav).

    For formats 1 and 2, base paths derive:
      - video:  path + ".mp4"
      - audio:  path + ".wav" or ".flac" (auto-detected)
      - frames: path           (directory of image sequences)
      - npz:    parent/features/name.npz
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_json": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a dataset.json file. Accepts SelVA, Foley, or compact format.",
                }),
                "index": ("INT", {
                    "default": 0, "min": 0, "max": 99999, "step": 1,
                    "tooltip": "Zero-based index of the entry to browse.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("video_path", "audio_path", "frames_dir", "npz_path", "prompt", "max_index")
    OUTPUT_TOOLTIPS = (
        "path + '.mp4'",
        "path + '.wav'  (or features/name.wav)",
        "path  (image-sequence directory)",
        "features/name.npz  (pre-extracted features)",
        "Text prompt / label for this clip",
        "count - 1 — wire to a primitive INT's max to constrain the index widget",
    )
    FUNCTION = "browse"
    CATEGORY = FOLEY_DS_CATEGORY
    DESCRIPTION = (
        "Reads a dataset.json and exposes one entry at a time. "
        "Accepts SelVA format (path/label), Foley format (video_path/prompt), "
        "or compact format (shared prompt + list of paths). "
        "Wire outputs to VHS_LoadVideoPath and Foley Feature Extractor."
    )

    IS_CHANGED = classmethod(lambda cls, **_: float("nan"))

    def browse(self, dataset_json: str, index: int):
        import json as _json

        p = Path(dataset_json.strip())
        if not p.exists():
            raise FileNotFoundError(f"[FoleyDatasetBrowser] File not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            data = _json.load(f)

        # Parse the three supported formats into a uniform list
        entries = []
        default_prompt = ""
        clips_dir = None
        audio_dir = None
        features_dir = None

        if isinstance(data, dict):
            # Compact format
            default_prompt = data.get("prompt", data.get("label", ""))
            clips_dir = Path(data["clips_dir"]) if "clips_dir" in data else None
            audio_dir = Path(data["audio_dir"]) if "audio_dir" in data else clips_dir
            if "features_dir" in data:
                features_dir = Path(data["features_dir"])
            elif clips_dir:
                features_dir = clips_dir / "features"

            clips = data.get("clips", [])
            for c in clips:
                if isinstance(c, str):
                    base = str(clips_dir / c) if clips_dir else c
                    entries.append({"name": c, "base": base, "prompt": default_prompt})
                elif isinstance(c, dict):
                    name = c.get("name", "")
                    base = c.get("path", c.get("video_path", ""))
                    if not base and name and clips_dir:
                        base = str(clips_dir / name)
                    prompt = c.get("prompt", c.get("label", default_prompt))
                    entries.append({"name": name or Path(base).stem, "base": base, "prompt": prompt})
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    entries.append({"name": Path(item).stem, "base": item, "prompt": ""})
                elif isinstance(item, dict):
                    # SelVA format: path/label  or  Foley format: video_path/prompt
                    base = item.get("path", item.get("video_path", ""))
                    prompt = item.get("label", item.get("prompt", ""))
                    entries.append({"name": Path(base).stem, "base": base, "prompt": prompt})

        if not entries:
            raise ValueError(f"[FoleyDatasetBrowser] No entries found in {p}")

        count = len(entries)
        if index >= count:
            raise IndexError(
                f"[FoleyDatasetBrowser] index {index} out of range "
                f"(dataset has {count} entries, last index is {count - 1})"
            )

        entry = entries[index]
        name = entry["name"]
        base = entry["base"]
        prompt = entry["prompt"]
        p_base = Path(base)

        # Video / frames
        video_path = base + ".mp4" if not p_base.suffix else base
        frames_dir = str(p_base) if not p_base.suffix else str(p_base.with_suffix(""))

        # Audio: use audio_dir if set, otherwise derive from base
        if audio_dir:
            audio_base = audio_dir / name
        else:
            audio_base = p_base

        # Auto-detect .flac or .wav
        audio_path = ""
        for ext in (".flac", ".wav"):
            candidate = audio_base.with_suffix(ext)
            if candidate.exists():
                audio_path = str(candidate)
                break
        if not audio_path:
            audio_path = str(audio_base.with_suffix(".wav"))

        # NPZ: use features_dir if set, otherwise derive from base
        if features_dir:
            npz_path = str(features_dir / name) + ".npz"
        else:
            npz_path = str(p_base.parent / "features" / p_base.stem) + ".npz"

        print(
            f"[FoleyDatasetBrowser] [{index}/{count - 1}]  prompt='{prompt}'  "
            f"base={base}",
            flush=True,
        )

        return (video_path, audio_path, frames_dir, npz_path, prompt, count - 1)


# ─── Node Mappings ───────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "FoleyDatasetLoader": FoleyDatasetLoader,
    "FoleyDatasetResampler": FoleyDatasetResampler,
    "FoleyDatasetLUFSNormalizer": FoleyDatasetLUFSNormalizer,
    "FoleyDatasetCompressor": FoleyDatasetCompressor,
    "FoleyDatasetInspector": FoleyDatasetInspector,
    "FoleyDatasetHfSmoother": FoleyDatasetHfSmoother,
    "FoleyDatasetAugmenter": FoleyDatasetAugmenter,
    "FoleyDatasetSaver": FoleyDatasetSaver,
    "FoleyDatasetItemExtractor": FoleyDatasetItemExtractor,
    "FoleyDatasetSpectralMatcher": FoleyDatasetSpectralMatcher,
    "FoleyHfSmoother": FoleyHfSmoother,
    "FoleyHarmonicExciter": FoleyHarmonicExciter,
    "FoleyOutputNormalizer": FoleyOutputNormalizer,
    "FoleyDatasetBrowser": FoleyDatasetBrowser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoleyDatasetLoader": "Foley Dataset Loader",
    "FoleyDatasetResampler": "Foley Dataset Resampler",
    "FoleyDatasetLUFSNormalizer": "Foley Dataset LUFS Normalizer",
    "FoleyDatasetCompressor": "Foley Dataset Compressor",
    "FoleyDatasetInspector": "Foley Dataset Inspector",
    "FoleyDatasetHfSmoother": "Foley Dataset HF Smoother",
    "FoleyDatasetAugmenter": "Foley Dataset Augmenter",
    "FoleyDatasetSaver": "Foley Dataset Saver",
    "FoleyDatasetItemExtractor": "Foley Dataset Item Extractor",
    "FoleyDatasetSpectralMatcher": "Foley Dataset Spectral Matcher",
    "FoleyHfSmoother": "Foley HF Smoother",
    "FoleyHarmonicExciter": "Foley Harmonic Exciter",
    "FoleyOutputNormalizer": "Foley Output Normalizer",
    "FoleyDatasetBrowser": "Foley Dataset Browser",
}
