"""Foley Audio Dataset Pipeline — chainable in-memory preprocessing nodes.

Typical chain:
  FoleyTuneVideoQualityFilter
      ↓ FOLEYTUNE_AUDIO_DATASET  (+ val clip from rejected)
  FoleyTuneBatchFeatureExtractor
      ↓ FOLEYTUNE_AUDIO_DATASET  (+ features attached)
  FoleyTuneDatasetResampler       (optional)
      ↓ FOLEYTUNE_AUDIO_DATASET
  FoleyTuneDatasetLUFSNormalizer  (optional)
      ↓ FOLEYTUNE_AUDIO_DATASET
  FoleyTuneDatasetCompressor      (optional)
      ↓ FOLEYTUNE_AUDIO_DATASET
  FoleyTuneDatasetHfSmoother      (optional)
      ↓ FOLEYTUNE_AUDIO_DATASET
  FoleyTuneDatasetAugmenter       (optional)
      ↓ FOLEYTUNE_AUDIO_DATASET
  FoleyTuneDatasetInspector       (optional)
      ↓ FOLEYTUNE_AUDIO_DATASET  +  STRING report
  FoleyTuneDatasetSaver
      ↓ STRING report  +  dataset.json  +  val/ subfolder

Alternative entry points:
  FoleyTuneDatasetLoader          → FOLEYTUNE_AUDIO_DATASET (from saved FLAC files)
  FoleyTuneDatasetItemExtractor   → AUDIO (bridges to standard nodes)
"""

import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

FOLEYTUNE_AUDIO_DATASET = "FOLEYTUNE_AUDIO_DATASET"
FOLEYTUNE_DS_CATEGORY = "FoleyTune"
FOLEYTUNE_AUDIO_CATEGORY = "FoleyTune"

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aac", ".m4a", ".aiff", ".aif"}
_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}
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

class FoleyTuneDatasetLoader:
    """Load all audio files in a folder into an in-memory FOLEYTUNE_AUDIO_DATASET."""

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

    RETURN_TYPES = (FOLEYTUNE_AUDIO_DATASET,)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "load"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
    DESCRIPTION = "Load all audio files from a folder into memory as a FOLEYTUNE_AUDIO_DATASET."

    def load(self, folder: str):
        folder = Path(folder.strip())
        if not folder.exists():
            raise FileNotFoundError(f"[FoleyTuneDatasetLoader] Folder not found: {folder}")

        files = [f for f in folder.rglob("*") if f.suffix.lower() in _AUDIO_EXTS]
        if not files:
            raise RuntimeError(f"[FoleyTuneDatasetLoader] No audio files found in {folder}")

        dataset = []
        for f in sorted(files):
            try:
                wav, sr = _load_audio(f)
                dataset.append({"waveform": wav, "sample_rate": sr, "name": f.stem})
            except Exception as e:
                print(f"[FoleyTuneDatasetLoader] Skipping {f.name}: {e}", flush=True)

        if not dataset:
            raise RuntimeError(f"[FoleyTuneDatasetLoader] All {len(files)} files failed to load from {folder}")

        print(f"[FoleyTuneDatasetLoader] Loaded {len(dataset)} clips from {folder}", flush=True)
        return (dataset,)


# ─── Node 2: Dataset Resampler ───────────────────────────────────────────────

class FoleyTuneDatasetResampler:
    """Resample all clips in a dataset to a target sample rate using soxr VHQ."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEYTUNE_AUDIO_DATASET,),
                "target_sr": ("INT", {
                    "default": 48000, "min": 8000, "max": 192000,
                    "tooltip": "Target sample rate. 48000 for Foley (DAC codec).",
                }),
            }
        }

    RETURN_TYPES = (FOLEYTUNE_AUDIO_DATASET,)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "resample"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
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

        print(f"[FoleyTuneDatasetResampler] {changed}/{len(dataset)} clips resampled -> {target_sr} Hz", flush=True)
        return (out,)


# ─── Node 3: Dataset LUFS Normalizer ─────────────────────────────────────────

class FoleyTuneDatasetLUFSNormalizer:
    """Normalize each clip to a target integrated LUFS level + true peak limit."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEYTUNE_AUDIO_DATASET,),
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

    RETURN_TYPES = (FOLEYTUNE_AUDIO_DATASET,)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "normalize"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
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
            f"[FoleyTuneDatasetLUFSNormalizer] {len(dataset) - skipped}/{len(dataset)} clips normalized  "
            f"target={target_lufs} LUFS  TP={true_peak_dbtp} dBTP  skipped={skipped}",
            flush=True,
        )
        return (out,)


# ─── Node 4: Dataset Compressor ──────────────────────────────────────────────

class FoleyTuneDatasetCompressor:
    """Apply mild parallel compression to reduce within-clip dynamic range.

    Uses pedalboard.Compressor. Parallel (New York) style:
    blends compressed signal with dry so transients are preserved while
    the dynamic range is gently tightened. Apply after LUFS normalization.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEYTUNE_AUDIO_DATASET,),
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

    RETURN_TYPES = (FOLEYTUNE_AUDIO_DATASET,)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "compress"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
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
            f"[FoleyTuneDatasetCompressor] {len(out)} clips compressed  "
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


class FoleyTuneDatasetInspector:
    """Analyze each clip for quality issues and optionally filter out flagged clips."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEYTUNE_AUDIO_DATASET,),
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

    RETURN_TYPES = (FOLEYTUNE_AUDIO_DATASET, "STRING")
    RETURN_NAMES = ("dataset", "report")
    FUNCTION = "inspect"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
    DESCRIPTION = (
        "Analyze each clip for clipping, low SNR, codec artifacts, and silence. "
        "Outputs a filtered FOLEYTUNE_AUDIO_DATASET and a text report."
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
            duration = wav.shape[-1] / sr
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
                lines.append(f"  FLAGGED  {name} ({duration:.2f}s): {', '.join(issues)}")
                existing_reasons = item.get("reject_reasons", [])
                item = dict(item)
                item["rejected"] = True
                item["reject_reasons"] = existing_reasons + issues
                if item.get("val") or not skip_rejected:
                    clean.append(item)
            else:
                item = dict(item)
                item.setdefault("rejected", False)
                item.setdefault("reject_reasons", [])
                clean.append(item)
                lines.append(f"  OK       {name} ({duration:.2f}s)")

        lines.append("=" * 40)
        lines.append(
            f"Total: {len(dataset)}  Clean: {len(clean)}  Flagged: {len(flagged)}"
            + (" (removed)" if skip_rejected else " (kept)")
        )
        report = "\n".join(lines)
        print(f"[FoleyTuneDatasetInspector]\n{report}", flush=True)
        return (clean, report)


# ─── Node 5b: Dataset Quality Filter ────────────────────────────────────────


def _bandwidth_score(wav: torch.Tensor, sr: int) -> float:
    """Score 0-1 based on effective audio bandwidth via spectral rolloff at 85%.

    Maps rolloff >= 16 kHz -> 1.0, <= 4 kHz -> 0.0, linear between.
    Detects bandwidth-limited or upsampled clips.
    """
    mono = wav[0].mean(0)
    n_fft = 2048
    hop = 512
    if mono.shape[0] < n_fft:
        return 0.0
    window = torch.hann_window(n_fft, device=mono.device)
    stft = torch.stft(mono, n_fft, hop, n_fft, window, return_complex=True)
    power = stft.abs().pow(2).mean(-1)  # average over time -> [n_freqs]
    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1, device=mono.device)
    cumsum = torch.cumsum(power, dim=0)
    total = cumsum[-1].clamp(min=1e-12)
    rolloff_idx = torch.searchsorted(cumsum, 0.85 * total).item()
    rolloff_hz = freqs[min(rolloff_idx, len(freqs) - 1)].item()
    # Linear map: 2 kHz -> 0.0, 16 kHz -> 1.0
    score = (rolloff_hz - 2000.0) / (16000.0 - 2000.0)
    return max(0.0, min(1.0, score))


def _spectral_quality_score(wav: torch.Tensor, sr: int) -> float:
    """Score 0-1 based on spectral naturalness.

    Average of three sub-metrics (each min-max scaled to 0-1):
    - Spectral flatness: higher = more natural broadband content
    - Temporal variance: higher = dynamic audio
    - HF energy ratio: presence of high-frequency content
    """
    mono = wav[0].mean(0).float().cpu().numpy()
    n_fft = 2048
    hop = 512
    n_frames = 1 + (len(mono) - n_fft) // hop
    if n_frames < 1:
        return 0.0

    window = np.hanning(n_fft)
    frames = np.stack([mono[i * hop: i * hop + n_fft] * window for i in range(n_frames)])
    spec = np.abs(np.fft.rfft(frames, n=n_fft))
    power = spec ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    total_energy = power.sum()
    if total_energy < 1e-10:
        return 0.0

    # Spectral flatness (Wiener entropy) — typical range 0.0001-0.1 for audio
    mean_power = power.mean(axis=0) + 1e-10
    geo_mean = np.exp(np.mean(np.log(mean_power)))
    arith_mean = np.mean(mean_power)
    flatness = geo_mean / (arith_mean + 1e-10)
    # Map: 0.0 -> 0.0, 0.05+ -> 1.0 (over-cleaned audio has flatness < 0.001)
    flatness_score = min(1.0, flatness / 0.05)

    # Temporal variance (CoV of frame RMS) — typical range 0.1-2.0
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    mean_rms = frame_rms.mean()
    temporal_var = frame_rms.std() / (mean_rms + 1e-10)
    # Map: 0.0 -> 0.0, 0.5+ -> 1.0
    temporal_score = min(1.0, temporal_var / 0.5)

    # HF energy ratio (>4kHz) — typical range 0.01-0.3
    hf_mask = freqs > 4000
    hf_ratio = float(power[:, hf_mask].sum() / total_energy)
    # Map: 0.0 -> 0.0, 0.1+ -> 1.0
    hf_score = min(1.0, hf_ratio / 0.1)

    return (flatness_score + temporal_score + hf_score) / 3.0


class FoleyTuneDatasetQualityFilter:
    """Research-backed quality scoring and filtering for Foley audio datasets.

    Computes three quality sub-scores per clip:
    - Bandwidth: effective audio bandwidth via spectral rolloff
    - Spectral quality: naturalness via flatness, temporal variance, HF energy
    - CLAP similarity: text-audio alignment (optional, requires clap_prompt)

    Clips are rejected if their composite score or any individual sub-score
    falls below the configured thresholds.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEYTUNE_AUDIO_DATASET,),
                "min_quality_score": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Minimum composite quality score to pass.",
                }),
                "skip_rejected": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If True, rejected clips are removed from output.",
                }),
            },
            "optional": {
                "clap_prompt": ("STRING", {
                    "default": "",
                    "tooltip": "Global text prompt for CLAP similarity. Empty = skip CLAP.",
                }),
                "min_bandwidth_score": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Minimum bandwidth sub-score (~0.3 = 6.2 kHz effective).",
                }),
                "min_spectral_score": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Minimum spectral quality sub-score.",
                }),
                "min_clap_score": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Minimum CLAP similarity sub-score (per GenAU paper).",
                }),
                "weight_bandwidth": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Weight of bandwidth score in composite.",
                }),
                "weight_spectral": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Weight of spectral quality score in composite.",
                }),
                "weight_clap": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Weight of CLAP similarity score in composite.",
                }),
            },
        }

    RETURN_TYPES = (FOLEYTUNE_AUDIO_DATASET, "STRING")
    RETURN_NAMES = ("dataset", "report")
    FUNCTION = "filter_quality"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
    DESCRIPTION = (
        "Research-backed quality filtering: scores clips on effective bandwidth, "
        "spectral naturalness, and optional CLAP text-audio similarity. "
        "Rejects clips below quality thresholds."
    )

    def filter_quality(self, dataset, min_quality_score: float,
                       skip_rejected: bool, clap_prompt: str = "",
                       min_bandwidth_score: float = 0.3,
                       min_spectral_score: float = 0.2,
                       min_clap_score: float = 0.1,
                       weight_bandwidth: float = 0.4,
                       weight_spectral: float = 0.4,
                       weight_clap: float = 0.2):
        use_clap = bool(clap_prompt.strip())
        clap_model = None
        clap_processor = None
        text_embed = None

        if use_clap:
            from transformers import ClapModel, ClapProcessor
            print("[QualityFilter] Loading CLAP model for text-audio scoring...",
                  flush=True)
            clap_model = ClapModel.from_pretrained("laion/larger_clap_general")
            clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
            clap_model.eval()
            # Pre-compute text embedding once
            text_inputs = clap_processor(
                text=[clap_prompt], return_tensors="pt", padding=True
            )
            with torch.no_grad():
                text_embed = clap_model.get_text_features(**text_inputs)
                if not isinstance(text_embed, torch.Tensor):
                    text_embed = text_embed.pooler_output
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        # Normalize weights
        if use_clap:
            w_total = weight_bandwidth + weight_spectral + weight_clap
        else:
            w_total = weight_bandwidth + weight_spectral
        if w_total < 1e-8:
            w_total = 1.0
        w_bw = weight_bandwidth / w_total
        w_sq = weight_spectral / w_total
        w_cl = (weight_clap / w_total) if use_clap else 0.0

        passed = []
        rejected = []
        lines = ["=== Quality Filter Report ==="]
        scores_all = []

        for item in dataset:
            wav = item["waveform"]
            sr = item["sample_rate"]
            name = item["name"]

            bw = _bandwidth_score(wav, sr)
            sq = _spectral_quality_score(wav, sr)

            cl = None
            if use_clap:
                # Resample to 48 kHz for CLAP if needed
                mono = wav[0].mean(0, keepdim=True)  # [1, L]
                if sr != 48000:
                    mono = torchaudio.functional.resample(mono, sr, 48000)
                audio_inputs = clap_processor(
                    audio=[mono.squeeze(0).numpy()],
                    sampling_rate=48000,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    audio_embed = clap_model.get_audio_features(**audio_inputs)
                    if not isinstance(audio_embed, torch.Tensor):
                        audio_embed = audio_embed.pooler_output
                    audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
                cl = float((audio_embed @ text_embed.T).squeeze())

            # Composite score
            composite = w_bw * bw + w_sq * sq
            if cl is not None:
                composite += w_cl * cl
            scores_all.append(composite)

            # Check rejection reasons
            reasons = []
            if composite < min_quality_score:
                reasons.append(f"below {min_quality_score:.2f}")
            if bw < min_bandwidth_score:
                reasons.append(f"bandwidth < {min_bandwidth_score:.2f}")
            if sq < min_spectral_score:
                reasons.append(f"spectral < {min_spectral_score:.2f}")
            if cl is not None and cl < min_clap_score:
                reasons.append(f"CLAP < {min_clap_score:.2f}")

            cl_str = f"{cl:.2f}" if cl is not None else "--"
            status = "PASS" if not reasons else f"REJECT: {', '.join(reasons)}"
            lines.append(
                f"  {name}: BW={bw:.2f} SQ={sq:.2f} CLAP={cl_str} "
                f"SCORE={composite:.2f} [{status}]"
            )

            if reasons:
                rejected.append(name)
                if not skip_rejected:
                    passed.append(item)
            else:
                passed.append(item)

        avg_score = sum(scores_all) / len(scores_all) if scores_all else 0.0
        lines.append("---")
        lines.append(
            f"Passed: {len(passed)}/{len(dataset)} | "
            f"Rejected: {len(rejected)} | Avg score: {avg_score:.2f}"
        )
        report = "\n".join(lines)
        print(f"[FoleyTuneDatasetQualityFilter]\n{report}", flush=True)
        return (passed, report)


# ─── Node 5c: Video Quality Filter ────────────────────────────────────────────


# Per-worker CLAP processor (initialized once per process via pool initializer)
_worker_clap_proc = None

def _init_clap_worker():
    global _worker_clap_proc
    from transformers import ClapProcessor
    _worker_clap_proc = ClapProcessor.from_pretrained("laion/larger_clap_general")

def _clap_preprocess(npy_path):
    """Compute CLAP mel spectrogram features for one clip. Runs in worker process.

    Takes path to a .npy file (not raw array) to avoid pipe serialization bottleneck.
    """
    mono_np = np.load(npy_path)
    inputs = _worker_clap_proc(
        audio=[mono_np], sampling_rate=48000, return_tensors="np",
    )
    feats = inputs["input_features"][0]  # (1, T, F) or (T, F)
    if feats.ndim == 3:
        feats = feats[0]  # squeeze channel dim → (T, F)
    return feats


def _extract_and_score(args):
    """Extract audio from video, score, and prepare CLAP mono.

    Top-level function so it's picklable for ProcessPoolExecutor.
    Returns numpy arrays (not torch tensors) to avoid shared memory issues.
    Resamples mono to 48kHz for CLAP in the worker (parallel).
    """
    f, folder_str, need_clap = args
    folder = Path(folder_str)
    try:
        rel = f.relative_to(folder)
    except ValueError:
        rel = Path(f.name)
    try:
        wav, sr = _extract_audio_from_video(f)
    except Exception as e:
        return {"path": str(f), "rel": str(rel), "error": str(e)}
    duration = wav.shape[-1] / sr
    bw = _bandwidth_score(wav, sr)
    sq = _spectral_quality_score(wav, sr)
    result = {
        "path": str(f), "rel": str(rel), "duration": duration,
        "bw": bw, "sq": sq, "wav_np": wav.numpy(), "sr": sr,
    }
    # Resample mono to 48kHz for CLAP (parallel in worker, not main process)
    if need_clap:
        mono = wav[0].mean(0, keepdim=True)
        if sr != 48000:
            mono_48k = torchaudio.functional.resample(mono, sr, 48000)
        else:
            mono_48k = mono
        result["mono_48k_np"] = mono_48k.numpy()
    return result


def _extract_audio_from_video(path: Path):
    """Extract audio from a video file via FFmpeg subprocess.

    Returns (waveform [1, C, L], sample_rate) or raises on failure.
    Uses native sample rate — no resampling.
    """
    import io
    import subprocess
    import soundfile as sf

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vn",          # no video
        "-f", "wav",    # output WAV at native sr/channels
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())

    wav_np, sr = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav_np).T.unsqueeze(0)  # [1, C, L]
    return wav, sr


def _scan_video_folder(folder: Path):
    """Scan folder + 1 level of subfolders for video files."""
    files = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in _VIDEO_EXTS:
            files.append(f)
        elif f.is_dir():
            for child in sorted(f.iterdir()):
                if child.is_file() and child.suffix.lower() in _VIDEO_EXTS:
                    files.append(child)
    return files


class FoleyTuneVideoQualityFilter:
    """Quality-filter video clips by analyzing their audio track only.

    Extracts audio via FFmpeg (no video frame decoding), scores each clip
    using bandwidth and spectral quality metrics, and optionally copies
    passing videos to an output folder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Folder containing video files. Scans 1 level of subfolders.",
                }),
                "min_quality_score": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Minimum composite quality score to pass.",
                }),
                "skip_rejected": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If True, only passing clips are copied to output.",
                }),
                "num_workers": ("INT", {
                    "default": 4, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Parallel workers for FFmpeg extraction + spectral scoring.",
                }),
            },
            "optional": {
                "output_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Copy passing videos here. Empty = inspect only, no copy.",
                }),
                "clap_prompt": ("STRING", {
                    "default": "",
                    "tooltip": "Global text prompt for CLAP similarity. Empty = skip CLAP.",
                }),
                "clap_negative_prompt": ("STRING", {
                    "default": "",
                    "tooltip": "Reject clips matching this description (e.g. 'voice, speech, talking'). Empty = skip.",
                }),
                "max_negative_clap_score": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Reject clips scoring above this against the negative prompt.",
                }),
                "min_bandwidth_score": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Minimum bandwidth sub-score (~0.3 = 6.2 kHz effective).",
                }),
                "min_spectral_score": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Minimum spectral quality sub-score.",
                }),
                "min_clap_score": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Minimum CLAP similarity sub-score.",
                }),
                "weight_bandwidth": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Weight of bandwidth score in composite.",
                }),
                "weight_spectral": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Weight of spectral quality score in composite.",
                }),
                "weight_clap": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Weight of CLAP similarity score in composite.",
                }),
                "seed": ("INT", {
                    "default": 42,
                    "tooltip": "Seed for random val clip selection from rejected clips.",
                }),
            },
        }

    RETURN_TYPES = (FOLEYTUNE_AUDIO_DATASET, "STRING")
    RETURN_NAMES = ("dataset", "report")
    FUNCTION = "filter_videos"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
    DESCRIPTION = (
        "Quality-filter video clips by audio analysis only (no video frame loading). "
        "Scores bandwidth and spectral quality, optionally copies passing clips. "
        "Outputs a FOLEYTUNE_AUDIO_DATASET of accepted clips (plus one rejected "
        "clip marked as val) alongside the text report."
    )

    def filter_videos(self, video_folder: str, min_quality_score: float,
                      skip_rejected: bool, num_workers: int = 4,
                      output_folder: str = "",
                      clap_prompt: str = "",
                      clap_negative_prompt: str = "",
                      max_negative_clap_score: float = 0.3,
                      min_bandwidth_score: float = 0.3,
                      min_spectral_score: float = 0.2,
                      min_clap_score: float = 0.1,
                      weight_bandwidth: float = 0.4,
                      weight_spectral: float = 0.4,
                      weight_clap: float = 0.2,
                      seed: int = 42):
        import shutil

        folder = Path(video_folder.strip())
        if not folder.exists():
            raise FileNotFoundError(f"[VideoQualityFilter] Folder not found: {folder}")

        files = _scan_video_folder(folder)
        if not files:
            raise RuntimeError(f"[VideoQualityFilter] No video files found in {folder}")

        do_copy = bool(output_folder.strip())
        out_dir = Path(output_folder.strip()) if do_copy else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        # CLAP setup
        use_clap = bool(clap_prompt.strip())
        use_neg_clap = bool(clap_negative_prompt.strip())
        clap_model = None
        clap_processor = None
        text_embed = None
        neg_text_embed = None
        if use_clap or use_neg_clap:
            from transformers import ClapModel, ClapProcessor
            print("[VideoQualityFilter] Loading CLAP model...", flush=True)
            clap_model = ClapModel.from_pretrained("laion/larger_clap_general")
            clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
            clap_model.eval()
            if use_clap:
                text_inputs = clap_processor(
                    text=[clap_prompt], return_tensors="pt", padding=True
                )
                with torch.no_grad():
                    text_embed = clap_model.get_text_features(**text_inputs)
                    if not isinstance(text_embed, torch.Tensor):
                        text_embed = text_embed.pooler_output
                    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            if use_neg_clap:
                neg_inputs = clap_processor(
                    text=[clap_negative_prompt], return_tensors="pt", padding=True
                )
                with torch.no_grad():
                    neg_text_embed = clap_model.get_text_features(**neg_inputs)
                    if not isinstance(neg_text_embed, torch.Tensor):
                        neg_text_embed = neg_text_embed.pooler_output
                    neg_text_embed = neg_text_embed / neg_text_embed.norm(dim=-1, keepdim=True)

        # Normalize weights
        if use_clap:
            w_total = weight_bandwidth + weight_spectral + weight_clap
        else:
            w_total = weight_bandwidth + weight_spectral
        if w_total < 1e-8:
            w_total = 1.0
        w_bw = weight_bandwidth / w_total
        w_sq = weight_spectral / w_total
        w_cl = (weight_clap / w_total) if use_clap else 0.0

        # --- Phase 1: parallel extraction + scoring via ProcessPoolExecutor ---
        # Separate processes bypass the GIL so both FFmpeg I/O and
        # numpy/torch spectral scoring run truly in parallel.
        import time
        from concurrent.futures import ProcessPoolExecutor, as_completed

        need_clap = use_clap or use_neg_clap
        t0 = time.time()
        print(f"[VideoQualityFilter] Phase 1: extracting + scoring + resample "
              f"{len(files)} clips with {num_workers} workers...", flush=True)
        folder_str = str(folder)
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(_extract_and_score, (f, folder_str, need_clap)) for f in files]
            for i, future in enumerate(as_completed(futures)):
                r = future.result()
                if "error" not in r:
                    r["wav"] = torch.from_numpy(r["wav_np"])
                    del r["wav_np"]
                    r["path"] = Path(r["path"])
                    r["rel"] = Path(r["rel"])
                else:
                    r["path"] = Path(r["path"])
                    r["rel"] = Path(r["rel"])
                results.append(r)
                if (i + 1) % 10 == 0 or (i + 1) == len(futures):
                    print(f"  [{i+1}/{len(futures)}] done ({time.time()-t0:.1f}s)", flush=True)

        t1 = time.time()
        print(f"[VideoQualityFilter] Phase 1 done in {t1-t0:.1f}s", flush=True)

        # Sort by relative path to keep report deterministic
        results.sort(key=lambda r: str(r["rel"]))

        # --- Phase 2: parallel CLAP preprocessing + batched GPU inference ---
        valid_results = [r for r in results if "error" not in r]

        clap_scores = {}   # idx -> cl
        neg_clap_scores = {}  # idx -> neg_cl
        if need_clap and valid_results:
            # Collect mono arrays from workers
            batch_indices = [i for i, r in enumerate(valid_results) if "mono_48k_np" in r]
            batch_monos = [valid_results[i]["mono_48k_np"].squeeze(0) for i in batch_indices]

            # Step 1: parallel mel spectrogram computation across workers
            # Save mono arrays to temp .npy files to avoid pipe serialization bottleneck
            # (710 clips × ~2MB each = ~1.4GB through pipes causes hangs)
            import tempfile
            print(f"[VideoQualityFilter] Phase 2a: CLAP preprocessing {len(batch_monos)} clips "
                  f"with {num_workers} workers...", flush=True)
            t_pre = time.time()
            with tempfile.TemporaryDirectory() as tmpdir:
                npy_paths = []
                for i, mono in enumerate(batch_monos):
                    p = os.path.join(tmpdir, f"{i}.npy")
                    np.save(p, mono)
                    npy_paths.append(p)
                with ProcessPoolExecutor(max_workers=num_workers,
                                         initializer=_init_clap_worker) as pool:
                    mel_features = list(pool.map(_clap_preprocess, npy_paths, chunksize=8))
            print(f"[VideoQualityFilter] Phase 2a done in {time.time()-t_pre:.1f}s", flush=True)

            # Step 2: single batched GPU forward pass
            print(f"[VideoQualityFilter] Phase 2b: CLAP model inference (batched)...", flush=True)
            t_gpu = time.time()
            # Stack all mel features into one batch tensor
            # Each feature is [T, F] from ClapProcessor — pad on T (axis 0)
            max_t = max(f.shape[0] for f in mel_features)
            n_mel = mel_features[0].shape[1]
            padded = np.zeros((len(mel_features), max_t, n_mel),
                              dtype=mel_features[0].dtype)
            for j, f in enumerate(mel_features):
                padded[j, :f.shape[0], :] = f
            input_features = torch.from_numpy(padded)

            with torch.no_grad():
                audio_embeds = clap_model.get_audio_features(input_features=input_features)
                if not isinstance(audio_embeds, torch.Tensor):
                    audio_embeds = audio_embeds.pooler_output
                audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)

            for j, bi in enumerate(batch_indices):
                embed = audio_embeds[j:j+1]
                if use_clap:
                    clap_scores[bi] = float((embed @ text_embed.T).squeeze())
                if use_neg_clap:
                    neg_clap_scores[bi] = float((embed @ neg_text_embed.T).squeeze())

            print(f"[VideoQualityFilter] Phase 2b done in {time.time()-t_gpu:.1f}s", flush=True)

        # --- Phase 3: decisions ---
        print(f"[VideoQualityFilter] Phase 3: filtering...", flush=True)
        accepted_items = []
        rejected_items = []
        n_passed = 0
        n_rejected = 0
        n_skipped = 0
        scores_all = []
        lines = ["=== Video Quality Filter Report ==="]

        valid_idx = 0
        for r in results:
            rel = r["rel"]

            if "error" in r:
                lines.append(f"  SKIP  {rel}: FFmpeg error — {r['error']}")
                n_skipped += 1
                continue

            bw, sq, duration = r["bw"], r["sq"], r["duration"]
            cl = clap_scores.get(valid_idx)
            neg_cl = neg_clap_scores.get(valid_idx)
            valid_idx += 1

            # Free CLAP mono
            r.pop("mono_48k_np", None)

            composite = w_bw * bw + w_sq * sq
            if cl is not None:
                composite += w_cl * cl
            scores_all.append(composite)

            # Rejection check
            reasons = []
            if composite < min_quality_score:
                reasons.append(f"below {min_quality_score:.2f}")
            if bw < min_bandwidth_score:
                reasons.append(f"bandwidth < {min_bandwidth_score:.2f}")
            if sq < min_spectral_score:
                reasons.append(f"spectral < {min_spectral_score:.2f}")
            if cl is not None and cl < min_clap_score:
                reasons.append(f"CLAP < {min_clap_score:.2f}")
            if neg_cl is not None and neg_cl > max_negative_clap_score:
                reasons.append(f"NEG_CLAP {neg_cl:.2f} > {max_negative_clap_score:.2f}")

            cl_str = f"{cl:.2f}" if cl is not None else "--"
            neg_cl_str = f"{neg_cl:.2f}" if neg_cl is not None else "--"
            clip_passed = not reasons

            if clip_passed:
                n_passed += 1
                lines.append(
                    f"  PASS    {rel} ({duration:.1f}s): "
                    f"BW={bw:.2f} SQ={sq:.2f} CLAP={cl_str} NEG={neg_cl_str} SCORE={composite:.2f}"
                )
                if do_copy:
                    dst = out_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(r["path"], dst)
                accepted_items.append({
                    "waveform": r["wav"],
                    "sample_rate": r["sr"],
                    "name": str(r["rel"].with_suffix("")).replace(os.sep, "_"),
                    "video_path": str(r["path"]),
                })
            else:
                n_rejected += 1
                status = f"REJECT: {', '.join(reasons)}"
                lines.append(
                    f"  {status}  {rel} ({duration:.1f}s): "
                    f"BW={bw:.2f} SQ={sq:.2f} CLAP={cl_str} NEG={neg_cl_str} SCORE={composite:.2f}"
                )
                if do_copy and not skip_rejected:
                    dst = out_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(r["path"], dst)
                rejected_items.append({
                    "waveform": r["wav"],
                    "sample_rate": r["sr"],
                    "name": str(r["rel"].with_suffix("")).replace(os.sep, "_"),
                    "video_path": str(r["path"]),
                })

        t2 = time.time()
        print(f"[VideoQualityFilter] Phase 3 done in {t2-t1:.1f}s", flush=True)

        # --- Build dataset with optional val clip from rejected ---
        import random
        rng = random.Random(seed)

        dataset = list(accepted_items)
        if rejected_items:
            val_pick = rng.choice(rejected_items)
            val_pick["val"] = True
            dataset.append(val_pick)
            rejected_items.clear()  # free waveforms of non-selected rejects
        results.clear()  # free Phase 1 result dicts

        avg_score = sum(scores_all) / len(scores_all) if scores_all else 0.0
        lines.append("---")
        lines.append(
            f"Passed: {n_passed} | Rejected: {n_rejected} | "
            f"Skipped: {n_skipped} | Avg score: {avg_score:.2f}"
        )
        if do_copy:
            copied = n_passed if skip_rejected else (n_passed + n_rejected)
            lines.append(f"Copied {copied} files to {out_dir}")

        t3 = time.time()
        print(f"[VideoQualityFilter] Total: {t3-t0:.1f}s "
              f"(extract+score: {t1-t0:.1f}s, filter: {t2-t1:.1f}s, finalize: {t3-t2:.1f}s)",
              flush=True)

        report = "\n".join(lines)
        print(f"[FoleyTuneVideoQualityFilter]\n{report}", flush=True)
        return (dataset, report)


# ─── Node 6: Dataset HF Smoother (batch) ─────────────────────────────────────

class FoleyTuneDatasetHfSmoother:
    """Apply soft high-frequency attenuation to every clip in a dataset.

    Blends a low-pass filtered copy with the original. Default cutoff is 16 kHz
    (vs 12 kHz in SelVA) because DAC's neural codec handles HF content better
    than BigVGAN's mel-spectrogram vocoder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEYTUNE_AUDIO_DATASET,),
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

    RETURN_TYPES = (FOLEYTUNE_AUDIO_DATASET,)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "process"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
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

        print(f"[FoleyTuneDatasetHfSmoother] {len(out)} clips processed  "
              f"cutoff={cutoff_hz:.0f}Hz  blend={blend:.2f}", flush=True)
        return (out,)


# ─── Node 7: Dataset Augmenter ───────────────────────────────────────────────

class FoleyTuneDatasetAugmenter:
    """Create augmented variants of each clip to expand a small dataset.

    Supports gain variation (always available) and optionally pitch shift
    and time stretch via audiomentations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEYTUNE_AUDIO_DATASET,),
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

    RETURN_TYPES = (FOLEYTUNE_AUDIO_DATASET,)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "augment"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
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
                print("[FoleyTuneDatasetAugmenter] audiomentations not installed — "
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
                new_item.pop("val", None)
                out.append(new_item)

        print(f"[FoleyTuneDatasetAugmenter] {len(dataset)} originals -> {len(out)} total clips  "
              f"gain=+/-{gain_range_db:.1f}dB"
              + (f"  pitch=+/-{pitch_range_semitones:.1f}st" if pitch_range_semitones > 0 else "")
              + (f"  stretch=+/-{time_stretch_range:.0%}" if time_stretch_range > 0 else ""),
              flush=True)
        return (out,)


# ─── Node 8: Dataset Saver ───────────────────────────────────────────────────

class FoleyTuneDatasetSaver:
    """Save all clips in a FOLEYTUNE_AUDIO_DATASET to disk as FLAC files."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEYTUNE_AUDIO_DATASET,),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to output folder. Created if it does not exist.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dataset_json", "report")
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
    DESCRIPTION = (
        "Save every clip in a FOLEYTUNE_AUDIO_DATASET to output_dir as 24-bit FLAC. "
        "Writes .npz feature files from item data and generates dataset.json with train/val split. "
        "The global prompt is taken from the first training clip's prompt (set by BatchFeatureExtractor)."
    )

    def save(self, dataset, output_dir: str):
        import json
        import soundfile as sf

        out_path = Path(output_dir.strip())
        out_path.mkdir(parents=True, exist_ok=True)

        train_names = []
        val_name = None
        saved = 0
        features_saved = 0

        for item in dataset:
            name = item["name"]
            wav = item["waveform"][0]  # [C, L]
            sr = item["sample_rate"]
            is_val = item.get("val", False)

            # Determine output directory
            if is_val:
                item_dir = out_path / "val"
                item_dir.mkdir(exist_ok=True)
            else:
                item_dir = out_path

            # Write FLAC
            wav_np = wav.permute(1, 0).float().numpy()  # [L, C]
            if wav_np.shape[1] == 1:
                wav_np = wav_np[:, 0]  # [L] mono
            flac_path = item_dir / f"{name}.flac"
            sf.write(str(flac_path), wav_np, sr, subtype="PCM_24")
            saved += 1

            # Write .npz from features if present
            if "features" in item:
                feats = item["features"]
                npz_path = item_dir / f"{name}.npz"
                save_kwargs = {}
                for key in ("clip_features", "sync_features", "text_embedding"):
                    feat_val = feats.get(key)
                    if feat_val is not None:
                        save_kwargs[key] = feat_val.float().numpy() if hasattr(feat_val, 'numpy') else feat_val
                if "duration" in feats:
                    save_kwargs["duration"] = feats["duration"]
                if "fps" in feats:
                    save_kwargs["fps"] = feats["fps"]
                if item.get("prompt"):
                    save_kwargs["prompt"] = item["prompt"]
                np.savez(str(npz_path), **save_kwargs)
                features_saved += 1

            # Track names for dataset.json
            if is_val:
                val_name = f"val/{name}"
            else:
                train_names.append(name)

        # Derive global prompt from first training clip
        prompt = ""
        for item in dataset:
            if not item.get("val") and item.get("prompt"):
                prompt = item["prompt"]
                break

        # Write dataset.json
        ds_json = {"train": train_names}
        if prompt:
            ds_json["prompt"] = prompt
        if val_name:
            ds_json["val"] = val_name
        json_path = out_path / "dataset.json"
        with open(json_path, "w") as f:
            json.dump(ds_json, f, indent=2)

        lines = [f"[FoleyTuneDatasetSaver] Saved {saved} clips -> {out_path}"]
        lines.append(f"  FLAC: {saved}  NPZ: {features_saved}")
        lines.append(f"  Train: {len(train_names)}  Val: {1 if val_name else 0}")
        lines.append(f"  dataset.json -> {json_path}")

        report = "\n".join(lines)
        print(report, flush=True)
        return (str(json_path), report)


# ─── Node 9: Dataset Item Extractor ──────────────────────────────────────────

class FoleyTuneDatasetItemExtractor:
    """Extract a single AUDIO item from a FOLEYTUNE_AUDIO_DATASET by index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEYTUNE_AUDIO_DATASET,),
                "index": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "0-based index. Wraps around if index >= dataset length.",
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "INT")
    RETURN_NAMES = ("audio", "name", "total")
    FUNCTION = "extract"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
    DESCRIPTION = (
        "Extract one clip from a FOLEYTUNE_AUDIO_DATASET by index. "
        "Returns standard AUDIO (compatible with all audio nodes), "
        "the clip name, and the total dataset length."
    )

    def extract(self, dataset, index: int):
        if not dataset:
            raise RuntimeError("[FoleyTuneDatasetItemExtractor] Dataset is empty.")
        idx = index % len(dataset)
        item = dataset[idx]
        audio = {"waveform": item["waveform"], "sample_rate": item["sample_rate"]}
        print(
            f"[FoleyTuneDatasetItemExtractor] [{idx}/{len(dataset)-1}] {item['name']}  "
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

class FoleyTuneDatasetSpectralMatcher:
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
                "dataset": (FOLEYTUNE_AUDIO_DATASET,),
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

    RETURN_TYPES = (FOLEYTUNE_AUDIO_DATASET,)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "process"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
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
                f"[FoleyTuneDatasetSpectralMatcher] No audio files found in reference dir: {ref_dir}"
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
                print(f"[FoleyTuneDatasetSpectralMatcher] Skipping ref {f.name}: {e}", flush=True)

        if not all_means:
            raise RuntimeError("[FoleyTuneDatasetSpectralMatcher] Could not process any reference files")

        target_mean = torch.stack(all_means).mean(dim=0)  # [n_mels]
        print(f"[FoleyTuneDatasetSpectralMatcher] Computed reference profile from "
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
                print(f"[FoleyTuneDatasetSpectralMatcher] Warning: stereo clip '{item['name']}' "
                      f"collapsed to dual-mono (spectral matching operates on mono downmix)", flush=True)
                result = result.expand(-1, wav.shape[1], -1).clone()

            new_item = dict(item)
            new_item["waveform"] = result
            out.append(new_item)

        print(f"[FoleyTuneDatasetSpectralMatcher] {len(out)} clips processed  "
              f"strength={strength}  n_mels={n_mels}  ref={ref_dir.name}", flush=True)
        return (out,)


# ─── Single-audio pre/post-processor nodes ───────────────────────────────────

# ─── Node 11: HF Smoother (single audio) ─────────────────────────────────────

class FoleyTuneHfSmoother:
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
    CATEGORY = FOLEYTUNE_AUDIO_CATEGORY
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

        print(f"[FoleyTuneHfSmoother] cutoff={cutoff_hz:.0f} Hz  blend={blend:.2f}  "
              f"rms={rms_in:.4f}->{out.pow(2).mean().sqrt():.4f}", flush=True)

        return ({"waveform": out, "sample_rate": sr},)


# ─── Node 12: Harmonic Exciter ───────────────────────────────────────────────

class FoleyTuneHarmonicExciter:
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
    CATEGORY = FOLEYTUNE_AUDIO_CATEGORY
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
        print(f"[FoleyTuneHarmonicExciter] cutoff={cutoff_hz}Hz  drive={drive}  mix={mix:.0%}", flush=True)
        return ({"waveform": wav_out, "sample_rate": sr},)


# ─── Node 13: Output Normalizer ──────────────────────────────────────────────

class FoleyTuneOutputNormalizer:
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
    CATEGORY = FOLEYTUNE_AUDIO_CATEGORY
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
            print("[FoleyTuneOutputNormalizer] Could not measure loudness — passing through.", flush=True)
            return (audio,)

        gain_db = target_lufs - loudness
        gain_linear = 10.0 ** (gain_db / 20.0)

        wav_out = wav * gain_linear

        peak = wav_out.abs().max().item()
        if peak > tp_linear:
            wav_out = wav_out * (tp_linear / peak)

        print(
            f"[FoleyTuneOutputNormalizer] {loudness:.1f} LUFS -> {target_lufs} LUFS  "
            f"gain={gain_db:+.1f}dB  TP={true_peak_dbtp}dBTP",
            flush=True,
        )
        return ({"waveform": wav_out.unsqueeze(0), "sample_rate": sr},)


# ─── Node 14: Dataset Browser ───────────────────────────────────────────────

class FoleyTuneDatasetBrowser:
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
         "raw_audio_dir": "/path/to/raw/audio",
         "audio_dir": "/path/to/cleaned/audio",
         "features_dir": "/path/to/features",
         "clips": ["clip_001", "clip_002"]
       }
       All dirs are independent. audio_dir and raw_audio_dir default
       to clips_dir if omitted. features_dir defaults to clips_dir/features.
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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("video_path", "raw_audio_dir", "clean_audio_dir", "audio_path", "features_dir", "frames_dir", "npz_path", "dataset_dir", "prompt", "max_index")
    OUTPUT_TOOLTIPS = (
        "path + '.mp4'",
        "raw_audio_dir folder path — wire to Foley Dataset Loader",
        "audio_dir folder path — cleaned audio directory",
        "audio_dir/name.flac or .wav  (per-clip cleaned audio file)",
        "features_dir folder path — wire to Feature Extractor cache_dir",
        "clips_dir/name  (image-sequence directory)",
        "features_dir/name.npz  (per-clip .npz features file)",
        "dataset_dir folder path — final dir with .npz + audio for trainer data_dir",
        "Text prompt / label for this clip",
        "count - 1 — wire to a primitive INT's max to constrain the index widget",
    )
    FUNCTION = "browse"
    CATEGORY = FOLEYTUNE_DS_CATEGORY
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
            raise FileNotFoundError(f"[FoleyTuneDatasetBrowser] File not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            data = _json.load(f)

        # Parse the three supported formats into a uniform list
        entries = []
        default_prompt = ""
        clips_dir = None
        raw_audio_dir = None
        audio_dir = None
        features_dir = None
        dataset_dir = None

        if isinstance(data, dict) and "train" in data:
            # FoleyTune train/val format: {"train": [...], "val": "val/name", "prompt": "..."}
            default_prompt = data.get("prompt", "")
            dataset_dir = p.parent
            audio_dir = p.parent
            features_dir = p.parent
            for name in data["train"]:
                base = str(p.parent / name)
                entries.append({"name": name, "base": base, "prompt": default_prompt})
            if data.get("val"):
                val_name = data["val"]
                base = str(p.parent / val_name)
                entries.append({"name": val_name, "base": base, "prompt": default_prompt})
        elif isinstance(data, dict):
            # Compact format
            default_prompt = data.get("prompt", data.get("label", ""))
            clips_dir = Path(data["clips_dir"]) if "clips_dir" in data else None
            raw_audio_dir = Path(data["raw_audio_dir"]) if "raw_audio_dir" in data else clips_dir
            audio_dir = Path(data["audio_dir"]) if "audio_dir" in data else clips_dir
            if "features_dir" in data:
                features_dir = Path(data["features_dir"])
            elif clips_dir:
                features_dir = clips_dir / "features"
            dataset_dir = Path(data["dataset_dir"]) if "dataset_dir" in data else None

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
            raise ValueError(f"[FoleyTuneDatasetBrowser] No entries found in {p}")

        count = len(entries)
        if index >= count:
            raise IndexError(
                f"[FoleyTuneDatasetBrowser] index {index} out of range "
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

        def _find_audio(base_dir, stem, prefer_flac=True):
            """Auto-detect .flac or .wav in base_dir."""
            if base_dir:
                ab = base_dir / stem
            else:
                ab = p_base
            exts = (".flac", ".wav") if prefer_flac else (".wav", ".flac")
            for ext in exts:
                candidate = ab.with_suffix(ext)
                if candidate.exists():
                    return str(candidate)
            return str(ab.with_suffix(exts[0]))

        # Raw audio dir: folder path for Dataset Loader
        raw_dir_str = str(raw_audio_dir) if raw_audio_dir else str(clips_dir or p_base.parent)

        # Clean audio dir: folder path
        clean_dir_str = str(audio_dir) if audio_dir else raw_dir_str

        # Audio path: per-clip file path with auto extension
        audio_path = _find_audio(audio_dir, name, prefer_flac=True)

        # NPZ: use features_dir if set, otherwise derive from base
        if features_dir:
            npz_path = str(features_dir / name) + ".npz"
        else:
            npz_path = str(p_base.parent / "features" / p_base.stem) + ".npz"

        feat_dir_str = str(features_dir) if features_dir else ""
        ds_dir_str = str(dataset_dir) if dataset_dir else ""

        print(
            f"[FoleyTuneDatasetBrowser] [{index}/{count - 1}]\n"
            f"  prompt      = {prompt}\n"
            f"  video_path  = {video_path}\n"
            f"  raw_dir     = {raw_dir_str}\n"
            f"  clean_dir   = {clean_dir_str}\n"
            f"  audio_path  = {audio_path}\n"
            f"  feat_dir    = {feat_dir_str}\n"
            f"  frames_dir  = {frames_dir}\n"
            f"  npz_path    = {npz_path}\n"
            f"  dataset_dir = {ds_dir_str}",
            flush=True,
        )

        return (video_path, raw_dir_str, clean_dir_str, audio_path, feat_dir_str, frames_dir, npz_path, ds_dir_str, prompt, count - 1)


# ─── Node Mappings ───────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "FoleyTuneDatasetLoader": FoleyTuneDatasetLoader,
    "FoleyTuneDatasetResampler": FoleyTuneDatasetResampler,
    "FoleyTuneDatasetLUFSNormalizer": FoleyTuneDatasetLUFSNormalizer,
    "FoleyTuneDatasetCompressor": FoleyTuneDatasetCompressor,
    "FoleyTuneDatasetInspector": FoleyTuneDatasetInspector,
    "FoleyTuneDatasetQualityFilter": FoleyTuneDatasetQualityFilter,
    "FoleyTuneVideoQualityFilter": FoleyTuneVideoQualityFilter,
    "FoleyTuneDatasetHfSmoother": FoleyTuneDatasetHfSmoother,
    "FoleyTuneDatasetAugmenter": FoleyTuneDatasetAugmenter,
    "FoleyTuneDatasetSaver": FoleyTuneDatasetSaver,
    "FoleyTuneDatasetItemExtractor": FoleyTuneDatasetItemExtractor,
    "FoleyTuneDatasetSpectralMatcher": FoleyTuneDatasetSpectralMatcher,
    "FoleyTuneHfSmoother": FoleyTuneHfSmoother,
    "FoleyTuneHarmonicExciter": FoleyTuneHarmonicExciter,
    "FoleyTuneOutputNormalizer": FoleyTuneOutputNormalizer,
    "FoleyTuneDatasetBrowser": FoleyTuneDatasetBrowser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoleyTuneDatasetLoader": "FoleyTune Dataset Loader",
    "FoleyTuneDatasetResampler": "FoleyTune Dataset Resampler",
    "FoleyTuneDatasetLUFSNormalizer": "FoleyTune Dataset LUFS Normalizer",
    "FoleyTuneDatasetCompressor": "FoleyTune Dataset Compressor",
    "FoleyTuneDatasetInspector": "FoleyTune Dataset Inspector",
    "FoleyTuneDatasetQualityFilter": "FoleyTune Dataset Quality Filter",
    "FoleyTuneVideoQualityFilter": "FoleyTune Video Quality Filter",
    "FoleyTuneDatasetHfSmoother": "FoleyTune Dataset HF Smoother",
    "FoleyTuneDatasetAugmenter": "FoleyTune Dataset Augmenter",
    "FoleyTuneDatasetSaver": "FoleyTune Dataset Saver",
    "FoleyTuneDatasetItemExtractor": "FoleyTune Dataset Item Extractor",
    "FoleyTuneDatasetSpectralMatcher": "FoleyTune Dataset Spectral Matcher",
    "FoleyTuneHfSmoother": "FoleyTune HF Smoother",
    "FoleyTuneHarmonicExciter": "FoleyTune Harmonic Exciter",
    "FoleyTuneOutputNormalizer": "FoleyTune Output Normalizer",
    "FoleyTuneDatasetBrowser": "FoleyTune Dataset Browser",
}
