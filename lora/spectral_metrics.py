"""Spectral analysis utilities for evaluating audio generation quality."""

import numpy as np
import torch


def spectral_metrics(wav: np.ndarray, sr: int) -> dict:
    """Compute spectral descriptors for a mono waveform.

    Args:
        wav: 1D numpy array of audio samples
        sr: sample rate

    Returns:
        dict with keys: hf_energy_ratio, spectral_centroid_hz,
        spectral_rolloff_hz, spectral_flatness, temporal_variance
    """
    # STFT: 2048-point FFT, 512-sample hop, Hann window
    n_fft = 2048
    hop = 512
    window = np.hanning(n_fft)

    # Frame the signal
    n_frames = 1 + (len(wav) - n_fft) // hop
    if n_frames < 1:
        return {
            "hf_energy_ratio": 0.0,
            "spectral_centroid_hz": 0.0,
            "spectral_rolloff_hz": 0.0,
            "spectral_flatness": 0.0,
            "temporal_variance": 0.0,
        }

    frames = np.stack([wav[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    spec = np.abs(np.fft.rfft(frames, n=n_fft))
    power = spec ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    total_energy = power.sum()
    if total_energy < 1e-10:
        return {k: 0.0 for k in ["hf_energy_ratio", "spectral_centroid_hz",
                                   "spectral_rolloff_hz", "spectral_flatness",
                                   "temporal_variance"]}

    # HF energy ratio (>4kHz)
    hf_mask = freqs > 4000
    hf_energy_ratio = float(power[:, hf_mask].sum() / total_energy)

    # Spectral centroid
    mean_power = power.mean(axis=0)
    centroid = float(np.sum(freqs * mean_power) / (mean_power.sum() + 1e-10))

    # Spectral rolloff (85%)
    cumsum = np.cumsum(mean_power)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # Spectral flatness (Wiener entropy)
    mp = mean_power + 1e-10
    geo_mean = np.exp(np.mean(np.log(mp)))
    arith_mean = np.mean(mp)
    flatness = float(geo_mean / (arith_mean + 1e-10))

    # Temporal variance
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    mean_rms = frame_rms.mean()
    temporal_var = float(frame_rms.std() / (mean_rms + 1e-10))

    return {
        "hf_energy_ratio": hf_energy_ratio,
        "spectral_centroid_hz": centroid,
        "spectral_rolloff_hz": rolloff,
        "spectral_flatness": flatness,
        "temporal_variance": temporal_var,
    }


def reference_metrics(gen_wav: np.ndarray, ref_wav: np.ndarray, sr: int) -> dict:
    """Compute distance metrics between generated and reference audio.

    Returns:
        dict with keys: log_spectral_distance_db, mel_cepstral_distortion,
        per_band_correlation
    """
    n_fft = 2048
    hop = 512
    window = np.hanning(n_fft)

    min_len = min(len(gen_wav), len(ref_wav))
    gen_wav = gen_wav[:min_len]
    ref_wav = ref_wav[:min_len]

    n_frames = 1 + (min_len - n_fft) // hop
    if n_frames < 1:
        return {"log_spectral_distance_db": 0.0, "mel_cepstral_distortion": 0.0,
                "per_band_correlation": 0.0}

    gen_frames = np.stack([gen_wav[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    ref_frames = np.stack([ref_wav[i * hop : i * hop + n_fft] * window for i in range(n_frames)])

    gen_spec = np.abs(np.fft.rfft(gen_frames, n=n_fft)) + 1e-10
    ref_spec = np.abs(np.fft.rfft(ref_frames, n=n_fft)) + 1e-10

    # Log spectral distance
    lsd = float(np.sqrt(np.mean((20 * np.log10(gen_spec) - 20 * np.log10(ref_spec)) ** 2)))

    # Mel cepstral distortion (simplified: log-mel space L2)
    n_mels = 80
    mel_fmin, mel_fmax = 0, sr // 2
    mel_points = np.linspace(2595 * np.log10(1 + mel_fmin / 700),
                              2595 * np.log10(1 + mel_fmax / 700), n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    mel_fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, mid, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        mel_fb[i] = np.where((freqs >= lo) & (freqs <= mid),
                              (freqs - lo) / (mid - lo + 1e-10), 0)
        mel_fb[i] += np.where((freqs > mid) & (freqs <= hi),
                               (hi - freqs) / (hi - mid + 1e-10), 0)

    gen_mel = np.log(gen_spec @ mel_fb.T + 1e-10)
    ref_mel = np.log(ref_spec @ mel_fb.T + 1e-10)
    mcd = float(np.sqrt(np.mean((gen_mel - ref_mel) ** 2)) * (10 / np.log(10)))

    # Per-band correlation
    cors = []
    for b in range(n_mels):
        g, r = gen_mel[:, b], ref_mel[:, b]
        if g.std() < 1e-8 or r.std() < 1e-8:
            cors.append(0.0)
        else:
            cors.append(float(np.corrcoef(g, r)[0, 1]))
    pbc = float(np.mean(cors))

    return {
        "log_spectral_distance_db": lsd,
        "mel_cepstral_distortion": mcd,
        "per_band_correlation": pbc,
    }
