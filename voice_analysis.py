"""Voice analysis utilities for speaker clustering and descriptor generation."""

import re
import numpy as np


def extract_voice_features(waveform: np.ndarray, sr: int) -> dict:
    """Extract vocal features from a mono waveform using Parselmouth.

    Args:
        waveform: 1D numpy array of audio samples
        sr: sample rate

    Returns:
        dict with keys: median_f0, mean_hnr, jitter, shimmer, spectral_centroid
    """
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(waveform, sampling_frequency=sr)

    # Pitch (F0) — use default Praat settings
    pitch = snd.to_pitch()
    f0_values = pitch.selected_array["frequency"]
    voiced = f0_values[f0_values > 0]
    median_f0 = float(np.median(voiced)) if len(voiced) > 0 else 0.0

    # Harmonics-to-Noise Ratio — breathiness vs clarity
    hnr = snd.to_harmonicity()
    hnr_values = hnr.values[0]
    valid_hnr = hnr_values[hnr_values != -200]  # Praat uses -200 for unvoiced
    mean_hnr = float(np.mean(valid_hnr)) if len(valid_hnr) > 0 else 0.0

    # Jitter — pitch perturbation (smooth vs rough/raspy)
    point_process = call(snd, "To PointProcess (periodic, cc)...", 75, 600)
    try:
        jitter = call(point_process, "Get jitter (local)...", 0, 0, 0.0001, 0.02, 1.3)
    except Exception:
        jitter = 0.0
    if np.isnan(jitter):
        jitter = 0.0

    # Shimmer — amplitude perturbation (steady vs unstable)
    try:
        shimmer = call([snd, point_process], "Get shimmer (local)...",
                       0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        shimmer = 0.0
    if np.isnan(shimmer):
        shimmer = 0.0

    # Spectral centroid — brightness vs warmth
    spectrum = snd.to_spectrum()
    centroid = call(spectrum, "Get centre of gravity...", 2)
    if np.isnan(centroid):
        centroid = 0.0

    return {
        "median_f0": median_f0,
        "mean_hnr": mean_hnr,
        "jitter": jitter,
        "shimmer": shimmer,
        "spectral_centroid": centroid,
    }


def group_by_source(names: list[str]) -> dict[str, list[int]]:
    """Group item indices by source video prefix.

    Strips trailing _NN suffix: clip_001_03 -> clip_001

    Args:
        names: list of clip names

    Returns:
        dict mapping source prefix -> list of item indices
    """
    groups = {}
    for idx, name in enumerate(names):
        prefix = re.sub(r"_\d+$", "", name)
        groups.setdefault(prefix, []).append(idx)
    return groups


def sample_indices(group_size: int, samples_per_source: int) -> list[int]:
    """Pick evenly-spaced indices from a group.

    Args:
        group_size: number of items in the group
        samples_per_source: how many to pick

    Returns:
        list of integer indices
    """
    n = min(samples_per_source, group_size)
    return [int(i) for i in np.linspace(0, group_size - 1, n)]


def generate_descriptor(median_f0: float, mean_hnr: float,
                        min_f0_female: float = 165.0,
                        mode: str = "auto",
                        jitter: float = 0.0,
                        shimmer: float = 0.0,
                        spectral_centroid: float = 0.0) -> str:
    """Generate a CLAP-compatible voice descriptor string.

    Uses vocal pedagogy terminology for richer, more distinctive descriptors:
    - Register: soprano (>250Hz), mezzo-soprano (190-250Hz), contralto (165-190Hz)
    - Breathiness: breathy vs clear (HNR)
    - Texture: raspy vs smooth (jitter)
    - Brightness: bright vs warm (spectral centroid)

    Args:
        median_f0: median fundamental frequency in Hz
        mean_hnr: mean harmonics-to-noise ratio in dB
        min_f0_female: F0 threshold for male/female split
        mode: "auto" for full descriptors, "label_only" for gender only
        jitter: pitch perturbation (0-1 scale, >0.02 = raspy)
        shimmer: amplitude perturbation (0-1 scale)
        spectral_centroid: center of spectral gravity in Hz

    Returns:
        descriptor string, e.g. "breathy bright soprano voice"
    """
    is_female = median_f0 >= min_f0_female

    if mode == "label_only":
        return "female voice" if is_female else "male voice"

    # Vocal register (female classification)
    if is_female:
        if median_f0 > 250:
            register = "soprano"
        elif median_f0 >= 190:
            register = "mezzo-soprano"
        else:
            register = "contralto"
    else:
        if median_f0 > 140:
            register = "tenor"
        elif median_f0 >= 100:
            register = "baritone"
        else:
            register = "bass"

    # Breathiness (HNR) — most distinctive axis
    breath = "breathy" if mean_hnr < 10 else "clear"

    # Texture (jitter) — smooth vs raspy
    texture = "raspy" if jitter > 0.02 else "smooth"

    # Brightness (spectral centroid) — bright vs warm
    # Typical female speech centroid: 1500-3000 Hz
    if spectral_centroid > 0:
        brightness = "bright" if spectral_centroid > 2000 else "warm"
    else:
        brightness = ""

    # Build descriptor — pick the 2 most distinctive qualities + register + "voice"
    parts = []
    parts.append(breath)
    if brightness:
        parts.append(brightness)
    if texture == "raspy":
        parts.append(texture)
    parts.append(register)
    parts.append("voice")

    return " ".join(parts)


def waveform_to_mono_numpy(wav) -> "np.ndarray":
    """Convert a waveform (torch tensor or numpy array) to mono 1D numpy.

    Handles [1, C, L] torch tensors, [C, L] numpy arrays, etc.
    """
    if hasattr(wav, "numpy"):
        wav_np = wav[0].cpu().numpy()  # [C, L]
    elif isinstance(wav, np.ndarray):
        wav_np = wav[0]
    else:
        wav_np = np.array(wav)[0]
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=0)  # mono [L]
    return wav_np


def tag_prompt(prompt: str, descriptor: str, position: str = "prepend") -> str:
    """Prepend or append a voice descriptor to a prompt.

    Args:
        prompt: existing prompt string
        descriptor: voice descriptor to add
        position: "prepend" or "append"

    Returns:
        tagged prompt string
    """
    if not descriptor:
        return prompt
    if not prompt:
        return descriptor
    if position == "append":
        return f"{prompt}, {descriptor}"
    return f"{descriptor}, {prompt}"
