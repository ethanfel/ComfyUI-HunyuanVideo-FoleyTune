"""Voice analysis utilities for speaker clustering and descriptor generation."""

import re
import numpy as np


def extract_voice_features(waveform: np.ndarray, sr: int) -> dict:
    """Extract F0 and HNR from a mono waveform using Parselmouth.

    Args:
        waveform: 1D numpy array of audio samples
        sr: sample rate

    Returns:
        dict with keys: median_f0, mean_hnr
    """
    import parselmouth

    snd = parselmouth.Sound(waveform, sampling_frequency=sr)

    # Pitch (F0) — use default Praat settings
    pitch = snd.to_pitch()
    f0_values = pitch.selected_array["frequency"]
    voiced = f0_values[f0_values > 0]
    median_f0 = float(np.median(voiced)) if len(voiced) > 0 else 0.0

    # Harmonics-to-Noise Ratio
    hnr = snd.to_harmonicity()
    hnr_values = hnr.values[0]
    valid_hnr = hnr_values[hnr_values != -200]  # Praat uses -200 for unvoiced
    mean_hnr = float(np.mean(valid_hnr)) if len(valid_hnr) > 0 else 0.0

    return {"median_f0": median_f0, "mean_hnr": mean_hnr}


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
                        mode: str = "auto") -> str:
    """Generate a CLAP-compatible voice descriptor string.

    Args:
        median_f0: median fundamental frequency in Hz
        mean_hnr: mean harmonics-to-noise ratio in dB
        min_f0_female: F0 threshold for male/female split
        mode: "auto" for full descriptors, "label_only" for gender only

    Returns:
        descriptor string, e.g. "breathy high-pitched female"
    """
    is_female = median_f0 >= min_f0_female

    if mode == "label_only":
        return "female voice" if is_female else "male voice"

    # Pitch label
    if median_f0 > 250:
        pitch = "high-pitched"
    elif median_f0 >= min_f0_female:
        pitch = "mid-pitched"
    else:
        pitch = "deep"

    # Breathiness label
    breath = "breathy" if mean_hnr < 10 else "clear"

    gender = "female" if is_female else "male"

    return f"{breath} {pitch} {gender}"


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
