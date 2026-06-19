"""FoleyTune mastering node — the three training-free "confident wins" from the
post-BWE audio research (docs/plans/2026-06-19-post-bwe-audio-mastering-research.md),
implemented in pure numpy/scipy (+ pyloudnorm) so there is NO native dependency
(pedalboard is not required).

Three stages, applied in the research-recommended internal order, each independently
toggleable so you can A/B by ear:

  1. HARMONIC EXCITER  — HPF -> asymmetric soft-clip (tanh, +even) -> attenuate ->
     parallel mix. SYNTHESIZES harmonics from existing content (unlike BWE, which
     extrapolates a brand-new band), so it adds brightness/presence with a cleaner
     character than BWE fizz. Targets the DARK-tail / wash problem.
  2. TRANSIENT SHAPER  — gain driven by (fast envelope - slow envelope), level-
     independent (no threshold), so it PUNCHES skin-slaps without pumping the moans.
  3. LOUDNESS + TRUE-PEAK LIMITER — pyloudnorm (ITU-R BS.1770-4) LUFS normalize
     (opt-in level change) followed by a 4x-oversampled true-peak limiter (safety,
     on by default; transparent unless peaks exceed the ceiling).

Validation is by ear + the render-battery (HF flatness, HNR, gap-band breath, crest,
LUFS) — PESQ/STOI are invalid for non-speech foley.
"""

import numpy as np
import torch
from scipy import signal as sps
from scipy.ndimage import minimum_filter1d
from loguru import logger

try:
    import pyloudnorm as pyln
    _HAS_PYLN = True
except Exception:  # pragma: no cover
    _HAS_PYLN = False


# --------------------------------------------------------------------------- #
# DSP helpers (operate on 1-D float64 numpy arrays)                            #
# --------------------------------------------------------------------------- #
def _onepole(x, tau_s, sr):
    """Causal one-pole low-pass (envelope follower). tau = time constant in seconds."""
    a = float(np.exp(-1.0 / (max(tau_s, 1e-6) * sr)))
    return sps.lfilter([1.0 - a], [1.0, -a], x)


def _fit_len(g, n):
    """Pad (edge) or truncate a 1-D array to length n."""
    if len(g) == n:
        return g
    if len(g) > n:
        return g[:n]
    return np.concatenate([g, np.full(n - len(g), g[-1] if len(g) else 1.0)])


def _exciter(x, sr, freq, drive, amount, even):
    """Aural-exciter: HPF -> (a)symmetric soft-clip -> rescale -> parallel add.

    Generates new harmonics from the existing high band; `amount` is the parallel
    blend (0.2-0.7 classic). Harmonics are rescaled to the HF band's own RMS so the
    excitation is level-aware (quiet passages get less) and `amount` stays intuitive.
    """
    if amount <= 0.0 or len(x) < 32:
        return x
    nyq = sr * 0.5
    fc = float(np.clip(freq, 100.0, nyq * 0.95))
    sos = sps.butter(4, fc / nyq, btype="high", output="sos")
    hp = sps.sosfiltfilt(sos, x).astype(np.float64)
    driven = drive * hp
    bias = 0.5 * even                      # asymmetry -> even harmonics
    harm = np.tanh(driven + bias)          # tanh -> odd harmonics
    if even > 0.0:
        harm = harm - float(np.mean(harm))  # strip the DC the asymmetry introduced
    rhp = float(np.sqrt(np.mean(hp ** 2))) + 1e-9
    rh = float(np.sqrt(np.mean(harm ** 2))) + 1e-9
    harm *= rhp / rh                        # match HF energy -> level-aware
    return x + amount * harm


def _transient(x, sr, attack, sustain, attack_ms, release_ms):
    """Differential-envelope transient shaper.

    gain = ratio**attack on onsets (fast>slow), ratio**sustain on decays (fast<slow),
    where ratio = fast_env / slow_env. Level-independent: no threshold, so it boosts
    slap attacks regardless of how loud the moan bed is, and doesn't pump the body.
    """
    if (attack == 0.0 and sustain == 0.0) or len(x) < 32:
        return x
    env = np.abs(x).astype(np.float64)
    fast = _onepole(env, attack_ms / 1000.0, sr)
    slow = _onepole(env, release_ms / 1000.0, sr)
    eps = 1e-6
    log_r = np.log((fast + eps) / (slow + eps))
    g_log = np.where(log_r > 0.0, attack * log_r, sustain * log_r)
    gain = np.exp(g_log)
    gain = np.clip(gain, 0.5, 2.0)          # +/-6 dB safety
    gain = _onepole(gain, 0.002, sr)        # 2 ms de-zipper
    return x * gain


def _lufs_normalize(x_2d, sr, target):
    """LUFS normalize [T, C] (or [T]) to target LUFS via pyloudnorm. Returns (out, measured)."""
    if not _HAS_PYLN:
        return x_2d, None
    try:
        meter = pyln.Meter(sr)
        loud = meter.integrated_loudness(x_2d)
        if not np.isfinite(loud) or loud < -70.0:   # silence / too short
            return x_2d, loud
        return pyln.normalize.loudness(x_2d, loud, target), loud
    except Exception as e:                            # e.g. clip shorter than a 400ms block
        logger.warning(f"FoleyTune Master: LUFS skipped ({e})")
        return x_2d, None


def _true_peak_limit(x, sr, ceiling_db, release_ms, lookahead_ms, oversample=4):
    """4x-oversampled lookahead true-peak limiter. Transparent unless peaks exceed ceiling."""
    if len(x) < 32:
        return np.clip(x, -1.0, 1.0)
    ceiling = float(10.0 ** (ceiling_db / 20.0))
    up = sps.resample_poly(x, oversample, 1)
    desired_up = np.minimum(1.0, ceiling / (np.abs(up) + 1e-12))
    # collapse each oversample block by its MIN -> base-rate gain that respects true peak
    n = (len(desired_up) // oversample) * oversample
    g = desired_up[:n].reshape(-1, oversample).min(axis=1)
    g = _fit_len(g, len(x))
    la = max(1, int(lookahead_ms / 1000.0 * sr))
    g = minimum_filter1d(g, size=2 * la + 1)          # pre-dip before peaks (lookahead)
    rel = max(release_ms / 1000.0, 1e-3)
    a = float(np.exp(-1.0 / (rel * sr)))
    g_s = sps.filtfilt([1 - a], [1.0, -a], g)          # smooth the gain envelope
    g = np.minimum(g, g_s)                             # smoothing may only ADD reduction
    g = np.clip(g, 0.0, 1.0)
    return np.clip(x * g, -ceiling, ceiling)


# --------------------------------------------------------------------------- #
# Node                                                                        #
# --------------------------------------------------------------------------- #
class FoleyTuneMaster:
    """Mastering chain for generated foley: exciter + transient shaper + loudness/limiter.

    Drop in AFTER BWE (or instead of it, for the exciter-only brightness path). Each
    stage toggles independently so you can ear-A/B. Runs per-channel; stereo preserved.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Foley audio to master (typically post-BWE, or raw)."}),

                # --- Stage 1: harmonic exciter (darkness / wash) ---
                "exciter_enable": ("BOOLEAN", {"default": True, "tooltip": "Stage 1: add brightness/presence via synthesized harmonics (cleaner than BWE fizz)."}),
                "exciter_amount": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Parallel blend of the excited harmonics. 0.2-0.3 = subtle air; >0.5 gets aggressive."}),
                "exciter_freq": ("FLOAT", {"default": 4000.0, "min": 1000.0, "max": 12000.0, "step": 250.0,
                    "tooltip": "HPF cutoff (Hz). Only content above this is excited. Lower = brighten lower-mids too."}),
                "exciter_drive": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 8.0, "step": 0.25,
                    "tooltip": "Soft-clip drive = harmonic density. Higher = more harmonics but harsher; keep low on moans."}),
                "exciter_even": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Asymmetry -> EVEN harmonics (warmth/tube-like). 0 = pure odd (tanh). Small values add body."}),

                # --- Stage 2: transient shaper (slaps) ---
                "transient_enable": ("BOOLEAN", {"default": True, "tooltip": "Stage 2: punch up skin-slap attacks without pumping the moans."}),
                "transient_attack": ("FLOAT", {"default": 0.25, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Onset boost. +0.2-0.4 = punchier slaps; negative softens attacks."}),
                "transient_sustain": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Decay/body. +adds tail/room, -tightens. 0 = leave sustain alone."}),
                "transient_attack_ms": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 20.0, "step": 0.5,
                    "tooltip": "Fast envelope time constant (ms). Smaller = sharper transient detection."}),
                "transient_release_ms": ("FLOAT", {"default": 80.0, "min": 20.0, "max": 300.0, "step": 10.0,
                    "tooltip": "Slow envelope time constant (ms). The attack/sustain split is fast vs this."}),

                # --- Stage 3: loudness + true-peak limiter ---
                "loudness_enable": ("BOOLEAN", {"default": False, "tooltip": "Stage 3a: LUFS-normalize to target (a deliberate level change; off by default to respect your global_peak pipeline)."}),
                "target_lufs": ("FLOAT", {"default": -16.0, "min": -30.0, "max": -6.0, "step": 0.5,
                    "tooltip": "Integrated loudness target (ITU-R BS.1770-4). -16 typical for content; -23 broadcast."}),
                "limiter_enable": ("BOOLEAN", {"default": True, "tooltip": "Stage 3b: true-peak safety limiter. Transparent unless peaks exceed the ceiling (catches exciter/transient overshoot)."}),
                "true_peak_db": ("FLOAT", {"default": -1.0, "min": -3.0, "max": 0.0, "step": 0.1,
                    "tooltip": "True-peak ceiling (dBTP). -1.0 leaves headroom for lossy encode; 0 = brickwall at full-scale."}),
                "limiter_release_ms": ("FLOAT", {"default": 60.0, "min": 10.0, "max": 300.0, "step": 10.0,
                    "tooltip": "Limiter gain release (ms). Shorter = louder/grabbier; longer = cleaner."}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "master"
    CATEGORY = "FoleyTune"

    def master(self, audio, exciter_enable, exciter_amount, exciter_freq, exciter_drive, exciter_even,
               transient_enable, transient_attack, transient_sustain, transient_attack_ms, transient_release_ms,
               loudness_enable, target_lufs, limiter_enable, true_peak_db, limiter_release_ms):
        wav = audio["waveform"]
        sr = int(audio["sample_rate"])
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)
        arr = wav.detach().cpu().float().numpy()          # [B, C, T]
        B, C, T = arr.shape
        out = np.empty_like(arr)
        lufs_in = lufs_out = None

        for b in range(B):
            # Stages 1 & 2 run per-channel.
            chans = []
            for c in range(C):
                x = arr[b, c].astype(np.float64)
                if exciter_enable:
                    x = _exciter(x, sr, exciter_freq, exciter_drive, exciter_amount, exciter_even)
                if transient_enable:
                    x = _transient(x, sr, transient_attack, transient_sustain,
                                   transient_attack_ms, transient_release_ms)
                chans.append(x)
            y = np.stack(chans, axis=0)                   # [C, T]

            # Stage 3a: LUFS across channels (perceptual loudness is a whole-signal measure).
            if loudness_enable:
                if not _HAS_PYLN:
                    logger.warning("FoleyTune Master: loudness_enable but pyloudnorm missing; skipping.")
                else:
                    y2d, lufs_in = _lufs_normalize(y.T, sr, target_lufs)   # [T, C]
                    y = np.ascontiguousarray(np.asarray(y2d).T)

            # Stage 3b: true-peak limiter per-channel.
            if limiter_enable:
                y = np.stack([_true_peak_limit(y[c], sr, true_peak_db, limiter_release_ms, 1.5)
                              for c in range(C)], axis=0)
            else:
                y = np.clip(y, -1.0, 1.0)                  # avoid downstream hard-clip artifacts

            out[b] = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        stages = []
        if exciter_enable:   stages.append(f"exciter(a={exciter_amount},f={exciter_freq:.0f})")
        if transient_enable: stages.append(f"transient(atk={transient_attack})")
        if loudness_enable and _HAS_PYLN and lufs_in is not None:
            stages.append(f"lufs({lufs_in:.1f}->{target_lufs:.1f})")
        if limiter_enable:   stages.append(f"tplimit({true_peak_db}dBTP)")
        logger.info(f"FoleyTune Master: {B}x{C}@{sr}Hz -> [{' -> '.join(stages) or 'bypass'}]")

        return ({"waveform": torch.from_numpy(out), "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {"FoleyTuneMaster": FoleyTuneMaster}
NODE_DISPLAY_NAME_MAPPINGS = {"FoleyTuneMaster": "FoleyTune Master (Exciter / Transient / Loudness)"}
