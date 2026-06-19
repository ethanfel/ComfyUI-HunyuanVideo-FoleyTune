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

PROFILES drive stages 1-2 without slider-fiddling: a safe->strong strength ladder,
content-tuned presets (moaning / wet_oral / slaps_sex), an `auto` mode that measures
the clip and adapts, and `manual` to use the sliders. Loudness + limiter sliders apply
regardless of profile. Validation is by ear + the render-battery (HF flatness, HNR,
gap-band breath, crest, LUFS) — PESQ/STOI are invalid for non-speech foley.
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
# Profiles — resolve exciter + transient params without slider-fiddling       #
# Each entry: exciter (enable, amount, freq, drive, even) + transient          #
# (enable, attack, sustain, attack_ms, release_ms). Content-tuned.            #
# --------------------------------------------------------------------------- #
def _prof(ex_en, amt, freq, drive, even, tr_en, atk, sus, ams, rms):
    return dict(ex_en=ex_en, ex_amt=amt, ex_freq=freq, ex_drive=drive, ex_even=even,
                tr_en=tr_en, tr_atk=atk, tr_sus=sus, tr_ams=ams, tr_rms=rms)

PROFILE_NAMES = ["manual", "auto", "safe", "balanced", "strong", "moaning", "wet_oral", "slaps_sex"]

PROFILES = {
    # strength ladder
    "safe":      _prof(True, 0.15, 4500.0, 1.5, 0.00, True,  0.15,  0.00, 3.0, 80.0),
    "balanced":  _prof(True, 0.25, 4000.0, 2.0, 0.00, True,  0.25,  0.00, 3.0, 80.0),
    "strong":    _prof(True, 0.40, 3500.0, 2.5, 0.10, True,  0.40,  0.05, 2.0, 70.0),
    # content-tuned (I know the material)
    "moaning":   _prof(True, 0.18, 5000.0, 1.5, 0.00, False, 0.00,  0.00, 3.0, 80.0),  # tonal/breath-safe: just air, NO transient
    "wet_oral":  _prof(True, 0.28, 4000.0, 2.0, 0.05, True,  0.35,  0.00, 2.0, 70.0),  # gags/slurps are transient-rich
    "slaps_sex": _prof(True, 0.22, 3500.0, 2.0, 0.00, True,  0.45, -0.05, 1.5, 60.0),  # punch slaps, slightly tighten tails
}


def _measure(x, sr):
    """Quick descriptors on a 1-D signal for the auto profile."""
    n, hop = 2048, 512
    if len(x) < n:
        return dict(cent=2000.0, hf=3.0, fizz=0.01, crest=18.0, floor=-50.0)
    win = np.hanning(n)
    P = np.stack([np.abs(np.fft.rfft(x[i:i + n] * win)) ** 2 for i in range(0, len(x) - n, hop)], 1) + 1e-10
    f = np.fft.rfftfreq(n, 1 / sr)
    cent = float(((f[:, None] * P).sum(0) / P.sum(0)).mean())
    hf = float(P[f > 4000].sum() / P.sum() * 100)
    Phf = P[f > 4000] + 1e-12
    fizz = float((np.exp(np.log(Phf).mean(0)) / (Phf.mean(0) + 1e-12)).mean())
    peak = float(np.abs(x).max()); rms = float(np.sqrt(np.mean(x ** 2))) + 1e-9
    crest = float(20 * np.log10(peak / rms))
    fr = np.array([np.sqrt(np.mean(c ** 2)) for c in np.array_split(x, 100) if len(c)])
    floor = float(20 * np.log10(np.percentile(fr, 10) + 1e-9))
    return dict(cent=cent, hf=hf, fizz=fizz, crest=crest, floor=floor)


def _auto_profile(x, sr):
    """Measure the clip and pick exciter+transient settings adaptively.

    Exciter amount scales with DARKNESS (low centroid -> more), is pulled back when the
    HF is already FIZZY (don't add grain) or the NOISE FLOOR is high (don't amplify hiss).
    Transient attack scales with the PUNCH deficit (low crest -> more).
    """
    m = _measure(x, sr)
    amt = 0.32 if m["cent"] < 1500 else (0.20 if m["cent"] < 2200 else 0.10)
    drive, freq = 2.0, 4000.0
    if m["fizz"] > 0.020:          # already grainy -> gentler, only the very top
        amt *= 0.5; drive = 1.5; freq = 5000.0
    if m["floor"] > -45.0:         # noisy -> don't excite the hiss
        amt *= 0.6
    atk = 0.35 if m["crest"] < 18 else (0.20 if m["crest"] < 24 else 0.10)
    p = _prof(True, round(amt, 3), freq, drive, 0.0, atk > 0.0, atk, 0.0, 2.5, 80.0)
    p["_dbg"] = (f"cent{m['cent']:.0f} hf{m['hf']:.1f} fizz{m['fizz']:.3f} "
                 f"crest{m['crest']:.1f} floor{m['floor']:.0f} -> ex{p['ex_amt']} atk{atk}")
    return p


# --------------------------------------------------------------------------- #
# Node                                                                        #
# --------------------------------------------------------------------------- #
class FoleyTuneMaster:
    """Mastering chain for generated foley: exciter + transient shaper + loudness/limiter.

    Drop in AFTER BWE (or instead of it, for the exciter-only brightness path). Use a
    `profile` for one-knob operation, or `manual` to drive the sliders. Loudness +
    limiter always follow their own sliders. Runs per-channel; stereo preserved.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Foley audio to master (typically post-BWE, or raw)."}),

                "profile": (PROFILE_NAMES, {"default": "balanced",
                    "tooltip": "Preset for the exciter+transient stages. 'manual' = use the sliders below. "
                               "'auto' = measure the clip and adapt (darker->more exciter, fizzy/noisy->less, "
                               "flatter->more punch). safe<balanced<strong. moaning=tonal/breath-safe (no "
                               "transient). wet_oral=gags/slurps (punchy). slaps_sex=slap-forward. "
                               "Loudness+limiter sliders ALWAYS apply."}),

                # --- Stage 1: harmonic exciter (used when profile=manual) ---
                "exciter_enable": ("BOOLEAN", {"default": True, "tooltip": "[manual] Stage 1: brightness/presence via synthesized harmonics (cleaner than BWE fizz)."}),
                "exciter_amount": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "[manual] Parallel blend of excited harmonics. 0.2-0.3 = subtle air; >0.5 aggressive."}),
                "exciter_freq": ("FLOAT", {"default": 4000.0, "min": 1000.0, "max": 12000.0, "step": 250.0,
                    "tooltip": "[manual] HPF cutoff (Hz). Only content above this is excited."}),
                "exciter_drive": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 8.0, "step": 0.25,
                    "tooltip": "[manual] Soft-clip drive = harmonic density. Higher = more but harsher."}),
                "exciter_even": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "[manual] Asymmetry -> EVEN harmonics (warmth). 0 = pure odd (tanh)."}),

                # --- Stage 2: transient shaper (used when profile=manual) ---
                "transient_enable": ("BOOLEAN", {"default": True, "tooltip": "[manual] Stage 2: punch slap attacks without pumping moans."}),
                "transient_attack": ("FLOAT", {"default": 0.25, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "[manual] Onset boost. +0.2-0.4 punchier; negative softens."}),
                "transient_sustain": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "[manual] Decay/body. + adds tail/room, - tightens. 0 = leave alone."}),
                "transient_attack_ms": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 20.0, "step": 0.5,
                    "tooltip": "[manual] Fast envelope time constant (ms)."}),
                "transient_release_ms": ("FLOAT", {"default": 80.0, "min": 20.0, "max": 300.0, "step": 10.0,
                    "tooltip": "[manual] Slow envelope time constant (ms)."}),

                # --- Stage 3: loudness + true-peak limiter (ALWAYS apply) ---
                "loudness_enable": ("BOOLEAN", {"default": False, "tooltip": "Stage 3a: LUFS-normalize to target (a deliberate level change; off by default to respect your global_peak pipeline)."}),
                "target_lufs": ("FLOAT", {"default": -16.0, "min": -30.0, "max": -6.0, "step": 0.5,
                    "tooltip": "Integrated loudness target (ITU-R BS.1770-4). -16 typical for content; -23 broadcast."}),
                "limiter_enable": ("BOOLEAN", {"default": True, "tooltip": "Stage 3b: true-peak safety limiter. Transparent unless peaks exceed the ceiling."}),
                "true_peak_db": ("FLOAT", {"default": -1.0, "min": -3.0, "max": 0.0, "step": 0.1,
                    "tooltip": "True-peak ceiling (dBTP). -1.0 leaves headroom for lossy encode."}),
                "limiter_release_ms": ("FLOAT", {"default": 60.0, "min": 10.0, "max": 300.0, "step": 10.0,
                    "tooltip": "Limiter gain release (ms). Shorter = louder/grabbier; longer = cleaner."}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "master"
    CATEGORY = "FoleyTune"

    def master(self, audio, profile,
               exciter_enable, exciter_amount, exciter_freq, exciter_drive, exciter_even,
               transient_enable, transient_attack, transient_sustain, transient_attack_ms, transient_release_ms,
               loudness_enable, target_lufs, limiter_enable, true_peak_db, limiter_release_ms):
        wav = audio["waveform"]
        sr = int(audio["sample_rate"])
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)
        arr = wav.detach().cpu().float().numpy()          # [B, C, T]
        B, C, T = arr.shape
        out = np.empty_like(arr)

        manual_p = _prof(exciter_enable, exciter_amount, exciter_freq, exciter_drive, exciter_even,
                         transient_enable, transient_attack, transient_sustain,
                         transient_attack_ms, transient_release_ms)

        for b in range(B):
            # Resolve exciter+transient params for this clip.
            if profile == "manual":
                p, dbg = manual_p, ""
            elif profile == "auto":
                p = _auto_profile(arr[b].mean(axis=0).astype(np.float64), sr)
                dbg = p.get("_dbg", "")
            else:
                p, dbg = PROFILES[profile], ""

            # Stages 1 & 2 per-channel.
            chans = []
            for c in range(C):
                x = arr[b, c].astype(np.float64)
                if p["ex_en"]:
                    x = _exciter(x, sr, p["ex_freq"], p["ex_drive"], p["ex_amt"], p["ex_even"])
                if p["tr_en"]:
                    x = _transient(x, sr, p["tr_atk"], p["tr_sus"], p["tr_ams"], p["tr_rms"])
                chans.append(x)
            y = np.stack(chans, axis=0)                   # [C, T]

            # Stage 3a: LUFS across channels.
            lufs_in = None
            if loudness_enable:
                if not _HAS_PYLN:
                    logger.warning("FoleyTune Master: loudness_enable but pyloudnorm missing; skipping.")
                else:
                    y2d, lufs_in = _lufs_normalize(y.T, sr, target_lufs)
                    y = np.ascontiguousarray(np.asarray(y2d).T)

            # Stage 3b: true-peak limiter per-channel.
            if limiter_enable:
                y = np.stack([_true_peak_limit(y[c], sr, true_peak_db, limiter_release_ms, 1.5)
                              for c in range(C)], axis=0)
            else:
                y = np.clip(y, -1.0, 1.0)

            out[b] = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

            stages = []
            if p["ex_en"] and p["ex_amt"] > 0: stages.append(f"exciter(a={p['ex_amt']},f={p['ex_freq']:.0f})")
            if p["tr_en"] and (p["tr_atk"] or p["tr_sus"]): stages.append(f"transient(atk={p['tr_atk']})")
            if loudness_enable and _HAS_PYLN and lufs_in is not None: stages.append(f"lufs({lufs_in:.1f}->{target_lufs:.1f})")
            if limiter_enable: stages.append(f"tplimit({true_peak_db}dBTP)")
            logger.info(f"FoleyTune Master[{profile}] item{b} {C}ch@{sr}Hz -> "
                        f"[{' -> '.join(stages) or 'bypass'}]" + (f"  [{dbg}]" if dbg else ""))

        return ({"waveform": torch.from_numpy(out), "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {"FoleyTuneMaster": FoleyTuneMaster}
NODE_DISPLAY_NAME_MAPPINGS = {"FoleyTuneMaster": "FoleyTune Master (Exciter / Transient / Loudness)"}
