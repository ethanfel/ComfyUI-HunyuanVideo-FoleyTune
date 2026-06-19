"""FoleyTune character/colour nodes — TRANSFORMATIVE audio treatments (they reshape
the sound, accepting it's no longer a faithful copy of the model output). Pure
numpy/scipy, no native dep. Four single-purpose nodes you chain as you like:

  - FoleyTuneDeHarsh   — spectral de-harsher ("soothe-lite"): dynamically attenuates
                         bins that protrude above a smoothed spectral envelope ->
                         removes the fizzy/brittle HF EDGE without dulling the tone.
  - FoleyTuneSaturate  — broadband soft saturation (tanh + even): analog/organic
                         WARMTH and density for synthetic audio.
  - FoleyTuneTilt      — zero-phase spectral TILT EQ: darken (tame edge / carve) or
                         brighten, pivoting around a centre frequency.
  - FoleyTuneGlue      — gentle GLUE COMPRESSION: cohesion/density (binds moans+slaps).

Each node: `mode` = off | auto | manual | <presets>. `auto` measures the clip and
adapts. Validate by ear + the render-battery (HF flatness, HNR, gap-band breath,
crest) — PESQ/STOI are invalid for non-speech foley.
"""

import numpy as np
import torch
from scipy import signal as sps
from loguru import logger

# Shown on every FoleyTune audio node's `audio` input so the chain order is discoverable.
_CHAIN = "Recommended FoleyTune chain: BWE → De-Harsh → Tilt → Saturate → Glue → Master (loudness/limiter always last)."


# --------------------------------------------------------------------------- #
# shared helpers                                                              #
# --------------------------------------------------------------------------- #
def _onepole(x, tau_s, sr):
    a = float(np.exp(-1.0 / (max(tau_s, 1e-6) * sr)))
    return sps.lfilter([1.0 - a], [1.0, -a], x)


def _fit_len(g, n):
    if len(g) == n:
        return g
    if len(g) > n:
        return g[:n]
    return np.concatenate([g, np.zeros(n - len(g))])


def _measure(x, sr):
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


def _run(audio, resolve, apply_fn, label):
    """B/C loop with per-item param resolution (so `auto` adapts per clip) + logging."""
    wav = audio["waveform"]
    sr = int(audio["sample_rate"])
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)
    arr = wav.detach().cpu().float().numpy()
    B, C, T = arr.shape
    out = np.empty_like(arr)
    for b in range(B):
        params, dbg = resolve(arr[b].mean(0).astype(np.float64), sr)
        for c in range(C):
            y = arr[b, c].astype(np.float64) if params is None else apply_fn(arr[b, c].astype(np.float64), sr, params)
            out[b, c] = _fit_len(np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0), T).astype(np.float32)
        logger.info(f"FoleyTune {label} item{b} {C}ch@{sr}Hz -> "
                    f"{'bypass' if params is None else params}" + (f"  [{dbg}]" if dbg else ""))
    return ({"waveform": torch.from_numpy(out), "sample_rate": sr},)


# --------------------------------------------------------------------------- #
# DSP                                                                         #
# --------------------------------------------------------------------------- #
def _spec_envelope(logmag, n_cep):
    """Cepstral-smoothed spectral envelope along the frequency axis (per frame)."""
    cep = np.fft.rfft(logmag, axis=0)
    cep[n_cep:, :] = 0.0
    return np.fft.irfft(cep, n=logmag.shape[0], axis=0)


def _deharsh(x, sr, amount, freq_lo, n_cep, nfft=2048, hop=512):
    """Dynamic de-harsher: attenuate STFT bins that protrude above the smoothed
    spectral envelope (resonances / fizz / harsh edges), only above freq_lo."""
    if amount <= 0.0 or len(x) < nfft:
        return x
    f, _, Z = sps.stft(x, fs=sr, nperseg=nfft, noverlap=nfft - hop, window="hann")
    logmag = np.log(np.abs(Z) + 1e-9)
    env = _spec_envelope(logmag, max(6, int(n_cep)))
    excess = np.clip(logmag - env, 0.0, None)        # how far each bin pokes above the envelope
    gain = np.exp(-amount * excess)                  # attenuate proportional to protrusion
    gain = np.maximum(gain, 0.1)                      # floor at -20 dB
    gain = np.where((f >= freq_lo)[:, None], gain, 1.0)
    _, xr = sps.istft(Z * gain, fs=sr, nperseg=nfft, noverlap=nfft - hop, window="hann")
    return xr


def _saturate(x, amount, warmth):
    """Broadband soft saturation: tanh (odd) + asymmetry (even), level-matched.

    Pre-normalizes by peak so the signal actually reaches tanh's bend regardless of how
    quiet the clip is (otherwise a -16 dB foley clip stays in the linear region = no effect).
    """
    if amount <= 0.0:
        return x
    pk = float(np.abs(x).max()) + 1e-9
    g = 0.7 / pk                                     # bring peak to ~0.7 so tanh bends
    drive = 1.0 + amount * 4.0
    pre = float(np.sqrt(np.mean(x ** 2))) + 1e-9
    y = np.tanh(drive * g * x + 0.3 * warmth)
    if warmth > 0.0:
        y = y - float(np.mean(y))                    # strip DC from asymmetry
    return y * (pre / (float(np.sqrt(np.mean(y ** 2))) + 1e-9))   # restore original RMS


def _tilt(x, sr, tilt_db, pivot):
    """Zero-phase spectral tilt: gain linear in log-frequency, pivoting at `pivot`."""
    if abs(tilt_db) < 1e-3 or len(x) < 16:
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    g_db = np.clip(tilt_db * np.log2((f + 1e-6) / pivot), -12.0, 12.0)
    return np.fft.irfft(X * (10.0 ** (g_db / 20.0)), n=len(x))


def _glue(x, sr, amount):
    """Gentle glue compressor: adaptive threshold, low ratio, smoothed gain reduction.

    RMS-matched to the input — glue changes DYNAMICS (cohesion), not level. (Auto-makeup
    from the gain-reduction signal under-compensates because GR is mostly zero, so the
    peak reduction would otherwise drop the level a few dB.)
    """
    if amount <= 0.0:
        return x
    ratio = 1.0 + amount * 3.0
    det_db = 20.0 * np.log10(_onepole(np.abs(x), 0.010, sr) + 1e-9)   # 10 ms detector
    thresh = float(np.percentile(det_db, 75)) - 3.0
    over = np.clip(det_db - thresh, 0.0, None)
    gr_db = -over * (1.0 - 1.0 / ratio)
    gr_db = _onepole(gr_db, 0.080, sr)                                # 80 ms smoothing
    y = x * (10.0 ** (gr_db / 20.0))
    pre = float(np.sqrt(np.mean(x ** 2)))
    post = float(np.sqrt(np.mean(y ** 2))) + 1e-9
    return y * (pre / post)                                           # restore input RMS


# --------------------------------------------------------------------------- #
# Nodes                                                                       #
# --------------------------------------------------------------------------- #
class FoleyTuneDeHarsh:
    """Spectral de-harsher — tames the fizzy/brittle HF edge dynamically, per-frequency,
    without darkening the tone (cuts only what protrudes above the smoothed envelope)."""

    _PRE = {"subtle": (0.15, 3000.0, 50), "medium": (0.30, 2500.0, 40), "strong": (0.50, 2200.0, 30)}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO", {"tooltip": _CHAIN}),
            "mode": (["off", "auto", "manual", "subtle", "medium", "strong"], {"default": "auto",
                "tooltip": "auto = scale by measured fizz/brightness. manual = use the sliders. subtle<medium<strong."}),
            "amount": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "[manual] Depth of attenuation on protruding bins."}),
            "freq": ("FLOAT", {"default": 2500.0, "min": 500.0, "max": 10000.0, "step": 250.0, "tooltip": "[manual] Only de-harsh above this freq (Hz) — protects the moan body."}),
            "smooth": ("INT", {"default": 40, "min": 8, "max": 120, "step": 4, "tooltip": "[manual] Envelope detail (cepstral coeffs). Lower = smoother envelope = catches broader humps; higher = only narrow spikes."}),
        }}
    RETURN_TYPES = ("AUDIO",); FUNCTION = "run"; CATEGORY = "FoleyTune"

    def run(self, audio, mode, amount, freq, smooth):
        def resolve(mono, sr):
            if mode == "off":
                return None, ""
            if mode == "manual":
                return dict(a=amount, f=freq, s=int(smooth)), ""
            if mode == "auto":
                m = _measure(mono, sr)
                a = 0.40 if m["fizz"] > 0.020 else (0.25 if m["fizz"] > 0.010 else 0.12)
                if m["cent"] > 2400 and m["hf"] > 5:
                    a = max(a, 0.30)
                return dict(a=a, f=2500.0, s=40), f"fizz{m['fizz']:.3f} cent{m['cent']:.0f} -> a{a}"
            p = self._PRE[mode]
            return dict(a=p[0], f=p[1], s=p[2]), ""
        return _run(audio, resolve, lambda x, sr, p: _deharsh(x, sr, p["a"], p["f"], p["s"]), "DeHarsh")


class FoleyTuneSaturate:
    """Broadband soft saturation — adds analog/organic warmth + density to synthetic audio."""

    _PRE = {"subtle": (0.10, 0.15), "medium": (0.20, 0.20), "strong": (0.35, 0.30)}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO", {"tooltip": _CHAIN}),
            "mode": (["off", "auto", "manual", "subtle", "medium", "strong"], {"default": "subtle",
                "tooltip": "auto = more warmth on dull/thin clips. manual = sliders. subtle<medium<strong."}),
            "amount": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "[manual] Saturation depth (drive + parallel mix)."}),
            "warmth": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "[manual] Even-harmonic asymmetry (tube-like body). 0 = pure odd (tape-like)."}),
        }}
    RETURN_TYPES = ("AUDIO",); FUNCTION = "run"; CATEGORY = "FoleyTune"

    def run(self, audio, mode, amount, warmth):
        def resolve(mono, sr):
            if mode == "off":
                return None, ""
            if mode == "manual":
                return dict(a=amount, w=warmth), ""
            if mode == "auto":
                m = _measure(mono, sr)
                a = 0.28 if m["cent"] < 1600 else (0.18 if m["cent"] < 2200 else 0.12)
                return dict(a=a, w=0.20), f"cent{m['cent']:.0f} -> a{a}"
            p = self._PRE[mode]
            return dict(a=p[0], w=p[1]), ""
        return _run(audio, resolve, lambda x, sr, p: _saturate(x, p["a"], p["w"]), "Saturate")


class FoleyTuneTilt:
    """Zero-phase spectral tilt EQ — darken (tame edge) or brighten around a pivot."""

    _PRE = {"darken": -1.0, "darken_strong": -2.0, "brighten": 0.8}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO", {"tooltip": _CHAIN}),
            "mode": (["off", "auto", "manual", "darken", "darken_strong", "brighten"], {"default": "auto",
                "tooltip": "auto = darken if bright/edgy, lift if dull. manual = sliders. darken/brighten = fixed tilts."}),
            "tilt_db": ("FLOAT", {"default": -0.5, "min": -4.0, "max": 4.0, "step": 0.1, "tooltip": "[manual] dB/octave tilt. Negative = darker (cut highs, lift lows); positive = brighter."}),
            "pivot": ("FLOAT", {"default": 1000.0, "min": 250.0, "max": 6000.0, "step": 50.0, "tooltip": "[manual] Pivot frequency (Hz) — gain is 0 here, tilts away from it."}),
        }}
    RETURN_TYPES = ("AUDIO",); FUNCTION = "run"; CATEGORY = "FoleyTune"

    def run(self, audio, mode, tilt_db, pivot):
        def resolve(mono, sr):
            if mode == "off":
                return None, ""
            if mode == "manual":
                return dict(t=tilt_db, p=pivot), ""
            if mode == "auto":
                m = _measure(mono, sr)
                t = -0.8 if m["cent"] > 2200 else (0.5 if m["cent"] < 1500 else 0.0)
                return dict(t=t, p=1000.0), f"cent{m['cent']:.0f} -> tilt{t}"
            return dict(t=self._PRE[mode], p=1000.0), ""
        return _run(audio, resolve, lambda x, sr, p: _tilt(x, sr, p["t"], p["p"]), "Tilt")


class FoleyTuneGlue:
    """Gentle glue compression — cohesion/density; binds moans+slaps into one sound."""

    _PRE = {"subtle": 0.10, "medium": 0.20, "strong": 0.35}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO", {"tooltip": _CHAIN}),
            "mode": (["off", "auto", "manual", "subtle", "medium", "strong"], {"default": "subtle",
                "tooltip": "auto = scale by crest (don't over-glue already-flat material). manual = slider. subtle<medium<strong."}),
            "amount": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "[manual] Glue depth (ratio + drive). Gentle by design."}),
        }}
    RETURN_TYPES = ("AUDIO",); FUNCTION = "run"; CATEGORY = "FoleyTune"

    def run(self, audio, mode, amount):
        def resolve(mono, sr):
            if mode == "off":
                return None, ""
            if mode == "manual":
                return dict(a=amount), ""
            if mode == "auto":
                m = _measure(mono, sr)
                a = 0.25 if m["crest"] > 22 else (0.15 if m["crest"] > 16 else 0.05)
                return dict(a=a), f"crest{m['crest']:.1f} -> a{a}"
            return dict(a=self._PRE[mode]), ""
        return _run(audio, resolve, lambda x, sr, p: _glue(x, sr, p["a"]), "Glue")


NODE_CLASS_MAPPINGS = {
    "FoleyTuneDeHarsh": FoleyTuneDeHarsh,
    "FoleyTuneSaturate": FoleyTuneSaturate,
    "FoleyTuneTilt": FoleyTuneTilt,
    "FoleyTuneGlue": FoleyTuneGlue,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FoleyTuneDeHarsh": "FoleyTune De-Harsh (HF edge)",
    "FoleyTuneSaturate": "FoleyTune Saturate (warmth)",
    "FoleyTuneTilt": "FoleyTune Tilt EQ",
    "FoleyTuneGlue": "FoleyTune Glue Comp",
}
