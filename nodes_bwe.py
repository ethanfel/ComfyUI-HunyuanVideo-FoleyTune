"""FoleyTune bandwidth-extension node — post-hoc HF restoration via UniverSR.

The foley LoRA produces excellent temporal sync but muffled audio (HF energy
~0.005 vs ~0.045 in bright ground truth). Training-side HF recovery was
exhausted (spectral bias of diffusion); the pragmatic fix is post-hoc blind
bandwidth extension. UniverSR (ICASSP 2026, vocoder-free flow matching in the
complex STFT domain) brightens the generated 48kHz audio after the fact.

Validated settings for foley: input_sr=16000 + guidance_scale=3.0 regenerates
only the top octave (>8kHz), reading as natural air rather than hallucinated
texture. Lower input_sr / higher guidance brighten more but risk artifacts;
the blend knob dials the effect down if needed.

Requires `universr` in the ComfyUI environment:
    pip install git+https://github.com/woongzip1/UniverSR.git
"""

import os
import tempfile

import torch
import torchaudio
import comfy.model_management as mm
from loguru import logger

TARGET_SR = 48000

# Cache loaded UniverSR models by (model_path, device) so the 229MB weights load once.
_UNIVERSR_CACHE = {}


def _get_universr(model_path, device):
    key = (model_path, str(device))
    if key not in _UNIVERSR_CACHE:
        try:
            from universr import UniverSR
        except ImportError as e:
            raise RuntimeError(
                "UniverSR is not installed in this environment. Install with:\n"
                "    pip install git+https://github.com/woongzip1/UniverSR.git\n"
                f"(import error: {e})"
            )
        logger.info(f"Loading UniverSR from {model_path} on {device}...")
        _UNIVERSR_CACHE[key] = UniverSR.from_pretrained(model_path, device=str(device))
        logger.info("UniverSR loaded.")
    return _UNIVERSR_CACHE[key]


class FoleyTuneBWE:
    """Brighten muffled foley audio with UniverSR bandwidth extension.

    Restores high-frequency content (>4-8kHz) that the LoRA under-generates,
    keeping the model's superior temporal sync intact. Runs per-channel so
    stereo is preserved, and blends with the dry signal for intensity control.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Generated foley audio to brighten."}),
                "input_sr": ([8000, 12000, 16000, 24000], {
                    "default": 16000,
                    "tooltip": "Effective input bandwidth (Hz). The model treats content as valid up to input_sr/2 and regenerates above it. 16000 = regenerate only >8kHz (most natural). 8000 = regenerate >4kHz (brighter, more aggressive).",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 6.0, "step": 0.25,
                    "tooltip": "Classifier-free guidance. Higher = more aggressive HF generation. 3.0 validated for foley; 1.5 (default) is too weak; >4 risks hiss/artifacts.",
                }),
                "blend": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Wet/dry mix. 1.0 = full bandwidth extension, 0.0 = bypass (original). Lower to dial back artifacts.",
                }),
                "ode_steps": ("INT", {
                    "default": 4, "min": 1, "max": 32, "step": 1,
                    "tooltip": "ODE integration steps. 4 is fast and validated; more steps can sharpen but interact unpredictably with high guidance.",
                }),
                "ode_method": (["midpoint", "euler", "rk4"], {"default": "midpoint"}),
            },
            "optional": {
                "model_path": ("STRING", {
                    "default": "woongzip1/universr-audio",
                    "tooltip": "HF model id or local checkpoint dir. Default downloads the general-audio model on first use.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "enhance"
    CATEGORY = "FoleyTune"

    def enhance(self, audio, input_sr, guidance_scale, blend, ode_steps,
                ode_method, model_path="woongzip1/universr-audio"):
        waveform = audio["waveform"]  # [B, C, T]
        sr = audio["sample_rate"]

        if blend <= 0.0:
            return (audio,)  # bypass

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        device = mm.get_torch_device()

        # UniverSR targets 48kHz; resample dry signal so blend stays aligned.
        dry = waveform.float().cpu()
        if sr != TARGET_SR:
            dry = torchaudio.functional.resample(dry, sr, TARGET_SR)

        model = _get_universr(model_path, device)
        cfg = guidance_scale if guidance_scale and guidance_scale > 0 else None

        # UniverSR's tensor path assumes the tensor is *at* input_sr; only its file
        # path correctly treats a 48kHz signal as band-limited and applies the
        # internal low-pass. Hand it a 48kHz temp WAV per channel so behaviour
        # matches the validated file-based result exactly.
        B, C, T = dry.shape
        out_chans = []
        with tempfile.TemporaryDirectory() as td:
            for b in range(B):
                chans = []
                for c in range(C):
                    wav_path = os.path.join(td, f"ch_{b}_{c}.wav")
                    torchaudio.save(wav_path, dry[b, c:c + 1], TARGET_SR)
                    wet = model.enhance(
                        wav_path,
                        input_sr=int(input_sr),
                        ode_method=ode_method,
                        ode_steps=int(ode_steps),
                        guidance_scale=cfg,
                    ).detach().cpu().float().reshape(-1)  # (T',)
                    d = dry[b, c]
                    n = min(d.shape[-1], wet.shape[-1])
                    mixed = (1.0 - blend) * d[:n] + blend * wet[:n]
                    chans.append(mixed)
                m = min(x.shape[-1] for x in chans)
                out_chans.append(torch.stack([x[:m] for x in chans], dim=0))
        # align batch items to equal length
        mb = min(x.shape[-1] for x in out_chans)
        out = torch.stack([x[:, :mb] for x in out_chans], dim=0)  # [B, C, T]

        logger.info(f"FoleyTune BWE: input_sr={input_sr}, cfg={guidance_scale}, "
                    f"blend={blend}, {B}x{C} channels -> {out.shape[-1]} samples @48kHz")
        return ({"waveform": out.cpu(), "sample_rate": TARGET_SR},)


NODE_CLASS_MAPPINGS = {"FoleyTuneBWE": FoleyTuneBWE}
NODE_DISPLAY_NAME_MAPPINGS = {"FoleyTuneBWE": "FoleyTune BWE (UniverSR)"}
