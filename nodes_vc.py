"""Voice Conversion nodes — zero-shot voice transfer using EZ-VC, Seed-VC, or Vevo."""

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
from comfy.model_management import throw_exception_if_processing_interrupted

FOLEYTUNE_AUDIO_DATASET = "FOLEYTUNE_AUDIO_DATASET"
FOLEYTUNE_VC_MODEL = "FOLEYTUNE_VC_MODEL"
FOLEYTUNE_VC_CATEGORY = "FoleyTune/VoiceConversion"

_SOUNDFILE_EXTS = {".wav", ".flac", ".ogg"}

_cached_backend = None
_cached_backend_name = None


def _write_temp_wav(waveform: torch.Tensor, sr: int, tmpdir: str, name: str) -> str:
    """Write a dataset waveform tensor to a temporary WAV file."""
    wav = waveform.squeeze(0)  # [C, L]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    path = os.path.join(tmpdir, f"{name}.wav")
    torchaudio.save(path, wav, sr)
    return path


class FoleyTuneVCModelLoader:
    """Load a zero-shot voice conversion model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["ezvc", "seedvc", "vevo"], {
                    "default": "ezvc",
                    "tooltip": "Which VC backend to use. Each requires its own dependencies installed.",
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                }),
            },
            "optional": {
                "model_path": ("STRING", {
                    "default": "",
                    "tooltip": "Override checkpoint path. Leave empty for default HuggingFace weights.",
                }),
            },
        }

    RETURN_TYPES = (FOLEYTUNE_VC_MODEL,)
    RETURN_NAMES = ("vc_model",)
    FUNCTION = "load_model"
    CATEGORY = FOLEYTUNE_VC_CATEGORY
    DESCRIPTION = (
        "Load a zero-shot voice conversion model. Supports EZ-VC (16kHz), "
        "Seed-VC (22kHz), and Vevo (24kHz). Model stays cached between runs."
    )

    def load_model(self, model: str, device: str, model_path: str = ""):
        global _cached_backend, _cached_backend_name

        if _cached_backend_name == model and _cached_backend is not None:
            print(f"[VC Loader] Reusing cached {model} backend", flush=True)
            return ({"backend": _cached_backend, "name": model},)

        if _cached_backend is not None:
            print(f"[VC Loader] Unloading {_cached_backend_name}", flush=True)
            _cached_backend.unload()
            _cached_backend = None
            _cached_backend_name = None

        from vc_backends import get_backend
        backend_cls = get_backend(model)
        backend = backend_cls()
        backend.load(device=device, model_path=model_path or None)

        _cached_backend = backend
        _cached_backend_name = model
        return ({"backend": backend, "name": model},)


class FoleyTuneVoiceConverter:
    """Convert voices in a dataset to match a reference speaker."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (FOLEYTUNE_AUDIO_DATASET,),
                "vc_model": (FOLEYTUNE_VC_MODEL,),
                "reference_audio": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a WAV/FLAC file of the target voice (a few seconds is enough).",
                }),
                "diffusion_steps": ("INT", {
                    "default": 25, "min": 4, "max": 100,
                    "tooltip": "More steps = better quality but slower. EZ-VC: 12, Seed-VC: 30, Vevo: 32.",
                }),
                "cfg_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance strength. Higher = more similar to reference. EZ-VC: 2.0, Seed-VC: 0.5, Vevo: 1.0.",
                }),
            },
        }

    RETURN_TYPES = (FOLEYTUNE_AUDIO_DATASET,)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "convert"
    CATEGORY = FOLEYTUNE_VC_CATEGORY
    DESCRIPTION = (
        "Convert all clips in a dataset to match a reference speaker's voice. "
        "Preserves timing and content, changes only timbre."
    )

    def convert(self, dataset, vc_model, reference_audio: str,
                diffusion_steps: int, cfg_strength: float):
        backend = vc_model["backend"]
        model_name = vc_model["name"]
        ref_path = reference_audio.strip()

        if not ref_path or not os.path.isfile(ref_path):
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")

        out = []
        total = len(dataset)
        print(f"[VC Converter] Converting {total} clips with {model_name} "
              f"(steps={diffusion_steps}, cfg={cfg_strength})", flush=True)

        with tempfile.TemporaryDirectory(prefix="foleytune_vc_") as tmpdir:
            for i, item in enumerate(dataset):
                throw_exception_if_processing_interrupted()

                name = item.get("name", f"clip_{i:04d}")
                sr = item["sample_rate"]
                src_path = _write_temp_wav(item["waveform"], sr, tmpdir, f"src_{i:04d}")

                try:
                    audio_np, out_sr = backend.convert(
                        source_path=src_path,
                        reference_path=ref_path,
                        diffusion_steps=diffusion_steps,
                        cfg_strength=cfg_strength,
                    )
                except Exception as e:
                    print(f"[VC Converter] Failed on {name}: {e}", flush=True)
                    out.append(item)
                    continue

                if audio_np.ndim == 1:
                    audio_np = audio_np[np.newaxis, :]  # [1, L]
                wav_t = torch.from_numpy(audio_np).float().unsqueeze(0)  # [1, 1, L]

                new_item = dict(item)
                new_item["waveform"] = wav_t
                new_item["sample_rate"] = out_sr
                out.append(new_item)

                if (i + 1) % 10 == 0 or i == total - 1:
                    print(f"[VC Converter] {i + 1}/{total} done", flush=True)

        print(f"[VC Converter] Finished — {len(out)} clips at {backend.native_sr} Hz", flush=True)
        return (out,)


class FoleyTuneVCSingleConverter:
    """Convert a single audio file's voice to match a reference speaker."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vc_model": (FOLEYTUNE_VC_MODEL,),
                "source_audio": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the source audio file to convert.",
                }),
                "reference_audio": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the target voice reference file.",
                }),
                "diffusion_steps": ("INT", {
                    "default": 25, "min": 4, "max": 100,
                }),
                "cfg_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                }),
            },
            "optional": {
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Save converted audio here. Leave empty for temp file.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("output_path", "sample_rate")
    FUNCTION = "convert_single"
    CATEGORY = FOLEYTUNE_VC_CATEGORY
    DESCRIPTION = "Convert a single audio file's voice to match a reference speaker."

    def convert_single(self, vc_model, source_audio: str, reference_audio: str,
                       diffusion_steps: int, cfg_strength: float,
                       output_path: str = ""):
        backend = vc_model["backend"]
        src = source_audio.strip()
        ref = reference_audio.strip()

        if not src or not os.path.isfile(src):
            raise FileNotFoundError(f"Source audio not found: {src}")
        if not ref or not os.path.isfile(ref):
            raise FileNotFoundError(f"Reference audio not found: {ref}")

        audio_np, sr = backend.convert(
            source_path=src,
            reference_path=ref,
            diffusion_steps=diffusion_steps,
            cfg_strength=cfg_strength,
        )

        if not output_path.strip():
            output_path = os.path.splitext(src)[0] + f"_vc_{vc_model['name']}.wav"

        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]
        wav_t = torch.from_numpy(audio_np).float()
        torchaudio.save(output_path, wav_t, sr)
        print(f"[VC Single] Saved to {output_path} ({sr} Hz)", flush=True)
        return (output_path, sr)


NODE_CLASS_MAPPINGS = {
    "FoleyTuneVCModelLoader": FoleyTuneVCModelLoader,
    "FoleyTuneVoiceConverter": FoleyTuneVoiceConverter,
    "FoleyTuneVCSingleConverter": FoleyTuneVCSingleConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FoleyTuneVCModelLoader": "FoleyTune VC Model Loader",
    "FoleyTuneVoiceConverter": "FoleyTune Voice Converter (Dataset)",
    "FoleyTuneVCSingleConverter": "FoleyTune Voice Converter (Single)",
}
