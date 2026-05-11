"""EZ-VC backend — DiT-based flow matching voice conversion.

Requires: pip install the EZ-VC repo + espnet SSL branch.
See https://github.com/EZ-VC/EZ-VC for installation.
"""

import os
import sys
import tempfile
from typing import Optional

import numpy as np
import torch

from vc_backends import VCBackend, register_backend

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)


def _find_ezvc_root():
    candidates = [
        os.path.join(_PROJECT, "vc_repos", "EZ-VC"),
        os.path.join(_PROJECT, "EZ-VC"),
        os.path.expanduser("~/EZ-VC"),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "src", "f5_tts")):
            return c
    return None


@register_backend
class EZVCBackend(VCBackend):
    name = "ezvc"
    native_sr = 16000

    def __init__(self):
        self._model = None
        self._vocoder = None
        self._xeus = None
        self._kmeans = None
        self._device = "cuda"

    def load(self, device: str = "cuda", model_path: Optional[str] = None):
        self._device = device

        ezvc_root = _find_ezvc_root()
        if ezvc_root is None:
            raise FileNotFoundError(
                "EZ-VC repo not found. Run: python install_vc.py ezvc"
            )
        src_dir = os.path.join(ezvc_root, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from f5_tts.infer.utils_infer import load_vocoder, load_model
        from f5_tts.infer.utils_xeus import load_xeus_model, ApplyKmeans
        from f5_tts.model import get_class
        from omegaconf import OmegaConf

        self._vocoder = load_vocoder(vocoder_name="vocos", device=device)

        self._xeus = load_xeus_model(device).eval()
        self._kmeans = ApplyKmeans(device)

        ckpt = model_path or "hf://SPRINGLab/EZ-VC/model_2700000.safetensors"
        cfg_path = os.path.join(src_dir, "f5_tts", "configs", "F5TTS_Base_EZ-VC.yaml")

        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"EZ-VC config not found at {cfg_path}")

        model_cfg = OmegaConf.load(cfg_path)
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        self._model = load_model(
            model_cls, model_cfg.model.arch, ckpt,
            mel_spec_type="vocos",
            vocab_file="hf://SPRINGLab/EZ-VC/vocab.txt",
            device=device,
        )
        print(f"[EZ-VC] Model loaded on {device}", flush=True)

    def convert(
        self,
        source_path: str,
        reference_path: str,
        *,
        diffusion_steps: int = 12,
        cfg_strength: float = 2.0,
    ) -> tuple[np.ndarray, int]:
        from f5_tts.infer.utils_infer import infer_process
        from f5_tts.infer.utils_xeus import extract_units

        ref_text = extract_units(reference_path, self._xeus, self._kmeans, self._device)
        src_text = extract_units(source_path, self._xeus, self._kmeans, self._device)

        audio, sr, _ = infer_process(
            ref_audio=reference_path,
            ref_text=ref_text,
            gen_text=src_text,
            model_obj=self._model,
            vocoder=self._vocoder,
            mel_spec_type="vocos",
            target_rms=0.1,
            cross_fade_duration=0.15,
            nfe_step=diffusion_steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=-1.0,
            speed=1.0,
            fix_duration=None,
            device=self._device,
        )
        return audio.astype(np.float32), self.native_sr

    def unload(self):
        del self._model, self._vocoder, self._xeus, self._kmeans
        self._model = self._vocoder = self._xeus = self._kmeans = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[EZ-VC] Unloaded", flush=True)
