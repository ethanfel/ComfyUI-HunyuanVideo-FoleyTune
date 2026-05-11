"""Seed-VC v2 backend — zero-shot voice conversion with AR + CFM.

Requires: pip install the Seed-VC repo dependencies.
See https://github.com/Plachtaa/seed-vc for installation.
"""

import os
import sys
from typing import Optional

import numpy as np
import torch

from vc_backends import VCBackend, register_backend

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)


def _find_seedvc_root():
    candidates = [
        os.path.join(_PROJECT, "vc_repos", "seed-vc"),
        os.path.join(_PROJECT, "seed-vc"),
        os.path.expanduser("~/seed-vc"),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "configs")):
            return c
    return None


@register_backend
class SeedVCBackend(VCBackend):
    name = "seedvc"
    native_sr = 22050

    def __init__(self):
        self._wrapper = None
        self._device = "cuda"

    def load(self, device: str = "cuda", model_path: Optional[str] = None):
        import yaml
        from hydra.utils import instantiate
        from omegaconf import DictConfig

        self._device = device

        seedvc_root = _find_seedvc_root()
        if seedvc_root is None:
            raise FileNotFoundError(
                "Seed-VC repo not found. Run: python install_vc.py seedvc"
            )
        if seedvc_root not in sys.path:
            sys.path.insert(0, seedvc_root)

        cfg_path = os.path.join(seedvc_root, "configs", "v2", "vc_wrapper.yaml")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Seed-VC config not found at {cfg_path}")

        with open(cfg_path) as f:
            cfg = DictConfig(yaml.safe_load(f))

        self._wrapper = instantiate(cfg)
        self._wrapper.load_checkpoints(
            ar_checkpoint_path=model_path,
            cfm_checkpoint_path=None,
        )

        if hasattr(self._wrapper, 'sr'):
            self.native_sr = self._wrapper.sr

        print(f"[Seed-VC] Model loaded on {device}, sr={self.native_sr}", flush=True)

    def convert(
        self,
        source_path: str,
        reference_path: str,
        *,
        diffusion_steps: int = 30,
        cfg_strength: float = 0.5,
    ) -> tuple[np.ndarray, int]:
        audio = self._wrapper.convert_voice(
            source_audio_path=source_path,
            target_audio_path=reference_path,
            diffusion_steps=diffusion_steps,
            length_adjust=1.0,
            inference_cfg_rate=cfg_strength,
            top_p=0.7,
            temperature=0.7,
            repetition_penalty=1.5,
            use_sway_sampling=False,
            use_amo_sampling=False,
            device=torch.device(self._device),
            dtype=torch.float32,
        )
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        return audio.astype(np.float32).squeeze(), self.native_sr

    def unload(self):
        del self._wrapper
        self._wrapper = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Seed-VC] Unloaded", flush=True)
