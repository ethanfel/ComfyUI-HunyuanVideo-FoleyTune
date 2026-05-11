"""Vevo backend — timbre conversion via flow matching (Amphion toolkit).

Requires: pip install Amphion dependencies + espeak-ng system package.
See https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md
"""

import os
import sys
from typing import Optional

import numpy as np
import torch

from vc_backends import VCBackend, register_backend

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)


def _find_amphion_root():
    candidates = [
        os.path.join(_PROJECT, "vc_repos", "Amphion"),
        os.path.join(_PROJECT, "Amphion"),
        os.path.expanduser("~/Amphion"),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "models", "vc", "vevo")):
            return c
    return None


@register_backend
class VevoBackend(VCBackend):
    name = "vevo"
    native_sr = 24000

    def __init__(self):
        self._pipeline = None
        self._device = "cuda"

    def load(self, device: str = "cuda", model_path: Optional[str] = None):
        self._device = device

        from huggingface_hub import snapshot_download

        hf_dir = snapshot_download("amphion/Vevo", allow_patterns=[
            "tokenizer/vq8192/*",
            "acoustic_modeling/Vq8192ToMels/*",
            "acoustic_modeling/Vocoder/*",
        ])

        tokenizer_ckpt = os.path.join(hf_dir, "tokenizer", "vq8192")
        fmt_ckpt = os.path.join(hf_dir, "acoustic_modeling", "Vq8192ToMels")
        vocoder_ckpt = os.path.join(hf_dir, "acoustic_modeling", "Vocoder")

        amphion_root = _find_amphion_root()
        if amphion_root is None:
            raise FileNotFoundError(
                "Amphion repo not found. Run: python install_vc.py vevo"
            )

        import sys
        if amphion_root not in sys.path:
            sys.path.insert(0, amphion_root)

        from models.vc.vevo.vevo_utils import VevoInferencePipeline

        fmt_cfg = os.path.join(amphion_root, "models", "vc", "vevo", "config", "Vq8192ToMels.json")
        vocoder_cfg = os.path.join(amphion_root, "models", "vc", "vevo", "config", "Vocoder.json")

        self._pipeline = VevoInferencePipeline(
            content_style_tokenizer_ckpt_path=tokenizer_ckpt,
            fmt_cfg_path=fmt_cfg,
            fmt_ckpt_path=fmt_ckpt,
            vocoder_cfg_path=vocoder_cfg,
            vocoder_ckpt_path=vocoder_ckpt,
            device=device,
        )
        print(f"[Vevo] Timbre pipeline loaded on {device}", flush=True)

    def convert(
        self,
        source_path: str,
        reference_path: str,
        *,
        diffusion_steps: int = 32,
        cfg_strength: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        gen_audio = self._pipeline.inference_fm(
            src_wav_path=source_path,
            timbre_ref_wav_path=reference_path,
            flow_matching_steps=diffusion_steps,
            display_audio=False,
        )
        if isinstance(gen_audio, torch.Tensor):
            gen_audio = gen_audio.cpu().numpy()
        audio = gen_audio.astype(np.float32).squeeze()
        return audio, self.native_sr

    def unload(self):
        del self._pipeline
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Vevo] Unloaded", flush=True)
