"""Voice conversion backends — common interface for EZ-VC, Seed-VC, Vevo."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class VCBackend(ABC):
    """Base class for zero-shot voice conversion backends."""

    name: str = "base"
    native_sr: int = 16000

    @abstractmethod
    def load(self, device: str = "cuda", model_path: Optional[str] = None):
        """Load model weights and any auxiliary components."""

    @abstractmethod
    def convert(
        self,
        source_path: str,
        reference_path: str,
        *,
        diffusion_steps: int = 25,
        cfg_strength: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """Convert source audio to match reference voice.

        Returns (audio_numpy, sample_rate). Audio is float32 mono.
        """

    @abstractmethod
    def unload(self):
        """Free GPU memory."""


BACKENDS: dict[str, type[VCBackend]] = {}


def register_backend(cls: type[VCBackend]) -> type[VCBackend]:
    BACKENDS[cls.name] = cls
    return cls


def get_backend(name: str) -> type[VCBackend]:
    if name not in BACKENDS:
        _lazy_import(name)
    return BACKENDS[name]


def _lazy_import(name: str):
    if name == "ezvc":
        import vc_backends.ezvc
    elif name == "seedvc":
        import vc_backends.seedvc
    elif name == "vevo":
        import vc_backends.vevo
    else:
        raise ValueError(f"Unknown VC backend: {name}")


def available_backends() -> list[str]:
    return ["ezvc", "seedvc", "vevo"]
