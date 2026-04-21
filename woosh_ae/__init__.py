"""Vendored subset of Sony AI's Woosh audio autoencoder.

Only the AE (encode/decode) path is vendored. See THIRD_PARTY_NOTICES.md at
the repository root for the upstream commit and license.
"""

from .components.autoencoders import AudioAutoEncoder
from .components.base import LoadConfig

__all__ = ["AudioAutoEncoder", "LoadConfig"]
