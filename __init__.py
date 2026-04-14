# ComfyUI-HunyuanVideoFoley/__init__.py

# This line makes the 'hunyuanvideo_foley' sub-directory importable by your nodes.py
import sys
import os
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except ImportError:
    pass

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .nodes_lora import (
    NODE_CLASS_MAPPINGS as LORA_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as LORA_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_dataset import (
    NODE_CLASS_MAPPINGS as DATASET_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as DATASET_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **LORA_NODE_CLASS_MAPPINGS, **DATASET_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **LORA_NODE_DISPLAY_NAME_MAPPINGS, **DATASET_NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
