"""Pytest configuration for tests/ directory.

Add the repo root to sys.path so `lora.merge_math` is importable.
"""

import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import safetensors once, globally, BEFORE any test module runs its
# `patch.dict("sys.modules", ...)` block. Those blocks snapshot and restore
# sys.modules; if safetensors were first imported inside such a block it would
# be evicted on exit, and re-importing the PyO3 extension a second time raises
# "PyO3 modules ... may only be initialized once per interpreter process".
# Loading it here keeps it in the base sys.modules so every snapshot preserves it.
try:
    import safetensors.torch  # noqa: F401
except Exception:
    pass
