"""Root conftest: prevent pytest from importing the root __init__.py.

The root __init__.py uses relative imports (from .nodes import ...) that only
work inside the ComfyUI runtime. We monkey-patch Package.setup() globally
to skip __init__.py imports for directories that aren't test packages.
"""

import sys
import os

# Add repo root to sys.path for `lora.merge_math` imports
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

# Monkey-patch Package.setup to skip root __init__.py
import _pytest.python as _pp

_orig_package_setup = _pp.Package.setup


def _safe_package_setup(self):
    """Skip importing __init__.py for the root package (ComfyUI node)."""
    if str(self.path) == _ROOT:
        return
    return _orig_package_setup(self)


_pp.Package.setup = _safe_package_setup
