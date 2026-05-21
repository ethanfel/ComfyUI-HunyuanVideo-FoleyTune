"""Pytest configuration for tests/ directory.

Add the repo root to sys.path so `lora.merge_math` is importable.
"""

import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
