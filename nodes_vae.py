"""VAE round-trip nodes: DAC and Woosh-AE.

A/B tool for comparing reconstruction ceilings. Each node runs an AUDIO
input through an encode->decode cycle of its VAE and returns the output.
"""
import os
import sys
import tempfile
import urllib.request
import zipfile
import shutil
from loguru import logger

import torch
import torchaudio
import folder_paths
import comfy.model_management as mm

# --- Woosh-AE weight auto-download -------------------------------------------

WOOSH_AE_RELEASE_URL = "https://github.com/SonyResearch/Woosh/releases/download/v1.0.0/Woosh-AE.zip"
WOOSH_AE_ZIP_NAME = "Woosh-AE.zip"
WOOSH_AE_EXPECTED_SIZE = 822991075  # bytes


def _woosh_ae_dir() -> str:
    return os.path.join(folder_paths.models_dir, "foley", "Woosh-AE")


def _infer_zip_layout(namelist):
    """Return ('prefix', prefix_str) describing how to relocate zip contents into
    the final Woosh-AE/ folder.

    Three supported layouts:
      1. Zip contains a top-level ``Woosh-AE/`` folder.
         -> extract to parent of target (``foley/``); zip's top dir becomes target.
      2. Zip contains loose files (no common prefix).
         -> extract directly into target.
      3. Zip has some other top-level folder (e.g. ``checkpoints/Woosh-AE/``).
         -> extract to a temp dir, then move the inner ``Woosh-AE/`` (or, if no
            such inner dir exists, the deepest single-child chain) into target.

    Returns one of:
      ('parent', None)            # layout 1
      ('target', None)            # layout 2
      ('relocate', 'inner/path/') # layout 3; path is the subdir inside the
                                   # extraction root whose contents should be
                                   # moved into target.
    """
    # Strip empty entries and normalize separators
    entries = [n.replace("\\", "/") for n in namelist if n]

    # Find all top-level components (first path segment of every entry)
    top_components = set()
    for e in entries:
        head = e.split("/", 1)[0]
        top_components.add(head)

    # Loose files: everything is at the root (no directory segment before any file)
    # Heuristic: if all entries have no '/' OR have only a trailing '/' (empty dir entry),
    # treat as loose.
    has_nested = any("/" in e.rstrip("/") for e in entries)
    if not has_nested:
        return ("target", None)

    # Single top-level folder
    if len(top_components) == 1:
        top = next(iter(top_components))
        if top == "Woosh-AE":
            return ("parent", None)
        # Look for a "Woosh-AE/" directory anywhere inside
        for e in entries:
            parts = e.split("/")
            for i, p in enumerate(parts):
                if p == "Woosh-AE":
                    # Prefix of everything up to and including this segment
                    inner = "/".join(parts[: i + 1]) + "/"
                    return ("relocate", inner)
        # No Woosh-AE subfolder; relocate the single top-level folder's contents
        return ("relocate", top + "/")

    # Multiple top-level entries with some nesting; treat as loose into target.
    return ("target", None)


def ensure_woosh_ae() -> str:
    """Download + extract Woosh-AE weights if missing. Returns absolute path to the checkpoint folder."""
    target = _woosh_ae_dir()
    if os.path.isdir(target) and os.listdir(target):
        return target

    foley_dir = os.path.dirname(target)
    os.makedirs(foley_dir, exist_ok=True)
    zip_path = os.path.join(foley_dir, WOOSH_AE_ZIP_NAME)

    logger.info(
        f"Woosh-AE weights not found. Downloading {WOOSH_AE_ZIP_NAME} "
        f"(~{WOOSH_AE_EXPECTED_SIZE // (1024 * 1024)} MB)"
    )
    urllib.request.urlretrieve(WOOSH_AE_RELEASE_URL, zip_path)

    actual_size = os.path.getsize(zip_path)
    if actual_size != WOOSH_AE_EXPECTED_SIZE:
        os.remove(zip_path)
        raise RuntimeError(
            f"Woosh-AE download size mismatch: got {actual_size}, expected {WOOSH_AE_EXPECTED_SIZE}. "
            f"Try deleting {zip_path} and re-running."
        )

    logger.info(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        layout, inner_prefix = _infer_zip_layout(names)

        if layout == "parent":
            # Zip already has Woosh-AE/ at the top; extract directly into foley/.
            zf.extractall(foley_dir)
        elif layout == "target":
            # Loose files — drop them straight into target.
            os.makedirs(target, exist_ok=True)
            zf.extractall(target)
        else:  # 'relocate'
            # Extract into a temp dir, then move the inner subtree to target.
            with tempfile.TemporaryDirectory(dir=foley_dir, prefix=".woosh-ae-extract-") as tmp:
                zf.extractall(tmp)
                src = os.path.join(tmp, inner_prefix.rstrip("/"))
                if not os.path.isdir(src):
                    raise RuntimeError(
                        f"Expected inner directory {inner_prefix!r} in {WOOSH_AE_ZIP_NAME} "
                        f"but found {os.listdir(tmp)!r}"
                    )
                # Move: if target exists but is empty (from the mkdirs above), remove first
                if os.path.isdir(target):
                    shutil.rmtree(target)
                shutil.move(src, target)

    os.remove(zip_path)

    if not (os.path.isdir(target) and os.listdir(target)):
        raise RuntimeError(f"Extraction succeeded but {target} is empty. Zip layout may have changed.")

    logger.info(f"Woosh-AE ready at {target}")
    return target
