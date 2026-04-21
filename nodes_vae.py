"""VAE round-trip nodes: DAC and Woosh-AE.

A/B tool for comparing reconstruction ceilings. Each node runs an AUDIO
input through an encode->decode cycle of its VAE and returns the output.
"""
import os
import sys
import tempfile
import threading
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

# Serializes concurrent ensure_woosh_ae() calls (e.g. two ComfyUI worker threads
# hitting a cold cache at once) so only one thread downloads and extracts.
_LOCK = threading.Lock()


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


def _make_download_reporthook(total_hint: int):
    """Build a urlretrieve reporthook that logs progress every 10% (of the
    expected total). We prefer percentage over a fixed byte stride so the cadence
    stays readable regardless of final file size; ~10 log lines total.
    """
    state = {"last_pct_bucket": -1}
    total_mb = max(total_hint, 1) / (1024 * 1024)

    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        # total_size from the server can be -1 when Content-Length is missing;
        # fall back to the expected size so the percentage is still meaningful.
        total = total_size if total_size and total_size > 0 else total_hint
        downloaded = min(block_num * block_size, total) if total > 0 else block_num * block_size
        if total <= 0:
            return
        pct = int(downloaded * 100 / total)
        bucket = pct // 10
        if bucket > state["last_pct_bucket"]:
            state["last_pct_bucket"] = bucket
            downloaded_mb = downloaded / (1024 * 1024)
            logger.info(
                f"Woosh-AE download: {downloaded_mb:.1f}/{total_mb:.1f} MB ({pct}%)"
            )

    return _hook


def ensure_woosh_ae() -> str:
    """Download + extract Woosh-AE weights if missing. Returns absolute path to the checkpoint folder."""
    with _LOCK:
        # Re-check inside the lock (standard double-check): another thread may
        # have finished the download while we waited on the lock.
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
        urllib.request.urlretrieve(
            WOOSH_AE_RELEASE_URL,
            zip_path,
            reporthook=_make_download_reporthook(WOOSH_AE_EXPECTED_SIZE),
        )
        logger.info(f"Woosh-AE download complete: {zip_path}")

        actual_size = os.path.getsize(zip_path)
        if actual_size != WOOSH_AE_EXPECTED_SIZE:
            os.remove(zip_path)
            raise RuntimeError(
                f"Woosh-AE download size mismatch: got {actual_size}, expected {WOOSH_AE_EXPECTED_SIZE}. "
                f"Try deleting {zip_path} and re-running."
            )

        logger.info(f"Extracting {zip_path}")
        # Always extract into a tempdir first, then atomically move the final
        # Woosh-AE/ subtree into `target`. This keeps every layout crash-safe:
        # a partial extract lives entirely inside the tempdir and gets cleaned
        # up by TemporaryDirectory on exception, so a subsequent call sees no
        # half-populated `target` and re-downloads cleanly.
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            layout, inner_prefix = _infer_zip_layout(names)

            with tempfile.TemporaryDirectory(dir=foley_dir, prefix=".woosh-ae-extract-") as tmp:
                zf.extractall(tmp)

                if layout == "parent":
                    # Zip had top-level Woosh-AE/; source is <tmp>/Woosh-AE.
                    src = os.path.join(tmp, "Woosh-AE")
                elif layout == "target":
                    # Loose files at the zip root: the tmp dir itself holds the
                    # final layout. shutil.move of a directory into a fresh
                    # destination path renames it, which is what we want.
                    src = tmp
                else:  # 'relocate'
                    src = os.path.join(tmp, inner_prefix.rstrip("/"))

                if not os.path.isdir(src):
                    raise RuntimeError(
                        f"Expected source directory {src!r} after extracting "
                        f"{WOOSH_AE_ZIP_NAME} but it does not exist. Zip layout may have changed."
                    )

                # Target must not exist for shutil.move to rename-into-place.
                if os.path.isdir(target):
                    shutil.rmtree(target)
                shutil.move(src, target)

        os.remove(zip_path)

        if not (os.path.isdir(target) and os.listdir(target)):
            raise RuntimeError(f"Extraction succeeded but {target} is empty. Zip layout may have changed.")

        logger.info(f"Woosh-AE ready at {target}")
        return target


# --- Shared audio helpers ----------------------------------------------------

TARGET_SR = 48000

def _prep_audio(audio: dict) -> tuple[torch.Tensor, int, int]:
    """Normalize ComfyUI AUDIO to [B, 1, L] mono @ 48 kHz. Returns (wave, orig_sr, orig_len)."""
    wave = audio["waveform"]  # [B, C, L]
    sr = int(audio["sample_rate"])
    orig_len = wave.shape[-1]

    if wave.shape[1] > 1:
        logger.warning(f"Stereo input, downmixing to mono")
        wave = wave.mean(dim=1, keepdim=True)

    if sr != TARGET_SR:
        logger.warning(f"Resampling {sr} Hz -> {TARGET_SR} Hz")
        wave = torchaudio.functional.resample(wave, sr, TARGET_SR)

    return wave, sr, orig_len


def _pad_to_min(wave: torch.Tensor, min_len: int) -> tuple[torch.Tensor, int]:
    """Right-pad with silence if shorter than min_len. Returns (padded_wave, pad_applied)."""
    pad = max(0, min_len - wave.shape[-1])
    if pad:
        wave = torch.nn.functional.pad(wave, (0, pad))
    return wave, pad


def _finalize(wave: torch.Tensor, orig_len_at_target_sr: int) -> dict:
    """Trim to original duration (at target SR) and package as ComfyUI AUDIO."""
    wave = wave[..., :orig_len_at_target_sr]
    return {"waveform": wave.contiguous().cpu(), "sample_rate": TARGET_SR}
