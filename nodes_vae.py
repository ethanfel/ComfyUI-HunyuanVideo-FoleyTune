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


# --- Woosh node --------------------------------------------------------------

_WOOSH_AE_SINGLETON = {"model": None, "device": None, "dtype": None}
_WOOSH_ALIASED = False


def _alias_woosh_modules() -> None:
    """Register ``sys.modules['woosh.*']`` -> ``woosh_ae.*`` so that hydra's
    ``instantiate`` can resolve the upstream ``_target_`` paths embedded in the
    vendored ``config.yaml`` (e.g. ``woosh.module.model.VocosAutoEncoder``).

    Idempotent: guarded by a module-level flag so repeated node invocations do
    not re-import or overwrite existing aliases.
    """
    global _WOOSH_ALIASED
    if _WOOSH_ALIASED:
        return

    import woosh_ae
    import woosh_ae.module
    import woosh_ae.module.model
    import woosh_ae.module.model.vocos

    # Covers the _target_ paths observed in config.yaml:
    #   woosh.module.model.VocosAutoEncoder
    #   woosh.module.model.vocos.ISTFTCircleHead
    #   woosh.module.model.vocos.ZeroDropoutTransform
    # Destructive assignment: if the upstream `woosh` package is installed in
    # the same venv, setdefault would preserve it and hydra would instantiate
    # upstream classes instead of our vendored ones (subtle divergence, hard
    # to diagnose). Overwrite and warn loudly so shadowing is visible.
    existing = sys.modules.get("woosh")
    if existing is not None and existing is not woosh_ae:
        logger.warning("Overriding pre-existing 'woosh' module with vendored woosh_ae (FoleyTune)")
    sys.modules["woosh"] = woosh_ae
    sys.modules["woosh.module"] = woosh_ae.module
    sys.modules["woosh.module.model"] = woosh_ae.module.model
    sys.modules["woosh.module.model.vocos"] = woosh_ae.module.model.vocos

    _WOOSH_ALIASED = True


def _get_woosh_ae(device: str, dtype_str: str):
    """Lazy-load + cache the Woosh-AE. Reloads if device or dtype changed."""
    _alias_woosh_modules()
    from woosh_ae import AudioAutoEncoder, LoadConfig

    dtype = {"fp32": torch.float32, "fp16": torch.float16}[dtype_str]
    target_device = torch.device(device)

    cached = _WOOSH_AE_SINGLETON
    if (cached["model"] is not None
            and cached["device"] == target_device
            and cached["dtype"] == dtype):
        return cached["model"]

    # Free old model before loading a new one on device/dtype change so we
    # don't hold 2x memory transiently. A local torch.cuda.empty_cache() on
    # explicit reload is fine (unlike mm.soft_empty_cache(), which touches
    # ComfyUI's global manager and is disallowed here).
    if cached["model"] is not None:
        cached["model"] = None
        cached["device"] = None
        cached["dtype"] = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ckpt_dir = ensure_woosh_ae()
    logger.info(f"Loading Woosh-AE from {ckpt_dir} on {target_device} ({dtype})")
    ae = AudioAutoEncoder(LoadConfig(path=ckpt_dir))
    # The constructor only sets up the graph; weights are pulled from disk
    # by this explicit call (mirrors upstream test_Woosh-AE.py).
    ae.load_from_config()
    ae = ae.to(target_device).to(dtype).eval()
    ae.requires_grad_(False)

    cached.update(model=ae, device=target_device, dtype=dtype)
    return ae


class FoleyTuneWooshVAERoundTrip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["fp32", "fp16"], {"default": "fp32"}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "round_trip"
    CATEGORY = "FoleyTune/Utils"
    DESCRIPTION = "Encode->decode audio through Woosh-AE (VOCOS, 128-d, 48 kHz). For A/B comparison against DAC-VAE."

    def round_trip(self, audio, device, dtype):
        wave, _, orig_len = _prep_audio(audio)
        orig_len_48k = wave.shape[-1]

        # Woosh VOCOS hop_length=480. Pad to 6x hop for safety margin over
        # any unknown upsampling/downsampling ratios. Silence is trimmed by _finalize.
        wave, _ = _pad_to_min(wave, 2880)

        ae = _get_woosh_ae(device, dtype)
        torch_dtype = {"fp32": torch.float32, "fp16": torch.float16}[dtype]
        x = wave.to(ae.device_any() if hasattr(ae, 'device_any') else next(ae.parameters()).device).to(torch_dtype)

        with torch.no_grad():
            z = ae.forward(x)
            y = ae.inverse(z)

        y = y.float()
        if not torch.isfinite(y).all():
            raise RuntimeError(
                "Woosh-AE round-trip produced non-finite values (NaN/Inf). "
                "Try switching dtype to fp32 or check input audio for silence/extreme values."
            )
        return (_finalize(y, orig_len_48k),)


# --- DAC node ----------------------------------------------------------------

# DAC-VAE stride at 48 kHz: product of encoder_rates [2, 3, 4, 5, 8] = 960.
# preprocess() inside DAC auto-pads to hop_length, so strictly speaking the
# external pre-pad is belt-and-suspenders; we keep it to mirror Woosh and
# guarantee a usable minimum length (4 frames of latent) regardless of
# _prep_audio output.
_DAC_HOP_LENGTH = 960
_DAC_MIN_LEN = _DAC_HOP_LENGTH * 4  # 3840 samples @ 48 kHz (~80 ms)

_DAC_SINGLETON = {"model": None, "device": None}


def _get_dac(device: str):
    """Lazy-load + cache DAC-VAE. Reloads if the device changed.

    Uses the project's ``load_dac_any`` helper rather than the upstream
    ``DAC.load`` classmethod: the vendored DAC inherits from a stubbed
    ``audiotools.ml.BaseModel`` that does not implement ``load``, and the
    real DAC checkpoint ships with ``continuous=True`` + the specific
    encoder/decoder rates pinned in ``utils._DAC_KWARGS``.
    """
    # Try relative import first (production: loaded as part of the
    # ComfyUI custom_nodes package). Fall back to absolute for standalone
    # imports (smoke tests that just prepend the repo dir to sys.path).
    try:
        from .utils import load_dac_any
    except ImportError:
        from utils import load_dac_any

    target_device = torch.device(device)
    cached = _DAC_SINGLETON
    if cached["model"] is not None and cached["device"] == target_device:
        return cached["model"]

    # Free old model before loading a new one on device change so we don't
    # hold 2x memory transiently (mirrors _get_woosh_ae).
    if cached["model"] is not None:
        cached["model"] = None
        cached["device"] = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    weights_path = os.path.join(folder_paths.models_dir, "foley", "vae_128d_48k.pth")
    if not os.path.exists(weights_path):
        from huggingface_hub import hf_hub_download
        logger.info("DAC-VAE weights not found. Downloading from Tencent/HunyuanVideo-Foley")
        weights_path = hf_hub_download(
            repo_id="Tencent/HunyuanVideo-Foley",
            filename="vae_128d_48k.pth",
            local_dir=os.path.join(folder_paths.models_dir, "foley"),
        )

    logger.info(f"Loading DAC-VAE from {weights_path} on {target_device} (fp32)")
    dac = load_dac_any(weights_path, device=target_device)
    dac.requires_grad_(False)

    cached.update(model=dac, device=target_device)
    return dac


class FoleyTuneDACVAERoundTrip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "round_trip"
    CATEGORY = "FoleyTune/Utils"
    DESCRIPTION = "Encode->decode audio through HunyuanVideo-Foley's DAC-VAE (128-d, 48 kHz). For A/B comparison against Woosh-AE."

    def round_trip(self, audio, device):
        wave, _, _ = _prep_audio(audio)
        orig_len_48k = wave.shape[-1]

        # Guarantee at least 4 latent frames (see _DAC_MIN_LEN above).
        wave, _ = _pad_to_min(wave, _DAC_MIN_LEN)

        dac = _get_dac(device)
        x = wave.to(next(dac.parameters()).device).float()

        with torch.no_grad():
            # DAC with continuous=True: encode() returns a 5-tuple
            # (z, codes, latents, commitment_loss, codebook_loss) where `z`
            # is a DiagonalGaussianDistribution. Use .mode() (the posterior
            # mean) for deterministic, reproducible round-trips — this is
            # what audio2audio uses for the same reason.
            z_dist, _, _, _, _ = dac.encode(x)
            z = z_dist.mode()
            y = dac.decode(z)

        y = y.float()
        if not torch.isfinite(y).all():
            raise RuntimeError(
                "DAC-VAE round-trip produced non-finite values (NaN/Inf). "
                "Check input audio for silence/extreme values."
            )
        return (_finalize(y, orig_len_48k),)
