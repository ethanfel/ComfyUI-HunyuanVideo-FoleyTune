#!/usr/bin/env python3
"""Install voice conversion backends for FoleyTune.

Usage:
    python install_vc.py                # install all backends
    python install_vc.py ezvc           # install EZ-VC only
    python install_vc.py seedvc         # install Seed-VC only
    python install_vc.py vevo           # install Vevo only
    python install_vc.py ezvc seedvc    # install multiple

Protects ComfyUI core packages from being downgraded by pinning them
during installation. Risky packages are installed with --no-deps.
"""

import subprocess
import sys
import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOS_DIR = os.path.join(SCRIPT_DIR, "vc_repos")

# Packages that must NOT be changed — pinned to whatever is currently installed
PROTECTED = [
    "torch", "torchaudio", "torchvision", "numpy", "scipy",
    "transformers", "accelerate", "pydantic", "pydantic_core",
    "safetensors", "aiohttp", "pillow", "huggingface-hub",
    "librosa", "soundfile", "einops", "hydra-core", "omegaconf",
    "gradio", "fastapi", "starlette", "uvicorn",
]


def run(cmd, check=True):
    print(f"\n>>> {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def pip_install(*packages, no_deps=False):
    nd = " --no-deps" if no_deps else ""
    run(f"{sys.executable} -m pip install{nd} {' '.join(packages)}")


def pip_install_safe(*packages):
    """Install packages but prevent any PROTECTED package from changing."""
    constraints = []
    for pkg in PROTECTED:
        try:
            ver = subprocess.check_output(
                [sys.executable, "-m", "pip", "show", pkg],
                stderr=subprocess.DEVNULL, text=True
            )
            for line in ver.splitlines():
                if line.startswith("Version:"):
                    v = line.split(":", 1)[1].strip()
                    constraints.append(f"{pkg}=={v}")
                    break
        except subprocess.CalledProcessError:
            pass

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix="vc_constraints_") as f:
        f.write("\n".join(constraints))
        constraints_path = f.name

    try:
        run(f"{sys.executable} -m pip install -c {constraints_path} {' '.join(packages)}")
    finally:
        os.unlink(constraints_path)


def clone_repo(url, dirname):
    dest = os.path.join(REPOS_DIR, dirname)
    if os.path.isdir(dest):
        print(f"[skip] {dirname} already cloned at {dest}", flush=True)
        return dest
    os.makedirs(REPOS_DIR, exist_ok=True)
    run(f"git clone --depth 1 {url} {dest}")
    return dest


def check_system_dep(cmd, name, install_hint):
    if shutil.which(cmd) is None:
        print(f"\n[WARNING] {name} not found. Install with: {install_hint}")
        return False
    return True


def install_ezvc():
    print("\n" + "=" * 60)
    print("Installing EZ-VC backend")
    print("=" * 60, flush=True)

    check_system_dep("sox", "SoX", "sudo pacman -S sox  /  sudo apt install sox")
    check_system_dep("ffmpeg", "FFmpeg", "sudo pacman -S ffmpeg  /  sudo apt install ffmpeg")

    pip_install_safe("vocos", "x_transformers>=1.31.14", "torchdiffeq", "cached_path", "ema_pytorch>=0.5.2")

    pip_install(
        "'espnet @ git+https://github.com/wanchichen/espnet.git@ssl'",
        no_deps=True,
    )

    # Check if espnet missing sub-deps that are safe to add
    missing = []
    for mod, pkg in [("kaldiio", "kaldiio"), ("typeguard", "typeguard")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        pip_install_safe(*missing)

    repo = clone_repo("https://github.com/EZ-VC/EZ-VC.git", "EZ-VC")

    # Init submodules (BigVGAN vocoder)
    run(f"cd {repo} && git submodule update --init --recursive", check=False)

    # Make EZ-VC importable
    src_dir = os.path.join(repo, "src")
    if src_dir not in sys.path:
        print(f"\n[NOTE] Add to your Python path or PYTHONPATH: {src_dir}")

    print("\n[OK] EZ-VC installed", flush=True)


def install_seedvc():
    print("\n" + "=" * 60)
    print("Installing Seed-VC backend")
    print("=" * 60, flush=True)

    pip_install_safe("munch", "resemblyzer", "pyyaml", "python-dotenv")

    # These are heavy — install with --no-deps to avoid torch/numpy conflicts
    pip_install("descript-audio-codec", no_deps=True)
    pip_install("funasr", no_deps=True)
    pip_install("modelscope", no_deps=True)

    # Check what funasr/modelscope actually need that's missing
    safe_subdeps = []
    for mod, pkg in [
        ("oss2", "oss2"), ("datasets", "datasets"),
        ("jiwer", "jiwer"), ("editdistance", "editdistance"),
        ("dac", "descript-audio-codec"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            if pkg != "descript-audio-codec":
                safe_subdeps.append(pkg)
    if safe_subdeps:
        pip_install_safe(*safe_subdeps)

    repo = clone_repo("https://github.com/Plachtaa/seed-vc.git", "seed-vc")

    print("\n[OK] Seed-VC installed", flush=True)


def install_vevo():
    print("\n" + "=" * 60)
    print("Installing Vevo backend")
    print("=" * 60, flush=True)

    if not check_system_dep("espeak-ng", "espeak-ng",
                            "sudo pacman -S espeak-ng  /  sudo apt install espeak-ng"):
        print("[WARNING] Vevo requires espeak-ng — install it before using this backend")

    pip_install_safe("encodec", "phonemizer", "g2p_en", "openai-whisper")

    repo = clone_repo("https://github.com/open-mmlab/Amphion.git", "Amphion")

    print("\n[OK] Vevo installed", flush=True)


BACKENDS = {
    "ezvc": install_ezvc,
    "seedvc": install_seedvc,
    "vevo": install_vevo,
}


def main():
    targets = sys.argv[1:] or list(BACKENDS.keys())

    for t in targets:
        if t not in BACKENDS:
            print(f"Unknown backend: {t}. Choose from: {list(BACKENDS.keys())}")
            sys.exit(1)

    print("Protected packages (will NOT be changed):")
    for pkg in PROTECTED:
        try:
            ver = subprocess.check_output(
                [sys.executable, "-m", "pip", "show", pkg],
                stderr=subprocess.DEVNULL, text=True
            )
            for line in ver.splitlines():
                if line.startswith("Version:"):
                    print(f"  {pkg}=={line.split(':', 1)[1].strip()}")
                    break
        except subprocess.CalledProcessError:
            pass

    for t in targets:
        BACKENDS[t]()

    print("\n" + "=" * 60)
    print("Installation complete!")
    print(f"Repos cloned to: {REPOS_DIR}")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
