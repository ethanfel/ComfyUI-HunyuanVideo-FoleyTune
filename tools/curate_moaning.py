#!/usr/bin/env python3
"""Curate a feature dataset: keep moaning-dominant clips, drop too-breathy ones.

Moaning is VOICED (periodic/harmonic); breathing is UNVOICED (aperiodic noise).
Per clip we compute `moan_score` = the fraction of AUDIBLE energy that lands in
voiced frames (autocorrelation periodicity above a threshold). Breathy clips score
low. Level is ignored (clips are LUFS-normalized), voicing is not.

Dry-run by default: prints the score distribution + lowest/highest example clips so
you can spot-check by ear and pick a threshold. Add --apply to build the clone dir.

The clone hardlinks (fallback symlink) the kept clips' .npz/.flac into a new dir and
writes a filtered dataset.json — no data duplication, self-contained for training.

Usage:
  python tools/curate_moaning.py <src_features_dir>                 # dry-run, show distribution
  python tools/curate_moaning.py <src_features_dir> -t 0.5          # dry-run at threshold 0.5
  python tools/curate_moaning.py <src_features_dir> -t 0.5 --apply  # build the clone
  python tools/curate_moaning.py <src> -t 0.5 --apply --out /path/to/custom_dir
"""
import argparse, json, os, glob, sys, shutil
import numpy as np
import soundfile as sf

AUDIO_EXTS = (".flac", ".wav", ".ogg", ".aiff", ".aif")


def moan_score(path, nfft=2048, hop=512, fmin=80, fmax=500, gate_rel=0.15, voice_thr=0.45):
    """Fraction of audible energy that is voiced (moaning). 0=all breath/noise, 1=all voiced."""
    w, sr = sf.read(path)
    if w.ndim > 1:
        w = w.mean(1)
    w = w.astype(np.float64)
    if len(w) < nfft:
        return None
    win = np.hanning(nfft)
    n = 1 + (len(w) - nfft) // hop
    frames = np.stack([w[i * hop:i * hop + nfft] * win for i in range(n)])  # (n, nfft)
    rms = np.sqrt((frames ** 2).mean(1))
    if rms.max() <= 0:
        return 0.0
    gated = rms > rms.max() * gate_rel                    # ignore silence between sounds
    # FFT-based autocorrelation (fast, vectorized over frames)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    ac = np.fft.irfft(power, axis=1)                       # (n, nfft)
    ac0 = ac[:, 0].copy(); ac0[ac0 <= 0] = 1e-9
    acn = ac / ac0[:, None]
    lo, hi = int(sr / fmax), int(sr / fmin)
    peak = acn[:, lo:hi].max(axis=1) if hi > lo else np.zeros(n)
    voiced = gated & (peak > voice_thr)                   # periodic + audible = moaning
    te = rms[gated].sum()
    return float(rms[voiced].sum() / (te + 1e-9))


def find_audio(d, stem):
    for ext in AUDIO_EXTS:
        p = os.path.join(d, stem + ext)
        if os.path.exists(p):
            return p
    return None


def place(src, dst, mode):
    """copy (portable across machines), hardlink (same fs, no space), or symlink."""
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)            # fallback if cross-device/FUSE
    else:  # symlink (NOT portable if mounts differ across machines)
        os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-t", "--threshold", type=float, default=0.5,
                    help="keep clips with moan_score >= threshold (default 0.5)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: src with _features -> _nobreath_features)")
    ap.add_argument("--apply", action="store_true", help="actually build the clone (else dry-run)")
    ap.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy",
                    help="populate the clone via copy (default, portable across machines), "
                         "hardlink (same fs, no extra disk), or symlink (NOT cross-machine safe)")
    args = ap.parse_args()

    src = args.src.rstrip("/")
    if not os.path.isfile(os.path.join(src, "dataset.json")):
        sys.exit(f"No dataset.json in {src}")
    ds = json.load(open(os.path.join(src, "dataset.json")))
    train = ds.get("train", [])

    scores = {}
    for stem in train:
        ap_ = find_audio(src, stem)
        if ap_ is None:
            print(f"  WARN no audio for {stem}, skipping"); continue
        s = moan_score(ap_)
        if s is not None:
            scores[stem] = s
    if not scores:
        sys.exit("No scorable clips.")

    vals = np.array(list(scores.values()))
    pct = lambda p: np.percentile(vals, p)
    print(f"\n{src.split('/')[-1]}: {len(scores)} clips scored (moan_score = voiced-energy fraction)")
    print(f"  distribution: min {vals.min():.2f}  p10 {pct(10):.2f}  p25 {pct(25):.2f}  "
          f"median {pct(50):.2f}  p75 {pct(75):.2f}  max {vals.max():.2f}")
    ordered = sorted(scores.items(), key=lambda kv: kv[1])
    print("  --- 8 LOWEST (most breathy, drop candidates) — spot-check these by ear:")
    for stem, s in ordered[:8]:
        print(f"      {s:.2f}  {stem}")
    print("  --- 8 HIGHEST (most moaning):")
    for stem, s in ordered[-8:]:
        print(f"      {s:.2f}  {stem}")
    boundary = sorted(scores.items(), key=lambda kv: abs(kv[1] - args.threshold))[:8]
    print(f"  --- 8 NEAREST threshold {args.threshold} (the cutoff judgment calls — listen to set -t):")
    for stem, s in sorted(boundary, key=lambda kv: kv[1]):
        print(f"      {s:.2f}  {stem}")

    keep = [stem for stem, s in scores.items() if s >= args.threshold]
    drop = [stem for stem, s in scores.items() if s < args.threshold]
    print(f"\n  threshold {args.threshold}: KEEP {len(keep)} / DROP {len(drop)} "
          f"({100*len(drop)/len(scores):.0f}% dropped)")

    out = args.out or src.replace("_features", "_nobreath_features")
    if out == src:
        out = src + "_nobreath"
    if not args.apply:
        print(f"\n  DRY-RUN. Would write {len(keep)} clips -> {out}")
        print("  Re-run with --apply (tune -t first by listening to the lowest-score clips).")
        return

    os.makedirs(out, exist_ok=True)
    linked = 0
    for stem in keep:
        npz = os.path.join(src, stem + ".npz")
        if os.path.exists(npz):
            dst = os.path.join(out, stem + ".npz")
            if not os.path.exists(dst):
                place(npz, dst, args.mode)
        a = find_audio(src, stem)
        if a:
            dst = os.path.join(out, os.path.basename(a))
            if not os.path.exists(dst):
                place(a, dst, args.mode)
        linked += 1
    new_ds = dict(ds)
    new_ds["train"] = keep
    json.dump(new_ds, open(os.path.join(out, "dataset.json"), "w"), indent=2)
    print(f"\n  APPLIED ({args.mode}): {linked} clips -> {out} (dataset.json written, {len(drop)} dropped)")


if __name__ == "__main__":
    main()
