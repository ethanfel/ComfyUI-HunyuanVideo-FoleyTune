"""Standalone clip labeler — n-stage prompt builder via keyboard.

Usage:
    python server.py --root /path/to/mp4_dir --prompts prompts.bj.json [--port 8765]

Scans <root> for .mp4 files, serves a web UI that plays each clip and accepts
keyboard input to compose a prompt from a JSON-defined n-stage template.
Saves <clipname>.txt sidecars next to each .mp4.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory


VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}


def _load_prompts(path: Path) -> dict:
    with path.open() as f:
        cfg = json.load(f)
    if "prompt_template" not in cfg or "stages" not in cfg:
        raise ValueError("prompts JSON must have 'prompt_template' and 'stages'")
    for stage in cfg["stages"]:
        for required in ("id", "title", "options"):
            if required not in stage:
                raise ValueError(f"stage missing '{required}': {stage}")
    return cfg


def _load_train_stems(dataset_json: Path) -> list[str]:
    """Read train stems from a dataset.json (filter-output format)."""
    with dataset_json.open() as f:
        cfg = json.load(f)
    if isinstance(cfg, dict) and isinstance(cfg.get("train"), list):
        return cfg["train"]
    raise ValueError(f"dataset.json must have a 'train' list: {dataset_json}")


def _build_stem_index(root: Path) -> dict[str, Path]:
    """Map clip stem -> video path for all videos under root (recursive)."""
    index = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            index[p.stem] = p
    return index


def _scan_clips(root: Path, stem_index: dict[str, Path],
                train_stems: list[str]) -> list[dict]:
    clips = []
    for stem in train_stems:
        p = stem_index.get(stem)
        if p is None:
            continue
        txt_path = p.with_suffix(".txt")
        current = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
        clips.append({
            "name": p.name,
            "rel": str(p.relative_to(root)),
            "tagged": bool(current),
            "current_prompt": current,
        })
    return clips


def _compose(template: str, selections: dict[str, str]) -> str:
    """Substitute {stage_id} placeholders, dropping any whose value is empty."""
    out = template
    for k, v in selections.items():
        out = out.replace("{" + k + "}", v if v else "")
    # Collapse `, , ` from empty substitutions and trim leading/trailing commas
    out = re.sub(r",\s*,", ",", out)
    out = re.sub(r",\s*$", "", out).strip(", ").strip()
    return out


def _match_existing(prompt: str, stages: list[dict]) -> dict[str, str]:
    """Best-effort: pre-select options whose 'value' substring is in prompt."""
    selections = {}
    for stage in stages:
        for opt in stage["options"]:
            if opt["value"] and opt["value"] in prompt:
                selections[stage["id"]] = opt["key"]
                break
    return selections


def create_app(root: Path, prompts_cfg: dict, train_stems: list[str]) -> Flask:
    app = Flask(__name__, static_folder=None)
    here = Path(__file__).parent
    stem_index = _build_stem_index(root)
    missing = [s for s in train_stems if s not in stem_index]
    if missing:
        print(f"Warning: {len(missing)} train stems not found in {root} "
              f"(first 5: {missing[:5]})")

    @app.route("/")
    def index():
        return send_from_directory(str(here), "index.html")

    @app.route("/api/config")
    def api_config():
        return jsonify(prompts_cfg)

    @app.route("/api/clips")
    def api_clips():
        clips = _scan_clips(root, stem_index, train_stems)
        for c in clips:
            c["preselect"] = _match_existing(c["current_prompt"], prompts_cfg["stages"])
        return jsonify({
            "clips": clips,
            "total": len(clips),
            "tagged": sum(1 for c in clips if c["tagged"]),
        })

    @app.route("/clips/<path:clip_rel>")
    def serve_clip(clip_rel: str):
        # Defend against path traversal
        target = (root / clip_rel).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            abort(403)
        if not target.exists() or target.suffix.lower() not in VIDEO_EXTS:
            abort(404)
        return send_from_directory(str(root), clip_rel, conditional=True)

    @app.route("/api/save", methods=["POST"])
    def api_save():
        data = request.get_json(force=True)
        rel = data.get("rel", "")
        selections = data.get("selections", {})
        if not rel:
            return jsonify({"error": "missing rel"}), 400

        target = (root / rel).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            abort(403)
        if not target.exists():
            return jsonify({"error": "clip not found"}), 404

        # Map stage option keys back to values
        values = {}
        for stage in prompts_cfg["stages"]:
            sel_key = selections.get(stage["id"])
            opt = next((o for o in stage["options"] if o["key"] == sel_key), None)
            values[stage["id"]] = opt["value"] if opt else ""

        prompt = _compose(prompts_cfg["prompt_template"], values)
        txt = target.with_suffix(".txt")
        txt.write_text(prompt + "\n", encoding="utf-8")
        return jsonify({"ok": True, "prompt": prompt, "path": str(txt)})

    @app.route("/api/delete", methods=["POST"])
    def api_delete():
        """Remove a clip's .txt sidecar (undo)."""
        data = request.get_json(force=True)
        rel = data.get("rel", "")
        target = (root / rel).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            abort(403)
        txt = target.with_suffix(".txt")
        if txt.exists():
            txt.unlink()
        return jsonify({"ok": True})

    @app.route("/api/delete-clip", methods=["POST"])
    def api_delete_clip():
        """Delete a clip and all same-stem siblings (.txt, .npz, .flac, etc)."""
        data = request.get_json(force=True)
        rel = data.get("rel", "")
        if not rel:
            return jsonify({"error": "missing rel"}), 400
        target = (root / rel).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            abort(403)
        if not target.exists():
            return jsonify({"error": "clip not found"}), 404

        deleted = []
        stem = target.stem
        parent = target.parent
        for sibling in parent.iterdir():
            if sibling.stem == stem and sibling.is_file():
                sibling.unlink()
                deleted.append(sibling.name)
        return jsonify({"ok": True, "deleted": deleted})

    return app


def main():
    ap = argparse.ArgumentParser(
        description="Clip labeler — tag training clips with keyboard-driven prompts.",
        usage="%(prog)s DATASET_JSON MP4_ROOT [--prompts FILE] [--port N]",
    )
    here = Path(__file__).parent
    ap.add_argument("dataset_json", help="Path to dataset.json (must have a 'train' list)")
    ap.add_argument("mp4_root", help="Directory containing .mp4 clips (searched recursively)")
    ap.add_argument("--prompts", default=str(here / "prompts.bj.json"),
                    help="Path to prompts JSON (default: ./prompts.bj.json)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    ds_path = Path(args.dataset_json).resolve()
    if not ds_path.is_file():
        sys.exit(f"dataset.json not found: {ds_path}")
    root = Path(args.mp4_root).resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")
    prompts_path = Path(args.prompts).resolve()
    if not prompts_path.is_file():
        sys.exit(f"Prompts JSON not found: {prompts_path}")

    train_stems = _load_train_stems(ds_path)
    print(f"Loaded {len(train_stems)} train clips from {ds_path}")

    cfg = _load_prompts(prompts_path)
    app = create_app(root, cfg, train_stems)
    print(f"Labeler serving {root}")
    print(f"Open http://{args.host}:{args.port}/ in a browser")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
