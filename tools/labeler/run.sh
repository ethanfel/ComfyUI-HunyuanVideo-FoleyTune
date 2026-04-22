#!/usr/bin/env bash
# One-shot launcher. Edit ROOT below for other datasets.
# Usage:  ./run.sh           # serves on 8765
#         ./run.sh 9000      # custom port
#         ./run.sh 9000 path/to/other/prompts.json

set -euo pipefail

ROOT="${LABELER_ROOT:-/media/unraid/davinci/Foley/AD/blowjob/mp4}"
PORT="${1:-8765}"
PROMPTS="${2:-$(dirname "$0")/prompts.bj.json}"

cd "$(dirname "$0")"

# Open browser after server is reachable
( for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${PORT}/" > /dev/null; then
      xdg-open "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 || true
      break
    fi
    sleep 0.2
  done ) &

exec python server.py --root "$ROOT" --prompts "$PROMPTS" --port "$PORT"
