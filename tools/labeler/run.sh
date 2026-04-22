#!/usr/bin/env bash
# Usage:  ./run.sh DATASET_JSON MP4_ROOT [PORT] [PROMPTS_JSON]
#
# Example:
#   ./run.sh /media/unraid/davinci/Foley/AD/blowjob/features_v3/dataset.json \
#            /media/unraid/davinci/Foley/AD/blowjob/mp4

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 DATASET_JSON MP4_ROOT [PORT] [PROMPTS_JSON]"
  exit 1
fi

DATASET_JSON="$1"
MP4_ROOT="$2"
PORT="${3:-8765}"
PROMPTS="${4:-$(dirname "$0")/prompts.bj.json}"

cd "$(dirname "$0")"

# Open browser after server is reachable
( for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${PORT}/" > /dev/null; then
      xdg-open "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 || true
      break
    fi
    sleep 0.2
  done ) &

exec python server.py "$DATASET_JSON" "$MP4_ROOT" --prompts "$PROMPTS" --port "$PORT"
