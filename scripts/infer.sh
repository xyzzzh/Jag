#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$task_root/scripts/docker.sh" --eval run --rm jag \
  python -m groundingjev.predict --checkpoint /models/Jag --device cuda:0 "$@"
