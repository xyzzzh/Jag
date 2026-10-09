#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_model="${1:-jag}"
if [[ "$task_model" != jag && "$task_model" != base ]]; then
  printf '%s\n' 'Usage: bash scripts/evaluate.sh jag|base [evaluation arguments]' >&2
  exit 2
fi
if (($#)); then shift; fi
exec bash "$task_root/scripts/docker.sh" --eval run --rm jag \
  python scripts/evaluate_all.py "$task_model" "$@"
