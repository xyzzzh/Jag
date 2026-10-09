#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_stamp="$(date -u +%Y%m%dT%H%M%S)-$$"
bash "$task_root/scripts/docker.sh" --eval run --rm jag \
  python scripts/prepare_cost_set.py --output "/outputs/benchmark/$task_stamp/data"
exec bash "$task_root/scripts/docker.sh" --eval run --rm jag \
  python -m jag.benchmark \
  --base-model /models/Qwen3.5-0.8B --checkpoint /models/Jag \
  --jsonl "/outputs/benchmark/$task_stamp/data/cost.jsonl" \
  --warmup-jsonl "/outputs/benchmark/$task_stamp/data/warmup.jsonl" \
  --source-order --weight-dtype bf16 \
  --output "/outputs/benchmark/$task_stamp" --device cuda:0 \
  "$@"
