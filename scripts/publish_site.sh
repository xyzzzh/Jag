#!/usr/bin/env bash
# Publish the static project page without build dependencies.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
if [[ -n "$(git status --porcelain -- docs/site)" ]]; then
  echo "Commit docs/site changes before publishing." >&2
  exit 1
fi
git subtree push --prefix docs/site origin gh-pages
