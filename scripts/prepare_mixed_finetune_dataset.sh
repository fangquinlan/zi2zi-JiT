#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

python3 scripts/prepare_mixed_finetune_dataset.py \
  --input-root train \
  --output-dir data/mixed_finetune_dataset \
  --force \
  "$@"
