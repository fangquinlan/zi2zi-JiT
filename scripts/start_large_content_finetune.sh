#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

python3 scripts/start_large_content_finetune.py \
  --dataset-dir data/mixed_finetune_dataset \
  --base-checkpoint models/zi2zi-JiT-L-16.pth \
  "$@"
