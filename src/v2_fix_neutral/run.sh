#!/usr/bin/env bash
# Wrapper around build_final_dataset.py
#
# Full pipeline (swap → ROT/158 → Canada/10 → story↔background validate → explanations):
#   bash src/v2_fix_neutral/run.sh
#   bash src/v2_fix_neutral/run.sh --workers 32
#
# Sample / smoke test:
#   bash src/v2_fix_neutral/run.sh --sample_size 25 --workers 16
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
python build_final_dataset.py "$@"
