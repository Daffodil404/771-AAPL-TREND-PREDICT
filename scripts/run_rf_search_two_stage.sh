#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$ROOT_DIR/src/train/train_rf_search.py" --mode fast
python "$ROOT_DIR/src/train/train_rf_search.py" --mode slow
