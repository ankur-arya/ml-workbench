#!/usr/bin/env bash
# Train one experiment. Extra args are forwarded to `python -m workbench train`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}"
exec python -m workbench train "$@"
