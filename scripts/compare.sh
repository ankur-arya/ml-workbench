#!/usr/bin/env bash
# Compare several sklearn models and log nested MLflow runs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}"
exec python -m workbench compare "$@"
