#!/usr/bin/env bash
# Start the local MLflow tracking UI against the workbench SQLite store.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
HOST="${MLFLOW_HOST:-127.0.0.1}"
PORT="${MLFLOW_PORT:-5000}"
BACKEND="${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}"
echo "MLflow UI  backend=${BACKEND}  http://${HOST}:${PORT}"
exec mlflow ui --backend-store-uri "$BACKEND" --host "$HOST" --port "$PORT"
