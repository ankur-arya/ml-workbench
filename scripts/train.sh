#!/usr/bin/env bash
# Train one experiment. Extra args are forwarded to `python -m workbench train`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CLEARML_WEB_HOST="${CLEARML_WEB_HOST:-http://localhost:8080}"
export CLEARML_API_HOST="${CLEARML_API_HOST:-http://localhost:8008}"
export CLEARML_FILES_HOST="${CLEARML_FILES_HOST:-http://localhost:8081}"
exec python -m workbench train "$@"
