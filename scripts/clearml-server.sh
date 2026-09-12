#!/usr/bin/env bash
# Start / stop the local ClearML Server (WebApp on :8080).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_DIR="$ROOT/deploy/clearml"
COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"

usage() {
  cat <<'EOF'
Usage: ./scripts/clearml-server.sh <up|down|status|logs|init-config>

  up           Start ClearML Server (WebApp, API, files, mongo, redis, elastic)
  down         Stop the stack (keeps ./deploy/clearml/data)
  status       docker compose ps
  logs         Tail compose logs
  init-config  Print env exports + remind how to write ~/clearml.conf

WebApp:  http://localhost:8080
API:     http://localhost:8008
Files:   http://localhost:8081
EOF
}

compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose -f "$COMPOSE_FILE" "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose -f "$COMPOSE_FILE" "$@"
  else
    echo "docker compose is required. Install Docker Engine + Compose v2." >&2
    exit 1
  fi
}

check_vm_map() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    return 0
  fi
  local current
  current="$(sysctl -n vm.max_map_count 2>/dev/null || echo 0)"
  if [[ "${current}" -lt 262144 ]]; then
    echo "WARNING: vm.max_map_count=${current} (Elasticsearch wants >= 262144)."
    echo "         sudo sysctl -w vm.max_map_count=262144"
  fi
}

cmd="${1:-}"
case "$cmd" in
  up)
    if ! command -v docker >/dev/null 2>&1; then
      echo "docker is not installed. Install Docker, then re-run this script." >&2
      exit 1
    fi
    check_vm_map
    mkdir -p \
      "$COMPOSE_DIR/data/logs" \
      "$COMPOSE_DIR/data/config" \
      "$COMPOSE_DIR/data/fileserver" \
      "$COMPOSE_DIR/data/elastic" \
      "$COMPOSE_DIR/data/mongo/db" \
      "$COMPOSE_DIR/data/mongo/configdb" \
      "$COMPOSE_DIR/data/redis"
    compose up -d
    echo
    echo "ClearML WebApp:  http://localhost:8080"
    echo "API server:      http://localhost:8008"
    echo "File server:     http://localhost:8081"
    echo
    echo "First visit the WebApp, create a local user, then"
    echo "Settings → Workspace → Create new credentials."
    echo "Copy them into ~/clearml.conf (see deploy/clearml/clearml.conf.example)"
    echo "or export CLEARML_API_ACCESS_KEY / CLEARML_API_SECRET_KEY."
    ;;
  down)
    compose down
    ;;
  status)
    compose ps
    ;;
  logs)
    compose logs -f --tail=200
    ;;
  init-config)
    cat <<'EOF'
# Point the ClearML SDK at the local server (run after `up` and creating credentials):

export CLEARML_WEB_HOST=http://localhost:8080
export CLEARML_API_HOST=http://localhost:8008
export CLEARML_FILES_HOST=http://localhost:8081
# export CLEARML_API_ACCESS_KEY=...
# export CLEARML_API_SECRET_KEY=...

# Or: cp deploy/clearml/clearml.conf.example ~/clearml.conf  and edit the keys.
# Or: clearml-init   (paste WebApp / hosted credentials)

# Quick start without a local server — hosted ClearML:
#   clearml-init     # workspace https://app.clear.ml
#
# Offline (no UI until you import the session):
#   CLEARML_OFFLINE_MODE=1 python -m workbench train --config configs/iris_random_forest.yaml
#   # later: python -c "from clearml import Task; Task.import_offline_session('<zip>')"
EOF
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 2
    ;;
esac
