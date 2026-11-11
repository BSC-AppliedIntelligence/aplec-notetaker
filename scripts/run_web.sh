#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${APLEC_WEB_HOST:-127.0.0.1}"
PORT="${APLEC_WEB_PORT:-18091}"

exec "$ROOT_DIR/.venv/bin/uvicorn" aplec_web.app:app --host "$HOST" --port "$PORT"
