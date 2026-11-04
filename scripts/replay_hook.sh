#!/usr/bin/env bash
set -euo pipefail

HOOK_FILE="${1:-}"
if [[ -z "$HOOK_FILE" || "$HOOK_FILE" == "-h" || "$HOOK_FILE" == "--help" ]]; then
  cat >&2 <<'USAGE'
Usage: scripts/replay_hook.sh <path/to/hook.json>

POSTs a previously saved Vexa "meeting.completed" payload back to the running worker,
triggering a full re-process (ASR + LLM) of the underlying recording.

Env:
  FRED_LIVE_WORKER_URL  Base URL of the worker (default: http://127.0.0.1:18088)
  FRED_LIVE_AUTH_TOKEN  Auth token (default: read from .env.worker if present)

Example:
  scripts/replay_hook.sh data/meetings/2/20260415T190607Z/hook.json
USAGE
  exit 1
fi

if [[ ! -f "$HOOK_FILE" ]]; then
  echo "File not found: $HOOK_FILE" >&2
  exit 1
fi

WORKER_URL="${FRED_LIVE_WORKER_URL:-http://127.0.0.1:18088}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKEN="${FRED_LIVE_AUTH_TOKEN:-}"
if [[ -z "$TOKEN" && -f "$ROOT_DIR/.env.worker" ]]; then
  TOKEN="$(grep -E '^FRED_LIVE_AUTH_TOKEN=' "$ROOT_DIR/.env.worker" | head -n1 | cut -d= -f2-)"
fi

CURL_AUTH=()
if [[ -n "$TOKEN" ]]; then
  CURL_AUTH=(-H "X-Fred-Live-Token: $TOKEN")
fi

# Sign the payload like Vexa does (HMAC-SHA256 over "{ts}.{body}") when the
# worker enforces FRED_LIVE_WEBHOOK_SECRET.
SECRET="${FRED_LIVE_WEBHOOK_SECRET:-}"
if [[ -z "$SECRET" && -f "$ROOT_DIR/.env.worker" ]]; then
  SECRET="$(grep -E '^FRED_LIVE_WEBHOOK_SECRET=' "$ROOT_DIR/.env.worker" | head -n1 | cut -d= -f2-)"
fi
CURL_SIG=()
if [[ -n "$SECRET" ]]; then
  TS="$(date +%s)"
  SIG="$( { printf '%s.' "$TS"; cat "$HOOK_FILE"; } | openssl dgst -sha256 -hmac "$SECRET" -hex | sed 's/^.*= //')"
  CURL_SIG=(-H "X-Webhook-Signature: sha256=$SIG" -H "X-Webhook-Timestamp: $TS")
fi

curl --fail --show-error --silent \
  -X POST \
  -H "Content-Type: application/json" \
  "${CURL_AUTH[@]}" \
  "${CURL_SIG[@]}" \
  --data-binary "@$HOOK_FILE" \
  "${WORKER_URL%/}/internal/vexa/meeting-completed"
printf '\n'
