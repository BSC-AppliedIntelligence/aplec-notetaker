#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${APLEC_LLM_GGUF_PATH:-$ROOT_DIR/models/unsloth__Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf}"
# Loopback by default: the worker connects to 127.0.0.1:18081. Bind anything
# wider only together with an API key (below): an open model server is GPU DoS
# plus arbitrary prompts for anyone who can reach it.
HOST="${APLEC_LLAMACPP_HOST:-127.0.0.1}"
PORT="${APLEC_LLAMACPP_PORT:-18081}"
MODEL_ALIAS="${APLEC_LLAMACPP_MODEL_ALIAS:-qwen3.8-27b}"
CTX_SIZE="${APLEC_LLAMACPP_CTX_SIZE:-65536}"
GPU_LAYERS="${APLEC_LLAMACPP_GPU_LAYERS:-999}"
PARALLEL="${APLEC_LLAMACPP_PARALLEL:-1}"
# 0 disables reasoning/thinking server-side (model-agnostic); empty = model default.
REASONING_BUDGET="${APLEC_LLAMACPP_REASONING_BUDGET:-}"
# Speculative decoding: the model's official MTP drafter (auto-enabled when the
# file exists; set APLEC_LLAMACPP_DRAFT_GGUF="" to disable). Output is
# verified against the target model, so it is bit-identical to non-speculative
# decoding; this is purely a speed feature (~1GB extra memory).
# NOTE: a drafter belongs to ONE model family (vocabulary must match). When
# changing MODEL_PATH, change this too or the server refuses to start.
DEFAULT_DRAFT_GGUF="$ROOT_DIR/models/unsloth__Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
DRAFT_GGUF="${APLEC_LLAMACPP_DRAFT_GGUF-$DEFAULT_DRAFT_GGUF}"
SPEC_TYPE="${APLEC_LLAMACPP_SPEC_TYPE:-draft-mtp}"
SPEC_DRAFT_MAX="${APLEC_LLAMACPP_SPEC_DRAFT_MAX:-4}"
LLAMA_SERVER_BIN="${APLEC_LLAMACPP_BIN:-$ROOT_DIR/vendor/llama.cpp/build/bin/llama-server}"

if [[ -x "$LLAMA_SERVER_BIN" ]]; then
  :
elif command -v llama-server >/dev/null 2>&1; then
  LLAMA_SERVER_BIN="$(command -v llama-server)"
else
  echo "llama-server not found at $LLAMA_SERVER_BIN or in PATH" >&2
  exit 1
fi

export GGML_CUDA_ENABLE_UNIFIED_MEMORY="${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}"

# Optional API keys (one per line). Clients then send "Authorization: Bearer
# <key>"; the worker reads its key from APLEC_LLM_API_KEY. A configured but
# missing file is fatal: silently starting without auth would expose the server.
API_KEY_FILE="${APLEC_LLAMACPP_API_KEY_FILE:-}"

EXTRA_ARGS=()
if [[ -n "$API_KEY_FILE" ]]; then
  if [[ ! -r "$API_KEY_FILE" ]]; then
    echo "APLEC_LLAMACPP_API_KEY_FILE=$API_KEY_FILE is not readable; refusing to start without auth" >&2
    exit 1
  fi
  EXTRA_ARGS+=(--api-key-file "$API_KEY_FILE")
fi
if [[ -n "$REASONING_BUDGET" ]]; then
  EXTRA_ARGS+=(--reasoning-budget "$REASONING_BUDGET")
fi
if [[ -n "$DRAFT_GGUF" && -f "$DRAFT_GGUF" ]]; then
  EXTRA_ARGS+=(-md "$DRAFT_GGUF" --spec-type "$SPEC_TYPE" --spec-draft-n-max "$SPEC_DRAFT_MAX")
fi

exec "$LLAMA_SERVER_BIN" \
  -m "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --alias "$MODEL_ALIAS" \
  -c "$CTX_SIZE" \
  --parallel "$PARALLEL" \
  --n-gpu-layers "$GPU_LAYERS" \
  "${EXTRA_ARGS[@]}"
