#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The worker reads .env.worker from its working directory.
cd "$ROOT_DIR"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"

if ! command -v ffmpeg >/dev/null 2>&1 && [[ -x "$VENV_PYTHON" ]]; then
  FFMPEG_BIN="$("$VENV_PYTHON" - <<'PY'
try:
    import imageio_ffmpeg
except Exception:
    print("")
else:
    print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)"
  if [[ -n "$FFMPEG_BIN" && -x "$FFMPEG_BIN" ]]; then
    ln -sf "$FFMPEG_BIN" "$ROOT_DIR/.venv/bin/ffmpeg"
    export PATH="$ROOT_DIR/.venv/bin:$PATH"
  fi
fi

# The venv's own uvicorn: the PATH above only gains .venv/bin when ffmpeg is missing.
UVICORN="$ROOT_DIR/.venv/bin/uvicorn"
[[ -x "$UVICORN" ]] || UVICORN=uvicorn

exec "$UVICORN" \
  aplec_worker.main:app \
  --app-dir "$ROOT_DIR" \
  --host "${APLEC_APP_HOST:-0.0.0.0}" \
  --port "${APLEC_APP_PORT:-18088}"
