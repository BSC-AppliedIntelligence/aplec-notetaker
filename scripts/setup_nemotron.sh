#!/usr/bin/env bash
# Environment and weights of the second diarizer, which runs as a subprocess of
# the worker (tools/nemotron/diarize.py, FRED_LIVE_DIARIZATION_NEMOTRON). Run once,
# after the worker's .venv exists. Downloads ~700 MB of packages and 379 MB of
# weights; rerunning skips what is already there.
#
#   ./scripts/setup_nemotron.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv-nemotron"
MODEL="${FRED_LIVE_NEMOTRON_MODEL_DIR:-$ROOT/models/nvidia__Nemotron-3-Diarization}"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "the worker's .venv is missing: set it up first (README.md, Quickstart)" >&2
    exit 1
fi

python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --requirement "$ROOT/requirements.nemotron.txt"

# torch and accelerate come from the worker: several GB, and the CUDA build known
# to work here. A .pth adds its site-packages AFTER this environment's own, so
# transformers 5 wins and everything missing here is found there.
worker_site="$("$ROOT/.venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
own_site="$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
echo "$worker_site" > "$own_site/zz_worker_packages.pth"

"$VENV/bin/python" - "$MODEL" <<'PY'
import sys

from huggingface_hub import snapshot_download

snapshot_download(
    "nvidia/Nemotron-3-Diarization",
    local_dir=sys.argv[1],
    allow_patterns=["config.json", "processor_config.json", "model.safetensors"],
)
PY
"$VENV/bin/python" -c 'import torch, transformers; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__)'
echo "ready: set FRED_LIVE_DIARIZATION_NEMOTRON=true in .env.worker and restart the worker"
