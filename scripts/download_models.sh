#!/usr/bin/env bash
# Downloads the model weights Aplec reads from models/. Resumable, and skips files
# that are already there.
#
#   ./scripts/download_models.sh                Whisper large-v3 for whisper.cpp (~3 GB)
#   ./scripts/download_models.sh --llm          + the local summarizer, Qwen3.8-27B (UD-Q6_K),
#                                                 and its speculative-decoding drafter (~23 GB)
#   ./scripts/download_models.sh --voiceprints  + ReDimNet2 for voiceprints (needs wespeaker)
#
# Not here: pyannote's diarization model and the SpeechBrain models download
# themselves on first use (pyannote's is gated: accept its conditions on Hugging
# Face and set FRED_LIVE_HF_TOKEN), and the optional second diarizer comes with
# scripts/setup_nemotron.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${FRED_LIVE_MODELS_DIR:-$ROOT_DIR/models}"

WANT_LLM=0
WANT_VOICEPRINTS=0
for arg in "$@"; do
  case "$arg" in
    --llm) WANT_LLM=1 ;;
    --voiceprints) WANT_VOICEPRINTS=1 ;;
    *) echo "usage: $0 [--llm] [--voiceprints]" >&2; exit 2 ;;
  esac
done

download() {  # <hf repo> <file in the repo> <local path>
  local repo="$1" remote="$2" local_path="$3"
  mkdir -p "$(dirname "$local_path")"
  if [[ -s "$local_path" ]]; then
    echo "already there: $local_path"
    return
  fi
  echo "downloading $repo/$remote"
  curl --fail --location --retry 5 --continue-at - \
    "https://huggingface.co/$repo/resolve/main/$remote" \
    --output "$local_path.part"
  mv "$local_path.part" "$local_path"
}

download ggerganov/whisper.cpp ggml-large-v3.bin "$MODELS_DIR/ggml-whisper/ggml-large-v3.bin"

if (( WANT_LLM )); then
  QWEN="$MODELS_DIR/unsloth__Qwen3.8-27B-GGUF"
  download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q6_K.gguf "$QWEN/Qwen3.8-27B-UD-Q6_K.gguf"
  download unsloth/Qwen3.8-27B-GGUF MTP/mtp-Qwen3.8-27B-Q4_0.gguf "$QWEN/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
fi

if (( WANT_VOICEPRINTS )); then
  REDIMNET="$MODELS_DIR/wespeaker__voxceleb-redimnet2-B6-LM"
  download Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM avg_model.pt "$REDIMNET/avg_model.pt"
  download Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM config.yaml "$REDIMNET/config.yaml"
fi

echo "models under $MODELS_DIR"
