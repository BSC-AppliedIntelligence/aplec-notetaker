#!/usr/bin/env bash
# Build tools/whisper-batch against the whisper.cpp build whisper-cli comes
# from, and install it next to whisper-cli, where the worker looks for it.
#
# It links the very same shared libraries, which is what keeps its output
# identical to whisper-cli's; rerun this after rebuilding or updating
# whisper.cpp, then check with scripts/verify_whisper_batch.py. Takes seconds,
# except when it first applies patches/whisper-cpp-second-language.patch (a few
# minutes: whisper.cpp recompiles libwhisper). If a future whisper.cpp refuses
# that patch, whisper-batch still builds, without the second language tag.
#
#   ./scripts/build_whisper_batch.sh
#   WHISPER_CPP_DIR=/path/to/whisper.cpp ./scripts/build_whisper_batch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHISPER_CPP_DIR="${WHISPER_CPP_DIR:-$ROOT/vendor/whisper.cpp}"
BUILD="${WHISPER_CPP_BUILD:-$WHISPER_CPP_DIR/build}"
OUT="$BUILD/bin/whisper-batch"

if [[ ! -x "$BUILD/bin/whisper-cli" ]]; then
    echo "whisper-cli not found in $BUILD/bin: build whisper.cpp first (docs/INSTALL.md, section 3)" >&2
    exit 1
fi
if [[ ! -e "$BUILD/bin/libwhisper.so" || ! -f "$BUILD/examples/libcommon.a" ]]; then
    echo "expected a shared-library build (libwhisper.so in $BUILD/bin, libcommon.a in $BUILD/examples)." >&2
    echo "A static build (-DBUILD_SHARED_LIBS=OFF) builds whisper-batch as an example instead: see Dockerfile.worker." >&2
    exit 1
fi

PATCH="$ROOT/patches/whisper-cpp-second-language.patch"
if ! grep -q WHISPER_HAS_SECOND_LANGUAGE "$WHISPER_CPP_DIR/include/whisper.h"; then
    if git -C "$WHISPER_CPP_DIR" apply --check "$PATCH" 2>/dev/null; then
        git -C "$WHISPER_CPP_DIR" apply "$PATCH"
        cmake --build "$BUILD" --target whisper whisper-cli -j"$(nproc)"
    else
        echo "warning: $PATCH does not apply to this whisper.cpp; building without the second language tag" >&2
    fi
fi

# $ORIGIN: the binary finds libwhisper/libggml* beside itself, exactly the
# ones whisper-cli loads.
g++ -O2 -std=c++17 "$ROOT/tools/whisper-batch/whisper_batch.cpp" \
    -I"$WHISPER_CPP_DIR/include" -I"$WHISPER_CPP_DIR/ggml/include" -I"$WHISPER_CPP_DIR/examples" \
    -L"$BUILD/examples" -lcommon \
    -L"$BUILD/bin" -lwhisper -lggml -lggml-base \
    -Wl,-rpath,'$ORIGIN' -lpthread \
    -o "$OUT"
echo "built $OUT"
