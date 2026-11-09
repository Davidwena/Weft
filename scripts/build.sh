#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${WEFT_BUILD_DIR:-"$ROOT/build"}
cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$BUILD_DIR" --parallel "${JOBS:-4}"
