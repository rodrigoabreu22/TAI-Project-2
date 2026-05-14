#!/usr/bin/env bash
# Build all three compressors (ratio + balanced + fast)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" > /dev/null
cmake --build build -j"$(nproc)"
echo "Built all targets in build/"
ls -1 build/compress_astro_* build/decompress_astro_*
