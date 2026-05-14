#!/usr/bin/env bash
# Build ox-astro-ratio (compress + decompress)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" > /dev/null
cmake --build build --target compress_astro_ratio decompress_astro_ratio -j"$(nproc)"
echo "Built: build/compress_astro_ratio  build/decompress_astro_ratio"
