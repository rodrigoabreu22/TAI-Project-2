# TAI - Project 2
**Algorithmic Information Theory (2025/26) — Universidade de Aveiro**

Data-specific lossless compressor for raw astronomical images. Input is a raw raster of unsigned 16-bit integers (big-endian). Three independent compressors are provided, each targeting a different trade-off between compression ratio and speed.

| Tool | Strategy | bits/byte |
|------|----------|----------:|
| `ox-astro-ratio`    | GAP/mean adaptive predictor + 365-context spatial model + bias correction | **3.427** |
| `ox-astro-balanced` | GAP/mean adaptive predictor + 16 hi-contexts + 287 lo-contexts + tile-parallel | **3.430** |
| `ox-astro-fast`     | Horizontal delta + 2 ORDER-0 models + chunk-parallel                      | **3.608** |

## Authors

| Name | GitHub | Assignment |
|------|--------|------------|
| Eduardo Lopes | [@odraude23](https://github.com/odraude23) | `ox-astro-balanced` |
| Rodrigo Abreu | [@rodrigoabreu22](https://github.com/rodrigoabreu22) | `ox-astro-ratio` |
| Hugo Ribeiro  | [@xHuGODx](https://github.com/xHuGODx) | `ox-astro-fast` |

## Build

```bash
# Configure (first time only)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all targets
cmake --build build -j$(nproc)

# Build a single target
cmake --build build --target compress_astro_ratio -j$(nproc)
```

Produces binaries in `build/`:
- `compress_astro_ratio`    / `decompress_astro_ratio`
- `compress_astro_balanced` / `decompress_astro_balanced`
- `compress_astro_fast`     / `decompress_astro_fast`

## Usage

```bash
# Compress — explicit geometry (required interface)
./build/compress_astro_balanced n_rows n_cols <input> <output>
./build/compress_astro_ratio    n_rows n_cols <input> <output>
./build/compress_astro_fast     n_rows n_cols <input> <output>

# Compress — two-argument form (defaults to 1500×1500)
./build/compress_astro_balanced <input> <output>

# Decompress (all variants — reads geometry from header)
./build/decompress_astro_balanced <compressed> <output>
./build/decompress_astro_ratio    <compressed> <output>
./build/decompress_astro_fast     <compressed> <output>

# Example
./build/compress_astro_balanced 1000 3500 image.raw image.enc
./build/decompress_astro_balanced image.enc image.dec
```

## Data

Place the benchmark files (A–H, each 4,500,000 bytes = 1500×1500 × 16-bit) in `data/`.
Download `data2.zip` from the Moodle "Project #02" folder and unzip it here.

## Benchmark

Run against `gzip`, `bzip2`, `lzma`, `xz`, and `zstd` (per-file mean over files A–H):

```bash
./benchmark.sh -e -q \
  -o "ox-astro-ratio:./build/compress_astro_ratio %i %o:./build/decompress_astro_ratio %i %o" \
  -o "ox-astro-balanced:./build/compress_astro_balanced %i %o:./build/decompress_astro_balanced %i %o" \
  -o "ox-astro-fast:./build/compress_astro_fast %i %o:./build/decompress_astro_fast %i %o"
```

### Current results (per-file mean over A–H, 34.33 MB total)

| Rank | Compressor | bits/byte | t_comp (s) | t_decomp (s) | t_total (s) | Lossless |
|-----:|:-----------|----------:|-----------:|-------------:|------------:|:--------:|
| **1** | **ox-astro-ratio**    | **3.427** | **0.150** | **0.163** | **0.313** | **YES** |
| **2** | **ox-astro-balanced** | **3.430** | **0.046** | **0.037** | **0.083** | **YES** |
|     3 | bzip2                 |     3.579 |      0.247 |        0.129 |       0.376 |   YES    |
| **4** | **ox-astro-fast**     | **3.608** | **0.019** | **0.021** | **0.040** | **YES** |
|     5 | lzma-5                |     3.679 |      1.595 |        0.086 |       1.681 |   YES    |
|     6 | lzma-9                |     3.684 |      1.559 |        0.086 |       1.645 |   YES    |
|     7 | xz-6                  |     3.685 |      1.564 |        0.090 |       1.654 |   YES    |
|     8 | lzma-1                |     3.916 |      0.366 |        0.085 |       0.451 |   YES    |
|     9 | zstd-19               |     4.031 |      1.433 |        0.010 |       1.443 |   YES    |
|    10 | zstd-3                |     4.323 |      0.031 |        0.010 |       0.041 |   YES    |
|    11 | gzip                  |     4.376 |      0.306 |        0.022 |       0.328 |   YES    |
|    12 | zstd-1                |     4.499 |      0.015 |        0.010 |       0.025 |   YES    |

## Algorithm

All three variants share the same two-stage architecture: a **modeling stage** that predicts each pixel from its neighbours, and a **coding stage** that entropy-codes the residuals.

### Common components

**Input:** raw big-endian unsigned 16-bit samples; dimensions supplied at invocation (default 1500×1500).

**Modular zigzag:** the wrapping residual `u = (pixel − pred) mod 65536` is remapped so that small-magnitude errors (positive and negative) land near 0: even values encode positive errors, odd values encode negative errors. This concentrates probability mass near 0 regardless of sign.

**Byte split:** each 16-bit zigzag value is split into an upper byte and a lower byte, encoded with separate adaptive models. For smooth astronomical backgrounds the upper byte is zero for 90–100% of pixels.

**Entropy coder:** LZMA-style byte-aligned range coder with Fenwick-tree frequency tables (logarithmic update and query cost). All models are adaptive and updated symbol by symbol.

**Compressed format:** 4-byte magic identifier + width (4 B LE) + height (4 B LE) + variant-specific header + range-coded stream.

### Variant differences

| | Predictor | hi model | lo model | Models total |
|---|-----------|----------|----------|--------------|
| **ratio** | Adaptive (MED or global-mean, chosen per-file by entropy pre-scan) | MED mode: 365 spatial contexts + 4 class priors. Mean mode: 16 hi-neighbor contexts | 32 tables (smooth pixels) + 255 tables (non-smooth pixels) | 381 or 287 |
| **balanced** | Adaptive (GAP or global-mean, chosen per-tile by entropy pre-scan) | 16 hi-neighbor contexts (both modes) | 32 tables (smooth pixels) + 255 tables (non-smooth pixels) | 303 |
| **fast** | Horizontal delta (left neighbour only) | ORDER-0 (1 table) | ORDER-0 (1 table) | 2 |

**Predictors:** MED (JPEG-LS) reads three neighbours (W, N, NW) and clips the prediction to the local range. GAP reads six neighbours and adapts to dominant gradient direction before falling back to MED. Mean uses the global image mean as a constant (effective for flat-field calibration images where MED produces oscillatory residuals).
