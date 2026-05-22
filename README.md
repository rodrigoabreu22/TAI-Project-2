# TAI - Project 2
**Algorithmic Information Theory (2025/26) — Universidade de Aveiro**

Data-specific lossless compressor for raw astronomical images. Input is a raw raster of **1500×1500 unsigned 16-bit integers (big-endian)**. Three independent compressors are provided, each targeting a different trade-off between compression ratio and speed.

| Tool | Strategy | bits/byte (baseline) |
|------|----------|---------------------|
| `ox-astro-ratio`    | JPEG-LS MED + ORDER-1 hi + ORDER-1(hi) lo | 3.609 |
| `ox-astro-balanced` | JPEG-LS MED + ORDER-1 hi + ORDER-0 lo     | 3.611 |
| `ox-astro-fast`     | Horizontal delta + ORDER-0                | 3.601 |

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
# Compress with explicit geometry
./build/compress_astro_fast <n_rows> <n_cols> <input_file> <output_file>

# Decompress
./build/decompress_astro_fast <compressed_file> <output_file>

# Example
./build/compress_astro_fast 1000 3500 x x.enc
./build/decompress_astro_fast x.enc x.dec
```

The fast encoder also accepts the older `<input_file> <output_file>` form for the local benchmark script, but the required submission interface is the explicit `<n_rows> <n_cols> <input_file> <output_file>` form.

## Data

Place the benchmark files (A–H, each 4,500,000 bytes = 1500×1500 × 16-bit) in `data2/`.
Download `data2.zip` from the Moodle "Project #02" folder and unzip it here.

## Benchmark

Run against `gzip`, `bzip2`, `lzma`, `xz`, and `zstd` (per-file mean over files A–H):

```bash
./benchmark.sh -d data2 -m -q \
  -o "ox-astro-ratio:./build/compress_astro_ratio %i %o:./build/decompress_astro_ratio %i %o" \
  -o "ox-astro-balanced:./build/compress_astro_balanced %i %o:./build/decompress_astro_balanced %i %o" \
  -o "ox-astro-fast:./build/compress_astro_fast %i %o:./build/decompress_astro_fast %i %o"
```

### Baseline results (per-file mean over A–H, 34.33 MB total)

| Rank | Compressor | bits/byte | t_comp (s) | t_decomp (s) | Lossless |
|-----:|:-----------|----------:|-----------:|-------------:|:--------:|
| 1 | bzip2 | 3.579 | 0.38 | 0.26 | YES |
| **2** | **ox-astro-fast** | **3.601** | **1.16** | **1.20** | **YES** |
| **3** | **ox-astro-ratio** | **3.609** | **1.09** | **1.24** | **YES** |
| **4** | **ox-astro-balanced** | **3.611** | **1.09** | **1.32** | **YES** |
| 5 | lzma-5 | 3.679 | 1.95 | 0.19 | YES |
| 6 | lzma-9 | 3.684 | 2.09 | 0.18 | YES |
| 7 | xz-6 | 3.685 | 2.04 | 0.18 | YES |
| 8 | lzma-1 | 3.916 | 0.57 | 0.20 | YES |
| 9 | zstd-19 | 4.156 | 1.89 | 0.02 | YES |
| 10 | zstd-3 | 4.324 | 0.10 | 0.05 | YES |
| 11 | gzip | 4.376 | 0.45 | 0.09 | YES |
| 12 | zstd-1 | 4.487 | 0.06 | 0.02 | YES |

## Algorithm

All three variants share the same two-stage architecture: a **modeling stage** that predicts each pixel from its neighbours, and a **coding stage** that entropy-codes the residuals.

### Common components

**Input parsing:** raw bytes decoded as big-endian `uint16_t`, stored as a 2D pixel array.

**Modular zigzag:** wrapping residual `u = pixel − pred (mod 2¹⁶)` is remapped so that small-magnitude values (both positive and negative) land near 0:
```
z = (u ≤ 32767) ?  2·u  :  (65536−u)·2 − 1
```
This concentrates probability mass near 0 regardless of residual sign.

**Byte split:** the 16-bit zigzag value `z` is split into `hi = z >> 8` and `lo = z & 0xFF`, encoded with separate models. For smooth astronomical backgrounds, `hi = 0x00` for 90–100% of pixels.

**Compressed format:** 4-byte magic + width (4 B LE) + height (4 B LE) + range-coded stream.

### Variant differences

| | Predictor | hi model | lo model | Models total | Magic |
|---|-----------|----------|----------|--------------|-------|
| **ratio** | JPEG-LS MED | ORDER-1 (256 tables, ctx=prev hi) | ORDER-1 (256 tables, ctx=current hi) | 512 | `TA2A` |
| **balanced** | JPEG-LS MED | ORDER-1 (256 tables, ctx=prev hi) | ORDER-0 (1 shared table) | 257 | `TA2B` |
| **fast** | Horizontal delta (left neighbour) | ORDER-0 (1 table) | ORDER-0 (1 table) | 2 | `TA2F` |

**JPEG-LS MED predictor** (`W`=left, `N`=above, `NW`=above-left):
```
if   NW ≥ max(W, N):  pred = min(W, N)
elif NW ≤ min(W, N):  pred = max(W, N)
else:                 pred = W + N − NW
```
Selects horizontal/vertical prediction at edges and bilinear interpolation in smooth regions.
