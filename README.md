# TAI - Project 2
**Algorithmic Information Theory (2025/26) — Universidade de Aveiro**

Data-specific lossless compressor for raw astronomical images. Input is a raw raster of **1500×1500 unsigned 16-bit integers (big-endian)**. Three independent compressors are provided, each targeting a different trade-off between compression ratio and speed.


## Authors

| Name | GitHub | Assignment |
|------|--------|------------|
| Eduardo Lopes | [@odraude23](https://github.com/odraude23) | `ox-astro-fast` |
| Rodrigo Abreu | [@rodrigoabreu22](https://github.com/rodrigoabreu22) | `ox-astro-ratio` |
| Hugo Ribeiro  | [@xHuGODx](https://github.com/xHuGODx) | `ox-astro-balanced` |

## Build

```bash
# Configure (first time only)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all targets
cmake --build build -j$(nproc)

# Build a specific target
cmake --build build --target compress_astro_balanced -j$(nproc)
```

Produces binaries in `build/`:
- `compress_astro` / `decompress_astro`  (balanced — already working)

> Ratio and fast binaries will be added as `compress_astro_ratio` / `compress_astro_fast` etc.

## Usage

```bash
# Compress
./build/compress_astro <input_file> <output_file>

# Decompress
./build/decompress_astro <compressed_file> <output_file>

# Stdin/stdout mode
./build/compress_astro < input_file > output_file
```

## Data

Place the benchmark files (A–H, each 4,500,000 bytes = 1500×1500 × 16-bit) in `data2/`.
Download `data2.zip` from the Moodle "Project #02" folder and unzip it here.

## Benchmark

Run against `gzip`, `bzip2`, `lzma`, `xz`, and `zstd` (per-file mean over files A–H):

```bash
# Balanced only
./benchmark.sh -d data2 -m -q \
  -o "ox-astro:./build/compress_astro %i %o:./build/decompress_astro %i %o"

# All three (once ratio and fast are implemented)
./benchmark.sh -d data2 -m -q \
  -o "ox-astro:./build/compress_astro %i %o:./build/decompress_astro %i %o" \
  -o "ox-astro-ratio:./build/compress_astro_ratio %i %o:./build/decompress_astro_ratio %i %o" \
  -o "ox-astro-fast:./build/compress_astro_fast %i %o:./build/decompress_astro_fast %i %o"
```

### Current results (balanced baseline, per-file mean over A–H)

| Rank | Compressor | bits/byte | t_comp (s) | t_decomp (s) | Lossless |
|------|-----------|-----------|------------|--------------|----------|
| 1 | bzip2 | 3.579 | 0.33 | 0.24 | YES |
| **2** | **ox-astro** | **3.609** | **0.94** | **1.00** | **YES** |
| 3 | lzma-9 | 3.684 | 1.49 | 0.17 | YES |
| 4 | xz-6 | 3.685 | 1.48 | 0.18 | YES |
| 5 | lzma-1 | 3.916 | 0.42 | 0.18 | YES |
| 6 | zstd-19 | 4.156 | 1.38 | 0.02 | YES |
| 7 | zstd-3 | 4.324 | 0.08 | 0.06 | YES |
| 8 | gzip | 4.376 | 0.37 | 0.08 | YES |
| 9 | zstd-1 | 4.487 | 0.05 | 0.02 | YES |


## Algorithm — Balanced Variant

**Pipeline:**
1. Read raw bytes → decode as 2D array of big-endian `uint16_t`
2. Raster-scan each pixel; predict with **JPEG-LS MED**:
   - `W` = left, `N` = above, `NW` = above-left
   - `pred = median-edge(W, N, NW)` — handles edges and smooth regions
3. Compute wrapping residual: `u = pixel − pred  (mod 2¹⁶)`
4. **Modular zigzag**: map `u` so small-magnitude values (both positive and negative) map near 0
5. Split each 16-bit zigzag value into `hi` (upper byte) and `lo` (lower byte)
6. Encode with adaptive range coder:
   - `hi` → model indexed by previous `hi` byte (ORDER-1)
   - `lo` → model indexed by current `hi` byte (strong correlation)

**Compressed format:** `"TA2A"` magic (4 B) + width (4 B LE) + height (4 B LE) + range-coded stream.
