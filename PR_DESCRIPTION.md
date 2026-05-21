## ox-astro-ratio — compression-focused astronomical image compressor

### Summary

- Implement `ox-astro-ratio` (`TA2A`), a lossless compressor for raw 16-bit big-endian astronomical rasters (1500×1500) targeting minimum bits/byte
- Achieve **3.427 bits/byte** mean over 8 benchmark files, beating bzip2 (3.579) and all other reference tools
- Add `FastFrequencyTable` (inline Fenwick tree, O(8) per op) and `MixedFrequencyTable` (joint Fenwick descent) as new coder primitives
- Add adaptive predictor pre-scan, 365-context spatial model, and context mixing to `ImagePredictor.hpp`
- Update `README.md`, `benchmarks.md`, and build scripts

---

### Algorithm overview

**Two-pass architecture**

1. **Pre-scan** — iterate over all pixels once, accumulating per-gradient-class symbol histograms for both the JPEG-LS MED predictor and the global-mean predictor. Compute class-weighted entropy estimates and select the predictor whose estimate is lower by at least 0.30 bits/pixel.
2. **Encoding pass** — encode residuals in raster order using the chosen predictor and full context models.

**Predictor selection**

The entropy estimate uses four gradient-magnitude classes (thresholds 32 / 128 / 512 on `max(|D1|,|D2|,|D3|)`), giving a class-conditional estimate that captures the spatial-context benefit before a single bit is coded. The 0.30 b/px threshold switches correctly: global-mean wins for noise-dominated flat fields (C, D, E, H) where MED creates oscillatory residuals with r₁ ≈ −0.4, while MED wins for structured images (A, B, F, G).

**Residual encoding**

Wrapping residual `u = (pixel − pred) mod 65536` is mapped through modular zigzag (`z = 2u` if `u ≤ 32767`, else `2(65536−u)−1`) and split into `hi = z >> 8` and `lo = z & 0xFF`. For smooth backgrounds hi = 0 for 90–100% of pixels.

**Hi-byte context — MED mode (365 contexts + context mixing)**

Gradients `D1 = N−NW`, `D2 = NW−W`, `D3 = W−WW` are each quantised to nine levels {−4…4}. Sign-symmetry folds 9³ = 729 indices to 365: when the leading non-zero gradient is negative the residual is negated before encoding. Because 365 fine contexts are sparse, each hi symbol is drawn from a mixed distribution:

```
P_mixed(s) ∝ f_ctx(s) + f_prior(s)
```

where `f_ctx` is the fine per-context table (one of 365) and `f_prior` is one of four gradient-class prior tables (capped at 1024, halved on overflow). The prior provides warm-start while the fine context accumulates observations.

**Hi-byte context — mean mode (16 contexts)**

Under a constant predictor, pixel gradients carry no information. Instead, residual persistence is captured by `ctx = q(hi_W) × 4 + q(hi_N)` where `hi_W`/`hi_N` are the hi bytes of encoded west/north neighbours and `q` is a four-level quantiser (0 / 1 / 2–7 / ≥8).

**Lo-byte models**

- **hi = 0** (32 tables): indexed by `gc × 8 + q_lo(prev_lo_hi0)` where `gc` is the gradient-magnitude class from raw pixel differences and `q_lo` is an eight-level log-scale quantiser of the previous hi=0 lo byte.
- **hi > 0** (255 tables): one per distinct hi value, indexed by `hi − 1`.

**Bias correction**

LOCO-I accumulators `C[ctx]`, `B[ctx]`, `Nc[ctx]` per context. Adjusted prediction: `pred_adj = pred + sign × C[ctx]`. After each pixel, `B` accumulates the signed residual; when `|B| > Nc`, `C` is nudged ±1. Window halved at `Nc = 512`.

**Frequency tables**

`FastFrequencyTable`: inline Fenwick BIT, O(8) for `increment`, `getLow`, `findSymbol`, and batch `halve()` (O(N) tree rebuild). `MixedFrequencyTable`: read-only sum of two `FastFrequencyTable` instances with joint Fenwick descent so the mixed lookup stays O(8).

---

### Per-file results

| File | bits/byte | Mode |
|------|----------:|------|
| A | 4.399 | MED |
| B | 3.252 | MED |
| C | 2.419 | mean |
| D | 2.714 | mean |
| E | 2.563 | mean |
| F | 6.245 | MED |
| G | 2.564 | MED |
| H | 3.254 | mean |
| **Mean** | **3.427** | |

All 8 files verified lossless (byte-for-byte roundtrip).

---

### File format (`TA2A`)

```
Bytes  0–3  : magic "TA2A"
Bytes  4–7  : width   uint32_t LE
Bytes  8–11 : height  uint32_t LE
Byte    12  : pred_mode  (0 = MED, 1 = global-mean)
Bytes 13–14 : global_mean  uint16_t LE
Bytes  15+  : LZMA-style range-coded bitstream
```

---

### Test plan

- [ ] `cmake --build build --target compress_astro_ratio decompress_astro_ratio` builds cleanly
- [ ] Lossless roundtrip on all 8 files: `cmp data2/X <(decompress | compress)` passes for A–H
- [ ] `./benchmark.sh` shows ox-astro-ratio at rank 1 with ~3.43 bits/byte
- [ ] No memory errors under `valgrind` on at least one file
