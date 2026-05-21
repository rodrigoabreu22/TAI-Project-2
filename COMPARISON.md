# ox-astro-ratio vs ox-astro-balanced — Side-by-side Comparison

**Result summary (mean over 8 files, lower bits/byte = better)**

| | ox-astro-ratio | ox-astro-balanced |
|---|:-:|:-:|
| **bits/byte** | **3.427** | 3.465 |
| t_compress | 0.24 s | 0.18 s |
| t_decompress | 0.21 s | 0.21 s |
| Magic | `TA2A` | `TA2B` |

---

## Pipeline — step by step

| Step | ox-astro-ratio | ox-astro-balanced | Same? |
|------|---------------|-------------------|:-----:|
| 1. Load image | Read raw bytes → 2,250,000 big-endian uint16_t pixels | identical | ✓ |
| 2. Pre-scan | Compute residual entropy for **MED** and for **global-mean** | Compute residual entropy for **GAP** and for **global-mean** | ≈ (different candidate) |
| 3. Predictor choice | Pick lower-entropy candidate (threshold 0.30 b/px) | Pick lower-entropy candidate (no threshold — any gain switches) | ≈ (threshold differs) |
| 4. Residual | `(pixel − pred) mod 65536` → modular zigzag → hi/lo byte split | identical | ✓ |
| 5. Hi context | **Mode-dependent** — gradient (365) in MED mode, hi-neighbor (16) in mean mode | **Always hi-neighbor (16)** regardless of mode | ✗ |
| 6. Hi encoding | Range-code hi with selected context model | identical range coder | ✓ |
| 7. Lo encoding | 32 models (hi=0) + 255 models (hi>0) | identical counts | ✓ |
| 8. Lo index (hi=0) | gradient class (from raw pixel D1/D2/D3) × prev_lo_bin | hi-neighbor sum class × prev_lo_bin | ✗ |
| 9. Bias correction | C/B/Nc nudge per context | identical formula | ✓ |
| 10. Frequency tables | **FastFrequencyTable** — inline Fenwick tree (header-only) | **FenwickFrequencyTable** — Fenwick tree (separate .cpp) | ✓ same O(8) idea |
| 11. File header | magic + width + height + pred_mode (1 B) + global_mean (2 B) | identical layout | ✓ |
| 12. Decoding | Exact mirror of encoder | Exact mirror of encoder | ✓ |

---

## The key structural difference — hi context

### ratio — two separate context systems

```
if use_mean:
    hi_ctx = quantize(hi_W) × 4 + quantize(hi_N)    ← 16 hi-neighbor contexts
    encode hi  →  hi_nbr_models[hi_ctx]
    bias state →  C_nbr / B_nbr / Nc_nbr  (16 states)

else (MED mode):
    D1 = N−NW,  D2 = NW−W,  D3 = W−WW
    ctx, sign = spatial_context(D1, D2, D3)           ← 365 gradient contexts
    mixed = MixedFrequencyTable(hi_models[ctx],
                                hi_class_priors[grad_class])
    encode hi  →  mixed                               ← context mixing (fine + prior)
    bias state →  C / B / Nc  (365 states)
```

### balanced — one context system for both modes

```
hi_ctx = quantize(hi_W) × 4 + quantize(hi_N)         ← 16 hi-neighbor contexts, always
encode hi  →  hi_models[hi_ctx]
bias state →  C / B / Nc  (16 states, always)
```

---

## What is shared

- **Two-pass architecture**: pre-scan to pick the predictor, then a single coding pass.
- **Entropy-based predictor selection**: both compute `H(hi) + frac_hi0×H(lo|hi=0) + frac_hip×H(lo|hi>0)` for each candidate and pick the lower one.
- **Global-mean mode is identical**: constant predictor, 16 hi-neighbor contexts, same bias formula, same lo models. When both implementations choose global-mean for a file, they produce nearly identical results.
- **Lo model structure**: 32 tables for hi=0 (4 classes × 8 prev_lo bins) and 255 tables for hi>0.
- **Modular zigzag encoding** and byte split.
- **Fenwick tree** for O(log 256) = O(8) frequency table operations — both implementations use it, implemented independently.
- **LZMA-style range coder**.
- **Bias correction**: same C/B/Nc formula and halving schedule (period 512).
- **File format**: `magic (4B)` + `width (4B LE)` + `height (4B LE)` + `pred_mode (1B)` + `global_mean (2B LE)` + bitstream.

---

## What differs

### 1 — Neighbours read per pixel

| | ratio | balanced |
|---|---|---|
| Prediction | W, N, NW | W, WW, N, NN, NW, NE |
| Gradient (for lo index) | W, WW, N, NW only (D1, D2, D3) | same 4 (D1, D2, D3) |
| **Total neighbours read** | **4** (W, WW, N, NW) | **6** (adds NN, NE) |

ratio removed NN and NE — they were left over from an earlier GAP predictor experiment and contributed no information to the final model.

### 2 — Non-mean predictor

| | ratio | balanced |
|---|---|---|
| **Predictor** | JPEG-LS MED (3 neighbours: W, N, NW) | GAP (6 neighbours: W, WW, N, NN, NW, NE) |
| **Robust fallback** | — | When hi_W or hi_N ≥ 8, uses cleaner neighbour (`robust_gap_predict`) |

GAP measures horizontal and vertical gradient activity and switches between predicting from the left column or the row above, falling back to MED in smooth areas. The robust fallback prevents star-pixel residuals from contaminating the prediction of their spatial neighbours.

### 3 — Hi context in non-mean mode

| | ratio | balanced |
|---|---|---|
| **Context source** | Raw pixel gradients D1=N−NW, D2=NW−W, D3=W−WW | Hi byte of left (hi_W) and above (hi_N) neighbours |
| **Number of contexts** | 365 (sign-folded 3-gradient quantisation) | 16 (4 bins × 4 bins of hi-neighbour) |
| **Samples per context** | ~6,000 | ~140,000 |
| **Warm-up fix** | Context mixing: MixedFrequencyTable(fine[365], prior[4]) | No mixing needed — 16 models converge fast |
| **What it captures** | Texture type: edge direction, gradient magnitude | Residual persistence: were neighbouring coded residuals large? |

**Why ratio wins on A, B, G:** strong spatial autocorrelation means 365 fine gradient contexts provide far more discriminating power. A steep vertical edge gets a different model from a gentle horizontal one. The cold-start overhead is solved by context mixing.

**Why balanced wins on F:** GAP uses two extra neighbours (NN, NE) and the robust fallback, giving slightly better prediction on this particular file.

### 4 — Lo index for hi=0 pixels

| | ratio | balanced |
|---|---|---|
| **4-class index source** | `max(|D1|,|D2|,|D3|)` thresholded at 32/128/512 (raw pixel gradients) | `min(quantize(hi_W) + quantize(hi_N), 3)` (coded residuals) |

Empirical analysis confirms the gradient-class index gives lower conditional H(lo) on most files, so ratio retains it in both modes.

### 5 — Number of bias correction states

| | ratio MED mode | ratio mean mode | balanced |
|---|:-:|:-:|:-:|
| **Bias states** | 365 | 16 | 16 |

---

## Per-file winner

| File | ratio | balanced | Winner | Reason |
|------|------:|----------:|:------:|--------|
| A | 4.399 | 4.492 | **ratio** | High spatial autocorrelation (r₁=0.91); 365 fine contexts exploit it fully |
| B | 3.252 | 3.379 | **ratio** | Moderate spatial autocorrelation; fine gradient contexts give clear advantage |
| C | 2.419 | 2.419 | **tie** | Both choose global-mean + 16 hi-neighbor; both at entropy floor |
| D | 2.714 | 2.713 | **tie** | Same mode; within rounding |
| E | 2.563 | 2.561 | **tie** | Same mode; essentially at floor |
| F | 6.245 | 6.228 | balanced | GAP + robust_gap_predict gives marginally better prediction |
| G | 2.564 | 2.669 | **ratio** | Very strong vertical autocorrelation (r₁=0.75); 365 contexts capture it |
| H | 3.254 | 3.252 | **tie** | Both choose global-mean + 16 hi-neighbor; essentially tied |
| **Mean** | **3.427** | **3.465** | **ratio** | −0.038 b/B advantage |

---

## Summary

```
Shared by both:
  Two-pass architecture  (pre-scan → pick predictor → encode)
  Entropy-based predictor selection  (MED/GAP vs global-mean)
  Global-mean mode  — identical (16 hi-neighbor contexts, same lo models)
  Lo model structure  — 32 tables (hi=0) + 255 tables (hi>0)
  Fenwick-tree frequency tables  — O(8) per operation
  LZMA range coder + bias correction  — same formula
  File format  — magic + dims + pred_mode + global_mean + bitstream

Non-mean mode only (A, B, F, G):
  ratio:    4 neighbours  →  MED(W,N,NW)  →  365 gradient contexts  →  context mixing
  balanced: 6 neighbours  →  GAP(W,WW,N,NN,NW,NE) + robust fallback  →  16 hi-neighbor contexts
```