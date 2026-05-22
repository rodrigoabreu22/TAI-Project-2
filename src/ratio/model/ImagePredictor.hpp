#pragma once

#include <algorithm>
#include <cstdint>

// ── JPEG-LS MED predictor ─────────────────────────────────────────────────────
// W=left, N=above, NW=above-left.
inline uint16_t med_predict(uint16_t W, uint16_t N, uint16_t NW) {
    if (NW >= std::max(W, N)) return std::min(W, N);
    if (NW <= std::min(W, N)) return std::max(W, N);
    return static_cast<uint16_t>(W + N - NW);
}

// ── GAP predictor (CALIC-style) ───────────────────────────────────────────────
// Uses a 6-neighbour window (W, WW, N, NN, NW, NE) to measure horizontal and
// vertical activity, then selects the predictor from the lower-activity axis.
// Falls back to MED for smooth/ambiguous regions.
//
// dh = horizontal activity = |W-WW| + |N-NW| + |N-NE|
// dv = vertical   activity = |N-NN| + |W-NW| + |NE-N|
//
//   dv - dh > GAP_THRESH  →  pred = W   (vertical changes dominate → row is stable)
//   dh - dv > GAP_THRESH  →  pred = N   (horizontal changes dominate → column is stable)
//   otherwise             →  pred = MED(W, N, NW)
//
// GAP_THRESH = 80: chosen to be robust across 16-bit astronomical images of
// varying noise levels. Large enough to ignore noise-scale fluctuations
// (~10-20 ADU in smooth images), small enough to catch real gradients (>80 ADU).
static constexpr int GAP_THRESH = 128;

inline uint16_t gap_predict(uint16_t W, uint16_t WW,
                             uint16_t N, uint16_t NN,
                             uint16_t NW, uint16_t NE) {
    int dh = std::abs((int)W  - (int)WW)
           + std::abs((int)N  - (int)NW)
           + std::abs((int)N  - (int)NE);
    int dv = std::abs((int)N  - (int)NN)
           + std::abs((int)W  - (int)NW)
           + std::abs((int)NE - (int)N);

    if (dv - dh > GAP_THRESH) return W;
    if (dh - dv > GAP_THRESH) return N;
    return med_predict(W, N, NW);
}

// ── Modular zigzag ────────────────────────────────────────────────────────────
// Maps wrapping residual u = (pixel - pred) mod 2^16 so that small-magnitude
// values (both positive and negative) land near 0.
//   0 → 0,  65535 → 1,  1 → 2,  65534 → 3,  ...
inline uint16_t zigzag_encode(uint16_t u) {
    return (u <= 32767u) ? static_cast<uint16_t>(u * 2u)
                         : static_cast<uint16_t>((65536u - u) * 2u - 1u);
}
inline uint16_t zigzag_decode(uint16_t z) {
    return (z & 1u) ? static_cast<uint16_t>(65536u - (z + 1u) / 2u)
                    : static_cast<uint16_t>(z / 2u);
}

// ── JPEG-LS 3-gradient spatial context ───────────────────────────────────────
//
// Computes a context index in [0, 364] from three directional gradients:
//   D1 = N  - NW   (vertical)
//   D2 = NW - W    (diagonal)
//   D3 = W  - WW   (horizontal, requires one extra left neighbour WW)
//
// Each gradient is quantized to {-4..4} using thresholds T1 < T2 < T3.
// Sign symmetry folds (q1,q2,q3) and (-q1,-q2,-q3) into the same context,
// halving the total from 729 to 365 and doubling samples per context.
// When sign==-1 the caller must negate the residual before encoding/decoding.
//
// Tunable thresholds for 16-bit astronomical images:
static constexpr int GRAD_T1 =  32;
static constexpr int GRAD_T2 = 128;
static constexpr int GRAD_T3 = 512;
static constexpr int NUM_CONTEXTS = 365;

inline int quantize_grad(int d) {
    if (d <= -GRAD_T3) return -4;
    if (d <= -GRAD_T2) return -3;
    if (d <= -GRAD_T1) return -2;
    if (d <   0)       return -1;
    if (d ==  0)       return  0;
    if (d <  GRAD_T1)  return  1;
    if (d <  GRAD_T2)  return  2;
    if (d <  GRAD_T3)  return  3;
    return 4;
}

// Returns a context index in [0, 364] and sets `sign` to +1 or -1.
// Count breakdown: q1>0: 4*9*9=324; q1=0,q2>0: 4*9=36; q1=q2=0: 5 → total 365.
inline int spatial_context(int D1, int D2, int D3, int& sign) {
    int q1 = quantize_grad(D1);
    int q2 = quantize_grad(D2);
    int q3 = quantize_grad(D3);

    sign = 1;
    if (q1 < 0 || (q1 == 0 && q2 < 0) || (q1 == 0 && q2 == 0 && q3 < 0)) {
        q1 = -q1; q2 = -q2; q3 = -q3;
        sign = -1;
    }

    if (q1 > 0) return (q1 - 1) * 81 + (q2 + 4) * 9 + (q3 + 4);  // [0, 323]
    if (q2 > 0) return 324 + (q2 - 1) * 9 + (q3 + 4);             // [324, 359]
    return 324 + 36 + q3;                                           // [360, 364]
}
