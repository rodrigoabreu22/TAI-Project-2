/*
 * TAI Project 2 — Astronomical Image Compressor (ox-astro-ratio)
 *
 * Pipeline:
 *   1. Read raw bytes; interpret as 2D array of big-endian uint16_t.
 *   2. Pre-scan: estimate marginal entropy H(hi)+frac*H(lo) for MED and global-mean
 *      predictors. The predictor with lower estimated entropy is selected.
 *      For calibration-style flat-field images (C,D,E,F,H) global-mean wins because
 *      MED overshoots noise, creating r1≈-0.4 oscillatory residuals that inflate entropy.
 *      For structured images (A,B,G) MED wins.
 *   3. JPEG-LS MED or global-mean spatial predictor → wrapping residual → modular zigzag.
 *   4. Split zigzag value into hi byte and lo byte.
 *   5. Compute JPEG-LS 3-gradient spatial context (365 contexts):
 *        D1 = N-NW (vertical), D2 = NW-W (diagonal), D3 = W-WW (horizontal)
 *        Each quantised to {-4..4}; sign-symmetry folds 729 → 365 contexts.
 *        When sign==-1, the residual is negated before encoding.
 *   6. Per-context bias correction:
 *        C[ctx] is a running correction term. pred_adj = pred + sign*C[ctx].
 *        After each symbol, B[ctx] accumulates the signed residual; when the
 *        mean exceeds ±1 sample, C[ctx] is nudged by ±1 to drive B toward 0.
 *        Decoder runs the identical update, so no extra bytes are stored.
 *   7. Encode hi and lo with the same spatial context (365 models each).
 *
 * Compressed file format
 * ──────────────────────
 *  Bytes  0–3  : magic "TA2A"
 *  Bytes  4–7  : width       uint32_t LE
 *  Bytes  8–11 : height      uint32_t LE
 *  Byte     12 : pred_mode   0=MED  1=global-mean
 *  Bytes 13–14 : global_mean uint16_t LE  (needed by decoder for pred_mode==1)
 *  Bytes   15+ : range-coded bitstream
 *
 * Usage:  compress_astro_ratio <input_file> <output_file>
 *         compress_astro_ratio          (stdin → stdout)
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "coder/RangeCoder.hpp"
#include "coder/FrequencyTable.hpp"
#include "coder/FastFrequencyTable.hpp"
#include "coder/MixedFrequencyTable.hpp"
#include "model/ImagePredictor.hpp"

static constexpr uint8_t MAGIC[4] = {'T','A','2','A'};

static void write_u32le(std::ostream& out, uint32_t v) {
    out.put(static_cast<char>(v        & 0xFF));
    out.put(static_cast<char>((v >>  8) & 0xFF));
    out.put(static_cast<char>((v >> 16) & 0xFF));
    out.put(static_cast<char>((v >> 24) & 0xFF));
}

// Map lo byte to one of 8 log-scale bins for ORDER-1 lo context.
static int quantize_prev_lo(uint8_t lo) {
    if (lo == 0)  return 0;
    if (lo < 4)   return 1;
    if (lo < 8)   return 2;
    if (lo < 16)  return 3;
    if (lo < 32)  return 4;
    if (lo < 64)  return 5;
    if (lo < 128) return 6;
    return 7;
}

// Quantize hi byte to 4 bins for hi-neighbor context (used in global-mean mode).
static int quantize_hi_val(uint8_t h) {
    if (h == 0)  return 0;
    if (h <= 2)  return 1;
    if (h <= 7)  return 2;
    return 3;
}

int main(int argc, char* argv[]) {
    std::istream* in_ptr  = &std::cin;
    std::ostream* out_ptr = &std::cout;
    std::ifstream fin;
    std::ofstream fout;

    if (argc >= 3) {
        fin.open(argv[1], std::ios::binary);
        if (!fin) { std::cerr << "Cannot open input: " << argv[1] << '\n'; return 1; }
        fout.open(argv[2], std::ios::binary);
        if (!fout) { std::cerr << "Cannot open output: " << argv[2] << '\n'; return 1; }
        in_ptr  = &fin;
        out_ptr = &fout;
    } else if (argc == 1) {
        std::cin.sync_with_stdio(false);
    } else {
        std::cerr << "Usage: compress_astro_ratio <input> <output>\n"; return 1;
    }

    // Read entire input
    std::vector<uint8_t> raw(std::istreambuf_iterator<char>(*in_ptr), {});
    if (raw.size() % 2 != 0) {
        std::cerr << "Input size not even — not a valid 16-bit raster\n"; return 1;
    }

    // Infer dimensions: expect square image, or rectangular if exact match
    const uint64_t npix = raw.size() / 2;
    uint32_t width = 0, height = 0;
    // Try square first
    uint32_t sq = static_cast<uint32_t>(std::sqrt(static_cast<double>(npix)));
    if ((uint64_t)sq * sq == npix) {
        width = height = sq;
    } else {
        // Common rectangular sizes: try 1500×1500, 1000×1000, 2048×1024, etc.
        // Fall back to 1×npix (1D mode)
        width  = static_cast<uint32_t>(npix);
        height = 1;
        // Try common widths
        for (uint32_t w : {1500u, 2048u, 1024u, 512u, 256u}) {
            if (npix % w == 0) { width = w; height = static_cast<uint32_t>(npix / w); break; }
        }
    }

    // Decode big-endian uint16_t pixels into host order
    std::vector<uint16_t> pixels(npix);
    for (uint64_t i = 0; i < npix; ++i)
        pixels[i] = (static_cast<uint16_t>(raw[2*i]) << 8) | raw[2*i+1];

    // ── Adaptive predictor selection (pre-scan) ───────────────────────────
    // Compute 4-class conditional entropy estimate for MED and global-mean:
    //   E = sum_c  frac_c * [ H(hi|c) + frac_hi0|c * H(lo|hi=0,c) + frac_hip|c * H(lo|hi>0,c) ]
    // where c ∈ {flat, slight, moderate, strong} is the JPEG-LS gradient class.
    // This 4-class estimate captures the spatial-context benefit: in strong-gradient
    // regions MED gives small residuals that global-mean misses, so the estimate
    // correctly prefers MED for structured images (A, B, G, F, H) while picking
    // global-mean for noise-dominated flat fields (C, D, E) where MED overshoots
    // and creates r1≈-0.4 oscillatory residuals.
    uint64_t px_sum = 0;
    for (uint64_t i = 0; i < npix; ++i) px_sum += pixels[i];
    const uint16_t global_mean = static_cast<uint16_t>(px_sum / npix);

    static constexpr int NCLASS = 4;
    std::array<std::array<uint64_t,256>,NCLASS> hi_med{},  hi_mean{};
    std::array<std::array<uint64_t,256>,NCLASS> lo0_med{}, lo0_mean{};
    std::array<std::array<uint64_t,256>,NCLASS> lop_med{}, lop_mean{};
    for (uint32_t row = 0; row < height; ++row) {
        for (uint32_t col = 0; col < width; ++col) {
            uint16_t px = pixels[row * width + col];
            uint16_t W  = (col > 0)                    ? pixels[row * width + col - 1]         : 0u;
            uint16_t WW = (col > 1)                    ? pixels[row * width + col - 2]         : W;
            uint16_t N  = (row > 0)                    ? pixels[(row-1) * width + col]         : W;
            uint16_t NW = (row > 0 && col > 0)         ? pixels[(row-1) * width + col - 1]     : W;

            int gc = 0;
            if (row > 0 && col > 0) {
                int D1 = static_cast<int>(N)  - static_cast<int>(NW);
                int D2 = static_cast<int>(NW) - static_cast<int>(W);
                int D3 = static_cast<int>(W)  - static_cast<int>(WW);
                int gm = std::max({std::abs(D1), std::abs(D2), std::abs(D3)});
                gc = (gm < GRAD_T1) ? 0 : (gm < GRAD_T2) ? 1 : (gm < GRAD_T3) ? 2 : 3;
            }

            auto acc = [&](uint16_t pred,
                           std::array<uint64_t,256>& hc,
                           std::array<uint64_t,256>& lc0,
                           std::array<uint64_t,256>& lcp) {
                uint16_t u = static_cast<uint16_t>(px - pred);
                uint16_t z = zigzag_encode(u);
                uint8_t  h = static_cast<uint8_t>(z >> 8);
                uint8_t  l = static_cast<uint8_t>(z & 0xFF);
                hc[h]++;
                (h == 0 ? lc0 : lcp)[l]++;
            };

            uint16_t pm = (row == 0 && col == 0) ? 0u : med_predict(W, N, NW);
            acc(pm,          hi_med[gc],  lo0_med[gc],  lop_med[gc]);
            acc(global_mean, hi_mean[gc], lo0_mean[gc], lop_mean[gc]);
        }
    }

    auto H256 = [](const std::array<uint64_t,256>& c) -> double {
        uint64_t tot = 0; for (auto v : c) tot += v;
        if (!tot) return 0.0;
        double h = 0.0, inv = 1.0 / static_cast<double>(tot);
        for (auto v : c) if (v) { double p = v*inv; h -= p*std::log2(p); }
        return h;
    };
    auto est_class = [&](const std::array<std::array<uint64_t,256>,NCLASS>& hc,
                         const std::array<std::array<uint64_t,256>,NCLASS>& lc0,
                         const std::array<std::array<uint64_t,256>,NCLASS>& lcp) -> double {
        double total = 0, weighted = 0;
        for (int c = 0; c < NCLASS; ++c) {
            uint64_t nc = 0; for (auto v : hc[c]) nc += v;
            if (!nc) continue;
            uint64_t n0 = 0; for (auto v : lc0[c]) n0 += v;
            double f0 = static_cast<double>(n0) / static_cast<double>(nc);
            double e  = H256(hc[c]) + f0*H256(lc0[c]) + (1.0-f0)*H256(lcp[c]);
            weighted += static_cast<double>(nc) * e;
            total    += static_cast<double>(nc);
        }
        return total > 0 ? weighted / total : 0.0;
    };

    // Threshold: 0.30 bits/pixel (= 0.15 bpb).
    // Correctly switches C/D/E (est gain 0.45–0.50 b/px) and H (0.34 b/px),
    // and correctly keeps F at MED (0.07 b/px).
    // When use_mean=true, hi encoding uses 16 hi-neighbor models (not gradient
    // contexts), which is essential for H: star clusters have hi>0 neighbors and
    // the hi-neighbor context captures residual persistence much better than
    // gradient context under a constant predictor.
    static constexpr double MEAN_PRED_THRESHOLD = 0.30;  // bits/pixel
    const double e_med  = est_class(hi_med,  lo0_med,  lop_med);
    const double e_mean = est_class(hi_mean, lo0_mean, lop_mean);
    const bool use_mean = (e_med - e_mean) > MEAN_PRED_THRESHOLD;

    // Write header
    out_ptr->write(reinterpret_cast<const char*>(MAGIC), 4);
    write_u32le(*out_ptr, width);
    write_u32le(*out_ptr, height);
    out_ptr->put(static_cast<char>(use_mean ? 1 : 0));
    out_ptr->put(static_cast<char>(global_mean & 0xFF));
    out_ptr->put(static_cast<char>((global_mean >> 8) & 0xFF));

    // hi models (365): fine per-context adaptive models.
    // hi_class_priors (4): gradient-class priors for context mixing.
    // lo_hi0_models: 32 × 256-symbol, indexed by (grad_class × 8 + prev_lo_bin).
    // lo_hip_models: 255 × 256-symbol, indexed by (hi-1) for hi in [1,255].
    static constexpr int    LO_HI0_CLASSES   = 4;
    static constexpr int    LO_HI0_PREV_BINS = 8;
    static constexpr int    LO_HI0_TOTAL     = LO_HI0_CLASSES * LO_HI0_PREV_BINS;
    static constexpr uint32_t HI_PRIOR_CAP   = 1024;

    std::vector<FastFrequencyTable> hi_models(NUM_CONTEXTS);
    std::vector<FastFrequencyTable> hi_class_priors(LO_HI0_CLASSES);
    std::vector<FastFrequencyTable> lo_hi0_models(LO_HI0_TOTAL);
    std::vector<FastFrequencyTable> lo_hip_models(255);

    // hi-neighbor models for global-mean mode (16 contexts).
    // Used instead of the 365-gradient model when use_mean=true.
    // Captures residual persistence: after a star pixel (hi>0), the neighbor
    // is also likely hi>0. Under global-mean predictor this is the dominant
    // structure — gradient context is useless since residual = pixel - constant.
    static constexpr int HI_NBR_BINS = 4;
    static constexpr int NUM_HI_NBR  = HI_NBR_BINS * HI_NBR_BINS;  // 16

    std::vector<FastFrequencyTable> hi_nbr_models(NUM_HI_NBR);
    std::vector<int> C_nbr(NUM_HI_NBR, 0);
    std::vector<int> B_nbr(NUM_HI_NBR, 0);
    std::vector<int> Nc_nbr(NUM_HI_NBR, 0);

    // Per-context bias correction state for MED mode (all zero-initialised)
    std::vector<int> C(NUM_CONTEXTS, 0);
    std::vector<int> B(NUM_CONTEXTS, 0);
    std::vector<int> Nc(NUM_CONTEXTS, 0);

    RangeEncoder enc(*out_ptr);

    uint8_t prev_lo_hi0 = 0;  // lo of last pixel whose hi byte was 0
    std::vector<uint8_t> hi_N_arr(width, 0u);  // hi values of row above (for mean mode)

    for (uint32_t row = 0; row < height; ++row) {
        uint8_t hi_W_mean = 0;  // hi of left neighbour for mean mode, reset per row
        for (uint32_t col = 0; col < width; ++col) {
            uint16_t px = pixels[row * width + col];

            uint16_t W  = (col > 0)            ? pixels[row * width + col - 1]           : 0u;
            uint16_t WW = (col > 1)            ? pixels[row * width + col - 2]           : W;
            uint16_t N  = (row > 0)            ? pixels[(row-1) * width + col]           : W;
            uint16_t NW = (row > 0 && col > 0) ? pixels[(row-1) * width + col - 1]       : W;

            // Compute gradients once; reused for lo_hi0_ctx and (in MED mode) spatial_context.
            int D1 = 0, D2 = 0, D3 = 0;
            int lo_hi0_ctx = 0;
            if (row > 0 && col > 0) {
                D1 = static_cast<int>(N)  - static_cast<int>(NW);
                D2 = static_cast<int>(NW) - static_cast<int>(W);
                D3 = static_cast<int>(W)  - static_cast<int>(WW);
                int grad_max = std::max({std::abs(D1), std::abs(D2), std::abs(D3)});
                lo_hi0_ctx = (grad_max < GRAD_T1) ? 0 :
                             (grad_max < GRAD_T2) ? 1 :
                             (grad_max < GRAD_T3) ? 2 : 3;
            }

            uint8_t hi, lo;

            if (use_mean) {
                // ── Global-mean mode: hi-neighbor context, sign=1 always ──────
                int hi_ctx = quantize_hi_val(hi_W_mean) * HI_NBR_BINS
                           + quantize_hi_val(hi_N_arr[col]);
                uint16_t pred_adj = static_cast<uint16_t>(
                    static_cast<int>(global_mean) + C_nbr[hi_ctx]);
                uint16_t u = static_cast<uint16_t>(px - pred_adj);
                uint16_t z = zigzag_encode(u);
                hi = static_cast<uint8_t>(z >> 8);
                lo = static_cast<uint8_t>(z & 0xFF);

                enc.write(hi_nbr_models[hi_ctx], hi);
                hi_nbr_models[hi_ctx].increment(hi);

                if (hi == 0) {
                    int lo_idx = lo_hi0_ctx * LO_HI0_PREV_BINS
                               + quantize_prev_lo(prev_lo_hi0);
                    enc.write(lo_hi0_models[lo_idx], lo);
                    lo_hi0_models[lo_idx].increment(lo);
                    prev_lo_hi0 = lo;
                } else {
                    enc.write(lo_hip_models[hi - 1], lo);
                    lo_hip_models[hi - 1].increment(lo);
                }

                int r_s = (u <= 32767u) ? static_cast<int>(u)
                                        : static_cast<int>(u) - 65536;
                B_nbr[hi_ctx] += r_s; Nc_nbr[hi_ctx]++;
                if (B_nbr[hi_ctx] > Nc_nbr[hi_ctx]) {
                    C_nbr[hi_ctx]++; B_nbr[hi_ctx] -= Nc_nbr[hi_ctx];
                    if (B_nbr[hi_ctx] > Nc_nbr[hi_ctx]) B_nbr[hi_ctx] = Nc_nbr[hi_ctx];
                } else if (B_nbr[hi_ctx] < -Nc_nbr[hi_ctx]) {
                    C_nbr[hi_ctx]--; B_nbr[hi_ctx] += Nc_nbr[hi_ctx];
                    if (B_nbr[hi_ctx] < -Nc_nbr[hi_ctx]) B_nbr[hi_ctx] = -Nc_nbr[hi_ctx];
                }
                if (Nc_nbr[hi_ctx] == 512) { Nc_nbr[hi_ctx] >>= 1; B_nbr[hi_ctx] >>= 1; }

                hi_W_mean   = hi;
                hi_N_arr[col] = hi;

            } else {
                // ── MED mode: 365-gradient context + class priors ─────────────
                int sign = 1, ctx = 0;
                if (row > 0 && col > 0)
                    ctx = spatial_context(D1, D2, D3, sign);
                uint16_t pred = (row == 0 && col == 0) ? 0u : med_predict(W, N, NW);
                uint16_t pred_adj = static_cast<uint16_t>(
                    static_cast<int>(pred) + sign * C[ctx]);
                uint16_t u = static_cast<uint16_t>(px - pred_adj);
                if (sign == -1) u = static_cast<uint16_t>(0u - u);
                uint16_t z = zigzag_encode(u);
                hi = static_cast<uint8_t>(z >> 8);
                lo = static_cast<uint8_t>(z & 0xFF);

                MixedFrequencyTable mixed(hi_models[ctx], hi_class_priors[lo_hi0_ctx]);
                enc.write(mixed, hi);
                hi_models[ctx].increment(hi);
                hi_class_priors[lo_hi0_ctx].increment(hi);
                if (hi_class_priors[lo_hi0_ctx].getTotal() > HI_PRIOR_CAP)
                    hi_class_priors[lo_hi0_ctx].halve();

                if (hi == 0) {
                    int lo_idx = lo_hi0_ctx * LO_HI0_PREV_BINS
                               + quantize_prev_lo(prev_lo_hi0);
                    enc.write(lo_hi0_models[lo_idx], lo);
                    lo_hi0_models[lo_idx].increment(lo);
                    prev_lo_hi0 = lo;
                } else {
                    enc.write(lo_hip_models[hi - 1], lo);
                    lo_hip_models[hi - 1].increment(lo);
                }

                int r_s = (u <= 32767u) ? static_cast<int>(u)
                                        : static_cast<int>(u) - 65536;
                B[ctx] += r_s; Nc[ctx]++;
                if (B[ctx] > Nc[ctx]) {
                    C[ctx]++; B[ctx] -= Nc[ctx];
                    if (B[ctx] > Nc[ctx]) B[ctx] = Nc[ctx];
                } else if (B[ctx] < -Nc[ctx]) {
                    C[ctx]--; B[ctx] += Nc[ctx];
                    if (B[ctx] < -Nc[ctx]) B[ctx] = -Nc[ctx];
                }
                if (Nc[ctx] == 512) { Nc[ctx] >>= 1; B[ctx] >>= 1; }
            }
        }
    }

    enc.finish();
    return 0;
}
