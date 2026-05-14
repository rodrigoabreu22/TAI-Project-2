/*
 * TAI Project 2 — Astronomical Image Compressor (ox-astro-ratio)
 *
 * Pipeline:
 *   1. Read raw bytes; interpret as 2D array of big-endian uint16_t.
 *   2. JPEG-LS MED spatial predictor → wrapping residual → modular zigzag.
 *   3. Split zigzag value into hi byte and lo byte.
 *   4. Compute JPEG-LS 3-gradient spatial context (365 contexts):
 *        D1 = N-NW (vertical), D2 = NW-W (diagonal), D3 = W-WW (horizontal)
 *        Each quantised to {-4..4}; sign-symmetry folds 729 → 365 contexts.
 *        When sign==-1, the residual is negated before encoding.
 *   5. Per-context bias correction (Phase 2):
 *        C[ctx] is a running correction term. pred_adj = pred + sign*C[ctx].
 *        After each symbol, B[ctx] accumulates the signed residual; when the
 *        mean exceeds ±1 sample, C[ctx] is nudged by ±1 to drive B toward 0.
 *        Decoder runs the identical update, so no extra bytes are stored.
 *   6. Encode hi and lo with the same spatial context (365 models each).
 *
 * Compressed file format
 * ──────────────────────
 *  Bytes  0–3  : magic "TA2A"
 *  Bytes  4–7  : width   uint32_t LE
 *  Bytes  8–11 : height  uint32_t LE
 *  Bytes 12+   : range-coded bitstream
 *
 * Usage:  compress_astro_ratio <input_file> <output_file>
 *         compress_astro_ratio          (stdin → stdout)
 */

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "coder/RangeCoder.hpp"
#include "coder/FrequencyTable.hpp"
#include "coder/MixedFrequencyTable.hpp"
#include "model/ImagePredictor.hpp"

static constexpr uint8_t MAGIC[4] = {'T','A','2','A'};

static void write_u32le(std::ostream& out, uint32_t v) {
    out.put(static_cast<char>(v        & 0xFF));
    out.put(static_cast<char>((v >>  8) & 0xFF));
    out.put(static_cast<char>((v >> 16) & 0xFF));
    out.put(static_cast<char>((v >> 24) & 0xFF));
}

static SimpleFrequencyTable make_flat256() {
    return SimpleFrequencyTable(std::vector<uint32_t>(256, 1u));
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

    // Write header
    out_ptr->write(reinterpret_cast<const char*>(MAGIC), 4);
    write_u32le(*out_ptr, width);
    write_u32le(*out_ptr, height);

    // hi models        : 365 × 256-symbol fine per-context adaptive models.
    // hi_class_priors  : 4 × 256-symbol class-level priors (flat/slight/moderate/strong).
    //   Each prior is a bounded running summary of hi statistics for its gradient
    //   class, capped at HI_PRIOR_CAP total counts by periodic halving.
    //   At encode time hi is coded via MixedFrequencyTable(fine, class_prior):
    //     • Sparse fine context  (few visits): class prior dominates → correct prior
    //     • Dense  fine context  (many visits): fine model dominates → no dilution
    //   This is context mixing: Strategy A.
    // lo_hi0_models: 32 × 256-symbol, indexed by (grad_class × 8 + prev_lo_bin).
    //   grad_class  [0-3]: coarse gradient magnitude (flat/slight/moderate/strong).
    //   prev_lo_bin [0-7]: log-scale quantisation of the previous hi=0 lo byte.
    //   ORDER-1 on the lo byte when hi=0 captures the persistence of small
    //   residuals: if the last small-residual pixel had lo≈5, the current one
    //   is likely also near 5. 32 tables keep each well-populated (~56K samples).
    // lo_hip_models: 255 × 256-symbol, indexed by (hi-1) for hi in [1,255].
    static constexpr int    LO_HI0_CLASSES   = 4;
    static constexpr int    LO_HI0_PREV_BINS = 8;
    static constexpr int    LO_HI0_TOTAL     = LO_HI0_CLASSES * LO_HI0_PREV_BINS;
    static constexpr uint32_t HI_PRIOR_CAP   = 1024;  // keeps class priors bounded

    std::vector<SimpleFrequencyTable> hi_models, hi_class_priors,
                                      lo_hi0_models, lo_hip_models;
    hi_models.reserve(NUM_CONTEXTS);
    for (int i = 0; i < NUM_CONTEXTS; ++i)
        hi_models.push_back(make_flat256());
    hi_class_priors.reserve(LO_HI0_CLASSES);
    for (int i = 0; i < LO_HI0_CLASSES; ++i)
        hi_class_priors.push_back(make_flat256());
    lo_hi0_models.reserve(LO_HI0_TOTAL);
    for (int i = 0; i < LO_HI0_TOTAL; ++i)
        lo_hi0_models.push_back(make_flat256());
    lo_hip_models.reserve(255);
    for (int i = 0; i < 255; ++i)
        lo_hip_models.push_back(make_flat256());

    // Per-context bias correction state (all zero-initialised)
    // C[ctx] : correction added to sign*pred before residual computation
    // B[ctx] : running sum of sign-normalised residuals (drives C updates)
    // Nc[ctx]: sample count per context (halved periodically)
    std::vector<int> C(NUM_CONTEXTS, 0);
    std::vector<int> B(NUM_CONTEXTS, 0);
    std::vector<int> Nc(NUM_CONTEXTS, 0);

    RangeEncoder enc(*out_ptr);

    uint8_t prev_lo_hi0 = 0;  // lo of last pixel whose hi byte was 0

    for (uint32_t row = 0; row < height; ++row) {
        for (uint32_t col = 0; col < width; ++col) {
            uint16_t px = pixels[row * width + col];

            uint16_t W  = (col > 0)                     ? pixels[row * width + col - 1]               : 0u;
            uint16_t WW = (col > 1)                     ? pixels[row * width + col - 2]               : W;
            uint16_t N  = (row > 0)                     ? pixels[(row-1) * width + col]               : W;
            uint16_t NN = (row > 1)                     ? pixels[(row-2) * width + col]               : N;
            uint16_t NW = (row > 0 && col > 0)          ? pixels[(row-1) * width + col - 1]           : W;
            uint16_t NE = (row > 0 && col < width - 1)  ? pixels[(row-1) * width + col + 1]           : N;

            uint16_t pred = (row == 0 && col == 0) ? 0u : med_predict(W, N, NW);

            // Spatial context from three directional gradients
            int sign = 1;
            int ctx  = 0;
            int lo_hi0_ctx = 0;  // coarse activity class for lo when hi==0
            if (row > 0 && col > 0) {
                int D1 = static_cast<int>(N)  - static_cast<int>(NW);
                int D2 = static_cast<int>(NW) - static_cast<int>(W);
                int D3 = static_cast<int>(W)  - static_cast<int>(WW);
                ctx = spatial_context(D1, D2, D3, sign);
                int grad_max = std::max({std::abs(D1), std::abs(D2), std::abs(D3)});
                lo_hi0_ctx = (grad_max < GRAD_T1) ? 0 :
                             (grad_max < GRAD_T2) ? 1 :
                             (grad_max < GRAD_T3) ? 2 : 3;
            }

            // Apply bias correction: shift prediction by sign * C[ctx]
            uint16_t pred_adj = static_cast<uint16_t>(
                static_cast<int>(pred) + sign * C[ctx]);

            // Wrapping residual; negate when sign==-1 (canonical context form)
            uint16_t u = static_cast<uint16_t>(px - pred_adj);
            if (sign == -1) u = static_cast<uint16_t>(0u - u);

            uint16_t z  = zigzag_encode(u);
            uint8_t  hi = static_cast<uint8_t>(z >> 8);
            uint8_t  lo = static_cast<uint8_t>(z & 0xFF);

            // Context mixing: blend fine spatial model with class-level prior.
            // The prior converges fast (sees all pixels in its class) and provides
            // a good distribution for fine contexts that have been visited rarely.
            {
                MixedFrequencyTable mixed(hi_models[ctx], hi_class_priors[lo_hi0_ctx]);
                enc.write(mixed, hi);
            }
            hi_models[ctx].increment(hi);
            hi_class_priors[lo_hi0_ctx].increment(hi);
            // Cap class prior total to stay bounded and avoid diluting fine models.
            if (hi_class_priors[lo_hi0_ctx].getTotal() > HI_PRIOR_CAP) {
                for (uint32_t s = 0; s < 256; ++s) {
                    hi_class_priors[lo_hi0_ctx].set(
                        s, std::max(1u, hi_class_priors[lo_hi0_ctx].get(s) >> 1));
                }
            }

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

            // Update bias: accumulate sign-normalised residual and nudge C
            int r_s = (u <= 32767u) ? static_cast<int>(u)
                                    : static_cast<int>(u) - 65536;
            B[ctx] += r_s;
            Nc[ctx]++;
            if (B[ctx] > Nc[ctx]) {
                C[ctx]++;
                B[ctx] -= Nc[ctx];
                if (B[ctx] > Nc[ctx]) B[ctx] = Nc[ctx];
            } else if (B[ctx] < -Nc[ctx]) {
                C[ctx]--;
                B[ctx] += Nc[ctx];
                if (B[ctx] < -Nc[ctx]) B[ctx] = -Nc[ctx];
            }
            // Halve counters to weight recent samples more
            if (Nc[ctx] == 512) { Nc[ctx] >>= 1; B[ctx] >>= 1; }
        }
    }

    enc.finish();
    return 0;
}
