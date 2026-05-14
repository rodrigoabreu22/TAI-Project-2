/*
 * TAI Project 2 — Astronomical Image Compressor (ox-astro-balanced)
 *
 * Pipeline:
 *   1. Read raw bytes; interpret as 2D big-endian uint16_t array.
 *   2. Adaptive predictor selection (scan full image before coding):
 *        Compute MAE of GAP predictor and global-mean predictor.
 *        If MAE(mean) < 0.9 × MAE(GAP), use mean predictor (better for
 *        weakly-correlated / noise-dominated images such as flat fields).
 *        Otherwise use robust GAP (better for spatially structured images).
 *   3. Wrapping residual → modular zigzag → hi/lo byte split.
 *   4. Per-context bias correction (16 contexts, LOCO-I style).
 *   5. Encode hi with spatial context: quantize(hi_W)*4 + quantize(hi_N).
 *   6. Encode lo:
 *        hi == 0 : model[(grad_class)*8 + quantize_prev_lo]  (32 models)
 *        hi  > 0 : model[hi - 1]                             (255 models)
 *
 * Compressed file format
 * ──────────────────────
 *  Bytes  0– 3 : magic "TA2B"
 *  Bytes  4– 7 : width        uint32_t LE
 *  Bytes  8–11 : height       uint32_t LE
 *  Byte     12 : pred_mode    0=robust-GAP  1=global-mean
 *  Bytes 13–14 : global_mean  uint16_t LE  (used when pred_mode==1)
 *  Bytes   15+ : range-coded bitstream
 *
 * Usage:  compress_astro_balanced <input_file> <output_file>
 *         compress_astro_balanced          (stdin → stdout)
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
#include "coder/FenwickFrequencyTable.hpp"
#include "model/ImagePredictor.hpp"

static constexpr uint8_t MAGIC[4] = {'T','A','2','B'};

static void write_u32le(std::ostream& out, uint32_t v) {
    out.put(static_cast<char>(v         & 0xFF));
    out.put(static_cast<char>((v >>  8) & 0xFF));
    out.put(static_cast<char>((v >> 16) & 0xFF));
    out.put(static_cast<char>((v >> 24) & 0xFF));
}

static void write_u16le(std::ostream& out, uint16_t v) {
    out.put(static_cast<char>(v       & 0xFF));
    out.put(static_cast<char>((v >> 8) & 0xFF));
}

static FenwickFrequencyTable make_fenwick256() {
    FenwickFrequencyTable t(256);
    for (int i = 0; i < 256; ++i) t.increment(i);
    return t;
}

int main(int argc, char* argv[]) {
    std::istream* in_ptr  = &std::cin;
    std::ostream* out_ptr = &std::cout;
    std::ifstream fin;
    std::ofstream fout;

    if (argc >= 3) {
        fin.open(argv[1], std::ios::binary);
        if (!fin)  { std::cerr << "Cannot open input: "  << argv[1] << '\n'; return 1; }
        fout.open(argv[2], std::ios::binary);
        if (!fout) { std::cerr << "Cannot open output: " << argv[2] << '\n'; return 1; }
        in_ptr  = &fin;
        out_ptr = &fout;
    } else if (argc == 1) {
        std::cin.sync_with_stdio(false);
    } else {
        std::cerr << "Usage: compress_astro_balanced <input> <output>\n"; return 1;
    }

    std::vector<uint8_t> raw(std::istreambuf_iterator<char>(*in_ptr), {});
    if (raw.size() % 2 != 0) {
        std::cerr << "Input size not even\n"; return 1;
    }

    const uint64_t npix = raw.size() / 2;
    uint32_t width = 0, height = 0;
    uint32_t sq = static_cast<uint32_t>(std::sqrt(static_cast<double>(npix)));
    if ((uint64_t)sq * sq == npix) {
        width = height = sq;
    } else {
        width = static_cast<uint32_t>(npix); height = 1;
        for (uint32_t w : {1500u, 2048u, 1024u, 512u, 256u})
            if (npix % w == 0) { width = w; height = static_cast<uint32_t>(npix / w); break; }
    }

    std::vector<uint16_t> pixels(npix);
    for (uint64_t i = 0; i < npix; ++i)
        pixels[i] = (static_cast<uint16_t>(raw[2*i]) << 8) | raw[2*i+1];

    // ── Adaptive predictor selection ───────────────────────────────────────
    // Scan full image and estimate the marginal entropy of residuals after
    // zigzag + hi/lo split for both GAP and global-mean predictors.
    // Selects whichever predictor produces lower estimated entropy.
    // This is more accurate than MAE comparison: an image with bright saturated
    // stars has low GAP-MAE (smooth background is easy) but high GAP-entropy
    // (huge residuals near stars create a bimodal distribution).
    uint64_t px_sum = 0;
    for (uint64_t i = 0; i < npix; ++i) px_sum += pixels[i];
    const uint16_t global_mean = static_cast<uint16_t>(px_sum / npix);

    std::array<uint64_t, 256> hi_cnt_gap{},    hi_cnt_mean{};
    std::array<uint64_t, 256> lo_hi0_cnt_gap{}, lo_hi0_cnt_mean{};
    std::array<uint64_t, 256> lo_hip_cnt_gap{}, lo_hip_cnt_mean{};

    for (uint32_t row = 0; row < height; ++row) {
        for (uint32_t col = 0; col < width; ++col) {
            uint16_t px = pixels[row * width + col];
            uint16_t W  = (col > 0)                    ? pixels[row * width + col - 1]             : 0u;
            uint16_t WW = (col > 1)                    ? pixels[row * width + col - 2]             : W;
            uint16_t N  = (row > 0)                    ? pixels[(row-1) * width + col]             : W;
            uint16_t NN = (row > 1)                    ? pixels[(row-2) * width + col]             : N;
            uint16_t NW = (row > 0 && col > 0)         ? pixels[(row-1) * width + col - 1]         : W;
            uint16_t NE = (row > 0 && col < width - 1) ? pixels[(row-1) * width + col + 1]         : N;

            // GAP residual
            uint16_t pg = (row == 0 && col == 0) ? 0u : gap_predict(W, WW, N, NN, NW, NE);
            uint16_t zg = zigzag_encode(static_cast<uint16_t>(static_cast<int>(px) - static_cast<int>(pg)));
            uint8_t hig = static_cast<uint8_t>(zg >> 8);
            uint8_t log_ = static_cast<uint8_t>(zg & 0xFF);
            hi_cnt_gap[hig]++;
            if (hig == 0) lo_hi0_cnt_gap[log_]++; else lo_hip_cnt_gap[log_]++;

            // Mean residual
            uint16_t zm = zigzag_encode(static_cast<uint16_t>(static_cast<int>(px) - static_cast<int>(global_mean)));
            uint8_t him = static_cast<uint8_t>(zm >> 8);
            uint8_t lom = static_cast<uint8_t>(zm & 0xFF);
            hi_cnt_mean[him]++;
            if (him == 0) lo_hi0_cnt_mean[lom]++; else lo_hip_cnt_mean[lom]++;
        }
    }

    // Compute Shannon entropy from a histogram
    auto entropy = [](const std::array<uint64_t, 256>& cnt) -> double {
        uint64_t total = 0;
        for (auto c : cnt) total += c;
        if (total == 0) return 0.0;
        double h = 0.0;
        const double inv = 1.0 / static_cast<double>(total);
        for (auto c : cnt) {
            if (c > 0) {
                double p = static_cast<double>(c) * inv;
                h -= p * std::log2(p);
            }
        }
        return h;
    };

    // Marginal entropy estimate: H(hi) + frac_hi0 * H(lo|hi=0) + frac_hip * H(lo|hi>0)
    auto est_entropy = [&](const std::array<uint64_t,256>& hc,
                           const std::array<uint64_t,256>& lc0,
                           const std::array<uint64_t,256>& lcp) -> double {
        uint64_t n_hi0 = 0; for (auto c : lc0) n_hi0 += c;
        double frac0 = static_cast<double>(n_hi0) / static_cast<double>(npix);
        return entropy(hc) + frac0 * entropy(lc0) + (1.0 - frac0) * entropy(lcp);
    };

    const double est_gap  = est_entropy(hi_cnt_gap,  lo_hi0_cnt_gap,  lo_hip_cnt_gap);
    const double est_mean = est_entropy(hi_cnt_mean, lo_hi0_cnt_mean, lo_hip_cnt_mean);

    // Use mean predictor when its estimated entropy is strictly lower
    const bool use_mean_pred = (est_mean < est_gap);

    // ── Header ─────────────────────────────────────────────────────────────
    out_ptr->write(reinterpret_cast<const char*>(MAGIC), 4);
    write_u32le(*out_ptr, width);
    write_u32le(*out_ptr, height);
    out_ptr->put(static_cast<char>(use_mean_pred ? 1 : 0));
    write_u16le(*out_ptr, global_mean);

    // ── Models (all Fenwick for O(log 256) operations) ─────────────────────
    std::vector<FenwickFrequencyTable> hi_models(16, make_fenwick256());
    std::vector<FenwickFrequencyTable> lo_hi0_models(32, make_fenwick256());
    std::vector<FenwickFrequencyTable> lo_hip_models(255, make_fenwick256());

    auto quantize = [](uint8_t h) -> int {
        if (h == 0) return 0;
        if (h <= 2) return 1;
        if (h <= 7) return 2;
        return 3;
    };

    auto quantize_prev_lo = [](uint8_t lo) -> int {
        if (lo == 0)   return 0;
        if (lo < 4)    return 1;
        if (lo < 8)    return 2;
        if (lo < 16)   return 3;
        if (lo < 32)   return 4;
        if (lo < 64)   return 5;
        if (lo < 128)  return 6;
        return 7;
    };

    std::vector<int> C(16, 0), B(16, 0), Nc(16, 0);
    std::vector<uint8_t> hi_N(width, 0);
    uint8_t prev_lo_hi0 = 0;

    RangeEncoder enc(*out_ptr);

    for (uint32_t row = 0; row < height; ++row) {
        uint8_t hi_W = 0;
        for (uint32_t col = 0; col < width; ++col) {
            uint16_t px = pixels[row * width + col];

            uint16_t W  = (col > 0)                    ? pixels[row * width + col - 1]             : 0u;
            uint16_t WW = (col > 1)                    ? pixels[row * width + col - 2]             : W;
            uint16_t N  = (row > 0)                    ? pixels[(row-1) * width + col]             : W;
            uint16_t NN = (row > 1)                    ? pixels[(row-2) * width + col]             : N;
            uint16_t NW = (row > 0 && col > 0)         ? pixels[(row-1) * width + col - 1]         : W;
            uint16_t NE = (row > 0 && col < width - 1) ? pixels[(row-1) * width + col + 1]         : N;

            uint8_t ctx_hi = static_cast<uint8_t>(quantize(hi_W) * 4 + quantize(hi_N[col]));

            uint16_t pred;
            if (use_mean_pred) {
                pred = global_mean;
            } else {
                pred = (row == 0 && col == 0)
                    ? 0u
                    : robust_gap_predict(W, WW, N, NN, NW, NE, hi_W, hi_N[col]);
            }
            uint16_t pred_adj = static_cast<uint16_t>(static_cast<int>(pred) + C[ctx_hi]);

            uint16_t u  = static_cast<uint16_t>(px - pred_adj);
            uint16_t z  = zigzag_encode(u);
            uint8_t  hi = static_cast<uint8_t>(z >> 8);
            uint8_t  lo = static_cast<uint8_t>(z & 0xFF);

            enc.write(hi_models[ctx_hi], hi);
            hi_models[ctx_hi].increment(hi);

            if (hi == 0) {
                int grad_class = std::min(quantize(hi_W) + quantize(hi_N[col]), 3);
                int lo_idx = grad_class * 8 + quantize_prev_lo(prev_lo_hi0);
                enc.write(lo_hi0_models[lo_idx], lo);
                lo_hi0_models[lo_idx].increment(lo);
                prev_lo_hi0 = lo;
            } else {
                enc.write(lo_hip_models[hi - 1], lo);
                lo_hip_models[hi - 1].increment(lo);
            }

            int r_s = (u <= 32767u) ? static_cast<int>(u) : static_cast<int>(u) - 65536;
            B[ctx_hi] += r_s;
            Nc[ctx_hi]++;
            if (B[ctx_hi] > Nc[ctx_hi]) {
                C[ctx_hi]++;
                B[ctx_hi] -= Nc[ctx_hi];
                if (B[ctx_hi] > Nc[ctx_hi]) B[ctx_hi] = Nc[ctx_hi];
            } else if (B[ctx_hi] < -Nc[ctx_hi]) {
                C[ctx_hi]--;
                B[ctx_hi] += Nc[ctx_hi];
                if (B[ctx_hi] < -Nc[ctx_hi]) B[ctx_hi] = -Nc[ctx_hi];
            }
            if (Nc[ctx_hi] == 512) { Nc[ctx_hi] >>= 1; B[ctx_hi] >>= 1; }

            hi_W      = hi;
            hi_N[col] = hi;
        }
    }

    enc.finish();
    return 0;
}
