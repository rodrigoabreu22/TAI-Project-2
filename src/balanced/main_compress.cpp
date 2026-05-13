/*
 * TAI Project 2 — Astronomical Image Compressor (ox-astro-balanced)
 *
 * Pipeline:
 *   1. Read raw bytes; interpret as 2D big-endian uint16_t array.
 *   2. JPEG-LS MED spatial predictor → wrapping residual → modular zigzag.
 *   3. Split zigzag value into hi byte and lo byte.
 *   4. GAP predictor using 6 neighbours (W, WW, N, NN, NW, NE).
 *   5. Per-context bias correction (C/B/Nc arrays, 16 contexts):
 *        pred_adj = pred + C[ctx].  After each pixel, the signed residual
 *        updates B[ctx]; when the mean exceeds ±1, C[ctx] is nudged ±1.
 *        Both sides run identical updates — zero overhead in the bitstream.
 *   6. Encode hi with spatial context: quantize(hi_W)*4 + quantize(hi_N) → 16 models.
 *   7. Encode lo with a split strategy:
 *        hi == 0 : model[(grad_class)*8 + quantize_prev_lo]  (32 models)
 *        hi  > 0 : model[hi - 1]                             (255 models)
 *
 * Compressed file format
 * ──────────────────────
 *  Bytes  0–3  : magic "TA2B"
 *  Bytes  4–7  : width   uint32_t LE
 *  Bytes  8–11 : height  uint32_t LE
 *  Bytes 12+   : range-coded bitstream
 *
 * Usage:  compress_astro_balanced <input_file> <output_file>
 *         compress_astro_balanced          (stdin → stdout)
 */

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

    out_ptr->write(reinterpret_cast<const char*>(MAGIC), 4);
    write_u32le(*out_ptr, width);
    write_u32le(*out_ptr, height);

    // All three model arrays now use Fenwick for O(log 256) operations.
    std::vector<FenwickFrequencyTable> hi_models(16, make_fenwick256());
    std::vector<FenwickFrequencyTable> lo_hi0_models(32, make_fenwick256());
    std::vector<FenwickFrequencyTable> lo_hip_models(255, make_fenwick256());

    // Maps residual hi byte to one of 4 activity levels: 0 | 1-2 | 3-7 | 8+
    auto quantize = [](uint8_t h) -> int {
        if (h == 0) return 0;
        if (h <= 2) return 1;
        if (h <= 7) return 2;
        return 3;
    };

    // Log-scale quantisation of the previous hi=0 lo byte (8 bins)
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

    // Per-context bias correction state (16 contexts, all zero-initialised)
    std::vector<int> C(16, 0), B(16, 0), Nc(16, 0);

    std::vector<uint8_t> hi_N(width, 0);  // hi bytes of N neighbours (previous row)
    uint8_t prev_lo_hi0 = 0;              // lo of last pixel whose hi was 0

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

            // Bias-corrected prediction
            uint16_t pred = (row == 0 && col == 0) ? 0u : gap_predict(W, WW, N, NN, NW, NE);
            uint16_t pred_adj = static_cast<uint16_t>(static_cast<int>(pred) + C[ctx_hi]);

            uint16_t u  = static_cast<uint16_t>(px - pred_adj);  // wrapping residual
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

            // Bias correction update
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
