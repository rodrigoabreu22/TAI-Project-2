/*
 * TAI Project 2 — Astronomical Image Decompressor (ox-astro-ratio)
 *
 * Inverse of main_compress.cpp: reads header, reconstructs pixels using
 * the same JPEG-LS MED predictor and spatial context as the encoder.
 *
 * Usage:  decompress_astro_ratio <input_file> <output_file>
 *         decompress_astro_ratio          (stdin → stdout)
 */

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

static uint32_t read_u32le(std::istream& in) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i)
        v |= static_cast<uint32_t>(static_cast<uint8_t>(in.get())) << (8 * i);
    return v;
}

static FastFrequencyTable make_flat256() { return FastFrequencyTable{}; }

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
        std::cerr << "Usage: decompress_astro_ratio <input> <output>\n"; return 1;
    }

    // Verify magic
    uint8_t magic[4];
    in_ptr->read(reinterpret_cast<char*>(magic), 4);
    for (int i = 0; i < 4; ++i) {
        if (magic[i] != MAGIC[i]) {
            std::cerr << "Bad magic number\n"; return 1;
        }
    }

    const uint32_t width  = read_u32le(*in_ptr);
    const uint32_t height = read_u32le(*in_ptr);
    const uint64_t npix   = static_cast<uint64_t>(width) * height;

    std::vector<uint16_t> pixels(npix);

    static constexpr int      LO_HI0_CLASSES   = 4;
    static constexpr int      LO_HI0_PREV_BINS = 8;
    static constexpr int      LO_HI0_TOTAL     = LO_HI0_CLASSES * LO_HI0_PREV_BINS;
    static constexpr uint32_t HI_PRIOR_CAP     = 1024;

    std::vector<FastFrequencyTable> hi_models, hi_class_priors,
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

    // Bias correction state — must be identical to encoder
    std::vector<int> C(NUM_CONTEXTS, 0);
    std::vector<int> B(NUM_CONTEXTS, 0);
    std::vector<int> Nc(NUM_CONTEXTS, 0);

    RangeDecoder dec(*in_ptr);

    uint8_t prev_lo_hi0 = 0;

    for (uint32_t row = 0; row < height; ++row) {
        for (uint32_t col = 0; col < width; ++col) {
            uint16_t W  = (col > 0)                    ? pixels[row * width + col - 1]              : 0u;
            uint16_t WW = (col > 1)                    ? pixels[row * width + col - 2]              : W;
            uint16_t N  = (row > 0)                    ? pixels[(row-1) * width + col]              : W;
            uint16_t NN = (row > 1)                    ? pixels[(row-2) * width + col]              : N;
            uint16_t NW = (row > 0 && col > 0)         ? pixels[(row-1) * width + col - 1]          : W;
            uint16_t NE = (row > 0 && col < width - 1) ? pixels[(row-1) * width + col + 1]          : N;

            // Same context computation as encoder (pixels decoded left-to-right)
            int sign = 1;
            int ctx  = 0;
            int lo_hi0_ctx = 0;
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

            MixedFrequencyTable mixed_hi(hi_models[ctx], hi_class_priors[lo_hi0_ctx]);
            uint8_t hi = static_cast<uint8_t>(dec.read(mixed_hi));
            hi_models[ctx].increment(hi);
            hi_class_priors[lo_hi0_ctx].increment(hi);
            if (hi_class_priors[lo_hi0_ctx].getTotal() > HI_PRIOR_CAP) {
                for (uint32_t s = 0; s < 256; ++s) {
                    hi_class_priors[lo_hi0_ctx].set(
                        s, std::max(1u, hi_class_priors[lo_hi0_ctx].get(s) >> 1));
                }
            }

            uint8_t lo;
            if (hi == 0) {
                int lo_idx = lo_hi0_ctx * LO_HI0_PREV_BINS
                           + quantize_prev_lo(prev_lo_hi0);
                lo = static_cast<uint8_t>(dec.read(lo_hi0_models[lo_idx]));
                lo_hi0_models[lo_idx].increment(lo);
                prev_lo_hi0 = lo;
            } else {
                lo = static_cast<uint8_t>(dec.read(lo_hip_models[hi - 1]));
                lo_hip_models[hi - 1].increment(lo);
            }

            uint16_t z     = (static_cast<uint16_t>(hi) << 8) | lo;
            uint16_t u_norm = zigzag_decode(z);

            // Apply bias correction with the CURRENT C[ctx] — same as encoder did
            // before its own update. Pixel must be reconstructed first.
            uint16_t pred = (row == 0 && col == 0) ? 0u : med_predict(W, N, NW);
            uint16_t pred_adj = static_cast<uint16_t>(
                static_cast<int>(pred) + sign * C[ctx]);

            uint16_t u = u_norm;
            if (sign == -1) u = static_cast<uint16_t>(0u - u_norm);
            pixels[row * width + col] = static_cast<uint16_t>(pred_adj + u);

            // Bias update — identical to encoder, runs after pixel is written
            int r_s = (u_norm <= 32767u) ? static_cast<int>(u_norm)
                                         : static_cast<int>(u_norm) - 65536;
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
            if (Nc[ctx] == 512) { Nc[ctx] >>= 1; B[ctx] >>= 1; }
        }
    }

    // Write as big-endian uint16_t
    std::vector<uint8_t> out_bytes(npix * 2);
    for (uint64_t i = 0; i < npix; ++i) {
        out_bytes[2*i]   = static_cast<uint8_t>(pixels[i] >> 8);
        out_bytes[2*i+1] = static_cast<uint8_t>(pixels[i] & 0xFF);
    }
    out_ptr->write(reinterpret_cast<const char*>(out_bytes.data()),
                   static_cast<std::streamsize>(out_bytes.size()));
    return 0;
}
