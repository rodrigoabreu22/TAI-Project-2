/*
 * TAI Project 2 — Astronomical Image Compressor (ox-astro-balanced)
 *
 * Pipeline:
 *   1. Read raw bytes; interpret as 2D big-endian uint16_t array.
 *   2. JPEG-LS MED spatial predictor → wrapping residual → modular zigzag.
 *   3. Split zigzag value into hi byte and lo byte.
 *   4. Encode hi with ORDER-1 (context = previous hi byte).
 *      Encode lo with ORDER-0 (single shared model — simpler/faster than ratio).
 *
 * Difference from ratio: lo byte uses a single adaptive model instead of
 * 256 per-hi-context models. This halves memory and improves cache behaviour
 * at a small cost in compression ratio.
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
#include "model/ImagePredictor.hpp"

static constexpr uint8_t MAGIC[4] = {'T','A','2','B'};

static void write_u32le(std::ostream& out, uint32_t v) {
    out.put(static_cast<char>(v         & 0xFF));
    out.put(static_cast<char>((v >>  8) & 0xFF));
    out.put(static_cast<char>((v >> 16) & 0xFF));
    out.put(static_cast<char>((v >> 24) & 0xFF));
}

static SimpleFrequencyTable make_flat256() {
    return SimpleFrequencyTable(std::vector<uint32_t>(256, 1u));
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

    // ORDER-1 models for hi byte (context = previous hi)
    std::vector<SimpleFrequencyTable> hi_models;
    hi_models.reserve(256);
    for (int i = 0; i < 256; ++i)
        hi_models.push_back(make_flat256());

    // Single ORDER-0 model for lo byte
    SimpleFrequencyTable lo_model = make_flat256();

    RangeEncoder enc(*out_ptr);

    uint8_t prev_hi = 0;
    for (uint32_t row = 0; row < height; ++row) {
        for (uint32_t col = 0; col < width; ++col) {
            uint16_t px = pixels[row * width + col];

            uint16_t W  = (col > 0)            ? pixels[row * width + col - 1]         : 0u;
            uint16_t N  = (row > 0)            ? pixels[(row-1) * width + col]          : W;
            uint16_t NW = (row > 0 && col > 0) ? pixels[(row-1) * width + col - 1]     : W;

            uint16_t pred = (row == 0 && col == 0) ? 0u : med_predict(W, N, NW);
            uint16_t u    = zigzag_encode(static_cast<uint16_t>(px - pred));

            uint8_t hi = static_cast<uint8_t>(u >> 8);
            uint8_t lo = static_cast<uint8_t>(u & 0xFF);

            enc.write(hi_models[prev_hi], hi);
            hi_models[prev_hi].increment(hi);

            enc.write(lo_model, lo);
            lo_model.increment(lo);

            prev_hi = hi;
        }
    }

    enc.finish();
    return 0;
}
