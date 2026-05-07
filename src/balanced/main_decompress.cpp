/*
 * TAI Project 1 — MTF+ORDER=0 Decompressor (BWT whole-file + parallel MTF chunks)
 *
 * Usage:  decompress <compressed_file> <output_file>
 *         decompress          (stdin → stdout)
 */

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include "coder/RangeCoder.hpp"
#include "coder/FrequencyTable.hpp"
#include "coder/FenwickFrequencyTable.hpp"
#include "model/BwtTransform.hpp"


// ── Per-chunk data ────────────────────────────────────────────────────────────
struct ChunkData {
    uint32_t k;
    std::vector<uint8_t> alphabet;
    std::vector<uint8_t> bitstream;
};


// ── Decode one chunk -> portion of bwt_data (runs in a worker thread) ─────────
static std::vector<uint8_t> decompress_chunk(const ChunkData &cd)
{
    // MTF list: [0, 1, ..., k-1]
    std::vector<uint32_t> mtf_list(cd.k);
    std::iota(mtf_list.begin(), mtf_list.end(), 0u);

    // Same models as compressor — Fenwick for O(log k) findSymbol
    FenwickFrequencyTable rank_model(cd.k + 1);
    for (uint32_t i = 0; i <= cd.k; i++) rank_model.increment(i);
    constexpr uint32_t COUNT_CTXS = 4;
    constexpr uint32_t THRESH = 8u;
    auto rank_ctx = [](uint32_t r) -> uint32_t {
        if (r == 0) return 0;
        if (r == 1) return 1;
        if (r <= 3) return 2;
        return 3;
    };
    std::vector<uint32_t> cnt_init(THRESH + 1, 1u);
    std::vector<SimpleFrequencyTable> cnt_models(COUNT_CTXS, SimpleFrequencyTable(cnt_init));
    std::vector<uint32_t> exp_init(32, 1u);
    SimpleFrequencyTable exp_model(exp_init);
    std::vector<uint32_t> bit_init(2, 1u);
    SimpleFrequencyTable bit_model(bit_init);

    std::string raw(cd.bitstream.begin(), cd.bitstream.end());
    std::istringstream buf(raw, std::ios::binary);
    RangeDecoder dec(buf);

    std::vector<uint8_t> bwt_portion;

    while (true) {
        uint32_t r = dec.read(rank_model);
        if (r == cd.k) break;  // EOF marker
        rank_model.increment(r);

        // Recover symbol at rank r, move to front
        uint32_t sym = mtf_list[r];
        for (uint32_t i = r; i > 0; i--) mtf_list[i] = mtf_list[i - 1];
        mtf_list[0] = sym;

        // Decode count via direct adaptive model
        uint32_t ctx = rank_ctx(r);
        uint32_t sym_cnt = dec.read(cnt_models[ctx]);
        cnt_models[ctx].increment(sym_cnt);

        uint32_t cnt;
        if (sym_cnt < THRESH) {
            cnt = sym_cnt + 1u;  // counts 1..THRESH
        } else {
            // Elias-gamma for overflow counts
            uint32_t b = dec.read(exp_model);
            exp_model.increment(b);
            uint32_t residual = 0;
            for (uint32_t j = 0; j < b; j++)
                residual = (residual << 1) | dec.read(bit_model);
            cnt = (1u << b) | residual;
        }

        uint8_t byte = cd.alphabet[sym];
        for (uint32_t j = 0; j < cnt; j++)
            bwt_portion.push_back(byte);
    }

    return bwt_portion;
}


// ── ORDER-0 adaptive range decoding of raw bytes ─────────────────────────────
// Uses FenwickFrequencyTable for O(log 256) findSymbol per symbol.
static std::vector<uint8_t> order0_decode(const std::vector<uint8_t>& bitstream,uint32_t original_size) 
{
    FenwickFrequencyTable model(256);
    for (int i = 0; i < 256; i++) model.increment(i);
    std::string raw(bitstream.begin(), bitstream.end());
    std::istringstream buf(raw, std::ios::binary);
    RangeDecoder dec(buf);
    std::vector<uint8_t> result;
    result.reserve(original_size);
    for (uint32_t i = 0; i < original_size; i++) {
        uint32_t sym = dec.read(model);
        model.increment(sym);
        result.push_back(static_cast<uint8_t>(sym));
    }
    return result;
}


// ── Read a uint32_t little-endian ─────────────────────────────────────────────
static bool read_u32le(std::istream &in, uint32_t &v) {
    v = 0;
    for (int i = 0; i < 4; i++) {
        int b = in.get();
        if (!in) return false;
        v |= static_cast<uint32_t>(static_cast<uint8_t>(b)) << (i * 8);
    }
    return true;
}


int main(int argc, char *argv[]) {
    if (argc != 1 && argc != 3) {
        std::cerr << "Usage: decompress <compressed_file> <output_file>\n"
                     "       decompress          (stdin -> stdout)\n";
        return EXIT_FAILURE;
    }

    // ── Open input ────────────────────────────────────────────────────────────
    std::ifstream file_in;
    if (argc == 3) {
        file_in.open(argv[1], std::ios::binary);
        if (!file_in) {
            std::cerr << "Error: cannot open input file: " << argv[1] << "\n";
            return EXIT_FAILURE;
        }
    }
    std::istream &in = (argc == 3) ? static_cast<std::istream &>(file_in) : std::cin;

    // ── Read mode byte ────────────────────────────────────────────────────────
    int mode_byte = in.get();
    if (!in) {
        std::cerr << "Error: empty or truncated input.\n";
        return EXIT_FAILURE;
    }
    uint8_t mode = static_cast<uint8_t>(mode_byte);

    // ── Open output ───────────────────────────────────────────────────────────
    std::ofstream file_out;
    if (argc == 3) {
        file_out.open(argv[2], std::ios::binary);
        if (!file_out) {
            std::cerr << "Error: cannot open output file: " << argv[2] << "\n";
            return EXIT_FAILURE;
        }
    }
    std::ostream &out = (argc == 3) ? static_cast<std::ostream &>(file_out) : std::cout;

    if (mode == 1) {
        // ── ORDER-0 path ──────────────────────────────────────────────────────
        uint32_t original_size = 0, bitstream_size = 0;
        if (!read_u32le(in, original_size) || !read_u32le(in, bitstream_size)) {
            std::cerr << "Error: truncated ORDER-0 header.\n";
            return EXIT_FAILURE;
        }
        std::vector<uint8_t> bitstream(bitstream_size);
        in.read(reinterpret_cast<char *>(bitstream.data()),
                static_cast<std::streamsize>(bitstream_size));
        if (!in) {
            std::cerr << "Error: truncated ORDER-0 bitstream.\n";
            return EXIT_FAILURE;
        }
        std::vector<uint8_t> original = order0_decode(bitstream, original_size);
        out.write(reinterpret_cast<const char *>(original.data()),
                  static_cast<std::streamsize>(original.size()));
        return EXIT_SUCCESS;
    }

    // ── BWT+MTF path (mode == 0) ──────────────────────────────────────────────
    uint32_t primary_index = 0, num_chunks = 0;
    if (!read_u32le(in, primary_index) || !read_u32le(in, num_chunks)) {
        std::cerr << "Error: truncated global header.\n";
        return EXIT_FAILURE;
    }

    // ── Read all chunk headers + bitstreams ───────────────────────────────────
    std::vector<ChunkData> chunks(num_chunks);
    for (uint32_t i = 0; i < num_chunks; i++) {
        ChunkData &cd = chunks[i];

        uint32_t bitstream_size = 0;
        if (!read_u32le(in, bitstream_size)) {
            std::cerr << "Error: truncated bitstream_size (chunk " << i << ").\n";
            return EXIT_FAILURE;
        }

        uint32_t k_raw = static_cast<uint8_t>(in.get());
        if (!in) {
            std::cerr << "Error: truncated k_raw (chunk " << i << ").\n";
            return EXIT_FAILURE;
        }
        cd.k = (k_raw == 0u) ? 256u : k_raw;

        cd.alphabet.resize(cd.k);
        if (k_raw != 0u) {
            in.read(reinterpret_cast<char *>(cd.alphabet.data()),
                    static_cast<std::streamsize>(cd.k));
            if (!in) {
                std::cerr << "Error: truncated alphabet (chunk " << i << ").\n";
                return EXIT_FAILURE;
            }
        } else {
            for (uint32_t j = 0; j < 256u; j++)
                cd.alphabet[j] = static_cast<uint8_t>(j);
        }

        cd.bitstream.resize(bitstream_size);
        in.read(reinterpret_cast<char *>(cd.bitstream.data()),
                static_cast<std::streamsize>(bitstream_size));
        if (!in) {
            std::cerr << "Error: truncated bitstream (chunk " << i << ").\n";
            return EXIT_FAILURE;
        }
    }

    // ── Decode chunks in parallel ─────────────────────────────────────────────
    std::vector<std::future<std::vector<uint8_t>>> futures;
    futures.reserve(num_chunks);
    for (const auto &cd : chunks)
        futures.push_back(std::async(std::launch::async, decompress_chunk, cd));

    // ── Assemble full bwt_buf in order ────────────────────────────────────────
    std::vector<uint8_t> bwt_buf;
    for (auto &f : futures) {
        std::vector<uint8_t> portion = f.get();
        bwt_buf.insert(bwt_buf.end(), portion.begin(), portion.end());
    }

    // ── BWT inverse on the whole buffer ───────────────────────────────────────
    std::vector<uint8_t> original = bwt_inverse(bwt_buf, primary_index);
    out.write(reinterpret_cast<const char *>(original.data()),
              static_cast<std::streamsize>(original.size()));

    return EXIT_SUCCESS;
}
