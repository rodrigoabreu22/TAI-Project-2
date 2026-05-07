/*
 * TAI Project 1 — Compressor (Block BWT + MTF + Context Mixing + Range Coding)
 *
 * Pipeline:
 *   1. Read input into memory.
 *   2. For each 4 MiB block:
 *        - Compute bigram mutual information (MI) of the block.
 *        - If MI < kDirectThreshold: encode raw bytes directly with CM
 *          (BWT produces no useful clustering when bytes are independent).
 *        - Otherwise: apply BWT then MTF, then encode with CM.
 *   3. Encode each (transformed) block bit-by-bit with a context-mixing model
 *      and range coder.
 *
 * File format (TA10)
 * ──────────────────
 *  Bytes 0-3    Magic "TA10"
 *  Bytes 4-11   uint64_t original_size  (little-endian)
 *  Bytes 12-15  uint32_t block_size     (little-endian)
 *  Bytes 16-19  uint32_t block_count    (little-endian)
 *  Next         uint32_t tag per block  (little-endian)
 *                 bit 31 = 1 → direct CM mode (no BWT+MTF)
 *                 bit 31 = 0 → BWT+MTF mode; bits 30-0 = BWT primary_index
 *  Rest         Range-coded CM bitstream of the blocks
 *
 * Usage:  compress <input_file> <output_file> [block_size_kib]
 *         compress          (stdin -> stdout)
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "coder/RangeCoder.hpp"
#include "model/BwtTransform.hpp"
#include "model/ContextMixModel.hpp"
#include "model/MoveToFront.hpp"

namespace {

// Bigram mutual information: MI = 2·H(bytes) - H(bigrams).
// Measures how strongly adjacent bytes are correlated.
// MI ≈ 0  → bytes are nearly independent; BWT produces no useful clustering.
// MI > 0.1 → strong inter-byte correlations; BWT+MTF is beneficial.
static constexpr float kDirectThreshold = 0.10f;
static constexpr std::uint32_t kDirectFlag = 0x80000000u;  // bit 31

float computeMI(const std::vector<std::uint8_t> &block) {
    const std::size_t n = block.size();
    if (n < 2) return 0.0f;

    std::uint32_t uni[256] = {};
    for (std::uint8_t b : block) uni[b]++;

    std::vector<std::uint32_t> bi(65536, 0);
    for (std::size_t i = 0; i + 1 < n; ++i)
        bi[(static_cast<std::uint32_t>(block[i]) << 8) | block[i + 1]]++;

    const double fn = static_cast<double>(n);
    double h1 = 0.0;
    for (int i = 0; i < 256; ++i) {
        if (!uni[i]) continue;
        const double p = uni[i] / fn;
        h1 -= p * log2(p);
    }

    const double fm = static_cast<double>(n - 1);
    double h2 = 0.0;
    for (int i = 0; i < 65536; ++i) {
        if (!bi[i]) continue;
        const double p = bi[i] / fm;
        h2 -= p * std::log2(p);
    }

    return static_cast<float>(2.0 * h1 - h2);
}

void writeUint64LE(std::ostream &out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i)
        out.put(static_cast<char>((v >> (i * 8)) & 0xFFu));
}

void writeUint32LE(std::ostream &out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i)
        out.put(static_cast<char>((v >> (i * 8)) & 0xFFu));
}

}  // namespace

int main(int argc, char *argv[]) {
    if (argc != 1 && argc != 3 && argc != 4) {
        std::cerr << "Usage: compress <input_file> <output_file> [block_size_kib]\n"
                     "       compress          (stdin -> stdout)\n";
        return EXIT_FAILURE;
    }

    // --- Read input ---
    std::ifstream file_in;
    if (argc >= 3) {
        file_in.open(argv[1], std::ios::binary);
        if (!file_in) {
            std::cerr << "Error: cannot open input file: " << argv[1] << "\n";
            return EXIT_FAILURE;
        }
    }
    std::istream &in = (argc >= 3) ? static_cast<std::istream &>(file_in) : std::cin;

    std::vector<std::uint8_t> raw_data;
    for (int b; (b = in.get()) != std::char_traits<char>::eof();)
        raw_data.push_back(static_cast<std::uint8_t>(b));

    // --- Parse optional block size ---
    constexpr std::uint32_t kDefaultBlockSize = 4u * 1024u * 1024u;
    std::uint32_t block_size = kDefaultBlockSize;
    if (argc == 4) {
        const long long kib = std::atoll(argv[3]);
        const unsigned long long bytes = static_cast<unsigned long long>(kib) * 1024ull;
        if (kib <= 0 || bytes == 0 || bytes > 0xFFFFFFFFull) {
            std::cerr << "Error: invalid block_size_kib.\n";
            return EXIT_FAILURE;
        }
        block_size = static_cast<std::uint32_t>(bytes);
    }

    // --- Per-block: MI filter, then BWT+MTF or direct ---
    const std::size_t n_blocks = (raw_data.size() + block_size - 1) / block_size;
    std::vector<std::vector<std::uint8_t>> blocks;
    std::vector<std::uint32_t> tags;  // bit 31: direct flag; bits 30-0: BWT primary_index
    blocks.reserve(n_blocks);
    tags.reserve(n_blocks);

    for (std::size_t offset = 0; offset < raw_data.size(); offset += block_size) {
        const std::size_t len = std::min<std::size_t>(block_size, raw_data.size() - offset);
        std::vector<std::uint8_t> block(raw_data.begin() + offset,
                                        raw_data.begin() + offset + len);

        if (computeMI(block) < kDirectThreshold) {
            // Bytes are nearly independent: BWT would not cluster them usefully.
            // Encode raw bytes directly with the CM model.
            blocks.push_back(std::move(block));
            tags.push_back(kDirectFlag);
        } else {
            auto [bwt_out, primary_index] = bwt_forward(block);
            blocks.push_back(mtf_forward(bwt_out));
            tags.push_back(primary_index);  // bit 31 is 0 (primary_index < block_size < 2^22)
        }
    }

    // --- Open output and write header ---
    std::ofstream file_out;
    if (argc >= 3) {
        file_out.open(argv[2], std::ios::binary);
        if (!file_out) {
            std::cerr << "Error: cannot open output file: " << argv[2] << "\n";
            return EXIT_FAILURE;
        }
    }
    std::ostream &out = (argc >= 3) ? static_cast<std::ostream &>(file_out) : std::cout;

    out.write("TA10", 4);
    writeUint64LE(out, static_cast<std::uint64_t>(raw_data.size()));
    writeUint32LE(out, block_size);
    writeUint32LE(out, static_cast<std::uint32_t>(n_blocks));
    for (std::uint32_t tag : tags)
        writeUint32LE(out, tag);

    // --- Range-encode each block ---
    try {
        RangeEncoder encoder(out);
        for (const auto &block : blocks) {
            ContextMixModel model;
            for (std::uint8_t byte : block) {
                for (int bit = 7; bit >= 0; --bit) {
                    const std::uint32_t prob1 = model.predict();
                    BinaryFrequencyTable freqs(prob1);
                    encoder.write(freqs, (byte >> bit) & 1u);
                    model.update((byte >> bit) & 1u);
                }
            }
        }
        encoder.finish();
    } catch (const std::exception &e) {
        std::cerr << "Encoding error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
