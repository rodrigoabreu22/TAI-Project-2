/*
 * TAI Project 1 — Decompressor (Block BWT + MTF + Context Mixing + Range Coding)
 *
 * Reads a TA10 file and reconstructs the original data exactly.
 * Uses the same context-mixing model as the compressor to reproduce
 * identical probability predictions without any extra data in the file.
 *
 * Usage:  decompress <compressed_file> <output_file>
 *         decompress          (stdin -> stdout)
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "coder/RangeCoder.hpp"
#include "model/BwtTransform.hpp"
#include "model/ContextMixModel.hpp"
#include "model/MoveToFront.hpp"

namespace {

bool readUint64LE(std::istream &in, std::uint64_t &v) {
    v = 0;
    for (int i = 0; i < 8; ++i) {
        const int b = in.get();
        if (!in) return false;
        v |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(b)) << (i * 8);
    }
    return true;
}

bool readUint32LE(std::istream &in, std::uint32_t &v) {
    v = 0;
    for (int i = 0; i < 4; ++i) {
        const int b = in.get();
        if (!in) return false;
        v |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(b)) << (i * 8);
    }
    return true;
}

}  // namespace

int main(int argc, char *argv[]) {
    if (argc != 1 && argc != 3) {
        std::cerr << "Usage: decompress <compressed_file> <output_file>\n"
                     "       decompress          (stdin -> stdout)\n";
        return EXIT_FAILURE;
    }

    // --- Open input ---
    std::ifstream file_in;
    if (argc == 3) {
        file_in.open(argv[1], std::ios::binary);
        if (!file_in) {
            std::cerr << "Error: cannot open input file: " << argv[1] << "\n";
            return EXIT_FAILURE;
        }
    }
    std::istream &in = (argc == 3) ? static_cast<std::istream &>(file_in) : std::cin;

    // --- Read and validate header ---
    char magic[4];
    in.read(magic, 4);
    if (!in || magic[0] != 'T' || magic[1] != 'A' || magic[2] != '1' || magic[3] != '0') {
        std::cerr << "Error: not a TA10 file.\n";
        return EXIT_FAILURE;
    }

    std::uint64_t original_size = 0;
    std::uint32_t block_size   = 0;
    std::uint32_t block_count  = 0;
    if (!readUint64LE(in, original_size) ||
        !readUint32LE(in, block_size)    ||
        !readUint32LE(in, block_count)) {
        std::cerr << "Error: truncated header.\n";
        return EXIT_FAILURE;
    }
    if (original_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        std::cerr << "Error: file too large for this platform.\n";
        return EXIT_FAILURE;
    }
    if (block_size == 0) {
        std::cerr << "Error: invalid block size in header.\n";
        return EXIT_FAILURE;
    }

    const std::uint64_t expected_blocks =
        original_size == 0 ? 0u : (original_size + block_size - 1) / block_size;
    if (block_count != expected_blocks) {
        std::cerr << "Error: inconsistent block count in header.\n";
        return EXIT_FAILURE;
    }

    static constexpr std::uint32_t kDirectFlag = 0x80000000u;

    std::vector<std::uint32_t> tags(block_count, 0);
    for (std::uint32_t i = 0; i < block_count; ++i) {
        if (!readUint32LE(in, tags[i])) {
            std::cerr << "Error: truncated block tags.\n";
            return EXIT_FAILURE;
        }
        // Validate primary index for BWT+MTF blocks (bit 31 = 0).
        if (!(tags[i] & kDirectFlag)) {
            const std::uint64_t block_len =
                std::min<std::uint64_t>(block_size, original_size - static_cast<std::uint64_t>(i) * block_size);
            if (block_len > 0 && tags[i] >= block_len) {
                std::cerr << "Error: invalid primary index.\n";
                return EXIT_FAILURE;
            }
        }
    }

    // --- Open output ---
    std::ofstream file_out;
    if (argc == 3) {
        file_out.open(argv[2], std::ios::binary);
        if (!file_out) {
            std::cerr << "Error: cannot open output file: " << argv[2] << "\n";
            return EXIT_FAILURE;
        }
    }
    std::ostream &out = (argc == 3) ? static_cast<std::ostream &>(file_out) : std::cout;

    // --- Range-decode and invert BWT+MTF per block ---
    try {
        RangeDecoder decoder(in);
        std::uint64_t remaining = original_size;

        for (std::uint32_t b = 0; b < block_count; ++b) {
            const std::size_t block_len =
                static_cast<std::size_t>(std::min<std::uint64_t>(block_size, remaining));

            // Decode bits into the BWT+MTF block using the same model as the compressor
            ContextMixModel model;
            std::vector<std::uint8_t> encoded_block(block_len, 0);
            for (std::size_t i = 0; i < block_len; ++i) {
                std::uint8_t byte = 0;
                for (int bit = 7; bit >= 0; --bit) {
                    const std::uint32_t prob1 = model.predict();
                    BinaryFrequencyTable freqs(prob1);
                    const std::uint32_t decoded = decoder.read(freqs);
                    byte = static_cast<std::uint8_t>((byte << 1) | decoded);
                    model.update(decoded);
                }
                encoded_block[i] = byte;
            }

            // Invert transforms to recover original data
            if (tags[b] & kDirectFlag) {
                // Direct CM mode: no transforms were applied, write decoded bytes as-is
                out.write(reinterpret_cast<const char *>(encoded_block.data()),
                          static_cast<std::streamsize>(encoded_block.size()));
            } else {
                const auto original = bwt_inverse(mtf_inverse(encoded_block), tags[b]);
                out.write(reinterpret_cast<const char *>(original.data()),
                          static_cast<std::streamsize>(original.size()));
            }
            remaining -= block_len;
        }
    } catch (const std::exception &e) {
        std::cerr << "Decoding error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
