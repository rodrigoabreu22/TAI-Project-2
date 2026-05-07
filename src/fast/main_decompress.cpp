#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/FastFormat.hpp"
#include "common/OrderedParallelProcessor.hpp"
#include "coder/RangeCoder.hpp"
#include "model/FastOrder0Model.hpp"

namespace {

constexpr std::size_t kWorkerCount = 8;
constexpr std::size_t kMaxPendingBlocks = 16;

struct DecompressionTask {
    fast::BlockMode mode;
    std::uint32_t raw_size;
    std::string payload;
};

std::string processBlock(DecompressionTask task) {
    if (task.mode == fast::BlockMode::Raw) {
        if (task.payload.size() != task.raw_size)
            throw std::runtime_error("invalid raw block.");
        return task.payload;
    }

    if (task.mode != fast::BlockMode::RangeCoded)
        throw std::runtime_error("invalid block mode.");

    std::istringstream payload_in(task.payload, std::ios::binary | std::ios::in);
    RangeDecoder decoder(payload_in);
    FastOrder0Model model;
    std::string block(task.raw_size, '\0');

    for (std::uint32_t i = 0; i < task.raw_size; i++)
        block[static_cast<std::size_t>(i)] = static_cast<char>(model.decodeSymbol(decoder));

    return block;
}

}  // namespace

int main(int argc, char *argv[]) {
    if (argc != 1 && argc != 3) {
        std::cerr << "Usage: decompress <compressed_file> <output_file>\n"
                     "       decompress          (stdin -> stdout)\n";
        return EXIT_FAILURE;
    }

    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::ifstream file_in;
    if (argc == 3) {
        file_in.open(argv[1], std::ios::binary);
        if (!file_in) {
            std::cerr << "Error: cannot open input file: " << argv[1] << "\n";
            return EXIT_FAILURE;
        }
    }
    std::istream &in = (argc == 3) ? static_cast<std::istream &>(file_in) : std::cin;

    std::ofstream file_out;
    if (argc == 3) {
        file_out.open(argv[2], std::ios::binary);
        if (!file_out) {
            std::cerr << "Error: cannot open output file: " << argv[2] << "\n";
            return EXIT_FAILURE;
        }
    }
    std::ostream &out = (argc == 3) ? static_cast<std::ostream &>(file_out) : std::cout;

    char magic[4];
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != std::string(fast::kMagic, 4)) {
        std::cerr << "Error: not a TAF1 compressed file.\n";
        return EXIT_FAILURE;
    }

    std::uint32_t block_size = 0;
    if (!fast::readUint32(in, block_size) || !fast::isValidBlockSize(block_size)) {
        std::cerr << "Error: invalid header.\n";
        return EXIT_FAILURE;
    }

    try {
        OrderedParallelProcessor<DecompressionTask, std::string> processor(
            kWorkerCount, processBlock);

        std::size_t submitted = 0;
        std::size_t next_to_write = 0;

        while (true) {
            int mode_value = in.get();
            if (mode_value == std::char_traits<char>::eof())
                break;

            std::uint32_t raw_size = 0;
            std::uint32_t payload_size = 0;
            if (!fast::readUint32(in, raw_size) || !fast::readUint32(in, payload_size)) {
                std::cerr << "Error: truncated block header.\n";
                return EXIT_FAILURE;
            }
            if (raw_size > block_size || payload_size > (block_size * 2u + 64u)) {
                std::cerr << "Error: invalid block sizes.\n";
                return EXIT_FAILURE;
            }

            std::string payload(payload_size, '\0');
            in.read(payload.data(), static_cast<std::streamsize>(payload_size));
            if (!in) {
                std::cerr << "Error: truncated block payload.\n";
                return EXIT_FAILURE;
            }

            while (submitted - next_to_write >= kMaxPendingBlocks) {
                std::string ready_block = processor.takeNext(next_to_write);
                out.write(ready_block.data(), static_cast<std::streamsize>(ready_block.size()));
                next_to_write++;
            }

            processor.submit(submitted, DecompressionTask{
                                            static_cast<fast::BlockMode>(
                                                static_cast<std::uint8_t>(mode_value)),
                                            raw_size,
                                            std::move(payload),
                                        });
            submitted++;

            std::string ready_block;
            while (processor.tryTakeNext(next_to_write, ready_block)) {
                out.write(ready_block.data(), static_cast<std::streamsize>(ready_block.size()));
                next_to_write++;
            }
        }

        processor.closeInput();
        while (next_to_write < submitted) {
            std::string ready_block = processor.takeNext(next_to_write);
            out.write(ready_block.data(), static_cast<std::streamsize>(ready_block.size()));
            next_to_write++;
        }
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return out ? EXIT_SUCCESS : EXIT_FAILURE;
}
