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

struct CompressionTask {
    std::string data;
};

struct EncodedBlock {
    fast::BlockMode mode;
    std::uint32_t raw_size;
    std::string payload;
};

static std::string compressBlock(const char *data, std::size_t size) {
    std::ostringstream payload(std::ios::binary | std::ios::out);
    RangeEncoder encoder(payload);
    FastOrder0Model model;

    for (std::size_t i = 0; i < size; i++)
        model.encodeSymbol(encoder, static_cast<std::uint8_t>(data[i]));

    encoder.finish();
    return payload.str();
}

EncodedBlock processBlock(CompressionTask task) {
    const std::uint32_t raw_size = static_cast<std::uint32_t>(task.data.size());
    std::string compressed = compressBlock(task.data.data(), task.data.size());

    if (compressed.size() >= task.data.size()) {
        return EncodedBlock{fast::BlockMode::Raw, raw_size, std::move(task.data)};
    }

    return EncodedBlock{fast::BlockMode::RangeCoded, raw_size, std::move(compressed)};
}

}  // namespace

int main(int argc, char *argv[]) {
    if (argc != 1 && argc != 3) {
        std::cerr << "Usage: compress <input_file> <output_file>\n"
                     "       compress          (stdin -> stdout)\n";
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

    out.write(fast::kMagic, 4);
    fast::writeUint32(out, fast::kBlockSize);

    try {
        OrderedParallelProcessor<CompressionTask, EncodedBlock> processor(
            kWorkerCount, processBlock);

        std::size_t submitted = 0;
        std::size_t next_to_write = 0;

        while (true) {
            std::string block(fast::kBlockSize, '\0');
            in.read(block.data(), static_cast<std::streamsize>(block.size()));
            std::streamsize got = in.gcount();
            if (got <= 0)
                break;

            while (submitted - next_to_write >= kMaxPendingBlocks) {
                EncodedBlock ready_block = processor.takeNext(next_to_write);
                out.put(static_cast<char>(ready_block.mode));
                fast::writeUint32(out, ready_block.raw_size);
                fast::writeUint32(out, static_cast<std::uint32_t>(ready_block.payload.size()));
                out.write(ready_block.payload.data(),
                          static_cast<std::streamsize>(ready_block.payload.size()));
                next_to_write++;
            }

            block.resize(static_cast<std::size_t>(got));
            processor.submit(submitted, CompressionTask{std::move(block)});
            submitted++;

            EncodedBlock ready_block{};
            while (processor.tryTakeNext(next_to_write, ready_block)) {
                out.put(static_cast<char>(ready_block.mode));
                fast::writeUint32(out, ready_block.raw_size);
                fast::writeUint32(out, static_cast<std::uint32_t>(ready_block.payload.size()));
                out.write(ready_block.payload.data(),
                          static_cast<std::streamsize>(ready_block.payload.size()));
                next_to_write++;
            }
        }

        processor.closeInput();
        while (next_to_write < submitted) {
            EncodedBlock ready_block = processor.takeNext(next_to_write);
            out.put(static_cast<char>(ready_block.mode));
            fast::writeUint32(out, ready_block.raw_size);
            fast::writeUint32(out, static_cast<std::uint32_t>(ready_block.payload.size()));
            out.write(ready_block.payload.data(),
                      static_cast<std::streamsize>(ready_block.payload.size()));
            next_to_write++;
        }
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return out ? EXIT_SUCCESS : EXIT_FAILURE;
}
