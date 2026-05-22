#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "coder/RangeCoder.hpp"
#include "common/FastFormat.hpp"
#include "common/OrderedParallelProcessor.hpp"
#include "model/FastByteModel.hpp"
#include "model/ImagePredictor.hpp"

namespace {

constexpr std::size_t kWorkerCount = 8;
constexpr std::size_t kMaxPendingChunks = 16;

struct DecompressionTask {
    fast::BlockMode mode;
    std::uint32_t width;
    std::uint32_t raw_size;
    std::string payload;
};

std::string processChunk(DecompressionTask task) {
    if (task.mode == fast::BlockMode::Raw) {
        if (task.payload.size() != task.raw_size)
            throw std::runtime_error("invalid raw chunk");
        return task.payload;
    }

    if (task.mode != fast::BlockMode::RangeCoded)
        throw std::runtime_error("invalid block mode");

    std::istringstream payload_in(task.payload, std::ios::binary | std::ios::in);
    RangeDecoder dec(payload_in);
    FastByteModel hi_model;
    FastByteModel lo_model;

    std::string raw(task.raw_size, '\0');
    const std::uint32_t row_bytes = task.width * 2u;
    const std::uint32_t rows = task.raw_size / row_bytes;

    for (std::uint32_t row = 0; row < rows; ++row) {
        char *row_ptr = raw.data() + static_cast<std::size_t>(row) * row_bytes;
        std::uint16_t prev = 0;
        for (std::uint32_t col = 0; col < task.width; ++col) {
            std::uint8_t hi = hi_model.decodeSymbol(dec);
            std::uint8_t lo = lo_model.decodeSymbol(dec);

            std::uint16_t z = static_cast<std::uint16_t>(
                (static_cast<std::uint16_t>(hi) << 8) | lo);
            std::uint16_t u = zigzag_decode(z);
            std::uint16_t px = static_cast<std::uint16_t>(prev + u);
            prev = px;

            row_ptr[2u * col] = static_cast<char>(px >> 8);
            row_ptr[2u * col + 1u] = static_cast<char>(px & 0xFF);
        }
    }

    return raw;
}

}  // namespace

int main(int argc, char *argv[]) {
    std::istream *in_ptr = &std::cin;
    std::ostream *out_ptr = &std::cout;
    std::ifstream fin;
    std::ofstream fout;

    if (argc >= 3) {
        fin.open(argv[1], std::ios::binary);
        if (!fin) {
            std::cerr << "Cannot open input: " << argv[1] << '\n';
            return 1;
        }
        fout.open(argv[2], std::ios::binary);
        if (!fout) {
            std::cerr << "Cannot open output: " << argv[2] << '\n';
            return 1;
        }
        in_ptr = &fin;
        out_ptr = &fout;
    } else if (argc == 1) {
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);
    } else {
        std::cerr << "Usage: decompress_astro_fast <input> <output>\n";
        return 1;
    }

    char magic[4];
    in_ptr->read(magic, 4);
    if (!*in_ptr || std::string(magic, 4) != std::string(fast::kMagic, 4)) {
        std::cerr << "Bad magic\n";
        return 1;
    }

    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t chunk_layout_hint = 0;
    if (!fast::readUint32(*in_ptr, width) ||
        !fast::readUint32(*in_ptr, height) ||
        !fast::readUint32(*in_ptr, chunk_layout_hint) ||
        width == 0 || height == 0 || chunk_layout_hint == 0) {
        std::cerr << "Invalid header\n";
        return 1;
    }

    try {
        OrderedParallelProcessor<DecompressionTask, std::string> processor(
            kWorkerCount, processChunk);

        std::size_t submitted = 0;
        std::size_t next_to_write = 0;
        std::uint64_t total_written = 0;
        const std::uint64_t expected_bytes =
            static_cast<std::uint64_t>(width) * height * 2u;

        while (true) {
            int mode_value = in_ptr->get();
            if (mode_value == std::char_traits<char>::eof())
                break;

            std::uint32_t raw_size = 0;
            std::uint32_t payload_size = 0;
            if (!fast::readUint32(*in_ptr, raw_size) ||
                !fast::readUint32(*in_ptr, payload_size) ||
                raw_size == 0 || (raw_size % (width * 2u)) != 0) {
                std::cerr << "Invalid chunk header\n";
                return 1;
            }

            std::string payload(payload_size, '\0');
            in_ptr->read(payload.data(), static_cast<std::streamsize>(payload_size));
            if (!*in_ptr) {
                std::cerr << "Truncated chunk payload\n";
                return 1;
            }

            while (submitted - next_to_write >= kMaxPendingChunks) {
                std::string ready = processor.takeNext(next_to_write);
                out_ptr->write(ready.data(), static_cast<std::streamsize>(ready.size()));
                total_written += ready.size();
                next_to_write++;
            }

            processor.submit(submitted, DecompressionTask{
                                            static_cast<fast::BlockMode>(
                                                static_cast<std::uint8_t>(mode_value)),
                                            width,
                                            raw_size,
                                            std::move(payload),
                                        });
            submitted++;

            std::string ready;
            while (processor.tryTakeNext(next_to_write, ready)) {
                out_ptr->write(ready.data(), static_cast<std::streamsize>(ready.size()));
                total_written += ready.size();
                next_to_write++;
            }
        }

        processor.closeInput();
        while (next_to_write < submitted) {
            std::string ready = processor.takeNext(next_to_write);
            out_ptr->write(ready.data(), static_cast<std::streamsize>(ready.size()));
            total_written += ready.size();
            next_to_write++;
        }

        if (total_written != expected_bytes) {
            std::cerr << "Decoded size mismatch\n";
            return 1;
        }
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }

    return *out_ptr ? 0 : 1;
}
