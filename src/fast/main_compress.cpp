#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "coder/RangeCoder.hpp"
#include "common/FastFormat.hpp"
#include "common/OrderedParallelProcessor.hpp"
#include "model/FastByteModel.hpp"
#include "model/ImagePredictor.hpp"

namespace {

constexpr std::size_t kWorkerCount = 8;
constexpr std::size_t kMaxPendingChunks = 16;

struct CompressionTask {
    std::uint32_t width;
    std::string raw_bytes;
};

struct EncodedChunk {
    fast::BlockMode mode;
    std::uint32_t raw_size;
    std::string payload;
};

std::uint16_t read_be_u16(const char *ptr) {
    return static_cast<std::uint16_t>((static_cast<std::uint16_t>(
                                          static_cast<std::uint8_t>(ptr[0])) << 8) |
                                      static_cast<std::uint8_t>(ptr[1]));
}

std::string compressChunk(const CompressionTask &task) {
    std::ostringstream payload(std::ios::binary | std::ios::out);
    RangeEncoder enc(payload);
    FastByteModel hi_model;
    FastByteModel lo_model;

    const std::uint32_t row_bytes = task.width * 2u;
    const std::uint32_t rows = static_cast<std::uint32_t>(task.raw_bytes.size()) / row_bytes;

    for (std::uint32_t row = 0; row < rows; ++row) {
        const char *row_ptr = task.raw_bytes.data() + static_cast<std::size_t>(row) * row_bytes;
        std::uint16_t prev = 0;
        for (std::uint32_t col = 0; col < task.width; ++col) {
            std::uint16_t px = read_be_u16(row_ptr + static_cast<std::size_t>(col) * 2u);
            std::uint16_t u = zigzag_encode(static_cast<std::uint16_t>(px - prev));
            prev = px;

            hi_model.encodeSymbol(enc, static_cast<std::uint8_t>(u >> 8));
            lo_model.encodeSymbol(enc, static_cast<std::uint8_t>(u & 0xFF));
        }
    }

    enc.finish();
    return payload.str();
}

EncodedChunk processChunk(CompressionTask task) {
    const std::uint32_t raw_size = static_cast<std::uint32_t>(task.raw_bytes.size());
    std::string compressed = compressChunk(task);

    if (compressed.size() >= task.raw_bytes.size()) {
        return EncodedChunk{fast::BlockMode::Raw, raw_size, std::move(task.raw_bytes)};
    }

    return EncodedChunk{fast::BlockMode::RangeCoded, raw_size, std::move(compressed)};
}

void inferImageShape(std::uint64_t npix, std::uint32_t &width, std::uint32_t &height) {
    std::uint32_t sq = static_cast<std::uint32_t>(std::sqrt(static_cast<double>(npix)));
    if (static_cast<std::uint64_t>(sq) * sq == npix) {
        width = sq;
        height = sq;
        return;
    }

    width = static_cast<std::uint32_t>(npix);
    height = 1;
    for (std::uint32_t w : {1500u, 2048u, 1024u, 512u, 256u}) {
        if (npix % w == 0) {
            width = w;
            height = static_cast<std::uint32_t>(npix / w);
            return;
        }
    }
}

void writeChunk(std::ostream &out, const EncodedChunk &chunk) {
    out.put(static_cast<char>(chunk.mode));
    fast::writeUint32(out, chunk.raw_size);
    fast::writeUint32(out, static_cast<std::uint32_t>(chunk.payload.size()));
    out.write(chunk.payload.data(), static_cast<std::streamsize>(chunk.payload.size()));
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
        std::cerr << "Usage: compress_astro_fast <input> <output>\n";
        return 1;
    }

    std::string raw(std::istreambuf_iterator<char>(*in_ptr), {});
    if (raw.size() % 2 != 0) {
        std::cerr << "Input size not even\n";
        return 1;
    }

    const std::uint64_t npix = raw.size() / 2u;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    inferImageShape(npix, width, height);

    out_ptr->write(fast::kMagic, 4);
    fast::writeUint32(*out_ptr, width);
    fast::writeUint32(*out_ptr, height);
    fast::writeUint32(*out_ptr, fast::kRowsPerChunk);

    try {
        OrderedParallelProcessor<CompressionTask, EncodedChunk> processor(
            kWorkerCount, processChunk);

        const std::size_t row_bytes = static_cast<std::size_t>(width) * 2u;
        std::size_t submitted = 0;
        std::size_t next_to_write = 0;

        for (std::uint32_t row = 0; row < height; row += fast::kRowsPerChunk) {
            std::uint32_t rows_this_chunk = std::min(fast::kRowsPerChunk, height - row);
            std::size_t chunk_size = static_cast<std::size_t>(rows_this_chunk) * row_bytes;
            std::size_t offset = static_cast<std::size_t>(row) * row_bytes;

            while (submitted - next_to_write >= kMaxPendingChunks) {
                EncodedChunk ready = processor.takeNext(next_to_write);
                writeChunk(*out_ptr, ready);
                next_to_write++;
            }

            processor.submit(submitted, CompressionTask{
                                            width,
                                            raw.substr(offset, chunk_size),
                                        });
            submitted++;

            EncodedChunk ready{};
            while (processor.tryTakeNext(next_to_write, ready)) {
                writeChunk(*out_ptr, ready);
                next_to_write++;
            }
        }

        processor.closeInput();
        while (next_to_write < submitted) {
            EncodedChunk ready = processor.takeNext(next_to_write);
            writeChunk(*out_ptr, ready);
            next_to_write++;
        }
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }

    return *out_ptr ? 0 : 1;
}
