#pragma once

#include <array>
#include <cstdint>

#include "common/FastFormat.hpp"

class FastByteModel final {
    static constexpr std::uint32_t kSymbolLimit = 256;

    struct SymbolRange {
        std::uint32_t symbol;
        std::uint32_t low;
        std::uint32_t high;
    };

    std::array<std::uint16_t, kSymbolLimit> frequencies_{};
    std::array<std::uint32_t, kSymbolLimit + 1> tree_{};
    std::uint32_t total_ = 0;

    void update(std::uint32_t index, std::uint32_t delta) {
        for (std::uint32_t i = index; i <= kSymbolLimit; i += i & -i)
            tree_[i] += delta;
    }

    std::uint32_t query(std::uint32_t index) const {
        std::uint32_t sum = 0;
        for (std::uint32_t i = index; i > 0; i -= i & -i)
            sum += tree_[i];
        return sum;
    }

    void rebuild() {
        tree_.fill(0);
        total_ = 0;
        for (std::uint32_t symbol = 0; symbol < kSymbolLimit; symbol++) {
            total_ += frequencies_[symbol];
            update(symbol + 1, frequencies_[symbol]);
        }
    }

    void rescale() {
        for (auto &freq : frequencies_) {
            freq = static_cast<std::uint16_t>((freq + 1u) >> 1);
            if (freq == 0)
                freq = 1;
        }
        rebuild();
    }

    SymbolRange findSymbolRange(std::uint32_t value) const {
        std::uint32_t pos = 0;
        std::uint32_t low_sum = 0;
        for (std::uint32_t bit = 256; bit > 0; bit >>= 1) {
            std::uint32_t next = pos + bit;
            if (next <= kSymbolLimit && low_sum + tree_[next] <= value) {
                pos = next;
                low_sum += tree_[next];
            }
        }
        return SymbolRange{pos, low_sum, low_sum + frequencies_[pos]};
    }

public:
    FastByteModel() {
        reset();
    }

    void reset() {
        frequencies_.fill(1);
        rebuild();
    }

    template <typename Encoder>
    void encodeSymbol(Encoder &enc, std::uint8_t symbol) {
        if (total_ >= fast::kRescaleThreshold)
            rescale();

        std::uint32_t low = query(symbol);
        std::uint32_t high = low + frequencies_[symbol];
        enc.write(low, high, total_);

        frequencies_[symbol]++;
        total_++;
        update(symbol + 1, 1);
    }

    template <typename Decoder>
    std::uint8_t decodeSymbol(Decoder &dec) {
        if (total_ >= fast::kRescaleThreshold)
            rescale();

        std::uint32_t value = dec.getTarget(total_);
        SymbolRange range = findSymbolRange(value);
        dec.consume(range.low, range.high, total_);

        frequencies_[range.symbol]++;
        total_++;
        update(range.symbol + 1, 1);
        return static_cast<std::uint8_t>(range.symbol);
    }
};
