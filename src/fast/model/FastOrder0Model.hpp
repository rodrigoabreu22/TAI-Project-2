#pragma once

#include <array>
#include <cstdint>

#include "common/FastFormat.hpp"
#include "coder/RangeCoder.hpp"

class FastOrder0Model final {
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

public:
    FastOrder0Model() {
        reset();
    }

    void reset() {
        frequencies_.fill(1);
        rebuild();
    }

    std::uint32_t total() const {
        return total_;
    }

    std::uint32_t low(std::uint32_t symbol) const {
        return query(symbol);
    }

    std::uint32_t high(std::uint32_t symbol) const {
        return query(symbol + 1);
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

    void increment(std::uint32_t symbol) {
        if (total_ >= fast::kRescaleThreshold)
            rescale();
        frequencies_[symbol]++;
        total_++;
        update(symbol + 1, 1);
    }

    template <typename Encoder>
    void encodeSymbol(Encoder &enc, std::uint8_t symbol) {
        std::uint32_t lo = low(symbol);
        std::uint32_t hi = lo + frequencies_[symbol];
        enc.write(lo, hi, total_);
        increment(symbol);
    }

    template <typename Decoder>
    std::uint8_t decodeSymbol(Decoder &dec) {
        std::uint32_t value = dec.getTarget(total_);
        SymbolRange range = findSymbolRange(value);
        dec.consume(range.low, range.high, total_);
        increment(range.symbol);
        return static_cast<std::uint8_t>(range.symbol);
    }
};
