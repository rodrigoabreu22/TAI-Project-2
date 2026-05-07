/*
 * Byte-level range coder (LZMA-style carry propagation).
 *
 * RangeEncoder / RangeDecoder are templated on the frequency table type so
 * the compiler can inline and constant-fold all frequency lookups.
 * For BinaryFrequencyTable (kTotal=4096=2^12), the range_ division becomes
 * a right-shift, removing the only division from the hot encode/decode loop.
 *
 * Invariant: range_ always in [RC_TOP, 0xFFFFFFFF] after each renorm.
 * RC_TOP = 2^24.  Correct as long as FreqTable::getTotal() <= RC_TOP.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <istream>
#include <ostream>

static constexpr uint32_t RC_TOP = 1u << 24;


// ── Range Encoder ─────────────────────────────────────────────────────────────
class RangeEncoder {
    std::ostream& out_;
    uint64_t      low_;
    uint32_t      range_;
    uint8_t       cache_;
    uint32_t      pending_;

    void shift_low() {
        const bool    carry = (low_ >> 32) != 0;
        const uint8_t top   = static_cast<uint8_t>(static_cast<uint32_t>(low_) >> 24);
        if (top < 0xFFu || carry) {
            out_.put(static_cast<char>(cache_ + (carry ? 1u : 0u)));
            const uint8_t fill = carry ? 0x00u : 0xFFu;
            for (uint32_t i = 0; i < pending_; ++i)
                out_.put(static_cast<char>(fill));
            pending_ = 0;
            cache_   = top;
        } else {
            ++pending_;
        }
        low_ = static_cast<uint64_t>(static_cast<uint32_t>(low_) << 8);
    }

public:
    explicit RangeEncoder(std::ostream& o)
        : out_(o), low_(0), range_(0xFFFFFFFFu), cache_(0), pending_(0) {}

    template<typename FreqTable>
    void write(const FreqTable& freqs, uint32_t symbol) {
        const uint32_t total    = freqs.getTotal();
        const uint32_t sym_low  = freqs.getLow(symbol);
        const uint32_t sym_high = freqs.getHigh(symbol);
        const uint32_t r = range_ / total;
        low_   += static_cast<uint64_t>(sym_low) * r;
        range_  = (sym_high < total) ? (sym_high - sym_low) * r
                                     : range_ - sym_low * r;
        while (range_ < RC_TOP) { range_ <<= 8; shift_low(); }
    }

    void finish() {
        for (int i = 0; i < 5; ++i) shift_low();
    }
};


// ── Range Decoder ─────────────────────────────────────────────────────────────
class RangeDecoder {
    std::istream& in_;
    uint32_t      range_;
    uint32_t      code_;

    uint8_t read_byte() {
        const int b = in_.get();
        return b >= 0 ? static_cast<uint8_t>(b) : 0u;
    }

public:
    explicit RangeDecoder(std::istream& i) : in_(i), range_(0xFFFFFFFFu), code_(0) {
        for (int k = 0; k < 5; ++k)
            code_ = (code_ << 8) | read_byte();
    }

    template<typename FreqTable>
    uint32_t read(const FreqTable& freqs) {
        const uint32_t total = freqs.getTotal();
        const uint32_t r     = range_ / total;
        const uint32_t value = std::min(code_ / r, total - 1u);
        const uint32_t sym   = freqs.findSymbol(value);
        const uint32_t sym_low  = freqs.getLow(sym);
        const uint32_t sym_high = freqs.getHigh(sym);
        code_  -= sym_low * r;
        range_  = (sym_high < total) ? (sym_high - sym_low) * r
                                     : range_ - sym_low * r;
        while (range_ < RC_TOP) { code_ = (code_ << 8) | read_byte(); range_ <<= 8; }
        return sym;
    }
};
