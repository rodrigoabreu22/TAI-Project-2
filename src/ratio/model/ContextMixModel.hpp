#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

// Immutable binary frequency table with a fixed total of 4096.
// Constructed once per bit from the predicted probability of bit=1.
class BinaryFrequencyTable {
public:
    static constexpr std::uint32_t kTotal = 1u << 12;  // 4096

    explicit BinaryFrequencyTable(std::uint32_t prob1) noexcept {
        if (prob1 == 0) prob1 = 1;
        if (prob1 >= kTotal) prob1 = kTotal - 1;
        freq0_ = static_cast<std::uint16_t>(kTotal - prob1);
        freq1_ = static_cast<std::uint16_t>(prob1);
    }

    std::uint32_t getTotal()                        const noexcept { return kTotal; }
    std::uint32_t getLow (std::uint32_t sym)        const noexcept { return sym == 0 ? 0u : freq0_; }
    std::uint32_t getHigh(std::uint32_t sym)        const noexcept { return sym == 0 ? freq0_ : kTotal; }
    std::uint32_t findSymbol(std::uint32_t value)   const noexcept { return value < freq0_ ? 0u : 1u; }

private:
    std::uint16_t freq0_;
    std::uint16_t freq1_;
};


class ContextMixModel final {
public:
    ContextMixModel();

    std::uint32_t predict() const;
    void update(std::uint32_t bit);

private:
    struct Entry {
        std::uint32_t key    = 0;
        std::uint16_t count0 = 1;
        std::uint16_t count1 = 1;
    };

    class ContextMap {
    public:
        explicit ContextMap(std::size_t size_power_of_two);
        Entry *lookup(std::uint32_t key);
    private:
        std::vector<Entry> table_;
        std::size_t mask_;
    };

    static std::uint32_t hash(std::uint32_t x);
    static std::uint32_t estimate(const Entry &entry);
    void updateEntry(Entry &entry, std::uint32_t bit);
    std::uint32_t prefixKey() const;

    static constexpr std::size_t  kSseBuckets  = 32;
    static constexpr std::int32_t kWeightScale = 1 << 16;

    // Stationary model: per (bitPos, MSB-of-prev0) counters
    std::array<std::uint16_t, 16> bit0Counts_;
    std::array<std::uint16_t, 16> bit1Counts_;

    // SSE: calibrates the mixed probability per (bitPos, bucket)
    std::array<std::array<std::uint16_t, kSseBuckets>, 8> sse0Counts_;
    std::array<std::array<std::uint16_t, kSseBuckets>, 8> sse1Counts_;

    // Second SSE layer: conditioned on (prev0 high nibble, bitPos, bucket)
    std::array<std::array<std::array<std::uint16_t, kSseBuckets>, 8>, 16> sse2_0Counts_;
    std::array<std::array<std::array<std::uint16_t, kSseBuckets>, 8>, 16> sse2_1Counts_;

    // Context maps: ctx0 = 1-byte ctx, ctx1-ctx4 grow by 1 extra prev byte each
    ContextMap ctx0_;   // 4K entries
    ContextMap ctx1_;   // 64K
    ContextMap ctx2_;   // 256K
    ContextMap ctx3_;   // 1M
    ContextMap ctx4_;   // 4M
    ContextMap runCtx_; // 128K — conditioned on run-length + prev0

    // 7 models: stationary + ctx0..ctx4 + runCtx
    mutable std::array<Entry *, 6>          activeEntries_;
    mutable std::uint8_t                    lastSseBucket_;
    mutable std::uint8_t                    lastSse2Bucket_;
    mutable std::uint32_t                   lastMixedProb_;
    mutable std::array<std::int32_t,  7>    weights_;
    mutable std::array<std::uint32_t, 7>    lastProbs_;

    std::uint8_t  prev0_;
    std::uint8_t  prev1_;
    std::uint8_t  prev2_;
    std::uint8_t  prev3_;
    std::uint8_t  currentByte_;
    std::uint8_t  bitPos_;
    std::uint16_t runLength_;
};
