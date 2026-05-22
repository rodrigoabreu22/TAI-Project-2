#pragma once
#include <algorithm>
#include <cstdint>
#include "FrequencyTable.hpp"

// 256-symbol frequency table backed by a Fenwick tree (Binary Indexed Tree).
//
// SimpleFrequencyTable clears and rebuilds its 257-element cumulative array
// (O(256) ops) on every increment/set call.  With ~4 model updates per pixel
// at 2.25 M pixels, that is ~2.3 B ops just in cumulative maintenance.
//
// This implementation costs O(log 256) = O(8) for every update, prefix query,
// and symbol lookup — roughly 16–32x fewer operations.
//
// findSymbol uses Fenwick descent (not binary search), so it is also O(8).
// A tree_node(j) accessor lets MixedFrequencyTable do joint descent on two
// tables in O(8) instead of O(8 * 16).
//
// Starts with flat distribution (all frequencies = 1, total = 256).
class FastFrequencyTable final : public FrequencyTable {
    static constexpr uint32_t N = 256;

    uint32_t freq_[N];     // raw per-symbol frequencies
    uint32_t tree_[N + 1]; // 1-indexed Fenwick tree (tree_[0] unused)
    uint32_t total_;

    static constexpr uint32_t lowbit(uint32_t j) { return j & (uint32_t)(-(int32_t)j); }

public:
    FastFrequencyTable() : total_(N) {
        for (uint32_t i = 0; i < N; ++i)  freq_[i] = 1;
        tree_[0] = 0;
        // For flat distribution, tree_[j] = lowbit(j)
        for (uint32_t j = 1; j <= N; ++j) tree_[j] = lowbit(j);
    }

    // Expose Fenwick node for joint descent in MixedFrequencyTable::findSymbol.
    uint32_t tree_node(uint32_t j) const { return tree_[j]; }

    uint32_t getSymbolLimit() const override { return N; }
    uint32_t getTotal()       const override { return total_; }
    uint32_t get(uint32_t s)  const override { return freq_[s]; }

    // prefix(s) = sum of freq[0..s-1]  (number of elements before symbol s)
    uint32_t getLow (uint32_t s) const override {
        uint32_t sum = 0;
        for (uint32_t j = s; j > 0; j -= lowbit(j)) sum += tree_[j];
        return sum;
    }
    uint32_t getHigh(uint32_t s) const override {
        uint32_t sum = 0;
        for (uint32_t j = s + 1; j > 0; j -= lowbit(j)) sum += tree_[j];
        return sum;
    }

    void increment(uint32_t s) override {
        freq_[s]++;
        total_++;
        for (uint32_t j = s + 1; j <= N; j += lowbit(j)) tree_[j]++;
    }

    // Used during periodic halving.  Signed delta handles both increase and decrease.
    void set(uint32_t s, uint32_t f) override {
        int32_t delta = (int32_t)f - (int32_t)freq_[s];
        freq_[s] = f;
        total_ = (uint32_t)((int32_t)total_ + delta);
        for (uint32_t j = s + 1; j <= N; j += lowbit(j))
            tree_[j] = (uint32_t)((int32_t)tree_[j] + delta);
    }

    // Halve all frequencies (min 1) and rebuild the Fenwick tree in O(N)
    // instead of N individual set() calls (each O(log N)).
    void halve() {
        total_ = 0;
        for (uint32_t i = 0; i < N; ++i) {
            freq_[i] = std::max(1u, freq_[i] >> 1);
            total_ += freq_[i];
        }
        tree_[0] = 0;
        for (uint32_t j = 1; j <= N; ++j) tree_[j] = freq_[j - 1];
        for (uint32_t j = 1; j <= N; ++j) {
            uint32_t parent = j + lowbit(j);
            if (parent <= N) tree_[parent] += tree_[j];
        }
    }

    // O(8) descent — no binary search, no getLow calls.
    uint32_t findSymbol(uint32_t value) const override {
        uint32_t pos = 0;
        for (uint32_t bit = 128; bit > 0; bit >>= 1) {
            uint32_t nxt = pos + bit;
            if (nxt <= N && tree_[nxt] <= value) {
                pos = nxt;
                value -= tree_[nxt];
            }
        }
        return pos;
    }
};
