#pragma once

#include "FastFrequencyTable.hpp"

// Read-only linear mixture of two FastFrequencyTable instances:
//   mixed_freq(s) = a.get(s) + b.get(s)
//   mixed_total   = a.getTotal() + b.getTotal()
//
// findSymbol uses O(8) joint Fenwick descent on both trees simultaneously,
// avoiding the O(8 * 16) cost of binary-searching on getLow calls.
//
// Design: b is a bounded class-level prior (total <= HI_PRIOR_CAP).
// a is the fine per-context adaptive model.  For dense contexts a dominates;
// for sparse fine contexts b provides the right warm-start distribution.
class MixedFrequencyTable final : public FrequencyTable {
    const FastFrequencyTable& a_;
    const FastFrequencyTable& b_;

public:
    MixedFrequencyTable(const FastFrequencyTable& a, const FastFrequencyTable& b)
        : a_(a), b_(b) {}

    uint32_t getSymbolLimit() const override { return 256; }
    uint32_t getTotal()       const override { return a_.getTotal() + b_.getTotal(); }

    uint32_t get (uint32_t s) const override { return a_.get(s)  + b_.get(s);  }
    uint32_t getLow (uint32_t s) const override { return a_.getLow(s)  + b_.getLow(s);  }
    uint32_t getHigh(uint32_t s) const override { return a_.getHigh(s) + b_.getHigh(s); }

    void set      (uint32_t, uint32_t) override {}
    void increment(uint32_t)           override {}

    // Joint Fenwick descent: O(8) — avoids binary search + getLow calls entirely.
    uint32_t findSymbol(uint32_t value) const override {
        uint32_t pos = 0;
        for (uint32_t bit = 128; bit > 0; bit >>= 1) {
            uint32_t nxt = pos + bit;
            if (nxt <= 256) {
                uint32_t combined = a_.tree_node(nxt) + b_.tree_node(nxt);
                if (combined <= value) {
                    pos = nxt;
                    value -= combined;
                }
            }
        }
        return pos;
    }
};
