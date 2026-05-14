#pragma once

#include "FrequencyTable.hpp"

// Read-only linear mixture of two frequency tables:
//   mixed_freq(s) = a.get(s) + b.get(s)
//   mixed_total   = a.getTotal() + b.getTotal()
//
// getLow/getHigh/findSymbol are derived from this without any extra memory.
// Increment and set are no-ops — update the constituent tables separately.
//
// Design note: b is a bounded "class-level prior" with total <= HI_PRIOR_CAP
// (kept small by periodic halving in the caller). a is the fine per-context
// adaptive model. For dense contexts (a.getTotal() >> b.getTotal()) the prior
// is negligible; for sparse fine contexts it provides the right distribution.
class MixedFrequencyTable final : public FrequencyTable {
    const SimpleFrequencyTable& a_;
    const SimpleFrequencyTable& b_;

public:
    MixedFrequencyTable(const SimpleFrequencyTable& a, const SimpleFrequencyTable& b)
        : a_(a), b_(b) {}

    uint32_t getSymbolLimit() const override { return a_.getSymbolLimit(); }
    uint32_t getTotal()       const override { return a_.getTotal() + b_.getTotal(); }

    uint32_t get (uint32_t s) const override { return a_.get(s)  + b_.get(s);  }
    uint32_t getLow (uint32_t s) const override { return a_.getLow(s)  + b_.getLow(s);  }
    uint32_t getHigh(uint32_t s) const override { return a_.getHigh(s) + b_.getHigh(s); }

    void set      (uint32_t, uint32_t) override {}
    void increment(uint32_t)           override {}

    uint32_t findSymbol(uint32_t value) const override {
        uint32_t lo = 0, hi = getSymbolLimit();
        while (hi - lo > 1) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (getLow(mid) > value) hi = mid;
            else                     lo = mid;
        }
        return lo;
    }
};
