#pragma once

#include <cstdint>
#include <utility>
#include <vector>

// Forward RLE: collapses runs of identical bytes into (symbol, count) pairs.
inline std::vector<std::pair<uint8_t, uint32_t>>
rle_encode(const std::vector<uint8_t>& data)
{
    std::vector<std::pair<uint8_t, uint32_t>> runs;
    if (data.empty()) return runs;

    uint32_t count = 1;
    for (size_t i = 1; i < data.size(); i++) {
        if (data[i] == data[i - 1]) {
            ++count;
        } else {
            runs.emplace_back(data[i - 1], count);
            count = 1;
        }
    }
    runs.emplace_back(data.back(), count);
    return runs;
}

// Inverse RLE: expands (symbol, count) pairs back to a byte sequence.
inline std::vector<uint8_t>
rle_decode(const std::vector<std::pair<uint8_t, uint32_t>>& runs)
{
    size_t total = 0;
    for (auto& [sym, cnt] : runs) total += cnt;

    std::vector<uint8_t> out;
    out.reserve(total);
    for (auto& [sym, cnt] : runs)
        for (uint32_t i = 0; i < cnt; i++)
            out.push_back(sym);
    return out;
}
