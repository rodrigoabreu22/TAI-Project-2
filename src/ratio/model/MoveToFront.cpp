#include "model/MoveToFront.hpp"

#include <array>
#include <cstddef>

std::vector<std::uint8_t> mtf_forward(const std::vector<std::uint8_t> &input) {
    std::vector<std::uint8_t> output;
    output.reserve(input.size());

    std::array<std::uint8_t, 256> table{};
    for (std::size_t i = 0; i < table.size(); ++i)
        table[i] = static_cast<std::uint8_t>(i);

    for (std::uint8_t sym : input) {
        std::uint8_t idx = 0;
        while (table[idx] != sym)
            ++idx;
        output.push_back(idx);

        // Move symbol to front.
        for (std::uint8_t j = idx; j > 0; --j)
            table[j] = table[j - 1];
        table[0] = sym;
    }

    return output;
}

std::vector<std::uint8_t> mtf_inverse(const std::vector<std::uint8_t> &input) {
    std::vector<std::uint8_t> output;
    output.reserve(input.size());

    std::array<std::uint8_t, 256> table{};
    for (std::size_t i = 0; i < table.size(); ++i)
        table[i] = static_cast<std::uint8_t>(i);

    for (std::uint8_t idx : input) {
        std::uint8_t sym = table[idx];
        output.push_back(sym);

        // Move symbol to front.
        for (std::uint8_t j = idx; j > 0; --j)
            table[j] = table[j - 1];
        table[0] = sym;
    }

    return output;
}
