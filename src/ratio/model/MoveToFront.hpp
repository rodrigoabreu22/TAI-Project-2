#pragma once

#include <cstdint>
#include <vector>

// Simple Move-To-Front transform for 8-bit symbols.
std::vector<std::uint8_t> mtf_forward(const std::vector<std::uint8_t> &input);
std::vector<std::uint8_t> mtf_inverse(const std::vector<std::uint8_t> &input);
