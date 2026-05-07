#pragma once

#include <cstdint>
#include <istream>
#include <ostream>

namespace fast {

inline constexpr char kMagic[4] = {'T', 'A', 'F', '1'};
inline constexpr std::uint32_t kBlockSize = 256u * 1024u;   // 256 KiB
inline constexpr std::uint32_t kRescaleThreshold = 16384u;

enum class BlockMode : std::uint8_t {
    Raw = 0,
    RangeCoded = 1,
};

inline bool isValidBlockSize(std::uint32_t value) {
    return value >= 4u * 1024u && value <= 64u * 1024u * 1024u;
}

inline void writeUint32(std::ostream &out, std::uint32_t value) {
    out.put(static_cast<char>((value >>  0) & 0xFF));
    out.put(static_cast<char>((value >>  8) & 0xFF));
    out.put(static_cast<char>((value >> 16) & 0xFF));
    out.put(static_cast<char>((value >> 24) & 0xFF));
}

inline bool readUint32(std::istream &in, std::uint32_t &value) {
    value = 0;
    for (int i = 0; i < 4; i++) {
        int b = in.get();
        if (b == std::char_traits<char>::eof())
            return false;
        value |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(b)) << (i * 8);
    }
    return true;
}

}  // namespace fast
