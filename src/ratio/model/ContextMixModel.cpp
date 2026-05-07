#include "model/ContextMixModel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------------
// ContextMap
// ---------------------------------------------------------------------------

ContextMixModel::ContextMap::ContextMap(std::size_t size_power_of_two)
    : table_(size_power_of_two),
      mask_(size_power_of_two - 1) {}

ContextMixModel::Entry *ContextMixModel::ContextMap::lookup(std::uint32_t key) {
    Entry &entry = table_[ContextMixModel::hash(key) & mask_];
    if (entry.key != key) {
        entry.key    = key;
        entry.count0 = 1;
        entry.count1 = 1;
    }
    return &entry;
}

// ---------------------------------------------------------------------------
// ContextMixModel
// ---------------------------------------------------------------------------

ContextMixModel::ContextMixModel()
    : bit0Counts_{},
      bit1Counts_{},
      sse0Counts_{},
      sse1Counts_{},
      sse2_0Counts_{},
      sse2_1Counts_{},
      ctx0_(1u << 12),
      ctx1_(1u << 16),
      ctx2_(1u << 18),
      ctx3_(1u << 20),
      ctx4_(1u << 22),
      runCtx_(1u << 17),
      activeEntries_{},
      lastSseBucket_(0),
      lastSse2Bucket_(0),
      lastMixedProb_(BinaryFrequencyTable::kTotal / 2),
      // Initial weights favour longer-context models; all adapt per block.
      weights_{1 * kWeightScale, 2 * kWeightScale, 5 * kWeightScale,
               8 * kWeightScale, 10 * kWeightScale, 8 * kWeightScale,
               6 * kWeightScale},
      lastProbs_{},
      prev0_(0), prev1_(0), prev2_(0), prev3_(0),
      currentByte_(0), bitPos_(0), runLength_(0) {
    bit0Counts_.fill(1);
    bit1Counts_.fill(1);
    for (auto &row : sse0Counts_) row.fill(1);
    for (auto &row : sse1Counts_) row.fill(1);
    for (auto &nibble : sse2_0Counts_)
        for (auto &row : nibble) row.fill(1);
    for (auto &nibble : sse2_1Counts_)
        for (auto &row : nibble) row.fill(1);
}

// ---------------------------------------------------------------------------
// Precomputed LUTs for stretch (log-odds) and squash (sigmoid).
//
// stretch() is called 7× per bit and squash() once per bit, totalling ~256M
// transcendental function calls per 4 MB block.  Precomputing them at startup
// replaces those calls with cache-hot table lookups.
//
// stretch LUT:  4096 floats (16 KB) — exact, same values as runtime log().
// squash  LUT:  4096 uint32_t (16 KB) — quantised over [-10, +10] logit range;
//               both compressor and decompressor use the same LUT so the
//               probabilities agree and the format is unchanged.
// ---------------------------------------------------------------------------

static constexpr int   kSquashLUTSize  = 4096;
static constexpr float kSquashLUTRange = 10.0f;  // logit range covered

// Initialised once at program start via static constructor.
static const struct Luts {
    float    stretch[BinaryFrequencyTable::kTotal];
    std::uint32_t squash[kSquashLUTSize];

    Luts() noexcept {
        constexpr float kT = static_cast<float>(BinaryFrequencyTable::kTotal);

        // stretch[p] = log(p / (kTotal - p)), clamped to avoid ±inf at edges
        for (std::uint32_t p = 0; p < BinaryFrequencyTable::kTotal; ++p) {
            float fp = std::max(static_cast<float>(p) / kT, 1e-6f);
            fp = std::min(fp, 1.0f - 1e-6f);
            stretch[p] = std::log(fp / (1.0f - fp));
        }

        // squash[i] = sigmoid(x) * kTotal, where x maps linearly over [-range, +range]
        constexpr float step = 2.0f * kSquashLUTRange / (kSquashLUTSize - 1);
        for (int i = 0; i < kSquashLUTSize; ++i) {
            const float x = -kSquashLUTRange + i * step;
            const float p = 1.0f / (1.0f + std::exp(-x));
            std::uint32_t r = static_cast<std::uint32_t>(p * kT);
            if (r == 0) r = 1;
            if (r >= BinaryFrequencyTable::kTotal) r = BinaryFrequencyTable::kTotal - 1;
            squash[i] = r;
        }
    }
} kLuts;

// Inline wrappers used in the hot path.
static inline float stretchFast(std::uint32_t p) {
    return kLuts.stretch[p];
}

static inline std::uint32_t squashFast(float x) {
    constexpr float scale = (kSquashLUTSize - 1) / (2.0f * kSquashLUTRange);
    int idx = static_cast<int>((x + kSquashLUTRange) * scale);
    if (idx < 0)               idx = 0;
    if (idx >= kSquashLUTSize) idx = kSquashLUTSize - 1;
    return kLuts.squash[idx];
}

// Finalised hash (avalanche) used to index context maps
std::uint32_t ContextMixModel::hash(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// Estimate P(bit=1) for a context map entry
std::uint32_t ContextMixModel::estimate(const Entry &entry) {
    return static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(entry.count1) * BinaryFrequencyTable::kTotal) /
        (entry.count0 + entry.count1));
}

// Increment the winning counter and halve both when total exceeds 1024
void ContextMixModel::updateEntry(Entry &entry, std::uint32_t bit) {
    if (bit == 0) entry.count0++;
    else          entry.count1++;
    if (entry.count0 + entry.count1 > 1024) {
        entry.count0 = static_cast<std::uint16_t>((entry.count0 + 1) >> 1);
        entry.count1 = static_cast<std::uint16_t>((entry.count1 + 1) >> 1);
    }
}

// Key encoding the current bit position within the current byte.
// Encodes as (1 << bitPos) | bits_decoded_so_far, giving unique keys per position.
std::uint32_t ContextMixModel::prefixKey() const {
    return (1u << bitPos_) | currentByte_;
}

// ---------------------------------------------------------------------------
// predict() — called before encoding/decoding each bit
// ---------------------------------------------------------------------------

std::uint32_t ContextMixModel::predict() const {
    const std::uint32_t prefix    = prefixKey();
    const std::uint32_t baseCtx   = (static_cast<std::uint32_t>(bitPos_) << 9) | prefix;
    const std::uint32_t runBucket = std::min<std::uint32_t>(runLength_, 63u);

    // Look up each context map — collision evicts the old entry (open addressing)
    activeEntries_[0] = const_cast<ContextMap &>(ctx0_).lookup(baseCtx);
    activeEntries_[1] = const_cast<ContextMap &>(ctx1_).lookup(
        (static_cast<std::uint32_t>(prev0_) << 12) ^ baseCtx ^ 0x13579bdu);
    activeEntries_[2] = const_cast<ContextMap &>(ctx2_).lookup(
        (static_cast<std::uint32_t>(prev1_) << 20) ^
        (static_cast<std::uint32_t>(prev0_) << 8)  ^ baseCtx ^ 0x2468aceu);
    activeEntries_[3] = const_cast<ContextMap &>(ctx3_).lookup(
        (static_cast<std::uint32_t>(prev2_) << 24) ^
        (static_cast<std::uint32_t>(prev1_) << 16) ^
        (static_cast<std::uint32_t>(prev0_) << 8)  ^ baseCtx ^ 0x9e3779b9u);
    activeEntries_[4] = const_cast<ContextMap &>(ctx4_).lookup(
        (static_cast<std::uint32_t>(prev3_) << 24) ^
        (static_cast<std::uint32_t>(prev2_) << 16) ^
        (static_cast<std::uint32_t>(prev1_) << 8)  ^
         static_cast<std::uint32_t>(prev0_)         ^ baseCtx ^ 0xb7e15163u);
    activeEntries_[5] = const_cast<ContextMap &>(runCtx_).lookup(
        (runBucket << 16) ^ (static_cast<std::uint32_t>(prev0_) << 8) ^ baseCtx ^ 0xa511e9b3u);

    // Stationary model: global bit statistics per (bitPos, MSB of prev0)
    const std::uint32_t stationaryIndex = bitPos_ * 2u + ((prev0_ >> 7) & 1u);
    lastProbs_[0] = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(bit1Counts_[stationaryIndex]) * BinaryFrequencyTable::kTotal) /
        (bit0Counts_[stationaryIndex] + bit1Counts_[stationaryIndex]));

    // Context map probabilities
    lastProbs_[1] = estimate(*activeEntries_[0]);
    lastProbs_[2] = estimate(*activeEntries_[1]);
    lastProbs_[3] = estimate(*activeEntries_[2]);
    lastProbs_[4] = estimate(*activeEntries_[3]);
    lastProbs_[5] = estimate(*activeEntries_[4]);
    lastProbs_[6] = estimate(*activeEntries_[5]);

    // Mix all 7 models in log-odds (logit) space with adaptive weights
    constexpr float kWS = static_cast<float>(kWeightScale);
    float logitMix = 0.0f;
    float weightSum = 0.0f;
    for (int i = 0; i < 7; ++i) {
        const float w = static_cast<float>(weights_[i]) / kWS;
        logitMix  += w * stretchFast(lastProbs_[i]);
        weightSum += w;
    }
    const std::uint32_t mixedProb = squashFast(logitMix / weightSum);

    // SSE layer 1: calibrate mixedProb using per-(bitPos, bucket) statistics
    const std::uint32_t bucket = std::min<std::uint32_t>(
        (mixedProb * kSseBuckets) / BinaryFrequencyTable::kTotal,
        static_cast<std::uint32_t>(kSseBuckets - 1));
    const std::uint32_t sseProb = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(sse1Counts_[bitPos_][bucket]) * BinaryFrequencyTable::kTotal) /
        (sse0Counts_[bitPos_][bucket] + sse1Counts_[bitPos_][bucket]));

    lastSseBucket_  = static_cast<std::uint8_t>(bucket);
    lastMixedProb_  = mixedProb;

    // SSE layer 2: further calibrate using high nibble of prev0
    const std::uint32_t nibble   = (prev0_ >> 4) & 0xFu;
    const std::uint32_t sse2Prob = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(sse2_1Counts_[nibble][bitPos_][bucket]) * BinaryFrequencyTable::kTotal) /
        (sse2_0Counts_[nibble][bitPos_][bucket] + sse2_1Counts_[nibble][bitPos_][bucket]));
    lastSse2Bucket_ = static_cast<std::uint8_t>(bucket);

    // Blend SSE layers (2:1 weighting towards SSE1)
    return (2u * sseProb + sse2Prob) / 3u;
}

// ---------------------------------------------------------------------------
// update() — called after encoding/decoding each bit with the actual bit value
// ---------------------------------------------------------------------------

void ContextMixModel::update(std::uint32_t bit) {
    // Update stationary model
    const std::uint32_t stationaryIndex = bitPos_ * 2u + ((prev0_ >> 7) & 1u);
    if (bit == 0) bit0Counts_[stationaryIndex]++;
    else          bit1Counts_[stationaryIndex]++;
    if (bit0Counts_[stationaryIndex] + bit1Counts_[stationaryIndex] > 2048) {
        bit0Counts_[stationaryIndex] = static_cast<std::uint16_t>((bit0Counts_[stationaryIndex] + 1) >> 1);
        bit1Counts_[stationaryIndex] = static_cast<std::uint16_t>((bit1Counts_[stationaryIndex] + 1) >> 1);
    }

    // Update SSE layer 1
    if (bit == 0) sse0Counts_[bitPos_][lastSseBucket_]++;
    else          sse1Counts_[bitPos_][lastSseBucket_]++;
    if (sse0Counts_[bitPos_][lastSseBucket_] + sse1Counts_[bitPos_][lastSseBucket_] > 1024) {
        sse0Counts_[bitPos_][lastSseBucket_] = static_cast<std::uint16_t>((sse0Counts_[bitPos_][lastSseBucket_] + 1) >> 1);
        sse1Counts_[bitPos_][lastSseBucket_] = static_cast<std::uint16_t>((sse1Counts_[bitPos_][lastSseBucket_] + 1) >> 1);
    }

    // Update SSE layer 2
    const std::uint32_t nibble = (prev0_ >> 4) & 0xFu;
    if (bit == 0) sse2_0Counts_[nibble][bitPos_][lastSse2Bucket_]++;
    else          sse2_1Counts_[nibble][bitPos_][lastSse2Bucket_]++;
    if (sse2_0Counts_[nibble][bitPos_][lastSse2Bucket_] + sse2_1Counts_[nibble][bitPos_][lastSse2Bucket_] > 1024) {
        sse2_0Counts_[nibble][bitPos_][lastSse2Bucket_] = static_cast<std::uint16_t>((sse2_0Counts_[nibble][bitPos_][lastSse2Bucket_] + 1) >> 1);
        sse2_1Counts_[nibble][bitPos_][lastSse2Bucket_] = static_cast<std::uint16_t>((sse2_1Counts_[nibble][bitPos_][lastSse2Bucket_] + 1) >> 1);
    }

    // Update context map entries
    for (Entry *entry : activeEntries_)
        updateEntry(*entry, bit);

    // Update adaptive weights: increase weight of models that predicted well
    constexpr std::int32_t kLR = kWeightScale / 128;
    const std::int32_t kT = static_cast<std::int32_t>(BinaryFrequencyTable::kTotal);
    for (int i = 0; i < 7; ++i) {
        const std::int32_t pred  = static_cast<std::int32_t>(lastProbs_[i]);
        const std::int32_t error = (bit == 1) ? pred : (kT - pred);
        weights_[i] += kLR * (error - kT / 2) / (kT / 2);
        weights_[i] = std::max(weights_[i], kWeightScale / 16);
        weights_[i] = std::min(weights_[i], kWeightScale * 16);
    }

    // Advance byte state
    currentByte_ = static_cast<std::uint8_t>((currentByte_ << 1) | (bit & 1u));
    bitPos_++;
    if (bitPos_ == 8) {
        runLength_ = (currentByte_ == prev0_) ? std::min<std::uint16_t>(runLength_ + 1, 65535) : 0;
        prev3_ = prev2_;
        prev2_ = prev1_;
        prev1_ = prev0_;
        prev0_ = currentByte_;
        currentByte_ = 0;
        bitPos_      = 0;
    }
}
