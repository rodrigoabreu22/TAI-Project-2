#pragma once

/*
 * BWT Forward + Inverse Transform
 *
 * Forward: SA-IS algorithm (Nong, Zhang & Chan, 2009) — O(N) suffix array.
 *   Maps input bytes to 1-256, appends sentinel 0 (unique minimum), builds
 *   suffix array of the extended string, then extracts the BWT output and
 *   primary_index from the SA.
 *
 * Inverse: LF-mapping — O(N).
 */

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace detail {

// Fill bkt[] with bucket heads (end=false) or tails (end=true).
inline void getBuckets(const std::vector<int32_t>& s,
                       std::vector<int32_t>& bkt, int alpha, bool end) {
    std::fill(bkt.begin(), bkt.end(), 0);
    for (auto c : s) bkt[c]++;
    int32_t sum = 0;
    for (int i = 0; i < alpha; i++) {
        sum += bkt[i];
        bkt[i] = end ? sum - 1 : sum - bkt[i];
    }
}

// Induce L-type (left→right) then S-type (right→left) entries into SA.
inline void induceSA(const std::vector<int32_t>& s, std::vector<int32_t>& SA,
                     const std::vector<bool>& t, int alpha) {
    int n = static_cast<int>(s.size());
    std::vector<int32_t> bkt(alpha);

    // L-type: scan left→right; if SA[i]-1 is L-type, place it at bucket head
    getBuckets(s, bkt, alpha, false);
    for (int i = 0; i < n; i++) {
        if (SA[i] > 0 && !t[SA[i] - 1])
            SA[bkt[s[SA[i] - 1]]++] = SA[i] - 1;
    }

    // S-type: scan right→left; if SA[i]-1 is S-type, place it at bucket tail
    getBuckets(s, bkt, alpha, true);
    for (int i = n - 1; i >= 0; i--) {
        if (SA[i] > 0 && t[SA[i] - 1])
            SA[bkt[s[SA[i] - 1]]--] = SA[i] - 1;
    }
}

// SA-IS: build the suffix array of s[] in O(n) time.
//   s[n-1] must be 0 — the unique sentinel (smallest value, not elsewhere).
//   alpha = alphabet size = max(s)+1.
//   Result written into SA (resized to n).
inline void sais(const std::vector<int32_t>& s, std::vector<int32_t>& SA, int alpha) {
    int n = static_cast<int>(s.size());
    SA.assign(n, -1);
    if (n == 1) { SA[0] = 0; return; }

    // ── Step 1: classify S/L types ───────────────────────────────────────────
    // t[i] = true  → S-type (suffix[i] < suffix[i+1])
    // t[i] = false → L-type (suffix[i] > suffix[i+1])
    std::vector<bool> t(n, false);
    t[n - 1] = true;  // sentinel is always S-type (smallest)
    for (int i = n - 2; i >= 0; i--)
        t[i] = (s[i] < s[i + 1]) || (s[i] == s[i + 1] && t[i + 1]);

    // LMS (Left-Most S-type): S-type position whose left neighbour is L-type.
    auto isLMS = [&](int i) { return i > 0 && t[i] && !t[i - 1]; };

    // Collect LMS positions in input order (needed later to map SA1 → original)
    std::vector<int32_t> lmsPos;
    for (int i = 1; i < n; i++)
        if (isLMS(i)) lmsPos.push_back(i);
    int n1 = static_cast<int>(lmsPos.size());

    // ── Step 2: approximate SA via initial induced sort ──────────────────────
    {
        std::vector<int32_t> bkt(alpha);
        getBuckets(s, bkt, alpha, true);  // tails
        // Place LMS suffixes at the tail of their character buckets
        for (int i = n1 - 1; i >= 0; i--)
            SA[bkt[s[lmsPos[i]]]--] = lmsPos[i];
    }
    induceSA(s, SA, t, alpha);

    // ── Step 3: name LMS substrings ──────────────────────────────────────────
    // Two LMS substrings are equal iff they have the same sequence of
    // (character, type) pairs up to and including the next LMS position.
    std::vector<int32_t> name(n, -1);
    int32_t cur = -1, prev = -1;
    for (int i = 0; i < n; i++) {
        if (!isLMS(SA[i])) continue;
        bool diff = (prev == -1);
        if (!diff) {
            for (int d = 0; d < n; d++) {
                if (s[SA[i] + d] != s[prev + d] || t[SA[i] + d] != t[prev + d]) {
                    diff = true; break;
                }
                if (d > 0 && (isLMS(SA[i] + d) || isLMS(prev + d))) break;
            }
        }
        if (diff) { cur++; prev = SA[i]; }
        name[SA[i]] = cur;
    }

    // ── Step 4: build reduced problem s1 ─────────────────────────────────────
    // s1[i] = name of the LMS substring starting at lmsPos[i].
    // The last entry is always name[sentinel] = 0 — acts as s1's sentinel.
    std::vector<int32_t> s1;
    s1.reserve(n1);
    for (int i = 0; i < n; i++)
        if (name[i] >= 0) s1.push_back(name[i]);

    // ── Step 5: solve reduced problem ────────────────────────────────────────
    std::vector<int32_t> SA1(n1);
    int alpha1 = cur + 1;
    if (alpha1 < n1) {
        sais(s1, SA1, alpha1);  // recurse
    } else {
        // All LMS substrings are unique → SA1 is the inverse permutation of s1
        for (int i = 0; i < n1; i++) SA1[s1[i]] = i;
    }

    // ── Step 6: final induced sort using exact LMS order ─────────────────────
    // SA1[i] = j means the i-th LMS in sorted order is lmsPos[j]
    std::vector<int32_t> sortedLms(n1);
    for (int i = 0; i < n1; i++) sortedLms[i] = lmsPos[SA1[i]];

    SA.assign(n, -1);
    {
        std::vector<int32_t> bkt(alpha);
        getBuckets(s, bkt, alpha, true);  // tails
        for (int i = n1 - 1; i >= 0; i--)
            SA[bkt[s[sortedLms[i]]]--] = sortedLms[i];
    }
    induceSA(s, SA, t, alpha);
}

} // namespace detail


// ── Forward BWT ───────────────────────────────────────────────────────────────
// Uses SA-IS on the DOUBLED string T+T+'$' to obtain the cyclic rotation order.
//
// Why doubling: SA-IS builds a linear suffix array. For pairs of positions i,j
// where one suffix is shorter, the sentinel '$' resolves ties differently from
// the cyclic wrap-around. Doubling ensures the first N characters of each suffix
// at positions 0..N-1 fully reproduce the cyclic rotation, so the suffix order
// of positions 0..N-1 in T+T+'$' equals the cyclic rotation sort of T.
//
// Memory: 2*(N+1) ints for s, 2*(N+1) ints for SA → ~4*(2N+1) bytes overhead.
// For N=12.5 MB: ~100 MB, well within the 8 GB limit.
//
// Returns (bwt_output, primary_index).
inline std::pair<std::vector<uint8_t>, uint32_t>
bwt_forward(const std::vector<uint8_t>& data)
{
    size_t N = data.size();
    if (N == 0) return {{}, 0};
    if (N == 1) return {{data[0]}, 0};

    // Build T+T+'$': map bytes to 1-256, repeat twice, append sentinel 0.
    // Length = 2N+1.
    std::vector<int32_t> s(2 * N + 1);
    for (size_t i = 0; i < N; i++) {
        s[i]     = static_cast<int32_t>(data[i]) + 1;
        s[N + i] = s[i];
    }
    s[2 * N] = 0;  // unique sentinel

    // Build suffix array of length 2N+1
    std::vector<int32_t> SA;
    detail::sais(s, SA, 257);  // alphabet: 0 (sentinel) + 1-256

    // Extract BWT: scan SA, keep only positions 0..N-1 (cyclic rotations of T).
    // The character at each row is the cyclic predecessor: data[(SA[i]-1+N) % N].
    std::vector<uint8_t> L;
    L.reserve(N);
    uint32_t primary_index = 0;
    for (size_t i = 0; i <= 2 * N; i++) {
        if (SA[i] < 0 || static_cast<size_t>(SA[i]) >= N) continue;
        if (SA[i] == 0) {
            L.push_back(data[N - 1]);
            primary_index = static_cast<uint32_t>(L.size() - 1);
        } else {
            L.push_back(data[SA[i] - 1]);
        }
    }
    return {std::move(L), primary_index};
}


// ── Inverse BWT ───────────────────────────────────────────────────────────────
// LF-mapping reconstruction. O(N).
inline std::vector<uint8_t>
bwt_inverse(const std::vector<uint8_t>& L, uint32_t primary_index)
{
    size_t N = L.size();
    if (N == 0) return {};

    uint32_t cnt[256] = {};
    for (uint8_t b : L) cnt[b]++;

    uint32_t C[256] = {};
    for (int i = 1; i < 256; i++) C[i] = C[i - 1] + cnt[i - 1];

    std::vector<uint32_t> LF(N);
    uint32_t rank_cnt[256] = {};
    for (size_t i = 0; i < N; i++) {
        uint8_t c = L[i];
        LF[i] = C[c] + rank_cnt[c]++;
    }

    std::vector<uint8_t> original(N);
    uint32_t pos = primary_index;
    for (size_t j = N; j-- > 0;) {
        original[j] = L[pos];
        pos = LF[pos];
        // Prefetch the LF entry we will need two iterations from now.
        // The LF array is 4×N bytes (16 MB for a 4 MB block) — mostly in DRAM.
        // Without prefetch each access stalls ~100 cycles waiting for DRAM.
        __builtin_prefetch(&LF[LF[pos]], 0 /* read */, 0 /* no temporal locality */);
    }
    return original;
}
