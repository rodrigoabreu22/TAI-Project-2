/*
 * TAI Project 1 — MTF+ORDER=0 Compressor (BWT whole-file + parallel MTF chunks)
 *
 * Pipeline:
 *   1. BWT (SA-IS) on the ENTIRE input.
 *   2. RLE of BWT output.
 *   3. Split runs into N chunks (N = hardware thread count).
 *   4. Each chunk: MTF on run symbols → adaptive ORDER=0 range-code → buffer.
 *   5. Write header + per-chunk data.
 *
 * MTF (Move-To-Front) converts the run-symbol stream to rank indices
 * heavily concentrated near 0 (BWT clusters identical symbols → same symbol
 * repeats → MTF rank is 0 almost always).  ORDER=0 codes this very cheaply.
 *
 * Compressed file format
 * ──────────────────────
 *  [Global header]
 *  Byte  0     uint8_t  mode  (0 = BWT+MTF, 1 = ORDER-0)
 *
 *  mode=0 (BWT+MTF):
 *  Bytes 1–4   uint32_t primary_index (LE) — for whole-file BWT inverse
 *  Bytes 5–8   uint32_t num_chunks (LE)
 *  [Per chunk, repeated num_chunks times]
 *  4 bytes  uint32_t bitstream_size (LE)
 *  1 byte   uint8_t  k_raw  (0 = full 256; 1–255 = actual k)
 *  k bytes  sorted alphabet (omitted if k_raw == 0)
 *  <bitstream_size bytes of range-coded data>
 *
 *  mode=1 (ORDER-0):
 *  Bytes 1–4   uint32_t original_size (LE)
 *  Bytes 5–8   uint32_t bitstream_size (LE)
 *  <bitstream_size bytes of ORDER-0 range-coded data>
 *
 * Usage:  compress <input_file> <output_file>
 *         compress          (stdin → stdout)
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

#include "coder/RangeCoder.hpp"
#include "coder/FrequencyTable.hpp"
#include "coder/FenwickFrequencyTable.hpp"
#include "model/BwtTransform.hpp"
#include "model/RleTransform.hpp"


// ── ORDER-0 adaptive range coding of raw bytes ───────────────────────────────
// Uses FenwickFrequencyTable for O(log 256) getLow/getHigh/findSymbol per symbol.
static std::vector<uint8_t> order0_encode(const std::vector<uint8_t>& data) {
    FenwickFrequencyTable model(256);
    for (int i = 0; i < 256; i++) model.increment(i);  
    std::ostringstream buf(std::ios::binary);
    RangeEncoder enc(buf);
    for (uint8_t b : data) {
        enc.write(model, b);
        model.increment(b);
    }
    enc.finish();
    std::string s = buf.str();
    return std::vector<uint8_t>(s.begin(), s.end());
}


// ── Per-chunk result ──────────────────────────────────────────────────────────
struct ChunkResult {
    std::vector<uint8_t> bitstream;
    uint8_t k_raw;
    std::vector<uint8_t> alphabet;
};


// ── Encode one chunk of runs via MTF + ORDER=0 (runs in a worker thread) ──────
static ChunkResult compress_chunk(
    std::vector<std::pair<uint8_t, uint32_t>> runs)
{
    // ── Build compact alphabet ────────────────────────────────────────────────
    bool seen[256] = {};
    for (auto& [sym, cnt] : runs) seen[sym] = true;

    std::vector<uint8_t> alphabet;
    std::array<uint32_t, 256> encode_map{};
    for (int i = 0; i < 256; i++) {
        if (seen[i]) {
            encode_map[i] = static_cast<uint32_t>(alphabet.size());
            alphabet.push_back(static_cast<uint8_t>(i));
        }
    }
    uint32_t k = static_cast<uint32_t>(alphabet.size());

    // ── MTF list initialised as [0, 1, ..., k-1] ─────────────────────────────
    std::vector<uint32_t> mtf_list(k);
    std::iota(mtf_list.begin(), mtf_list.end(), 0u);

    // ── Adaptive ORDER=0 model for MTF ranks: symbols 0..k-1 + k as EOF ──────
    FenwickFrequencyTable rank_model(k + 1);
    for (uint32_t i = 0; i <= k; i++) rank_model.increment(i);

    // ── Count models: direct adaptive model for small counts ─────────────────
    // Symbols 0..THRESH-1 represent counts 1..THRESH.
    // Symbol THRESH = "large" (count > THRESH) → followed by Elias-gamma.
    constexpr uint32_t COUNT_CTXS = 4;
    constexpr uint32_t THRESH = 8u;  
    auto rank_ctx = [](uint32_t r) -> uint32_t {
        if (r == 0) return 0;
        if (r == 1) return 1;
        if (r <= 3) return 2;
        return 3;
    };
    std::vector<uint32_t> cnt_init(THRESH + 1, 1u);
    std::vector<SimpleFrequencyTable> cnt_models(COUNT_CTXS, SimpleFrequencyTable(cnt_init));
    std::vector<uint32_t> exp_init(32, 1u);
    SimpleFrequencyTable  exp_model(exp_init);  // for large counts only
    std::vector<uint32_t> bit_init(2, 1u);
    SimpleFrequencyTable  bit_model(bit_init);

    std::ostringstream buf(std::ios::binary);
    RangeEncoder enc(buf);

    for (auto& [sym, cnt] : runs) {
        uint32_t s = encode_map[sym];

        // Find rank of s in MTF list (linear scan, k ≤ 256 → fast)
        uint32_t r = 0;
        while (mtf_list[r] != s) r++;

        // Encode rank
        enc.write(rank_model, r);
        rank_model.increment(r);

        // Move s to front: shift elements 0..r-1 right by one
        for (uint32_t i = r; i > 0; i--) mtf_list[i] = mtf_list[i - 1];
        mtf_list[0] = s;

        // Encode count via direct adaptive model (counts 1..THRESH) or Elias-gamma
        uint32_t ctx = rank_ctx(r);
        uint32_t sym_cnt = (cnt <= THRESH) ? (cnt - 1u) : THRESH;
        enc.write(cnt_models[ctx], sym_cnt);
        cnt_models[ctx].increment(sym_cnt);

        if (cnt > THRESH) {
            // Elias-gamma for overflow counts
            int b = 31 - __builtin_clz(cnt);
            enc.write(exp_model, static_cast<uint32_t>(b));
            exp_model.increment(static_cast<uint32_t>(b));
            uint32_t residual = cnt - (1u << b);
            for (int i = b - 1; i >= 0; i--)
                enc.write(bit_model, (residual >> i) & 1u);
        }
    }

    enc.write(rank_model, k);  // EOF marker
    enc.finish();

    ChunkResult result;
    std::string s = buf.str();
    result.bitstream.assign(s.begin(), s.end());
    result.k_raw = (k == 256u) ? 0u : static_cast<uint8_t>(k);
    result.alphabet = (k < 256u) ? alphabet : std::vector<uint8_t>{};
    return result;
}


// ── Write a uint32_t little-endian ───────────────────────────────────────────
static void write_u32le(std::ostream &out, uint32_t v) {
    out.put(static_cast<char>((v >>  0) & 0xFF));
    out.put(static_cast<char>((v >>  8) & 0xFF));
    out.put(static_cast<char>((v >> 16) & 0xFF));
    out.put(static_cast<char>((v >> 24) & 0xFF));
}


int main(int argc, char *argv[]) {
    if (argc != 1 && argc != 3) {
        std::cerr << "Usage: compress <input_file> <output_file>\n"
                     "       compress          (stdin -> stdout)\n";
        return EXIT_FAILURE;
    }

    // ── Open input ────────────────────────────────────────────────────────────
    std::ifstream file_in;
    if (argc >= 3) {
        file_in.open(argv[1], std::ios::binary);
        if (!file_in) {
            std::cerr << "Error: cannot open input file: " << argv[1] << "\n";
            return EXIT_FAILURE;
        }
    }
    std::istream &in = (argc >= 3) ? static_cast<std::istream &>(file_in) : std::cin;

    // ── Read entire input ─────────────────────────────────────────────────────
    std::vector<uint8_t> raw_data;
    if (argc >= 3) {
        file_in.seekg(0, std::ios::end);
        std::streamsize sz = file_in.tellg();
        file_in.seekg(0, std::ios::beg);
        if (sz > 0) {
            raw_data.resize(static_cast<size_t>(sz));
            file_in.read(reinterpret_cast<char *>(raw_data.data()), sz);
            raw_data.resize(static_cast<size_t>(file_in.gcount()));
        }
    } else {
        int b;
        while ((b = in.get()) != std::char_traits<char>::eof())
            raw_data.push_back(static_cast<uint8_t>(b));
    }

    uint32_t original_size = static_cast<uint32_t>(raw_data.size());

    // ── Single-pass: byte entropy + bigram MI pre-filter ─────────────────────
    // MI = 2*H(bytes) - H(bigrams): if MI < 0.45 bytes are near-independent
    // and BWT won't improve compression → use ORDER-0 directly.
    double H_bytes = 0.0;
    bool use_order0 = false;
    {
        uint64_t freq[256] = {};
        std::vector<uint32_t> bg(65536u, 0u);
        for (size_t i = 0; i < raw_data.size(); i++) {
            uint8_t b = raw_data[i];
            freq[b]++;
            if (i > 0)
                bg[(static_cast<uint32_t>(raw_data[i - 1]) << 8) | b]++;
        }
        double N = static_cast<double>(raw_data.size());
        for (int i = 0; i < 256; i++) {
            if (freq[i] > 0) {
                double p = freq[i] / N;
                H_bytes -= p * std::log2(p);
            }
        }
        if (H_bytes > 7.5) {
            use_order0 = true;
        } else if (raw_data.size() > 1) {
            double N2 = static_cast<double>(raw_data.size() - 1);
            double H_bigrams = 0.0;
            for (uint32_t c = 0; c < 65536u; c++) {
                if (bg[c] > 0) {
                    double p = static_cast<double>(bg[c]) / N2;
                    H_bigrams -= p * std::log2(p);
                }
            }
            double MI = 2.0 * H_bytes - H_bigrams;
            if (MI < 0.45)
                use_order0 = true;
        }
    }

    uint32_t primary_index = 0;
    uint32_t num_chunks = 0;
    std::vector<ChunkResult> results;
    std::vector<uint8_t> order0_bs;

    if (use_order0) {
        // Near-random file: encode directly with ORDER-0, skip BWT.
        order0_bs = order0_encode(raw_data);
        raw_data.clear();
        raw_data.shrink_to_fit();
    } else {
        // Launch ORDER-0 speculatively in the background while BWT runs.
        // Both only read raw_data; both finish before raw_data.clear() below.
        std::future<std::vector<uint8_t>> order0_future =
            std::async(std::launch::async, order0_encode, std::cref(raw_data));

        // ── BWT forward on the whole file ─────────────────────────────────────
        auto [bwt_data, pi] = bwt_forward(raw_data);
        primary_index = pi;

        // ── RLE of BWT output ─────────────────────────────────────────────────
        auto runs = rle_encode(bwt_data);
        bwt_data.clear();
        bwt_data.shrink_to_fit();

        // If avg run length is high, BWT clearly helps → skip ORDER-0 comparison.
        // Otherwise encode both and pick the smaller output.
        double avg_run = static_cast<double>(original_size) / static_cast<double>(runs.size());
        bool compare_order0 = (avg_run < 2.0);

        if (compare_order0)
            order0_bs = order0_future.get();
        else
            order0_future.get();  // wait for thread to finish before clearing raw_data
        raw_data.clear();
        raw_data.shrink_to_fit();

        // ── Split runs into chunks (one per hardware thread) ───────────────────
        uint32_t hw = std::max(1u, std::thread::hardware_concurrency());
        num_chunks = static_cast<uint32_t>(
            std::min(static_cast<size_t>(hw), runs.size()));
        if (num_chunks == 0) num_chunks = 1;

        size_t runs_per_chunk = (runs.size() + num_chunks - 1) / num_chunks;
        std::vector<std::vector<std::pair<uint8_t, uint32_t>>> chunks(num_chunks);
        for (uint32_t i = 0; i < num_chunks; i++) {
            size_t start = i * runs_per_chunk;
            size_t end = std::min(start + runs_per_chunk, runs.size());
            if (start >= runs.size()) { num_chunks = i; break; }
            chunks[i].assign(runs.begin() + static_cast<std::ptrdiff_t>(start),
                             runs.begin() + static_cast<std::ptrdiff_t>(end));
        }
        runs.clear();
        runs.shrink_to_fit();

        // ── Encode chunks in parallel ──────────────────────────────────────────
        std::vector<std::future<ChunkResult>> futures;
        futures.reserve(num_chunks);
        for (uint32_t i = 0; i < num_chunks; i++)
            futures.push_back(std::async(std::launch::async, compress_chunk, chunks[i]));
        results.reserve(num_chunks);
        for (auto &f : futures)
            results.push_back(f.get());

        // ── Compare sizes if we encoded both ──────────────────────────────────
        if (compare_order0) {
            size_t bwt_size = 1 + 4 + 4;  // mode + primary_index + num_chunks
            for (const auto &r : results)
                bwt_size += 4 + 1 + r.alphabet.size() + r.bitstream.size();
            size_t o0_size = 1 + 4 + 4 + order0_bs.size();  // mode + original_size + bitstream_size
            use_order0 = (o0_size < bwt_size);
        }
    }

    // ── Open output ───────────────────────────────────────────────────────────
    std::ofstream file_out;
    if (argc >= 3) {
        file_out.open(argv[2], std::ios::binary);
        if (!file_out) {
            std::cerr << "Error: cannot open output file: " << argv[2] << "\n";
            return EXIT_FAILURE;
        }
    }
    std::ostream &out = (argc >= 3) ? static_cast<std::ostream &>(file_out) : std::cout;

    if (use_order0) {
        // ── Write ORDER-0 output ──────────────────────────────────────────────
        // Header: mode=1 + original_size + bitstream_size
        out.put(1);  
        write_u32le(out, original_size);
        write_u32le(out, static_cast<uint32_t>(order0_bs.size()));
        out.write(reinterpret_cast<const char *>(order0_bs.data()),
                  static_cast<std::streamsize>(order0_bs.size()));
    } else {
        // ── Write BWT+MTF output ──────────────────────────────────────────────
        // Header: mode=0 + primary_index + num_chunks
        out.put(0);  
        write_u32le(out, primary_index);
        write_u32le(out, num_chunks);
        for (const auto &r : results) {
            write_u32le(out, static_cast<uint32_t>(r.bitstream.size()));
            out.put(static_cast<char>(r.k_raw));
            
            if (r.k_raw != 0)
                out.write(reinterpret_cast<const char *>(r.alphabet.data()),
                          static_cast<std::streamsize>(r.alphabet.size()));
            out.write(reinterpret_cast<const char *>(r.bitstream.data()),
                      static_cast<std::streamsize>(r.bitstream.size()));
        }
    }

    return EXIT_SUCCESS;
}
