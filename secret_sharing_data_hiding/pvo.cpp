#include "pvo.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cstdint>
#include <cmath>

using namespace std;

// ---------------------------------------------------------------------------
// PVO pair expansion / contraction (shared by both variants)
//
// Given a sorted pair (s1 <= s2), embed one bit by expanding the difference:
//   l  = (s2 + s1) / 2   (integer average, floor)
//   h  = s2 - s1         (non-negative difference)
//   h' = 2*h + bit
//   new_s2 = l + ceil(h'/2)
//   new_s1 = l - floor(h'/2)
//
// Returns false if the expanded values would leave [0, 255].
// ---------------------------------------------------------------------------
static inline bool pvo_expand(unsigned char s1, unsigned char s2,
    unsigned char bit,
    unsigned char& out_s1, unsigned char& out_s2)
{
    int l  = (static_cast<int>(s2) + static_cast<int>(s1)) / 2;
    int h  = static_cast<int>(s2) - static_cast<int>(s1); // >= 0
    long long hprime = static_cast<long long>(2 * h) + bit;

    long long hf = hprime / 2;          // floor(h'/2)
    long long hc = (hprime + 1) / 2;    // ceil(h'/2)

    long long ns2 = static_cast<long long>(l) + hc;
    long long ns1 = static_cast<long long>(l) - hf;

    if (ns1 < 0 || ns1 > 255 || ns2 < 0 || ns2 > 255) return false;

    out_s2 = static_cast<unsigned char>(ns2);
    out_s1 = static_cast<unsigned char>(ns1);
    return true;
}

// ---------------------------------------------------------------------------
// PVO pair contraction (recovery)
//
// Given the stego sorted pair (s1' <= s2'), recover the embedded bit and
// the original sorted pair (s1, s2):
//   l' = (s2' + s1') / 2
//   h' = s2' - s1'
//   bit = h' & 1
//   h   = (h' - bit) / 2
//   s2  = l' + ceil(h/2)
//   s1  = l' - floor(h/2)
// ---------------------------------------------------------------------------
static inline void pvo_contract(unsigned char s1p, unsigned char s2p,
    unsigned char& out_bit,
    unsigned char& out_s1, unsigned char& out_s2)
{
    int lp = (static_cast<int>(s2p) + static_cast<int>(s1p)) / 2;
    int hp = static_cast<int>(s2p) - static_cast<int>(s1p); // >= 0

    out_bit = static_cast<unsigned char>(hp & 1u);
    long long h = (static_cast<long long>(hp) - out_bit) / 2;

    long long hf = h / 2;
    long long hc = (h + 1) / 2;

    long long s2 = static_cast<long long>(lp) + hc;
    long long s1 = static_cast<long long>(lp) - hf;

    out_s2 = static_cast<unsigned char>(max(0LL, min(255LL, s2)));
    out_s1 = static_cast<unsigned char>(max(0LL, min(255LL, s1)));
}

// ---------------------------------------------------------------------------
// Bit accessor: extract bit bit_idx from share (LSB-first within each byte)
// ---------------------------------------------------------------------------
static inline unsigned char get_share_bit(const vector<unsigned char>& share, size_t bit_idx)
{
    return static_cast<unsigned char>((share[bit_idx / 8] >> (bit_idx % 8)) & 1u);
}

// ---------------------------------------------------------------------------
// Core embed loop - shared by both sequential and sorted variants.
//
// pair_indices: sequence of pair indices to process (in embedding order).
// cover        : original cover pixel data.
// modified     : working buffer (modified in-place).
// used_map[rank]: 1 if pair_indices[rank] was successfully embedded.
// perm_map[rank]: 1 if the original pair had v0 > v1.
// ---------------------------------------------------------------------------
static bool embed_pairs(const vector<size_t>& pair_indices,
    const vector<unsigned char>& cover,
    const vector<unsigned char>& share,
    vector<unsigned char>& modified,
    vector<uint8_t>& used_map,
    vector<uint8_t>& perm_map)
{
    const size_t required_bits = share.size() * 8;
    const size_t n_pairs = pair_indices.size();

    used_map.assign(n_pairs, 0);
    perm_map.assign(n_pairs, 0);

    size_t embedded_bits = 0;

    for (size_t rank = 0; rank < n_pairs && embedded_bits < required_bits; ++rank) {
        size_t pi  = pair_indices[rank];
        size_t idx = pi * 2;

        unsigned char v0 = cover[idx];
        unsigned char v1 = cover[idx + 1];

        // Sort into (smaller, larger); record original order
        uint8_t perm = (v0 > v1) ? 1u : 0u;
        unsigned char s1 = perm ? v1 : v0; // smaller
        unsigned char s2 = perm ? v0 : v1; // larger

        unsigned char bit = get_share_bit(share, embedded_bits);
        unsigned char new_s1 = 0, new_s2 = 0;

        if (pvo_expand(s1, s2, bit, new_s1, new_s2)) {
            // Write back in original positional order
            modified[idx]     = perm ? new_s2 : new_s1;
            modified[idx + 1] = perm ? new_s1 : new_s2;
            used_map[rank] = 1;
            perm_map[rank] = perm;
            ++embedded_bits;
        } else {
            // Overflow - leave pair unchanged, mark as unused
            modified[idx]     = v0;
            modified[idx + 1] = v1;
            used_map[rank] = 0;
            perm_map[rank] = perm;
        }
    }

    if (embedded_bits != required_bits) {
        cerr << "PVO embed: embedded " << embedded_bits
             << " of " << required_bits << " bits\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Core recover loop - shared by both sequential and sorted variants.
// ---------------------------------------------------------------------------
static bool recover_pairs(const vector<size_t>& pair_indices,
    const vector<unsigned char>& stego,
    const vector<uint8_t>& used_map,
    const vector<uint8_t>& perm_map,
    size_t share_len,
    vector<unsigned char>& out_share,
    vector<unsigned char>& recovered)
{
    const size_t n_pairs = pair_indices.size();
    if (used_map.size() != n_pairs || perm_map.size() != n_pairs) {
        cerr << "PVO recover: map size mismatch\n";
        return false;
    }

    out_share.clear();
    out_share.reserve(share_len);

    const size_t needed_bits = share_len * 8;
    size_t extracted_bits = 0;
    unsigned char cur_byte = 0;
    int cur_bitpos = 0;

    for (size_t rank = 0; rank < n_pairs && extracted_bits < needed_bits; ++rank) {
        size_t pi  = pair_indices[rank];
        size_t idx = pi * 2;

        uint8_t perm = perm_map[rank];
        unsigned char rp = stego[idx];
        unsigned char rq = stego[idx + 1];

        // Map back to (smaller, larger) using the stored permutation
        unsigned char s1p = perm ? rq : rp; // smaller position
        unsigned char s2p = perm ? rp : rq; // larger position

        if (used_map[rank]) {
            unsigned char bit = 0, orig_s1 = 0, orig_s2 = 0;
            pvo_contract(s1p, s2p, bit, orig_s1, orig_s2);

            // Restore original positional order
            recovered[idx]     = perm ? orig_s2 : orig_s1;
            recovered[idx + 1] = perm ? orig_s1 : orig_s2;

            cur_byte |= static_cast<unsigned char>(bit << cur_bitpos);
            if (++cur_bitpos == 8) {
                out_share.push_back(cur_byte);
                cur_byte = 0;
                cur_bitpos = 0;
            }
            ++extracted_bits;
        } else {
            // Pair was not used - stego values are original values
            recovered[idx]     = rp;
            recovered[idx + 1] = rq;
        }
    }

    if (extracted_bits != needed_bits) {
        cerr << "PVO recover: extracted " << extracted_bits
             << " of " << needed_bits << " bits\n";
        return false;
    }
    return true;
}

// ===========================================================================
// Sequential PVO (original behaviour, renamed for clarity)
// ===========================================================================

bool embed_share_pvo_sequential(const TiffInfo& cover_info,
    const vector<unsigned char>& share,
    TiffInfo& out_info,
    vector<uint8_t>& used_map,
    vector<uint8_t>& perm_map)
{
    const size_t cover_pixels = cover_info.size();
    if (cover_pixels < 2) { cerr << "PVO-seq: cover too small\n"; return false; }

    const size_t n_pairs = cover_pixels / 2;

    // Sequential order: 0, 1, 2, ..., n_pairs-1
    vector<size_t> order(n_pairs);
    iota(order.begin(), order.end(), 0);

    vector<unsigned char> modified = cover_info.data;
    if (!embed_pairs(order, cover_info.data, share, modified, used_map, perm_map))
        return false;

    out_info = cover_info;
    out_info.data = std::move(modified);
    return true;
}

bool recover_share_pvo_sequential(const TiffInfo& stego_info,
    const vector<uint8_t>& used_map,
    const vector<uint8_t>& perm_map,
    size_t share_len,
    vector<unsigned char>& out_share,
    TiffInfo& recovered_cover)
{
    const size_t cover_pixels = stego_info.size();
    if (cover_pixels < 2) { cerr << "PVO-seq recover: stego too small\n"; return false; }

    const size_t n_pairs = cover_pixels / 2;
    vector<size_t> order(n_pairs);
    iota(order.begin(), order.end(), 0);

    vector<unsigned char> recovered = stego_info.data;
    if (!recover_pairs(order, stego_info.data, used_map, perm_map,
                       share_len, out_share, recovered))
        return false;

    recovered_cover = stego_info;
    recovered_cover.data = std::move(recovered);
    return true;
}

// ===========================================================================
// Sorting-based PVO
//
// Smoothness metric: absolute difference |v0 - v1| of the pair.
// Pairs with smaller difference are processed first - they have smaller h,
// so expansion is less likely to overflow and the distortion per embedded
// bit is lower.
//
// Tie-breaking by pair index keeps the sort stable and deterministic.
// ===========================================================================

// Build the sort order from a pixel buffer (cover or stego).
// Used by both embed and recover so the order is always consistent.
static vector<size_t> build_sort_order(const vector<unsigned char>& pixels,
    size_t n_pairs)
{
    vector<size_t> order(n_pairs);
    iota(order.begin(), order.end(), 0);

    sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        int da = abs(static_cast<int>(pixels[a * 2]) - static_cast<int>(pixels[a * 2 + 1]));
        int db = abs(static_cast<int>(pixels[b * 2]) - static_cast<int>(pixels[b * 2 + 1]));
        if (da != db) return da < db;
        return a < b; // stable tie-break
    });

    return order;
}

bool embed_share_pvo_sorted(const TiffInfo& cover_info,
    const vector<unsigned char>& share,
    TiffInfo& out_info,
    vector<size_t>& sort_order,
    vector<uint8_t>& used_map,
    vector<uint8_t>& perm_map)
{
    const size_t cover_pixels = cover_info.size();
    if (cover_pixels < 2) { cerr << "PVO-sorted: cover too small\n"; return false; }

    const size_t n_pairs = cover_pixels / 2;
    sort_order = build_sort_order(cover_info.data, n_pairs);

    vector<unsigned char> modified = cover_info.data;
    if (!embed_pairs(sort_order, cover_info.data, share, modified, used_map, perm_map))
        return false;

    out_info = cover_info;
    out_info.data = std::move(modified);
    return true;
}

bool recover_share_pvo_sorted(const TiffInfo& stego_info,
    const vector<size_t>& sort_order,
    const vector<uint8_t>& used_map,
    const vector<uint8_t>& perm_map,
    size_t share_len,
    vector<unsigned char>& out_share,
    TiffInfo& recovered_cover)
{
    const size_t cover_pixels = stego_info.size();
    if (cover_pixels < 2) { cerr << "PVO-sorted recover: stego too small\n"; return false; }

    vector<unsigned char> recovered = stego_info.data;
    if (!recover_pairs(sort_order, stego_info.data, used_map, perm_map,
                       share_len, out_share, recovered))
        return false;

    recovered_cover = stego_info;
    recovered_cover.data = std::move(recovered);
    return true;
}
