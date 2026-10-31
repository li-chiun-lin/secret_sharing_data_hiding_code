#include "lsb_matching.h"
#include <iostream>
#include <cstdint>
#include <vector>

using namespace std;

// Simple xorshift64 PRNG for the random +-1 decision.
static inline uint64_t xorshift64(uint64_t& state)
{
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

bool embed_share_lsb_matching(const TiffInfo& cover_info,
    const vector<unsigned char>& share,
    TiffInfo& out_info,
    uint64_t seed)
{
    const size_t cover_pixels = cover_info.size();
    const size_t required_bits = share.size() * 8;

    if (cover_pixels < required_bits) {
        cerr << "LSB-Matching embed: cover too small ("
             << cover_pixels << " pixels, need " << required_bits << ")\n";
        return false;
    }

    vector<unsigned char> modified = cover_info.data;

    // Seed the PRNG; use a non-zero state.
    uint64_t rng = (seed == 0) ? 0xDEADBEEFCAFEBABEULL : seed;

    for (size_t bit_idx = 0; bit_idx < required_bits; ++bit_idx) {
        unsigned char desired_bit =
            static_cast<unsigned char>((share[bit_idx / 8] >> (bit_idx % 8)) & 1u);
        unsigned char pixel = modified[bit_idx];
        unsigned char current_lsb = pixel & 1u;

        if (current_lsb == desired_bit) {
            // Already matches - leave unchanged.
            continue;
        }

        // Need to flip LSB: randomly choose +1 or -1.
        uint64_t r = xorshift64(rng);
        if ((r & 1u) == 0u) {
            // +1, clamped to 255
            modified[bit_idx] = static_cast<unsigned char>(pixel < 255 ? pixel + 1 : pixel - 1);
        } else {
            // -1, clamped to 0
            modified[bit_idx] = static_cast<unsigned char>(pixel > 0 ? pixel - 1 : pixel + 1);
        }
    }

    out_info = cover_info;
    out_info.data = std::move(modified);
    return true;
}

bool extract_share_lsb_matching(const TiffInfo& stego_info,
    size_t share_len,
    vector<unsigned char>& out_share)
{
    const size_t cover_pixels = stego_info.size();
    const size_t required_bits = share_len * 8;

    if (cover_pixels < required_bits) {
        cerr << "LSB-Matching extract: stego too small\n";
        return false;
    }

    out_share.assign(share_len, 0);
    for (size_t bit_idx = 0; bit_idx < required_bits; ++bit_idx) {
        unsigned char bit = stego_info.data[bit_idx] & 1u;
        out_share[bit_idx / 8] |= static_cast<unsigned char>(bit << (bit_idx % 8));
    }
    return true;
}
