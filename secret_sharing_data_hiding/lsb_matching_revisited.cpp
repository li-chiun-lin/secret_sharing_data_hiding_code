#include "lsb_matching_revisited.h"
#include <iostream>
#include <cstdint>
#include <vector>

using namespace std;

static inline uint64_t xorshift64(uint64_t& state)
{
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

// f(x1, x2) = (floor(x1/2) + x2) mod 2
static inline unsigned char lsbmr_f(unsigned char x1, unsigned char x2)
{
    return static_cast<unsigned char>(((x1 >> 1) + x2) & 1u);
}

bool embed_share_lsbmr(const TiffInfo& cover_info,
    const vector<unsigned char>& share,
    TiffInfo& out_info,
    uint64_t seed)
{
    const size_t cover_pixels = cover_info.size();
    const size_t total_bits   = share.size() * 8;

    // LSBMR embeds 2 bits per pair, so we need ceil(total_bits/2) pairs
    // = ceil(total_bits/2)*2 pixels.
    const size_t pairs_needed = (total_bits + 1) / 2;
    if (cover_pixels < pairs_needed * 2) {
        cerr << "LSBMR embed: cover too small ("
             << cover_pixels << " pixels, need " << pairs_needed * 2 << ")\n";
        return false;
    }

    vector<unsigned char> modified = cover_info.data;
    uint64_t rng = (seed == 0) ? 0xDEADBEEFCAFEBABEULL : seed;

    size_t bit_idx = 0;

    for (size_t pair = 0; pair < pairs_needed && bit_idx < total_bits; ++pair) {
        size_t idx1 = pair * 2;
        size_t idx2 = pair * 2 + 1;

        unsigned char x1 = modified[idx1];
        unsigned char x2 = modified[idx2];

        // Read the two bits to embed
        unsigned char b1 = static_cast<unsigned char>(
            (share[bit_idx / 8] >> (bit_idx % 8)) & 1u);
        ++bit_idx;

        unsigned char b2 = 0;
        if (bit_idx < total_bits) {
            b2 = static_cast<unsigned char>(
                (share[bit_idx / 8] >> (bit_idx % 8)) & 1u);
            ++bit_idx;
        }
        // If only one bit remains, b2 stays 0 and we only embed b1.

        // Step 1: embed b1 into LSB(x1)
        if ((x1 & 1u) != b1) {
            uint64_t r = xorshift64(rng);
            if ((r & 1u) == 0u)
                x1 = static_cast<unsigned char>(x1 < 255 ? x1 + 1 : x1 - 1);
            else
                x1 = static_cast<unsigned char>(x1 > 0   ? x1 - 1 : x1 + 1);
        }

        // Step 2: embed b2 into f(x1, x2) by adjusting x2 only
        if (lsbmr_f(x1, x2) != b2) {
            uint64_t r = xorshift64(rng);
            if ((r & 1u) == 0u)
                x2 = static_cast<unsigned char>(x2 < 255 ? x2 + 1 : x2 - 1);
            else
                x2 = static_cast<unsigned char>(x2 > 0   ? x2 - 1 : x2 + 1);
        }

        modified[idx1] = x1;
        modified[idx2] = x2;
    }

    out_info = cover_info;
    out_info.data = std::move(modified);
    return true;
}

bool extract_share_lsbmr(const TiffInfo& stego_info,
    size_t share_len,
    vector<unsigned char>& out_share)
{
    const size_t cover_pixels = stego_info.size();
    const size_t total_bits   = share_len * 8;
    const size_t pairs_needed = (total_bits + 1) / 2;

    if (cover_pixels < pairs_needed * 2) {
        cerr << "LSBMR extract: stego too small\n";
        return false;
    }

    out_share.assign(share_len, 0);
    size_t bit_idx = 0;

    for (size_t pair = 0; pair < pairs_needed && bit_idx < total_bits; ++pair) {
        unsigned char x1 = stego_info.data[pair * 2];
        unsigned char x2 = stego_info.data[pair * 2 + 1];

        // Extract b1 from LSB(x1)
        unsigned char b1 = x1 & 1u;
        out_share[bit_idx / 8] |= static_cast<unsigned char>(b1 << (bit_idx % 8));
        ++bit_idx;

        // Extract b2 from f(x1, x2)
        if (bit_idx < total_bits) {
            unsigned char b2 = lsbmr_f(x1, x2);
            out_share[bit_idx / 8] |= static_cast<unsigned char>(b2 << (bit_idx % 8));
            ++bit_idx;
        }
    }

    return true;
}
