#include "mlsb_mixed.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>

using namespace std;

bool g_use_shuffled = false;

// ---------------------------------------------------------------------------
// Helper: apply Optimal Pixel Adjustment (OPA) to a stego pixel
// ---------------------------------------------------------------------------
static void apply_opa_pixel(unsigned char& stego, unsigned char orig, int t)
{
    if (t < 8 && t > 0) {
        unsigned int flip_mask = 1u << t;
        unsigned char stego_flipped = static_cast<unsigned char>(stego ^ flip_mask);
        int err_before  = abs(static_cast<int>(stego)         - static_cast<int>(orig));
        int err_flipped = abs(static_cast<int>(stego_flipped) - static_cast<int>(orig));
        if (err_flipped < err_before) {
            stego = stego_flipped;
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: serialize share, r, and bs components into a bitstream.
// bs contributes exactly z bits (no byte-padding).
// ---------------------------------------------------------------------------
static vector<uint8_t> payload_to_bits(const vector<unsigned char>& share,
                                       unsigned char r,
                                       const vector<unsigned char>& bs,
                                       size_t z)
{
    vector<uint8_t> bits;
    bits.reserve(8 * (share.size() + 1) + z);

    // 1. Shamir share bytes
    for (unsigned char b : share) {
        for (int i = 0; i < 8; ++i) {
            bits.push_back((b >> i) & 1u);
        }
    }

    // 2. r byte
    for (int i = 0; i < 8; ++i) {
        bits.push_back((r >> i) & 1u);
    }

    // 3. bs: exactly z bits (LSB-first within each byte), no padding
    for (size_t bit = 0; bit < z; ++bit) {
        size_t byte_idx = bit / 8;
        int    bit_idx  = static_cast<int>(bit % 8);
        bits.push_back((bs[byte_idx] >> bit_idx) & 1u);
    }

    return bits;
}

// ---------------------------------------------------------------------------
// embed_share_mlsb_bs_mixed
// ---------------------------------------------------------------------------
bool embed_share_mlsb_bs_mixed(const TiffInfo& cover_info,
                               const vector<unsigned char>& share,
                               unsigned char r,
                               const vector<unsigned char>& bs,
                               size_t z,
                               TiffInfo& out_info)
{
    if (g_use_shuffled) {
        return embed_share_mlsb_bs_mixed_shuffled(cover_info, share, r, bs, z, out_info);
    }

    if (cover_info.size() < 6) {
        cerr << "Mixed embed: cover image too small even for header\n";
        return false;
    }

    size_t N_cover = cover_info.size();
    size_t P = N_cover - 6; // payload pixels

    // L = 8*(share + r) + z  -- bs contributes exactly z bits, no byte-padding
    size_t L = 8 * (share.size() + 1) + z;

    // Compute t* and split n_t* using bit-by-bit formulas
    size_t t_star = (L + P - 1) / P; // ceil(L / P)
    if (t_star > 8) {
        cerr << "Mixed embed: payload too large even for t=8. "
             << "L=" << L << " bits, P=" << P << " pixels.\n";
        return false;
    }

    long long diff = static_cast<long long>(L) - static_cast<long long>(P) * (t_star - 1);
    size_t n_t_star = static_cast<size_t>(max(0LL, diff));

    // Serialize payload into bit vector
    vector<uint8_t> payload_bits = payload_to_bits(share, r, bs, z);
    assert(payload_bits.size() == L);

    vector<unsigned char> modified = cover_info.data;

    // 1. Embed header: z (24-bit, little-endian) in LSB4 of first 6 pixels
    unsigned int mask_hdr = 15; // 4 bits
    for (int i = 0; i < 6; ++i) {
        unsigned char val = static_cast<unsigned char>((z >> (i * 4)) & mask_hdr);
        unsigned char orig = cover_info.data[i];
        unsigned char stego = static_cast<unsigned char>((orig & ~mask_hdr) | val);
        apply_opa_pixel(stego, orig, 4);
        modified[i] = stego;
    }

    // 2. Embed payload: pixels 6 onwards
    size_t pos = 6;
    size_t bit_idx = 0;
    for (size_t i = 0; i < P; ++i) {
        int current_t = (i < n_t_star) ? static_cast<int>(t_star) : static_cast<int>(t_star - 1);
        if (current_t > 0 && bit_idx < payload_bits.size()) {
            unsigned int mask_payload = (1u << current_t) - 1u;
            unsigned int val = 0;
            for (int b = 0; b < current_t; ++b) {
                if (bit_idx < payload_bits.size()) {
                    val |= (payload_bits[bit_idx++] << b);
                }
            }
            unsigned char orig = cover_info.data[pos];
            unsigned char stego = static_cast<unsigned char>((orig & ~mask_payload) | val);
            apply_opa_pixel(stego, orig, current_t);
            modified[pos] = stego;
        } else {
            // current_t == 0 or all bits embedded: leave unchanged
            modified[pos] = cover_info.data[pos];
        }
        ++pos;
    }

    out_info = cover_info;
    out_info.data = std::move(modified);
    return true;
}

// ---------------------------------------------------------------------------
// extract_share_mlsb_bs_mixed
// ---------------------------------------------------------------------------
bool extract_share_mlsb_bs_mixed(const TiffInfo& stego_info,
                                 size_t share_len,
                                 vector<unsigned char>& out_share,
                                 unsigned char& out_r,
                                 vector<unsigned char>& out_bs)
{
    if (g_use_shuffled) {
        return extract_share_mlsb_bs_mixed_shuffled(stego_info, share_len, out_share, out_r, out_bs);
    }

    if (stego_info.size() < 6) {
        cerr << "Mixed extract: stego image too small even for header\n";
        return false;
    }

    size_t N_cover = stego_info.size();
    size_t P = N_cover - 6;

    // 1. Read header: recover z (24-bit, little-endian) from LSB4 of first 6 pixels
    size_t z = 0;
    unsigned int mask_hdr = 15;
    for (int i = 0; i < 6; ++i) {
        unsigned char val = stego_info.data[i] & mask_hdr;
        z |= (static_cast<size_t>(val) << (i * 4));
    }

    // L = 8*(share + r) + z  -- bs contributes exactly z bits, no byte-padding
    size_t L = 8 * (share_len + 1) + z;

    // Compute t* and split n_t*
    size_t t_star = (L + P - 1) / P; // ceil(L / P)
    if (t_star > 8) {
        cerr << "Mixed extract: payload too large even for t=8. "
             << "L=" << L << " bits, P=" << P << " pixels.\n";
        return false;
    }

    long long diff = static_cast<long long>(L) - static_cast<long long>(P) * (t_star - 1);
    size_t n_t_star = static_cast<size_t>(max(0LL, diff));

    // 2. Extract payload bits: pixels 6 onwards
    vector<uint8_t> payload_bits(L, 0);
    size_t bit_idx = 0;
    size_t pos = 6;
    for (size_t i = 0; i < P; ++i) {
        int current_t = (i < n_t_star) ? static_cast<int>(t_star) : static_cast<int>(t_star - 1);
        if (current_t > 0 && bit_idx < L) {
            unsigned int mask_payload = (1u << current_t) - 1u;
            unsigned int val = stego_info.data[pos] & mask_payload;
            for (int b = 0; b < current_t; ++b) {
                if (bit_idx < payload_bits.size()) {
                    payload_bits[bit_idx++] = (val >> b) & 1u;
                }
            }
        }
        ++pos;
    }

    // 3. Deserialize: first (share_len+1)*8 bits -> share bytes + r byte
    size_t byte_bits = 8 * (share_len + 1);
    vector<unsigned char> share_and_r(share_len + 1, 0);
    for (size_t i = 0; i < share_len + 1; ++i) {
        unsigned char b = 0;
        for (int bit = 0; bit < 8; ++bit) {
            b |= (payload_bits[i * 8 + bit] << bit);
        }
        share_and_r[i] = b;
    }

    // 4. Remaining z bits -> bs (packed into ceil(z/8) bytes, LSB-first)
    size_t bs_bytes = (z + 7) / 8;
    vector<unsigned char> bs_out(bs_bytes, 0);
    for (size_t bit = 0; bit < z; ++bit) {
        if (payload_bits[byte_bits + bit])
            bs_out[bit / 8] |= static_cast<unsigned char>(1u << (bit % 8));
    }

    out_share.assign(share_and_r.begin(), share_and_r.begin() + share_len);
    out_r = share_and_r[share_len];
    out_bs = std::move(bs_out);

    return true;
}

// ---------------------------------------------------------------------------
// Helper: generate a pseudo-random permutation of length P using seed z
// ---------------------------------------------------------------------------
static std::vector<size_t> generate_permutation(size_t P, uint32_t seed)
{
    std::vector<size_t> perm(P);
    for (size_t i = 0; i < P; ++i) {
        perm[i] = i;
    }
    uint32_t state = seed;
    for (size_t i = P - 1; i > 0; --i) {
        // LCG next state
        state = state * 1664525u + 1013904223u;
        size_t j = state % (i + 1);
        std::swap(perm[i], perm[j]);
    }
    return perm;
}

// ---------------------------------------------------------------------------
// embed_share_mlsb_bs_mixed_shuffled
// ---------------------------------------------------------------------------
bool embed_share_mlsb_bs_mixed_shuffled(const TiffInfo& cover_info,
                                        const std::vector<unsigned char>& share,
                                        unsigned char r,
                                        const std::vector<unsigned char>& bs,
                                        size_t z,
                                        TiffInfo& out_info)
{
    if (cover_info.size() < 6) {
        cerr << "Mixed shuffled embed: cover image too small even for header\n";
        return false;
    }

    size_t N_cover = cover_info.size();
    size_t P = N_cover - 6; // payload pixels

    // L = 8*(share + r) + z  -- bs contributes exactly z bits, no byte-padding
    size_t L = 8 * (share.size() + 1) + z;

    // Compute t* and split n_t* using bit-by-bit formulas
    size_t t_star = (L + P - 1) / P; // ceil(L / P)
    if (t_star > 8) {
        cerr << "Mixed shuffled embed: payload too large even for t=8. "
             << "L=" << L << " bits, P=" << P << " pixels.\n";
        return false;
    }

    long long diff = static_cast<long long>(L) - static_cast<long long>(P) * (t_star - 1);
    size_t n_t_star = static_cast<size_t>(max(0LL, diff));

    // Serialize payload into bit vector
    vector<uint8_t> payload_bits = payload_to_bits(share, r, bs, z);
    assert(payload_bits.size() == L);

    vector<unsigned char> modified = cover_info.data;

    // 1. Embed header: z (24-bit, little-endian) in LSB4 of first 6 pixels
    unsigned int mask_hdr = 15; // 4 bits
    for (int i = 0; i < 6; ++i) {
        unsigned char val = static_cast<unsigned char>((z >> (i * 4)) & mask_hdr);
        unsigned char orig = cover_info.data[i];
        unsigned char stego = static_cast<unsigned char>((orig & ~mask_hdr) | val);
        apply_opa_pixel(stego, orig, 4);
        modified[i] = stego;
    }

    // Generate permutation seeded by z
    vector<size_t> perm = generate_permutation(P, static_cast<uint32_t>(z));

    // 2. Embed payload: pixels 6 onwards
    size_t pos = 6;
    size_t bit_idx = 0;
    for (size_t i = 0; i < P; ++i) {
        int current_t = (perm[i] < n_t_star) ? static_cast<int>(t_star) : static_cast<int>(t_star - 1);
        if (current_t > 0 && bit_idx < payload_bits.size()) {
            unsigned int mask_payload = (1u << current_t) - 1u;
            unsigned int val = 0;
            for (int b = 0; b < current_t; ++b) {
                if (bit_idx < payload_bits.size()) {
                    val |= (payload_bits[bit_idx++] << b);
                }
            }
            unsigned char orig = cover_info.data[pos];
            unsigned char stego = static_cast<unsigned char>((orig & ~mask_payload) | val);
            apply_opa_pixel(stego, orig, current_t);
            modified[pos] = stego;
        } else {
            // current_t == 0 or all bits embedded: leave unchanged
            modified[pos] = cover_info.data[pos];
        }
        ++pos;
    }

    out_info = cover_info;
    out_info.data = std::move(modified);
    return true;
}

// ---------------------------------------------------------------------------
// extract_share_mlsb_bs_mixed_shuffled
// ---------------------------------------------------------------------------
bool extract_share_mlsb_bs_mixed_shuffled(const TiffInfo& stego_info,
                                          size_t share_len,
                                          std::vector<unsigned char>& out_share,
                                          unsigned char& out_r,
                                          std::vector<unsigned char>& out_bs)
{
    if (stego_info.size() < 6) {
        cerr << "Mixed shuffled extract: stego image too small even for header\n";
        return false;
    }

    size_t N_cover = stego_info.size();
    size_t P = N_cover - 6;

    // 1. Read header: recover z (24-bit, little-endian) from LSB4 of first 6 pixels
    size_t z = 0;
    unsigned int mask_hdr = 15;
    for (int i = 0; i < 6; ++i) {
        unsigned char val = stego_info.data[i] & mask_hdr;
        z |= (static_cast<size_t>(val) << (i * 4));
    }

    // L = 8*(share + r) + z  -- bs contributes exactly z bits, no byte-padding
    size_t L = 8 * (share_len + 1) + z;

    // Compute t* and split n_t*
    size_t t_star = (L + P - 1) / P; // ceil(L / P)
    if (t_star > 8) {
        cerr << "Mixed shuffled extract: payload too large even for t=8. "
             << "L=" << L << " bits, P=" << P << " pixels.\n";
        return false;
    }

    long long diff = static_cast<long long>(L) - static_cast<long long>(P) * (t_star - 1);
    size_t n_t_star = static_cast<size_t>(max(0LL, diff));

    // Generate same permutation seeded by z
    vector<size_t> perm = generate_permutation(P, static_cast<uint32_t>(z));

    // 2. Extract payload bits: pixels 6 onwards
    vector<uint8_t> payload_bits(L, 0);
    size_t bit_idx = 0;
    size_t pos = 6;
    for (size_t i = 0; i < P; ++i) {
        int current_t = (perm[i] < n_t_star) ? static_cast<int>(t_star) : static_cast<int>(t_star - 1);
        if (current_t > 0 && bit_idx < L) {
            unsigned int mask_payload = (1u << current_t) - 1u;
            unsigned int val = stego_info.data[pos] & mask_payload;
            for (int b = 0; b < current_t; ++b) {
                if (bit_idx < payload_bits.size()) {
                    payload_bits[bit_idx++] = (val >> b) & 1u;
                }
            }
        }
        ++pos;
    }

    // 3. Deserialize: first (share_len+1)*8 bits -> share bytes + r byte
    size_t byte_bits = 8 * (share_len + 1);
    vector<unsigned char> share_and_r(share_len + 1, 0);
    for (size_t i = 0; i < share_len + 1; ++i) {
        unsigned char b = 0;
        for (int bit = 0; bit < 8; ++bit) {
            b |= (payload_bits[i * 8 + bit] << bit);
        }
        share_and_r[i] = b;
    }

    // 4. Remaining z bits -> bs (packed into ceil(z/8) bytes, LSB-first)
    size_t bs_bytes = (z + 7) / 8;
    vector<unsigned char> bs_out(bs_bytes, 0);
    for (size_t bit = 0; bit < z; ++bit) {
        if (payload_bits[byte_bits + bit])
            bs_out[bit / 8] |= static_cast<unsigned char>(1u << (bit % 8));
    }

    out_share.assign(share_and_r.begin(), share_and_r.begin() + share_len);
    out_r = share_and_r[share_len];
    out_bs = std::move(bs_out);

    return true;
}

