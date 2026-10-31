#include "mlsb_all.h"
#include "../finite_field/binary_field.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>

using namespace std;

// xorshift64 implementation for random coefficients
static inline uint64_t all_xorshift64(uint64_t& s)
{
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return s;
}

// OPA correction helper
static void apply_opa_pixel_all(unsigned char& stego, unsigned char orig, int t)
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

// LCG permutation generator seeded by z
static std::vector<size_t> generate_permutation_all(size_t P, uint32_t seed)
{
    std::vector<size_t> perm(P);
    for (size_t i = 0; i < P; ++i) {
        perm[i] = i;
    }
    uint32_t state = seed;
    for (size_t i = P - 1; i > 0; --i) {
        state = state * 1664525u + 1013904223u;
        size_t j = state % (i + 1);
        std::swap(perm[i], perm[j]);
    }
    return perm;
}

void share_z(size_t z, int k, int n, std::vector<std::vector<unsigned char>>& out_z_shares)
{
    out_z_shares.assign(n, vector<unsigned char>(3, 0));
    uint64_t rng = 0x5EED0F2B00C0FFEEULL; // unique seed for z sharing

    // z fits in 3 bytes
    unsigned char z_bytes[3];
    z_bytes[0] = z & 0xFF;
    z_bytes[1] = (z >> 8) & 0xFF;
    z_bytes[2] = (z >> 16) & 0xFF;

    for (int j = 0; j < 3; ++j) {
        vector<unsigned char> coef(k);
        coef[0] = z_bytes[j]; // constant term
        for (int c = 1; c < k; ++c) {
            coef[c] = static_cast<unsigned char>(all_xorshift64(rng) & 0xFF);
        }
        for (int x = 1; x <= n; ++x) {
            out_z_shares[x - 1][j] = poly_eval(coef, static_cast<unsigned char>(x));
        }
    }
}

size_t reconstruct_z(const std::vector<std::vector<unsigned char>>& z_shares, int k, const std::vector<int>& participant_ids)
{
    assert(static_cast<int>(z_shares.size()) >= k);
    assert(static_cast<int>(participant_ids.size()) >= k);

    vector<unsigned char> xs(k);
    for (int i = 0; i < k; ++i) {
        xs[i] = static_cast<unsigned char>(participant_ids[i]);
    }

    size_t z = 0;
    for (int j = 0; j < 3; ++j) {
        vector<unsigned char> ys(k);
        for (int i = 0; i < k; ++i) {
            ys[i] = z_shares[i][j];
        }
        vector<unsigned char> coef(k);
        vector<unsigned char> numer, tmp, factor;
        lagrange_interpolate(xs, ys, coef, numer, tmp, factor);
        z |= (static_cast<size_t>(coef[0]) << (j * 8));
    }
    return z;
}

bool embed_share_mlsb_all(const TiffInfo& cover_info,
                          const std::vector<unsigned char>& main_share,
                          const std::vector<unsigned char>& z_share,
                          size_t z,
                          TiffInfo& out_info)
{
    if (cover_info.size() < 6) {
        cerr << "mlsb_all embed: cover too small for header\n";
        return false;
    }
    assert(z_share.size() == 3);

    size_t N_cover = cover_info.size();
    size_t P = N_cover - 6;

    // L = 8 * M (main share size in bits)
    size_t L = 8 * main_share.size();

    // Compute t* and split n_t*
    size_t t_star = (L + P - 1) / P;
    if (t_star > 8) {
        cerr << "mlsb_all embed: payload too large even for t=8\n";
        return false;
    }

    long long diff = static_cast<long long>(L) - static_cast<long long>(P) * (t_star - 1);
    size_t n_t_star = static_cast<size_t>(max(0LL, diff));

    // Serialize main share to bits
    vector<uint8_t> payload_bits;
    payload_bits.reserve(L);
    for (unsigned char b : main_share) {
        for (int i = 0; i < 8; ++i) {
            payload_bits.push_back((b >> i) & 1u);
        }
    }
    assert(payload_bits.size() == L);

    vector<unsigned char> modified = cover_info.data;

    // 1. Embed header: z_share (3 bytes) into LSB4 of first 6 pixels
    for (int j = 0; j < 3; ++j) {
        unsigned char b = z_share[j];
        unsigned char b_low = b & 15;
        unsigned char b_high = (b >> 4) & 15;

        // Pixel 2*j (lower half-byte)
        unsigned char orig_low = cover_info.data[2 * j];
        unsigned char stego_low = (orig_low & ~15) | b_low;
        apply_opa_pixel_all(stego_low, orig_low, 4);
        modified[2 * j] = stego_low;

        // Pixel 2*j + 1 (upper half-byte)
        unsigned char orig_high = cover_info.data[2 * j + 1];
        unsigned char stego_high = (orig_high & ~15) | b_high;
        apply_opa_pixel_all(stego_high, orig_high, 4);
        modified[2 * j + 1] = stego_high;
    }

    // Generate permutation seeded by z
    vector<size_t> perm = generate_permutation_all(P, static_cast<uint32_t>(z));

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
            apply_opa_pixel_all(stego, orig, current_t);
            modified[pos] = stego;
        } else {
            modified[pos] = cover_info.data[pos];
        }
        ++pos;
    }

    out_info = cover_info;
    out_info.data = std::move(modified);
    return true;
}

bool extract_z_share_mlsb_all(const TiffInfo& stego_info,
                              std::vector<unsigned char>& out_z_share)
{
    if (stego_info.size() < 6) {
        cerr << "mlsb_all extract: stego too small for header\n";
        return false;
    }
    out_z_share.resize(3);
    for (int j = 0; j < 3; ++j) {
        unsigned char b_low  = stego_info.data[2 * j] & 15;
        unsigned char b_high = stego_info.data[2 * j + 1] & 15;
        out_z_share[j] = b_low | (b_high << 4);
    }
    return true;
}

bool extract_main_share_mlsb_all(const TiffInfo& stego_info,
                                 size_t z,
                                 size_t main_share_len,
                                 std::vector<unsigned char>& out_main_share)
{
    if (stego_info.size() < 6) {
        cerr << "mlsb_all extract: stego too small for payload\n";
        return false;
    }

    size_t N_cover = stego_info.size();
    size_t P = N_cover - 6;

    // L = 8 * main_share_len
    size_t L = 8 * main_share_len;

    // Compute t* and split n_t*
    size_t t_star = (L + P - 1) / P;
    if (t_star > 8) {
        cerr << "mlsb_all extract: payload too large even for t=8\n";
        return false;
    }

    long long diff = static_cast<long long>(L) - static_cast<long long>(P) * (t_star - 1);
    size_t n_t_star = static_cast<size_t>(max(0LL, diff));

    // Generate permutation seeded by z
    vector<size_t> perm = generate_permutation_all(P, static_cast<uint32_t>(z));

    // Extract payload bits
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

    // Pack bits back to bytes
    out_main_share.assign(main_share_len, 0);
    for (size_t i = 0; i < main_share_len; ++i) {
        unsigned char b = 0;
        for (int bit = 0; bit < 8; ++bit) {
            b |= (payload_bits[i * 8 + bit] << bit);
        }
        out_main_share[i] = b;
    }

    return true;
}
