#include "pvo_inband.h"
#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>

using namespace std;

namespace {
constexpr uint32_t kMagic = 0x50494E42u; // "PINB"
constexpr int kHeaderBits = 32 + 20;     // magic + n_data_pairs

static inline bool pvo_expand(unsigned char s1, unsigned char s2, unsigned char bit,
    unsigned char& out_s1, unsigned char& out_s2)
{
    int l = (static_cast<int>(s2) + static_cast<int>(s1)) / 2;
    int h = static_cast<int>(s2) - static_cast<int>(s1);
    long long hp = static_cast<long long>(2 * h) + bit;
    long long hf = hp / 2;
    long long hc = (hp + 1) / 2;
    long long ns2 = static_cast<long long>(l) + hc;
    long long ns1 = static_cast<long long>(l) - hf;
    if (ns1 < 0 || ns1 > 255 || ns2 < 0 || ns2 > 255) return false;
    out_s1 = static_cast<unsigned char>(ns1);
    out_s2 = static_cast<unsigned char>(ns2);
    return true;
}

static inline void pvo_contract(unsigned char s1p, unsigned char s2p,
    unsigned char& out_bit, unsigned char& out_s1, unsigned char& out_s2)
{
    int lp = (static_cast<int>(s2p) + static_cast<int>(s1p)) / 2;
    int hp = static_cast<int>(s2p) - static_cast<int>(s1p);
    out_bit = static_cast<unsigned char>(hp & 1u);
    long long h = (static_cast<long long>(hp) - out_bit) / 2;
    long long hf = h / 2;
    long long hc = (h + 1) / 2;
    long long s2 = static_cast<long long>(lp) + hc;
    long long s1 = static_cast<long long>(lp) - hf;
    out_s2 = static_cast<unsigned char>(max(0LL, min(255LL, s2)));
    out_s1 = static_cast<unsigned char>(max(0LL, min(255LL, s1)));
}

static inline unsigned char get_share_bit(const vector<unsigned char>& share, size_t bit_idx)
{
    return static_cast<unsigned char>((share[bit_idx / 8] >> (bit_idx % 8)) & 1u);
}

static inline void write_u32_bits(vector<uint8_t>& bits, uint32_t v, int n)
{
    for (int i = 0; i < n; ++i) bits.push_back(static_cast<uint8_t>((v >> i) & 1u));
}

static inline uint32_t read_u32_bits(const vector<uint8_t>& bits, size_t& p, int n)
{
    uint32_t v = 0;
    for (int i = 0; i < n; ++i) {
        if (p < bits.size() && bits[p]) v |= (1u << i);
        ++p;
    }
    return v;
}

static void write_bits_lsb2(vector<unsigned char>& pixels, size_t pixel_start, const vector<uint8_t>& bits)
{
    size_t bit_pos = 0;
    for (size_t i = 0; i < bits.size();) {
        uint8_t two = 0;
        if (i < bits.size()) two |= static_cast<uint8_t>(bits[i++] & 1u);
        if (i < bits.size()) two |= static_cast<uint8_t>((bits[i++] & 1u) << 1);
        size_t p = pixel_start + bit_pos;
        pixels[p] = static_cast<unsigned char>((pixels[p] & 0xFCu) | two);
        ++bit_pos;
    }
}

static vector<uint8_t> read_bits_lsb2(const vector<unsigned char>& pixels, size_t pixel_start, size_t bit_len)
{
    vector<uint8_t> bits;
    bits.reserve(bit_len);
    for (size_t i = 0; bits.size() < bit_len; ++i) {
        uint8_t two = static_cast<uint8_t>(pixels[pixel_start + i] & 0x3u);
        bits.push_back(two & 1u);
        if (bits.size() < bit_len) bits.push_back((two >> 1) & 1u);
    }
    return bits;
}
} // namespace

bool embed_share_pvo_inband(const TiffInfo& cover_info, const vector<unsigned char>& share, TiffInfo& out_info)
{
    const size_t n_pixels = cover_info.size();
    if (n_pixels < 4) return false;
    const size_t n_pairs = n_pixels / 2;
    const size_t req_bits = share.size() * 8;

    // Choose reserved prefix pairs so metadata (header + 2 maps) fits in 2-LSB channel.
    // meta_bits = kHeaderBits + 2 * n_data_pairs, n_data_pairs = n_pairs - R
    // capacity_bits = reserved_pixels * 2 = (2R)*2 = 4R
    // 4R >= kHeaderBits + 2*(n_pairs - R)
    size_t R = (kHeaderBits + 2 * n_pairs + 5) / 6; // ceil division
    if (R >= n_pairs) return false;
    const size_t n_data_pairs = n_pairs - R;

    vector<unsigned char> modified = cover_info.data;
    vector<uint8_t> used_map(n_data_pairs, 0), perm_map(n_data_pairs, 0);
    size_t embedded = 0;

    for (size_t rank = 0; rank < n_data_pairs && embedded < req_bits; ++rank) {
        size_t pi = R + rank;
        size_t idx = pi * 2;
        unsigned char v0 = cover_info.data[idx], v1 = cover_info.data[idx + 1];
        uint8_t perm = (v0 > v1) ? 1u : 0u;
        unsigned char s1 = perm ? v1 : v0;
        unsigned char s2 = perm ? v0 : v1;
        unsigned char bit = get_share_bit(share, embedded);
        unsigned char ns1 = 0, ns2 = 0;
        if (pvo_expand(s1, s2, bit, ns1, ns2)) {
            modified[idx] = perm ? ns2 : ns1;
            modified[idx + 1] = perm ? ns1 : ns2;
            used_map[rank] = 1;
            perm_map[rank] = perm;
            ++embedded;
        } else {
            modified[idx] = v0;
            modified[idx + 1] = v1;
            used_map[rank] = 0;
            perm_map[rank] = perm;
        }
    }

    if (embedded != req_bits) {
        cerr << "PVO-inband embed: embedded " << embedded << " of " << req_bits << " bits\n";
        return false;
    }

    vector<uint8_t> meta_bits;
    meta_bits.reserve(static_cast<size_t>(kHeaderBits) + 2 * n_data_pairs);
    write_u32_bits(meta_bits, kMagic, 32);
    write_u32_bits(meta_bits, static_cast<uint32_t>(n_data_pairs), 20);
    for (size_t i = 0; i < n_data_pairs; ++i) meta_bits.push_back(used_map[i]);
    for (size_t i = 0; i < n_data_pairs; ++i) meta_bits.push_back(perm_map[i]);

    const size_t meta_cap_bits = (R * 2) * 2; // reserved pixels * 2 bits
    if (meta_bits.size() > meta_cap_bits) {
        cerr << "PVO-inband embed: metadata overflow\n";
        return false;
    }
    meta_bits.resize(meta_cap_bits, 0);
    write_bits_lsb2(modified, 0, meta_bits);

    out_info = cover_info;
    out_info.data = std::move(modified);
    return true;
}

bool recover_share_pvo_inband(const TiffInfo& stego_info, size_t share_len, vector<unsigned char>& out_share, TiffInfo& recovered_cover)
{
    const size_t n_pixels = stego_info.size();
    if (n_pixels < 4) return false;
    const size_t n_pairs = n_pixels / 2;

    // Try consistent R from formula.
    size_t R = (kHeaderBits + 2 * n_pairs + 5) / 6;
    if (R >= n_pairs) return false;
    size_t meta_cap_bits = (R * 2) * 2;
    vector<uint8_t> meta = read_bits_lsb2(stego_info.data, 0, meta_cap_bits);
    size_t p = 0;
    uint32_t magic = read_u32_bits(meta, p, 32);
    if (magic != kMagic) {
        cerr << "PVO-inband recover: bad magic\n";
        return false;
    }
    uint32_t n_data_pairs_u32 = read_u32_bits(meta, p, 20);
    size_t n_data_pairs = static_cast<size_t>(n_data_pairs_u32);
    if (n_data_pairs > n_pairs || n_pairs - n_data_pairs != R) return false;

    vector<uint8_t> used_map(n_data_pairs, 0), perm_map(n_data_pairs, 0);
    for (size_t i = 0; i < n_data_pairs; ++i) used_map[i] = (p < meta.size() ? meta[p++] : 0);
    for (size_t i = 0; i < n_data_pairs; ++i) perm_map[i] = (p < meta.size() ? meta[p++] : 0);

    const size_t need_bits = share_len * 8;
    out_share.assign(share_len, 0);
    size_t extracted = 0;
    vector<unsigned char> recovered = stego_info.data;

    for (size_t rank = 0; rank < n_data_pairs && extracted < need_bits; ++rank) {
        size_t pi = R + rank;
        size_t idx = pi * 2;
        uint8_t perm = perm_map[rank];
        unsigned char rp = stego_info.data[idx], rq = stego_info.data[idx + 1];
        unsigned char s1p = perm ? rq : rp;
        unsigned char s2p = perm ? rp : rq;

        if (used_map[rank]) {
            unsigned char bit = 0, os1 = 0, os2 = 0;
            pvo_contract(s1p, s2p, bit, os1, os2);
            recovered[idx] = perm ? os2 : os1;
            recovered[idx + 1] = perm ? os1 : os2;
            if (bit) out_share[extracted / 8] |= static_cast<unsigned char>(1u << (extracted % 8));
            ++extracted;
        } else {
            recovered[idx] = rp;
            recovered[idx + 1] = rq;
        }
    }

    if (extracted != need_bits) {
        cerr << "PVO-inband recover: extracted " << extracted << " of " << need_bits << " bits\n";
        return false;
    }

    // Prefix metadata region is not restored in this prototype.
    recovered_cover = stego_info;
    recovered_cover.data = std::move(recovered);
    return true;
}
