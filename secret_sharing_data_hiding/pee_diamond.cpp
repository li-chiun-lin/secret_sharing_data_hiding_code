#include "pee_diamond.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace std;

namespace {
constexpr uint8_t kPixNone  = 0;
constexpr uint8_t kPixEmbed = 1;
constexpr uint8_t kPixShift = 2;
} // namespace

static inline int diamond_predict(const vector<unsigned char>& img,
    uint32_t W, uint32_t H,
    uint32_t row, uint32_t col)
{
    int sum = 0, count = 0;

    if (row > 0)     { sum += img[(row - 1) * W + col]; ++count; }
    if (row < H - 1) { sum += img[(row + 1) * W + col]; ++count; }
    if (col > 0)     { sum += img[row * W + col - 1];   ++count; }
    if (col < W - 1) { sum += img[row * W + col + 1];   ++count; }

    if (count == 0) return 128;
    return sum / count;
}

static inline unsigned char get_share_bit(const vector<unsigned char>& share, size_t bit_idx)
{
    return static_cast<unsigned char>((share[bit_idx / 8] >> (bit_idx % 8)) & 1u);
}

static void embed_one_pixel(vector<unsigned char>& img,
    const vector<unsigned char>& pred_buf,
    const vector<unsigned char>& orig_buf,
    vector<uint8_t>& pix_kind,
    uint32_t W, uint32_t H,
    uint32_t row, uint32_t col,
    int T,
    const vector<unsigned char>& share,
    size_t& embedded_bits,
    size_t required_bits)
{
    const size_t pos = static_cast<size_t>(row) * W + col;

    const int p = diamond_predict(pred_buf, W, H, row, col);
    const int c = static_cast<int>(orig_buf[pos]);
    const int e = c - p;

    if (embedded_bits < required_bits && abs(e) <= T) {
        const unsigned char bit = get_share_bit(share, embedded_bits);
        const long long eprime = static_cast<long long>(2 * e) + bit;
        const long long cprime = static_cast<long long>(p) + eprime;
        if (cprime >= 0 && cprime <= 255) {
            img[pos] = static_cast<unsigned char>(cprime);
            pix_kind[pos] = kPixEmbed;
            ++embedded_bits;
            return;
        }
        for (int sgn : { e >= 0 ? 1 : -1, e >= 0 ? -1 : 1 }) {
            const long long cps =
                static_cast<long long>(p) + e + static_cast<long long>(sgn) * (2 * T + 1);
            if (cps >= 0 && cps <= 255) {
                img[pos] = static_cast<unsigned char>(cps);
                pix_kind[pos] = kPixShift;
                return;
            }
        }
        return;
    }

    if (T < abs(e) && abs(e) <= 2 * T + 1) {
        const int sign = (e > 0) ? 1 : -1;
        const long long cprime =
            static_cast<long long>(p) + e + static_cast<long long>(sign) * (2 * T + 1);
        if (cprime >= 0 && cprime <= 255) {
            img[pos] = static_cast<unsigned char>(cprime);
            pix_kind[pos] = kPixShift;
        }
    }
}

static void recover_one_pixel(const vector<unsigned char>& stego,
    vector<unsigned char>& recovered,
    const vector<uint8_t>& pix_kind,
    uint32_t W, uint32_t H,
    uint32_t row, uint32_t col,
    int T,
    vector<unsigned char>& out_share,
    unsigned char& cur_byte,
    int& cur_bitpos,
    size_t& extracted_bits)
{
    const size_t pos = static_cast<size_t>(row) * W + col;

    const int p = diamond_predict(recovered, W, H, row, col);
    const int sp = static_cast<int>(stego[pos]);
    const uint8_t kind = pix_kind[pos];

    if (kind == kPixEmbed) {
        const int eprime = sp - p;
        const unsigned char bit = static_cast<unsigned char>(eprime & 1u);
        int e = (eprime - static_cast<int>(bit)) / 2;
        int orig = p + e;
        orig = max(0, min(255, orig));
        recovered[pos] = static_cast<unsigned char>(orig);

        cur_byte |= static_cast<unsigned char>(bit << cur_bitpos);
        if (++cur_bitpos == 8) {
            out_share.push_back(cur_byte);
            cur_byte = 0;
            cur_bitpos = 0;
        }
        ++extracted_bits;
    } else if (kind == kPixShift) {
        const int eprime = sp - p;
        const int sign = (eprime > 0) ? 1 : -1;
        int e = eprime - sign * (2 * T + 1);
        int orig = p + e;
        orig = max(0, min(255, orig));
        recovered[pos] = static_cast<unsigned char>(orig);
    } else {
        recovered[pos] = static_cast<unsigned char>(sp);
    }
}

bool embed_share_pee_diamond(const TiffInfo& cover_info,
    const vector<unsigned char>& share,
    TiffInfo& out_info,
    vector<uint8_t>& pix_kind,
    int T)
{
    if (T < 0) return false;

    const uint32_t W = cover_info.width;
    const uint32_t H = cover_info.height;
    if (W == 0 || H == 0) return false;

    const size_t n = cover_info.size();
    const size_t required_bits = share.size() * 8;

    pix_kind.assign(n, kPixNone);
    vector<unsigned char> modified = cover_info.data;
    size_t embedded_bits = 0;

    for (uint32_t row = 0; row < H && embedded_bits < required_bits; ++row) {
        for (uint32_t col = (row % 2 == 0) ? 1u : 0u;
             col < W && embedded_bits < required_bits;
             col += 2)
        {
            embed_one_pixel(modified, cover_info.data, cover_info.data, pix_kind, W, H, row, col, T,
                            share, embedded_bits, required_bits);
        }
    }

    for (uint32_t row = 0; row < H && embedded_bits < required_bits; ++row) {
        for (uint32_t col = (row % 2 == 0) ? 0u : 1u;
             col < W && embedded_bits < required_bits;
             col += 2)
        {
            embed_one_pixel(modified, modified, cover_info.data, pix_kind, W, H, row, col, T,
                            share, embedded_bits, required_bits);
        }
    }

    if (embedded_bits != required_bits) {
        cerr << "PEE-Diamond embed: embedded " << embedded_bits
             << " of " << required_bits << " bits\n";
        return false;
    }

    out_info = cover_info;
    out_info.data = std::move(modified);
    return true;
}

bool recover_share_pee_diamond(const TiffInfo& stego_info,
    size_t share_len,
    const vector<uint8_t>& pix_kind,
    vector<unsigned char>& out_share,
    TiffInfo& recovered_cover,
    int T)
{
    if (T < 0) return false;

    const uint32_t W = stego_info.width;
    const uint32_t H = stego_info.height;
    if (W == 0 || H == 0) return false;

    if (pix_kind.size() != stego_info.size()) {
        cerr << "PEE-Diamond recover: pix_kind size mismatch\n";
        return false;
    }

    const size_t needed_bits = share_len * 8;

    vector<unsigned char> recovered = stego_info.data;

    out_share.clear();
    out_share.reserve(share_len);

    size_t extracted_bits = 0;
    unsigned char cur_byte = 0;
    int cur_bitpos = 0;

    for (uint32_t row = 0; row < H; ++row) {
        for (uint32_t col = (row % 2 == 0) ? 0u : 1u; col < W; col += 2) {
            recover_one_pixel(stego_info.data, recovered, pix_kind, W, H, row, col, T,
                              out_share, cur_byte, cur_bitpos, extracted_bits);
        }
    }

    for (uint32_t row = 0; row < H; ++row) {
        for (uint32_t col = (row % 2 == 0) ? 1u : 0u; col < W; col += 2) {
            recover_one_pixel(stego_info.data, recovered, pix_kind, W, H, row, col, T,
                              out_share, cur_byte, cur_bitpos, extracted_bits);
        }
    }

    if (extracted_bits != needed_bits) {
        cerr << "PEE-Diamond recover: extracted " << extracted_bits
             << " of " << needed_bits << " bits\n";
        return false;
    }

    recovered_cover = stego_info;
    recovered_cover.data = std::move(recovered);
    return true;
}

static bool embed_one_pixel_no_map(vector<unsigned char>& img,
    const vector<unsigned char>& pred_buf,
    const vector<unsigned char>& orig_buf,
    uint32_t W, uint32_t H,
    uint32_t row, uint32_t col,
    int T,
    const vector<unsigned char>& share,
    size_t& embedded_bits,
    size_t required_bits)
{
    const size_t pos = static_cast<size_t>(row) * W + col;
    const int p = diamond_predict(pred_buf, W, H, row, col);
    const int c = static_cast<int>(orig_buf[pos]);
    const int e = c - p;

    const bool embed_safe = (p >= 2 * T) && (p <= 254 - 2 * T);
    const int pos_shift_max_e = 254 - p - T; // max e for e' = e + (T+1) to stay <= 255-p
    const int neg_shift_min_e = T + 1 - p;   // min e for e' = e - (T+1) to stay >= -p

    int eprime = e; // default: unchanged branch
    if (embed_safe && abs(e) <= T) {
        // Ambiguous interval: must always occupy with embed mapping.
        int bit = 0;
        if (embedded_bits < required_bits) {
            bit = static_cast<int>(get_share_bit(share, embedded_bits));
            ++embedded_bits;
        }
        eprime = 2 * e + bit;
    } else if (e >= T + 1 && e <= pos_shift_max_e) {
        eprime = e + (T + 1);
    } else if (e <= -(T + 1) && e >= neg_shift_min_e) {
        eprime = e - (T + 1);
    } // otherwise keep unchanged (eprime = e)

    const int cprime = p + eprime;
    if (cprime < 0 || cprime > 255) return false;
    img[pos] = static_cast<unsigned char>(cprime);
    return true;
}

bool embed_share_pee_diamond_no_map(const TiffInfo& cover_info,
    const vector<unsigned char>& share,
    TiffInfo& out_info,
    int T)
{
    if (T < 0) return false;
    const uint32_t W = cover_info.width;
    const uint32_t H = cover_info.height;
    if (W == 0 || H == 0) return false;

    // Robust no-map mode: repeat each payload bit 3 times.
    // This mitigates occasional decision drift without any side information.
    const size_t payload_bits = share.size() * 8;
    const size_t required_bits = payload_bits * 3;
    vector<unsigned char> rep_share((required_bits + 7) / 8, 0);
    for (size_t i = 0; i < payload_bits; ++i) {
        unsigned char b = get_share_bit(share, i);
        if (b) {
            size_t j0 = 3 * i + 0;
            size_t j1 = 3 * i + 1;
            size_t j2 = 3 * i + 2;
            rep_share[j0 / 8] |= static_cast<unsigned char>(1u << (j0 % 8));
            rep_share[j1 / 8] |= static_cast<unsigned char>(1u << (j1 % 8));
            rep_share[j2 / 8] |= static_cast<unsigned char>(1u << (j2 % 8));
        }
    }
    vector<unsigned char> modified = cover_info.data;
    size_t embedded_bits = 0;

    // Phase 1 (same checkerboard order as original implementation)
    for (uint32_t row = 0; row < H; ++row) {
        for (uint32_t col = (row % 2 == 0) ? 1u : 0u; col < W; col += 2) {
            if (!embed_one_pixel_no_map(modified, cover_info.data, cover_info.data,
                                        W, H, row, col, T, rep_share, embedded_bits, required_bits)) return false;
        }
    }

    // Keep phase-2 unchanged in robust no-map mode.

    if (embedded_bits < required_bits) {
        cerr << "PEE-Diamond(no-map) embed: embedded " << embedded_bits
             << " of " << required_bits << " bits\n";
        return false;
    }

    out_info = cover_info;
    out_info.data = std::move(modified);
    return true;
}

static void recover_one_pixel_no_map(const vector<unsigned char>& stego,
    vector<unsigned char>& recovered,
    uint32_t W, uint32_t H,
    uint32_t row, uint32_t col,
    int T)
{
    const size_t pos = static_cast<size_t>(row) * W + col;
    const int p = diamond_predict(recovered, W, H, row, col);
    const int sp = static_cast<int>(stego[pos]);
    const int eprime = sp - p;

    int e = 0;
    const bool embed_safe = (p >= 2 * T) && (p <= 254 - 2 * T);
    if (embed_safe && eprime >= -2 * T && eprime <= 2 * T + 1) {
        const unsigned char bit = static_cast<unsigned char>(eprime & 1u);
        e = (eprime - static_cast<int>(bit)) / 2;
    } else if (eprime >= 2 * T + 2 && eprime <= 255 - p) {
        e = eprime - (T + 1);
    } else if (eprime <= -2 * T - 2 && eprime >= -p) {
        e = eprime + (T + 1);
    } else {
        e = eprime; // unchanged branch
    }

    int orig = p + e;
    orig = max(0, min(255, orig));
    recovered[pos] = static_cast<unsigned char>(orig);
}

bool recover_share_pee_diamond_no_map(const TiffInfo& stego_info,
    size_t share_len,
    vector<unsigned char>& out_share,
    TiffInfo& recovered_cover,
    int T)
{
    if (T < 0) return false;
    const uint32_t W = stego_info.width;
    const uint32_t H = stego_info.height;
    if (W == 0 || H == 0) return false;

    const size_t payload_bits = share_len * 8;
    const size_t needed_bits = payload_bits * 3;
    vector<unsigned char> recovered = stego_info.data;

    // Reverse order corresponding to two embedding phases
    for (uint32_t row = 0; row < H; ++row) {
        for (uint32_t col = (row % 2 == 0) ? 0u : 1u; col < W; col += 2) {
            recover_one_pixel_no_map(stego_info.data, recovered, W, H, row, col, T);
        }
    }

    for (uint32_t row = 0; row < H; ++row) {
        for (uint32_t col = (row % 2 == 0) ? 1u : 0u; col < W; col += 2) {
            recover_one_pixel_no_map(stego_info.data, recovered, W, H, row, col, T);
        }
    }

    // Extract bits from phase-1 carrier set only.
    vector<unsigned char> rep_bits((needed_bits + 7) / 8, 0);
    size_t extracted_bits = 0;
    auto try_extract_bit = [&](int p, int sp, size_t bit_index) {
        if (bit_index >= needed_bits) return;
        const bool embed_safe = (p >= 2 * T) && (p <= 254 - 2 * T);
        const int eprime = sp - p;
        if (embed_safe && eprime >= -2 * T && eprime <= 2 * T + 1) {
            const unsigned char bit = static_cast<unsigned char>(eprime & 1u);
            if (bit) rep_bits[bit_index / 8] |= static_cast<unsigned char>(1u << (bit_index % 8));
            ++extracted_bits;
        }
    };

    for (uint32_t row = 0; row < H && extracted_bits < needed_bits; ++row) {
        for (uint32_t col = (row % 2 == 0) ? 1u : 0u; col < W && extracted_bits < needed_bits; col += 2) {
            const size_t pos = static_cast<size_t>(row) * W + col;
            int p = diamond_predict(recovered, W, H, row, col);
            int sp = static_cast<int>(stego_info.data[pos]);
            try_extract_bit(p, sp, extracted_bits);
        }
    }
    if (extracted_bits != needed_bits) {
        cerr << "PEE-Diamond(no-map) recover: extracted " << extracted_bits
             << " of " << needed_bits << " bits\n";
        return false;
    }

    // Majority-vote decode (3x repetition) back to payload bits.
    out_share.assign(share_len, 0);
    for (size_t i = 0; i < payload_bits; ++i) {
        size_t j0 = 3 * i + 0;
        size_t j1 = 3 * i + 1;
        size_t j2 = 3 * i + 2;
        int v0 = (rep_bits[j0 / 8] >> (j0 % 8)) & 1;
        int v1 = (rep_bits[j1 / 8] >> (j1 % 8)) & 1;
        int v2 = (rep_bits[j2 / 8] >> (j2 % 8)) & 1;
        int sum = v0 + v1 + v2;
        if (sum >= 2) out_share[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
    }

    recovered_cover = stego_info;
    recovered_cover.data = std::move(recovered);
    return true;
}
