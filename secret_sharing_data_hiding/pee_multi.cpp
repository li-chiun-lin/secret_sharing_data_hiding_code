#include "pee_multi.h"
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <vector>

using namespace std;

static inline int med_predict(const vector<unsigned char>& img,
    uint32_t W, uint32_t H,
    uint32_t row, uint32_t col)
{
    (void)H;
    const bool have_left = (col > 0);
    const bool have_top  = (row > 0);

    if (!have_left && !have_top) return 128;

    const int a = have_left ? static_cast<int>(img[row * W + col - 1])       : 0;
    const int b = have_top  ? static_cast<int>(img[(row - 1) * W + col])     : 0;
    const int c = (have_left && have_top)
                ? static_cast<int>(img[(row - 1) * W + col - 1]) : 0;

    if (!have_left) return b;
    if (!have_top)  return a;

    const int lo = min(a, b);
    const int hi = max(a, b);
    if (c >= hi) return lo;
    if (c <= lo) return hi;
    return a + b - c;
}

static inline unsigned char get_share_bit(const vector<unsigned char>& share,
    size_t bit_idx)
{
    return static_cast<unsigned char>((share[bit_idx / 8] >> (bit_idx % 8)) & 1u);
}

static size_t embed_one_layer(vector<unsigned char>& img,
    const vector<unsigned char>& orig,
    uint32_t W, uint32_t H,
    const vector<unsigned char>& share,
    size_t share_bit_offset,
    size_t required_bits,
    int T, int B)
{
    const int skip_lo = B + 2 * T;
    const int skip_hi = 255 - B - 2 * T;

    size_t embedded = 0;
    const int shift_delta = T + 1;

    for (uint32_t row = 0; row < H && embedded < required_bits; ++row) {
        for (uint32_t col = 0; col < W && embedded < required_bits; ++col) {
            const size_t pos = static_cast<size_t>(row) * W + col;
            const int c = static_cast<int>(orig[pos]);

            if (c < skip_lo || c > skip_hi) {
                img[pos] = static_cast<unsigned char>(c);
                continue;
            }

            const int p = med_predict(orig, W, H, row, col);
            const int e = c - p;

            if (abs(e) <= T) {
                const unsigned char bit =
                    get_share_bit(share, share_bit_offset + embedded);
                const long long eprime = static_cast<long long>(2 * e) + bit;
                const long long cprime = static_cast<long long>(p) + eprime;
                if (cprime >= B && cprime <= 255 - B) {
                    img[pos] = static_cast<unsigned char>(cprime);
                    ++embedded;
                    continue;
                }
                // Embed overflow fallback: deterministic shift (no bit consumed)
                bool shifted = false;
                for (int sgn : { e >= 0 ? 1 : -1, e >= 0 ? -1 : 1 }) {
                    const long long cps =
                        static_cast<long long>(p) + e + static_cast<long long>(sgn) * shift_delta;
                    if (cps >= B && cps <= 255 - B) {
                        img[pos] = static_cast<unsigned char>(cps);
                        shifted = true;
                        break;
                    }
                }
                if (!shifted) {
                    // Deterministic no-map mode cannot leave ambiguous pixels
                    // untouched while payload is still pending.
                    return embedded;
                }
                continue;
            }

            if (abs(e) > T) {
                const int sign = (e >= 0) ? 1 : -1;
                const long long cprime =
                    static_cast<long long>(p) + e + static_cast<long long>(sign) * shift_delta;
                if (cprime >= B && cprime <= 255 - B) {
                    img[pos] = static_cast<unsigned char>(cprime);
                } else {
                    // If error sits in decoder's embedded band and cannot be shifted
                    // out, we must stop to preserve deterministic decoding.
                    if (abs(e) <= 2 * T + 1) return embedded;
                    img[pos] = static_cast<unsigned char>(c);
                }
            }
        }
    }

    return embedded;
}

static size_t recover_one_layer(const vector<unsigned char>& stego,
    vector<unsigned char>& prev_layer,
    uint32_t W, uint32_t H,
    vector<unsigned char>& out_share_bytes,
    unsigned char& cur_byte,
    int& cur_bitpos,
    size_t required_bits,
    int T, int B)
{
    size_t extracted = 0;
    const int shift_delta = T + 1;

    for (uint32_t row = 0; row < H; ++row) {
        for (uint32_t col = 0; col < W; ++col) {
            const size_t pos = static_cast<size_t>(row) * W + col;
            const int sp = static_cast<int>(stego[pos]);
            if (extracted >= required_bits) {
                // Embedder stops modifying once quota is met; remaining pixels
                // are untouched for this layer.
                prev_layer[pos] = static_cast<unsigned char>(sp);
                continue;
            }
            if (sp < B || sp > 255 - B) {
                prev_layer[pos] = static_cast<unsigned char>(sp);
                continue;
            }

            const int p = med_predict(prev_layer, W, H, row, col);
            const int eprime = sp - p;
            if (abs(eprime) <= 2 * T + 1) {
                const unsigned char bit = static_cast<unsigned char>(eprime & 1u);
                int e = (eprime - static_cast<int>(bit)) / 2;
                int orig = p + e;
                orig = max(0, min(255, orig));
                prev_layer[pos] = static_cast<unsigned char>(orig);

                cur_byte |= static_cast<unsigned char>(bit << cur_bitpos);
                if (++cur_bitpos == 8) {
                    out_share_bytes.push_back(cur_byte);
                    cur_byte = 0;
                    cur_bitpos = 0;
                }
                ++extracted;
            } else {
                const int sign = (eprime >= 0) ? 1 : -1;
                int e = eprime - sign * shift_delta;
                // If |e| is still outside embeddable band, this was a shifted pixel.
                // Otherwise treat it as unchanged to avoid false inverse shifts.
                if (abs(e) > T) {
                    int orig = p + e;
                    orig = max(0, min(255, orig));
                    prev_layer[pos] = static_cast<unsigned char>(orig);
                } else {
                    prev_layer[pos] = static_cast<unsigned char>(sp);
                }
            }
        }
    }

    (void)required_bits; // layer caller checks sum
    return extracted;
}

bool embed_share_pee_multi(const TiffInfo& cover_info,
    const vector<unsigned char>& share,
    TiffInfo& out_info,
    vector<size_t>& bits_per_layer,
    int T, int B, int max_layers)
{
    if (T < 0 || B < 1 || max_layers < 1) return false;

    const uint32_t W = cover_info.width;
    const uint32_t H = cover_info.height;
    if (W == 0 || H == 0) return false;

    const int skip_lo = B + 2 * T;
    const int skip_hi = 255 - B - 2 * T;
    if (skip_lo > skip_hi) {
        cerr << "PEE-Multi embed: B=" << B << " T=" << T
             << " leaves no embeddable pixel range (skip_lo=" << skip_lo
             << " > skip_hi=" << skip_hi << ")\n";
        return false;
    }

    const size_t total_bits = share.size() * 8;
    bits_per_layer.clear();

    vector<unsigned char> current = cover_info.data;
    size_t bits_embedded_so_far = 0;

    for (int layer = 0; layer < max_layers && bits_embedded_so_far < total_bits; ++layer) {
        const size_t remaining = total_bits - bits_embedded_so_far;

        vector<unsigned char> next = current;
        const size_t embedded = embed_one_layer(next, current,
                                                W, H,
                                                share,
                                                bits_embedded_so_far, remaining,
                                                T, B);

        if (embedded == 0) {
            cerr << "PEE-Multi embed: layer " << layer
                 << " embedded 0 bits - capacity exhausted\n";
            return false;
        }

        bits_per_layer.push_back(embedded);
        bits_embedded_so_far += embedded;
        current = std::move(next);
    }

    if (bits_embedded_so_far != total_bits) {
        cerr << "PEE-Multi embed: embedded " << bits_embedded_so_far
             << " of " << total_bits << " bits in "
             << max_layers << " layers\n";
        return false;
    }

    out_info = cover_info;
    out_info.data = std::move(current);
    return true;
}

bool recover_share_pee_multi(const TiffInfo& stego_info,
    size_t share_len,
    const vector<size_t>& bits_per_layer,
    vector<unsigned char>& out_share,
    TiffInfo& recovered_cover,
    int T, int B)
{
    if (T < 0 || B < 1) return false;

    const uint32_t W = stego_info.width;
    const uint32_t H = stego_info.height;
    if (W == 0 || H == 0) return false;

    if (bits_per_layer.empty()) {
        cerr << "PEE-Multi recover: bits_per_layer is empty\n";
        return false;
    }

    const size_t total_bits = share_len * 8;

    size_t sum = 0;
    for (size_t b : bits_per_layer) sum += b;
    if (sum != total_bits) {
        cerr << "PEE-Multi recover: bits_per_layer sum=" << sum
             << " != total_bits=" << total_bits << "\n";
        return false;
    }

    const int num_layers = static_cast<int>(bits_per_layer.size());

    vector<vector<unsigned char>> layer_bits_rev;
    layer_bits_rev.reserve(num_layers);

    vector<unsigned char> current_stego = stego_info.data;

    for (int layer = num_layers - 1; layer >= 0; --layer) {
        const size_t layer_bits = bits_per_layer[layer];

        vector<unsigned char> prev_layer = current_stego;

        vector<unsigned char> layer_share_bytes;
        layer_share_bytes.reserve((layer_bits + 7) / 8);
        unsigned char cur_byte   = 0;
        int           cur_bitpos = 0;

        const size_t extracted = recover_one_layer(current_stego, prev_layer,
                                                   W, H,
                                                   layer_share_bytes,
                                                   cur_byte, cur_bitpos,
                                                   layer_bits,
                                                   T, B);

        if (cur_bitpos > 0)
            layer_share_bytes.push_back(cur_byte);

        if (extracted != layer_bits) {
            cerr << "PEE-Multi recover: layer " << layer
                 << " extracted " << extracted
                 << " of " << layer_bits << " bits\n";
            return false;
        }

        vector<unsigned char> layer_bits_vec;
        layer_bits_vec.reserve(layer_bits);
        for (size_t bi = 0; bi < layer_bits; ++bi) {
            layer_bits_vec.push_back(
                static_cast<unsigned char>((layer_share_bytes[bi / 8] >> (bi % 8)) & 1u));
        }
        layer_bits_rev.push_back(std::move(layer_bits_vec));

        current_stego = std::move(prev_layer);
    }

    vector<unsigned char> raw_bits;
    raw_bits.reserve(total_bits);
    for (auto it = layer_bits_rev.rbegin(); it != layer_bits_rev.rend(); ++it) {
        raw_bits.insert(raw_bits.end(), it->begin(), it->end());
    }

    out_share.assign(share_len, 0);
    for (size_t bi = 0; bi < total_bits; ++bi) {
        out_share[bi / 8] |= static_cast<unsigned char>(raw_bits[bi] << (bi % 8));
    }

    recovered_cover = stego_info;
    recovered_cover.data = std::move(current_stego);
    return true;
}
