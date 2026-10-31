#include "mlsb_bs.h"
#include "mlsb.h"
#include <iostream>
#include <cassert>

using namespace std;

// ---------------------------------------------------------------------------
// Helper: check whether a block has a constant polynomial, i.e. all
// coefficients of degree >= 1 are zero: bytes [1..k-1] are all 0x00.
// This means f(x) = a0 for all x, leaking that a1=...=a(k-1)=0.
// The value of byte[0] (a0) does not matter.
// ---------------------------------------------------------------------------
static bool is_constant_block(const vector<unsigned char>& px, size_t base, int k)
{
    for (int j = 1; j < k; ++j)
        if (px[base + static_cast<size_t>(j)] != 0x00) return false;
    return true;
}

// Check whether a block is the ambiguous pattern:
// bytes [1..k-2] are 0x00 and byte [k-1] == r (for any value of byte[0]).
static bool is_ambiguous_block(const vector<unsigned char>& px, size_t base,
                                int k, unsigned char r)
{
    for (int j = 1; j < k - 1; ++j)
        if (px[base + static_cast<size_t>(j)] != 0x00) return false;
    return px[base + static_cast<size_t>(k - 1)] == r;
}

// ---------------------------------------------------------------------------
// Count ambiguous blocks for a given r candidate (used to choose best r).
// ---------------------------------------------------------------------------
static int count_ambiguous(const vector<unsigned char>& px, int k,
                            size_t groups, unsigned char r)
{
    int cnt = 0;
    for (size_t g = 0; g < groups; ++g) {
        size_t base = g * static_cast<size_t>(k);
        if (is_ambiguous_block(px, base, k, r)) ++cnt;
    }
    return cnt;
}

// ---------------------------------------------------------------------------
// bs_preprocess
// ---------------------------------------------------------------------------
bool bs_preprocess(const vector<unsigned char>& pixels_in,
                   int k,
                   unsigned char& r_out,
                   size_t& z_out,
                   vector<unsigned char>& bs_out,
                   vector<unsigned char>& pixels_out)
{
    assert(k >= 2);
    assert(pixels_in.size() % static_cast<size_t>(k) == 0);

    size_t groups = pixels_in.size() / static_cast<size_t>(k);

    // Count constant-polynomial blocks (independent of r).
    // A constant block has bytes[1..k-1] all zero.
    int const_count = 0;
    for (size_t g = 0; g < groups; ++g)
        if (is_constant_block(pixels_in, g * static_cast<size_t>(k), k))
            ++const_count;

    // Choose r in [1..255] that minimises (const_count + ambiguous_count).
    unsigned char best_r = 1;
    int best_total = const_count + count_ambiguous(pixels_in, k, groups, 1);
    for (int rc = 2; rc <= 255; ++rc) {
        int total = const_count + count_ambiguous(pixels_in, k, groups,
                        static_cast<unsigned char>(rc));
        if (total < best_total) {
            best_total = total;
            best_r = static_cast<unsigned char>(rc);
        }
    }

    r_out = best_r;
    z_out = static_cast<size_t>(best_total);

    // Build modified pixel sequence and bs bitstream.
    pixels_out = pixels_in;
    bs_out.clear();

    int bit_pos = 0;

    auto write_bit = [&](int bit) {
        size_t byte_idx = static_cast<size_t>(bit_pos) / 8;
        int    bit_idx  = bit_pos % 8;
        if (bit_idx == 0) bs_out.push_back(0x00);
        if (bit) bs_out.back() |= static_cast<unsigned char>(1u << bit_idx);
        ++bit_pos;
    };

    for (size_t g = 0; g < groups; ++g) {
        size_t base = g * static_cast<size_t>(k);
        if (is_constant_block(pixels_out, base, k)) {
            // Replace byte[k-1] with r to make the polynomial non-constant.
            // Record bit 1: this was a constant block.
            pixels_out[base + static_cast<size_t>(k - 1)] = best_r;
            write_bit(1);
        } else if (is_ambiguous_block(pixels_out, base, k, best_r)) {
            // Genuine block with bytes[1..k-2]=0 and byte[k-1]=r.
            // Record bit 0: this is genuine.
            write_bit(0);
        }
        // All other blocks: no action, no bit written.
    }

    return true;
}

// ---------------------------------------------------------------------------
// bs_postprocess
// ---------------------------------------------------------------------------
bool bs_postprocess(const vector<unsigned char>& pixels_in,
                    int k,
                    unsigned char r,
                    const vector<unsigned char>& bs,
                    vector<unsigned char>& pixels_out)
{
    assert(k >= 2);
    assert(pixels_in.size() % static_cast<size_t>(k) == 0);

    if (r == 0) {
        cerr << "bs_postprocess: r=0 is invalid.\n";
        return false;
    }

    // Count ambiguous blocks dynamically in pixels_in to determine z (count of bs bits)
    size_t z = 0;
    size_t groups = pixels_in.size() / static_cast<size_t>(k);
    for (size_t g = 0; g < groups; ++g) {
        size_t base = g * static_cast<size_t>(k);
        if (is_ambiguous_block(pixels_in, base, k, r)) {
            ++z;
        }
    }

    pixels_out = pixels_in;

    int bit_pos = 0;
    int bits_read = 0;

    auto read_bit = [&]() -> int {
        if (bits_read >= static_cast<int>(z)) return -1; // no more bits
        size_t byte_idx = static_cast<size_t>(bit_pos) / 8;
        int    bit_idx  = bit_pos % 8;
        if (byte_idx >= bs.size()) return -1;
        int b = (bs[byte_idx] >> bit_idx) & 1;
        ++bit_pos;
        ++bits_read;
        return b;
    };

    for (size_t g = 0; g < groups; ++g) {
        size_t base = g * static_cast<size_t>(k);
        if (is_ambiguous_block(pixels_out, base, k, r)) {
            int bit = read_bit();
            if (bit < 0) {
                cerr << "bs_postprocess: ran out of bs bits at group " << g << "\n";
                return false;
            }
            if (bit == 1) {
                // Was a constant-block substitution: restore byte[k-1] to 0x00.
                pixels_out[base + static_cast<size_t>(k - 1)] = 0x00;
            }
            // bit == 0: genuine block, leave as-is.
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// embed_share_mlsb_bs
//
// Layout of embedded bytes:
//   [0..L-1]     share payload bytes (L = share.size())
//   [L]          r (8-bit)
//   [L+1..]      bs bytes (length = bs.size())
// ---------------------------------------------------------------------------
bool embed_share_mlsb_bs(const TiffInfo& cover_info,
                          const vector<unsigned char>& share,
                          unsigned char r,
                          const vector<unsigned char>& bs,
                          TiffInfo& out_info,
                          int t)
{
    if (r == 0) {
        cerr << "embed_share_mlsb_bs: r=0 is invalid.\n";
        return false;
    }

    // Build the full payload: share + r (1 byte) + bs.
    vector<unsigned char> payload;
    payload.reserve(share.size() + 1 + bs.size());

    payload.insert(payload.end(), share.begin(), share.end());
    payload.push_back(r);                                             // byte L: r
    payload.insert(payload.end(), bs.begin(), bs.end());

    // Embed via standard mLSB+OPA.
    return embed_share_mlsb(cover_info, payload, out_info, t);
}

// ---------------------------------------------------------------------------
// extract_share_mlsb_bs
// ---------------------------------------------------------------------------
bool extract_share_mlsb_bs(const TiffInfo& stego_info,
                             size_t total_embedded_len,
                             size_t share_len,
                             vector<unsigned char>& out_share,
                             unsigned char& out_r,
                             vector<unsigned char>& out_bs,
                             int t)
{
    if (total_embedded_len == 0) {
        cerr << "extract_share_mlsb_bs: total_embedded_len must be provided.\n";
        return false;
    }

    if (total_embedded_len < share_len + 1) {
        cerr << "extract_share_mlsb_bs: total_embedded_len too small.\n";
        return false;
    }

    // Extract raw bytes from stego image.
    vector<unsigned char> raw;
    if (!extract_share_mlsb(stego_info, total_embedded_len, raw, t))
        return false;

    if (raw.size() < share_len + 1) {
        cerr << "extract_share_mlsb_bs: extracted data too short.\n";
        return false;
    }

    // First share_len bytes are the share payload.
    out_share.assign(raw.begin(), raw.begin() + static_cast<ptrdiff_t>(share_len));

    // Next byte is r.
    out_r = raw[share_len];
    if (out_r == 0) {
        cerr << "extract_share_mlsb_bs: r=0 is invalid.\n";
        return false;
    }

    // Remaining bytes are the bs bit-stream.
    out_bs.assign(raw.begin() + static_cast<ptrdiff_t>(share_len + 1), raw.end());

    return true;
}
