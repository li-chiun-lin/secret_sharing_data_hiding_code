#include "mlsb.h"
#include <iostream>
#include <cassert>
#include <fstream>

using namespace std;

// ---------------------------------------------------------------------------
// compute_min_t
// ---------------------------------------------------------------------------
int compute_min_t(size_t share_bytes, size_t cover_pixels)
{
    for (int t = 1; t <= 8; ++t) {
        size_t pixels_per_byte = (8u + static_cast<size_t>(t) - 1u)
                                 / static_cast<size_t>(t); // ceil(8/t)
        size_t capacity_bytes  = cover_pixels / pixels_per_byte;
        if (capacity_bytes >= share_bytes)
            return t;
    }
    return 9; // infeasible: share too large even at t=8
}

void set_bits(unsigned char& px, unsigned char value, int t)
{
    unsigned int mask = (1u << t) - 1u;
    px = static_cast<unsigned char>((px & ~mask) | (value & mask));
}

static inline unsigned char get_share_bit_from_bytes(const vector<unsigned char>& share, size_t bit_idx)
{
    size_t byte_idx = bit_idx / 8;
    int b = static_cast<int>(bit_idx % 8);
    return static_cast<unsigned char>((share[byte_idx] >> b) & 1u);
}

static void write_bits_to_pixels(vector<unsigned char>& pixels, size_t pixel_start, size_t t, const vector<uint8_t>& bit_bytes, size_t bit_len)
{
    unsigned int mask = (1u << t) - 1u;
    size_t bit_idx = 0;
    size_t pixel_pos = pixel_start;
    while (bit_idx < bit_len) {
        unsigned int val = 0;
        for (unsigned int b = 0; b < (unsigned)t && bit_idx < bit_len; ++b, ++bit_idx) {
            size_t byte_idx = bit_idx / 8;
            int bitpos = static_cast<int>(bit_idx % 8);
            uint8_t bit = (bit_bytes[byte_idx] >> bitpos) & 1u;
            val |= (bit << b);
        }
        unsigned char px = pixels[pixel_pos];
        px = static_cast<unsigned char>((px & ~mask) | (val & mask));
        pixels[pixel_pos] = px;
        ++pixel_pos;
    }
}

static void read_bits_from_pixels(const vector<unsigned char>& pixels, size_t pixel_start, size_t t, vector<uint8_t>& out_bytes, size_t bit_len)
{
    out_bytes.clear();
    size_t bytes_needed = (bit_len + 7) / 8;
    out_bytes.assign(bytes_needed, 0);
    size_t bit_idx = 0;
    size_t pixel_pos = pixel_start;
    while (bit_idx < bit_len) {
        unsigned int val = static_cast<unsigned int>(pixels[pixel_pos] & ((1u << t) - 1u));
        for (unsigned int b = 0; b < (unsigned)t && bit_idx < bit_len; ++b, ++bit_idx) {
            size_t byte_idx = bit_idx / 8;
            int bitpos = static_cast<int>(bit_idx % 8);
            if ((val >> b) & 1u) out_bytes[byte_idx] |= static_cast<uint8_t>(1u << bitpos);
        }
        ++pixel_pos;
    }
}

bool embed_share_mlsb(const TiffInfo& cover_info, const vector<unsigned char>& share, TiffInfo& out_info, int t)
{
    // Resolve t=0 to minimal feasible t
    if (t == 0) {
        t = compute_min_t(share.size(), cover_info.size());
    }

    if (t > 8) {
        cerr << "mLSB embed: share too large for any t in [1..8]. "
             << "share=" << share.size() << " bytes, "
             << "cover=" << cover_info.size() << " pixels. "
             << "Increase k to reduce share size.\n";
        return false;
    }
    if (t <= 0) return false;
    const size_t bits_per_byte = 8;
    const size_t pixels_per_byte = (bits_per_byte + t - 1) / t; // ceil(8/t)
    size_t required_pixels = share.size() * pixels_per_byte;

    size_t cover_pixels = cover_info.size();
    if (cover_pixels < required_pixels) {
        cerr << "Cover image too small for mLSB embedding: need " << required_pixels << " pixels\n";
        return false;
    }

    vector<unsigned char> modified = cover_info.data;
    size_t pos = 0;
    unsigned int mask = (1u << t) - 1u;

    for (size_t i = 0; i < share.size(); ++i) {
        unsigned char byte = share[i];
        int bits_done = 0;
        while (bits_done < 8) {
            unsigned char chunk = static_cast<unsigned char>((byte >> bits_done) & mask);
            unsigned char orig  = cover_info.data[pos];

            // Step 1: substitute t LSBs with payload bits
            unsigned char stego = static_cast<unsigned char>((orig & ~mask) | chunk);

            // Step 2: Optimal Pixel Adjustment (OPA)
            // Flipping bit t (the next higher bit) may reduce |stego - orig|.
            // Maximum error without OPA: 2^t - 1.
            // Maximum error with OPA:    2^(t-1).
            // Only flip if it strictly reduces the error and stays in [0,255].
            if (t < 8) {
                unsigned int flip_mask = 1u << t;
                unsigned char stego_flipped = static_cast<unsigned char>(stego ^ flip_mask);
                int err_before  = abs(static_cast<int>(stego)         - static_cast<int>(orig));
                int err_flipped = abs(static_cast<int>(stego_flipped) - static_cast<int>(orig));
                if (err_flipped < err_before) {
                    stego = stego_flipped;
                }
            }

            modified[pos] = stego;
            ++pos;
            bits_done += t;
        }
    }

    out_info = cover_info;
    out_info.data = std::move(modified);
    return true;
}

bool extract_share_mlsb(const TiffInfo& stego_info, size_t share_len, vector<unsigned char>& out_share, int t, bool write_file, const string& out_path)
{
    // Resolve t=0 to minimal feasible t
    if (t == 0) {
        t = compute_min_t(share_len, stego_info.size());
    }

    if (t > 8) {
        cerr << "mLSB extract: share too large for any t in [1..8]. "
             << "share=" << share_len << " bytes, "
             << "stego=" << stego_info.size() << " pixels.\n";
        return false;
    }
    if (t <= 0) return false;
    const size_t bits_per_byte = 8;
    const size_t pixels_per_byte = (bits_per_byte + t - 1) / t; // ceil(8/t)
    size_t required_pixels = share_len * pixels_per_byte;

    size_t cover_pixels = stego_info.size();
    if (cover_pixels < required_pixels) {
        cerr << "Stego image too small for mLSB extraction: need " << required_pixels << " pixels\n";
        return false;
    }

    out_share.clear();
    out_share.reserve(share_len);

    size_t pos = 0;
    unsigned int mask = (1u << t) - 1u;

    for (size_t i = 0; i < share_len; ++i) {
        unsigned char byte = 0;
        int bits_done = 0;
        int shift = 0;
        while (bits_done < 8) {
            unsigned char chunk = static_cast<unsigned char>(stego_info.data[pos] & mask);
            byte |= static_cast<unsigned char>(chunk << shift);
            ++pos;
            bits_done += t;
            shift += t;
        }
        out_share.push_back(byte);
    }

    if (write_file && !out_path.empty()) {
        std::ofstream os(out_path, std::ios::binary);
        if (!os) {
            cerr << "extract_share_mlsb: cannot open " << out_path << " for writing\n";
            return false;
        }
        os.write(reinterpret_cast<const char*>(out_share.data()), static_cast<std::streamsize>(out_share.size()));
        os.close();
    }

    return true;
}