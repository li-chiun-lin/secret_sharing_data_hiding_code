#include "lsb_huffman.h"
#include <cstring>
#include <iostream>
#include <queue>
#include <utility>
#include <limits>
#include <algorithm>
#include <functional>
#include <fstream>

namespace {
constexpr uint32_t kBootMagic = 0x32424D48; // HMB2
constexpr uint32_t kMainMagic = 0x324D4248; // HBM2
constexpr int kBootPixels = 128;
constexpr int kBootBits = 4;
}

static void write_bits_to_pixels(std::vector<unsigned char>& pixels, size_t pixel_start, size_t t,
    const std::vector<uint8_t>& bit_bytes, size_t bit_len)
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

static void read_bits_from_pixels(const std::vector<unsigned char>& pixels, size_t pixel_start, size_t t,
    std::vector<uint8_t>& out_bytes, size_t bit_len)
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

static void append_bits(std::vector<uint8_t>& out, size_t& out_bits, const uint8_t* src, size_t src_bits)
{
    size_t need_bytes = (out_bits + src_bits + 7) / 8;
    if (out.size() < need_bytes) out.resize(need_bytes, 0);
    for (size_t i = 0; i < src_bits; ++i) {
        uint8_t bit = static_cast<uint8_t>((src[i / 8] >> (i % 8)) & 1u);
        size_t dst_bit = out_bits + i;
        if (bit) out[dst_bit / 8] |= static_cast<uint8_t>(1u << (dst_bit % 8));
    }
    out_bits += src_bits;
}

static uint32_t read_u32_bits(const std::vector<uint8_t>& in, size_t& bit_ofs)
{
    uint32_t v = 0;
    for (int i = 0; i < 32; ++i) {
        uint8_t bit = static_cast<uint8_t>((in[(bit_ofs + i) / 8] >> ((bit_ofs + i) % 8)) & 1u);
        v |= static_cast<uint32_t>(bit) << i;
    }
    bit_ofs += 32;
    return v;
}

bool embed_share_lsb_with_huffman_inband(const TiffInfo& cover_info,
    const std::vector<unsigned char>& share,
    TiffInfo& out_info,
    HuffmanCodec& codec_out,
    int t,
    bool write_sidecar,
    const std::string& sidecar_path)
{
    if (t <= 0 || t > 8) return false;
    if (kBootBits <= 0 || kBootBits > 8) return false;

    // Bootstrap region must hold: 32-bit boot magic + 32-bit compressed_bits_len
    // + frequency table (alphabet_size * 32 bits).
    // Compute the minimum number of pixels needed at kBootBits per pixel.
    const int alphabet_size_local = 1 << std::max(0, std::min(t, 8));
    const size_t boot_need_bits_min =
        32u + 32u + static_cast<size_t>(alphabet_size_local) * 32u;
    const int kBootPixelsDyn = static_cast<int>(
        (boot_need_bits_min + static_cast<size_t>(kBootBits) - 1u)
        / static_cast<size_t>(kBootBits));

    const size_t cover_pixels = cover_info.size();
    if (cover_pixels <= static_cast<size_t>(kBootPixelsDyn)) {
        std::cerr << "Huffman-LSB: cover too small for payload\n";
        return false;
    }

    const size_t regionB_pixels = cover_pixels - static_cast<size_t>(kBootPixelsDyn);
    const size_t share_bits = share.size() * 8;
    const size_t saved_boot_bits = static_cast<size_t>(kBootPixelsDyn) * kBootBits;
    const size_t main_meta_bits = 64;
    const size_t regionB_capacity_bits = regionB_pixels * static_cast<size_t>(t);

    const int alphabet_size = 1 << std::max(0, std::min(t, 8));
    std::vector<uint32_t> freq_u32(static_cast<size_t>(alphabet_size), 0u);
    std::vector<uint8_t> freq_bytes(static_cast<size_t>(alphabet_size) * sizeof(uint32_t), 0u);

    std::vector<uint8_t> symbols;
    symbols.reserve(regionB_pixels);
    unsigned int tmask = (1u << t) - 1u;
    for (size_t pos = static_cast<size_t>(kBootPixelsDyn); pos < cover_pixels; ++pos) {
        uint8_t sym = static_cast<uint8_t>(cover_info.data[pos] & tmask);
        symbols.push_back(sym);
        if (sym < freq_u32.size()) freq_u32[sym]++;
    }

    codec_out.build_from_symbols(symbols, t);
    std::vector<uint8_t> compressed_bytes;
    size_t compressed_bits_len = 0;
    codec_out.encode_bits(symbols, compressed_bytes, compressed_bits_len);

    for (size_t i = 0; i < freq_u32.size(); ++i) {
        std::memcpy(freq_bytes.data() + i * sizeof(uint32_t), &freq_u32[i], sizeof(uint32_t));
    }

    const size_t boot_need_bits = 32 + 32 + freq_bytes.size() * 8;
    const size_t boot_capacity_bits = static_cast<size_t>(kBootPixelsDyn) * kBootBits;
    if (boot_need_bits > boot_capacity_bits) {
        std::cerr << "Huffman-LSB: bootstrap region too small for model\n";
        return false;
    }

    const size_t total_main_bits = main_meta_bits + compressed_bits_len + saved_boot_bits + share_bits;
    if (total_main_bits > regionB_capacity_bits) {
        std::cerr << "Huffman-LSB: insufficient space for in-band model + payload\n";
        return false;
    }

    std::vector<uint8_t> saved_boot_bytes((saved_boot_bits + 7) / 8, 0u);
    size_t sb_bit = 0;
    for (int i = 0; i < kBootPixelsDyn; ++i) {
        uint8_t v = static_cast<uint8_t>(cover_info.data[static_cast<size_t>(i)] & ((1u << kBootBits) - 1u));
        for (int b = 0; b < kBootBits; ++b, ++sb_bit) {
            if ((v >> b) & 1u) saved_boot_bytes[sb_bit / 8] |= static_cast<uint8_t>(1u << (sb_bit % 8));
        }
    }

    uint8_t main_meta[8];
    uint32_t mm = kMainMagic;
    uint32_t cbits_u32 = static_cast<uint32_t>(compressed_bits_len);
    std::memcpy(main_meta, &mm, 4);
    std::memcpy(main_meta + 4, &cbits_u32, 4);

    std::vector<uint8_t> main_bytes;
    size_t main_bits = 0;
    append_bits(main_bytes, main_bits, main_meta, 64);
    append_bits(main_bytes, main_bits, compressed_bytes.data(), compressed_bits_len);
    append_bits(main_bytes, main_bits, saved_boot_bytes.data(), saved_boot_bits);
    append_bits(main_bytes, main_bits, share.data(), share_bits);

    std::vector<unsigned char> modified = cover_info.data;

    uint8_t boot_prefix[8];
    uint32_t bm = kBootMagic;
    std::memcpy(boot_prefix, &bm, 4);
    std::memcpy(boot_prefix + 4, &cbits_u32, 4);
    std::vector<uint8_t> boot_bytes;
    size_t boot_bits = 0;
    append_bits(boot_bytes, boot_bits, boot_prefix, 64);
    append_bits(boot_bytes, boot_bits, freq_bytes.data(), freq_bytes.size() * 8);
    write_bits_to_pixels(modified, 0, kBootBits, boot_bytes, boot_bits);

    write_bits_to_pixels(modified, static_cast<size_t>(kBootPixelsDyn), t, main_bytes, main_bits);

    if (write_sidecar && !sidecar_path.empty()) {
        std::ofstream os(sidecar_path, std::ios::binary);
        if (os) {
            uint32_t magic = kBootMagic;
            uint32_t version = 2;
            os.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
            os.write(reinterpret_cast<const char*>(&version), sizeof(version));
            os.close();
        }
    }

    out_info = cover_info;
    out_info.data = std::move(modified);
    return true;
}

bool recover_share_lsb_with_huffman_inband(const TiffInfo& stego_info,
    const HuffmanCodec& codec_in,
    size_t share_len,
    std::vector<unsigned char>& out_share,
    TiffInfo& recovered_cover,
    int t,
    bool use_sidecar,
    const std::string& sidecar_path)
{
    (void)codec_in;
    (void)use_sidecar;
    (void)sidecar_path;

    if (t <= 0 || t > 8) return false;
    if (kBootBits <= 0 || kBootBits > 8) return false;

    // Recompute dynamic boot region size (must match embed)
    const int alphabet_size_local = 1 << std::max(0, std::min(t, 8));
    const size_t boot_need_bits_min =
        32u + 32u + static_cast<size_t>(alphabet_size_local) * 32u;
    const int kBootPixelsDyn = static_cast<int>(
        (boot_need_bits_min + static_cast<size_t>(kBootBits) - 1u)
        / static_cast<size_t>(kBootBits));

    const size_t cover_pixels = stego_info.size();
    if (cover_pixels <= static_cast<size_t>(kBootPixelsDyn)) return false;

    const size_t regionB_pixels = cover_pixels - static_cast<size_t>(kBootPixelsDyn);
    const size_t share_bits = share_len * 8;
    const size_t saved_boot_bits = static_cast<size_t>(kBootPixelsDyn) * kBootBits;
    const size_t main_meta_bits = 64;

    const size_t boot_capacity_bits = static_cast<size_t>(kBootPixelsDyn) * kBootBits;
    std::vector<uint8_t> boot_bytes;
    read_bits_from_pixels(stego_info.data, 0, kBootBits, boot_bytes, boot_capacity_bits);

    size_t bo = 0;
    uint32_t boot_magic = read_u32_bits(boot_bytes, bo);
    uint32_t compressed_bits_len_u32 = read_u32_bits(boot_bytes, bo);
    if (boot_magic != kBootMagic) {
        std::cerr << "Huffman-LSB recover: invalid boot magic\n";
        return false;
    }
    size_t compressed_bits_len = static_cast<size_t>(compressed_bits_len_u32);

    const int alphabet_size = 1 << std::max(0, std::min(t, 8));
    if (bo + static_cast<size_t>(alphabet_size) * 32 > boot_capacity_bits) {
        std::cerr << "Huffman-LSB recover: bootstrap table overflow\n";
        return false;
    }

    std::vector<uint32_t> freq_u32(static_cast<size_t>(alphabet_size), 0u);
    for (int i = 0; i < alphabet_size; ++i) freq_u32[i] = read_u32_bits(boot_bytes, bo);

    const size_t total_main_bits = main_meta_bits + compressed_bits_len + saved_boot_bits + share_bits;
    if (total_main_bits > regionB_pixels * static_cast<size_t>(t)) {
        std::cerr << "Huffman-LSB recover: invalid main stream size\n";
        return false;
    }

    std::vector<uint8_t> main_bytes;
    read_bits_from_pixels(stego_info.data, static_cast<size_t>(kBootPixelsDyn), t, main_bytes, total_main_bits);

    size_t mo = 0;
    uint32_t main_magic = read_u32_bits(main_bytes, mo);
    uint32_t compressed_bits_check = read_u32_bits(main_bytes, mo);
    if (main_magic != kMainMagic || compressed_bits_check != compressed_bits_len_u32) {
        std::cerr << "Huffman-LSB recover: invalid main metadata\n";
        return false;
    }

    std::vector<uint8_t> compressed_bytes((compressed_bits_len + 7) / 8, 0u);
    for (size_t i = 0; i < compressed_bits_len; ++i) {
        uint8_t bit = static_cast<uint8_t>((main_bytes[(mo + i) / 8] >> ((mo + i) % 8)) & 1u);
        if (bit) compressed_bytes[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
    }
    mo += compressed_bits_len;

    std::vector<uint8_t> saved_boot_bytes((saved_boot_bits + 7) / 8, 0u);
    for (size_t i = 0; i < saved_boot_bits; ++i) {
        uint8_t bit = static_cast<uint8_t>((main_bytes[(mo + i) / 8] >> ((mo + i) % 8)) & 1u);
        if (bit) saved_boot_bytes[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
    }
    mo += saved_boot_bits;

    out_share.assign(share_len, 0u);
    for (size_t i = 0; i < share_bits; ++i) {
        uint8_t bit = static_cast<uint8_t>((main_bytes[(mo + i) / 8] >> ((mo + i) % 8)) & 1u);
        if (bit) out_share[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
    }

    HuffmanCodec local_codec;
    local_codec.build_from_frequencies(freq_u32, t);
    std::vector<uint8_t> symbols;
    if (!local_codec.decode_bits(compressed_bytes, compressed_bits_len, regionB_pixels, symbols)) {
        std::cerr << "Huffman-LSB recover: decode failed\n";
        return false;
    }

    std::vector<unsigned char> recovered = stego_info.data;
    unsigned int tmask = (1u << t) - 1u;
    for (size_t idx = 0; idx < regionB_pixels; ++idx) {
        size_t pos = static_cast<size_t>(kBootPixels) + idx;
        unsigned char px = recovered[pos];
        px = static_cast<unsigned char>((px & ~tmask) | (symbols[idx] & tmask));
        recovered[pos] = px;
    }

    unsigned int bmask = (1u << kBootBits) - 1u;
    size_t sb_bit = 0;
    for (int i = 0; i < kBootPixelsDyn; ++i) {
        unsigned int v = 0;
        for (int b = 0; b < kBootBits; ++b, ++sb_bit) {
            uint8_t bit = static_cast<uint8_t>((saved_boot_bytes[sb_bit / 8] >> (sb_bit % 8)) & 1u);
            v |= static_cast<unsigned int>(bit) << b;
        }
        unsigned char px = recovered[static_cast<size_t>(i)];
        px = static_cast<unsigned char>((px & ~bmask) | (static_cast<unsigned char>(v) & bmask));
        recovered[static_cast<size_t>(i)] = px;
    }

    recovered_cover = stego_info;
    recovered_cover.data = std::move(recovered);
    return true;
}

// ----------------- HuffmanCodec implementation -----------------

void HuffmanCodec::build_from_symbols(const std::vector<uint8_t>& symbols, int alphabet_bits)
{
    alphabet_size = 1 << std::max(0, std::min(alphabet_bits, 8));
    std::vector<uint32_t> freq_u32(static_cast<size_t>(alphabet_size), 0u);
    for (uint8_t s : symbols) {
        if (s < alphabet_size) freq_u32[s]++;
    }
    build_from_frequencies(freq_u32, alphabet_bits);
}

void HuffmanCodec::build_from_frequencies(const std::vector<uint32_t>& freq_u32, int alphabet_bits)
{
    alphabet_size = 1 << std::max(0, std::min(alphabet_bits, 8));
    nodes.clear();
    nodes.reserve(alphabet_size * 2);

    std::vector<uint64_t> freq(alphabet_size, 0);
    for (int s = 0; s < alphabet_size; ++s) {
        if (static_cast<size_t>(s) < freq_u32.size()) freq[s] = freq_u32[static_cast<size_t>(s)];
    }

    std::priority_queue<std::pair<uint64_t,int>, std::vector<std::pair<uint64_t,int>>, std::greater<>> pq;
    for (int s = 0; s < alphabet_size; ++s) {
        if (freq[s] > 0) {
            Node node;
            node.left = -1; node.right = -1; node.symbol = s; node.freq = freq[s];
            nodes.push_back(node);
            pq.push({node.freq, static_cast<int>(nodes.size() - 1)});
        }
    }

    if (pq.empty()) {
        Node node;
        node.left = -1; node.right = -1; node.symbol = 0; node.freq = 1;
        nodes.push_back(node);
        pq.push({1, static_cast<int>(nodes.size() - 1)});
    }

    while (pq.size() > 1) {
        auto a = pq.top(); pq.pop();
        auto b = pq.top(); pq.pop();
        Node parent;
        parent.left = a.second;
        parent.right = b.second;
        parent.symbol = -1;
        parent.freq = a.first + b.first;
        nodes.push_back(parent);
        int parent_idx = static_cast<int>(nodes.size() - 1);
        pq.push({parent.freq, parent_idx});
    }

    root = pq.top().second;

    code.assign(alphabet_size, 0u);
    code_len.assign(alphabet_size, 0);

    std::function<void(int, uint32_t, int)> dfs = [&](int node_idx, uint32_t cur_code, int cur_len) {
        const Node& nd = nodes[node_idx];
        if (nd.symbol >= 0) {
            if (nd.symbol < alphabet_size) {
                code[nd.symbol] = cur_code;
                code_len[nd.symbol] = cur_len;
            }
            return;
        }
        if (nd.left >= 0) dfs(nd.left, cur_code, cur_len + 1);
        if (nd.right >= 0) dfs(nd.right, cur_code | (1u << cur_len), cur_len + 1);
    };

    dfs(root, 0u, 0);
}

void HuffmanCodec::encode_bits(const std::vector<uint8_t>& symbols, std::vector<uint8_t>& out_bytes, size_t& out_bit_len) const
{
    out_bytes.clear();
    out_bit_len = 0;
    if (alphabet_size == 0) return;

    size_t est_bits = symbols.size() * 4;
    out_bytes.reserve((est_bits + 7) / 8);

    uint8_t cur_byte = 0;
    int cur_bit = 0;

    auto push_bit = [&](uint8_t b) {
        if (b) cur_byte |= static_cast<uint8_t>(1u << cur_bit);
        ++cur_bit;
        ++out_bit_len;
        if (cur_bit == 8) {
            out_bytes.push_back(cur_byte);
            cur_byte = 0;
            cur_bit = 0;
        }
    };

    for (uint8_t s : symbols) {
        if (s >= code.size() || code_len[s] == 0) continue;
        uint32_t sym_code = code[s];
        int len = code_len[s];
        for (int b = 0; b < len; ++b) {
            uint8_t bit = static_cast<uint8_t>((sym_code >> b) & 1u);
            push_bit(bit);
        }
    }

    if (cur_bit != 0) out_bytes.push_back(cur_byte);
}

bool HuffmanCodec::decode_bits(const std::vector<uint8_t>& in_bytes, size_t in_bit_len, size_t decode_count, std::vector<uint8_t>& out_symbols) const
{
    out_symbols.clear();
    if (alphabet_size == 0 || root < 0) return false;

    size_t bit_idx = 0;
    int node_idx = root;

    auto read_bit = [&](bool& ok)->int {
        if (bit_idx >= in_bit_len) { ok = false; return 0; }
        size_t byte_idx = bit_idx / 8;
        int bitpos = static_cast<int>(bit_idx % 8);
        uint8_t bit = (in_bytes[byte_idx] >> bitpos) & 1u;
        ++bit_idx;
        ok = true;
        return bit;
    };

    while (out_symbols.size() < decode_count) {
        bool ok;
        int bit = read_bit(ok);
        if (!ok) return false;
        const Node& nd = nodes[node_idx];
        if (bit == 0) {
            if (nd.left >= 0) node_idx = nd.left;
            else return false;
        } else {
            if (nd.right >= 0) node_idx = nd.right;
            else return false;
        }
        const Node& cur = nodes[node_idx];
        if (cur.symbol >= 0) {
            out_symbols.push_back(static_cast<uint8_t>(cur.symbol));
            node_idx = root;
        }
    }

    return true;
}
