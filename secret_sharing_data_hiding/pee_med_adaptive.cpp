#include "pee_med_adaptive.h"
#include <vector>
#include <queue>
#include <functional>
#include <algorithm>
#include <cstdint>
#include <iostream>

using namespace std;

namespace {
constexpr int kMBits = 20;
constexpr int kLsbBitsField = 4; // store lsb_bits in [1,8]

static inline int med_predict(const vector<unsigned char>& img, uint32_t W, uint32_t H, uint32_t row, uint32_t col)
{
    (void)H;
    bool have_left = (col > 0);
    bool have_top  = (row > 0);
    if (!have_left && !have_top) return 128;
    int a = have_left ? static_cast<int>(img[row * W + col - 1]) : 0;
    int b = have_top  ? static_cast<int>(img[(row - 1) * W + col]) : 0;
    int c = (have_left && have_top) ? static_cast<int>(img[(row - 1) * W + col - 1]) : 0;
    if (!have_left) return b;
    if (!have_top) return a;
    int lo = min(a, b), hi = max(a, b);
    if (c >= hi) return lo;
    if (c <= lo) return hi;
    return a + b - c;
}

struct HuffNode {
    int symbol = -1;
    int left = -1;
    int right = -1;
    uint64_t freq = 0;
};

struct DynHuff {
    int alphabet = 0;
    int sym_bits = 0;
    vector<HuffNode> nodes;
    int root = -1;
    vector<uint32_t> code;
    vector<int> codelen;

    void init(int alpha, int bits)
    {
        alphabet = alpha;
        sym_bits = bits;
        code.assign(alpha, 0);
        codelen.assign(alpha, 0);
    }

    void build(const vector<int>& freq)
    {
        nodes.clear();
        nodes.reserve(static_cast<size_t>(alphabet * 2));
        using P = pair<uint64_t, int>;
        priority_queue<P, vector<P>, greater<P>> pq;
        for (int s = 0; s < alphabet; ++s) {
            HuffNode n;
            n.symbol = s;
            n.freq = static_cast<uint64_t>(freq[s] > 0 ? freq[s] : 1);
            nodes.push_back(n);
            pq.push({n.freq, static_cast<int>(nodes.size()) - 1});
        }
        while (pq.size() > 1) {
            auto [fa, ia] = pq.top(); pq.pop();
            auto [fb, ib] = pq.top(); pq.pop();
            HuffNode p;
            p.left = ia;
            p.right = ib;
            p.freq = fa + fb;
            nodes.push_back(p);
            pq.push({p.freq, static_cast<int>(nodes.size()) - 1});
        }
        root = pq.top().second;
        function<void(int, uint32_t, int)> dfs = [&](int idx, uint32_t cur, int len) {
            const HuffNode& nd = nodes[idx];
            if (nd.symbol >= 0) {
                code[static_cast<size_t>(nd.symbol)] = cur;
                codelen[static_cast<size_t>(nd.symbol)] = (len == 0 ? 1 : len);
                return;
            }
            if (nd.left >= 0) dfs(nd.left, cur, len + 1);
            if (nd.right >= 0) dfs(nd.right, cur | (1u << len), len + 1);
        };
        dfs(root, 0u, 0);
    }

    size_t topo_bits() const { return static_cast<size_t>(2 * alphabet - 1); }
    size_t vals_bits() const { return static_cast<size_t>(alphabet * sym_bits); }

    void serialize(vector<uint8_t>& buf, size_t& bit_pos) const
    {
        function<void(int)> write_topo = [&](int idx) {
            const HuffNode& nd = nodes[idx];
            bool internal = (nd.symbol < 0);
            if (internal) buf[bit_pos / 8] |= static_cast<uint8_t>(1u << (bit_pos % 8));
            ++bit_pos;
            if (internal) {
                if (nd.left >= 0) write_topo(nd.left);
                if (nd.right >= 0) write_topo(nd.right);
            }
        };
        function<void(int)> write_vals = [&](int idx) {
            const HuffNode& nd = nodes[idx];
            if (nd.symbol >= 0) {
                for (int b = 0; b < sym_bits; ++b) {
                    if ((nd.symbol >> b) & 1) buf[bit_pos / 8] |= static_cast<uint8_t>(1u << (bit_pos % 8));
                    ++bit_pos;
                }
                return;
            }
            if (nd.left >= 0) write_vals(nd.left);
            if (nd.right >= 0) write_vals(nd.right);
        };
        write_topo(root);
        write_vals(root);
    }

    bool deserialize(const vector<uint8_t>& buf, size_t& bit_pos, size_t buf_bits)
    {
        nodes.clear();
        nodes.reserve(static_cast<size_t>(alphabet * 2));
        auto read_bit = [&]() -> int {
            if (bit_pos >= buf_bits) return -1;
            int b = (buf[bit_pos / 8] >> (bit_pos % 8)) & 1;
            ++bit_pos;
            return b;
        };
        function<int()> read_node = [&]() -> int {
            int internal = read_bit();
            if (internal < 0) return -1;
            HuffNode nd;
            if (internal) {
                nd.left = read_node();
                nd.right = read_node();
            }
            nodes.push_back(nd);
            return static_cast<int>(nodes.size()) - 1;
        };
        root = read_node();
        if (root < 0) return false;
        function<bool(int)> read_vals = [&](int idx) -> bool {
            HuffNode& nd = nodes[idx];
            if (nd.left < 0 && nd.right < 0) {
                int v = 0;
                for (int b = 0; b < sym_bits; ++b) {
                    int bit = read_bit();
                    if (bit < 0) return false;
                    v |= (bit << b);
                }
                nd.symbol = v;
                return true;
            }
            if (nd.left >= 0 && !read_vals(nd.left)) return false;
            if (nd.right >= 0 && !read_vals(nd.right)) return false;
            return true;
        };
        if (!read_vals(root)) return false;

        code.assign(alphabet, 0);
        codelen.assign(alphabet, 0);
        function<void(int, uint32_t, int)> dfs = [&](int idx, uint32_t cur, int len) {
            const HuffNode& nd = nodes[idx];
            if (nd.symbol >= 0) {
                code[static_cast<size_t>(nd.symbol)] = cur;
                codelen[static_cast<size_t>(nd.symbol)] = (len == 0 ? 1 : len);
                return;
            }
            if (nd.left >= 0) dfs(nd.left, cur, len + 1);
            if (nd.right >= 0) dfs(nd.right, cur | (1u << len), len + 1);
        };
        dfs(root, 0u, 0);
        return true;
    }

    int decode_one(const vector<uint8_t>& buf, size_t& bit_pos, size_t buf_bits) const
    {
        int idx = root;
        while (true) {
            const HuffNode& nd = nodes[idx];
            if (nd.symbol >= 0) return nd.symbol;
            if (bit_pos >= buf_bits) return -1;
            int bit = (buf[bit_pos / 8] >> (bit_pos % 8)) & 1;
            ++bit_pos;
            idx = bit ? nd.right : nd.left;
            if (idx < 0) return -1;
        }
    }
};

static void write_bits(vector<uint8_t>& buf, size_t& pos, uint32_t val, int nbits)
{
    size_t need = (pos + static_cast<size_t>(nbits) + 7) / 8;
    if (buf.size() < need) buf.resize(need, 0);
    for (int i = 0; i < nbits; ++i) {
        if ((val >> i) & 1u) buf[pos / 8] |= static_cast<uint8_t>(1u << (pos % 8));
        ++pos;
    }
}

static uint32_t read_bits(const vector<uint8_t>& buf, size_t& pos, int nbits)
{
    uint32_t v = 0;
    for (int i = 0; i < nbits; ++i) {
        if ((buf[pos / 8] >> (pos % 8)) & 1u) v |= (1u << i);
        ++pos;
    }
    return v;
}
} // namespace

bool embed_share_pee_med_adaptive(const TiffInfo& cover_info,
    const vector<unsigned char>& share,
    TiffInfo& out_info,
    int lsb_bits)
{
    const uint32_t W = cover_info.width;
    const uint32_t H = cover_info.height;
    const size_t N = cover_info.size();
    if (N == 0 || W == 0 || H == 0) return false;
    if (lsb_bits < 1 || lsb_bits > 8) return false;
    if (N > ((1u << kMBits) - 1u)) return false;

    const int alphabet = 1 << lsb_bits;
    const uint32_t mask = static_cast<uint32_t>(alphabet - 1);
    DynHuff tree;
    tree.init(alphabet, lsb_bits);

    const size_t tree_bits = tree.topo_bits() + tree.vals_bits();
    const size_t header_bits = static_cast<size_t>(kMBits + kLsbBitsField) + tree_bits;
    const size_t R = (header_bits + static_cast<size_t>(lsb_bits) - 1) / static_cast<size_t>(lsb_bits);
    if (R >= N) return false;

    vector<uint8_t> pe_all(N, 0);
    for (size_t pos = R; pos < N; ++pos) {
        uint32_t row = static_cast<uint32_t>(pos / W), col = static_cast<uint32_t>(pos % W);
        int p = med_predict(cover_info.data, W, H, row, col);
        int x = static_cast<int>(cover_info.data[pos]);
        pe_all[pos] = static_cast<uint8_t>((p ^ x) & static_cast<int>(mask));
    }

    vector<int> freq(static_cast<size_t>(alphabet), 0);
    for (size_t pos = R; pos < N; ++pos) freq[pe_all[pos]]++;
    tree.build(freq);

    const size_t share_bits = share.size() * 8;
    size_t pe_bits = 0;
    size_t M = R;
    for (size_t pos = R; pos < N; ++pos) {
        pe_bits += static_cast<size_t>(tree.codelen[pe_all[pos]]);
        M = pos + 1;
        size_t cap = M * static_cast<size_t>(lsb_bits);
        size_t payload = header_bits + pe_bits + share_bits + R * static_cast<size_t>(lsb_bits);
        if (cap >= payload) break;
    }

    {
        size_t cap = M * static_cast<size_t>(lsb_bits);
        size_t payload = header_bits + pe_bits + share_bits + R * static_cast<size_t>(lsb_bits);
        if (cap < payload) {
            cerr << "PEE-MED-adaptive embed: insufficient capacity\n";
            return false;
        }
    }

    size_t total_bits = M * static_cast<size_t>(lsb_bits);
    vector<uint8_t> stream((total_bits + 7) / 8, 0);
    size_t bit_pos = 0;
    write_bits(stream, bit_pos, static_cast<uint32_t>(M), kMBits);
    write_bits(stream, bit_pos, static_cast<uint32_t>(lsb_bits), kLsbBitsField);
    tree.serialize(stream, bit_pos);
    for (size_t pos = R; pos < M; ++pos) {
        uint8_t sym = pe_all[pos];
        write_bits(stream, bit_pos, tree.code[sym], tree.codelen[sym]);
    }
    for (size_t i = 0; i < share_bits; ++i) {
        uint32_t b = (share[i / 8] >> (i % 8)) & 1u;
        write_bits(stream, bit_pos, b, 1);
    }
    for (size_t i = 0; i < R; ++i) {
        uint32_t lsb = static_cast<uint32_t>(cover_info.data[i]) & mask;
        write_bits(stream, bit_pos, lsb, lsb_bits);
    }

    vector<unsigned char> modified = cover_info.data;
    size_t read_pos = 0;
    for (size_t pos = 0; pos < M; ++pos) {
        uint32_t v = read_bits(stream, read_pos, lsb_bits);
        modified[pos] = static_cast<unsigned char>((static_cast<uint32_t>(cover_info.data[pos]) & ~mask) | (v & mask));
    }

    out_info = cover_info;
    out_info.data = std::move(modified);
    return true;
}

bool recover_share_pee_med_adaptive(const TiffInfo& stego_info,
    size_t share_len,
    vector<unsigned char>& out_share,
    TiffInfo& recovered_cover)
{
    const uint32_t W = stego_info.width;
    const uint32_t H = stego_info.height;
    const size_t N = stego_info.size();
    if (N == 0 || W == 0 || H == 0) return false;

    // Read M and lsb_bits from 1-bit stream view first.
    size_t sb = 0;
    auto read_stream_bit = [&]() -> int {
        size_t pixel_idx = sb / 8; // temporarily assume worst-case nibble; replaced after lsb_bits known
        (void)pixel_idx;
        return 0;
    };
    (void)read_stream_bit;

    // To decode adaptive-width stream, read M and lsb_bits from pixel LSBs directly with width discovery.
    // M field is always first 20 bits; lsb_bits field next 4 bits, both packed in unknown width.
    // We brute-force lsb_bits in [1,8] and pick one where decoded field is self-consistent.
    int lsb_bits = -1;
    uint32_t M = 0;
    for (int b = 1; b <= 8; ++b) {
        uint32_t mask = static_cast<uint32_t>((1u << b) - 1u);
        size_t pos = 0;
        size_t bit_pos = 0;
        auto get_bit = [&](size_t idx) -> int {
            size_t p = idx / static_cast<size_t>(b);
            int off = static_cast<int>(idx % static_cast<size_t>(b));
            if (p >= N) return 0;
            return (stego_info.data[p] >> off) & 1;
        };
        uint32_t candM = 0;
        for (int i = 0; i < kMBits; ++i) candM |= static_cast<uint32_t>(get_bit(bit_pos++)) << i;
        uint32_t candB = 0;
        for (int i = 0; i < kLsbBitsField; ++i) candB |= static_cast<uint32_t>(get_bit(bit_pos++)) << i;
        if (candB == static_cast<uint32_t>(b) && candM >= 1 && candM <= N) {
            lsb_bits = b;
            M = candM;
            break;
        }
        (void)pos;
    }
    if (lsb_bits < 1 || lsb_bits > 8 || M == 0 || M > N) {
        cerr << "PEE-MED-adaptive recover: cannot determine lsb_bits/M\n";
        return false;
    }

    const int alphabet = 1 << lsb_bits;
    const uint32_t mask = static_cast<uint32_t>(alphabet - 1);
    DynHuff tree;
    tree.init(alphabet, lsb_bits);
    const size_t tree_bits = tree.topo_bits() + tree.vals_bits();
    const size_t header_bits = static_cast<size_t>(kMBits + kLsbBitsField) + tree_bits;
    const size_t R = (header_bits + static_cast<size_t>(lsb_bits) - 1) / static_cast<size_t>(lsb_bits);
    if (R >= M) return false;

    // Build full stream buffer from first M pixels.
    const size_t stream_bits = static_cast<size_t>(M) * static_cast<size_t>(lsb_bits);
    vector<uint8_t> buf((stream_bits + 7) / 8, 0);
    size_t wp = 0;
    for (size_t p = 0; p < M; ++p) {
        uint32_t v = static_cast<uint32_t>(stego_info.data[p]) & mask;
        for (int i = 0; i < lsb_bits; ++i) {
            if ((v >> i) & 1u) buf[wp / 8] |= static_cast<uint8_t>(1u << (wp % 8));
            ++wp;
        }
    }

    size_t rp = 0;
    uint32_t M2 = read_bits(buf, rp, kMBits);
    uint32_t b2 = read_bits(buf, rp, kLsbBitsField);
    if (M2 != M || b2 != static_cast<uint32_t>(lsb_bits)) return false;

    vector<uint8_t> tree_buf((tree_bits + 7) / 8, 0);
    for (size_t i = 0; i < tree_bits; ++i) {
        int bit = (buf[rp / 8] >> (rp % 8)) & 1;
        if (bit) tree_buf[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
        ++rp;
    }
    size_t tp = 0;
    if (!tree.deserialize(tree_buf, tp, tree_bits)) return false;

    const size_t remaining_bits = stream_bits - header_bits;
    vector<uint8_t> data_buf((remaining_bits + 7) / 8, 0);
    for (size_t i = 0; i < remaining_bits; ++i) {
        int bit = (buf[rp / 8] >> (rp % 8)) & 1;
        if (bit) data_buf[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
        ++rp;
    }

    size_t n_pe = static_cast<size_t>(M) - R;
    vector<uint8_t> pe_vals(n_pe, 0);
    size_t dp = 0;
    for (size_t i = 0; i < n_pe; ++i) {
        int sym = tree.decode_one(data_buf, dp, remaining_bits);
        if (sym < 0) return false;
        pe_vals[i] = static_cast<uint8_t>(sym);
    }

    const size_t share_bits = share_len * 8;
    out_share.assign(share_len, 0);
    for (size_t i = 0; i < share_bits; ++i) {
        if (dp >= remaining_bits) return false;
        int bit = (data_buf[dp / 8] >> (dp % 8)) & 1;
        ++dp;
        if (bit) out_share[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
    }

    vector<uint32_t> ref_lsb(R, 0);
    for (size_t i = 0; i < R; ++i) {
        uint32_t v = 0;
        for (int b = 0; b < lsb_bits; ++b) {
            if (dp >= remaining_bits) return false;
            int bit = (data_buf[dp / 8] >> (dp % 8)) & 1;
            ++dp;
            if (bit) v |= (1u << b);
        }
        ref_lsb[i] = v;
    }

    vector<unsigned char> recovered = stego_info.data;
    for (size_t i = 0; i < R; ++i) {
        recovered[i] = static_cast<unsigned char>((static_cast<uint32_t>(stego_info.data[i]) & ~mask) | ref_lsb[i]);
    }
    for (size_t i = 0; i < n_pe; ++i) {
        size_t pos = R + i;
        uint32_t row = static_cast<uint32_t>(pos / W), col = static_cast<uint32_t>(pos % W);
        int p = med_predict(recovered, W, H, row, col);
        uint32_t orig_lsb = static_cast<uint32_t>((p ^ pe_vals[i]) & static_cast<int>(mask));
        recovered[pos] = static_cast<unsigned char>((static_cast<uint32_t>(stego_info.data[pos]) & ~mask) | orig_lsb);
    }

    recovered_cover = stego_info;
    recovered_cover.data = std::move(recovered);
    return true;
}
