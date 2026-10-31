#include "pee_med.h"
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <queue>
#include <vector>
#include <functional>

using namespace std;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int kAlphabetSize = 16;   // pe in [0, 15]
static constexpr int kMBits        = 20;   // bits used to store M
static constexpr int kTreeTopoBits = 31;   // 31 nodes: 1 bit each (internal/leaf)
static constexpr int kTreeValBits  = 64;   // 16 leaves * 4 bits each
static constexpr int kTreeBits     = kTreeTopoBits + kTreeValBits; // 95
static constexpr int kHeaderBits   = kMBits + kTreeBits;           // 115
static constexpr int kR            = kPeeMedR;                     // 29

// ---------------------------------------------------------------------------
// MED predictor (operates on original cover values)
// ---------------------------------------------------------------------------
static inline int med_predict(const vector<unsigned char>& img,
    uint32_t W, uint32_t H,
    uint32_t row, uint32_t col)
{
    (void)H;
    const bool have_left = (col > 0);
    const bool have_top  = (row > 0);

    if (!have_left && !have_top) return 128;

    const int a = have_left ? static_cast<int>(img[row * W + col - 1])           : 0;
    const int b = have_top  ? static_cast<int>(img[(row - 1) * W + col])         : 0;
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

// ---------------------------------------------------------------------------
// Huffman codec (alphabet [0,15])
// ---------------------------------------------------------------------------
struct HuffNode {
    int symbol = -1;   // -1 for internal nodes
    int left   = -1;
    int right  = -1;
    uint64_t freq = 0;
};

struct HuffTree {
    vector<HuffNode> nodes;
    int root = -1;
    // code table
    uint32_t code[kAlphabetSize]  = {};
    int      codelen[kAlphabetSize] = {};

    void build(const int freq[kAlphabetSize]) {
        nodes.clear();
        nodes.reserve(kAlphabetSize * 2);

        // priority queue: (freq, node_index), min-heap
        using P = pair<uint64_t, int>;
        priority_queue<P, vector<P>, greater<P>> pq;

        for (int s = 0; s < kAlphabetSize; ++s) {
            HuffNode n;
            n.symbol = s;
            n.freq   = static_cast<uint64_t>(freq[s] > 0 ? freq[s] : 1);
            nodes.push_back(n);
            pq.push({n.freq, static_cast<int>(nodes.size()) - 1});
        }

        while (pq.size() > 1) {
            auto [fa, ia] = pq.top(); pq.pop();
            auto [fb, ib] = pq.top(); pq.pop();
            HuffNode parent;
            parent.left  = ia;
            parent.right = ib;
            parent.freq  = fa + fb;
            nodes.push_back(parent);
            int pi = static_cast<int>(nodes.size()) - 1;
            pq.push({parent.freq, pi});
        }
        root = pq.top().second;

        // build code table via DFS
        function<void(int, uint32_t, int)> dfs = [&](int idx, uint32_t cur, int len) {
            const HuffNode& nd = nodes[idx];
            if (nd.symbol >= 0) {
                code[nd.symbol]    = cur;
                codelen[nd.symbol] = len;
                return;
            }
            if (nd.left  >= 0) dfs(nd.left,  cur,          len + 1);
            if (nd.right >= 0) dfs(nd.right, cur | (1u << len), len + 1);
        };
        dfs(root, 0u, 0);
    }

    // Serialize tree into a bit buffer (LSB-first packing).
    // Format: DFS pre-order, 1 bit per node (1=internal, 0=leaf),
    //         then 4 bits per leaf value in the same DFS order.
    // Total: 31 topology bits + 16*4 = 64 value bits = 95 bits.
    void serialize(vector<uint8_t>& buf, size_t& bit_pos) const {
        // topology bits
        function<void(int)> write_topo = [&](int idx) {
            const HuffNode& nd = nodes[idx];
            bool is_internal = (nd.symbol < 0);
            // write 1 bit
            if (is_internal) buf[bit_pos / 8] |= static_cast<uint8_t>(1u << (bit_pos % 8));
            ++bit_pos;
            if (is_internal) {
                if (nd.left  >= 0) write_topo(nd.left);
                if (nd.right >= 0) write_topo(nd.right);
            }
        };
        write_topo(root);

        // leaf values in same DFS order
        function<void(int)> write_vals = [&](int idx) {
            const HuffNode& nd = nodes[idx];
            if (nd.symbol >= 0) {
                // write 4 bits (LSB-first)
                for (int b = 0; b < 4; ++b) {
                    if ((nd.symbol >> b) & 1)
                        buf[bit_pos / 8] |= static_cast<uint8_t>(1u << (bit_pos % 8));
                    ++bit_pos;
                }
                return;
            }
            if (nd.left  >= 0) write_vals(nd.left);
            if (nd.right >= 0) write_vals(nd.right);
        };
        write_vals(root);
    }

    // Deserialize from bit buffer. Returns true on success.
    bool deserialize(const vector<uint8_t>& buf, size_t& bit_pos, size_t buf_bits) {
        nodes.clear();
        nodes.reserve(kAlphabetSize * 2);

        auto read_bit = [&]() -> int {
            if (bit_pos >= buf_bits) return -1;
            int b = (buf[bit_pos / 8] >> (bit_pos % 8)) & 1;
            ++bit_pos;
            return b;
        };

        // Reconstruct tree from topology bits
        function<int()> read_node = [&]() -> int {
            int is_internal = read_bit();
            if (is_internal < 0) return -1;
            HuffNode nd;
            if (is_internal) {
                nd.left  = read_node();
                nd.right = read_node();
            }
            nodes.push_back(nd);
            return static_cast<int>(nodes.size()) - 1;
        };
        root = read_node();
        if (root < 0) return false;

        // Read leaf values
        function<bool(int)> read_vals = [&](int idx) -> bool {
            HuffNode& nd = nodes[idx];
            if (nd.left < 0 && nd.right < 0) {
                // leaf: read 4 bits
                int val = 0;
                for (int b = 0; b < 4; ++b) {
                    int bit = read_bit();
                    if (bit < 0) return false;
                    val |= (bit << b);
                }
                nd.symbol = val;
                return true;
            }
            if (nd.left  >= 0 && !read_vals(nd.left))  return false;
            if (nd.right >= 0 && !read_vals(nd.right)) return false;
            return true;
        };
        if (!read_vals(root)) return false;

        // Rebuild code table
        function<void(int, uint32_t, int)> dfs = [&](int idx, uint32_t cur, int len) {
            const HuffNode& nd = nodes[idx];
            if (nd.symbol >= 0) {
                code[nd.symbol]    = cur;
                codelen[nd.symbol] = len;
                return;
            }
            if (nd.left  >= 0) dfs(nd.left,  cur,          len + 1);
            if (nd.right >= 0) dfs(nd.right, cur | (1u << len), len + 1);
        };
        dfs(root, 0u, 0);
        return true;
    }

    // Decode one symbol from bit stream. Returns symbol or -1 on error.
    int decode_one(const vector<uint8_t>& buf, size_t& bit_pos, size_t buf_bits) const {
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

// ---------------------------------------------------------------------------
// Bit buffer helpers
// ---------------------------------------------------------------------------
static void write_bits(vector<uint8_t>& buf, size_t& pos, uint32_t val, int nbits) {
    size_t need = (pos + nbits + 7) / 8;
    if (buf.size() < need) buf.resize(need, 0);
    for (int i = 0; i < nbits; ++i) {
        if ((val >> i) & 1u)
            buf[pos / 8] |= static_cast<uint8_t>(1u << (pos % 8));
        ++pos;
    }
}

static uint32_t read_bits(const vector<uint8_t>& buf, size_t& pos, int nbits) {
    uint32_t val = 0;
    for (int i = 0; i < nbits; ++i) {
        if ((buf[pos / 8] >> (pos % 8)) & 1u)
            val |= (1u << i);
        ++pos;
    }
    return val;
}

// ---------------------------------------------------------------------------
// embed_share_pee_med
// ---------------------------------------------------------------------------
bool embed_share_pee_med(const TiffInfo& cover_info,
    const vector<unsigned char>& share,
    TiffInfo& out_info)
{
    const uint32_t W = cover_info.width;
    const uint32_t H = cover_info.height;
    const size_t N   = cover_info.size();

    if (N == 0 || W == 0 || H == 0) return false;
    if (N < static_cast<size_t>(kR)) {
        cerr << "PEE-MED embed: image too small\n";
        return false;
    }

    const size_t share_bits = share.size() * 8;

    // --- Step 1: compute pe for all non-header pixels ---
    // pe[i] = (MED_predict(i) XOR pixel[i]) & 0xF
    // Header pixels (0..kR-1) are excluded from pe computation.
    vector<uint8_t> pe_all(N, 0);
    for (size_t pos = static_cast<size_t>(kR); pos < N; ++pos) {
        uint32_t row = static_cast<uint32_t>(pos / W);
        uint32_t col = static_cast<uint32_t>(pos % W);
        int p = med_predict(cover_info.data, W, H, row, col);
        int x = static_cast<int>(cover_info.data[pos]);
        pe_all[pos] = static_cast<uint8_t>((p ^ x) & 0xF);
    }

    // --- Step 2: build Huffman tree from pe frequency ---
    int freq[kAlphabetSize] = {};
    for (size_t pos = static_cast<size_t>(kR); pos < N; ++pos)
        freq[pe_all[pos]]++;

    HuffTree tree;
    tree.build(freq);

    // --- Step 3: find M (smallest number of pixels to hold everything) ---
    // Payload = kHeaderBits + sum(codelen[pe[i]] for i=kR..M-1) + share_bits + kR*4
    // Capacity = M * 4
    // Find smallest M >= kR such that capacity >= payload.

    size_t pe_encoded_bits = 0;
    size_t M = static_cast<size_t>(kR);

    // Accumulate pe code lengths until capacity >= payload
    for (size_t pos = static_cast<size_t>(kR); pos < N; ++pos) {
        pe_encoded_bits += static_cast<size_t>(tree.codelen[pe_all[pos]]);
        M = pos + 1;
        size_t capacity = M * 4;
        size_t payload  = static_cast<size_t>(kHeaderBits)
                        + pe_encoded_bits
                        + share_bits
                        + static_cast<size_t>(kR) * 4;
        if (capacity >= payload) break;
    }

    // Final check
    {
        size_t capacity = M * 4;
        size_t payload  = static_cast<size_t>(kHeaderBits)
                        + pe_encoded_bits
                        + share_bits
                        + static_cast<size_t>(kR) * 4;
        if (capacity < payload) {
            cerr << "PEE-MED embed: insufficient capacity (need "
                 << payload << " bits, have " << capacity << ")\n";
            return false;
        }
    }

    if (M > (1u << kMBits) - 1u) {
        cerr << "PEE-MED embed: M=" << M << " exceeds " << kMBits << "-bit field\n";
        return false;
    }

    // --- Step 4: build the full bit stream ---
    // Total bits = M * 4 (we pad with zeros at the end if needed)
    size_t total_bits = M * 4;
    vector<uint8_t> stream((total_bits + 7) / 8, 0);
    size_t bit_pos = 0;

    // [M: 20 bits]
    write_bits(stream, bit_pos, static_cast<uint32_t>(M), kMBits);

    // [Huffman tree: 95 bits]
    tree.serialize(stream, bit_pos);
    assert(bit_pos == static_cast<size_t>(kHeaderBits));

    // [Huffman-encoded pe of pixels kR..M-1]
    for (size_t pos = static_cast<size_t>(kR); pos < M; ++pos) {
        uint8_t sym = pe_all[pos];
        write_bits(stream, bit_pos, tree.code[sym], tree.codelen[sym]);
    }

    // [share data: share_bits bits]
    for (size_t i = 0; i < share_bits; ++i) {
        uint32_t bit = (share[i / 8] >> (i % 8)) & 1u;
        write_bits(stream, bit_pos, bit, 1);
    }

    // [original LSB4 of pixels 0..kR-1: kR*4 bits]
    for (int i = 0; i < kR; ++i) {
        uint32_t lsb4 = static_cast<uint32_t>(cover_info.data[i]) & 0xFu;
        write_bits(stream, bit_pos, lsb4, 4);
    }

    // bit_pos should now be <= total_bits (remainder is zero-padding)

    // --- Step 5: write stream into LSB4 of pixels 0..M-1 ---
    vector<unsigned char> modified = cover_info.data;
    size_t read_pos = 0;
    for (size_t pos = 0; pos < M; ++pos) {
        uint32_t nibble = read_bits(stream, read_pos, 4);
        modified[pos] = static_cast<unsigned char>(
            (static_cast<unsigned int>(cover_info.data[pos]) & 0xF0u) | nibble);
    }

    out_info = cover_info;
    out_info.data = std::move(modified);
    return true;
}

// ---------------------------------------------------------------------------
// recover_share_pee_med
// ---------------------------------------------------------------------------
bool recover_share_pee_med(const TiffInfo& stego_info,
    size_t share_len,
    vector<unsigned char>& out_share,
    TiffInfo& recovered_cover)
{
    const uint32_t W = stego_info.width;
    const uint32_t H = stego_info.height;
    const size_t N   = stego_info.size();

    if (N == 0 || W == 0 || H == 0) return false;
    if (N < static_cast<size_t>(kR)) {
        cerr << "PEE-MED recover: image too small\n";
        return false;
    }

    // --- Step 1: extract the full bit stream from LSB4 of all pixels ---
    // We don't know M yet, so we read lazily from the stego image.
    // Helper: read 4 bits from pixel pos.
    auto get_nibble = [&](size_t pos) -> uint8_t {
        return static_cast<uint8_t>(stego_info.data[pos] & 0xFu);
    };

    // We'll build a virtual bit stream reader over the stego LSB4s.
    // bit_pos tracks position within the stream (4 bits per pixel).
    size_t stream_bit = 0; // current bit position in the virtual stream

    auto read_stream_bit = [&]() -> int {
        size_t pixel_idx = stream_bit / 4;
        int    bit_in_nibble = static_cast<int>(stream_bit % 4);
        if (pixel_idx >= N) return -1;
        ++stream_bit;
        return (get_nibble(pixel_idx) >> bit_in_nibble) & 1;
    };

    auto read_stream_bits = [&](int nbits) -> uint32_t {
        uint32_t val = 0;
        for (int i = 0; i < nbits; ++i) {
            int b = read_stream_bit();
            if (b < 0) return val;
            val |= static_cast<uint32_t>(b) << i;
        }
        return val;
    };

    // --- Step 2: read M ---
    uint32_t M = read_stream_bits(kMBits);
    if (M < static_cast<uint32_t>(kR) || M > N) {
        cerr << "PEE-MED recover: invalid M=" << M << "\n";
        return false;
    }

    // --- Step 3: read and deserialize Huffman tree ---
    // Tree is kTreeBits = 95 bits starting at bit kMBits = 20.
    // We need to feed these bits into the deserializer.
    // Build a temporary byte buffer for the tree bits.
    vector<uint8_t> tree_buf((kTreeBits + 7) / 8, 0);
    size_t tree_buf_pos = 0;
    for (int i = 0; i < kTreeBits; ++i) {
        int b = read_stream_bit();
        if (b < 0) { cerr << "PEE-MED recover: truncated tree\n"; return false; }
        if (b) tree_buf[tree_buf_pos / 8] |= static_cast<uint8_t>(1u << (tree_buf_pos % 8));
        ++tree_buf_pos;
    }

    HuffTree tree;
    size_t tree_parse_pos = 0;
    if (!tree.deserialize(tree_buf, tree_parse_pos, kTreeBits)) {
        cerr << "PEE-MED recover: tree deserialization failed\n";
        return false;
    }

    // --- Step 4: decode M-kR pe values ---
    size_t n_pe = static_cast<size_t>(M) - static_cast<size_t>(kR);
    vector<uint8_t> pe_vals(n_pe);

    // Build a byte buffer from the remaining stream bits for Huffman decoding.
    // We read enough bits to cover the pe stream + share + REF.
    // Upper bound: (M*4 - kHeaderBits) bits remain.
    size_t remaining_bits = static_cast<size_t>(M) * 4 - static_cast<size_t>(kHeaderBits);
    vector<uint8_t> data_buf((remaining_bits + 7) / 8, 0);
    size_t data_buf_pos = 0;
    for (size_t i = 0; i < remaining_bits; ++i) {
        int b = read_stream_bit();
        if (b < 0) break;
        if (b) data_buf[data_buf_pos / 8] |= static_cast<uint8_t>(1u << (data_buf_pos % 8));
        ++data_buf_pos;
    }

    size_t data_pos = 0;
    for (size_t i = 0; i < n_pe; ++i) {
        int sym = tree.decode_one(data_buf, data_pos, remaining_bits);
        if (sym < 0) {
            cerr << "PEE-MED recover: Huffman decode failed at pe " << i << "\n";
            return false;
        }
        pe_vals[i] = static_cast<uint8_t>(sym);
    }

    // --- Step 5: extract share data ---
    const size_t share_bits = share_len * 8;
    out_share.assign(share_len, 0);
    for (size_t i = 0; i < share_bits; ++i) {
        if (data_pos >= remaining_bits) {
            cerr << "PEE-MED recover: truncated share\n";
            return false;
        }
        int b = (data_buf[data_pos / 8] >> (data_pos % 8)) & 1;
        ++data_pos;
        if (b) out_share[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
    }

    // --- Step 6: extract original LSB4 of header pixels ---
    vector<uint8_t> ref_lsb4(kR);
    for (int i = 0; i < kR; ++i) {
        if (data_pos + 4 > remaining_bits) {
            cerr << "PEE-MED recover: truncated REF\n";
            return false;
        }
        uint8_t nibble = 0;
        for (int b = 0; b < 4; ++b) {
            int bit = (data_buf[data_pos / 8] >> (data_pos % 8)) & 1;
            ++data_pos;
            if (bit) nibble |= static_cast<uint8_t>(1u << b);
        }
        ref_lsb4[i] = nibble;
    }

    // --- Step 7: restore the cover image ---
    vector<unsigned char> recovered = stego_info.data;

    // Restore header pixels (0..kR-1): replace LSB4 with saved original
    for (int i = 0; i < kR; ++i) {
        recovered[i] = static_cast<unsigned char>(
            (static_cast<unsigned int>(stego_info.data[i]) & 0xF0u) | ref_lsb4[i]);
    }

    // Restore non-header embedded pixels (kR..M-1):
    // original_lsb4 = (MED_predict XOR pe) & 0xF
    // original_pixel = (stego_pixel & 0xF0) | original_lsb4
    for (size_t i = 0; i < n_pe; ++i) {
        size_t pos = static_cast<size_t>(kR) + i;
        uint32_t row = static_cast<uint32_t>(pos / W);
        uint32_t col = static_cast<uint32_t>(pos % W);
        // Predict from already-recovered neighbors
        int p = med_predict(recovered, W, H, row, col);
        uint8_t orig_lsb4 = static_cast<uint8_t>((p ^ pe_vals[i]) & 0xF);
        recovered[pos] = static_cast<unsigned char>(
            (static_cast<unsigned int>(stego_info.data[pos]) & 0xF0u) | orig_lsb4);
    }

    // Pixels M..N-1 are untouched (stego == original)

    recovered_cover = stego_info;
    recovered_cover.data = std::move(recovered);
    return true;
}

bool analyze_pee_med_capacity(const TiffInfo& cover_info,
    int lsb_bits,
    PeeMedCapacityReport& out_report)
{
    const uint32_t W = cover_info.width;
    const uint32_t H = cover_info.height;
    const size_t N   = cover_info.size();

    if (N == 0 || W == 0 || H == 0) return false;
    if (lsb_bits <= 0 || lsb_bits > 8) return false;

    const int alphabet_size = 1 << lsb_bits;
    const uint32_t lsb_mask = static_cast<uint32_t>(alphabet_size - 1);

    // Generalized header sizing:
    // M field remains 20 bits.
    // Full binary tree with L leaves uses (2L-1) topology bits and L*lsb_bits value bits.
    const size_t tree_topo_bits = static_cast<size_t>(2 * alphabet_size - 1);
    const size_t tree_val_bits  = static_cast<size_t>(alphabet_size * lsb_bits);
    const size_t header_bits    = static_cast<size_t>(kMBits) + tree_topo_bits + tree_val_bits;
    const size_t R = (header_bits + static_cast<size_t>(lsb_bits) - 1) / static_cast<size_t>(lsb_bits);

    if (R >= N) return false;

    vector<uint8_t> pe_all(N, 0);
    for (size_t pos = R; pos < N; ++pos) {
        uint32_t row = static_cast<uint32_t>(pos / W);
        uint32_t col = static_cast<uint32_t>(pos % W);
        int p = med_predict(cover_info.data, W, H, row, col);
        int x = static_cast<int>(cover_info.data[pos]);
        pe_all[pos] = static_cast<uint8_t>((p ^ x) & static_cast<int>(lsb_mask));
    }

    // Dynamic Huffman build for alphabet [0, 2^lsb_bits - 1].
    struct DynNode {
        int symbol = -1;
        int left = -1;
        int right = -1;
        uint64_t freq = 0;
    };
    vector<DynNode> nodes;
    nodes.reserve(static_cast<size_t>(alphabet_size * 2));
    vector<int> codelen(static_cast<size_t>(alphabet_size), 0);

    int freq[256] = {};
    for (size_t pos = R; pos < N; ++pos) {
        freq[pe_all[pos]]++;
    }

    using PQItem = pair<uint64_t, int>;
    priority_queue<PQItem, vector<PQItem>, greater<PQItem>> pq;

    for (int s = 0; s < alphabet_size; ++s) {
        DynNode nd;
        nd.symbol = s;
        nd.freq = static_cast<uint64_t>(freq[s] > 0 ? freq[s] : 1);
        nodes.push_back(nd);
        pq.push({nd.freq, static_cast<int>(nodes.size()) - 1});
    }

    while (pq.size() > 1) {
        auto [fa, ia] = pq.top(); pq.pop();
        auto [fb, ib] = pq.top(); pq.pop();
        DynNode parent;
        parent.left = ia;
        parent.right = ib;
        parent.freq = fa + fb;
        nodes.push_back(parent);
        int pi = static_cast<int>(nodes.size()) - 1;
        pq.push({parent.freq, pi});
    }
    if (pq.empty()) return false;
    const int root = pq.top().second;

    function<void(int, int)> dfs_len = [&](int idx, int len) {
        const DynNode& nd = nodes[idx];
        if (nd.symbol >= 0) {
            codelen[static_cast<size_t>(nd.symbol)] = max(1, len);
            return;
        }
        if (nd.left  >= 0) dfs_len(nd.left,  len + 1);
        if (nd.right >= 0) dfs_len(nd.right, len + 1);
    };
    dfs_len(root, 0);

    bool feasible_any = false;
    size_t best_secret_bits = 0;
    size_t best_M = R;
    size_t best_pe_bits = 0;
    size_t pe_prefix_bits = 0;
    const size_t ref_bits = R * static_cast<size_t>(lsb_bits);

    for (size_t pos = R; pos < N; ++pos) {
        pe_prefix_bits += static_cast<size_t>(codelen[pe_all[pos]]);
        const size_t M = pos + 1;
        const size_t capacity_bits = M * static_cast<size_t>(lsb_bits);
        const size_t payload_without_secret = header_bits + pe_prefix_bits + ref_bits;
        if (capacity_bits < payload_without_secret) continue;
        feasible_any = true;
        const size_t secret_bits = capacity_bits - payload_without_secret;
        if (secret_bits >= best_secret_bits) {
            best_secret_bits = secret_bits;
            best_M = M;
            best_pe_bits = pe_prefix_bits;
        }
    }

    if (!feasible_any) {
        best_M = N;
        best_pe_bits = 0;
        for (size_t pos = R; pos < N; ++pos) {
            best_pe_bits += static_cast<size_t>(codelen[pe_all[pos]]);
        }
    }

    out_report.feasible = feasible_any;
    out_report.lsb_bits = lsb_bits;
    out_report.image_pixels = N;
    out_report.chosen_M = best_M;
    out_report.overall_capacity_bits = best_M * static_cast<size_t>(lsb_bits);
    out_report.full_image_capacity_bits = N * static_cast<size_t>(lsb_bits);
    out_report.header_bits = header_bits;
    out_report.ref_bits = ref_bits;
    out_report.compressed_lsb_bits = best_pe_bits;
    out_report.usable_secret_bits = best_secret_bits;
    out_report.usable_secret_bytes = best_secret_bits / 8;
    out_report.header_pixels_R = R;
    return true;
}
