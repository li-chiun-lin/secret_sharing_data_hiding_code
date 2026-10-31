#include "../tiff_io/tiff_io.h"
#include "../finite_field/binary_field.h"

#include "lsb_huffman.h"
#include "mlsb.h"
#include "mlsb_bs.h"
#include "mlsb_mixed.h"
#include "mlsb_all.h"
#include "lsb_matching.h"
#include "lsb_matching_revisited.h"
#include "pee_med.h"
#include "pee_med_adaptive.h"
#include "pee_diamond.h"
#include "pee_multi.h"
#include "pee_checker_safe.h"
#include "histogram_shifting.h"
#include "pvo.h"
#include "pvo_inband.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <vector>
#include <cassert>
#include <string>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>

using namespace std;

// -------------------- Global dataset / path settings --------------------
// Modify these global settings to adapt to different secret and cover image sources/output directories
static const int g_image_size = 512;
static const string g_input_dir = "../images/USC-SIPI/" + to_string(g_image_size) + "/";
static const string g_output_dir = "../out/" + to_string(g_image_size);
static const string g_secret_name = "woman.tiff";
static const vector<string> g_cover_names = {
    "jetplane.tiff",
    "house.tiff",
    "girl.tiff",
    "baboon.tiff",
    "airplane.tiff",
    "boat.tiff",
    "bridge.tiff",
    "car.tiff",
    "lake.tiff",
    "milkdrop.tiff",
    "peppers.tiff",
    "tiffany.tiff"
};
// ------------------------------------------------------------------------

// helper: get basename from path (filename only)
static string get_basename(const string& path)
{
    size_t p = path.find_last_of("/\\");
    if (p == string::npos) return path;
    return path.substr(p + 1);
}

// generate n cover file paths (rotating selection from g_cover_names)
static vector<string> generate_cover_paths(int n)
{
    vector<string> covers;
    covers.reserve(n);
    int m = static_cast<int>(g_cover_names.size());
    for (int i = 0; i < n; ++i) {
        string cname = g_cover_names[i % m];
        covers.push_back(g_input_dir + cname);
    }
    return covers;
}

// produce output base path for a given module, index and cover file name (no extra extension)
static string make_output_base(const string& module, int index, const string& cover_name)
{
    return g_output_dir + "/" + module + "_" + to_string(index) + "_" + cover_name;
}

// ---------- Secret sharing (Shamir over GF(256)) ----------
//
// Revised scheme:
//   - Each block uses (k-1) secret pixels as coefficients a0..a(k-2).
//   - A random non-zero byte is appended as the leading coefficient a(k-1),
//     making the polynomial degree exactly (k-1) and ensuring it is never
//     the zero polynomial even when all secret pixels in the block are 0.
//   - If N is not divisible by (k-1), the secret is padded with random bytes
//     to the next multiple of (k-1). The padding is discarded after recovery.
//   - Reconstruction requires k shares (threshold = k) because the
//     polynomial now has degree k-1 (k coefficients).
//   - Share size = ceil(N / (k-1)) bytes per share.
//
// PRNG for padding and leading coefficient: xorshift64 seeded with a fixed
// value. Both sender and receiver use the same seed (it is not secret; the
// security comes from the secret sharing scheme itself).

static inline uint64_t ss_xorshift64(uint64_t& s)
{
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return s;
}

// Returns a non-zero random byte.
static inline unsigned char ss_rand_nonzero(uint64_t& rng)
{
    unsigned char v = 0;
    while (v == 0) v = static_cast<unsigned char>(ss_xorshift64(rng) & 0xFF);
    return v;
}

static vector<vector<unsigned char>> generate_shares(const TiffInfo& secret, int k, int n,
    bool use_nonzero_leading = true)
{
    // When use_nonzero_leading=true (default):
    //   Each block uses (k-1) secret pixels as coefficients a0..a(k-2),
    //   plus one random non-zero byte as the leading coefficient a(k-1).
    //   Polynomial degree = k-1, threshold = k.
    //   Share size = ceil(N/(k-1)) bytes.
    //
    // When use_nonzero_leading=false:
    //   Each block uses k secret pixels as coefficients a0..a(k-1).
    //   Polynomial degree = k-1, threshold = k (standard Shamir).
    //   Share size = ceil(N/k) bytes.
    //   All-zero blocks produce all-zero shares (security caveat).
    assert(k > 0 && n > 0 && k <= n);
    assert(k >= 2);

    const size_t N = secret.size();
    const int    block_secrets = use_nonzero_leading ? k - 1 : k;

    // Pad secret to a multiple of block_secrets
    size_t padded_N = ((N + static_cast<size_t>(block_secrets) - 1)
                       / static_cast<size_t>(block_secrets))
                      * static_cast<size_t>(block_secrets);
    size_t groups = padded_N / static_cast<size_t>(block_secrets);

    uint64_t rng = 0xDEADBEEFCAFEBABEULL;
    vector<unsigned char> padded = secret.data;
    while (padded.size() < padded_N)
        padded.push_back(static_cast<unsigned char>(ss_xorshift64(rng) & 0xFF));

    vector<vector<unsigned char>> shares(n);
    for (int i = 0; i < n; ++i) shares[i].reserve(groups);

    for (size_t g = 0; g < groups; ++g) {
        vector<unsigned char> coef;
        coef.reserve(k);
        size_t baseIdx = g * static_cast<size_t>(block_secrets);
        for (int j = 0; j < block_secrets; ++j)
            coef.push_back(padded[baseIdx + j]);
        if (use_nonzero_leading)
            coef.push_back(ss_rand_nonzero(rng));  // random non-zero leading coef

        for (int x = 1; x <= n; ++x) {
            unsigned char xv = static_cast<unsigned char>((((x - 1) + g) % 255) + 1);
            shares[x - 1].push_back(poly_eval(coef, xv));
        }
    }
    return shares;
}

static vector<unsigned char> reconstruct_secret(const vector<vector<unsigned char>>& shares, int k, size_t secret_size,
    const vector<int>& share_indices, bool use_nonzero_leading = true)
{
    assert(k >= 2 && static_cast<int>(shares.size()) >= k);

    const int block_secrets = use_nonzero_leading ? k - 1 : k;
    size_t groups = shares[0].size();

    vector<int> indices = share_indices;
    if (indices.empty()) {
        indices.resize(k);
        for (int i = 0; i < k; ++i) {
            indices[i] = i + 1;
        }
    }

    vector<unsigned char> result;
    result.reserve(groups * static_cast<size_t>(block_secrets));

    for (size_t g = 0; g < groups; ++g) {
        vector<unsigned char> xs, ys;
        xs.reserve(k);
        ys.reserve(k);
        for (int i = 0; i < k; ++i) {
            int participant_id = indices[i];
            unsigned char xv = static_cast<unsigned char>((((participant_id - 1) + g) % 255) + 1);
            xs.push_back(xv);
            ys.push_back(shares[i][g]);
        }
        vector<unsigned char> coef(k);
        vector<unsigned char> numer, tmp, factor;
        lagrange_interpolate(xs, ys, coef, numer, tmp, factor);

        // Recover secret pixels; discard leading coef if use_nonzero_leading
        for (int j = 0; j < block_secrets; ++j)
            result.push_back(coef[j]);
    }

    result.resize(secret_size);
    return result;
}

static vector<unsigned char> reconstruct_secret(const vector<vector<unsigned char>>& shares, int k, size_t secret_size,
    bool use_nonzero_leading = true)
{
    return reconstruct_secret(shares, k, secret_size, {}, use_nonzero_leading);
}


// ---------------- Per-module test functions (use globals) -----------------
// write_file determines whether to actually write stego/sidecar to disk (default false for fast testing)

static void test_mLSB(int k, int n, bool write_file = false, int mlsb_t = 0)
{
    if (k <= 0 || n <= 0 || k > n) { cerr << "test_mLSB: invalid k/n\n"; return; }

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Secret image validation failed\n"; return;
    }

    auto covers = generate_cover_paths(n);
    auto shares = generate_shares(secret, k, n);
    size_t share_len = shares[0].size();

    // Resolve t for display (use first cover as representative)
    int display_t = mlsb_t;
    if (display_t == 0) {
        TiffInfo first_cover = readTiff(covers[0]);
        display_t = compute_min_t(share_len, first_cover.size());
    }

    cout << "---- TEST: mLSB k=" << k << " n=" << n << " t=" << display_t
         << " (auto=" << (mlsb_t == 0 ? "YES" : "NO") << ")"
         << " write_file=" << (write_file ? "YES" : "NO")
         << " share_bytes=" << share_len << " ----\n";
    cout << "Method,Index,ImageName,MSE,PSNR(dB),ShareEqual,CoverIdentical,RecCoverMSE,RecCoverPSNR\n";

    for (int i = 0; i < n; ++i) {
        TiffInfo cover_orig = readTiff(covers[i]);
        if (!validate_gray(cover_orig, covers[i], cover_orig.width, cover_orig.height)) {
            cerr << "Cover validation failed: " << covers[i] << "\n"; continue;
        }
        string cname = get_basename(covers[i]);
        string out_base = make_output_base("stego_mlsb", i, cname);

        TiffInfo stego;
        bool ok = embed_share_mlsb(cover_orig, shares[i], stego, mlsb_t);
        if (!ok) { cout << "mLSB," << i << "," << cname << ",-,-,NO_EMBED,-,-,-\n"; continue; }

        if (write_file) writeTiff(stego, out_base + ".tiff");

        double mse = compute_mse(cover_orig.data, stego.data);
        double psnr = compute_psnr(cover_orig.data, stego.data);

        vector<unsigned char> extracted;
        bool ex_ok = extract_share_mlsb(stego, share_len, extracted, mlsb_t, /*write_file=*/false, /*out_path=*/"");
        bool share_equal = ex_ok && (extracted == shares[i]);
        bool cover_identical = false;

        cout << "mLSB" << "," << i << "," << cname << "," << mse << "," << psnr << ","
             << (share_equal ? "YES" : "NO") << "," << (cover_identical ? "YES" : "NO")
             << ",-,-\n";
    }
}

static int find_k_for_target_t_star(const TiffInfo& secret,
                                      const TiffInfo& cover,
                                      int target_t_star,
                                      int max_k = 16)
{
    if (target_t_star < 2 || target_t_star > 8) {
        return -1;
    }

    for (int k = 2; k <= max_k; ++k) {
        vector<unsigned char> padded_secret = secret.data;
        size_t padded_N = ((padded_secret.size() + static_cast<size_t>(k) - 1)
                           / static_cast<size_t>(k))
                          * static_cast<size_t>(k);
        uint64_t rng_pad = 0xDEADBEEFCAFEBABEULL;
        while (padded_secret.size() < padded_N) {
            rng_pad ^= rng_pad << 13; rng_pad ^= rng_pad >> 7; rng_pad ^= rng_pad << 17;
            padded_secret.push_back(static_cast<unsigned char>(rng_pad & 0xFF));
        }

        unsigned char r = 0;
        size_t z = 0;
        vector<unsigned char> bs, processed_secret;
        if (!bs_preprocess(padded_secret, k, r, z, bs, processed_secret)) {
            continue;
        }

        TiffInfo processed_info = secret;
        processed_info.data = processed_secret;
        auto shares = generate_shares(processed_info, k, k, /*use_nonzero_leading=*/false);
        if (shares.empty()) {
            continue;
        }

        size_t share_len = shares[0].size();
        size_t bs_bytes = (z + 7) / 8;
        size_t total_embed = 1 + bs_bytes + share_len;
        int min_t_star = compute_min_t(total_embed, cover.size());
        if (min_t_star == target_t_star) {
            return k;
        }
    }

    return -1;
}

static void generate_visual_attack_figure(const string& cover_name,
                                          const vector<int>& k_values)
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "generate_visual_attack_figure: secret validation failed\n";
        return;
    }

    string cover_path = g_input_dir + cover_name;
    TiffInfo cover = readTiff(cover_path);
    if (!validate_gray(cover, cover_path, cover.width, cover.height)) {
        cerr << "generate_visual_attack_figure: cover validation failed: " << cover_path << "\n";
        return;
    }

    std::filesystem::path out_dir = std::filesystem::current_path();
    out_dir /= "doc";
    out_dir /= "latex";
    out_dir /= "image";
    std::filesystem::create_directories(out_dir);

    for (int k : k_values) {
        vector<unsigned char> padded_secret = secret.data;
        size_t padded_N = ((padded_secret.size() + static_cast<size_t>(k) - 1)
                           / static_cast<size_t>(k))
                          * static_cast<size_t>(k);
        uint64_t rng_pad = 0xDEADBEEFCAFEBABEULL;
        while (padded_secret.size() < padded_N) {
            rng_pad ^= rng_pad << 13; rng_pad ^= rng_pad >> 7; rng_pad ^= rng_pad << 17;
            padded_secret.push_back(static_cast<unsigned char>(rng_pad & 0xFF));
        }

        unsigned char r = 0;
        size_t z = 0;
        vector<unsigned char> bs, processed_secret;
        if (!bs_preprocess(padded_secret, k, r, z, bs, processed_secret)) {
            cerr << "generate_visual_attack_figure: bs_preprocess failed for k=" << k << "\n";
            continue;
        }

        TiffInfo processed_info = secret;
        processed_info.data = processed_secret;
        auto shares = generate_shares(processed_info, k, k, /*use_nonzero_leading=*/false);
        if (shares.empty()) {
            cerr << "generate_visual_attack_figure: share generation failed for k=" << k << "\n";
            continue;
        }

        TiffInfo stego;
        bool ok = embed_share_mlsb_bs_mixed(cover, shares[0], r, bs, z, stego);
        if (!ok) {
            cerr << "generate_visual_attack_figure: mixed embedding failed for k=" << k << "\n";
            continue;
        }

        std::filesystem::path out_path = out_dir;
        out_path /= "jetplane_stego_k" + to_string(k) + "_mixed.tiff";
        writeTiff(stego, out_path.string());
        if (!std::filesystem::exists(out_path)) {
            cerr << "generate_visual_attack_figure: failed to write output TIFF: "
                 << out_path << "\n";
            continue;
        }
        size_t P = cover.size() - 6;
        size_t L = 8 * (shares[0].size() + 1) + z;
        size_t t_star = (L + P - 1) / P;
        long long diff = static_cast<long long>(L) - static_cast<long long>(P) * (t_star - 1);
        size_t n_t_star = static_cast<size_t>(max(0LL, diff));
        cout << "Generated visual attack stego image: " << out_path
             << " (k=" << k << ", t*=" << t_star
             << ", n_t*=" << n_t_star << ")\n";
    }
}

static void test_LSB_Matching(int k, int n, bool write_file = false, uint64_t seed = 0)
{
    if (k <= 0 || n <= 0 || k > n) { cerr << "test_LSB_Matching: invalid k/n\n"; return; }

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Secret image validation failed\n"; return;
    }

    auto covers = generate_cover_paths(n);
    auto shares = generate_shares(secret, k, n);
    size_t share_len = shares[0].size();

    cout << "---- TEST: LSB-Matching k=" << k << " n=" << n
         << " write_file=" << (write_file ? "YES" : "NO")
         << " share_bytes=" << share_len << " ----\n";
    cout << "Method,Index,ImageName,MSE,PSNR(dB),ShareEqual,CoverIdentical,RecCoverMSE,RecCoverPSNR\n";

    for (int i = 0; i < n; ++i) {
        TiffInfo cover_orig = readTiff(covers[i]);
        if (!validate_gray(cover_orig, covers[i], cover_orig.width, cover_orig.height)) {
            cerr << "Cover validation failed: " << covers[i] << "\n"; continue;
        }
        string cname = get_basename(covers[i]);
        string out_base = make_output_base("stego_lsb_matching", i, cname);

        TiffInfo stego;
        bool ok = embed_share_lsb_matching(cover_orig, shares[i], stego, seed);
        if (!ok) { cout << "LSB-Matching," << i << "," << cname << ",-,-,NO_EMBED,-,-,-\n"; continue; }

        if (write_file) writeTiff(stego, out_base + ".tiff");

        double mse  = compute_mse(cover_orig.data, stego.data);
        double psnr = compute_psnr(cover_orig.data, stego.data);

        vector<unsigned char> extracted;
        bool ex_ok = extract_share_lsb_matching(stego, share_len, extracted);
        bool share_equal = ex_ok && (extracted == shares[i]);

        cout << "LSB-Matching," << i << "," << cname << "," << mse << "," << psnr << ","
             << (share_equal ? "YES" : "NO") << ",NO,-,-\n";
    }
}

static void test_LSBMR(int k, int n, bool write_file = false, uint64_t seed = 0)
{
    if (k <= 0 || n <= 0 || k > n) { cerr << "test_LSBMR: invalid k/n\n"; return; }

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Secret image validation failed\n"; return;
    }

    auto covers = generate_cover_paths(n);
    auto shares = generate_shares(secret, k, n);
    size_t share_len = shares[0].size();

    cout << "---- TEST: LSBMR k=" << k << " n=" << n
         << " write_file=" << (write_file ? "YES" : "NO")
         << " share_bytes=" << share_len << " ----\n";
    cout << "Method,Index,ImageName,MSE,PSNR(dB),ShareEqual,CoverIdentical,RecCoverMSE,RecCoverPSNR\n";

    for (int i = 0; i < n; ++i) {
        TiffInfo cover_orig = readTiff(covers[i]);
        if (!validate_gray(cover_orig, covers[i], cover_orig.width, cover_orig.height)) {
            cerr << "Cover validation failed: " << covers[i] << "\n"; continue;
        }
        string cname = get_basename(covers[i]);
        string out_base = make_output_base("stego_lsbmr", i, cname);

        TiffInfo stego;
        bool ok = embed_share_lsbmr(cover_orig, shares[i], stego, seed);
        if (!ok) { cout << "LSBMR," << i << "," << cname << ",-,-,NO_EMBED,-,-,-\n"; continue; }

        if (write_file) writeTiff(stego, out_base + ".tiff");

        double mse  = compute_mse(cover_orig.data, stego.data);
        double psnr = compute_psnr(cover_orig.data, stego.data);

        vector<unsigned char> extracted;
        bool ex_ok = extract_share_lsbmr(stego, share_len, extracted);
        bool share_equal = ex_ok && (extracted == shares[i]);

        cout << "LSBMR," << i << "," << cname << "," << mse << "," << psnr << ","
             << (share_equal ? "YES" : "NO") << ",NO,-,-\n";
    }
}

static void test_LSB_HUFF(int k, int n, bool write_file = false, int mlsb_t = 2)
{
    if (k <= 0 || n <= 0 || k > n) { cerr << "test_LSB_HUFF: invalid k/n\n"; return; }

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Secret image validation failed\n"; return;
    }

    auto covers = generate_cover_paths(n);
    auto shares = generate_shares(secret, k, n);
    size_t share_len = shares[0].size();

    cout << "---- TEST: LSB_HUFF k=" << k << " n=" << n << " t=" << mlsb_t
         << " write_file=" << (write_file ? "YES" : "NO")
         << " share_bytes=" << share_len << " ----\n";
    cout << "Method,Index,ImageName,MSE,PSNR(dB),ShareEqual,CoverIdentical,RecCoverMSE,RecCoverPSNR\n";

    for (int i = 0; i < n; ++i) {
        TiffInfo cover_orig = readTiff(covers[i]);
        if (!validate_gray(cover_orig, covers[i], cover_orig.width, cover_orig.height)) {
            cerr << "Cover validation failed: " << covers[i] << "\n"; continue;
        }
        string cname = get_basename(covers[i]);
        string out_base = make_output_base("stego_LSB_HUFF", i, cname);
        string sidecar_path = out_base + ".sidecar";

        TiffInfo stego;
        HuffmanCodec codec;
        bool ok = embed_share_lsb_with_huffman_inband(cover_orig, shares[i], stego, codec, mlsb_t, /*write_sidecar=*/write_file, /*sidecar_path=*/sidecar_path);
        if (!ok) { cout << "LSB_HUFF," << i << "," << cname << ",-,-,NO_EMBED,-,-,-\n"; continue; }

        if (write_file) writeTiff(stego, out_base + ".huff.tiff");

        double mse = compute_mse(cover_orig.data, stego.data);
        double psnr = compute_psnr(cover_orig.data, stego.data);

        vector<unsigned char> rec_share;
        TiffInfo rec_cover;
        bool rec_ok = recover_share_lsb_with_huffman_inband(stego, codec, share_len, rec_share, rec_cover, mlsb_t, /*use_sidecar=*/write_file, /*sidecar_path=*/sidecar_path);

        bool share_equal = false;
        bool cover_identical = false;
        double rec_mse = -1.0;
        double rec_psnr = -1.0;
        if (rec_ok) {
            share_equal = (rec_share == shares[i]);
            cover_identical = (rec_cover.data == cover_orig.data);
            if (!cover_identical) {
                rec_mse = compute_mse(cover_orig.data, rec_cover.data);
                rec_psnr = compute_psnr(cover_orig.data, rec_cover.data);
            }
        } else {
            cerr << "LSB_HUFF recovery failed for index " << i << "\n";
        }

        cout << "LSB_HUFF" << "," << i << "," << cname << "," << mse << "," << psnr << ","
             << (share_equal ? "YES" : "NO") << "," << (cover_identical ? "YES" : "NO") << ",";
        if (rec_mse < 0) cout << "-" << ","; else cout << rec_mse << ",";
        if (rec_psnr < 0) cout << "-" << "\n"; else cout << rec_psnr << "\n";
    }
}

static void test_PVO(int k, int n, bool write_file = false)
{
    if (k <= 0 || n <= 0 || k > n) { cerr << "test_PVO: invalid k/n\n"; return; }

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Secret image validation failed\n"; return;
    }

    auto covers = generate_cover_paths(n);
    auto shares = generate_shares(secret, k, n);
    size_t share_len = shares[0].size();

    cout << "---- TEST: PVO k=" << k << " n=" << n
         << " write_file=" << (write_file ? "YES" : "NO")
         << " share_bytes=" << share_len << " ----\n";
    cout << "Method,Index,ImageName,MSE,PSNR(dB),ShareEqual,CoverIdentical,RecCoverMSE,RecCoverPSNR\n";

    for (int i = 0; i < n; ++i) {
        TiffInfo cover_orig = readTiff(covers[i]);
        if (!validate_gray(cover_orig, covers[i], cover_orig.width, cover_orig.height)) {
            cerr << "Cover validation failed: " << covers[i] << "\n"; continue;
        }
        string cname = get_basename(covers[i]);
        string out_base = make_output_base("stego_pvo", i, cname);

        TiffInfo stego;
        vector<uint8_t> used_map;
        vector<uint8_t> perm_map;
        bool ok = embed_share_pvo_sequential(cover_orig, shares[i], stego, used_map, perm_map);
        if (!ok) { cout << "PVO," << i << "," << cname << ",-,-,NO_EMBED,-,-,-\n"; continue; }

        if (write_file) writeTiff(stego, out_base + ".tiff");

        double mse = compute_mse(cover_orig.data, stego.data);
        double psnr = compute_psnr(cover_orig.data, stego.data);

        vector<unsigned char> rec_share;
        TiffInfo rec_cover;
        bool rec_ok = recover_share_pvo_sequential(stego, used_map, perm_map, share_len, rec_share, rec_cover);

        bool share_equal = false;
        bool cover_identical = false;
        double rec_mse = -1.0;
        double rec_psnr = -1.0;
        if (rec_ok) {
            share_equal = (rec_share == shares[i]);
            cover_identical = (rec_cover.data == cover_orig.data);
            if (!cover_identical) {
                rec_mse = compute_mse(cover_orig.data, rec_cover.data);
                rec_psnr = compute_psnr(cover_orig.data, rec_cover.data);
            }
        } else {
            cerr << "PVO recovery failed for index " << i << "\n";
        }

        cout << "PVO" << "," << i << "," << cname << "," << mse << "," << psnr << ","
             << (share_equal ? "YES" : "NO") << "," << (cover_identical ? "YES" : "NO") << ",";
        if (rec_mse < 0) cout << "-" << ","; else cout << rec_mse << ",";
        if (rec_psnr < 0) cout << "-" << "\n"; else cout << rec_psnr << "\n";
    }
}

static void test_PEE_MED(int k, int n, bool write_file = false)
{
    if (k <= 0 || n <= 0 || k > n) { cerr << "test_PEE_MED: invalid k/n\n"; return; }

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Secret image validation failed\n"; return;
    }

    auto covers = generate_cover_paths(n);
    auto shares = generate_shares(secret, k, n);
    size_t share_len = shares[0].size();

    cout << "---- TEST: PEE-MED k=" << k << " n=" << n
         << " write_file=" << (write_file ? "YES" : "NO")
         << " share_bytes=" << share_len << " ----\n";
    cout << "Method,Index,ImageName,MSE,PSNR(dB),ShareEqual,CoverIdentical,RecCoverMSE,RecCoverPSNR\n";

    for (int i = 0; i < n; ++i) {
        TiffInfo cover_orig = readTiff(covers[i]);
        if (!validate_gray(cover_orig, covers[i], cover_orig.width, cover_orig.height)) {
            cerr << "Cover validation failed: " << covers[i] << "\n"; continue;
        }
        string cname = get_basename(covers[i]);
        string out_base = make_output_base("stego_pee_med", i, cname);

        TiffInfo stego;
        bool ok = embed_share_pee_med(cover_orig, shares[i], stego);
        if (!ok) { cout << "PEE-MED," << i << "," << cname << ",-,-,NO_EMBED,-,-,-\n"; continue; }

        if (write_file) writeTiff(stego, out_base + ".tiff");

        double mse  = compute_mse(cover_orig.data, stego.data);
        double psnr = compute_psnr(cover_orig.data, stego.data);

        vector<unsigned char> rec_share;
        TiffInfo rec_cover;
        bool rec_ok = recover_share_pee_med(stego, share_len, rec_share, rec_cover);

        bool share_equal    = false;
        bool cover_identical = false;
        double rec_mse  = -1.0;
        double rec_psnr = -1.0;
        if (rec_ok) {
            share_equal     = (rec_share == shares[i]);
            cover_identical = (rec_cover.data == cover_orig.data);
            if (!cover_identical) {
                rec_mse  = compute_mse(cover_orig.data, rec_cover.data);
                rec_psnr = compute_psnr(cover_orig.data, rec_cover.data);
            }
        } else {
            cerr << "PEE-MED recovery failed for index " << i << "\n";
        }

        cout << "PEE-MED," << i << "," << cname << "," << mse << "," << psnr << ","
             << (share_equal ? "YES" : "NO") << "," << (cover_identical ? "YES" : "NO") << ",";
        if (rec_mse  < 0) cout << "-,"; else cout << rec_mse  << ",";
        if (rec_psnr < 0) cout << "-\n"; else cout << rec_psnr << "\n";
    }
}

static void test_PEE_MED_Adaptive(int k, int n, int lsb_bits, bool write_file = false)
{
    if (k <= 0 || n <= 0 || k > n) { cerr << "test_PEE_MED_Adaptive: invalid k/n\n"; return; }
    if (lsb_bits < 1 || lsb_bits > 8) { cerr << "test_PEE_MED_Adaptive: invalid lsb_bits\n"; return; }

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Secret image validation failed\n"; return;
    }

    auto covers = generate_cover_paths(n);
    auto shares = generate_shares(secret, k, n);
    size_t share_len = shares[0].size();

    cout << "---- TEST: PEE-MED-Adaptive k=" << k << " n=" << n
         << " lsb_bits=" << lsb_bits
         << " write_file=" << (write_file ? "YES" : "NO")
         << " share_bytes=" << share_len << " ----\n";
    cout << "Method,Index,ImageName,MSE,PSNR(dB),ShareEqual,CoverIdentical,RecCoverMSE,RecCoverPSNR\n";

    for (int i = 0; i < n; ++i) {
        TiffInfo cover_orig = readTiff(covers[i]);
        if (!validate_gray(cover_orig, covers[i], cover_orig.width, cover_orig.height)) {
            cerr << "Cover validation failed: " << covers[i] << "\n"; continue;
        }
        string cname = get_basename(covers[i]);
        string out_base = make_output_base("stego_pee_med_adaptive", i, cname);

        TiffInfo stego;
        bool ok = embed_share_pee_med_adaptive(cover_orig, shares[i], stego, lsb_bits);
        if (!ok) { cout << "PEE-MED-Adaptive," << i << "," << cname << ",-,-,NO_EMBED,-,-,-\n"; continue; }

        if (write_file) writeTiff(stego, out_base + ".tiff");

        double mse  = compute_mse(cover_orig.data, stego.data);
        double psnr = compute_psnr(cover_orig.data, stego.data);

        vector<unsigned char> rec_share;
        TiffInfo rec_cover;
        bool rec_ok = recover_share_pee_med_adaptive(stego, share_len, rec_share, rec_cover);

        bool share_equal    = false;
        bool cover_identical = false;
        double rec_mse  = -1.0;
        double rec_psnr = -1.0;
        if (rec_ok) {
            share_equal     = (rec_share == shares[i]);
            cover_identical = (rec_cover.data == cover_orig.data);
            if (!cover_identical) {
                rec_mse  = compute_mse(cover_orig.data, rec_cover.data);
                rec_psnr = compute_psnr(cover_orig.data, rec_cover.data);
            }
        }

        cout << "PEE-MED-Adaptive," << i << "," << cname << "," << mse << "," << psnr << ","
             << (share_equal ? "YES" : "NO") << "," << (cover_identical ? "YES" : "NO") << ",";
        if (rec_mse  < 0) cout << "-,"; else cout << rec_mse  << ",";
        if (rec_psnr < 0) cout << "-\n"; else cout << rec_psnr << "\n";
    }
}

static void test_PEE_Diamond(int k, int n, bool write_file = false, int pee_T = 1)
{
    if (k <= 0 || n <= 0 || k > n) { cerr << "test_PEE_Diamond: invalid k/n\n"; return; }

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Secret image validation failed\n"; return;
    }

    auto covers = generate_cover_paths(n);
    auto shares = generate_shares(secret, k, n);
    size_t share_len = shares[0].size();

    cout << "---- TEST: PEE-Diamond k=" << k << " n=" << n << " T=" << pee_T
         << " write_file=" << (write_file ? "YES" : "NO")
         << " share_bytes=" << share_len << " ----\n";
    cout << "Method,Index,ImageName,MSE,PSNR(dB),ShareEqual,CoverIdentical,RecCoverMSE,RecCoverPSNR\n";

    for (int i = 0; i < n; ++i) {
        TiffInfo cover_orig = readTiff(covers[i]);
        if (!validate_gray(cover_orig, covers[i], cover_orig.width, cover_orig.height)) {
            cerr << "Cover validation failed: " << covers[i] << "\n"; continue;
        }
        string cname = get_basename(covers[i]);
        string out_base = make_output_base("stego_pee_diamond", i, cname);

        TiffInfo stego;
        vector<uint8_t> pee_diamond_tags;
        bool ok = embed_share_pee_diamond(cover_orig, shares[i], stego, pee_diamond_tags, pee_T);
        if (!ok) { cout << "PEE-Diamond," << i << "," << cname << ",-,-,NO_EMBED,-,-,-\n"; continue; }

        if (write_file) writeTiff(stego, out_base + ".tiff");

        double mse  = compute_mse(cover_orig.data, stego.data);
        double psnr = compute_psnr(cover_orig.data, stego.data);

        vector<unsigned char> rec_share;
        TiffInfo rec_cover;
        bool rec_ok = recover_share_pee_diamond(stego, share_len, pee_diamond_tags, rec_share, rec_cover, pee_T);

        bool share_equal    = false;
        bool cover_identical = false;
        double rec_mse  = -1.0;
        double rec_psnr = -1.0;
        if (rec_ok) {
            share_equal     = (rec_share == shares[i]);
            cover_identical = (rec_cover.data == cover_orig.data);
            if (!cover_identical) {
                rec_mse  = compute_mse(cover_orig.data, rec_cover.data);
                rec_psnr = compute_psnr(cover_orig.data, rec_cover.data);
            }
        } else {
            cerr << "PEE-Diamond recovery failed for index " << i << "\n";
        }

        cout << "PEE-Diamond," << i << "," << cname << "," << mse << "," << psnr << ","
             << (share_equal ? "YES" : "NO") << "," << (cover_identical ? "YES" : "NO") << ",";
        if (rec_mse  < 0) cout << "-,"; else cout << rec_mse  << ",";
        if (rec_psnr < 0) cout << "-\n"; else cout << rec_psnr << "\n";
    }
}

static void test_PEE_Diamond_NoMap(int k, int n, bool write_file = false, int pee_T = 1)
{
    if (k <= 0 || n <= 0 || k > n) { cerr << "test_PEE_Diamond_NoMap: invalid k/n\n"; return; }

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Secret image validation failed\n"; return;
    }

    auto covers = generate_cover_paths(n);
    auto shares = generate_shares(secret, k, n);
    size_t share_len = shares[0].size();

    cout << "---- TEST: PEE-Diamond-NoMap k=" << k << " n=" << n << " T=" << pee_T
         << " write_file=" << (write_file ? "YES" : "NO")
         << " share_bytes=" << share_len << " ----\n";
    cout << "Method,Index,ImageName,MSE,PSNR(dB),ShareEqual,CoverIdentical,RecCoverMSE,RecCoverPSNR\n";

    for (int i = 0; i < n; ++i) {
        TiffInfo cover_orig = readTiff(covers[i]);
        if (!validate_gray(cover_orig, covers[i], cover_orig.width, cover_orig.height)) {
            cerr << "Cover validation failed: " << covers[i] << "\n"; continue;
        }
        string cname = get_basename(covers[i]);
        string out_base = make_output_base("stego_pee_diamond_nomap", i, cname);

        TiffInfo stego;
        bool ok = embed_share_pee_diamond_no_map(cover_orig, shares[i], stego, pee_T);
        if (!ok) { cout << "PEE-Diamond-NoMap," << i << "," << cname << ",-,-,NO_EMBED,-,-,-\n"; continue; }

        if (write_file) writeTiff(stego, out_base + ".tiff");

        double mse  = compute_mse(cover_orig.data, stego.data);
        double psnr = compute_psnr(cover_orig.data, stego.data);

        vector<unsigned char> rec_share;
        TiffInfo rec_cover;
        bool rec_ok = recover_share_pee_diamond_no_map(stego, share_len, rec_share, rec_cover, pee_T);

        bool share_equal    = false;
        bool cover_identical = false;
        double rec_mse  = -1.0;
        double rec_psnr = -1.0;
        if (rec_ok) {
            share_equal     = (rec_share == shares[i]);
            cover_identical = (rec_cover.data == cover_orig.data);
            if (!cover_identical) {
                rec_mse  = compute_mse(cover_orig.data, rec_cover.data);
                rec_psnr = compute_psnr(cover_orig.data, rec_cover.data);
            }
        } else {
            cerr << "PEE-Diamond-NoMap recovery failed for index " << i << "\n";
        }

        cout << "PEE-Diamond-NoMap," << i << "," << cname << "," << mse << "," << psnr << ","
             << (share_equal ? "YES" : "NO") << "," << (cover_identical ? "YES" : "NO") << ",";
        if (rec_mse  < 0) cout << "-,"; else cout << rec_mse  << ",";
        if (rec_psnr < 0) cout << "-\n"; else cout << rec_psnr << "\n";
    }
}

static void test_PVO_Sorted(int k, int n, bool write_file = false)
{
    if (k <= 0 || n <= 0 || k > n) { cerr << "test_PVO_Sorted: invalid k/n\n"; return; }

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Secret image validation failed\n"; return;
    }

    auto covers = generate_cover_paths(n);
    auto shares = generate_shares(secret, k, n);
    size_t share_len = shares[0].size();

    cout << "---- TEST: PVO-Sorted k=" << k << " n=" << n
         << " write_file=" << (write_file ? "YES" : "NO")
         << " share_bytes=" << share_len << " ----\n";
    cout << "Method,Index,ImageName,MSE,PSNR(dB),ShareEqual,CoverIdentical,RecCoverMSE,RecCoverPSNR\n";

    for (int i = 0; i < n; ++i) {
        TiffInfo cover_orig = readTiff(covers[i]);
        if (!validate_gray(cover_orig, covers[i], cover_orig.width, cover_orig.height)) {
            cerr << "Cover validation failed: " << covers[i] << "\n"; continue;
        }
        string cname = get_basename(covers[i]);
        string out_base = make_output_base("stego_pvo_sorted", i, cname);

        TiffInfo stego;
        vector<size_t> sort_order;
        vector<uint8_t> used_map;
        vector<uint8_t> perm_map;
        bool ok = embed_share_pvo_sorted(cover_orig, shares[i], stego,
                                         sort_order, used_map, perm_map);
        if (!ok) { cout << "PVO-Sorted," << i << "," << cname << ",-,-,NO_EMBED,-,-,-\n"; continue; }

        if (write_file) writeTiff(stego, out_base + ".tiff");

        double mse  = compute_mse(cover_orig.data, stego.data);
        double psnr = compute_psnr(cover_orig.data, stego.data);

        vector<unsigned char> rec_share;
        TiffInfo rec_cover;
        bool rec_ok = recover_share_pvo_sorted(stego, sort_order, used_map, perm_map,
                                               share_len, rec_share, rec_cover);

        bool share_equal    = false;
        bool cover_identical = false;
        double rec_mse  = -1.0;
        double rec_psnr = -1.0;
        if (rec_ok) {
            share_equal     = (rec_share == shares[i]);
            cover_identical = (rec_cover.data == cover_orig.data);
            if (!cover_identical) {
                rec_mse  = compute_mse(cover_orig.data, rec_cover.data);
                rec_psnr = compute_psnr(cover_orig.data, rec_cover.data);
            }
        } else {
            cerr << "PVO-Sorted recovery failed for index " << i << "\n";
        }

        cout << "PVO-Sorted," << i << "," << cname << "," << mse << "," << psnr << ","
             << (share_equal ? "YES" : "NO") << "," << (cover_identical ? "YES" : "NO") << ",";
        if (rec_mse  < 0) cout << "-,"; else cout << rec_mse  << ",";
        if (rec_psnr < 0) cout << "-\n"; else cout << rec_psnr << "\n";
    }
}

static void test_PVO_Inband(int k, int n, bool write_file = false)
{
    if (k <= 0 || n <= 0 || k > n) { cerr << "test_PVO_Inband: invalid k/n\n"; return; }
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Secret image validation failed\n"; return;
    }

    auto covers = generate_cover_paths(n);
    auto shares = generate_shares(secret, k, n);
    size_t share_len = shares[0].size();
    cout << "---- TEST: PVO-Inband k=" << k << " n=" << n
         << " write_file=" << (write_file ? "YES" : "NO")
         << " share_bytes=" << share_len << " ----\n";
    cout << "Method,Index,ImageName,MSE,PSNR(dB),ShareEqual,CoverIdentical,RecCoverMSE,RecCoverPSNR\n";

    for (int i = 0; i < n; ++i) {
        TiffInfo cover_orig = readTiff(covers[i]);
        if (!validate_gray(cover_orig, covers[i], cover_orig.width, cover_orig.height)) {
            cerr << "Cover validation failed: " << covers[i] << "\n"; continue;
        }
        string cname = get_basename(covers[i]);
        string out_base = make_output_base("stego_pvo_inband", i, cname);
        TiffInfo stego;
        bool ok = embed_share_pvo_inband(cover_orig, shares[i], stego);
        if (!ok) { cout << "PVO-Inband," << i << "," << cname << ",-,-,NO_EMBED,-,-,-\n"; continue; }
        if (write_file) writeTiff(stego, out_base + ".tiff");
        double mse = compute_mse(cover_orig.data, stego.data);
        double psnr = compute_psnr(cover_orig.data, stego.data);
        vector<unsigned char> rec_share;
        TiffInfo rec_cover;
        bool rec_ok = recover_share_pvo_inband(stego, share_len, rec_share, rec_cover);
        bool share_equal = rec_ok && (rec_share == shares[i]);
        bool cover_identical = rec_ok && (rec_cover.data == cover_orig.data);
        double rec_mse = -1.0, rec_psnr = -1.0;
        if (rec_ok && !cover_identical) {
            rec_mse = compute_mse(cover_orig.data, rec_cover.data);
            rec_psnr = compute_psnr(cover_orig.data, rec_cover.data);
        }
        cout << "PVO-Inband," << i << "," << cname << "," << mse << "," << psnr << ","
             << (share_equal ? "YES" : "NO") << "," << (cover_identical ? "YES" : "NO") << ",";
        if (rec_mse < 0) cout << "-,"; else cout << rec_mse << ",";
        if (rec_psnr < 0) cout << "-\n"; else cout << rec_psnr << "\n";
    }
}

static void test_PEE_Multi(int k, int n, bool write_file = false,
    int pee_T = 1, int pee_B = 2, int max_layers = 16)
{
    if (k <= 0 || n <= 0 || k > n) { cerr << "test_PEE_Multi: invalid k/n\n"; return; }

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Secret image validation failed\n"; return;
    }

    auto covers = generate_cover_paths(n);
    auto shares = generate_shares(secret, k, n);
    size_t share_len = shares[0].size();

    cout << "---- TEST: PEE-Multi k=" << k << " n=" << n
         << " T=" << pee_T << " B=" << pee_B << " max_layers=" << max_layers
         << " write_file=" << (write_file ? "YES" : "NO")
         << " share_bytes=" << share_len << " ----\n";
    cout << "Method,Index,ImageName,Layers,MSE,PSNR(dB),ShareEqual,CoverIdentical,RecCoverMSE,RecCoverPSNR\n";

    for (int i = 0; i < n; ++i) {
        TiffInfo cover_orig = readTiff(covers[i]);
        if (!validate_gray(cover_orig, covers[i], cover_orig.width, cover_orig.height)) {
            cerr << "Cover validation failed: " << covers[i] << "\n"; continue;
        }
        string cname = get_basename(covers[i]);
        string out_base = make_output_base("stego_pee_multi", i, cname);

        TiffInfo stego;
        vector<size_t> bits_per_layer;
        bool ok = embed_share_pee_multi(cover_orig, shares[i], stego,
                                        bits_per_layer,
                                        pee_T, pee_B, max_layers);
        if (!ok) {
            cout << "PEE-Multi," << i << "," << cname << ",-,-,-,NO_EMBED,-,-,-\n";
            continue;
        }

        if (write_file) writeTiff(stego, out_base + ".tiff");

        double mse  = compute_mse(cover_orig.data, stego.data);
        double psnr = compute_psnr(cover_orig.data, stego.data);
        int layers_used = static_cast<int>(bits_per_layer.size());

        vector<unsigned char> rec_share;
        TiffInfo rec_cover;
        bool rec_ok = recover_share_pee_multi(stego, share_len, bits_per_layer,
                                              rec_share, rec_cover, pee_T, pee_B);

        bool share_equal     = false;
        bool cover_identical = false;
        double rec_mse  = -1.0;
        double rec_psnr = -1.0;
        if (rec_ok) {
            share_equal     = (rec_share == shares[i]);
            cover_identical = (rec_cover.data == cover_orig.data);
            if (!cover_identical) {
                rec_mse  = compute_mse(cover_orig.data, rec_cover.data);
                rec_psnr = compute_psnr(cover_orig.data, rec_cover.data);
            }
        } else {
            cerr << "PEE-Multi recovery failed for index " << i << "\n";
        }

        cout << "PEE-Multi," << i << "," << cname << ","
             << layers_used << "," << mse << "," << psnr << ","
             << (share_equal ? "YES" : "NO") << ","
             << (cover_identical ? "YES" : "NO") << ",";
        if (rec_mse  < 0) cout << "-,"; else cout << rec_mse  << ",";
        if (rec_psnr < 0) cout << "-\n"; else cout << rec_psnr << "\n";
    }
}

static void test_PEE_CheckerSafe(int k, int n, bool write_file = false, int pee_T = 1, int pee_B = 2)
{
    if (k <= 0 || n <= 0 || k > n) { cerr << "test_PEE_CheckerSafe: invalid k/n\n"; return; }

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Secret image validation failed\n"; return;
    }

    auto covers = generate_cover_paths(n);
    auto shares = generate_shares(secret, k, n);
    size_t share_len = shares[0].size();

    cout << "---- TEST: PEE-CheckerSafe k=" << k << " n=" << n
         << " T=" << pee_T << " B=" << pee_B
         << " write_file=" << (write_file ? "YES" : "NO")
         << " share_bytes=" << share_len << " ----\n";
    cout << "Method,Index,ImageName,MSE,PSNR(dB),ShareEqual,CoverIdentical,RecCoverMSE,RecCoverPSNR\n";

    for (int i = 0; i < n; ++i) {
        TiffInfo cover_orig = readTiff(covers[i]);
        if (!validate_gray(cover_orig, covers[i], cover_orig.width, cover_orig.height)) {
            cerr << "Cover validation failed: " << covers[i] << "\n"; continue;
        }
        string cname = get_basename(covers[i]);
        string out_base = make_output_base("stego_pee_checker_safe", i, cname);

        TiffInfo stego;
        bool ok = embed_share_pee_checker_safe(cover_orig, shares[i], stego, pee_T, pee_B);
        if (!ok) { cout << "PEE-CheckerSafe," << i << "," << cname << ",-,-,NO_EMBED,-,-,-\n"; continue; }

        if (write_file) writeTiff(stego, out_base + ".tiff");

        double mse  = compute_mse(cover_orig.data, stego.data);
        double psnr = compute_psnr(cover_orig.data, stego.data);

        vector<unsigned char> rec_share;
        TiffInfo rec_cover;
        bool rec_ok = recover_share_pee_checker_safe(stego, share_len, rec_share, rec_cover, pee_T, pee_B);

        bool share_equal     = false;
        bool cover_identical = false;
        double rec_mse  = -1.0;
        double rec_psnr = -1.0;
        if (rec_ok) {
            share_equal     = (rec_share == shares[i]);
            cover_identical = (rec_cover.data == cover_orig.data);
            if (!cover_identical) {
                rec_mse  = compute_mse(cover_orig.data, rec_cover.data);
                rec_psnr = compute_psnr(cover_orig.data, rec_cover.data);
            }
        } else {
            cerr << "PEE-CheckerSafe recovery failed for index " << i << "\n";
        }

        cout << "PEE-CheckerSafe," << i << "," << cname << ","
             << fixed << mse << "," << psnr << ","
             << (share_equal ? "YES" : "NO") << ","
             << (cover_identical ? "YES" : "NO") << ",";
        if (rec_mse  < 0) cout << "-,"; else cout << fixed << rec_mse  << ",";
        if (rec_psnr < 0) cout << "-\n"; else cout << fixed << rec_psnr << "\n";
        cout.unsetf(ios::fixed);
    }
}

static void test_Histogram_Shifting(int k, int n, bool write_file = false)
{
    if (k <= 0 || n <= 0 || k > n) { cerr << "test_Histogram_Shifting: invalid k/n\n"; return; }

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Secret image validation failed\n"; return;
    }

    auto covers = generate_cover_paths(n);
    auto shares = generate_shares(secret, k, n);
    size_t share_len = shares[0].size();

    cout << "---- TEST: Histogram Shifting k=" << k << " n=" << n
         << " write_file=" << (write_file ? "YES" : "NO")
         << " share_bytes=" << share_len << " ----\n";
    cout << "Method,Index,ImageName,MSE,PSNR(dB),ShareEqual,CoverIdentical,RecCoverMSE,RecCoverPSNR\n";

    for (int i = 0; i < n; ++i) {
        TiffInfo cover_orig = readTiff(covers[i]);
        if (!validate_gray(cover_orig, covers[i], cover_orig.width, cover_orig.height)) {
            cerr << "Cover validation failed: " << covers[i] << "\n"; continue;
        }
        string cname = get_basename(covers[i]);
        string out_base = make_output_base("stego_hist_shift", i, cname);

        TiffInfo stego;
        bool ok = embed_share_histogram_shifting(cover_orig, shares[i], stego);
        if (!ok) { cout << "Histogram Shifting," << i << "," << cname << ",-,-,NO_EMBED,-,-,-\n"; continue; }

        if (write_file) writeTiff(stego, out_base + ".tiff");

        double mse = compute_mse(cover_orig.data, stego.data);
        double psnr = compute_psnr(cover_orig.data, stego.data);

        vector<unsigned char> rec_share;
        TiffInfo rec_cover;
        bool rec_ok = recover_share_histogram_shifting(stego, share_len, rec_share, rec_cover);

        bool share_equal = rec_ok && (rec_share == shares[i]);
        bool cover_identical = rec_ok && (rec_cover.data == cover_orig.data);
        double rec_mse = -1.0;
        double rec_psnr = -1.0;
        if (rec_ok && !cover_identical) {
            rec_mse = compute_mse(cover_orig.data, rec_cover.data);
            rec_psnr = compute_psnr(cover_orig.data, rec_cover.data);
        }

        cout << "Histogram Shifting," << i << "," << cname << "," << mse << "," << psnr << ","
             << (share_equal ? "YES" : "NO") << "," << (cover_identical ? "YES" : "NO") << ",";
        if (rec_mse < 0) cout << "-" << ","; else cout << rec_mse << ",";
        if (rec_psnr < 0) cout << "-" << "\n"; else cout << rec_psnr << "\n";
    }
}

// ==========================================================================
// minimal_k(): binary search for the minimum feasible k per module.
//
// "Feasible k" means the module can successfully embed a share of size
// pixels/k bytes into the cover image.  Larger k -> smaller share -> easier.
// The feasibility function is monotone, so binary search works.
//
// Search space: divisors of secret.size() in [1, secret.size()].
// For a 512x512 image (2^18 pixels) these are the 19 powers of 2.
//
// Probe image: baboon (most textured cover, hardest case for PEE capacity).
// The search is silent; results are printed once the minimum k is found,
// then the full test is run with (k, k) so n=k shares, one per cover.
// ==========================================================================

// Build sorted list of all divisors of n (ascending).
// Uses O(sqrt(n)) algorithm instead of O(n) scan.
static vector<int> divisors_of(int n)
{
    vector<int> lo_divs, hi_divs;
    for (int d = 1; static_cast<long long>(d) * d <= static_cast<long long>(n); ++d) {
        if (n % d == 0) {
            lo_divs.push_back(d);
            if (d != n / d)
                hi_divs.push_back(n / d);
        }
    }
    // hi_divs is in descending order; reverse and append to get full sorted list
    for (int i = static_cast<int>(hi_divs.size()) - 1; i >= 0; --i)
        lo_divs.push_back(hi_divs[i]);
    return lo_divs;
}

// ---------- per-module probe functions (silent, single cover) ----------
// Each returns true if embedding succeeds for the given k on ALL covers
// in g_cover_names.  This guarantees the found k is valid for every cover,
// including any covers added to g_cover_names in the future.

static bool probe_mLSB(int k, int mlsb_t = 0)
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (secret.data.empty()) return false;
    auto shares = generate_shares(secret, k, k);
    for (const string& cname : g_cover_names) {
        TiffInfo cover = readTiff(g_input_dir + cname);
        if (cover.data.empty()) continue;
        // Check feasibility without actually embedding
        int t = (mlsb_t == 0)
            ? compute_min_t(shares[0].size(), cover.size())
            : mlsb_t;
        if (t > 8) return false;
        TiffInfo stego;
        if (!embed_share_mlsb(cover, shares[0], stego, t)) return false;
    }
    return true;
}

static bool probe_LSB_Matching(int k)
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (secret.data.empty()) return false;
    auto shares = generate_shares(secret, k, k);
    for (const string& cname : g_cover_names) {
        TiffInfo cover = readTiff(g_input_dir + cname);
        if (cover.data.empty()) continue;
        TiffInfo stego;
        if (!embed_share_lsb_matching(cover, shares[0], stego)) return false;
    }
    return true;
}

static bool probe_LSBMR(int k)
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (secret.data.empty()) return false;
    auto shares = generate_shares(secret, k, k);
    for (const string& cname : g_cover_names) {
        TiffInfo cover = readTiff(g_input_dir + cname);
        if (cover.data.empty()) continue;
        TiffInfo stego;
        if (!embed_share_lsbmr(cover, shares[0], stego)) return false;
    }
    return true;
}

static bool probe_LSB_HUFF(int k, int mlsb_t = 2)
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (secret.data.empty()) return false;
    auto shares = generate_shares(secret, k, k);
    for (const string& cname : g_cover_names) {
        TiffInfo cover = readTiff(g_input_dir + cname);
        if (cover.data.empty()) continue;
        TiffInfo stego;
        HuffmanCodec codec;
        if (!embed_share_lsb_with_huffman_inband(cover, shares[0], stego, codec,
                                                  mlsb_t, false, "")) return false;
    }
    return true;
}

static bool probe_PVO(int k)
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (secret.data.empty()) return false;
    auto shares = generate_shares(secret, k, k);
    for (const string& cname : g_cover_names) {
        TiffInfo cover = readTiff(g_input_dir + cname);
        if (cover.data.empty()) continue;
        TiffInfo stego;
        vector<uint8_t> used_map, perm_map;
        if (!embed_share_pvo_sequential(cover, shares[0], stego, used_map, perm_map))
            return false;
    }
    return true;
}

static bool probe_PVO_Sorted(int k)
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (secret.data.empty()) return false;
    auto shares = generate_shares(secret, k, k);
    for (const string& cname : g_cover_names) {
        TiffInfo cover = readTiff(g_input_dir + cname);
        if (cover.data.empty()) continue;
        TiffInfo stego;
        vector<size_t> sort_order;
        vector<uint8_t> used_map, perm_map;
        if (!embed_share_pvo_sorted(cover, shares[0], stego, sort_order, used_map, perm_map))
            return false;
    }
    return true;
}

static bool probe_PVO_Inband(int k)
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (secret.data.empty()) return false;
    auto shares = generate_shares(secret, k, k);
    for (const string& cname : g_cover_names) {
        TiffInfo cover = readTiff(g_input_dir + cname);
        if (cover.data.empty()) continue;
        TiffInfo stego;
        if (!embed_share_pvo_inband(cover, shares[0], stego)) return false;
    }
    return true;
}

static bool probe_PEE_MED(int k)
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (secret.data.empty()) return false;
    auto shares = generate_shares(secret, k, k);
    for (const string& cname : g_cover_names) {
        TiffInfo cover = readTiff(g_input_dir + cname);
        if (cover.data.empty()) continue;
        TiffInfo stego;
        if (!embed_share_pee_med(cover, shares[0], stego)) return false;
    }
    return true;
}

static bool probe_PEE_MED_Adaptive(int k, int lsb_bits = 6)
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (secret.data.empty()) return false;
    auto shares = generate_shares(secret, k, k);
    for (const string& cname : g_cover_names) {
        TiffInfo cover = readTiff(g_input_dir + cname);
        if (cover.data.empty()) continue;
        TiffInfo stego;
        if (!embed_share_pee_med_adaptive(cover, shares[0], stego, lsb_bits)) return false;
    }
    return true;
}

static bool probe_PEE_Diamond(int k, int T = 1)
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (secret.data.empty()) return false;
    auto shares = generate_shares(secret, k, k);
    for (const string& cname : g_cover_names) {
        TiffInfo cover = readTiff(g_input_dir + cname);
        if (cover.data.empty()) continue;
        TiffInfo stego;
        vector<uint8_t> tags;
        if (!embed_share_pee_diamond(cover, shares[0], stego, tags, T)) return false;
    }
    return true;
}

static bool probe_PEE_Diamond_NoMap(int k, int T = 1)
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (secret.data.empty()) return false;
    auto shares = generate_shares(secret, k, k);
    for (const string& cname : g_cover_names) {
        TiffInfo cover = readTiff(g_input_dir + cname);
        if (cover.data.empty()) continue;
        TiffInfo stego;
        if (!embed_share_pee_diamond_no_map(cover, shares[0], stego, T)) return false;
    }
    return true;
}

static bool probe_PEE_Multi(int k, int T = 1, int B = 2, int max_layers = 16)
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (secret.data.empty()) return false;
    auto shares = generate_shares(secret, k, k);
    for (const string& cname : g_cover_names) {
        TiffInfo cover = readTiff(g_input_dir + cname);
        if (cover.data.empty()) continue;
        TiffInfo stego;
        vector<size_t> bits_per_layer;
        if (!embed_share_pee_multi(cover, shares[0], stego, bits_per_layer,
                                   T, B, max_layers)) return false;
    }
    return true;
}

static bool probe_PEE_CheckerSafe(int k, int T = 1, int B = 2)
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (secret.data.empty()) return false;
    auto shares = generate_shares(secret, k, k);
    for (const string& cname : g_cover_names) {
        TiffInfo cover = readTiff(g_input_dir + cname);
        if (cover.data.empty()) continue;
        TiffInfo stego;
        if (!embed_share_pee_checker_safe(cover, shares[0], stego, T, B)) return false;
    }
    return true;
}

static bool probe_Histogram_Shifting(int k)
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (secret.data.empty()) return false;
    auto shares = generate_shares(secret, k, k);
    for (const string& cname : g_cover_names) {
        TiffInfo cover = readTiff(g_input_dir + cname);
        if (cover.data.empty()) continue;
        vector<int> hist(256, 0);
        for (auto p : cover.data) hist[p]++;
        int peak = 0;
        int max_freq = 0;
        for (int i = 0; i < 256; ++i) {
            if (hist[i] > max_freq) {
                max_freq = hist[i];
                peak = i;
            }
        }
        size_t capacity_bytes = max_freq / 8;
        cout << "HS probe: " << cname << " peak=" << peak << " freq=" << max_freq << " capacity=" << capacity_bytes << " bytes, share_len=" << shares[0].size() << endl;
        if (shares[0].size() > capacity_bytes) {
            cout << "Insufficient capacity for " << cname << endl;
            return false;
        }
        TiffInfo stego;
        if (!embed_share_histogram_shifting(cover, shares[0], stego)) return false;
    }
    return true;
}

// ---------- linear search over all k in [2, max_k] ----------
// Returns the smallest k >= 2 such that probe(k) == true,
// or -1 if no such k exists.
// Linear search is used because share size = ceil(N/(k-1)) is not strictly
// monotone for all integers (consecutive k values may give the same share
// size), so binary search could miss the true minimum.
static int linear_search_min_k(int max_k, const function<bool(int)>& probe)
{
    for (int k = 2; k <= max_k; ++k) {
        if (probe(k)) return k;
    }
    return -1;
}

// ---------- minimal_k ----------
static void minimal_k(bool write_file = false)
{
    TiffInfo secret = readTiff(g_input_dir + g_secret_name);
    if (secret.data.empty()) { cerr << "minimal_k: cannot load secret\n"; return; }

    const int total_pixels = static_cast<int>(secret.size());

    // Search all k in [2, max_k]. With the revised scheme, share size is
    // ceil(N/(k-1)) bytes, which is not tied to divisors of N.
    // max_k=128: at k=128, share = ceil(262144/127) = 2064 bytes, small enough.
    const int max_k = 128;

    cout << "==== minimal_k: secret=" << g_secret_name
         << " pixels=" << total_pixels
         << " max_k=" << max_k << " ====\n\n";

    // ---- mLSB ----
    {
        int k = linear_search_min_k(max_k, [](int d){ return probe_mLSB(d); });
        cout << "mLSB        minimal k = " << k << "\n";
        if (k > 0) test_mLSB(k, k, write_file);
    }

#if 0
    // ---- LSB-Matching ----
    {
        int k = linear_search_min_k(max_k, [](int d){ return probe_LSB_Matching(d); });
        cout << "LSB-Matching minimal k = " << k << "\n";
        if (k > 0) test_LSB_Matching(k, k, write_file);
    }

    // ---- LSBMR ----
    {
        int k = linear_search_min_k(max_k, [](int d){ return probe_LSBMR(d); });
        cout << "LSBMR       minimal k = " << k << "\n";
        if (k > 0) test_LSBMR(k, k, write_file);
    }
#endif

#if 0
    // ---- LSB_HUFF ----
    {
        int t = 2;
        int k = linear_search_min_k(max_k, [&](int d){ return probe_LSB_HUFF(d, t); });
        cout << "LSB_HUFF   minimal k = " << k << "\n";
        if (k > 0) test_LSB_HUFF(k, k, write_file, t);
    }
#endif

#if 0
    // ---- PVO (sequential) ----
    {
        int k = linear_search_min_k(max_k, [](int d){ return probe_PVO(d); });
        cout << "PVO         minimal k = " << k << "\n";
        if (k > 0) test_PVO(k, k, write_file);
    }

    // ---- PVO-Sorted ----
    {
        int k = linear_search_min_k(max_k, [](int d){ return probe_PVO_Sorted(d); });
        cout << "PVO-Sorted  minimal k = " << k << "\n";
        if (k > 0) test_PVO_Sorted(k, k, write_file);
    }
#endif

#if 0
    // ---- PVO-Inband (self-contained prototype) ----
    {
        int k = linear_search_min_k(max_k, [](int d){ return probe_PVO_Inband(d); });
        cout << "PVO-Inband  minimal k = " << k << "\n";
        if (k > 0) test_PVO_Inband(k, k, write_file);
    }
#endif

#if 0
    // ---- PEE-MED ----
    {
        int k = linear_search_min_k(max_k, [](int d){ return probe_PEE_MED(d); });
        cout << "PEE-MED     minimal k = " << k << "\n";
        if (k > 0) test_PEE_MED(k, k, write_file);    
    }
#endif

#if 0
    // ---- PEE-MED-Adaptive (self-contained, variable lsb_bits) ----
    {
        int lsb_bits = 6;
        int k = linear_search_min_k(max_k, [&](int d){ return probe_PEE_MED_Adaptive(d, lsb_bits); });
        cout << "PEE-MED-Adaptive(l=" << lsb_bits << ") minimal k = " << k << "\n";
        if (k > 0) test_PEE_MED_Adaptive(k, k, lsb_bits, write_file);
    }
#endif

#if 0
    // ---- PEE-Diamond (NoMap) ----
    {
        int k = linear_search_min_k(max_k, [](int d){ return probe_PEE_Diamond_NoMap(d); });
        cout << "PEE-Diamond-NoMap minimal k = " << k << "\n";
        if (k > 0) test_PEE_Diamond_NoMap(k, k, write_file);
    }
#endif

#if 0
    // ---- PEE-Diamond (with side info) ----
    {
        int k = linear_search_min_k(max_k, [](int d){ return probe_PEE_Diamond(d); });
        cout << "PEE-Diamond minimal k = " << k << "\n";
        if (k > 0) test_PEE_Diamond(k, k, write_file);
    }
#endif

#if 0
    // ---- PEE-Multi ----
    {
        int k = linear_search_min_k(max_k, [](int d){ return probe_PEE_Multi(d); });
        cout << "PEE-Multi   minimal k = " << k << "\n";
        if (k > 0) test_PEE_Multi(k, k, write_file);
    }
#endif

#if 0
    // ---- PEE-CheckerSafe ----
    {
        int k = linear_search_min_k(max_k, [](int d){ return probe_PEE_CheckerSafe(d); });
        cout << "PEE-CheckerSafe minimal k = " << k << "\n";
        if (k > 0) test_PEE_CheckerSafe(k, k, write_file);
    }
#endif

#if 0
    // ---- Histogram Shifting ----
    {
        int k = linear_search_min_k(max_k, [](int d){ return probe_Histogram_Shifting(d); });
        cout << "Histogram Shifting minimal k = " << k << "\n";
        if (k > 0) test_Histogram_Shifting(k, k, write_file);
    }
#endif
}

// ---------- test secret sharing ----------
static void test_secret_sharing()
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Secret image validation failed\n"; return;
    }

    // k=3 means polynomial degree 2, threshold k=3, need n>=3
    int k = 3, n = 5;

    auto shares = generate_shares(secret, k, n);
    auto reconstructed = reconstruct_secret(shares, k, secret.size());

    bool correct = (reconstructed == secret.data);
    cout << "Secret sharing test: k=" << k << " n=" << n
         << " threshold=" << k
         << " share_bytes=" << shares[0].size()
         << " correct=" << (correct ? "YES" : "NO") << "\n";

    // Test subset reconstruction using shares 2, 3, 5
    vector<vector<unsigned char>> subset_shares;
    subset_shares.push_back(shares[1]); // Share 2 (index 1)
    subset_shares.push_back(shares[2]); // Share 3 (index 2)
    subset_shares.push_back(shares[4]); // Share 5 (index 4)
    auto reconstructed_subset = reconstruct_secret(subset_shares, k, secret.size(), {2, 3, 5});
    bool correct_subset = (reconstructed_subset == secret.data);
    cout << "Secret sharing subset test {2,3,5}: correct=" << (correct_subset ? "YES" : "NO") << "\n";
}

// ---------------------------------------------------------------------------
// Sweep test: for each n in [4..8], test every valid k (2 <= k <= n-1)
// using mLSB embedding. Reports share size, PSNR, and secret recovery.
// k=1 is excluded (block_secrets = k-1 = 0 makes no sense).
// k=n is excluded (threshold = k+1 = n+1 > n, impossible).
// ---------------------------------------------------------------------------
static void test_secret_sharing_sweep()
{
    cout << "\n==== Secret Sharing Sweep: mLSB, all valid k, n in [4..8] ====\n";
    cout << "n,k,threshold,share_bytes,cover,MSE,PSNR(dB),ShareEqual,SecretRecovered\n";

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Sweep: secret validation failed\n"; return;
    }
    const size_t N = secret.size();

    for (int n = 4; n <= 16; ++n) {
        // valid k: 2 <= k <= n  (k=n means all shares needed)
        for (int k = 2; k <= n; ++k) {
            auto shares = generate_shares(secret, k, n);
            size_t share_len = shares[0].size();

            // Test embedding into the first cover only (jetplane)
            // and verify secret recovery using the first k+1 shares.
            const string& cname = g_cover_names[0]; // jetplane
            TiffInfo cover = readTiff(g_input_dir + cname);
            if (cover.data.empty()) continue;

            TiffInfo stego;
            bool embed_ok = embed_share_mlsb(cover, shares[0], stego, 2);
            if (!embed_ok) {
                cout << n << "," << k << "," << k+1 << "," << share_len
                     << "," << cname << ",-,-,NO_EMBED,NO\n";
                continue;
            }

            double mse  = compute_mse(cover.data, stego.data);
            double psnr = compute_psnr(cover.data, stego.data);

            // Extract share from stego
            vector<unsigned char> extracted;
            bool ex_ok = extract_share_mlsb(stego, share_len, extracted, 2);
            bool share_equal = ex_ok && (extracted == shares[0]);

            // Reconstruct secret using first k shares (indices 0..k-1)
            // Replace shares[0] with the extracted version to simulate real use
            vector<vector<unsigned char>> recv_shares(shares.begin(), shares.begin() + k);
            recv_shares[0] = extracted;
            auto reconstructed = reconstruct_secret(recv_shares, k, N);
            bool secret_ok = (reconstructed == secret.data);

            cout << n << "," << k << "," << k+1 << "," << share_len
                 << "," << cname
                 << "," << mse << "," << psnr
                 << "," << (share_equal ? "YES" : "NO")
                 << "," << (secret_ok  ? "YES" : "NO") << "\n";
        }
    }
}

// ---------------------------------------------------------------------------
// Comparison test matching article.6 experimental setup.
//
// Secret: woman.tiff
// Covers: jetplane, house, girl, baboon, airplane (5 images)
// Configurations:
//   n=4, k in {2,3,4}
//   n=5, k in {2,3,4,5}
// Image sizes: 256x256 and 512x512 (uses g_image_size; run twice)
// Metrics: PSNR and SSIM of each stego-share vs original cover.
//
// Note: article.6 embeds the secret pixel at f(0) and cover pixels at f(i).
// Our method embeds k-1 secret pixels as coefficients and uses mLSB to hide
// shares in cover images. The comparison is on stego-image quality (PSNR/SSIM)
// and perfect secret recovery.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// mLSB parameter sweep: t in [1..5], n in [5..8], k in [minimal_k..n].
//
// For each (t, n) pair, find the minimal k by linear search (smallest k
// where mLSB can embed the share into ALL cover images).
// Then test every k from minimal_k up to n.
// Reports PSNR, share correctness, and secret recovery.
// ---------------------------------------------------------------------------
static void test_mlsb_parameter_sweep()
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "mLSB sweep: secret validation failed\n"; return;
    }
    const size_t N = secret.size();

    cout << "\n==== mLSB Parameter Sweep: t=[1..5], n=[5..8], k=[min_k..n] ====\n";
    cout << "t,n,k,min_k,share_bytes,cover,MSE,PSNR(dB),ShareEqual,SecretRecovered\n";

    for (int t = 1; t <= 5; ++t) {
        for (int n = 5; n <= 8; ++n) {

            // Find minimal k for this (t, n): smallest k in [2..n] where
            // mLSB with t bits can embed the share into every cover.
            // Correct capacity = floor(N / ceil(8/t)) bytes.
            // pixels_per_byte = ceil(8/t), capacity_bytes = N / pixels_per_byte.
            int pixels_per_byte = (8 + t - 1) / t;  // ceil(8/t)
            size_t capacity_bytes = N / static_cast<size_t>(pixels_per_byte);

            int min_k = -1;
            for (int k = 2; k <= n; ++k) {
                // share size = ceil(N / (k-1))
                size_t share_bytes = (N + static_cast<size_t>(k - 1) - 1)
                                     / static_cast<size_t>(k - 1);
                if (share_bytes <= capacity_bytes) {
                    min_k = k;
                    break;
                }
            }

            if (min_k < 0) {
                // No valid k found for this (t, n)
                cout << t << "," << n << ",-,-,-,-,-,-,-,-\n";
                continue;
            }

            // Test every k from min_k to n
            for (int k = min_k; k <= n; ++k) {
                auto shares = generate_shares(secret, k, n);
                size_t share_len = shares[0].size();

                // Test on first cover only (jetplane) for brevity
                const string& cname = g_cover_names[0];
                TiffInfo cover = readTiff(g_input_dir + cname);
                if (cover.data.empty()) continue;

                TiffInfo stego;
                bool embed_ok = embed_share_mlsb(cover, shares[0], stego, t);
                if (!embed_ok) {
                    cout << t << "," << n << "," << k << "," << min_k << ","
                         << share_len << "," << cname
                         << ",-,-,NO_EMBED,NO\n";
                    continue;
                }

                double mse  = compute_mse(cover.data, stego.data);
                double psnr = compute_psnr(cover.data, stego.data);

                vector<unsigned char> extracted;
                bool ex_ok = extract_share_mlsb(stego, share_len, extracted, t);
                bool share_equal = ex_ok && (extracted == shares[0]);

                // Reconstruct using first k shares, replacing share[0] with extracted
                vector<vector<unsigned char>> recv(shares.begin(),
                                                   shares.begin() + k);
                recv[0] = extracted;
                auto rec = reconstruct_secret(recv, k, N);
                bool secret_ok = (rec == secret.data);

                cout << t << "," << n << "," << k << "," << min_k << ","
                     << share_len << "," << cname
                     << "," << mse << "," << psnr
                     << "," << (share_equal ? "YES" : "NO")
                     << "," << (secret_ok  ? "YES" : "NO") << "\n";
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Minimal-t sweep: for every (k,n) with 2 <= k <= n <= 8,
// auto-select minimal t and report share_bytes, min_t, PSNR, MSE.
// Uses jetplane as the single cover (representative).
// ---------------------------------------------------------------------------
static void test_minimal_t_sweep()
{
    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "test_minimal_t_sweep: secret validation failed\n"; return;
    }

    cout << "\n==== Minimal-t Sweep: 2 <= k <= n <= 8, secret=" << g_secret_name << " ====\n";
    cout << "n,k,share_bytes,min_t,cover,MSE,PSNR(dB),ShareEqual,SecretRecovered\n";

    for (int n = 2; n <= 8; ++n) {
        for (int k = 2; k <= n; ++k) {
            auto shares = generate_shares(secret, k, n);
            size_t share_len = shares[0].size();

            // Use first cover to determine min_t (all covers are same size)
            TiffInfo first_cover = readTiff(g_input_dir + g_cover_names[0]);
            int min_t = compute_min_t(share_len, first_cover.size());

            if (min_t > 8) {
                cout << n << "," << k << "," << share_len
                     << ",INFEASIBLE,ALL,-,-,-,-\n";
                continue;
            }

            // Embed share i into cover i, verify each one
            for (int i = 0; i < n; ++i) {
                string cname = g_cover_names[i % static_cast<int>(g_cover_names.size())];
                TiffInfo cover = readTiff(g_input_dir + cname);
                if (!validate_gray(cover, g_input_dir + cname, cover.width, cover.height)) {
                    cerr << "test_minimal_t_sweep: cover validation failed: " << cname << "\n";
                    continue;
                }

                TiffInfo stego;
                bool embed_ok = embed_share_mlsb(cover, shares[i], stego, min_t);
                if (!embed_ok) {
                    cout << n << "," << k << "," << share_len
                         << "," << min_t << "," << cname << ",-,-,NO_EMBED,NO\n";
                    continue;
                }

                double mse  = compute_mse(cover.data, stego.data);
                double psnr = compute_psnr(cover.data, stego.data);

                vector<unsigned char> extracted;
                bool ex_ok = extract_share_mlsb(stego, share_len, extracted, min_t);
                bool share_equal = ex_ok && (extracted == shares[i]);

                // Reconstruct using all k shares (replace share i with extracted)
                bool secret_ok = false;
                if (ex_ok && i < k) {
                    vector<vector<unsigned char>> recv(shares.begin(),
                                                       shares.begin() + k);
                    recv[i] = extracted;
                    auto rec = reconstruct_secret(recv, k, secret.size());
                    secret_ok = (rec == secret.data);
                } else if (ex_ok) {
                    // share i >= k: not needed for reconstruction, just verify embed/extract
                    secret_ok = true; // extraction correct; reconstruction not tested for this share
                }

                cout << n << "," << k << "," << share_len
                     << "," << min_t
                     << "," << cname
                     << "," << mse << "," << psnr
                     << "," << (share_equal ? "YES" : "NO")
                     << "," << (secret_ok  ? "YES" : "NO") << "\n";
            }
        }
    }
}

static void test_comparison_article6()
{
    // Covers matching article.6 experimental setup
    const vector<string> cover_names = {
        "jetplane.tiff",
        "house.tiff",
        "girl.tiff",
        "baboon.tiff",
        "airplane.tiff"
    };

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Comparison test: secret validation failed\n"; return;
    }
    const size_t N = secret.size();

    cout << "\n==== Comparison with Lin et al. baseline (image_size=" << g_image_size << ") ====\n";
    cout << "BS scheme: preprocess secret pixels before sharing.\n";
    cout << "k=2 skipped (degenerate).\n\n";

    const vector<tuple<int,int,int>> cases = {
        {4, 3, 4},
        {4, 4, 4},
        {5, 3, 5},
        {5, 4, 5},
        {5, 5, 5},
    };

    cout << "n,k,t*,share_bytes,total_embed,cover,MSE,PSNR(dB),SSIM,ShareEqual,SecretRecovered\n";

    for (const auto& [n, k, num_covers] : cases) {
        // Step 1: preprocess secret pixels.
        vector<unsigned char> padded_secret = secret.data;
        size_t padded_N = ((padded_secret.size() + static_cast<size_t>(k) - 1)
                           / static_cast<size_t>(k))
                          * static_cast<size_t>(k);
        uint64_t rng_pad = 0xDEADBEEFCAFEBABEULL;
        while (padded_secret.size() < padded_N) {
            rng_pad ^= rng_pad << 13; rng_pad ^= rng_pad >> 7; rng_pad ^= rng_pad << 17;
            padded_secret.push_back(static_cast<unsigned char>(rng_pad & 0xFF));
        }

        unsigned char r = 0;
        size_t z = 0;
        vector<unsigned char> bs, processed_secret;
        if (!bs_preprocess(padded_secret, k, r, z, bs, processed_secret)) {
            cout << n << "," << k << ",-,-,-,ALL,BS_OVERFLOW,-,-,-,-\n";
            continue;
        }

        // Step 2: generate BS-mode shares from preprocessed secret.
        TiffInfo processed_info = secret;
        processed_info.data = processed_secret;
        auto shares = generate_shares(processed_info, k, n, /*use_nonzero_leading=*/false);
        size_t share_len = shares[0].size();

        size_t bs_bytes_count = (z + 7) / 8;
        size_t total_embed = 1 + bs_bytes_count + share_len;

        {
            TiffInfo fc = readTiff(g_input_dir + cover_names[0]);
            int min_t_check = compute_min_t(total_embed, fc.size());
            cout << "# k=" << k << " share=" << share_len
                 << " total_embed=" << total_embed
                 << " t*=" << min_t_check << " z=" << z << "\n";
        }

        for (int ci = 0; ci < num_covers && ci < static_cast<int>(cover_names.size()); ++ci) {
            TiffInfo cover = readTiff(g_input_dir + cover_names[ci]);
            if (!validate_gray(cover, g_input_dir + cover_names[ci],
                               cover.width, cover.height)) continue;

            int min_t = compute_min_t(total_embed, cover.size());

            TiffInfo stego;
            bool embed_ok = embed_share_mlsb_bs(cover, shares[ci],
                                                 r, bs, stego, min_t);
            if (!embed_ok) {
                cout << n << "," << k << "," << min_t << "," << share_len
                     << "," << total_embed << "," << cover_names[ci]
                     << ",-,-,-,NO_EMBED,NO\n";
                continue;
            }

            double mse  = compute_mse(cover.data, stego.data);
            double psnr = compute_psnr(cover.data, stego.data);
            double ssim = compute_ssim(cover.data, stego.data);

            vector<unsigned char> extracted_share;
            unsigned char ex_r = 0;
            vector<unsigned char> ex_bs;
            bool ex_ok = extract_share_mlsb_bs(stego, total_embed, share_len,
                                                extracted_share,
                                                ex_r, ex_bs, min_t);
            if (ex_ok && extracted_share.size() > share_len)
                extracted_share.resize(share_len);
            bool share_equal = ex_ok && (extracted_share == shares[ci]);

            bool secret_ok = false;
            if (ex_ok) {
                vector<vector<unsigned char>> recv(shares.begin(),
                                                   shares.begin() + k);
                if (ci < k) recv[ci] = extracted_share;
                auto rec_processed = reconstruct_secret(recv, k,
                                         processed_secret.size(), /*use_nonzero_leading=*/false);
                vector<unsigned char> rec_secret;
                if (bs_postprocess(rec_processed, k, ex_r, ex_bs,
                                   rec_secret)) {
                    rec_secret.resize(N);
                    secret_ok = (rec_secret == secret.data);
                }
            }

            cout << n << "," << k << "," << min_t << "," << share_len
                 << "," << total_embed
                 << "," << cover_names[ci]
                 << "," << mse << "," << psnr << "," << ssim
                 << "," << (share_equal ? "YES" : "NO")
                 << "," << (secret_ok  ? "YES" : "NO") << "\n";
        }
    }
}

static void test_mlsb_requested_pairs()
{
    cout << "\n==== mLSB requested runs ====\n";
    const vector<pair<int, int>> runs = {
        {4, 32},
        {8, 32},
        {16, 32},
        {32, 32}
    };

    for (const auto& [k, n] : runs) {
        cout << "\n[RUN] mLSB (k,n)=(" << k << "," << n << ")\n";
        test_mLSB(k, n, false, 2);
    }
}

static void test_PEE_MED_baboon_capacity()
{
    const string cover_path = g_input_dir + "baboon.tiff";
    TiffInfo cover = readTiff(cover_path);
    if (!validate_gray(cover, cover_path, cover.width, cover.height)) {
        cerr << "PEE-MED baboon capacity test: cover validation failed\n";
        return;
    }

    cout << "\n==== PEE-MED baboon.tiff capacity test ====\n";
    cout << "LSB,Feasible,ChosenM,HeaderPixelsR,OverallCapacityBits(M*LSB),FullImageCapacityBits,"
            "HeaderBits,CompressedLSBBits(Huffman),RefBits,UsableSecretBits,UsableSecretBytes\n";

    for (int lsb = 2; lsb <= 5; ++lsb) {
        PeeMedCapacityReport rep;
        bool ok = analyze_pee_med_capacity(cover, lsb, rep);
        if (!ok) {
            cout << lsb << ",NO,-,-,-,-,-,-,-,-,-\n";
            continue;
        }
        cout << rep.lsb_bits << ","
             << (rep.feasible ? "YES" : "NO") << ","
             << rep.chosen_M << ","
             << rep.header_pixels_R << ","
             << rep.overall_capacity_bits << ","
             << rep.full_image_capacity_bits << ","
             << rep.header_bits << ","
             << rep.compressed_lsb_bits << ","
             << rep.ref_bits << ","
             << rep.usable_secret_bits << ","
             << rep.usable_secret_bytes << "\n";
    }
}

static void analyze_PEE_Diamond_inband_overhead_baboon()
{
    const string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "PEE-Diamond inband analysis: secret validation failed\n";
        return;
    }

    const string cover_path = g_input_dir + "baboon.tiff";
    TiffInfo cover = readTiff(cover_path);
    if (!validate_gray(cover, cover_path, cover.width, cover.height)) {
        cerr << "PEE-Diamond inband analysis: cover validation failed\n";
        return;
    }

    cout << "\n==== PEE-Diamond inband overhead (baboon.tiff) ====\n";
    cout << "k,share_bytes,share_bits,embed_ok,sideinfo_raw_bits(2bpp),sideinfo_rle_bits,sideinfo_raw/share,sideinfo_rle/share\n";

    const vector<int> ks = {4, 8, 16, 32, 64, 128, 256, 512};
    for (int k : ks) {
    
        auto shares = generate_shares(secret, k, k);
        const vector<unsigned char>& share = shares[0];
        const size_t share_bits = share.size() * 8;

        TiffInfo stego;
        vector<uint8_t> tags;
        bool ok = embed_share_pee_diamond(cover, share, stego, tags, 1);

        if (!ok) {
            cout << k << "," << share.size() << "," << share_bits << ",NO,-,-,-,-\n";
            continue;
        }

        // Raw packed tags require 2 bits per pixel (values 0/1/2).
        const size_t raw_bits = cover.size() * 2;

        // Simple RLE upper-bounded estimate on [value(8b), run_len(16b)] tuples.
        size_t runs = 0;
        for (size_t i = 0; i < tags.size();) {
            ++runs;
            uint8_t v = tags[i];
            size_t j = i + 1;
            while (j < tags.size() && tags[j] == v && (j - i) < 65535) ++j;
            i = j;
        }
        const size_t rle_bits = runs * 24;

        double raw_ratio = static_cast<double>(raw_bits) / static_cast<double>(share_bits);
        double rle_ratio = static_cast<double>(rle_bits) / static_cast<double>(share_bits);
        cout << k << "," << share.size() << "," << share_bits << ",YES,"
             << raw_bits << "," << rle_bits << "," << raw_ratio << "," << rle_ratio << "\n";
    }
}

static void analyze_PEE_Diamond_inband_budget_all_covers()
{
    const string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "PEE-Diamond inband budget: secret validation failed\n";
        return;
    }

    cout << "\n==== PEE-Diamond inband budget (all covers, T=1) ====\n";
    cout << "k,cover,share_bytes,embed_ok,raw_map_bits,rle_bits,total_bits_raw,total_bits_rle,raw_ok,rle_ok\n";

    const vector<int> ks = {64, 128, 256, 512};
    for (int k : ks) {
        if ((secret.size() % static_cast<size_t>(k)) != 0) continue;
        auto shares = generate_shares(secret, k, k);
        const vector<unsigned char>& share = shares[0];
        const size_t share_bits = share.size() * 8;

        for (const string& cname : g_cover_names) {
            TiffInfo cover = readTiff(g_input_dir + cname);
            if (cover.data.empty()) {
                cout << k << "," << cname << "," << share.size() << ",NO,-,-,-,-,NO,NO\n";
                continue;
            }

            TiffInfo stego;
            vector<uint8_t> tags;
            bool ok = embed_share_pee_diamond(cover, share, stego, tags, 1);
            if (!ok) {
                cout << k << "," << cname << "," << share.size() << ",NO,-,-,-,-,NO,NO\n";
                continue;
            }

            const size_t raw_bits = cover.size() * 2; // 2-bit tag per pixel
            size_t runs = 0;
            for (size_t i = 0; i < tags.size();) {
                ++runs;
                uint8_t v = tags[i];
                size_t j = i + 1;
                while (j < tags.size() && tags[j] == v && (j - i) < 65535) ++j;
                i = j;
            }
            const size_t rle_bits = runs * 24; // (value,run_len) = 8 + 16 bits

            const size_t total_raw = share_bits + raw_bits;
            const size_t total_rle = share_bits + rle_bits;
            const size_t cover_cap_bits_lsb2 = cover.size() * 2;

            cout << k << "," << cname << "," << share.size() << ",YES,"
                 << raw_bits << "," << rle_bits << ","
                 << total_raw << "," << total_rle << ","
                 << (total_raw <= cover_cap_bits_lsb2 ? "YES" : "NO") << ","
                 << (total_rle <= cover_cap_bits_lsb2 ? "YES" : "NO") << "\n";
        }
    }
}

static void reevaluate_PEE_MED_self_contained()
{
    cout << "\n==== PEE-MED self-contained reevaluation ====\n";
    cout << "cover,lsb,feasible,usable_secret_bytes\n";

    for (const string& cname : g_cover_names) {
        TiffInfo cover = readTiff(g_input_dir + cname);
        if (cover.data.empty()) continue;
        for (int lsb = 2; lsb <= 8; ++lsb) {
            PeeMedCapacityReport rep;
            bool ok = analyze_pee_med_capacity(cover, lsb, rep);
            if (!ok) {
                cout << cname << "," << lsb << ",NO,-\n";
                continue;
            }
            cout << cname << "," << lsb << ","
                 << (rep.feasible ? "YES" : "NO") << ","
                 << rep.usable_secret_bytes << "\n";
        }
    }
}

static void test_minimal_t_sweep_bs(const string& secret_name)
{
    // Sweep using the BS scheme with preprocessing applied to SECRET PIXELS
    // before sharing (not to share bytes after sharing).
    //
    // Flow per (n,k):
    //   1. bs_preprocess(secret.data, k) -> S', z, r, bs  [done once per (n,k)]
    //   2. generate_shares(S', k, n)     -> shares
    //   3. For each share i:
    //      total_embed = 3 + ceil(z/8) + share_len
    //      embed_share_mlsb_bs(cover_i, shares[i], z, r, bs) -> stego
    //   4. extract_share_mlsb_bs(stego, total_embed) -> share_i', z', r', bs'
    //   5. reconstruct_secret(shares', k) -> S''
    //   6. bs_postprocess(S'', k, r', z', bs') -> recovered secret

    string secret_path = g_input_dir + secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "test_minimal_t_sweep_bs: secret validation failed: " << secret_name << "\n"; return;
    }

    cout << "\n==== Minimal-t Sweep (mlsb_bs): 2 <= k <= n <= 8, secret="
         << secret_name << " ====\n";
    cout << "n,k,share_bytes,total_embed_bytes,min_t,z,r,cover,"
            "MSE,PSNR(dB),ShareEqual,SecretRecovered\n";

    for (int n = 2; n <= 8; ++n) {
        for (int k = 2; k <= n; ++k) {

            // Step 1: preprocess secret pixels.
            // Pad to multiple of k first.
            vector<unsigned char> padded_secret = secret.data;
            size_t padded_N = ((padded_secret.size() + static_cast<size_t>(k) - 1)
                               / static_cast<size_t>(k))
                              * static_cast<size_t>(k);
            // Use same RNG as generate_shares for padding
            uint64_t rng_pad = 0xDEADBEEFCAFEBABEULL;
            while (padded_secret.size() < padded_N) {
                rng_pad ^= rng_pad << 13; rng_pad ^= rng_pad >> 7; rng_pad ^= rng_pad << 17;
                padded_secret.push_back(static_cast<unsigned char>(rng_pad & 0xFF));
            }

            unsigned char r = 0;
            size_t z = 0;
            vector<unsigned char> bs, processed_secret;
            if (!bs_preprocess(padded_secret, k, r, z, bs, processed_secret)) {
                cout << n << "," << k << ",-,-,-,-,-,ALL,BS_OVERFLOW,-,-\n";
                continue;
            }

            // Step 2: generate shares from preprocessed secret.
            // Wrap processed_secret back into a TiffInfo for generate_shares.
            TiffInfo processed_info = secret;
            processed_info.data = processed_secret;
            // generate_shares with use_nonzero_leading=false, block size=k.
            auto shares = generate_shares(processed_info, k, n,
                                          /*use_nonzero_leading=*/false);
            size_t share_len = shares[0].size();

            size_t bs_bytes = (static_cast<size_t>(z) + 7) / 8;
            size_t total_embed = 1 + bs_bytes + share_len;

            // Use first cover to determine min_t.
            TiffInfo first_cover = readTiff(g_input_dir + g_cover_names[0]);
            int min_t = compute_min_t(total_embed, first_cover.size());

            if (min_t > 8) {
                cout << n << "," << k << "," << share_len
                     << "," << total_embed << ",INFEASIBLE,"
                     << static_cast<int>(z) << "," << static_cast<int>(r)
                     << ",ALL,-,-,-,-\n";
                continue;
            }

            // Step 3: embed share i into cover i.
            for (int i = 0; i < n; ++i) {
                string cname = g_cover_names[i % static_cast<int>(g_cover_names.size())];
                TiffInfo cover = readTiff(g_input_dir + cname);
                if (!validate_gray(cover, g_input_dir + cname,
                                   cover.width, cover.height)) {
                    cerr << "test_minimal_t_sweep_bs: cover validation failed: "
                         << cname << "\n";
                    continue;
                }

                TiffInfo stego;
                bool embed_ok = embed_share_mlsb_bs(cover, shares[i],
                                                     r, bs, stego, min_t);
                if (!embed_ok) {
                    cout << n << "," << k << "," << share_len
                         << "," << total_embed << "," << min_t << ","
                         << static_cast<int>(z) << "," << static_cast<int>(r)
                         << "," << cname << ",-,-,NO_EMBED,NO\n";
                    continue;
                }

                double mse  = compute_mse(cover.data, stego.data);
                double psnr = compute_psnr(cover.data, stego.data);

                // Step 4: extract.
                vector<unsigned char> extracted_share;
                unsigned char ex_r = 0;
                vector<unsigned char> ex_bs;
                bool ex_ok = extract_share_mlsb_bs(stego, total_embed, share_len,
                                                    extracted_share,
                                                    ex_r, ex_bs, min_t);

                if (ex_ok && extracted_share.size() > share_len)
                    extracted_share.resize(share_len);

                bool share_equal = ex_ok && (extracted_share == shares[i]);

                // Step 5+6: reconstruct and postprocess.
                bool secret_ok = false;
                if (ex_ok && i < k) {
                    vector<vector<unsigned char>> recv(shares.begin(),
                                                       shares.begin() + k);
                    recv[i] = extracted_share;
                    // Reconstruct -> S'
                    auto rec_processed = reconstruct_secret(recv, k,
                                             processed_secret.size(),
                                             /*use_nonzero_leading=*/false);
                    // Postprocess -> original secret
                    vector<unsigned char> rec_secret;
                    if (bs_postprocess(rec_processed, k, ex_r, ex_bs,
                                       rec_secret)) {
                        rec_secret.resize(secret.size());
                        secret_ok = (rec_secret == secret.data);
                    }
                } else if (ex_ok) {
                    secret_ok = true;
                }

                cout << n << "," << k << "," << share_len
                     << "," << total_embed
                     << "," << min_t
                     << "," << static_cast<int>(z)
                     << "," << static_cast<int>(r)
                     << "," << cname
                     << "," << mse << "," << psnr
                     << "," << (share_equal ? "YES" : "NO")
                     << "," << (secret_ok  ? "YES" : "NO") << "\n";
            }
        }
    }
}

static bool embed_partial_payload_mlsb_bs(const TiffInfo& cover_info,
                                          const vector<unsigned char>& share,
                                          unsigned char r,
                                          const vector<unsigned char>& bs,
                                          double rate,
                                          int t,
                                          TiffInfo& out_info)
{
    if (rate <= 0.0 || rate > 1.0) {
        return false;
    }

    vector<unsigned char> payload;
    payload.reserve(share.size() + 1 + bs.size());
    payload.insert(payload.end(), share.begin(), share.end());
    payload.push_back(r);
    payload.insert(payload.end(), bs.begin(), bs.end());

    size_t payload_len = static_cast<size_t>(floor(payload.size() * rate));
    if (payload_len == 0) payload_len = 1;
    if (payload_len > payload.size()) payload_len = payload.size();

    vector<unsigned char> truncated(payload.begin(), payload.begin() + payload_len);
    return embed_share_mlsb(cover_info, truncated, out_info, t);
}

static void test_rs_partial_embedding_rates()
{
    const vector<double> rates = {0.25, 0.5, 0.75, 1.0};
    const vector<int> ks = {3, 4, 5};

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "test_rs_partial_embedding_rates: secret validation failed\n";
        return;
    }

    cout << "\n==== RS Steganalysis: Partial Embedding Rates ====" << "\n";
    cout << "k,rate,cover,t*,cover_gap,stego_gap,delta_gap,cover_Rm,cover_Sm,"
            "stego_Rm,stego_Sm,PSNR(dB)\n";

    for (int k : ks) {
        vector<unsigned char> padded_secret = secret.data;
        size_t padded_N = ((padded_secret.size() + static_cast<size_t>(k) - 1)
                           / static_cast<size_t>(k))
                          * static_cast<size_t>(k);
        uint64_t rng_pad = 0xDEADBEEFCAFEBABEULL;
        while (padded_secret.size() < padded_N) {
            rng_pad ^= rng_pad << 13; rng_pad ^= rng_pad >> 7; rng_pad ^= rng_pad << 17;
            padded_secret.push_back(static_cast<unsigned char>(rng_pad & 0xFF));
        }

        unsigned char r = 0;
        size_t z = 0;
        vector<unsigned char> bs, processed_secret;
        if (!bs_preprocess(padded_secret, k, r, z, bs, processed_secret)) {
            cerr << "test_rs_partial_embedding_rates: bs_preprocess failed for k=" << k << "\n";
            continue;
        }

        TiffInfo processed_info = secret;
        processed_info.data = processed_secret;
        auto shares = generate_shares(processed_info, k,
                                      static_cast<int>(g_cover_names.size()),
                                      /*use_nonzero_leading=*/false);
        if (shares.empty()) {
            cerr << "test_rs_partial_embedding_rates: share generation failed for k=" << k << "\n";
            continue;
        }

        TiffInfo first_cover = readTiff(g_input_dir + g_cover_names[0]);
        size_t share_len = shares[0].size();
        size_t bs_bytes = (z + 7) / 8;
        size_t total_embed = 1 + bs_bytes + share_len;

        int t_star = compute_min_t(total_embed, first_cover.size());
        if (t_star > 8) {
            cerr << "test_rs_partial_embedding_rates: k=" << k << " infeasible t*\n";
            continue;
        }

        for (size_t i = 0; i < g_cover_names.size(); ++i) {
            const string& cname = g_cover_names[i];
            TiffInfo cover = readTiff(g_input_dir + cname);
            if (!validate_gray(cover, g_input_dir + cname, cover.width, cover.height)) {
                cerr << "test_rs_partial_embedding_rates: cover validation failed: " << cname << "\n";
                continue;
            }

            RsAnalysisResult cover_rs = compute_rs_analysis(cover);
            for (double rate : rates) {
                TiffInfo stego;
                bool embed_ok = embed_partial_payload_mlsb_bs(cover, shares[i], r, bs,
                                                              rate, t_star, stego);
                if (!embed_ok) {
                    cout << k << "," << rate << "," << cname << "," << t_star
                         << ",NA,NA,NA,NA,NA,NA,NA,NA\n";
                    continue;
                }

                RsAnalysisResult stego_rs = compute_rs_analysis(stego);
                double mse = compute_mse(cover.data, stego.data);
                double psnr = compute_psnr(cover.data, stego.data);
                double delta_gap = stego_rs.gap - cover_rs.gap;

                cout << k << "," << rate << "," << cname << "," << t_star << ","
                     << cover_rs.gap << "," << stego_rs.gap << ","
                     << delta_gap << ","
                     << cover_rs.regular_m_ratio << ","
                     << cover_rs.singular_m_ratio << ","
                     << stego_rs.regular_m_ratio << ","
                     << stego_rs.singular_m_ratio << ","
                     << psnr << "\n";

                if (k == 5 && rate == 1.0 && cname == "jetplane.tiff") {
                    string out_path = "../../doc/latex/image/jetplane_stego_k5_t2.tiff";
                    writeTiff(stego, out_path);
                }
            }
        }
    }
}

static bool embed_partial_payload_mlsb_bs_mixed(const TiffInfo& cover_info,
                                                const vector<unsigned char>& share,
                                                unsigned char r,
                                                const vector<unsigned char>& bs,
                                                double rate,
                                                size_t z,
                                                TiffInfo& out_info)
{
    if (rate <= 0.0 || rate > 1.0) {
        return false;
    }

    vector<unsigned char> payload;
    payload.reserve(share.size() + 1 + bs.size());
    payload.insert(payload.end(), share.begin(), share.end());
    payload.push_back(r);
    payload.insert(payload.end(), bs.begin(), bs.end());

    size_t payload_len = static_cast<size_t>(floor(payload.size() * rate));
    if (payload_len == 0) payload_len = 1;
    if (payload_len > payload.size()) payload_len = payload.size();

    vector<unsigned char> part_share;
    unsigned char part_r = 0;
    vector<unsigned char> part_bs;
    size_t part_z = 0;

    if (payload_len <= share.size()) {
        part_share.assign(payload.begin(), payload.begin() + payload_len);
    } else if (payload_len == share.size() + 1) {
        part_share = share;
        part_r = r;
    } else {
        part_share = share;
        part_r = r;
        part_bs.assign(payload.begin() + share.size() + 1, payload.begin() + payload_len);
        part_z = part_bs.size() * 8;
        if (part_z > z) part_z = z;
    }

    return embed_share_mlsb_bs_mixed(cover_info, part_share, part_r, part_bs, part_z, out_info);
}

static void test_rs_partial_embedding_rates_mixed()
{
    const vector<double> rates = {0.25, 0.5, 0.75, 1.0};
    const vector<int> ks = {3, 4, 5};

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "test_rs_partial_embedding_rates_mixed: secret validation failed\n";
        return;
    }

    cout << "\n==== RS Steganalysis: Partial Embedding Rates [MIXED-DEPTH] ====\n";
    cout << "k,rate,cover,t*,n_t*,cover_gap,stego_gap,delta_gap,cover_Rm,cover_Sm,"
            "stego_Rm,stego_Sm,PSNR(dB)\n";

    for (int k : ks) {
        vector<unsigned char> padded_secret = secret.data;
        size_t padded_N = ((padded_secret.size() + static_cast<size_t>(k) - 1)
                           / static_cast<size_t>(k))
                          * static_cast<size_t>(k);
        uint64_t rng_pad = 0xDEADBEEFCAFEBABEULL;
        while (padded_secret.size() < padded_N) {
            rng_pad ^= rng_pad << 13; rng_pad ^= rng_pad >> 7; rng_pad ^= rng_pad << 17;
            padded_secret.push_back(static_cast<unsigned char>(rng_pad & 0xFF));
        }

        unsigned char r = 0;
        size_t z = 0;
        vector<unsigned char> bs, processed_secret;
        if (!bs_preprocess(padded_secret, k, r, z, bs, processed_secret)) {
            cerr << "test_rs_partial_embedding_rates_mixed: bs_preprocess failed for k=" << k << "\n";
            continue;
        }

        TiffInfo processed_info = secret;
        processed_info.data = processed_secret;
        auto shares = generate_shares(processed_info, k,
                                      static_cast<int>(g_cover_names.size()),
                                      /*use_nonzero_leading=*/false);
        if (shares.empty()) {
            cerr << "test_rs_partial_embedding_rates_mixed: share generation failed for k=" << k << "\n";
            continue;
        }

        for (size_t i = 0; i < g_cover_names.size(); ++i) {
            const string& cname = g_cover_names[i];
            TiffInfo cover = readTiff(g_input_dir + cname);
            if (!validate_gray(cover, g_input_dir + cname, cover.width, cover.height)) {
                cerr << "test_rs_partial_embedding_rates_mixed: cover validation failed: " << cname << "\n";
                continue;
            }

            RsAnalysisResult cover_rs = compute_rs_analysis(cover);
            for (double rate : rates) {
                TiffInfo stego;
                bool embed_ok = embed_partial_payload_mlsb_bs_mixed(cover, shares[i], r, bs,
                                                                    rate, z, stego);
                if (!embed_ok) {
                    cout << k << "," << rate << "," << cname << ",NA,NA,NA,NA,NA,NA,NA,NA,NA,NA\n";
                    continue;
                }

                RsAnalysisResult stego_rs = compute_rs_analysis(stego);
                double mse = compute_mse(cover.data, stego.data);
                double psnr = compute_psnr(cover.data, stego.data);
                double delta_gap = stego_rs.gap - cover_rs.gap;

                size_t share_len = shares[i].size();
                size_t bs_bytes = (z + 7) / 8;
                size_t total_embed_full = 1 + bs_bytes + share_len;
                size_t payload_len = static_cast<size_t>(floor(total_embed_full * rate));
                if (payload_len == 0) payload_len = 1;
                if (payload_len > total_embed_full) payload_len = total_embed_full;
                size_t L = payload_len * 8;
                size_t P_cover = cover.size() - 6;
                size_t t_star = (L + P_cover - 1) / P_cover;
                long long diff = static_cast<long long>(L) - static_cast<long long>(P_cover) * (t_star - 1);
                size_t n_t_star = static_cast<size_t>(max(0LL, diff));

                cout << k << "," << rate << "," << cname << "," << t_star << "," << n_t_star << ","
                     << cover_rs.gap << "," << stego_rs.gap << ","
                     << delta_gap << ","
                     << cover_rs.regular_m_ratio << ","
                     << cover_rs.singular_m_ratio << ","
                     << stego_rs.regular_m_ratio << ","
                     << stego_rs.singular_m_ratio << ","
                     << psnr << "\n";

                if (k == 5 && rate == 1.0 && cname == "jetplane.tiff") {
                    string out_path = "../../doc/latex/image/jetplane_stego_k5_t2.tiff";
                    writeTiff(stego, out_path);
                }
            }
        }
    }
}

static void test_ws_spa_partial_embedding_rates_mixed()
{
    const vector<double> rates = {0.25, 0.5, 0.75, 1.0};
    const vector<int> ks = {3, 4, 5};

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "test_ws_spa_partial_embedding_rates_mixed: secret validation failed\n";
        return;
    }

    cout << "\n==== WS/SPA Steganalysis: Partial Embedding Rates [MIXED-DEPTH] ====" << "\n";
    cout << "k,rate,cover,t*,n_t*,avg_t,cover_ws,stego_ws,delta_ws,ws_est_rate,"
            "cover_spa,stego_spa,delta_spa,spa_est_rate,PSNR(dB)\n";

    for (int k : ks) {
        vector<unsigned char> padded_secret = secret.data;
        size_t padded_N = ((padded_secret.size() + static_cast<size_t>(k) - 1)
                           / static_cast<size_t>(k))
                          * static_cast<size_t>(k);
        uint64_t rng_pad = 0xDEADBEEFCAFEBABEULL;
        while (padded_secret.size() < padded_N) {
            rng_pad ^= rng_pad << 13; rng_pad ^= rng_pad >> 7; rng_pad ^= rng_pad << 17;
            padded_secret.push_back(static_cast<unsigned char>(rng_pad & 0xFF));
        }

        unsigned char r = 0;
        size_t z = 0;
        vector<unsigned char> bs, processed_secret;
        if (!bs_preprocess(padded_secret, k, r, z, bs, processed_secret)) {
            cerr << "test_ws_spa_partial_embedding_rates_mixed: bs_preprocess failed for k=" << k << "\n";
            continue;
        }

        TiffInfo processed_info = secret;
        processed_info.data = processed_secret;
        auto shares = generate_shares(processed_info, k,
                                      static_cast<int>(g_cover_names.size()),
                                      /*use_nonzero_leading=*/false);
        if (shares.empty()) {
            cerr << "test_ws_spa_partial_embedding_rates_mixed: share generation failed for k=" << k << "\n";
            continue;
        }

        for (size_t i = 0; i < g_cover_names.size(); ++i) {
            const string& cname = g_cover_names[i];
            TiffInfo cover = readTiff(g_input_dir + cname);
            if (!validate_gray(cover, g_input_dir + cname, cover.width, cover.height)) {
                cerr << "test_ws_spa_partial_embedding_rates_mixed: cover validation failed: " << cname << "\n";
                continue;
            }

            WsAnalysisResult cover_ws = compute_ws_analysis(cover);
            SpaAnalysisResult cover_spa = compute_spa_analysis(cover);
            for (double rate : rates) {
                TiffInfo stego;
                bool embed_ok = embed_partial_payload_mlsb_bs_mixed(cover, shares[i], r, bs,
                                                                    rate, z, stego);
                if (!embed_ok) {
                    cout << k << "," << rate << "," << cname << ",NA,NA,NA,"
                         << "NA,NA,NA,NA,NA,NA,NA,NA,NA\n";
                    continue;
                }

                WsAnalysisResult stego_ws = compute_ws_analysis(stego);
                SpaAnalysisResult stego_spa = compute_spa_analysis(stego);
                double psnr = compute_psnr(cover.data, stego.data);

                size_t share_len = shares[i].size();
                size_t bs_bytes = (z + 7) / 8;
                size_t total_embed_full = 1 + bs_bytes + share_len;
                size_t payload_len = static_cast<size_t>(floor(total_embed_full * rate));
                if (payload_len == 0) payload_len = 1;
                if (payload_len > total_embed_full) payload_len = total_embed_full;
                size_t L = payload_len * 8;
                size_t P_cover = cover.size() - 6;
                size_t t_star = (L + P_cover - 1) / P_cover;
                long long diff = static_cast<long long>(L) - static_cast<long long>(P_cover) * (t_star - 1);
                size_t n_t_star = static_cast<size_t>(max(0LL, diff));
                double avg_t = static_cast<double>(L) / static_cast<double>(P_cover);

                cout << k << "," << rate << "," << cname << ","
                     << t_star << "," << n_t_star << "," << avg_t << ","
                     << cover_ws.score << "," << stego_ws.score << ","
                     << (stego_ws.score - cover_ws.score) << ","
                     << stego_ws.estimated_rate << ","
                     << cover_spa.score << "," << stego_spa.score << ","
                     << (stego_spa.score - cover_spa.score) << ","
                     << stego_spa.estimated_rate << ","
                     << psnr << "\n";
            }
        }
    }
}

static void test_zero_block_count()
{
    // For the secret image and k in [2..5], count how many blocks have a
    // constant polynomial, i.e. bytes[1..k-1] are all 0x00 (regardless of
    // byte[0]).  This is the actual condition that triggers bs_preprocess.
    // Also find the best r and count ambiguous blocks.

    cout << "\n==== Constant-polynomial block count: bytes[1..k-1] all 0x00 ====\n";
    cout << "image,k,pixels,zero_pixels,blocks,const_blocks,best_r,ambiguous,z_total\n";

    vector<string> all_names;
    all_names.push_back(g_secret_name);
    for (const string& c : g_cover_names) all_names.push_back(c);
    if (find(all_names.begin(), all_names.end(), "couple.tiff") == all_names.end()) {
        all_names.push_back("couple.tiff");
    }

    for (const string& iname : all_names) {
        TiffInfo img = readTiff(g_input_dir + iname);
        if (img.data.empty()) {
            cerr << "test_zero_block_count: failed to load " << iname << "\n";
            continue;
        }
        if (img.sampleperpixel != 1 || img.bitperpixel != 8) {
            cerr << "test_zero_block_count: " << iname << " is not 8-bit grayscale\n";
            continue;
        }

        const vector<unsigned char>& px = img.data;
        size_t N = px.size();

        int zero_pixels = 0;
        for (size_t i = 0; i < N; ++i)
            if (px[i] == 0x00) ++zero_pixels;

        for (int k = 2; k <= 5; ++k) {
            vector<unsigned char> padded_secret = img.data;
            size_t padded_N = ((N + static_cast<size_t>(k) - 1)
                               / static_cast<size_t>(k))
                              * static_cast<size_t>(k);
            size_t groups = padded_N / static_cast<size_t>(k);

            // Use same RNG for padding as the actual embedding
            uint64_t rng_pad = 0xDEADBEEFCAFEBABEULL;
            while (padded_secret.size() < padded_N) {
                rng_pad ^= rng_pad << 13; rng_pad ^= rng_pad >> 7; rng_pad ^= rng_pad << 17;
                padded_secret.push_back(static_cast<unsigned char>(rng_pad & 0xFF));
            }

            unsigned char r = 0;
            size_t z = 0;
            vector<unsigned char> bs_data, processed_secret;
            if (!bs_preprocess(padded_secret, k, r, z, bs_data, processed_secret)) {
                cerr << "Error running bs_preprocess for " << iname << ", k=" << k << "\n";
                continue;
            }

            int const_blocks = 0;
            for (size_t g = 0; g < groups; ++g) {
                bool constant = true;
                for (int j = 1; j < k; ++j) {
                    size_t idx = g * static_cast<size_t>(k) + static_cast<size_t>(j);
                    if (padded_secret[idx] != 0x00) { constant = false; break; }
                }
                if (constant) ++const_blocks;
            }

            int ambiguous = static_cast<int>(z) - const_blocks;

            cout << iname << "," << k << "," << N << ","
                 << zero_pixels << "," << groups << ","
                 << const_blocks << ","
                 << static_cast<int>(r) << ","
                 << ambiguous << ","
                 << z << "\n";
        }
    }
}

static void generate_stress_images()
{
    int width = 512;
    int height = 512;
    size_t N = static_cast<size_t>(width) * height;

    // 1. all_zero
    {
        TiffInfo img;
        img.width = width;
        img.height = height;
        img.sampleperpixel = 1;
        img.bitperpixel = 8;
        img.data.assign(N, 0x00);
        writeTiff(img, "../out/512/stress_secret_all_zero.tiff");
    }

    // 2. k2_tail_zero
    {
        TiffInfo img;
        img.width = width;
        img.height = height;
        img.sampleperpixel = 1;
        img.bitperpixel = 8;
        img.data.resize(N);
        for (size_t i = 0; i < N; ++i) {
            img.data[i] = (i % 2 == 1) ? 0x00 : 128;
        }
        writeTiff(img, "../out/512/stress_secret_k2_tail_zero.tiff");
    }

    // 3. large_flat_regions
    {
        TiffInfo img;
        img.width = width;
        img.height = height;
        img.sampleperpixel = 1;
        img.bitperpixel = 8;
        img.data.resize(N);
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                int block_r = r / 64;
                int block_c = c / 64;
                unsigned char val = static_cast<unsigned char>((block_r * 8 + block_c) * 3);
                img.data[static_cast<size_t>(r) * width + c] = val;
            }
        }
        writeTiff(img, "../out/512/stress_secret_large_flat_regions.tiff");
    }

    // 4. periodic_blocks
    {
        TiffInfo img;
        img.width = width;
        img.height = height;
        img.sampleperpixel = 1;
        img.bitperpixel = 8;
        img.data.resize(N);
        for (size_t i = 0; i < N; ++i) {
            img.data[i] = static_cast<unsigned char>((i % 4) * 60);
        }
        writeTiff(img, "../out/512/stress_secret_periodic_blocks.tiff");
    }

    // 5. checker_sparse
    {
        TiffInfo img;
        img.width = width;
        img.height = height;
        img.sampleperpixel = 1;
        img.bitperpixel = 8;
        img.data.resize(N);
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                int br = r / 8;
                int bc = c / 8;
                img.data[static_cast<size_t>(r) * width + c] = ((br + bc) % 2 == 1) ? 255 : 0x00;
            }
        }
        writeTiff(img, "../out/512/stress_secret_checker_sparse.tiff");
    }
}

static void test_bs_stress_synthetic()
{
    generate_stress_images();

    vector<pair<string, string>> patterns = {
        {"all_zero", "stress_secret_all_zero.tiff"},
        {"k2_tail_zero", "stress_secret_k2_tail_zero.tiff"},
        {"large_flat_regions", "stress_secret_large_flat_regions.tiff"},
        {"periodic_blocks", "stress_secret_periodic_blocks.tiff"},
        {"checker_sparse", "stress_secret_checker_sparse.tiff"}
    };

    cout << "==== BS Preprocessing Stress Test ====\n";
    cout << "pattern,k,pixels,groups,status,z,r,bs_bytes,share_bytes,total_embed,t*,SecretRecovered\n";

    for (const auto& p : patterns) {
        string path = "../out/512/" + p.second;
        TiffInfo secret = readTiff(path);
        if (secret.data.empty()) {
            cerr << "Failed to read " << path << "\n";
            continue;
        }

        for (int k = 2; k <= 8; ++k) {
            // Step 1: preprocess secret pixels.
            vector<unsigned char> padded_secret = secret.data;
            size_t padded_N = ((padded_secret.size() + static_cast<size_t>(k) - 1)
                               / static_cast<size_t>(k))
                              * static_cast<size_t>(k);
            uint64_t rng_pad = 0xDEADBEEFCAFEBABEULL;
            while (padded_secret.size() < padded_N) {
                rng_pad ^= rng_pad << 13; rng_pad ^= rng_pad >> 7; rng_pad ^= rng_pad << 17;
                padded_secret.push_back(static_cast<unsigned char>(rng_pad & 0xFF));
            }

            unsigned char r = 0;
            size_t z = 0;
            vector<unsigned char> bs, processed_secret;
            if (!bs_preprocess(padded_secret, k, r, z, bs, processed_secret)) {
                cout << p.first << "," << k << "," << secret.size() << ","
                     << padded_N / k << ",BS_OVERFLOW,-,-,-,-,-,-,NO\n";
                continue;
            }

            // Step 2: generate shares.
            TiffInfo processed_info = secret;
            processed_info.data = processed_secret;
            auto shares = generate_shares(processed_info, k, k, /*use_nonzero_leading=*/false);
            size_t share_len = shares[0].size();

            size_t bs_bytes = (z + 7) / 8;
            size_t total_embed = 1 + bs_bytes + share_len;

            // Compute min_t (using cover[0] placeholder)
            TiffInfo cover = readTiff(g_input_dir + g_cover_names[0]);
            int min_t = compute_min_t(total_embed, cover.size());

            // Verify recoverability
            // Reconstruct S'
            auto rec_processed = reconstruct_secret(shares, k,
                                     processed_secret.size(),
                                     /*use_nonzero_leading=*/false);
            // Postprocess -> original secret
            vector<unsigned char> rec_secret;
            bool secret_ok = false;
            if (bs_postprocess(rec_processed, k, r, bs, rec_secret)) {
                rec_secret.resize(secret.size());
                secret_ok = (rec_secret == secret.data);
            }

            cout << p.first << "," << k << "," << secret.size() << ","
                 << padded_N / k << ",OK," << z << "," << static_cast<int>(r) << ","
                 << bs_bytes << "," << share_len << "," << total_embed << ","
                 << (min_t > 8 ? "INFEASIBLE" : to_string(min_t)) << ","
                 << (secret_ok ? "YES" : "NO") << "\n";
        }
    }
}

static void test_minimal_t_sweep_bs_mixed(const string& secret_name)
{
    string secret_path = g_input_dir + secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "test_minimal_t_sweep_bs_mixed: secret validation failed: " << secret_name << "\n"; return;
    }

    cout << "\n==== Minimal-t Sweep (mlsb_bs_mixed): 2 <= k <= n <= 8, secret="
         << secret_name << " ====\n";
    cout << "n,k,share_bytes,total_embed_bytes,t_star,n_t_star,z,r,cover,"
            "MSE,PSNR(dB),ShareEqual,SecretRecovered\n";

    for (int n = 2; n <= 8; ++n) {
        for (int k = 2; k <= n; ++k) {

            // Step 1: preprocess secret pixels.
            vector<unsigned char> padded_secret = secret.data;
            size_t padded_N = ((padded_secret.size() + static_cast<size_t>(k) - 1)
                               / static_cast<size_t>(k))
                              * static_cast<size_t>(k);
            uint64_t rng_pad = 0xDEADBEEFCAFEBABEULL;
            while (padded_secret.size() < padded_N) {
                rng_pad ^= rng_pad << 13; rng_pad ^= rng_pad >> 7; rng_pad ^= rng_pad << 17;
                padded_secret.push_back(static_cast<unsigned char>(rng_pad & 0xFF));
            }

            unsigned char r = 0;
            size_t z = 0;
            vector<unsigned char> bs, processed_secret;
            if (!bs_preprocess(padded_secret, k, r, z, bs, processed_secret)) {
                cout << n << "," << k << ",-,-,-,-,-,-,ALL,BS_OVERFLOW,-,-\n";
                continue;
            }

            // Step 2: generate shares from preprocessed secret.
            TiffInfo processed_info = secret;
            processed_info.data = processed_secret;
            auto shares = generate_shares(processed_info, k, n, /*use_nonzero_leading=*/false);
            size_t share_len = shares[0].size();

            size_t bs_bytes = (z + 7) / 8;
            size_t S_total = share_len + 1 + bs_bytes;
            size_t L = S_total * 8;

            // Use first cover to determine mixed parameters.
            TiffInfo first_cover = readTiff(g_input_dir + g_cover_names[0]);
            size_t P = first_cover.size() - 6;
            size_t t_star = (L + P - 1) / P;

            if (t_star > 8) {
                cout << n << "," << k << "," << share_len
                     << "," << S_total << ",INFEASIBLE,-,"
                     << static_cast<int>(z) << "," << static_cast<int>(r)
                     << ",ALL,-,-,-,-\n";
                continue;
            }

            long long diff = static_cast<long long>(L) - static_cast<long long>(P) * (t_star - 1);
            size_t n_t_star = static_cast<size_t>(max(0LL, diff));

            // Step 3: embed share i into cover i.
            for (int i = 0; i < n; ++i) {
                string cname = g_cover_names[i % static_cast<int>(g_cover_names.size())];
                TiffInfo cover = readTiff(g_input_dir + cname);
                if (!validate_gray(cover, g_input_dir + cname, cover.width, cover.height)) {
                    cerr << "test_minimal_t_sweep_bs_mixed: cover validation failed: " << cname << "\n";
                    continue;
                }

                TiffInfo stego;
                bool embed_ok = embed_share_mlsb_bs_mixed(cover, shares[i], r, bs, z, stego);
                if (!embed_ok) {
                    cout << n << "," << k << "," << share_len
                         << "," << S_total << "," << t_star << "," << n_t_star << ","
                         << static_cast<int>(z) << "," << static_cast<int>(r)
                         << "," << cname << ",-,-,NO_EMBED,NO\n";
                    continue;
                }

                double mse  = compute_mse(cover.data, stego.data);
                double psnr = compute_psnr(cover.data, stego.data);

                // Step 4: extract.
                vector<unsigned char> extracted_share;
                unsigned char ex_r = 0;
                vector<unsigned char> ex_bs;
                bool ex_ok = extract_share_mlsb_bs_mixed(stego, share_len, extracted_share, ex_r, ex_bs);

                bool share_equal = ex_ok && (extracted_share == shares[i]);

                // Reconstruct secret using first k shares (replace share i with extracted)
                bool secret_ok = false;
                if (ex_ok && i < k) {
                    vector<vector<unsigned char>> recv(shares.begin(), shares.begin() + k);
                    recv[i] = extracted_share;
                    auto rec_processed = reconstruct_secret(recv, k, processed_secret.size(), /*use_nonzero_leading=*/false);
                    vector<unsigned char> rec_secret;
                    if (bs_postprocess(rec_processed, k, ex_r, ex_bs, rec_secret)) {
                        rec_secret.resize(secret.size());
                        secret_ok = (rec_secret == secret.data);
                    }
                } else if (ex_ok) {
                    secret_ok = true; // share i >= k is not needed for reconstruction, extract ok is enough
                }

                cout << n << "," << k << "," << share_len
                     << "," << S_total << "," << t_star << "," << n_t_star << ","
                     << static_cast<int>(z) << "," << static_cast<int>(r)
                     << "," << cname
                     << "," << mse << "," << psnr
                     << "," << (share_equal ? "YES" : "NO")
                     << "," << (secret_ok  ? "YES" : "NO") << "\n";
            }
        }
    }
}

static void test_minimal_t_sweep_all(const string& secret_name)
{
    string secret_path = g_input_dir + secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "test_minimal_t_sweep_all: secret validation failed: " << secret_name << "\n"; return;
    }

    cout << "\n==== Minimal-t Sweep (mlsb_all): 2 <= k <= n <= 8, secret="
         << secret_name << " ====\n";
    cout << "n,k,share_bytes,total_embed_bytes,t_star,n_t_star,z,r,cover,"
            "MSE,PSNR(dB),ShareEqual,SecretRecovered\n";

    for (int n = 2; n <= 8; ++n) {
        for (int k = 2; k <= n; ++k) {

            // Step 1: preprocess secret pixels.
            vector<unsigned char> padded_secret = secret.data;
            size_t padded_N = ((padded_secret.size() + static_cast<size_t>(k) - 1)
                               / static_cast<size_t>(k))
                              * static_cast<size_t>(k);
            uint64_t rng_pad = 0xDEADBEEFCAFEBABEULL;
            while (padded_secret.size() < padded_N) {
                rng_pad ^= rng_pad << 13; rng_pad ^= rng_pad >> 7; rng_pad ^= rng_pad << 17;
                padded_secret.push_back(static_cast<unsigned char>(rng_pad & 0xFF));
            }

            unsigned char r = 0;
            size_t z = 0;
            vector<unsigned char> bs, processed_secret;
            if (!bs_preprocess(padded_secret, k, r, z, bs, processed_secret)) {
                cout << n << "," << k << ",-,-,-,-,-,-,ALL,BS_OVERFLOW,-,-\n";
                continue;
            }

            // Step 2: Consolidate metadata (r and bs) into processed_secret to form S_shared
            vector<unsigned char> S_shared = processed_secret;
            S_shared.push_back(r);
            S_shared.insert(S_shared.end(), bs.begin(), bs.end());

            // Pad S_shared to a multiple of k
            size_t padded_shared_N = ((S_shared.size() + static_cast<size_t>(k) - 1)
                                      / static_cast<size_t>(k))
                                     * static_cast<size_t>(k);
            uint64_t rng_shared = 0xDEADBEEFCAFEBABEULL;
            while (S_shared.size() < padded_shared_N) {
                rng_shared ^= rng_shared << 13; rng_shared ^= rng_shared >> 7; rng_shared ^= rng_shared << 17;
                S_shared.push_back(static_cast<unsigned char>(rng_shared & 0xFF));
            }

            // Generate main shares from S_shared
            TiffInfo shared_info = secret;
            shared_info.data = S_shared;
            shared_info.width = static_cast<uint32_t>(S_shared.size());
            shared_info.height = 1;
            shared_info.sampleperpixel = 1;
            auto main_shares = generate_shares(shared_info, k, n, /*use_nonzero_leading=*/false);
            size_t main_share_len = main_shares[0].size();

            // Share z using standard (k,n) Shamir sharing (3 bytes each)
            vector<vector<unsigned char>> z_shares;
            share_z(z, k, n, z_shares);

            // Calculate total embed bytes for stats comparison.
            size_t S_total = main_share_len;
            size_t L = S_total * 8;

            // Use first cover to determine mixed parameters.
            TiffInfo first_cover = readTiff(g_input_dir + g_cover_names[0]);
            size_t P = first_cover.size() - 6;
            size_t t_star = (L + P - 1) / P;

            if (t_star > 8) {
                cout << n << "," << k << "," << main_share_len
                     << "," << S_total << ",INFEASIBLE,-,"
                     << static_cast<int>(z) << "," << static_cast<int>(r)
                     << ",ALL,-,-,-,-\n";
                continue;
            }

            long long diff = static_cast<long long>(L) - static_cast<long long>(P) * (t_star - 1);
            size_t n_t_star = static_cast<size_t>(max(0LL, diff));

            // Embed for each participant
            vector<TiffInfo> stego_images(n);
            vector<bool> embed_ok(n, false);
            for (int i = 0; i < n; ++i) {
                string cname = g_cover_names[i % static_cast<int>(g_cover_names.size())];
                TiffInfo cover = readTiff(g_input_dir + cname);
                if (!validate_gray(cover, g_input_dir + cname, cover.width, cover.height)) {
                    cerr << "test_minimal_t_sweep_all: cover validation failed: " << cname << "\n";
                    continue;
                }

                embed_ok[i] = embed_share_mlsb_all(cover, main_shares[i], z_shares[i], z, stego_images[i]);
            }

            // Verify recovery for each participant position
            for (int i = 0; i < n; ++i) {
                string cname = g_cover_names[i % static_cast<int>(g_cover_names.size())];
                if (!embed_ok[i]) {
                    cout << n << "," << k << "," << main_share_len
                         << "," << S_total << "," << t_star << "," << n_t_star << ","
                         << static_cast<int>(z) << "," << static_cast<int>(r)
                         << "," << cname << ",-,-,NO_EMBED,NO\n";
                    continue;
                }

                double mse  = compute_mse(readTiff(g_input_dir + cname).data, stego_images[i].data);
                double psnr = compute_psnr(readTiff(g_input_dir + cname).data, stego_images[i].data);

                // Reconstruct using first k stego images
                bool secret_ok = false;
                bool share_equal = false;

                // Step A: Extract z_shares from first k stego images
                vector<vector<unsigned char>> collected_z_shares(k);
                vector<int> participant_ids(k);
                bool header_extract_ok = true;
                for (int idx = 0; idx < k; ++idx) {
                    int p_idx = (i < k) ? idx : (idx == k - 1 ? i : idx);
                    participant_ids[idx] = p_idx + 1; // 1-based index
                    if (!extract_z_share_mlsb_all(stego_images[p_idx], collected_z_shares[idx])) {
                        header_extract_ok = false;
                    }
                }

                if (header_extract_ok) {
                    // Step B: Reconstruct z
                    size_t rec_z = reconstruct_z(collected_z_shares, k, participant_ids);
                    if (rec_z != z) {
                        cout << "[ERROR] z mismatch! original: " << z << ", reconstructed: " << rec_z << "\n";
                    }

                    // Step C: Extract main shares
                    vector<vector<unsigned char>> collected_main_shares(k);
                    bool main_extract_ok = true;
                    for (int idx = 0; idx < k; ++idx) {
                        int p_idx = (i < k) ? idx : (idx == k - 1 ? i : idx);
                        if (!extract_main_share_mlsb_all(stego_images[p_idx], rec_z, main_share_len, collected_main_shares[idx])) {
                            main_extract_ok = false;
                        }
                    }

                    if (main_extract_ok) {
                        // Verify the current participant's extracted main share is correct
                        int current_idx_in_k = (i < k) ? i : k - 1;
                        share_equal = (collected_main_shares[current_idx_in_k] == main_shares[i]);

                        // Step D: Lagrange interpolate main shares to get S_shared
                        auto rec_shared = reconstruct_secret(collected_main_shares, k, S_shared.size(), participant_ids, /*use_nonzero_leading=*/false);
                        
                        if (i == 0) {
                            cout << "[DEBUG] S_shared size: " << S_shared.size() << ", rec_shared size: " << rec_shared.size() << "\n";
                            cout << "[DEBUG] S_shared: ";
                            for (size_t d = 0; d < 10 && d < S_shared.size(); ++d) cout << (int)S_shared[d] << " ";
                            cout << "\n[DEBUG] rec_shared: ";
                            for (size_t d = 0; d < 10 && d < rec_shared.size(); ++d) cout << (int)rec_shared[d] << " ";
                            cout << "\n";
                            cout << "[DEBUG] S_shared r: " << (int)S_shared[processed_secret.size()] 
                                 << ", rec_shared r: " << (int)rec_shared[processed_secret.size()] << "\n";
                        }

                        // Step E: De-concatenate S_shared -> S', r, bs
                        if (rec_shared.size() >= processed_secret.size() + 1 + bs.size()) {
                            vector<unsigned char> rec_processed(rec_shared.begin(), rec_shared.begin() + processed_secret.size());
                            unsigned char rec_r = rec_shared[processed_secret.size()];
                            vector<unsigned char> rec_bs(rec_shared.begin() + processed_secret.size() + 1, rec_shared.begin() + processed_secret.size() + 1 + bs.size());

                            // Step F: CPP postprocess
                            vector<unsigned char> rec_secret;
                            if (bs_postprocess(rec_processed, k, rec_r, rec_bs, rec_secret)) {
                                rec_secret.resize(secret.size());
                                secret_ok = (rec_secret == secret.data);
                            }
                        }
                    }
                }

                cout << n << "," << k << "," << main_share_len
                     << "," << S_total << "," << t_star << "," << n_t_star << ","
                     << static_cast<int>(z) << "," << static_cast<int>(r)
                     << "," << cname
                     << "," << mse << "," << psnr
                     << "," << (share_equal ? "YES" : "NO")
                     << "," << (secret_ok  ? "YES" : "NO") << "\n";
            }
        }
    }
}

static void test_multisize_sweep_mixed()
{
    cout << "\n==== Multi-size Sweep (MIXED-DEPTH): image sizes 256, 512, 1024 ====\n";
    cout << "size,k,share_bytes,total_embed,t*,cover,MSE,PSNR(dB),ShareEqual,SecretRecovered\n";
    
    vector<int> sizes = {256, 512, 1024};
    vector<int> ks = {3, 4, 5};
    
    for (int sz : sizes) {
        string input_dir = "../images/USC-SIPI/" + to_string(sz) + "/";
        string secret_path = input_dir + "woman.tiff";
        TiffInfo secret = readTiff(secret_path);
        if (secret.data.empty()) {
            cerr << "multisize_mixed: cannot load secret for size " << sz << endl;
            continue;
        }
        
        string cover_path = input_dir + "jetplane.tiff";
        TiffInfo cover = readTiff(cover_path);
        if (cover.data.empty()) {
            cerr << "multisize_mixed: cannot load cover for size " << sz << endl;
            continue;
        }
        
        const size_t N = secret.size();
        
        for (int k : ks) {
            int n = k; // threshold k, n=k participants
            
            // Step 1: preprocess
            vector<unsigned char> padded_secret = secret.data;
            size_t padded_N = ((padded_secret.size() + static_cast<size_t>(k) - 1)
                               / static_cast<size_t>(k))
                              * static_cast<size_t>(k);
            uint64_t rng_pad = 0xDEADBEEFCAFEBABEULL;
            while (padded_secret.size() < padded_N) {
                rng_pad ^= rng_pad << 13; rng_pad ^= rng_pad >> 7; rng_pad ^= rng_pad << 17;
                padded_secret.push_back(static_cast<unsigned char>(rng_pad & 0xFF));
            }
            
            unsigned char r = 0;
            size_t z = 0;
            vector<unsigned char> bs, processed_secret;
            if (!bs_preprocess(padded_secret, k, r, z, bs, processed_secret)) {
                cerr << "Preprocessing failed for size " << sz << " k=" << k << endl;
                continue;
            }
            
            // Step 2: generate shares
            TiffInfo processed_info = secret;
            processed_info.data = processed_secret;
            auto shares = generate_shares(processed_info, k, n, /*use_nonzero_leading=*/false);
            size_t share_len = shares[0].size();
            
            size_t bs_bytes = (z + 7) / 8;
            size_t S_total = share_len + 1 + bs_bytes;
            
            TiffInfo stego;
            bool ok = embed_share_mlsb_bs_mixed(cover, shares[0], r, bs, z, stego);
            if (!ok) {
                cerr << "Embedding failed for size " << sz << " k=" << k << endl;
                continue;
            }
            
            double mse = compute_mse(cover.data, stego.data);
            double psnr = compute_psnr(cover.data, stego.data);
            
            // Extract
            vector<unsigned char> extracted_share;
            unsigned char ex_r = 0;
            vector<unsigned char> ex_bs;
            bool ex_ok = extract_share_mlsb_bs_mixed(stego, share_len, extracted_share, ex_r, ex_bs);
            bool share_equal = ex_ok && (extracted_share == shares[0]);
            
            bool secret_ok = false;
            if (ex_ok) {
                vector<vector<unsigned char>> recv(shares.begin(), shares.begin() + k);
                recv[0] = extracted_share;
                auto rec_processed = reconstruct_secret(recv, k, processed_secret.size(), /*use_nonzero_leading=*/false);
                vector<unsigned char> rec_secret;
                if (bs_postprocess(rec_processed, k, ex_r, ex_bs, rec_secret)) {
                    rec_secret.resize(N);
                    secret_ok = (rec_secret == secret.data);
                }
            }
            
            // Calculate t* from the formula in the stego header (P = cover.size() - 6)
            size_t P = cover.size() - 6;
            size_t L = S_total * 8;
            size_t t_star = (L + P - 1) / P;
            
            cout << sz << "," << k << "," << share_len << "," << S_total << "," << t_star
                 << ",jetplane.tiff," << mse << "," << psnr << ","
                 << (share_equal ? "YES" : "NO") << ","
                 << (secret_ok ? "YES" : "NO") << endl;
        }
    }
}

static void test_comparison_article6_mixed()
{
    // Covers matching article.6 experimental setup
    const vector<string> cover_names = {
        "jetplane.tiff",
        "house.tiff",
        "girl.tiff",
        "baboon.tiff",
        "airplane.tiff"
    };

    string secret_path = g_input_dir + g_secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (!validate_gray(secret, secret_path, secret.width, secret.height)) {
        cerr << "Comparison test mixed: secret validation failed\n"; return;
    }
    const size_t N = secret.size();

    cout << "\n==== Comparison with Lin et al. baseline (image_size=" << g_image_size << ") [MIXED-DEPTH] ====\n";
    cout << "BS scheme with mixed-depth embedding.\n";
    cout << "k=2 skipped (degenerate).\n\n";

    const vector<tuple<int,int,int>> cases = {
        {4, 3, 4},
        {4, 4, 4},
        {5, 3, 5},
        {5, 4, 5},
        {5, 5, 5},
    };

    cout << "n,k,t*,n_t*,share_bytes,total_embed,cover,MSE,PSNR(dB),SSIM,ShareEqual,SecretRecovered\n";

    for (const auto& [n, k, num_covers] : cases) {
        // Step 1: preprocess secret pixels.
        vector<unsigned char> padded_secret = secret.data;
        size_t padded_N = ((padded_secret.size() + static_cast<size_t>(k) - 1)
                           / static_cast<size_t>(k))
                          * static_cast<size_t>(k);
        uint64_t rng_pad = 0xDEADBEEFCAFEBABEULL;
        while (padded_secret.size() < padded_N) {
            rng_pad ^= rng_pad << 13; rng_pad ^= rng_pad >> 7; rng_pad ^= rng_pad << 17;
            padded_secret.push_back(static_cast<unsigned char>(rng_pad & 0xFF));
        }

        unsigned char r = 0;
        size_t z = 0;
        vector<unsigned char> bs, processed_secret;
        if (!bs_preprocess(padded_secret, k, r, z, bs, processed_secret)) {
            cout << n << "," << k << ",-,-,-,-,ALL,BS_OVERFLOW,-,-,-,-\n";
            continue;
        }

        // Step 2: generate BS-mode shares from preprocessed secret.
        TiffInfo processed_info = secret;
        processed_info.data = processed_secret;
        auto shares = generate_shares(processed_info, k, n, /*use_nonzero_leading=*/false);
        size_t share_len = shares[0].size();

        size_t bs_bytes_count = (z + 7) / 8;
        size_t total_embed = share_len + 1 + bs_bytes_count;

        for (int ci = 0; ci < num_covers && ci < static_cast<int>(cover_names.size()); ++ci) {
            TiffInfo cover = readTiff(g_input_dir + cover_names[ci]);
            if (!validate_gray(cover, g_input_dir + cover_names[ci], cover.width, cover.height)) continue;

            size_t P = cover.size() - 6;
            size_t t_star = (total_embed * 8 + P - 1) / P;
            long long diff = static_cast<long long>(total_embed * 8) - static_cast<long long>(P) * (t_star - 1);
            size_t n_t_star = static_cast<size_t>(max(0LL, diff));

            TiffInfo stego;
            bool embed_ok = embed_share_mlsb_bs_mixed(cover, shares[ci], r, bs, z, stego);
            if (!embed_ok) {
                cout << n << "," << k << "," << t_star << "," << n_t_star << "," << share_len
                     << "," << total_embed << "," << cover_names[ci]
                     << ",-,-,-,NO_EMBED,NO\n";
                continue;
            }

            double mse  = compute_mse(cover.data, stego.data);
            double psnr = compute_psnr(cover.data, stego.data);
            double ssim = compute_ssim(cover.data, stego.data);

            vector<unsigned char> extracted_share;
            unsigned char ex_r = 0;
            vector<unsigned char> ex_bs;
            bool ex_ok = extract_share_mlsb_bs_mixed(stego, share_len, extracted_share, ex_r, ex_bs);

            bool share_equal = ex_ok && (extracted_share == shares[ci]);

            bool secret_ok = false;
            if (ex_ok) {
                vector<vector<unsigned char>> recv(shares.begin(), shares.begin() + k);
                if (ci < k) recv[ci] = extracted_share;
                auto rec_processed = reconstruct_secret(recv, k, processed_secret.size(), /*use_nonzero_leading=*/false);
                vector<unsigned char> rec_secret;
                if (bs_postprocess(rec_processed, k, ex_r, ex_bs, rec_secret)) {
                    rec_secret.resize(N);
                    secret_ok = (rec_secret == secret.data);
                }
            }

            cout << n << "," << k << "," << t_star << "," << n_t_star << "," << share_len
                 << "," << total_embed
                 << "," << cover_names[ci]
                 << "," << mse << "," << psnr << "," << ssim
                 << "," << (share_equal ? "YES" : "NO")
                 << "," << (secret_ok  ? "YES" : "NO") << "\n";
        }
    }
}

// ---------------------------------------------------------------------------
// test_block_shift_repeat_risk
//   For each k in [2..8], groups the secret into blocks of k pixels,
//   then counts:
//     - identical block pairs at distance exactly 255 (wraparound risk)
//     - the maximum run length of consecutive identical blocks
// ---------------------------------------------------------------------------
static void test_block_shift_repeat_risk(const string& secret_name)
{
    string secret_path = g_input_dir + secret_name;
    TiffInfo secret = readTiff(secret_path);
    if (secret.data.empty()) {
        cerr << "test_block_shift_repeat_risk: cannot load " << secret_name << "\n";
        return;
    }

    cout << "\n==== Block-Shifted Repeat Risk: " << secret_name << " ====\n";
    cout << "k,blocks_per_group,identical_pairs_at_255,max_repeated_run\n";

    for (int k = 2; k <= 8; ++k) {
        const size_t N = secret.size();
        size_t block_size = static_cast<size_t>(k);
        size_t groups = (N + block_size - 1) / block_size;

        // Build padded pixel sequence (same padding as generate_shares)
        vector<unsigned char> padded = secret.data;
        uint64_t rng = 0xDEADBEEFCAFEBABEULL;
        while (padded.size() < groups * block_size) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            padded.push_back(static_cast<unsigned char>(rng & 0xFF));
        }

        // Extract blocks
        vector<vector<unsigned char>> blocks(groups);
        for (size_t g = 0; g < groups; ++g) {
            blocks[g].assign(padded.begin() + g * block_size,
                             padded.begin() + g * block_size + block_size);
        }

        // Count identical pairs at distance exactly 255
        int pairs_at_255 = 0;
        for (size_t g = 0; g + 255 < groups; ++g) {
            if (blocks[g] == blocks[g + 255])
                ++pairs_at_255;
        }

        // Find maximum run length of consecutive identical blocks
        int max_run = 1;
        int cur_run = 1;
        for (size_t g = 1; g < groups; ++g) {
            if (blocks[g] == blocks[g - 1]) {
                ++cur_run;
                if (cur_run > max_run) max_run = cur_run;
            } else {
                cur_run = 1;
            }
        }

        cout << k << "," << groups << "," << pairs_at_255 << "," << max_run << "\n";
    }
}

int main(int argc, char* argv[])
{
    bool visual_attack_only = false;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--visual-attack-only" || arg == "--generate-visual-attack") {
            visual_attack_only = true;
        } else if (arg == "--shuffled") {
            g_use_shuffled = true;
            cout << "[INFO] Shuffled mixed-depth mode ENABLED.\n";
        }
    }

    if (visual_attack_only) {
        generate_visual_attack_figure("jetplane.tiff", {2, 3, 4, 5, 8});
        return 0;
    }

    //test_secret_sharing();
    //test_secret_sharing_sweep();
    //test_mlsb_parameter_sweep();
    //test_minimal_t_sweep();
    test_minimal_t_sweep_bs("woman.tiff");
    test_minimal_t_sweep_bs_mixed("woman.tiff");
    test_minimal_t_sweep_all("woman.tiff");
    test_minimal_t_sweep_bs("couple.tiff");
    test_minimal_t_sweep_bs_mixed("couple.tiff");
    test_minimal_t_sweep_all("couple.tiff");
    generate_visual_attack_figure("jetplane.tiff", {2, 3, 4, 5, 8});
    test_rs_partial_embedding_rates();
    test_rs_partial_embedding_rates_mixed();
    test_ws_spa_partial_embedding_rates_mixed();
    test_zero_block_count();
    test_comparison_article6();
    test_comparison_article6_mixed();
    test_multisize_sweep_mixed();
    //minimal_k(false);
    test_bs_stress_synthetic();
    test_block_shift_repeat_risk("woman.tiff");
    test_block_shift_repeat_risk("couple.tiff");
    return 0;
}
