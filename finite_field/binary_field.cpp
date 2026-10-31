#include "binary_field.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <mutex>
#include <vector>

namespace
{
    // Tables initialized on first use (thread-safe)
    static std::once_flag g_tables_init;
    static std::array<std::array<unsigned char, 256>, 256> g_mul_table;
    static std::array<unsigned char, 256> g_inv_table;

    // naive multiplication used only during table init
    inline unsigned char gf_mul_no_table(unsigned char a, unsigned char b)
    {
        unsigned char p = 0;
        while (b)
        {
            if (b & 1u)
                p ^= a;
            bool hi = (a & 0x80u) != 0;
            a <<= 1;
            if (hi)
                a ^= 0x1Bu;
            b >>= 1;
        }
        return p;
    }

    void init_tables()
    {
        // build multiplication table
        for (int a = 0; a < 256; ++a)
        {
            for (int b = 0; b < 256; ++b)
                g_mul_table[a][b] = gf_mul_no_table(static_cast<unsigned char>(a), static_cast<unsigned char>(b));
        }

        // build inverse table (0 -> 0 by convention)
        g_inv_table[0] = 0;
        for (int a = 1; a < 256; ++a)
        {
            unsigned char inv = 0;
            for (int b = 1; b < 256; ++b)
            {
                if (g_mul_table[a][b] == 1)
                {
                    inv = static_cast<unsigned char>(b);
                    break;
                }
            }
            g_inv_table[a] = inv;
        }
    }

    // ensure tables are initialized before use
    inline void ensure_tables()
    {
        std::call_once(g_tables_init, init_tables);
    }
} // namespace

// --- Galois Field GF(2^8) with modulus 0x11B ---
// addition = XOR
unsigned char gf_add(unsigned char a, unsigned char b)
{
    return static_cast<unsigned char>(a ^ b);
}

unsigned char gf_mul(unsigned char a, unsigned char b)
{
    ensure_tables();
    return g_mul_table[a][b];
}

unsigned char gf_pow(unsigned char a, unsigned int e)
{
    ensure_tables();
    if (e == 0) return 1;
    if (a == 0) return 0;

    unsigned char result = 1;
    unsigned char base = a;
    while (e)
    {
        if (e & 1u)
            result = g_mul_table[result][base];
        base = g_mul_table[base][base];
        e >>= 1u;
    }
    return result;
}

unsigned char gf_inv(unsigned char a)
{
    ensure_tables();
    return g_inv_table[a]; // 0 -> 0 by convention
}

// Polynomial utilities over GF(2^8)
// polynomials represented as vector<unsigned char> coefficients, lowest degree first

void poly_add(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b, std::vector<unsigned char>& out)
{
    out.clear();
    size_t n = std::max(a.size(), b.size());
    if (n == 0) return;
    out.resize(n, 0);
    for (size_t i = 0; i < a.size(); ++i)
        out[i] = static_cast<unsigned char>(out[i] ^ a[i]);
    for (size_t i = 0; i < b.size(); ++i)
        out[i] = static_cast<unsigned char>(out[i] ^ b[i]);

    // trim trailing zeros
    while (!out.empty() && out.back() == 0) out.pop_back();
}

void poly_mul(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b, std::vector<unsigned char>& out)
{
    out.clear();
    if (a.empty() || b.empty())
        return;

    ensure_tables();

    out.assign(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
    {
        unsigned char ai = a[i];
        if (ai == 0) continue;
        for (size_t j = 0; j < b.size(); ++j)
        {
            unsigned char bj = b[j];
            if (bj == 0) continue;
            out[i + j] = static_cast<unsigned char>(out[i + j] ^ g_mul_table[ai][bj]);
        }
    }

    // trim trailing zeros
    while (!out.empty() && out.back() == 0) out.pop_back();
}

void poly_scale(const std::vector<unsigned char>& p, unsigned char scalar, std::vector<unsigned char>& out)
{
    out.clear();
    if (p.empty())
        return;

    if (scalar == 0)
    {
        // zero polynomial -> leave out empty (normalized)
        return;
    }

    ensure_tables();

    out.resize(p.size());
    for (size_t i = 0; i < p.size(); ++i)
    {
        if (p[i] == 0) out[i] = 0;
        else out[i] = g_mul_table[scalar][p[i]];
    }

    // trim trailing zeros
    while (!out.empty() && out.back() == 0) out.pop_back();
}

unsigned char poly_eval(const std::vector<unsigned char>& p, unsigned char x)
{
    if (p.empty()) return 0;
    ensure_tables();

    unsigned char acc = 0;
    for (int i = static_cast<int>(p.size()) - 1; i >= 0; --i)
    {
        if (acc == 0)
            acc = p[i];
        else
            acc = static_cast<unsigned char>(g_mul_table[acc][x] ^ p[i]);
    }
    return acc;
}

unsigned char eval_horner(const std::vector<unsigned char>& coef, const std::vector<uint8_t>& xPowers) {
    unsigned char result = coef.back();
    int k = (int)size(coef);
    for (int d = k - 2; d >= 0; d--) {
        result = gf_mul(result, xPowers[1]) ^ coef[d];  // xPowers[1] = x
    }
    return result;
}

// Lagrange interpolation: given k points (xs[i], ys[i]) with distinct xs,
// produce polynomial coefficients of degree < k (size == k) such that P(xs[i]) == ys[i]
void lagrange_interpolate(
    const std::vector<unsigned char>& xs,
    const std::vector<unsigned char>& ys,
    std::vector<unsigned char>& out,
    std::vector<unsigned char>& numer,
    std::vector<unsigned char>& tmp,
    std::vector<unsigned char>& factor)
{
    const size_t k = xs.size();
    assert(k == ys.size());
    assert(out.size() == k);

    ensure_tables();

    // clear output
    for (size_t i = 0; i < k; ++i)
        out[i] = 0;

    for (size_t i = 0; i < k; ++i)
    {
        numer.clear();
        numer.push_back(1);   // numer = [1]
        unsigned char denom = 1;

        for (size_t j = 0; j < k; ++j)
        {
            if (j == i) continue;

            factor.clear();
            factor.push_back(xs[j]); // (x - xs[j]) = (x + xs[j])
            factor.push_back(1);

            poly_mul(numer, factor, tmp);
            numer.swap(tmp);

            unsigned char diff = xs[i] ^ xs[j];
            denom = g_mul_table[denom][diff];
        }

        unsigned char inv = g_inv_table[denom];
        unsigned char scale = g_mul_table[ys[i]][inv];

        poly_scale(numer, scale, tmp);

        const size_t sz = tmp.size();
        for (size_t t = 0; t < sz; ++t)
            out[t] ^= tmp[t];
    }
}