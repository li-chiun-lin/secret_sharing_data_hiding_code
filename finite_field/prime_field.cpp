#include "prime_field.h"

#include <cassert>
#include <algorithm>
#include <vector>

const int P = 257;

int pf_norm(int v)
{
    v %= P;
    if (v < 0) v += P;
    return v;
}

int pf_add(int a, int b) { return pf_norm(a + b); }
int pf_sub(int a, int b) { return pf_norm(a - b); }
int pf_mul(int a, int b) { return pf_norm((a * b) % P); }

int pf_pow(int a, unsigned int e)
{
    int base = pf_norm(a);
    int r = 1;
    while (e)
    {
        if (e & 1u) r = pf_mul(r, base);
        base = pf_mul(base, base);
        e >>= 1u;
    }
    return r;
}

int pf_inv(int a)
{
    a = pf_norm(a);
    if (a == 0) return 0; // caller must avoid dividing by zero
    return pf_pow(a, P - 2);
}

// Polynomials over F_p, coefficients lowest degree first (int in [0,P-1])
std::vector<int> poly_add_p(const std::vector<int>& a, const std::vector<int>& b)
{
    size_t n = std::max(a.size(), b.size());
    std::vector<int> r(n, 0);
    for (size_t i = 0; i < a.size(); ++i) r[i] = pf_add(r[i], a[i]);
    for (size_t i = 0; i < b.size(); ++i) r[i] = pf_add(r[i], b[i]);
    return r;
}

std::vector<int> poly_mul_p(const std::vector<int>& a, const std::vector<int>& b)
{
    if (a.empty() || b.empty()) return {};
    std::vector<int> r(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            r[i + j] = pf_add(r[i + j], pf_mul(a[i], b[j]));
    return r;
}

std::vector<int> poly_scale_p(const std::vector<int>& p, int scalar)
{
    std::vector<int> r(p.size(), 0);
    for (size_t i = 0; i < p.size(); ++i) r[i] = pf_mul(p[i], scalar);
    return r;
}

int poly_eval_p(const std::vector<int>& p, int x)
{
    int acc = 0;
    for (int i = static_cast<int>(p.size()) - 1; i >= 0; --i)
    {
        acc = pf_mul(acc, x);
        acc = pf_add(acc, p[i]);
    }
    return acc;
}

// Lagrange interpolation over F_p: given k points (xs[i], ys[i]) with distinct xs,
// return polynomial coefficients of degree < k (size == k)
std::vector<int> lagrange_interpolate_p(const std::vector<int>& xs, const std::vector<int>& ys)
{
    assert(xs.size() == ys.size());
    size_t k = xs.size();
    std::vector<int> result(k, 0);

    for (size_t i = 0; i < k; ++i)
    {
        std::vector<int> numer = { 1 }; // polynomial "1"
        int denom = 1;
        for (size_t j = 0; j < k; ++j)
        {
            if (j == i) continue;
            int c0 = pf_sub(0, xs[j]); // -xs[j]
            std::vector<int> factor = { c0, 1 };
            numer = poly_mul_p(numer, factor);
            int diff = pf_sub(xs[i], xs[j]);
            denom = pf_mul(denom, diff);
        }
        int denom_inv = pf_inv(denom);
        int scalar = pf_mul(ys[i], denom_inv);
        std::vector<int> scaled = poly_scale_p(numer, scalar);
        if (result.size() < scaled.size()) result.resize(scaled.size(), 0);
        result = poly_add_p(result, scaled);
    }

    if (result.size() < k) result.resize(k, 0);
    else if (result.size() > k) result.resize(k);
    return result;
}