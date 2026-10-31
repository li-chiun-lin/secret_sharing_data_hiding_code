#pragma once

#include <vector>

// Galois Field GF(2^8) operations (modulus 0x11B)
unsigned char gf_add(unsigned char a, unsigned char b);
unsigned char gf_mul(unsigned char a, unsigned char b);
unsigned char gf_pow(unsigned char a, unsigned int e);
unsigned char gf_inv(unsigned char a);

// Polynomial utilities over GF(2^8) (coefficients lowest degree first)
// Out-parameter APIs (preferred for performance)
void poly_add(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b, std::vector<unsigned char>& out);
void poly_mul(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b, std::vector<unsigned char>& out);
void poly_scale(const std::vector<unsigned char>& p, unsigned char scalar, std::vector<unsigned char>& out);
//void lagrange_interpolate(const std::vector<unsigned char>& xs, const std::vector<unsigned char>& ys, std::vector<unsigned char>& out);
void lagrange_interpolate(
    const std::vector<unsigned char>& xs,
    const std::vector<unsigned char>& ys,
    std::vector<unsigned char>& out,
    std::vector<unsigned char>& numer,
    std::vector<unsigned char>& tmp,
    std::vector<unsigned char>& factor);

// Evaluation (small return value)
unsigned char poly_eval(const std::vector<unsigned char>& p, unsigned char x);
unsigned char eval_horner(const std::vector<unsigned char>& coef, const std::vector<uint8_t>& xPowers);
