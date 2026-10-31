#pragma once

#include <vector>

// Prime field parameter and operations for F_p (p = 257).
// Implementations live in finite_field_p.cpp

// Prime modulus
extern const int P;

// Normalize integer into [0, P-1]
int pf_norm(int v);

// Field arithmetic
int pf_add(int a, int b);
int pf_sub(int a, int b);
int pf_mul(int a, int b);

// Exponentiation and multiplicative inverse
int pf_pow(int a, unsigned int e);
int pf_inv(int a);

// Polynomial utilities over F_p (coefficients lowest degree first)
std::vector<int> poly_add_p(const std::vector<int>& a, const std::vector<int>& b);
std::vector<int> poly_mul_p(const std::vector<int>& a, const std::vector<int>& b);
std::vector<int> poly_scale_p(const std::vector<int>& p, int scalar);
int poly_eval_p(const std::vector<int>& p, int x);

// Lagrange interpolation over F_p
std::vector<int> lagrange_interpolate_p(const std::vector<int>& xs, const std::vector<int>& ys);
