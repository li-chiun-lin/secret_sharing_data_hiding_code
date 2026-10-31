#include "tiff_io.h"

#include <cmath>
#include <vector>
#include <iostream>

using std::vector;

double compute_ssim(const Pixels& a, const Pixels& b)
{
	if (a.size() != b.size() || a.empty())
		return -1.0;

	const size_t n = a.size();

	// Step 1: compute means
	double mean_a = 0.0;
	double mean_b = 0.0;

	for (size_t i = 0; i < n; ++i)
	{
		mean_a += static_cast<double>(a[i]);
		mean_b += static_cast<double>(b[i]);
	}

	mean_a /= static_cast<double>(n);
	mean_b /= static_cast<double>(n);

	// Step 2: compute variances and covariance
	double var_a = 0.0;
	double var_b = 0.0;
	double cov_ab = 0.0;

	for (size_t i = 0; i < n; ++i)
	{
		double da = static_cast<double>(a[i]) - mean_a;
		double db = static_cast<double>(b[i]) - mean_b;

		var_a += da * da;
		var_b += db * db;
		cov_ab += da * db;
	}

	var_a /= static_cast<double>(n);
	var_b /= static_cast<double>(n);
	cov_ab /= static_cast<double>(n);

	// Step 3: SSIM constants (standard)
	const double L = 255.0;
	const double C1 = (0.01 * L) * (0.01 * L);
	const double C2 = (0.03 * L) * (0.03 * L);

	// Step 4: SSIM formula
	double numerator =
		(2.0 * mean_a * mean_b + C1) *
		(2.0 * cov_ab + C2);

	double denominator =
		(mean_a * mean_a + mean_b * mean_b + C1) *
		(var_a + var_b + C2);

	if (denominator == 0.0)
		return -1.0;

	return numerator / denominator;
}