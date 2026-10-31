#include "tiff_io.h"

#include <cmath>
#include <vector>
#include <iostream>

using std::vector;

double compute_mse(const Pixels &a, const Pixels &b)
{
	if (a.size() != b.size() || a.empty())
		return -1.0;

	double mse = 0.0;
	const size_t n = a.size();
	for (size_t i = 0; i < n; ++i)
	{
		int diff = static_cast<int>(a[i]) - static_cast<int>(b[i]);
		mse += static_cast<double>(diff) * static_cast<double>(diff);
	}
	mse /= static_cast<double>(n);
	return mse;
}

double compute_psnr(const Pixels &a, const Pixels &b)
{
	double mse = compute_mse(a, b);
	if (mse < 0.0)
		return -1.0;
	if (mse == 0.0)
		return INFINITY;

	const double maxI = 255.0;
	return 10.0 * log10((maxI * maxI) / mse);
}