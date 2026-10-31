#include "tiff_io.h"

#include <cmath>
#include <iostream>

using namespace std;

SpaAnalysisResult compute_spa_analysis(const TiffInfo& info) {
    SpaAnalysisResult result;
    if (info.data.empty()) {
        cerr << "Error: Image data is empty." << endl;
        return result;
    }
    if (info.width < 2 || info.height == 0) {
        cerr << "Error: Image is too small for SPA analysis." << endl;
        return result;
    }

    long long same_even_odd = 0;
    long long same_odd_even = 0;
    long long adjacent = 0;
    int pairs = 0;

    for (uint32_t y = 0; y < info.height; ++y) {
        for (uint32_t x = 0; x + 1 < info.width; x += 2) {
            int u = info.data[y * info.width + x];
            int v = info.data[y * info.width + (x + 1)];
            int u2 = u / 2;
            int v2 = v / 2;
            bool u_even = (u % 2) == 0;
            bool v_even = (v % 2) == 0;

            if (u_even && !v_even && u2 == v2) {
                ++same_even_odd;
            } else if (!u_even && v_even && u2 == v2) {
                ++same_odd_even;
            }
            if (abs(u2 - v2) == 1) {
                ++adjacent;
            }
            ++pairs;
        }
    }

    result.pairs = pairs;
    if (pairs > 0) {
        result.pair_imbalance = static_cast<double>(same_even_odd - same_odd_even);
        result.score = result.pair_imbalance / static_cast<double>(pairs);
        double denom = static_cast<double>(same_even_odd + same_odd_even);
        if (denom > 0.0) {
            result.estimated_rate = static_cast<double>(same_even_odd - same_odd_even) / denom;
        }
    }
    return result;
}

void spa_analysis(const TiffInfo& info) {
    SpaAnalysisResult result = compute_spa_analysis(info);
    cout << "--- SPA Analysis for " << info.width << "x" << info.height << " ---" << endl;
    cout << "Pairs: " << result.pairs << endl;
    cout << "PairImbalance: " << result.pair_imbalance << endl;
    cout << "Score: " << result.score << endl;
    cout << "EstimatedRate: " << result.estimated_rate << endl;
}
