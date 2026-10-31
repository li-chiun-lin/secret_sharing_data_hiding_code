#include "tiff_io.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

using namespace std;

// Discrimination function: sum of absolute differences in a block
static double get_smoothness(const vector<int>& block) {
    double smoothness = 0;
    for (size_t i = 0; i < block.size() - 1; ++i) {
        smoothness += abs(block[i + 1] - block[i]);
    }
    return smoothness;
}

// Flipping function F1 (Positive): 0<->1, 2<->3...
static int flip_positive(int val) {
    return (val % 2 == 0) ? (val + 1) : (val - 1);
}

// Flipping function F-1 (Negative): -1<->0, 1<->2...
static int flip_negative(int val) {
    return (val % 2 == 0) ? (val - 1) : (val + 1);
}

static bool apply_flip(int value, int mask, bool negative, int& out)
{
    if (mask == 0) {
        out = value;
        return true;
    }

    out = negative ? flip_negative(value) : flip_positive(value);
    return out >= 0 && out <= 255;
}

RsAnalysisResult compute_rs_analysis(const TiffInfo& info) {
    RsAnalysisResult result;
    if (info.data.empty()) {
        cerr << "Error: Image data is empty." << endl;
        return result;
    }
    if (info.width < 4 || info.height == 0) {
        cerr << "Error: Image is too small for RS analysis." << endl;
        return result;
    }

    int rm = 0, sm = 0, r_m = 0, s_m = 0;
    int total_blocks = 0;
    const int blockSize = 4;

    // Standard Mask M = [0, 1, 1, 0]
    const vector<int> mask = { 0, 1, 1, 0 };

    // Iterate through rows and columns using linear indexing
    for (uint32_t r = 0; r < info.height; ++r) {
        // Stop before the end to ensure a full block of 4 can be read
        for (uint32_t c = 0; c <= info.width - blockSize; c += blockSize) {

            // 1. Extract block of 4 pixels
            vector<int> block(blockSize);
            for (int i = 0; i < blockSize; ++i) {
                // Linear indexing for vector<Pixel>
                block[i] = static_cast<int>(info.data[r * info.width + (c + i)]);
            }

            double f_orig = get_smoothness(block);

            // 2. Positive Flipping Analysis
            vector<int> block_p = block;
            bool valid_p = true;
            for (int i = 0; i < blockSize; ++i) {
                int flipped = 0;
                if (!apply_flip(block_p[i], mask[i], false, flipped)) {
                    valid_p = false;
                    break;
                }
                block_p[i] = flipped;
            }
            if (!valid_p) {
                continue;
            }
            double f_p = get_smoothness(block_p);
            if (f_p > f_orig) rm++;
            else if (f_p < f_orig) sm++;

            // 3. Negative Flipping Analysis
            vector<int> block_n = block;
            bool valid_n = true;
            for (int i = 0; i < blockSize; ++i) {
                int flipped = 0;
                if (!apply_flip(block_n[i], mask[i], true, flipped)) {
                    valid_n = false;
                    break;
                }
                block_n[i] = flipped;
            }
            if (!valid_n) {
                continue;
            }
            double f_n = get_smoothness(block_n);
            if (f_n > f_orig) r_m++;
            else if (f_n < f_orig) s_m++;

            total_blocks++;
        }
    }

    result.total_blocks = total_blocks;
    result.regular_m = rm;
    result.singular_m = sm;
    result.regular_neg_m = r_m;
    result.singular_neg_m = s_m;

    if (total_blocks > 0) {
        result.regular_m_ratio = static_cast<double>(rm) / total_blocks;
        result.singular_m_ratio = static_cast<double>(sm) / total_blocks;
        result.regular_neg_m_ratio = static_cast<double>(r_m) / total_blocks;
        result.singular_neg_m_ratio = static_cast<double>(s_m) / total_blocks;
        result.gap = abs((result.regular_m_ratio - result.singular_m_ratio)
                       - (result.regular_neg_m_ratio - result.singular_neg_m_ratio));
    }

    return result;
}

void rs_analysis(const TiffInfo& info) {
    RsAnalysisResult result = compute_rs_analysis(info);

    cout << "--- RS Analysis for " << info.width << "x" << info.height << " ---" << endl;
    cout << "Blocks: " << result.total_blocks << endl;
    cout << "Rm:   " << result.regular_m_ratio << endl;
    cout << "Sm:   " << result.singular_m_ratio << endl;
    cout << "R-m:  " << result.regular_neg_m_ratio << endl;
    cout << "S-m:  " << result.singular_neg_m_ratio << endl;
    cout << "Linearity Gap: " << result.gap << endl;
}
