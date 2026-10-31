#include "histogram_shifting.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cstdint>

using namespace std;

// Compute histogram of pixel values
static vector<int> compute_histogram(const vector<unsigned char>& data)
{
    vector<int> hist(256, 0);
    for (auto p : data) hist[p]++;
    return hist;
}

// Find peak and zero bins: peak has max frequency, zero has min frequency != peak
static pair<int, int> find_peak_zero(const vector<int>& hist)
{
    int peak = 0, zero = 0;
    int max_freq = 0, min_freq = INT_MAX;
    for (int i = 0; i < 256; ++i) {
        if (hist[i] > max_freq) {
            max_freq = hist[i];
            peak = i;
        }
        if (hist[i] < min_freq) {
            min_freq = hist[i];
            zero = i;
        }
    }
    // If zero == peak, find another zero
    if (zero == peak) {
        min_freq = INT_MAX;
        for (int i = 0; i < 256; ++i) {
            if (i != peak && hist[i] < min_freq) {
                min_freq = hist[i];
                zero = i;
            }
        }
    }
    return {peak, zero};
}

bool embed_share_histogram_shifting(const TiffInfo& cover_info,
    const vector<unsigned char>& share,
    TiffInfo& out_info)
{
    vector<unsigned char> modified = cover_info.data;
    vector<int> hist = compute_histogram(cover_info.data);
    auto [peak, zero] = find_peak_zero(hist);

    // Shift bins: pixels > peak shift right (+1), pixels < peak shift left (-1)
    for (auto& p : modified) {
        if (p > peak) p++;
        else if (p < peak) p--;
    }

    // Now, embed in the peak bin: for bit 0, change to peak-1, for bit 1, change to peak+1
    size_t bit_idx = 0;
    size_t total_bits = share.size() * 8;
    for (auto& p : modified) {
        if (p == peak && bit_idx < total_bits) {
            int bit = (share[bit_idx / 8] >> (bit_idx % 8)) & 1;
            if (bit == 0) p = peak - 1;
            else p = peak + 1;
            bit_idx++;
        }
    }

    if (bit_idx < total_bits) {
        cerr << "Not enough peak pixels for embedding\n";
        return false;
    }

    out_info = cover_info;
    out_info.data = move(modified);
    return true;
}

bool recover_share_histogram_shifting(const TiffInfo& stego_info,
    size_t share_len,
    vector<unsigned char>& out_share,
    TiffInfo& recovered_cover)
{
    // The stego histogram has a gap at (peak+1) or (peak-1) depending on
    // shift direction. We find peak by looking for the bin adjacent to a
    // zero (or minimum) bin that has the highest count.
    // More robustly: the embed always shifts pixels > peak up by 1 and
    // pixels < peak down by 1. So in the stego histogram:
    //   - bin (peak-1) and bin (peak+1) each contain embedded bits
    //   - the original zero bin is now empty (or near-empty)
    //   - peak itself may be empty (all peak pixels were embedded)
    //
    // Recovery strategy: find the bin with minimum count (the zero bin),
    // then determine peak as the adjacent bin with higher count.
    // But this is fragile. The embed stores peak and zero implicitly.
    //
    // Simpler: scan for the gap. After embedding, the bin at (zero) is
    // still zero (no pixel was shifted to it). The peak bin is adjacent.
    // We find zero as the bin with count 0 (or minimum), then peak is
    // the neighbor with the higher count.

    vector<unsigned char> recovered = stego_info.data;
    vector<int> hist = compute_histogram(stego_info.data);

    // Find the zero bin (count == 0 preferred, else minimum)
    int zero = -1;
    for (int i = 0; i < 256; ++i) {
        if (hist[i] == 0) { zero = i; break; }
    }
    if (zero < 0) {
        // No exact zero bin; find minimum
        int min_freq = INT_MAX;
        for (int i = 0; i < 256; ++i) {
            if (hist[i] < min_freq) { min_freq = hist[i]; zero = i; }
        }
    }

    // Peak is the neighbor of zero with higher count
    int peak = -1;
    int best = -1;
    if (zero > 0   && hist[zero - 1] > best) { best = hist[zero - 1]; peak = zero - 1; }
    if (zero < 255 && hist[zero + 1] > best) { best = hist[zero + 1]; peak = zero + 1; }
    if (peak < 0) return false;

    // Extract bits and restore
    out_share.assign(share_len, 0);
    size_t bit_idx = 0;
    const size_t total_bits = share_len * 8;

    for (auto& p : recovered) {
        int v = static_cast<int>(p);
        if (v == peak - 1 && bit_idx < total_bits) {
            // bit 0 was embedded here
            p = static_cast<unsigned char>(peak);
            bit_idx++;
        } else if (v == peak + 1 && bit_idx < total_bits) {
            // bit 1 was embedded here
            out_share[bit_idx / 8] |= static_cast<unsigned char>(1u << (bit_idx % 8));
            p = static_cast<unsigned char>(peak);
            bit_idx++;
        } else {
            // Undo the shift
            if (v > peak) p = static_cast<unsigned char>(v - 1);
            else if (v < peak) p = static_cast<unsigned char>(v + 1);
        }
    }

    recovered_cover = stego_info;
    recovered_cover.data = std::move(recovered);
    return bit_idx >= total_bits;
}