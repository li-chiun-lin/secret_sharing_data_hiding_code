#pragma once
#include "../tiff_io/tiff_io.h"
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// LSB Matching Revisited (LSBMR)
//
// Embeds 2 bits across each pair of consecutive pixels (x1, x2) using:
//   - LSB(x1) carries the first bit directly.
//   - The function f(x1, x2) = floor(x1/2) + x2 (mod 2) carries the second.
//
// Embedding rule for pair (x1, x2) carrying bits (b1, b2):
//   1. If LSB(x1) != b1: adjust x1 by +-1 (random choice, clamped).
//      This changes f(x1,x2) as a side effect.
//   2. After step 1, if f(x1,x2) != b2: adjust x2 by +-1 (random choice).
//      Step 2 never changes LSB(x1) because only x2 is modified.
//
// On average only 0.5 pixels are modified per 2 bits embedded (vs. 0.5
// pixels per 1 bit for plain LSB matching), giving fewer total changes.
//
// The method is NOT reversible.
//
// Capacity: 2 bits per pixel pair = 1 bit per pixel (same raw rate as LSB),
// but with fewer modifications per bit.
// ---------------------------------------------------------------------------
bool embed_share_lsbmr(const TiffInfo& cover_info,
    const std::vector<unsigned char>& share,
    TiffInfo& out_info,
    uint64_t seed = 0);

bool extract_share_lsbmr(const TiffInfo& stego_info,
    size_t share_len,
    std::vector<unsigned char>& out_share);
