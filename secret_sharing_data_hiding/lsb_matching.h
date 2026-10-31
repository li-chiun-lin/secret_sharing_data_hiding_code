#pragma once
#include "../tiff_io/tiff_io.h"
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// LSB Matching (+-1 steganography)
//
// Instead of overwriting the LSB directly, each pixel is adjusted by at
// most +-1 so that its LSB matches the desired payload bit.
// If the LSB already matches, the pixel is left unchanged.
// If it does not match, the pixel is randomly incremented or decremented
// by 1 (with equal probability, clamped to [0,255]).
//
// This breaks the statistical artifacts of direct LSB replacement and is
// harder to detect by RS analysis or chi-square attacks.
//
// The method is NOT reversible: the original pixel values cannot be
// recovered from the stego image.
//
// Capacity: 1 bit per pixel (same as plain LSB).
// ---------------------------------------------------------------------------
bool embed_share_lsb_matching(const TiffInfo& cover_info,
    const std::vector<unsigned char>& share,
    TiffInfo& out_info,
    uint64_t seed = 0);

bool extract_share_lsb_matching(const TiffInfo& stego_info,
    size_t share_len,
    std::vector<unsigned char>& out_share);
