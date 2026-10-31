#pragma once
#include "../tiff_io/tiff_io.h"
#include <vector>
#include <cstdint>

// PEE with Diamond (rhombus) predictor - two-phase checkerboard scan.
// pix_kind must be passed to recover (one byte per pixel: 0 none, 1 embed, 2 shift).

bool embed_share_pee_diamond(const TiffInfo& cover_info,
    const std::vector<unsigned char>& share,
    TiffInfo& out_info,
    std::vector<uint8_t>& pix_kind,
    int T = 1);

bool recover_share_pee_diamond(const TiffInfo& stego_info,
    size_t share_len,
    const std::vector<uint8_t>& pix_kind,
    std::vector<unsigned char>& out_share,
    TiffInfo& recovered_cover,
    int T = 1);

// Side-info-free variant:
// No pix_kind / location map is used. Decoding classifies pixels by disjoint
// prediction-error intervals from the stego image.
bool embed_share_pee_diamond_no_map(const TiffInfo& cover_info,
    const std::vector<unsigned char>& share,
    TiffInfo& out_info,
    int T = 1);

bool recover_share_pee_diamond_no_map(const TiffInfo& stego_info,
    size_t share_len,
    std::vector<unsigned char>& out_share,
    TiffInfo& recovered_cover,
    int T = 1);
