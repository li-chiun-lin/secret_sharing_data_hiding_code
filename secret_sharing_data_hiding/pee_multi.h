#pragma once
#include "../tiff_io/tiff_io.h"
#include <vector>
#include <cstdint>

// Multi-layer PEE with MED + deterministic selection (no side info map).
// Recovery relies only on the same (T,B) and bits_per_layer.

bool embed_share_pee_multi(const TiffInfo& cover_info,
    const std::vector<unsigned char>& share,
    TiffInfo& out_info,
    std::vector<size_t>& bits_per_layer,
    int T = 1,
    int B = 2,
    int max_layers = 16);

bool recover_share_pee_multi(const TiffInfo& stego_info,
    size_t share_len,
    const std::vector<size_t>& bits_per_layer,
    std::vector<unsigned char>& out_share,
    TiffInfo& recovered_cover,
    int T = 1,
    int B = 2);
