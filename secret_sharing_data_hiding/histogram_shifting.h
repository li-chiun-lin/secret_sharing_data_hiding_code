#pragma once
#include "../tiff_io/tiff_io.h"
#include <vector>
#include <cstdint>
#include <map>

// Histogram Shifting Reversible Data Hiding (Ni et al.)
// Embeds data by shifting histogram bins, using the peak bin for embedding.
// No side information required - parameters are recoverable from the stego image.

bool embed_share_histogram_shifting(const TiffInfo& cover_info,
    const std::vector<unsigned char>& share,
    TiffInfo& out_info);

bool recover_share_histogram_shifting(const TiffInfo& stego_info,
    size_t share_len,
    std::vector<unsigned char>& out_share,
    TiffInfo& recovered_cover);