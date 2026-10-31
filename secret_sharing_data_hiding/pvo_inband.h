#pragma once
#include "../tiff_io/tiff_io.h"
#include <vector>
#include <cstddef>

// PVO in-band variant:
// - embeds share using PVO on data pairs
// - stores PVO side info (used_map/perm_map) inside reserved prefix pixels
// - self-contained for share extraction (no external maps)
//
// Note: cover recovery is exact for data pairs; prefix metadata region is not
// restored losslessly in this prototype.
bool embed_share_pvo_inband(const TiffInfo& cover_info,
    const std::vector<unsigned char>& share,
    TiffInfo& out_info);

bool recover_share_pvo_inband(const TiffInfo& stego_info,
    size_t share_len,
    std::vector<unsigned char>& out_share,
    TiffInfo& recovered_cover);
