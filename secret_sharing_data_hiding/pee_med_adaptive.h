#pragma once
#include "../tiff_io/tiff_io.h"
#include <vector>
#include <cstddef>

// Self-contained adaptive PEE-MED:
// - in-band stream stores M, lsb_bits, Huffman tree, encoded pe, share bits, REF bits
// - no external sidecar file
bool embed_share_pee_med_adaptive(const TiffInfo& cover_info,
    const std::vector<unsigned char>& share,
    TiffInfo& out_info,
    int lsb_bits);

bool recover_share_pee_med_adaptive(const TiffInfo& stego_info,
    size_t share_len,
    std::vector<unsigned char>& out_share,
    TiffInfo& recovered_cover);
