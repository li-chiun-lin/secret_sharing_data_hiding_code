#pragma once
#include "../tiff_io/tiff_io.h"
#include <vector>
#include <cstdint>
#include <string>

// ---------------------------------------------------------------------------
// mlsb_mixed: Mixed-Depth mLSB embedding with a 6-pixel LSB4 header.
//
// Layout of stego image:
//   [0..5]       Header pixels containing z (24-bit, LSB4)
//   [6..P+5]     Payload pixels containing:
//                - share bytes (first ceil(N/k) bytes)
//                - r byte (1 byte)
//                - bs bytes (ceil(z/8) bytes)
// ---------------------------------------------------------------------------

extern bool g_use_shuffled;

bool embed_share_mlsb_bs_mixed(const TiffInfo& cover_info,
                               const std::vector<unsigned char>& share,
                               unsigned char r,
                               const std::vector<unsigned char>& bs,
                               size_t z,
                               TiffInfo& out_info);

bool extract_share_mlsb_bs_mixed(const TiffInfo& stego_info,
                                 size_t share_len,
                                 std::vector<unsigned char>& out_share,
                                 unsigned char& out_r,
                                 std::vector<unsigned char>& out_bs);

bool embed_share_mlsb_bs_mixed_shuffled(const TiffInfo& cover_info,
                                        const std::vector<unsigned char>& share,
                                        unsigned char r,
                                        const std::vector<unsigned char>& bs,
                                        size_t z,
                                        TiffInfo& out_info);

bool extract_share_mlsb_bs_mixed_shuffled(const TiffInfo& stego_info,
                                          size_t share_len,
                                          std::vector<unsigned char>& out_share,
                                          unsigned char& out_r,
                                          std::vector<unsigned char>& out_bs);

