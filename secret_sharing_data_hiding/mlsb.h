#pragma once
#include "../tiff_io/tiff_io.h"
#include <vector>
#include <cstdint>
#include <string>

// ---------------------------------------------------------------------------
// compute_min_t: find the smallest t in [1..8] such that a share of
// share_bytes bytes fits in a cover of cover_pixels pixels.
//
// Capacity at t bits per pixel = floor(cover_pixels / ceil(8/t)) bytes.
//
// Returns the minimal t if feasible.
// Returns 9 if no t in [1..8] provides enough capacity (share too large).
// ---------------------------------------------------------------------------
int compute_min_t(size_t share_bytes, size_t cover_pixels);

// embed_share_mlsb:
//   t = 0  -> auto-select minimal t via compute_min_t (recommended).
//   t > 0  -> use the specified t (must be in [1..8]).
//   Returns false with a cerr message if minimal t > 8 (infeasible).
bool embed_share_mlsb(const TiffInfo& cover_info,
    const std::vector<unsigned char>& share,
    TiffInfo& out_info,
    int t = 0);

// extract_share_mlsb:
//   t = 0  -> auto-select minimal t via compute_min_t.
//   t > 0  -> use the specified t.
//   If write_file=true and out_path is non-empty, writes extracted share to file.
bool extract_share_mlsb(const TiffInfo& stego_info,
    size_t share_len,
    std::vector<unsigned char>& out_share,
    int t = 0,
    bool write_file = false,
    const std::string& out_path = "");

void set_bits(unsigned char& px, unsigned char value, int t);
