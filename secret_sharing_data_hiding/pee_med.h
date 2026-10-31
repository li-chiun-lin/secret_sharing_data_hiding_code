#pragma once
#include "../tiff_io/tiff_io.h"
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// PEE-MED: Reversible LSB4 data hiding using MED predictor + Huffman coding.
// No side information or location map required.
//
// Scheme overview:
//   For each of the first M pixels (raster order), the 4 LSBs are replaced
//   by a packed bit stream.  M is the smallest number of pixels sufficient
//   to hold the entire payload.  All pixels beyond M are untouched.
//
// Bit stream layout (packed into LSB4 of pixels 1..M):
//   [M          : 20 bits  ]  pixels 1..5   (covers images up to ~1M pixels)
//   [Huffman tree: 95 bits ]  pixels 6..29  (17 topology + 16*4 leaf values)
//   [Huffman-encoded pe    ]  pixels 30..M  (pe = (MED_predict XOR pixel) & 0xF)
//   [share data : s*8 bits ]  (immediately after pe stream)
//   [REF LSB4   : R*4 bits ]  original LSB4 of pixels 1..R (R = 29)
//
// Recovery reads the same bit stream in the same order:
//   1. Decode M (20 bits) from pixels 1..5.
//   2. Decode Huffman tree (95 bits) from pixels 6..29.
//   3. Decode M-R pe values from pixels 30..M using the Huffman tree.
//   4. Collect s*8 bits as share data.
//   5. Collect R*4 bits as original LSB4 of pixels 1..R.
//   6. Restore all M pixels to their original values.
// ---------------------------------------------------------------------------

// Number of header pixels (fixed).
// Pixels 1..5  : M field (20 bits)
// Pixels 6..29 : Huffman tree (95 bits)
static constexpr int kPeeMedR = 29;

struct PeeMedCapacityReport {
    bool feasible = false;                // true if any M can carry header+pe+ref
    int lsb_bits = 0;
    size_t image_pixels = 0;
    size_t chosen_M = 0;
    size_t overall_capacity_bits = 0;      // chosen_M * lsb_bits
    size_t full_image_capacity_bits = 0;   // N * lsb_bits
    size_t header_bits = 0;                // M-field + serialized Huffman tree
    size_t ref_bits = 0;                   // saved original LSBs for header pixels
    size_t compressed_lsb_bits = 0;        // Huffman-coded pe stream length over [R, M)
    size_t usable_secret_bits = 0;         // remaining payload budget for secret bits
    size_t usable_secret_bytes = 0;        // floor(usable_secret_bits / 8)
    size_t header_pixels_R = 0;            // ceil(header_bits / lsb_bits)
};

// embed_share_pee_med:
//   Embeds share into cover_info using the LSB4 + Huffman scheme.
//   Returns false if the image has insufficient capacity for the given share.
bool embed_share_pee_med(const TiffInfo& cover_info,
    const std::vector<unsigned char>& share,
    TiffInfo& out_info);

// Analyze maximum payload capacity of the PEE-MED design on a given cover
// for an arbitrary LSB depth (e.g., 2/3/4/5).
// Returns false if input is invalid or if the image is too small.
bool analyze_pee_med_capacity(const TiffInfo& cover_info,
    int lsb_bits,
    PeeMedCapacityReport& out_report);

// recover_share_pee_med:
//   Recovers share and original cover from stego_info.
//   share_len must match the length used during embedding.
bool recover_share_pee_med(const TiffInfo& stego_info,
    size_t share_len,
    std::vector<unsigned char>& out_share,
    TiffInfo& recovered_cover);
