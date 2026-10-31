#pragma once
#include "../tiff_io/tiff_io.h"
#include <vector>
#include <cstdint>

// Deterministic checkerboard PEE with guaranteed decode partition (no map).
// Two-set order:
//   - phase 1: odd parity (row+col odd), predictor from original cover
//   - phase 2: even parity (row+col even), predictor from phase-1 stego
//
// Decoder mirrors in reverse dependency order:
//   - recover phase 2 first (using phase-1 stego as predictor), then phase 1.
//
// Decoder partition (no side info needed):
//   - boundary skip: stego pixel outside [B, 255-B] -> orig = stego
//   - |e'| <= 2T+1  -> was embedded, extract bit
//   - |e'| > 2T+1   -> was shifted by sign(e)*(2T+2), undo shift
//
// INVARIANT: B is silently clamped to max(B, 4*T+4) inside both functions.
// This guarantees that shifted pixels never land in the boundary zone,
// making all three decoder cases unambiguous without any side information.
bool embed_share_pee_checker_safe(const TiffInfo& cover_info,
    const std::vector<unsigned char>& share,
    TiffInfo& out_info,
    int T = 1,
    int B = 2);

bool recover_share_pee_checker_safe(const TiffInfo& stego_info,
    size_t share_len,
    std::vector<unsigned char>& out_share,
    TiffInfo& recovered_cover,
    int T = 1,
    int B = 2);

