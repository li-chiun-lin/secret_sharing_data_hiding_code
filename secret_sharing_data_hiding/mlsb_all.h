#pragma once
#include "../tiff_io/tiff_io.h"
#include <vector>
#include <cstdint>
#include <string>

// ---------------------------------------------------------------------------
// mlsb_all: New robust SIS scheme where:
//   - CPP metadata (r and bs) is shared together with S' as S_shared.
//   - The 24-bit block count z is shared into n header shares (3 bytes each)
//     using standard (k,n) Shamir secret sharing.
//   - The z-share is embedded in the first 6 pixels of the cover using LSB4
//     (fixed depth 4, no shuffle).
//   - The main share is embedded in pixels 6 onwards using mixed-depth
//     shuffled mLSB-OPA (seeded by z).
// ---------------------------------------------------------------------------

// Shares the 24-bit block count z into n shares (each share is 3 bytes).
// x-coordinates for shares will be 1 to n.
void share_z(size_t z, int k, int n, std::vector<std::vector<unsigned char>>& out_z_shares);

// Reconstructs the 24-bit block count z from k shares.
// participant_ids contains the 1-based indices of the participants.
size_t reconstruct_z(const std::vector<std::vector<unsigned char>>& z_shares, int k, const std::vector<int>& participant_ids);

// Embeds the main_share and z_share into a cover image.
// z is passed to compute the mixed-depth split parameters and seed the shuffle.
bool embed_share_mlsb_all(const TiffInfo& cover_info,
                          const std::vector<unsigned char>& main_share,
                          const std::vector<unsigned char>& z_share,
                          size_t z,
                          TiffInfo& out_info);

// Extracts the 3-byte z-share from the LSB4 of the first 6 pixels of a stego image.
bool extract_z_share_mlsb_all(const TiffInfo& stego_info,
                              std::vector<unsigned char>& out_z_share);

// Extracts the main share of length main_share_len from the stego image.
// z is passed to compute the mixed-depth split parameters and seed the shuffle.
bool extract_main_share_mlsb_all(const TiffInfo& stego_info,
                                 size_t z,
                                 size_t main_share_len,
                                 std::vector<unsigned char>& out_main_share);
