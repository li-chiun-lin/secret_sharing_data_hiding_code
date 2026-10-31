#pragma once
#include "../tiff_io/tiff_io.h"
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// Original PVO: sequential pixel pairs in raster order.
//
// Pairs pixels as (0,1), (2,3), (4,5), ... regardless of local smoothness.
// used_map[i] = 1 if pair i was used for embedding (no overflow).
// perm_map[i] = 1 if the original pair had v0 > v1 (needed for recovery).
// ---------------------------------------------------------------------------
bool embed_share_pvo_sequential(const TiffInfo& cover_info,
    const std::vector<unsigned char>& share,
    TiffInfo& out_info,
    std::vector<uint8_t>& used_map,
    std::vector<uint8_t>& perm_map);

bool recover_share_pvo_sequential(const TiffInfo& stego_info,
    const std::vector<uint8_t>& used_map,
    const std::vector<uint8_t>& perm_map,
    size_t share_len,
    std::vector<unsigned char>& out_share,
    TiffInfo& recovered_cover);

// ---------------------------------------------------------------------------
// Sorting-based PVO: pairs sorted by local smoothness before embedding.
//
// Motivation: PVO expansion succeeds only when the difference h = |v1 - v0|
// is small (smooth pair). Sorting pairs by their local variance and
// processing the smoothest pairs first maximises the number of successful
// embeddings and improves PSNR, because distortion is concentrated in the
// pairs that are already similar.
//
// Smoothness metric: absolute difference |v0 - v1| of the pair itself.
// Pairs with smaller |v0 - v1| are processed first.
//
// The sort order is deterministic from the cover image, so the receiver
// can reconstruct it without any side information.
//
// sort_order: output vector of pair indices in the order they were processed.
//   The receiver needs this to know which pair index corresponds to which
//   position in used_map / perm_map.  It is fully determined by the cover,
//   so in practice the receiver recomputes it from the stego image after
//   undoing the embedding - but we return it here for convenience and
//   to make the interface explicit.
//
// used_map[rank] = 1 if the pair at sort_order[rank] was embedded.
// perm_map[rank] = 1 if that pair had v0 > v1.
// ---------------------------------------------------------------------------
bool embed_share_pvo_sorted(const TiffInfo& cover_info,
    const std::vector<unsigned char>& share,
    TiffInfo& out_info,
    std::vector<size_t>& sort_order,
    std::vector<uint8_t>& used_map,
    std::vector<uint8_t>& perm_map);

bool recover_share_pvo_sorted(const TiffInfo& stego_info,
    const std::vector<size_t>& sort_order,
    const std::vector<uint8_t>& used_map,
    const std::vector<uint8_t>& perm_map,
    size_t share_len,
    std::vector<unsigned char>& out_share,
    TiffInfo& recovered_cover);

// ---------------------------------------------------------------------------
// Backward-compatible aliases (keep existing call sites working)
// ---------------------------------------------------------------------------
inline bool embed_share_pvo_pair_rammap(const TiffInfo& cover_info,
    const std::vector<unsigned char>& share,
    TiffInfo& out_info,
    std::vector<uint8_t>& used_map,
    std::vector<uint8_t>& perm_map)
{
    return embed_share_pvo_sequential(cover_info, share, out_info, used_map, perm_map);
}

inline bool recover_share_pvo_pair_rammap(const TiffInfo& stego_info,
    const std::vector<uint8_t>& used_map,
    const std::vector<uint8_t>& perm_map,
    size_t share_len,
    std::vector<unsigned char>& out_share,
    TiffInfo& recovered_cover)
{
    return recover_share_pvo_sequential(stego_info, used_map, perm_map,
                                        share_len, out_share, recovered_cover);
}
