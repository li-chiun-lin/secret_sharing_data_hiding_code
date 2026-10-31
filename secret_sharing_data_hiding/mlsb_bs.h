#pragma once
#include "../tiff_io/tiff_io.h"
#include <vector>
#include <cstdint>
#include <string>

// ---------------------------------------------------------------------------
// mlsb_bs: mLSB embedding with Bit-Stream constant-polynomial fix.
//
// The BS preprocessing is applied to the SECRET PIXELS before Shamir sharing,
// not to the share bytes.  This correctly eliminates constant-polynomial
// blocks (a1=...=a(k-1)=0) at the source.
//
// Workflow:
//   Sender:
//     1. bs_preprocess(secret_pixels, k) -> S', z, r, bs
//     2. generate_shares(S', k, n)       -> shares[0..n-1]
//     3. For each share i:
//        embed_share_mlsb_bs(cover_i, shares[i], r, bs) -> stego_i
//
//   Receiver (collects k stego images):
//     1. extract_share_mlsb_bs(stego_i, total_len, share_len) -> share_i, r, bs
//     2. reconstruct_secret(shares, k)                        -> S'
//     3. bs_postprocess(S', k, r, bs)                         -> original secret
//
//   Parameter r (1 byte, appended after share, before bs):
//     Layout of payload: [share (L bytes)] [r (1 byte)] [bs (ceil(z/8) bytes)]
//     Where L = ceil(N/k) bytes.
//
//   Bit-stream bs (z bits, packed into ceil(z/8) bytes):
//     bit 1 -> block was a constant-polynomial block (byte[k-1] was 0x00,
//              replaced with r)
//     bit 0 -> block was a genuine ambiguous block (bytes[1..k-2]=0, byte[k-1]=r)
//
//   Total embedded per share = L + 1 + ceil(z/8) bytes.
// ---------------------------------------------------------------------------

// Preprocess secret pixels: scan for zero-blocks and ambiguous-blocks,
// choose r, build bs, and return the modified pixel sequence.
// k         -- threshold (block size = k)
// pixels_in -- original secret pixels (padded to multiple of k if needed)
// r_out     -- chosen replacement value (1..255)
// z_out     -- total count of modified/ambiguous blocks
// bs_out    -- packed bitstream (ceil(z/8) bytes)
bool bs_preprocess(const std::vector<unsigned char>& pixels_in,
                   int k,
                   unsigned char& r_out,
                   size_t& z_out,
                   std::vector<unsigned char>& bs_out,
                   std::vector<unsigned char>& pixels_out);

// Reverse the preprocessing: given reconstructed pixels (which may contain
// substituted blocks), restore original zero-blocks using bs.
// The count z is computed dynamically inside bs_postprocess by counting
// ambiguous blocks in pixels_in.
// Returns false on error.
bool bs_postprocess(const std::vector<unsigned char>& pixels_in,
                    int k,
                    unsigned char r,
                    const std::vector<unsigned char>& bs,
                    std::vector<unsigned char>& pixels_out);

// embed_share_mlsb_bs:
//   Embeds the payload [share | r(1B) | bs(ceil(z/8)B)] into the cover.
//   r and bs are passed explicitly; the share is the raw Shamir share.
//   t = 0  -> auto-select minimal t (recommended).
//   t > 0  -> use specified t.
//   Returns false on error.
bool embed_share_mlsb_bs(const TiffInfo& cover_info,
                         const std::vector<unsigned char>& share,
                         unsigned char r,
                         const std::vector<unsigned char>& bs,
                         TiffInfo& out_info,
                         int t = 0);

// extract_share_mlsb_bs:
//   Extracts the raw bytes (share + r + bs payload) from the stego image,
//   then parses the components.
//   total_embedded_len -- total bytes embedded (share_len + 1 + ceil(z/8)).
//   share_len -- length of the Shamir share (known to the receiver).
//   out_share -- the raw Shamir share (first share_len bytes).
//   out_r -- the parsed replacement value.
//   out_bs -- the parsed bit-stream, to be passed to bs_postprocess.
bool extract_share_mlsb_bs(const TiffInfo& stego_info,
                            size_t total_embedded_len,
                            size_t share_len,
                            std::vector<unsigned char>& out_share,
                            unsigned char& out_r,
                            std::vector<unsigned char>& out_bs,
                            int t = 0);
