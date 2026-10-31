#pragma once
#include "../tiff_io/tiff_io.h"
#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>

struct HuffmanCodec {
    // build from symbol frequency or symbol sequence
    void build_from_symbols(const std::vector<uint8_t>& symbols, int alphabet_bits);
    void build_from_frequencies(const std::vector<uint32_t>& freq_u32, int alphabet_bits);

    // encode symbols -> bitpacked bytes (LSB-first packing of bitstream)
    // returns bit length in out_bit_len, bytes in out_bytes
    void encode_bits(const std::vector<uint8_t>& symbols, std::vector<uint8_t>& out_bytes, size_t& out_bit_len) const;

    // decode from bitpacked bytes -> symbols (stops when decoded_count reached)
    bool decode_bits(const std::vector<uint8_t>& in_bytes, size_t in_bit_len, size_t decode_count, std::vector<uint8_t>& out_symbols) const;

private:
    struct Node { int left = -1, right = -1; int symbol = -1; uint64_t freq = 0; };
    std::vector<Node> nodes;
    int root = -1;
    int alphabet_size = 0;

    // code table
    std::vector<uint32_t> code; // lower bits contain code
    std::vector<int> code_len;

    void build_codes();
};

// Huffman in-band LSB embed/recover
// embed: If write_sidecar is true and sidecar_path is not empty, before writing compressed header/meta,
// the overwritten least significant bits will be written to the sidecar (which can be used during recovery to restore the cover image).
bool embed_share_lsb_with_huffman_inband(const TiffInfo& cover_info,
    const std::vector<unsigned char>& share,
    TiffInfo& out_info,
    HuffmanCodec& codec_out,
    int t = 2,
    bool write_sidecar = false,
    const std::string& sidecar_path = "");

// recover: If use_sidecar is true and sidecar_path is not empty, it will read the sidecar and before returning recovered_cover,
// it will write back the overwritten least significant bits to recovered_cover (thus achieving a cover-identical recovery).
bool recover_share_lsb_with_huffman_inband(const TiffInfo& stego_info,
    const HuffmanCodec& codec_in,
    size_t share_len,
    std::vector<unsigned char>& out_share,
    TiffInfo& recovered_cover,
    int t = 2,
    bool use_sidecar = false,
    const std::string& sidecar_path = "");
