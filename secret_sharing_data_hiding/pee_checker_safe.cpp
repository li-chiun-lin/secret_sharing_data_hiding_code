#include "pee_checker_safe.h"
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

// ---------------------------------------------------------------------------
// Design invariant: B >= 2*T + 2
//
// This guarantees that for any non-boundary pixel c in [B, 255-B]:
//   - Expansion: e' = 2e+bit, |e'| <= 2T+1, so c' = p+e' stays in
//     [p-(2T+1), p+(2T+1)]. Since c = p+e and |e|<=T, p is in
//     [c-T, c+T] subset [B-T, 255-B+T]. With B >= 2T+2:
//     c'_min >= B-T-(2T+1) = B-3T-1 >= 2T+2-3T-1 = 1-T >= 0 for T<=1.
//     More precisely: c' = p+2e+bit. Since c in [B,255-B] and |e|<=T:
//     c'_min = c - e + 2e + 0 - ... actually c' = p+2e+bit = c+e+bit.
//     c'_min = c - T + 0 >= B - T >= 2T+2-T = T+2 >= B for T>=0. Wait:
//     c' = p + 2e + bit. p = c - e. c' = c - e + 2e + bit = c + e + bit.
//     c'_max = c + T + 1 <= 255-B+T+1 = 255-(B-T-1) <= 255-B for B>=T+1. OK.
//     c'_min = c - T + 0 >= B - T >= 2T+2-T = T+2 >= B for T>=0. OK.
//   - Shift: c' = c + sign*(2T+2). c'_min = B-(2T+2) >= 0 for B>=2T+2. OK.
//     c'_max = 255-B+(2T+2) <= 255 for B>=2T+2. OK.
//     c' stays in [B-(2T+2), 255-B+(2T+2)] but we need c' in [0,255].
//     With B=2T+2: c'_min = 0, c'_max = 255. Exactly at boundary.
//     With B>2T+2: strictly inside. No overflow possible.
//
// Decoder partition (no side info needed):
//   stego sp in [0,B) or (255-B,255]  -> boundary-skipped, orig = sp
//   |e'| = |sp-p| <= 2T+1             -> was embedded, extract bit
//   |e'| > 2T+1                        -> was shifted, undo shift
//
// The shift of 2T+2 (not 2T+1) ensures the shifted range [3T+3,...] does
// not overlap the embedded range [-2T-1, 2T+1]. Specifically:
//   shifted e' = e + sign*(2T+2), |e| > T -> |e'| >= 2T+2 > 2T+1. OK.
//   shifted e' for exhausted embeddable: e in [-T,T], e' = e+sign*(2T+2),
//   |e'| >= 2T+2-T = T+2 > 2T+1 only if T < 2. For T=1: T+2=3 > 3? No, 3=3.
//   Hmm: for T=1, e=1 (max embeddable), shift up: e'=1+4=5 > 3=2T+1. OK.
//   For T=1, e=-1, shift down: e'=-1-4=-5, |e'|=5 > 3. OK.
//   For T=1, e=0, shift up: e'=0+4=4 > 3. OK. All clear.
// ---------------------------------------------------------------------------

static inline int diamond_predict(const vector<unsigned char>& img,
    uint32_t W, uint32_t H, uint32_t row, uint32_t col)
{
    int sum = 0, cnt = 0;
    if (row > 0)     { sum += img[(row - 1) * W + col]; ++cnt; }
    if (row + 1 < H) { sum += img[(row + 1) * W + col]; ++cnt; }
    if (col > 0)     { sum += img[row * W + col - 1];   ++cnt; }
    if (col + 1 < W) { sum += img[row * W + col + 1];   ++cnt; }
    if (cnt == 0) return 128;
    return sum / cnt;
}

static inline unsigned char get_share_bit(const vector<unsigned char>& share, size_t bit_idx)
{
    return static_cast<unsigned char>((share[bit_idx / 8] >> (bit_idx % 8)) & 1u);
}

// ---------------------------------------------------------------------------
// embed_phase
//
// parity=0: odd set (row+col odd)  - Phase 1, pred from original cover
// parity=1: even set (row+col even) - Phase 2, pred from Phase-1 stego
//
// With B >= 2T+2, no overflow is possible, so all branches are clean.
// ---------------------------------------------------------------------------
static void embed_phase(vector<unsigned char>& modified,
    const vector<unsigned char>& pred_buf,
    const vector<unsigned char>& orig_buf,
    uint32_t W, uint32_t H,
    int parity,
    const vector<unsigned char>& share,
    size_t& embedded_bits,
    size_t required_bits,
    int T, int B)
{
    const int shift_delta = 2 * T + 2;

    for (uint32_t row = 0; row < H; ++row) {
        uint32_t col0 = ((row % 2) == static_cast<uint32_t>(parity)) ? 1u : 0u;
        for (uint32_t col = col0; col < W; col += 2) {
            size_t pos = static_cast<size_t>(row) * W + col;
            int c = static_cast<int>(orig_buf[pos]);

            // Boundary skip
            if (c < B || c > 255 - B) {
                modified[pos] = static_cast<unsigned char>(c);
                continue;
            }

            int p = diamond_predict(pred_buf, W, H, row, col);
            int e = c - p;

            if (abs(e) <= T && embedded_bits < required_bits) {
                // Embed payload bit
                unsigned char bit = get_share_bit(share, embedded_bits);
                // c' = c + e + bit  (since c' = p + 2e + bit = (c-e) + 2e + bit)
                int cprime = c + e + static_cast<int>(bit);
                modified[pos] = static_cast<unsigned char>(cprime);
                ++embedded_bits;
            } else {
                // Shift: covers non-embeddable AND embeddable-but-exhausted
                int sign = (e >= 0) ? 1 : -1;
                int cprime = c + sign * shift_delta;
                modified[pos] = static_cast<unsigned char>(cprime);
            }
        }
    }
}

bool embed_share_pee_checker_safe(const TiffInfo& cover_info,
    const vector<unsigned char>& share,
    TiffInfo& out_info,
    int T, int B)
{
    if (T < 0 || B < 1) return false;
    // Enforce invariant: B >= 2*(2T+2) to guarantee that shifted pixels
    // never land in the boundary zone [0,B) or (255-B,255].
    // A pixel c in [B, 255-B] shifted by +(2T+2) gives c' = c+(2T+2).
    // We need c' <= 255-B, i.e. c <= 255-B-(2T+2) = 255-B-2T-2.
    // Since c <= 255-B, we need 255-B >= 255-B-2T-2, always true.
    // For downward shift: c' = c-(2T+2) >= B-(2T+2). We need c' >= B,
    // i.e. c >= B+(2T+2) = B+2T+2. Since c >= B, we need B >= 2T+2
    // AND c >= B+2T+2. The latter requires B >= 2*(2T+2) = 4T+4.
    const int min_B = 2 * (2 * T + 2); // = 4T+4
    if (B < min_B) B = min_B;

    const uint32_t W = cover_info.width;
    const uint32_t H = cover_info.height;
    if (W == 0 || H == 0) return false;

    // Phase 1 and Phase 2 each use a SEPARATE copy of modified so that
    // Phase 2's predictor reads Phase-1 stego values, not Phase-2 stego values.
    // We keep orig_phase1 = cover for Phase-1 orig lookup,
    // and stego_phase1 = result of Phase 1 for Phase-2 predictor.
    vector<unsigned char> stego_phase1 = cover_info.data;
    size_t embedded_bits = 0;
    const size_t required_bits = share.size() * 8;

    // Phase 1: odd set (row+col odd), predictor AND orig from original cover
    embed_phase(stego_phase1, cover_info.data, cover_info.data,
                W, H, 0, share, embedded_bits, required_bits, T, B);

    // Phase 2: even set (row+col even)
    // pred_buf = stego_phase1 (Phase-1 stego, frozen snapshot)
    // orig_buf = cover_info.data (original cover values)
    // We write into a new buffer so stego_phase1 stays unchanged as predictor
    vector<unsigned char> stego_final = stego_phase1;
    embed_phase(stego_final, stego_phase1, cover_info.data,
                W, H, 1, share, embedded_bits, required_bits, T, B);

    if (embedded_bits != required_bits) {
        cerr << "PEE-CheckerSafe embed: embedded " << embedded_bits
             << " of " << required_bits << " bits\n";
        return false;
    }

    out_info = cover_info;
    out_info.data = std::move(stego_final);
    return true;
}

// ---------------------------------------------------------------------------
// recover_phase
//
// Decoder partition (no side info):
//   sp in [0,B) or (255-B,255]  -> boundary-skipped: orig = sp
//   |e'| <= 2T+1                -> was embedded: extract bit, restore orig
//   |e'| > 2T+1                 -> was shifted: undo shift, restore orig
// ---------------------------------------------------------------------------
static void recover_phase(const vector<unsigned char>& stego,
    vector<unsigned char>& recovered,
    uint32_t W, uint32_t H,
    int parity,
    vector<unsigned char>& out_share,
    unsigned char& cur_byte,
    int& cur_bitpos,
    size_t& extracted_bits,
    size_t needed_bits,
    int T, int B)
{
    const int shift_delta = 2 * T + 2;

    for (uint32_t row = 0; row < H; ++row) {
        uint32_t col0 = ((row % 2) == static_cast<uint32_t>(parity)) ? 1u : 0u;
        for (uint32_t col = col0; col < W; col += 2) {
            size_t pos = static_cast<size_t>(row) * W + col;
            int sp = static_cast<int>(stego[pos]);

            // Boundary skip
            if (sp < B || sp > 255 - B) {
                recovered[pos] = static_cast<unsigned char>(sp);
                continue;
            }

            int p = diamond_predict(recovered, W, H, row, col);
            int eprime = sp - p;

            if (abs(eprime) <= 2 * T + 1) {
                // Was embedded: extract bit and restore original
                // c' = c + e + bit  =>  orig = c' - e - bit
                // e' = 2e + bit  =>  bit = e' & 1,  e = (e'-bit)/2
                // orig = p + e = p + (e'-bit)/2
                unsigned char bit = static_cast<unsigned char>(eprime & 1u);
                int e = (eprime - static_cast<int>(bit)) / 2;
                int orig = max(0, min(255, p + e));
                recovered[pos] = static_cast<unsigned char>(orig);

                if (extracted_bits < needed_bits) {
                    cur_byte |= static_cast<unsigned char>(bit << cur_bitpos);
                    if (++cur_bitpos == 8) {
                        out_share.push_back(cur_byte);
                        cur_byte = 0;
                        cur_bitpos = 0;
                    }
                    ++extracted_bits;
                }
            } else {
                // Was shifted: undo shift
                // c' = c + sign*(2T+2)  =>  orig = c' - sign*(2T+2)
                // sign of shift = sign of e' (since shift moves away from 0)
                int sign = (eprime >= 0) ? 1 : -1;
                // orig = p + e,  e' = e + sign*(2T+2)  =>  e = e' - sign*(2T+2)
                int e = eprime - sign * shift_delta;
                int orig = max(0, min(255, p + e));
                recovered[pos] = static_cast<unsigned char>(orig);
            }
        }
    }
}

bool recover_share_pee_checker_safe(const TiffInfo& stego_info,
    size_t share_len,
    vector<unsigned char>& out_share,
    TiffInfo& recovered_cover,
    int T, int B)
{
    if (T < 0 || B < 1) return false;
    const int min_B = 2 * (2 * T + 2);
    if (B < min_B) B = min_B;

    const uint32_t W = stego_info.width;
    const uint32_t H = stego_info.height;
    if (W == 0 || H == 0) return false;

    const size_t needed_bits = share_len * 8;
    // recovered starts as stego. Phase-1 stego values for odd pixels are
    // already correct here. Phase-2 stego values for even pixels are also here.
    vector<unsigned char> recovered = stego_info.data;
    // Keep a frozen copy of Phase-1 stego for use as Phase-2 predictor.
    // This matches what embed did: Phase-2 pred_buf = stego_phase1 (frozen).
    const vector<unsigned char> stego_phase1 = stego_info.data;

    out_share.clear();
    out_share.reserve(share_len);

    size_t extracted_bits = 0;
    unsigned char cur_byte = 0;
    int cur_bitpos = 0;

    // Recover Phase 2 first (even set).
    // pred_buf = stego_phase1 (frozen Phase-1 stego, same as embed used).
    // We pass stego_phase1 as the stego buffer AND as the predictor source.
    // recover_phase reads stego[pos] for the stego value and
    // diamond_predict(recovered,...) for the predictor.
    // But we need the predictor to come from stego_phase1, not recovered.
    // Solution: at the start of Phase-2 recovery, recovered == stego_info.data
    // == stego_phase1 for odd pixels (Phase-1 stego, unchanged by Phase 2).
    // Even pixels in recovered hold Phase-2 stego values.
    // The predictor for even pixels uses only odd neighbors -> stego_phase1 values. OK.
    // As we recover even pixels, we write original values into recovered.
    // Subsequent even pixels' predictors still only use odd neighbors -> still OK.
    recover_phase(stego_info.data, recovered, W, H, 1,
                  out_share, cur_byte, cur_bitpos, extracted_bits, needed_bits, T, B);

    // Recover Phase 1 (odd set).
    // At this point, recovered has: original even pixels (Phase-2 recovered),
    // Phase-1 stego odd pixels (not yet recovered).
    // Predictor for odd pixels uses even neighbors -> original values. OK.
    // During embed, Phase-1 pred_buf = cover_info.data -> original even values. OK.
    recover_phase(stego_info.data, recovered, W, H, 0,
                  out_share, cur_byte, cur_bitpos, extracted_bits, needed_bits, T, B);

    // Flush partial byte (needed_bits is always divisible by 8 for whole shares,
    // but guard anyway)
    if (cur_bitpos > 0) {
        out_share.push_back(cur_byte);
    }

    if (extracted_bits != needed_bits) {
        cerr << "PEE-CheckerSafe recover: extracted " << extracted_bits
             << " of " << needed_bits << " bits\n";
        return false;
    }

    recovered_cover = stego_info;
    recovered_cover.data = std::move(recovered);
    return true;
}
