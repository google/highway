// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// SIMD interleaved range decoder. A single source that runs on every Highway
// target, decoding the bitstream produced by hwy::EncodeInterleaved (see
// range_coder.h). Ported from Richard Geldreich's public-domain
// "sserangecoding" (https://github.com/richgel999/sserangecoding); the SSE4.1
// kernel worked on four u32 lanes at a time, and this keeps that shape: the 16
// interleaved streams are four groups of four lanes, held in four vector
// locals and driven from an unrolled loop. The 4-lane vector is FixedTag, so
// the same kernel runs on RVV and SVE (masked to four lanes) as well as the
// fixed-size targets; only HWY_SCALAR, which cannot form a 4-lane vector, uses
// the scalar reference decoder. Widening the group to Lanes(d) would need the
// renormalization step (below) to be reworked away from its 4-lane pshufb
// tables; that is a possible follow-up.
//
// The float divide for value/range is exact because `value` is always <= 24
// bits and `range` <= 12 bits; ConvertTo truncates toward zero, as suggested by
// Jan Wassenberg for the original.

// Include guard (still compiled once per target)
#if defined(HIGHWAY_HWY_CONTRIB_CODER_RANGE_CODER_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_CODER_RANGE_CODER_INL_H_
#undef HIGHWAY_HWY_CONTRIB_CODER_RANGE_CODER_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_CODER_RANGE_CODER_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>
#include <string.h>  // memcpy, memset

#include "hwy/contrib/coder/range_coder.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace range_coder {

// HWY_SCALAR has a single lane and cannot form the 4-lane vector the kernel
// needs, so it decodes with the scalar reference implementation, which produces
// identical output.
#if HWY_TARGET == HWY_SCALAR

HWY_INLINE bool DecodeInterleaved(const uint8_t* HWY_RESTRICT src,
                                  size_t comp_size, uint8_t* HWY_RESTRICT dst,
                                  size_t orig_size,
                                  const uint32_t* HWY_RESTRICT table) {
  return DecodeInterleavedScalar(src, comp_size, dst, orig_size, table);
}

#else

namespace detail {

// Decodes one symbol per lane (4 total), writing the 4 symbol bytes to `out4`.
template <class D>
HWY_INLINE void Decode(D d, VFromD<D>& value, VFromD<D>& length,
                       const uint32_t* HWY_RESTRICT table,
                       uint8_t* HWY_RESTRICT out4) {
  const RebindToSigned<D> di;
  const RebindToFloat<D> df;

  const VFromD<D> r = ShiftRight<kRangeProbBits>(length);
  const VFromD<D> q =
      And(BitCast(d, ConvertTo(di, Div(ConvertTo(df, BitCast(di, value)),
                                       ConvertTo(df, BitCast(di, r))))),
          Set(d, kRangeProbScale - 1));

  const VFromD<D> e = GatherIndex(d, table, BitCast(di, q));

  // Byte 0 of each u32 lane is the symbol; keep the low byte of each lane.
  TruncateStore(e, d, out4);

  const VFromD<D> low_prob = And(ShiftRight<8>(e), Set(d, kRangeProbScale - 1));
  const VFromD<D> prob_range = ShiftRight<20>(e);  // 8 + kRangeProbBits

  value = Sub(value, Mul(low_prob, r));
  length = Mul(prob_range, r);
}

// Renormalizes 4 lanes, consuming up to 2 bytes per lane (<= 8 total) from
// `src`.
template <class D>
HWY_INLINE void Normalize(D d, VFromD<D>& value, VFromD<D>& length,
                          const uint8_t*& src, const RangeShuffleTables& sh) {
  const Repartition<uint8_t, D> d8;

  const uint64_t b0 = BitsFromMask(d, Lt(length, Set(d, kRangeMinLen)));
  const uint64_t b1 = BitsFromMask(d, Lt(length, Set(d, uint32_t{256})));
  const size_t msk = static_cast<size_t>(b0 | (b1 << 4));

  HWY_ALIGN uint8_t sb[16];
  memset(sb, 0, sizeof(sb));
  memcpy(sb, src, 8);

  const VFromD<decltype(d8)> shift = LoadU(d8, sh.shift[msk]);
  const VFromD<decltype(d8)> dist = LoadU(d8, sh.dist[msk]);

  value = BitCast(d, Or(TableLookupBytesOr0(BitCast(d8, value), shift),
                        TableLookupBytesOr0(LoadU(d8, sb), dist)));
  length = BitCast(d, TableLookupBytesOr0(BitCast(d8, length), shift));

  src += sh.num_bytes[msk];
}

}  // namespace detail

// Decodes data produced by hwy::EncodeInterleaved. `table` is from
// hwy::BuildDecodeTable. Returns false if the input is truncated. Output is
// identical to hwy::DecodeInterleavedScalar.
HWY_INLINE bool DecodeInterleaved(const uint8_t* HWY_RESTRICT src_start,
                                  size_t comp_size,
                                  uint8_t* HWY_RESTRICT dst_start,
                                  size_t orig_size,
                                  const uint32_t* HWY_RESTRICT table) {
  // 16 big-endian 24-bit words = 48 header bytes precede the renorm stream.
  if (comp_size < kRangeLanes * 3u) return false;

  // FixedTag: exactly 4 u32 lanes on every target, including RVV/SVE.
  const FixedTag<uint32_t, 4> d;
  const Repartition<uint8_t, decltype(d)> d8;

  const RangeShuffleTables& sh = GetRangeShuffleTables();

  const uint8_t* src = src_start + kRangeLanes * 3u;
  const uint8_t* const src_end = src_start + comp_size;

  // Load the four groups' initial `value` words from a zero-padded copy of the
  // 48-byte header, one 16-byte load + shuffle per group (the last group's
  // load would otherwise reach 4 bytes past the header). Each big-endian 24-bit
  // word b0 b1 b2 becomes the u32 (b0 << 16) | (b1 << 8) | b2.
  HWY_ALIGN uint8_t hdr[64];
  memset(hdr, 0, sizeof(hdr));
  memcpy(hdr, src_start, kRangeLanes * 3u);
  const VFromD<decltype(d8)> be24 = Dup128VecFromValues(
      d8, 2, 1, 0, 0x80, 5, 4, 3, 0x80, 8, 7, 6, 0x80, 11, 10, 9, 0x80);

  VFromD<decltype(d)> value0 =
      BitCast(d, TableLookupBytesOr0(LoadU(d8, hdr + 0), be24));
  VFromD<decltype(d)> value1 =
      BitCast(d, TableLookupBytesOr0(LoadU(d8, hdr + 12), be24));
  VFromD<decltype(d)> value2 =
      BitCast(d, TableLookupBytesOr0(LoadU(d8, hdr + 24), be24));
  VFromD<decltype(d)> value3 =
      BitCast(d, TableLookupBytesOr0(LoadU(d8, hdr + 36), be24));

  const VFromD<decltype(d)> full = Set(d, kRangeMaxLen);
  VFromD<decltype(d)> length0 = full;
  VFromD<decltype(d)> length1 = full;
  VFromD<decltype(d)> length2 = full;
  VFromD<decltype(d)> length3 = full;

  size_t dst_ofs = 0;
  for (; dst_ofs + kRangeLanes <= orig_size && src + 32 <= src_end;
       dst_ofs += kRangeLanes) {
    uint8_t* const out = dst_start + dst_ofs;
    detail::Decode(d, value0, length0, table, out + 0);
    detail::Decode(d, value1, length1, table, out + 4);
    detail::Decode(d, value2, length2, table, out + 8);
    detail::Decode(d, value3, length3, table, out + 12);
    detail::Normalize(d, value0, length0, src, sh);
    detail::Normalize(d, value1, length1, src, sh);
    detail::Normalize(d, value2, length2, src, sh);
    detail::Normalize(d, value3, length3, src, sh);
  }

  // Scalar tail. The vector loop stopped within 32 bytes of the end (or ran out
  // of output), so the remaining input is tiny; copy it into a zero-padded
  // buffer to keep every read in bounds, then finish byte-by-byte.
  const size_t tail_avail = static_cast<size_t>(src_end - src);
  uint8_t tail[64];
  memset(tail, 0, sizeof(tail));
  if (tail_avail > sizeof(tail)) return false;  // unreachable for valid input
  memcpy(tail, src, tail_avail);
  const uint8_t* tp = tail;

  HWY_ALIGN uint32_t vals[kRangeLanes];
  HWY_ALIGN uint32_t lens[kRangeLanes];
  Store(value0, d, vals + 0);
  Store(value1, d, vals + 4);
  Store(value2, d, vals + 8);
  Store(value3, d, vals + 12);
  Store(length0, d, lens + 0);
  Store(length1, d, lens + 4);
  Store(length2, d, lens + 8);
  Store(length3, d, lens + 12);

  RangeDecoder dec;
  for (; dst_ofs < orig_size; ++dst_ofs) {
    const uint32_t s = static_cast<uint32_t>(dst_ofs) & kRangeLaneMask;
    dec.length_ = lens[s];
    dec.value_ = vals[s];
    dst_start[dst_ofs] = static_cast<uint8_t>(dec.DecodeSymbol(table, tp));
    lens[s] = dec.length_;
    vals[s] = dec.value_;
  }

  // A valid stream consumes no more than the bytes it actually contains.
  return static_cast<size_t>(tp - tail) <= tail_avail;
}

#endif  // scalar fallback

}  // namespace range_coder
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // include guard
