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
// kernel worked on four u32 lanes at a time, and this keeps that shape
// (CappedTag<uint32_t, 4>), so the 16 interleaved streams are decoded as four
// groups of four lanes.
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

// The kernel keeps four u32 lanes in local variables and drives them from a
// small loop, which needs a fixed-size (non-sizeless) 4-lane vector. That rules
// out HWY_SCALAR (1 lane) and the scalable targets (RVV, SVE), whose vector
// types cannot be array elements. Those get the scalar reference decoder, which
// produces identical output; a scalable kernel is a possible follow-up.
#if HWY_TARGET == HWY_SCALAR || HWY_HAVE_SCALABLE || HWY_TARGET_IS_SVE

HWY_INLINE bool DecodeInterleaved(const uint8_t* HWY_RESTRICT src,
                                  size_t comp_size, uint8_t* HWY_RESTRICT dst,
                                  size_t orig_size,
                                  const uint32_t* HWY_RESTRICT table) {
  return DecodeInterleavedScalar(src, comp_size, dst, orig_size, table);
}

#else

namespace detail {

HWY_INLINE uint32_t ReadBE24(const uint8_t*& src) {
  const uint32_t res = (static_cast<uint32_t>(src[0]) << 16) |
                       (static_cast<uint32_t>(src[1]) << 8) |
                       static_cast<uint32_t>(src[2]);
  src += 3;
  return res;
}

// Decodes one symbol per lane (4 total), writing the 4 symbol bytes to `out4`.
template <class D>
HWY_INLINE void Decode(D d, VFromD<D>& value, VFromD<D>& length,
                       const uint32_t* HWY_RESTRICT table,
                       VFromD<Repartition<uint8_t, D> > pack_idx,
                       uint8_t* HWY_RESTRICT out4) {
  const RebindToSigned<D> di;
  const RebindToFloat<D> df;
  const Repartition<uint8_t, D> d8;

  const VFromD<D> r = ShiftRight<kRangeProbBits>(length);
  const VFromD<D> q =
      And(BitCast(d, ConvertTo(di, Div(ConvertTo(df, BitCast(di, value)),
                                       ConvertTo(df, BitCast(di, r))))),
          Set(d, kRangeProbScale - 1));

  const VFromD<D> e = GatherIndex(d, table, BitCast(di, q));

  // Byte 0 of each u32 lane -> low 4 bytes.
  HWY_ALIGN uint8_t sym_bytes[16];
  Store(TableLookupBytesOr0(BitCast(d8, e), pack_idx), d8, sym_bytes);
  memcpy(out4, sym_bytes, 4);

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
  // CappedTag gives exactly 4 lanes on every fixed-size target.
  const CappedTag<uint32_t, 4> d;
  const Repartition<uint8_t, decltype(d)> d8;

  // Gathers byte 0 of each u32 lane into the low 4 bytes (rest zeroed).
  const VFromD<decltype(d8)> pack_idx =
      Dup128VecFromValues(d8, 0, 4, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
                          0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

  const RangeShuffleTables& sh = GetRangeShuffleTables();

  const uint8_t* src = src_start;
  const uint8_t* const src_end = src_start + comp_size;

  VFromD<decltype(d)> value[4];
  VFromD<decltype(d)> length[4];
  for (int g = 0; g < 4; ++g) {
    HWY_ALIGN uint32_t v4[4];
    for (int l = 0; l < 4; ++l) v4[l] = detail::ReadBE24(src);
    value[g] = Load(d, v4);
    length[g] = Set(d, kRangeMaxLen);
  }

  size_t dst_ofs = 0;
  for (; dst_ofs + kRangeLanes <= orig_size && src + 32 <= src_end;
       dst_ofs += kRangeLanes) {
    for (int g = 0; g < 4; ++g) {
      detail::Decode(d, value[g], length[g], table, pack_idx,
                     dst_start + dst_ofs + g * 4);
    }
    for (int g = 0; g < 4; ++g) {
      detail::Normalize(d, value[g], length[g], src, sh);
    }
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
  for (int g = 0; g < 4; ++g) {
    Store(value[g], d, vals + g * 4);
    Store(length[g], d, lens + g * 4);
  }

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

#endif  // scalar / scalable fallback

}  // namespace range_coder
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // include guard
