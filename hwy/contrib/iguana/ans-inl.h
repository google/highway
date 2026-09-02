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

// SIMD decoder for Iguana's ANS32 entropy stage (32-way interleaved rANS). A
// single source that runs on every Highway target and returns output identical
// to hwy::iguana::Ans32DecodeScalar. Ported from the Go reference (ans32.go).
//
// Each round decodes 32 symbols (16 "forward" lanes + 16 "reverse" lanes) with
// a gather into the dense table and one multiply-add, then renormalizes the
// lanes whose state fell below 2^16 by pulling one 16-bit word each from the
// forward / reverse halves of the payload - vectorized with Expand.
//
// State is held in up to 4 named vectors per half (fwd0..fwd3, rev0..rev3),
// not an array: RVV/SVE vectors are sizeless types and cannot be array
// elements. HWY_SCALAR is excluded at compile time (its 1-lane-only ops
// can't instantiate RenormLane's multi-lane Repartition/LoadU at all, not
// just uselessly -- confirmed by trying, not assumed). `Ans32DecodePayload`
// additionally falls back to the scalar reference decoder at *runtime* for
// a scalable (RVV/SVE) target whose hardware vector length is unusually
// small -- everywhere else, including RVV/SVE with a typical vector length,
// this vectorizes.

#if defined(HIGHWAY_HWY_CONTRIB_IGUANA_ANS_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_IGUANA_ANS_INL_H_
#undef HIGHWAY_HWY_CONTRIB_IGUANA_ANS_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_IGUANA_ANS_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>
#include <string.h>  // memcpy

#include "hwy/contrib/iguana/ans.h"
#include "hwy/contrib/iguana/ans_detail.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace iguana_ans {
namespace HWY_NAMESPACE {

namespace hi = hwy::iguana;
namespace hn = hwy::HWY_NAMESPACE;

// HWY_SCALAR is fundamentally 1-lane and doesn't support the multi-lane
// Repartition/LoadU that RenormLane needs (confirmed by trying to compile
// it, not assumed: D::kPrivateLanes==1 is baked into HWY_SCALAR's LoadU
// overload set, so instantiating RenormLane<HWY_SCALAR's D> is a hard
// compile error, independent of whether that code path ever runs -- C++
// instantiates called templates whether or not they're reachable at
// runtime). Scalable targets (RVV/SVE) don't have this problem -- they're
// genuine multi-lane SIMD, just with a width unknown until runtime -- so
// only HWY_SCALAR needs a compile-time exclusion; the runtime nv check in
// Ans32DecodePayload below (not a compile-time one) is what protects
// scalable targets with an unusually small hardware vector length.
#if HWY_TARGET == HWY_SCALAR

HWY_INLINE bool Ans32Decode(const uint8_t* src, size_t src_size, uint8_t* dst,
                            size_t orig_size) {
  return hi::Ans32DecodeScalar(src, src_size, dst, orig_size);
}

#else

// Decodes one symbol per lane of `x`; returns the updated state and writes the
// `Lanes(d)` symbol bytes to `out`.
template <class D>
HWY_INLINE hn::VFromD<D> DecodeLane(D d, hn::VFromD<D> x,
                                    const uint32_t* HWY_RESTRICT table,
                                    uint8_t* HWY_RESTRICT out) {
  const hn::RebindToSigned<D> di;
  const hn::VFromD<D> slot = hn::And(x, hn::Set(d, hi::kAnsFreqMask));
  const hn::VFromD<D> t = hn::GatherIndex(d, table, hn::BitCast(di, slot));
  const hn::VFromD<D> freq = hn::And(t, hn::Set(d, hi::kAnsFreqMask));
  const hn::VFromD<D> bias = hn::And(hn::ShiftRight<hi::kAnsWordMBits>(t),
                                     hn::Set(d, hi::kAnsFreqMask));

  const hn::Rebind<uint8_t, D> d8;
  hn::StoreU(hn::TruncateTo(d8, hn::ShiftRight<24>(t)), d8, out);

  return hn::MulAdd(freq, hn::ShiftRight<hi::kAnsWordMBits>(x), bias);
}

// Renormalizes the lanes of `x` whose state < 2^16, consuming 16-bit words from
// `p` (advanced by the number consumed). `forward` selects the read direction.
template <bool kForward, class D>
HWY_INLINE hn::VFromD<D> RenormLane(D d, hn::VFromD<D> x, const uint8_t*& p) {
  const hn::Rebind<uint16_t, D> d16;
  const hn::Repartition<uint8_t, decltype(d16)> d16_bytes;

  const hn::MFromD<D> mask = hn::Lt(x, hn::Set(d, hi::kAnsWordL));
  const size_t cnt = hn::CountTrue(d, mask);

  hn::VFromD<decltype(d16)> words;
  HWY_IF_CONSTEXPR(kForward) {
    words = hn::BitCast(d16, hn::LoadU(d16_bytes, p));
    p += 2 * cnt;
  }
  else {
    words = hn::Reverse(
        d16, hn::BitCast(d16, hn::LoadU(d16_bytes, p - 2 * hn::Lanes(d))));
    p -= 2 * cnt;
  }
  const hn::VFromD<D> expanded = hn::Expand(hn::PromoteTo(d, words), mask);
  return hn::IfThenElse(
      mask, hn::Or(hn::ShiftLeft<hi::kAnsWordLBits>(x), expanded), x);
}

// Decodes `payload` (the rANS data, without the frequency table) using an
// already-built dense `table`. Mirrors Ans32DecodePayloadScalar.
HWY_INLINE bool Ans32DecodePayload(const uint8_t* HWY_RESTRICT payload,
                                   size_t payload_size,
                                   const hi::AnsDenseTable& table,
                                   uint8_t* HWY_RESTRICT dst,
                                   size_t orig_size) {
  if (payload_size < 128) return false;
  const uint32_t* HWY_RESTRICT tab = table.data();

  const hn::CappedTag<uint32_t, 16> d;
  const size_t n = hn::Lanes(d);  // <= 16; typically 4, 8 or 16
  const size_t nv = n == 0 ? 0 : 16 / n;
  // Falls back to the scalar reference decoder for lane counts that don't
  // divide 16 into at most 4 groups -- only reachable here on a scalable
  // (RVV/SVE) target with an unusually narrow hardware vector length
  // (HWY_SCALAR is excluded at compile time above, never reaches this).
  // Produces identical output, just without vectorization for that rare case.
  if (n == 0 || nv == 0 || nv > 4 || n * nv != 16) {
    return hi::Ans32DecodePayloadScalar(payload, payload_size, table, dst,
                                        orig_size);
  }

  // Named variables, not an array: RVV/SVE vectors are sizeless types and
  // cannot be array elements. A fixed array of POINTERS to them is fine
  // (pointers are ordinary, fixed-size objects) and lets the loops below
  // stay index-based like the original, unrolled-by-hand version -- only
  // `nv` (<=4) of each are ever read.
  using V = hn::VFromD<decltype(d)>;
  V fwd0 = hn::Zero(d), fwd1 = hn::Zero(d), fwd2 = hn::Zero(d),
    fwd3 = hn::Zero(d);
  V rev0 = hn::Zero(d), rev1 = hn::Zero(d), rev2 = hn::Zero(d),
    rev3 = hn::Zero(d);
  V* const fwd[4] = {&fwd0, &fwd1, &fwd2, &fwd3};
  V* const rev[4] = {&rev0, &rev1, &rev2, &rev3};

  {
    HWY_ALIGN uint32_t s[32];
    const size_t rev_off = payload_size - 64;
    for (int lane = 0; lane < 16; ++lane) {
      s[lane] = static_cast<uint32_t>(payload[lane * 4]) |
                (static_cast<uint32_t>(payload[lane * 4 + 1]) << 8) |
                (static_cast<uint32_t>(payload[lane * 4 + 2]) << 16) |
                (static_cast<uint32_t>(payload[lane * 4 + 3]) << 24);
      const size_t o = rev_off + static_cast<size_t>(lane) * 4;
      s[lane + 16] = static_cast<uint32_t>(payload[o]) |
                     (static_cast<uint32_t>(payload[o + 1]) << 8) |
                     (static_cast<uint32_t>(payload[o + 2]) << 16) |
                     (static_cast<uint32_t>(payload[o + 3]) << 24);
    }
    for (size_t g = 0; g < nv; ++g) {
      *fwd[g] = hn::LoadU(d, s + g * n);
      *rev[g] = hn::LoadU(d, s + 16 + g * n);
    }
  }

  const uint8_t* pf = payload + 64;
  const uint8_t* pr = payload + payload_size - 64;
  size_t pos = 0;

  // Vectorized rounds, kept clear of the point where the two halves meet.
  // fwd/rev decode fused into one loop (likewise for renorm below): the two
  // halves are independent within a round, so interleaving them changes
  // neither the result nor which bytes of `pf`/`pr` each touches.
  while (pos + 32 <= orig_size && pf + 64 <= pr - 64) {
    for (size_t g = 0; g < nv; ++g) {
      *fwd[g] = DecodeLane(d, *fwd[g], tab, dst + pos + g * n);
      *rev[g] = DecodeLane(d, *rev[g], tab, dst + pos + 16 + g * n);
    }
    pos += 32;
    for (size_t g = 0; g < nv; ++g) {
      *fwd[g] = RenormLane<true>(d, *fwd[g], pf);
      *rev[g] = RenormLane<false>(d, *rev[g], pr);
    }
  }

  // Scalar tail: spill state and finish exactly like the reference.
  HWY_ALIGN uint32_t state[32];
  for (size_t g = 0; g < nv; ++g) {
    hn::StoreU(*fwd[g], d, state + g * n);
    hn::StoreU(*rev[g], d, state + 16 + g * n);
  }
  size_t cursor_fwd = static_cast<size_t>(pf - payload);
  size_t cursor_rev = static_cast<size_t>(pr - payload);

  for (;;) {
    bool stop = false;
    for (int lane = 0; lane < 32; ++lane) {
      const uint32_t x = state[lane];
      const uint32_t t = tab[x & hi::kAnsFreqMask];
      const uint32_t freq = t & hi::kAnsFreqMask;
      const uint32_t bias = (t >> hi::kAnsWordMBits) & hi::kAnsFreqMask;
      state[lane] = freq * (x >> hi::kAnsWordMBits) + bias;
      if (pos < orig_size) {
        dst[pos++] = static_cast<uint8_t>(t >> 24);
      } else {
        stop = true;
        break;
      }
    }
    if (stop) break;
    for (int lane = 0; lane < 16; ++lane) {
      if (state[lane] < hi::kAnsWordL) {
        if (cursor_fwd + 2 > cursor_rev) return false;
        state[lane] = (state[lane] << hi::kAnsWordLBits) |
                      (static_cast<uint32_t>(payload[cursor_fwd]) |
                       (static_cast<uint32_t>(payload[cursor_fwd + 1]) << 8));
        cursor_fwd += 2;
      }
    }
    for (int lane = 16; lane < 32; ++lane) {
      if (state[lane] < hi::kAnsWordL) {
        if (cursor_rev < cursor_fwd + 2) return false;
        state[lane] = (state[lane] << hi::kAnsWordLBits) |
                      (static_cast<uint32_t>(payload[cursor_rev - 2]) |
                       (static_cast<uint32_t>(payload[cursor_rev - 1]) << 8));
        cursor_rev -= 2;
      }
    }
  }
  return true;
}

// Decodes a full ANS32 block (rANS payload + serialized frequency table).
HWY_INLINE bool Ans32Decode(const uint8_t* HWY_RESTRICT src, size_t src_size,
                            uint8_t* HWY_RESTRICT dst, size_t orig_size) {
  hi::AnsDenseTable table;
  const size_t payload = hi::DeserializeAnsTable(table, src, src_size);
  if (payload == SIZE_MAX) return false;
  return Ans32DecodePayload(src, payload, table, dst, orig_size);
}

#endif  // HWY_TARGET == HWY_SCALAR

}  // namespace HWY_NAMESPACE
}  // namespace iguana_ans
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // include guard
