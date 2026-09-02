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
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace iguana_ans {

namespace hi = hwy::iguana;

// The kernel keeps the 32 rANS states in fixed-size 4-lane-or-wider u32 vectors
// held in local arrays, which rules out HWY_SCALAR and the scalable targets
// (RVV/SVE, whose vectors cannot be array elements). Those use the scalar
// reference decoder, which produces identical output.
#if HWY_TARGET == HWY_SCALAR || HWY_HAVE_SCALABLE || HWY_TARGET_IS_SVE

HWY_INLINE bool Ans32Decode(const uint8_t* src, size_t src_size, uint8_t* dst,
                            size_t orig_size) {
  return hi::Ans32DecodeScalar(src, src_size, dst, orig_size);
}

#else

// Decodes one symbol per lane of `x`; returns the updated state and writes the
// `Lanes(d)` symbol bytes to `out`.
template <class D>
HWY_INLINE VFromD<D> DecodeLane(D d, VFromD<D> x,
                                const uint32_t* HWY_RESTRICT table,
                                uint8_t* HWY_RESTRICT out) {
  const RebindToSigned<D> di;
  const VFromD<D> slot = And(x, Set(d, hi::kAnsFreqMask));
  const VFromD<D> t = GatherIndex(d, table, BitCast(di, slot));
  const VFromD<D> freq = And(t, Set(d, hi::kAnsFreqMask));
  const VFromD<D> bias =
      And(ShiftRight<hi::kAnsWordMBits>(t), Set(d, hi::kAnsFreqMask));

  HWY_ALIGN uint32_t syms[16];
  StoreU(ShiftRight<24>(t), d, syms);
  for (size_t i = 0; i < Lanes(d); ++i) out[i] = static_cast<uint8_t>(syms[i]);

  return Add(Mul(freq, ShiftRight<hi::kAnsWordMBits>(x)), bias);
}

// Renormalizes the lanes of `x` whose state < 2^16, consuming 16-bit words from
// `p` (advanced by the number consumed). `forward` selects the read direction.
template <bool kForward, class D>
HWY_INLINE VFromD<D> RenormLane(D d, VFromD<D> x, const uint8_t*& p) {
  const Rebind<uint16_t, D> d16;
  const Repartition<uint8_t, decltype(d16)> d16_bytes;

  const MFromD<D> mask = Lt(x, Set(d, hi::kAnsWordL));
  const size_t cnt = CountTrue(d, mask);

  VFromD<decltype(d16)> words;
  HWY_IF_CONSTEXPR(kForward) {
    words = BitCast(d16, LoadU(d16_bytes, p));
    p += 2 * cnt;
  }
  else {
    words = Reverse(d16, BitCast(d16, LoadU(d16_bytes, p - 2 * Lanes(d))));
    p -= 2 * cnt;
  }
  const VFromD<D> expanded = Expand(PromoteTo(d, words), mask);
  return IfThenElse(mask, Or(ShiftLeft<hi::kAnsWordLBits>(x), expanded), x);
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

  const CappedTag<uint32_t, 16> d;
  const size_t n = Lanes(d);  // 4, 8 or 16
  const size_t nv = 16 / n;

  VFromD<decltype(d)> fwd[4];
  VFromD<decltype(d)> rev[4];
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
      fwd[g] = LoadU(d, s + g * n);
      rev[g] = LoadU(d, s + 16 + g * n);
    }
  }

  const uint8_t* pf = payload + 64;
  const uint8_t* pr = payload + payload_size - 64;
  size_t pos = 0;

  // Vectorized rounds, kept clear of the point where the two halves meet.
  while (pos + 32 <= orig_size && pf + 64 <= pr - 64) {
    for (size_t g = 0; g < nv; ++g) {
      fwd[g] = DecodeLane(d, fwd[g], tab, dst + pos + g * n);
    }
    for (size_t g = 0; g < nv; ++g) {
      rev[g] = DecodeLane(d, rev[g], tab, dst + pos + 16 + g * n);
    }
    pos += 32;
    for (size_t g = 0; g < nv; ++g) fwd[g] = RenormLane<true>(d, fwd[g], pf);
    for (size_t g = 0; g < nv; ++g) rev[g] = RenormLane<false>(d, rev[g], pr);
  }

  // Scalar tail: spill state and finish exactly like the reference.
  HWY_ALIGN uint32_t state[32];
  for (size_t g = 0; g < nv; ++g) {
    StoreU(fwd[g], d, state + g * n);
    StoreU(rev[g], d, state + 16 + g * n);
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

#endif  // scalar / scalable fallback

}  // namespace iguana_ans
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // include guard
