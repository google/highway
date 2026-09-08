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
// The 16+16 states are held in named vector locals (fwd0..fwd3, rev0..rev3):
// RVV/SVE vectors are sizeless and cannot be array elements. The number of
// groups (16 / Lanes(d), so 1, 2 or 4) is a template argument kNumVectors, so
// each instantiation is straight-line `if constexpr`-guarded code the compiler
// keeps in registers -- no pointer-to-vector array. `Ans32DecodePayload`
// dispatches on it: a fixed target instantiates only the one group count its
// (compile-time) Lanes(d) produces; a scalable target instantiates all three.
// HWY_SCALAR is excluded at compile time (its 1-lane ops can't instantiate
// RenormLane's multi-lane Repartition/LoadU -- confirmed by trying), and a
// scalable target with an unusually small hardware vector length falls back to
// the scalar reference decoder at runtime.

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

// Decodes `payload` with `kNumVectors` (1, 2 or 4) vector groups per half.
// The group count is a template argument so the unrolled body is straight-line
// code in named locals rather than a runtime loop over an array of pointers.
template <size_t kNumVectors, class D>
HWY_INLINE bool Ans32DecodePayloadT(D d, size_t n,
                                    const uint8_t* HWY_RESTRICT payload,
                                    size_t payload_size,
                                    const uint32_t* HWY_RESTRICT tab,
                                    uint8_t* HWY_RESTRICT dst,
                                    size_t orig_size) {
  using V = hn::VFromD<D>;
  V fwd0 = hn::Zero(d), rev0 = hn::Zero(d);
  HWY_MAYBE_UNUSED V fwd1 = hn::Zero(d), fwd2 = hn::Zero(d), fwd3 = hn::Zero(d);
  HWY_MAYBE_UNUSED V rev1 = hn::Zero(d), rev2 = hn::Zero(d), rev3 = hn::Zero(d);

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
    fwd0 = hn::LoadU(d, s + 0 * n);
    rev0 = hn::LoadU(d, s + 16 + 0 * n);
    if constexpr (kNumVectors >= 2) {
      fwd1 = hn::LoadU(d, s + 1 * n);
      rev1 = hn::LoadU(d, s + 16 + 1 * n);
    }
    if constexpr (kNumVectors >= 4) {
      fwd2 = hn::LoadU(d, s + 2 * n);
      rev2 = hn::LoadU(d, s + 16 + 2 * n);
      fwd3 = hn::LoadU(d, s + 3 * n);
      rev3 = hn::LoadU(d, s + 16 + 3 * n);
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
    fwd0 = DecodeLane(d, fwd0, tab, dst + pos + 0 * n);
    rev0 = DecodeLane(d, rev0, tab, dst + pos + 16 + 0 * n);
    if constexpr (kNumVectors >= 2) {
      fwd1 = DecodeLane(d, fwd1, tab, dst + pos + 1 * n);
      rev1 = DecodeLane(d, rev1, tab, dst + pos + 16 + 1 * n);
    }
    if constexpr (kNumVectors >= 4) {
      fwd2 = DecodeLane(d, fwd2, tab, dst + pos + 2 * n);
      rev2 = DecodeLane(d, rev2, tab, dst + pos + 16 + 2 * n);
      fwd3 = DecodeLane(d, fwd3, tab, dst + pos + 3 * n);
      rev3 = DecodeLane(d, rev3, tab, dst + pos + 16 + 3 * n);
    }
    pos += 32;
    fwd0 = RenormLane<true>(d, fwd0, pf);
    rev0 = RenormLane<false>(d, rev0, pr);
    if constexpr (kNumVectors >= 2) {
      fwd1 = RenormLane<true>(d, fwd1, pf);
      rev1 = RenormLane<false>(d, rev1, pr);
    }
    if constexpr (kNumVectors >= 4) {
      fwd2 = RenormLane<true>(d, fwd2, pf);
      rev2 = RenormLane<false>(d, rev2, pr);
      fwd3 = RenormLane<true>(d, fwd3, pf);
      rev3 = RenormLane<false>(d, rev3, pr);
    }
  }

  // Scalar tail: spill state and finish exactly like the reference.
  HWY_ALIGN uint32_t state[32];
  hn::StoreU(fwd0, d, state + 0 * n);
  hn::StoreU(rev0, d, state + 16 + 0 * n);
  if constexpr (kNumVectors >= 2) {
    hn::StoreU(fwd1, d, state + 1 * n);
    hn::StoreU(rev1, d, state + 16 + 1 * n);
  }
  if constexpr (kNumVectors >= 4) {
    hn::StoreU(fwd2, d, state + 2 * n);
    hn::StoreU(rev2, d, state + 16 + 2 * n);
    hn::StoreU(fwd3, d, state + 3 * n);
    hn::StoreU(rev3, d, state + 16 + 3 * n);
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

// Decodes `payload` (the rANS data, without the frequency table) using an
// already-built dense `table`. Mirrors Ans32DecodePayloadScalar. Computes the
// group count (16 / Lanes(d)) at runtime and dispatches to the matching
// Ans32DecodePayloadT instantiation; the `if constexpr` guards keep a
// fixed-size target from compiling the group counts it can never see.
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

  if constexpr (!HWY_HAVE_SCALABLE) {
    // Fixed target: Lanes(d) is a compile-time constant, so only the one
    // group count it produces is instantiated.
    constexpr size_t kNV = 16 / size_t{HWY_MAX_LANES_D(decltype(d))};
    if (n * kNV == 16) {
      return Ans32DecodePayloadT<kNV>(d, n, payload, payload_size, tab, dst,
                                      orig_size);
    }
  } else if (n * nv == 16) {
    // Scalable target: the hardware vector length picks nv at runtime.
    if (nv == 4) {
      return Ans32DecodePayloadT<4>(d, n, payload, payload_size, tab, dst,
                                    orig_size);
    }
    if (nv == 2) {
      return Ans32DecodePayloadT<2>(d, n, payload, payload_size, tab, dst,
                                    orig_size);
    }
    if (nv == 1) {
      return Ans32DecodePayloadT<1>(d, n, payload, payload_size, tab, dst,
                                    orig_size);
    }
  }

  // Lane counts that don't split 16 into 1/2/4 groups -- only reachable on a
  // scalable target with an unusually narrow vector length. Identical output,
  // just not vectorized.
  return hi::Ans32DecodePayloadScalar(payload, payload_size, table, dst,
                                      orig_size);
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
