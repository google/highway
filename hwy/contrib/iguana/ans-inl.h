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
// vectors (16 / Lanes(d), so 1, 2 or 4) is a template argument kNumVectors, so
// each instantiation is straight-line `if constexpr`-guarded code the compiler
// keeps in registers. `Ans32DecodePayload` dispatches on it: a fixed target
// instantiates only the one group count its (compile-time) Lanes(d) produces; a
// scalable target instantiates all three. HWY_SCALAR is excluded at compile
// time because we require Repartition.

#if defined(HIGHWAY_HWY_CONTRIB_IGUANA_ANS_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
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

// HWY_SCALAR doesn't support Repartition for RenormLane.
#if HWY_TARGET == HWY_SCALAR

HWY_INLINE bool Ans32Decode(Span<const uint8_t> src, Span<uint8_t> dst) {
  return hi::Ans32DecodeScalar(src, dst);
}

#else

// Renormalizes the lanes of `x` whose state < 2^16, consuming 16-bit words from
// `p` (advanced by the number consumed). `forward` selects the read direction.
template <bool kForward, class D, class V = hn::VFromD<D> >
HWY_INLINE V RenormLane(D d, V x, const uint8_t*& p) {
  const hn::Rebind<uint16_t, D> du16;
  const hn::Repartition<uint8_t, decltype(du16)> du16_bytes;

  const hn::MFromD<D> mask = hn::Lt(x, hn::Set(d, hi::kAnsWordL));
  const size_t cnt = hn::CountTrue(d, mask);

  hn::VFromD<decltype(du16)> words;
  if constexpr (kForward) {
    words = hn::BitCast(du16, hn::LoadU(du16_bytes, p));
    p += 2 * cnt;
  } else {
    words = hn::Reverse(
        du16, hn::BitCast(du16, hn::LoadU(du16_bytes, p - 2 * hn::Lanes(d))));
    p -= 2 * cnt;
  }
#if HWY_IS_BIG_ENDIAN
  // The format stores these words little-endian (Read16LE in the scalar
  // decoder and in the scalar tail below), but BitCast reinterprets the loaded
  // bytes in host order, so swap them back. Lane order and byte order within a
  // lane are independent, hence this also applies after the Reverse above.
  words = hn::ReverseLaneBytes(words);
#endif
  const V expanded = hn::Expand(hn::PromoteTo(d, words), mask);
  return hn::IfThenElse(
      mask, hn::Or(hn::ShiftLeft<hi::kAnsWordLBits>(x), expanded), x);
}

// Decodes a pair of `N`-lane uint32_t state vectors `(x0, x1)` (`2 * N` lanes
// total) without `GatherIndex`, using the algebraic split:
//   state' = freq[sym] * (x >> 12) + (x & 0xFFF) - cum_freq[sym]
// where `sym = slot_to_sym[x & 0xFFF]` is written directly to `out0` and `out1`
// (`N` bytes each), and `(freq[sym], cum_freq[sym])` are looked up via
// `TwoTablesLookupLanes` (when the active alphabet fits in 2 vectors) or from
// the 1 KiB `sym_fc` L1 table.
template <class D, class V = hn::VFromD<D>,
          class DU16 = hn::Repartition<uint16_t, D>,
          class V16 = hn::VFromD<DU16>>
HWY_INLINE void DecodePairAlgebraic(
    D d, DU16 du16, size_t n, V& x0, V& x1,
    const uint8_t* HWY_RESTRICT slot_to_sym,
    const uint8_t* HWY_RESTRICT slot_to_compact, bool use_two_tables,
    V16 cf0, V16 cf1, V16 cc0, V16 cc1,
    const uint32_t* HWY_RESTRICT sym_fc, uint8_t* HWY_RESTRICT out0,
    uint8_t* HWY_RESTRICT out1) {
  const hn::Rebind<uint8_t, DU16> du8_2n;
  const V freq_mask = hn::Set(d, hi::kAnsFreqMask);
  const V slot0 = hn::And(x0, freq_mask);
  const V slot1 = hn::And(x1, freq_mask);

  HWY_ALIGN uint32_t slots[32];
  hn::Store(slot0, d, slots);
  hn::Store(slot1, d, slots + n);

  // When the active alphabet fits in two `du16` registers (`<= 4 * n` symbols,
  // i.e. <= 64 symbols on AVX-512, <= 32 on AVX2), a single
  // `TwoTablesLookupLanes` looks up all `2 * n` lanes in registers with zero
  // register spills.
  if (use_two_tables) {
    HWY_ALIGN uint8_t c_idx[32];
    for (size_t i = 0; i < n; ++i) {
      const uint32_t s0 = slots[i];
      const uint32_t s1 = slots[n + i];
      out0[i] = slot_to_sym[s0];
      out1[i] = slot_to_sym[s1];
      c_idx[i] = slot_to_compact[s0];
      c_idx[n + i] = slot_to_compact[s1];
    }
    const auto v_idx = hn::PromoteTo(du16, hn::Load(du8_2n, c_idx));
    const auto indices = hn::IndicesFromVec(du16, v_idx);
    const V16 freq16 = hn::TwoTablesLookupLanes(du16, cf0, cf1, indices);
    const V16 cum16 = hn::TwoTablesLookupLanes(du16, cc0, cc1, indices);

    const V bias0 = hn::Sub(slot0, hn::PromoteLowerTo(d, cum16));
    const V bias1 = hn::Sub(slot1, hn::PromoteUpperTo(d, cum16));
    x0 = hn::MulAdd(hn::PromoteLowerTo(d, freq16),
                    hn::ShiftRight<hi::kAnsWordMBits>(x0), bias0);
    x1 = hn::MulAdd(hn::PromoteUpperTo(d, freq16),
                    hn::ShiftRight<hi::kAnsWordMBits>(x1), bias1);
    return;
  }

  // For larger alphabets (> 4 * n unique symbols), use the algebraic split
  // (`4 KiB slot_to_sym` + `1 KiB sym_fc`) with L1D loads, avoiding both
  // `vpgatherdd` and multi-register shuffle cascades.
  HWY_ALIGN uint32_t fc[32];
  for (size_t i = 0; i < n; ++i) {
    const uint8_t sym0 = slot_to_sym[slots[i]];
    const uint8_t sym1 = slot_to_sym[slots[n + i]];
    out0[i] = sym0;
    out1[i] = sym1;
    fc[i] = sym_fc[sym0];
    fc[n + i] = sym_fc[sym1];
  }
  const V fc0 = hn::Load(d, fc);
  const V fc1 = hn::Load(d, fc + n);
  const V freq0 = hn::And(fc0, freq_mask);
  const V freq1 = hn::And(fc1, freq_mask);
  const V bias0 = hn::Sub(slot0, hn::ShiftRight<16>(fc0));
  const V bias1 = hn::Sub(slot1, hn::ShiftRight<16>(fc1));
  x0 = hn::MulAdd(freq0, hn::ShiftRight<hi::kAnsWordMBits>(x0), bias0);
  x1 = hn::MulAdd(freq1, hn::ShiftRight<hi::kAnsWordMBits>(x1), bias1);
}

// Decodes `payload` with `kNumVectors` (1, 2 or 4) vector groups per half.
template <size_t kNumVectors, class D>
HWY_INLINE bool Ans32DecodePayloadT(D d, size_t n,
                                    const uint8_t* HWY_RESTRICT payload,
                                    size_t payload_size,
                                    const uint32_t* HWY_RESTRICT tab,
                                    uint8_t* HWY_RESTRICT dst,
                                    size_t orig_size) {
  using V = hn::VFromD<D>;
  const hn::Repartition<uint16_t, D> du16;
  using V16 = hn::VFromD<decltype(du16)>;
  V fwd0 = hn::Zero(d), rev0 = hn::Zero(d);
  HWY_MAYBE_UNUSED V fwd1 = hn::Zero(d), fwd2 = hn::Zero(d), fwd3 = hn::Zero(d);
  HWY_MAYBE_UNUSED V rev1 = hn::Zero(d), rev2 = hn::Zero(d), rev3 = hn::Zero(d);

  {
    HWY_ALIGN uint32_t s[32];
    const size_t rev_off = payload_size - 64;
    for (int lane = 0; lane < 16; ++lane) {
      s[lane] = LoadLE32(payload + lane * 4);
      const size_t o = rev_off + static_cast<size_t>(lane) * 4;
      s[lane + 16] = LoadLE32(payload + o);
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

  // Build the algebraic split tables (`slot_to_sym[4096]` + 256-entry symbol
  // tables) in <= 256 steps (one per active symbol run). Skip the vectorized
  // loop only for the degenerate 1-symbol edge case where slot 4095 has a
  // synthetic sentinel entry.
  const bool is_single_sym =
      (tab[hi::kAnsWordM - 1] & hi::kAnsFreqMask) == 1 &&
      (tab[0] & hi::kAnsFreqMask) == hi::kAnsWordM - 1;

  HWY_ALIGN uint8_t slot_to_sym[hi::kAnsWordM];
  HWY_ALIGN uint8_t slot_to_compact[hi::kAnsWordM];
  HWY_ALIGN uint16_t compact_freq[64] = {};
  HWY_ALIGN uint16_t compact_cum[64] = {};
  HWY_ALIGN uint32_t sym_fc[256] = {};
  size_t num_unique = 0;
  if (HWY_LIKELY(!is_single_sym)) {
    uint32_t slot = 0;
    while (slot < hi::kAnsWordM) {
      const uint32_t t = tab[slot];
      const uint32_t sym = t >> 24;
      const uint32_t freq = t & hi::kAnsFreqMask;
      sym_fc[sym] = (slot << 16) | freq;
      if (num_unique < 64) {
        compact_freq[num_unique] = static_cast<uint16_t>(freq);
        compact_cum[num_unique] = static_cast<uint16_t>(slot);
      }
      memset(slot_to_sym + slot, static_cast<int>(sym), freq);
      ++num_unique;
      slot += freq;
    }
  }
  const bool use_two_tables = !is_single_sym && (num_unique <= 4 * n);
  V16 cf0 = hn::Zero(du16), cf1 = hn::Zero(du16);
  V16 cc0 = hn::Zero(du16), cc1 = hn::Zero(du16);
  if (use_two_tables) {
    uint32_t slot = 0;
    for (size_t idx = 0; idx < num_unique; ++idx) {
      const uint32_t freq = compact_freq[idx];
      memset(slot_to_compact + slot, static_cast<int>(idx), freq);
      slot += freq;
    }
    cf0 = hn::Load(du16, compact_freq);
    cf1 = hn::Load(du16, compact_freq + 2 * n);
    cc0 = hn::Load(du16, compact_cum);
    cc1 = hn::Load(du16, compact_cum + 2 * n);
  }

  // Vectorized rounds, kept clear of the point where the two halves meet.
  while (HWY_LIKELY(!is_single_sym) && pos + 32 <= orig_size &&
         pf + 64 <= pr - 64) {
    DecodePairAlgebraic(d, du16, n, fwd0, rev0, slot_to_sym, slot_to_compact,
                        use_two_tables, cf0, cf1, cc0, cc1, sym_fc,
                        dst + pos + 0 * n, dst + pos + 16 + 0 * n);
    if constexpr (kNumVectors >= 2) {
      DecodePairAlgebraic(d, du16, n, fwd1, rev1, slot_to_sym, slot_to_compact,
                          use_two_tables, cf0, cf1, cc0, cc1, sym_fc,
                          dst + pos + 1 * n, dst + pos + 16 + 1 * n);
    }
    if constexpr (kNumVectors >= 4) {
      DecodePairAlgebraic(d, du16, n, fwd2, rev2, slot_to_sym, slot_to_compact,
                          use_two_tables, cf0, cf1, cc0, cc1, sym_fc,
                          dst + pos + 2 * n, dst + pos + 16 + 2 * n);
      DecodePairAlgebraic(d, du16, n, fwd3, rev3, slot_to_sym, slot_to_compact,
                          use_two_tables, cf0, cf1, cc0, cc1, sym_fc,
                          dst + pos + 3 * n, dst + pos + 16 + 3 * n);
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
HWY_INLINE bool Ans32DecodePayload(Span<const uint8_t> src,
                                   const hi::AnsDenseTable& table,
                                   Span<uint8_t> out) {
  const size_t payload_size = src.size();
  if (payload_size < 128) return false;
  // Every lookup is an unchecked gather at (state & kAnsFreqMask); the table
  // size is ensured via std::array.
  const uint32_t* HWY_RESTRICT tab = table.data();
  // Unpacked here so the inner loop keeps the restrict qualifiers.
  const uint8_t* HWY_RESTRICT payload = src.data();
  uint8_t* HWY_RESTRICT dst = out.data();
  const size_t orig_size = out.size();

  const hn::CappedTag<uint32_t, 16> d;
  HWY_LANES_CONSTEXPR size_t N = hn::Lanes(d);
  HWY_DASSERT(N == 4 || N == 8 || N == 16);
  const size_t nv = 16 / N;  // 1..4

  if constexpr (!HWY_HAVE_SCALABLE) {
    // Non-scalable and full-width: Lanes(d) is constexpr, and we only
    // instantiate once.
    constexpr size_t kNV = 16 / size_t{HWY_MAX_LANES_D(decltype(d))};
    if constexpr (hn::detail::IsFull(d)) {
      return Ans32DecodePayloadT<kNV>(d, N, payload, payload_size, tab, dst,
                                      orig_size);
    } else {
      // Partial vector, but still enough for SIMD.
      if (N * kNV == 16) {
        return Ans32DecodePayloadT<kNV>(d, N, payload, payload_size, tab, dst,
                                        orig_size);
      } else {
        return hi::Ans32DecodePayloadScalar(src, table, out);
      }
    }
  }

  if constexpr (HWY_HAVE_SCALABLE) {
    if (nv == 4) {
      return Ans32DecodePayloadT<4>(d, N, payload, payload_size, tab, dst,
                                    orig_size);
    }
    if (nv == 2) {
      return Ans32DecodePayloadT<2>(d, N, payload, payload_size, tab, dst,
                                    orig_size);
    }
    if (nv == 1) {
      return Ans32DecodePayloadT<1>(d, N, payload, payload_size, tab, dst,
                                    orig_size);
    }

    if constexpr (hn::detail::IsFull(d)) {
      HWY_DASSERT(false);  // unreachable: fulls scalable >= 128 bits.
      return false;
    } else {  // < 128 bit vectors: fall back to scalar.
      return hi::Ans32DecodePayloadScalar(src, table, out);
    }
  }
}

// Decodes a full ANS32 block (rANS payload + serialized frequency table).
HWY_INLINE bool Ans32Decode(Span<const uint8_t> src, Span<uint8_t> dst) {
  HWY_ALIGN hi::AnsDenseTable table;
  const size_t payload = hi::DeserializeAnsTable(table, src.data(), src.size());
  if (payload == SIZE_MAX) return false;
  return Ans32DecodePayload(src.first(payload), table, dst);
}

#endif  // HWY_TARGET == HWY_SCALAR

}  // namespace HWY_NAMESPACE
}  // namespace iguana_ans
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // include guard
