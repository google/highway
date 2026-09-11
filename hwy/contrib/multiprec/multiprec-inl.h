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

// Per-target include guard
#if defined(HIGHWAY_HWY_CONTRIB_MULTIPREC_MULTIPREC_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_MULTIPREC_MULTIPREC_INL_H_
#undef HIGHWAY_HWY_CONTRIB_MULTIPREC_MULTIPREC_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_MULTIPREC_MULTIPREC_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

#if HWY_TARGET != HWY_SCALAR

// Fixed-width unsigned integer multiplication for arbitrary precision.
//
// Digits are 52 bits ("nails"), the largest width for which the IFMA
// MulAdd52Lo/Hi ops are exact: for inputs < 2^52 the two together return the
// full 104-bit product exactly. Those ops exist on every target (native
// HWY_NATIVE_MULADD52 where available, the generic fallback otherwise), so
// this works everywhere without a separate fallback.
//
// Layout is SoA: digit i (little-endian) of each operand is passed in its own
// vector, `a[i]` / `b[i]`. The lanes of those vectors hold independent
// multiplications, which is how crypto (several RSA/ECC instances) uses it.

constexpr size_t kWideMulDigitBits = 52;
constexpr uint64_t kWideMulDigitMask = (uint64_t{1} << kWideMulDigitBits) - 1;

// Multiplies two kDigits-digit unsigned integers, each digit < 2^52, into
// 2*kDigits digits. `d` must be a u64 vector descriptor; `out` must have room
// for 2*kDigits digits. Input digits at or above 2^52 are ignored (as in
// MulAdd52*).
//
// kDigits is limited so that the per-digit accumulators cannot overflow 64
// bits: each accumulator holds at most kDigits products < 2^52.
template <size_t kDigits>
struct WideMul {
  static_assert(kDigits >= 1, "need at least one digit");
  static_assert(kDigits <= 2048, "accumulators may overflow");
  static constexpr size_t kNumResultDigits = 2 * kDigits;

  template <class D>
  static HWY_INLINE void Mul(D d, const Vec<D>* HWY_RESTRICT a,
                             const Vec<D>* HWY_RESTRICT b,
                             Vec<D>* HWY_RESTRICT out) {
    // low[k] is the sum of the low 52 bits of a[i]*b[j] over all i+j == k;
    // high[k] is the sum of bits 52..103 of the same products. Both fit in 64
    // bits because there are at most kDigits terms, each < 2^52.
    Vec<D> low[kNumResultDigits - 1];
    Vec<D> high[kNumResultDigits - 1];
    for (size_t k = 0; k < kNumResultDigits - 1; ++k) {
      low[k] = Zero(d);
      high[k] = Zero(d);
    }
    for (size_t i = 0; i < kDigits; ++i) {
      for (size_t j = 0; j < kDigits; ++j) {
        low[i + j] = MulAdd52Lo(a[i], b[j], low[i + j]);
        high[i + j] = MulAdd52Hi(a[i], b[j], high[i + j]);
      }
    }

    // dig_k = low[k] + high[k]*2^52 + carry, of which the low 52 bits are the
    // result digit and the rest carries into the next digit.
    const Vec<D> mask = Set(d, kWideMulDigitMask);
    Vec<D> carry = Zero(d);
    for (size_t k = 0; k < kNumResultDigits - 1; ++k) {
      const Vec<D> sum = Add(low[k], carry);
      out[k] = And(sum, mask);
      carry = Add(ShiftRight<kWideMulDigitBits>(sum), high[k]);
    }
    out[kNumResultDigits - 1] = And(carry, mask);
  }
};

// Bit-width form of the API above: kDigits = ceil(kBits / 52), so e.g.
// `WideMulBits<128>` multiplies 128-bit values held as 3 digits each.
template <size_t kBits>
using WideMulBits =
    WideMul<(kBits + kWideMulDigitBits - 1) / kWideMulDigitBits>;

#endif  // HWY_TARGET != HWY_SCALAR

namespace detail {

// Extracts kDigits 52-bit digits from kLimbs little-endian 64-bit limbs.
template <size_t kDigits, size_t kLimbs>
HWY_INLINE void DigitsFromLimbs(const uint64_t* HWY_RESTRICT limbs,
                                uint64_t* HWY_RESTRICT digits) {
  static_assert((kDigits - 1) * kWideMulDigitBits < kLimbs * 64,
                "digits do not fit in the limbs");
  for (size_t i = 0; i < kDigits; ++i) {
    const size_t bit = i * kWideMulDigitBits;
    const size_t limb = bit / 64;
    const size_t off = bit % 64;
    uint64_t d = limbs[limb] >> off;
    if (off + kWideMulDigitBits > 64) {
      d |= (limb + 1 < kLimbs ? limbs[limb + 1] : uint64_t{0})
           << (64 - off);
    }
    digits[i] = d & kWideMulDigitMask;
  }
}

// Packs kDigits 52-bit digits into kLimbs little-endian 64-bit limbs.
template <size_t kDigits, size_t kLimbs>
HWY_INLINE void LimbsFromDigits(const uint64_t* HWY_RESTRICT digits,
                                uint64_t* HWY_RESTRICT limbs) {
  size_t di = 0;
  size_t dbit = 0;
  for (size_t j = 0; j < kLimbs; ++j) {
    uint64_t v = 0;
    size_t vbits = 0;
    size_t need = 64;
    while (need != 0 && di < kDigits) {
      const size_t avail = kWideMulDigitBits - dbit;
      const size_t take = HWY_MIN(need, avail);
      const uint64_t part =
          (digits[di] >> dbit) & ((uint64_t{1} << take) - 1);
      v |= part << vbits;
      vbits += take;
      need -= take;
      dbit += take;
      if (dbit == kWideMulDigitBits) {
        ++di;
        dbit = 0;
      }
    }
    limbs[j] = v;
  }
}

// Scalar schoolbook on digits; same math as WideMul, one lane at a time.
template <size_t kDigits>
HWY_INLINE void MulDigits(const uint64_t* HWY_RESTRICT a,
                          const uint64_t* HWY_RESTRICT b,
                          uint64_t* HWY_RESTRICT out) {
  uint64_t low[2 * kDigits - 1] = {0};
  uint64_t high[2 * kDigits - 1] = {0};
  for (size_t i = 0; i < kDigits; ++i) {
    for (size_t j = 0; j < kDigits; ++j) {
      uint64_t hi;
      const uint64_t lo = Mul128(a[i], b[j], &hi);
      low[i + j] += lo & kWideMulDigitMask;
      high[i + j] +=
          (lo >> kWideMulDigitBits) | (hi << (64 - kWideMulDigitBits));
    }
  }
  uint64_t carry = 0;
  for (size_t k = 0; k + 1 < 2 * kDigits; ++k) {
    const uint64_t sum = low[k] + carry;
    out[k] = sum & kWideMulDigitMask;
    carry = (sum >> kWideMulDigitBits) + high[k];
  }
  out[2 * kDigits - 1] = carry & kWideMulDigitMask;
}

}  // namespace detail

// Multiplies two kBits-bit unsigned integers, both given as little-endian
// 64-bit limbs (kBits/64 each), and writes the 2*kBits/64-limb product.
// Scalar convenience over the 52-bit digit core, e.g. for u128/u192/u256.
// kBits must be a multiple of 64.
template <size_t kBits>
HWY_INLINE void WideMulLimbs(const uint64_t* HWY_RESTRICT a,
                             const uint64_t* HWY_RESTRICT b,
                             uint64_t* HWY_RESTRICT out) {
  static_assert(kBits % 64 == 0, "kBits must be a multiple of 64");
  constexpr size_t kLimbs = kBits / 64;
  constexpr size_t kDigits =
      (kBits + kWideMulDigitBits - 1) / kWideMulDigitBits;
  uint64_t da[kDigits];
  uint64_t db[kDigits];
  uint64_t dr[2 * kDigits];
  detail::DigitsFromLimbs<kDigits, kLimbs>(a, da);
  detail::DigitsFromLimbs<kDigits, kLimbs>(b, db);
  detail::MulDigits<kDigits>(da, db, dr);
  detail::LimbsFromDigits<2 * kDigits, 2 * kLimbs>(dr, out);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_MULTIPREC_MULTIPREC_INL_H_
