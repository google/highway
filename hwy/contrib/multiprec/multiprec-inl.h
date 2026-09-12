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

// The schoolbook kernel costs O(kDigits^2). Karatsuba's 3-way split lowers
// that to ~O(kDigits^1.585) but adds additions, subtractions and copies, so
// it only pays off above this many digits.
constexpr size_t kWideMulKaratsubaMinDigits = 8;

#if HWY_TARGET != HWY_SCALAR

namespace detail {

// out[0..2*kDigits) = a * b, schoolbook. Both operands have kDigits digits,
// each < 2^52.
template <size_t kDigits, class D>
HWY_INLINE void SchoolbookMul(D d, const Vec<D>* HWY_RESTRICT a,
                              const Vec<D>* HWY_RESTRICT b,
                              Vec<D>* HWY_RESTRICT out) {
  // low[k] is the sum of the low 52 bits of a[i]*b[j] over all i+j == k;
  // high[k] is the sum of bits 52..103 of the same products. Both fit in 64
  // bits because there are at most kDigits terms, each < 2^52.
  Vec<D> low[2 * kDigits - 1];
  Vec<D> high[2 * kDigits - 1];
  for (size_t k = 0; k < 2 * kDigits - 1; ++k) {
    low[k] = Zero(d);
    high[k] = Zero(d);
  }
  for (size_t i = 0; i < kDigits; ++i) {
    for (size_t j = 0; j < kDigits; ++j) {
      low[i + j] = MulAdd52Lo(a[i], b[j], low[i + j]);
      high[i + j] = MulAdd52Hi(a[i], b[j], high[i + j]);
    }
  }

  const Vec<D> mask = Set(d, kWideMulDigitMask);
  Vec<D> carry = Zero(d);
  for (size_t k = 0; k + 1 < 2 * kDigits; ++k) {
    const Vec<D> sum = Add(low[k], carry);
    out[k] = And(sum, mask);
    carry = Add(ShiftRight<kWideMulDigitBits>(sum), high[k]);
  }
  out[2 * kDigits - 1] = And(carry, mask);
}

// out[i] = a[i] + b[i] for i < kCount, returning the carry-out digit.
template <class D>
HWY_INLINE Vec<D> AddDigitsCarry(D d, const Vec<D>* HWY_RESTRICT a,
                                 const Vec<D>* HWY_RESTRICT b, size_t kCount,
                                 Vec<D>* HWY_RESTRICT out) {
  const Vec<D> mask = Set(d, kWideMulDigitMask);
  Vec<D> carry = Zero(d);
  for (size_t i = 0; i < kCount; ++i) {
    const Vec<D> sum = Add(Add(a[i], b[i]), carry);
    out[i] = And(sum, mask);
    carry = ShiftRight<kWideMulDigitBits>(sum);
  }
  return carry;
}

// out[kOffset..kTotal) += src[0..kCount), with carry ripple. kCount is
// clamped so that nothing is written at or past kTotal.
template <size_t kOffset, size_t kCount, size_t kTotal, class D>
HWY_INLINE void AddInto(D d, Vec<D>* HWY_RESTRICT out,
                        const Vec<D>* HWY_RESTRICT src) {
  constexpr size_t kUsed = kCount <= kTotal - kOffset ? kCount : kTotal - kOffset;
  const Vec<D> mask = Set(d, kWideMulDigitMask);
  Vec<D> carry = Zero(d);
  for (size_t i = 0; i < kUsed; ++i) {
    const Vec<D> sum = Add(Add(out[kOffset + i], src[i]), carry);
    out[kOffset + i] = And(sum, mask);
    carry = ShiftRight<kWideMulDigitBits>(sum);
  }
  for (size_t i = kOffset + kUsed; i < kTotal; ++i) {
    const Vec<D> sum = Add(out[i], carry);
    out[i] = And(sum, mask);
    carry = ShiftRight<kWideMulDigitBits>(sum);
  }
}

// dst[0..kTotal) -= src[kOffset..kOffset + kCount), zero-extending src, with
// borrow ripple. The result must be non-negative.
template <size_t kOffset, size_t kCount, size_t kTotal, class D>
HWY_INLINE void SubInto(D d, Vec<D>* HWY_RESTRICT dst,
                        const Vec<D>* HWY_RESTRICT src) {
  const Vec<D> mask = Set(d, kWideMulDigitMask);
  const Vec<D> zero = Zero(d);
  const Vec<D> one = Set(d, 1);
  Vec<D> borrow = Zero(d);
  for (size_t i = 0; i < kTotal; ++i) {
    const Vec<D> sv = i < kCount ? src[kOffset + i] : zero;
    const Vec<D> sub = Add(sv, borrow);
    const Mask<D> underflow = Lt(dst[i], sub);
    dst[i] = And(Sub(dst[i], sub), mask);
    borrow = IfThenElse(underflow, one, zero);
  }
}

// out[0..2*kDigits) = a * b. Karatsuba's 3-way split above the threshold;
// schoolbook for small or odd sizes (whose split would be unbalanced).
template <size_t kDigits, class D>
HWY_INLINE void MulRec(D d, const Vec<D>* HWY_RESTRICT a,
                       const Vec<D>* HWY_RESTRICT b, Vec<D>* HWY_RESTRICT out) {
  if constexpr (kDigits < kWideMulKaratsubaMinDigits || (kDigits & 1) != 0) {
    SchoolbookMul<kDigits>(d, a, b, out);
  } else {
    constexpr size_t kHalf = kDigits / 2;
    constexpr size_t kTotal = 2 * kDigits;
    constexpr size_t kSumDigits = 2 * (kHalf + 1);

    for (size_t k = 0; k < kTotal; ++k) out[k] = Zero(d);

    {  // z0 = a_lo * b_lo, at digit 0.
      Vec<D> z0[2 * kHalf];
      MulRec<kHalf>(d, a, b, z0);
      AddInto<0, 2 * kHalf, kTotal>(d, out, z0);
    }
    {  // z2 = a_hi * b_hi, at digit 2*kHalf.
      Vec<D> z2[2 * kHalf];
      MulRec<kHalf>(d, a + kHalf, b + kHalf, z2);
      AddInto<2 * kHalf, 2 * kHalf, kTotal>(d, out, z2);
    }

    // s = a_lo + a_hi and t = b_lo + b_hi, each kHalf + 1 digits.
    Vec<D> s[kHalf + 1];
    Vec<D> t[kHalf + 1];
    s[kHalf] = AddDigitsCarry(d, a, a + kHalf, kHalf, s);
    t[kHalf] = AddDigitsCarry(d, b, b + kHalf, kHalf, t);

    // z1 = s*t - z0 - z2 = a_lo*b_hi + a_hi*b_lo. z0 and z2 are already in
    // `out` at digits 0 and 2*kHalf and nothing else has been added yet, so
    // subtract them from there.
    Vec<D> st[kSumDigits];
    MulRec<kHalf + 1>(d, s, t, st);
    SubInto<0, 2 * kHalf, kSumDigits>(d, st, out);
    SubInto<2 * kHalf, 2 * kHalf, kSumDigits>(d, st, out);
    AddInto<kHalf, kSumDigits, kTotal>(d, out, st);
  }
}

}  // namespace detail

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
    detail::MulRec<kDigits>(d, a, b, out);
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
