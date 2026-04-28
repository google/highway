// Copyright 2024 Google LLC
// Copyright 2026 Fujitsu Limited
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
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

/**********************************************************************************
 ** Integer division
 **********************************************************************************
 *
 * Most SIMD ISAs do not provide a portable integer vector division operation.
 * This implementation replaces division by a run-time invariant scalar divisor
 * with multiplication by precomputed reciprocal parameters and shifts.
 *
 * The method is based on T. Granlund and P. L. Montgomery,
 * "Division by invariant integers using multiplication" (Figures 4.1 and 5.1):
 *   https://gmplib.org/~tege/divcnst-pldi94.pdf
 *
 * hwy/base.h already provides scalar classes `hwy::Divisor` (uint32_t) and
 * `hwy::Divisor64` (uint64_t, gated on HWY_HAVE_DIV128) implementing the same
 * scheme. This file extends it to SIMD vectors and adds:
 *  - Signed division (Figure 5.1).
 *  - 8-bit and 16-bit lanes via widening multiply.
 *  - Power-of-two fast path.
 *  - Floor-division variant (Python/NumPy semantics).
 *
 * Usage is split into two steps:
 *  1) Precompute parameters from the scalar divisor:
 *       DivisorParams{U,S}<T> ComputeDivisorParams(T divisor);
 *
 *  2) Use those parameters to divide vector lanes:
 *       Vec<D> IntDiv(D d, Vec<D> dividend,
 *                     const DivisorParams{U,S}<T>& params);
 *
 * Computing divisor parameters is relatively expensive, so this is intended
 * for divisors reused across multiple vector operations.
 *
 * For 64-bit lanes, some targets use a scalar fallback. This is required when
 * 128-bit arithmetic is unavailable, and is also used for NEON/PPC8/VSX where
 * the vectorized 64-bit reciprocal-multiply path is not expected to outperform
 * scalar division. Array-level APIs skip the vector round-trip in this case.
 *
 ***************************************************************
 ** Figure 4.1: Unsigned division by run-time invariant divisor
 ***************************************************************
 * Initialization (given uword d with 1 <= d < 2^N):
 *    int l   = ceil(log2(d));
 *    uword m = 2^N * (2^l - d) / d + 1;
 *    int sh1 = min(l, 1);
 *    int sh2 = max(l - 1, 0);
 *
 * For q = FLOOR(a/d), all uword:
 *    uword t1 = MULUH(m, a);
 *    q = SRL(t1 + SRL(a - t1, sh1), sh2);
 *
 ************************************************************************************
 ** Figure 5.1: Signed division by run-time invariant divisor, rounded toward zero
 ************************************************************************************
 * Initialization (given sword d with d != 0):
 *    int l       = max(ceil(log2(abs(d))), 1);
 *    udword m0   = 1 + (2^(N+l-1)) / abs(d);
 *    sword  m    = m0 - 2^N;
 *    sword dsign = XSIGN(d);
 *    int sh      = l - 1;
 *
 * For q = TRUNC(a/d), all sword:
 *    sword q0 = a + MULSH(m, a);
 *          q0 = SRA(q0, sh) - XSIGN(a);
 *    q = EOR(q0, dsign) - dsign;
 *
 **********************************************************************************/


// Per-target include guard
#if defined(HIGHWAY_HWY_CONTRIB_INTDIV_INTDIV_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_INTDIV_INTDIV_INL_H_
#undef HIGHWAY_HWY_CONTRIB_INTDIV_INTDIV_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_INTDIV_INTDIV_INL_H_
#endif

#include <cstddef>
#include <cstdint>
#include <limits>
#include "hwy/highway.h"

#ifdef HWY_INTDIV_SCALAR64
#undef HWY_INTDIV_SCALAR64
#endif

#if !HWY_HAVE_DIV128 || HWY_TARGET_IS_NEON || HWY_TARGET == HWY_PPC8 || HWY_TARGET == HWY_VSX
#define HWY_INTDIV_SCALAR64 1
#else
#define HWY_INTDIV_SCALAR64 0
#endif

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

namespace detail {

template <typename T, bool kNeedsWiden = (sizeof(T) < 4)>
struct MultiplierType {
  using type = T;
};
template <typename T>
struct MultiplierType<T, true> {
  using type = MakeWide<T>;
};
template <typename T>
using MultiplierType_T = typename MultiplierType<T>::type;

template <typename T>
HWY_INLINE constexpr bool IsPow2(T x) {
  return x > 0 && (x & (x - 1)) == 0;
}

HWY_INLINE constexpr int CountTrailingZeros32(uint32_t x) {
  return x == 0 ? 32 : static_cast<int>(Num0BitsBelowLS1Bit_Nonzero32(x));
}
HWY_INLINE constexpr int CountTrailingZeros64(uint64_t x) {
  return x == 0 ? 64 : static_cast<int>(Num0BitsBelowLS1Bit_Nonzero64(x));
}
HWY_INLINE constexpr unsigned LeadingZeroCount32(uint32_t x) {
  return x == 0 ? 32u : static_cast<unsigned>(Num0BitsAboveMS1Bit_Nonzero32(x));
}
HWY_INLINE constexpr unsigned LeadingZeroCount64(uint64_t x) {
  return x == 0 ? 64u : static_cast<unsigned>(Num0BitsAboveMS1Bit_Nonzero64(x));
}

#if HWY_HAVE_DIV128
HWY_INLINE uint64_t DivideHighBy(uint64_t high, uint64_t divisor) {
  HWY_DASSERT(divisor != 0);
#if HWY_COMPILER_MSVC >= 1920 && HWY_ARCH_X86_64
  unsigned __int64 remainder;
  return _udiv128(high, uint64_t{0}, divisor, &remainder);
#else
  using u128 = unsigned __int128;
  const u128 hi128 = static_cast<u128>(high) << 64;
  return static_cast<uint64_t>(hi128 / static_cast<u128>(divisor));
#endif
}
#endif  // HWY_HAVE_DIV128

template <class D, class V = Vec<D>>
HWY_INLINE V ShiftRightUniform(D d, V v, int sh) {
  HWY_DASSERT(sh >= 0);
  HWY_DASSERT(sh < static_cast<int>(sizeof(TFromD<D>) * 8));
#if HWY_TARGET_IS_NEON || HWY_TARGET == HWY_AVX2 || HWY_TARGET <= HWY_AVX3
  (void)d;
  return ShiftRightSame(v, sh);
#else
  using T = TFromD<D>;
  return Shr(v, Set(d, static_cast<T>(sh)));
#endif
}

template <class D, class V = Vec<D>, typename T = TFromD<D>>
HWY_INLINE V ScalarDivPerLane(D d, V dividend, T divisor) {
  const size_t N = Lanes(d);
  HWY_ALIGN T buf[HWY_MAX_BYTES / sizeof(T)];
  StoreU(dividend, d, buf);
  for (size_t i = 0; i < N; ++i) {
    buf[i] = static_cast<T>(buf[i] / divisor);
  }
  return LoadU(d, buf);
}

}  // namespace detail

template <typename T>
struct DivisorParamsU {
  detail::MultiplierType_T<T> multiplier;
  int shift2;
  bool is_pow2;
  int pow2_shift;
  T divisor;
};

template <typename T>
struct DivisorParamsS {
  detail::MultiplierType_T<T> multiplier;
  int shift;
  T divisor;
  T dsign;
  bool is_pow2;
  bool is_neg_one;
  int pow2_shift;
};

template <typename T, HWY_IF_T_SIZE(T, 1), HWY_IF_UNSIGNED(T)>
HWY_INLINE constexpr DivisorParamsU<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  DivisorParamsU<T> params{};
  params.divisor = divisor;

  if (detail::IsPow2(divisor)) {  // also catches divisor == 1
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(divisor);
    params.multiplier = 1;
    params.shift2 = 0;
    return params;
  }
  params.is_pow2 = false;
  params.pow2_shift = 0;

  const unsigned l = 32u - detail::LeadingZeroCount32(divisor - 1u);
  const uint16_t two_l = static_cast<uint16_t>(1U << l);
  const uint32_t m = ((static_cast<uint32_t>(two_l - divisor) << 8) / divisor) + 1u;
  params.multiplier = static_cast<uint16_t>(m);
  params.shift2 = static_cast<int>(l) - 1;
  return params;
}

template <typename T, HWY_IF_T_SIZE(T, 2), HWY_IF_UNSIGNED(T)>
HWY_INLINE constexpr DivisorParamsU<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  DivisorParamsU<T> params{};
  params.divisor = divisor;

  if (detail::IsPow2(divisor)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(divisor);
    params.multiplier = 1;
    params.shift2 = 0;
    return params;
  }
  params.is_pow2 = false;
  params.pow2_shift = 0;

  const unsigned l = 32u - detail::LeadingZeroCount32(divisor - 1u);
  const uint32_t two_l = 1U << l;
  const uint64_t tmp = ((static_cast<uint64_t>(two_l - divisor) << 16) / divisor) + 1u;
  params.multiplier = static_cast<uint32_t>(tmp);
  params.shift2 = static_cast<int>(l) - 1;
  return params;
}

template <typename T, HWY_IF_T_SIZE(T, 4), HWY_IF_UNSIGNED(T)>
HWY_INLINE constexpr DivisorParamsU<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  DivisorParamsU<T> params{};
  params.divisor = divisor;

  if (detail::IsPow2(divisor)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(divisor);
    params.multiplier = 1;
    params.shift2 = 0;
    return params;
  }
  params.is_pow2 = false;
  params.pow2_shift = 0;

  const unsigned l = 32u - detail::LeadingZeroCount32(divisor - 1u);
  const uint64_t two_l = 1ULL << l;
  const uint64_t m = ((two_l - divisor) << 32) / divisor + 1u;
  params.multiplier = static_cast<T>(m);
  params.shift2 = static_cast<int>(l) - 1;
  return params;
}

template <typename T, HWY_IF_T_SIZE(T, 8), HWY_IF_UNSIGNED(T)>
HWY_INLINE
#if !HWY_INTDIV_SCALAR64 && (!HWY_COMPILER_MSVC || !HWY_ARCH_X86_64)
    constexpr
#endif
    DivisorParamsU<T>
    ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  DivisorParamsU<T> params{};
  params.divisor = divisor;

  if (detail::IsPow2(divisor)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros64(divisor);
    params.multiplier = 1;
    params.shift2 = 0;
    return params;
  }
  params.is_pow2 = false;
  params.pow2_shift = 0;

#if HWY_INTDIV_SCALAR64
  params.multiplier = 1;
  params.shift2 = 0;
#else
  const unsigned l = 64u - detail::LeadingZeroCount64(divisor - 1u);
  const uint64_t two_l_minus_d = (l < 64) ? ((1ULL << l) - divisor) : (0 - divisor);
  const uint64_t m = detail::DivideHighBy(two_l_minus_d, divisor) + 1u;
  params.multiplier = m;
  params.shift2 = static_cast<int>(l) - 1;
#endif
  return params;
}

template <typename T, HWY_IF_T_SIZE(T, 1), HWY_IF_SIGNED(T)>
HWY_INLINE constexpr DivisorParamsS<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  using UT = MakeUnsigned<T>;
  DivisorParamsS<T> params{};
  params.divisor = divisor;
  params.dsign = (divisor < 0) ? static_cast<T>(-1) : static_cast<T>(0);
  params.is_neg_one = (divisor == T(-1));

  const UT abs_d = divisor < 0 ? static_cast<UT>(UT{0} - static_cast<UT>(divisor))
                               : static_cast<UT>(divisor);

  if (detail::IsPow2(abs_d)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(abs_d);
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  params.is_pow2 = false;
  params.pow2_shift = 0;

  const unsigned sh = 31u - detail::LeadingZeroCount32(static_cast<uint32_t>(abs_d - 1u));
  const uint32_t m = (256U << sh) / abs_d + 1u;
  params.multiplier = static_cast<int16_t>(static_cast<T>(m));
  params.shift = static_cast<int>(sh);
  return params;
}

template <typename T, HWY_IF_T_SIZE(T, 2), HWY_IF_SIGNED(T)>
HWY_INLINE constexpr DivisorParamsS<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  using UT = MakeUnsigned<T>;
  DivisorParamsS<T> params{};
  params.divisor = divisor;
  params.dsign = (divisor < 0) ? static_cast<T>(-1) : static_cast<T>(0);
  params.is_neg_one = (divisor == T(-1));

  const UT abs_d = divisor < 0 ? static_cast<UT>(UT{0} - static_cast<UT>(divisor))
                               : static_cast<UT>(divisor);

  if (detail::IsPow2(abs_d)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(abs_d);
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  params.is_pow2 = false;
  params.pow2_shift = 0;

  const unsigned sh = 31u - detail::LeadingZeroCount32(static_cast<uint32_t>(abs_d - 1u));
  const uint64_t tmp = ((uint64_t{1} << 16) << sh) / abs_d + 1u;
  params.multiplier = static_cast<int32_t>(static_cast<T>(static_cast<uint32_t>(tmp)));
  params.shift = static_cast<int>(sh);
  return params;
}

template <typename T, HWY_IF_T_SIZE(T, 4), HWY_IF_SIGNED(T)>
HWY_INLINE constexpr DivisorParamsS<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  using UT = MakeUnsigned<T>;
  DivisorParamsS<T> params{};
  params.divisor = divisor;
  params.dsign = (divisor < 0) ? static_cast<T>(-1) : static_cast<T>(0);
  params.is_neg_one = (divisor == T(-1));

  const UT abs_d = divisor < 0 ? static_cast<UT>(UT{0} - static_cast<UT>(divisor))
                               : static_cast<UT>(divisor);

  if (detail::IsPow2(abs_d)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(abs_d);
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  params.is_pow2 = false;
  params.pow2_shift = 0;

  const unsigned sh = 31u - detail::LeadingZeroCount32(abs_d - 1u);
  const uint64_t m = (0x100000000ULL << sh) / abs_d + 1u;
  params.multiplier = static_cast<T>(m);
  params.shift = static_cast<int>(sh);
  return params;
}

template <typename T, HWY_IF_T_SIZE(T, 8), HWY_IF_SIGNED(T)>
HWY_INLINE
#if !HWY_INTDIV_SCALAR64 && (!HWY_COMPILER_MSVC || !HWY_ARCH_X86_64)
    constexpr
#endif
    DivisorParamsS<T>
    ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  using UT = MakeUnsigned<T>;
  DivisorParamsS<T> params{};
  params.divisor = divisor;
  params.dsign = (divisor < 0) ? static_cast<T>(-1) : static_cast<T>(0);
  params.is_neg_one = (divisor == T(-1));

  const UT abs_d = divisor < 0 ? static_cast<UT>(UT{0} - static_cast<UT>(divisor))
                               : static_cast<UT>(divisor);

  if (detail::IsPow2(abs_d)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros64(abs_d);
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  params.is_pow2 = false;
  params.pow2_shift = 0;

#if HWY_INTDIV_SCALAR64
  params.multiplier = 1;
  params.shift = 0;
#else
  const unsigned sh = 63u - detail::LeadingZeroCount64(abs_d - 1u);
  const uint64_t m = detail::DivideHighBy(1ULL << sh, abs_d) + 1u;
  params.multiplier = static_cast<T>(m);
  params.shift = static_cast<int>(sh);
#endif
  return params;
}

template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_UNSIGNED_D(D)>
HWY_INLINE V IntDiv(D d, V dividend, const DivisorParamsU<T>& params) {
  HWY_DASSERT(params.divisor != 0);

  if (params.is_pow2) {
    if (params.pow2_shift == 0) return dividend;  // divisor == 1
    return detail::ShiftRightUniform(d, dividend, params.pow2_shift);
  }

#if HWY_INTDIV_SCALAR64
  if constexpr (sizeof(T) == 8) {
    return detail::ScalarDivPerLane(d, dividend, params.divisor);
  }
#endif

  if constexpr (sizeof(T) <= 2) {
    if constexpr (D::kPrivateLanes < 2) {
      return detail::ScalarDivPerLane(d, dividend, params.divisor);
    } else {
      using TWide = detail::MultiplierType_T<T>;
      const Repartition<TWide, D> d_wide;

      const auto lo_wide = PromoteLowerTo(d_wide, dividend);
      const auto hi_wide = PromoteUpperTo(d_wide, dividend);

      const auto mul_wide = Set(d_wide, static_cast<TWide>(params.multiplier));

      const auto prod_lo = Mul(lo_wide, mul_wide);
      const auto prod_hi = Mul(hi_wide, mul_wide);

      constexpr int kShift = static_cast<int>(sizeof(T) * 8);

#if defined(HWY_HAVE_ORDEREDDEMOTE2TO)
      const V t1 =
          OrderedDemote2To(d, ShiftRight<kShift>(prod_lo), ShiftRight<kShift>(prod_hi));
#else
      const Half<D> d_half;
      const auto t1_lo = DemoteTo(d_half, ShiftRight<kShift>(prod_lo));
      const auto t1_hi = DemoteTo(d_half, ShiftRight<kShift>(prod_hi));
      const V t1 = Combine(d, t1_hi, t1_lo);
#endif

      const V diff = Sub(dividend, t1);
      const V shifted = ShiftRight<1>(diff);
      const V sum = Add(t1, shifted);
      return detail::ShiftRightUniform(d, sum, params.shift2);
    }
  } else {
    const V multiplier = Set(d, params.multiplier);
    const V t1 = MulHigh(dividend, multiplier);
    const V diff = Sub(dividend, t1);
    const V shifted = ShiftRight<1>(diff);
    const V sum = Add(t1, shifted);
    return detail::ShiftRightUniform(d, sum, params.shift2);
  }
}

template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_SIGNED_D(D)>
HWY_INLINE V IntDiv(D d, V dividend, const DivisorParamsS<T>& params) {
  HWY_DASSERT(params.divisor != 0);
  const V dsign_vec = Set(d, params.dsign);

  if (params.is_pow2) {
    if (params.pow2_shift == 0) {  // |divisor| == 1
      return Sub(Xor(dividend, dsign_vec), dsign_vec);
    }

    using UT = MakeUnsigned<T>;
    HWY_DASSERT(params.pow2_shift > 0 &&
                params.pow2_shift < static_cast<int>(8 * sizeof(T)));
    const T mask_val = static_cast<T>(
        (static_cast<UT>(1) << static_cast<unsigned>(params.pow2_shift)) - 1);
    const V mask = Set(d, mask_val);
    constexpr int kSignBit = int(sizeof(T) * 8) - 1;
    const V sign = ShiftRight<kSignBit>(dividend);
    const V bias = And(sign, mask);

    V q = detail::ShiftRightUniform(d, Add(dividend, bias), params.pow2_shift);
    return Sub(Xor(q, dsign_vec), dsign_vec);
  }

#if HWY_INTDIV_SCALAR64
  if constexpr (sizeof(T) == 8) {
    return detail::ScalarDivPerLane(d, dividend, params.divisor);
  }
#endif

  V q0;

  if constexpr (sizeof(T) <= 2) {
    if constexpr (D::kPrivateLanes < 2) {
      return detail::ScalarDivPerLane(d, dividend, params.divisor);
    } else {
      using TWide = detail::MultiplierType_T<T>;
      const Repartition<TWide, D> d_wide;

      const auto lo_wide = PromoteLowerTo(d_wide, dividend);
      const auto hi_wide = PromoteUpperTo(d_wide, dividend);

      const auto mul_wide = Set(d_wide, static_cast<TWide>(params.multiplier));

      const auto prod_lo = Mul(lo_wide, mul_wide);
      const auto prod_hi = Mul(hi_wide, mul_wide);

      constexpr int kShift = static_cast<int>(sizeof(T) * 8);

#if defined(HWY_HAVE_ORDEREDDEMOTE2TO)
      const auto high =
          OrderedDemote2To(d, ShiftRight<kShift>(prod_lo), ShiftRight<kShift>(prod_hi));
#else
      const Half<D> d_half;
      const auto high_lo = DemoteTo(d_half, ShiftRight<kShift>(prod_lo));
      const auto high_hi = DemoteTo(d_half, ShiftRight<kShift>(prod_hi));
      const auto high = Combine(d, high_hi, high_lo);
#endif

      q0 = Add(dividend, high);
    }
  } else {
    const V multiplier = Set(d, params.multiplier);
    const V mulh = MulHigh(dividend, multiplier);
    q0 = Add(dividend, mulh);
  }

  q0 = detail::ShiftRightUniform(d, q0, params.shift);

  constexpr int kSignBit2 = int(sizeof(T) * 8) - 1;
  const V sign_dividend = ShiftRight<kSignBit2>(dividend);
  q0 = Sub(q0, sign_dividend);

  return Sub(Xor(q0, dsign_vec), dsign_vec);
}

template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_SIGNED_D(D)>
HWY_INLINE V IntDivFloor(D d, V dividend, const DivisorParamsS<T>& params) {
  if (params.is_neg_one) {
    const V kMin = Set(d, std::numeric_limits<T>::min());
    const auto kMinMask = Eq(dividend, kMin);
    return IfThenElse(kMinMask, Zero(d), Neg(dividend));
  }
  V q = IntDiv(d, dividend, params);

  const V divisor = Set(d, params.divisor);
  const V prod = Mul(q, divisor);
  const auto neq = Ne(dividend, prod);
  const auto sdiff = Xor(Lt(dividend, Zero(d)), Lt(divisor, Zero(d)));
  const V one = Set(d, static_cast<T>(1));

  return Sub(q, IfThenElse(And(neq, sdiff), one, Zero(d)));
}

template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_UNSIGNED_D(D)>
HWY_INLINE V IntDivFloor(D d, V dividend, const DivisorParamsU<T>& params) {
  return IntDiv(d, dividend, params);
}

template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_UNSIGNED_D(D)>
HWY_INLINE V DivideByScalar(D d, V dividend, T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) HWY_ABORT("intdiv: division by zero");
  const auto params = ComputeDivisorParams(divisor);
  return IntDiv(d, dividend, params);
}

template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_SIGNED_D(D)>
HWY_INLINE V DivideByScalar(D d, V dividend, T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) HWY_ABORT("intdiv: division by zero");
  const auto params = ComputeDivisorParams(divisor);
  return IntDiv(d, dividend, params);
}

template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_SIGNED_D(D)>
HWY_INLINE V FloorDivideByScalar(D d, V dividend, T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) HWY_ABORT("intdiv: division by zero");
  const auto params = ComputeDivisorParams(divisor);
  return IntDivFloor(d, dividend, params);
}

template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_UNSIGNED_D(D)>
HWY_INLINE V FloorDivideByScalar(D d, V dividend, T divisor) {
  return DivideByScalar(d, dividend, divisor);
}

template <typename T, HWY_IF_UNSIGNED(T)>
HWY_INLINE void DivideArrayByScalar(T* HWY_RESTRICT array, size_t count, T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) HWY_ABORT("intdiv: division by zero");

#if HWY_INTDIV_SCALAR64
  if constexpr (sizeof(T) == 8) {
    for (size_t i = 0; i < count; ++i) {
      array[i] = static_cast<T>(array[i] / divisor);
    }
    return;
  }
#endif

  const ScalableTag<T> d;
  const size_t N = Lanes(d);
  const auto params = ComputeDivisorParams(divisor);

  size_t i = 0;
  for (; i + N <= count; i += N) {
    const auto vec = LoadU(d, array + i);
    const auto result = IntDiv(d, vec, params);
    StoreU(result, d, array + i);
  }
  if (i < count) {
    const size_t remaining = count - i;
    const auto vec = LoadN(d, array + i, remaining);
    const auto result = IntDiv(d, vec, params);
    StoreN(result, d, array + i, remaining);
  }
}

template <typename T, HWY_IF_SIGNED(T)>
HWY_INLINE void DivideArrayByScalar(T* HWY_RESTRICT array, size_t count, T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) HWY_ABORT("intdiv: division by zero");

#if HWY_INTDIV_SCALAR64
  if constexpr (sizeof(T) == 8) {
    for (size_t i = 0; i < count; ++i) {
      array[i] = static_cast<T>(array[i] / divisor);
    }
    return;
  }
#endif

  const ScalableTag<T> d;
  const size_t N = Lanes(d);
  const auto params = ComputeDivisorParams(divisor);

  size_t i = 0;
  for (; i + N <= count; i += N) {
    const auto vec = LoadU(d, array + i);
    const auto result = IntDiv(d, vec, params);
    StoreU(result, d, array + i);
  }
  if (i < count) {
    const size_t remaining = count - i;
    const auto vec = LoadN(d, array + i, remaining);
    const auto result = IntDiv(d, vec, params);
    StoreN(result, d, array + i, remaining);
  }
}

template <typename T, HWY_IF_SIGNED(T)>
HWY_INLINE void FloorDivideArrayByScalar(T* HWY_RESTRICT array, size_t count, T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) HWY_ABORT("intdiv: division by zero");

#if HWY_INTDIV_SCALAR64
  if constexpr (sizeof(T) == 8) {
    for (size_t i = 0; i < count; ++i) {
      const T a = array[i];
      T q = static_cast<T>(a / divisor);
      const T r = static_cast<T>(a % divisor);
      if (r != 0 && ((a < 0) != (divisor < 0))) --q;
      array[i] = q;
    }
    return;
  }
#endif

  const ScalableTag<T> d;
  const size_t N = Lanes(d);
  const auto params = ComputeDivisorParams(divisor);

  size_t i = 0;
  for (; i + N <= count; i += N) {
    const auto vec = LoadU(d, array + i);
    const auto result = IntDivFloor(d, vec, params);
    StoreU(result, d, array + i);
  }
  if (i < count) {
    const size_t remaining = count - i;
    const auto vec = LoadN(d, array + i, remaining);
    const auto result = IntDivFloor(d, vec, params);
    StoreN(result, d, array + i, remaining);
  }
}

template <typename T, HWY_IF_UNSIGNED(T)>
HWY_INLINE void FloorDivideArrayByScalar(T* HWY_RESTRICT array, size_t count, T divisor) {
  DivideArrayByScalar(array, count, divisor);  // Same for unsigned
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // per-target include guard
