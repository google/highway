// Copyright 2024 Google LLC
// Copyright 2026 Fujitsu Limited
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: BSD-3-Clause
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
 * Almost all architectures (except Power10) don't support integer vector division,
 * and the cost of scalar division on architectures like x86 is too high - it can
 * take 30 to 40 cycles on modern chips and up to 100 on older ones.
 *
 * Therefore we use division by multiplying with a precomputed reciprocal. The
 * method used in this implementation is based on T. Granlund and P. L. Montgomery,
 * "Division by invariant integers using multiplication" (see [Figure 4.1, 5.1])
 * https://gmplib.org/~tege/divcnst-pldi94.pdf
 * https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.2556
 *
 * It shows good performance gains for all architectures, especially on x86.
 * However, computing divisor parameters is expensive, so this implementation
 * should only be used when the divisor is a scalar and used multiple times.
 *
 * We split the work into two steps:
 *   1) Precompute parameters from the scalar divisor (multiplier + shifts).
 *  DivisorParams{U,S}<T> ComputeDivisorParams(T divisor);
 *     Computes the divisor parameters (multiplier + shifts + sign of divisor).
 *
 *   2) Use those parameters to replace per-lane division with MulHigh + shifts.
 *   Vec<D> IntDiv(D d, Vec<D> dividend, const DivisorParams{U,S}<T>& params);
 *     Performs the actual division using the precomputed parameters.
 *
 ** NOTES:
 *  - For 64-bit division on platforms without 128-bit division support (including
 *    Aarch64, IBM/Power, and clang-cl on Windows), we fall back to scalar division
 *    since emulating multiply-high is expensive and both architectures have very
 *    fast hardware dividers.
 *  - Power-of-two divisors are optimized to simple shifts.
 *  - Edge cases like INT_MIN / -1 are handled specially.
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
 ** Figure 5.1: Signed division by run-time invariant divisor, rounded towards zero
 ************************************************************************************
 * Initialization (given constant sword d with d != 0):
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

#include "hwy/highway.h"

#ifndef HWY_INTDIV_SCALAR64
  #if !HWY_HAVE_DIV128 || HWY_TARGET_IS_NEON || HWY_TARGET == HWY_PPC8 || HWY_TARGET == HWY_VSX
    #define HWY_INTDIV_SCALAR64 1
  #else
    #define HWY_INTDIV_SCALAR64 0
  #endif
#endif

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

/** 
 * Type used for the precomputed multiplier constant.
 * For 8/16-bit lanes: next wider type (widening multiply path).
 * For 32/64-bit lanes: same type (MulHigh path).
 */
namespace detail {
template <typename T>
struct MultiplierType {
  using type = T;
};

template <>
struct MultiplierType<uint8_t> {
  using type = uint16_t;
};

template <>
struct MultiplierType<int8_t> {
  using type = int16_t;
};

template <>
struct MultiplierType<uint16_t> {
  using type = uint32_t;
};

template <>
struct MultiplierType<int16_t> {
  using type = int32_t;
};

template <>
struct MultiplierType<uint32_t> {
  using type = uint32_t;
};

template <>
struct MultiplierType<int32_t> {
  using type = int32_t;
};

template <>
struct MultiplierType<uint64_t> {
  using type = uint64_t;
};

template <>
struct MultiplierType<int64_t> {
  using type = int64_t;
};

template <typename T>
using MultiplierType_T = typename MultiplierType<T>::type;

template <typename T>
HWY_INLINE bool IsPow2(T x) {
  return x > 0 && (x & (x - 1)) == 0;
}

HWY_INLINE int CountTrailingZeros32(uint32_t x) {
  return x == 0 ? 32 : static_cast<int>(Num0BitsBelowLS1Bit_Nonzero32(x));
}

HWY_INLINE int CountTrailingZeros64(uint64_t x) {
  return x == 0 ? 64 : static_cast<int>(Num0BitsBelowLS1Bit_Nonzero64(x));
}

HWY_INLINE unsigned LeadingZeroCount32(uint32_t x) {
  return x == 0 ? 32u : static_cast<unsigned>(Num0BitsAboveMS1Bit_Nonzero32(x));
}

HWY_INLINE unsigned LeadingZeroCount64(uint64_t x) {
  return x == 0 ? 64u : static_cast<unsigned>(Num0BitsAboveMS1Bit_Nonzero64(x));
}

/**
 * Divides a 128-bit unsigned integer (high:0) by a 64-bit divisor.
 * Computes: (high << 64) / divisor
 *
 * Only defined when 128-bit division is available (HWY_HAVE_DIV128).
 * When HWY_INTDIV_SCALAR64 is set, this function is not needed because
 * 64-bit division uses the scalar per-lane fallback.
 */
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

/** 
 * Shifts all lanes of 'v' right by runtime value sh
 * Dispatches to ShiftRightSame on targets that support it for all lane sizes; 
 * otherwise falls back to decomposed compile-time shifts.
 */
template <class D, class V = Vec<D>>
HWY_INLINE V ShiftRightUniform(D d, V v, int sh) {
  using T = TFromD<D>;
  const int kBits = int(sizeof(T) * 8);
  if (sh <= 0) return v;
  if (sh >= kBits) sh = kBits - 1;
  (void)d;

#if HWY_TARGET_IS_NEON || HWY_TARGET == HWY_AVX2 || HWY_TARGET <= HWY_AVX3
  return ShiftRightSame(v, sh);
#endif
  if constexpr (kBits > 32) { if (sh & 32) v = ShiftRight<32>(v); }
  if constexpr (kBits > 16) { if (sh & 16) v = ShiftRight<16>(v); }
  if constexpr (kBits > 8) { if (sh & 8)  v = ShiftRight<8>(v); }
  if (sh & 4)  v = ShiftRight<4>(v);
  if (sh & 2)  v = ShiftRight<2>(v);
  if (sh & 1)  v = ShiftRight<1>(v);
  return v;
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
  int shift1;
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
  bool is_pow2;
  int pow2_shift;
};


template <typename T, HWY_IF_T_SIZE(T, 1), HWY_IF_UNSIGNED(T)>
HWY_INLINE DivisorParamsU<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  DivisorParamsU<T> params;
  params.divisor = divisor;
  
  if (detail::IsPow2(divisor)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(divisor);
    params.multiplier = 1;
    params.shift1 = params.shift2 = 0;
    return params;
  }
  
  params.is_pow2 = false;
  
  if (divisor == 1) {
    params.multiplier = 1;
    params.shift1 = params.shift2 = 0;
    return params;
  }
  
  unsigned l = 32 - detail::LeadingZeroCount32(divisor - 1);
  uint16_t two_l = static_cast<uint16_t>(1U << l);
  uint32_t m = ((static_cast<uint32_t>(two_l - divisor) << 8) / divisor) + 1;
  params.multiplier = static_cast<uint16_t>(m);
  params.shift1 = 1;
  params.shift2 = static_cast<int>(l) - 1;
  return params;
}

template <typename T, HWY_IF_T_SIZE(T, 2), HWY_IF_UNSIGNED(T)>
HWY_INLINE DivisorParamsU<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  DivisorParamsU<T> params;
  params.divisor = divisor;
  
  if (detail::IsPow2(divisor)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(divisor);
    params.multiplier = 1;
    params.shift1 = params.shift2 = 0;
    return params;
  }
  
  params.is_pow2 = false;
  
  if (divisor == 1) {
    params.multiplier = 1;
    params.shift1 = params.shift2 = 0;
    return params;
  }
  
  unsigned l = 32 - detail::LeadingZeroCount32(divisor - 1);
  uint32_t two_l = 1U << l;
  const uint64_t tmp = ((static_cast<uint64_t>(two_l - divisor) << 16) / divisor) + 1;
  const uint32_t m = static_cast<uint32_t>(tmp);
  params.multiplier = m;
  params.shift1 = 1;
  params.shift2 = static_cast<int>(l) - 1;
  
  return params;
}

template <typename T, HWY_IF_T_SIZE(T, 4), HWY_IF_UNSIGNED(T)>
HWY_INLINE DivisorParamsU<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  DivisorParamsU<T> params;
  params.divisor = divisor;
  
  if (detail::IsPow2(divisor)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(divisor);
    params.multiplier = 1;
    params.shift1 = params.shift2 = 0;
    return params;
  }
  
  params.is_pow2 = false;
  
  if (divisor == 1) {
    params.multiplier = 1;
    params.shift1 = params.shift2 = 0;
    return params;
  }
  
  unsigned l = 32 - detail::LeadingZeroCount32(divisor - 1);
  uint64_t two_l = 1ULL << l;
  uint64_t m = ((two_l - divisor) << 32) / divisor + 1;
  params.multiplier = static_cast<T>(m);
  params.shift1 = 1;
  params.shift2 = static_cast<int>(l) - 1;
  
  return params;
}

template <typename T, HWY_IF_T_SIZE(T, 8), HWY_IF_UNSIGNED(T)>
HWY_INLINE DivisorParamsU<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  DivisorParamsU<T> params;
  params.divisor = divisor;
  
  if (detail::IsPow2(divisor)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros64(divisor);
    params.multiplier = 1;
    params.shift1 = params.shift2 = 0;
    return params;
  }
  
  params.is_pow2 = false;
  
  if (divisor == 1) {
    params.multiplier = 1;
    params.shift1 = params.shift2 = 0;
    return params;
  }
  
#if HWY_INTDIV_SCALAR64
  params.multiplier = 1;
  params.shift1 = 0;
  params.shift2 = 0;
#else
  unsigned l = 64 - detail::LeadingZeroCount64(divisor - 1);
  uint64_t two_l_minus_d = (l < 64) ? ((1ULL << l) - divisor) : (0 - divisor);
  uint64_t m = detail::DivideHighBy(two_l_minus_d, divisor) + 1;
  
  params.multiplier = m;
  params.shift1 = 1;
  params.shift2 = static_cast<int>(l) - 1;
#endif
  
  return params;
}

template <typename T, HWY_IF_T_SIZE(T, 1), HWY_IF_SIGNED(T)>
HWY_INLINE DivisorParamsS<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  using UT = MakeUnsigned<T>;
  DivisorParamsS<T> params;
  params.divisor = divisor;
  
  UT abs_d = divisor < 0 ? static_cast<UT>(0) - static_cast<UT>(divisor) 
                       : static_cast<UT>(divisor);
  
  if (detail::IsPow2(abs_d)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(abs_d);
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  
  params.is_pow2 = false;
  
  if (abs_d == 1) {
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  
  if (static_cast<UT>(divisor) == 0x80U) {
    params.multiplier = static_cast<int16_t>(0x81);
    params.shift = 6;
    return params;
  }
  
  unsigned sh = 31 - detail::LeadingZeroCount32(static_cast<uint32_t>(abs_d - 1));
  uint32_t m = (256U << sh) / abs_d + 1;

  params.multiplier = static_cast<int16_t>(static_cast<T>(m));
  params.shift = static_cast<int>(sh);
  
  return params;
}

template <typename T, HWY_IF_T_SIZE(T, 2), HWY_IF_SIGNED(T)>
HWY_INLINE DivisorParamsS<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  using UT = MakeUnsigned<T>;
  DivisorParamsS<T> params;
  params.divisor = divisor;
  
  UT abs_d = divisor < 0 ? static_cast<UT>(0) - static_cast<UT>(divisor) 
                       : static_cast<UT>(divisor);
  
  if (detail::IsPow2(abs_d)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(abs_d);
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  
  params.is_pow2 = false;
  
  if (abs_d == 1) {
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  
  if (static_cast<UT>(divisor) == 0x8000U) {
    params.multiplier = static_cast<int32_t>(0x8001);
    params.shift = 14;
    return params;
  }
  
  unsigned sh = 31 - detail::LeadingZeroCount32(static_cast<uint32_t>(abs_d - 1));
  const uint64_t tmp = ((uint64_t{1} << 16) << sh) / abs_d + 1;
  const uint32_t m = static_cast<uint32_t>(tmp);
  
  params.multiplier = static_cast<int32_t>(static_cast<T>(m));
  params.shift = static_cast<int>(sh);
  
  return params;
}

template <typename T, HWY_IF_T_SIZE(T, 4), HWY_IF_SIGNED(T)>
HWY_INLINE DivisorParamsS<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  using UT = MakeUnsigned<T>;
  DivisorParamsS<T> params;
  params.divisor = divisor;
  
  UT abs_d = divisor < 0 ? static_cast<UT>(0) - static_cast<UT>(divisor) 
                       : static_cast<UT>(divisor);
  
  if (detail::IsPow2(abs_d)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(abs_d);
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  
  params.is_pow2 = false;
  
  if (abs_d == 1) {
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  
  if (static_cast<UT>(divisor) == 0x80000000U) {
    params.multiplier = static_cast<T>(0x80000001);
    params.shift = 30;
    return params;
  }
  
  unsigned sh = 31 - detail::LeadingZeroCount32(abs_d - 1);
  uint64_t m = (0x100000000ULL << sh) / abs_d + 1;
  params.multiplier = static_cast<T>(m);
  params.shift = static_cast<int>(sh);
  
  return params;
}

template <typename T, HWY_IF_T_SIZE(T, 8), HWY_IF_SIGNED(T)>
HWY_INLINE DivisorParamsS<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  using UT = MakeUnsigned<T>;
  DivisorParamsS<T> params;
  params.divisor = divisor;
  
  UT abs_d = divisor < 0 ? static_cast<UT>(0) - static_cast<UT>(divisor) 
                       : static_cast<UT>(divisor);
  
  if (detail::IsPow2(abs_d)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros64(abs_d);
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  
  params.is_pow2 = false;
  
  if (abs_d == 1) {
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  
  if (static_cast<UT>(divisor) == 0x8000000000000000ULL) {
    params.multiplier = static_cast<T>(0x8000000000000001LL);
    params.shift = 62;
    return params;
  }

#if HWY_INTDIV_SCALAR64
  params.multiplier = 1;
  params.shift = 0;
#else
  unsigned sh = 63 - detail::LeadingZeroCount64(abs_d - 1);
  uint64_t m = detail::DivideHighBy(1ULL << sh, abs_d) + 1;
  params.multiplier = static_cast<T>(m);
  params.shift = static_cast<int>(sh);
#endif
  
  return params;
}

template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_UNSIGNED_D(D)>
HWY_INLINE V IntDiv(D d, V dividend, const DivisorParamsU<T>& params) {
  HWY_DASSERT(params.divisor != 0);
  if (params.is_pow2) {
    return detail::ShiftRightUniform(d, dividend, params.pow2_shift);
  }
  
  #if HWY_INTDIV_SCALAR64
    if constexpr (sizeof(T) == 8) {
      return detail::ScalarDivPerLane(d, dividend, params.divisor);
    }
  #endif

  if (params.shift1 == 0 && params.shift2 == 0 && params.multiplier == 1) {
    return dividend;
  }

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
        const V t1 = OrderedDemote2To(d, ShiftRight<kShift>(prod_lo), ShiftRight<kShift>(prod_hi));
      #else
        const Half<D> d_half;
        const auto t1_lo = DemoteTo(d_half, ShiftRight<kShift>(prod_lo));
        const auto t1_hi = DemoteTo(d_half, ShiftRight<kShift>(prod_hi));
        const V t1 = Combine(d, t1_hi, t1_lo);
      #endif
      
      const V diff = Sub(dividend, t1);
      const V shifted = detail::ShiftRightUniform(d, diff, params.shift1);
      const V sum = Add(t1, shifted);
      return detail::ShiftRightUniform(d, sum, params.shift2);
    }
} else {
    const V multiplier = Set(d, params.multiplier);
    const V t1 = MulHigh(dividend, multiplier);
    const V diff = Sub(dividend, t1);
    const V shifted = detail::ShiftRightUniform(d, diff, params.shift1);
    const V sum = Add(t1, shifted);
    return detail::ShiftRightUniform(d, sum, params.shift2);
  }
}

template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_SIGNED_D(D)>
HWY_INLINE V IntDiv(D d, V dividend, const DivisorParamsS<T>& params) {
  const bool neg_divisor = params.divisor < 0;
  HWY_DASSERT(params.divisor != 0);
  
  if (params.is_pow2) {
    
    using UT = MakeUnsigned<T>;
    
    HWY_DASSERT(params.pow2_shift >= 0 && params.pow2_shift < static_cast<int>(8 * sizeof(T)));
    const T mask_val = static_cast<T>((static_cast<UT>(1) << static_cast<unsigned>(params.pow2_shift)) - 1);

    const V mask = Set(d, mask_val);
    constexpr int kSignBit = int(sizeof(T) * 8) - 1;
    const V sign = ShiftRight<kSignBit>(dividend);
    
    const V bias = And(sign, mask);
    
    V q = detail::ShiftRightUniform(d, Add(dividend, bias), params.pow2_shift);
    
    if (neg_divisor) {
      q = Neg(q);
    }
    
    return q;
  }

  #if HWY_INTDIV_SCALAR64
    if constexpr (sizeof(T) == 8) {
      return detail::ScalarDivPerLane(d, dividend, params.divisor);
    }
  #endif

  if (params.shift == 0 && params.multiplier == 1) {
    if (neg_divisor) {
      return Neg(dividend);
    }
    return dividend;
  }

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
        const auto high = OrderedDemote2To(d, ShiftRight<kShift>(prod_lo), ShiftRight<kShift>(prod_hi));
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
  
  if (neg_divisor) {
    const V neg_one = Set(d, static_cast<T>(-1));
    q0 = Xor(q0, neg_one);
    q0 = Sub(q0, neg_one);
  }
  
  return q0;
}

template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_SIGNED_D(D)>
HWY_INLINE V IntDivFloor(D d, V dividend, const DivisorParamsS<T>& params) {
  if (params.divisor == T(-1)) {
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
  
  if (detail::IsPow2(divisor)) {
    const int ctz = (sizeof(T) == 8)
        ? detail::CountTrailingZeros64(static_cast<uint64_t>(divisor))
        : detail::CountTrailingZeros32(static_cast<uint32_t>(divisor));
    return detail::ShiftRightUniform(d, dividend, ctz);
  }
  
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
