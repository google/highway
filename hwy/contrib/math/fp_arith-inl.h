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

// Building blocks for floating-point arithmetic: error-free transforms and
// double-double helpers. The exact-arithmetic primitives are adapted from
// gemma.cpp

// Include guard (still compiled once per target)
#if defined(HIGHWAY_HWY_CONTRIB_MATH_FP_ARITH_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_MATH_FP_ARITH_INL_H_
#undef HIGHWAY_HWY_CONTRIB_MATH_FP_ARITH_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_MATH_FP_ARITH_INL_H_
#endif

#include <stddef.h>

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

//------------------------------------------------------------------------------
// Exact multiplication

// Scalar variant of TwoSums; see below. Knuth98/Moller65, 6 ops.
template <typename T, HWY_IF_FLOAT3264(T)>
static inline T TwoSum(T a, T b, T& err) {
  const T sum = a + b;
  const T a2 = sum - b;
  const T b2 = sum - a2;
  const T err_a = a - a2;
  const T err_b = b - b2;
  err = err_a + err_b;
  return sum;
}

// Returns `prod` and `err` such that `prod + err` is exactly equal to `a * b`,
// despite floating-point rounding, assuming that `err` is not subnormal, i.e.,
// the sum of exponents >= min exponent + mantissa bits. 2..17 ops. Useful for
// compensated dot products and polynomial evaluation.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = Vec<DF>>
static HWY_INLINE VF TwoProducts(DF df, VF a, VF b, VF& err) {
  const VF prod = Mul(a, b);
  if constexpr (HWY_NATIVE_FMA) {
    err = MulSub(a, b, prod);
  } else {
    // Non-FMA fallback, splits into half-mantissa hi/lo parts so the partial
    // products are exact
    using TF = TFromD<DF>;
    const RebindToUnsigned<DF> du;
    constexpr int kMant = hwy::MantissaBits<TF>();
    // Bits to drop so each half is narrow enough to multiply exactly
    constexpr int kHalf = (kMant | 1) - (kMant >> 1);
    // Split a and b into hi (high half) + lo (low half)
    const VF a_hi = BitCast(
        df, ShiftLeft<kHalf>(RoundingShiftRight<kHalf>(BitCast(du, a))));
    const VF b_hi = BitCast(
        df, ShiftLeft<kHalf>(RoundingShiftRight<kHalf>(BitCast(du, b))));
    const VF a_lo = Sub(a, a_hi);
    const VF b_lo = Sub(b, b_hi);
    // Sum the exact partial products into the low part
    err = MulAdd(
        a_lo, b_lo,
        Add(MulAdd(a_hi, b_lo, Mul(a_lo, b_hi)), MulSub(a_hi, b_hi, prod)));
  }
  return prod;
}

//------------------------------------------------------------------------------
// Exact addition

// Returns `sum` and `err` such that `sum + err` is exactly equal to `a + b`,
// despite floating-point rounding. `sum` is already the best estimate for the
// addition, so do not directly add `err` to it. `UpdateCascadedSums` instead
// accumulates multiple `err`, which are then later added to the total `sum`.
//
// Knuth98/Moller65. Unlike FastTwoSums, this does not require any relative
// ordering of the exponents of a and b. 6 ops.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = Vec<DF>>
static HWY_INLINE VF TwoSums(DF /*df*/, VF a, VF b, VF& err) {
  const VF sum = Add(a, b);
  const VF a2 = Sub(sum, b);
  const VF b2 = Sub(sum, a2);
  const VF err_a = Sub(a, a2);
  const VF err_b = Sub(b, b2);
  err = Add(err_a, err_b);
  return sum;
}

// As above, but only exact if the exponent of `a` >= that of `b`, which is the
// case if |a| >= |b|. Dekker71, also used in Kahan65 compensated sum. 3 ops.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = Vec<DF>>
static HWY_INLINE VF FastTwoSums(DF /*df*/, VF a, VF b, VF& err) {
  const VF sum = Add(a, b);
  const VF b2 = Sub(sum, a);
  err = Sub(b, b2);
  return sum;
}

//------------------------------------------------------------------------------
// Cascaded summation (twice working precision)

// Accumulates numbers with about twice the precision of T using 7 * n FLOPS.
// Rump/Ogita/Oishi08, Algorithm 6.11 in Handbook of Floating-Point Arithmetic.
//
// Because vectors generally cannot be wrapped in a class, we use functions.
// `sum` and `sum_err` must be initially zero. Each lane is an independent sum.
// To reduce them into a single result, use `ReduceCascadedSums`.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = Vec<DF>>
void UpdateCascadedSums(DF df, VF v, VF& sum, VF& sum_err) {
  VF err;
  sum = TwoSums(df, sum, v, err);
  sum_err = Add(sum_err, err);
}

// Combines two cascaded sum vectors, typically from unrolling/parallelization.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = Vec<DF>>
void AssimilateCascadedSums(DF df, const VF& other_sum, const VF& other_sum_err,
                            VF& sum, VF& sum_err) {
  sum_err = Add(sum_err, other_sum_err);
  UpdateCascadedSums(df, other_sum, sum, sum_err);
}

// Reduces cascaded sums, to a single value. Slow, call outside of loops.
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = Vec<DF>>
TFromD<DF> ReduceCascadedSums(DF df, const VF sum, VF sum_err) {
  const size_t N = Lanes(df);
  using TF = TFromD<DF>;
  // For non-scalable wide vectors, reduce loop iterations below by recursing
  // once or twice for halves of 256-bit or 512-bit vectors.
  if constexpr (HWY_HAVE_CONSTEXPR_LANES) {
    if constexpr (Lanes(df) > 16 / sizeof(TF)) {
      const Half<DF> dfh;
      using VFH = Vec<decltype(dfh)>;

      VFH sum0 = LowerHalf(dfh, sum);
      VFH sum_err0 = LowerHalf(dfh, sum_err);
      const VFH sum1 = UpperHalf(dfh, sum);
      const VFH sum_err1 = UpperHalf(dfh, sum_err);
      AssimilateCascadedSums(dfh, sum1, sum_err1, sum0, sum_err0);
      return ReduceCascadedSums(dfh, sum0, sum_err0);
    }
  }

  TF total = TF{0.0};
  TF total_err = TF{0.0};
  for (size_t i = 0; i < N; ++i) {
    TF err;
    total_err += ExtractLane(sum_err, i);
    total = TwoSum(total, ExtractLane(sum, i), err);
    total_err += err;
  }
  return total + total_err;
}

//------------------------------------------------------------------------------
// Double-double helpers
// One value stored in two doubles (hi + lo) for ~2x precision, lo is hi's
// rounding error.

// Returns hi = (a_hi + a_lo) + (b_hi + b_lo), r_lo = low part
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = Vec<DF>>
HWY_INLINE HWY_MAYBE_UNUSED VF DDAdd(DF df, VF a_hi, VF a_lo, VF b_hi, VF b_lo,
                                     VF& r_lo) {
  VF e;
  const VF s = TwoSums(df, a_hi, b_hi, e);
  e = Add(e, Add(a_lo, b_lo));
  return FastTwoSums(df, s, e, r_lo);
}

// Returns hi = (a_hi + a_lo) * b, r_lo = low part
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = Vec<DF>>
HWY_INLINE HWY_MAYBE_UNUSED VF DDMul1(DF df, VF a_hi, VF a_lo, VF b, VF& r_lo) {
  VF p_lo;
  const VF p_hi = TwoProducts(df, a_hi, b, p_lo);
  p_lo = MulAdd(a_lo, b, p_lo);
  return FastTwoSums(df, p_hi, p_lo, r_lo);
}

// Returns hi = (a_hi + a_lo) * (b_hi + b_lo), r_lo = low part
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = Vec<DF>>
HWY_INLINE HWY_MAYBE_UNUSED VF DDMul2(DF df, VF a_hi, VF a_lo, VF b_hi, VF b_lo,
                                      VF& r_lo) {
  VF p_lo;
  const VF p_hi = TwoProducts(df, a_hi, b_hi, p_lo);
  p_lo = MulAdd(a_hi, b_lo, MulAdd(a_lo, b_hi, p_lo));
  return FastTwoSums(df, p_hi, p_lo, r_lo);
}

// Returns hi = (a_hi + a_lo) / (b_hi + b_lo), r_lo = low part
template <class DF, HWY_IF_FLOAT3264_D(DF), class VF = Vec<DF>>
HWY_INLINE HWY_MAYBE_UNUSED VF DDDiv(DF df, VF a_hi, VF a_lo, VF b_hi, VF b_lo,
                                     VF& r_lo) {
  const VF q1 = Div(a_hi, b_hi);
  VF qb_lo;
  const VF qb_hi = DDMul1(df, b_hi, b_lo, q1, qb_lo);
  VF r_hi_lo;
  const VF r_hi = DDAdd(df, a_hi, a_lo, Neg(qb_hi), Neg(qb_lo), r_hi_lo);
  const VF q2 = Div(r_hi, b_hi);
  return FastTwoSums(df, q1, q2, r_lo);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
