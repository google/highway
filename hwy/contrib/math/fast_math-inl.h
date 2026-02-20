// Copyright 2024 Google LLC
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

// Include guard (still compiled once per target)
#if defined(HIGHWAY_HWY_CONTRIB_MATH_FAST_MATH_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_MATH_FAST_MATH_INL_H_
#undef HIGHWAY_HWY_CONTRIB_MATH_FAST_MATH_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_MATH_FAST_MATH_INL_H_
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

namespace impl {

// Port of reduce_angle_tan_SIMD
template <class D, class V>
HWY_INLINE void ReduceAngleTan(D d, V ang, V& x_red, V& sign) {
  using T = TFromD<D>;
  const auto pi = Set(d, static_cast<T>(3.14159265358979323846));
  const auto zero = Set(d, static_cast<T>(0.0));
  const auto one = Set(d, static_cast<T>(1.0));
  const auto minus_one = Set(d, static_cast<T>(-1.0));

  const auto inv_pi = Set(d, static_cast<T>(0.31830988618379067153777));

  // Modulo pi
  auto quotient = Mul(ang, inv_pi);
  quotient = Round(quotient);
  auto ang_mod = NegMulAdd(quotient, pi, ang);

  // Determine sign
  auto mask_neg = Lt(ang_mod, zero);
  sign = IfThenElse(mask_neg, minus_one, one);

  // Absolute value
  x_red = Abs(ang_mod);
}

}  // namespace impl

/**
 * Fast approximation of tan(x).
 *
 * Valid Lane Types: float32, float64
 * Max Relative Error: < 0.35% for angles equivalent to falling between [-89.99,
 * +89.99] degrees (float32) and
 *                     [-89.9999999, +89.9999999] degrees (float64).
 * Valid Range: float32 : [-20, +20]rads
 *              float64 : [-39000, +39000]rads
 *
 * Note: Inputs extremely close to asymptotes may result in
 * a sign flip due to precision limits.
 *
 * @return tangent of 'x'
 */
template <class D, class V>
HWY_INLINE V FastTan(D d, V x) {
  using T = TFromD<D>;

  // Reduction
  V x_red, sign;
  impl::ReduceAngleTan(d, x, x_red, sign);

  constexpr size_t kLanes = HWY_MAX_LANES_D(D);
  V a, b, c, d_val;

  if constexpr ((kLanes >= 4 && !HWY_HAVE_SCALABLE) ||
                (HWY_HAVE_SCALABLE && sizeof(T) == 4)) {
    // --- Table Lookup ---
    const auto scale = Set(d, static_cast<T>(3.8197186342));
    auto idx_float = Floor(Mul(x_red, scale));

    // Convert to Integer Vector (Signed)
    auto idx_int = ConvertTo(RebindToSigned<D>(), idx_float);

    HWY_ALIGN static constexpr T arr_a[] = {
        630.25357464271012, 572.95779513082321, 343.77467707849392,
        572.95779513082321, 229.18311805232929, 57.295779513082323,
        57.295779513082323, 57.295779513082323};

    HWY_ALIGN static constexpr T arr_b[] = {
        0.0000000000000000, 10.0000000000000000, 46.0000000000000000,
        217.00000000000000, 297.00000000000000,  542.00000000000000,
        542.00000000000000, 542.00000000000000};

    HWY_ALIGN static constexpr T arr_c[] = {
        -57.295779513082323, -229.18311805232929, -286.47889756541161,
        -744.84513367007019, -572.95779513082321, -630.25357464271012,
        -630.25357464271012, -630.25357464271012};

    HWY_ALIGN static constexpr T arr_d[] = {
        632.00000000000000, 657.00000000000000, 541.00000000000000,
        1252.0000000000000, 910.00000000000000, 990.00000000000000,
        990.00000000000000, 990.00000000000000};

    if constexpr (kLanes >= 8 && !HWY_HAVE_SCALABLE) {
      // Cast to "Indices" Type
      auto idx = IndicesFromVec(d, idx_int);
      a = TableLookupLanes(Load(d, arr_a), idx);
      b = TableLookupLanes(Load(d, arr_b), idx);
      c = TableLookupLanes(Load(d, arr_c), idx);
      d_val = TableLookupLanes(Load(d, arr_d), idx);
    } else {
      auto idx = IndicesFromVec(d, idx_int);
      FixedTag<T, 4> d4;
      a = TwoTablesLookupLanes(d, Load(d4, arr_a), Load(d4, arr_a + 4), idx);
      b = TwoTablesLookupLanes(d, Load(d4, arr_b), Load(d4, arr_b + 4), idx);
      c = TwoTablesLookupLanes(d, Load(d4, arr_c), Load(d4, arr_c + 4), idx);
      d_val =
          TwoTablesLookupLanes(d, Load(d4, arr_d), Load(d4, arr_d + 4), idx);
    }
  } else {
    // --- FALLBACK PATH: Blend Chain ---
    a = Set(d, static_cast<T>(57.295779513082323));
    b = Set(d, static_cast<T>(542.00000000000000));
    c = Set(d, static_cast<T>(-630.25357464271012));
    d_val = Set(d, static_cast<T>(990.00000000000000));

    auto mask = Lt(x_red, Set(d, static_cast<T>(1.305)));
    a = IfThenElse(mask, Set(d, static_cast<T>(229.18311805232929)), a);
    b = IfThenElse(mask, Set(d, static_cast<T>(297.00000000000000)), b);
    c = IfThenElse(mask, Set(d, static_cast<T>(-572.95779513082321)), c);
    d_val = IfThenElse(mask, Set(d, static_cast<T>(910.00000000000000)), d_val);

    mask = Lt(x_red, Set(d, static_cast<T>(1.044)));
    a = IfThenElse(mask, Set(d, static_cast<T>(572.95779513082321)), a);
    b = IfThenElse(mask, Set(d, static_cast<T>(217.00000000000000)), b);
    c = IfThenElse(mask, Set(d, static_cast<T>(-744.84513367007019)), c);
    d_val = IfThenElse(mask, Set(d, static_cast<T>(1252.0000000000000)), d_val);

    mask = Lt(x_red, Set(d, static_cast<T>(0.783)));
    a = IfThenElse(mask, Set(d, static_cast<T>(343.77467707849392)), a);
    b = IfThenElse(mask, Set(d, static_cast<T>(46.0000000000000000)), b);
    c = IfThenElse(mask, Set(d, static_cast<T>(-286.47889756541161)), c);
    d_val = IfThenElse(mask, Set(d, static_cast<T>(541.00000000000000)), d_val);

    mask = Lt(x_red, Set(d, static_cast<T>(0.522)));
    a = IfThenElse(mask, Set(d, static_cast<T>(572.95779513082321)), a);
    b = IfThenElse(mask, Set(d, static_cast<T>(10.0000000000000000)), b);
    c = IfThenElse(mask, Set(d, static_cast<T>(-229.18311805232929)), c);
    d_val = IfThenElse(mask, Set(d, static_cast<T>(657.00000000000000)), d_val);

    mask = Lt(x_red, Set(d, static_cast<T>(0.261)));
    a = IfThenElse(mask, Set(d, static_cast<T>(630.25357464271012)), a);
    b = IfThenZeroElse(mask, b);
    c = IfThenElse(mask, Set(d, static_cast<T>(-57.295779513082323)), c);
    d_val = IfThenElse(mask, Set(d, static_cast<T>(632.00000000000000)), d_val);
  }

  // Math: y=(ax + b)/(cx + d)
  auto num = MulAdd(a, x_red, b);
  auto den = MulAdd(c, x_red, d_val);

  // Guard against denominator underflow/sign-flip near singularities
  T epsilon_val;
  if constexpr (sizeof(T) == 8) {
    epsilon_val = 1e-15;
  } else {
    epsilon_val = 1e-6;
  }
  const auto kMinDenom = Set(d, epsilon_val);
  // We use Abs() because on the reduced interval [0, pi/2], the tangent
  // magnitude must be positive. If the polynomial approximation calculates a
  // negative denominator (overshoot), it is an error, and we force it to be
  // positive.
  den = Max(Abs(den), kMinDenom);

  auto result = Div(num, den);

  // Apply Sign
  return CopySign(result, sign);
}

/**
 * Fast approximation of atan(x).
 *
 * Valid Lane Types: float32, float64
 * Max Relative Error: < 0.35%
 * Valid Range: float32: [-1e35, +1e35]
 *              float64: [-1e305, +1e305]
 *
 * @return arctangent of 'x'
 */
template <class D, class V>
HWY_INLINE V FastAtan(D d, V val) {
  using T = TFromD<D>;

  // Abs(val) and preserve sign for later
  auto y = Abs(val);

  // Constants for thresholds (tan 15, 30, 45, 60, 75)
  const auto tan15 = Set(d, static_cast<T>(0.26794919243));
  const auto tan30 = Set(d, static_cast<T>(0.57735026919));
  const auto tan45 = Set(d, static_cast<T>(1.0));
  const auto tan60 = Set(d, static_cast<T>(1.73205080757));
  const auto tan75 = Set(d, static_cast<T>(3.73205080757));

  constexpr size_t kLanes = HWY_MAX_LANES_D(D);
  V a, b, c, d_coef;

  if constexpr ((kLanes >= 4 && !HWY_HAVE_SCALABLE) ||
                (HWY_HAVE_SCALABLE && sizeof(T) == 4)) {
    // Index calculation by counting thresholds crossed
    // We want:
    // y < tan15 -> idx 0
    // tan15 <= y < tan30 -> idx 1
    // ...
    // y >= tan75 -> idx 5

    using DI = RebindToSigned<D>;
    auto idx_i = Zero(DI());
    const auto one_i = Set(DI(), 1);

    // Rebind masks to integer comparisons
    auto mask15 = RebindMask(DI(), Ge(y, tan15));
    auto mask30 = RebindMask(DI(), Ge(y, tan30));
    auto mask45 = RebindMask(DI(), Ge(y, tan45));
    auto mask60 = RebindMask(DI(), Ge(y, tan60));
    auto mask75 = RebindMask(DI(), Ge(y, tan75));

    idx_i = Add(idx_i, And(VecFromMask(DI(), mask15), one_i));
    idx_i = Add(idx_i, And(VecFromMask(DI(), mask30), one_i));
    idx_i = Add(idx_i, And(VecFromMask(DI(), mask45), one_i));
    idx_i = Add(idx_i, And(VecFromMask(DI(), mask60), one_i));
    idx_i = Add(idx_i, And(VecFromMask(DI(), mask75), one_i));

    HWY_ALIGN static constexpr T arr_a[] = {
        630.25357464271012, 572.95779513082321, 343.77467707849392,
        572.95779513082321, 229.18311805232929, 57.295779513082323,
        57.295779513082323, 57.295779513082323};
    HWY_ALIGN static constexpr T arr_b[] = {
        0.0000000000000000, 10.0000000000000000, 46.0000000000000000,
        217.00000000000000, 297.00000000000000,  542.00000000000000,
        542.00000000000000, 542.00000000000000};
    HWY_ALIGN static constexpr T arr_c[] = {
        -57.295779513082323, -229.18311805232929, -286.47889756541161,
        -744.84513367007019, -572.95779513082321, -630.25357464271012,
        -630.25357464271012, -630.25357464271012};
    HWY_ALIGN static constexpr T arr_d[] = {
        632.00000000000000, 657.00000000000000, 541.00000000000000,
        1252.0000000000000, 910.00000000000000, 990.00000000000000,
        990.00000000000000, 990.00000000000000};

    if constexpr (kLanes >= 8 && !HWY_HAVE_SCALABLE) {
      auto idx = IndicesFromVec(d, idx_i);
      a = TableLookupLanes(Load(d, arr_a), idx);
      b = TableLookupLanes(Load(d, arr_b), idx);
      c = TableLookupLanes(Load(d, arr_c), idx);
      d_coef = TableLookupLanes(Load(d, arr_d), idx);
    } else {
      auto idx = IndicesFromVec(d, idx_i);
      FixedTag<T, 4> d4;
      a = TwoTablesLookupLanes(d, Load(d4, arr_a), Load(d4, arr_a + 4), idx);
      b = TwoTablesLookupLanes(d, Load(d4, arr_b), Load(d4, arr_b + 4), idx);
      c = TwoTablesLookupLanes(d, Load(d4, arr_c), Load(d4, arr_c + 4), idx);
      d_coef =
          TwoTablesLookupLanes(d, Load(d4, arr_d), Load(d4, arr_d + 4), idx);
    }
  } else {
    // --- FALLBACK PATH: Blend Chain ---
    // Start with highest index (5)
    a = Set(d, static_cast<T>(57.295779513082323));
    b = Set(d, static_cast<T>(542.00000000000000));
    c = Set(d, static_cast<T>(-630.25357464271012));
    d_coef = Set(d, static_cast<T>(990.00000000000000));

    // If y < tan75 (idx 4)
    auto mask = Lt(y, tan75);
    a = IfThenElse(mask, Set(d, static_cast<T>(229.18311805232929)), a);
    b = IfThenElse(mask, Set(d, static_cast<T>(297.00000000000000)), b);
    c = IfThenElse(mask, Set(d, static_cast<T>(-572.95779513082321)), c);
    d_coef =
        IfThenElse(mask, Set(d, static_cast<T>(910.00000000000000)), d_coef);

    // If y < tan60 (idx 3)
    mask = Lt(y, tan60);
    a = IfThenElse(mask, Set(d, static_cast<T>(572.95779513082321)), a);
    b = IfThenElse(mask, Set(d, static_cast<T>(217.00000000000000)), b);
    c = IfThenElse(mask, Set(d, static_cast<T>(-744.84513367007019)), c);
    d_coef =
        IfThenElse(mask, Set(d, static_cast<T>(1252.0000000000000)), d_coef);

    // If y < tan45 (idx 2)
    mask = Lt(y, tan45);
    a = IfThenElse(mask, Set(d, static_cast<T>(343.77467707849392)), a);
    b = IfThenElse(mask, Set(d, static_cast<T>(46.0000000000000000)), b);
    c = IfThenElse(mask, Set(d, static_cast<T>(-286.47889756541161)), c);
    d_coef =
        IfThenElse(mask, Set(d, static_cast<T>(541.00000000000000)), d_coef);

    // If y < tan30 (idx 1)
    mask = Lt(y, tan30);
    a = IfThenElse(mask, Set(d, static_cast<T>(572.95779513082321)), a);
    b = IfThenElse(mask, Set(d, static_cast<T>(10.0000000000000000)), b);
    c = IfThenElse(mask, Set(d, static_cast<T>(-229.18311805232929)), c);
    d_coef =
        IfThenElse(mask, Set(d, static_cast<T>(657.00000000000000)), d_coef);

    // If y < tan15 (idx 0)
    mask = Lt(y, tan15);
    a = IfThenElse(mask, Set(d, static_cast<T>(630.25357464271012)), a);
    b = IfThenZeroElse(mask, b);
    c = IfThenElse(mask, Set(d, static_cast<T>(-57.295779513082323)), c);
    d_coef =
        IfThenElse(mask, Set(d, static_cast<T>(632.00000000000000)), d_coef);
  }

  // Math: x = (dy - b)/(a - cy)
  // num = d*y - b = MulAdd(d, y, -b)
  auto num = MulAdd(d_coef, y, Neg(b));
  // den = a - c*y = NegMulAdd(c, y, a)
  auto den = NegMulAdd(c, y, a);

  auto result = Div(num, den);
  return CopySign(result, val);
}

template <class D, class V>
HWY_NOINLINE V CallFastAtan(const D d, VecArg<V> x) {
  return FastAtan(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastTan(const D d, VecArg<V> x) {
  return FastTan(d, x);
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_MATH_FAST_MATH_INL_H_
