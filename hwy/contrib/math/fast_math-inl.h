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

namespace impl {

template <class T>
struct FastExpImpl {};

template <>
struct FastExpImpl<float> {
  // Rounds float toward zero and returns as int32_t.
  template <class D, class V>
  HWY_INLINE Vec<Rebind<int32_t, D>> ToInt32(D /*unused*/, V x) {
    return ConvertInRangeTo(Rebind<int32_t, D>(), x);
  }

  // Computes 2^x, where x is an integer.
  template <class D, class VI32>
  HWY_INLINE Vec<D> Pow2I(D d, VI32 x) {
    const Rebind<int32_t, D> di32;
    const VI32 kOffset = Set(di32, 0x7F);
    return BitCast(d, ShiftLeft<23>(Add(x, kOffset)));
  }

  // Sets the exponent of 'x' to 2^e.
  template <class D, class V, class VI32>
  HWY_INLINE V LoadExpShortRange(D d, V x, VI32 e) {
    const VI32 y = ShiftRight<1>(e);
    return Mul(Mul(x, Pow2I(d, y)), Pow2I(d, Sub(e, y)));
  }

  template <class D, class V, class VI32>
  HWY_INLINE V ExpReduce(D d, V x, VI32 q) {
    // kMinusLn2 ~= -ln(2)
    const V kMinusLn2 = Set(d, -0.69314718056f);

    // Extended precision modular arithmetic.
    const V qf = ConvertTo(d, q);
    return MulAdd(qf, kMinusLn2, x);
  }
};

#if HWY_HAVE_FLOAT64 && HWY_HAVE_INTEGER64
template <>
struct FastExpImpl<double> {
  // Rounds double toward zero and returns as int32_t.
  template <class D, class V>
  HWY_INLINE Vec<Rebind<int32_t, D>> ToInt32(D /*unused*/, V x) {
    return DemoteInRangeTo(Rebind<int32_t, D>(), x);
  }

  // Computes 2^x, where x is an integer.
  template <class D, class VI32>
  HWY_INLINE Vec<D> Pow2I(D d, VI32 x) {
    const Rebind<int32_t, D> di32;
    const Rebind<int64_t, D> di64;
    const VI32 kOffset = Set(di32, 0x3FF);
    return BitCast(d, ShiftLeft<52>(PromoteTo(di64, Add(x, kOffset))));
  }

  // Sets the exponent of 'x' to 2^e.
  template <class D, class V, class VI32>
  HWY_INLINE V LoadExpShortRange(D d, V x, VI32 e) {
    const VI32 y = ShiftRight<1>(e);
    return Mul(Mul(x, Pow2I(d, y)), Pow2I(d, Sub(e, y)));
  }

  template <class D, class V, class VI32>
  HWY_INLINE V ExpReduce(D d, V x, VI32 q) {
    // kMinusLn2 ~= -ln(2)
    const V kMinusLn2 = Set(d, -0.6931471805599453);

    // Extended precision modular arithmetic.
    const V qf = PromoteTo(d, q);
    return MulAdd(qf, kMinusLn2, x);
  }
};
#endif

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
        static_cast<T>(630.25357464271012), static_cast<T>(572.95779513082321),
        static_cast<T>(343.77467707849392), static_cast<T>(572.95779513082321),
        static_cast<T>(229.18311805232929), static_cast<T>(57.295779513082323),
        static_cast<T>(57.295779513082323), static_cast<T>(57.295779513082323)};

    HWY_ALIGN static constexpr T arr_b[] = {static_cast<T>(0.0000000000000000),
                                            static_cast<T>(10.0000000000000000),
                                            static_cast<T>(46.0000000000000000),
                                            static_cast<T>(217.00000000000000),
                                            static_cast<T>(297.00000000000000),
                                            static_cast<T>(542.00000000000000),
                                            static_cast<T>(542.00000000000000),
                                            static_cast<T>(542.00000000000000)};

    HWY_ALIGN static constexpr T arr_c[] = {
        static_cast<T>(-57.295779513082323),
        static_cast<T>(-229.18311805232929),
        static_cast<T>(-286.47889756541161),
        static_cast<T>(-744.84513367007019),
        static_cast<T>(-572.95779513082321),
        static_cast<T>(-630.25357464271012),
        static_cast<T>(-630.25357464271012),
        static_cast<T>(-630.25357464271012)};

    HWY_ALIGN static constexpr T arr_d[] = {
        static_cast<T>(632.00000000000000), static_cast<T>(657.00000000000000),
        static_cast<T>(541.00000000000000), static_cast<T>(1252.0000000000000),
        static_cast<T>(910.00000000000000), static_cast<T>(990.00000000000000),
        static_cast<T>(990.00000000000000), static_cast<T>(990.00000000000000)};

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
    epsilon_val = static_cast<T>(1e-15);
  } else {
    epsilon_val = static_cast<T>(1e-6);
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
        static_cast<T>(630.25357464271012), static_cast<T>(572.95779513082321),
        static_cast<T>(343.77467707849392), static_cast<T>(572.95779513082321),
        static_cast<T>(229.18311805232929), static_cast<T>(57.295779513082323),
        static_cast<T>(57.295779513082323), static_cast<T>(57.295779513082323)};
    HWY_ALIGN static constexpr T arr_b[] = {static_cast<T>(0.0000000000000000),
                                            static_cast<T>(10.0000000000000000),
                                            static_cast<T>(46.0000000000000000),
                                            static_cast<T>(217.00000000000000),
                                            static_cast<T>(297.00000000000000),
                                            static_cast<T>(542.00000000000000),
                                            static_cast<T>(542.00000000000000),
                                            static_cast<T>(542.00000000000000)};
    HWY_ALIGN static constexpr T arr_c[] = {
        static_cast<T>(-57.295779513082323),
        static_cast<T>(-229.18311805232929),
        static_cast<T>(-286.47889756541161),
        static_cast<T>(-744.84513367007019),
        static_cast<T>(-572.95779513082321),
        static_cast<T>(-630.25357464271012),
        static_cast<T>(-630.25357464271012),
        static_cast<T>(-630.25357464271012)};
    HWY_ALIGN static constexpr T arr_d[] = {
        static_cast<T>(632.00000000000000), static_cast<T>(657.00000000000000),
        static_cast<T>(541.00000000000000), static_cast<T>(1252.0000000000000),
        static_cast<T>(910.00000000000000), static_cast<T>(990.00000000000000),
        static_cast<T>(990.00000000000000), static_cast<T>(990.00000000000000)};

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

/**
 * Fast approximation of atan2(y, x).
 *
 * Valid Lane Types: float32, float64
 * Valid Range: As long as y/x is in Valid Range for FastAtan()
 * Correctly handles negative zero, infinities, and NaN.
 * @return atan2 of 'y', 'x'
 */
template <class D, class V>
HWY_INLINE V FastAtan2(const D d, V y, V x) {
  using T = TFromD<D>;
  using M = MFromD<D>;

  const V kHalf = Set(d, static_cast<T>(+0.5));
  const V kPi = Set(d, static_cast<T>(+3.14159265358979323846264));
  const V kPi2 = Mul(kPi, kHalf);

  const V k0 = Zero(d);
  const M y_0 = Eq(y, k0);
  const M x_0 = Eq(x, k0);
  const M x_neg = Lt(x, k0);
  const M y_inf = IsInf(y);
  const M x_inf = IsInf(x);
  const M nan = Or(IsNaN(y), IsNaN(x));

  const V if_xneg_pi = IfThenElseZero(x_neg, kPi);
  // x= +inf: pi/4; -inf: 3*pi/4; else: pi/2
  const V if_yinf = Mul(kHalf, IfThenElse(x_inf, Add(kPi2, if_xneg_pi), kPi));

  V t = FastAtan(d, Div(y, x));
  // Disambiguate between quadrants 1/3 and 2/4 by adding (Q2: Pi; Q3: -Pi).
  t = Add(t, CopySignToAbs(if_xneg_pi, y));
  // Special cases for 0 and infinity:
  t = IfThenElse(x_inf, if_xneg_pi, t);
  t = IfThenElse(x_0, kPi2, t);
  t = IfThenElse(y_inf, if_yinf, t);
  t = IfThenElse(y_0, if_xneg_pi, t);
  // Any input NaN => NaN, otherwise fix sign.
  return IfThenElse(nan, NaN(d), CopySign(t, y));
}

/**
 * Fast approximation of tanh(x).
 *
 * Valid Lane Types: float32, float64
 * Max Relative Error : 0.08% for float32, 0.08% for float64
 * Average Relative Error : 0.0007% for float32, 0.00009% for float64
 * Max Relative Error for [-0.01, 0.01] : 0.003%
 * Average Relative Error for [-0.01, 0.01] : 0.00001%
 * Valid Range: float32: [-1e35, +1e35]
 *              float64: [-1e305, +1e305]
 *
 * @return hyperbolic tangent of 'x'
 */
template <class D, class V>
HWY_INLINE V FastTanh(D d, V val) {
  using T = TFromD<D>;

  // Abs(val) and preserve sign for later
  auto y = Abs(val);

  // Thresholds for intervals (atanh(0.13), atanh(2/6), ..., atanh(0.99))
  const auto t0 = Set(d, static_cast<T>(0.130739850028878));
  const auto t1 = Set(d, static_cast<T>(0.346573590279973));
  const auto t2 = Set(d, static_cast<T>(0.549306144334055));
  const auto t3 = Set(d, static_cast<T>(0.80471895621705));
  const auto t4 = Set(d, static_cast<T>(1.19894763639919));
  const auto t5 = Set(d, static_cast<T>(1.56774710796457));
  const auto t6 = Set(d, static_cast<T>(2.64665241236225));

  constexpr size_t kLanes = HWY_MAX_LANES_D(D);
  V a, c, d_coef;

  if constexpr ((kLanes >= 4 && !HWY_HAVE_SCALABLE) ||
                (HWY_HAVE_SCALABLE && sizeof(T) == 4)) {
    // Index calculation by counting thresholds crossed
    using DI = RebindToSigned<D>;
    auto idx_i = Zero(DI());
    auto one_i = Set(DI(), 1);

    // Rebind masks to integer comparisons
    auto mask0 = RebindMask(DI(), Ge(y, t0));
    auto mask1 = RebindMask(DI(), Ge(y, t1));
    auto mask2 = RebindMask(DI(), Ge(y, t2));
    auto mask3 = RebindMask(DI(), Ge(y, t3));
    auto mask4 = RebindMask(DI(), Ge(y, t4));
    auto mask5 = RebindMask(DI(), Ge(y, t5));
    auto mask6 = RebindMask(DI(), Ge(y, t6));

#ifdef HWY_NATIVE_MASK
    // Adder tree for native masks.
    const auto sum0 = IfThenElseZero(mask0, one_i);
    const auto sum01 = MaskedAddOr(sum0, mask1, sum0, one_i);

    const auto sum2 = IfThenElseZero(mask2, one_i);
    const auto sum23 = MaskedAddOr(sum2, mask3, sum2, one_i);

    const auto sum4 = IfThenElseZero(mask4, one_i);
    const auto sum45 = MaskedAddOr(sum4, mask5, sum4, one_i);

    const auto sum6 = IfThenElseZero(mask6, one_i);

    const auto sum03 = Add(sum01, sum23);
    const auto sum46 = Add(sum45, sum6);
    idx_i = Add(sum03, sum46);
#else
    (void)one_i;
    // VecFromMask returns -1 if true, 0 otherwise.
    // We accumulate these -1s in a tree dependency to reduce latency.
    const auto m0 = VecFromMask(DI(), mask0);
    const auto m1 = VecFromMask(DI(), mask1);
    const auto m2 = VecFromMask(DI(), mask2);
    const auto m3 = VecFromMask(DI(), mask3);
    const auto m4 = VecFromMask(DI(), mask4);
    const auto m5 = VecFromMask(DI(), mask5);
    const auto m6 = VecFromMask(DI(), mask6);

    const auto sum01 = Add(m0, m1);
    const auto sum23 = Add(m2, m3);
    const auto sum45 = Add(m4, m5);

    const auto sum03 = Add(sum01, sum23);
    const auto sum46 = Add(sum45, m6);

    // idx_i = - (sum of -1s)
    idx_i = Neg(Add(sum03, sum46));
#endif

    // Clamp index to 7 to handle precision overshoots
    idx_i = Min(idx_i, Set(DI(), 7));

    HWY_ALIGN static constexpr T arr_a[] = {
        static_cast<T>(-4804.5175138358197516),
        static_cast<T>(-269.9235517815573846),
        static_cast<T>(-37.8618164320350153),
        static_cast<T>(-11.9109724888716321),
        static_cast<T>(-4.4134909385266750),
        static_cast<T>(-2.0551302095484554),
        static_cast<T>(-1.0106145266876491),
        static_cast<T>(-0.4536618986892263)};
    // arr_b is not needed since its always 1.0
    HWY_ALIGN static constexpr T arr_c[] = {
        static_cast<T>(-432.7725967893319362),
        static_cast<T>(-62.9921535198310236),
        static_cast<T>(-15.8649345215830344),
        static_cast<T>(-6.9975511052974487),
        static_cast<T>(-3.3464404050751531),
        static_cast<T>(-1.8074175146735112),
        static_cast<T>(-0.9753809406887446),
        static_cast<T>(-0.4526547639551327)};

    HWY_ALIGN static constexpr T arr_d[] = {
        static_cast<T>(-4767.3861985656808412),
        static_cast<T>(-255.7317317756317322),
        static_cast<T>(-30.8751156500003356),
        static_cast<T>(-7.2442407076571424),
        static_cast<T>(-1.1360085675341221),
        static_cast<T>(0.4099624066839623),
        static_cast<T>(0.8911139501703692),
        static_cast<T>(0.9952482080672984)};

    if constexpr (kLanes >= 8 && !HWY_HAVE_SCALABLE) {
      auto idx = IndicesFromVec(d, idx_i);
      a = TableLookupLanes(Load(d, arr_a), idx);
      c = TableLookupLanes(Load(d, arr_c), idx);
      d_coef = TableLookupLanes(Load(d, arr_d), idx);
    } else {
      auto idx = IndicesFromVec(d, idx_i);
      FixedTag<T, 4> d4;
      a = TwoTablesLookupLanes(d, Load(d4, arr_a), Load(d4, arr_a + 4), idx);
      c = TwoTablesLookupLanes(d, Load(d4, arr_c), Load(d4, arr_c + 4), idx);
      d_coef =
          TwoTablesLookupLanes(d, Load(d4, arr_d), Load(d4, arr_d + 4), idx);
    }
  } else {
    // --- FALLBACK PATH: Blend Chain ---
    // Start with highest index (7)
    a = Set(d, static_cast<T>(-0.4536618986892263));
    c = Set(d, static_cast<T>(-0.4526547639551327));
    d_coef = Set(d, static_cast<T>(0.9952482080672984));

    // If y < t6 (idx 6)
    auto mask = Lt(y, t6);
    a = IfThenElse(mask, Set(d, static_cast<T>(-1.0106145266876491)), a);
    c = IfThenElse(mask, Set(d, static_cast<T>(-0.9753809406887446)), c);
    d_coef =
        IfThenElse(mask, Set(d, static_cast<T>(0.8911139501703692)), d_coef);

    // If y < t5 (idx 5)
    mask = Lt(y, t5);
    a = IfThenElse(mask, Set(d, static_cast<T>(-2.0551302095484554)), a);
    c = IfThenElse(mask, Set(d, static_cast<T>(-1.8074175146735112)), c);
    d_coef =
        IfThenElse(mask, Set(d, static_cast<T>(0.4099624066839623)), d_coef);

    // If y < t4 (idx 4)
    mask = Lt(y, t4);
    a = IfThenElse(mask, Set(d, static_cast<T>(-4.4134909385266750)), a);
    c = IfThenElse(mask, Set(d, static_cast<T>(-3.3464404050751531)), c);
    d_coef =
        IfThenElse(mask, Set(d, static_cast<T>(-1.1360085675341221)), d_coef);

    // If y < t3 (idx 3)
    mask = Lt(y, t3);
    a = IfThenElse(mask, Set(d, static_cast<T>(-11.9109724888716321)), a);
    c = IfThenElse(mask, Set(d, static_cast<T>(-6.9975511052974487)), c);
    d_coef =
        IfThenElse(mask, Set(d, static_cast<T>(-7.2442407076571424)), d_coef);

    // If y < t2 (idx 2)
    mask = Lt(y, t2);
    a = IfThenElse(mask, Set(d, static_cast<T>(-37.8618164320350153)), a);
    c = IfThenElse(mask, Set(d, static_cast<T>(-15.8649345215830344)), c);
    d_coef =
        IfThenElse(mask, Set(d, static_cast<T>(-30.8751156500003356)), d_coef);

    // If y < t1 (idx 1)
    mask = Lt(y, t1);
    a = IfThenElse(mask, Set(d, static_cast<T>(-269.9235517815573846)), a);
    c = IfThenElse(mask, Set(d, static_cast<T>(-62.9921535198310236)), c);
    d_coef =
        IfThenElse(mask, Set(d, static_cast<T>(-255.7317317756317322)), d_coef);

    // If y < t0 (idx 0)
    mask = Lt(y, t0);
    a = IfThenElse(mask, Set(d, static_cast<T>(-4804.5175138358197516)), a);
    c = IfThenElse(mask, Set(d, static_cast<T>(-432.7725967893319362)), c);
    d_coef = IfThenElse(mask, Set(d, static_cast<T>(-4767.3861985656808412)),
                        d_coef);
  }

  // Math: y = (ax + 1.0)/(cx + d)
  auto num = MulAdd(a, y, Set(d, static_cast<T>(1.0)));
  auto den = MulAdd(c, y, d_coef);

  auto result = Div(num, den);

  const auto kSmall = Set(d, static_cast<T>(0.05));
  result = IfThenElse(Lt(y, kSmall), y, result);

  const auto one = Set(d, static_cast<T>(1.0));
  // Clamp the value to 1
  result = Min(result, one);

  return CopySign(result, val);  // Restore sign
}

/**
 * Fast approximation of log(x).
 *
 * Valid Lane Types: float32, float64
 * Max Relative Error: 0.008%
 * Average Relative Error : 0.000014%
 * Valid Range: float32: (0, +FLT_MAX]
 *              float64: (0, +DBL_MAX]
 *
 * @return natural logarithm of 'x'
 */
template <class D, class V>
HWY_INLINE V FastLog(D d, V x) {
  using T = TFromD<D>;
  using TI = MakeSigned<T>;
  using TU = MakeUnsigned<T>;
  const Rebind<TI, D> di;
  const Rebind<TU, D> du;
  using VI = decltype(Zero(di));

  constexpr bool kIsF32 = (sizeof(T) == 4);

  // Constants for Range Reduction
  // kMagic is approx 1/sqrt(2). It is used to center the mantissa interval
  // around 1.0 (specifically [0.707, 1.414])
  const VI kMagic = Set(di, kIsF32 ? static_cast<TI>(0x3F3504F3L)
                                   : static_cast<TI>(0x3FE6A09E00000000LL));
  // Bit pattern for 1.0. Used in the integer arithmetic to extract the
  // exponent.
  const VI kExpMask = Set(di, kIsF32 ? static_cast<TI>(0x3F800000L)
                                     : static_cast<TI>(0x3FF0000000000000LL));
  // Integer exponent adjustment (-25 or -54) corresponding to kScale.
  const VI kExpScale =
      Set(di, kIsF32 ? static_cast<TI>(-25) : static_cast<TI>(-54));
  // Mantissa mask.
  const VI kManMask = Set(di, kIsF32 ? static_cast<TI>(0x7FFFFFL)
                                     : static_cast<TI>(0xFFFFF00000000LL));
  // Mask for lower 32 or 64 bits.
  const VI kLowerBits = Set(di, kIsF32 ? static_cast<TI>(0x00000000L)
                                       : static_cast<TI>(0xFFFFFFFFLL));
  const V kMinNormal = Set(d, kIsF32 ? static_cast<T>(1.175494351e-38f)
                                     : static_cast<T>(2.2250738585072014e-308));
  // Scale to normalize subnormal inputs: 2^25 (f32) or 2^54 (f64)
  const V kScale = Set(d, kIsF32 ? static_cast<T>(3.355443200e+7f)
                                 : static_cast<T>(1.8014398509481984e+16));
  const V kLn2 = Set(d, static_cast<T>(0.6931471805599453));

  // Handle Subnormals
  const auto is_denormal = Lt(x, kMinNormal);
  x = MaskedMulOr(x, is_denormal, x, kScale);

  // Compute exponent
  auto exp_bits = Add(BitCast(di, x), Sub(kExpMask, kMagic));
  const VI exp_scale =
      BitCast(di, IfThenElseZero(is_denormal, BitCast(d, kExpScale)));

  constexpr int kMantissaShift = kIsF32 ? 23 : 52;
  const auto kBias = Set(di, kIsF32 ? 0x7F : 0x3FF);
  const auto exp_int = Sub(BitCast(di, ShiftRight<kMantissaShift>(
                                           BitCast(du, BitCast(d, exp_bits)))),
                           kBias);
  const auto exp = ConvertTo(d, Add(exp_scale, exp_int));

  // Renormalize x to y in [0.707, 1.414]
  const auto x_bits = BitCast(di, x);
  const auto y_bits =
      OrAnd(Add(And(exp_bits, kManMask), kMagic), x_bits, kLowerBits);
  const V y = BitCast(d, y_bits);

  // Polynomial Approximation
  const auto t0 = Set(d, static_cast<T>(0.7954951275));
  const auto t1 = Set(d, static_cast<T>(0.883883475));
  const auto t2 = Set(d, static_cast<T>(0.9722718225));
  const auto t3 = Set(d, static_cast<T>(1.06066017));
  const auto t4 = Set(d, static_cast<T>(1.1490485175));
  const auto t5 = Set(d, static_cast<T>(1.237436865));
  const auto t6 = Set(d, static_cast<T>(1.3258252125));

  constexpr size_t kLanes = HWY_MAX_LANES_D(D);
  V a, c, d_coef;

  if constexpr ((kLanes >= 4 && !HWY_HAVE_SCALABLE) ||
                (HWY_HAVE_SCALABLE && sizeof(T) == 4)) {
    // --- Table Lookup ---
    const auto scale = Set(d, static_cast<T>(11.3137085));
    // Input is always non-negative, so Floor() + ConvertTo()
    // can be replaced by direct ConvertTo() (truncation), which is faster.
    // We use MulAdd(y, scale, -8.0) instead of Mul(Sub(y, lower_bound), scale)
    // to save instructions. 0.70710678 * 11.3137085 ~= 8.0.
    auto idx_i = ConvertInRangeTo(
        RebindToSigned<D>(), MulAdd(y, scale, Set(d, static_cast<T>(-8.0))));

    // Clamp index to 7 to handle overshoots
    idx_i = Min(idx_i, Set(RebindToSigned<D>(), 7));

    HWY_ALIGN static constexpr T arr_a[] = {
        static_cast<T>(-9.9805647568302591e-01),
        static_cast<T>(-9.9957356952094290e-01),
        static_cast<T>(-9.9997448030468128e-01),
        static_cast<T>(-1.0000000000000000e+00),
        static_cast<T>(-1.0000708413493518e+00),
        static_cast<T>(-1.0004412247700072e+00),
        static_cast<T>(-1.0012578436820159e+00),
        static_cast<T>(-1.0026088937292035e+00)};
    // b array is not needed since b is always 1.0.
    HWY_ALIGN static constexpr T arr_c[] = {
        static_cast<T>(-5.8272115256950630e-01),
        static_cast<T>(-5.4794075644717266e-01),
        static_cast<T>(-5.1959981902435026e-01),
        static_cast<T>(-4.9736724255016224e-01),
        static_cast<T>(-4.7642542599075438e-01),
        static_cast<T>(-4.5972782480224245e-01),
        static_cast<T>(-4.4546134537646059e-01),
        static_cast<T>(-4.3319821691832594e-01)};
    HWY_ALIGN static constexpr T arr_d[] = {
        static_cast<T>(-4.3704086438791473e-01),
        static_cast<T>(-4.5946229210571821e-01),
        static_cast<T>(-4.8168192392472370e-01),
        static_cast<T>(-5.0257424895983926e-01),
        static_cast<T>(-5.2595942907640092e-01),
        static_cast<T>(-5.4819049252707497e-01),
        static_cast<T>(-5.7057755922517284e-01),
        static_cast<T>(-5.9318108813974268e-01)};

    if constexpr (kLanes >= 8 && !HWY_HAVE_SCALABLE) {
      auto idx = IndicesFromVec(d, idx_i);
      a = TableLookupLanes(Load(d, arr_a), idx);
      c = TableLookupLanes(Load(d, arr_c), idx);
      d_coef = TableLookupLanes(Load(d, arr_d), idx);
    } else {
      auto idx = IndicesFromVec(d, idx_i);
      FixedTag<T, 4> d4;
      a = TwoTablesLookupLanes(d, Load(d4, arr_a), Load(d4, arr_a + 4), idx);
      c = TwoTablesLookupLanes(d, Load(d4, arr_c), Load(d4, arr_c + 4), idx);
      d_coef =
          TwoTablesLookupLanes(d, Load(d4, arr_d), Load(d4, arr_d + 4), idx);
    }
  } else {
    // --- FALLBACK PATH: Blend Chain ---
    // Start with highest index (7)
    a = Set(d, static_cast<T>(-1.0026088937292035e+00));
    c = Set(d, static_cast<T>(-4.3319821691832594e-01));
    d_coef = Set(d, static_cast<T>(-5.9318108813974268e-01));

    // If y < t6 (idx 6)
    auto mask = Lt(y, t6);
    a = IfThenElse(mask, Set(d, static_cast<T>(-1.0012578436820159e+00)), a);
    c = IfThenElse(mask, Set(d, static_cast<T>(-4.4546134537646059e-01)), c);
    d_coef = IfThenElse(mask, Set(d, static_cast<T>(-5.7057755922517284e-01)),
                        d_coef);

    // If y < t5 (idx 5)
    mask = Lt(y, t5);
    a = IfThenElse(mask, Set(d, static_cast<T>(-1.0004412247700072e+00)), a);
    c = IfThenElse(mask, Set(d, static_cast<T>(-4.5972782480224245e-01)), c);
    d_coef = IfThenElse(mask, Set(d, static_cast<T>(-5.4819049252707497e-01)),
                        d_coef);

    // If y < t4 (idx 4)
    mask = Lt(y, t4);
    a = IfThenElse(mask, Set(d, static_cast<T>(-1.0000708413493518e+00)), a);
    c = IfThenElse(mask, Set(d, static_cast<T>(-4.7642542599075438e-01)), c);
    d_coef = IfThenElse(mask, Set(d, static_cast<T>(-5.2595942907640092e-01)),
                        d_coef);

    // If y < t3 (idx 3)
    mask = Lt(y, t3);
    a = IfThenElse(mask, Set(d, static_cast<T>(-1.0000000000000000e+00)), a);
    c = IfThenElse(mask, Set(d, static_cast<T>(-4.9736724255016224e-01)), c);
    d_coef = IfThenElse(mask, Set(d, static_cast<T>(-5.0257424895983926e-01)),
                        d_coef);

    // If y < t2 (idx 2)
    mask = Lt(y, t2);
    a = IfThenElse(mask, Set(d, static_cast<T>(-9.9997448030468128e-01)), a);
    c = IfThenElse(mask, Set(d, static_cast<T>(-5.1959981902435026e-01)), c);
    d_coef = IfThenElse(mask, Set(d, static_cast<T>(-4.8168192392472370e-01)),
                        d_coef);

    // If y < t1 (idx 1)
    mask = Lt(y, t1);
    a = IfThenElse(mask, Set(d, static_cast<T>(-9.9957356952094290e-01)), a);
    c = IfThenElse(mask, Set(d, static_cast<T>(-5.4794075644717266e-01)), c);
    d_coef = IfThenElse(mask, Set(d, static_cast<T>(-4.5946229210571821e-01)),
                        d_coef);

    // If y < t0 (idx 0)
    mask = Lt(y, t0);
    a = IfThenElse(mask, Set(d, static_cast<T>(-9.9805647568302591e-01)), a);
    c = IfThenElse(mask, Set(d, static_cast<T>(-5.8272115256950630e-01)), c);
    d_coef = IfThenElse(mask, Set(d, static_cast<T>(-4.3704086438791473e-01)),
                        d_coef);
  }

  // Math: y = (ax + 1.0)/(cx + d_coef)
  auto num = MulAdd(a, y, Set(d, static_cast<T>(1.0)));
  auto den = MulAdd(c, y, d_coef);

  auto approx = Div(num, den);

  return MulAdd(exp, kLn2, approx);
}

/**
 * Fast approximation of exp(x).
 *
 * Valid Lane Types: float32, float64
 * Max ULP Error: 1 for float32 [-FLT_MAX, -87]
 * Max ULP Error: 1 for float64 [-DBL_MAX, -708]
 * Max Relative Error: 0.0007% for float32 [-87, 88]
 * Max Relative Error: 0.0007% for float64 [-708, 706]
 * Average Relative Error: 0.00002% for float32 [-87, 88]
 * Average Relative Error: 0.00001% for float64 [-708, 706]
 * Valid Range: float32[-FLT_MAX, +88], float64[-DBL_MAX, +706]
 *
 * @return e^x
 */
template <class D, class V>
HWY_INLINE V FastExp(D d, V x) {
  using T = TFromD<D>;
  impl::FastExpImpl<T> impl;

  const V kHalf = Set(d, static_cast<T>(+0.5));
  const V kLowerBound =
      Set(d, static_cast<T>((sizeof(T) == 4 ? -104.0 : -1000.0)));
  const V kNegZero = Set(d, static_cast<T>(-0.0));

  const V kOneOverLog2 = Set(d, static_cast<T>(+1.442695040888963407359924681));

  using TI = MakeSigned<T>;
  const Rebind<TI, D> di;
  const auto rounded_offs = BitCast(
      d, OrAnd(BitCast(di, kHalf), BitCast(di, x), BitCast(di, kNegZero)));

  const auto q = impl.ToInt32(d, MulAdd(x, kOneOverLog2, rounded_offs));

  const auto x_red = impl.ExpReduce(d, x, q);

  // Degree 4 polynomial approximation of e^x on [-ln2/2, ln2/2]
  // Generated via Caratheodory-Fejer approximation.
  const auto c0 = Set(d, static_cast<T>(1.0000001510806224569));
  const auto c1 = Set(d, static_cast<T>(0.99996228117046825901));
  const auto c2 = Set(d, static_cast<T>(0.49998365704575670199));
  const auto c3 = Set(d, static_cast<T>(0.16792157982876812494));
  const auto c4 = Set(d, static_cast<T>(0.041959439862987071845));

  // Estrin's scheme
  const auto x2 = Mul(x_red, x_red);
  // term0 = c1*x + c0
  const auto term0 = MulAdd(c1, x_red, c0);
  // term1 = c3*x + c2
  const auto term1 = MulAdd(c3, x_red, c2);
  // term2 = c4*x^2 + term1
  const auto term2 = MulAdd(c4, x2, term1);
  // approx = term2 * x^2 + term0
  const auto approx = MulAdd(term2, x2, term0);

  const V res = impl.LoadExpShortRange(d, approx, q);

  // Handle underflow
  return IfThenElseZero(Ge(x, kLowerBound), res);
}

/**
 * Fast approximation of exp(x) for x <= 0.
 *
 * Valid Lane Types: float32, float64
 * Max ULP Error: 1 for float32 [-FLT_MAX, -87]
 * Max ULP Error: 1 for float64 [-DBL_MAX, -708]
 * Max Relative Error: 0.0007% for float32 [-87, 0]
 * Max Relative Error: 0.0007% for float64 [-708, 0]
 * Average Relative Error: 0.00002% for float32 [-87, 0]
 * Average Relative Error: 0.00001% for float64 [-708, 0]
 * Valid Range: float32[-FLT_MAX, +0.0], float64[-DBL_MAX, +0.0]
 *
 * @return e^x
 */
template <class D, class V>
HWY_INLINE V FastExpMinusOrZero(D d, V x) {
  using T = TFromD<D>;
  impl::FastExpImpl<T> impl;

  const V kHalfMinus = Set(d, static_cast<T>(-0.5));
  const V kLowerBound =
      Set(d, static_cast<T>((sizeof(T) == 4 ? -88.0 : -709.0)));

  const V kOneOverLog2 = Set(d, static_cast<T>(+1.442695040888963407359924681));

  // Optimization for x <= 0:
  // FastExp computes `rounded_offs = sign(x) ? -0.5 : 0.5` to round the
  // multiplied argument towards zero. Since x <= 0, we avoid the dynamic
  // calculation and simply use a constant -0.5 (kHalfMinus).
  //
  // We clamp x to be >= kLowerBound. For x < kLowerBound, the remapped
  // exponent q becomes -127 (f32) or -1023 (f64), which Pow2I converts to
  // exactly 0.0. This avoids subnormals and the need for a final mask.
  const auto x_clamped = Max(x, kLowerBound);
  const auto q = impl.ToInt32(d, MulAdd(x_clamped, kOneOverLog2, kHalfMinus));

  const auto x_red = impl.ExpReduce(d, x_clamped, q);

  // Degree 4 polynomial approximation of e^x on [-ln2/2, ln2/2]
  // Generated via Caratheodory-Fejer approximation.
  const auto c0 = Set(d, static_cast<T>(1.0000001510806224569));
  const auto c1 = Set(d, static_cast<T>(0.99996228117046825901));
  const auto c2 = Set(d, static_cast<T>(0.49998365704575670199));
  const auto c3 = Set(d, static_cast<T>(0.16792157982876812494));
  const auto c4 = Set(d, static_cast<T>(0.041959439862987071845));

  // Estrin's scheme
  const auto x2 = Mul(x_red, x_red);
  // term0 = c1*x + c0
  const auto term0 = MulAdd(c1, x_red, c0);
  // term1 = c3*x + c2
  const auto term1 = MulAdd(c3, x_red, c2);
  // term2 = c4*x^2 + term1
  const auto term2 = MulAdd(c4, x2, term1);
  // approx = term2 * x^2 + term0
  const auto approx = MulAdd(term2, x2, term0);

  // Since inputs < -88.0 (f32) and < -709.0 (f64) are flushed to zero,
  // we do not generate subnormals. Therefore, q is guaranteed to be >= -127
  // and we can use Pow2I directly without splitting the exponent computation.
  return Mul(approx, impl.Pow2I(d, q));
}

/**
 * Fast approximation of log2(x).
 *
 * Valid Lane Types: float32, float64
 * Max Relative Error: 0.008%
 * Valid Range: float32: (0, +FLT_MAX]
 *              float64: (0, +DBL_MAX]
 *
 * @return base 2 logarithm of 'x'
 */
template <class D, class V>
HWY_INLINE V FastLog2(D d, V x) {
  using T = TFromD<D>;
  const auto kInvLn2 = Set(d, static_cast<T>(1.4426950408889634));
  return Mul(FastLog(d, x), kInvLn2);
}

/**
 * Fast approximation of log10(x).
 *
 * Valid Lane Types: float32, float64
 * Max Relative Error: 0.008%
 * Valid Range: float32: (0, +FLT_MAX]
 *              float64: (0, +DBL_MAX]
 *
 * @return base 10 logarithm of 'x'
 */
template <class D, class V>
HWY_INLINE V FastLog10(D d, V x) {
  using T = TFromD<D>;
  const auto kInvLn10 = Set(d, static_cast<T>(0.4342944819032518));
  return Mul(FastLog(d, x), kInvLn10);
}

/**
 * Fast approximation of log(1 + x).
 *
 * Valid Lane Types: float32, float64
 * Max Relative Error: 0.0081%
 * Valid Range: float32: [-1 + epsilon, +FLT_MAX]
 *              float64: [-1 + epsilon, +DBL_MAX]
 *
 * @return natural logarithm of '1 + x'
 */
template <class D, class V>
HWY_INLINE V FastLog1p(const D d, V x) {
  using T = TFromD<D>;
  const V kOne = Set(d, static_cast<T>(+1.0));

  const V y = Add(x, kOne);
  const Mask<D> not_pole = Ne(y, kOne);
  // If y == 1, divisor becomes -1 (dummy), avoiding division by zero.
  const V kMinusOne = Set(d, static_cast<T>(-1.0));
  const V divisor = MaskedSubOr(kMinusOne, not_pole, y, kOne);
  const V non_pole = Mul(FastLog(d, y), Div(x, divisor));
  return IfThenElse(not_pole, non_pole, x);
}

/**
 * Fast approximation of base^exp.
 *
 * Valid Lane Types: float32, float64
 * Valid Range: float32: base in (0, +FLT_MAX], exp * log(base) in [-25.0,
 * +25.0] float64: base in (0, +DBL_MAX], exp * log(base) in [-25.0, +25.0] Max
 * Relative Error for Valid Range: float32 : 0.27%, float64 : 0.22%
 * @return base^exp
 */
template <class D, class V>
HWY_INLINE V FastPow(D d, V base, V exp) {
  return FastExp(d, Mul(exp, FastLog(d, base)));
}

template <class D, class V>
HWY_NOINLINE V CallFastAtan(const D d, VecArg<V> x) {
  return FastAtan(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastTan(const D d, VecArg<V> x) {
  return FastTan(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastAtan2(const D d, VecArg<V> y, VecArg<V> x) {
  return FastAtan2(d, y, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastTanh(const D d, VecArg<V> x) {
  return FastTanh(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastLog(const D d, VecArg<V> x) {
  return FastLog(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastExp(const D d, VecArg<V> x) {
  return FastExp(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastExpMinusOrZero(const D d, VecArg<V> x) {
  return FastExpMinusOrZero(d, x);
}
template <class D, class V>
HWY_NOINLINE V CallFastLog2(const D d, VecArg<V> x) {
  return FastLog2(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastLog10(const D d, VecArg<V> x) {
  return FastLog10(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastLog1p(const D d, VecArg<V> x) {
  return FastLog1p(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastPow(const D d, VecArg<V> base, VecArg<V> exp) {
  return FastPow(d, base, exp);
}
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_MATH_FAST_MATH_INL_H_
