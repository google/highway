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

#include <stddef.h>
#include <stdint.h>

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
  V b, c, d_val;

  if constexpr ((kLanes >= 4 && !HWY_HAVE_SCALABLE) ||
                (HWY_HAVE_SCALABLE && sizeof(T) == 4 && detail::IsFull(d))) {
    // --- Table Lookup ---
    const auto scale = Set(d, static_cast<T>(3.8197186342));
    auto idx_float = Floor(Mul(x_red, scale));

    // Convert to Integer Vector (Signed)
    auto idx_int = ConvertTo(RebindToSigned<D>(), idx_float);

    HWY_ALIGN static constexpr T arr_b[8] = {
        static_cast<T>(0),
        static_cast<T>(0.0174532925199432955),
        static_cast<T>(0.133808575986231942),
        static_cast<T>(0.378736447682769484),
        static_cast<T>(1.29590696960578966),
        static_cast<T>(9.45968454580926554),
        static_cast<T>(9.45968454580926554),
        static_cast<T>(9.45968454580926554)};

    HWY_ALIGN static constexpr T arr_c[8] = {
        static_cast<T>(-0.0909090909092633431),
        static_cast<T>(-0.400000000000000022),
        static_cast<T>(-0.83333333333333337),
        static_cast<T>(-1.29999999999999982),
        static_cast<T>(-2.5),
        static_cast<T>(-10.9999999999791349),
        static_cast<T>(-10.9999999999791349),
        static_cast<T>(-10.9999999999791349)};

    HWY_ALIGN static constexpr T arr_d[8] = {
        static_cast<T>(1.00277098842046231),
        static_cast<T>(1.14668131856027444),
        static_cast<T>(1.57370520888155374),
        static_cast<T>(2.18515222349690053),
        static_cast<T>(3.97062404828709958),
        static_cast<T>(17.2787595947438639),
        static_cast<T>(17.2787595947438639),
        static_cast<T>(17.2787595947438639)};

    if constexpr (kLanes >= 8 && !HWY_HAVE_SCALABLE) {
      // Cast to "Indices" Type
      auto idx = IndicesFromVec(d, idx_int);
      CappedTag<T, 8> d8;
      b = TableLookupLanes(ResizeBitCast(d, Load(d8, arr_b)), idx);
      c = TableLookupLanes(ResizeBitCast(d, Load(d8, arr_c)), idx);
      d_val = TableLookupLanes(ResizeBitCast(d, Load(d8, arr_d)), idx);
    } else {
      auto idx = IndicesFromVec(d, idx_int);
      FixedTag<T, 4> d4;
      b = TwoTablesLookupLanes(d, Load(d4, arr_b), Load(d4, arr_b + 4), idx);
      c = TwoTablesLookupLanes(d, Load(d4, arr_c), Load(d4, arr_c + 4), idx);
      d_val =
          TwoTablesLookupLanes(d, Load(d4, arr_d), Load(d4, arr_d + 4), idx);
    }
  } else {
    // --- FALLBACK PATH: Blend Chain ---
    if constexpr (HWY_REGISTERS >= 32) {
      // Split into two parallel chains to reduce dependency latency.
      const auto t0 = Set(d, static_cast<T>(0.2617993877995256));
      const auto t1 = Set(d, static_cast<T>(0.5235987755990512));
      const auto t2 = Set(d, static_cast<T>(0.7853981633985767));
      const auto t3 = Set(d, static_cast<T>(1.0471975511981024));
      const auto t4 = Set(d, static_cast<T>(1.3089969389976279));

      // -- Chain 1: Indices 0 to 2 (Evaluated starting from t1 down to t0)
      auto b_low = Set(d, static_cast<T>(0.133808575986231942));  // idx 2
      auto c_low = Set(d, static_cast<T>(-0.83333333333333337));
      auto d_low = Set(d, static_cast<T>(1.57370520888155374));

      auto mask = Lt(x_red, t1);
      b_low = IfThenElse(mask, Set(d, static_cast<T>(0.0174532925199432955)),
                         b_low);
      c_low = IfThenElse(mask, Set(d, static_cast<T>(-0.400000000000000022)),
                         c_low);
      d_low =
          IfThenElse(mask, Set(d, static_cast<T>(1.14668131856027444)), d_low);

      mask = Lt(x_red, t0);
      b_low = IfThenZeroElse(mask, b_low);
      c_low = IfThenElse(mask, Set(d, static_cast<T>(-0.0909090909092633431)),
                         c_low);
      d_low =
          IfThenElse(mask, Set(d, static_cast<T>(1.00277098842046231)), d_low);

      // -- Chain 2: Indices 3 to 5 (Evaluated starting from t4 down to t3)
      auto b_high = Set(d, static_cast<T>(9.45968454580926554));  // idx 5
      auto c_high = Set(d, static_cast<T>(-10.9999999999791349));
      auto d_high = Set(d, static_cast<T>(17.2787595947438639));

      mask = Lt(x_red, t4);
      b_high =
          IfThenElse(mask, Set(d, static_cast<T>(1.29590696960578966)), b_high);
      c_high = IfThenElse(mask, Set(d, static_cast<T>(-2.5)), c_high);
      d_high =
          IfThenElse(mask, Set(d, static_cast<T>(3.97062404828709958)), d_high);

      mask = Lt(x_red, t3);
      b_high = IfThenElse(mask, Set(d, static_cast<T>(0.378736447682769484)),
                          b_high);
      c_high = IfThenElse(mask, Set(d, static_cast<T>(-1.29999999999999982)),
                          c_high);
      d_high =
          IfThenElse(mask, Set(d, static_cast<T>(2.18515222349690053)), d_high);

      // -- Merge the two chains
      auto merge_mask = Lt(x_red, t2);
      b = IfThenElse(merge_mask, b_low, b_high);
      c = IfThenElse(merge_mask, c_low, c_high);
      d_val = IfThenElse(merge_mask, d_low, d_high);
    } else {
      b = Set(d, static_cast<T>(9.45968454580926554));
      c = Set(d, static_cast<T>(-10.9999999999791349));
      d_val = Set(d, static_cast<T>(17.2787595947438639));

      auto mask = Lt(x_red, Set(d, static_cast<T>(1.3089969389976279)));
      b = IfThenElse(mask, Set(d, static_cast<T>(1.29590696960578966)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(-2.5)), c);
      d_val =
          IfThenElse(mask, Set(d, static_cast<T>(3.97062404828709958)), d_val);

      mask = Lt(x_red, Set(d, static_cast<T>(1.0471975511981024)));
      b = IfThenElse(mask, Set(d, static_cast<T>(0.378736447682769484)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(-1.29999999999999982)), c);
      d_val =
          IfThenElse(mask, Set(d, static_cast<T>(2.18515222349690053)), d_val);

      mask = Lt(x_red, Set(d, static_cast<T>(0.7853981633985767)));
      b = IfThenElse(mask, Set(d, static_cast<T>(0.133808575986231942)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(-0.83333333333333337)), c);
      d_val =
          IfThenElse(mask, Set(d, static_cast<T>(1.57370520888155374)), d_val);

      mask = Lt(x_red, Set(d, static_cast<T>(0.5235987755990512)));
      b = IfThenElse(mask, Set(d, static_cast<T>(0.0174532925199432955)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(-0.400000000000000022)), c);
      d_val =
          IfThenElse(mask, Set(d, static_cast<T>(1.14668131856027444)), d_val);

      mask = Lt(x_red, Set(d, static_cast<T>(0.2617993877995256)));
      b = IfThenZeroElse(mask, b);
      c = IfThenElse(mask, Set(d, static_cast<T>(-0.0909090909092633431)), c);
      d_val =
          IfThenElse(mask, Set(d, static_cast<T>(1.00277098842046231)), d_val);
    }
  }

  // Math: y=(x + b)/(cx + d)
  auto num = Add(x_red, b);
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
 * Max Relative Error: 0.0034%
 * Average Relative Error : 0.0002% for float32
 *                          0.0002% for float64
 * Valid Range: float32: [-1e35, +1e35]
 *              float64: [-1e305, +1e305]
 *
 * @return arctangent of 'x'
 */
// if kAssumePositive is true, we assume inputs are non-negative.
template <class D, class V, bool kAssumePositive = false>
HWY_INLINE V FastAtan(D d, V val) {
  using T = TFromD<D>;

  // Abs(val) and preserve sign for later (if needed)
  V y;
  if constexpr (kAssumePositive) {
    y = val;
  } else {
    y = Abs(val);
  }

  const V kOne = Set(d, static_cast<T>(1.0));
  const auto gt1_mask = Gt(y, kOne);
  // Domain reduction: map [1, inf) to [0, 1]
  const V mapped_y = MaskedDivOr(y, gt1_mask, kOne, y);

  // Degree 4 polynomial for atan(x) / x over [0, 1]
  const V c0 = Set(d, static_cast<T>(0.9999653683169244));
  const V c1 = Set(d, static_cast<T>(-0.3315525587266785));
  const V c2 = Set(d, static_cast<T>(0.1844770291758270));
  const V c3 = Set(d, static_cast<T>(-0.0907475543745560));
  const V c4 = Set(d, static_cast<T>(0.0232748721030191));

  const V z = Mul(mapped_y, mapped_y);
  const V z2 = Mul(z, z);

  const V p01 = MulAdd(c1, z, c0);
  const V p23 = MulAdd(c3, z, c2);
  const V p234 = MulAdd(z2, c4, p23);
  const V p = MulAdd(z2, p234, p01);

  const V poly = Mul(mapped_y, p);

  const V kPiOverTwo = Set(d, static_cast<T>(1.57079632679489661923));
  auto result = MaskedSubOr(poly, gt1_mask, kPiOverTwo, poly);

  if constexpr (kAssumePositive) {
    return result;
  } else {
    return CopySign(result, val);
  }
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

  const V kPi = Set(d, static_cast<T>(3.14159265358979323846264));
  const V kPiOverTwo = Set(d, static_cast<T>(1.57079632679489661923));
  const V kOne = Set(d, static_cast<T>(1.0));
  const V k0 = Zero(d);

  const V ax = Abs(x);
  const V ay = Abs(y);

  const V num = Min(ax, ay);
  const V den = Max(ax, ay);

  const M is_inf = IsInf(num);
  V mapped_y = MaskedDivOr(k0, Ne(den, k0), num, den);
  mapped_y = IfThenElse(is_inf, kOne, mapped_y);

  // Degree 4 polynomial for atan(x) / x over [0, 1]
  const V c0 = Set(d, static_cast<T>(0.9999653683169244));
  const V c1 = Set(d, static_cast<T>(-0.3315525587266785));
  const V c2 = Set(d, static_cast<T>(0.1844770291758270));
  const V c3 = Set(d, static_cast<T>(-0.0907475543745560));
  const V c4 = Set(d, static_cast<T>(0.0232748721030191));

  const V z = Mul(mapped_y, mapped_y);
  const V z2 = Mul(z, z);

  const V p01 = MulAdd(c1, z, c0);
  const V p23 = MulAdd(c3, z, c2);
  const V p234 = MulAdd(z2, c4, p23);
  const V p = MulAdd(z2, p234, p01);

  const V poly = Mul(mapped_y, p);

  const M ay_gt_ax = Gt(ay, ax);
  V angle = MaskedSubOr(poly, ay_gt_ax, kPiOverTwo, poly);

  const M x_neg = Lt(x, k0);
  angle = MaskedSubOr(angle, x_neg, kPi, angle);

  const M is_nan = IsEitherNaN(y, x);
  return IfThenElse(is_nan, NaN(d), CopySign(angle, y));
}

/**
 * Fast approximation of tanh(x).
 *
 * Valid Lane Types: float32, float64
 * Max Relative Error : 0.08% for float32, 0.08% for float64
 * Average Relative Error : 0.0008% for float32, 0.0001% for float64
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

  constexpr size_t kLanes = HWY_MAX_LANES_D(D);
  V b, c, d_coef;

  if constexpr ((kLanes >= 4 && !HWY_HAVE_SCALABLE) ||
                (HWY_HAVE_SCALABLE && sizeof(T) == 4 && detail::IsFull(d))) {
    // Coefficients for P(y) ~ index using CF algo
    const auto k0 = Set(d, static_cast<T>(-0.1145426548151546));
    const auto k1 = Set(d, static_cast<T>(6.911330994691481));
    const auto k2 = Set(d, static_cast<T>(-2.511392313950185));
    const auto k3 = Set(d, static_cast<T>(0.3465107224049977));

    // Index calculation: idx = P(y)
    // Estrin's scheme
    // k0 + y * k1 + y^2 * (k2 + y * k3)
    const auto y2 = Mul(y, y);
    const auto p01 = MulAdd(k1, y, k0);
    const auto p23 = MulAdd(k3, y, k2);
    auto idx_poly = MulAdd(y2, p23, p01);

    // Convert to integer index
    using DI = RebindToSigned<D>;
    auto idx_i = ConvertTo(DI(), idx_poly);

    // Clamp index to 7
    idx_i = Min(idx_i, Set(DI(), 7));

    HWY_ALIGN static constexpr T arr_b[8] = {
        static_cast<T>(-0.000348352759899830738),
        static_cast<T>(-0.00515752779573849796),
        static_cast<T>(-0.026839995105239111),
        static_cast<T>(-0.0847183209617155403),
        static_cast<T>(-0.212185036493306028),
        static_cast<T>(-0.477817475977543593),
        static_cast<T>(-1.04124155564303233),
        static_cast<T>(-2.34441629280816644)};

    HWY_ALIGN static constexpr T arr_c[8] = {
        static_cast<T>(0.110275977731797495),
        static_cast<T>(0.253460214460919264),
        static_cast<T>(0.421190493260395371),
        static_cast<T>(0.588781480536245128),
        static_cast<T>(0.745461145870127129),
        static_cast<T>(0.878791191811599481),
        static_cast<T>(0.968182025925170109),
        static_cast<T>(0.99926173798378537)};

    HWY_ALIGN static constexpr T arr_d[8] = {
        static_cast<T>(0.98871519384438189),
        static_cast<T>(0.936262153484913595),
        static_cast<T>(0.813465176556373737),
        static_cast<T>(0.606020356299582064),
        static_cast<T>(0.289127308048047149),
        static_cast<T>(-0.188516967205459129),
        static_cast<T>(-0.941802400116966476),
        static_cast<T>(-2.33926181246965115)};

    if constexpr (kLanes >= 8 && !HWY_HAVE_SCALABLE) {
      auto idx = IndicesFromVec(d, idx_i);
      CappedTag<T, 8> d8;
      b = TableLookupLanes(ResizeBitCast(d, Load(d8, arr_b)), idx);
      c = TableLookupLanes(ResizeBitCast(d, Load(d8, arr_c)), idx);
      d_coef = TableLookupLanes(ResizeBitCast(d, Load(d8, arr_d)), idx);
    } else {
      auto idx = IndicesFromVec(d, idx_i);
      FixedTag<T, 4> d4;
      b = TwoTablesLookupLanes(d, Load(d4, arr_b), Load(d4, arr_b + 4), idx);
      c = TwoTablesLookupLanes(d, Load(d4, arr_c), Load(d4, arr_c + 4), idx);
      d_coef =
          TwoTablesLookupLanes(d, Load(d4, arr_d), Load(d4, arr_d + 4), idx);
    }
  } else {
    // --- FALLBACK PATH: Blend Chain ---
    // Thresholds for intervals
    const auto t0 = Set(d, static_cast<T>(0.1717248723716211));
    const auto t1 = Set(d, static_cast<T>(0.3477988003593246));
    const auto t2 = Set(d, static_cast<T>(0.5534457063834466));
    const auto t3 = Set(d, static_cast<T>(0.8043240819114703));
    const auto t4 = Set(d, static_cast<T>(1.1345195608232461));
    const auto t5 = Set(d, static_cast<T>(1.6442012736785494));
    const auto t6 = Set(d, static_cast<T>(2.6358900094985716));

    if constexpr (HWY_REGISTERS >= 32) {
      // Split into two parallel chains to reduce dependency latency.

      // -- Chain 1: Indices 0 to 3 (Evaluated starting from t3 down to t0)
      auto b_low = Set(d, static_cast<T>(-0.0847183209617155403));  // idx 3
      auto c_low = Set(d, static_cast<T>(0.588781480536245128));
      auto d_low = Set(d, static_cast<T>(0.606020356299582064));

      auto mask = Lt(y, t2);
      b_low = IfThenElse(mask, Set(d, static_cast<T>(-0.026839995105239111)),
                         b_low);
      c_low =
          IfThenElse(mask, Set(d, static_cast<T>(0.421190493260395371)), c_low);
      d_low =
          IfThenElse(mask, Set(d, static_cast<T>(0.813465176556373737)), d_low);

      mask = Lt(y, t1);
      b_low = IfThenElse(mask, Set(d, static_cast<T>(-0.00515752779573849796)),
                         b_low);
      c_low =
          IfThenElse(mask, Set(d, static_cast<T>(0.253460214460919264)), c_low);
      d_low =
          IfThenElse(mask, Set(d, static_cast<T>(0.936262153484913595)), d_low);

      mask = Lt(y, t0);
      b_low = IfThenElse(mask, Set(d, static_cast<T>(-0.000348352759899830738)),
                         b_low);
      c_low =
          IfThenElse(mask, Set(d, static_cast<T>(0.110275977731797495)), c_low);
      d_low =
          IfThenElse(mask, Set(d, static_cast<T>(0.98871519384438189)), d_low);

      // -- Chain 2: Indices 4 to 7 (Evaluated starting from t6 down to t4)
      auto b_high = Set(d, static_cast<T>(-2.34441629280816644));  // idx 7
      auto c_high = Set(d, static_cast<T>(0.99926173798378537));
      auto d_high = Set(d, static_cast<T>(-2.33926181246965115));

      mask = Lt(y, t6);
      b_high = IfThenElse(mask, Set(d, static_cast<T>(-1.04124155564303233)),
                          b_high);
      c_high = IfThenElse(mask, Set(d, static_cast<T>(0.968182025925170109)),
                          c_high);
      d_high = IfThenElse(mask, Set(d, static_cast<T>(-0.941802400116966476)),
                          d_high);

      mask = Lt(y, t5);
      b_high = IfThenElse(mask, Set(d, static_cast<T>(-0.477817475977543593)),
                          b_high);
      c_high = IfThenElse(mask, Set(d, static_cast<T>(0.878791191811599481)),
                          c_high);
      d_high = IfThenElse(mask, Set(d, static_cast<T>(-0.188516967205459129)),
                          d_high);

      mask = Lt(y, t4);
      b_high = IfThenElse(mask, Set(d, static_cast<T>(-0.212185036493306028)),
                          b_high);
      c_high = IfThenElse(mask, Set(d, static_cast<T>(0.745461145870127129)),
                          c_high);
      d_high = IfThenElse(mask, Set(d, static_cast<T>(0.289127308048047149)),
                          d_high);

      // -- Merge the two chains
      auto merge_mask = Lt(y, t3);
      b = IfThenElse(merge_mask, b_low, b_high);
      c = IfThenElse(merge_mask, c_low, c_high);
      d_coef = IfThenElse(merge_mask, d_low, d_high);
    } else {
      // Start with highest index (7)
      b = Set(d, static_cast<T>(-2.34441629280816644));
      c = Set(d, static_cast<T>(0.99926173798378537));
      d_coef = Set(d, static_cast<T>(-2.33926181246965115));

      // If y < t6 (idx 6)
      auto mask = Lt(y, t6);
      b = IfThenElse(mask, Set(d, static_cast<T>(-1.04124155564303233)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(0.968182025925170109)), c);
      d_coef = IfThenElse(mask, Set(d, static_cast<T>(-0.941802400116966476)),
                          d_coef);

      // If y < t5 (idx 5)
      mask = Lt(y, t5);
      b = IfThenElse(mask, Set(d, static_cast<T>(-0.477817475977543593)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(0.878791191811599481)), c);
      d_coef = IfThenElse(mask, Set(d, static_cast<T>(-0.188516967205459129)),
                          d_coef);

      // If y < t4 (idx 4)
      mask = Lt(y, t4);
      b = IfThenElse(mask, Set(d, static_cast<T>(-0.212185036493306028)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(0.745461145870127129)), c);
      d_coef = IfThenElse(mask, Set(d, static_cast<T>(0.289127308048047149)),
                          d_coef);

      // If y < t3 (idx 3)
      mask = Lt(y, t3);
      b = IfThenElse(mask, Set(d, static_cast<T>(-0.0847183209617155403)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(0.588781480536245128)), c);
      d_coef = IfThenElse(mask, Set(d, static_cast<T>(0.606020356299582064)),
                          d_coef);

      // If y < t2 (idx 2)
      mask = Lt(y, t2);
      b = IfThenElse(mask, Set(d, static_cast<T>(-0.026839995105239111)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(0.421190493260395371)), c);
      d_coef = IfThenElse(mask, Set(d, static_cast<T>(0.813465176556373737)),
                          d_coef);

      // If y < t1 (idx 1)
      mask = Lt(y, t1);
      b = IfThenElse(mask, Set(d, static_cast<T>(-0.00515752779573849796)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(0.253460214460919264)), c);
      d_coef = IfThenElse(mask, Set(d, static_cast<T>(0.936262153484913595)),
                          d_coef);

      // If y < t0 (idx 0)
      mask = Lt(y, t0);
      b = IfThenElse(mask, Set(d, static_cast<T>(-0.000348352759899830738)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(0.110275977731797495)), c);
      d_coef =
          IfThenElse(mask, Set(d, static_cast<T>(0.98871519384438189)), d_coef);
    }
  }

  // Math: y = (x + b)/(cx + d_coef)
  auto num = Add(y, b);
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
 * Max Relative Error: 0.0095%
 * Average Relative Error : 0.000014%
 * Valid Range: float32: (0, +FLT_MAX]
 *              float64: (0, +DBL_MAX]
 *
 * @return natural logarithm of 'x'
 */
// If false, subnormals are treated as zero.
template <bool kHandleSubnormals = true, class D, class V>
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

  MFromD<D> is_denormal;
  if constexpr (kHandleSubnormals) {
    // Handle Subnormals
    is_denormal = Lt(x, kMinNormal);
    x = MaskedMulOr(x, is_denormal, x, kScale);
  } else {
    (void)is_denormal;
    (void)kMinNormal;
    (void)kScale;
    (void)kExpScale;
  }

  // Compute exponent
  auto exp_bits = Add(BitCast(di, x), Sub(kExpMask, kMagic));
  VI exp_scale = Zero(di);
  if constexpr (kHandleSubnormals) {
    exp_scale = BitCast(di, IfThenElseZero(is_denormal, BitCast(d, kExpScale)));
  }

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

  constexpr size_t kLanes = HWY_MAX_LANES_D(D);
  V b, c, d_coef;

  if constexpr ((kLanes >= 4 && !HWY_HAVE_SCALABLE) ||
                (HWY_HAVE_SCALABLE && sizeof(T) == 4 && detail::IsFull(d))) {
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

    HWY_ALIGN static constexpr T arr_b[8] = {
        static_cast<T>(-1.00194730895928918),
        static_cast<T>(-1.00042661239958708),
        static_cast<T>(-1.0000255203465902),
        static_cast<T>(-1),
        static_cast<T>(-0.999929163668789478),
        static_cast<T>(-0.999558969823431065),
        static_cast<T>(-0.998743736501089163),
        static_cast<T>(-0.997397894886509873)};

    HWY_ALIGN static constexpr T arr_c[8] = {
        static_cast<T>(0.58385589069067223),
        static_cast<T>(0.548174514768112076),
        static_cast<T>(0.519613079391819999),
        static_cast<T>(0.497367242550162236),
        static_cast<T>(0.476391677761481835),
        static_cast<T>(0.459525070958496262),
        static_cast<T>(0.44490172854808846),
        static_cast<T>(0.432070989622927948)};

    HWY_ALIGN static constexpr T arr_d[8] = {
        static_cast<T>(0.437891917978712797),
        static_cast<T>(0.459658304416673158),
        static_cast<T>(0.481694216614368509),
        static_cast<T>(0.502574248959839265),
        static_cast<T>(0.525922172040079627),
        static_cast<T>(0.547948723977362273),
        static_cast<T>(0.569860763464220654),
        static_cast<T>(0.591637568597068619)};

    if constexpr (kLanes >= 8 && !HWY_HAVE_SCALABLE) {
      auto idx = IndicesFromVec(d, idx_i);
      CappedTag<T, 8> d8;
      b = TableLookupLanes(ResizeBitCast(d, Load(d8, arr_b)), idx);
      c = TableLookupLanes(ResizeBitCast(d, Load(d8, arr_c)), idx);
      d_coef = TableLookupLanes(ResizeBitCast(d, Load(d8, arr_d)), idx);
    } else {
      auto idx = IndicesFromVec(d, idx_i);
      FixedTag<T, 4> d4;
      b = TwoTablesLookupLanes(d, Load(d4, arr_b), Load(d4, arr_b + 4), idx);
      c = TwoTablesLookupLanes(d, Load(d4, arr_c), Load(d4, arr_c + 4), idx);
      d_coef =
          TwoTablesLookupLanes(d, Load(d4, arr_d), Load(d4, arr_d + 4), idx);
    }
  } else {
    // --- FALLBACK PATH: Blend Chain ---
    // Polynomial Approximation
    const auto t0 = Set(d, static_cast<T>(0.7954951287634819));
    const auto t1 = Set(d, static_cast<T>(0.8838834764038688));
    const auto t2 = Set(d, static_cast<T>(0.9722718240442556));
    const auto t3 = Set(d, static_cast<T>(1.0606601716846424));
    const auto t4 = Set(d, static_cast<T>(1.1490485193250295));
    const auto t5 = Set(d, static_cast<T>(1.2374368669654163));
    const auto t6 = Set(d, static_cast<T>(1.3258252146058032));

    if constexpr (HWY_REGISTERS >= 32) {
      // Split into two parallel chains to reduce dependency latency.

      // -- Chain 1: Indices 0 to 3 (Evaluated starting from t3 down to t0)
      auto b_low = Set(d, static_cast<T>(-1));  // idx 3
      auto c_low = Set(d, static_cast<T>(0.497367242550162236));
      auto d_low = Set(d, static_cast<T>(0.502574248959839265));

      auto mask = Lt(y, t2);
      b_low =
          IfThenElse(mask, Set(d, static_cast<T>(-1.0000255203465902)), b_low);
      c_low =
          IfThenElse(mask, Set(d, static_cast<T>(0.519613079391819999)), c_low);
      d_low =
          IfThenElse(mask, Set(d, static_cast<T>(0.481694216614368509)), d_low);

      mask = Lt(y, t1);
      b_low =
          IfThenElse(mask, Set(d, static_cast<T>(-1.00042661239958708)), b_low);
      c_low =
          IfThenElse(mask, Set(d, static_cast<T>(0.548174514768112076)), c_low);
      d_low =
          IfThenElse(mask, Set(d, static_cast<T>(0.459658304416673158)), d_low);

      mask = Lt(y, t0);
      b_low =
          IfThenElse(mask, Set(d, static_cast<T>(-1.00194730895928918)), b_low);
      c_low =
          IfThenElse(mask, Set(d, static_cast<T>(0.58385589069067223)), c_low);
      d_low =
          IfThenElse(mask, Set(d, static_cast<T>(0.437891917978712797)), d_low);

      // -- Chain 2: Indices 4 to 7 (Evaluated starting from t6 down to t4)
      auto b_high = Set(d, static_cast<T>(-0.997397894886509873));  // idx 7
      auto c_high = Set(d, static_cast<T>(0.432070989622927948));
      auto d_high = Set(d, static_cast<T>(0.591637568597068619));

      mask = Lt(y, t6);
      b_high = IfThenElse(mask, Set(d, static_cast<T>(-0.998743736501089163)),
                          b_high);
      c_high =
          IfThenElse(mask, Set(d, static_cast<T>(0.44490172854808846)), c_high);
      d_high = IfThenElse(mask, Set(d, static_cast<T>(0.569860763464220654)),
                          d_high);

      mask = Lt(y, t5);
      b_high = IfThenElse(mask, Set(d, static_cast<T>(-0.999558969823431065)),
                          b_high);
      c_high = IfThenElse(mask, Set(d, static_cast<T>(0.459525070958496262)),
                          c_high);
      d_high = IfThenElse(mask, Set(d, static_cast<T>(0.547948723977362273)),
                          d_high);

      mask = Lt(y, t4);
      b_high = IfThenElse(mask, Set(d, static_cast<T>(-0.999929163668789478)),
                          b_high);
      c_high = IfThenElse(mask, Set(d, static_cast<T>(0.476391677761481835)),
                          c_high);
      d_high = IfThenElse(mask, Set(d, static_cast<T>(0.525922172040079627)),
                          d_high);

      // -- Merge the two chains
      auto merge_mask = Lt(y, t3);
      b = IfThenElse(merge_mask, b_low, b_high);
      c = IfThenElse(merge_mask, c_low, c_high);
      d_coef = IfThenElse(merge_mask, d_low, d_high);
    } else {
      // Start with highest index (7)
      b = Set(d, static_cast<T>(-0.997397894886509873));
      c = Set(d, static_cast<T>(0.432070989622927948));
      d_coef = Set(d, static_cast<T>(0.591637568597068619));

      // If y < t6 (idx 6)
      auto mask = Lt(y, t6);
      b = IfThenElse(mask, Set(d, static_cast<T>(-0.998743736501089163)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(0.44490172854808846)), c);
      d_coef = IfThenElse(mask, Set(d, static_cast<T>(0.569860763464220654)),
                          d_coef);

      // If y < t5 (idx 5)
      mask = Lt(y, t5);
      b = IfThenElse(mask, Set(d, static_cast<T>(-0.999558969823431065)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(0.459525070958496262)), c);
      d_coef = IfThenElse(mask, Set(d, static_cast<T>(0.547948723977362273)),
                          d_coef);

      // If y < t4 (idx 4)
      mask = Lt(y, t4);
      b = IfThenElse(mask, Set(d, static_cast<T>(-0.999929163668789478)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(0.476391677761481835)), c);
      d_coef = IfThenElse(mask, Set(d, static_cast<T>(0.525922172040079627)),
                          d_coef);

      // If y < t3 (idx 3)
      mask = Lt(y, t3);
      b = IfThenElse(mask, Set(d, static_cast<T>(-1)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(0.497367242550162236)), c);
      d_coef = IfThenElse(mask, Set(d, static_cast<T>(0.502574248959839265)),
                          d_coef);

      // If y < t2 (idx 2)
      mask = Lt(y, t2);
      b = IfThenElse(mask, Set(d, static_cast<T>(-1.0000255203465902)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(0.519613079391819999)), c);
      d_coef = IfThenElse(mask, Set(d, static_cast<T>(0.481694216614368509)),
                          d_coef);

      // If y < t1 (idx 1)
      mask = Lt(y, t1);
      b = IfThenElse(mask, Set(d, static_cast<T>(-1.00042661239958708)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(0.548174514768112076)), c);
      d_coef = IfThenElse(mask, Set(d, static_cast<T>(0.459658304416673158)),
                          d_coef);

      // If y < t0 (idx 0)
      mask = Lt(y, t0);
      b = IfThenElse(mask, Set(d, static_cast<T>(-1.00194730895928918)), b);
      c = IfThenElse(mask, Set(d, static_cast<T>(0.58385589069067223)), c);
      d_coef = IfThenElse(mask, Set(d, static_cast<T>(0.437891917978712797)),
                          d_coef);
    }
  }

  // Math: y = (x + b)/(cx + d_coef)
  auto num = Add(y, b);
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
// If false, subnormals are treated as zero.
template <bool kHandleSubnormals = true, class D, class V>
HWY_INLINE V FastLog2(D d, V x) {
  using T = TFromD<D>;
  const auto kInvLn2 = Set(d, static_cast<T>(1.4426950408889634));
  return Mul(FastLog<kHandleSubnormals>(d, x), kInvLn2);
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
// If false, subnormals are treated as zero.
template <bool kHandleSubnormals = true, class D, class V>
HWY_INLINE V FastLog10(D d, V x) {
  using T = TFromD<D>;
  const auto kInvLn10 = Set(d, static_cast<T>(0.4342944819032518));
  return Mul(FastLog<kHandleSubnormals>(d, x), kInvLn10);
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
// If false, subnormals are treated as zero.
template <bool kHandleSubnormals = true, class D, class V>
HWY_INLINE V FastLog1p(const D d, V x) {
  using T = TFromD<D>;
  const V kOne = Set(d, static_cast<T>(+1.0));

  const V y = Add(x, kOne);
  const Mask<D> not_pole = Ne(y, kOne);
  // If y == 1, divisor becomes -1 (dummy), avoiding division by zero.
  const V kMinusOne = Set(d, static_cast<T>(-1.0));
  const V divisor = MaskedSubOr(kMinusOne, not_pole, y, kOne);
  // Ensure exactly 1.0 when x == divisor. This is necessary because some
  // platforms (like Armv7) use Newton-Raphson for division, which can return
  // 0.0, instead of 1.0 when the reciprocal calculation underflows
  // for very large x.
  const V div_res = MaskedDivOr(kOne, Ne(x, divisor), x, divisor);
  const V non_pole = Mul(FastLog<kHandleSubnormals>(d, y), div_res);
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
// If false, subnormals are treated as zero.
template <bool kHandleSubnormals = true, class D, class V>
HWY_INLINE V FastPow(D d, V base, V exp) {
  return FastExp(d, Mul(exp, FastLog<kHandleSubnormals>(d, base)));
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
  return FastLog<>(d, x);
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
  return FastLog2<>(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastLog10(const D d, VecArg<V> x) {
  return FastLog10<>(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastLog1p(const D d, VecArg<V> x) {
  return FastLog1p<>(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastPow(const D d, VecArg<V> base, VecArg<V> exp) {
  return FastPow<>(d, base, exp);
}

template <class D, class V>
HWY_NOINLINE V CallFastLogPositiveNormal(const D d, VecArg<V> x) {
  return FastLog</*kHandleSubnormals=*/false>(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastLog2PositiveNormal(const D d, VecArg<V> x) {
  return FastLog2</*kHandleSubnormals=*/false>(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastLog10PositiveNormal(const D d, VecArg<V> x) {
  return FastLog10</*kHandleSubnormals=*/false>(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastLog1pPositiveNormal(const D d, VecArg<V> x) {
  return FastLog1p</*kHandleSubnormals=*/false>(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastAtanPositive(const D d, VecArg<V> x) {
  return FastAtan<D, V, /*kAssumePositive=*/true>(d, x);
}
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_MATH_FAST_MATH_INL_H_
