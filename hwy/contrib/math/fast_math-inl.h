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

// Reduces input angle `ang` modulo pi into [-pi/2, +pi/2], outputting:
//   - `x_red`: |ang - q * pi| in [0, pi/2], the reduced angle magnitude.
//   - `dx`:    (pi/2 - x_red), the signed distance from x_red to the pole at
//              pi/2, computed with extended precision to avoid cancellation.
//   - `sign`:  vector whose sign bit matches tan(ang) on (-pi/2, +pi/2).
template <class D, class V = VFromD<D>>
HWY_INLINE void ReduceAngleTan(D d, V ang, V& x_red, V& dx, V& sign) {
  using T = TFromD<D>;
  const V inv_pi = Set(d, static_cast<T>(0.31830988618379067153777));

  // Step 1: Find the nearest integer multiple of pi:
  //   quotient (q) = round(ang / pi).
  const V quotient = Round(Mul(ang, inv_pi));

  // Step 2: Cody-Waite multi-word range reduction:
  // Subtract q * pi in stages using a high-precision split of pi so that
  // (ang - q * pi) does not lose mantissa bits when |ang| is large:
  //   - `t1`:      partially reduced angle after subtracting the high bits of
  //                q * pi. Because q * pi_hi is close to ang, this subtraction
  //                is exact (zero rounding error).
  //   - `pi_tail`: the remaining low-order bits of pi (pi = pi_hi + pi_tail).
  V t1;
  V kHalfPiHi;
  V pi_3;
  V pi_4 = Zero(d);
  V pi_tail;
  if constexpr (HWY_NATIVE_FMA) {
    // With hardware FMA, each NegMulAdd computes (quotient * pi_word) to
    // double-width precision before subtracting, so each constant can use all
    // mantissa bits (24 bits in f32, 53 bits in f64). Using 3 words in f32
    // gives 73 bits of pi (so even the worst-case float32 near 5152*pi =
    // 16185.48535f, where 36 bits cancel, retains full accuracy).
    const V pi_hi =
        Set(d, sizeof(T) == 8
                   ? static_cast<T>(3.14159265358979311599796346854418516)
                   : static_cast<T>(3.1415927410125732421875f));
    kHalfPiHi =
        Set(d, sizeof(T) == 8
                   ? static_cast<T>(0.5 * 3.14159265358979311599796346854418516)
                   : static_cast<T>(0.5f * 3.1415927410125732421875f));
    pi_3 =
        Set(d, sizeof(T) == 8
                   ? static_cast<T>(1.22464679914735320717376402945839660e-16)
                   : static_cast<T>(-8.742277657347585773097e-08f));
    pi_tail =
        Set(d, sizeof(T) == 8
                   ? static_cast<T>(-2.99476980971833966613301542650298899e-33)
                   : static_cast<T>(-3.430248993517596376012e-15f));
    t1 = NegMulAdd(quotient, pi_hi, ang);
  } else {
    // Without FMA, quotient * pi_k would round before subtracting if pi_k used
    // all mantissa bits. Thus pi_1..pi_4 have 14-16 trailing zero bits in their
    // binary mantissas so that each product (quotient * pi_k) and
    // ((quotient +/- 0.5) * pi_k) is 100% exact in standard floating-point
    // multiplication up to |ang| = 39000.
    // Note: Only pi_1 and pi_2 (>= 2^-23) are subtracted in `t1`; pi_3, pi_4,
    // and pi_tail (< 2^-21) are applied after `t1` so they are not rounded off
    // when |t1| ~= 1.5708 near an asymptote.
    const V pi_1 =
        Set(d, sizeof(T) == 8 ? static_cast<T>(3.141592502593994140625)
                              : static_cast<T>(3.140625f));
    const V pi_2 =
        Set(d, sizeof(T) == 8 ? static_cast<T>(1.509957883172319270673e-07)
                              : static_cast<T>(0.00096797943115234375f));
    kHalfPiHi = Set(
        d, sizeof(T) == 8
               ? static_cast<T>(0.5 * (3.141592502593994140625 +
                                       1.509957883172319270673e-07))
               : static_cast<T>(0.5f * (3.140625f + 0.00096797943115234375f)));
    pi_3 =
        Set(d, sizeof(T) == 8 ? static_cast<T>(1.078060559385155819106e-14)
                              : static_cast<T>(-3.2596290111541748046875e-07f));
    pi_4 = Set(d, sizeof(T) == 8
                      ? static_cast<T>(0.0)
                      : static_cast<T>(1.2141754268668591976165771484375e-10f));
    pi_tail =
        Set(d, sizeof(T) == 8 ? static_cast<T>(1.224646799147353207174e-22)
                              : static_cast<T>(1.2448399344364623564e-13f));
    t1 = NegMulAdd(quotient, pi_2, NegMulAdd(quotient, pi_1, ang));
  }

  // Subtract the low tails of q * pi in descending magnitude order to get the
  // signed reduced angle in [-pi/2, +pi/2]:
  //   ang_mod = ang - quotient * pi.
  V ang_mod = NegMulAdd(quotient, pi_3, t1);
  if constexpr (!HWY_NATIVE_FMA) {
    ang_mod = NegMulAdd(quotient, pi_4, ang_mod);
  }
  ang_mod = NegMulAdd(quotient, pi_tail, ang_mod);

  // Step 3: Extract sign and magnitude x_red = |ang_mod| in [0, pi/2].
  // Preserve signed zero when ang_mod == 0 so that tan(-0.0) == -0.0.
  sign = IfThenElse(Eq(ang_mod, Zero(d)), ang, ang_mod);
  x_red = Abs(ang_mod);

  // Step 4: Compute dx = (pi/2 - x_red) without losing low-order bits near the
  // asymptote x_red -> pi/2.
  // Since pi = pi_top + pi_low (where kHalfPiHi = 0.5 * pi_top exactly, so
  // pi/2 = kHalfPiHi + 0.5 * pi_low) and x_red = t1_signed - q_signed * pi_low,
  // we have the exact identity:
  //   dx = (pi/2 - x_red) = (kHalfPiHi - t1_signed) + (q_signed + 0.5) * pi_low
  // Because t1_signed ~= 1.5708 is within 2x of kHalfPiHi near any asymptote,
  // (kHalfPiHi - t1_signed) cancels the leading 1.5708 with zero rounding error
  // (by Sterbenz's lemma), and chaining (q_signed + 0.5) * pi_low in descending
  // magnitude order preserves full 73-bit accuracy at every odd multiple of
  // pi/2.
  const V sign_bit = And(sign, SignBit(d));
  const V t1_signed = Xor(t1, sign_bit);       // sgn(ang_mod) * t1
  const V q_signed = Xor(quotient, sign_bit);  // sgn(ang_mod) * quotient
  const V q_half = Add(q_signed, Set(d, static_cast<T>(0.5)));
  V dx_val = Sub(kHalfPiHi, t1_signed);
  dx_val = MulAdd(q_half, pi_3, dx_val);
  if constexpr (!HWY_NATIVE_FMA) {
    dx_val = MulAdd(q_half, pi_4, dx_val);
  }
  dx = MulAdd(q_half, pi_tail, dx_val);
}

// Range reduction and exponent extraction for logarithm functions.
//
// Mathematical Goal:
// Every positive float x > 0 is represented in standard binary scientific
// notation (buckets [2^k, 2^{k+1})) as:
//   x = 2^k * m,   where k in Z and m = x / 2^k in [1.0, 2.0).
// Taking the logarithm yields:
//   ln(x) = k * ln(2) + ln(m).
//
// Why we shift the buckets by 0.75 to [0.75 * 2^e, 1.50 * 2^e):
// If we used m in [1.0, 2.0) directly, x = 1.0 (where ln(1.0) = 0) would sit on
// a bucket boundary: inputs x = 1 - eps just below 1.0 fall into the k = -1
// bucket with m = 2 * x -> 2.0, causing (-1)*ln(2) + P(m) to suffer
// catastrophic cancellation and huge relative error as ln(x) -> 0. Multiplying
// all bucket boundaries by 0.75 shifts the buckets to [0.75 * 2^e, 1.50 * 2^e),
// placing 1.0 safely inside the e = 0 bucket [0.75, 1.50). Dividing x by 2^e
// gives:
//   x = 2^e * y,   where e in Z and y = x / 2^e in [0.75, 1.50),
//   ln(x) = e * ln(2) + ln(y).
//
// Specifically, starting from x = 2^k * m with m in [1.0, 2.0):
//   - Case 1 (1.0 <= m < 1.50): e = k,     y = m     in [1.0, 1.50)
//   - Case 2 (1.50 <= m < 2.0): e = k + 1, y = m / 2 in [0.75, 1.0)
template <bool kHandleSubnormals = true, class D, class V>
HWY_INLINE void FastLogRangeReduction(D d, V x, V& y, V& exp) {
  if constexpr (HWY_NATIVE_GET_MANTISSA) {
    // Path 1: Hardware Instructions on AVX-512 (VGETMANTPS/PD + VGETEXPPS/PD).
    // Step 1: GetMantissa0p75_1p5(x) directly extracts y in [0.75, 1.50) in one
    //         instruction (keeping y = m if m < 1.5, or halving to y = m / 2 if
    //         m >= 1.5, while normalizing subnormals in hardware).
    // Step 2: GetExponent(x) returns the original exponent k, and
    //         GetExponent(y) returns 0 when y in [1.0, 1.50) or -1 when y in
    //         [0.75, 1.0). Therefore:
    //           exp = GetExponent(x) - GetExponent(y)
    //               = k - 0    = k     (when m < 1.50)
    //               = k - (-1) = k + 1 (when m >= 1.50)
    //         yielding the exact shifted exponent e in both cases.
    y = GetMantissa0p75_1p5(x);
    exp = Sub(GetExponent(x), GetExponent(y));
  } else {
    // Path 2: Software Bit-Manipulation Fallback.
    // In IEEE-754, x = 2^k * m has integer bit representation:
    //   bits(x) = (k + bias) * 2^p + M,
    // where p = 23 (float32) or 52 (float64), and M in [0, 2^p) are the
    // fractional bits of m = 1 + M / 2^p in [1.0, 2.0).
    // Note that m >= 1.50 iff the top mantissa bit (bit p-1, value 2^{p-1})
    // is 1.
    using T = TFromD<D>;
    const RebindToSigned<D> di;
    const RebindToUnsigned<D> du;
    using TI = TFromD<decltype(di)>;
    using VI = decltype(Zero(di));

    constexpr bool kIsF32 = (sizeof(T) == 4);

    // Step 1: Compute shifted exponent e = k (if m < 1.5) or k + 1 (if m
    // >= 1.5). kExpMagicDiff = bits(1.0) - bits(0.75) = 2^{p-1} (0x00400000 for
    // float32, which is a single '1' at the top mantissa bit p-1, representing
    // 0.5). Adding 2^{p-1} to bits(x) adds 1 to the top mantissa bit of M:
    //   - If m < 1.50 (top mantissa bit is 0): no carry into bit p; exponent
    //     field remains (k + bias).
    //   - If m >= 1.50 (top mantissa bit is 1): 1 + 1 carries +1 into bit p;
    //     exponent field becomes (k + 1 + bias).
    const VI kExpMagicDiff = Set(
        di, kIsF32
                ? static_cast<TI>(0x3F800000L - 0x3F400000L)
                : static_cast<TI>(0x3FF0000000000000LL - 0x3FE8000000000000LL));

    MFromD<D> is_denormal;
    if constexpr (kHandleSubnormals) {
      const V kMinNormal =
          Set(d, kIsF32 ? static_cast<T>(1.175494351e-38f)
                        : static_cast<T>(2.2250738585072014e-308));
      const V kScale = Set(d, kIsF32 ? static_cast<T>(3.355443200e+7f)
                                     : static_cast<T>(1.8014398509481984e+16));
      is_denormal = Lt(x, kMinNormal);
      x = MaskedMulOr(x, is_denormal, x, kScale);
    } else {
      (void)is_denormal;
    }

    auto exp_bits = Add(BitCast(di, x), kExpMagicDiff);

    // Step 2: Shift right by p bits and subtract bias to obtain integer e.
    constexpr int kMantissaShift = kIsF32 ? 23 : 52;
    const auto kBias = Set(di, kIsF32 ? 0x7F : 0x3FF);
    const auto exp_int = Sub(
        BitCast(di, ShiftRight<kMantissaShift>(BitCast(du, exp_bits))), kBias);
    exp = ConvertTo(d, exp_int);

    if constexpr (kHandleSubnormals) {
      const V kExpScaleFloat =
          Set(d, kIsF32 ? static_cast<T>(-25.0) : static_cast<T>(-54.0));
      exp = MaskedAddOr(exp, is_denormal, exp, kExpScaleFloat);
    }

    // Step 3: Compute y = x / 2^e in [0.75, 1.50).
    // Dividing x by 2^e in IEEE-754 simply subtracts e from x's exponent field
    // (i.e., subtracting e * 2^p from bits(x)):
    //   y_bits = bits(x) - e * 2^p = ((k - e) + bias) * 2^p + M
    //   - If m < 1.50 (e = k):     k - e = 0  -> exponent is 0  -> y = m in
    //   [1.0, 1.50)
    //   - If m >= 1.50 (e = k + 1): k - e = -1 -> exponent is -1 -> y = m/2 in
    //   [0.75, 1.0)
    const VI exp_int_shifted = ShiftLeft<kMantissaShift>(exp_int);
    const VI y_bits = Sub(BitCast(di, x), exp_int_shifted);
    y = BitCast(d, y_bits);
  }
}

}  // namespace impl

namespace impl {

template <class T>
struct FastExpImpl {};

template <>
struct FastExpImpl<float> {
  // Rounds float toward zero and returns as int32_t.
  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE Vec<Rebind<int32_t, D>> ToInt32(D /*unused*/, V x) {
    return ConvertInRangeTo(Rebind<int32_t, D>(), x);
  }

  // Computes 2^x, where x is an integer.
  template <class D, class VI32 = Vec<Rebind<int32_t, D>>, HWY_IF_F32_D(D)>
  HWY_INLINE Vec<D> Pow2I(D d, VI32 x) {
    const Rebind<int32_t, D> di32;
    const VI32 kOffset = Set(di32, 0x7F);
    return BitCast(d, ShiftLeft<23>(Add(x, kOffset)));
  }

  // Sets the exponent of 'x' to 2^e.
  template <class D, class V = VFromD<D>, class VI32 = Vec<Rebind<int32_t, D>>,
            HWY_IF_F32_D(D)>
  HWY_INLINE V LoadExpShortRange(D d, V x, VI32 e) {
    const VI32 y = ShiftRight<1>(e);
    return Mul(Mul(x, Pow2I(d, y)), Pow2I(d, Sub(e, y)));
  }

  template <class D, class V = VFromD<D>, class VI32 = Vec<Rebind<int32_t, D>>,
            HWY_IF_F32_D(D)>
  HWY_INLINE V ExpReduce(D d, V x, VI32 q) {
    // kMinusLn2 ~= -ln(2)
    const V kMinusLn2 = Set(d, -0.69314718056f);

    // Extended precision modular arithmetic.
    const V qf = ConvertTo(d, q);
    return MulAdd(qf, kMinusLn2, x);
  }

  template <class D, class V = VFromD<D>, class VI32 = Vec<Rebind<int32_t, D>>,
            HWY_IF_F32_D(D)>
  HWY_INLINE V Exp2Reduce(D d, V x, VI32 q) {
    const V qf = ConvertTo(d, q);
    return Sub(x, qf);
  }
};

#if HWY_HAVE_FLOAT64 && HWY_HAVE_INTEGER64
template <>
struct FastExpImpl<double> {
  // Rounds double toward zero and returns as int32_t.
  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE Vec<Rebind<int32_t, D>> ToInt32(D /*unused*/, V x) {
    return DemoteInRangeTo(Rebind<int32_t, D>(), x);
  }

  // Computes 2^x, where x is an integer.
  template <class D, class VI32 = Vec<Rebind<int32_t, D>>, HWY_IF_F64_D(D)>
  HWY_INLINE Vec<D> Pow2I(D d, VI32 x) {
    const Rebind<int32_t, D> di32;
    const Rebind<int64_t, D> di64;
    const VI32 kOffset = Set(di32, 0x3FF);
    return BitCast(d, ShiftLeft<52>(PromoteTo(di64, Add(x, kOffset))));
  }

  // Sets the exponent of 'x' to 2^e.
  template <class D, class V = VFromD<D>, class VI32 = Vec<Rebind<int32_t, D>>,
            HWY_IF_F64_D(D)>
  HWY_INLINE V LoadExpShortRange(D d, V x, VI32 e) {
    const VI32 y = ShiftRight<1>(e);
    return Mul(Mul(x, Pow2I(d, y)), Pow2I(d, Sub(e, y)));
  }

  template <class D, class V = VFromD<D>, class VI32 = Vec<Rebind<int32_t, D>>,
            HWY_IF_F64_D(D)>
  HWY_INLINE V ExpReduce(D d, V x, VI32 q) {
    // kMinusLn2 ~= -ln(2)
    const V kMinusLn2 = Set(d, -0.6931471805599453);

    // Extended precision modular arithmetic.
    const V qf = PromoteTo(d, q);
    return MulAdd(qf, kMinusLn2, x);
  }

  template <class D, class V = VFromD<D>, class VI32 = Vec<Rebind<int32_t, D>>,
            HWY_IF_F64_D(D)>
  HWY_INLINE V Exp2Reduce(D d, V x, VI32 q) {
    const V qf = PromoteTo(d, q);
    return Sub(x, qf);
  }
};
#endif

}  // namespace impl

/**
 * Fast approximation of tan(x).
 *
 * Valid Lane Types: float32, float64
 * Max Relative Error: < 0.00005% (4.5e-7) for float32 in
 *                     [-89.999999, +89.999999] degrees and
 *                     < 0.000015% (1.5e-7) for float64 in
 *                     [-89.999999999999, +89.999999999999] degrees.
 * Valid Range: float32 : [-39000, +39000] rads
 *              float64 : [-1e10, +1e10] rads
 *
 * @return tangent of 'x'
 */
template <class D, class V>
HWY_INLINE V FastTan(D d, V x) {
  using T = TFromD<D>;

  // Step 1: Reduce `x` modulo pi to:
  //   - `x_red` in [0, pi/2]: reduced angle magnitude.
  //   - `dx`    = pi/2 - x_red: high-precision distance to the asymptote at
  //   pi/2.
  //   - `sign`  : sign of the reduced angle in [-pi/2, +pi/2].
  V x_red, dx, sign;
  impl::ReduceAngleTan(d, x, x_red, dx, sign);

  // Step 2: Rational approximation with an exact pole at x = pi/2.
  // Why tan(x) blows up like 1 / (pi/2 - x) near x = pi/2:
  //   tan(x) = sin(x) / cos(x). Let d = pi/2 - x. As x -> pi/2 (d -> 0),
  //   sin(x) = cos(d) -> 1, while cos(x) = sin(d) = d - d^3/6 + ... ~= d.
  //   Therefore, tan(x) ~= 1 / d = 1 / (pi/2 - x).
  //
  // Why we multiply (tan(x) / x) by (pi^2/4 - x^2) = (pi/2 - x)(pi/2 + x):
  //   1) Dividing by x removes the odd symmetry so tan(x) / x = 1 + x^2/3 + ...
  //      is a pure function of u = x^2 (halving the polynomial degree).
  //   2) Multiplying by (pi/2 - x)(pi/2 + x) cancels the 1/(pi/2 - x) infinity
  //      at both x = +pi/2 and x = -pi/2 while keeping only even powers (x^2).
  //   This yields a smooth, bounded, strictly positive function of u = x_red^2:
  //     h(u) = (pi^2/4 - u) * (tan(x_red) / x_red) ~= P2(u) / Q1(u).
  //   Dividing back by the pole factor reconstructs tan(x_red):
  //     tan(x_red) ~= (x_red * P2(u)) / ((pi/2 - x_red) * (pi/2 + x_red) *
  //     Q1(u)).
  const V u = Mul(x_red, x_red);  // u = x_red^2

  const V p2 = Set(d, static_cast<T>(0.0011413062935132635));
  const V p1 = Set(d, static_cast<T>(-0.1122671620241909));
  const V p0 = Set(d, static_cast<T>(1.0));

  const V q1 = Set(d, static_cast<T>(-0.016338901306484525));
  const V q0 = Set(d, static_cast<T>(0.40528469663390199));

  // Evaluate numerator polynomial P2(u) = (p2 * u + p1) * u + p0
  // and denominator polynomial Q1(u) = q1 * u + q0.
  const V p2_u = MulAdd(MulAdd(p2, u, p1), u, p0);
  const V q1_u = MulAdd(q1, u, q0);

  // Compute the pole factor: pole = (pi/2 - x_red) * (pi/2 + x_red) = dx *
  // sum_x.
  const V kHalfPiHi = Set(
      d, sizeof(T) == 8 ? static_cast<T>(1.57079632679489655799898173427209258)
                        : static_cast<T>(1.5707962512969970703125f));
  const V sum_x = Add(kHalfPiHi, x_red);  // pi/2 + x_red
  const V pole = Mul(dx, sum_x);          // (pi/2 - x_red) * (pi/2 + x_red)

  // Step 3: Reconstruct signed tan(x) = num / den.
  // Apply `sign` to x_red in the numerator, and do NOT take Abs(den):
  // P2(u), Q1(u), and sum_x are always positive, while dx = pi/2 - x_red
  // becomes negative if and only if x_red slightly overshoots pi/2 (when
  // round(x/pi) has not yet incremented across an asymptote). Leaving den
  // signed lets dx automatically flip the output sign at the exact pole.
  const V num = Mul(CopySign(x_red, sign), p2_u);
  const V den = Mul(pole, q1_u);

  return Div(num, den);
}

/**
 * Fast approximation of atan(x).
 *
 * Valid Lane Types: float32, float64
 * Max Relative Error: 0.0006%
 * Average Relative Error : 0.00014% for float32
 *                          0.000024% for float64
 * Valid Range: float32: [-1e35, +1e35]
 *              float64: [-1e305, +1e305]
 *
 * @return arctangent of 'x'
 */
// if kAssumePositive is true, we assume inputs are non-negative.
template <bool kAssumePositive = false, class D, class V>
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

  // Degree 5 polynomial in z = x^2 for atan(x) / x over [0, 1]
  const V p0 = Set(d, static_cast<T>(0.99999612569809));
  const V p1 = Set(d, static_cast<T>(-0.333017796278));
  const V p2 = Set(d, static_cast<T>(0.195822671055794));
  const V p3 = Set(d, static_cast<T>(-0.121763534843922));
  const V p4 = Set(d, static_cast<T>(0.0580778792500496));
  const V p5 = Set(d, static_cast<T>(-0.0137210292741656));

  const V z = Mul(mapped_y, mapped_y);
  const V z2 = Mul(z, z);
  const V z4 = Mul(z2, z2);

  // Estrin scheme for polynomial in z
  // term0 = p1*z + p0
  const V term0 = MulAdd(p1, z, p0);
  // term1 = p3*z + p2
  const V term1 = MulAdd(p3, z, p2);
  // term2 = p5*z + p4
  const V term2 = MulAdd(p5, z, p4);
  // term3 = term1 * z^2 + term0
  const V term3 = MulAdd(term1, z2, term0);
  // p_val = term2 * z^4 + term3
  const V p_val = MulAdd(term2, z4, term3);

  const V poly = Mul(mapped_y, p_val);

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

  // Pre-sort to ensure num <= den, mapping the input to the [0, 1] range.
  // This avoids a second division that would otherwise occur inside FastAtan()
  // flow for domain reduction.
  const V num = Min(ax, ay);
  const V den = Max(ax, ay);

  const M is_inf = IsInf(num);
  V mapped_y = MaskedDivOr(k0, Ne(den, k0), num, den);
  mapped_y = IfThenElse(is_inf, kOne, mapped_y);

  // Degree 5 polynomial in z = x^2 for atan(x) / x over [0, 1]
  const V p0 = Set(d, static_cast<T>(0.99999612569809));
  const V p1 = Set(d, static_cast<T>(-0.333017796278));
  const V p2 = Set(d, static_cast<T>(0.195822671055794));
  const V p3 = Set(d, static_cast<T>(-0.121763534843922));
  const V p4 = Set(d, static_cast<T>(0.0580778792500496));
  const V p5 = Set(d, static_cast<T>(-0.0137210292741656));

  const V z = Mul(mapped_y, mapped_y);
  const V z2 = Mul(z, z);
  const V z4 = Mul(z2, z2);

  // Estrin scheme for polynomial in z
  // term0 = p1*z + p0
  const V term0 = MulAdd(p1, z, p0);
  // term1 = p3*z + p2
  const V term1 = MulAdd(p3, z, p2);
  // term2 = p5*z + p4
  const V term2 = MulAdd(p5, z, p4);
  // term3 = term1 * z^2 + term0
  const V term3 = MulAdd(term1, z2, term0);
  // p_val = term2 * z^4 + term3
  const V p_val = MulAdd(term2, z4, term3);

  const V poly = Mul(mapped_y, p_val);

  const M ay_gt_ax = Gt(ay, ax);
  V angle = MaskedSubOr(poly, ay_gt_ax, kPiOverTwo, poly);

  // Test the sign bit (not Lt) so a negative-zero x reflects into the
  // negative-x half-plane, giving atan2(+/-0, -0) = +/-pi.
  const M x_neg = IsNegative(x);
  angle = MaskedSubOr(angle, x_neg, kPi, angle);

  const M is_nan = IsEitherNaN(y, x);
  return IfThenElse(is_nan, NaN(d), CopySign(angle, y));
}

namespace impl {

// Computes the index vector required for Lookup8 when the
// intervals are uneven. Runs either an adder tree or a sequential add chain
// depending on the number of registers.
template <class D, class V>
HWY_INLINE Vec<RebindToSigned<D>> ComputeIndices8Intervals(
    D d, V y, const TFromD<D>* HWY_RESTRICT thresholds) {
  using DI = RebindToSigned<D>;
  auto idx_i = Zero(DI());
  const auto one_i = Set(DI(), 1);

  const auto t0 = Set(d, thresholds[0]);
  const auto t1 = Set(d, thresholds[1]);
  const auto t2 = Set(d, thresholds[2]);
  const auto t3 = Set(d, thresholds[3]);
  const auto t4 = Set(d, thresholds[4]);
  const auto t5 = Set(d, thresholds[5]);
  const auto t6 = Set(d, thresholds[6]);

  const auto mask0 = RebindMask(DI(), Ge(y, t0));
  const auto mask1 = RebindMask(DI(), Ge(y, t1));
  const auto mask2 = RebindMask(DI(), Ge(y, t2));
  const auto mask3 = RebindMask(DI(), Ge(y, t3));
  const auto mask4 = RebindMask(DI(), Ge(y, t4));
  const auto mask5 = RebindMask(DI(), Ge(y, t5));
  const auto mask6 = RebindMask(DI(), Ge(y, t6));

  if constexpr (HWY_NATIVE_MASK) {
    if constexpr (HWY_REGISTERS >= 32) {
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
    } else {
      // 2x unrolled sequential chain.
      const auto sum0 = IfThenElseZero(mask0, one_i);
      const auto sum02 = MaskedAddOr(sum0, mask2, sum0, one_i);
      const auto sum024 = MaskedAddOr(sum02, mask4, sum02, one_i);
      const auto sum0246 = MaskedAddOr(sum024, mask6, sum024, one_i);

      const auto sum1 = IfThenElseZero(mask1, one_i);
      const auto sum13 = MaskedAddOr(sum1, mask3, sum1, one_i);
      const auto sum135 = MaskedAddOr(sum13, mask5, sum13, one_i);

      idx_i = Add(sum0246, sum135);
    }
  } else {
    (void)one_i;
    if constexpr (HWY_REGISTERS >= 32) {
      // Accummulate -1s in a tree to reduce latency
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

      idx_i = Neg(Add(sum03, sum46));
    } else {
      // Subtract in a 2x unrolled chain
      auto sum0246 = Sub(idx_i, VecFromMask(DI(), mask0));
      sum0246 = Sub(sum0246, VecFromMask(DI(), mask2));
      sum0246 = Sub(sum0246, VecFromMask(DI(), mask4));
      sum0246 = Sub(sum0246, VecFromMask(DI(), mask6));

      auto sum135 = Zero(DI());
      sum135 = Sub(sum135, VecFromMask(DI(), mask1));
      sum135 = Sub(sum135, VecFromMask(DI(), mask3));
      sum135 = Sub(sum135, VecFromMask(DI(), mask5));

      idx_i = Add(sum0246, sum135);
    }
  }
  return idx_i;
}

}  // namespace impl

/**
 * Fast approximation of tanh(x).
 *
 * Valid Lane Types: float32, float64
 * Max Relative Error : 0.00034% for float32, 0.00033% for float64
 * Average Relative Error : 6.7e-6% for float32, 3.4e-6% for float64
 * Max Relative Error for [-0.01, 0.01] : 0.00002% for float32, 2.2e-8% for
 * float64 Average Relative Error for [-0.01, 0.01] : 1.9e-7% for
 * float32, 1.6e-11% for float64 Valid Range: float32: [-1e35, +1e35] float64:
 * [-1e305, +1e305]
 * Note: FastSigmoid in third_party/gemma_cpp/ops/fast_ops-inl.h is derived
 * from FastTanh. If any changes are made to FastTanh (coefficients, clamping
 * range etc.), check if FastSigmoid also needs to be updated.
 *
 * @return hyperbolic tangent of 'x'
 */
template <class D, class V>
HWY_INLINE V FastTanh(D d, V val) {
  using T = TFromD<D>;
  // Clamp |val| to kMax = 6.65 before squaring so that numerator and
  // denominator cannot overflow to Inf / Inf = NaN for large inputs.
  const auto kMax = Set(d, static_cast<T>(6.65));
  const auto kOne = Set(d, static_cast<T>(1.0));
  const auto y = Min(Abs(val), kMax);
  const auto u = Mul(y, y);

  // Mathematical derivation of the approximation for y in [0, 6.65]:
  // 1. Taylor expansion of tanh(y) around y = 0:
  //      tanh(y) = y - (1/3)y^3 + (2/15)y^5 - (17/315)y^7 + ...
  // 2. Factor out y to get an even function containing only powers of y^2:
  //      tanh(y) / y = 1 - (1/3)y^2 + (2/15)y^4 - (17/315)y^6 + ...
  // 3. Substitute u = y^2 (for u in [0, 6.65^2]):
  //      g(u) = tanh(sqrt(u)) / sqrt(u) = 1 - (1/3)u + (2/15)u^2 - (17/315)u^3
  //      + ...
  // 4. Approximate g(u) with a degree-(3, 3) rational function P3(u) / Q3(u)
  //    (fitted via Caratheodory-Fejer):
  //      P3(u) / Q3(u) = (p3*u^3 + p2*u^2 + p1*u + 1) /
  //                      (q3*u^3 + q2*u^2 + q1*u + 1)
  // 5. Multiply P3(u) by y to obtain the final degree-(7, 6) approximation:
  //      tanh(y) ~= y * P3(u) / Q3(u)
  const auto p1 = Set(d, static_cast<T>(0.1241054959918859));
  const auto p2 = Set(d, static_cast<T>(0.002373373217532085));
  const auto p3 = Set(d, static_cast<T>(4.384877434150902e-06));

  const auto q1 = Set(d, static_cast<T>(0.4574366996671967));
  const auto q2 = Set(d, static_cast<T>(0.02152238002247929));
  const auto q3 = Set(d, static_cast<T>(0.0001528102763059944));

  // Evaluate P3(u) and Q3(u) using Estrin's scheme maximizing ILP:
  const auto u2 = Mul(u, u);

  // p_term0 = p1 * u + 1
  const auto p_term0 = MulAdd(p1, u, kOne);
  // p_term1 = p3 * u + p2
  const auto p_term1 = MulAdd(p3, u, p2);
  // q_term0 = q1 * u + 1
  const auto q_term0 = MulAdd(q1, u, kOne);
  // q_term1 = q3 * u + q2
  const auto q_term1 = MulAdd(q3, u, q2);

  // p3_u = p_term1 * u^2 + p_term0 = p3*u^3 + p2*u^2 + p1*u + 1
  const auto p3_u = MulAdd(p_term1, u2, p_term0);
  // q3_u = q_term1 * u^2 + q_term0 = q3*u^3 + q2*u^2 + q1*u + 1
  const auto q3_u = MulAdd(q_term1, u2, q_term0);
  const auto num = Mul(y, p3_u);

  // Although y * P3(u) / Q3(u) at kMax = 6.65 is designed to evaluate to 1.0
  //, we clamp with Min(..., kOne) for safety in case of FMA differences on some
  // architectures which could hypothetically cause a slight overshoot over 1.0
  const auto result = Min(Div(num, q3_u), kOne);
  return CopySign(result, val);
}

namespace impl {

// Fallback path used when Lookup8 cannot be used. Computes 4 final coefficient
// vectors by running a blend chain (either serially or in parallel depending
// on the number of registers)
template <class D, class V>
HWY_INLINE void FallbackBlendChain4Coeff(
    D d, V y, const TFromD<D>* HWY_RESTRICT thresholds,
    const TFromD<D>* HWY_RESTRICT arr_a, const TFromD<D>* HWY_RESTRICT arr_b,
    const TFromD<D>* HWY_RESTRICT arr_c, const TFromD<D>* HWY_RESTRICT arr_d,
    V& a, V& b, V& c, V& d_val) {
  const auto t0 = Set(d, thresholds[0]);
  const auto t1 = Set(d, thresholds[1]);
  const auto t2 = Set(d, thresholds[2]);
  const auto t3 = Set(d, thresholds[3]);
  const auto t4 = Set(d, thresholds[4]);
  const auto t5 = Set(d, thresholds[5]);
  const auto t6 = Set(d, thresholds[6]);

  if constexpr (HWY_REGISTERS >= 32) {
    // Split into two parallel chains to reduce dependency latency.
    // -- Chain 1: Indices 0 to 3 (Evaluated starting from t3 down to t0)
    auto a_low = Set(d, arr_a[3]);
    auto b_low = Set(d, arr_b[3]);
    auto c_low = Set(d, arr_c[3]);
    auto d_low = Set(d, arr_d[3]);

    auto mask = Lt(y, t2);
    a_low = IfThenElse(mask, Set(d, arr_a[2]), a_low);
    b_low = IfThenElse(mask, Set(d, arr_b[2]), b_low);
    c_low = IfThenElse(mask, Set(d, arr_c[2]), c_low);
    d_low = IfThenElse(mask, Set(d, arr_d[2]), d_low);

    mask = Lt(y, t1);
    a_low = IfThenElse(mask, Set(d, arr_a[1]), a_low);
    b_low = IfThenElse(mask, Set(d, arr_b[1]), b_low);
    c_low = IfThenElse(mask, Set(d, arr_c[1]), c_low);
    d_low = IfThenElse(mask, Set(d, arr_d[1]), d_low);

    mask = Lt(y, t0);
    a_low = IfThenElse(mask, Set(d, arr_a[0]), a_low);
    b_low = IfThenElse(mask, Set(d, arr_b[0]), b_low);
    c_low = IfThenElse(mask, Set(d, arr_c[0]), c_low);
    d_low = IfThenElse(mask, Set(d, arr_d[0]), d_low);

    // -- Chain 2: Indices 4 to 7 (Evaluated starting from t6 down to t4)
    auto a_high = Set(d, arr_a[7]);
    auto b_high = Set(d, arr_b[7]);
    auto c_high = Set(d, arr_c[7]);
    auto d_high = Set(d, arr_d[7]);

    mask = Lt(y, t6);
    a_high = IfThenElse(mask, Set(d, arr_a[6]), a_high);
    b_high = IfThenElse(mask, Set(d, arr_b[6]), b_high);
    c_high = IfThenElse(mask, Set(d, arr_c[6]), c_high);
    d_high = IfThenElse(mask, Set(d, arr_d[6]), d_high);

    mask = Lt(y, t5);
    a_high = IfThenElse(mask, Set(d, arr_a[5]), a_high);
    b_high = IfThenElse(mask, Set(d, arr_b[5]), b_high);
    c_high = IfThenElse(mask, Set(d, arr_c[5]), c_high);
    d_high = IfThenElse(mask, Set(d, arr_d[5]), d_high);

    mask = Lt(y, t4);
    a_high = IfThenElse(mask, Set(d, arr_a[4]), a_high);
    b_high = IfThenElse(mask, Set(d, arr_b[4]), b_high);
    c_high = IfThenElse(mask, Set(d, arr_c[4]), c_high);
    d_high = IfThenElse(mask, Set(d, arr_d[4]), d_high);

    // -- Merge the two chains
    auto merge_mask = Lt(y, t3);
    a = IfThenElse(merge_mask, a_low, a_high);
    b = IfThenElse(merge_mask, b_low, b_high);
    c = IfThenElse(merge_mask, c_low, c_high);
    d_val = IfThenElse(merge_mask, d_low, d_high);
  } else {
    // Start with highest index (7)
    a = Set(d, arr_a[7]);
    b = Set(d, arr_b[7]);
    c = Set(d, arr_c[7]);
    d_val = Set(d, arr_d[7]);

    // If y < t6 (idx 6)
    auto mask = Lt(y, t6);
    a = IfThenElse(mask, Set(d, arr_a[6]), a);
    b = IfThenElse(mask, Set(d, arr_b[6]), b);
    c = IfThenElse(mask, Set(d, arr_c[6]), c);
    d_val = IfThenElse(mask, Set(d, arr_d[6]), d_val);

    // If y < t5 (idx 5)
    mask = Lt(y, t5);
    a = IfThenElse(mask, Set(d, arr_a[5]), a);
    b = IfThenElse(mask, Set(d, arr_b[5]), b);
    c = IfThenElse(mask, Set(d, arr_c[5]), c);
    d_val = IfThenElse(mask, Set(d, arr_d[5]), d_val);

    // If y < t4 (idx 4)
    mask = Lt(y, t4);
    a = IfThenElse(mask, Set(d, arr_a[4]), a);
    b = IfThenElse(mask, Set(d, arr_b[4]), b);
    c = IfThenElse(mask, Set(d, arr_c[4]), c);
    d_val = IfThenElse(mask, Set(d, arr_d[4]), d_val);

    // If y < t3 (idx 3)
    mask = Lt(y, t3);
    a = IfThenElse(mask, Set(d, arr_a[3]), a);
    b = IfThenElse(mask, Set(d, arr_b[3]), b);
    c = IfThenElse(mask, Set(d, arr_c[3]), c);
    d_val = IfThenElse(mask, Set(d, arr_d[3]), d_val);

    // If y < t2 (idx 2)
    mask = Lt(y, t2);
    a = IfThenElse(mask, Set(d, arr_a[2]), a);
    b = IfThenElse(mask, Set(d, arr_b[2]), b);
    c = IfThenElse(mask, Set(d, arr_c[2]), c);
    d_val = IfThenElse(mask, Set(d, arr_d[2]), d_val);

    // If y < t1 (idx 1)
    mask = Lt(y, t1);
    a = IfThenElse(mask, Set(d, arr_a[1]), a);
    b = IfThenElse(mask, Set(d, arr_b[1]), b);
    c = IfThenElse(mask, Set(d, arr_c[1]), c);
    d_val = IfThenElse(mask, Set(d, arr_d[1]), d_val);

    // If y < t0 (idx 0)
    mask = Lt(y, t0);
    a = IfThenElse(mask, Set(d, arr_a[0]), a);
    b = IfThenElse(mask, Set(d, arr_b[0]), b);
    c = IfThenElse(mask, Set(d, arr_c[0]), c);
    d_val = IfThenElse(mask, Set(d, arr_d[0]), d_val);
  }
}

}  // namespace impl

namespace impl {

enum class LogScale { kLn, kLog2, kLog10 };

constexpr double GetLogScale(LogScale s) {
  switch (s) {
    case LogScale::kLn:
      return 1.0;
    case LogScale::kLog2:
      return 1.4426950408889634;
    case LogScale::kLog10:
      return 0.4342944819032518;
  }
}

template <LogScale S, class D, class V>
HWY_INLINE V FastLogPoly(D d, V z) {
  using T = TFromD<D>;

  constexpr double scale = GetLogScale(S);

  // Centering the approximation around z = y - 1 significantly improves
  // accuracy for low-degree polynomials compared to approximating log(y)
  // directly. This evaluates a degree-6 minimax polynomial for
  // log(1+z) on z in [-0.25, 0.50] (corresponding to y in [0.75, 1.50)).
  const auto c1 = Set(d, static_cast<T>(0.99999706695836765 * scale));
  const auto c2 = Set(d, static_cast<T>(-0.49987522853513588 * scale));
  const auto c3 = Set(d, static_cast<T>(0.33362333005435102 * scale));
  const auto c4 = Set(d, static_cast<T>(-0.2562350754778312 * scale));
  const auto c5 = Set(d, static_cast<T>(0.20391195859159791 * scale));
  const auto c6 = Set(d, static_cast<T>(-0.10415052602438714 * scale));

  const auto z2 = Mul(z, z);
  const auto z4 = Mul(z2, z2);
  // t0 = c1 * z
  const auto t0 = Mul(c1, z);
  // t1 = c3 * z + c2
  const auto t1 = MulAdd(c3, z, c2);
  // t2 = c5 * z + c4
  const auto t2 = MulAdd(c5, z, c4);
  // t01 = t1 * z^2 + t0 = c3*z^3 + c2*z^2 + c1*z
  const auto t01 = MulAdd(z2, t1, t0);
  // t23 = c6 * z^2 + t2 = c6*z^2 + c5*z + c4
  const auto t23 = MulAdd(z2, c6, t2);
  // result = t23 * z^4 + t01 = c6*z^6 + c5*z^5 + c4*z^4 + c3*z^3 + c2*z^2 +
  // c1*z
  return MulAdd(z4, t23, t01);
}
}  // namespace impl

/**
 * Fast approximation of log(x).
 *
 * Valid Lane Types: float32, float64
 * Max Relative Error: 0.00081% for float32, 0.00079% for float64
 * Average Relative Error: 7.4e-6% for float32, 1.2e-6% for float64
 * Valid Range: float32: (0, +FLT_MAX]
 *              float64: (0, +DBL_MAX]
 *
 * @return natural logarithm of 'x'
 */
// If false, subnormals are treated as zero.
template <bool kHandleSubnormals = true, class D, class V>
HWY_INLINE V FastLog(D d, V x) {
  using T = TFromD<D>;
  const V kLn2 = Set(d, static_cast<T>(0.6931471805599453));
  V y, exp;
  impl::FastLogRangeReduction<kHandleSubnormals>(d, x, y, exp);

  // Centering the approximation around y=1.0 by using z = y - 1.0 significantly
  // improves accuracy for low-degree polynomials compared to approximating
  // log(y) directly.
  const V z = Sub(y, Set(d, static_cast<T>(1.0)));
  const V approx = impl::FastLogPoly<impl::LogScale::kLn>(d, z);

  return MulAdd(exp, kLn2, approx);
}

/**
 * Fast approximation of exp(x).
 *
 * Valid Lane Types: float32, float64
 * Max Relative Error: 0.0007% for float32 [-87, 88]
 * Max Relative Error: 0.0007% for float64 [-708, 706]
 * Average Relative Error: 0.00002% for float32 [-87, 88]
 * Average Relative Error: 0.00001% for float64 [-708, 706]
 * Max Relative Error for Subnormals: 2.4% for float32 [-FLT_MAX, -87]
 * Max Relative Error for Subnormals: 0.006% for float64 [-DBL_MAX, -708]
 * Valid Range: float32[-FLT_MAX, +88], float64[-DBL_MAX, +706]
 *
 * @return e^x
 */
template <bool kHandleSubnormals = true, class D, class V>
HWY_INLINE V FastExp(D d, V x) {
  using T = TFromD<D>;
  impl::FastExpImpl<T> impl;

  T lower_bound_val;
  if constexpr (kHandleSubnormals) {
    lower_bound_val = sizeof(T) == 4 ? -104.0 : -1000.0;
  } else {
    lower_bound_val = sizeof(T) == 4 ? -88.0 : -709.0;
  }
  const V kLowerBound = Set(d, static_cast<T>(lower_bound_val));

  const V kHalf = Set(d, static_cast<T>(+0.5));
  const V kNegZero = Set(d, static_cast<T>(-0.0));

  const V kOneOverLog2 = Set(d, static_cast<T>(+1.442695040888963407359924681));

  using TI = MakeSigned<T>;
  const Rebind<TI, D> di;

  V x_clamped = x;
  if constexpr (!kHandleSubnormals) {
    x_clamped = Max(x, kLowerBound);
  }

  const auto rounded_offs = BitCast(
      d,
      OrAnd(BitCast(di, kHalf), BitCast(di, x_clamped), BitCast(di, kNegZero)));

  const auto q = impl.ToInt32(d, MulAdd(x_clamped, kOneOverLog2, rounded_offs));

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

  if constexpr (kHandleSubnormals) {
    const V res = impl.LoadExpShortRange(d, approx, q);
    // Handle underflow
    return IfThenElseZero(Ge(x, kLowerBound), res);
  } else {
    // Optimization: avoid splitting the exponent since 'q' is guaranteed
    // to fall within the normal floating-point ranges.
    return Mul(approx, impl.Pow2I(d, q));
  }
}

/**
 * Fast approximation of exp2(x).
 *
 * Valid Lane Types: float32, float64
 * Max Relative Error: 0.0007% for float32 [-150, 128]
 * Max Relative Error: 0.0007% for float64 [-1075, 1024]
 * Average Relative Error: 0.00002% for float32 [-150, 128]
 * Average Relative Error: 0.00001% for float64 [-1075, 1024]
 * Max Relative Error for Subnormals: 0.08% for float32 [-FLT_MAX, -150]
 * Max Relative Error for Subnormals: 0.03% for float64 [-DBL_MAX, -1075]
 * Valid Range: float32[-FLT_MAX, +128], float64[-DBL_MAX, +1024]
 *
 * @return 2^x
 */
template <bool kHandleSubnormals = true, class D, class V>
HWY_INLINE V FastExp2(D d, V x) {
  using T = TFromD<D>;
  impl::FastExpImpl<T> impl;

  T lower_bound_val;
  if constexpr (kHandleSubnormals) {
    // FastExp uses kLowerBound = -104.0 / -1000.0 since it operates on e^x. For
    // FastExp2, we use lower limits correspondingly to -150.0 and -1075.0.
    lower_bound_val = sizeof(T) == 4 ? -150.0 : -1075.0;
  } else {
    lower_bound_val = sizeof(T) == 4 ? -127.0 : -1023.0;
  }
  const V kLowerBound = Set(d, static_cast<T>(lower_bound_val));

  const V kHalf = Set(d, static_cast<T>(+0.5));
  const V kNegZero = Set(d, static_cast<T>(-0.0));

  using TI = MakeSigned<T>;
  const Rebind<TI, D> di;

  V x_clamped = x;
  if constexpr (!kHandleSubnormals) {
    x_clamped = Max(x, kLowerBound);
  }

  const auto rounded_offs = BitCast(
      d,
      OrAnd(BitCast(di, kHalf), BitCast(di, x_clamped), BitCast(di, kNegZero)));

  // FastExp calculates q = ToInt32(x * (1/ln(2)) + rounded_offs)
  // FastExp2 does not need the (1/ln(2)) scaling factor since the input is
  // already in base 2.
  const auto q = impl.ToInt32(d, Add(x_clamped, rounded_offs));

  const auto x_red = impl.Exp2Reduce(d, x_clamped, q);

  // Degree 4 polynomial approximation of 2^x on [-1/2, 1/2]
  // Derived from FastExp coefficients by pre-absorbing ln2:
  // c_fast_exp2[i] = c_fast_exp[i] * (ln2)^i.
  const auto c0 = Set(d, static_cast<T>(1.0000001510806224569));
  const auto c1 = Set(d, static_cast<T>(0.69312104523363065471));
  const auto c2 = Set(d, static_cast<T>(0.24021865239713606622));
  const auto c3 = Set(d, static_cast<T>(0.05592203117565365516));
  const auto c4 = Set(d, static_cast<T>(0.00968574163456345638));

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

  if constexpr (kHandleSubnormals) {
    const V res = impl.LoadExpShortRange(d, approx, q);
    // Handle underflow
    return IfThenElseZero(Ge(x, kLowerBound), res);
  } else {
    // Optimization: avoid splitting the exponent since 'q' is guaranteed
    // to fall within the normal floating-point ranges.
    return Mul(approx, impl.Pow2I(d, q));
  }
}

/**
 * Fast approximation of exp(x) for x <= 0. Subnormals are flushed to zero.
 *
 * Valid Lane Types: float32, float64
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
 * Max Relative Error: 0.00081% for float32, 0.00079% for float64
 * Average Relative Error: 7.4e-6% for float32, 1.2e-6% for float64
 * Valid Range: float32: (0, +FLT_MAX]
 *              float64: (0, +DBL_MAX]
 *
 * @return base 2 logarithm of 'x'
 */
// If false, subnormals are treated as zero.
template <bool kHandleSubnormals = true, class D, class V>
HWY_INLINE V FastLog2(D d, V x) {
  using T = TFromD<D>;
  V y, exp;
  impl::FastLogRangeReduction<kHandleSubnormals>(d, x, y, exp);

  // Centering the approximation around y=1.0 by using z = y - 1.0 significantly
  // improves accuracy for low-degree polynomials compared to approximating
  // log(y) directly.
  const V z = Sub(y, Set(d, static_cast<T>(1.0)));
  const V approx = impl::FastLogPoly<impl::LogScale::kLog2>(d, z);  // 1 / ln(2)

  return Add(exp, approx);
}

/**
 * Fast approximation of log10(x).
 *
 * Valid Lane Types: float32, float64
 * Max Relative Error: 0.00080% for float32, 0.00079% for float64
 * Average Relative Error: 1.0e-5% for float32, 1.2e-6% for float64
 * Valid Range: float32: (0, +FLT_MAX]
 *              float64: (0, +DBL_MAX]
 *
 * @return base 10 logarithm of 'x'
 */
// If false, subnormals are treated as zero.
template <bool kHandleSubnormals = true, class D, class V>
HWY_INLINE V FastLog10(D d, V x) {
  using T = TFromD<D>;
  V y, exp;
  impl::FastLogRangeReduction<kHandleSubnormals>(d, x, y, exp);

  // Centering the approximation around y=1.0 by using z = y - 1.0 significantly
  // improves accuracy for low-degree polynomials compared to approximating
  // log(y) directly.
  const V z = Sub(y, Set(d, static_cast<T>(1.0)));
  const V approx =
      impl::FastLogPoly<impl::LogScale::kLog10>(d, z);  // 1 / ln(10)

  const auto kLog10_2 = Set(d, static_cast<T>(0.3010299956639812));  // log10(2)
  // Computes exp * log10(2) + approx. Since approx was scaled by 1/Ln(10)
  // via the pre-scaled coefficients, this yields the correct log10 result
  // using a single MulAdd instruction.
  return MulAdd(exp, kLog10_2, approx);
}

/**
 * Fast approximation of log(1 + x).
 *
 * Valid Lane Types: float32, float64
 * Max Relative Error: 0.00081% for float32, 0.00079% for float64
 * Average Relative Error: 4.9e-5% for float32, 1.2e-5% for float64
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
  // If y == 1, divisor becomes 1 (dummy), avoiding division by zero.
  const V divisor = MaskedSubOr(y, not_pole, y, kOne);
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
 * Relative Error for Valid Range: float32 : 0.015%, float64 : 0.015%
 * @return base^exp
 */
// If false, subnormals are treated as zero.
template <bool kHandleSubnormals = true, class D, class V>
HWY_INLINE V FastPow(D d, V base, V exp) {
  return FastExp<kHandleSubnormals>(
      d, Mul(exp, FastLog<kHandleSubnormals>(d, base)));
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
HWY_NOINLINE V CallFastExp2(const D d, VecArg<V> x) {
  return FastExp2(d, x);
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
HWY_NOINLINE V CallFastExpNormal(const D d, VecArg<V> x) {
  return FastExp</*kHandleSubnormals=*/false>(d, x);
}

template <class D, class V>
HWY_NOINLINE V CallFastExp2Normal(const D d, VecArg<V> x) {
  return FastExp2</*kHandleSubnormals=*/false>(d, x);
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
  return FastAtan</*kAssumePositive=*/true>(d, x);
}
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_MATH_FAST_MATH_INL_H_
