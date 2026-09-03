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

// Floating-point decomposition / manipulation, i.e. the <cmath> functions that
// take a float apart or rebuild it from its exponent and significand rather
// than compute a transcendental: ldexp / scalbn, ilogb, logb, modf, nextafter,
// fmod. These are exact (0 ULP) and operate on the IEEE-754 representation, so
// they need no polynomial and are defined for float32 and float64. Resolves the
// "fmod, ilogb, logb, modf, nextafter, scalbn" item in g3doc/op_wishlist.md.

// Include guard (still compiled once per target)
#if defined(HIGHWAY_HWY_CONTRIB_MATH_FLOAT_MANIP_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_MATH_FLOAT_MANIP_INL_H_
#undef HIGHWAY_HWY_CONTRIB_MATH_FLOAT_MANIP_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_MATH_FLOAT_MANIP_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

namespace detail {

// floor(log2(|x|)) as a signed-integer lane, valid for any non-zero finite x
// including subnormals. Undefined (caller must mask) for x in {0, inf, NaN}.
template <class D, HWY_IF_FLOAT3264_D(D)>
HWY_INLINE VFromD<RebindToSigned<D>> RawExponent(D d, VFromD<D> x) {
  using T = TFromD<D>;
  const RebindToUnsigned<D> du;
  const RebindToSigned<D> di;
  using TU = TFromD<decltype(du)>;
  constexpr int kMantBits = MantissaBits<T>();

  const VFromD<D> ax = Abs(x);
  // Subnormals have a biased exponent of 0; scaling by 2^kMantBits makes them
  // normal (the multiply is exact) and we undo the bias shift afterwards.
  const MFromD<decltype(du)> is_sub =
      Lt(BitCast(du, ax), Set(du, TU{1} << kMantBits));
  const VFromD<D> scaled =
      IfThenElse(RebindMask(d, is_sub),
                 MulByPow2(ax, Set(di, TFromD<decltype(di)>{kMantBits})), ax);

  const VFromD<decltype(di)> biased = BitCast(di, GetBiasedExponent(scaled));
  const VFromD<decltype(di)> kBias =
      Set(di, static_cast<TFromD<decltype(di)>>(MaxExponentField<T>() >> 1));
  VFromD<decltype(di)> e = Sub(biased, kBias);
  e = IfThenElse(RebindMask(di, is_sub),
                 Sub(e, Set(di, TFromD<decltype(di)>{kMantBits})), e);
  return e;
}

}  // namespace detail

// ------------------------------ Ldexp / Scalbn

// std::ldexp(x, exp) == std::scalbn(x, exp) for binary floating-point:
// returns x[i] * 2^exp[i], with `exp` a signed-integer vector. Overflow
// saturates to +/-inf and underflow to +/-0, and the exponent is clamped to a
// range wide enough that any finite result is exact (same as MulByPow2).
template <class D, HWY_IF_FLOAT3264_D(D)>
HWY_INLINE VFromD<D> Ldexp(D /*d*/, VFromD<D> x,
                           VFromD<RebindToSigned<D>> exp) {
  return MulByPow2(x, exp);
}

template <class D, HWY_IF_FLOAT3264_D(D)>
HWY_INLINE VFromD<D> Scalbn(D d, VFromD<D> x, VFromD<RebindToSigned<D>> exp) {
  return Ldexp(d, x, exp);
}

// ------------------------------ Ilogb

// std::ilogb(x): the unbiased base-2 exponent of |x| as a signed-integer lane
// (int32 for float32, int64 for float64), i.e. floor(log2(|x|)). Special cases
// match glibc: ilogb(+/-0) and ilogb(NaN) return the minimum lane value,
// ilogb(+/-inf) returns the maximum lane value.
template <class D, HWY_IF_FLOAT3264_D(D)>
HWY_INLINE VFromD<RebindToSigned<D>> Ilogb(D d, VFromD<D> x) {
  const RebindToSigned<D> di;
  using TI = TFromD<decltype(di)>;

  VFromD<decltype(di)> e = detail::RawExponent(d, x);
  e = IfThenElse(RebindMask(di, Eq(Abs(x), Zero(d))), Set(di, LimitsMin<TI>()),
                 e);
  e = IfThenElse(RebindMask(di, IsInf(x)), Set(di, LimitsMax<TI>()), e);
  e = IfThenElse(RebindMask(di, IsNaN(x)), Set(di, LimitsMin<TI>()), e);
  return e;
}

// ------------------------------ Logb

// std::logb(x): floor(log2(|x|)) as a floating-point value of the same type.
// logb(+/-0) == -inf, logb(+/-inf) == +inf, logb(NaN) == NaN.
template <class D, HWY_IF_FLOAT3264_D(D)>
HWY_INLINE VFromD<D> Logb(D d, VFromD<D> x) {
  VFromD<D> e = ConvertTo(d, detail::RawExponent(d, x));
  e = IfThenElse(Eq(Abs(x), Zero(d)), Neg(Inf(d)), e);
  e = IfThenElse(IsInf(x), Inf(d), e);
  e = IfThenElse(IsNaN(x), NaN(d), e);
  return e;
}

// ------------------------------ Modf

// std::modf(x, &int_part): splits x into its integer part (truncated toward
// zero, written to `int_part`) and the returned fractional part. Both carry the
// sign of x, including for +/-0. modf(+/-inf) writes +/-inf and returns +/-0.
template <class D, HWY_IF_FLOAT3264_D(D)>
HWY_INLINE VFromD<D> Modf(D d, VFromD<D> x, VFromD<D>& int_part) {
  // Trunc keeps +/-inf and NaN unchanged, but may not preserve the sign of a
  // zero (e.g. HWY_EMU128), which modf must for both outputs.
  const VFromD<D> ip = CopySign(Trunc(x), x);
  VFromD<D> frac = Sub(x, ip);
  // inf - inf is NaN; the fractional part of +/-inf is defined as +/-0.
  frac = IfThenElse(IsInf(x), Zero(d), frac);
  // Give a zero fractional part the sign of x (C requires "same sign as x").
  frac = CopySign(frac, x);
  int_part = ip;
  return frac;
}

// ------------------------------ NextAfter

// std::nextafter(from, to): the representable value adjacent to `from` in the
// direction of `to` (== std::nexttoward for float32/float64). Returns `to` when
// from == to (including +-0), NaN if either input is NaN. Stepping past the
// largest finite magnitude yields +/-inf; stepping below the smallest
// subnormal yields +/-0 with the sign of `from`.
template <class D, HWY_IF_FLOAT3264_D(D)>
HWY_INLINE VFromD<D> NextAfter(D d, VFromD<D> from, VFromD<D> to) {
  using T = TFromD<D>;
  const RebindToUnsigned<D> du;
  using TU = TFromD<decltype(du)>;

  const VFromD<decltype(du)> kSign = Set(du, SignMask<T>());
  const VFromD<decltype(du)> bits = BitCast(du, from);
  const VFromD<decltype(du)> magn = AndNot(kSign, bits);  // |from| as bits
  const VFromD<decltype(du)> sbit = And(bits, kSign);

  // Magnitude grows when `to` is farther from zero than `from` on from's side.
  const MFromD<D> up = Or(And(Ge(from, Zero(d)), Gt(to, from)),
                          And(Le(from, Zero(d)), Lt(to, from)));
  const VFromD<decltype(du)> stepped = IfThenElse(
      RebindMask(du, up), Add(magn, Set(du, TU{1})), Sub(magn, Set(du, TU{1})));
  VFromD<D> result = BitCast(d, Or(stepped, sbit));

  // from == 0: the neighbour is the smallest subnormal with the sign of `to`.
  const VFromD<D> tiny =
      BitCast(d, Or(Set(du, TU{1}), And(BitCast(du, to), kSign)));
  result = IfThenElse(Eq(from, Zero(d)), tiny, result);

  result = IfThenElse(Eq(from, to), to, result);
  result = IfThenElse(Or(IsNaN(from), IsNaN(to)), NaN(d), result);
  return result;
}

// ------------------------------ Fmod

// std::fmod(x, y): the exact floating-point remainder of x / y with the sign of
// x, i.e. x - n*y where n is x/y truncated toward zero. Computed from the
// integer significands so it is exact for every input (no cancellation from a
// rounded quotient). fmod(x, 0), fmod(+/-inf, y) and fmod(x, NaN) are NaN;
// fmod(x, +/-inf) is x for finite x.
//
// The core reduction runs once per bit of the exponent gap ilogb(x)-ilogb(y),
// masked per lane; inputs whose magnitudes are within a few powers of two
// finish in a handful of iterations, but a very large ratio (e.g. DBL_MAX
// mod DBL_TRUE_MIN) is correspondingly slow.
template <class D, HWY_IF_FLOAT3264_D(D)>
HWY_INLINE VFromD<D> Fmod(D d, VFromD<D> x, VFromD<D> y) {
  using T = TFromD<D>;
  const RebindToUnsigned<D> du;
  const RebindToSigned<D> di;
  using TU = TFromD<decltype(du)>;
  using TI = TFromD<decltype(di)>;
  constexpr int kMantBits = MantissaBits<T>();
  const VFromD<decltype(du)> kMantMask = Set(du, MantissaMask<T>());
  const VFromD<decltype(du)> kImplicit = Set(du, TU{1} << kMantBits);

  const VFromD<D> ax = Abs(x);
  const VFromD<D> ay = Abs(y);

  // Significand with the implicit leading bit, left-justified so the leading
  // bit sits at position kMantBits; `e` is its unbiased exponent. Works for
  // normals and subnormals (HighestSetBitIndex avoids a normalization loop).
  const VFromD<decltype(du)> kTopPos = Set(du, static_cast<TU>(kMantBits));
  auto split = [&](VFromD<D> a, VFromD<decltype(du)>& m,
                   VFromD<decltype(di)>& e) HWY_ATTR {
    const VFromD<decltype(du)> abits = BitCast(du, a);
    const VFromD<decltype(di)> biased =
        BitCast(di, ShiftRight<kMantBits>(abits));
    const MFromD<decltype(di)> sub = Eq(biased, Zero(di));
    const VFromD<decltype(du)> frac = And(abits, kMantMask);
    // Normal: m = frac | implicit, e = biased - bias.
    // Subnormal: shift frac up so its top set bit reaches position kMantBits.
    const VFromD<decltype(du)> top =
        Min(BitCast(du, HighestSetBitIndex(Or(frac, Set(du, TU{1})))), kTopPos);
    const VFromD<decltype(du)> shift = Sub(kTopPos, top);
    const VFromD<decltype(du)> m_sub = Shl(frac, shift);
    // Smallest subnormal has unbiased exponent 1-(bias)-kMantBits; shifting the
    // significand left by `shift` raises floor(log2) by the same amount, giving
    // e = (1 - bias) - shift.
    const VFromD<decltype(di)> kSubBase = Set(
        di, static_cast<TI>(1) - static_cast<TI>(MaxExponentField<T>() >> 1));
    const VFromD<decltype(di)> e_sub = Sub(kSubBase, BitCast(di, shift));
    const VFromD<decltype(di)> e_norm =
        Sub(biased, Set(di, static_cast<TI>(MaxExponentField<T>() >> 1)));
    m = IfThenElse(RebindMask(du, sub), m_sub, Or(frac, kImplicit));
    e = IfThenElse(sub, e_sub, e_norm);
  };

  VFromD<decltype(du)> mx, my;
  VFromD<decltype(di)> ex, ey;
  split(ax, mx, ex);
  split(ay, my, ey);

  // Reduce: for each step while ex > ey, mx = (mx >= my ? mx - my : mx) << 1.
  const VFromD<decltype(di)> one_i = Set(di, TI{1});
  auto active = Gt(ex, ey);
  // Bound the loop: exponent gap never exceeds the full dynamic range.
  constexpr int kMaxSteps = (int{1} << ExponentBits<T>()) + kMantBits + 2;
  for (int step = 0; step < kMaxSteps; ++step) {
    if (AllFalse(di, active)) break;
    const MFromD<decltype(du)> ge = Ge(mx, my);
    const MFromD<decltype(du)> act_u = RebindMask(du, active);
    mx = IfThenElse(And(act_u, ge), Sub(mx, my), mx);
    mx = IfThenElse(act_u, Add(mx, mx), mx);  // << 1
    ex = IfThenElse(active, Sub(ex, one_i), ex);
    active = Gt(ex, ey);
  }
  // One final subtract at ex == ey.
  {
    const MFromD<decltype(du)> ge = Ge(mx, my);
    mx = IfThenElse(ge, Sub(mx, my), mx);
  }

  // Renormalize the remainder: shift mx up until its bit kMantBits is set,
  // decreasing ex; mx == 0 means an exact multiple, remainder +/-0.
  const MFromD<decltype(du)> is_zero_rem = Eq(mx, Zero(du));
  for (int step = 0; step < kMantBits + 1; ++step) {
    const MFromD<decltype(du)> need =
        AndNot(is_zero_rem, Eq(ShiftRight<kMantBits>(mx), Zero(du)));
    if (AllFalse(du, need)) break;
    mx = IfThenElse(need, Add(mx, mx), mx);
    ex = IfThenElse(RebindMask(di, need), Sub(ex, one_i), ex);
  }

  // Rebuild the float: value = mx * 2^(ex - kMantBits). If ex is a valid
  // normal exponent, pack it; otherwise (subnormal result) shift mx down.
  const VFromD<decltype(di)> kBias =
      Set(di, static_cast<TI>(MaxExponentField<T>() >> 1));
  const VFromD<decltype(du)> biased_ex =
      BitCast(du, Add(ex, kBias));  // exponent field if normal
  const VFromD<decltype(du)> packed_normal =
      Or(And(mx, kMantMask), Shl(biased_ex, Set(du, TU{kMantBits})));
  // Subnormal result: ex + kBias <= 0, so shift right by (1 - (ex+bias)).
  const VFromD<decltype(di)> sub_shift = Sub(one_i, Add(ex, kBias));
  const VFromD<decltype(du)> packed_sub = Shr(mx, BitCast(du, sub_shift));
  const MFromD<decltype(di)> ex_normal = Gt(Add(ex, kBias), Zero(di));
  VFromD<decltype(du)> out_bits =
      IfThenElse(RebindMask(du, ex_normal), packed_normal, packed_sub);
  out_bits = IfThenElse(is_zero_rem, Zero(du), out_bits);
  VFromD<D> result = CopySign(BitCast(d, out_bits), x);

  // |x| <= |y| shortcuts (must come before the specials so NaN still wins).
  result = IfThenElse(Lt(ax, ay), x, result);
  result = IfThenElse(Eq(ax, ay), CopySign(Zero(d), x), result);
  // Specials.
  result = IfThenElse(IsInf(y), x, result);  // finite x mod +/-inf == x
  const MFromD<D> nan_case =
      Or(Or(IsNaN(x), IsNaN(y)), Or(IsInf(x), Eq(ay, Zero(d))));
  result = IfThenElse(nan_case, NaN(d), result);
  return result;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
