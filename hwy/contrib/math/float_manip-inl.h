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
// than compute a transcendental: ilogb, logb, modf, nextafter. These are exact
// (0 ULP) and operate on the IEEE-754 representation, so they need no
// polynomial and are defined for float32 and float64. Part of the
// "fmod, ilogb, logb, modf, nextafter, scalbn" op_wishlist item (scalbn is
// MulByPow2; fmod is left for a follow-up).

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

// std::ldexp / std::scalbn (x * 2^exp with a signed-integer exponent) are
// already covered by MulByPow2 in the core ops.

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
  // 0 and NaN both map to the minimum lane value.
  e = IfThenElse(RebindMask(di, Or(Eq(Abs(x), Zero(d)), IsNaN(x))),
                 Set(di, LimitsMin<TI>()), e);
  e = IfThenElse(RebindMask(di, IsInf(x)), Set(di, LimitsMax<TI>()), e);
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
  result = IfThenElse(IsEitherNaN(from, to), NaN(d), result);
  return result;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // NOLINT
