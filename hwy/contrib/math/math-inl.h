// Copyright 2020 Google LLC
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

// TODO(rhettstucki): Make compatible with SVE vector types.

// Include guard (still compiled once per target)
#if defined(HIGHWAY_HWY_CONTRIB_MATH_MATH_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_MATH_MATH_INL_H_
#undef HIGHWAY_HWY_CONTRIB_MATH_MATH_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_MATH_MATH_INL_H_
#endif

#include "hwy/highway.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/math/math-inl.h"
#include "hwy/foreach_target.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// Computes e^x for each lane. Highway version of std::exp(x).
//
// Valid Types: F32xN and F64xN
//   Max Error: ULP = 1
// Valid Range: float32[-FLT_MAX, +104], float64[-DBL_MAX, +706]
HWY_NOINLINE F32xN Exp(F32xN x);
HWY_NOINLINE F64xN Exp(F64xN x);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////
namespace impl {

// Half Types
using I32xH = Vec<Simd<int32_t, (sizeof(F64xN) / sizeof(double))>>;

template <class V>
struct TagTypeFor {
  using Lane = LaneType<V>;
  using Type = Simd<Lane, sizeof(V) / sizeof(Lane)>;
};
template <>
struct TagTypeFor<I32xH> {
  using Type = Simd<int32_t, sizeof(I64xN) / sizeof(int64_t)>;
};

template <class V>
HWY_INLINE typename TagTypeFor<V>::Type Tag() {
  return typename TagTypeFor<V>::Type();
}

// ConvertToAny
template <class Out, class In>
HWY_INLINE Out ConvertToAny(In x) {
  return ConvertTo(Tag<Out>(), x);
}
template <>
HWY_INLINE I32xH ConvertToAny<I32xH, F64xN>(F64xN x) {
  return DemoteTo(Tag<I32xH>(), x);
}
template <>
HWY_INLINE F64xN ConvertToAny<F64xN, I32xH>(I32xH x) {
  return PromoteTo(Tag<F64xN>(), x);
}
template <>
HWY_INLINE I64xN ConvertToAny<I64xN, I32xH>(I32xH x) {
  return PromoteTo(Tag<I64xN>(), x);
}

// Make
template <class V>
HWY_INLINE V Make(LaneType<V> x) {
  return Set(Tag<V>(), x);
}

// Estrin's Scheme is a faster method for evaluating large polynomials on
// super scalar architectures. It works by factoring the Horner's Method
// polynomial into power of two sub-trees that can be evaluated in parallel.
// Wikipedia Link: https://en.wikipedia.org/wiki/Estrin%27s_scheme
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1) {
  return MulAdd(c1, x, c0);
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2) {
  T x2(x * x);
  return MulAdd(x2, c2, MulAdd(c1, x, c0));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3) {
  T x2(x * x);
  return MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4) {
  T x2(x * x), x4(x2 * x2);
  return MulAdd(x4, c4, MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5) {
  T x2(x * x), x4(x2 * x2);
  return MulAdd(x4, MulAdd(c5, x, c4),
                MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6) {
  T x2(x * x), x4(x2 * x2);
  return MulAdd(x4, MulAdd(x2, c6, MulAdd(c5, x, c4)),
                MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7) {
  T x2(x * x), x4(x2 * x2);
  return MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8) {
  T x2(x * x), x4(x2 * x2), x8(x4 * x4);
  return MulAdd(x8, c8,
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9) {
  T x2(x * x), x4(x2 * x2), x8(x4 * x4);
  return MulAdd(x8, MulAdd(c9, x, c8),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10) {
  T x2(x * x), x4(x2 * x2), x8(x4 * x4);
  return MulAdd(x8, MulAdd(x2, c10, MulAdd(c9, x, c8)),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11) {
  T x2(x * x), x4(x2 * x2), x8(x4 * x4);
  return MulAdd(x8, MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8)),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12) {
  T x2(x * x), x4(x2 * x2), x8(x4 * x4);
  return MulAdd(
      x8, MulAdd(x4, c12, MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
      MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
             MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13) {
  T x2(x * x), x4(x2 * x2), x8(x4 * x4);
  return MulAdd(x8,
                MulAdd(x4, MulAdd(c13, x, c12),
                       MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13, T c14) {
  T x2(x * x), x4(x2 * x2), x8(x4 * x4);
  return MulAdd(x8,
                MulAdd(x4, MulAdd(x2, c14, MulAdd(c13, x, c12)),
                       MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13, T c14, T c15) {
  T x2(x * x), x4(x2 * x2), x8(x4 * x4);
  return MulAdd(x8,
                MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                       MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13, T c14, T c15, T c16) {
  T x2(x * x), x4(x2 * x2), x8(x4 * x4), x16(x8 * x8);
  return MulAdd(
      x16, c16,
      MulAdd(x8,
             MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                    MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
             MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                    MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13, T c14, T c15, T c16, T c17) {
  T x2(x * x), x4(x2 * x2), x8(x4 * x4), x16(x8 * x8);
  return MulAdd(
      x16, MulAdd(c17, x, c16),
      MulAdd(x8,
             MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                    MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
             MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                    MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13, T c14, T c15, T c16, T c17,
                                     T c18) {
  T x2(x * x), x4(x2 * x2), x8(x4 * x4), x16(x8 * x8);
  return MulAdd(
      x16, MulAdd(x2, c18, MulAdd(c17, x, c16)),
      MulAdd(x8,
             MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                    MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
             MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                    MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)))));
}

HWY_INLINE HWY_MAYBE_UNUSED F32xN ExpPoly(F32xN x) {
  const F32xN k0 = Make<F32xN>(+0.5f);
  const F32xN k1 = Make<F32xN>(+0.166666671633720397949219f);
  const F32xN k2 = Make<F32xN>(+0.0416664853692054748535156f);
  const F32xN k3 = Make<F32xN>(+0.00833336077630519866943359f);
  const F32xN k4 = Make<F32xN>(+0.00139304355252534151077271f);
  const F32xN k5 = Make<F32xN>(+0.000198527617612853646278381f);

  return MulAdd(Estrin(x, k0, k1, k2, k3, k4, k5), (x * x), x);
}

HWY_INLINE HWY_MAYBE_UNUSED F64xN ExpPoly(F64xN x) {
  const F64xN k0 = Make<F64xN>(+0.5);
  const F64xN k1 = Make<F64xN>(+0.166666666666666851703837);
  const F64xN k2 = Make<F64xN>(+0.0416666666666665047591422);
  const F64xN k3 = Make<F64xN>(+0.00833333333331652721664984);
  const F64xN k4 = Make<F64xN>(+0.00138888888889774492207962);
  const F64xN k5 = Make<F64xN>(+0.000198412698960509205564975);
  const F64xN k6 = Make<F64xN>(+2.4801587159235472998791e-5);
  const F64xN k7 = Make<F64xN>(+2.75572362911928827629423e-6);
  const F64xN k8 = Make<F64xN>(+2.75573911234900471893338e-7);
  const F64xN k9 = Make<F64xN>(+2.51112930892876518610661e-8);
  const F64xN k10 = Make<F64xN>(+2.08860621107283687536341e-9);

  return MulAdd(Estrin(x, k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10), (x * x),
                x);
}

// Computes 2^x, where x is an integer.
HWY_INLINE HWY_MAYBE_UNUSED F32xN F32_Pow2I(I32xN x) {
  const I32xN kOffset = Make<I32xN>(0x7F);
  return BitCast(Tag<F32xN>(), ShiftLeft<23>(x + kOffset));
}
HWY_INLINE HWY_MAYBE_UNUSED F64xN F64_Pow2I(I32xH x) {
  const I32xH kOffset = Make<I32xH>(0x3FF);
  return BitCast(Tag<F64xN>(), ShiftLeft<52>(ConvertToAny<I64xN>(x + kOffset)));
}

// Sets the exponent of 'x' to 2^e.
HWY_INLINE HWY_MAYBE_UNUSED F32xN LoadExpShortRange(F32xN x, I32xN e) {
  const I32xN y = ShiftRight<1>(e);
  return (x * F32_Pow2I(y) * F32_Pow2I(e - y));
}
HWY_INLINE HWY_MAYBE_UNUSED F64xN LoadExpShortRange(F64xN x, I32xH e) {
  const I32xH y = ShiftRight<1>(e);
  return (x * F64_Pow2I(y) * F64_Pow2I(e - y));
}

HWY_INLINE HWY_MAYBE_UNUSED F32xN ExpReduce(F32xN x, I32xN q) {
  // kLn2Part0f + kLn2Part1f ~= -ln(2)
  const F32xN kLn2Part0f = Make<F32xN>(-0.693145751953125f);
  const F32xN kLn2Part1f = Make<F32xN>(-1.428606765330187045e-6f);

  // Extended precision modular arithmetic.
  const F32xN qf = ConvertToAny<F32xN>(q);
  x = MulAdd(qf, kLn2Part0f, x);
  x = MulAdd(qf, kLn2Part1f, x);
  return x;
}
HWY_INLINE HWY_MAYBE_UNUSED F64xN ExpReduce(F64xN x, I32xH q) {
  // kLn2Part0d + kLn2Part1d ~= -ln(2)
  const F64xN kLn2Part0d = Make<F64xN>(-0.6931471805596629565116018);
  const F64xN kLn2Part1d = Make<F64xN>(-0.28235290563031577122588448175e-12);

  // Extended precision modular arithmetic.
  const F64xN qf = ConvertToAny<F64xN>(q);
  x = MulAdd(qf, kLn2Part0d, x);
  x = MulAdd(qf, kLn2Part1d, x);
  return x;
}

template <class V>
HWY_INLINE V Exp(V x) {
  constexpr size_t N = (sizeof(V) / sizeof(LaneType<V>));
  using I32Type = Vec<Simd<int32_t, N>>;

  // clang-format off
  const V kLowerBound  = Make<V>(-104.0);
  const V kHalf        = Make<V>(+0.5);
  const V kNegZero     = Make<V>(-0.0);
  const V kOne         = Make<V>(+1.0);
  const V kOneOverLog2 = Make<V>(+1.442695040888963407359924681);
  // clang-format on

  // q = static_cast<int32>((x / log(2)) + ((x < 0) ? -0.5 : +0.5))
  const I32Type q = ConvertToAny<I32Type>(
      MulAdd(x, kOneOverLog2, Or(kHalf, And(x, kNegZero))));

  // Reduce, approximate, and then reconstruct.
  const V y = LoadExpShortRange((ExpPoly(ExpReduce(x, q)) + kOne), q);
  return IfThenElseZero(x >= kLowerBound, y);
}

}  // namespace impl

HWY_NOINLINE F32xN Exp(F32xN x) { return impl::Exp(x); }  // NOLINT
HWY_NOINLINE F64xN Exp(F64xN x) { return impl::Exp(x); }  // NOLINT

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_MATH_MATH_INL_H_
