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

// Include guard (still compiled once per target)
#if defined(HIGHWAY_CONTRIB_MATH_MATH_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_CONTRIB_MATH_MATH_INL_H_
#undef HIGHWAY_CONTRIB_MATH_MATH_INL_H_
#else
#define HIGHWAY_CONTRIB_MATH_MATH_INL_H_
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

/**
 * Highway SIMD version of std::acosh(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 3
 *      Valid Range: float32[1, +FLT_MAX], float64[1, +DBL_MAX]
 * @return hyperbolic arc cosine of 'x'
 */
template <class D, class V>
HWY_NOINLINE V Acosh(const D d, V x);

/**
 * Highway SIMD version of std::asinh(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 3
 *      Valid Range: float32[-FLT_MAX, +FLT_MAX], float64[-DBL_MAX, +DBL_MAX]
 * @return hyperbolic arc sine of 'x'
 */
template <class D, class V>
HWY_NOINLINE V Asinh(const D d, V x);

/**
 * Highway SIMD version of std::atanh(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 3
 *      Valid Range: (-1, +1)
 * @return hyperbolic arc tangent of 'x'
 */
template <class D, class V>
HWY_NOINLINE V Atanh(const D d, V x);

/**
 * Highway SIMD version of std::exp(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 1
 *      Valid Range: float32[-FLT_MAX, +104], float64[-DBL_MAX, +706]
 * @return e^x
 */
template <class D, class V>
HWY_NOINLINE V Exp(const D d, V x);

/**
 * Highway SIMD version of std::expm1(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 4
 *      Valid Range: float32[-FLT_MAX, +104], float64[-DBL_MAX, +706]
 * @return e^x - 1
 */
template <class D, class V>
HWY_NOINLINE V Expm1(const D d, V x);

/**
 * Highway SIMD version of std::log(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 4
 *      Valid Range: float32(0, +FLT_MAX], float64(0, +DBL_MAX]
 * @return natural logarithm of 'x'
 */
template <class D, class V>
HWY_NOINLINE V Log(const D d, V x);

/**
 * Highway SIMD version of std::log10(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 2
 *      Valid Range: float32(0, +FLT_MAX], float64(0, +DBL_MAX]
 * @return base 10 logarithm of 'x'
 */
template <class D, class V>
HWY_NOINLINE V Log10(const D d, V x);

/**
 * Highway SIMD version of std::log1p(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 2
 *      Valid Range: float32[0, +FLT_MAX], float64[0, +DBL_MAX]
 * @return log(1 + x)
 */
template <class D, class V>
HWY_NOINLINE V Log1p(const D d, V x);

/**
 * Highway SIMD version of std::log2(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 2
 *      Valid Range: float32(0, +FLT_MAX], float64(0, +DBL_MAX]
 * @return base 2 logarithm of 'x'
 */
template <class D, class V>
HWY_NOINLINE V Log2(const D d, V x);

/**
 * Highway SIMD version of std::sinh(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 4
 *      Valid Range: float32[-88.7228, +88.7228], float64[-709, +709]
 * @return hyperbolic sine of 'x'
 */
template <class D, class V>
HWY_NOINLINE V Sinh(const D d, V x);

/**
 * Highway SIMD version of std::tanh(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 4
 *      Valid Range: float32[-FLT_MAX, +FLT_MAX], float64[-DBL_MAX, +DBL_MAX]
 * @return hyperbolic tangent of 'x'
 */
template <class D, class V>
HWY_NOINLINE V Tanh(const D d, V x);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////
namespace impl {

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

template <typename FloatOrDouble>
struct ExpImpl {};
template <typename FloatOrDouble>
struct LogImpl {};

template <>
struct ExpImpl<float> {
  // Rounds float toward zero and returns as int32_t.
  template <class D, class V>
  HWY_INLINE Vec<Rebind<int32_t, D>> ToInt32(D /*unused*/, V x) {
    return ConvertTo(Rebind<int32_t, D>(), x);
  }

  template <class D, class V>
  HWY_INLINE V ExpPoly(D d, V x) {
    const auto k0 = Set(d, +0.5f);
    const auto k1 = Set(d, +0.166666671633720397949219f);
    const auto k2 = Set(d, +0.0416664853692054748535156f);
    const auto k3 = Set(d, +0.00833336077630519866943359f);
    const auto k4 = Set(d, +0.00139304355252534151077271f);
    const auto k5 = Set(d, +0.000198527617612853646278381f);

    return MulAdd(Estrin(x, k0, k1, k2, k3, k4, k5), (x * x), x);
  }

  // Computes 2^x, where x is an integer.
  template <class D, class VI32>
  HWY_INLINE Vec<D> Pow2I(D d, VI32 x) {
    const Rebind<int32_t, D> di32;
    const VI32 kOffset = Set(di32, 0x7F);
    return BitCast(d, ShiftLeft<23>(x + kOffset));
  }

  // Sets the exponent of 'x' to 2^e.
  template <class D, class V, class VI32>
  HWY_INLINE V LoadExpShortRange(D d, V x, VI32 e) {
    const VI32 y = ShiftRight<1>(e);
    return x * Pow2I(d, y) * Pow2I(d, e - y);
  }

  template <class D, class V, class VI32>
  HWY_INLINE V ExpReduce(D d, V x, VI32 q) {
    // kLn2Part0f + kLn2Part1f ~= -ln(2)
    const V kLn2Part0f = Set(d, -0.693145751953125f);
    const V kLn2Part1f = Set(d, -1.428606765330187045e-6f);

    // Extended precision modular arithmetic.
    const V qf = ConvertTo(d, q);
    x = MulAdd(qf, kLn2Part0f, x);
    x = MulAdd(qf, kLn2Part1f, x);
    return x;
  }
};

template <>
struct LogImpl<float> {
  template <class D, class V>
  HWY_INLINE Vec<Rebind<int32_t, D>> Log2p1NoSubnormal(D d, V x) {
    const Rebind<int32_t, D> di32;
    const Rebind<uint32_t, D> du32;
    return BitCast(di32, ShiftRight<23>(BitCast(du32, x))) - Set(di32, 0x7F);
  }

  // Approximates Log(x) over the range [sqrt(2) / 2, sqrt(2)].
  template <class D, class V>
  HWY_INLINE V LogPoly(D d, V x) {
    const V k0 = Set(d, 0.66666662693f);
    const V k1 = Set(d, 0.40000972152f);
    const V k2 = Set(d, 0.28498786688f);
    const V k3 = Set(d, 0.24279078841f);

    const V x2 = (x * x);
    const V x4 = (x2 * x2);
    return MulAdd(MulAdd(k2, x4, k0), x2, (MulAdd(k3, x4, k1) * x4));
  }
};

#if HWY_CAP_FLOAT64 && HWY_CAP_INTEGER64
template <>
struct ExpImpl<double> {
  // Rounds double toward zero and returns as int32_t.
  template <class D, class V>
  HWY_INLINE Vec<Rebind<int32_t, D>> ToInt32(D /*unused*/, V x) {
    return DemoteTo(Rebind<int32_t, D>(), x);
  }

  template <class D, class V>
  HWY_INLINE V ExpPoly(D d, V x) {
    const auto k0 = Set(d, +0.5);
    const auto k1 = Set(d, +0.166666666666666851703837);
    const auto k2 = Set(d, +0.0416666666666665047591422);
    const auto k3 = Set(d, +0.00833333333331652721664984);
    const auto k4 = Set(d, +0.00138888888889774492207962);
    const auto k5 = Set(d, +0.000198412698960509205564975);
    const auto k6 = Set(d, +2.4801587159235472998791e-5);
    const auto k7 = Set(d, +2.75572362911928827629423e-6);
    const auto k8 = Set(d, +2.75573911234900471893338e-7);
    const auto k9 = Set(d, +2.51112930892876518610661e-8);
    const auto k10 = Set(d, +2.08860621107283687536341e-9);

    return MulAdd(Estrin(x, k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10),
                  (x * x), x);
  }

  // Computes 2^x, where x is an integer.
  template <class D, class VI32>
  HWY_INLINE Vec<D> Pow2I(D d, VI32 x) {
    const Rebind<int32_t, D> di32;
    const Rebind<int64_t, D> di64;
    const VI32 kOffset = Set(di32, 0x3FF);
    return BitCast(d, ShiftLeft<52>(PromoteTo(di64, x + kOffset)));
  }

  // Sets the exponent of 'x' to 2^e.
  template <class D, class V, class VI32>
  HWY_INLINE V LoadExpShortRange(D d, V x, VI32 e) {
    const VI32 y = ShiftRight<1>(e);
    return (x * Pow2I(d, y) * Pow2I(d, e - y));
  }

  template <class D, class V, class VI32>
  HWY_INLINE V ExpReduce(D d, V x, VI32 q) {
    // kLn2Part0d + kLn2Part1d ~= -ln(2)
    const V kLn2Part0d = Set(d, -0.6931471805596629565116018);
    const V kLn2Part1d = Set(d, -0.28235290563031577122588448175e-12);

    // Extended precision modular arithmetic.
    const V qf = PromoteTo(d, q);
    x = MulAdd(qf, kLn2Part0d, x);
    x = MulAdd(qf, kLn2Part1d, x);
    return x;
  }
};

template <>
struct LogImpl<double> {
  template <class D, class V>
  HWY_INLINE Vec<Rebind<int64_t, D>> Log2p1NoSubnormal(D d, V x) {
    const Rebind<int64_t, D> di64;
    const Rebind<uint64_t, D> du64;
    return BitCast(di64, ShiftRight<52>(BitCast(du64, x))) - Set(di64, 0x3FF);
  }

  // Approximates Log(x) over the range [sqrt(2) / 2, sqrt(2)].
  template <class D, class V>
  HWY_INLINE V LogPoly(D d, V x) {
    const V k0 = Set(d, 0.6666666666666735130);
    const V k1 = Set(d, 0.3999999999940941908);
    const V k2 = Set(d, 0.2857142874366239149);
    const V k3 = Set(d, 0.2222219843214978396);
    const V k4 = Set(d, 0.1818357216161805012);
    const V k5 = Set(d, 0.1531383769920937332);
    const V k6 = Set(d, 0.1479819860511658591);

    const V x2 = (x * x);
    const V x4 = (x2 * x2);
    return MulAdd(MulAdd(MulAdd(MulAdd(k6, x4, k4), x4, k2), x4, k0), x2,
                  (MulAdd(MulAdd(k5, x4, k3), x4, k1) * x4));
  }
};

#endif

template <class D, class V, bool kAllowSubnormals = true>
HWY_INLINE V Log(const D d, V x) {
  // http://git.musl-libc.org/cgit/musl/tree/src/math/log.c for more info.
  using LaneType = LaneType<V>;
  impl::LogImpl<LaneType> impl;

  // clang-format off
  constexpr bool kIsF32 = (sizeof(LaneType) == 4);

  // Float Constants
  const V kLn2Hi     = Set(d, (kIsF32 ? 0.69313812256f   :
                                        0.693147180369123816490   ));
  const V kLn2Lo     = Set(d, (kIsF32 ? 9.0580006145e-6f :
                                        1.90821492927058770002e-10));
  const V kOne       = Set(d, +1.0);
  const V kMinNormal = Set(d, (kIsF32 ? 1.175494351e-38f :
                                        2.2250738585072014e-308   ));
  const V kScale     = Set(d, (kIsF32 ? 3.355443200e+7f  :
                                        1.8014398509481984e+16    ));

  // Integer Constants
  const Rebind<MakeSigned<LaneType>, D> di;
  using VI = decltype(Zero(di));
  const VI kLowerBits = Set(di, (kIsF32 ? 0x00000000L : 0xFFFFFFFFLL));
  const VI kMagic     = Set(di, (kIsF32 ? 0x3F3504F3L : 0x3FE6A09E00000000LL));
  const VI kExpMask   = Set(di, (kIsF32 ? 0x3F800000L : 0x3FF0000000000000LL));
  const VI kExpScale  = Set(di, (kIsF32 ? -25         : -54));
  const VI kManMask   = Set(di, (kIsF32 ? 0x7FFFFFL   : 0xFFFFF00000000LL));
  // clang-format on

  // Scale up 'x' so that it is no longer denormalized.
  VI exp_bits;
  V exp;
  if (kAllowSubnormals == true) {
    const auto is_denormal = (x < kMinNormal);
    x = IfThenElse(is_denormal, (x * kScale), x);

    // Compute the new exponent.
    exp_bits = (BitCast(di, x) + (kExpMask - kMagic));
    const VI exp_scale =
        BitCast(di, IfThenElseZero(is_denormal, BitCast(d, kExpScale)));
    exp = ConvertTo(
        d, exp_scale + impl.Log2p1NoSubnormal(d, BitCast(d, exp_bits)));
  } else {
    // Compute the new exponent.
    exp_bits = (BitCast(di, x) + (kExpMask - kMagic));
    exp = ConvertTo(d, impl.Log2p1NoSubnormal(d, BitCast(d, exp_bits)));
  }

  // Renormalize.
  const V y = Or(And(x, BitCast(d, kLowerBits)),
                 BitCast(d, ((exp_bits & kManMask) + kMagic)));

  // Approximate and reconstruct.
  const V ym1 = (y - kOne);
  const V z = (ym1 / (y + kOne));

  return MulSub(exp, kLn2Hi,
                (MulSub(z, (ym1 - impl.LogPoly(d, z)), (exp * kLn2Lo)) - ym1));
}

}  // namespace impl

template <class D, class V>
HWY_NOINLINE V Acosh(const D d, V x) {
  const V kLarge = Set(d, 268435456.0);
  const V kLog2 = Set(d, 0.693147180559945286227);
  const V kOne = Set(d, +1.0);
  const V kTwo = Set(d, +2.0);

  const auto is_x_large = (x > kLarge);
  const auto is_x_gt_2 = (x > kTwo);

  const V x_minus_1 = (x - kOne);
  const V y0 = MulSub(kTwo, x, (kOne / (Sqrt(MulSub(x, x, kOne)) + x)));
  const V y1 =
      (Sqrt(MulAdd(x_minus_1, kTwo, (x_minus_1 * x_minus_1))) + x_minus_1);
  const V y2 =
      IfThenElse(is_x_gt_2, IfThenElse(is_x_large, x, y0), (y1 + kOne));
  const V z = impl::Log<D, V, /*kAllowSubnormals=*/false>(d, y2);

  return IfThenElse(is_x_gt_2, z,
                    IfThenElse(y2 == kOne, y1, (z * y1 / (y2 - kOne)))) +
         IfThenElseZero(is_x_large, kLog2);
}

template <class D, class V>
HWY_NOINLINE V Asinh(const D d, V x) {
  const V kSmall = Set(d, 1.0 / 268435456.0);
  const V kLarge = Set(d, 268435456.0);
  const V kLog2 = Set(d, 0.693147180559945286227);
  const V kOne = Set(d, +1.0);
  const V kTwo = Set(d, +2.0);

  const V sign_x = And(SignBit(d), x);  // Extract the sign bit
  const V abs_x = Xor(x, sign_x);

  const auto is_x_large = (abs_x > kLarge);
  const auto is_x_lt_2 = (abs_x < kTwo);

  const V x2 = (x * x);
  const V sqrt_x2_plus_1 = Sqrt(x2 + kOne);

  const V y0 = MulAdd(abs_x, kTwo, (kOne / (sqrt_x2_plus_1 + abs_x)));
  const V y1 = ((x2 / (sqrt_x2_plus_1 + kOne)) + abs_x);
  const V y2 =
      IfThenElse(is_x_lt_2, (y1 + kOne), IfThenElse(is_x_large, abs_x, y0));
  const V z = impl::Log<D, V, /*kAllowSubnormals=*/false>(d, y2);

  const V y = IfThenElse((abs_x < kSmall), x,
                         IfThenElse(y2 == kOne, y1, (z * y1 / (y2 - kOne))));
  return Or((IfThenElse(is_x_lt_2, y, z) + IfThenElseZero(is_x_large, kLog2)),
            sign_x);
}

template <class D, class V>
HWY_NOINLINE V Atanh(const D d, V x) {
  const V kHalf = Set(d, +0.5);
  const V kOne = Set(d, +1.0);

  const V sign = And(SignBit(d), x);  // Extract the sign bit
  const V abs_x = Xor(x, sign);
  return Log1p(d, ((abs_x + abs_x) / (kOne - abs_x))) * Xor(kHalf, sign);
}

template <class D, class V>
HWY_NOINLINE V Exp(const D d, V x) {
  using LaneType = LaneType<V>;

  // clang-format off
  const V kHalf        = Set(d, +0.5);
  const V kLowerBound  = Set(d, (sizeof(LaneType) == 4 ? -104.0 : -1000.0));
  const V kNegZero     = Set(d, -0.0);
  const V kOne         = Set(d, +1.0);
  const V kOneOverLog2 = Set(d, +1.442695040888963407359924681);
  // clang-format on

  impl::ExpImpl<LaneType> impl;

  // q = static_cast<int32>((x / log(2)) + ((x < 0) ? -0.5 : +0.5))
  const auto q =
      impl.ToInt32(d, MulAdd(x, kOneOverLog2, Or(kHalf, And(x, kNegZero))));

  // Reduce, approximate, and then reconstruct.
  const V y = impl.LoadExpShortRange(
      d, (impl.ExpPoly(d, impl.ExpReduce(d, x, q)) + kOne), q);
  return IfThenElseZero(x >= kLowerBound, y);
}

template <class D, class V>
HWY_NOINLINE V Expm1(const D d, V x) {
  using LaneType = LaneType<V>;

  // clang-format off
  const V kHalf        = Set(d, +0.5);
  const V kLowerBound  = Set(d, (sizeof(LaneType) == 4 ? -104.0 : -1000.0));
  const V kLn2Over2    = Set(d, +0.346573590279972654708616);
  const V kNegOne      = Set(d, -1.0);
  const V kNegZero     = Set(d, -0.0);
  const V kOne         = Set(d, +1.0);
  const V kOneOverLog2 = Set(d, +1.442695040888963407359924681);
  // clang-format on

  impl::ExpImpl<LaneType> impl;

  // q = static_cast<int32>((x / log(2)) + ((x < 0) ? -0.5 : +0.5))
  const auto q =
      impl.ToInt32(d, MulAdd(x, kOneOverLog2, Or(kHalf, And(x, kNegZero))));

  // Reduce, approximate, and then reconstruct.
  const V y = impl.ExpPoly(d, impl.ExpReduce(d, x, q));
  const V z = IfThenElse(Abs(x) < kLn2Over2, y,
                         impl.LoadExpShortRange(d, (y + kOne), q) - kOne);
  return IfThenElse(x < kLowerBound, kNegOne, z);
}

template <class D, class V>
HWY_NOINLINE V Log(const D d, V x) {
  return impl::Log<D, V, /*kAllowSubnormals=*/true>(d, x);
}

template <class D, class V>
HWY_NOINLINE V Log10(const D d, V x) {
  return Log(d, x) * Set(d, 0.4342944819032518276511);
}

template <class D, class V>
HWY_NOINLINE V Log1p(const D d, V x) {
  const V kOne = Set(d, +1.0);

  const V y = (x + kOne);
  return IfThenElse(
      (y == kOne), x,
      (impl::Log<D, V, /*kAllowSubnormals=*/false>(d, y) * (x / (y - kOne))));
}

template <class D, class V>
HWY_NOINLINE V Log2(const D d, V x) {
  return Log(d, x) * Set(d, 1.44269504088896340735992);
}

template <class D, class V>
HWY_NOINLINE V Sinh(const D d, V x) {
  const V kHalf = Set(d, +0.5);
  const V kOne = Set(d, +1.0);
  const V kTwo = Set(d, +2.0);

  const V sign = And(SignBit(d), x);  // Extract the sign bit
  const V abs_x = Xor(x, sign);
  const V y = Expm1(d, abs_x);
  const V z = ((y + kTwo) / (y + kOne) * (y * kHalf));
  return Xor(z, sign);  // Reapply the sign bit
}

template <class D, class V>
HWY_NOINLINE V Tanh(const D d, V x) {
  const V kLimit = Set(d, 18.714973875);
  const V kOne = Set(d, +1.0);
  const V kTwo = Set(d, +2.0);

  const V sign = And(SignBit(d), x);  // Extract the sign bit
  const V abs_x = Xor(x, sign);
  const V y = Expm1(d, abs_x * kTwo);
  const V z = IfThenElse((abs_x > kLimit), kOne, (y / (y + kTwo)));
  return Xor(z, sign);  // Reapply the sign bit
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_CONTRIB_MATH_MATH_INL_H_
