// Copyright 2020 Google LLC
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
#if defined(HIGHWAY_HWY_CONTRIB_MATH_MATH_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_MATH_MATH_INL_H_
#undef HIGHWAY_HWY_CONTRIB_MATH_MATH_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_MATH_MATH_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>

#include "hwy/contrib/math/fp_arith-inl.h"
#include "hwy/highway.h"

// Disable FMA contraction so each mul/add in the error-free transforms 
// rounds on its own.
#if HWY_COMPILER_GCC && !HWY_COMPILER_CLANG
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

/**
 * Highway SIMD version of std::acos(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 2
 *      Valid Range: [-1, +1]
 * @return arc cosine of 'x'
 */
template <class D, class V>
HWY_INLINE V Acos(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallAcos(const D d, VecArg<V> x) {
  return Acos(d, x);
}

/**
 * Highway SIMD version of std::acosh(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 3
 *      Valid Range: float32[1, +FLT_MAX], float64[1, +DBL_MAX]
 * @return hyperbolic arc cosine of 'x'
 */
template <class D, class V>
HWY_INLINE V Acosh(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallAcosh(const D d, VecArg<V> x) {
  return Acosh(d, x);
}

/**
 * Highway SIMD version of std::asin(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 2
 *      Valid Range: [-1, +1]
 * @return arc sine of 'x'
 */
template <class D, class V>
HWY_INLINE V Asin(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallAsin(const D d, VecArg<V> x) {
  return Asin(d, x);
}

/**
 * Highway SIMD version of std::asinh(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 3
 *      Valid Range: float32[-FLT_MAX, +FLT_MAX], float64[-DBL_MAX, +DBL_MAX]
 * @return hyperbolic arc sine of 'x'
 */
template <class D, class V>
HWY_INLINE V Asinh(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallAsinh(const D d, VecArg<V> x) {
  return Asinh(d, x);
}

/**
 * Highway SIMD version of std::atan(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 3
 *      Valid Range: float32[-FLT_MAX, +FLT_MAX], float64[-DBL_MAX, +DBL_MAX]
 * @return arc tangent of 'x'
 */
template <class D, class V>
HWY_INLINE V Atan(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallAtan(const D d, VecArg<V> x) {
  return Atan(d, x);
}

/**
 * Highway SIMD version of std::atanh(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 3
 *      Valid Range: (-1, +1)
 * @return hyperbolic arc tangent of 'x'
 */
template <class D, class V>
HWY_INLINE V Atanh(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallAtanh(const D d, VecArg<V> x) {
  return Atanh(d, x);
}

// Atan2 was added later and some users may be implementing it themselves, so
// notify them that this version of Highway defines it already.
#ifndef HWY_HAVE_ATAN2
#define HWY_HAVE_ATAN2 1
#endif

/**
 * Highway SIMD version of std::atan2(x).
 *
 * Valid Lane Types: float32, float64
 * Correctly handles negative zero, infinities, and NaN.
 * @return atan2 of 'y', 'x'
 */
template <class D, class V>
HWY_INLINE V Atan2(D d, V y, V x);
template <class D, class V>
HWY_NOINLINE V CallAtan2(const D d, VecArg<V> y, VecArg<V> x) {
  return Atan2(d, y, x);
}

/**
 * Highway SIMD version of std::cbrt(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 6
 *      Valid Range: float32[-FLT_MAX, +FLT_MAX], float64[-DBL_MAX, +DBL_MAX]
 * @return cube root of 'x'
 */
template <bool kHandleSubnormals = true, class D, class V>
HWY_INLINE V Cbrt(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallCbrt(const D d, VecArg<V> x) {
  return Cbrt<true>(d, x);
}

/**
 * Highway SIMD version of std::cos(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 3
 *      Valid Range: [-39000, +39000]
 * @return cosine of 'x'
 */
template <class D, class V>
HWY_INLINE V Cos(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallCos(const D d, VecArg<V> x) {
  return Cos(d, x);
}

/**
 * Highway SIMD version of std::tan(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = ~300 (float32), 2 (float64)
 *                   Note: On float32, error is ~64 ULP on targets with FMA.
 *                   Without FMA (e.g. SSE4), rounding errors accumulate up to
 * ~300 ULP. Valid Range: [-39000, +39000]
 * @return tangent of 'x'
 */
template <class D, class V>
HWY_INLINE V Tan(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallTan(const D d, VecArg<V> x) {
  return Tan(d, x);
}

/**
 * Highway SIMD version of std::erf(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 4
 *      Valid Range: float32[-FLT_MAX, +FLT_MAX], float64[-DBL_MAX, +DBL_MAX]
 * @return error function of 'x'
 */
template <class D, class V>
HWY_INLINE V Erf(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallErf(const D d, VecArg<V> x) {
  return Erf(d, x);
}

/**
 * Highway SIMD version of std::exp(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 1
 *      Valid Range: float32[-FLT_MAX, +104], float64[-DBL_MAX, +706]
 * @return e^x
 */
template <class D, class V>
HWY_INLINE V Exp(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallExp(const D d, VecArg<V> x) {
  return Exp(d, x);
}

/**
 * Highway SIMD version of std::exp2(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 2
 *      Valid Range: float32[-FLT_MAX, +128], float64[-DBL_MAX, +1024]
 * @return 2^x
 */
template <class D, class V>
HWY_INLINE V Exp2(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallExp2(const D d, VecArg<V> x) {
  return Exp2(d, x);
}

/**
 * Highway SIMD version of std::expm1(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 4
 *      Valid Range: float32[-FLT_MAX, +104], float64[-DBL_MAX, +706]
 * @return e^x - 1
 */
template <class D, class V>
HWY_INLINE V Expm1(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallExpm1(const D d, VecArg<V> x) {
  return Expm1(d, x);
}

/**
 * Highway SIMD version of std::log(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 4
 *      Valid Range: float32(0, +FLT_MAX], float64(0, +DBL_MAX]
 * @return natural logarithm of 'x'
 */
template <class D, class V>
HWY_INLINE V Log(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallLog(const D d, VecArg<V> x) {
  return Log(d, x);
}

/**
 * Highway SIMD version of std::log10(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 2
 *      Valid Range: float32(0, +FLT_MAX], float64(0, +DBL_MAX]
 * @return base 10 logarithm of 'x'
 */
template <class D, class V>
HWY_INLINE V Log10(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallLog10(const D d, VecArg<V> x) {
  return Log10(d, x);
}

/**
 * Highway SIMD version of std::log1p(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 2
 *      Valid Range: float32[0, +FLT_MAX], float64[0, +DBL_MAX]
 * @return log(1 + x)
 */
template <class D, class V>
HWY_INLINE V Log1p(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallLog1p(const D d, VecArg<V> x) {
  return Log1p(d, x);
}

/**
 * Highway SIMD version of std::log2(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 2
 *      Valid Range: float32(0, +FLT_MAX], float64(0, +DBL_MAX]
 * @return base 2 logarithm of 'x'
 */
template <class D, class V>
HWY_INLINE V Log2(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallLog2(const D d, VecArg<V> x) {
  return Log2(d, x);
}

/**
 * Highway SIMD version of std::sin(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 3
 *      Valid Range: [-39000, +39000]
 * @return sine of 'x'
 */
template <class D, class V>
HWY_INLINE V Sin(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallSin(const D d, VecArg<V> x) {
  return Sin(d, x);
}

/**
 * Highway SIMD version of std::sinh(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 4
 *      Valid Range: float32[-88.7228, +88.7228], float64[-709, +709]
 * @return hyperbolic sine of 'x'
 */
template <class D, class V>
HWY_INLINE V Sinh(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallSinh(const D d, VecArg<V> x) {
  return Sinh(d, x);
}

/**
 * Highway SIMD version of std::cosh(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 4
 *      Valid Range: float32[-88.7228, +88.7228], float64[-709, +709]
 * @return hyperbolic cosine of 'x'
 */
template <class D, class V>
HWY_INLINE V Cosh(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallCosh(const D d, VecArg<V> x) {
  return Cosh(d, x);
}

/**
 * Highway SIMD version of std::tanh(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 4
 *      Valid Range: float32[-FLT_MAX, +FLT_MAX], float64[-DBL_MAX, +DBL_MAX]
 * @return hyperbolic tangent of 'x'
 */
template <class D, class V>
HWY_INLINE V Tanh(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallTanh(const D d, VecArg<V> x) {
  return Tanh(d, x);
}

/**
 * Highway SIMD version of std::tgamma(x).
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 4
 *      Valid Range: float32(0, +35], float64(0, +171.6]
 * @return gamma function of 'x'
 */
template <class D, class V>
HWY_INLINE V Tgamma(D d, V x);
template <class D, class V>
HWY_NOINLINE V CallTgamma(const D d, VecArg<V> x) {
  return Tgamma(d, x);
}

/**
 * Highway SIMD version of SinCos.
 * Compute the sine and cosine at the same time
 * The performance should be around the same as calling Sin.
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 1
 *      Valid Range: [-39000, +39000]
 */
template <class D, class V>
HWY_INLINE void SinCos(D d, V x, V& s, V& c);
template <class D, class V>
HWY_NOINLINE void CallSinCos(const D d, VecArg<V> x, V& s, V& c) {
  SinCos(d, x, s, c);
}

/**
 * Highway SIMD version of Hypot
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 4
 *      Valid Range: float32[-FLT_MAX, +FLT_MAX], float64[-DBL_MAX, +DBL_MAX]
 * @return hypotenuse of a and b
 */
template <class D, class V>
HWY_INLINE V Hypot(D d, V a, V b);
template <class D, class V>
HWY_NOINLINE V CallHypot(const D d, VecArg<V> a, VecArg<V> b) {
  return Hypot(d, a, b);
}

/**
 * Highway SIMD version of Pow
 *
 * Valid Lane Types: float32, float64
 *        Max Error: ULP = 5
 *      Valid Range: float32[-FLT_MAX, +FLT_MAX], float64[-DBL_MAX, +DBL_MAX]
 * @return a raised to b
 */
template <class D, class V>
HWY_INLINE V Pow(D d, V a, V b);
template <class D, class V>
HWY_NOINLINE V CallPow(const D d, VecArg<V> a, VecArg<V> b) {
  return Pow(d, a, b);
}

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
  T x2 = Mul(x, x);
  return MulAdd(x2, c2, MulAdd(c1, x, c0));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3) {
  T x2 = Mul(x, x);
  return MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4) {
  T x2 = Mul(x, x);
  T x4 = Mul(x2, x2);
  return MulAdd(x4, c4, MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5) {
  T x2 = Mul(x, x);
  T x4 = Mul(x2, x2);
  return MulAdd(x4, MulAdd(c5, x, c4),
                MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6) {
  T x2 = Mul(x, x);
  T x4 = Mul(x2, x2);
  return MulAdd(x4, MulAdd(x2, c6, MulAdd(c5, x, c4)),
                MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7) {
  T x2 = Mul(x, x);
  T x4 = Mul(x2, x2);
  return MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8) {
  T x2 = Mul(x, x);
  T x4 = Mul(x2, x2);
  T x8 = Mul(x4, x4);
  return MulAdd(x8, c8,
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9) {
  T x2 = Mul(x, x);
  T x4 = Mul(x2, x2);
  T x8 = Mul(x4, x4);
  return MulAdd(x8, MulAdd(c9, x, c8),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10) {
  T x2 = Mul(x, x);
  T x4 = Mul(x2, x2);
  T x8 = Mul(x4, x4);
  return MulAdd(x8, MulAdd(x2, c10, MulAdd(c9, x, c8)),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11) {
  T x2 = Mul(x, x);
  T x4 = Mul(x2, x2);
  T x8 = Mul(x4, x4);
  return MulAdd(x8, MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8)),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12) {
  T x2 = Mul(x, x);
  T x4 = Mul(x2, x2);
  T x8 = Mul(x4, x4);
  return MulAdd(
      x8, MulAdd(x4, c12, MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
      MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
             MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13) {
  T x2 = Mul(x, x);
  T x4 = Mul(x2, x2);
  T x8 = Mul(x4, x4);
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
  T x2 = Mul(x, x);
  T x4 = Mul(x2, x2);
  T x8 = Mul(x4, x4);
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
  T x2 = Mul(x, x);
  T x4 = Mul(x2, x2);
  T x8 = Mul(x4, x4);
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
  T x2 = Mul(x, x);
  T x4 = Mul(x2, x2);
  T x8 = Mul(x4, x4);
  T x16 = Mul(x8, x8);
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
  T x2 = Mul(x, x);
  T x4 = Mul(x2, x2);
  T x8 = Mul(x4, x4);
  T x16 = Mul(x8, x8);
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
  T x2 = Mul(x, x);
  T x4 = Mul(x2, x2);
  T x8 = Mul(x4, x4);
  T x16 = Mul(x8, x8);
  return MulAdd(
      x16, MulAdd(x2, c18, MulAdd(c17, x, c16)),
      MulAdd(x8,
             MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                    MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
             MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                    MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)))));
}

template <class V, HWY_IF_FLOAT_V(V)>
HWY_INLINE HWY_MAYBE_UNUSED V InRangeLoProduct(V a, V b, V p_hi) {
  // InRangeLoProduct assumes that either a[i], b[i], p_hi[i] and the result
  // returned by InRangeLoProduct are all finite or that we do not care about
  // the result.

#if HWY_NATIVE_FMA
  return MulSub(a, b, p_hi);
#else
  const DFromV<decltype(a)> d;
  const RebindToUnsigned<decltype(d)> du;

  using T = TFromD<decltype(d)>;

  constexpr int kNumOfMantFracBits = MantissaBits<T>();
  static_assert(kNumOfMantFracBits > 0, "kNumOfMantFracBits > 0 must be true");

  constexpr int kCeilHalfNumOfMantBits =
      (kNumOfMantFracBits | 1) - (kNumOfMantFracBits >> 1);
  static_assert(kCeilHalfNumOfMantBits > 0,
                "kCeilHalfNumOfMantBits > 0 must be true");

  const auto a_bits = BitCast(du, a);
  const auto b_bits = BitCast(du, b);

  // a_hi[i] is equal to the result of rounding a[i] to the nearest value that
  // has at most kNumOfMantFracBits - kCeilHalfNumOfMantBits + 1 bits of
  // precision, with ties rounded away from zero.

  // b_hi[i] is equal to the result of rounding b[i] to the nearest value that
  // has at most kNumOfMantFracBits - kCeilHalfNumOfMantBits + 1 bits of
  // precision, with ties rounded away from zero.

  // RoundingShiftRight + ShiftLeft is used instead of Veltkamp splitting as
  // RoundingShiftRight + ShiftLeft will only overflow to infinity if |a[i]|
  // or |b[i]| is sufficiently close to LimitsMax<T>().

  // For F16 values, RoundingShiftRight + ShiftLeft will not overflow to
  // infinity for |a[i]| <= 64480, whereas Veltkamp splitting will
  // overflow for |a[i]| > 1007.

  // For F32 values, RoundingShiftRight + ShiftLeft will not overflow to
  // infinity for |a[i]| < 3.4028235E+38, whereas Veltkamp splitting will
  // overflow for |a[i]| > 8.3056467E+34.

  // For F64 values, RoundingShiftRight + ShiftLeft will not overflow to
  // infinity for |a[i]| < 1.7976931214684583E+308, whereas Veltkamp splitting
  // will overflow for |a[i]| >= 1.3393857490036326E300.

  const auto a_hi =
      BitCast(d, ShiftLeft<kCeilHalfNumOfMantBits>(
                     RoundingShiftRight<kCeilHalfNumOfMantBits>(a_bits)));
  const auto b_hi =
      BitCast(d, ShiftLeft<kCeilHalfNumOfMantBits>(
                     RoundingShiftRight<kCeilHalfNumOfMantBits>(b_bits)));

  // InRangeLoProduct assumes that we only care about the result if a_hi[i] and
  // b_hi[i] are both finite.

  const auto a_lo = Sub(a, a_hi);
  const auto b_lo = Sub(b, b_hi);

  // All of the multiplications below are exact if x[i] and y[i] (where x and y
  // are the multiplicands) are both finite values and either the exact value of
  // x[i] * y[i] is a normal finite value, at least one of x[i] or y[i] is zero,
  // or at least one of |x[i]| or |y[i]| is greater than or equal to 1.

  // In addition, a_hi[i] * b_lo[i] + a_lo[i] * b_hi[i] is exact if
  // a_hi[i] * b_lo[i] and a_lo[i] * b_hi[i] are both exact finite values.

  const auto p_lo = MulAdd(
      a_lo, b_lo,
      Add(MulAdd(a_hi, b_lo, Mul(a_lo, b_hi)), MulSub(a_hi, b_hi, p_hi)));

  // p_lo[i] is exact if a_hi[i] * b_hi[i], a_hi[i] * b_lo[i],
  // a_lo[i] * b_hi[i], and a_lo[i] * b_lo[i] are all exact finite values
  // and p_hi[i] is finite (which will be the case if none of the
  // multiplications overflow to infinity or underflow to a subnormal or zero).

  return p_lo;
#endif
}

template <class V, HWY_IF_FLOAT_V(V)>
HWY_INLINE HWY_MAYBE_UNUSED V InRangeFloatDivRem(V a, V b, V q) {
  // InRangeFloatDivRem assumes that either a[i], b[i], q[i] and the result
  // returned by InRangeFloatDivRem are all finite or that we do not care about
  // the result.

#if HWY_NATIVE_FMA
  return NegMulAdd(q, b, a);
#else
  const auto p_hi = Mul(q, b);
  const auto p_lo = InRangeLoProduct(q, b, p_hi);
  return Sub(Sub(a, p_hi), p_lo);
#endif
}

template <class FloatOrDouble>
struct AsinImpl {};
template <class FloatOrDouble>
struct AtanImpl {};
template <class FloatOrDouble>
struct CosSinImpl {};
template <class FloatOrDouble>
struct ErfImpl {};
template <class FloatOrDouble>
struct ExpImpl {};
template <class FloatOrDouble>
struct GammaImpl {};
template <class FloatOrDouble>
struct LogImpl {};
template <class FloatOrDouble>
struct ExtPrecLog2ForPowImpl;
template <class FloatOrDouble>
struct SinCosImpl {};

template <>
struct AsinImpl<float> {
  // Polynomial approximation for asin(x) over the range [0, 0.5).
  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE V AsinPoly(D d, V x2, V /*x*/) {
    const auto k0 = Set(d, +0.1666677296f);
    const auto k1 = Set(d, +0.07495029271f);
    const auto k2 = Set(d, +0.04547423869f);
    const auto k3 = Set(d, +0.02424046025f);
    const auto k4 = Set(d, +0.04197454825f);

    return Estrin(x2, k0, k1, k2, k3, k4);
  }
};

#if HWY_HAVE_FLOAT64 && HWY_HAVE_INTEGER64

template <>
struct AsinImpl<double> {
  // Polynomial approximation for asin(x) over the range [0, 0.5).
  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE V AsinPoly(D d, V x2, V /*x*/) {
    const auto k0 = Set(d, +0.1666666666666497543);
    const auto k1 = Set(d, +0.07500000000378581611);
    const auto k2 = Set(d, +0.04464285681377102438);
    const auto k3 = Set(d, +0.03038195928038132237);
    const auto k4 = Set(d, +0.02237176181932048341);
    const auto k5 = Set(d, +0.01735956991223614604);
    const auto k6 = Set(d, +0.01388715184501609218);
    const auto k7 = Set(d, +0.01215360525577377331);
    const auto k8 = Set(d, +0.006606077476277170610);
    const auto k9 = Set(d, +0.01929045477267910674);
    const auto k10 = Set(d, -0.01581918243329996643);
    const auto k11 = Set(d, +0.03161587650653934628);

    return Estrin(x2, k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11);
  }
};

#endif

template <>
struct AtanImpl<float> {
  // Polynomial approximation for atan(x) over the range [0, 1.0).
  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE V AtanPoly(D d, V x) {
    const auto k0 = Set(d, -0.333331018686294555664062f);
    const auto k1 = Set(d, +0.199926957488059997558594f);
    const auto k2 = Set(d, -0.142027363181114196777344f);
    const auto k3 = Set(d, +0.106347933411598205566406f);
    const auto k4 = Set(d, -0.0748900920152664184570312f);
    const auto k5 = Set(d, +0.0425049886107444763183594f);
    const auto k6 = Set(d, -0.0159569028764963150024414f);
    const auto k7 = Set(d, +0.00282363896258175373077393f);

    const auto y = Mul(x, x);
    return MulAdd(Estrin(y, k0, k1, k2, k3, k4, k5, k6, k7), Mul(y, x), x);
  }
};

#if HWY_HAVE_FLOAT64 && HWY_HAVE_INTEGER64

template <>
struct AtanImpl<double> {
  // Polynomial approximation for atan(x) over the range [0, 1.0).
  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE V AtanPoly(D d, V x) {
    const auto k0 = Set(d, -0.333333333333311110369124);
    const auto k1 = Set(d, +0.199999999996591265594148);
    const auto k2 = Set(d, -0.14285714266771329383765);
    const auto k3 = Set(d, +0.111111105648261418443745);
    const auto k4 = Set(d, -0.090908995008245008229153);
    const auto k5 = Set(d, +0.0769219538311769618355029);
    const auto k6 = Set(d, -0.0666573579361080525984562);
    const auto k7 = Set(d, +0.0587666392926673580854313);
    const auto k8 = Set(d, -0.0523674852303482457616113);
    const auto k9 = Set(d, +0.0466667150077840625632675);
    const auto k10 = Set(d, -0.0407629191276836500001934);
    const auto k11 = Set(d, +0.0337852580001353069993897);
    const auto k12 = Set(d, -0.0254517624932312641616861);
    const auto k13 = Set(d, +0.016599329773529201970117);
    const auto k14 = Set(d, -0.00889896195887655491740809);
    const auto k15 = Set(d, +0.00370026744188713119232403);
    const auto k16 = Set(d, -0.00110611831486672482563471);
    const auto k17 = Set(d, +0.000209850076645816976906797);
    const auto k18 = Set(d, -1.88796008463073496563746e-5);

    const auto y = Mul(x, x);
    return MulAdd(Estrin(y, k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11,
                         k12, k13, k14, k15, k16, k17, k18),
                  Mul(y, x), x);
  }
};

#endif

template <>
struct ErfImpl<float> {
  // Polynomial approximation for erf(x) over |x| < 1
  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE V SmallErf(D d, V x, V z) {
    const V kT0 = Set(d, +1.128379165726710E+0f);
    const V kT1 = Set(d, -3.761262582423300E-1f);
    const V kT2 = Set(d, +1.128358514861418E-1f);
    const V kT3 = Set(d, -2.685381193529856E-2f);
    const V kT4 = Set(d, +5.188327685732524E-3f);
    const V kT5 = Set(d, -8.010193625184903E-4f);
    const V kT6 = Set(d, +7.853861353153693E-5f);
    return Mul(x, Estrin(z, kT0, kT1, kT2, kT3, kT4, kT5, kT6));
  }

  // erfc(x) / exp(-x*x) via polynomial in 1/(x*x), |x| in [1, Limit)
  // Uses P for |x| in [1, 2), R for |x| in [2, Limit)
  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE V ErfcFactor(D d, V x, V /*z*/) {
    const V kOne = Set(d, +1.0f);
    const V inv_x = Div(kOne, x);
    const V w = Mul(inv_x, inv_x);

    const V kP0 = Set(d, +5.638259427386472E-1f);
    const V kP1 = Set(d, -2.741127028184656E-1f);
    const V kP2 = Set(d, +3.404879937665872E-1f);
    const V kP3 = Set(d, -4.944515323274145E-1f);
    const V kP4 = Set(d, +6.210004621745983E-1f);
    const V kP5 = Set(d, -5.824733027278666E-1f);
    const V kP6 = Set(d, +3.687424674597105E-1f);
    const V kP7 = Set(d, -1.387039388740657E-1f);
    const V kP8 = Set(d, +2.326819970068386E-2f);
    const V poly_P = Estrin(w, kP0, kP1, kP2, kP3, kP4, kP5, kP6, kP7, kP8);

    const V kR0 = Set(d, +5.641895067754075E-1f);
    const V kR1 = Set(d, -2.820767439740514E-1f);
    const V kR2 = Set(d, +4.218463358204948E-1f);
    const V kR3 = Set(d, -1.015265279202700E+0f);
    const V kR4 = Set(d, +2.921019019210786E+0f);
    const V kR5 = Set(d, -7.495518717768503E+0f);
    const V kR6 = Set(d, +1.297719955372516E+1f);
    const V kR7 = Set(d, -1.047766399936249E+1f);
    const V poly_R = Estrin(w, kR0, kR1, kR2, kR3, kR4, kR5, kR6, kR7);

    const auto is_mid = Lt(x, Set(d, +2.0f));
    return Mul(inv_x, IfThenElse(is_mid, poly_P, poly_R));
  }

  // |x| >= 14: erfc underflows f32, erf saturates to +/-1
  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE V Limit(D d) {
    return Set(d, +14.0f);
  }
};

#if HWY_HAVE_FLOAT64 && HWY_HAVE_INTEGER64

template <>
struct ErfImpl<double> {
  // T(z) / U(z) approximation for erf(x) over |x| < 1
  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE V SmallErf(D d, V x, V z) {
    const V kT0 = Set(d, +5.55923013010394962768E4);
    const V kT1 = Set(d, +7.00332514112805075473E3);
    const V kT2 = Set(d, +2.23200534594684319226E3);
    const V kT3 = Set(d, +9.00260197203842689217E1);
    const V kT4 = Set(d, +9.60497373987051638749E0);
    const V T_poly = Estrin(z, kT0, kT1, kT2, kT3, kT4);

    const V kU0 = Set(d, +4.92673942608635921086E4);
    const V kU1 = Set(d, +2.26290000613890934246E4);
    const V kU2 = Set(d, +4.59432382970980127987E3);
    const V kU3 = Set(d, +5.21357949780152679795E2);
    const V kU4 = Set(d, +3.35617141647503099647E1);
    const V kU5 = Set(d, +1.0);
    const V U_poly = Estrin(z, kU0, kU1, kU2, kU3, kU4, kU5);

    return Mul(x, Div(T_poly, U_poly));
  }

  // erfc(x) / exp(-x*x) via P(x)/Q(x) and R(x)/S(x), |x| in [1, Limit)
  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE V ErfcFactor(D d, V x, V /*z*/) {
    const V kP0 = Set(d, +5.57535335369399327526E2);
    const V kP1 = Set(d, +1.02755188689515710272E3);
    const V kP2 = Set(d, +9.34528527171957607540E2);
    const V kP3 = Set(d, +5.26445194995477358631E2);
    const V kP4 = Set(d, +1.96520832956077098242E2);
    const V kP5 = Set(d, +4.86371970985681366614E1);
    const V kP6 = Set(d, +7.46321056442269912687E0);
    const V kP7 = Set(d, +5.64189564831068821977E-1);
    const V kP8 = Set(d, +2.46196981473530512524E-10);
    const V poly_P = Estrin(x, kP0, kP1, kP2, kP3, kP4, kP5, kP6, kP7, kP8);

    const V kQ0 = Set(d, +5.57535340817727675546E2);
    const V kQ1 = Set(d, +1.65666309194161350182E3);
    const V kQ2 = Set(d, +2.24633760818710981792E3);
    const V kQ3 = Set(d, +1.82390916687909736289E3);
    const V kQ4 = Set(d, +9.75708501743205489753E2);
    const V kQ5 = Set(d, +3.54937778887819891062E2);
    const V kQ6 = Set(d, +8.67072140885989742329E1);
    const V kQ7 = Set(d, +1.32281951154744992508E1);
    const V kQ8 = Set(d, +1.0);
    const V poly_Q = Estrin(x, kQ0, kQ1, kQ2, kQ3, kQ4, kQ5, kQ6, kQ7, kQ8);

    const V kR0 = Set(d, +2.97886665372100240670E0);
    const V kR1 = Set(d, +7.40974269950448939160E0);
    const V kR2 = Set(d, +6.16021097993053585195E0);
    const V kR3 = Set(d, +5.01905042251180477414E0);
    const V kR4 = Set(d, +1.27536670759978104416E0);
    const V kR5 = Set(d, +5.64189583547755073984E-1);
    const V poly_R = Estrin(x, kR0, kR1, kR2, kR3, kR4, kR5);

    const V kS0 = Set(d, +3.36907645100081516050E0);
    const V kS1 = Set(d, +9.60896809063285878198E0);
    const V kS2 = Set(d, +1.70814450747565897222E1);
    const V kS3 = Set(d, +1.20489539808096656605E1);
    const V kS4 = Set(d, +9.39603524938001434673E0);
    const V kS5 = Set(d, +2.26052863220117276590E0);
    const V kS6 = Set(d, +1.0);
    const V poly_S = Estrin(x, kS0, kS1, kS2, kS3, kS4, kS5, kS6);

    const auto is_mid = Lt(x, Set(d, +8.0));
    const V num = IfThenElse(is_mid, poly_P, poly_R);
    const V den = IfThenElse(is_mid, poly_Q, poly_S);
    return Div(num, den);
  }

  // |x| >= 37.519: erfc underflows f64, erf saturates to +/-1
  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE V Limit(D d) {
    return Set(d, +37.519379347);
  }
};

#endif

template <>
struct GammaImpl<float> {
  // Stirling series via polynomial in 1/(x*x), for x >= StirlingLimit().
  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE V StirlingPoly(D d, V u) {
    const V c0 = Set(d, +0.08333332931388039f);
    const V c1 = Set(d, -0.0027766148725720703f);
    const V c2 = Set(d, +0.0007427214288710828f);
    return Estrin(u, c0, c1, c2);
  }

  // Use the Stirling asymptotic path for w >= StirlingLimit().
  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE V StirlingLimit(D d) {
    return Set(d, +4.0f);
  }
  // Recurrence steps to reduce w into [2, 3): ceil(StirlingLimit - 3).
  static constexpr int kReduceSteps = 1;

  // Polynomial approximation for Gamma(x) on [2, 3], argument t = x - 2.5.
  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE V GammaPoly(D d, V t) {
    const V c0 = Set(d, +0.001607034915482715f);
    const V c1 = Set(d, +0.010780565616863308f);
    const V c2 = Set(d, +0.028360953629043274f);
    const V c3 = Set(d, +0.10961204737926883f);
    const V c4 = Set(d, +0.2538712358604304f);
    const V c5 = Set(d, +0.6545616480372625f);
    const V c6 = Set(d, +0.934734521715248f);
    const V c7 = Set(d, +1.3293403641787918f);
    return Estrin(t, c7, c6, c5, c4, c3, c2, c1, c0);
  }
};

#if HWY_HAVE_FLOAT64 && HWY_HAVE_INTEGER64

template <>
struct GammaImpl<double> {
  // Stirling series via polynomial in 1/(x*x), for x >= StirlingLimit().
  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE V StirlingPoly(D d, V u) {
    const V c0 = Set(d, +0.0833333333333333);
    const V c1 = Set(d, -0.002777777777606244);
    const V c2 = Set(d, +0.0007936506649488252);
    const V c3 = Set(d, -0.0005952025951802712);
    const V c4 = Set(d, +0.0008372834512071336);
    const V c5 = Set(d, -0.0016526004455595122);
    return Estrin(u, c0, c1, c2, c3, c4, c5);
  }

  // Use the Stirling asymptotic path for w >= StirlingLimit().
  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE V StirlingLimit(D d) {
    return Set(d, +8.0);
  }
  // Recurrence steps to reduce w into [2, 3): ceil(StirlingLimit - 3).
  static constexpr int kReduceSteps = 5;

  // Polynomial approximation for Gamma(x) on [2, 3], argument t = x - 2.5.
  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE V GammaPoly(D d, V t) {
    const V c0 = Set(d, +2.038631996005556e-07);
    const V c1 = Set(d, -5.060382483924732e-07);
    const V c2 = Set(d, +1.0675361551610872e-06);
    const V c3 = Set(d, -2.5237611386457594e-06);
    const V c4 = Set(d, +7.257335209971111e-06);
    const V c5 = Set(d, -1.2810259356350584e-05);
    const V c6 = Set(d, +6.132951527532448e-05);
    const V c7 = Set(d, +3.942933739732899e-06);
    const V c8 = Set(d, +0.0007544741634534283);
    const V c9 = Set(d, +0.001607400398278153);
    const V c10 = Set(d, +0.010392403424525895);
    const V c11 = Set(d, +0.028360780598296692);
    const V c12 = Set(d, +0.10967323400218487);
    const V c13 = Set(d, +0.253871246841081);
    const V c14 = Set(d, +0.6545585779813367);
    const V c15 = Set(d, +0.9347345216260858);
    const V c16 = Set(d, +1.329340388179137);
    return Estrin(t, c16, c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4,
                  c3, c2, c1, c0);
  }
};

#endif

template <>
struct CosSinImpl<float> {
  // Rounds float toward zero and returns as int32_t.
  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE Vec<Rebind<int32_t, D>> ToInt32(D /*unused*/, V x) {
    return ConvertTo(Rebind<int32_t, D>(), x);
  }

  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE V Poly(D d, V x) {
    const auto k0 = Set(d, -1.66666597127914428710938e-1f);
    const auto k1 = Set(d, +8.33307858556509017944336e-3f);
    const auto k2 = Set(d, -1.981069071916863322258e-4f);
    const auto k3 = Set(d, +2.6083159809786593541503e-6f);

    const auto y = Mul(x, x);
    return MulAdd(Estrin(y, k0, k1, k2, k3), Mul(y, x), x);
  }

  template <class D, class V = VFromD<D>, class VI32 = Vec<Rebind<int32_t, D>>,
            HWY_IF_F32_D(D)>
  HWY_INLINE V CosReduce(D d, V x, VI32 q) {
    // kHalfPiPart0f + kHalfPiPart1f + kHalfPiPart2f + kHalfPiPart3f ~= -pi/2
    const V kHalfPiPart0f = Set(d, -0.5f * 3.140625f);
    const V kHalfPiPart1f = Set(d, -0.5f * 0.0009670257568359375f);
    const V kHalfPiPart2f = Set(d, -0.5f * 6.2771141529083251953e-7f);
    const V kHalfPiPart3f = Set(d, -0.5f * 1.2154201256553420762e-10f);

    // Extended precision modular arithmetic.
    const V qf = ConvertTo(d, q);
    x = MulAdd(qf, kHalfPiPart0f, x);
    x = MulAdd(qf, kHalfPiPart1f, x);
    x = MulAdd(qf, kHalfPiPart2f, x);
    x = MulAdd(qf, kHalfPiPart3f, x);
    return x;
  }

  template <class D, class V = VFromD<D>, class VI32 = Vec<Rebind<int32_t, D>>,
            HWY_IF_F32_D(D)>
  HWY_INLINE V SinReduce(D d, V x, VI32 q) {
    // kPiPart0f + kPiPart1f + kPiPart2f + kPiPart3f ~= -pi
    const V kPiPart0f = Set(d, -3.140625f);
    const V kPiPart1f = Set(d, -0.0009670257568359375f);
    const V kPiPart2f = Set(d, -6.2771141529083251953e-7f);
    const V kPiPart3f = Set(d, -1.2154201256553420762e-10f);

    // Extended precision modular arithmetic.
    const V qf = ConvertTo(d, q);
    x = MulAdd(qf, kPiPart0f, x);
    x = MulAdd(qf, kPiPart1f, x);
    x = MulAdd(qf, kPiPart2f, x);
    x = MulAdd(qf, kPiPart3f, x);
    return x;
  }

  // (q & 2) == 0 ? -0.0 : +0.0
  template <class D, class VI32 = Vec<Rebind<int32_t, D>>, HWY_IF_F32_D(D)>
  HWY_INLINE Vec<Rebind<float, D>> CosSignFromQuadrant(D d, VI32 q) {
    const VI32 kTwo = Set(Rebind<int32_t, D>(), 2);
    return BitCast(d, ShiftLeft<30>(AndNot(q, kTwo)));
  }

  // ((q & 1) ? -0.0 : +0.0)
  template <class D, class VI32 = Vec<Rebind<int32_t, D>>, HWY_IF_F32_D(D)>
  HWY_INLINE Vec<Rebind<float, D>> SinSignFromQuadrant(D d, VI32 q) {
    const VI32 kOne = Set(Rebind<int32_t, D>(), 1);
    return BitCast(d, ShiftLeft<31>(And(q, kOne)));
  }
};

#if HWY_HAVE_FLOAT64 && HWY_HAVE_INTEGER64

template <>
struct CosSinImpl<double> {
  // Rounds double toward zero and returns as int32_t.
  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE Vec<Rebind<int32_t, D>> ToInt32(D /*unused*/, V x) {
    return DemoteTo(Rebind<int32_t, D>(), x);
  }

  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE V Poly(D d, V x) {
    const auto k0 = Set(d, -0.166666666666666657414808);
    const auto k1 = Set(d, +0.00833333333333332974823815);
    const auto k2 = Set(d, -0.000198412698412696162806809);
    const auto k3 = Set(d, +2.75573192239198747630416e-6);
    const auto k4 = Set(d, -2.50521083763502045810755e-8);
    const auto k5 = Set(d, +1.60590430605664501629054e-10);
    const auto k6 = Set(d, -7.64712219118158833288484e-13);
    const auto k7 = Set(d, +2.81009972710863200091251e-15);
    const auto k8 = Set(d, -7.97255955009037868891952e-18);

    const auto y = Mul(x, x);
    return MulAdd(Estrin(y, k0, k1, k2, k3, k4, k5, k6, k7, k8), Mul(y, x), x);
  }

  template <class D, class V = VFromD<D>, class VI32 = Vec<Rebind<int32_t, D>>,
            HWY_IF_F64_D(D)>
  HWY_INLINE V CosReduce(D d, V x, VI32 q) {
    // kHalfPiPart0d + kHalfPiPart1d + kHalfPiPart2d + kHalfPiPart3d ~= -pi/2
    const V kHalfPiPart0d = Set(d, -0.5 * 3.1415926218032836914);
    const V kHalfPiPart1d = Set(d, -0.5 * 3.1786509424591713469e-8);
    const V kHalfPiPart2d = Set(d, -0.5 * 1.2246467864107188502e-16);
    const V kHalfPiPart3d = Set(d, -0.5 * 1.2736634327021899816e-24);

    // Extended precision modular arithmetic.
    const V qf = PromoteTo(d, q);
    x = MulAdd(qf, kHalfPiPart0d, x);
    x = MulAdd(qf, kHalfPiPart1d, x);
    x = MulAdd(qf, kHalfPiPart2d, x);
    x = MulAdd(qf, kHalfPiPart3d, x);
    return x;
  }

  template <class D, class V = VFromD<D>, class VI32 = Vec<Rebind<int32_t, D>>,
            HWY_IF_F64_D(D)>
  HWY_INLINE V SinReduce(D d, V x, VI32 q) {
    // kPiPart0d + kPiPart1d + kPiPart2d + kPiPart3d ~= -pi
    const V kPiPart0d = Set(d, -3.1415926218032836914);
    const V kPiPart1d = Set(d, -3.1786509424591713469e-8);
    const V kPiPart2d = Set(d, -1.2246467864107188502e-16);
    const V kPiPart3d = Set(d, -1.2736634327021899816e-24);

    // Extended precision modular arithmetic.
    const V qf = PromoteTo(d, q);
    x = MulAdd(qf, kPiPart0d, x);
    x = MulAdd(qf, kPiPart1d, x);
    x = MulAdd(qf, kPiPart2d, x);
    x = MulAdd(qf, kPiPart3d, x);
    return x;
  }

  // (q & 2) == 0 ? -0.0 : +0.0
  template <class D, class VI32 = Vec<Rebind<int32_t, D>>, HWY_IF_F64_D(D)>
  HWY_INLINE Vec<Rebind<double, D>> CosSignFromQuadrant(D d, VI32 q) {
    const VI32 kTwo = Set(Rebind<int32_t, D>(), 2);
    return BitCast(
        d, ShiftLeft<62>(PromoteTo(Rebind<int64_t, D>(), AndNot(q, kTwo))));
  }

  // ((q & 1) ? -0.0 : +0.0)
  template <class D, class VI32 = Vec<Rebind<int32_t, D>>, HWY_IF_F64_D(D)>
  HWY_INLINE Vec<Rebind<double, D>> SinSignFromQuadrant(D d, VI32 q) {
    const VI32 kOne = Set(Rebind<int32_t, D>(), 1);
    return BitCast(
        d, ShiftLeft<63>(PromoteTo(Rebind<int64_t, D>(), And(q, kOne))));
  }
};

#endif

template <>
struct ExpImpl<float> {
  // Rounds float toward zero and returns as int32_t.
  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE Vec<Rebind<int32_t, D>> ToInt32(D /*unused*/, V x) {
    return ConvertTo(Rebind<int32_t, D>(), x);
  }

  // Rounds float to nearest int32_t
  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE Vec<Rebind<int32_t, D>> ToNearestInt32(D /*unused*/, V x) {
    return NearestInt(x);
  }

  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE V ExpPoly(D d, V x) {
    const auto k0 = Set(d, +0.5f);
    const auto k1 = Set(d, +0.166666671633720397949219f);
    const auto k2 = Set(d, +0.0416664853692054748535156f);
    const auto k3 = Set(d, +0.00833336077630519866943359f);
    const auto k4 = Set(d, +0.00139304355252534151077271f);
    const auto k5 = Set(d, +0.000198527617612853646278381f);

    return MulAdd(Estrin(x, k0, k1, k2, k3, k4, k5), Mul(x, x), x);
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
    // kLn2Part0f + kLn2Part1f ~= -ln(2)
    const V kLn2Part0f = Set(d, -0.693145751953125f);
    const V kLn2Part1f = Set(d, -1.428606765330187045e-6f);

    // Extended precision modular arithmetic.
    const V qf = ConvertTo(d, q);
    x = MulAdd(qf, kLn2Part0f, x);
    x = MulAdd(qf, kLn2Part1f, x);
    return x;
  }

  template <class D, class V = VFromD<D>, class VI32 = Vec<Rebind<int32_t, D>>,
            HWY_IF_F32_D(D)>
  HWY_INLINE V Exp2Reduce(D d, V x, VI32 q) {
    const V x_frac = Sub(x, ConvertTo(d, q));
    return MulAdd(x_frac, Set(d, 0.193147182464599609375f),
                  Mul(x_frac, Set(d, 0.5f)));
  }

  template <class D, class V = VFromD<D>, class VI32 = Vec<Rebind<int32_t, D>>,
            HWY_IF_F32_D(D)>
  HWY_INLINE V Exp2ReduceForPow(D d, V x_hi, V x_lo, VI32 q) {
    const V x_frac = Add(Sub(x_hi, ConvertTo(d, q)), x_lo);
    return MulAdd(x_frac, Set(d, 0.193147182464599609375f),
                  Mul(x_frac, Set(d, 0.5f)));
  }
};

template <>
struct LogImpl<float> {
  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE Vec<Rebind<int32_t, D>> Log2p1NoSubnormal(
      D /*d*/, Vec<Rebind<int32_t, D>> x) {
    const Rebind<int32_t, D> di32;
    const Rebind<uint32_t, D> du32;
    const auto kBias = Set(di32, 0x7F);
    return Sub(BitCast(di32, ShiftRight<23>(BitCast(du32, x))), kBias);
  }

  // Approximates Log(x) over the range [sqrt(2) / 2, sqrt(2)].
  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE V LogPoly(D d, V x) {
    const V k0 = Set(d, 0.66666662693f);
    const V k1 = Set(d, 0.40000972152f);
    const V k2 = Set(d, 0.28498786688f);
    const V k3 = Set(d, 0.24279078841f);

    const V x2 = Mul(x, x);
    const V x4 = Mul(x2, x2);
    return MulAdd(MulAdd(k2, x4, k0), x2, Mul(MulAdd(k3, x4, k1), x4));
  }
};

template <>
struct ExtPrecLog2ForPowImpl<float> {
  // Approximates Log2(x) over the range [sqrt(2) / 2, sqrt(2)].
  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE V ExtPrecLog2Poly(D d, V z_sqr_hi) {
    const auto k0 = Set(d, 0.5770779f);
    const auto k1 = Set(d, 0.41221106f);
    const auto k2 = Set(d, 0.31983307f);
    const auto k3 = Set(d, 0.28389898f);

    return Estrin(z_sqr_hi, k0, k1, k2, k3);
  }
};

#if HWY_HAVE_FLOAT64 && HWY_HAVE_INTEGER64
template <>
struct ExpImpl<double> {
  // Rounds double toward zero and returns as int32_t.
  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE Vec<Rebind<int32_t, D>> ToInt32(D /*unused*/, V x) {
    return DemoteTo(Rebind<int32_t, D>(), x);
  }

  // Rounds double to nearest int32_t
  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE Vec<Rebind<int32_t, D>> ToNearestInt32(D /*unused*/, V x) {
    return DemoteToNearestInt(Rebind<int32_t, D>(), x);
  }

  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
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
                  Mul(x, x), x);
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
    // kLn2Part0d + kLn2Part1d ~= -ln(2)
    const V kLn2Part0d = Set(d, -0.6931471805596629565116018);
    const V kLn2Part1d = Set(d, -0.28235290563031577122588448175e-12);

    // Extended precision modular arithmetic.
    const V qf = PromoteTo(d, q);
    x = MulAdd(qf, kLn2Part0d, x);
    x = MulAdd(qf, kLn2Part1d, x);
    return x;
  }

  template <class D, class V = VFromD<D>, class VI32 = Vec<Rebind<int32_t, D>>,
            HWY_IF_F64_D(D)>
  HWY_INLINE V Exp2Reduce(D d, V x, VI32 q) {
    const V x_frac = Sub(x, PromoteTo(d, q));
    return MulAdd(x_frac, Set(d, 0.1931471805599453139823396),
                  Mul(x_frac, Set(d, 0.5)));
  }

  template <class D, class V = VFromD<D>, class VI32 = Vec<Rebind<int32_t, D>>,
            HWY_IF_F64_D(D)>
  HWY_INLINE V Exp2ReduceForPow(D d, V x_hi, V x_lo, VI32 q) {
    const V x_frac = Add(Sub(x_hi, PromoteTo(d, q)), x_lo);
    return MulAdd(x_frac, Set(d, 0.1931471805599453139823396),
                  Mul(x_frac, Set(d, 0.5)));
  }
};

template <>
struct LogImpl<double> {
  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE Vec<Rebind<int64_t, D>> Log2p1NoSubnormal(
      D /*d*/, Vec<Rebind<int64_t, D>> x) {
    const Rebind<int64_t, D> di64;
    const Rebind<uint64_t, D> du64;
    return Sub(BitCast(di64, ShiftRight<52>(BitCast(du64, x))),
               Set(di64, 0x3FF));
  }

  // Approximates Log(x) over the range [sqrt(2) / 2, sqrt(2)].
  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE V LogPoly(D d, V x) {
    const V k0 = Set(d, 0.6666666666666735130);
    const V k1 = Set(d, 0.3999999999940941908);
    const V k2 = Set(d, 0.2857142874366239149);
    const V k3 = Set(d, 0.2222219843214978396);
    const V k4 = Set(d, 0.1818357216161805012);
    const V k5 = Set(d, 0.1531383769920937332);
    const V k6 = Set(d, 0.1479819860511658591);

    const V x2 = Mul(x, x);
    const V x4 = Mul(x2, x2);
    return MulAdd(MulAdd(MulAdd(MulAdd(k6, x4, k4), x4, k2), x4, k0), x2,
                  (Mul(MulAdd(MulAdd(k5, x4, k3), x4, k1), x4)));
  }
};

#if HWY_HAVE_FLOAT64
template <>
struct ExtPrecLog2ForPowImpl<double> {
  // Approximates Log2(x) over the range [sqrt(2) / 2, sqrt(2)].
  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE V ExtPrecLog2Poly(D d, V z_sqr_hi) {
    const auto k0 = Set(d, 0.5770780163556751);
    const auto k1 = Set(d, 0.41219858307744156);
    const auto k2 = Set(d, 0.3205989040855153);
    const auto k3 = Set(d, 0.26230757701579316);
    const auto k4 = Set(d, 0.2219887477464986);
    const auto k5 = Set(d, 0.19115982151710184);
    const auto k6 = Set(d, 0.19116733197473582);

    return Estrin(z_sqr_hi, k0, k1, k2, k3, k4, k5, k6);
  }
};
#endif

#endif

template <class D, class V, bool kAllowSubnormals = true>
HWY_INLINE V Log(const D d, V x) {
  // http://git.musl-libc.org/cgit/musl/tree/src/math/log.c for more info.
  using T = TFromD<D>;
  impl::LogImpl<T> impl;

  constexpr bool kIsF32 = (sizeof(T) == 4);

  // Float Constants
  const V kLn2Hi = Set(d, kIsF32 ? static_cast<T>(0.69313812256f)
                                 : static_cast<T>(0.693147180369123816490));
  const V kLn2Lo = Set(d, kIsF32 ? static_cast<T>(9.0580006145e-6f)
                                 : static_cast<T>(1.90821492927058770002e-10));
  const V kOne = Set(d, static_cast<T>(+1.0));
  const V kMinNormal = Set(d, kIsF32 ? static_cast<T>(1.175494351e-38f)
                                     : static_cast<T>(2.2250738585072014e-308));
  const V kScale = Set(d, kIsF32 ? static_cast<T>(3.355443200e+7f)
                                 : static_cast<T>(1.8014398509481984e+16));

  // Integer Constants
  using TI = MakeSigned<T>;
  const Rebind<TI, D> di;
  using VI = decltype(Zero(di));
  const VI kLowerBits = Set(di, kIsF32 ? static_cast<TI>(0x00000000L)
                                       : static_cast<TI>(0xFFFFFFFFLL));
  const VI kMagic = Set(di, kIsF32 ? static_cast<TI>(0x3F3504F3L)
                                   : static_cast<TI>(0x3FE6A09E00000000LL));
  const VI kExpMagicDiff = Set(
      di, kIsF32
              ? static_cast<TI>(0x3F800000L - 0x3F3504F3L)
              : static_cast<TI>(0x3FF0000000000000LL - 0x3FE6A09E00000000LL));
  const VI kExpScale =
      Set(di, kIsF32 ? static_cast<TI>(-25) : static_cast<TI>(-54));
  const VI kManMask = Set(di, kIsF32 ? static_cast<TI>(0x7FFFFFL)
                                     : static_cast<TI>(0xFFFFF00000000LL));

  // Scale up 'x' so that it is no longer denormalized.
  VI exp_bits;
  V exp;
  if (kAllowSubnormals == true) {
    const auto is_denormal = Lt(x, kMinNormal);
    x = IfThenElse(is_denormal, Mul(x, kScale), x);

    // Compute the new exponent.
    exp_bits = Add(BitCast(di, x), kExpMagicDiff);
    const VI exp_scale =
        BitCast(di, IfThenElseZero(is_denormal, BitCast(d, kExpScale)));
    exp = ConvertTo(d, Add(exp_scale, impl.Log2p1NoSubnormal(d, exp_bits)));
  } else {
    // Compute the new exponent.
    exp_bits = Add(BitCast(di, x), kExpMagicDiff);
    exp = ConvertTo(d, impl.Log2p1NoSubnormal(d, exp_bits));
  }

  // Renormalize.
  const V y = Or(And(x, BitCast(d, kLowerBits)),
                 BitCast(d, Add(And(exp_bits, kManMask), kMagic)));

  // Approximate and reconstruct.
  const V ym1 = Sub(y, kOne);
  const V z = Div(ym1, Add(y, kOne));

  return MulSub(
      exp, kLn2Hi,
      Sub(MulSub(z, Sub(ym1, impl.LogPoly(d, z)), Mul(exp, kLn2Lo)), ym1));
}

template <class D, class V = VFromD<D>, HWY_IF_FLOAT3264_D(D)>
HWY_INLINE V ExtPrecLog2OfMantForPow(D d, V x, V& log2_x_lo) {
  // sqrt(2) / 2 <= x[i] <= sqrt(2) should be true

  using T = TFromD<D>;
  impl::ExtPrecLog2ForPowImpl<T> impl;

  const V one = Set(d, ConvertScalarTo<T>(1));

  const V x_minus_1 = Sub(x, one);

  const V x_plus_1_hi = Add(x, one);
  const V x_plus_1_v = Sub(x_plus_1_hi, x);
  const V x_plus_1_lo =
      Add(Sub(x, Sub(x_plus_1_hi, x_plus_1_v)), Sub(one, x_plus_1_v));

  const V z_hi0 = Div(x_minus_1, x_plus_1_hi);
  const V z_lo0 =
      Div(NegMulAdd(z_hi0, x_plus_1_lo,
                    InRangeFloatDivRem(x_minus_1, x_plus_1_hi, z_hi0)),
          x_plus_1_hi);

  const V z_hi = Add(z_hi0, z_lo0);
  const V z_lo = Add(Sub(z_hi0, z_hi), z_lo0);

  const V z_sqr_hi = Mul(z_hi, z_hi);
  const V z_sqr_lo = InRangeLoProduct(z_hi, z_hi, z_sqr_hi);

  const V p0 = Mul(z_sqr_hi, impl.ExtPrecLog2Poly(d, z_sqr_hi));

  constexpr bool kIsF32 = (sizeof(T) == 4);
  const V c0_hi = Set(d, kIsF32 ? static_cast<T>(2.88539f)
                                : static_cast<T>(2.8853900817779268));
  const V c0_lo = Set(d, kIsF32 ? static_cast<T>(3.851926E-8f)
                                : static_cast<T>(4.0710547481862066E-17));

  const V c1_hi = Set(d, kIsF32 ? static_cast<T>(0.9617967f)
                                : static_cast<T>(0.9617966939259756));
  const V c1_lo = Set(d, kIsF32 ? static_cast<T>(-6.8015265E-9f)
                                : static_cast<T>(-3.1125578659356493E-17));

  const V s0 = Add(p0, c1_lo);
  const V v0 = Sub(s0, p0);
  const V e0 = Add(Sub(p0, Sub(s0, v0)), Sub(c1_lo, v0));

  const V s1 = Add(c1_hi, s0);
  const V e1 = Add(Add(Sub(c1_hi, s1), s0), e0);

  const V p1_hi = Mul(s1, z_sqr_hi);
  const V p1_lo =
      MulAdd(e1, z_sqr_hi,
             MulAdd(s1, z_sqr_lo, InRangeLoProduct(s1, z_sqr_hi, p1_hi)));

  const V s2 = Add(p1_hi, c0_lo);
  const V v2 = Sub(s2, p1_hi);
  const V e2 = Add(Sub(p1_hi, Sub(s2, v2)), Sub(c0_lo, v2));

  const V s3 = Add(c0_hi, s2);
  const V e3 = Add(Add(Sub(c0_hi, s3), s2), Add(e2, p1_lo));

  const V log2_x_hi = Mul(s3, z_hi);
  log2_x_lo =
      MulAdd(e3, z_hi, MulAdd(s3, z_lo, InRangeLoProduct(s3, z_hi, log2_x_hi)));

  return log2_x_hi;
}

// ExtPrecLog2OfMantForPow(x) computes Log2(x) to extra precision, which ensures
// that the fractional portion of Log2(a) * b is sufficiently accurate for
// Pow(a, b) if |Log2(a) * b| > 0.5 is true.
template <class D, class V = VFromD<D>, HWY_IF_FLOAT3264_D(D)>
HWY_INLINE V ExtPrecLog2ForPow(D d, V x, V& log2_x_lo) {
  // x[i] should have its sign bit cleared

  using T = TFromD<D>;
  using TU = MakeUnsigned<T>;
  using TI = MakeSigned<T>;

  constexpr bool kIsF32 = (sizeof(T) == 4);

  // kExpAdjBitIncr is equal to the fractional mantissa bits of the largest
  // floating point value that is less than the exact value of sqrt(2).
  constexpr TU kExpAdjBitIncr =
      static_cast<TU>(kIsF32 ? 0x004AFB0Cu : 0x00095F619980C434u);

  const RebindToUnsigned<decltype(d)> du;
  const RebindToSigned<decltype(d)> di;

  constexpr int kNumOfMantFracBits = MantissaBits<T>();
  static_assert(kNumOfMantFracBits > 0, "kNumOfMantFracBits > 0 must be true");

  constexpr TI kExpBias = static_cast<TI>(MaxExponentField<T>() >> 1);
  static_assert(kExpBias > 0, "kExpBias > 0 must be true");

  constexpr TU kExponentMask = ExponentMask<T>();
  constexpr TU kSignificandMask = static_cast<TU>(LimitsMax<TI>());
  constexpr TU kMaxLt2FloatValBits = (kSignificandMask >> 1);
  constexpr TU kPos1FloatValBits = kMaxLt2FloatValBits & kExponentMask;
  constexpr TU kNegInfBits = kExponentMask | SignMask<T>();
  constexpr TU kLessThanMinNormalScaleFactorBits =
      static_cast<TU>(kExpBias + kNumOfMantFracBits) << kNumOfMantFracBits;

  const auto exp_lsb = Set(du, TU{1} << kNumOfMantFracBits);
  const auto exp_bias = Set(di, kExpBias);
  const auto neg_inf = BitCast(d, Set(du, kNegInfBits));

  const auto one = BitCast(d, Set(du, kPos1FloatValBits));
  const auto min_normal = BitCast(d, exp_lsb);

  const auto x_is_zero_or_subnormal = Lt(x, min_normal);
  const auto x_normalize_scale_factor =
      IfThenElse(x_is_zero_or_subnormal,
                 BitCast(d, Set(du, kLessThanMinNormalScaleFactorBits)), one);

  // normalized_x[i] is equal to the normalized value of x[i], which ensures
  // that subnormal numbers are either scaled up to a normal number (which is
  // usually the case on most targets) or flushed to zero (which occurs on Armv7
  // NEON).

  const auto normalized_x = Mul(x, x_normalize_scale_factor);
  const auto x_denormalize_exp = IfThenElseZero(
      RebindMask(di, x_is_zero_or_subnormal), Set(di, -kNumOfMantFracBits));

  const auto normalized_x_bits = BitCast(du, normalized_x);

  // x_mant_exp_adj[i] is equal to the decrement that needs to be made to
  // the mantissa bits to get x_mant[i] in the (sqrt(2) / 2, sqrt(2)) range.
  const auto x_mant_exp_adj = And(
      Xor(normalized_x_bits, Add(normalized_x_bits, Set(du, kExpAdjBitIncr))),
      exp_lsb);

  // If normalized_x[i] is a normal floating-point number, then x_exp[i] is
  // equal to lrint(log2(x[i])).
  const auto x_exp =
      Add(Add(BitCast(di, GetBiasedExponent(normalized_x)), x_denormalize_exp),
          Sub(BitCast(di, ShiftRight<kNumOfMantFracBits>(x_mant_exp_adj)),
              exp_bias));
  const auto x_exp_as_float = ConvertTo(d, x_exp);

  // x_mant[i] is equal to the mantissa of x[i] in the (sqrt(2) / 2, sqrt(2))
  // range.
  const auto x_mant =
      BitCast(d, And(Xor(Or(normalized_x_bits, Set(du, kPos1FloatValBits)),
                         x_mant_exp_adj),
                     Set(du, kMaxLt2FloatValBits)));

  // If normalized_x[i] is a normal floating-point number, then x will be equal
  // to calbn(x_mant[i], x_exp[i]).

  RemoveCvRef<V> log2_x_mant_lo;
  const V log2_x_mant_hi = ExtPrecLog2OfMantForPow(d, x_mant, log2_x_mant_lo);

  const V log2_x_hi0 = Add(x_exp_as_float, log2_x_mant_hi);
  const V log2_x_lo0 =
      Add(Add(Sub(x_exp_as_float, log2_x_hi0), log2_x_mant_hi), log2_x_mant_lo);

  const V log2_x_hi1 = Add(log2_x_hi0, log2_x_lo0);
  const V log2_x_lo1 = Add(Sub(log2_x_hi0, log2_x_hi1), log2_x_lo0);

  const auto x_is_zero = Eq(normalized_x, Zero(d));
  const auto x_is_finite = IsFinite(normalized_x);

  // If normalized_x[i] is a nonzero finite number, then log2(x[i]) will be
  // equal to log2_x_hi1[i] + log2_x_hi0[i].

  // Otherwise, return negative infinity if normalized_x[i] is equal to zero
  // and return normalized_x[i] if normalized_x[i] is non-finite.

  const V log2_x_hi = IfThenElse(
      x_is_finite, IfThenElse(x_is_zero, neg_inf, log2_x_hi1), normalized_x);
  log2_x_lo = IfThenElseZero(AndNot(x_is_zero, x_is_finite), log2_x_lo1);

  return log2_x_hi;
}

// Returns ln(x) in double-double for x > 0; lo = low part.
template <class D, class V = VFromD<D>>
HWY_INLINE V DDLog(D d, V x, V& lo) {
  using T = TFromD<D>;
  constexpr bool kIsF32 = (sizeof(T) == 4);
  const V kLn2Hi = Set(d, kIsF32 ? static_cast<T>(0.6931471824645996f)
                                 : static_cast<T>(0.6931471805599453));
  const V kLn2Lo = Set(d, kIsF32 ? static_cast<T>(-1.9046542121259336e-9f)
                                 : static_cast<T>(2.3190468138462996e-17));
  V log2_lo;
  const V log2_hi = ExtPrecLog2ForPow(d, x, log2_lo);
  return DDMul2(d, log2_hi, log2_lo, kLn2Hi, kLn2Lo, lo);
}

// Returns logGamma(w) in double-double via Stirling's series for
// w >= StirlingLimit(); lo = low part.
template <class D, class V = VFromD<D>>
HWY_INLINE V StirlingLogGamma(D d, V w, V& lo) {
  using T = TFromD<D>;
  constexpr bool kIsF32 = (sizeof(T) == 4);
  GammaImpl<T> impl;
  const V kHalf = Set(d, static_cast<T>(0.5));
  const V kZero = Zero(d);
  const V kHalfLn2PiHi = Set(d, kIsF32 ? static_cast<T>(0.9189385175704956f)
                                       : static_cast<T>(0.9189385332046728));
  const V kHalfLn2PiLo =
      Set(d, kIsF32 ? static_cast<T>(1.563417661998301e-8f)
                    : static_cast<T>(-3.8782941580672414e-17));
  // (w-0.5)*ln(w) - w + 0.5*ln(2pi) + (1/w)*poly(1/w^2).
  const V inv_w = Div(Set(d, static_cast<T>(1.0)), w);
  const V u = Mul(inv_w, inv_w);
  V lnw_lo;
  const V lnw_hi = DDLog(d, w, lnw_lo);
  V hi = DDMul1(d, lnw_hi, lnw_lo, Sub(w, kHalf), lo);
  hi = DDAdd(d, hi, lo, Neg(w), kZero, lo);
  hi = DDAdd(d, hi, lo, kHalfLn2PiHi, kHalfLn2PiLo, lo);
  const V series = Mul(inv_w, impl.StirlingPoly(d, u));
  return DDAdd(d, hi, lo, series, kZero, lo);
}

// Computes signed Gamma(a). For a < 0.5 uses w = 1 - a; then a [2, 3)
// polynomial below kStirlingLimit, Stirling above.
template <class D, class V = VFromD<D>, class M = MFromD<D>>
HWY_INLINE V Gamma(D d, V a) {
  using T = TFromD<D>;
  static_assert(IsFloat<T>(), "Only makes sense for floating-point");
  GammaImpl<T> impl;

  const V kHalf = Set(d, static_cast<T>(0.5));
  const V kOne = Set(d, static_cast<T>(1.0));
  const V kZero = Zero(d);
  const V kPi = Set(d, static_cast<T>(+3.14159265358979323846264));
  const V kTwo = Set(d, static_cast<T>(2.0));
  const V kThree = Set(d, static_cast<T>(3.0));
  const V kStirlingLimit = impl.StirlingLimit(d);

  // Reduce to w >= 0.5: w = 1 - a when a < 0.5.
  const M neg = Lt(a, kHalf);
  V w = MaskedSubOr(a, neg, kOne, a);

  // Shift w into [2, 3) via Gamma(x+1) = x*Gamma(x).
  V wa = w;
  V num_hi = kOne, num_lo = kZero;
  V den_hi = kOne, den_lo = kZero;

  for (int i = 0; i < 2; ++i) {
    const M up = Lt(wa, kTwo);
    V d_lo;
    const V d_hi = DDMul1(d, den_hi, den_lo, wa, d_lo);
    den_hi = IfThenElse(up, d_hi, den_hi);
    den_lo = IfThenElse(up, d_lo, den_lo);
    wa = MaskedAddOr(wa, up, wa, kOne);
  }

  for (int i = 0; i < GammaImpl<T>::kReduceSteps; ++i) {
    const M down = Ge(wa, kThree);
    wa = MaskedSubOr(wa, down, wa, kOne);
    V m_lo;
    const V m_hi = DDMul1(d, num_hi, num_lo, wa, m_lo);
    num_hi = IfThenElse(down, m_hi, num_hi);
    num_lo = IfThenElse(down, m_lo, num_lo);
  }

  const V poly = impl.GammaPoly(d, Sub(wa, Set(d, static_cast<T>(2.5))));
  V np_lo;
  const V np_hi = DDMul1(d, num_hi, num_lo, poly, np_lo);
  V gA_lo;
  const V gamma_a = DDDiv(d, np_hi, np_lo, den_hi, den_lo, gA_lo);

  // Gamma via Stirling logGamma (dd): exp(hi+lo) = exp(hi)*(1+lo).
  V lo;
  const V hi = StirlingLogGamma(d, w, lo);
  const V exp_hi = Exp(d, hi);
  const V gamma_b = MulAdd(exp_hi, lo, exp_hi);

  const V gamma_w = IfThenElse(Lt(w, kStirlingLimit), gamma_a, gamma_b);

  // For a < 0.5: Gamma(a) = pi / (sin(pi*a) * Gamma(w)).
  const V ra = Round(a);
  const V frac = Sub(a, ra);
  const V s_mag = Sin(d, Mul(kPi, frac));
  const RebindToSigned<decltype(d)> di;
  const M odd =
      RebindMask(d, Ne(And(ConvertTo(di, ra), Set(di, 1)), Zero(di)));
  const V s_signed = IfThenElse(odd, Neg(s_mag), s_mag);
  const V refl = Div(kPi, Mul(s_signed, gamma_w));

  return IfThenElse(neg, refl, gamma_w);
}

// SinCos
// Based on "sse_mathfun.h", by Julien Pommier
// http://gruntthepeon.free.fr/ssemath/

// Third degree poly
template <class D, class V = VFromD<D>>
HWY_INLINE void SinCos3(D d, TFromD<D> dp1, TFromD<D> dp2, TFromD<D> dp3, V x,
                        V& s, V& c) {
  using T = TFromD<D>;
  using TI = MakeSigned<T>;
  using DI = Rebind<TI, D>;
  const DI di;
  using VI = decltype(Zero(di));
  using M = Mask<D>;

  static constexpr size_t bits = sizeof(TI) * 8;
  const VI sign_mask = SignBit(di);
  const VI ci_0 = Zero(di);
  const VI ci_1 = Set(di, 1);
  const VI ci_2 = Set(di, 2);
  const VI ci_4 = Set(di, 4);
  const V cos_p0 = Set(d, ConvertScalarTo<T>(2.443315711809948E-005));
  const V cos_p1 = Set(d, ConvertScalarTo<T>(-1.388731625493765E-003));
  const V cos_p2 = Set(d, ConvertScalarTo<T>(4.166664568298827E-002));
  const V sin_p0 = Set(d, ConvertScalarTo<T>(-1.9515295891E-4));
  const V sin_p1 = Set(d, ConvertScalarTo<T>(8.3321608736E-3));
  const V sin_p2 = Set(d, ConvertScalarTo<T>(-1.6666654611E-1));
  const V FOPI = Set(d, ConvertScalarTo<T>(1.27323954473516));  // 4 / M_PI
  const V DP1 = Set(d, dp1);
  const V DP2 = Set(d, dp2);
  const V DP3 = Set(d, dp3);

  V xmm1, xmm2, sign_bit_sin, y;
  VI imm0, imm2, imm4;

  sign_bit_sin = x;
  x = Abs(x);

  /* extract the sign bit (upper one) */
  sign_bit_sin = And(sign_bit_sin, BitCast(d, sign_mask));

  /* scale by 4/Pi */
  y = Mul(x, FOPI);

  /* store the integer part of y in imm2 */
  imm2 = ConvertTo(di, y);

  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = Add(imm2, ci_1);
  imm2 = AndNot(ci_1, imm2);

  y = ConvertTo(d, imm2);
  imm4 = imm2;

  /* get the swap sign flag for the sine */
  imm0 = And(imm2, ci_4);
  imm0 = ShiftLeft<bits - 3>(imm0);

  V swap_sign_bit_sin = BitCast(d, imm0);

  /* get the polynomial selection mask for the sine*/
  imm2 = And(imm2, ci_2);
  M poly_mask = RebindMask(d, Eq(imm2, ci_0));

  /* The magic pass: "Extended precision modular arithmetic"
  x = ((x - y * DP1) - y * DP2) - y * DP3; */
  x = MulAdd(y, DP1, x);
  x = MulAdd(y, DP2, x);
  x = MulAdd(y, DP3, x);

  imm4 = Sub(imm4, ci_2);
  imm4 = AndNot(imm4, ci_4);
  imm4 = ShiftLeft<bits - 3>(imm4);

  V sign_bit_cos = BitCast(d, imm4);

  sign_bit_sin = Xor(sign_bit_sin, swap_sign_bit_sin);

  /* Evaluate the first polynomial  (0 <= x <= Pi/4) */
  V z = Mul(x, x);

  y = MulAdd(cos_p0, z, cos_p1);
  y = MulAdd(y, z, cos_p2);
  y = Mul(y, z);
  y = Mul(y, z);
  y = NegMulAdd(z, Set(d, 0.5f), y);
  y = Add(y, Set(d, 1));

  /* Evaluate the second polynomial  (Pi/4 <= x <= 0) */
  V y2 = MulAdd(sin_p0, z, sin_p1);
  y2 = MulAdd(y2, z, sin_p2);
  y2 = Mul(y2, z);
  y2 = MulAdd(y2, x, x);

  /* select the correct result from the two polynomials */
  xmm1 = IfThenElse(poly_mask, y2, y);
  xmm2 = IfThenElse(poly_mask, y, y2);

  /* update the sign */
  s = Xor(xmm1, sign_bit_sin);
  c = Xor(xmm2, sign_bit_cos);
}

// Sixth degree poly
template <class D, class V = VFromD<D>>
HWY_INLINE void SinCos6(D d, TFromD<D> dp1, TFromD<D> dp2, TFromD<D> dp3, V x,
                        V& s, V& c) {
  using T = TFromD<D>;
  using TI = MakeSigned<T>;
  using DI = Rebind<TI, D>;
  const DI di;
  using VI = decltype(Zero(di));
  using M = Mask<D>;

  static constexpr size_t bits = sizeof(TI) * 8;
  const VI sign_mask = SignBit(di);
  const VI ci_0 = Zero(di);
  const VI ci_1 = Set(di, 1);
  const VI ci_2 = Set(di, 2);
  const VI ci_4 = Set(di, 4);
  const V cos_p0 = Set(d, ConvertScalarTo<T>(-1.13585365213876817300E-11));
  const V cos_p1 = Set(d, ConvertScalarTo<T>(2.08757008419747316778E-9));
  const V cos_p2 = Set(d, ConvertScalarTo<T>(-2.75573141792967388112E-7));
  const V cos_p3 = Set(d, ConvertScalarTo<T>(2.48015872888517045348E-5));
  const V cos_p4 = Set(d, ConvertScalarTo<T>(-1.38888888888730564116E-3));
  const V cos_p5 = Set(d, ConvertScalarTo<T>(4.16666666666665929218E-2));
  const V sin_p0 = Set(d, ConvertScalarTo<T>(1.58962301576546568060E-10));
  const V sin_p1 = Set(d, ConvertScalarTo<T>(-2.50507477628578072866E-8));
  const V sin_p2 = Set(d, ConvertScalarTo<T>(2.75573136213857245213E-6));
  const V sin_p3 = Set(d, ConvertScalarTo<T>(-1.98412698295895385996E-4));
  const V sin_p4 = Set(d, ConvertScalarTo<T>(8.33333333332211858878E-3));
  const V sin_p5 = Set(d, ConvertScalarTo<T>(-1.66666666666666307295E-1));
  const V FOPI =  // 4 / M_PI
      Set(d, ConvertScalarTo<T>(1.2732395447351626861510701069801148));
  const V DP1 = Set(d, dp1);
  const V DP2 = Set(d, dp2);
  const V DP3 = Set(d, dp3);

  V xmm1, xmm2, sign_bit_sin, y;
  VI imm0, imm2, imm4;

  sign_bit_sin = x;
  x = Abs(x);

  /* extract the sign bit (upper one) */
  sign_bit_sin = And(sign_bit_sin, BitCast(d, sign_mask));

  /* scale by 4/Pi */
  y = Mul(x, FOPI);

  /* store the integer part of y in imm2 */
  imm2 = ConvertTo(di, y);

  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = Add(imm2, ci_1);
  imm2 = AndNot(ci_1, imm2);

  y = ConvertTo(d, imm2);
  imm4 = imm2;

  /* get the swap sign flag for the sine */
  imm0 = And(imm2, ci_4);
  imm0 = ShiftLeft<bits - 3>(imm0);

  V swap_sign_bit_sin = BitCast(d, imm0);

  /* get the polynomial selection mask for the sine*/
  imm2 = And(imm2, ci_2);
  M poly_mask = RebindMask(d, Eq(imm2, ci_0));

  /* The magic pass: "Extended precision modular arithmetic"
    x = ((x - y * DP1) - y * DP2) - y * DP3; */
  x = MulAdd(y, DP1, x);
  x = MulAdd(y, DP2, x);
  x = MulAdd(y, DP3, x);

  imm4 = Sub(imm4, ci_2);
  imm4 = AndNot(imm4, ci_4);
  imm4 = ShiftLeft<bits - 3>(imm4);

  V sign_bit_cos = BitCast(d, imm4);
  sign_bit_sin = Xor(sign_bit_sin, swap_sign_bit_sin);

  /* Evaluate the first polynomial  (0 <= x <= Pi/4) */
  V z = Mul(x, x);

  y = MulAdd(cos_p0, z, cos_p1);
  y = MulAdd(y, z, cos_p2);
  y = MulAdd(y, z, cos_p3);
  y = MulAdd(y, z, cos_p4);
  y = MulAdd(y, z, cos_p5);
  y = Mul(y, z);
  y = Mul(y, z);
  y = NegMulAdd(z, Set(d, 0.5f), y);
  y = Add(y, Set(d, 1.0f));

  /* Evaluate the second polynomial  (Pi/4 <= x <= 0) */
  V y2 = MulAdd(sin_p0, z, sin_p1);
  y2 = MulAdd(y2, z, sin_p2);
  y2 = MulAdd(y2, z, sin_p3);
  y2 = MulAdd(y2, z, sin_p4);
  y2 = MulAdd(y2, z, sin_p5);
  y2 = Mul(y2, z);
  y2 = MulAdd(y2, x, x);

  /* select the correct result from the two polynomials */
  xmm1 = IfThenElse(poly_mask, y2, y);
  xmm2 = IfThenElse(poly_mask, y, y2);

  /* update the sign */
  s = Xor(xmm1, sign_bit_sin);
  c = Xor(xmm2, sign_bit_cos);
}

template <>
struct SinCosImpl<float> {
  template <class D, class V = VFromD<D>, HWY_IF_F32_D(D)>
  HWY_INLINE void SinCos(D d, V x, V& s, V& c) {
    SinCos3(d, -0.78515625f, -2.4187564849853515625e-4f,
            -3.77489497744594108e-8f, x, s, c);
  }
};

#if HWY_HAVE_FLOAT64 && HWY_HAVE_INTEGER64
template <>
struct SinCosImpl<double> {
  template <class D, class V = VFromD<D>, HWY_IF_F64_D(D)>
  HWY_INLINE void SinCos(D d, V x, V& s, V& c) {
    SinCos6(d, -7.85398125648498535156E-1, -3.77489470793079817668E-8,
            -2.69515142907905952645E-15, x, s, c);
  }
};
#endif

}  // namespace impl

template <class D, class V>
HWY_INLINE V Acos(const D d, V x) {
  using T = TFromD<D>;

  const V kZero = Zero(d);
  const V kHalf = Set(d, static_cast<T>(+0.5));
  const V kPi = Set(d, static_cast<T>(+3.14159265358979323846264));
  const V kPiOverTwo = Set(d, static_cast<T>(+1.57079632679489661923132169));

  const V sign_x = And(SignBit(d), x);
  const V abs_x = Xor(x, sign_x);
  const auto mask = Lt(abs_x, kHalf);
  const V yy =
      IfThenElse(mask, Mul(abs_x, abs_x), NegMulAdd(abs_x, kHalf, kHalf));
  const V y = IfThenElse(mask, abs_x, Sqrt(yy));

  impl::AsinImpl<T> impl;
  const V t = Mul(impl.AsinPoly(d, yy, y), Mul(y, yy));

  const V t_plus_y = Add(t, y);
  const V z =
      IfThenElse(mask, Sub(kPiOverTwo, Add(Xor(y, sign_x), Xor(t, sign_x))),
                 Add(t_plus_y, t_plus_y));
  return IfThenElse(Or(mask, Ge(x, kZero)), z, Sub(kPi, z));
}

template <class D, class V>
HWY_INLINE V Acosh(const D d, V x) {
  using T = TFromD<D>;

  const V kLarge = Set(d, static_cast<T>(268435456.0));
  const V kLog2 = Set(d, static_cast<T>(0.693147180559945286227));
  const V kOne = Set(d, static_cast<T>(+1.0));
  const V kTwo = Set(d, static_cast<T>(+2.0));

  const auto is_x_large = Gt(x, kLarge);
  const auto is_x_gt_2 = Gt(x, kTwo);

  const V x_minus_1 = Sub(x, kOne);
  const V y0 = MulSub(kTwo, x, Div(kOne, Add(Sqrt(MulSub(x, x, kOne)), x)));
  const V y1 =
      Add(Sqrt(MulAdd(x_minus_1, kTwo, Mul(x_minus_1, x_minus_1))), x_minus_1);
  const V y2 =
      IfThenElse(is_x_gt_2, IfThenElse(is_x_large, x, y0), Add(y1, kOne));
  const V z = impl::Log<D, V, /*kAllowSubnormals=*/false>(d, y2);

  const auto is_pole = Eq(y2, kOne);
  const auto divisor = Sub(IfThenZeroElse(is_pole, y2), kOne);
  return Add(IfThenElse(is_x_gt_2, z,
                        IfThenElse(is_pole, y1, Div(Mul(z, y1), divisor))),
             IfThenElseZero(is_x_large, kLog2));
}

template <class D, class V>
HWY_INLINE V Asin(const D d, V x) {
  using T = TFromD<D>;

  const V kHalf = Set(d, static_cast<T>(+0.5));
  const V kTwo = Set(d, static_cast<T>(+2.0));
  const V kPiOverTwo = Set(d, static_cast<T>(+1.57079632679489661923132169));

  const V sign_x = And(SignBit(d), x);
  const V abs_x = Xor(x, sign_x);
  const auto mask = Lt(abs_x, kHalf);
  const V yy =
      IfThenElse(mask, Mul(abs_x, abs_x), NegMulAdd(abs_x, kHalf, kHalf));
  const V y = IfThenElse(mask, abs_x, Sqrt(yy));

  impl::AsinImpl<T> impl;
  const V z0 = MulAdd(impl.AsinPoly(d, yy, y), Mul(yy, y), y);
  const V z1 = NegMulAdd(z0, kTwo, kPiOverTwo);
  return Or(IfThenElse(mask, z0, z1), sign_x);
}

template <class D, class V>
HWY_INLINE V Asinh(const D d, V x) {
  using T = TFromD<D>;

  const V kSmall = Set(d, static_cast<T>(1.0 / 268435456.0));
  const V kLarge = Set(d, static_cast<T>(268435456.0));
  const V kLog2 = Set(d, static_cast<T>(0.693147180559945286227));
  const V kOne = Set(d, static_cast<T>(+1.0));
  const V kTwo = Set(d, static_cast<T>(+2.0));

  const V sign_x = And(SignBit(d), x);  // Extract the sign bit
  const V abs_x = Xor(x, sign_x);

  const auto is_x_large = Gt(abs_x, kLarge);
  const auto is_x_lt_2 = Lt(abs_x, kTwo);

  const V x2 = Mul(x, x);
  const V sqrt_x2_plus_1 = Sqrt(Add(x2, kOne));

  const V y0 = MulAdd(abs_x, kTwo, Div(kOne, Add(sqrt_x2_plus_1, abs_x)));
  const V y1 = Add(Div(x2, Add(sqrt_x2_plus_1, kOne)), abs_x);
  const V y2 =
      IfThenElse(is_x_lt_2, Add(y1, kOne), IfThenElse(is_x_large, abs_x, y0));
  const V z = impl::Log<D, V, /*kAllowSubnormals=*/false>(d, y2);

  const auto is_pole = Eq(y2, kOne);
  const auto divisor = Sub(IfThenZeroElse(is_pole, y2), kOne);
  const auto large = IfThenElse(is_pole, y1, Div(Mul(z, y1), divisor));
  const V y = IfThenElse(Lt(abs_x, kSmall), x, large);
  return Or(Add(IfThenElse(is_x_lt_2, y, z), IfThenElseZero(is_x_large, kLog2)),
            sign_x);
}

template <class D, class V>
HWY_INLINE V Atan(const D d, V x) {
  using T = TFromD<D>;

  const V kOne = Set(d, static_cast<T>(+1.0));
  const V kPiOverTwo = Set(d, static_cast<T>(+1.57079632679489661923132169));

  const V sign = And(SignBit(d), x);
  const V abs_x = Xor(x, sign);
  const auto mask = Gt(abs_x, kOne);

  impl::AtanImpl<T> impl;
  const auto divisor = IfThenElse(mask, abs_x, kOne);
  const V y = impl.AtanPoly(d, IfThenElse(mask, Div(kOne, divisor), abs_x));
  return Or(IfThenElse(mask, Sub(kPiOverTwo, y), y), sign);
}

template <class D, class V>
HWY_INLINE V Atan2(const D d, V y, V x) {
  using T = TFromD<D>;
  using M = MFromD<D>;

  const V kPi = Set(d, static_cast<T>(3.14159265358979323846264));
  const V kPiOverTwo = Set(d, static_cast<T>(1.57079632679489661923132169));
  const V kOne = Set(d, static_cast<T>(1.0));
  const V k0 = Zero(d);

  const V ax = Abs(x);
  const V ay = Abs(y);

  // Pre-sort to ensure num <= den, mapping the input to the [0, 1] range.
  // This avoids a second division that would otherwise occur inside Atan()
  // flow for domain reduction.
  const V num = Min(ax, ay);
  const V den = Max(ax, ay);

  const M is_inf = IsInf(num);
  V mapped_y = MaskedDivOr(k0, Ne(den, k0), num, den);
  mapped_y = IfThenElse(is_inf, kOne, mapped_y);

  impl::AtanImpl<T> impl;
  const V poly = impl.AtanPoly(d, mapped_y);

  const M ay_gt_ax = Gt(ay, ax);
  V angle = MaskedSubOr(poly, ay_gt_ax, kPiOverTwo, poly);

  const M x_neg = Lt(x, k0);
  angle = MaskedSubOr(angle, x_neg, kPi, angle);

  const M is_nan = IsEitherNaN(y, x);
  return IfThenElse(is_nan, NaN(d), CopySign(angle, y));
}

template <class D, class V>
HWY_INLINE V Atanh(const D d, V x) {
  using T = TFromD<D>;

  const V kHalf = Set(d, static_cast<T>(+0.5));
  const V kOne = Set(d, static_cast<T>(+1.0));

  const V sign = And(SignBit(d), x);  // Extract the sign bit
  const V abs_x = Xor(x, sign);
  return Mul(Log1p(d, Div(Add(abs_x, abs_x), Sub(kOne, abs_x))),
             Xor(kHalf, sign));
}

namespace impl {

// Barrett reduction (n/3) via MulHigh by 0x5556, repartitions to u16 lanes
template <class DI, class VI = decltype(Zero(DI()))>
HWY_INLINE void CbrtDivMod3(DI di, VI exp_shifted, VI& div, VI& mod) {
  const Repartition<uint16_t, DI> du16;
  using VU16 = decltype(Zero(du16));
  const VU16 exp_u16 = BitCast(du16, exp_shifted);
  const VU16 div_u16 =
      MulHigh(exp_u16, Set(du16, static_cast<uint16_t>(0x5556)));
  const VU16 mod_u16 =
      Sub(exp_u16, Mul(div_u16, Set(du16, static_cast<uint16_t>(3))));
  div = BitCast(di, div_u16);
  mod = BitCast(di, mod_u16);
}

// Single-lane fallback, Barrett reduction on the int lane with no Repartition
template <class DI, class VI = decltype(Zero(DI()))>
HWY_INLINE void CbrtDivMod3Scalar(DI di, VI exp_shifted, VI& div, VI& mod) {
  using TI = TFromD<DI>;
  div = ShiftRight<16>(Mul(exp_shifted, Set(di, static_cast<TI>(0x5556))));
  mod = Sub(exp_shifted, Mul(div, Set(di, static_cast<TI>(3))));
}

}  // namespace impl

// Modified from BSD-licensed code
// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// See https://github.com/libjxl/libjxl/blob/main/LICENSE.
template <bool kHandleSubnormals, class D, class V>
HWY_INLINE V Cbrt(const D d, V x) {
  using T = TFromD<D>;

  const V sign = And(SignBit(d), x);
  const V abs_x = Xor(x, sign);
  x = abs_x;

  constexpr bool kIsF32 = (sizeof(T) == 4);

  MFromD<D> is_denormal;
  if constexpr (kHandleSubnormals) {
    const V kMinNormal = Set(d, SmallestNormal<T>());
    // Exponent to scale subnormals that is divisible by 3, 2^24 or 2^54
    const V kScale = Set(d, kIsF32 ? static_cast<T>(16777216.0f)
                                   : static_cast<T>(18014398509481984.0));
    is_denormal = Lt(x, kMinNormal);
    x = MaskedMulOr(x, is_denormal, x, kScale);
  } else {
    (void)is_denormal;
  }

  V y;
  const RebindToSigned<D> di;
  using TI = TFromD<decltype(di)>;
  using VI = decltype(Zero(di));

  const VI x_int = BitCast(di, x);

  // Extract exponent and shift (3*128 or 3*512) to keep non-negative for
  // Barrett reduction
  const VI exp_shifted =
      Add(ShiftRight<kIsF32 ? 23 : 52>(x_int),
          Set(di, kIsF32 ? static_cast<TI>(257) : static_cast<TI>(513)));

  VI exp_shifted_div_3;
  VI exp_mod_3;
  if constexpr ((HWY_MAX_LANES_D(D) > 1 && !HWY_HAVE_SCALABLE) ||
                (HWY_HAVE_SCALABLE && detail::IsFull(d))) {
    impl::CbrtDivMod3(di, exp_shifted, exp_shifted_div_3, exp_mod_3);
  } else if constexpr (HWY_MAX_LANES_D(D) > 1) {
    if (Lanes(d) > 1) {
      impl::CbrtDivMod3(di, exp_shifted, exp_shifted_div_3, exp_mod_3);
    } else {
      impl::CbrtDivMod3Scalar(di, exp_shifted, exp_shifted_div_3, exp_mod_3);
    }
  } else {
    impl::CbrtDivMod3Scalar(di, exp_shifted, exp_shifted_div_3, exp_mod_3);
  }

  // Undo constant shift to ensure non negative
  const VI neg_exp_div_3 =
      Sub(Set(di, kIsF32 ? static_cast<TI>(128) : static_cast<TI>(512)),
          exp_shifted_div_3);
  // Combine exp mod 3 index with the top mantissa bits
  const VI top_mant =
      And(ShiftRight<kIsF32 ? 22 : 50>(x_int),
          Set(di, kIsF32 ? static_cast<TI>(1) : static_cast<TI>(3)));
  const VI idx = Add(ShiftLeft<kIsF32 ? 1 : 2>(exp_mod_3), top_mant);

  V r;
  if constexpr (kIsF32) {
    // (1/cbrt(lo) + 1/cbrt(hi))/2 over 6 bins of [1,8)
    HWY_ALIGN static constexpr float initial_guess[8] = {
        0.92807984f, 0.81504166f, 0.73603648f, 0.65004617f,
        0.58375800f, 0.51406258f, 0.0f,        0.0f};
    if constexpr ((HWY_MAX_LANES_D(D) >= 4 && !HWY_HAVE_SCALABLE) ||
                  (HWY_HAVE_SCALABLE && sizeof(T) == 4 && detail::IsFull(d))) {
      r = Lookup8(d, initial_guess, idx);
    } else {
      r = GatherIndex(d, initial_guess, idx);
    }
  } else {
    // (1/cbrt(lo) + 1/cbrt(hi))/2 over 12 bins of [1,8)
    HWY_ALIGN static constexpr double initial_guess[12] = {
        9.6415888336127797e-01, 9.0094911572942737e-01, 8.5170349905127118e-01,
        8.1176352967517162e-01, 7.6525341285608861e-01, 7.1508378703935604e-01,
        6.7599751517949214e-01, 6.4429714047789299e-01, 6.0738203629500487e-01,
        5.6756237789583885e-01, 5.3653958336190732e-01, 5.1137897928735510e-01};
    r = GatherIndex(d, initial_guess, idx);
  }

  // Apply 2^(-exp/3) to scale lookup result to 1/cbrt(x).
  r = MulByPow2(r, neg_exp_div_3);

  const V kOneThird = Set(d, static_cast<T>(1.0 / 3.0));
  const V kFourThirds = Set(d, static_cast<T>(4.0 / 3.0));
  const V x_div_3 = Mul(kOneThird, x);
  constexpr size_t kIters = kIsF32 ? 2 : 3;
  // Newton iteration for 1/cbrt(x): r = r * (4/3 - (x/3) * r^3).
  for (size_t i = 0; i < kIters; ++i) {
    const V r2 = Mul(r, r);
    const V x_div_3_r = Mul(x_div_3, r);
    r = Mul(r, NegMulAdd(x_div_3_r, r2, kFourThirds));
  }

  // Fused finalizer: y = r*r*x * (5/3 - (2/3) * r*r*r*x).
  const V kFiveThirds = Set(d, static_cast<T>(5.0 / 3.0));
  const V kTwoThirds = Set(d, static_cast<T>(2.0 / 3.0));
  const V y0 = Mul(Mul(r, r), x);
  const V h = Mul(y0, r);
  y = Mul(y0, NegMulAdd(kTwoThirds, h, kFiveThirds));

  if constexpr (kHandleSubnormals) {
    // 1 / cbrt(kScale), 1 / 2^8 or 1 / 2^18
    const auto kUnscale = Set(d, kIsF32 ? static_cast<T>(1.0 / 256.0)
                                        : static_cast<T>(1.0 / 262144.0));
    y = MaskedMulOr(y, is_denormal, y, kUnscale);
  }

  y = IfThenElse(Or(Eq(abs_x, Zero(d)), Not(IsFinite(abs_x))), abs_x, y);

  y = Or(y, sign);
  return y;
}

template <class D, class V>
HWY_INLINE V Cos(const D d, V x) {
  using T = TFromD<D>;
  impl::CosSinImpl<T> impl;

  // Float Constants
  const V kOneOverPi = Set(d, static_cast<T>(0.31830988618379067153));

  // Integer Constants
  const Rebind<int32_t, D> di32;
  using VI32 = decltype(Zero(di32));
  const VI32 kOne = Set(di32, 1);

  const V y = Abs(x);  // cos(x) == cos(|x|)

  // Compute the quadrant, q = int(|x| / pi) * 2 + 1
  const VI32 q = Add(ShiftLeft<1>(impl.ToInt32(d, Mul(y, kOneOverPi))), kOne);

  // Reduce range, apply sign, and approximate.
  return impl.Poly(
      d, Xor(impl.CosReduce(d, y, q), impl.CosSignFromQuadrant(d, q)));
}

// Erf
// Based on Cephes erff/erf by Stephen Moshier (public domain, 1989)
// See https://www.netlib.org/cephes/ - single/ndtrf.c (f32), cprob/ndtr.c (f64)
template <class D, class V>
HWY_INLINE V Erf(const D d, V x) {
  using T = TFromD<D>;
  impl::ErfImpl<T> impl;
  const V kOne = Set(d, static_cast<T>(1));
  const V kLimit = impl.Limit(d);

  const V sign = And(SignBit(d), x);
  x = Xor(x, sign);

  const V x_clamped = Min(x, kLimit);
  const V z = Mul(x_clamped, x_clamped);

  const V small = impl.SmallErf(d, x_clamped, z);
  const V exp_neg_z = Exp(d, Neg(z));
  const V large = NegMulAdd(exp_neg_z, impl.ErfcFactor(d, x_clamped, z), kOne);

  const auto is_small = Lt(x_clamped, kOne);
  V result = IfThenElse(is_small, small, large);
  result = IfThenElse(IsNaN(x), x, result);

  return Or(result, sign);
}

template <class D, class V>
HWY_INLINE V Exp(const D d, V x) {
  using T = TFromD<D>;

  const V kHalf = Set(d, static_cast<T>(+0.5));
  const V kLowerBound =
      Set(d, static_cast<T>((sizeof(T) == 4 ? -104.0 : -1000.0)));
  const V kNegZero = Set(d, static_cast<T>(-0.0));
  const V kOne = Set(d, static_cast<T>(+1.0));
  const V kOneOverLog2 = Set(d, static_cast<T>(+1.442695040888963407359924681));

  impl::ExpImpl<T> impl;

  // q = static_cast<int32>((x / log(2)) + ((x < 0) ? -0.5 : +0.5))
  const auto q =
      impl.ToInt32(d, MulAdd(x, kOneOverLog2, Or(kHalf, And(x, kNegZero))));

  // Reduce, approximate, and then reconstruct.
  const V y = impl.LoadExpShortRange(
      d, Add(impl.ExpPoly(d, impl.ExpReduce(d, x, q)), kOne), q);
  return IfThenElseZero(Ge(x, kLowerBound), y);
}

template <class D, class V>
HWY_INLINE V Exp2(const D d, V x) {
  using T = TFromD<D>;

  const V kLowerBound =
      Set(d, static_cast<T>((sizeof(T) == 4 ? -150.0 : -1075.0)));
  const V kOne = Set(d, static_cast<T>(+1.0));

  impl::ExpImpl<T> impl;

  // q = static_cast<int32_t>(std::lrint(x))
  const auto q = impl.ToNearestInt32(d, x);

  // Reduce, approximate, and then reconstruct.
  const V y = impl.LoadExpShortRange(
      d, Add(impl.ExpPoly(d, impl.Exp2Reduce(d, x, q)), kOne), q);
  return IfThenElseZero(Ge(x, kLowerBound), y);
}

template <class D, class V>
HWY_INLINE V Expm1(const D d, V x) {
  using T = TFromD<D>;

  const V kHalf = Set(d, static_cast<T>(+0.5));
  const V kLowerBound =
      Set(d, static_cast<T>((sizeof(T) == 4 ? -104.0 : -1000.0)));
  const V kLn2Over2 = Set(d, static_cast<T>(+0.346573590279972654708616));
  const V kNegOne = Set(d, static_cast<T>(-1.0));
  const V kNegZero = Set(d, static_cast<T>(-0.0));
  const V kOne = Set(d, static_cast<T>(+1.0));
  const V kOneOverLog2 = Set(d, static_cast<T>(+1.442695040888963407359924681));

  impl::ExpImpl<T> impl;

  // q = static_cast<int32>((x / log(2)) + ((x < 0) ? -0.5 : +0.5))
  const auto q =
      impl.ToInt32(d, MulAdd(x, kOneOverLog2, Or(kHalf, And(x, kNegZero))));

  // Reduce, approximate, and then reconstruct.
  const V y = impl.ExpPoly(d, impl.ExpReduce(d, x, q));
  const V z = IfThenElse(Lt(Abs(x), kLn2Over2), y,
                         Sub(impl.LoadExpShortRange(d, Add(y, kOne), q), kOne));
  return IfThenElse(Lt(x, kLowerBound), kNegOne, z);
}

template <class D, class V>
HWY_INLINE V Log(const D d, V x) {
  return impl::Log<D, V, /*kAllowSubnormals=*/true>(d, x);
}

template <class D, class V>
HWY_INLINE V Log10(const D d, V x) {
  using T = TFromD<D>;
  return Mul(Log(d, x), Set(d, static_cast<T>(0.4342944819032518276511)));
}

template <class D, class V>
HWY_INLINE V Log1p(const D d, V x) {
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
  const auto non_pole =
      Mul(impl::Log<D, V, /*kAllowSubnormals=*/false>(d, y), div_res);
  return IfThenElse(not_pole, non_pole, x);
}

template <class D, class V>
HWY_INLINE V Log2(const D d, V x) {
  using T = TFromD<D>;
  return Mul(Log(d, x), Set(d, static_cast<T>(1.44269504088896340735992)));
}

template <class D, class V>
HWY_INLINE V Pow(D d, V a, V b) {
  using T = TFromD<decltype(d)>;
  using TI = MakeSigned<T>;

  const RebindToSigned<decltype(d)> di;

  const auto kZero = Zero(d);
  const auto kOne = Set(d, static_cast<T>(1));

  const auto a_is_nonzero_negative_finite =
      And(Gt(a, Set(d, NegativeInfOrLowestValue<T>())), Lt(a, kZero));

  const auto abs_b = Abs(b);
  const auto b_int = ConvertTo(di, b);
  const auto b_is_integer_or_inf =
      Or(Eq(b, ConvertTo(d, b_int)), Ge(abs_b, Set(d, MantissaEnd<T>())));

  // If a[i] is negative (including negative zero) and b[i] is an odd integer,
  // then the result of pow(a[i], b[i]) will be negative.

  // b[i] is an odd integer if and only if b_int[i] != LimitsMax<TI>(),
  // (b_int[i] & 1) != 0, and b_is_integer_or_inf[i] are all true.

  // Otherwise, the result of pow(a[i], b[i]) will be positive.

  // result_sign[i] is equal to the sign of pow(a[i], b[i]) in the sign bit.
  const auto result_sign = BitCast(
      d, And(BitCast(di, a),
             ShiftLeft<(sizeof(TI) * 8 - 1)>(
                 And(IfThenZeroElse(Eq(b_int, Set(di, LimitsMax<TI>())), b_int),
                     IfThenElseZero(RebindMask(di, b_is_integer_or_inf),
                                    Set(di, 1))))));

  // If -inf < a[i] < 0 and b[i] is not an integer or infinity, then force a2[i]
  // to NaN as the result of pow(a[i], b[i]) will be NaN in this case.

  // Otherwise, if b[i] == 0, force a2[i] to 1 as pow(a[i], 0) is equal to 1 for
  // all values of a[i].

  // If neither of the above cases are true, then a2[i] will be equal to |a[i]|.

  // If |a[i]| == 1, then force b2[i] to 0 as pow(1, b[i]) is equal to 1 for all
  // values of a[i] and as |pow(-1, b[i])| == 1 is true if b[i] is an integer or
  // infinity.

  const auto abs_a = Abs(a);
  const auto a2 =
      IfThenElse(AndNot(b_is_integer_or_inf, a_is_nonzero_negative_finite),
                 NaN(d), IfThenElse(Eq(abs_b, kZero), kOne, abs_a));
  const auto b2 = IfThenZeroElse(Eq(abs_a, kOne), b);

  // The absolute value of the result is equal to a2[i] raised to b2[i].

  // Compute log2(a2[i]) using extra precision to ensure that extra accuracy
  // is there in the fractional part of log2(a2[i]) * b2[i] if
  // |log2(a2[i]) * b2[i]| > 0.5.

  VFromD<D> log2_a2_lo;
  const auto log2_a2_hi = impl::ExtPrecLog2ForPow(d, a2, log2_a2_lo);

  constexpr TI kExpBias = static_cast<TI>(MaxExponentField<T>() >> 1);
  static_assert(kExpBias > 0, "kExpBias > 0 must be true");

  constexpr int kNumOfMantFracBits = MantissaBits<T>();
  static_assert(kNumOfMantFracBits > 0, "kNumOfMantFracBits > 0 must be true");

  constexpr TI kMinInRangeExp = -kExpBias - kNumOfMantFracBits - 1;
  static_assert(kMinInRangeExp < 0, "kMinInRangeExp < 0 must be true");

  constexpr TI kMaxInRangeExp = kExpBias + 2;
  static_assert(kMaxInRangeExp > 0, "kMaxInRangeExp > 0 must be true");

  const auto min_in_range_exp = Set(d, static_cast<T>(kMinInRangeExp));
  const auto max_in_range_exp = Set(d, static_cast<T>(kMaxInRangeExp));

  const auto base2_exp0 = Mul(log2_a2_hi, b2);
  const auto base2_exp0_is_nan = IsNaN(base2_exp0);

  // If base2_exp0[i] is NaN, clamped_base2_exp0[i] will be equal to NaN.
  // Otherwise, if base2_exp0[i] is non-NaN, then clamped_base2_exp0[i] will be
  // equal to min(max(base2_exp0[i], kMinInRangeExp), kMaxInRangeExp).
  const auto clamped_base2_exp0 =
      Clamp(IfThenZeroElse(base2_exp0_is_nan, base2_exp0), min_in_range_exp,
            max_in_range_exp);

  // If base2_exp0[i] is equal to clamped_base2_exp0[i], then base2_exp1[i] will
  // be equal to the lower bits of log2(a2[i]) * b2[i], which ensures that
  // base_e_exp[i] has sufficient accuracy if |log2(a2[i]) * b2[i]| > 0.5.

  // Otherwise, if base2_exp0[i] is not equal to clamped_base2_exp0[i], then
  // either base2_exp0[i] is too small (which will result in a zero result),
  // base2_exp0[i] is too large (which will result in an infinite result), or
  // base2_exp0[i] is NaN (which will result in a NaN result).
  const auto base2_exp1 = IfThenElseZero(
      Eq(base2_exp0, clamped_base2_exp0),
      MulAdd(log2_a2_lo, b2,
             impl::InRangeLoProduct(log2_a2_hi, b2, clamped_base2_exp0)));

  impl::ExpImpl<T> exp_impl;

  // Compute exp2(clamped_base2_exp0[i] + base2_exp1[i])

  // Reduce, approximate, and then reconstruct.
  const auto q = exp_impl.ToNearestInt32(d, clamped_base2_exp0);
  const auto base_e_exp =
      exp_impl.Exp2ReduceForPow(d, clamped_base2_exp0, base2_exp1, q);

  // abs_result[i] is equal to the result of |pow(a[i], b[i])|
  const auto abs_result =
      IfThenElse(base2_exp0_is_nan, base2_exp0,
                 exp_impl.LoadExpShortRange(
                     d, Add(exp_impl.ExpPoly(d, base_e_exp), kOne), q));

  // Bitwise OR result_sign[i] to abs_result[i] to get the final result
  const auto result = Or(abs_result, result_sign);

  return result;
}

template <class D, class V>
HWY_INLINE V Sin(const D d, V x) {
  using T = TFromD<D>;
  impl::CosSinImpl<T> impl;

  // Float Constants
  const V kOneOverPi = Set(d, static_cast<T>(0.31830988618379067153));
  const V kHalf = Set(d, static_cast<T>(0.5));

  // Integer Constants
  const Rebind<int32_t, D> di32;
  using VI32 = decltype(Zero(di32));

  const V abs_x = Abs(x);
  const V sign_x = Xor(abs_x, x);

  // Compute the quadrant, q = int((|x| / pi) + 0.5)
  const VI32 q = impl.ToInt32(d, MulAdd(abs_x, kOneOverPi, kHalf));

  // Reduce range, apply sign, and approximate.
  return impl.Poly(d, Xor(impl.SinReduce(d, abs_x, q),
                          Xor(impl.SinSignFromQuadrant(d, q), sign_x)));
}

template <class D, class V>
HWY_INLINE V Sinh(const D d, V x) {
  using T = TFromD<D>;
  const V kHalf = Set(d, static_cast<T>(+0.5));
  const V kOne = Set(d, static_cast<T>(+1.0));
  const V kTwo = Set(d, static_cast<T>(+2.0));

  const V sign = And(SignBit(d), x);  // Extract the sign bit
  const V abs_x = Xor(x, sign);
  const V y = Expm1(d, abs_x);
  const V z = Mul(Div(Add(y, kTwo), Add(y, kOne)), Mul(y, kHalf));
  return Xor(z, sign);  // Reapply the sign bit
}

template <class D, class V>
HWY_INLINE V Cosh(const D d, V x) {
  using T = TFromD<D>;
  const V kHalf = Set(d, static_cast<T>(+0.5));
  const V kOne = Set(d, static_cast<T>(+1.0));
  const V kTwo = Set(d, static_cast<T>(+2.0));

  const V y = Expm1(d, Abs(x));
  const V z = Mul(Div(Add(y, kTwo), Add(y, kOne)), Mul(y, kHalf));

  // cosh(x) == cosh(|x|) == (expm1(|x|) - sinh(|x|)) + 1 == (y - z) + 1
  return Add(Sub(y, z), kOne);
}

template <class D, class V>
HWY_INLINE V Tanh(const D d, V x) {
  using T = TFromD<D>;
  const V kLimit = Set(d, static_cast<T>(18.714973875));
  const V kOne = Set(d, static_cast<T>(+1.0));
  const V kTwo = Set(d, static_cast<T>(+2.0));

  const V sign = And(SignBit(d), x);  // Extract the sign bit
  const V abs_x = Xor(x, sign);
  const V y = Expm1(d, Mul(abs_x, kTwo));
  const V z = IfThenElse(Gt(abs_x, kLimit), kOne, Div(y, Add(y, kTwo)));
  return Xor(z, sign);  // Reapply the sign bit
}

template <class D, class V>
HWY_INLINE void SinCos(const D d, V x, V& s, V& c) {
  using T = TFromD<D>;
  impl::SinCosImpl<T> impl;
  impl.SinCos(d, x, s, c);
}

template <class D, class V>
HWY_INLINE V Tan(const D d, V x) {
  V s, c;
  SinCos(d, x, s, c);
  return Div(s, c);
}

template <class D, class V>
HWY_INLINE V Hypot(const D d, V a, V b) {
  using T = TFromD<D>;
  using TI = MakeSigned<T>;
  const RebindToUnsigned<decltype(d)> du;
  const RebindToSigned<decltype(d)> di;
  using VI = VFromD<decltype(di)>;

  constexpr int kMaxBiasedExp = static_cast<int>(MaxExponentField<T>());
  static_assert(kMaxBiasedExp > 0, "kMaxBiasedExp > 0 must be true");

  constexpr int kNumOfMantBits = MantissaBits<T>();
  static_assert(kNumOfMantBits > 0, "kNumOfMantBits > 0 must be true");

  constexpr int kExpBias = kMaxBiasedExp / 2;

  static_assert(
      static_cast<unsigned>(kExpBias) + static_cast<unsigned>(kNumOfMantBits) <
          static_cast<unsigned>(kMaxBiasedExp),
      "kExpBias + kNumOfMantBits < kMaxBiasedExp must be true");

  // kMinValToSquareBiasedExp is the smallest biased exponent such that
  // pow(pow(2, kMinValToSquareBiasedExp - kExpBias) * x, 2) is either a normal
  // floating-point value or infinity if x is a non-zero, non-NaN value
  constexpr int kMinValToSquareBiasedExp = (kExpBias / 2) + kNumOfMantBits;
  static_assert(kMinValToSquareBiasedExp < kExpBias,
                "kMinValToSquareBiasedExp < kExpBias must be true");

  // kMaxValToSquareBiasedExp is the largest biased exponent such that
  // pow(pow(2, kMaxValToSquareBiasedExp - kExpBias) * x, 2) * 2 is guaranteed
  // to be a finite value if x is a finite value
  constexpr int kMaxValToSquareBiasedExp = kExpBias + ((kExpBias / 2) - 1);
  static_assert(kMaxValToSquareBiasedExp > kExpBias,
                "kMaxValToSquareBiasedExp > kExpBias must be true");
  static_assert(kMaxValToSquareBiasedExp < kMaxBiasedExp,
                "kMaxValToSquareBiasedExp < kMaxBiasedExp must be true");

#if HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128 || \
    HWY_TARGET == HWY_Z14 || HWY_TARGET == HWY_Z15
  using TExpSatSub = MakeUnsigned<T>;
  using TExpMinMax = TI;
#else
  using TExpSatSub = uint16_t;
  using TExpMinMax = int16_t;
#endif

  const Repartition<TExpSatSub, decltype(d)> d_exp_sat_sub;
  const Repartition<TExpMinMax, decltype(d)> d_exp_min_max;

  const V abs_a = Abs(a);
  const V abs_b = Abs(b);

  const MFromD<D> either_inf = Or(IsInf(a), IsInf(b));

  const VI zero = Zero(di);

  // exp_a[i] is the biased exponent of abs_a[i]
  const VI exp_a = BitCast(di, ShiftRight<kNumOfMantBits>(BitCast(du, abs_a)));

  // exp_b[i] is the biased exponent of abs_b[i]
  const VI exp_b = BitCast(di, ShiftRight<kNumOfMantBits>(BitCast(du, abs_b)));

  // max_exp[i] is equal to HWY_MAX(exp_a[i], exp_b[i])

  // If abs_a[i] and abs_b[i] are both NaN values, max_exp[i] will be equal to
  // the biased exponent of the larger value. Otherwise, if either abs_a[i] or
  // abs_b[i] is NaN, max_exp[i] will be equal to kMaxBiasedExp.
  const VI max_exp = BitCast(
      di, Max(BitCast(d_exp_min_max, exp_a), BitCast(d_exp_min_max, exp_b)));

  // If either abs_a[i] or abs_b[i] is zero, min_exp[i] is equal to max_exp[i].
  // Otherwise, if abs_a[i] and abs_b[i] are both nonzero, min_exp[i] is equal
  // to HWY_MIN(exp_a[i], exp_b[i]).
  const VI min_exp = IfThenElse(
      Or(Eq(BitCast(di, abs_a), zero), Eq(BitCast(di, abs_b), zero)), max_exp,
      BitCast(di, Min(BitCast(d_exp_min_max, exp_a),
                      BitCast(d_exp_min_max, exp_b))));

  // scl_pow2[i] is the power of 2 to scale abs_a[i] and abs_b[i] by

  // abs_a[i] and abs_b[i] should be scaled by a factor that is greater than
  // zero but less than or equal to
  // pow(2, kMaxValToSquareBiasedExp - max_exp[i]) to ensure that that the
  // multiplications or addition operations do not overflow if
  // std::hypot(abs_a[i], abs_b[i]) is finite

  // If either abs_a[i] or abs_b[i] is a a positive value that is less than
  // pow(2, kMinValToSquareBiasedExp - kExpBias), then scaling up abs_a[i] and
  // abs_b[i] by pow(2, kMinValToSquareBiasedExp - min_exp[i]) will ensure that
  // the multiplications and additions result in normal floating point values,
  // infinities, or NaNs.

  // If HWY_MAX(kMinValToSquareBiasedExp - min_exp[i], 0) is greater than
  // kMaxValToSquareBiasedExp - max_exp[i], scale abs_a[i] and abs_b[i] up by
  // pow(2, kMaxValToSquareBiasedExp - max_exp[i]) to ensure that the
  // multiplication and addition operations result in a finite result if
  // std::hypot(abs_a[i], abs_b[i]) is finite.

  const VI scl_pow2 = BitCast(
      di,
      Min(BitCast(d_exp_min_max,
                  SaturatedSub(BitCast(d_exp_sat_sub,
                                       Set(di, static_cast<TI>(
                                                   kMinValToSquareBiasedExp))),
                               BitCast(d_exp_sat_sub, min_exp))),
          BitCast(d_exp_min_max,
                  Sub(Set(di, static_cast<TI>(kMaxValToSquareBiasedExp)),
                      max_exp))));

  const VI exp_bias = Set(di, static_cast<TI>(kExpBias));

  const V ab_scl_factor =
      BitCast(d, ShiftLeft<kNumOfMantBits>(Add(exp_bias, scl_pow2)));
  const V hypot_scl_factor =
      BitCast(d, ShiftLeft<kNumOfMantBits>(Sub(exp_bias, scl_pow2)));

  const V scl_a = Mul(abs_a, ab_scl_factor);
  const V scl_b = Mul(abs_b, ab_scl_factor);

  const V scl_hypot = Sqrt(MulAdd(scl_a, scl_a, Mul(scl_b, scl_b)));
  // std::hypot returns inf if one input is +/- inf, even if the other is NaN.
  return IfThenElse(either_inf, Inf(d), Mul(scl_hypot, hypot_scl_factor));
}

template <class D, class V>
HWY_INLINE V Tgamma(const D d, V x) {
  using T = TFromD<D>;
  const V kZero = Zero(d);
  const V kOverflow =
      Set(d, static_cast<T>(sizeof(T) == 4 ? 35.040095 : 171.61447887182298));

  V result = impl::Gamma(d, x);

  result = IfThenElse(Gt(x, kOverflow), Inf(d), result);
  const MFromD<D> is_neg_int = And(Eq(x, Round(x)), Lt(x, kZero));
  result = IfThenElse(is_neg_int, NaN(d), result);
  result = IfThenElse(Eq(x, kZero), CopySign(Inf(d), x), result);
  result = IfThenElse(IsNaN(x), x, result);
  return result;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_COMPILER_GCC && !HWY_COMPILER_CLANG
#pragma GCC pop_options
#endif

#endif  // HIGHWAY_HWY_CONTRIB_MATH_MATH_INL_H_
