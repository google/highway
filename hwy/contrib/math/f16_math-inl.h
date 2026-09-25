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
#if defined(HIGHWAY_HWY_CONTRIB_MATH_F16_MATH_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_MATH_F16_MATH_INL_H_
#undef HIGHWAY_HWY_CONTRIB_MATH_F16_MATH_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_MATH_F16_MATH_INL_H_
#endif

#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// float16_t overloads of the math functions in math-inl.h. There are no
// float16 kernels yet, so these promote to float32, evaluate the float32
// kernel, and demote the result. The ops used here support float16_t lanes
// on all targets, even when HWY_HAVE_FLOAT16 is 0. As with float16_t
// conversions, behavior for non-finite inputs is implementation-defined.

// Not named `impl`: unqualified calls such as Log() from inside that
// namespace would also find math-inl.h's impl::Log and be ambiguous.
namespace f16_impl {

// Promotes the lower and upper halves of the float16 vector `v` to float32,
// evaluates `kernel` (a functor wrapping one float32 math function) on each,
// then demotes and recombines. There is no OrderedDemote2To for f32->f16,
// hence DemoteTo + Combine.
//
// This is used for every non-fractional vector with more than one lane. A
// single promotion via Rebind<float, D> would be cheaper for small vectors,
// but on scalable targets it is not always lane-exact: MaxLanes() is only an
// upper bound, so a CappedTag whose cap does not bind is a full vector at run
// time, and Rebind then runs into the cap (SVE additionally clamps kPow2 to
// 0). For example, on SVE2 with 512-bit vectors, CappedTag<float16_t, 32> has
// Lanes() == 32 but Rebind<float, decltype(d)> has Lanes() == 16, so a single
// promotion would cover only the lower half. RepartitionToWide keeps kPow2
// and therefore always has exactly Lanes(d) / 2 lanes.
//
// Fractional tags (kPow2 < 0) take the single-promotion overload below
// instead. Their Rebind is lane-exact because the widening is absorbed by
// kPow2, and per-half is not even instantiable at kPow2 == -3, where
// RepartitionToWide<D> would need a float32 tag below the smallest supported
// fraction. This is the same split as IotaForSpecial in test_util-inl.h.
// Neither condition compares MaxLanes against a vector size, so both hold at
// every vector length.
template <class D, class Kernel, class V = VFromD<D>, HWY_IF_F16_D(D),
          HWY_IF_LANES_GT_D(D, 1), HWY_IF_POW2_GT_D(D, -1)>
HWY_INLINE V F16ViaF32(D d, V v, Kernel kernel) {
  const Half<D> dh;
  const RepartitionToWide<D> df32;
  HWY_DASSERT(Lanes(df32) == Lanes(d) / 2);
  const VFromD<decltype(df32)> lo = kernel(df32, PromoteLowerTo(df32, v));
  const VFromD<decltype(df32)> hi = kernel(df32, PromoteUpperTo(df32, v));
  return Combine(d, DemoteTo(dh, hi), DemoteTo(dh, lo));
}

// Single-lane vectors (including all of HWY_SCALAR, where PromoteUpperTo and
// Combine do not exist) and fractional tags: one Rebind promotion is
// lane-exact for both.
template <class D, class Kernel, class V = VFromD<D>, HWY_IF_F16_D(D),
          hwy::EnableIf<(HWY_MAX_LANES_D(D) == 1) ||
                        (HWY_POW2_D(D) < 0)>* = nullptr>
HWY_INLINE V F16ViaF32(D d, V v, Kernel kernel) {
  const Rebind<float, D> df32;
  HWY_DASSERT(Lanes(df32) == Lanes(d));
  return DemoteTo(d, kernel(df32, PromoteTo(df32, v)));
}

// Two-input counterparts of F16ViaF32: promote corresponding lanes of both
// inputs before evaluating the float32 kernel. Use the same tag split as the
// unary adapter so that both operands retain all lanes on scalable targets.
template <class D, class Kernel, class V = VFromD<D>, HWY_IF_F16_D(D),
          HWY_IF_LANES_GT_D(D, 1), HWY_IF_POW2_GT_D(D, -1)>
HWY_INLINE V F16ViaF32TwoIn(D d, V a, V b, Kernel kernel) {
  const Half<D> dh;
  const RepartitionToWide<D> df32;
  HWY_DASSERT(Lanes(df32) == Lanes(d) / 2);
  const VFromD<decltype(df32)> lo =
      kernel(df32, PromoteLowerTo(df32, a), PromoteLowerTo(df32, b));
  const VFromD<decltype(df32)> hi =
      kernel(df32, PromoteUpperTo(df32, a), PromoteUpperTo(df32, b));
  return Combine(d, DemoteTo(dh, hi), DemoteTo(dh, lo));
}

template <class D, class Kernel, class V = VFromD<D>, HWY_IF_F16_D(D),
          hwy::EnableIf<(HWY_MAX_LANES_D(D) == 1) ||
                        (HWY_POW2_D(D) < 0)>* = nullptr>
HWY_INLINE V F16ViaF32TwoIn(D d, V a, V b, Kernel kernel) {
  const Rebind<float, D> df32;
  HWY_DASSERT(Lanes(df32) == Lanes(d));
  return DemoteTo(d, kernel(df32, PromoteTo(df32, a), PromoteTo(df32, b)));
}

// Two-output counterparts of F16ViaF32: `kernel` writes two float32 results,
// each of which is demoted (and, per half, recombined) separately.
template <class D, class Kernel, class V = VFromD<D>, HWY_IF_F16_D(D),
          HWY_IF_LANES_GT_D(D, 1), HWY_IF_POW2_GT_D(D, -1)>
HWY_INLINE void F16ViaF32TwoOut(D d, V v, Kernel kernel, V& out0, V& out1) {
  const Half<D> dh;
  const RepartitionToWide<D> df32;
  HWY_DASSERT(Lanes(df32) == Lanes(d) / 2);
  using VF32 = VFromD<decltype(df32)>;
  VF32 lo0, lo1, hi0, hi1;
  kernel(df32, PromoteLowerTo(df32, v), lo0, lo1);
  kernel(df32, PromoteUpperTo(df32, v), hi0, hi1);
  out0 = Combine(d, DemoteTo(dh, hi0), DemoteTo(dh, lo0));
  out1 = Combine(d, DemoteTo(dh, hi1), DemoteTo(dh, lo1));
}

template <class D, class Kernel, class V = VFromD<D>, HWY_IF_F16_D(D),
          hwy::EnableIf<(HWY_MAX_LANES_D(D) == 1) ||
                        (HWY_POW2_D(D) < 0)>* = nullptr>
HWY_INLINE void F16ViaF32TwoOut(D d, V v, Kernel kernel, V& out0, V& out1) {
  const Rebind<float, D> df32;
  HWY_DASSERT(Lanes(df32) == Lanes(d));
  VFromD<decltype(df32)> f0, f1;
  kernel(df32, PromoteTo(df32, v), f0, f1);
  out0 = DemoteTo(d, f0);
  out1 = DemoteTo(d, f1);
}

struct AcosKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Acos(df, x);
  }
};

struct AcoshKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Acosh(df, x);
  }
};

struct AsinKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Asin(df, x);
  }
};

struct AsinhKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Asinh(df, x);
  }
};

struct AtanKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Atan(df, x);
  }
};

struct Atan2Kernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF a, VF b) const {
    return Atan2(df, a, b);
  }
};

struct AtanhKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Atanh(df, x);
  }
};

template <bool kHandleSubnormals>
struct CbrtKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Cbrt<kHandleSubnormals>(df, x);
  }
};

struct CosKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Cos(df, x);
  }
};

struct CoshKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Cosh(df, x);
  }
};

struct ErfKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Erf(df, x);
  }
};

struct ExpKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Exp(df, x);
  }
};

struct Exp2Kernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Exp2(df, x);
  }
};

struct Expm1Kernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Expm1(df, x);
  }
};

struct HypotKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF a, VF b) const {
    return Hypot(df, a, b);
  }
};

struct LogKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Log(df, x);
  }
};

struct Log10Kernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Log10(df, x);
  }
};

struct Log1pKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Log1p(df, x);
  }
};

struct Log2Kernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Log2(df, x);
  }
};

struct PowKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF a, VF b) const {
    return Pow(df, a, b);
  }
};

struct SinKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Sin(df, x);
  }
};

struct SinCosKernel {
  template <class DF, class VF>
  HWY_INLINE void operator()(DF df, VF x, VF& s, VF& c) const {
    SinCos(df, x, s, c);
  }
};

struct SinhKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Sinh(df, x);
  }
};

// Calls the float32 Tan, which divides in float32; dividing the demoted
// float16 sine and cosine would instead lose accuracy near the poles.
struct TanKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Tan(df, x);
  }
};

struct TanhKernel {
  template <class DF, class VF>
  HWY_INLINE VF operator()(DF df, VF x) const {
    return Tanh(df, x);
  }
};

}  // namespace f16_impl

// The generic templates in math-inl.h are constrained with
// HWY_IF_NOT_SPECIAL_FLOAT_D, so these HWY_IF_F16_D overloads partition the
// overload set rather than being ambiguous.

/**
 * Highway SIMD version of std::acos(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-1, +1]
 * @return arc cosine of 'x'
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Acos(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::AcosKernel());
}

/**
 * Highway SIMD version of std::acosh(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[1, +65504]
 * @return hyperbolic arc cosine of 'x'
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Acosh(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::AcoshKernel());
}

/**
 * Highway SIMD version of std::asin(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-1, +1]
 * @return arc sine of 'x'
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Asin(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::AsinKernel());
}

/**
 * Highway SIMD version of std::asinh(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-65504, +65504]
 * @return hyperbolic arc sine of 'x'
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Asinh(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::AsinhKernel());
}

/**
 * Highway SIMD version of std::atan(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-65504, +65504]
 * @return arc tangent of 'x'
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Atan(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::AtanKernel());
}

/**
 * Highway SIMD version of std::atan2(y, x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-65504, +65504] for both inputs
 * @return arc tangent of 'y' / 'x', with the quadrant determined by both signs
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Atan2(D d, V y, V x) {
  return f16_impl::F16ViaF32TwoIn(d, y, x, f16_impl::Atan2Kernel());
}

/**
 * Highway SIMD version of std::atanh(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16(-1, +1)
 * @return hyperbolic arc tangent of 'x'
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Atanh(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::AtanhKernel());
}

/**
 * Highway SIMD version of std::cbrt(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-65504, +65504]
 * @return cube root of 'x'
 */
template <bool kHandleSubnormals = true, class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Cbrt(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::CbrtKernel<kHandleSubnormals>());
}

/**
 * Highway SIMD version of std::cos(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-39000, +39000]
 * @return cosine of 'x'
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Cos(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::CosKernel());
}

/**
 * Highway SIMD version of std::cosh(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-80, +80]
 * @return hyperbolic cosine of 'x'
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Cosh(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::CoshKernel());
}

/**
 * Highway SIMD version of std::erf(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-65504, +65504]
 * @return error function of 'x'
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Erf(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::ErfKernel());
}

/**
 * Highway SIMD version of std::exp(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-65504, +104]
 * @return e^x
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Exp(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::ExpKernel());
}

/**
 * Highway SIMD version of std::exp2(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-65504, +128]
 * @return 2^x
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Exp2(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::Exp2Kernel());
}

/**
 * Highway SIMD version of std::expm1(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-65504, +104]
 * @return e^x - 1
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Expm1(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::Expm1Kernel());
}

/**
 * Highway SIMD version of std::hypot(a, b) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-65504, +65504] for both inputs
 * @return hypotenuse of a and b
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Hypot(D d, V a, V b) {
  return f16_impl::F16ViaF32TwoIn(d, a, b, f16_impl::HypotKernel());
}

/**
 * Highway SIMD version of std::log(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16(0, +65504]
 * @return natural logarithm of 'x'
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Log(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::LogKernel());
}

/**
 * Highway SIMD version of std::log10(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16(0, +65504]
 * @return base 10 logarithm of 'x'
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Log10(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::Log10Kernel());
}

/**
 * Highway SIMD version of std::log1p(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[0, +65504]
 * @return log(1 + x)
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Log1p(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::Log1pKernel());
}

/**
 * Highway SIMD version of std::log2(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16(0, +65504]
 * @return base 2 logarithm of 'x'
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Log2(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::Log2Kernel());
}

/**
 * Highway SIMD version of std::pow(a, b) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-65504, +65504] for both inputs
 * @return a raised to b
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Pow(D d, V a, V b) {
  return f16_impl::F16ViaF32TwoIn(d, a, b, f16_impl::PowKernel());
}

/**
 * Highway SIMD version of std::sin(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-39000, +39000]
 * @return sine of 'x'
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Sin(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::SinKernel());
}

/**
 * Highway SIMD version of SinCos for float16 lanes.
 * Compute the sine and cosine at the same time
 * The performance should be around the same as calling Sin.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-39000, +39000]
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE void SinCos(D d, V x, V& s, V& c) {
  f16_impl::F16ViaF32TwoOut(d, x, f16_impl::SinCosKernel(), s, c);
}

/**
 * Highway SIMD version of std::sinh(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-80, +80]
 * @return hyperbolic sine of 'x'
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Sinh(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::SinhKernel());
}

/**
 * Highway SIMD version of std::tan(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-39000, +39000]
 * @return tangent of 'x'
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Tan(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::TanKernel());
}

/**
 * Highway SIMD version of std::tanh(x) for float16 lanes.
 *
 * Valid Lane Types: float16
 *        Max Error: ULP = 1
 *      Valid Range: float16[-65504, +65504]
 * @return hyperbolic tangent of 'x'
 */
template <class D, class V, HWY_IF_F16_D(D)>
HWY_INLINE V Tanh(D d, V x) {
  return f16_impl::F16ViaF32(d, x, f16_impl::TanhKernel());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_MATH_F16_MATH_INL_H_
