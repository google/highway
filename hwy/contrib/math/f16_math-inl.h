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
// on all targets, even when HWY_HAVE_FLOAT16 is 0.

// Not named `impl`: unqualified calls such as Log() from inside that
// namespace would also find math-inl.h's impl::Log and be ambiguous.
namespace f16_impl {

// Promotes lower/upper halves of the float16 vector `v` to float32,
// evaluates `kernel` (a functor wrapping one float32 math function) twice,
// then demotes and recombines. Never instantiated on HWY_SCALAR, so
// PromoteUpperTo and Combine are safe to name here. There is no
// OrderedDemote2To for f32->f16, hence DemoteTo + Combine.
template <class D, class Kernel, class V = VFromD<D>>
HWY_INLINE V F16ViaF32PerHalf(D d, V v, Kernel kernel) {
  const Half<D> dh;
  const RepartitionToWide<D> df32;
  HWY_DASSERT(Lanes(df32) == Lanes(d) / 2);
  const VFromD<decltype(df32)> lo = kernel(df32, PromoteLowerTo(df32, v));
  const VFromD<decltype(df32)> hi = kernel(df32, PromoteUpperTo(df32, v));
  return Combine(d, DemoteTo(dh, hi), DemoteTo(dh, lo));
}

// Whether `Rebind<float, D>` is guaranteed to have exactly `Lanes(d)` lanes,
// i.e. one promotion covers every lane of `v`.
//
// Rebind raises kPow2 by one because float lanes are twice as large. On
// fixed-size targets Lanes() == MaxLanes(), so comparing sizes is enough. On
// scalable targets it is not: MaxLanes() is only an upper bound, so a
// CappedTag whose cap does not bind is a full vector at run time. Rebind then
// runs into the cap, and SVE additionally clamps kPow2 to 0, leaving the
// float32 tag with half as many lanes as `d`. For example, on SVE2 with
// 512-bit vectors, CappedTag<float16_t, 32> has Lanes() == 32, but
// Rebind<float, decltype(d)> has Lanes() == 16: the kernel would see only
// lanes 0..15, and DemoteTo, which duplicates its result into both halves of
// the register, would store those same results again for lanes 16..31.
//
// Fractional tags absorb the widening in kPow2 and are safe, as is any tag
// whose cap binds even for the smallest possible vector. Everything else uses
// the per-half path, whose RepartitionToWide keeps kPow2 and therefore always
// has exactly Lanes(d) / 2 lanes.
template <class D>
constexpr bool F16OnePromotionCoversAllLanes() {
  return (HWY_POW2_D(D) + 1 <= HWY_MAX_POW2) &&
         (HWY_HAVE_SCALABLE
              ? (HWY_POW2_D(D) < 0 || HWY_V_SIZE_D(D) * 2 <= HWY_MIN_BYTES)
              : (HWY_V_SIZE_D(D) <= HWY_MAX_BYTES / 2));
}

// Evaluates `kernel` on all lanes of `v` with a single promotion and one
// kernel evaluation.
template <class D, class Kernel, class V = VFromD<D>, HWY_IF_F16_D(D),
          hwy::EnableIf<F16OnePromotionCoversAllLanes<D>()>* = nullptr>
HWY_INLINE V F16ViaF32(D d, V v, Kernel kernel) {
  const Rebind<float, D> df32;
  HWY_DASSERT(Lanes(df32) == Lanes(d));
  return DemoteTo(d, kernel(df32, PromoteTo(df32, v)));
}

// Everything else needs two promotions, one per half.
template <class D, class Kernel, class V = VFromD<D>, HWY_IF_F16_D(D),
          hwy::EnableIf<!F16OnePromotionCoversAllLanes<D>()>* = nullptr>
HWY_INLINE V F16ViaF32(D d, V v, Kernel kernel) {
  return F16ViaF32PerHalf(d, v, kernel);
}

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

}  // namespace f16_impl

// The generic templates in math-inl.h are constrained with
// HWY_IF_NOT_SPECIAL_FLOAT_D, so these HWY_IF_F16_D overloads partition the
// overload set rather than being ambiguous.

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

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_MATH_F16_MATH_INL_H_
