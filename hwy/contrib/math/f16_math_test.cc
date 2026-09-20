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

#include <stdint.h>
#include <stdio.h>

#include <cmath>  // std::exp, std::sin

// For faster tests. Not using AES, hence NEON_WITHOUT_AES is sufficient.
// SVE is mostly superseded by SVE2.
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_NEON | HWY_SVE)
#endif  // HWY_DISABLED_TARGETS

#include "hwy/base.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/math/f16_math_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/math/f16_math-inl.h"
#include "hwy/contrib/math/math-inl.h"  // CallExp
#include "hwy/contrib/math/math_test-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

// The float16 min/max bounds mirror the float32 math test bounds where
// the kernel is validated (e.g. +104 for Exp), clamped to the float16 finite
// range [-65504, +65504]. The smallest positive float16 subnormal is 2^-24.
// clang-format off
DEFINE_F16_MATH_TEST(Acos,
  std::acos,  CallAcos,  -1.0f,            +1.0f,     1)
DEFINE_F16_MATH_TEST(Asin,
  std::asin,  CallAsin,  -1.0f,            +1.0f,     1)
DEFINE_F16_MATH_TEST(Atan,
  std::atan,  CallAtan,  -65504.0f,        +65504.0f, 1)
DEFINE_F16_MATH_TEST(Cbrt,
  std::cbrt,  CallCbrt,  -65504.0f,        +65504.0f, 1)
DEFINE_F16_MATH_TEST(Erf,
  std::erf,   CallErf,   -65504.0f,        +65504.0f, 1)
DEFINE_F16_MATH_TEST(Exp,
  std::exp,   CallExp,   -65504.0f,        +104.0f,   1)
DEFINE_F16_MATH_TEST(Exp2,
  std::exp2,  CallExp2,  -65504.0f,        +128.0f,   1)
DEFINE_F16_MATH_TEST(Expm1,
  std::expm1, CallExpm1, -65504.0f,        +104.0f,   1)
DEFINE_F16_MATH_TEST(Log,
  std::log,   CallLog,   +5.960464478E-8f, +65504.0f, 1)
DEFINE_F16_MATH_TEST(Log10,
  std::log10, CallLog10, +5.960464478E-8f, +65504.0f, 1)
DEFINE_F16_MATH_TEST(Log1p,
  std::log1p, CallLog1p, +0.0f,            +65504.0f, 1)
DEFINE_F16_MATH_TEST(Log2,
  std::log2,  CallLog2,  +5.960464478E-8f, +65504.0f, 1)
// clang-format on

// Bounds match math_hyper_test.cc, restricted to finite float16 inputs.
// 0.99951171875 is the largest float16 value below 1, excluding Atanh's poles.
// Sinh/Cosh use the upstream-tested float32 range and include float16 overflow.
// clang-format off
DEFINE_F16_MATH_TEST(Acosh,
  std::acosh, CallAcosh, +1.0f,          +65504.0f,       1)
DEFINE_F16_MATH_TEST(Asinh,
  std::asinh, CallAsinh, -65504.0f,      +65504.0f,       1)
DEFINE_F16_MATH_TEST(Atanh,
  std::atanh, CallAtanh, -0.99951171875f, +0.99951171875f, 1)
DEFINE_F16_MATH_TEST(Cosh,
  std::cosh,  CallCosh,  -80.0f,         +80.0f,          1)
DEFINE_F16_MATH_TEST(Sinh,
  std::sinh,  CallSinh,  -80.0f,         +80.0f,          1)
DEFINE_F16_MATH_TEST(Tanh,
  std::tanh,  CallTanh,  -65504.0f,      +65504.0f,       1)
// clang-format on

// Sweep every float16 bit pattern against boundary values in both argument
// positions, then against equal, opposite, and permuted values. This samples
// the two-input domain, rather than claiming to exhaust all 2^32 pairs.
template <class D>
HWY_NOINLINE void TestF16BinaryMath(
    const char* name, double (*fx1)(double, double),
    Vec<D> (*fxN)(D, VecArg<Vec<D>>, VecArg<Vec<D>>), D d) {
  if (HWY_MATH_TEST_EXCESS_PRECISION) return;

  const size_t N = Lanes(d);
  auto a = AllocateAligned<float16_t>(N);
  auto b = AllocateAligned<float16_t>(N);
  auto actual = AllocateAligned<float16_t>(N);
  HWY_ASSERT(a && b && actual);
  // Zero, subnormal/normal boundary, fractions, neighbors of 1, integer
  // exponents, and the largest finite value. Non-finite inputs still go
  // through the kernel, but their results are not checked: float16 conversion
  // behavior for such inputs is implementation-defined (quick_reference.md).
  const uint16_t anchors[] = {0x0000, 0x0001, 0x03FF, 0x0400,
                             0x3800, 0x3BFF, 0x3C00, 0x3C01,
                             0x4000, 0x4200, 0x7BFF};
  constexpr size_t kAnchoredSweeps = 2 * sizeof(anchors) / sizeof(anchors[0]);
  uint64_t max_ulp = 0;
  for (size_t sweep = 0; sweep < kAnchoredSweeps + 3; ++sweep) {
    for (uint32_t base = 0; base <= 0xFFFF; base += static_cast<uint32_t>(N)) {
      for (size_t i = 0; i < N; ++i) {
        const uint32_t bits = base + static_cast<uint32_t>(i);
        uint16_t a_bits = static_cast<uint16_t>(bits);
        uint16_t b_bits;
        if (sweep < kAnchoredSweeps) {
          b_bits = anchors[sweep / 2];
          if (sweep & 1) {
            a_bits = b_bits;
            b_bits = static_cast<uint16_t>(bits);
          }
        } else if (sweep == kAnchoredSweeps) {
          b_bits = a_bits;
        } else if (sweep == kAnchoredSweeps + 1) {
          b_bits = static_cast<uint16_t>(bits ^ 0x8000);
        } else {
          // Odd multiplier permutes all 16-bit patterns. Both inputs vary
          // across lanes, exposing lost halves and mixed operand lanes.
          b_bits = static_cast<uint16_t>((bits * 40503 + 12345) & 0xFFFF);
        }
        a[i] = BitCastScalar<float16_t>(a_bits);
        b[i] = BitCastScalar<float16_t>(b_bits);
      }
      Store(fxN(d, Load(d, a.get()), Load(d, b.get())), d, actual.get());
      for (size_t i = 0; i < N; ++i) {
        if ((BitCastScalar<uint16_t>(a[i]) & 0x7C00) == 0x7C00 ||
            (BitCastScalar<uint16_t>(b[i]) & 0x7C00) == 0x7C00) {
          continue;
        }
        const double a_value = static_cast<double>(F32FromF16(a[i]));
        const double b_value = static_cast<double>(F32FromF16(b[i]));
        const float16_t expected = F16FromF64(fx1(a_value, b_value));
        const uint16_t a_bits = BitCastScalar<uint16_t>(actual[i]);
        const uint16_t e_bits = BitCastScalar<uint16_t>(expected);
        const uint64_t ulp = F16UlpDelta(actual[i], expected);
        const bool wrong_zero_sign = (a_bits & 0x7FFF) == 0 &&
                                    (e_bits & 0x7FFF) == 0 && a_bits != e_bits;
        if (ulp > 1 || wrong_zero_sign) {
          HWY_ABORT("%s (MaxLanes %d, pow2 %d): %s(%g, %g) lane %d "
                    "expected 0x%04x actual 0x%04x ulp %g",
                    hwy::TypeName(float16_t(), N).c_str(),
                    static_cast<int>(HWY_MAX_LANES_D(D)), int{HWY_POW2_D(D)},
                    name, a_value, b_value, static_cast<int>(i),
                    static_cast<unsigned>(e_bits), static_cast<unsigned>(a_bits),
                    static_cast<double>(ulp));
        }
        max_ulp = HWY_MAX(max_ulp, ulp);
      }
    }
  }
  fprintf(stderr, "%s: %s max_ulp %g\n",
          hwy::TypeName(float16_t(), N).c_str(), name,
          static_cast<double>(max_ulp));
}

struct TestF16Atan2 {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    TestF16BinaryMath("Atan2", std::atan2, CallAtan2, d);
  }
};

struct TestF16Hypot {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    TestF16BinaryMath("Hypot", std::hypot, CallHypot, d);
  }
};

struct TestF16Pow {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    TestF16BinaryMath("Pow", std::pow, CallPow, d);
  }
};

HWY_NOINLINE void TestAllF16Atan2() {
  ForPartialVectors<TestF16Atan2>()(float16_t());
}

HWY_NOINLINE void TestAllF16Hypot() {
  ForPartialVectors<TestF16Hypot>()(float16_t());
}

HWY_NOINLINE void TestAllF16Pow() {
  ForPartialVectors<TestF16Pow>()(float16_t());
}

struct F16BinaryCase {
  uint16_t a;
  uint16_t b;
  uint16_t expected;
};

// Exact expectations cover cases where a 1-ULP budget would hide an incorrect
// zero sign, flushed subnormal, underflow tie, or saturation at overflow.
template <class D, size_t kCases>
void CheckF16BinaryCases(D d, const F16BinaryCase (&cases)[kCases],
                        Vec<D> (*fxN)(D, VecArg<Vec<D>>, VecArg<Vec<D>>)) {
  const size_t N = Lanes(d);
  const RebindToUnsigned<D> du;
  auto a = AllocateAligned<uint16_t>(N);
  auto b = AllocateAligned<uint16_t>(N);
  auto expected = AllocateAligned<uint16_t>(N);
  HWY_ASSERT(a && b && expected);
  for (size_t base = 0; base < kCases; base += N) {
    for (size_t i = 0; i < N; ++i) {
      const auto& c = cases[(base + i) % kCases];
      a[i] = c.a;
      b[i] = c.b;
      expected[i] = c.expected;
    }
    const auto actual = fxN(d, BitCast(d, Load(du, a.get())),
                           BitCast(d, Load(du, b.get())));
    HWY_ASSERT_VEC_EQ(du, Load(du, expected.get()), BitCast(du, actual));
  }
}

struct TestF16BinaryBoundaries {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    if (HWY_MATH_TEST_EXCESS_PRECISION) return;
    const F16BinaryCase atan2[] = {
        {0x0000, 0x0000, 0x0000}, {0x8000, 0x0000, 0x8000},
        {0x0000, 0x8000, 0x4248}, {0x8000, 0x8000, 0xC248},
        {0x3C00, 0x0000, 0x3E48}, {0xBC00, 0x0000, 0xBE48},
        {0x3C00, 0x3C00, 0x3A48}, {0xBC00, 0x3C00, 0xBA48},
        {0x3C00, 0xBC00, 0x40B6}, {0xBC00, 0xBC00, 0xC0B6},
        {0x0001, 0x3C00, 0x0001}, {0x8001, 0x3C00, 0x8001}};
    CheckF16BinaryCases(d, atan2, CallAtan2);

    const F16BinaryCase hypot[] = {
        {0x0000, 0x8000, 0x0000}, {0x8000, 0x8000, 0x0000},
        {0x0001, 0x8001, 0x0001}, {0x8001, 0x0000, 0x0001},
        {0x4200, 0x4400, 0x4500}, {0xC200, 0xC400, 0x4500},
        {0x7BFF, 0x0000, 0x7BFF}, {0x7BFF, 0x7BFF, 0x7C00}};
    CheckF16BinaryCases(d, hypot, CallHypot);

    const F16BinaryCase pow[] = {
        {0x4000, 0x4C00, 0x7C00},  // 2^16 overflows.
        {0x4000, 0xCE00, 0x0001},  // 2^-24 is the smallest subnormal.
        {0x4000, 0xCE40, 0x0000},  // 2^-25 ties to even (zero).
        {0xC000, 0xCE40, 0x8000},  // Negative base, odd exponent.
        {0x8000, 0x4200, 0x8000}, {0x8000, 0xC200, 0xFC00},
        {0x8000, 0x4000, 0x0000}, {0x8000, 0xC000, 0x7C00},
        {0xBC00, 0x67FF, 0xBC00},  // (-1)^2047, last odd F16 integer.
        {0xBC00, 0x6800, 0x3C00}, {0xBC00, 0x7BFF, 0x3C00},
        {0x0000, 0x0000, 0x3C00}, {0x3C00, 0x7BFF, 0x3C00},
        {0x0001, 0x3C00, 0x0001}, {0x8001, 0x3C00, 0x8001}};
    CheckF16BinaryCases(d, pow, CallPow);
  }
};

HWY_NOINLINE void TestAllF16BinaryBoundaries() {
  ForPartialVectors<TestF16BinaryBoundaries>()(float16_t());
}

// Even subnormal float16 inputs become normal float32 inputs, so both
// Cbrt modes should cover the entire finite float16 range.
template <class D>
static Vec<D> F16CbrtNoSubnormals(const D d, VecArg<Vec<D>> x) {
  return Cbrt<false>(d, x);
}

DEFINE_F16_MATH_TEST(CbrtNoSubnormals, std::cbrt, F16CbrtNoSubnormals, -65504.0f,
                     +65504.0f, 1)

// SinCos has two outputs, so test each separately, as math_trig_test.cc does.
template <class D>
static Vec<D> F16SinCosSin(const D d, VecArg<Vec<D>> x) {
  Vec<D> s, c;
  CallSinCos(d, x, s, c);
  return s;
}

template <class D>
static Vec<D> F16SinCosCos(const D d, VecArg<Vec<D>> x) {
  Vec<D> s, c;
  CallSinCos(d, x, s, c);
  return c;
}

// Unlike the bounds above, these come from math_trig_test.cc rather than from
// the float16 finite range: the trig kernels are only accurate on
// [-39000, +39000], which is narrower than float16 can represent. Beyond it
// the Cody-Waite range reduction loses exactness on targets without FMA.
// clang-format off
DEFINE_F16_MATH_TEST(Sin,
  std::sin,   CallSin,      -39000.0f,        +39000.0f, 1)
DEFINE_F16_MATH_TEST(Cos,
  std::cos,   CallCos,      -39000.0f,        +39000.0f, 1)
DEFINE_F16_MATH_TEST(Tan,
  std::tan,   CallTan,      -39000.0f,        +39000.0f, 1)
DEFINE_F16_MATH_TEST(SinCosSin,
  std::sin,   F16SinCosSin, -39000.0f,        +39000.0f, 1)
DEFINE_F16_MATH_TEST(SinCosCos,
  std::cos,   F16SinCosCos, -39000.0f,        +39000.0f, 1)
// clang-format on

// The exhaustive ULP test treats +0 and -0 as equal. Check that these odd
// functions preserve their bits in every lane.
struct TestF16UnarySignedZero {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const RebindToUnsigned<D> du;
    const uint16_t zero_bits[] = {0x0000, 0x8000};
    for (const uint16_t bits : zero_bits) {
      const auto expected = Set(du, bits);
      const auto x = BitCast(d, expected);
      HWY_ASSERT_VEC_EQ(du, expected, BitCast(du, CallAsin(d, x)));
      HWY_ASSERT_VEC_EQ(du, expected, BitCast(du, CallAtan(d, x)));
      HWY_ASSERT_VEC_EQ(du, expected, BitCast(du, CallCbrt(d, x)));
      HWY_ASSERT_VEC_EQ(du, expected, BitCast(du, Cbrt<false>(d, x)));
      HWY_ASSERT_VEC_EQ(du, expected, BitCast(du, CallErf(d, x)));
    }
  }
};

HWY_NOINLINE void TestAllF16UnarySignedZero() {
  ForPartialVectors<TestF16UnarySignedZero>()(float16_t());
}

// These exact results need stricter checks than the exhaustive 1-ULP test:
// signed zeros, the smallest subnormals, the Acosh endpoint, and saturation.
struct TestF16HyperbolicBoundaries {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const RebindToUnsigned<D> du;
    const auto one_bits = Set(du, uint16_t{0x3C00});
    const auto one = BitCast(d, one_bits);
    HWY_ASSERT_VEC_EQ(du, Zero(du), BitCast(du, CallAcosh(d, one)));

    const uint16_t tiny_bits[] = {0x0000, 0x8000, 0x0001, 0x8001};
    for (const uint16_t bits : tiny_bits) {
      const auto expected = Set(du, bits);
      const auto x = BitCast(d, expected);
      HWY_ASSERT_VEC_EQ(du, expected, BitCast(du, CallAsinh(d, x)));
      HWY_ASSERT_VEC_EQ(du, expected, BitCast(du, CallAtanh(d, x)));
      HWY_ASSERT_VEC_EQ(du, expected, BitCast(du, CallSinh(d, x)));
      HWY_ASSERT_VEC_EQ(du, expected, BitCast(du, CallTanh(d, x)));
      HWY_ASSERT_VEC_EQ(du, one_bits, BitCast(du, CallCosh(d, x)));
    }

    const uint16_t signs[] = {0x0000, 0x8000};
    for (const uint16_t sign : signs) {
      // +/-12: Sinh and Cosh overflow float16, and Tanh rounds to +/-1.
      const auto x = BitCast(d, Set(du, static_cast<uint16_t>(sign | 0x4A00)));
      const auto inf_bits = Set(du, uint16_t{0x7C00});
      const auto signed_inf = Set(du, static_cast<uint16_t>(sign | 0x7C00));
      const auto signed_one = Set(du, static_cast<uint16_t>(sign | 0x3C00));
      HWY_ASSERT_VEC_EQ(du, signed_inf, BitCast(du, CallSinh(d, x)));
      HWY_ASSERT_VEC_EQ(du, inf_bits, BitCast(du, CallCosh(d, x)));
      HWY_ASSERT_VEC_EQ(du, signed_one, BitCast(du, CallTanh(d, x)));
    }
  }
};

HWY_NOINLINE void TestAllF16HyperbolicBoundaries() {
  ForPartialVectors<TestF16HyperbolicBoundaries>()(float16_t());
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(HwyF16MathTest);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Atan2);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Hypot);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Pow);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16BinaryBoundaries);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Acosh);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Asinh);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Atanh);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Cosh);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Sinh);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Tanh);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16HyperbolicBoundaries);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Acos);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Asin);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Atan);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Cbrt);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16CbrtNoSubnormals);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Erf);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16UnarySignedZero);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Exp);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Exp2);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Expm1);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Log);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Log10);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Log1p);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Log2);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Sin);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Cos);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Tan);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16SinCosSin);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16SinCosCos);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
