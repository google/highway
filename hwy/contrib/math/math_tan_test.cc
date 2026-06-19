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

#include <stdint.h>
#include <stdio.h>

#include <cfloat>  // FLT_MAX
#include <cmath>   // std::abs

#include "hwy/base.h"
#include "hwy/nanobenchmark.h"

// Clang build timeout on RVV as of 2025-09-19.
#if !HWY_ARCH_RVV

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/math/math_tan_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/contrib/math/fast_math-inl.h"
#include "hwy/contrib/math/math_test-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

// clang-format off
DEFINE_MATH_TEST(Atan,
  std::atan,  CallAtan,  -FLT_MAX,   +FLT_MAX,    3,
  std::atan,  CallAtan,  -DBL_MAX,   +DBL_MAX,    3)

// 300 ULP max error for float32 accommodates for architectures without FMA (like SSE4)
// where rounding errors accumulate higher due to separate multiply and add instructions.
// On hardware with FMA, the max error is ~64 ULP.
DEFINE_MATH_TEST(Tan,
  std::tan,  CallTan,  -39000.0,   +39000.0,    (HWY_NATIVE_FMA ? 64 : 300),
  std::tan,  CallTan,  -39000.0,   +39000.0,    2)
// clang-format on

template <typename T, class D>
void Atan2TestCases(T /*unused*/, D d, size_t& padded,
                    AlignedFreeUniquePtr<T[]>& out_y,
                    AlignedFreeUniquePtr<T[]>& out_x,
                    AlignedFreeUniquePtr<T[]>& out_expected) {
  struct YX {
    T y;
    T x;
    T expected;
  };
  const T pos = ConvertScalarTo<T>(1E5);
  const T neg = ConvertScalarTo<T>(-1E7);
  const T p0 = ConvertScalarTo<T>(0);
  // -0 is not enough to get an actual negative zero.
  const T n0 = ConvertScalarTo<T>(-0.0);
  const T p1 = ConvertScalarTo<T>(1);
  const T n1 = ConvertScalarTo<T>(-1);
  const T p2 = ConvertScalarTo<T>(2);
  const T n2 = ConvertScalarTo<T>(-2);
  const T inf = GetLane(Inf(d));
  const T nan = GetLane(NaN(d));

  const T pi = ConvertScalarTo<T>(3.141592653589793238);
  const YX test_cases[] = {                        // 45 degree steps:
                           {p0, p1, p0},           // E
                           {n1, p1, -pi / 4},      // SE
                           {n1, p0, -pi / 2},      // S
                           {n1, n1, -3 * pi / 4},  // SW
                           {p0, n1, pi},           // W
                           {p1, n1, 3 * pi / 4},   // NW
                           {p1, p0, pi / 2},       // N
                           {p1, p1, pi / 4},       // NE

                           // y = ±0, x < 0 or -0
                           {p0, n1, pi},
                           {n0, n2, -pi},
                           // y = ±0, x > 0 or +0
                           {p0, p2, p0},
                           {n0, p2, n0},
                           // y = ±∞, x finite
                           {inf, p2, pi / 2},
                           {-inf, p2, -pi / 2},
                           // y = ±∞, x = -∞
                           {inf, -inf, 3 * pi / 4},
                           {-inf, -inf, -3 * pi / 4},
                           // y = ±∞, x = +∞
                           {inf, inf, pi / 4},
                           {-inf, inf, -pi / 4},
                           // y < 0, x = ±0
                           {n2, p0, -pi / 2},
                           {n1, n0, -pi / 2},
                           // y > 0, x = ±0
                           {pos, p0, pi / 2},
                           {p2, n0, pi / 2},
                           // finite y > 0, x = -∞
                           {pos, -inf, pi},
                           // finite y < 0, x = -∞
                           {neg, -inf, -pi},
                           // finite y > 0, x = +∞
                           {pos, inf, p0},
                           // finite y < 0, x = +∞
                           {neg, inf, n0},
                           // y NaN xor x NaN
                           {nan, p0, nan},
                           {pos, nan, nan}};
  const size_t kNumTestCases = sizeof(test_cases) / sizeof(test_cases[0]);
  const size_t N = Lanes(d);
  padded = RoundUpTo(kNumTestCases, N);  // allow loading whole vectors
  out_y = AllocateAligned<T>(padded);
  out_x = AllocateAligned<T>(padded);
  out_expected = AllocateAligned<T>(padded);
  HWY_ASSERT(out_y && out_x && out_expected);
  size_t i = 0;
  for (; i < kNumTestCases; ++i) {
    out_y[i] = test_cases[i].y;
    out_x[i] = test_cases[i].x;
    out_expected[i] = test_cases[i].expected;
  }
  for (; i < padded; ++i) {
    out_y[i] = p0;
    out_x[i] = p0;
    out_expected[i] = p0;
  }
}

struct TestAtan2 {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T t, D d) {
    const size_t N = Lanes(d);

    size_t padded;
    AlignedFreeUniquePtr<T[]> in_y, in_x, expected;
    Atan2TestCases(t, d, padded, in_y, in_x, expected);

    const Vec<D> tolerance = Set(d, ConvertScalarTo<T>(1E-5));

    for (size_t i = 0; i < padded; ++i) {
      const T actual = ConvertScalarTo<T>(atan2(in_y[i], in_x[i]));
      // fprintf(stderr, "%zu: table %f atan2 %f\n", i, expected[i], actual);
      HWY_ASSERT_EQ(expected[i], actual);
    }
    for (size_t i = 0; i < padded; i += N) {
      const Vec<D> y = Load(d, &in_y[i]);
      const Vec<D> x = Load(d, &in_x[i]);
#if HWY_ARCH_ARM_A64
      // TODO(b/287462770): inline to work around incorrect SVE codegen
      const Vec<D> actual = Atan2(d, y, x);
#else
      const Vec<D> actual = CallAtan2(d, y, x);
#endif
      const Vec<D> vexpected = Load(d, &expected[i]);

      const Mask<D> exp_nan = IsNaN(vexpected);
      const Mask<D> act_nan = IsNaN(actual);
      HWY_ASSERT_MASK_EQ(d, exp_nan, act_nan);

      // If not NaN, then compare with tolerance
      const Mask<D> ge = Ge(actual, Sub(vexpected, tolerance));
      const Mask<D> le = Le(actual, Add(vexpected, tolerance));
      const Mask<D> ok = Or(act_nan, And(le, ge));
      if (!AllTrue(d, ok)) {
        const size_t mismatch =
            static_cast<size_t>(FindKnownFirstTrue(d, Not(ok)));
        fprintf(stderr, "Mismatch for i=%d expected %E actual %E\n",
                static_cast<int>(i + mismatch), expected[i + mismatch],
                ExtractLane(actual, mismatch));
        HWY_ASSERT(0);
      }
    }
  }
};

HWY_NOINLINE void TestAllAtan2() {
  if (HWY_MATH_TEST_EXCESS_PRECISION) return;

  ForFloat3264Types(ForPartialVectors<TestAtan2>());
}

template <typename T, class D>
void HypotTestCases(T /*unused*/, D d, size_t& padded,
                    AlignedFreeUniquePtr<T[]>& out_a,
                    AlignedFreeUniquePtr<T[]>& out_b,
                    AlignedFreeUniquePtr<T[]>& out_expected) {
  using TU = MakeUnsigned<T>;

  struct AB {
    T a;
    T b;
  };

  constexpr int kNumOfMantBits = MantissaBits<T>();
  static_assert(kNumOfMantBits > 0, "kNumOfMantBits > 0 must be true");

  // Ensures inputs are not constexpr.
  const TU u1 = static_cast<TU>(hwy::Unpredictable1());
  const double k1 = static_cast<double>(u1);

  const T pos = ConvertScalarTo<T>(1E5 * k1);
  const T neg = ConvertScalarTo<T>(-1E7 * k1);
  const T p0 = ConvertScalarTo<T>(k1 - 1.0);
  // -0 is not enough to get an actual negative zero.
  const T n0 = ScalarCopySign<T>(p0, neg);
  const T p1 = ConvertScalarTo<T>(k1);
  const T n1 = ConvertScalarTo<T>(-k1);
  const T p2 = ConvertScalarTo<T>(2 * k1);
  const T n2 = ConvertScalarTo<T>(-2 * k1);
  const T inf = BitCastScalar<T>(ExponentMask<T>() * u1);
  const T neg_inf = ScalarCopySign(inf, n1);
  const T nan = BitCastScalar<T>(
      static_cast<TU>(ExponentMask<T>() | (u1 << (kNumOfMantBits - 1))));

  const double max_as_f64 = ConvertScalarTo<double>(HighestValue<T>()) * k1;
  const T max = ConvertScalarTo<T>(max_as_f64);

  const T huge = ConvertScalarTo<T>(max_as_f64 * 0.25);
  const T neg_huge = ScalarCopySign(huge, n1);

  const T huge2 = ConvertScalarTo<T>(max_as_f64 * 0.039415044328304796);

  const T large = ConvertScalarTo<T>(3.512227595593985E18 * k1);
  const T neg_large = ScalarCopySign(large, n1);
  const T large2 = ConvertScalarTo<T>(2.1190576943127544E16 * k1);

  const T small = ConvertScalarTo<T>(1.067033284841808E-11 * k1);
  const T neg_small = ScalarCopySign(small, n1);
  const T small2 = ConvertScalarTo<T>(1.9401409532292856E-12 * k1);

  const T tiny = BitCastScalar<T>(static_cast<TU>(u1 << kNumOfMantBits));
  const T neg_tiny = ScalarCopySign(tiny, n1);

  const T tiny2 =
      ConvertScalarTo<T>(78.68466968859765 * ConvertScalarTo<double>(tiny));

  const AB test_cases[] = {{p0, p0},          {p0, n0},
                           {n0, n0},          {p1, p1},
                           {p1, n1},          {n1, n1},
                           {p2, p2},          {p2, n2},
                           {p2, pos},         {p2, neg},
                           {n2, pos},         {n2, neg},
                           {n2, n2},          {p0, tiny},
                           {p0, neg_tiny},    {n0, tiny},
                           {n0, neg_tiny},    {p1, tiny},
                           {p1, neg_tiny},    {n1, tiny},
                           {n1, neg_tiny},    {tiny, p0},
                           {tiny2, p0},       {tiny, tiny2},
                           {neg_tiny, tiny2}, {huge, huge2},
                           {neg_huge, huge2}, {huge, p0},
                           {huge, tiny},      {huge2, tiny2},
                           {large, p0},       {large, large2},
                           {neg_large, p0},   {neg_large, large2},
                           {small, p0},       {small, small2},
                           {neg_small, p0},   {neg_small, small2},
                           {max, p0},         {max, huge},
                           {max, max},        {p0, inf},
                           {n0, inf},         {p1, inf},
                           {n1, inf},         {p2, inf},
                           {n2, inf},         {p0, neg_inf},
                           {n0, neg_inf},     {p1, neg_inf},
                           {n1, neg_inf},     {p2, neg_inf},
                           {n2, neg_inf},     {p0, nan},
                           {n0, nan},         {p1, nan},
                           {n1, nan},         {p2, nan},
                           {n2, nan},         {huge, inf},
                           {inf, nan},        {neg_inf, nan},
                           {nan, nan}};

  const size_t kNumTestCases = sizeof(test_cases) / sizeof(test_cases[0]);
  const size_t N = Lanes(d);
  padded = RoundUpTo(kNumTestCases, N);  // allow loading whole vectors
  out_a = AllocateAligned<T>(padded);
  out_b = AllocateAligned<T>(padded);
  out_expected = AllocateAligned<T>(padded);
  HWY_ASSERT(out_a && out_b && out_expected);

  size_t i = 0;
  for (; i < kNumTestCases; ++i) {
    const T a =
        test_cases[i].a * hwy::ConvertScalarTo<T>(hwy::Unpredictable1());
    const T b = test_cases[i].b;

#if HWY_TARGET <= HWY_NEON_WITHOUT_AES && HWY_ARCH_ARM_V7
    // Ignore test cases that have infinite or NaN inputs on Armv7 NEON
    if (!ScalarIsFinite(a) || !ScalarIsFinite(b)) {
      out_a[i] = p0;
      out_b[i] = p0;
      out_expected[i] = p0;
      continue;
    }
#endif

    out_a[i] = a;
    out_b[i] = b;

    if (ScalarIsInf(a) || ScalarIsInf(b)) {
      out_expected[i] = inf;
    } else if (ScalarIsNaN(a) || ScalarIsNaN(b)) {
      out_expected[i] = nan;
    } else {
      out_expected[i] = std::hypot(a, b);
    }
  }
  for (; i < padded; ++i) {
    out_a[i] = p0;
    out_b[i] = p0;
    out_expected[i] = p0;
  }
}

struct TestHypot {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T t, D d) {
    if (HWY_MATH_TEST_EXCESS_PRECISION) {
      return;
    }

    const size_t N = Lanes(d);

    constexpr uint64_t kMaxErrorUlp = 4;

    size_t padded;
    AlignedFreeUniquePtr<T[]> in_a, in_b, expected;
    HypotTestCases(t, d, padded, in_a, in_b, expected);

    auto actual1_lanes = AllocateAligned<T>(N);
    auto actual2_lanes = AllocateAligned<T>(N);
    HWY_ASSERT(actual1_lanes && actual2_lanes);

    uint64_t max_ulp = 0;
    for (size_t i = 0; i < padded; i += N) {
      const auto a = Load(d, in_a.get() + i);
      const auto b = Load(d, in_b.get() + i);

#if HWY_ARCH_ARM_A64
      // TODO(b/287462770): inline to work around incorrect SVE codegen
      const auto actual1 = Hypot(d, a, b);
      const auto actual2 = Hypot(d, b, a);
#else
      const auto actual1 = CallHypot(d, a, b);
      const auto actual2 = CallHypot(d, b, a);
#endif

      Store(actual1, d, actual1_lanes.get());
      Store(actual2, d, actual2_lanes.get());

      for (size_t j = 0; j < N; j++) {
        const T val_a = in_a[i + j];
        const T val_b = in_b[i + j];
        const T expected_val = expected[i + j];
        const T actual1_val = actual1_lanes[j];
        const T actual2_val = actual2_lanes[j];

        const auto ulp1 =
            hwy::detail::ComputeUlpDelta(actual1_val, expected_val);
        if (ulp1 > kMaxErrorUlp) {
          fprintf(stderr,
                  "%s: Hypot(%e, %e) lane %d expected %E actual %E ulp %g max "
                  "ulp %u\n",
                  hwy::TypeName(T(), Lanes(d)).c_str(), val_a, val_b,
                  static_cast<int>(j), expected_val, actual1_val,
                  static_cast<double>(ulp1),
                  static_cast<uint32_t>(kMaxErrorUlp));
        }

        const auto ulp2 =
            hwy::detail::ComputeUlpDelta(actual2_val, expected_val);
        if (ulp2 > kMaxErrorUlp) {
          fprintf(stderr,
                  "%s: Hypot(%e, %e) expected %E actual %E ulp %g max ulp %u\n",
                  hwy::TypeName(T(), Lanes(d)).c_str(), val_b, val_a,
                  expected_val, actual2_val, static_cast<double>(ulp2),
                  static_cast<uint32_t>(kMaxErrorUlp));
        }

        max_ulp = HWY_MAX(max_ulp, HWY_MAX(ulp1, ulp2));
      }
    }

    if (max_ulp != 0) {
      fprintf(stderr, "%s: Hypot max_ulp %g\n",
              hwy::TypeName(T(), Lanes(d)).c_str(),
              static_cast<double>(max_ulp));
      HWY_ASSERT(max_ulp <= kMaxErrorUlp);
    }
  }
};

HWY_NOINLINE void TestAllHypot() {
  if (HWY_MATH_TEST_EXCESS_PRECISION) return;

  ForFloat3264Types(ForPartialVectors<TestHypot>());
}

struct TestFastTanRelative {
  template <class T, class D>
  HWY_NOINLINE void operator()(T, D d) {
    if (sizeof(T) == 4) {
      // Float: [-89.999999, +89.999999] deg
      // Max relative error is 0.002 to tolerate accuracy drop on architectures
      // without FMA support (like SSE4). On targets with FMA, the actual max
      // rel error is ~0.00045.
      TestMathRelative<T, D>(
          "FastTan", std::tan, CallFastTan, d, static_cast<T>(-1.570796309),
          static_cast<T>(1.570796309), (HWY_NATIVE_FMA ? 0.00045 : 0.002),
          4000, 1e-20);
    } else {
      // Double: [-89.999999999999, +89.999999999999] deg
      // Max relative error is 0.002 to tolerate accuracy drop on architectures
      // without FMA support (like SSE4). On targets with FMA, the actual max
      // rel error is ~0.00045.
      TestMathRelative<T, D>(
          "FastTan", std::tan, CallFastTan, d, static_cast<T>(-1.5707963267948),
          static_cast<T>(1.5707963267948), (HWY_NATIVE_FMA ? 0.00045 : 0.002),
          4000, 1e-20);
    }
  }
};

HWY_NOINLINE void TestAllFastTan() {
  if (HWY_MATH_TEST_EXCESS_PRECISION) return;
  ForFloat3264Types(ForPartialVectors<TestFastTanRelative>());
}

struct TestFastAtanRelative {
  template <class T, class D>
  HWY_NOINLINE void operator()(T, D d) {
    if (sizeof(T) == 4) {
      // Float: [-1e35, +1e35]
      TestMathRelative<T, D>("FastAtan", std::atan, CallFastAtan, d,
                             static_cast<T>(-1e35), static_cast<T>(1e35),
                             6e-6, 1000000, 1e-20);
      // Float: [0, +1e35]
      TestMathRelative<T, D>("FastAtanPositive", std::atan,
                             CallFastAtanPositive, d, static_cast<T>(0),
                             static_cast<T>(1e35), 6e-6, 1000000, 1e-20);
    } else {
      // Double: [-1e305, +1e305]
      TestMathRelative<T, D>("FastAtan", std::atan, CallFastAtan, d,
                             static_cast<T>(-1e305), static_cast<T>(1e305),
                             6e-6, 1000000, 1e-20);
      // Double: [0, +1e305]
      TestMathRelative<T, D>("FastAtanPositive", std::atan,
                             CallFastAtanPositive, d, static_cast<T>(0),
                             static_cast<T>(1e305), 6e-6, 1000000, 1e-20);
    }
  }
};

HWY_NOINLINE void TestAllFastAtan() {
  if (HWY_MATH_TEST_EXCESS_PRECISION) return;
  ForFloat3264Types(ForPartialVectors<TestFastAtanRelative>());
}

struct TestFastAtan2 {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T t, D d) {
    const size_t N = Lanes(d);

    size_t padded;
    AlignedFreeUniquePtr<T[]> in_y, in_x, expected;
    Atan2TestCases(t, d, padded, in_y, in_x, expected);

    // Constants for error checking
    const T rel_limit = static_cast<T>(6e-6);
    const T tiny_threshold = static_cast<T>(1e-20);
    const Vec<D> v_rel_limit = Set(d, rel_limit);
    const Vec<D> v_tiny_threshold = Set(d, tiny_threshold);

    for (size_t i = 0; i < padded; i += N) {
      const Vec<D> y = Load(d, &in_y[i]);
      const Vec<D> x = Load(d, &in_x[i]);
#if HWY_ARCH_ARM_A64
      // TODO(b/287462770): inline to work around incorrect SVE codegen
      const Vec<D> actual = FastAtan2(d, y, x);
#else
      const Vec<D> actual = CallFastAtan2(d, y, x);
#endif
      const Vec<D> vexpected = Load(d, &expected[i]);

      // 1. Check NaNs match exactly
      const Mask<D> exp_nan = IsNaN(vexpected);
      const Mask<D> act_nan = IsNaN(actual);
      HWY_ASSERT_MASK_EQ(d, exp_nan, act_nan);

      // 2. Calculate Error
      const Vec<D> abs_exp = Abs(vexpected);
      const Vec<D> diff = Abs(Sub(actual, vexpected));

      // 3. Determine Tolerance
      // If abs_exp > 1e-20, tolerance = abs_exp * 8e-8.
      // Else, tolerance = 8e-8 (effectively treating 'err' as the relative
      // error metric).
      const Mask<D> use_relative = Gt(abs_exp, v_tiny_threshold);
      const Vec<D> tolerance =
          IfThenElse(use_relative, Mul(abs_exp, v_rel_limit), v_rel_limit);

      // 4. Verify
      // Pass if it's NaN (checked above) OR if within tolerance
      const Mask<D> ok = Or(act_nan, Le(diff, tolerance));

      if (!AllTrue(d, ok)) {
        const size_t mismatch =
            static_cast<size_t>(FindKnownFirstTrue(d, Not(ok)));
        fprintf(stderr, "Mismatch for i=%d expected %E actual %E\n",
                static_cast<int>(i + mismatch), expected[i + mismatch],
                ExtractLane(actual, mismatch));
        HWY_ASSERT(0);
      }
    }
  }
};

HWY_NOINLINE void TestAllFastAtan2() {
  if (HWY_MATH_TEST_EXCESS_PRECISION) return;
  ForFloat3264Types(ForPartialVectors<TestFastAtan2>());
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(HwyMathTanTest);
HWY_EXPORT_AND_TEST_P(HwyMathTanTest, TestAllTan);
HWY_EXPORT_AND_TEST_P(HwyMathTanTest, TestAllAtan);
HWY_EXPORT_AND_TEST_P(HwyMathTanTest, TestAllAtan2);
HWY_EXPORT_AND_TEST_P(HwyMathTanTest, TestAllHypot);
HWY_EXPORT_AND_TEST_P(HwyMathTanTest, TestAllFastTan);
HWY_EXPORT_AND_TEST_P(HwyMathTanTest, TestAllFastAtan);
HWY_EXPORT_AND_TEST_P(HwyMathTanTest, TestAllFastAtan2);

HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE

#endif  // HWY_ARCH_RVV
