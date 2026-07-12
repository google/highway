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

// For faster tests. Not using AES, hence NEON_WITHOUT_AES is sufficient.
// SVE is mostly superseded by SVE2.
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_NEON | HWY_SVE)
#endif  // HWY_DISABLED_TARGETS

#include "hwy/base.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/math/math_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/contrib/math/math_test-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

// clang-format off
DEFINE_MATH_TEST(Erf,
  std::erf,   CallErf,   -FLT_MAX,   +FLT_MAX,    4,
  std::erf,   CallErf,   -DBL_MAX,   +DBL_MAX,    4)
DEFINE_MATH_TEST(Exp,
  std::exp,   CallExp,   -FLT_MAX,   +104.0f,     1,
  std::exp,   CallExp,   -DBL_MAX,   +104.0,      1)
DEFINE_MATH_TEST(Exp2,
  std::exp2,  CallExp2,  -FLT_MAX,   +128.0f,     2,
  std::exp2,  CallExp2,  -DBL_MAX,   +128.0,      2)
DEFINE_MATH_TEST(Expm1,
  std::expm1, CallExpm1, -FLT_MAX,   +104.0f,     4,
  std::expm1, CallExpm1, -DBL_MAX,   +104.0,      4)
DEFINE_MATH_TEST(Log,
  std::log,   CallLog,   +FLT_MIN,   +FLT_MAX,    1,
  std::log,   CallLog,   +DBL_MIN,   +DBL_MAX,    1)
DEFINE_MATH_TEST(Log10,
  std::log10, CallLog10, +FLT_MIN,   +FLT_MAX,    2,
  std::log10, CallLog10, +DBL_MIN,   +DBL_MAX,    2)
DEFINE_MATH_TEST(Log1p,
  std::log1p, CallLog1p, +0.0f,      +FLT_MAX,    3,  // NEON is 3 instead of 2
  std::log1p, CallLog1p, +0.0,       +DBL_MAX,    2)
DEFINE_MATH_TEST(Log2,
  std::log2,  CallLog2,  +FLT_MIN,   +FLT_MAX,    2,
  std::log2,  CallLog2,  +DBL_MIN,   +DBL_MAX,    2)
DEFINE_MATH_TEST(Cbrt,
  std::cbrt, CallCbrt, -FLT_MAX, +FLT_MAX, 6,
  std::cbrt, CallCbrt, -DBL_MAX, +DBL_MAX, 6)
DEFINE_MATH_TEST(Tgamma,
  std::tgamma, CallTgamma, +0.5f, +35.0f,  6,
  std::tgamma, CallTgamma, +0.5,  +171.6,  8)
DEFINE_MATH_TEST(Lgamma,
  std::lgamma, CallLgamma, +0.5f, +1000.0f,  6,
  std::lgamma, CallLgamma, +0.5,  +1000.0,   10)
// clang-format on

struct TestPow {
  template <class D>
  static HWY_INLINE void CheckPowResult(const D d, const TFromD<D> expected,
                                        const TFromD<D> actual,
                                        const TFromD<D> a, const TFromD<D> b,
                                        uint64_t& max_ulp,
                                        const uint64_t max_error_ulp) {
    using T = TFromD<D>;

    const auto ulp = hwy::detail::ComputeUlpDelta(actual, expected);
    max_ulp = HWY_MAX(max_ulp, ulp);
    if (ulp > max_error_ulp) {
      fprintf(stderr,
              "%s: Pow(%f, %f) expected %E actual %E ulp %g max ulp %u\n",
              hwy::TypeName(T(), Lanes(d)).c_str(), static_cast<double>(a),
              static_cast<double>(b), static_cast<double>(expected),
              static_cast<double>(actual), static_cast<double>(ulp),
              static_cast<uint32_t>(max_error_ulp));
    }
  }

  template <class T, class D>
  HWY_NOINLINE void operator()(T, D d) {
    if (HWY_MATH_TEST_EXCESS_PRECISION) {
      return;
    }

    using UintT = MakeUnsigned<T>;

    constexpr uint64_t kMaxErrorUlp = 8;

    uint64_t max_ulp = 0;
    // Emulation is slower, so cannot afford as many.
    constexpr UintT kSamplesPerRange =
        static_cast<UintT>(AdjustedReps(static_cast<size_t>(32)));

    constexpr int kNumOfMantFracBits = MantissaBits<T>();
    static_assert(kNumOfMantFracBits > 0,
                  "kNumOfMantFracBits > 0 must be true");

    constexpr int kExpBias = static_cast<int>(MaxExponentField<T>() >> 1);
    static_assert(kExpBias > 0, "kExpBias > 0 must be true");

    constexpr UintT kMaxMantFracBits = MantissaMask<T>();
    static_assert(kMaxMantFracBits > 1, "kMaxMantFracBits > 1 must be true");
    static_assert(kMaxMantFracBits < LimitsMax<UintT>(),
                  "kMaxMantFracBits < LimitsMax<UintT>() must be true");

    for (int base_exp = -3; base_exp <= 3; base_exp++) {
      const UintT base_exp_bits = static_cast<UintT>(base_exp + kExpBias)
                                  << kNumOfMantFracBits;

      constexpr UintT step =
          HWY_MAX(1, ((kMaxMantFracBits - 1) / kSamplesPerRange));
      for (UintT base_mant_frac_bits = 1;
           base_mant_frac_bits <= kMaxMantFracBits;
           base_mant_frac_bits += step) {
        const T base = BitCastScalar<T>(
            static_cast<UintT>(base_exp_bits | base_mant_frac_bits));

#if HWY_TARGET <= HWY_NEON_WITHOUT_AES && HWY_ARCH_ARM_V7
        if (std::abs(base) < 1e-37f) {
          continue;
        }
#endif

        const auto v_base = Set(d, base);
        const auto v_neg_base = Neg(v_base);

        for (int exp_of_pow_exp = -4; exp_of_pow_exp <= 4; exp_of_pow_exp++) {
          const UintT exp_bits_of_pow_exp =
              static_cast<UintT>(exp_of_pow_exp + kExpBias)
              << kNumOfMantFracBits;
          for (UintT pow_exp_mant_frac_bits = 1;
               pow_exp_mant_frac_bits <= kMaxMantFracBits;
               pow_exp_mant_frac_bits += step) {
            const T pow_exp = BitCastScalar<T>(static_cast<UintT>(
                exp_bits_of_pow_exp | pow_exp_mant_frac_bits));
            const T int_pow_exp = static_cast<T>(static_cast<int>(pow_exp));

            const T expected0 = std::pow(base, pow_exp);
            const T expected1 = std::pow(-base, int_pow_exp);

#if HWY_TARGET <= HWY_NEON_WITHOUT_AES && HWY_ARCH_ARM_V7
            if ((std::abs(pow_exp) < 1e-37f) ||
                (std::abs(expected0) < 1e-37f) ||
                (std::abs(expected1) < 1e-37f)) {
              continue;
            }
#endif

            const T actual0 = GetLane(CallPow(d, v_base, Set(d, pow_exp)));
            CheckPowResult(d, expected0, actual0, base, pow_exp, max_ulp,
                           kMaxErrorUlp);

            const T actual1 =
                GetLane(CallPow(d, v_neg_base, Set(d, int_pow_exp)));
            CheckPowResult(d, expected1, actual1, -base, int_pow_exp, max_ulp,
                           kMaxErrorUlp);
          }
        }
      }
    }

    fprintf(stderr, "%s: Pow max_ulp %g\n",
            hwy::TypeName(T(), Lanes(d)).c_str(), static_cast<double>(max_ulp));
    HWY_ASSERT(max_ulp <= kMaxErrorUlp);
  }
};

HWY_NOINLINE void TestAllPow() {
  ForFloat3264Types(ForPartialVectors<TestPow>());
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(HwyMathTest);
HWY_EXPORT_AND_TEST_P(HwyMathTest, TestAllErf);
HWY_EXPORT_AND_TEST_P(HwyMathTest, TestAllExp);
HWY_EXPORT_AND_TEST_P(HwyMathTest, TestAllExp2);
HWY_EXPORT_AND_TEST_P(HwyMathTest, TestAllExpm1);
HWY_EXPORT_AND_TEST_P(HwyMathTest, TestAllLog);
HWY_EXPORT_AND_TEST_P(HwyMathTest, TestAllLog10);
HWY_EXPORT_AND_TEST_P(HwyMathTest, TestAllLog1p);
HWY_EXPORT_AND_TEST_P(HwyMathTest, TestAllLog2);
HWY_EXPORT_AND_TEST_P(HwyMathTest, TestAllCbrt);
HWY_EXPORT_AND_TEST_P(HwyMathTest, TestAllTgamma);
HWY_EXPORT_AND_TEST_P(HwyMathTest, TestAllLgamma);
HWY_EXPORT_AND_TEST_P(HwyMathTest, TestAllPow);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
