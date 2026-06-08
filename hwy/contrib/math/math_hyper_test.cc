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

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/math/math_hyper_test.cc"
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

// Floating point values closest to but less than 1.0. Avoid variables with
// static initializers inside HWY_BEFORE_NAMESPACE/HWY_AFTER_NAMESPACE to
// ensure target-specific code does not leak into startup code.
float kNearOneF() { return BitCastScalar<float>(0x3F7FFFFF); }
double kNearOneD() { return BitCastScalar<double>(0x3FEFFFFFFFFFFFFFULL); }

constexpr uint64_t ACosh32ULP() {
#if defined(__MINGW32__)
  return 8;
#else
  return 3;
#endif
}

// clang-format off
DEFINE_MATH_TEST(Acosh,
  std::acosh, CallAcosh, +1.0f,      +FLT_MAX,    ACosh32ULP(),
  std::acosh, CallAcosh, +1.0,       +DBL_MAX,    3)
DEFINE_MATH_TEST(Asinh,
  std::asinh, CallAsinh, -FLT_MAX,   +FLT_MAX,    3,
  std::asinh, CallAsinh, -DBL_MAX,   +DBL_MAX,    3)
// NEON has ULP 4 instead of 3
DEFINE_MATH_TEST(Atanh,
  std::atanh, CallAtanh, -kNearOneF(), +kNearOneF(),  4,
  std::atanh, CallAtanh, -kNearOneD(), +kNearOneD(),  3)
DEFINE_MATH_TEST(Sinh,
  std::sinh,  CallSinh,  -80.0f,     +80.0f,      4,
  std::sinh,  CallSinh,  -709.0,     +709.0,      4)
DEFINE_MATH_TEST(Cosh,
  std::cosh,  CallCosh,  -80.0f,     +80.0f,      4,
  std::cosh,  CallCosh,  -709.0,     +709.0,      4)
DEFINE_MATH_TEST(Tanh,
  std::tanh,  CallTanh,  -FLT_MAX,   +FLT_MAX,    4,
  std::tanh,  CallTanh,  -DBL_MAX,   +DBL_MAX,    4)
// clang-format on

struct TestFastTanh {
  template <class T, class D>
  HWY_NOINLINE void operator()(T, D d) {
    const double max_relative_error_float = 0.000007;
    const double max_relative_error_double = 0.000007;
    const double max_relative_error_small = 0.0000004;
    const uint64_t samples = 1000000;
    const uint64_t samples_small = 10000;
    TestMathRelative<T, D>("FastTanh Small", std::tanh, CallFastTanh, d,
                             static_cast<T>(-1e-2), static_cast<T>(1e-2),
                             max_relative_error_small, samples_small);
    if (sizeof(T) == 4) {
      TestMathRelative<T, D>("FastTanh Float", std::tanh, CallFastTanh, d,
                             static_cast<T>(-1e35), static_cast<T>(1e35),
                             max_relative_error_float, samples);
    } else {
      TestMathRelative<T, D>("FastTanh Double", std::tanh, CallFastTanh, d,
                             static_cast<T>(-1e305), static_cast<T>(1e305),
                             max_relative_error_double, samples);
    }
  }
};

HWY_NOINLINE void TestAllFastTanh() {
  if (HWY_MATH_TEST_EXCESS_PRECISION) return;
  ForFloat3264Types(ForPartialVectors<TestFastTanh>());
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(HwyMathHyperTest);
HWY_EXPORT_AND_TEST_P(HwyMathHyperTest, TestAllAcosh);
HWY_EXPORT_AND_TEST_P(HwyMathHyperTest, TestAllAsinh);
HWY_EXPORT_AND_TEST_P(HwyMathHyperTest, TestAllAtanh);
HWY_EXPORT_AND_TEST_P(HwyMathHyperTest, TestAllSinh);
HWY_EXPORT_AND_TEST_P(HwyMathHyperTest, TestAllCosh);
HWY_EXPORT_AND_TEST_P(HwyMathHyperTest, TestAllTanh);
HWY_EXPORT_AND_TEST_P(HwyMathHyperTest, TestAllFastTanh);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
