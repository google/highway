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

#include <cmath>   // std::abs

#include "hwy/base.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/math/math_trig_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/contrib/math/math_test-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

// The discrepancy is unacceptably large for MSYS2 (less accurate libm?), so
// only increase the error tolerance there.
constexpr uint64_t Cos64ULP() {
#if defined(__MINGW32__)
  return 23;
#else
  return 3;
#endif
}

template <class D>
static Vec<D> SinCosSin(const D d, VecArg<Vec<D>> x) {
  Vec<D> s, c;
  CallSinCos(d, x, s, c);
  return s;
}

template <class D>
static Vec<D> SinCosCos(const D d, VecArg<Vec<D>> x) {
  Vec<D> s, c;
  CallSinCos(d, x, s, c);
  return c;
}

// on targets without FMA the result is less inaccurate
constexpr uint64_t SinCosSin32ULP() {
#if !(HWY_NATIVE_FMA)
  return 256;
#else
  return 3;
#endif
}

constexpr uint64_t SinCosCos32ULP() {
#if !(HWY_NATIVE_FMA)
  return 64;
#else
  return 3;
#endif
}

// clang-format off
DEFINE_MATH_TEST(Acos,
  std::acos,  CallAcos,  -1.0f,      +1.0f,       3,  // NEON is 3 instead of 2
  std::acos,  CallAcos,  -1.0,       +1.0,        2)
DEFINE_MATH_TEST(Asin,
  std::asin,  CallAsin,  -1.0f,      +1.0f,       4,  // 4 ulp on Armv7, not 2
  std::asin,  CallAsin,  -1.0,       +1.0,        2)
// NEON has ULP 4 instead of 3
DEFINE_MATH_TEST(Cos,
  std::cos,   CallCos,   -39000.0f,  +39000.0f,   3,
  std::cos,   CallCos,   -39000.0,   +39000.0,    Cos64ULP())
DEFINE_MATH_TEST(Sin,
  std::sin,   CallSin,   -39000.0f,  +39000.0f,   3,
  std::sin,   CallSin,   -39000.0,   +39000.0,    4)  // MSYS is 4 instead of 3
DEFINE_MATH_TEST(SinCosSin,
  std::sin,   SinCosSin,   -39000.0f,  +39000.0f,   SinCosSin32ULP(),
  std::sin,   SinCosSin,   -39000.0,   +39000.0,    1)
DEFINE_MATH_TEST(SinCosCos,
  std::cos,   SinCosCos,   -39000.0f,  +39000.0f,   SinCosCos32ULP(),
  std::cos,   SinCosCos,   -39000.0,   +39000.0,    1)
// clang-format on

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(HwyMathTrigTest);
HWY_EXPORT_AND_TEST_P(HwyMathTrigTest, TestAllAcos);
HWY_EXPORT_AND_TEST_P(HwyMathTrigTest, TestAllAsin);
HWY_EXPORT_AND_TEST_P(HwyMathTrigTest, TestAllCos);
HWY_EXPORT_AND_TEST_P(HwyMathTrigTest, TestAllSin);
HWY_EXPORT_AND_TEST_P(HwyMathTrigTest, TestAllSinCosSin);
HWY_EXPORT_AND_TEST_P(HwyMathTrigTest, TestAllSinCosCos);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
