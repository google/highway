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

#include <cmath>  // std::exp

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

// The float16 min/max bounds mirror the float32 bounds in math_test.cc where
// the kernel is validated (e.g. +104 for Exp), clamped to the float16 finite
// range [-65504, +65504]. The smallest positive float16 subnormal is 2^-24.
// clang-format off
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

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(HwyF16MathTest);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Exp);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Exp2);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Expm1);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Log);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Log10);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Log1p);
HWY_EXPORT_AND_TEST_P(HwyF16MathTest, TestAllF16Log2);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
