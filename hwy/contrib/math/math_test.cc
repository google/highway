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

#include <cmath>

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/math/math_test.cc"
#include "hwy/foreach_target.h"

#include "hwy/contrib/math/math-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

template <class Out, class In>
inline Out BitCast(const In& in) {
  static_assert(sizeof(Out) == sizeof(In), "");
  Out out;
  std::memcpy(&out, &in, sizeof(out));
  return out;
}

// Computes the difference in units of last place between x and y.
inline int32_t ComputeUlpDelta(float x, float y) {
  const uint32_t ux = BitCast<uint32_t>(x);
  const uint32_t uy = BitCast<uint32_t>(y);
  return std::abs(BitCast<int32_t>(ux - uy));
}
inline int64_t ComputeUlpDelta(double x, double y) {
  const uint64_t ux = BitCast<uint64_t>(x);
  const uint64_t uy = BitCast<uint64_t>(y);
  return std::abs(BitCast<int64_t>(ux - uy));
}

struct TestExp {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    constexpr bool kIsF32 = (sizeof(T) == 4);
    constexpr T kMin = (kIsF32 ? -FLT_MAX : -DBL_MAX);
    constexpr T kMax = (kIsF32 ? +104.0 : +706.0);

    // clang-format off
    constexpr int kValueCount = 13;
    const T kTestValues[kValueCount] {
      kMin, -1e20, -1e10, -1000, -100, -10, -1, 0, 0.123456, +1, +10, +100, kMax
    };  // clang-format on

    uint64_t max_ulp = 0;
    for (int i = 0; i < kValueCount; ++i) {
      const T value = kTestValues[i];
      const auto actual = GetLane(Exp(Set(d, value)));
      const auto expected = std::exp(value);
      const auto ulp = ComputeUlpDelta(actual, expected);
      max_ulp = std::max<uint64_t>(max_ulp, ulp);
      ASSERT_LE(ulp, 1) << "expected: " << expected << " actual: " << actual;
    }
    std::cout << (kIsF32 ? "F32x" : "F64x") << Lanes(d)
              << ", Max Error(ULP): " << max_ulp << std::endl;
  }
};

HWY_NOINLINE void TestAllExp() { ForFloatTypes(ForFullVectors<TestExp>()); }

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {

class HwyMathTest : public hwy::TestWithParamTarget {};

HWY_TARGET_INSTANTIATE_TEST_SUITE_P(HwyMathTest);
HWY_EXPORT_AND_TEST_P(HwyMathTest, TestAllExp);

}  // namespace hwy
#endif
