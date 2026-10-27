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

#include "hwy/base.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/convert_fast_math_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

#if HWY_HAVE_FLOAT64 && HWY_HAVE_INTEGER64
HWY_NOINLINE double ScalarConvert(int64_t value) {
  return static_cast<double>(value);
}

struct TestI64ToDouble {
  template <class D>
  HWY_NOINLINE void operator()(double /*unused*/, D d) {
    const RebindToSigned<D> di;
    const RebindToUnsigned<D> du;
    const size_t N = Lanes(d);
    auto input = AllocateAligned<int64_t>(N);
    auto expected = AllocateAligned<double>(N);
    HWY_ASSERT(input && expected);

    const int64_t cases[] = {0,
                             -13,
                             13,
                             -1,
                             1,
                             LimitsMin<int64_t>(),
                             LimitsMax<int64_t>(),
                             -(int64_t{1} << 32) - 1,
                             -(int64_t{1} << 32),
                             -(int64_t{1} << 32) + 1,
                             (int64_t{1} << 32) - 1,
                             int64_t{1} << 32,
                             (int64_t{1} << 32) + 1,
                             -(int64_t{1} << 53) - 1,
                             -(int64_t{1} << 53),
                             -(int64_t{1} << 53) + 1,
                             (int64_t{1} << 53) - 1,
                             int64_t{1} << 53,
                             (int64_t{1} << 53) + 1};
    constexpr size_t kNumCases = sizeof(cases) / sizeof(cases[0]);
    RandomState rng;
    for (size_t rep = 0; rep < kNumCases + AdjustedReps(1000); ++rep) {
      for (size_t i = 0; i < N; ++i) {
        if (rep < kNumCases) {
          input[i] = cases[(rep + i) % kNumCases];
        } else {
          const uint64_t bits = rng();
          CopyBytes<sizeof(bits)>(&bits, &input[i]);
        }
        expected[i] = ScalarConvert(input[i]);
      }
      HWY_ASSERT_VEC_EQ(du, BitCast(du, Load(d, expected.get())),
                        BitCast(du, ConvertTo(d, Load(di, input.get()))));
    }
  }
};
#endif

HWY_NOINLINE void TestAllI64ToDouble() {
#if HWY_HAVE_FLOAT64 && HWY_HAVE_INTEGER64
  ForPartialVectors<TestI64ToDouble>()(double());
#endif
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(HwyConvertFastMathTest);
HWY_EXPORT_AND_TEST_P(HwyConvertFastMathTest, TestAllI64ToDouble);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
