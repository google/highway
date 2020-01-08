// Copyright 2019 Google LLC
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

#ifndef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/compare_test.cc"

#include "hwy/tests/test_util.h"
struct CompareTest {
  HWY_DECLARE(void, ())
};
TEST(HwyCompareTest, Run) { hwy::RunTests<CompareTest>(); }

#endif  // HWY_TARGET_INCLUDE
#include "hwy/tests/test_target_util.h"

namespace hwy {
namespace HWY_NAMESPACE {
namespace {

constexpr HWY_FULL(uint8_t) du8;
constexpr HWY_FULL(uint16_t) du16;
constexpr HWY_FULL(uint32_t) du32;
constexpr HWY_FULL(uint64_t) du64;
constexpr HWY_FULL(int8_t) di8;
constexpr HWY_FULL(int16_t) di16;
constexpr HWY_FULL(int32_t) di32;
constexpr HWY_FULL(int64_t) di64;
constexpr HWY_FULL(float) df;
constexpr HWY_FULL(double) dd;

template <class D>
HWY_NOINLINE HWY_ATTR void TestEquality(D d) {
  const auto v2 = Iota(d, 2);
  const auto v2b = Iota(d, 2);
  const auto v3 = Iota(d, 3);

  HWY_ASSERT_EQ(false, ext::AllTrue(v2 == v3));
  HWY_ASSERT_EQ(true, ext::AllFalse(v2 == v3));
  HWY_ASSERT_EQ(true, ext::AllTrue(v2 == v2b));
  HWY_ASSERT_EQ(false, ext::AllFalse(v2 == v2b));
}

// Integer and floating-point.
template <class D>
HWY_NOINLINE HWY_ATTR void TestInequality(D d) {
  using T = typename D::T;
  const auto v2 = Iota(d, 2);
  const auto vn = Iota(d, -T(d.N));

  HWY_ASSERT_EQ(true, ext::AllTrue(v2 > vn));
  HWY_ASSERT_EQ(true, ext::AllTrue(vn < v2));
  HWY_ASSERT_EQ(true, ext::AllFalse(v2 < vn));
  HWY_ASSERT_EQ(true, ext::AllFalse(vn > v2));
}

HWY_NOINLINE HWY_ATTR void TestCompare() {
  (void)dd;
  (void)di64;
  (void)du64;

  HWY_FOREACH_UIF(TestEquality);

  HWY_FOREACH_F(TestInequality);
  TestInequality(di8);
  TestInequality(di16);
  TestInequality(di32);
#if HWY_HAS_CMP64
  TestInequality(di64);
#endif
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy

// Instantiate for the current target.
void CompareTest::HWY_FUNC() { hwy::HWY_NAMESPACE::TestCompare(); }
