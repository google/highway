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

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/compare_test.cc"
#include "hwy/foreach_target.h"
// ^ must come before highway.h and any *-inl.h.

#include <string.h>  // memset

#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// All types.
struct TestEquality {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v2 = Iota(d, 2);
    const auto v2b = Iota(d, 2);
    const auto v3 = Iota(d, 3);

    HWY_ALIGN const T all_false[MaxLanes(d)] = {};
    HWY_ALIGN T all_true[MaxLanes(d)];
    memset(all_true, 0xFF, sizeof(all_true));

    HWY_ASSERT_VEC_EQ(d, all_false, VecFromMask(v2 == v3));
    HWY_ASSERT_VEC_EQ(d, all_true, VecFromMask(v2 == v2));
    HWY_ASSERT_VEC_EQ(d, all_true, VecFromMask(v2 == v2b));
  }
};

// Integer and floating-point.
struct TestStrictT {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v2 = Iota(d, 2);
    const auto vn = Iota(d, -T(Lanes(d)));

    HWY_ALIGN const T all_false[MaxLanes(d)] = {};
    HWY_ALIGN T all_true[MaxLanes(d)];
    memset(all_true, 0xFF, sizeof(all_true));

    HWY_ASSERT_VEC_EQ(d, all_true, VecFromMask(v2 > vn));
    HWY_ASSERT_VEC_EQ(d, all_true, VecFromMask(vn < v2));
    HWY_ASSERT_VEC_EQ(d, all_false, VecFromMask(v2 < vn));
    HWY_ASSERT_VEC_EQ(d, all_false, VecFromMask(vn > v2));
  }
};

HWY_NOINLINE void TestStrict() {
  const ForPartialVectors<TestStrictT> test;

  // Cannot use ForSignedTypes - need to check HWY_COMPARE64_LANES.
  test(int8_t());
  test(int16_t());
  test(int32_t());
#if HWY_COMPARE64_LANES > 1
  test(int64_t());
#endif

  ForFloatTypes(test);
}

// Floating-point.
struct TestWeak {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v2 = Iota(d, 2);
    const auto vn = Iota(d, -T(Lanes(d)));

    HWY_ALIGN const T all_false[MaxLanes(d)] = {};
    HWY_ALIGN T all_true[MaxLanes(d)];
    memset(all_true, 0xFF, sizeof(all_true));

    HWY_ASSERT_VEC_EQ(d, all_true, VecFromMask(v2 >= v2));
    HWY_ASSERT_VEC_EQ(d, all_true, VecFromMask(vn <= vn));

    HWY_ASSERT_VEC_EQ(d, all_true, VecFromMask(v2 >= vn));
    HWY_ASSERT_VEC_EQ(d, all_true, VecFromMask(vn <= v2));

    HWY_ASSERT_VEC_EQ(d, all_false, VecFromMask(v2 <= vn));
    HWY_ASSERT_VEC_EQ(d, all_false, VecFromMask(vn >= v2));
  }
};

HWY_NOINLINE void TestAllEquality() {
  ForAllTypes(ForPartialVectors<TestEquality>());
}

HWY_NOINLINE void TestAllWeak() {
  ForFloatTypes(ForPartialVectors<TestWeak>());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {

class HwyCompareTest : public hwy::TestWithParamTarget {};

HWY_TARGET_INSTANTIATE_TEST_SUITE_P(HwyCompareTest);

HWY_EXPORT_AND_TEST_P(HwyCompareTest, TestAllEquality);
HWY_EXPORT_AND_TEST_P(HwyCompareTest, TestStrict);
HWY_EXPORT_AND_TEST_P(HwyCompareTest, TestAllWeak);

}  // namespace hwy
#endif
