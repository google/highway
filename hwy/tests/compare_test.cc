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

#include <stddef.h>
#include <stdint.h>
#include <string.h>  // memset

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/compare_test.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// All types.
struct TestMask {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);

    std::fill(lanes.get(), lanes.get() + N, T(0));
    const auto actual_false = MaskFromVec(Load(d, lanes.get()));
    HWY_ASSERT_MASK_EQ(d, MaskFalse(d), actual_false);

    memset(lanes.get(), 0xFF, N * sizeof(T));
    const auto actual_true = MaskFromVec(Load(d, lanes.get()));
    HWY_ASSERT_MASK_EQ(d, MaskTrue(d), actual_true);
  }
};

HWY_NOINLINE void TestAllMask() { ForAllTypes(ForPartialVectors<TestMask>()); }

// All types.
struct TestEquality {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v2 = Iota(d, 2);
    const auto v2b = Iota(d, 2);
    const auto v3 = Iota(d, 3);

    const auto mask_false = MaskFalse(d);
    const auto mask_true = MaskTrue(d);

    HWY_ASSERT_MASK_EQ(d, mask_false, Eq(v2, v3));
    HWY_ASSERT_MASK_EQ(d, mask_true, Eq(v2, v2));
    HWY_ASSERT_MASK_EQ(d, mask_true, Eq(v2, v2b));
  }
};

HWY_NOINLINE void TestAllEquality() {
  ForAllTypes(ForPartialVectors<TestEquality>());
}

// a > b should be true, verify that for Gt/Lt and with swapped args.
template <class D>
void EnsureGreater(D d, TFromD<D> a, TFromD<D> b) {
  const auto mask_false = MaskFalse(d);
  const auto mask_true = MaskTrue(d);

  const auto va = Set(d, a);
  const auto vb = Set(d, b);
  HWY_ASSERT_MASK_EQ(d, mask_true, Gt(va, vb));
  HWY_ASSERT_MASK_EQ(d, mask_false, Lt(va, vb));

  // Swapped order
  HWY_ASSERT_MASK_EQ(d, mask_false, Gt(vb, va));
  HWY_ASSERT_MASK_EQ(d, mask_true, Lt(vb, va));

  // Also ensure irreflexive
  HWY_ASSERT_MASK_EQ(d, mask_false, Gt(va, va));
  HWY_ASSERT_MASK_EQ(d, mask_false, Gt(vb, vb));
  HWY_ASSERT_MASK_EQ(d, mask_false, Lt(va, va));
  HWY_ASSERT_MASK_EQ(d, mask_false, Lt(vb, vb));
}

// Integer and floating-point.
struct TestStrict {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const T min = LimitsMin<T>();
    const T max = LimitsMax<T>();
    const auto v0 = Zero(d);
    const auto v2 = And(Iota(d, T(2)), Set(d, 127));  // 0..127
    const auto vn = Neg(v2) - Set(d, 1);              // -1..-128

    const auto mask_false = MaskFalse(d);
    const auto mask_true = MaskTrue(d);

    // Individual values of interest
    EnsureGreater(d, 2, 1);
    EnsureGreater(d, 1, 0);
    EnsureGreater(d, 0, -1);
    EnsureGreater(d, -1, -2);
    EnsureGreater(d, max, max / 2);
    EnsureGreater(d, max, 1);
    EnsureGreater(d, max, 0);
    EnsureGreater(d, max, -1);
    EnsureGreater(d, max, min);
    EnsureGreater(d, 0, min);
    EnsureGreater(d, min / 2, min);

    // Also use Iota to ensure lanes are independent
    HWY_ASSERT_MASK_EQ(d, mask_true, Gt(v2, vn));
    HWY_ASSERT_MASK_EQ(d, mask_true, Lt(vn, v2));
    HWY_ASSERT_MASK_EQ(d, mask_false, Lt(v2, vn));
    HWY_ASSERT_MASK_EQ(d, mask_false, Gt(vn, v2));

    HWY_ASSERT_MASK_EQ(d, mask_false, Lt(v0, v0));
    HWY_ASSERT_MASK_EQ(d, mask_false, Lt(v2, v2));
    HWY_ASSERT_MASK_EQ(d, mask_false, Lt(vn, vn));
    HWY_ASSERT_MASK_EQ(d, mask_false, Gt(v0, v0));
    HWY_ASSERT_MASK_EQ(d, mask_false, Gt(v2, v2));
    HWY_ASSERT_MASK_EQ(d, mask_false, Gt(vn, vn));
  }
};

HWY_NOINLINE void TestAllStrict() {
  const ForExtendableVectors<TestStrict> test;
  ForSignedTypes(test);
  ForFloatTypes(test);
}

// Floating-point.
struct TestWeak {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v2 = Iota(d, 2);
    const auto vn = Iota(d, -T(Lanes(d)));

    const auto mask_false = MaskFalse(d);
    const auto mask_true = MaskTrue(d);

    HWY_ASSERT_MASK_EQ(d, mask_true, Ge(v2, v2));
    HWY_ASSERT_MASK_EQ(d, mask_true, Le(vn, vn));

    HWY_ASSERT_MASK_EQ(d, mask_true, Ge(v2, vn));
    HWY_ASSERT_MASK_EQ(d, mask_true, Le(vn, v2));

    HWY_ASSERT_MASK_EQ(d, mask_false, Le(v2, vn));
    HWY_ASSERT_MASK_EQ(d, mask_false, Ge(vn, v2));
  }
};

HWY_NOINLINE void TestAllWeak() {
  ForFloatTypes(ForPartialVectors<TestWeak>());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
HWY_BEFORE_TEST(HwyCompareTest);
HWY_EXPORT_AND_TEST_P(HwyCompareTest, TestAllEquality);
HWY_EXPORT_AND_TEST_P(HwyCompareTest, TestAllStrict);
HWY_EXPORT_AND_TEST_P(HwyCompareTest, TestAllWeak);
HWY_AFTER_TEST();
#endif
