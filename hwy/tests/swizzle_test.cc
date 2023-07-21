// Copyright 2019 Google LLC
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

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/swizzle_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

struct TestGetLane {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v = Iota(d, T(1));
    HWY_ASSERT_EQ(T(1), GetLane(v));
  }
};

HWY_NOINLINE void TestAllGetLane() {
  ForAllTypes(ForPartialVectors<TestGetLane>());
}

struct TestExtractLane {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v = Iota(d, T(1));
    for (size_t i = 0; i < Lanes(d); ++i) {
      const T actual = ExtractLane(v, i);
      HWY_ASSERT_EQ(static_cast<T>(i + 1), actual);
    }
  }
};

HWY_NOINLINE void TestAllExtractLane() {
  ForAllTypes(ForPartialVectors<TestExtractLane>());
}

struct TestInsertLane {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    using V = Vec<D>;
    const V v = Iota(d, T(1));
    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);
    HWY_ASSERT(lanes);
    Store(v, d, lanes.get());

    for (size_t i = 0; i < Lanes(d); ++i) {
      lanes[i] = T{0};
      const V actual = InsertLane(v, i, static_cast<T>(i + 1));
      HWY_ASSERT_VEC_EQ(d, v, actual);
      Store(v, d, lanes.get());  // restore lane i
    }
  }
};

HWY_NOINLINE void TestAllInsertLane() {
  ForAllTypes(ForPartialVectors<TestInsertLane>());
}

struct TestDupEven {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(expected);
    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>((i & ~size_t{1}) + 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), DupEven(Iota(d, 1)));
  }
};

HWY_NOINLINE void TestAllDupEven() {
  ForAllTypes(ForShrinkableVectors<TestDupEven>());
}

struct TestDupOdd {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
#if HWY_TARGET != HWY_SCALAR
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(expected);
    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>((i & ~size_t{1}) + 2);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), DupOdd(Iota(d, 1)));
#else
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllDupOdd() {
  ForAllTypes(ForShrinkableVectors<TestDupOdd>());
}

struct TestOddEven {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    const auto even = Iota(d, 1);
    const auto odd = Iota(d, static_cast<T>(1 + N));
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(expected);
    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>(1 + i + ((i & 1) ? N : 0));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), OddEven(odd, even));
  }
};

HWY_NOINLINE void TestAllOddEven() {
  ForAllTypes(ForShrinkableVectors<TestOddEven>());
}

class TestBroadcastLane {
 private:
  template <int kLane, class D,
            HWY_IF_LANES_GT_D(D, static_cast<size_t>(kLane))>
  static HWY_INLINE void DoTestBroadcastLane(D d, const size_t N) {
    using T = TFromD<D>;
    using TU = MakeUnsigned<T>;
    // kLane < HWY_MAX_LANES_D(D) is true
    if (kLane >= N) return;

    constexpr T kExpectedVal = static_cast<T>(static_cast<TU>(kLane) + 1u);
    const auto expected = Set(d, kExpectedVal);

    const BlockDFromD<decltype(d)> d_block;
    static_assert(d_block.MaxLanes() <= d.MaxLanes(),
                  "d_block.MaxLanes() <= d.MaxLanes() must be true");
    constexpr size_t kLanesPer16ByteBlk = 16 / sizeof(T);
    constexpr int kBlockIdx = kLane / static_cast<int>(kLanesPer16ByteBlk);
    constexpr int kLaneInBlkIdx =
        kLane & static_cast<int>(kLanesPer16ByteBlk - 1);

    const Vec<D> v = Iota(d, T{1});
    const Vec<D> actual = BroadcastLane<kLane>(v);
    const Vec<decltype(d_block)> actual_block =
        ExtractBlock<kBlockIdx>(Broadcast<kLaneInBlkIdx>(v));

    HWY_ASSERT_VEC_EQ(d, expected, actual);
    HWY_ASSERT_VEC_EQ(d_block, ResizeBitCast(d_block, expected), actual_block);
  }
  template <int kLane, class D,
            HWY_IF_LANES_LE_D(D, static_cast<size_t>(kLane))>
  static HWY_INLINE void DoTestBroadcastLane(D /*d*/, const size_t /*N*/) {
    // If kLane >= HWY_MAX_LANES_D(D) is true, do nothing
  }

 public:
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto N = Lanes(d);

    DoTestBroadcastLane<0>(d, N);
    DoTestBroadcastLane<1>(d, N);
    DoTestBroadcastLane<2>(d, N);
    DoTestBroadcastLane<3>(d, N);
    DoTestBroadcastLane<6>(d, N);
    DoTestBroadcastLane<14>(d, N);
    DoTestBroadcastLane<29>(d, N);
    DoTestBroadcastLane<53>(d, N);
    DoTestBroadcastLane<115>(d, N);
    DoTestBroadcastLane<251>(d, N);
    DoTestBroadcastLane<257>(d, N);
  }
};

HWY_NOINLINE void TestAllBroadcastLane() {
  ForAllTypes(ForPartialFixedOrFullScalableVectors<TestBroadcastLane>());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
HWY_BEFORE_TEST(HwySwizzleTest);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllGetLane);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllExtractLane);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllInsertLane);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllDupEven);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllDupOdd);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllOddEven);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllBroadcastLane);
}  // namespace hwy

#endif
