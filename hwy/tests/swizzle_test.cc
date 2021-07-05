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
#include <string.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/swizzle_test.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

struct TestTableLookupLanes {
#if HWY_TARGET == HWY_RVV
  using Index = uint32_t;
#else
  using Index = int32_t;
#endif

  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
#if HWY_TARGET != HWY_SCALAR
    const size_t N = Lanes(d);
    auto idx = AllocateAligned<Index>(N);
    std::fill(idx.get(), idx.get() + N, Index(0));
    auto expected = AllocateAligned<T>(N);
    const auto v = Iota(d, 1);

    if (N <= 8) {  // Test all permutations
      for (size_t i0 = 0; i0 < N; ++i0) {
        idx[0] = static_cast<Index>(i0);

        for (size_t i1 = 0; i1 < N; ++i1) {
          if (N >= 2) idx[1] = static_cast<Index>(i1);
          for (size_t i2 = 0; i2 < N; ++i2) {
            if (N >= 4) idx[2] = static_cast<Index>(i2);
            for (size_t i3 = 0; i3 < N; ++i3) {
              if (N >= 4) idx[3] = static_cast<Index>(i3);

              for (size_t i = 0; i < N; ++i) {
                expected[i] = static_cast<T>(idx[i] + 1);  // == v[idx[i]]
              }

              const auto opaque = SetTableIndices(d, idx.get());
              const auto actual = TableLookupLanes(v, opaque);
              HWY_ASSERT_VEC_EQ(d, expected.get(), actual);
            }
          }
        }
      }
    } else {
      // Too many permutations to test exhaustively; choose one with repeated
      // and cross-block indices and ensure indices do not exceed #lanes.
      // For larger vectors, upper lanes will be zero.
      HWY_ALIGN Index idx_source[16] = {1,  3,  2,  2,  8, 1, 7, 6,
                                        15, 14, 14, 15, 4, 9, 8, 5};
      for (size_t i = 0; i < N; ++i) {
        idx[i] = (i < 16) ? idx_source[i] : 0;
        // Avoid undefined results / asan error for scalar by capping indices.
        if (idx[i] >= static_cast<Index>(N)) {
          idx[i] = static_cast<Index>(N - 1);
        }
        expected[i] = static_cast<T>(idx[i] + 1);  // == v[idx[i]]
      }

      const auto opaque = SetTableIndices(d, idx.get());
      const auto actual = TableLookupLanes(v, opaque);
      HWY_ASSERT_VEC_EQ(d, expected.get(), actual);
    }
#else
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllTableLookupLanes() {
  const ForPartialVectors<TestTableLookupLanes> test;
  test(uint32_t());
  test(int32_t());
  test(float());
}

struct TestConcatHalves {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // TODO(janwas): fix
#if HWY_TARGET != HWY_RVV
    // Construct inputs such that interleaved halves == iota.
    const auto expected = Iota(d, 1);

    const size_t N = Lanes(d);
    auto lo = AllocateAligned<T>(N);
    auto hi = AllocateAligned<T>(N);
    size_t i;
    for (i = 0; i < N / 2; ++i) {
      lo[i] = static_cast<T>(1 + i);
      hi[i] = static_cast<T>(lo[i] + T(N) / 2);
    }
    for (; i < N; ++i) {
      lo[i] = hi[i] = 0;
    }

    HWY_ASSERT_VEC_EQ(d, expected,
                      ConcatLowerLower(Load(d, hi.get()), Load(d, lo.get())));

    // Same for high blocks.
    for (i = 0; i < N / 2; ++i) {
      lo[i] = hi[i] = 0;
    }
    for (; i < N; ++i) {
      lo[i] = static_cast<T>(1 + i - N / 2);
      hi[i] = static_cast<T>(lo[i] + T(N) / 2);
    }

    HWY_ASSERT_VEC_EQ(d, expected,
                      ConcatUpperUpper(Load(d, hi.get()), Load(d, lo.get())));
#else
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllConcatHalves() {
  ForAllTypes(ForGE128Vectors<TestConcatHalves>());
}

struct TestConcatLowerUpper {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // TODO(janwas): fix
#if HWY_TARGET != HWY_RVV
    const size_t N = Lanes(d);
    // Middle part of Iota(1) == Iota(1 + N / 2).
    const auto lo = Iota(d, 1);
    const auto hi = Iota(d, 1 + N);
    HWY_ASSERT_VEC_EQ(d, Iota(d, 1 + N / 2), ConcatLowerUpper(hi, lo));
#else
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllConcatLowerUpper() {
  ForAllTypes(ForGE128Vectors<TestConcatLowerUpper>());
}

struct TestConcatUpperLower {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    const auto lo = Iota(d, 1);
    const auto hi = Iota(d, 1 + N);
    auto expected = AllocateAligned<T>(N);
    size_t i = 0;
    for (; i < N / 2; ++i) {
      expected[i] = static_cast<T>(1 + i);
    }
    for (; i < N; ++i) {
      expected[i] = static_cast<T>(1 + i + N);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), ConcatUpperLower(hi, lo));
  }
};

HWY_NOINLINE void TestAllConcatUpperLower() {
  ForAllTypes(ForGE128Vectors<TestConcatUpperLower>());
}

struct TestOddEven {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    const auto even = Iota(d, 1);
    const auto odd = Iota(d, 1 + N);
    auto expected = AllocateAligned<T>(N);
    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>(1 + i + ((i & 1) ? N : 0));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), OddEven(odd, even));
  }
};

HWY_NOINLINE void TestAllOddEven() {
  ForAllTypes(ForGE128Vectors<TestOddEven>());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(HwySwizzleTest);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllTableLookupLanes);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllConcatHalves);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllConcatLowerUpper);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllConcatUpperLower);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllOddEven);
}  // namespace hwy
#endif
