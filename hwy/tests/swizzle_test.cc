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
  ForAllTypes(ForShrinkableVectors<TestOddEven>());
}

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

struct TestCompress {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    RandomState rng;

    using TI = MakeSigned<T>;  // For mask > 0 comparison
    const Rebind<TI, D> di;
    const size_t N = Lanes(d);
    auto in_lanes = AllocateAligned<T>(N);
    auto mask_lanes = AllocateAligned<TI>(N);
    auto expected = AllocateAligned<T>(N);
    auto actual = AllocateAligned<T>(N);

    // Each lane should have a chance of having mask=true.
    for (size_t rep = 0; rep < 100; ++rep) {
      size_t expected_pos = 0;
      for (size_t i = 0; i < N; ++i) {
        const uint64_t bits = Random32(&rng);
        in_lanes[i] = T();  // cannot initialize float16_t directly.
        CopyBytes<sizeof(T)>(&bits, &in_lanes[i]);
        mask_lanes[i] = (Random32(&rng) & 1024) ? TI(1) : TI(0);
        if (mask_lanes[i] > 0) {
          expected[expected_pos++] = in_lanes[i];
        }
      }

      const auto in = Load(d, in_lanes.get());
      const auto mask = RebindMask(d, Gt(Load(di, mask_lanes.get()), Zero(di)));

      Store(Compress(in, mask), d, actual.get());
      // Upper lanes are undefined. Modified from AssertVecEqual.
      for (size_t i = 0; i < expected_pos; ++i) {
        if (!IsEqual(expected[i], actual[i])) {
          fprintf(stderr, "Mismatch at i=%zu of %zu:\n\n", i, expected_pos);
          Print(di, "mask", Load(di, mask_lanes.get()), 0, N);
          Print(d, "in", in, 0, N);
          Print(d, "expect", Load(d, expected.get()), 0, N);
          Print(d, "actual", Load(d, actual.get()), 0, N);
          HWY_ASSERT(false);
        }
      }

      // Also check CompressStore in the same way.
      memset(actual.get(), 0, N * sizeof(T));
      const size_t num_written = CompressStore(in, mask, d, actual.get());
      HWY_ASSERT_EQ(expected_pos, num_written);
      for (size_t i = 0; i < expected_pos; ++i) {
        if (!IsEqual(expected[i], actual[i])) {
          fprintf(stderr, "Mismatch at i=%zu of %zu:\n\n", i, expected_pos);
          Print(di, "mask", Load(di, mask_lanes.get()), 0, N);
          Print(d, "in", in, 0, N);
          Print(d, "expect", Load(d, expected.get()), 0, N);
          Print(d, "actual", Load(d, actual.get()), 0, N);
          HWY_ASSERT(false);
        }
      }
    }
  }
};

// For regenerating tables used in the implementation
#if 0
namespace detail {  // for code folding
void PrintCompress16x8Tables() {
  constexpr size_t N = 8;  // 128-bit SIMD
  for (uint64_t code = 0; code < 1ull << N; ++code) {
    std::array<uint8_t, N> indices{0};
    size_t pos = 0;
    for (size_t i = 0; i < N; ++i) {
      if (code & (1ull << i)) {
        indices[pos++] = i;
      }
    }

    // Doubled (for converting lane to byte indices)
    for (size_t i = 0; i < N; ++i) {
      printf("%d,", 2 * indices[i]);
    }
  }
  printf("\n");
}

// Compressed to nibbles
void PrintCompress32x8Tables() {
  constexpr size_t N = 8;  // AVX2
  for (uint64_t code = 0; code < 1ull << N; ++code) {
    std::array<uint32_t, N> indices{0};
    size_t pos = 0;
    for (size_t i = 0; i < N; ++i) {
      if (code & (1ull << i)) {
        indices[pos++] = i;
      }
    }

    // Convert to nibbles
    uint64_t packed = 0;
    for (size_t i = 0; i < N; ++i) {
      HWY_ASSERT(indices[i] < 16);
      packed += indices[i] << (i * 4);
    }

    HWY_ASSERT(packed < (1ull << 32));
    printf("0x%08x,", static_cast<uint32_t>(packed));
  }
  printf("\n");
}

// Pairs of 32-bit lane indices
void PrintCompress64x4Tables() {
  constexpr size_t N = 4;  // AVX2
  for (uint64_t code = 0; code < 1ull << N; ++code) {
    std::array<uint32_t, N> indices{0};
    size_t pos = 0;
    for (size_t i = 0; i < N; ++i) {
      if (code & (1ull << i)) {
        indices[pos++] = i;
      }
    }

    for (size_t i = 0; i < N; ++i) {
      printf("%d,%d,", 2 * indices[i], 2 * indices[i] + 1);
    }
  }
  printf("\n");
}

// 4-tuple of byte indices
void PrintCompress32x4Tables() {
  using T = uint32_t;
  constexpr size_t N = 4;  // SSE4
  for (uint64_t code = 0; code < 1ull << N; ++code) {
    std::array<uint32_t, N> indices{0};
    size_t pos = 0;
    for (size_t i = 0; i < N; ++i) {
      if (code & (1ull << i)) {
        indices[pos++] = i;
      }
    }

    for (size_t i = 0; i < N; ++i) {
      for (size_t idx_byte = 0; idx_byte < sizeof(T); ++idx_byte) {
        printf("%zu,", sizeof(T) * indices[i] + idx_byte);
      }
    }
  }
  printf("\n");
}

// 8-tuple of byte indices
void PrintCompress64x2Tables() {
  using T = uint64_t;
  constexpr size_t N = 2;  // SSE4
  for (uint64_t code = 0; code < 1ull << N; ++code) {
    std::array<uint32_t, N> indices{0};
    size_t pos = 0;
    for (size_t i = 0; i < N; ++i) {
      if (code & (1ull << i)) {
        indices[pos++] = i;
      }
    }

    for (size_t i = 0; i < N; ++i) {
      for (size_t idx_byte = 0; idx_byte < sizeof(T); ++idx_byte) {
        printf("%zu,", sizeof(T) * indices[i] + idx_byte);
      }
    }
  }
  printf("\n");
}
}  // namespace detail
#endif

HWY_NOINLINE void TestAllCompress() {
  // detail::PrintCompress32x8Tables();
  // detail::PrintCompress64x4Tables();
  // detail::PrintCompress32x4Tables();
  // detail::PrintCompress64x2Tables();
  // detail::PrintCompress16x8Tables();

  const ForPartialVectors<TestCompress> test;

  test(uint16_t());
  test(int16_t());
#if HWY_CAP_FLOAT16
  test(float16_t());
#endif

  test(uint32_t());
  test(int32_t());
  test(float());

#if HWY_CAP_INTEGER64
  test(uint64_t());
  test(int64_t());
#endif
#if HWY_CAP_FLOAT64
  test(double());
#endif
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
HWY_BEFORE_TEST(HwySwizzleTest);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllGetLane);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllOddEven);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllTableLookupLanes);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllCompress);
}  // namespace hwy

// Ought not to be necessary, but without this, no tests run on RVV.
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif
