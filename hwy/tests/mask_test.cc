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
#include <string.h>  // memcmp

#include "hwy/base.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/mask_test.cc"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// All types.
struct TestFromVec {
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

HWY_NOINLINE void TestAllFromVec() {
  ForAllTypes(ForPartialVectors<TestFromVec>());
}

struct TestFirstN {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto mask_lanes = AllocateAligned<T>(N);

    // GCC workaround: we previously used zero to indicate true because we can
    // safely compare with that value. However, that hits an ICE for u64x1 on
    // GCC 8.3 but not 8.4, even if the implementation of operator== is
    // simplified to return zero. Using MaskFromVec avoids this, and requires
    // FF..FF and 0 constants.
    T on;
    memset(&on, 0xFF, sizeof(on));
    const T off = 0;

    for (size_t len = 0; len <= N; ++len) {
      for (size_t i = 0; i < N; ++i) {
        mask_lanes[i] = i < len ? on : off;
      }
      const auto mask_vals = Load(d, mask_lanes.get());
      const auto mask = MaskFromVec(mask_vals);
      HWY_ASSERT_MASK_EQ(d, mask, FirstN(d, len));
    }
  }
};

HWY_NOINLINE void TestAllFirstN() {
  ForAllTypes(ForPartialVectors<TestFirstN>());
}

struct TestIfThenElse {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    RandomState rng;

    const size_t N = Lanes(d);
    auto in1 = AllocateAligned<T>(N);
    auto in2 = AllocateAligned<T>(N);
    auto mask_lanes = AllocateAligned<T>(N);
    auto expected = AllocateAligned<T>(N);

    // NOTE: reverse polarity (mask is true iff lane == 0) because we cannot
    // reliably compare against all bits set (NaN for float types).
    const T off = 1;

    // Each lane should have a chance of having mask=true.
    for (size_t rep = 0; rep < 50; ++rep) {
      for (size_t i = 0; i < N; ++i) {
        in1[i] = static_cast<T>(Random32(&rng));
        in2[i] = static_cast<T>(Random32(&rng));
        mask_lanes[i] = (Random32(&rng) & 1024) ? off : T(0);
      }

      const auto v1 = Load(d, in1.get());
      const auto v2 = Load(d, in2.get());
      const auto mask = Eq(Load(d, mask_lanes.get()), Zero(d));

      for (size_t i = 0; i < N; ++i) {
        expected[i] = (mask_lanes[i] == off) ? in2[i] : in1[i];
      }
      HWY_ASSERT_VEC_EQ(d, expected.get(), IfThenElse(mask, v1, v2));

      for (size_t i = 0; i < N; ++i) {
        expected[i] = mask_lanes[i] ? T(0) : in1[i];
      }
      HWY_ASSERT_VEC_EQ(d, expected.get(), IfThenElseZero(mask, v1));

      for (size_t i = 0; i < N; ++i) {
        expected[i] = mask_lanes[i] ? in2[i] : T(0);
      }
      HWY_ASSERT_VEC_EQ(d, expected.get(), IfThenZeroElse(mask, v2));
    }
  }
};

HWY_NOINLINE void TestAllIfThenElse() {
  ForAllTypes(ForPartialVectors<TestIfThenElse>());
}

struct TestMaskVec {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    RandomState rng;

    const size_t N = Lanes(d);
    auto mask_lanes = AllocateAligned<T>(N);

    // Each lane should have a chance of having mask=true.
    for (size_t rep = 0; rep < 100; ++rep) {
      for (size_t i = 0; i < N; ++i) {
        mask_lanes[i] = static_cast<T>(Random32(&rng) & 1);
      }

      const auto mask = RebindMask(d, Eq(Load(d, mask_lanes.get()), Zero(d)));
      HWY_ASSERT_MASK_EQ(d, mask, MaskFromVec(VecFromMask(d, mask)));
    }
  }
};

HWY_NOINLINE void TestAllMaskVec() {
  const ForPartialVectors<TestMaskVec> test;

  test(uint16_t());
  test(int16_t());
  // TODO(janwas): float16_t - cannot compare yet

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

struct TestCompress {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    RandomState rng;

    using TU = MakeUnsigned<T>;
    const Rebind<TU, D> du;
    const size_t N = Lanes(d);
    auto in_lanes = AllocateAligned<T>(N);
    auto mask_lanes = AllocateAligned<TU>(N);
    auto expected = AllocateAligned<T>(N);
    auto actual = AllocateAligned<T>(N);

    // Each lane should have a chance of having mask=true.
    for (size_t rep = 0; rep < 100; ++rep) {
      size_t expected_pos = 0;
      for (size_t i = 0; i < N; ++i) {
        const uint64_t bits = Random32(&rng);
        in_lanes[i] = T();  // cannot initialize float16_t directly.
        CopyBytes<sizeof(T)>(&bits, &in_lanes[i]);
        mask_lanes[i] = static_cast<TU>(Random32(&rng) & 1);
        if (mask_lanes[i] == 0) {  // Zero means true (easier to compare)
          expected[expected_pos++] = in_lanes[i];
        }
      }

      const auto in = Load(d, in_lanes.get());
      const auto mask = RebindMask(d, Eq(Load(du, mask_lanes.get()), Zero(du)));

      Store(Compress(in, mask), d, actual.get());
      // Upper lanes are undefined.
      for (size_t i = 0; i < expected_pos; ++i) {
        HWY_ASSERT(memcmp(&actual[i], &expected[i], sizeof(T)) == 0);
      }

      // Also check CompressStore in the same way.
      memset(actual.get(), 0, N * sizeof(T));
      const size_t num_written = CompressStore(in, mask, d, actual.get());
      HWY_ASSERT_EQ(expected_pos, num_written);
      for (size_t i = 0; i < expected_pos; ++i) {
        HWY_ASSERT(memcmp(&actual[i], &expected[i], sizeof(T)) == 0);
      }
    }
  }
};

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
  test(float16_t());

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

struct TestAllTrueFalse {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto zero = Zero(d);
    auto v = zero;

    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);
    std::fill(lanes.get(), lanes.get() + N, T(0));

    auto mask_lanes = AllocateAligned<T>(N);

    HWY_ASSERT(AllTrue(d, Eq(v, zero)));
    HWY_ASSERT(!AllFalse(Eq(v, zero)));

    // Single lane implies AllFalse = !AllTrue. Otherwise, there are multiple
    // lanes and one is nonzero.
    const bool expected_all_false = (N != 1);

    // Set each lane to nonzero and back to zero
    for (size_t i = 0; i < N; ++i) {
      lanes[i] = T(1);
      v = Load(d, lanes.get());

      // GCC 10.2.1 workaround: AllTrue(Eq(v, zero)) is true but should not be.
      // Assigning to an lvalue is insufficient but storing to memory prevents
      // the bug; so does Print of VecFromMask(d, Eq(v, zero)).
      Store(VecFromMask(d, Eq(v, zero)), d, mask_lanes.get());
      HWY_ASSERT(!AllTrue(d, MaskFromVec(Load(d, mask_lanes.get()))));

      HWY_ASSERT(expected_all_false ^ AllFalse(Eq(v, zero)));

      lanes[i] = T(-1);
      v = Load(d, lanes.get());
      HWY_ASSERT(!AllTrue(d, Eq(v, zero)));
      HWY_ASSERT(expected_all_false ^ AllFalse(Eq(v, zero)));

      // Reset to all zero
      lanes[i] = T(0);
      v = Load(d, lanes.get());
      HWY_ASSERT(AllTrue(d, Eq(v, zero)));
      HWY_ASSERT(!AllFalse(Eq(v, zero)));
    }
  }
};

HWY_NOINLINE void TestAllAllTrueFalse() {
  ForAllTypes(ForPartialVectors<TestAllTrueFalse>());
}

class TestStoreMaskBits {
 public:
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*t*/, D d) {
    // TODO(janwas): remove once implemented (cast or vse1)
#if HWY_TARGET != HWY_RVV
    RandomState rng;
    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);
    const size_t expected_bytes = (N + 7) / 8;
    auto bits = AllocateAligned<uint8_t>(expected_bytes);

    for (size_t rep = 0; rep < 100; ++rep) {
      // Generate random mask pattern.
      for (size_t i = 0; i < N; ++i) {
        lanes[i] = static_cast<T>((rng() & 1024) ? 1 : 0);
      }
      const auto mask = Load(d, lanes.get()) == Zero(d);

      const size_t bytes_written = StoreMaskBits(d, mask, bits.get());

      HWY_ASSERT_EQ(expected_bytes, bytes_written);
      size_t i = 0;
      // Stored bits must match original mask
      for (; i < N; ++i) {
        const bool bit = (bits[i / 8] & (1 << (i % 8))) != 0;
        HWY_ASSERT_EQ(bit, lanes[i] == 0);
      }
      // Any partial bits in the last byte must be zero
      for (; i < 8 * bytes_written; ++i) {
        const int bit = (bits[i / 8] & (1 << (i % 8)));
        HWY_ASSERT_EQ(bit, 0);
      }
    }
#else
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllStoreMaskBits() {
  ForAllTypes(ForPartialVectors<TestStoreMaskBits>());
}

struct TestCountTrue {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    // For all combinations of zero/nonzero state of subset of lanes:
    const size_t max_lanes = HWY_MIN(N, size_t(10));

    auto lanes = AllocateAligned<T>(N);
    std::fill(lanes.get(), lanes.get() + N, T(1));

    for (size_t code = 0; code < (1ull << max_lanes); ++code) {
      // Number of zeros written = number of mask lanes that are true.
      size_t expected = 0;
      for (size_t i = 0; i < max_lanes; ++i) {
        lanes[i] = T(1);
        if (code & (1ull << i)) {
          ++expected;
          lanes[i] = T(0);
        }
      }

      const auto mask = Eq(Load(d, lanes.get()), Zero(d));
      const size_t actual = CountTrue(d, mask);
      HWY_ASSERT_EQ(expected, actual);
    }
  }
};

HWY_NOINLINE void TestAllCountTrue() {
  ForAllTypes(ForPartialVectors<TestCountTrue>());
}

struct TestLogicalMask {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto m0 = MaskFalse(d);
    const auto m_all = MaskTrue(d);

    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);
    std::fill(lanes.get(), lanes.get() + N, T(1));

    HWY_ASSERT_MASK_EQ(d, m0, Not(d, m_all));
    HWY_ASSERT_MASK_EQ(d, m_all, Not(d, m0));

    // For all combinations of zero/nonzero state of subset of lanes:
    const size_t max_lanes = HWY_MIN(N, size_t(6));
    for (size_t code = 0; code < (1ull << max_lanes); ++code) {
      for (size_t i = 0; i < max_lanes; ++i) {
        lanes[i] = T(1);
        if (code & (1ull << i)) {
          lanes[i] = T(0);
        }
      }

      const auto m = Eq(Load(d, lanes.get()), Zero(d));

      HWY_ASSERT_MASK_EQ(d, m0, Xor(m, m));
      HWY_ASSERT_MASK_EQ(d, m0, AndNot(m, m));
      HWY_ASSERT_MASK_EQ(d, m0, AndNot(m_all, m));

      HWY_ASSERT_MASK_EQ(d, m, Or(m, m));
      HWY_ASSERT_MASK_EQ(d, m, Or(m0, m));
      HWY_ASSERT_MASK_EQ(d, m, Or(m, m0));
      HWY_ASSERT_MASK_EQ(d, m, Xor(m0, m));
      HWY_ASSERT_MASK_EQ(d, m, Xor(m, m0));
      HWY_ASSERT_MASK_EQ(d, m, And(m, m));
      HWY_ASSERT_MASK_EQ(d, m, And(m_all, m));
      HWY_ASSERT_MASK_EQ(d, m, And(m, m_all));
      HWY_ASSERT_MASK_EQ(d, m, AndNot(m0, m));
    }
  }
};

HWY_NOINLINE void TestAllLogicalMask() {
  ForAllTypes(ForPartialVectors<TestLogicalMask>());
}
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(HwyLogicalTest);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllFromVec);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllFirstN);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllIfThenElse);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllMaskVec);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllCompress);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllAllTrueFalse);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllStoreMaskBits);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllCountTrue);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllLogicalMask);
}  // namespace hwy
#endif
