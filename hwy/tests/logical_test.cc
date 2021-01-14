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

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/logical_test.cc"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

struct TestLogicalInteger {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    const auto vi = Iota(d, 0);

    HWY_ASSERT_VEC_EQ(d, v0, And(v0, vi));
    HWY_ASSERT_VEC_EQ(d, v0, And(vi, v0));
    HWY_ASSERT_VEC_EQ(d, vi, And(vi, vi));

    HWY_ASSERT_VEC_EQ(d, vi, Or(v0, vi));
    HWY_ASSERT_VEC_EQ(d, vi, Or(vi, v0));
    HWY_ASSERT_VEC_EQ(d, vi, Or(vi, vi));

    HWY_ASSERT_VEC_EQ(d, vi, Xor(v0, vi));
    HWY_ASSERT_VEC_EQ(d, vi, Xor(vi, v0));
    HWY_ASSERT_VEC_EQ(d, v0, Xor(vi, vi));

    HWY_ASSERT_VEC_EQ(d, vi, AndNot(v0, vi));
    HWY_ASSERT_VEC_EQ(d, v0, AndNot(vi, v0));
    HWY_ASSERT_VEC_EQ(d, v0, AndNot(vi, vi));

    auto v = vi;
    v = And(v, vi);
    HWY_ASSERT_VEC_EQ(d, vi, v);
    v = And(v, v0);
    HWY_ASSERT_VEC_EQ(d, v0, v);

    v = Or(v, vi);
    HWY_ASSERT_VEC_EQ(d, vi, v);
    v = Or(v, v0);
    HWY_ASSERT_VEC_EQ(d, vi, v);

    v = Xor(v, vi);
    HWY_ASSERT_VEC_EQ(d, v0, v);
    v = Xor(v, v0);
    HWY_ASSERT_VEC_EQ(d, v0, v);
  }
};

HWY_NOINLINE void TestAllLogicalInteger() {
  ForIntegerTypes(ForPartialVectors<TestLogicalInteger>());
}

struct TestLogicalFloat {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    const auto vi = Iota(d, 0);

    HWY_ASSERT_VEC_EQ(d, v0, And(v0, vi));
    HWY_ASSERT_VEC_EQ(d, v0, And(vi, v0));
    HWY_ASSERT_VEC_EQ(d, vi, And(vi, vi));

    HWY_ASSERT_VEC_EQ(d, vi, Or(v0, vi));
    HWY_ASSERT_VEC_EQ(d, vi, Or(vi, v0));
    HWY_ASSERT_VEC_EQ(d, vi, Or(vi, vi));

    HWY_ASSERT_VEC_EQ(d, vi, Xor(v0, vi));
    HWY_ASSERT_VEC_EQ(d, vi, Xor(vi, v0));
    HWY_ASSERT_VEC_EQ(d, v0, Xor(vi, vi));

    HWY_ASSERT_VEC_EQ(d, vi, AndNot(v0, vi));
    HWY_ASSERT_VEC_EQ(d, v0, AndNot(vi, v0));
    HWY_ASSERT_VEC_EQ(d, v0, AndNot(vi, vi));

    auto v = vi;
    v = And(v, vi);
    HWY_ASSERT_VEC_EQ(d, vi, v);
    v = And(v, v0);
    HWY_ASSERT_VEC_EQ(d, v0, v);

    v = Or(v, vi);
    HWY_ASSERT_VEC_EQ(d, vi, v);
    v = Or(v, v0);
    HWY_ASSERT_VEC_EQ(d, vi, v);

    v = Xor(v, vi);
    HWY_ASSERT_VEC_EQ(d, v0, v);
    v = Xor(v, v0);
    HWY_ASSERT_VEC_EQ(d, v0, v);
  }
};

HWY_NOINLINE void TestAllLogicalFloat() {
  ForFloatTypes(ForPartialVectors<TestLogicalFloat>());
}

struct TestCopySign {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    const auto vp = Iota(d, 1);
    auto vn = Iota(d, -128);
    // Ensure all lanes are negative even if there are many lanes.
    vn = IfThenElse(Lt(vn, v0), vn, Neg(vn));

    // Zero remains zero regardless of sign
    HWY_ASSERT_VEC_EQ(d, v0, CopySign(v0, v0));
    HWY_ASSERT_VEC_EQ(d, v0, CopySign(v0, vp));
    HWY_ASSERT_VEC_EQ(d, v0, CopySign(v0, vn));
    HWY_ASSERT_VEC_EQ(d, v0, CopySignToAbs(v0, v0));
    HWY_ASSERT_VEC_EQ(d, v0, CopySignToAbs(v0, vp));
    HWY_ASSERT_VEC_EQ(d, v0, CopySignToAbs(v0, vn));

    // Positive input, positive sign => unchanged
    HWY_ASSERT_VEC_EQ(d, vp, CopySign(vp, vp));
    HWY_ASSERT_VEC_EQ(d, vp, CopySignToAbs(vp, vp));

    // Positive input, negative sign => negated
    HWY_ASSERT_VEC_EQ(d, Neg(vp), CopySign(vp, vn));
    HWY_ASSERT_VEC_EQ(d, Neg(vp), CopySignToAbs(vp, vn));

    // Negative input, negative sign => unchanged
    HWY_ASSERT_VEC_EQ(d, vn, CopySign(vn, vn));

    // Negative input, positive sign => negated
    HWY_ASSERT_VEC_EQ(d, Neg(vn), CopySign(vn, vp));
  }
};

HWY_NOINLINE void TestAllCopySign() {
  ForFloatTypes(ForPartialVectors<TestCopySign>());
}

// Tests MaskFromVec, VecFromMask, IfThen*
struct TestIfThenElse {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    RandomState rng{1234};
    const T no(0);
    T yes;
    memset(&yes, 0xFF, sizeof(yes));

    const size_t N = Lanes(d);
    auto in1 = AllocateAligned<T>(N);
    auto in2 = AllocateAligned<T>(N);
    auto mask_lanes = AllocateAligned<T>(N);
    for (size_t i = 0; i < N; ++i) {
      in1[i] = static_cast<T>(Random32(&rng));
      in2[i] = static_cast<T>(Random32(&rng));
      mask_lanes[i] = (Random32(&rng) & 1024) ? no : yes;
    }

    const auto vec = Load(d, mask_lanes.get());
    const auto mask = MaskFromVec(vec);
    // Separate lvalue works around clang-7 asan bug (unaligned spill).
    const auto vec2 = VecFromMask(d, mask);
    HWY_ASSERT_VEC_EQ(d, vec, vec2);

    auto out_lanes1 = AllocateAligned<T>(N);
    auto out_lanes2 = AllocateAligned<T>(N);
    auto out_lanes3 = AllocateAligned<T>(N);
    const auto v1 = Load(d, in1.get());
    const auto v2 = Load(d, in2.get());
    Store(IfThenElse(mask, v1, v2), d, out_lanes1.get());
    Store(IfThenElseZero(mask, v1), d, out_lanes2.get());
    Store(IfThenZeroElse(mask, v2), d, out_lanes3.get());
    for (size_t i = 0; i < N; ++i) {
      // Cannot reliably compare against yes (NaN).
      HWY_ASSERT_EQ((mask_lanes[i] == no) ? in2[i] : in1[i], out_lanes1[i]);
      HWY_ASSERT_EQ((mask_lanes[i] == no) ? no : in1[i], out_lanes2[i]);
      HWY_ASSERT_EQ((mask_lanes[i] == no) ? in2[i] : no, out_lanes3[i]);
    }
  }
};

HWY_NOINLINE void TestAllIfThenElse() {
  ForAllTypes(ForPartialVectors<TestIfThenElse>());
}

struct TestCompress {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    RandomState rng{1234};
    const T no(0);
    T yes;
    memset(&yes, 0xFF, sizeof(yes));

    const size_t N = Lanes(d);
    auto in_lanes = AllocateAligned<T>(N);
    auto mask_lanes = AllocateAligned<T>(N);
    auto expected = AllocateAligned<T>(N);
    auto actual = AllocateAligned<T>(N);

    // Exhaustive test of all mask combinations unless N is large.
    const uint64_t end = 1ull << std::min<size_t>(N, 16);
    for (uint64_t code = 0; code < end; ++code) {
      size_t expected_pos = 0;
      for (size_t i = 0; i < N; ++i) {
        in_lanes[i] = static_cast<T>(Random32(&rng));
        const bool on = (code & (1ull << i)) != 0;
        mask_lanes[i] = on ? yes : no;
        if (on) {
          expected[expected_pos++] = in_lanes[i];
        }
      }

      const auto in = Load(d, in_lanes.get());
      const auto mask = MaskFromVec(Load(d, mask_lanes.get()));
      Store(Compress(in, mask), d, actual.get());
      // Upper lanes are undefined.
      for (size_t i = 0; i < expected_pos; ++i) {
        HWY_ASSERT(actual[i] == expected[i]);
      }

      // Also check CompressStore in the same way.
      std::fill(actual.get(), actual.get() + N, T(0));
      const size_t num_written = CompressStore(in, mask, d, actual.get());
      HWY_ASSERT_EQ(expected_pos, num_written);
      for (size_t i = 0; i < expected_pos; ++i) {
        HWY_ASSERT_EQ(expected[i], actual[i]);
      }
    }
  }
};

#if 0
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

#endif

HWY_NOINLINE void TestAllCompress() {
  // PrintCompress32x8Tables();
  // PrintCompress64x4Tables();
  // PrintCompress32x4Tables();
  // PrintCompress64x2Tables();

  ForPartialVectors<TestCompress>()(uint32_t());
  ForPartialVectors<TestCompress>()(int32_t());
  ForPartialVectors<TestCompress>()(float());

#if HWY_CAP_INTEGER64
  ForPartialVectors<TestCompress>()(uint64_t());
  ForPartialVectors<TestCompress>()(int64_t());
#endif

#if HWY_CAP_FLOAT64
  ForPartialVectors<TestCompress>()(double());
#endif
}

struct TestZeroIfNegative {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    const auto vp = Iota(d, 1);
    auto vn = Iota(d, -128);
    // Ensure all lanes are negative even if there are many lanes.
    vn = IfThenElse(Lt(vn, v0), vn, Neg(vn));

    // Zero and positive remain unchanged
    HWY_ASSERT_VEC_EQ(d, v0, ZeroIfNegative(v0));
    HWY_ASSERT_VEC_EQ(d, vp, ZeroIfNegative(vp));

    // Negative are all replaced with zero
    HWY_ASSERT_VEC_EQ(d, v0, ZeroIfNegative(vn));
  }
};

HWY_NOINLINE void TestAllZeroIfNegative() {
  ForFloatTypes(ForPartialVectors<TestZeroIfNegative>());
}

struct TestTestBit {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t kNumBits = sizeof(T) * 8;
    for (size_t i = 0; i < kNumBits; ++i) {
      const auto bit1 = Set(d, 1ull << i);
      const auto bit2 = Set(d, 1ull << ((i + 1) % kNumBits));
      const auto bit3 = Set(d, 1ull << ((i + 2) % kNumBits));
      const auto bits12 = Or(bit1, bit2);
      const auto bits23 = Or(bit2, bit3);
      HWY_ASSERT(AllTrue(TestBit(bit1, bit1)));
      HWY_ASSERT(AllTrue(TestBit(bits12, bit1)));
      HWY_ASSERT(AllTrue(TestBit(bits12, bit2)));

      HWY_ASSERT(AllFalse(TestBit(bits12, bit3)));
      HWY_ASSERT(AllFalse(TestBit(bits23, bit1)));
      HWY_ASSERT(AllFalse(TestBit(bit1, bit2)));
      HWY_ASSERT(AllFalse(TestBit(bit2, bit1)));
      HWY_ASSERT(AllFalse(TestBit(bit1, bit3)));
      HWY_ASSERT(AllFalse(TestBit(bit3, bit1)));
      HWY_ASSERT(AllFalse(TestBit(bit2, bit3)));
      HWY_ASSERT(AllFalse(TestBit(bit3, bit2)));
    }
  }
};

HWY_NOINLINE void TestAllTestBit() {
  ForIntegerTypes(ForFullVectors<TestTestBit>());
}

struct TestAllTrueFalse {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto zero = Zero(d);
    const T max = LimitsMax<T>();
    const T min_nonzero = LimitsMin<T>() + 1;

    auto v = zero;

    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);
    std::fill(lanes.get(), lanes.get() + N, T(0));  // for clang-analyzer.
    Store(v, d, lanes.get());
    HWY_ASSERT(AllTrue(Eq(v, zero)));
    HWY_ASSERT(!AllFalse(Eq(v, zero)));

#if HWY_TARGET == HWY_SCALAR
    // Simple negation of the AllTrue result
    const bool expected_all_false = false;
#else
    // There are multiple lanes and one is nonzero
    const bool expected_all_false = true;
#endif

    // Set each lane to nonzero and back to zero
    for (size_t i = 0; i < N; ++i) {
      lanes[i] = max;
      v = Load(d, lanes.get());
      HWY_ASSERT(!AllTrue(Eq(v, zero)));
      HWY_ASSERT(expected_all_false ^ AllFalse(Eq(v, zero)));

      lanes[i] = min_nonzero;
      v = Load(d, lanes.get());
      HWY_ASSERT(!AllTrue(Eq(v, zero)));
      HWY_ASSERT(expected_all_false ^ AllFalse(Eq(v, zero)));

      // Reset to all zero
      lanes[i] = T(0);
      v = Load(d, lanes.get());
      HWY_ASSERT(AllTrue(Eq(v, zero)));
      HWY_ASSERT(!AllFalse(Eq(v, zero)));
    }
  }
};

HWY_NOINLINE void TestAllAllTrueFalse() {
  ForAllTypes(ForFullVectors<TestAllTrueFalse>());
}

class TestBitsFromMask {
 public:
  template <class T, class D>
  HWY_NOINLINE void operator()(T t, D d) {
    // Fixed patterns: all off/on/odd/even.
    Bits(t, d, 0);
    Bits(t, d, ~0ull);
    Bits(t, d, 0x5555555555555555ull);
    Bits(t, d, 0xAAAAAAAAAAAAAAAAull);

    // Random mask patterns.
    std::mt19937_64 rng;
    for (size_t i = 0; i < 100; ++i) {
      Bits(t, d, rng());
    }
  }

 private:
  template <typename T, class D>
  HWY_NOINLINE void Bits(T /*unused*/, D d, uint64_t bits) {
    // Generate a mask matching the given bits
    const size_t N = Lanes(d);
    auto mask_lanes = AllocateAligned<T>(N);
    memset(mask_lanes.get(), 0xFF, N * sizeof(T));
    for (size_t i = 0; i < N; ++i) {
      if ((bits & (1ull << i)) == 0) mask_lanes[i] = 0;
    }
    const auto mask = MaskFromVec(Load(d, mask_lanes.get()));

    const uint64_t actual_bits = BitsFromMask(mask);

    // Clear bits that cannot be returned for this D.
    const size_t shift = 64 - N;  // 0..63 - avoids UB for N == 64.
    HWY_ASSERT(shift < 64);
    const uint64_t expected_bits = (bits << shift) >> shift;

    HWY_ASSERT_EQ(expected_bits, actual_bits);
  }
};

HWY_NOINLINE void TestAllBitsFromMask() {
  ForAllTypes(ForPartialVectors<TestBitsFromMask>());
}

struct TestCountTrue {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    // For all combinations of zero/nonzero state of subset of lanes:
    const size_t max_lanes = std::min(N, size_t(10));

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
      const size_t actual = CountTrue(mask);
      HWY_ASSERT_EQ(expected, actual);
    }
  }
};

HWY_NOINLINE void TestAllCountTrue() {
  ForAllTypes(ForFullVectors<TestCountTrue>());
}

template <class D, class M>
void AssertMaskEq(D d, M a, M b) {
  HWY_ASSERT_EQ(CountTrue(a), CountTrue(b));
  HWY_ASSERT_VEC_EQ(d, VecFromMask(d, a), VecFromMask(d, b));
}

struct TestMaskLogical {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    // For all combinations of zero/nonzero state of subset of lanes:
    const size_t max_lanes = std::min(N, size_t(10));

    auto lanes = AllocateAligned<T>(N);
    std::fill(lanes.get(), lanes.get() + N, T(1));

    const auto v0 = Zero(d);
    const auto m0 = MaskFromVec(v0);
    T all;
    memset(&all, 0xFF, sizeof(T));
    const auto m_all = MaskFromVec(Set(d, all));

    for (size_t code = 0; code < (1ull << max_lanes); ++code) {
      for (size_t i = 0; i < max_lanes; ++i) {
        lanes[i] = T(1);
        if (code & (1ull << i)) {
          lanes[i] = T(0);
        }
      }

      const auto m = Eq(Load(d, lanes.get()), v0);

      AssertMaskEq(d, m0, Xor(m, m));
      AssertMaskEq(d, m0, AndNot(m, m));
      AssertMaskEq(d, m0, AndNot(m_all, m));

      AssertMaskEq(d, m, Or(m, m));
      AssertMaskEq(d, m, Or(m0, m));
      AssertMaskEq(d, m, Or(m, m0));
      AssertMaskEq(d, m, Xor(m0, m));
      AssertMaskEq(d, m, Xor(m, m0));
      AssertMaskEq(d, m, And(m, m));
      AssertMaskEq(d, m, And(m_all, m));
      AssertMaskEq(d, m, And(m, m_all));
      AssertMaskEq(d, m, AndNot(m0, m));
    }
  }
};

HWY_NOINLINE void TestAllMaskLogical() {
  ForAllTypes(ForFullVectors<TestMaskLogical>());
}
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {

class HwyLogicalTest : public hwy::TestWithParamTarget {};

HWY_TARGET_INSTANTIATE_TEST_SUITE_P(HwyLogicalTest);

HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllLogicalInteger);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllLogicalFloat);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllCopySign);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllIfThenElse);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllCompress);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllZeroIfNegative);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllTestBit);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllAllTrueFalse);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllBitsFromMask);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllCountTrue);
HWY_EXPORT_AND_TEST_P(HwyLogicalTest, TestAllMaskLogical);

}  // namespace hwy
#endif
