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
#define HWY_TARGET_INCLUDE "tests/logical_test.cc"

#include <random>

#include "hwy/tests/test_util.h"
struct LogicalTest {
  HWY_DECLARE(void, ())
};
TEST(HwyLogicalTest, Run) { hwy::RunTests<LogicalTest>(); }

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
HWY_NOINLINE HWY_ATTR void TestLogicalT(D d) {
  const auto v0 = Zero(d);
  const auto vi = Iota(d, 0);

  HWY_ASSERT_VEC_EQ(d, v0, v0 & vi);
  HWY_ASSERT_VEC_EQ(d, v0, vi & v0);
  HWY_ASSERT_VEC_EQ(d, vi, vi & vi);

  HWY_ASSERT_VEC_EQ(d, vi, v0 | vi);
  HWY_ASSERT_VEC_EQ(d, vi, vi | v0);
  HWY_ASSERT_VEC_EQ(d, vi, vi | vi);

  HWY_ASSERT_VEC_EQ(d, vi, v0 ^ vi);
  HWY_ASSERT_VEC_EQ(d, vi, vi ^ v0);
  HWY_ASSERT_VEC_EQ(d, v0, vi ^ vi);

  HWY_ASSERT_VEC_EQ(d, vi, AndNot(v0, vi));
  HWY_ASSERT_VEC_EQ(d, v0, AndNot(vi, v0));
  HWY_ASSERT_VEC_EQ(d, v0, AndNot(vi, vi));

  auto v = vi;
  v &= vi;
  HWY_ASSERT_VEC_EQ(d, vi, v);
  v &= v0;
  HWY_ASSERT_VEC_EQ(d, v0, v);

  v |= vi;
  HWY_ASSERT_VEC_EQ(d, vi, v);
  v |= v0;
  HWY_ASSERT_VEC_EQ(d, vi, v);

  v ^= vi;
  HWY_ASSERT_VEC_EQ(d, v0, v);
  v ^= v0;
  HWY_ASSERT_VEC_EQ(d, v0, v);
}

// Vec <-> Mask, IfThen*
template <class D>
HWY_NOINLINE HWY_ATTR void TestIfThenElse(D d) {
  using T = typename D::T;
  RandomState rng{1234};
  const T no(0);
  T yes;
  memset(&yes, 0xFF, sizeof(yes));

  HWY_ALIGN T in1[d.N] = {};         // Initialized for clang-analyzer.
  HWY_ALIGN T in2[d.N] = {};         // Initialized for clang-analyzer.
  HWY_ALIGN T mask_lanes[d.N] = {};  // Initialized for clang-analyzer.
  for (size_t i = 0; i < d.N; ++i) {
    in1[i] = int32_t(Random32(&rng));
    in2[i] = int32_t(Random32(&rng));
    mask_lanes[i] = (Random32(&rng) & 1024) ? no : yes;
  }

  const auto vec = Load(d, mask_lanes);
  const auto mask = MaskFromVec(vec);
  HWY_ASSERT_VEC_EQ(d, vec, VecFromMask(mask));

  HWY_ALIGN T out_lanes1[d.N];
  HWY_ALIGN T out_lanes2[d.N];
  HWY_ALIGN T out_lanes3[d.N];
  Store(IfThenElse(mask, Load(d, in1), Load(d, in2)), d, out_lanes1);
  Store(IfThenElseZero(mask, Load(d, in1)), d, out_lanes2);
  Store(IfThenZeroElse(mask, Load(d, in2)), d, out_lanes3);
  for (size_t i = 0; i < d.N; ++i) {
    // Cannot reliably compare against yes (NaN).
    HWY_ASSERT_EQ((mask_lanes[i] == no) ? in2[i] : in1[i], out_lanes1[i]);
    HWY_ASSERT_EQ((mask_lanes[i] == no) ? no : in1[i], out_lanes2[i]);
    HWY_ASSERT_EQ((mask_lanes[i] == no) ? in2[i] : no, out_lanes3[i]);
  }
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestTestBit(D d) {
  using T = typename D::T;
  const size_t kNumBits = sizeof(T) * 8;
  for (size_t i = 0; i < kNumBits; ++i) {
    const auto bit1 = Set(d, 1ull << i);
    const auto bit2 = Set(d, 1ull << ((i + 1) % kNumBits));
    const auto bit3 = Set(d, 1ull << ((i + 2) % kNumBits));
    const auto bits12 = bit1 | bit2;
    const auto bits23 = bit2 | bit3;
    HWY_ASSERT_EQ(true, ext::AllTrue(TestBit(bit1, bit1)));
    HWY_ASSERT_EQ(true, ext::AllTrue(TestBit(bits12, bit1)));
    HWY_ASSERT_EQ(true, ext::AllTrue(TestBit(bits12, bit2)));

    HWY_ASSERT_EQ(true, ext::AllFalse(TestBit(bits12, bit3)));
    HWY_ASSERT_EQ(true, ext::AllFalse(TestBit(bits23, bit1)));
    HWY_ASSERT_EQ(true, ext::AllFalse(TestBit(bit1, bit2)));
    HWY_ASSERT_EQ(true, ext::AllFalse(TestBit(bit2, bit1)));
    HWY_ASSERT_EQ(true, ext::AllFalse(TestBit(bit1, bit3)));
    HWY_ASSERT_EQ(true, ext::AllFalse(TestBit(bit3, bit1)));
    HWY_ASSERT_EQ(true, ext::AllFalse(TestBit(bit2, bit3)));
    HWY_ASSERT_EQ(true, ext::AllFalse(TestBit(bit3, bit2)));
  }
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestAllTrueFalse(D d) {
  using T = typename D::T;
  const auto zero = Zero(d);
  const T max = LimitsMax<T>();
  const T min_nonzero = LimitsMin<T>() + 1;

  auto v = zero;
  HWY_ALIGN T lanes[d.N] = {};  // Initialized for clang-analyzer.
  Store(v, d, lanes);
  HWY_ASSERT_EQ(true, ext::AllTrue(v == zero));
  HWY_ASSERT_EQ(false, ext::AllFalse(v == zero));

  // Set each lane to nonzero and back to zero
  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = max;
    v = Load(d, lanes);
    HWY_ASSERT_EQ(false, ext::AllTrue(v == zero));
#if HWY_BITS != 0
    HWY_ASSERT_EQ(false, ext::AllFalse(v == zero));
#endif

    lanes[i] = min_nonzero;
    v = Load(d, lanes);
    HWY_ASSERT_EQ(false, ext::AllTrue(v == zero));
#if HWY_BITS != 0
    HWY_ASSERT_EQ(false, ext::AllFalse(v == zero));
#endif

    // Reset to all zero
    lanes[i] = T(0);
    v = Load(d, lanes);
    HWY_ASSERT_EQ(true, ext::AllTrue(v == zero));
    HWY_ASSERT_EQ(false, ext::AllFalse(v == zero));
  }
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestBitsFromMask(D d, const uint64_t bits) {
  using T = typename D::T;

  // Generate a mask matching the given bits
  HWY_ALIGN T mask_lanes[d.N];
  memset(mask_lanes, 0xFF, sizeof(mask_lanes));
  for (size_t i = 0; i < d.N; ++i) {
    if ((bits & (1ull << i)) == 0) mask_lanes[i] = 0;
  }
  const auto mask = MaskFromVec(Load(d, mask_lanes));

  const uint64_t actual_bits = ext::BitsFromMask(mask);

  // Clear bits that cannot be returned for this D.
  constexpr size_t shift = 64 - d.N;  // 0..63 - avoids UB for d.N == 64.
  static_assert(shift < 64, "d.N out of range");  // Silences clang-tidy.
  // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
  const uint64_t expected_bits = (bits << shift) >> shift;

  HWY_ASSERT_EQ(expected_bits, actual_bits);
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestBitsFromMask(D d) {
  // Fixed patterns: all off/on/odd/even.
  TestBitsFromMask(d, 0);
  TestBitsFromMask(d, ~0ull);
  TestBitsFromMask(d, 0x5555555555555555ull);
  TestBitsFromMask(d, 0xAAAAAAAAAAAAAAAAull);

  // Random mask patterns.
  std::mt19937_64 rng;
  for (size_t i = 0; i < 100; ++i) {
    TestBitsFromMask(d, rng());
  }
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestCountTrue(D d) {
  using T = typename D::T;

  // For all combinations of zero/nonzero state of subset of lanes:
  const size_t max_lanes = std::min(d.N, size_t(10));

  HWY_ALIGN T lanes[d.N];
  std::fill(lanes, lanes + d.N, T(1));

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

    const auto mask = Load(d, lanes) == Zero(d);
    const size_t actual = ext::CountTrue(mask);
    HWY_ASSERT_VEC_EQ(d, Set(d, expected), Set(d, actual));
  }
}

HWY_NOINLINE HWY_ATTR void TestLogical() {
  (void)dd;
  (void)di64;
  (void)du64;

  HWY_FOREACH_UIF(TestLogicalT);
  HWY_FOREACH_UIF(TestIfThenElse);

  HWY_FOREACH_UI(TestTestBit);

  HWY_FOREACH_UIF(TestAllTrueFalse);
  HWY_FOREACH_UIF(TestBitsFromMask);
  HWY_FOREACH_UIF(TestCountTrue);
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy

// Instantiate for the current target.
void LogicalTest::HWY_FUNC() { hwy::HWY_NAMESPACE::TestLogical(); }
