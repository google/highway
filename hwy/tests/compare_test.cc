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
#if HWY_HAS_DOUBLE
constexpr HWY_FULL(double) dd;
#endif

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

// Returns "bits" after zeroing any upper bits that wouldn't be returned by
// movemask for the given vector "D".
template <class D>
uint64_t ValidBits(D d, const uint64_t bits) {
  constexpr size_t shift = 64 - d.N;  // 0..63 - avoids UB for d.N == 64.
  static_assert(shift < 64, "d.N out of range");  // Silences clang-tidy.
  // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
  return (bits << shift) >> shift;
}

HWY_NOINLINE HWY_ATTR void TestMovemask() {
  HWY_ALIGN constexpr uint8_t kBytes[kTestMaxVectorSize] = {
      0x80, 0xFF, 0x7F, 0x00, 0x01, 0x10, 0x20, 0x40, 0x80, 0x02, 0x04,
      0x08, 0xC0, 0xC1, 0xFE, 0x0F, 0x0F, 0xFE, 0xC1, 0xC0, 0x08, 0x04,
      0x02, 0x80, 0x40, 0x20, 0x10, 0x01, 0x00, 0x7F, 0xFF, 0x80, 0x0F,
      0xFE, 0xC1, 0xC0, 0x08, 0x04, 0x02, 0x80, 0x40, 0x20, 0x10, 0x01,
      0x00, 0x7F, 0xFF, 0x80, 0x80, 0xFF, 0x7F, 0x00, 0x01, 0x10, 0x20,
      0x40, 0x80, 0x02, 0x04, 0x08, 0xC0, 0xC1, 0xFE, 0x0F};
  const auto bytes = Load(du8, kBytes);
  HWY_ASSERT_EQ(ValidBits(du8, 0x7103C08EC08E7103ull), ext::movemask(bytes));

  HWY_ALIGN const float kLanesF[kTestMaxVectorSize / sizeof(float)] = {
      -1.0f,  1E30f, -0.0f, 1E-30f, 0.0f,  -0.0f, 1E30f, -1.0f,
      1E-30f, -0.0f, 1E30f, -1.0f,  -1.0f, 1E30f, -0.0f, 1E-30f};
  const auto vf = Load(df, kLanesF);
  HWY_ASSERT_EQ(ValidBits(df, 0x5aa5), ext::movemask(vf));

#if HWY_HAS_DOUBLE
  HWY_ALIGN const double kLanesD[kTestMaxVectorSize / sizeof(double)] = {
      1E300, -1E-300, -0.0, 1E-10, -0.0, 0.0, 1E300, -1E-10};
  const auto vd = Load(dd, kLanesD);
  HWY_ASSERT_EQ(ValidBits(dd, 0x96), ext::movemask(vd));
#endif
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

HWY_NOINLINE HWY_ATTR void TestCompare() {
  HWY_FOREACH_UIF(TestEquality);

  HWY_FOREACH_F(TestInequality);
  TestInequality(di8);
  TestInequality(di16);
  TestInequality(di32);
#if HWY_HAS_CMP64
  TestInequality(di64);
#endif

  TestMovemask();

  HWY_FOREACH_U(TestAllTrueFalse);
  HWY_FOREACH_I(TestAllTrueFalse);
  // No float.
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy

// Instantiate for the current target.
void CompareTest::HWY_FUNC() { hwy::HWY_NAMESPACE::TestCompare(); }
