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
#define HWY_TARGET_INCLUDE "tests/arithmetic_test.cc"

#include "hwy/tests/test_util.h"
struct ArithmeticTest {
  HWY_DECLARE(void, ())
};
TEST(HwyArithmeticTest, Run) { hwy::RunTests<ArithmeticTest>(); }

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
HWY_NOINLINE HWY_ATTR void TestPlusMinus(D d) {
  using T = typename D::T;
  const auto v2 = Iota(D(), 2);
  const auto v3 = Iota(D(), 3);
  const auto v4 = Iota(D(), 4);

  T lanes[d.N] = {};  // Initialized for static analysis.
  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = (2 + i) + (3 + i);
  }
  HWY_ASSERT_VEC_EQ(d, lanes, v2 + v3);
  HWY_ASSERT_VEC_EQ(d, v3, (v2 + v3) - v2);

  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = (2 + i) + (4 + i);
  }
  auto sum = v2;
  sum += v4;  // sum == 6,8..
  HWY_ASSERT_VEC_EQ(d, lanes, sum);

  sum -= v4;
  HWY_ASSERT_VEC_EQ(d, v2, sum);
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestUnsignedSaturatingArithmetic(D d) {
  using T = typename D::T;
  const auto v0 = Zero(d);
  const auto vi = Iota(d, 1);
  const auto vm = Set(d, LimitsMax<T>());

  HWY_ASSERT_VEC_EQ(d, v0 + v0, SaturatedAdd(v0, v0));
  HWY_ASSERT_VEC_EQ(d, v0 + vi, SaturatedAdd(v0, vi));
  HWY_ASSERT_VEC_EQ(d, v0 + vm, SaturatedAdd(v0, vm));
  HWY_ASSERT_VEC_EQ(d, vm, SaturatedAdd(vi, vm));
  HWY_ASSERT_VEC_EQ(d, vm, SaturatedAdd(vm, vm));

  HWY_ASSERT_VEC_EQ(d, v0, SaturatedSub(v0, v0));
  HWY_ASSERT_VEC_EQ(d, v0, SaturatedSub(v0, vi));
  HWY_ASSERT_VEC_EQ(d, v0, SaturatedSub(vi, vi));
  HWY_ASSERT_VEC_EQ(d, v0, SaturatedSub(vi, vm));
  HWY_ASSERT_VEC_EQ(d, vm - vi, SaturatedSub(vm, vi));
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestSignedSaturatingArithmetic(D d) {
  using T = typename D::T;
  const auto v0 = Zero(d);
  const auto vi = Iota(d, 1);
  const auto vpm = Set(d, LimitsMax<T>());
  const auto vn = Iota(d, -T(d.N));
  const auto vnm = Set(d, LimitsMin<T>());

  HWY_ASSERT_VEC_EQ(d, v0, SaturatedAdd(v0, v0));
  HWY_ASSERT_VEC_EQ(d, vi, SaturatedAdd(v0, vi));
  HWY_ASSERT_VEC_EQ(d, vpm, SaturatedAdd(v0, vpm));
  HWY_ASSERT_VEC_EQ(d, vpm, SaturatedAdd(vi, vpm));
  HWY_ASSERT_VEC_EQ(d, vpm, SaturatedAdd(vpm, vpm));

  HWY_ASSERT_VEC_EQ(d, v0, SaturatedSub(v0, v0));
  HWY_ASSERT_VEC_EQ(d, v0 - vi, SaturatedSub(v0, vi));
  HWY_ASSERT_VEC_EQ(d, vn, SaturatedSub(vn, v0));
  HWY_ASSERT_VEC_EQ(d, vnm, SaturatedSub(vnm, vi));
  HWY_ASSERT_VEC_EQ(d, vnm, SaturatedSub(vnm, vpm));
}

HWY_NOINLINE HWY_ATTR void TestSaturatingArithmetic() {
  TestUnsignedSaturatingArithmetic(du8);
  TestUnsignedSaturatingArithmetic(du16);
  TestSignedSaturatingArithmetic(di8);
  TestSignedSaturatingArithmetic(di16);
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestAverageT(D d) {
  using T = typename D::T;
  const auto v0 = Zero(d);
  const auto v1 = Set(d, T(1));
  const auto v2 = Set(d, T(2));

  HWY_ASSERT_VEC_EQ(d, v0, AverageRound(v0, v0));
  HWY_ASSERT_VEC_EQ(d, v1, AverageRound(v0, v1));
  HWY_ASSERT_VEC_EQ(d, v1, AverageRound(v1, v1));
  HWY_ASSERT_VEC_EQ(d, v2, AverageRound(v1, v2));
  HWY_ASSERT_VEC_EQ(d, v2, AverageRound(v2, v2));
}

HWY_NOINLINE HWY_ATTR void TestAverage() {
  TestAverageT(du8);
  TestAverageT(du16);
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestAbsT(D d) {
  using T = typename D::T;
  const auto v0 = Zero(d);
  const auto vp1 = Set(d, T(1));
  const auto vn1 = Set(d, T(-1));
  const auto vpm = Set(d, LimitsMax<T>());
  const auto vnm = Set(d, LimitsMin<T>());

  HWY_ASSERT_VEC_EQ(d, v0, Abs(v0));
  HWY_ASSERT_VEC_EQ(d, vp1, Abs(vp1));
  HWY_ASSERT_VEC_EQ(d, vp1, Abs(vn1));
  HWY_ASSERT_VEC_EQ(d, vpm, Abs(vpm));
  HWY_ASSERT_VEC_EQ(d, vnm, Abs(vnm));
}

HWY_NOINLINE HWY_ATTR void TestAbs() {
  TestAbsT(di8);
  TestAbsT(di16);
  TestAbsT(di32);
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestUnsignedShifts(D d) {
  using T = typename D::T;
  constexpr int kSign = (sizeof(T) * 8) - 1;
  const auto v0 = Zero(d);
  const auto vi = Iota(d, 0);
  HWY_ALIGN T expected[d.N] = {};  // Initialized for static analysis.

  // Shifting out of right side => zero
  HWY_ASSERT_VEC_EQ(d, v0, ShiftRight<7>(vi));
#if !HWY_HAS_VARIABLE_SHIFT || HWY_IDE
  HWY_ASSERT_VEC_EQ(d, v0, ShiftRightSame(vi, 7));
#endif

  // Simple left shift
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = T(i << 1);
  }
  HWY_ASSERT_VEC_EQ(d, expected, hwy::ShiftLeft<1>(vi));
#if !HWY_HAS_VARIABLE_SHIFT || HWY_IDE
  HWY_ASSERT_VEC_EQ(d, expected, ShiftLeftSame(vi, 1));
#endif

  // Simple right shift
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = T(i >> 1);
  }
  HWY_ASSERT_VEC_EQ(d, expected, ShiftRight<1>(vi));
#if !HWY_HAS_VARIABLE_SHIFT || HWY_IDE
  HWY_ASSERT_VEC_EQ(d, expected, ShiftRightSame(vi, 1));
#endif

  // Verify truncation for left-shift
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = (T(i) << kSign) & ~T(0);
  }
  HWY_ASSERT_VEC_EQ(d, expected, hwy::ShiftLeft<kSign>(vi));
#if !HWY_HAS_VARIABLE_SHIFT || HWY_IDE
  HWY_ASSERT_VEC_EQ(d, expected, ShiftLeftSame(vi, kSign));
#endif
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestSignedShifts(D d) {
  using T = typename D::T;
  const auto v0 = Zero(d);
  const auto vi = Iota(d, 0);
  HWY_ALIGN T expected[d.N] = {};  // Initialized for static analysis.

  // Shifting out of right side => zero
  HWY_ASSERT_VEC_EQ(d, v0, ShiftRight<7>(vi));
#if !HWY_HAS_VARIABLE_SHIFT || HWY_IDE
  HWY_ASSERT_VEC_EQ(d, v0, ShiftRightSame(vi, 7));
#endif

  // Simple left shift
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = T(i << 1);
  }
  HWY_ASSERT_VEC_EQ(d, expected, hwy::ShiftLeft<1>(vi));
#if !HWY_HAS_VARIABLE_SHIFT || HWY_IDE
  HWY_ASSERT_VEC_EQ(d, expected, ShiftLeftSame(vi, 1));
#endif

  // Simple right shift
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = T(i >> 1);
  }
  HWY_ASSERT_VEC_EQ(d, expected, ShiftRight<1>(vi));
#if !HWY_HAS_VARIABLE_SHIFT || HWY_IDE
  HWY_ASSERT_VEC_EQ(d, expected, ShiftRightSame(vi, 1));
#endif

  // Sign extension
  constexpr T min = LimitsMin<T>();
  const auto vn = Iota(d, min);
  for (size_t i = 0; i < d.N; ++i) {
    // We want a right-shift here, which is undefined behavior for negative
    // numbers. Since we want (-1)>>1 to be -1, we need to adjust rounding if
    // minT is odd and negative.
    T minT = min + i;
    expected[i] = T(minT / 2 + (minT < 0 ? minT % 2 : 0));
  }
  HWY_ASSERT_VEC_EQ(d, expected, ShiftRight<1>(vn));
#if !HWY_HAS_VARIABLE_SHIFT || HWY_IDE
  HWY_ASSERT_VEC_EQ(d, expected, ShiftRightSame(vn, 1));
#endif

  // Shifting negative left
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = T((min + i) << 1);
  }
  HWY_ASSERT_VEC_EQ(d, expected, hwy::ShiftLeft<1>(vn));
#if !HWY_HAS_VARIABLE_SHIFT || HWY_IDE
  HWY_ASSERT_VEC_EQ(d, expected, ShiftLeftSame(vn, 1));
#endif
}

#if HWY_HAS_VARIABLE_SHIFT || HWY_IDE

template <class D>
HWY_NOINLINE HWY_ATTR void TestUnsignedVarShifts(D d) {
  using T = typename D::T;
  constexpr int kSign = (sizeof(T) * 8) - 1;
  const auto v0 = Zero(d);
  const auto v1 = Set(d, 1);
  const auto vi = Iota(d, 0);
  HWY_ALIGN T expected[d.N];

  // Shifting out of right side => zero
  HWY_ASSERT_VEC_EQ(d, v0, vi >> Set(d, 7));

  // Simple left shift
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = T(i << 1);
  }
  HWY_ASSERT_VEC_EQ(d, expected, vi << Set(d, 1));

  // Simple right shift
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = T(i >> 1);
  }
  HWY_ASSERT_VEC_EQ(d, expected, vi >> Set(d, 1));

  // Verify truncation for left-shift
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = (T(i) << kSign) & ~T(0);
  }
  HWY_ASSERT_VEC_EQ(d, expected, vi << Set(d, kSign));

  // Verify variable left shift (assumes < 32 lanes)
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = T(1) << i;
  }
  HWY_ASSERT_VEC_EQ(d, expected, v1 << vi);
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestSignedVarLeftShifts(D d) {
  using T = typename D::T;
  const auto v1 = Set(d, 1);
  const auto vi = Iota(d, 0);

  HWY_ALIGN T expected[d.N];

  // Simple left shift
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = T(i << 1);
  }
  HWY_ASSERT_VEC_EQ(d, expected, vi << v1);

  // Shifting negative numbers left
  constexpr T min = LimitsMin<T>();
  const auto vn = Iota(d, min);
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = T((min + i) << 1);
  }
  HWY_ASSERT_VEC_EQ(d, expected, vn << v1);

  // Differing shift counts (assumes < 32 lanes)
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = T(1) << i;
  }
  HWY_ASSERT_VEC_EQ(d, expected, v1 << vi);
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestSignedVarRightShifts(D d) {
  using T = typename D::T;
  const auto v0 = Zero(d);
  const auto vi = Iota(d, 0);
  const auto vmax = Set(d, LimitsMax<T>());
  HWY_ALIGN T expected[d.N];

  // Shifting out of right side => zero
  HWY_ASSERT_VEC_EQ(d, v0, vi >> Set(d, 7));

  // Simple right shift
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = T(i >> 1);
  }
  HWY_ASSERT_VEC_EQ(d, expected, vi >> Set(d, 1));

  // Sign extension
  constexpr T min = LimitsMin<T>();
  const auto vn = Iota(d, min);
  for (size_t i = 0; i < d.N; ++i) {
    // We want a right-shift here, which is undefined behavior for negative
    // numbers. Since we want (-1)>>1 to be -1, we need to adjust rounding if
    // minT is odd and negative.
    T minT = min + i;
    expected[i] = T(minT / 2 + (minT < 0 ? minT % 2 : 0));
  }
  HWY_ASSERT_VEC_EQ(d, expected, vn >> Set(d, 1));

  // Differing shift counts (assumes < 32 lanes)
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = LimitsMax<T>() >> i;
  }
  HWY_ASSERT_VEC_EQ(d, expected, vmax >> vi);
}

#endif

HWY_NOINLINE HWY_ATTR void TestShifts() {
  // No u8.
  TestUnsignedShifts(du16);
  TestUnsignedShifts(du32);
#if HWY_HAS_INT64
  TestUnsignedShifts(du64);
#endif
  // No i8.
  TestSignedShifts(di16);
  TestSignedShifts(di32);
  // No i64/f32/f64.

#if HWY_HAS_VARIABLE_SHIFT || HWY_IDE
  TestUnsignedVarShifts(du32);
  TestUnsignedVarShifts(du64);
  TestSignedVarLeftShifts(di32);
  TestSignedVarRightShifts(di32);
  TestSignedVarLeftShifts(di64);
// No i64 (right-shift).
#endif
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestUnsignedMinMax(D d) {
  using T = typename D::T;
  const auto v0 = Zero(d);
  const auto v1 = Iota(d, 1);
  const auto v2 = Iota(d, 2);
  const auto v_max = Iota(d, LimitsMax<T>() - d.N + 1);
  HWY_ASSERT_VEC_EQ(d, v1, Min(v1, v2));
  HWY_ASSERT_VEC_EQ(d, v2, Max(v1, v2));
  HWY_ASSERT_VEC_EQ(d, v0, Min(v1, v0));
  HWY_ASSERT_VEC_EQ(d, v1, Max(v1, v0));
  HWY_ASSERT_VEC_EQ(d, v1, Min(v1, v_max));
  HWY_ASSERT_VEC_EQ(d, v_max, Max(v1, v_max));
  HWY_ASSERT_VEC_EQ(d, v0, Min(v0, v_max));
  HWY_ASSERT_VEC_EQ(d, v_max, Max(v0, v_max));
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestSignedMinMax(D d) {
  using T = typename D::T;
  const auto v1 = Iota(d, 1);
  const auto v2 = Iota(d, 2);
  const auto v_neg = Iota(d, -T(d.N));
  const auto v_neg_max = Iota(d, LimitsMin<T>());
  HWY_ASSERT_VEC_EQ(d, v1, Min(v1, v2));
  HWY_ASSERT_VEC_EQ(d, v2, Max(v1, v2));
  HWY_ASSERT_VEC_EQ(d, v_neg, Min(v1, v_neg));
  HWY_ASSERT_VEC_EQ(d, v1, Max(v1, v_neg));
  HWY_ASSERT_VEC_EQ(d, v_neg_max, Min(v1, v_neg_max));
  HWY_ASSERT_VEC_EQ(d, v1, Max(v1, v_neg_max));
  HWY_ASSERT_VEC_EQ(d, v_neg_max, Min(v_neg, v_neg_max));
  HWY_ASSERT_VEC_EQ(d, v_neg, Max(v_neg, v_neg_max));
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestFloatMinMax(D d) {
  using T = typename D::T;
  const auto v1 = Iota(d, 1);
  const auto v2 = Iota(d, 2);
  const auto v_neg = Iota(d, -T(d.N));
  HWY_ASSERT_VEC_EQ(d, v1, Min(v1, v2));
  HWY_ASSERT_VEC_EQ(d, v2, Max(v1, v2));
  HWY_ASSERT_VEC_EQ(d, v_neg, Min(v1, v_neg));
  HWY_ASSERT_VEC_EQ(d, v1, Max(v1, v_neg));
}

HWY_NOINLINE HWY_ATTR void TestMinMax() {
  TestUnsignedMinMax(du8);
  TestUnsignedMinMax(du16);
  TestUnsignedMinMax(du32);
  // No u64.
  TestSignedMinMax(di8);
  TestSignedMinMax(di16);
  TestSignedMinMax(di32);
  // No i64.
  HWY_FOREACH_F(TestFloatMinMax);
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestUnsignedMul(D d) {
  using T = typename D::T;
  const auto v0 = Zero(d);
  const auto v1 = Set(d, T(1));
  const auto vi = Iota(d, 1);
  const auto vj = Iota(d, 3);
  T lanes[d.N] = {};  // Initialized for static analysis.
  HWY_ASSERT_VEC_EQ(d, v0, v0 * v0);
  HWY_ASSERT_VEC_EQ(d, v1, v1 * v1);
  HWY_ASSERT_VEC_EQ(d, vi, v1 * vi);
  HWY_ASSERT_VEC_EQ(d, vi, vi * v1);

  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = (1 + i) * (1 + i);
  }
  HWY_ASSERT_VEC_EQ(d, lanes, vi * vi);

  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = (1 + i) * (3 + i);
  }
  HWY_ASSERT_VEC_EQ(d, lanes, vi * vj);

  const T max = LimitsMax<T>();
  const auto vmax = Set(d, max);
  HWY_ASSERT_VEC_EQ(d, vmax, vmax * v1);
  HWY_ASSERT_VEC_EQ(d, vmax, v1 * vmax);

  const size_t bits = sizeof(T) * 8;
  const uint64_t mask = (1ull << bits) - 1;
  const T max2 = (uint64_t(max) * max) & mask;
  HWY_ASSERT_VEC_EQ(d, Set(d, max2), vmax * vmax);
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestSignedMul(D d) {
  using T = typename D::T;
  const auto v0 = Zero(d);
  const auto v1 = Set(d, T(1));
  const auto vi = Iota(d, 1);
  const auto vn = Iota(d, -T(d.N));
  T lanes[d.N] = {};  // Initialized for static analysis.
  HWY_ASSERT_VEC_EQ(d, v0, v0 * v0);
  HWY_ASSERT_VEC_EQ(d, v1, v1 * v1);
  HWY_ASSERT_VEC_EQ(d, vi, v1 * vi);
  HWY_ASSERT_VEC_EQ(d, vi, vi * v1);

  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = (1 + i) * (1 + i);
  }
  HWY_ASSERT_VEC_EQ(d, lanes, vi * vi);

  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = (-T(d.N) + i) * (1 + i);
  }
  HWY_ASSERT_VEC_EQ(d, lanes, vn * vi);
  HWY_ASSERT_VEC_EQ(d, lanes, vi * vn);
}

HWY_NOINLINE HWY_ATTR void TestMul() {
  // No u8.
  TestUnsignedMul(du16);
  TestUnsignedMul(du32);
  // No u64,i8.
  TestSignedMul(di16);
  TestSignedMul(di32);
  // No i64.
}

template <class D, class D2>
HWY_NOINLINE HWY_ATTR void TestMulHighT(D d, D2 /*d2*/) {
  using T = typename D::T;
  using Wide = typename D2::T;
  HWY_ALIGN T in_lanes[d.N] = {};        // Initialized for static analysis.
  HWY_ALIGN T expected_lanes[d.N] = {};  // Initialized for static analysis.
  const auto vi = Iota(d, 1);
  const auto vni = Iota(d, -T(d.N));

  const auto v0 = Zero(d);
  HWY_ASSERT_VEC_EQ(d, v0, ext::MulHigh(v0, v0));
  HWY_ASSERT_VEC_EQ(d, v0, ext::MulHigh(v0, vi));
  HWY_ASSERT_VEC_EQ(d, v0, ext::MulHigh(vi, v0));

  // Large positive squared
  for (size_t i = 0; i < d.N; ++i) {
    in_lanes[i] = LimitsMax<T>() >> i;
    expected_lanes[i] = (Wide(in_lanes[i]) * in_lanes[i]) >> 16;
  }
  auto v = Load(d, in_lanes);
  HWY_ASSERT_VEC_EQ(d, expected_lanes, ext::MulHigh(v, v));

  // Large positive * small positive
  for (size_t i = 0; i < d.N; ++i) {
    expected_lanes[i] = (Wide(in_lanes[i]) * (1 + i)) >> 16;
  }
  HWY_ASSERT_VEC_EQ(d, expected_lanes, ext::MulHigh(v, vi));
  HWY_ASSERT_VEC_EQ(d, expected_lanes, ext::MulHigh(vi, v));

  // Large positive * small negative
  for (size_t i = 0; i < d.N; ++i) {
    expected_lanes[i] = (Wide(in_lanes[i]) * T(i - d.N)) >> 16;
  }
  HWY_ASSERT_VEC_EQ(d, expected_lanes, ext::MulHigh(v, vni));
  HWY_ASSERT_VEC_EQ(d, expected_lanes, ext::MulHigh(vni, v));
}

HWY_NOINLINE HWY_ATTR void TestMulHigh() {
  TestMulHighT(di16, di32);
  TestMulHighT(du16, du32);
}

template <class D, class D2>
HWY_NOINLINE HWY_ATTR void TestMulEvenT(D d, D2 d2) {
  using T = typename D::T;
  using Wide = typename D2::T;

  const auto v0 = Zero(d);
  HWY_ASSERT_VEC_EQ(d2, Zero(d2), MulEven(v0, v0));

  // scalar has N=1 and we write to "lane 1" below, though it isn't used by
  // the actual MulEven.
  HWY_ALIGN T in_lanes[HWY_MAX(d.N, 2)];
  HWY_ALIGN Wide expected[d2.N];
  for (size_t i = 0; i < d.N; i += 2) {
    in_lanes[i + 0] = LimitsMax<T>() >> i;
    in_lanes[i + 1] = 1;  // will be overwritten with upper half of result
    expected[i / 2] = Wide(in_lanes[i + 0]) * in_lanes[i + 0];
  }

  const auto v = Load(d, in_lanes);
  HWY_ASSERT_VEC_EQ(d2, expected, MulEven(v, v));
}

HWY_NOINLINE HWY_ATTR void TestMulEven() {
  TestMulEvenT(di32, di64);
  TestMulEvenT(du32, du64);
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestMulAdd(D d) {
  using T = typename D::T;
  const auto k0 = Zero(d);
  const auto v1 = Iota(d, 1);
  const auto v2 = Iota(d, 2);
  T lanes[d.N];
  HWY_ASSERT_VEC_EQ(d, k0, MulAdd(k0, k0, k0));
  HWY_ASSERT_VEC_EQ(d, v2, MulAdd(k0, v1, v2));
  HWY_ASSERT_VEC_EQ(d, v2, MulAdd(v1, k0, v2));
  HWY_ASSERT_VEC_EQ(d, k0, NegMulAdd(k0, k0, k0));
  HWY_ASSERT_VEC_EQ(d, v2, NegMulAdd(k0, v1, v2));
  HWY_ASSERT_VEC_EQ(d, v2, NegMulAdd(v1, k0, v2));

  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = (i + 1) * (i + 2);
  }
  HWY_ASSERT_VEC_EQ(d, lanes, MulAdd(v2, v1, k0));
  HWY_ASSERT_VEC_EQ(d, lanes, MulAdd(v1, v2, k0));
  HWY_ASSERT_VEC_EQ(d, lanes, NegMulAdd(Neg(v2), v1, k0));
  HWY_ASSERT_VEC_EQ(d, lanes, NegMulAdd(v1, Neg(v2), k0));

  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = (i + 2) * (i + 2) + (i + 1);
  }
  HWY_ASSERT_VEC_EQ(d, lanes, MulAdd(v2, v2, v1));
  HWY_ASSERT_VEC_EQ(d, lanes, NegMulAdd(Neg(v2), v2, v1));

  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = -T(i + 2) * (i + 2) + (1 + i);
  }
  HWY_ASSERT_VEC_EQ(d, lanes, NegMulAdd(v2, v2, v1));

  HWY_ASSERT_VEC_EQ(d, k0, ext::MulSub(k0, k0, k0));
  HWY_ASSERT_VEC_EQ(d, k0, ext::NegMulSub(k0, k0, k0));

  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = -T(i + 2);
  }
  HWY_ASSERT_VEC_EQ(d, lanes, ext::MulSub(k0, v1, v2));
  HWY_ASSERT_VEC_EQ(d, lanes, ext::MulSub(v1, k0, v2));
  HWY_ASSERT_VEC_EQ(d, lanes, ext::NegMulSub(Neg(k0), v1, v2));
  HWY_ASSERT_VEC_EQ(d, lanes, ext::NegMulSub(v1, Neg(k0), v2));

  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = (i + 1) * (i + 2);
  }
  HWY_ASSERT_VEC_EQ(d, lanes, ext::MulSub(v1, v2, k0));
  HWY_ASSERT_VEC_EQ(d, lanes, ext::MulSub(v2, v1, k0));
  HWY_ASSERT_VEC_EQ(d, lanes, ext::NegMulSub(Neg(v1), v2, k0));
  HWY_ASSERT_VEC_EQ(d, lanes, ext::NegMulSub(v2, Neg(v1), k0));

  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = (i + 2) * (i + 2) - (1 + i);
  }
  HWY_ASSERT_VEC_EQ(d, lanes, ext::MulSub(v2, v2, v1));
  HWY_ASSERT_VEC_EQ(d, lanes, ext::NegMulSub(Neg(v2), v2, v1));
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestSquareRoot(D d) {
  const auto vi = Iota(d, 0);
  HWY_ASSERT_VEC_EQ(d, vi, Sqrt(vi * vi));
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestReciprocalSquareRoot(D d) {
  const auto v = Set(d, 123.0f);
  HWY_ALIGN float lanes[d.N];
  Store(ApproximateReciprocalSqrt(v), d, lanes);
  for (size_t i = 0; i < d.N; ++i) {
    float err = lanes[i] - 0.090166f;
    if (err < 0.0f) err = -err;
    HWY_ASSERT_EQ(true, err < 1E-4f);
  }
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestRound(D d) {
  using T = typename D::T;
  // Numbers close to 0
  {
    const auto v = Set(d, 0.4);
    const auto zero = Set(d, 0);
    const auto one = Set(d, 1);
    HWY_ASSERT_VEC_EQ(d, one, Ceil(v));
    HWY_ASSERT_VEC_EQ(d, zero, Floor(v));
    HWY_ASSERT_VEC_EQ(d, zero, Round(v));
    HWY_ASSERT_VEC_EQ(d, zero, Trunc(v));
  }
  {
    const auto v = Set(d, -0.4);
    const auto zero = Set(d, 0);
    const auto one = Set(d, -1);
    HWY_ASSERT_VEC_EQ(d, zero, Ceil(v));
    HWY_ASSERT_VEC_EQ(d, one, Floor(v));
    HWY_ASSERT_VEC_EQ(d, zero, Round(v));
    HWY_ASSERT_VEC_EQ(d, zero, Trunc(v));
  }
  // Integer positive
  {
    const auto v = Iota(d, 4.0);
    HWY_ASSERT_VEC_EQ(d, v, Ceil(v));
    HWY_ASSERT_VEC_EQ(d, v, Floor(v));
    HWY_ASSERT_VEC_EQ(d, v, Round(v));
    HWY_ASSERT_VEC_EQ(d, v, Trunc(v));
  }

  // Integer negative
  {
    const auto v = Iota(d, T(-32.0));
    HWY_ASSERT_VEC_EQ(d, v, Ceil(v));
    HWY_ASSERT_VEC_EQ(d, v, Floor(v));
    HWY_ASSERT_VEC_EQ(d, v, Round(v));
    HWY_ASSERT_VEC_EQ(d, v, Trunc(v));
  }

  // Huge positive
  {
    const auto v = Set(d, T(1E15));
    HWY_ASSERT_VEC_EQ(d, v, Ceil(v));
    HWY_ASSERT_VEC_EQ(d, v, Floor(v));
  }

  // Huge negative
  {
    const auto v = Set(d, T(-1E15));
    HWY_ASSERT_VEC_EQ(d, v, Ceil(v));
    HWY_ASSERT_VEC_EQ(d, v, Floor(v));
  }

  // Above positive
  {
    const auto v = Iota(d, T(2.0001));
    const auto v3 = Iota(d, T(3));
    const auto v2 = Iota(d, T(2));
    HWY_ASSERT_VEC_EQ(d, v3, Ceil(v));
    HWY_ASSERT_VEC_EQ(d, v2, Floor(v));
    HWY_ASSERT_VEC_EQ(d, v2, Round(v));
    HWY_ASSERT_VEC_EQ(d, v2, Trunc(v));
  }

  // Below positive
  {
    const auto v = Iota(d, T(3.9999));
    const auto v4 = Iota(d, T(4));
    const auto v3 = Iota(d, T(3));
    HWY_ASSERT_VEC_EQ(d, v4, Ceil(v));
    HWY_ASSERT_VEC_EQ(d, v3, Floor(v));
    HWY_ASSERT_VEC_EQ(d, v4, Round(v));
    HWY_ASSERT_VEC_EQ(d, v3, Trunc(v));
  }

  // Above negative
  {
    // WARNING: using iota => ensure negative value is low enough that
    // even 16 lanes remain negative, otherwise trunc will behave differently
    // for positive/negative values.
    const auto v = Iota(d, T(-19.9999));
    const auto v3 = Iota(d, T(-19));
    const auto v4 = Iota(d, T(-20));
    HWY_ASSERT_VEC_EQ(d, v3, Ceil(v));
    HWY_ASSERT_VEC_EQ(d, v4, Floor(v));
    HWY_ASSERT_VEC_EQ(d, v4, Round(v));
    HWY_ASSERT_VEC_EQ(d, v3, Trunc(v));
  }

  // Below negative
  {
    const auto v = Iota(d, T(-18.0001));
    const auto v2 = Iota(d, T(-18));
    const auto v3 = Iota(d, T(-19));
    HWY_ASSERT_VEC_EQ(d, v2, Ceil(v));
    HWY_ASSERT_VEC_EQ(d, v3, Floor(v));
    HWY_ASSERT_VEC_EQ(d, v2, Round(v));
    HWY_ASSERT_VEC_EQ(d, v2, Trunc(v));
  }
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestIntFromFloat(D d) {
  // Integer positive
  HWY_ASSERT_VEC_EQ(d, Iota(d, 4), ConvertTo(d, Iota(df, 4.0f)));
  HWY_ASSERT_VEC_EQ(d, Iota(d, 4), NearestInt(Iota(df, 4.0f)));

  // Integer negative
  HWY_ASSERT_VEC_EQ(d, Iota(d, -32), ConvertTo(d, Iota(df, -32.0f)));
  HWY_ASSERT_VEC_EQ(d, Iota(d, -32), NearestInt(Iota(df, -32.0f)));

  // Above positive
  HWY_ASSERT_VEC_EQ(d, Iota(d, 2), ConvertTo(d, Iota(df, 2.001f)));
  HWY_ASSERT_VEC_EQ(d, Iota(d, 2), NearestInt(Iota(df, 2.001f)));

  // Below positive
  HWY_ASSERT_VEC_EQ(d, Iota(d, 3), ConvertTo(d, Iota(df, 3.9999f)));
  HWY_ASSERT_VEC_EQ(d, Iota(d, 4), NearestInt(Iota(df, 3.9999f)));

  // Above negative
  HWY_ASSERT_VEC_EQ(d, Iota(d, -23), ConvertTo(d, Iota(df, -23.9999f)));
  HWY_ASSERT_VEC_EQ(d, Iota(d, -24), NearestInt(Iota(df, -23.9999f)));

  // Below negative
  HWY_ASSERT_VEC_EQ(d, Iota(d, -24), ConvertTo(d, Iota(df, -24.001f)));
  HWY_ASSERT_VEC_EQ(d, Iota(d, -24), NearestInt(Iota(df, -24.001f)));
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestFloatFromInt(D d) {
  // Integer positive
  HWY_ASSERT_VEC_EQ(d, Iota(d, 4.0f), ConvertTo(d, Iota(di32, 4)));

  // Integer negative
  HWY_ASSERT_VEC_EQ(d, Iota(d, -32.0f), ConvertTo(d, Iota(di32, -32)));

  // Above positive
  HWY_ASSERT_VEC_EQ(d, Iota(d, 2.0f), ConvertTo(d, Iota(di32, 2)));

  // Below positive
  HWY_ASSERT_VEC_EQ(d, Iota(d, 4.0f), ConvertTo(d, Iota(di32, 4)));

  // Above negative
  HWY_ASSERT_VEC_EQ(d, Iota(d, -4.0f), ConvertTo(d, Iota(di32, -4)));

  // Below negative
  HWY_ASSERT_VEC_EQ(d, Iota(d, -2.0f), ConvertTo(d, Iota(di32, -2)));
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestSumsOfU8(D d) {
#if HWY_BITS != 0 || HWY_IDE
  HWY_ALIGN uint8_t in_bytes[du8.N];
  uint64_t sums[d.N] = {0};
  for (size_t i = 0; i < du8.N; ++i) {
    const size_t group = i / 8;
    in_bytes[i] = static_cast<uint8_t>(2 * i + 1);
    sums[group] += in_bytes[i];
  }
  const auto v = Load(du8, in_bytes);
  HWY_ASSERT_VEC_EQ(d, sums, ext::SumsOfU8x8(v));
#else
  (void)d;
#endif
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestHorzSumT(D d) {
  using T = typename D::T;

  HWY_ALIGN T in_lanes[d.N];
  double sum = 0.0;
  for (size_t i = 0; i < d.N; ++i) {
    in_lanes[i] = 1u << i;
    sum += in_lanes[i];
  }
  const auto v = Load(d, in_lanes);
  const auto expected = Set(d, T(sum));
  HWY_ASSERT_VEC_EQ(d, expected, ext::SumOfLanes(v));
}

HWY_NOINLINE HWY_ATTR void TestHorzSum() {
  // No u16.
  TestHorzSumT(du32);
#if HWY_HAS_INT64
  TestHorzSumT(du64);
#endif

  // No i8/i16.
  TestHorzSumT(di32);
#if HWY_HAS_INT64
  TestHorzSumT(di64);
#endif

  HWY_FOREACH_F(TestHorzSumT);
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestAbsDiffT(D d) {
  using T = typename D::T;

  HWY_ALIGN T in_lanes_a[d.N];
  HWY_ALIGN T in_lanes_b[d.N];
  HWY_ALIGN T out_lanes[d.N];
  for (size_t i = 0; i < d.N; ++i) {
    in_lanes_a[i] = (i ^ 1u) << i;
    in_lanes_b[i] = i << i;
    out_lanes[i] = fabsf(in_lanes_a[i] - in_lanes_b[i]);
  }
  const auto a = Load(d, in_lanes_a);
  const auto b = Load(d, in_lanes_b);
  const auto expected = Load(d, out_lanes);
  HWY_ASSERT_VEC_EQ(d, expected, ext::AbsDiff(a, b));
  HWY_ASSERT_VEC_EQ(d, expected, ext::AbsDiff(b, a));
}

HWY_NOINLINE HWY_ATTR void TestAbsDiff() { TestAbsDiffT(df); }

HWY_NOINLINE HWY_ATTR void TestArithmetic() {
  (void)dd;
  (void)di64;
  (void)du64;

  HWY_FOREACH_UIF(TestPlusMinus);
  TestSaturatingArithmetic();

  TestShifts();
  TestMinMax();
  TestAverage();
  TestAbs();
  TestMul();
  TestMulHigh();
  TestMulEven();

  HWY_FOREACH_F(TestMulAdd);
  HWY_FOREACH_F(TestSquareRoot);
  TestReciprocalSquareRoot(df);
  HWY_FOREACH_F(TestRound);
  TestIntFromFloat(di32);
  TestFloatFromInt(df);

  TestSumsOfU8(du64);
  TestHorzSum();

  TestAbsDiff();
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy

// Instantiate for the current target.
void ArithmeticTest::HWY_FUNC() { hwy::HWY_NAMESPACE::TestArithmetic(); }
