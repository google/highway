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

#include "highway/arithmetic_test.h"
#include "highway/test_target_util.h"

namespace jxl {
namespace SIMD_NAMESPACE {
namespace {

struct TestPlusMinus {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v2 = iota(D(), 2);
    const auto v3 = iota(D(), 3);
    const auto v4 = iota(D(), 4);

    T lanes[d.N] = {};  // Initialized for static analysis.
    for (size_t i = 0; i < d.N; ++i) {
      lanes[i] = (2 + i) + (3 + i);
    }
    SIMD_ASSERT_VEC_EQ(d, lanes, v2 + v3);
    SIMD_ASSERT_VEC_EQ(d, v3, (v2 + v3) - v2);

    for (size_t i = 0; i < d.N; ++i) {
      lanes[i] = (2 + i) + (4 + i);
    }
    auto sum = v2;
    sum += v4;  // sum == 6,8..
    SIMD_ASSERT_VEC_EQ(d, lanes, sum);

    sum -= v4;
    SIMD_ASSERT_VEC_EQ(d, v2, sum);
  }
};

struct TestUnsignedSaturatingArithmetic {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v0 = setzero(d);
    const auto vi = iota(d, 1);
    const auto vm = set1(d, LimitsMax<T>());

    SIMD_ASSERT_VEC_EQ(d, v0 + v0, saturated_add(v0, v0));
    SIMD_ASSERT_VEC_EQ(d, v0 + vi, saturated_add(v0, vi));
    SIMD_ASSERT_VEC_EQ(d, v0 + vm, saturated_add(v0, vm));
    SIMD_ASSERT_VEC_EQ(d, vm, saturated_add(vi, vm));
    SIMD_ASSERT_VEC_EQ(d, vm, saturated_add(vm, vm));

    SIMD_ASSERT_VEC_EQ(d, v0, saturated_subtract(v0, v0));
    SIMD_ASSERT_VEC_EQ(d, v0, saturated_subtract(v0, vi));
    SIMD_ASSERT_VEC_EQ(d, v0, saturated_subtract(vi, vi));
    SIMD_ASSERT_VEC_EQ(d, v0, saturated_subtract(vi, vm));
    SIMD_ASSERT_VEC_EQ(d, vm - vi, saturated_subtract(vm, vi));
  }
};

struct TestSignedSaturatingArithmetic {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v0 = setzero(d);
    const auto vi = iota(d, 1);
    const auto vpm = set1(d, LimitsMax<T>());
    const auto vn = iota(d, -T(d.N));
    const auto vnm = set1(d, LimitsMin<T>());

    SIMD_ASSERT_VEC_EQ(d, v0, saturated_add(v0, v0));
    SIMD_ASSERT_VEC_EQ(d, vi, saturated_add(v0, vi));
    SIMD_ASSERT_VEC_EQ(d, vpm, saturated_add(v0, vpm));
    SIMD_ASSERT_VEC_EQ(d, vpm, saturated_add(vi, vpm));
    SIMD_ASSERT_VEC_EQ(d, vpm, saturated_add(vpm, vpm));

    SIMD_ASSERT_VEC_EQ(d, v0, saturated_subtract(v0, v0));
    SIMD_ASSERT_VEC_EQ(d, v0 - vi, saturated_subtract(v0, vi));
    SIMD_ASSERT_VEC_EQ(d, vn, saturated_subtract(vn, v0));
    SIMD_ASSERT_VEC_EQ(d, vnm, saturated_subtract(vnm, vi));
    SIMD_ASSERT_VEC_EQ(d, vnm, saturated_subtract(vnm, vpm));
  }
};

SIMD_ATTR void TestSaturatingArithmetic() {
  Call<TestUnsignedSaturatingArithmetic, uint8_t>();
  Call<TestUnsignedSaturatingArithmetic, uint16_t>();
  Call<TestSignedSaturatingArithmetic, int8_t>();
  Call<TestSignedSaturatingArithmetic, int16_t>();
}

struct TestAverageT {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v0 = setzero(d);
    const auto v1 = set1(d, T(1));
    const auto v2 = set1(d, T(2));

    SIMD_ASSERT_VEC_EQ(d, v0, average_round(v0, v0));
    SIMD_ASSERT_VEC_EQ(d, v1, average_round(v0, v1));
    SIMD_ASSERT_VEC_EQ(d, v1, average_round(v1, v1));
    SIMD_ASSERT_VEC_EQ(d, v2, average_round(v1, v2));
    SIMD_ASSERT_VEC_EQ(d, v2, average_round(v2, v2));
  }
};

SIMD_ATTR void TestAverage() {
  Call<TestAverageT, uint8_t>();
  Call<TestAverageT, uint16_t>();
}

struct TestAbsT {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v0 = setzero(d);
    const auto vp1 = set1(d, T(1));
    const auto vn1 = set1(d, T(-1));
    const auto vpm = set1(d, LimitsMax<T>());
    const auto vnm = set1(d, LimitsMin<T>());

    SIMD_ASSERT_VEC_EQ(d, v0, abs(v0));
    SIMD_ASSERT_VEC_EQ(d, vp1, abs(vp1));
    SIMD_ASSERT_VEC_EQ(d, vp1, abs(vn1));
    SIMD_ASSERT_VEC_EQ(d, vpm, abs(vpm));
    SIMD_ASSERT_VEC_EQ(d, vnm, abs(vnm));
  }
};

SIMD_ATTR void TestAbs() {
  Call<TestAbsT, int8_t>();
  Call<TestAbsT, int16_t>();
  Call<TestAbsT, int32_t>();
}

struct TestUnsignedShifts {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    constexpr int kSign = (sizeof(T) * 8) - 1;
    const auto v0 = setzero(d);
    const auto vi = iota(d, 0);
    SIMD_ALIGN T expected[d.N] = {};  // Initialized for static analysis.

    // Shifting out of right side => zero
    SIMD_ASSERT_VEC_EQ(d, v0, shift_right<7>(vi));
    SIMD_ASSERT_VEC_EQ(d, v0,
                       shift_right_same(vi, set_shift_right_count(d, 7)));

    // Simple left shift
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = T(i << 1);
    }
    SIMD_ASSERT_VEC_EQ(d, expected, shift_left<1>(vi));
    SIMD_ASSERT_VEC_EQ(d, expected,
                       shift_left_same(vi, set_shift_left_count(d, 1)));

    // Simple right shift
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = T(i >> 1);
    }
    SIMD_ASSERT_VEC_EQ(d, expected, shift_right<1>(vi));
    SIMD_ASSERT_VEC_EQ(d, expected,
                       shift_right_same(vi, set_shift_right_count(d, 1)));

    // Verify truncation for left-shift
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = (T(i) << kSign) & ~T(0);
    }
    SIMD_ASSERT_VEC_EQ(d, expected, shift_left<kSign>(vi));
    SIMD_ASSERT_VEC_EQ(d, expected,
                       shift_left_same(vi, set_shift_left_count(d, kSign)));
  }
};

struct TestSignedShifts {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v0 = setzero(d);
    const auto vi = iota(d, 0);
    SIMD_ALIGN T expected[d.N] = {};  // Initialized for static analysis.

    // Shifting out of right side => zero
    SIMD_ASSERT_VEC_EQ(d, v0, shift_right<7>(vi));
    SIMD_ASSERT_VEC_EQ(d, v0,
                       shift_right_same(vi, set_shift_right_count(d, 7)));

    // Simple left shift
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = T(i << 1);
    }
    SIMD_ASSERT_VEC_EQ(d, expected, shift_left<1>(vi));
    SIMD_ASSERT_VEC_EQ(d, expected,
                       shift_left_same(vi, set_shift_left_count(d, 1)));

    // Simple right shift
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = T(i >> 1);
    }
    SIMD_ASSERT_VEC_EQ(d, expected, shift_right<1>(vi));
    SIMD_ASSERT_VEC_EQ(d, expected,
                       shift_right_same(vi, set_shift_right_count(d, 1)));

    // Sign extension
    constexpr T min = LimitsMin<T>();
    const auto vn = iota(d, min);
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = T((min + i) >> 1);
    }
    SIMD_ASSERT_VEC_EQ(d, expected, shift_right<1>(vn));
    SIMD_ASSERT_VEC_EQ(d, expected,
                       shift_right_same(vn, set_shift_right_count(d, 1)));

    // Shifting negative left
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = T((min + i) << 1);
    }
    SIMD_ASSERT_VEC_EQ(d, expected, shift_left<1>(vn));
    SIMD_ASSERT_VEC_EQ(d, expected,
                       shift_left_same(vn, set_shift_left_count(d, 1)));
  }
};

#if SIMD_HAS_VARIABLE_SHIFT

struct TestUnsignedVarShifts {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    constexpr int kSign = (sizeof(T) * 8) - 1;
    const auto v0 = setzero(d);
    const auto v1 = set1(d, 1);
    const auto vi = iota(d, 0);
    SIMD_ALIGN T expected[d.N];

    // Shifting out of right side => zero
    SIMD_ASSERT_VEC_EQ(d, v0, vi >> set1(d, 7));

    // Simple left shift
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = T(i << 1);
    }
    SIMD_ASSERT_VEC_EQ(d, expected, vi << set1(d, 1));

    // Simple right shift
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = T(i >> 1);
    }
    SIMD_ASSERT_VEC_EQ(d, expected, vi >> set1(d, 1));

    // Verify truncation for left-shift
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = (T(i) << kSign) & ~T(0);
    }
    SIMD_ASSERT_VEC_EQ(d, expected, vi << set1(d, kSign));

    // Verify variable left shift (assumes < 32 lanes)
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = T(1) << i;
    }
    SIMD_ASSERT_VEC_EQ(d, expected, v1 << vi);
  }
};

struct TestSignedVarLeftShifts {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v1 = set1(d, 1);
    const auto vi = iota(d, 0);

    SIMD_ALIGN T expected[d.N];

    // Simple left shift
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = T(i << 1);
    }
    SIMD_ASSERT_VEC_EQ(d, expected, vi << v1);

    // Shifting negative numbers left
    constexpr T min = LimitsMin<T>();
    const auto vn = iota(d, min);
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = T((min + i) << 1);
    }
    SIMD_ASSERT_VEC_EQ(d, expected, vn << v1);

    // Differing shift counts (assumes < 32 lanes)
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = T(1) << i;
    }
    SIMD_ASSERT_VEC_EQ(d, expected, v1 << vi);
  }
};

struct TestSignedVarRightShifts {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v0 = setzero(d);
    const auto vi = iota(d, 0);
    const auto vmax = set1(d, LimitsMax<T>());
    SIMD_ALIGN T expected[d.N];

    // Shifting out of right side => zero
    SIMD_ASSERT_VEC_EQ(d, v0, vi >> set1(d, 7));

    // Simple right shift
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = T(i >> 1);
    }
    SIMD_ASSERT_VEC_EQ(d, expected, vi >> set1(d, 1));

    // Sign extension
    constexpr T min = LimitsMin<T>();
    const auto vn = iota(d, min);
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = T((min + i) >> 1);
    }
    SIMD_ASSERT_VEC_EQ(d, expected, vn >> set1(d, 1));

    // Differing shift counts (assumes < 32 lanes)
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = LimitsMax<T>() >> i;
    }
    SIMD_ASSERT_VEC_EQ(d, expected, vmax >> vi);
  }
};

#endif

SIMD_ATTR void TestShifts() {
  // No u8.
  Call<TestUnsignedShifts, uint16_t>();
  Call<TestUnsignedShifts, uint32_t>();
  Call<TestUnsignedShifts, uint64_t>();
  // No i8.
  Call<TestSignedShifts, int16_t>();
  Call<TestSignedShifts, int32_t>();
  // No i64/f32/f64.

#if SIMD_HAS_VARIABLE_SHIFT
  Call<TestUnsignedVarShifts, uint32_t>();
  Call<TestUnsignedVarShifts, uint64_t>();
  Call<TestSignedVarLeftShifts, int32_t>();
  Call<TestSignedVarRightShifts, int32_t>();
  Call<TestSignedVarLeftShifts, int64_t>();
// No i64 (right-shift).
#endif
}

struct TestUnsignedMinMax {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v0 = setzero(d);
    const auto v1 = iota(d, 1);
    const auto v2 = iota(d, 2);
    const auto v_max = iota(d, LimitsMax<T>() - d.N + 1);
    SIMD_ASSERT_VEC_EQ(d, v1, min(v1, v2));
    SIMD_ASSERT_VEC_EQ(d, v2, max(v1, v2));
    SIMD_ASSERT_VEC_EQ(d, v0, min(v1, v0));
    SIMD_ASSERT_VEC_EQ(d, v1, max(v1, v0));
    SIMD_ASSERT_VEC_EQ(d, v1, min(v1, v_max));
    SIMD_ASSERT_VEC_EQ(d, v_max, max(v1, v_max));
    SIMD_ASSERT_VEC_EQ(d, v0, min(v0, v_max));
    SIMD_ASSERT_VEC_EQ(d, v_max, max(v0, v_max));
  }
};

struct TestSignedMinMax {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v1 = iota(d, 1);
    const auto v2 = iota(d, 2);
    const auto v_neg = iota(d, -T(d.N));
    const auto v_neg_max = iota(d, LimitsMin<T>());
    SIMD_ASSERT_VEC_EQ(d, v1, min(v1, v2));
    SIMD_ASSERT_VEC_EQ(d, v2, max(v1, v2));
    SIMD_ASSERT_VEC_EQ(d, v_neg, min(v1, v_neg));
    SIMD_ASSERT_VEC_EQ(d, v1, max(v1, v_neg));
    SIMD_ASSERT_VEC_EQ(d, v_neg_max, min(v1, v_neg_max));
    SIMD_ASSERT_VEC_EQ(d, v1, max(v1, v_neg_max));
    SIMD_ASSERT_VEC_EQ(d, v_neg_max, min(v_neg, v_neg_max));
    SIMD_ASSERT_VEC_EQ(d, v_neg, max(v_neg, v_neg_max));
  }
};

struct TestFloatMinMax {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v1 = iota(d, 1);
    const auto v2 = iota(d, 2);
    const auto v_neg = iota(d, -T(d.N));
    SIMD_ASSERT_VEC_EQ(d, v1, min(v1, v2));
    SIMD_ASSERT_VEC_EQ(d, v2, max(v1, v2));
    SIMD_ASSERT_VEC_EQ(d, v_neg, min(v1, v_neg));
    SIMD_ASSERT_VEC_EQ(d, v1, max(v1, v_neg));
  }
};

SIMD_ATTR void TestMinMax() {
  Call<TestUnsignedMinMax, uint8_t>();
  Call<TestUnsignedMinMax, uint16_t>();
  Call<TestUnsignedMinMax, uint32_t>();
  // No u64.
  Call<TestSignedMinMax, int8_t>();
  Call<TestSignedMinMax, int16_t>();
  Call<TestSignedMinMax, int32_t>();
  // No i64.
  Call<TestFloatMinMax, float>();
  Call<TestFloatMinMax, double>();
}

struct TestUnsignedMul {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v0 = setzero(d);
    const auto v1 = set1(d, T(1));
    const auto vi = iota(d, 1);
    const auto vj = iota(d, 3);
    T lanes[d.N] = {};  // Initialized for static analysis.
    SIMD_ASSERT_VEC_EQ(d, v0, v0 * v0);
    SIMD_ASSERT_VEC_EQ(d, v1, v1 * v1);
    SIMD_ASSERT_VEC_EQ(d, vi, v1 * vi);
    SIMD_ASSERT_VEC_EQ(d, vi, vi * v1);

    for (size_t i = 0; i < d.N; ++i) {
      lanes[i] = (1 + i) * (1 + i);
    }
    SIMD_ASSERT_VEC_EQ(d, lanes, vi * vi);

    for (size_t i = 0; i < d.N; ++i) {
      lanes[i] = (1 + i) * (3 + i);
    }
    SIMD_ASSERT_VEC_EQ(d, lanes, vi * vj);

    const T max = LimitsMax<T>();
    const auto vmax = set1(d, max);
    SIMD_ASSERT_VEC_EQ(d, vmax, vmax * v1);
    SIMD_ASSERT_VEC_EQ(d, vmax, v1 * vmax);

    const size_t bits = sizeof(T) * 8;
    const uint64_t mask = (1ull << bits) - 1;
    const T max2 = (uint64_t(max) * max) & mask;
    SIMD_ASSERT_VEC_EQ(d, set1(d, max2), vmax * vmax);
  }
};

struct TestSignedMul {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v0 = setzero(d);
    const auto v1 = set1(d, T(1));
    const auto vi = iota(d, 1);
    const auto vn = iota(d, -T(d.N));
    T lanes[d.N] = {};  // Initialized for static analysis.
    SIMD_ASSERT_VEC_EQ(d, v0, v0 * v0);
    SIMD_ASSERT_VEC_EQ(d, v1, v1 * v1);
    SIMD_ASSERT_VEC_EQ(d, vi, v1 * vi);
    SIMD_ASSERT_VEC_EQ(d, vi, vi * v1);

    for (size_t i = 0; i < d.N; ++i) {
      lanes[i] = (1 + i) * (1 + i);
    }
    SIMD_ASSERT_VEC_EQ(d, lanes, vi * vi);

    for (size_t i = 0; i < d.N; ++i) {
      lanes[i] = (-T(d.N) + i) * (1 + i);
    }
    SIMD_ASSERT_VEC_EQ(d, lanes, vn * vi);
    SIMD_ASSERT_VEC_EQ(d, lanes, vi * vn);
  }
};

SIMD_ATTR void TestMul() {
  // No u8.
  Call<TestUnsignedMul, uint16_t>();
  Call<TestUnsignedMul, uint32_t>();
  // No u64,i8.
  Call<TestSignedMul, int16_t>();
  Call<TestSignedMul, int32_t>();
  // No i64.
}

template <typename T, typename Wide>
SIMD_ATTR void TestMulHighT() {
  const SIMD_FULL(T) d;
  SIMD_ALIGN T in_lanes[d.N] = {};        // Initialized for static analysis.
  SIMD_ALIGN T expected_lanes[d.N] = {};  // Initialized for static analysis.
  const auto vi = iota(d, 1);
  const auto vni = iota(d, -T(d.N));

  const auto v0 = setzero(d);
  SIMD_ASSERT_VEC_EQ(d, v0, ext::mul_high(v0, v0));
  SIMD_ASSERT_VEC_EQ(d, v0, ext::mul_high(v0, vi));
  SIMD_ASSERT_VEC_EQ(d, v0, ext::mul_high(vi, v0));

  // Large positive squared
  for (size_t i = 0; i < d.N; ++i) {
    in_lanes[i] = LimitsMax<T>() >> i;
    expected_lanes[i] = (Wide(in_lanes[i]) * in_lanes[i]) >> 16;
  }
  auto v = load(d, in_lanes);
  SIMD_ASSERT_VEC_EQ(d, expected_lanes, ext::mul_high(v, v));

  // Large positive * small positive
  for (size_t i = 0; i < d.N; ++i) {
    expected_lanes[i] = (Wide(in_lanes[i]) * (1 + i)) >> 16;
  }
  SIMD_ASSERT_VEC_EQ(d, expected_lanes, ext::mul_high(v, vi));
  SIMD_ASSERT_VEC_EQ(d, expected_lanes, ext::mul_high(vi, v));

  // Large positive * small negative
  for (size_t i = 0; i < d.N; ++i) {
    expected_lanes[i] = (Wide(in_lanes[i]) * T(i - d.N)) >> 16;
  }
  SIMD_ASSERT_VEC_EQ(d, expected_lanes, ext::mul_high(v, vni));
  SIMD_ASSERT_VEC_EQ(d, expected_lanes, ext::mul_high(vni, v));
}

SIMD_ATTR void TestMulHigh() {
  TestMulHighT<int16_t, int32_t>();
  TestMulHighT<uint16_t, uint32_t>();
}

template <typename T1, typename T2>
SIMD_ATTR void TestMulEvenT() {
  const SIMD_FULL(T1) d1;
  const SIMD_FULL(T2) d2;  // wider type, half the lanes

  const auto v0 = setzero(d1);
  SIMD_ASSERT_VEC_EQ(d2, setzero(d2), mul_even(v0, v0));

  // scalar has N=1 and we write to "lane 1" below, though it isn't used by
  // the actual mul_even.
  SIMD_ALIGN T1 in_lanes[SIMD_MAX(d1.N, 2)];
  SIMD_ALIGN T2 expected[d2.N];
  for (size_t i = 0; i < d1.N; i += 2) {
    in_lanes[i + 0] = LimitsMax<T1>() >> i;
    in_lanes[i + 1] = 1;  // will be overwritten with upper half of result
    expected[i / 2] = T2(in_lanes[i + 0]) * in_lanes[i + 0];
  }

  const auto v = load(d1, in_lanes);
  SIMD_ASSERT_VEC_EQ(d2, expected, mul_even(v, v));
}

SIMD_ATTR void TestMulEven() {
  TestMulEvenT<int32_t, int64_t>();
  TestMulEvenT<uint32_t, uint64_t>();
}

struct TestMulAdd {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto k0 = setzero(d);
    const auto k1 = set1(d, 1);
    const auto v1 = iota(d, 1);
    const auto v2 = iota(d, 2);
    T lanes[d.N];
    SIMD_ASSERT_VEC_EQ(d, k0, mul_add(k0, k0, k0));
    SIMD_ASSERT_VEC_EQ(d, v2, mul_add(k0, v1, v2));
    SIMD_ASSERT_VEC_EQ(d, v2, mul_add(v1, k0, v2));
    SIMD_ASSERT_VEC_EQ(d, k0, nmul_add(k0, k0, k0));
    SIMD_ASSERT_VEC_EQ(d, v2, nmul_add(k0, v1, v2));
    SIMD_ASSERT_VEC_EQ(d, v2, nmul_add(v1, k0, v2));

    SIMD_ASSERT_VEC_EQ(d, v1, fadd(k0, k1, v1));
    SIMD_ASSERT_VEC_EQ(d, v2, fadd(k0, k1, v2));
    SIMD_ASSERT_VEC_EQ(d, v1, fadd(v1, k1, k0));
    SIMD_ASSERT_VEC_EQ(d, v2, fadd(v2, k1, k0));

    SIMD_ASSERT_VEC_EQ(d, v2, fsub(v2, k1, k0));
    SIMD_ASSERT_VEC_EQ(d, v1, fsub(v1, k1, k0));
    SIMD_ASSERT_VEC_EQ(d, v2, fsub(k0, k1, neg(v2)));
    SIMD_ASSERT_VEC_EQ(d, v1, fsub(k0, k1, neg(v1)));

    // Swapped arg order
    SIMD_ASSERT_VEC_EQ(d, v2, fnadd(k0, k1, v2));
    SIMD_ASSERT_VEC_EQ(d, v1, fnadd(k0, k1, v1));
    SIMD_ASSERT_VEC_EQ(d, v2, fnadd(neg(v2), k1, k0));
    SIMD_ASSERT_VEC_EQ(d, v1, fnadd(neg(v1), k1, k0));

    for (size_t i = 0; i < d.N; ++i) {
      lanes[i] = (i + 1) + (i + 2);
    }
    SIMD_ASSERT_VEC_EQ(d, lanes, fadd(v1, k1, v2));
    SIMD_ASSERT_VEC_EQ(d, lanes, fadd(v2, k1, v1));
    SIMD_ASSERT_VEC_EQ(d, lanes, fsub(v1, k1, neg(v2)));
    SIMD_ASSERT_VEC_EQ(d, lanes, fnadd(neg(v2), k1, v1));
    SIMD_ASSERT_VEC_EQ(d, k1, fsub(v2, k1, v1));
    SIMD_ASSERT_VEC_EQ(d, k1, fnadd(v1, k1, v2));

    for (size_t i = 0; i < d.N; ++i) {
      lanes[i] = (i + 1) * (i + 2);
    }
    SIMD_ASSERT_VEC_EQ(d, lanes, mul_add(v2, v1, k0));
    SIMD_ASSERT_VEC_EQ(d, lanes, mul_add(v1, v2, k0));
    SIMD_ASSERT_VEC_EQ(d, lanes, nmul_add(neg(v2), v1, k0));
    SIMD_ASSERT_VEC_EQ(d, lanes, nmul_add(v1, neg(v2), k0));

    for (size_t i = 0; i < d.N; ++i) {
      lanes[i] = (i + 2) * (i + 2) + (i + 1);
    }
    SIMD_ASSERT_VEC_EQ(d, lanes, mul_add(v2, v2, v1));
    SIMD_ASSERT_VEC_EQ(d, lanes, nmul_add(neg(v2), v2, v1));

    for (size_t i = 0; i < d.N; ++i) {
      lanes[i] = -T(i + 2) * (i + 2) + (1 + i);
    }
    SIMD_ASSERT_VEC_EQ(d, lanes, nmul_add(v2, v2, v1));

    SIMD_ASSERT_VEC_EQ(d, k0, ext::mul_subtract(k0, k0, k0));
    SIMD_ASSERT_VEC_EQ(d, k0, ext::nmul_subtract(k0, k0, k0));

    for (size_t i = 0; i < d.N; ++i) {
      lanes[i] = -T(i + 2);
    }
    SIMD_ASSERT_VEC_EQ(d, lanes, ext::mul_subtract(k0, v1, v2));
    SIMD_ASSERT_VEC_EQ(d, lanes, ext::mul_subtract(v1, k0, v2));
    SIMD_ASSERT_VEC_EQ(d, lanes, ext::nmul_subtract(neg(k0), v1, v2));
    SIMD_ASSERT_VEC_EQ(d, lanes, ext::nmul_subtract(v1, neg(k0), v2));

    for (size_t i = 0; i < d.N; ++i) {
      lanes[i] = (i + 1) * (i + 2);
    }
    SIMD_ASSERT_VEC_EQ(d, lanes, ext::mul_subtract(v1, v2, k0));
    SIMD_ASSERT_VEC_EQ(d, lanes, ext::mul_subtract(v2, v1, k0));
    SIMD_ASSERT_VEC_EQ(d, lanes, ext::nmul_subtract(neg(v1), v2, k0));
    SIMD_ASSERT_VEC_EQ(d, lanes, ext::nmul_subtract(v2, neg(v1), k0));

    for (size_t i = 0; i < d.N; ++i) {
      lanes[i] = (i + 2) * (i + 2) - (1 + i);
    }
    SIMD_ASSERT_VEC_EQ(d, lanes, ext::mul_subtract(v2, v2, v1));
    SIMD_ASSERT_VEC_EQ(d, lanes, ext::nmul_subtract(neg(v2), v2, v1));
  }
};

struct TestSquareRoot {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto vi = iota(d, 0);
    SIMD_ASSERT_VEC_EQ(d, vi, sqrt(vi * vi));
  }
};

struct TestReciprocalSquareRoot {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v = set1(d, 123.0f);
    SIMD_ALIGN float lanes[d.N];
    store(approximate_reciprocal_sqrt(v), d, lanes);
    for (size_t i = 0; i < d.N; ++i) {
      float err = lanes[i] - 0.090166f;
      if (err < 0.0f) err = -err;
      SIMD_ASSERT_EQ(true, err < 1E-4f);
    }
  }
};

struct TestRound {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    // Integer positive
    {
      const auto v = iota(d, 4.0);
      SIMD_ASSERT_VEC_EQ(d, v, ceil(v));
      SIMD_ASSERT_VEC_EQ(d, v, floor(v));
      SIMD_ASSERT_VEC_EQ(d, v, round(v));
      SIMD_ASSERT_VEC_EQ(d, v, trunc(v));
    }

    // Integer negative
    {
      const auto v = iota(d, T(-32.0));
      SIMD_ASSERT_VEC_EQ(d, v, ceil(v));
      SIMD_ASSERT_VEC_EQ(d, v, floor(v));
      SIMD_ASSERT_VEC_EQ(d, v, round(v));
      SIMD_ASSERT_VEC_EQ(d, v, trunc(v));
    }

    // Huge positive
    {
      const auto v = set1(d, T(1E15));
      SIMD_ASSERT_VEC_EQ(d, v, ceil(v));
      SIMD_ASSERT_VEC_EQ(d, v, floor(v));
    }

    // Huge negative
    {
      const auto v = set1(d, T(-1E15));
      SIMD_ASSERT_VEC_EQ(d, v, ceil(v));
      SIMD_ASSERT_VEC_EQ(d, v, floor(v));
    }

    // Above positive
    {
      const auto v = iota(d, T(2.0001));
      const auto v3 = iota(d, T(3));
      const auto v2 = iota(d, T(2));
      SIMD_ASSERT_VEC_EQ(d, v3, ceil(v));
      SIMD_ASSERT_VEC_EQ(d, v2, floor(v));
      SIMD_ASSERT_VEC_EQ(d, v2, round(v));
      SIMD_ASSERT_VEC_EQ(d, v2, trunc(v));
    }

    // Below positive
    {
      const auto v = iota(d, T(3.9999));
      const auto v4 = iota(d, T(4));
      const auto v3 = iota(d, T(3));
      SIMD_ASSERT_VEC_EQ(d, v4, ceil(v));
      SIMD_ASSERT_VEC_EQ(d, v3, floor(v));
      SIMD_ASSERT_VEC_EQ(d, v4, round(v));
      SIMD_ASSERT_VEC_EQ(d, v3, trunc(v));
    }

    // Above negative
    {
      // WARNING: using iota => ensure negative value is low enough that
      // even 16 lanes remain negative, otherwise trunc will behave differently
      // for positive/negative values.
      const auto v = iota(d, T(-19.9999));
      const auto v3 = iota(d, T(-19));
      const auto v4 = iota(d, T(-20));
      SIMD_ASSERT_VEC_EQ(d, v3, ceil(v));
      SIMD_ASSERT_VEC_EQ(d, v4, floor(v));
      SIMD_ASSERT_VEC_EQ(d, v4, round(v));
      SIMD_ASSERT_VEC_EQ(d, v3, trunc(v));
    }

    // Below negative
    {
      const auto v = iota(d, T(-18.0001));
      const auto v2 = iota(d, T(-18));
      const auto v3 = iota(d, T(-19));
      SIMD_ASSERT_VEC_EQ(d, v2, ceil(v));
      SIMD_ASSERT_VEC_EQ(d, v3, floor(v));
      SIMD_ASSERT_VEC_EQ(d, v2, round(v));
      SIMD_ASSERT_VEC_EQ(d, v2, trunc(v));
    }
  }
};

struct TestIntFromFloat {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const SIMD_FULL(float) df;

    // Integer positive
    SIMD_ASSERT_VEC_EQ(d, iota(d, 4), convert_to(d, iota(df, 4.0f)));
    SIMD_ASSERT_VEC_EQ(d, iota(d, 4), nearest_int(iota(df, 4.0f)));

    // Integer negative
    SIMD_ASSERT_VEC_EQ(d, iota(d, -32), convert_to(d, iota(df, -32.0f)));
    SIMD_ASSERT_VEC_EQ(d, iota(d, -32), nearest_int(iota(df, -32.0f)));

    // Above positive
    SIMD_ASSERT_VEC_EQ(d, iota(d, 2), convert_to(d, iota(df, 2.001f)));
    SIMD_ASSERT_VEC_EQ(d, iota(d, 2), nearest_int(iota(df, 2.001f)));

    // Below positive
    SIMD_ASSERT_VEC_EQ(d, iota(d, 3), convert_to(d, iota(df, 3.9999f)));
    SIMD_ASSERT_VEC_EQ(d, iota(d, 4), nearest_int(iota(df, 3.9999f)));

    // Above negative
    SIMD_ASSERT_VEC_EQ(d, iota(d, -23), convert_to(d, iota(df, -23.9999f)));
    SIMD_ASSERT_VEC_EQ(d, iota(d, -24), nearest_int(iota(df, -23.9999f)));

    // Below negative
    SIMD_ASSERT_VEC_EQ(d, iota(d, -24), convert_to(d, iota(df, -24.001f)));
    SIMD_ASSERT_VEC_EQ(d, iota(d, -24), nearest_int(iota(df, -24.001f)));
  }
};

struct TestFloatFromInt {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const SIMD_FULL(int32_t) di;

    // Integer positive
    SIMD_ASSERT_VEC_EQ(d, iota(d, 4.0f), convert_to(d, iota(di, 4)));

    // Integer negative
    SIMD_ASSERT_VEC_EQ(d, iota(d, -32.0f), convert_to(d, iota(di, -32)));

    // Above positive
    SIMD_ASSERT_VEC_EQ(d, iota(d, 2.0f), convert_to(d, iota(di, 2)));

    // Below positive
    SIMD_ASSERT_VEC_EQ(d, iota(d, 4.0f), convert_to(d, iota(di, 4)));

    // Above negative
    SIMD_ASSERT_VEC_EQ(d, iota(d, -4.0f), convert_to(d, iota(di, -4)));

    // Below negative
    SIMD_ASSERT_VEC_EQ(d, iota(d, -2.0f), convert_to(d, iota(di, -2)));
  }
};

struct TestSumsOfU8 {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
#if SIMD_BITS != 0
    const SIMD_FULL(uint8_t) d8;
    SIMD_ALIGN uint8_t in_bytes[d8.N];
    uint64_t sums[d.N] = {0};
    for (size_t i = 0; i < d8.N; ++i) {
      const size_t group = i / 8;
      in_bytes[i] = 2 * i + 1;
      sums[group] += in_bytes[i];
    }
    const auto v = load(d8, in_bytes);
    SIMD_ASSERT_VEC_EQ(d, sums, ext::sums_of_u8x8(v));
#endif
  }
};

struct TestMinPos {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
#if SIMD_BITS == 128
    const SIMD_FULL(uint16_t) d16;
    SIMD_ALIGN uint16_t in_bytes[d16.N];

    // Check for the minimum value in each position.
    for (uint16_t ret_pos = 0; ret_pos < d16.N; ret_pos++) {
      for (size_t i = 0; i < d16.N; ++i) {
        // The minimum value is when i == 0, since i < d16.N and 3 is coprime
        // with d16.N, therefore no other value of i gives i * 3 % d16.N == 0.
        in_bytes[(ret_pos + i) % d16.N] = 777U + i * 3 % d16.N;
      }

      const auto v = load(d16, in_bytes);
      auto ext_minpos_v = ext::minpos(v);

      EXPECT_EQ(777U, get_lane(ext_minpos_v)) << "Where ret_pos=" << ret_pos;
      ext_minpos_v = shift_right_lanes<1>(ext_minpos_v);
      EXPECT_EQ(ret_pos, get_lane(ext_minpos_v));
    }
#endif
  }
};

struct TestHorzSumT {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    SIMD_ALIGN T in_lanes[d.N];
    double sum = 0.0;
    for (size_t i = 0; i < d.N; ++i) {
      in_lanes[i] = 1u << i;
      sum += in_lanes[i];
    }
    const auto v = load(d, in_lanes);
    const auto expected = set1(d, T(sum));
    SIMD_ASSERT_VEC_EQ(d, expected, ext::sum_of_lanes(v));
  }
};

SIMD_ATTR void TestHorzSum() {
  // No u16.
  Call<TestHorzSumT, uint32_t>();
  Call<TestHorzSumT, uint64_t>();

  // No i8/i16.
  Call<TestHorzSumT, int32_t>();
  Call<TestHorzSumT, int64_t>();

  Call<TestHorzSumT, float>();
  Call<TestHorzSumT, double>();
}

SIMD_ATTR void TestArithmetic() {
  ForeachLaneType<TestPlusMinus>();
  TestSaturatingArithmetic();

  TestShifts();
  TestMinMax();
  TestAverage();
  TestAbs();
  TestMul();
  TestMulHigh();
  TestMulEven();

  ForeachFloatLaneType<TestMulAdd>();
  ForeachFloatLaneType<TestSquareRoot>();
  Call<TestReciprocalSquareRoot, float>();
  ForeachFloatLaneType<TestRound>();
  Call<TestIntFromFloat, int32_t>();
  Call<TestFloatFromInt, float>();

  Call<TestSumsOfU8, uint64_t>();
  Call<TestMinPos, uint16_t>();
  TestHorzSum();
}

}  // namespace
}  // namespace SIMD_NAMESPACE  // NOLINT(google-readability-namespace-comments)
}  // namespace jxl

// Instantiate for the current target.
template <>
void ArithmeticTest::operator()<jxl::SIMD_TARGET>() {
  jxl::SIMD_NAMESPACE::TestArithmetic();
}
