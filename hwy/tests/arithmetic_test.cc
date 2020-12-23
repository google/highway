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
#define HWY_TARGET_INCLUDE "tests/arithmetic_test.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

struct TestPlusMinus {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v2 = Iota(d, T(2));
    const auto v3 = Iota(d, T(3));
    const auto v4 = Iota(d, T(4));

    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);
    for (size_t i = 0; i < N; ++i) {
      lanes[i] = static_cast<T>((2 + i) + (3 + i));
    }
    HWY_ASSERT_VEC_EQ(d, Load(d, lanes.get()), v2 + v3);
    HWY_ASSERT_VEC_EQ(d, v3, (v2 + v3) - v2);

    for (size_t i = 0; i < N; ++i) {
      lanes[i] = static_cast<T>((2 + i) + (4 + i));
    }
    auto sum = v2;
    sum += v4;  // sum == 6,8..
    HWY_ASSERT_VEC_EQ(d, Load(d, lanes.get()), sum);

    sum -= v4;
    HWY_ASSERT_VEC_EQ(d, v2, sum);
  }
};

HWY_NOINLINE void TestAllPlusMinus() {
  ForAllTypes(ForPartialVectors<TestPlusMinus>());
}

struct TestUnsignedSaturatingArithmetic {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
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
};

struct TestSignedSaturatingArithmetic {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    const auto vi = Iota(d, 1);
    const auto vpm = Set(d, LimitsMax<T>());
    const auto vn = Iota(d, -T(Lanes(d)));
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
};

HWY_NOINLINE void TestAllSaturatingArithmetic() {
  const ForPartialVectors<TestUnsignedSaturatingArithmetic> test_unsigned;
  test_unsigned(uint8_t());
  test_unsigned(uint16_t());

  const ForPartialVectors<TestSignedSaturatingArithmetic> test_signed;
  test_signed(int8_t());
  test_signed(int16_t());
}

struct TestAverage {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
// Avoid "Do not know how to split the result of this operator"
#if !defined(HWY_DISABLE_BROKEN_AVX3_TESTS) || HWY_TARGET != HWY_AVX3
    const auto v0 = Zero(d);
    const auto v1 = Set(d, T(1));
    const auto v2 = Set(d, T(2));

    HWY_ASSERT_VEC_EQ(d, v0, AverageRound(v0, v0));
    HWY_ASSERT_VEC_EQ(d, v1, AverageRound(v0, v1));
    HWY_ASSERT_VEC_EQ(d, v1, AverageRound(v1, v1));
    HWY_ASSERT_VEC_EQ(d, v2, AverageRound(v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2, AverageRound(v2, v2));
#else  // HWY_DISABLE_BROKEN_AVX3_TESTS
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllAverage() {
  const ForPartialVectors<TestAverage> test;
  test(uint8_t());
  test(uint16_t());
}

struct TestAbs {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
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
};

struct TestFloatAbs {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    const auto vp1 = Set(d, T(1));
    const auto vn1 = Set(d, T(-1));
    const auto vp2 = Set(d, T(0.01));
    const auto vn2 = Set(d, T(-0.01));

    HWY_ASSERT_VEC_EQ(d, v0, Abs(v0));
    HWY_ASSERT_VEC_EQ(d, vp1, Abs(vp1));
    HWY_ASSERT_VEC_EQ(d, vp1, Abs(vn1));
    HWY_ASSERT_VEC_EQ(d, vp2, Abs(vp2));
    HWY_ASSERT_VEC_EQ(d, vp2, Abs(vn2));
  }
};

HWY_NOINLINE void TestAllAbs() {
  const ForPartialVectors<TestAbs> test;
  test(int8_t());
  test(int16_t());
  test(int32_t());

  const ForPartialVectors<TestFloatAbs> test_float;
  test_float(float());
#if HWY_CAP_FLOAT64
  test_float(double());
#endif
}

struct TestUnsignedShifts {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
// Avoid "Do not know how to split the result of this operator"
#if !defined(HWY_DISABLE_BROKEN_AVX3_TESTS) || HWY_TARGET != HWY_AVX3
    constexpr int kSign = (sizeof(T) * 8) - 1;
    const auto v0 = Zero(d);
    const auto vi = Iota(d, 0);
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);

    // Shifting out of right side => zero
    HWY_ASSERT_VEC_EQ(d, v0, ShiftRight<7>(vi));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, v0, ShiftRightSame(vi, 7));
#endif

    // Simple left shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(i << 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), ShiftLeft<1>(vi));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, expected.get(), ShiftLeftSame(vi, 1));
#endif

    // Simple right shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(i >> 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), ShiftRight<1>(vi));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, expected.get(), ShiftRightSame(vi, 1));
#endif

    // Verify truncation for left-shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = (T(i) << kSign) & ~T(0);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), ShiftLeft<kSign>(vi));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, expected.get(), ShiftLeftSame(vi, kSign));
#endif

#else  // HWY_DISABLE_BROKEN_AVX3_TESTS
    (void)d;
#endif
  }
};

struct TestSignedShifts {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
// Avoid "Do not know how to split the result of this operator"
#if !defined(HWY_DISABLE_BROKEN_AVX3_TESTS) || HWY_TARGET != HWY_AVX3
    using TU = MakeUnsigned<T>;
    const auto v0 = Zero(d);
    const auto vi = Iota(d, 0);
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);

    // Shifting out of right side => zero
    HWY_ASSERT_VEC_EQ(d, v0, ShiftRight<7>(vi));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, v0, ShiftRightSame(vi, 7));
#endif

    // Simple left shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(i << 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), ShiftLeft<1>(vi));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, expected.get(), ShiftLeftSame(vi, 1));
#endif

    // Simple right shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(i >> 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), ShiftRight<1>(vi));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, expected.get(), ShiftRightSame(vi, 1));
#endif

    // Sign extension
    constexpr T min = LimitsMin<T>();
    const auto vn = Iota(d, min);
    for (int i = 0; i < static_cast<int>(N); ++i) {
      // We want a right-shift here, which is undefined behavior for negative
      // numbers. Since we want (-1)>>1 to be -1, we need to adjust rounding if
      // minT is odd and negative.
      T minT = static_cast<T>(min + i);
      expected[i] = T(minT / 2 + (minT < 0 ? minT % 2 : 0));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), ShiftRight<1>(vn));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, expected.get(), ShiftRightSame(vn, 1));
#endif

    // Shifting negative left
    for (int i = 0; i < static_cast<int>(N); ++i) {
      expected[i] = T(TU(min + i) << 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), ShiftLeft<1>(vn));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, expected.get(), ShiftLeftSame(vn, 1));
#endif

#else  // HWY_DISABLE_BROKEN_AVX3_TESTS
    (void)d;
#endif
  }
};

struct TestUnsignedVarShifts {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
// Avoid "Do not know how to split the result of this operator"
#if !defined(HWY_DISABLE_BROKEN_AVX3_TESTS) || HWY_TARGET != HWY_AVX3
    constexpr int kSign = (sizeof(T) * 8) - 1;
    const auto v0 = Zero(d);
    const auto v1 = Set(d, 1);
    const auto vi = Iota(d, 0);
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);

    // Shifting out of right side => zero
    HWY_ASSERT_VEC_EQ(d, v0, vi >> Set(d, 7));

    // Simple left shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(i << 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), vi << Set(d, 1));

    // Simple right shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(i >> 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), vi >> Set(d, 1));

    // Verify truncation for left-shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = (T(i) << kSign) & ~T(0);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), vi << Set(d, kSign));

    // Verify variable left shift (assumes < 32 lanes)
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(1) << i;
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), v1 << vi);
#else  // HWY_DISABLE_BROKEN_AVX3_TESTS
    (void)d;
#endif
  }
};

struct TestSignedVarLeftShifts {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Avoid "Do not know how to split the result of this operator"
#if !defined(HWY_DISABLE_BROKEN_AVX3_TESTS) || HWY_TARGET != HWY_AVX3
    const auto v1 = Set(d, 1);
    const auto vi = Iota(d, 0);

    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);

    // Simple left shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(i << 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), vi << v1);

    // Shifting negative numbers left
    constexpr T min = LimitsMin<T>();
    const auto vn = Iota(d, min);
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T((min + i) << 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), vn << v1);

    // Differing shift counts (assumes < 32 lanes)
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(1) << i;
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), v1 << vi);
#else  // HWY_DISABLE_BROKEN_AVX3_TESTS
    (void)d;
#endif
  }
};

struct TestSignedVarRightShifts {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Avoid "Do not know how to split the result of this operator"
#if !defined(HWY_DISABLE_BROKEN_AVX3_TESTS) || HWY_TARGET != HWY_AVX3
    const auto v0 = Zero(d);
    const auto vi = Iota(d, 0);
    const auto vmax = Set(d, LimitsMax<T>());
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);

    // Shifting out of right side => zero
    HWY_ASSERT_VEC_EQ(d, v0, vi >> Set(d, 7));

    // Simple right shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(i >> 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), vi >> Set(d, 1));

    // Sign extension
    constexpr T min = LimitsMin<T>();
    const auto vn = Iota(d, min);
    for (size_t i = 0; i < N; ++i) {
      // We want a right-shift here, which is undefined behavior for negative
      // numbers. Since we want (-1)>>1 to be -1, we need to adjust rounding if
      // minT is odd and negative.
      T minT = min + i;
      expected[i] = T(minT / 2 + (minT < 0 ? minT % 2 : 0));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), vn >> Set(d, 1));

    // Differing shift counts (assumes < 32 lanes)
    for (size_t i = 0; i < N; ++i) {
      expected[i] = LimitsMax<T>() >> i;
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), vmax >> vi);
#else  // HWY_DISABLE_BROKEN_AVX3_TESTS
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllShifts() {
  const ForPartialVectors<TestUnsignedShifts> test_unsigned;
  // No u8.
  test_unsigned(uint16_t());
  test_unsigned(uint32_t());
#if HWY_CAP_INTEGER64
  test_unsigned(uint64_t());
#endif

  const ForPartialVectors<TestSignedShifts> test_signed;
  // No i8.
  test_signed(int16_t());
  test_signed(int32_t());
  // No i64/f32/f64.

  ForPartialVectors<TestUnsignedVarShifts, 1, 1,
                    HWY_VARIABLE_SHIFT_LANES(uint32_t)>()(uint32_t());
#if HWY_CAP_INTEGER64
  ForPartialVectors<TestUnsignedVarShifts, 1, 1,
                    HWY_VARIABLE_SHIFT_LANES(uint64_t)>()(uint64_t());
#endif

  ForPartialVectors<TestSignedVarLeftShifts, 1, 1,
                    HWY_VARIABLE_SHIFT_LANES(int32_t)>()(int32_t());
#if HWY_CAP_INTEGER64
  ForPartialVectors<TestSignedVarLeftShifts, 1, 1,
                    HWY_VARIABLE_SHIFT_LANES(int64_t)>()(int64_t());
#endif

  ForPartialVectors<TestSignedVarRightShifts, 1, 1,
                    HWY_VARIABLE_SHIFT_LANES(int32_t)>()(int32_t());
  // No i64 (right-shift).
}

struct TestUnsignedMinMax {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    const auto v1 = Iota(d, 1);
    const auto v2 = Iota(d, 2);
    const auto v_max = Iota(d, LimitsMax<T>() - Lanes(d) + 1);
    HWY_ASSERT_VEC_EQ(d, v1, Min(v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2, Max(v1, v2));
    HWY_ASSERT_VEC_EQ(d, v0, Min(v1, v0));
    HWY_ASSERT_VEC_EQ(d, v1, Max(v1, v0));
    HWY_ASSERT_VEC_EQ(d, v1, Min(v1, v_max));
    HWY_ASSERT_VEC_EQ(d, v_max, Max(v1, v_max));
    HWY_ASSERT_VEC_EQ(d, v0, Min(v0, v_max));
    HWY_ASSERT_VEC_EQ(d, v_max, Max(v0, v_max));
  }
};

struct TestSignedMinMax {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v1 = Iota(d, 1);
    const auto v2 = Iota(d, 2);
    const auto v_neg = Iota(d, -T(Lanes(d)));
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
};

struct TestFloatMinMax {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Avoid "Do not know how to split the result of this operator"
#if !defined(HWY_DISABLE_BROKEN_AVX3_TESTS) || HWY_TARGET != HWY_AVX3
    const auto v1 = Iota(d, 1);
    const auto v2 = Iota(d, 2);
    const auto v_neg = Iota(d, -T(Lanes(d)));
    HWY_ASSERT_VEC_EQ(d, v1, Min(v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2, Max(v1, v2));
    HWY_ASSERT_VEC_EQ(d, v_neg, Min(v1, v_neg));
    HWY_ASSERT_VEC_EQ(d, v1, Max(v1, v_neg));
#else  // HWY_DISABLE_BROKEN_AVX3_TESTS
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllMinMax() {
  const ForPartialVectors<TestUnsignedMinMax> test_unsigned;
  test_unsigned(uint8_t());
  test_unsigned(uint16_t());
  test_unsigned(uint32_t());
  // No u64.

  const ForPartialVectors<TestSignedMinMax> test_signed;
  test_signed(int8_t());
  test_signed(int16_t());
  test_signed(int32_t());
  // No i64.

  ForFloatTypes(ForPartialVectors<TestFloatMinMax>());
}

struct TestUnsignedMul {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    const auto v1 = Set(d, T(1));
    const auto vi = Iota(d, 1);
    const auto vj = Iota(d, 3);
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);

    HWY_ASSERT_VEC_EQ(d, v0, v0 * v0);
    HWY_ASSERT_VEC_EQ(d, v1, v1 * v1);
    HWY_ASSERT_VEC_EQ(d, vi, v1 * vi);
    HWY_ASSERT_VEC_EQ(d, vi, vi * v1);

    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>((1 + i) * (1 + i));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), vi * vi);

    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>((1 + i) * (3 + i));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), vi * vj);

    const T max = LimitsMax<T>();
    const auto vmax = Set(d, max);
    HWY_ASSERT_VEC_EQ(d, vmax, vmax * v1);
    HWY_ASSERT_VEC_EQ(d, vmax, v1 * vmax);

    const size_t bits = sizeof(T) * 8;
    const uint64_t mask = (1ull << bits) - 1;
    const T max2 = (uint64_t(max) * max) & mask;
    HWY_ASSERT_VEC_EQ(d, Set(d, max2), vmax * vmax);
  }
};

struct TestSignedMul {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);

    const auto v0 = Zero(d);
    const auto v1 = Set(d, T(1));
    const auto vi = Iota(d, 1);
    const auto vn = Iota(d, -T(N));
    HWY_ASSERT_VEC_EQ(d, v0, v0 * v0);
    HWY_ASSERT_VEC_EQ(d, v1, v1 * v1);
    HWY_ASSERT_VEC_EQ(d, vi, v1 * vi);
    HWY_ASSERT_VEC_EQ(d, vi, vi * v1);

    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>((1 + i) * (1 + i));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), vi * vi);

    for (int i = 0; i < static_cast<int>(N); ++i) {
      expected[i] = static_cast<T>((-T(N) + i) * (1 + i));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), vn * vi);
    HWY_ASSERT_VEC_EQ(d, expected.get(), vi * vn);
  }
};

HWY_NOINLINE void TestAllMul() {
  const ForPartialVectors<TestUnsignedMul> test_unsigned;
  // No u8.
  test_unsigned(uint16_t());
  test_unsigned(uint32_t());
  // No u64.

  const ForPartialVectors<TestSignedMul> test_signed;
  // No i8.
  test_signed(int16_t());
  test_signed(int32_t());
  // No i64.
}

template <typename Wide>
struct TestMulHigh {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Avoid "Do not know how to split the result of this operator"
#if !defined(HWY_DISABLE_BROKEN_AVX3_TESTS) || HWY_TARGET != HWY_AVX3
    const size_t N = Lanes(d);
    auto in_lanes = AllocateAligned<T>(N);
    auto expected_lanes = AllocateAligned<T>(N);

    const auto vi = Iota(d, 1);
    const auto vni = Iota(d, -T(N));

    const auto v0 = Zero(d);
    HWY_ASSERT_VEC_EQ(d, v0, MulHigh(v0, v0));
    HWY_ASSERT_VEC_EQ(d, v0, MulHigh(v0, vi));
    HWY_ASSERT_VEC_EQ(d, v0, MulHigh(vi, v0));

    // Large positive squared
    for (size_t i = 0; i < N; ++i) {
      in_lanes[i] = LimitsMax<T>() >> i;
      expected_lanes[i] = (Wide(in_lanes[i]) * in_lanes[i]) >> 16;
    }
    auto v = Load(d, in_lanes.get());
    HWY_ASSERT_VEC_EQ(d, expected_lanes.get(), MulHigh(v, v));

    // Large positive * small positive
    for (int i = 0; i < static_cast<int>(N); ++i) {
      expected_lanes[i] = static_cast<T>((Wide(in_lanes[i]) * T(1 + i)) >> 16);
    }
    HWY_ASSERT_VEC_EQ(d, expected_lanes.get(), MulHigh(v, vi));
    HWY_ASSERT_VEC_EQ(d, expected_lanes.get(), MulHigh(vi, v));

    // Large positive * small negative
    for (size_t i = 0; i < N; ++i) {
      expected_lanes[i] = (Wide(in_lanes[i]) * T(i - N)) >> 16;
    }
    HWY_ASSERT_VEC_EQ(d, expected_lanes.get(), MulHigh(v, vni));
    HWY_ASSERT_VEC_EQ(d, expected_lanes.get(), MulHigh(vni, v));
#else  // HWY_DISABLE_BROKEN_AVX3_TESTS
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllMulHigh() {
  ForPartialVectors<TestMulHigh<int32_t>>()(int16_t());
  ForPartialVectors<TestMulHigh<uint32_t>>()(uint16_t());
}

template <typename Wide>
struct TestMulEven {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    if (Lanes(d) == 1) return;
    static_assert(sizeof(Wide) == 2 * sizeof(T), "Expected double-width type");
    const Repartition<Wide, D> d2;
    const auto v0 = Zero(d);
    HWY_ASSERT_VEC_EQ(d2, Zero(d2), MulEven(v0, v0));

    // scalar has N=1 and we write to "lane 1" below, though it isn't used by
    // the actual MulEven.
    auto in_lanes = AllocateAligned<T>(HWY_MAX(Lanes(d), 2));
    auto expected = AllocateAligned<Wide>(Lanes(d2));
    for (size_t i = 0; i < Lanes(d); i += 2) {
      in_lanes[i + 0] = LimitsMax<T>() >> i;
      in_lanes[i + 1] = 1;  // will be overwritten with upper half of result
      expected[i / 2] = Wide(in_lanes[i + 0]) * in_lanes[i + 0];
    }

    const auto v = Load(d, in_lanes.get());
    HWY_ASSERT_VEC_EQ(d2, expected.get(), MulEven(v, v));
  }
};

HWY_NOINLINE void TestAllMulEven() {
  ForPartialVectors<TestMulEven<int64_t>>()(int32_t());
  ForPartialVectors<TestMulEven<uint64_t>>()(uint32_t());
}

struct TestMulAdd {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto k0 = Zero(d);
    const auto kNeg0 = Set(d, T(-0.0));
    const auto v1 = Iota(d, 1);
    const auto v2 = Iota(d, 2);
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT_VEC_EQ(d, k0, MulAdd(k0, k0, k0));
    HWY_ASSERT_VEC_EQ(d, v2, MulAdd(k0, v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2, MulAdd(v1, k0, v2));
    HWY_ASSERT_VEC_EQ(d, k0, NegMulAdd(k0, k0, k0));
    HWY_ASSERT_VEC_EQ(d, v2, NegMulAdd(k0, v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2, NegMulAdd(v1, k0, v2));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = (i + 1) * (i + 2);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulAdd(v2, v1, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulAdd(v1, v2, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulAdd(Neg(v2), v1, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulAdd(v1, Neg(v2), k0));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = (i + 2) * (i + 2) + (i + 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulAdd(v2, v2, v1));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulAdd(Neg(v2), v2, v1));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = -T(i + 2) * (i + 2) + (1 + i);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulAdd(v2, v2, v1));

    HWY_ASSERT_VEC_EQ(d, k0, MulSub(k0, k0, k0));
    HWY_ASSERT_VEC_EQ(d, kNeg0, NegMulSub(k0, k0, k0));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = -T(i + 2);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulSub(k0, v1, v2));
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulSub(v1, k0, v2));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulSub(Neg(k0), v1, v2));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulSub(v1, Neg(k0), v2));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = (i + 1) * (i + 2);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulSub(v1, v2, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulSub(v2, v1, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulSub(Neg(v1), v2, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulSub(v2, Neg(v1), k0));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = (i + 2) * (i + 2) - (1 + i);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulSub(v2, v2, v1));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulSub(Neg(v2), v2, v1));
  }
};

HWY_NOINLINE void TestAllMulAdd() {
  ForFloatTypes(ForPartialVectors<TestMulAdd>());
}

struct TestDiv {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v = Iota(d, T(-2));
    const auto v1 = Set(d, T(1));

    // Unchanged after division by 1.
    HWY_ASSERT_VEC_EQ(d, v, v / v1);

    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);
    for (size_t i = 0; i < N; ++i) {
      expected[i] = (T(i) - 2) / T(2);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), v / Set(d, T(2)));
  }
};

HWY_NOINLINE void TestAllDiv() { ForFloatTypes(ForPartialVectors<TestDiv>()); }

struct TestApproximateReciprocal {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v = Iota(d, T(-2));
    const auto nonzero = IfThenElse(v == Zero(d), Set(d, T(1)), v);
    const size_t N = Lanes(d);
    auto input = AllocateAligned<T>(N);
    Store(nonzero, d, input.get());

    auto actual = AllocateAligned<T>(N);
    Store(ApproximateReciprocal(nonzero), d, actual.get());

    double max_l1 = 0.0;
    for (size_t i = 0; i < N; ++i) {
      max_l1 = std::max<double>(max_l1, std::abs((1.0 / input[i]) - actual[i]));
    }
    const double max_rel = max_l1 / std::abs(1.0 / input[N - 1]);
    printf("max err %f\n", max_rel);

    HWY_ASSERT(max_rel < 0.002);
  }
};

HWY_NOINLINE void TestAllApproximateReciprocal() {
  ForPartialVectors<TestApproximateReciprocal>()(float());
}

struct TestSquareRoot {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto vi = Iota(d, 0);
    HWY_ASSERT_VEC_EQ(d, vi, Sqrt(vi * vi));
  }
};

HWY_NOINLINE void TestAllSquareRoot() {
  ForFloatTypes(ForPartialVectors<TestSquareRoot>());
}

struct TestReciprocalSquareRoot {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Avoid "Do not know how to split the result of this operator"
#if !defined(HWY_DISABLE_BROKEN_AVX3_TESTS) || HWY_TARGET != HWY_AVX3
    const auto v = Set(d, 123.0f);
    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);
    Store(ApproximateReciprocalSqrt(v), d, lanes.get());
    for (size_t i = 0; i < N; ++i) {
      float err = lanes[i] - 0.090166f;
      if (err < 0.0f) err = -err;
      HWY_ASSERT(err < 1E-4f);
    }
#else  // HWY_DISABLE_BROKEN_AVX3_TESTS
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllReciprocalSquareRoot() {
  ForPartialVectors<TestReciprocalSquareRoot>()(float());
}

struct TestRound {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Avoid "Do not know how to split the result of this operator"
#if !defined(HWY_DISABLE_BROKEN_AVX3_TESTS) || HWY_TARGET != HWY_AVX3
    // Numbers close to 0
    {
      const auto v = Set(d, T(0.4));
      const auto zero = Set(d, 0);
      const auto one = Set(d, 1);
      HWY_ASSERT_VEC_EQ(d, one, Ceil(v));
      HWY_ASSERT_VEC_EQ(d, zero, Floor(v));
      HWY_ASSERT_VEC_EQ(d, zero, Round(v));
      HWY_ASSERT_VEC_EQ(d, zero, Trunc(v));
    }
    {
      const auto v = Set(d, T(-0.4));
      const auto neg_zero = Set(d, T(-0.0));
      const auto neg_one = Set(d, -1);
      HWY_ASSERT_VEC_EQ(d, neg_zero, Ceil(v));
      HWY_ASSERT_VEC_EQ(d, neg_one, Floor(v));
      HWY_ASSERT_VEC_EQ(d, neg_zero, Round(v));
      HWY_ASSERT_VEC_EQ(d, neg_zero, Trunc(v));
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
#else  // HWY_DISABLE_BROKEN_AVX3_TESTS
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllRound() {
  ForFloatTypes(ForPartialVectors<TestRound>());
}

struct TestSumsOfU8 {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Avoid "Do not know how to split the result of this operator"
#if (!defined(HWY_DISABLE_BROKEN_AVX3_TESTS) || HWY_TARGET != HWY_AVX3) && \
    HWY_TARGET != HWY_SCALAR && HWY_CAP_INTEGER64
    const size_t N = Lanes(d);
    auto in_bytes = AllocateAligned<uint8_t>(N * sizeof(uint64_t));
    auto sums = AllocateAligned<T>(N);
    std::fill(sums.get(), sums.get() + N, 0);
    for (size_t i = 0; i < N * sizeof(uint64_t); ++i) {
      const size_t group = i / sizeof(uint64_t);
      in_bytes[i] = static_cast<uint8_t>(2 * i + 1);
      sums[group] += in_bytes[i];
    }
    const Repartition<uint8_t, D> du8;
    const auto v = Load(du8, in_bytes.get());
    HWY_ASSERT_VEC_EQ(d, sums.get(), SumsOfU8x8(v));
#else
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllSumsOfU8() {
#if HWY_CAP_INTEGER64
  ForPartialVectors<TestSumsOfU8>()(uint64_t());
#endif
}

struct TestSumOfLanes {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto in_lanes = AllocateAligned<T>(N);
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
      in_lanes[i] = static_cast<T>(1u << i);
      sum += static_cast<double>(in_lanes[i]);
    }
    const auto v = Load(d, in_lanes.get());
    const auto expected = Set(d, T(sum));
    HWY_ASSERT_VEC_EQ(d, expected, SumOfLanes(v));
  }
};

HWY_NOINLINE void TestAllSumOfLanes() {
  // Only full vectors because lanes in partial vectors are undefined.
  const ForFullVectors<TestSumOfLanes> sum;

  // No u8/u16/i8/i16.
  sum(uint32_t());
  sum(int32_t());

#if HWY_CAP_INTEGER64
  sum(uint64_t());
  sum(int64_t());
#endif

  ForFloatTypes(sum);
}

struct TestMinOfLanes {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Avoid "Do not know how to split the result of this operator"
#if !defined(HWY_DISABLE_BROKEN_AVX3_TESTS) || HWY_TARGET != HWY_AVX3
    const size_t N = Lanes(d);
    auto in_lanes = AllocateAligned<T>(N);

    T min = LimitsMax<T>();
    for (size_t i = 0; i < N; ++i) {
      in_lanes[i] = static_cast<T>(1u << i);
      min = std::min(min, in_lanes[i]);
    }
    const auto v = Load(d, in_lanes.get());
    const auto expected = Set(d, min);
    HWY_ASSERT_VEC_EQ(d, expected, MinOfLanes(v));
#else  // HWY_DISABLE_BROKEN_AVX3_TESTS
    (void)d;
#endif
  }
};

struct TestMaxOfLanes {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Avoid "Do not know how to split the result of this operator"
#if !defined(HWY_DISABLE_BROKEN_AVX3_TESTS) || HWY_TARGET != HWY_AVX3
    const size_t N = Lanes(d);
    auto in_lanes = AllocateAligned<T>(N);

    T max = LimitsMin<T>();
    for (size_t i = 0; i < N; ++i) {
      in_lanes[i] = static_cast<T>(1u << i);
      max = std::max(max, in_lanes[i]);
    }
    const auto v = Load(d, in_lanes.get());
    const auto expected = Set(d, max);
    HWY_ASSERT_VEC_EQ(d, expected, MaxOfLanes(v));
#else  // HWY_DISABLE_BROKEN_AVX3_TESTS
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllMinMaxOfLanes() {
  // Only full vectors because lanes in partial vectors are undefined.
  const ForFullVectors<TestMinOfLanes> min;
  const ForFullVectors<TestMaxOfLanes> max;

  // No u8/u16/i8/i16.
  min(uint32_t());
  max(uint32_t());
  min(int32_t());
  max(int32_t());

#if HWY_CAP_INTEGER64 && HWY_MINMAX64_LANES > 1
  min(uint64_t());
  max(uint64_t());
  min(int64_t());
  max(int64_t());
#endif

  ForFloatTypes(min);
  ForFloatTypes(max);
}

struct TestAbsDiff {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto in_lanes_a = AllocateAligned<T>(N);
    auto in_lanes_b = AllocateAligned<T>(N);
    auto out_lanes = AllocateAligned<T>(N);
    for (size_t i = 0; i < N; ++i) {
      in_lanes_a[i] = (i ^ 1u) << i;
      in_lanes_b[i] = i << i;
      out_lanes[i] = fabsf(in_lanes_a[i] - in_lanes_b[i]);
    }
    const auto a = Load(d, in_lanes_a.get());
    const auto b = Load(d, in_lanes_b.get());
    const auto expected = Load(d, out_lanes.get());
    HWY_ASSERT_VEC_EQ(d, expected, AbsDiff(a, b));
    HWY_ASSERT_VEC_EQ(d, expected, AbsDiff(b, a));
  }
};

HWY_NOINLINE void TestAllAbsDiff() {
  ForPartialVectors<TestAbsDiff>()(float());
}

struct TestNeg {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    const auto vn = Set(d, T(-3));
    const auto vp = Set(d, T(3));
    HWY_ASSERT_VEC_EQ(d, v0, Neg(v0));
    HWY_ASSERT_VEC_EQ(d, vp, Neg(vn));
    HWY_ASSERT_VEC_EQ(d, vn, Neg(vp));
  }
};

HWY_NOINLINE void TestAllNeg() {
  ForSignedTypes(ForPartialVectors<TestNeg>());
  ForFloatTypes(ForPartialVectors<TestNeg>());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {

class HwyArithmeticTest : public hwy::TestWithParamTarget {};

HWY_TARGET_INSTANTIATE_TEST_SUITE_P(HwyArithmeticTest);

HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllPlusMinus);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllSaturatingArithmetic);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllShifts);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllMinMax);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllAverage);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllAbs);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllMul);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllMulHigh);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllMulEven);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllMulAdd);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllDiv);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllApproximateReciprocal);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllSquareRoot);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllReciprocalSquareRoot);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllSumsOfU8);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllSumOfLanes);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllMinMaxOfLanes);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllRound);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllAbsDiff);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllNeg);

}  // namespace hwy
#endif
