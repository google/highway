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

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/arithmetic_test.cc"
#include "hwy/foreach_target.h"
// ^ must come before highway.h and any *-inl.h.

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

    HWY_ALIGN T lanes[MaxLanes(d)] = {};  // Initialized for static analysis.
    const size_t N = Lanes(d);
    for (size_t i = 0; i < N; ++i) {
      lanes[i] = static_cast<T>((2 + i) + (3 + i));
    }
    HWY_ASSERT_VEC_EQ(d, Load(d, lanes), v2 + v3);
    HWY_ASSERT_VEC_EQ(d, v3, (v2 + v3) - v2);

    for (size_t i = 0; i < N; ++i) {
      lanes[i] = static_cast<T>((2 + i) + (4 + i));
    }
    auto sum = v2;
    sum += v4;  // sum == 6,8..
    HWY_ASSERT_VEC_EQ(d, Load(d, lanes), sum);

    sum -= v4;
    HWY_ASSERT_VEC_EQ(d, v2, sum);
  }
};

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

HWY_NOINLINE void TestSaturatingArithmetic() {
  const ForPartialVectors<TestUnsignedSaturatingArithmetic> test_unsigned;
  test_unsigned(uint8_t());
  test_unsigned(uint16_t());

  const ForPartialVectors<TestSignedSaturatingArithmetic> test_signed;
  test_signed(int8_t());
  test_signed(int16_t());
}

struct TestAverageT {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    const auto v1 = Set(d, T(1));
    const auto v2 = Set(d, T(2));

    HWY_ASSERT_VEC_EQ(d, v0, AverageRound(v0, v0));
    HWY_ASSERT_VEC_EQ(d, v1, AverageRound(v0, v1));
    HWY_ASSERT_VEC_EQ(d, v1, AverageRound(v1, v1));
    HWY_ASSERT_VEC_EQ(d, v2, AverageRound(v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2, AverageRound(v2, v2));
  }
};

HWY_NOINLINE void TestAverage() {
  const ForPartialVectors<TestAverageT> test;
  test(uint8_t());
  test(uint16_t());
}

struct TestAbsT {
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

struct TestFloatAbsT {
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

HWY_NOINLINE void TestAbs() {
  const ForPartialVectors<TestAbsT> test;
  test(int8_t());
  test(int16_t());
  test(int32_t());

  const ForPartialVectors<TestFloatAbsT> test_float;
  test_float(float());
#if HWY_CAP_FLOAT64
  test_float(double());
#endif
}

struct TestUnsignedShifts {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    constexpr int kSign = (sizeof(T) * 8) - 1;
    const auto v0 = Zero(d);
    const auto vi = Iota(d, 0);
    HWY_ALIGN T expected[MaxLanes(d)] = {};  // Initialized for static analysis.
    const size_t N = Lanes(d);

    // Shifting out of right side => zero
    HWY_ASSERT_VEC_EQ(d, v0, ShiftRight<7>(vi));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, v0, ShiftRightSame(vi, 7));
#endif

    // Simple left shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(i << 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected, ShiftLeft<1>(vi));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, expected, ShiftLeftSame(vi, 1));
#endif

    // Simple right shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(i >> 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected, ShiftRight<1>(vi));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, expected, ShiftRightSame(vi, 1));
#endif

    // Verify truncation for left-shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = (T(i) << kSign) & ~T(0);
    }
    HWY_ASSERT_VEC_EQ(d, expected, ShiftLeft<kSign>(vi));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, expected, ShiftLeftSame(vi, kSign));
#endif
  }
};

struct TestSignedShifts {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    using TU = MakeUnsigned<T>;
    const auto v0 = Zero(d);
    const auto vi = Iota(d, 0);
    HWY_ALIGN T expected[MaxLanes(d)] = {};  // Initialized for static analysis.
    const size_t N = Lanes(d);

    // Shifting out of right side => zero
    HWY_ASSERT_VEC_EQ(d, v0, ShiftRight<7>(vi));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, v0, ShiftRightSame(vi, 7));
#endif

    // Simple left shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(i << 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected, ShiftLeft<1>(vi));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, expected, ShiftLeftSame(vi, 1));
#endif

    // Simple right shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(i >> 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected, ShiftRight<1>(vi));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, expected, ShiftRightSame(vi, 1));
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
    HWY_ASSERT_VEC_EQ(d, expected, ShiftRight<1>(vn));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, expected, ShiftRightSame(vn, 1));
#endif

    // Shifting negative left
    for (int i = 0; i < static_cast<int>(N); ++i) {
      expected[i] = T(TU(min + i) << 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected, ShiftLeft<1>(vn));
#if HWY_VARIABLE_SHIFT_LANES == 1 || HWY_IDE
    HWY_ASSERT_VEC_EQ(d, expected, ShiftLeftSame(vn, 1));
#endif
  }
};

struct TestUnsignedVarShifts {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    constexpr int kSign = (sizeof(T) * 8) - 1;
    const auto v0 = Zero(d);
    const auto v1 = Set(d, 1);
    const auto vi = Iota(d, 0);
    HWY_ALIGN T expected[MaxLanes(d)];
    const size_t N = Lanes(d);

    // Shifting out of right side => zero
    HWY_ASSERT_VEC_EQ(d, v0, vi >> Set(d, 7));

    // Simple left shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(i << 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected, vi << Set(d, 1));

    // Simple right shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(i >> 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected, vi >> Set(d, 1));

    // Verify truncation for left-shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = (T(i) << kSign) & ~T(0);
    }
    HWY_ASSERT_VEC_EQ(d, expected, vi << Set(d, kSign));

    // Verify variable left shift (assumes < 32 lanes)
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(1) << i;
    }
    HWY_ASSERT_VEC_EQ(d, expected, v1 << vi);
  }
};

struct TestSignedVarLeftShifts {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v1 = Set(d, 1);
    const auto vi = Iota(d, 0);

    HWY_ALIGN T expected[MaxLanes(d)];
    const size_t N = Lanes(d);

    // Simple left shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(i << 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected, vi << v1);

    // Shifting negative numbers left
    constexpr T min = LimitsMin<T>();
    const auto vn = Iota(d, min);
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T((min + i) << 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected, vn << v1);

    // Differing shift counts (assumes < 32 lanes)
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(1) << i;
    }
    HWY_ASSERT_VEC_EQ(d, expected, v1 << vi);
  }
};

struct TestSignedVarRightShifts {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    const auto vi = Iota(d, 0);
    const auto vmax = Set(d, LimitsMax<T>());
    HWY_ALIGN T expected[MaxLanes(d)];
    const size_t N = Lanes(d);

    // Shifting out of right side => zero
    HWY_ASSERT_VEC_EQ(d, v0, vi >> Set(d, 7));

    // Simple right shift
    for (size_t i = 0; i < N; ++i) {
      expected[i] = T(i >> 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected, vi >> Set(d, 1));

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
    HWY_ASSERT_VEC_EQ(d, expected, vn >> Set(d, 1));

    // Differing shift counts (assumes < 32 lanes)
    for (size_t i = 0; i < N; ++i) {
      expected[i] = LimitsMax<T>() >> i;
    }
    HWY_ASSERT_VEC_EQ(d, expected, vmax >> vi);
  }
};

HWY_NOINLINE void TestShifts() {
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
    const auto v1 = Iota(d, 1);
    const auto v2 = Iota(d, 2);
    const auto v_neg = Iota(d, -T(Lanes(d)));
    HWY_ASSERT_VEC_EQ(d, v1, Min(v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2, Max(v1, v2));
    HWY_ASSERT_VEC_EQ(d, v_neg, Min(v1, v_neg));
    HWY_ASSERT_VEC_EQ(d, v1, Max(v1, v_neg));
  }
};

HWY_NOINLINE void TestMinMax() {
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
    T lanes[MaxLanes(d)] = {};  // Initialized for static analysis.
    const size_t N = Lanes(d);

    HWY_ASSERT_VEC_EQ(d, v0, v0 * v0);
    HWY_ASSERT_VEC_EQ(d, v1, v1 * v1);
    HWY_ASSERT_VEC_EQ(d, vi, v1 * vi);
    HWY_ASSERT_VEC_EQ(d, vi, vi * v1);

    for (size_t i = 0; i < N; ++i) {
      lanes[i] = static_cast<T>((1 + i) * (1 + i));
    }
    HWY_ASSERT_VEC_EQ(d, lanes, vi * vi);

    for (size_t i = 0; i < N; ++i) {
      lanes[i] = static_cast<T>((1 + i) * (3 + i));
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
};

struct TestSignedMul {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    T lanes[MaxLanes(d)] = {};  // Initialized for static analysis.
    const size_t N = Lanes(d);

    const auto v0 = Zero(d);
    const auto v1 = Set(d, T(1));
    const auto vi = Iota(d, 1);
    const auto vn = Iota(d, -T(N));
    HWY_ASSERT_VEC_EQ(d, v0, v0 * v0);
    HWY_ASSERT_VEC_EQ(d, v1, v1 * v1);
    HWY_ASSERT_VEC_EQ(d, vi, v1 * vi);
    HWY_ASSERT_VEC_EQ(d, vi, vi * v1);

    for (size_t i = 0; i < N; ++i) {
      lanes[i] = static_cast<T>((1 + i) * (1 + i));
    }
    HWY_ASSERT_VEC_EQ(d, lanes, vi * vi);

    for (int i = 0; i < static_cast<int>(N); ++i) {
      lanes[i] = static_cast<T>((-T(N) + i) * (1 + i));
    }
    HWY_ASSERT_VEC_EQ(d, lanes, vn * vi);
    HWY_ASSERT_VEC_EQ(d, lanes, vi * vn);
  }
};

HWY_NOINLINE void TestMul() {
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
struct TestMulHighT {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    HWY_ALIGN T in_lanes[MaxLanes(d)] = {};  // Initialized for static analysis.
    const size_t N = Lanes(d);

    HWY_ALIGN T
        expected_lanes[MaxLanes(d)] = {};  // Initialized for static analysis.
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
    auto v = Load(d, in_lanes);
    HWY_ASSERT_VEC_EQ(d, expected_lanes, MulHigh(v, v));

    // Large positive * small positive
    for (int i = 0; i < static_cast<int>(N); ++i) {
      expected_lanes[i] = static_cast<T>((Wide(in_lanes[i]) * T(1 + i)) >> 16);
    }
    HWY_ASSERT_VEC_EQ(d, expected_lanes, MulHigh(v, vi));
    HWY_ASSERT_VEC_EQ(d, expected_lanes, MulHigh(vi, v));

    // Large positive * small negative
    for (size_t i = 0; i < N; ++i) {
      expected_lanes[i] = (Wide(in_lanes[i]) * T(i - N)) >> 16;
    }
    HWY_ASSERT_VEC_EQ(d, expected_lanes, MulHigh(v, vni));
    HWY_ASSERT_VEC_EQ(d, expected_lanes, MulHigh(vni, v));
  }
};

HWY_NOINLINE void TestMulHigh() {
  ForPartialVectors<TestMulHighT<int32_t>>()(int16_t());
  ForPartialVectors<TestMulHighT<uint32_t>>()(uint16_t());
}

template <typename Wide>
struct TestMulEvenT {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    if (Lanes(d) == 1) return;
    const HWY_CAPPED(Wide, (MaxLanes(d) + 1) / 2) d2;
    const auto v0 = Zero(d);
    HWY_ASSERT_VEC_EQ(d2, Zero(d2), MulEven(v0, v0));

    // scalar has N=1 and we write to "lane 1" below, though it isn't used by
    // the actual MulEven.
    HWY_ALIGN T in_lanes[HWY_MAX(MaxLanes(d), 2)];
    HWY_ALIGN Wide expected[MaxLanes(d2)];
    for (size_t i = 0; i < Lanes(d); i += 2) {
      in_lanes[i + 0] = LimitsMax<T>() >> i;
      in_lanes[i + 1] = 1;  // will be overwritten with upper half of result
      expected[i / 2] = Wide(in_lanes[i + 0]) * in_lanes[i + 0];
    }

    const auto v = Load(d, in_lanes);
    HWY_ASSERT_VEC_EQ(d2, expected, MulEven(v, v));
  }
};

HWY_NOINLINE void TestMulEven() {
  ForPartialVectors<TestMulEvenT<int64_t>>()(int32_t());
  ForPartialVectors<TestMulEvenT<uint64_t>>()(uint32_t());
}

struct TestMulAdd {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto k0 = Zero(d);
    const auto kNeg0 = Set(d, T(-0.0));
    const auto v1 = Iota(d, 1);
    const auto v2 = Iota(d, 2);
    T lanes[MaxLanes(d)];
    const size_t N = Lanes(d);
    HWY_ASSERT_VEC_EQ(d, k0, MulAdd(k0, k0, k0));
    HWY_ASSERT_VEC_EQ(d, v2, MulAdd(k0, v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2, MulAdd(v1, k0, v2));
    HWY_ASSERT_VEC_EQ(d, k0, NegMulAdd(k0, k0, k0));
    HWY_ASSERT_VEC_EQ(d, v2, NegMulAdd(k0, v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2, NegMulAdd(v1, k0, v2));

    for (size_t i = 0; i < N; ++i) {
      lanes[i] = (i + 1) * (i + 2);
    }
    HWY_ASSERT_VEC_EQ(d, lanes, MulAdd(v2, v1, k0));
    HWY_ASSERT_VEC_EQ(d, lanes, MulAdd(v1, v2, k0));
    HWY_ASSERT_VEC_EQ(d, lanes, NegMulAdd(Neg(v2), v1, k0));
    HWY_ASSERT_VEC_EQ(d, lanes, NegMulAdd(v1, Neg(v2), k0));

    for (size_t i = 0; i < N; ++i) {
      lanes[i] = (i + 2) * (i + 2) + (i + 1);
    }
    HWY_ASSERT_VEC_EQ(d, lanes, MulAdd(v2, v2, v1));
    HWY_ASSERT_VEC_EQ(d, lanes, NegMulAdd(Neg(v2), v2, v1));

    for (size_t i = 0; i < N; ++i) {
      lanes[i] = -T(i + 2) * (i + 2) + (1 + i);
    }
    HWY_ASSERT_VEC_EQ(d, lanes, NegMulAdd(v2, v2, v1));

    HWY_ASSERT_VEC_EQ(d, k0, MulSub(k0, k0, k0));
    HWY_ASSERT_VEC_EQ(d, kNeg0, NegMulSub(k0, k0, k0));

    for (size_t i = 0; i < N; ++i) {
      lanes[i] = -T(i + 2);
    }
    HWY_ASSERT_VEC_EQ(d, lanes, MulSub(k0, v1, v2));
    HWY_ASSERT_VEC_EQ(d, lanes, MulSub(v1, k0, v2));
    HWY_ASSERT_VEC_EQ(d, lanes, NegMulSub(Neg(k0), v1, v2));
    HWY_ASSERT_VEC_EQ(d, lanes, NegMulSub(v1, Neg(k0), v2));

    for (size_t i = 0; i < N; ++i) {
      lanes[i] = (i + 1) * (i + 2);
    }
    HWY_ASSERT_VEC_EQ(d, lanes, MulSub(v1, v2, k0));
    HWY_ASSERT_VEC_EQ(d, lanes, MulSub(v2, v1, k0));
    HWY_ASSERT_VEC_EQ(d, lanes, NegMulSub(Neg(v1), v2, k0));
    HWY_ASSERT_VEC_EQ(d, lanes, NegMulSub(v2, Neg(v1), k0));

    for (size_t i = 0; i < N; ++i) {
      lanes[i] = (i + 2) * (i + 2) - (1 + i);
    }
    HWY_ASSERT_VEC_EQ(d, lanes, MulSub(v2, v2, v1));
    HWY_ASSERT_VEC_EQ(d, lanes, NegMulSub(Neg(v2), v2, v1));
  }
};

struct TestSquareRoot {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto vi = Iota(d, 0);
    HWY_ASSERT_VEC_EQ(d, vi, Sqrt(vi * vi));
  }
};

struct TestReciprocalSquareRoot {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v = Set(d, 123.0f);
    HWY_ALIGN float lanes[MaxLanes(d)];
    Store(ApproximateReciprocalSqrt(v), d, lanes);
    for (size_t i = 0; i < Lanes(d); ++i) {
      float err = lanes[i] - 0.090166f;
      if (err < 0.0f) err = -err;
      HWY_ASSERT(err < 1E-4f);
    }
  }
};

struct TestRound {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
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
  }
};

struct TestSumsOfU8T {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
#if HWY_TARGET != HWY_SCALAR && HWY_CAP_INTEGER64
    const HWY_CAPPED(uint8_t, MaxLanes(d) * sizeof(uint64_t)) du8;
    HWY_ALIGN uint8_t in_bytes[MaxLanes(du8)];
    uint64_t sums[MaxLanes(d)] = {0};
    for (size_t i = 0; i < Lanes(du8); ++i) {
      const size_t group = i / 8;
      in_bytes[i] = static_cast<uint8_t>(2 * i + 1);
      sums[group] += in_bytes[i];
    }
    const auto v = Load(du8, in_bytes);
    HWY_ASSERT_VEC_EQ(d, sums, SumsOfU8x8(v));
#else
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestSumsOfU8() {
#if HWY_CAP_INTEGER64
  ForPartialVectors<TestSumsOfU8T>()(uint64_t());
#endif
}

struct TestSumOfLanesT {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    HWY_ALIGN T in_lanes[MaxLanes(d)];
    double sum = 0.0;
    for (size_t i = 0; i < Lanes(d); ++i) {
      in_lanes[i] = static_cast<T>(1u << i);
      sum += static_cast<double>(in_lanes[i]);
    }
    const auto v = Load(d, in_lanes);
    const auto expected = Set(d, T(sum));
    HWY_ASSERT_VEC_EQ(d, expected, SumOfLanes(v));
  }
};

struct TestMinOfLanesT {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    HWY_ALIGN T in_lanes[MaxLanes(d)];
    T min = LimitsMax<T>();
    for (size_t i = 0; i < Lanes(d); ++i) {
      in_lanes[i] = static_cast<T>(1u << i);
      min = std::min(min, in_lanes[i]);
    }
    const auto v = Load(d, in_lanes);
    const auto expected = Set(d, min);
    HWY_ASSERT_VEC_EQ(d, expected, MinOfLanes(v));
  }
};

struct TestMaxOfLanesT {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    HWY_ALIGN T in_lanes[MaxLanes(d)];
    T max = LimitsMin<T>();
    for (size_t i = 0; i < Lanes(d); ++i) {
      in_lanes[i] = static_cast<T>(1u << i);
      max = std::max(max, in_lanes[i]);
    }
    const auto v = Load(d, in_lanes);
    const auto expected = Set(d, max);
    HWY_ASSERT_VEC_EQ(d, expected, MaxOfLanes(v));
  }
};

HWY_NOINLINE void TestReductions() {
  // Only full vectors because lanes in partial vectors are undefined.
  const ForFullVectors<TestSumOfLanesT> sum;
  const ForFullVectors<TestMinOfLanesT> min;
  const ForFullVectors<TestMaxOfLanesT> max;

  // No u8/u16.
  sum(uint32_t());
  min(uint32_t());
  max(uint32_t());
#if HWY_CAP_INTEGER64
  sum(uint64_t());
#if HWY_MINMAX64_LANES > 1
  min(uint64_t());
  max(uint64_t());
#endif
#endif

  // No i8/i16.
  sum(int32_t());
  min(int32_t());
  max(int32_t());
#if HWY_CAP_INTEGER64
  sum(int64_t());
#if HWY_MINMAX64_LANES > 1
  min(int64_t());
  max(int64_t());
#endif
#endif

  sum(float());
  min(float());
  max(float());
#if HWY_CAP_FLOAT64
  sum(double());
  min(double());
  max(double());
#endif
}

struct TestAbsDiffT {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = MaxLanes(d);
    HWY_ALIGN T in_lanes_a[N];
    HWY_ALIGN T in_lanes_b[N];
    HWY_ALIGN T out_lanes[N];
    for (size_t i = 0; i < Lanes(d); ++i) {
      in_lanes_a[i] = (i ^ 1u) << i;
      in_lanes_b[i] = i << i;
      out_lanes[i] = fabsf(in_lanes_a[i] - in_lanes_b[i]);
    }
    const auto a = Load(d, in_lanes_a);
    const auto b = Load(d, in_lanes_b);
    const auto expected = Load(d, out_lanes);
    HWY_ASSERT_VEC_EQ(d, expected, AbsDiff(a, b));
    HWY_ASSERT_VEC_EQ(d, expected, AbsDiff(b, a));
  }
};

HWY_NOINLINE void TestAllPlusMinus() {
  ForAllTypes(ForPartialVectors<TestPlusMinus>());
}

HWY_NOINLINE void TestAllMulAdd() {
  ForFloatTypes(ForPartialVectors<TestMulAdd>());
}

HWY_NOINLINE void TestAllSquareRoot() {
  ForFloatTypes(ForPartialVectors<TestSquareRoot>());
}

HWY_NOINLINE void TestAllReciprocalSquareRoot() {
  ForPartialVectors<TestReciprocalSquareRoot>()(float());
}

HWY_NOINLINE void TestAllRound() {
  ForFloatTypes(ForPartialVectors<TestRound>());
}

HWY_NOINLINE void TestAllAbsDiff() {
  ForPartialVectors<TestAbsDiffT>()(float());
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
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestSaturatingArithmetic);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestShifts);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestMinMax);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAverage);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAbs);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestMul);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestMulHigh);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestMulEven);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllMulAdd);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllSquareRoot);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllReciprocalSquareRoot);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestSumsOfU8);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestReductions);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllRound);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllAbsDiff);

}  // namespace hwy
#endif
