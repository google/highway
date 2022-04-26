// Copyright 2019 Google LLC
// SPDX-License-Identifier: Apache-2.0
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

#include <inttypes.h>
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
    HWY_ASSERT_VEC_EQ(d, lanes.get(), Add(v2, v3));
    HWY_ASSERT_VEC_EQ(d, Set(d, 2), Sub(v4, v2));

    for (size_t i = 0; i < N; ++i) {
      lanes[i] = static_cast<T>((2 + i) + (4 + i));
    }
    auto sum = v2;
    sum = Add(sum, v4);  // sum == 6,8..
    HWY_ASSERT_VEC_EQ(d, Load(d, lanes.get()), sum);

    sum = Sub(sum, v4);
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

    HWY_ASSERT_VEC_EQ(d, Add(v0, v0), SaturatedAdd(v0, v0));
    HWY_ASSERT_VEC_EQ(d, Add(v0, vi), SaturatedAdd(v0, vi));
    HWY_ASSERT_VEC_EQ(d, Add(v0, vm), SaturatedAdd(v0, vm));
    HWY_ASSERT_VEC_EQ(d, vm, SaturatedAdd(vi, vm));
    HWY_ASSERT_VEC_EQ(d, vm, SaturatedAdd(vm, vm));

    HWY_ASSERT_VEC_EQ(d, v0, SaturatedSub(v0, v0));
    HWY_ASSERT_VEC_EQ(d, v0, SaturatedSub(v0, vi));
    HWY_ASSERT_VEC_EQ(d, v0, SaturatedSub(vi, vi));
    HWY_ASSERT_VEC_EQ(d, v0, SaturatedSub(vi, vm));
    HWY_ASSERT_VEC_EQ(d, Sub(vm, vi), SaturatedSub(vm, vi));
  }
};

struct TestSignedSaturatingArithmetic {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    const auto vpm = Set(d, LimitsMax<T>());
    // Ensure all lanes are positive, even if Iota wraps around
    const auto vi = Or(And(Iota(d, 0), vpm), Set(d, 1));
    const auto vn = Sub(v0, vi);
    const auto vnm = Set(d, LimitsMin<T>());
    HWY_ASSERT_MASK_EQ(d, MaskTrue(d), Gt(vi, v0));
    HWY_ASSERT_MASK_EQ(d, MaskTrue(d), Lt(vn, v0));

    HWY_ASSERT_VEC_EQ(d, v0, SaturatedAdd(v0, v0));
    HWY_ASSERT_VEC_EQ(d, vi, SaturatedAdd(v0, vi));
    HWY_ASSERT_VEC_EQ(d, vpm, SaturatedAdd(v0, vpm));
    HWY_ASSERT_VEC_EQ(d, vpm, SaturatedAdd(vi, vpm));
    HWY_ASSERT_VEC_EQ(d, vpm, SaturatedAdd(vpm, vpm));

    HWY_ASSERT_VEC_EQ(d, v0, SaturatedSub(v0, v0));
    HWY_ASSERT_VEC_EQ(d, Sub(v0, vi), SaturatedSub(v0, vi));
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
  ForSignedTypes(ForPartialVectors<TestAbs>());
  ForFloatTypes(ForPartialVectors<TestFloatAbs>());
}

struct TestUnsignedMinMax {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    // Leave headroom such that v1 < v2 even after wraparound.
    const auto mod = And(Iota(d, 0), Set(d, LimitsMax<T>() >> 1));
    const auto v1 = Add(mod, Set(d, 1));
    const auto v2 = Add(mod, Set(d, 2));
    HWY_ASSERT_VEC_EQ(d, v1, Min(v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2, Max(v1, v2));
    HWY_ASSERT_VEC_EQ(d, v0, Min(v1, v0));
    HWY_ASSERT_VEC_EQ(d, v1, Max(v1, v0));

    const auto vmin = Set(d, LimitsMin<T>());
    const auto vmax = Set(d, LimitsMax<T>());

    HWY_ASSERT_VEC_EQ(d, vmin, Min(vmin, vmax));
    HWY_ASSERT_VEC_EQ(d, vmin, Min(vmax, vmin));

    HWY_ASSERT_VEC_EQ(d, vmax, Max(vmin, vmax));
    HWY_ASSERT_VEC_EQ(d, vmax, Max(vmax, vmin));
  }
};

struct TestSignedMinMax {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Leave headroom such that v1 < v2 even after wraparound.
    const auto mod = And(Iota(d, 0), Set(d, LimitsMax<T>() >> 1));
    const auto v1 = Add(mod, Set(d, 1));
    const auto v2 = Add(mod, Set(d, 2));
    const auto v_neg = Sub(Zero(d), v1);
    HWY_ASSERT_VEC_EQ(d, v1, Min(v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2, Max(v1, v2));
    HWY_ASSERT_VEC_EQ(d, v_neg, Min(v1, v_neg));
    HWY_ASSERT_VEC_EQ(d, v1, Max(v1, v_neg));

    const auto v0 = Zero(d);
    const auto vmin = Set(d, LimitsMin<T>());
    const auto vmax = Set(d, LimitsMax<T>());
    HWY_ASSERT_VEC_EQ(d, vmin, Min(v0, vmin));
    HWY_ASSERT_VEC_EQ(d, vmin, Min(vmin, v0));
    HWY_ASSERT_VEC_EQ(d, v0, Max(v0, vmin));
    HWY_ASSERT_VEC_EQ(d, v0, Max(vmin, v0));

    HWY_ASSERT_VEC_EQ(d, vmin, Min(vmin, vmax));
    HWY_ASSERT_VEC_EQ(d, vmin, Min(vmax, vmin));

    HWY_ASSERT_VEC_EQ(d, vmax, Max(vmin, vmax));
    HWY_ASSERT_VEC_EQ(d, vmax, Max(vmax, vmin));
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

    const auto v0 = Zero(d);
    const auto vmin = Set(d, T(-1E30));
    const auto vmax = Set(d, T(1E30));
    HWY_ASSERT_VEC_EQ(d, vmin, Min(v0, vmin));
    HWY_ASSERT_VEC_EQ(d, vmin, Min(vmin, v0));
    HWY_ASSERT_VEC_EQ(d, v0, Max(v0, vmin));
    HWY_ASSERT_VEC_EQ(d, v0, Max(vmin, v0));

    HWY_ASSERT_VEC_EQ(d, vmin, Min(vmin, vmax));
    HWY_ASSERT_VEC_EQ(d, vmin, Min(vmax, vmin));

    HWY_ASSERT_VEC_EQ(d, vmax, Max(vmin, vmax));
    HWY_ASSERT_VEC_EQ(d, vmax, Max(vmax, vmin));
  }
};

HWY_NOINLINE void TestAllMinMax() {
  ForUnsignedTypes(ForPartialVectors<TestUnsignedMinMax>());
  ForSignedTypes(ForPartialVectors<TestSignedMinMax>());
  ForFloatTypes(ForPartialVectors<TestFloatMinMax>());
}

class TestMinMax128 {
  template <class D>
  static HWY_NOINLINE Vec<D> Make128(D d, uint64_t hi, uint64_t lo) {
    alignas(16) uint64_t in[2];
    in[0] = lo;
    in[1] = hi;
    return LoadDup128(d, in);
  }

 public:
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    using V = Vec<D>;
    const size_t N = Lanes(d);
    auto a_lanes = AllocateAligned<T>(N);
    auto b_lanes = AllocateAligned<T>(N);
    auto min_lanes = AllocateAligned<T>(N);
    auto max_lanes = AllocateAligned<T>(N);
    RandomState rng;

    const V v00 = Zero(d);
    const V v01 = Make128(d, 0, 1);
    const V v10 = Make128(d, 1, 0);
    const V v11 = Add(v01, v10);

    // Same arg
    HWY_ASSERT_VEC_EQ(d, v00, Min128(d, v00, v00));
    HWY_ASSERT_VEC_EQ(d, v01, Min128(d, v01, v01));
    HWY_ASSERT_VEC_EQ(d, v10, Min128(d, v10, v10));
    HWY_ASSERT_VEC_EQ(d, v11, Min128(d, v11, v11));
    HWY_ASSERT_VEC_EQ(d, v00, Max128(d, v00, v00));
    HWY_ASSERT_VEC_EQ(d, v01, Max128(d, v01, v01));
    HWY_ASSERT_VEC_EQ(d, v10, Max128(d, v10, v10));
    HWY_ASSERT_VEC_EQ(d, v11, Max128(d, v11, v11));

    // First arg less
    HWY_ASSERT_VEC_EQ(d, v00, Min128(d, v00, v01));
    HWY_ASSERT_VEC_EQ(d, v01, Min128(d, v01, v10));
    HWY_ASSERT_VEC_EQ(d, v10, Min128(d, v10, v11));
    HWY_ASSERT_VEC_EQ(d, v01, Max128(d, v00, v01));
    HWY_ASSERT_VEC_EQ(d, v10, Max128(d, v01, v10));
    HWY_ASSERT_VEC_EQ(d, v11, Max128(d, v10, v11));

    // Second arg less
    HWY_ASSERT_VEC_EQ(d, v00, Min128(d, v01, v00));
    HWY_ASSERT_VEC_EQ(d, v01, Min128(d, v10, v01));
    HWY_ASSERT_VEC_EQ(d, v10, Min128(d, v11, v10));
    HWY_ASSERT_VEC_EQ(d, v01, Max128(d, v01, v00));
    HWY_ASSERT_VEC_EQ(d, v10, Max128(d, v10, v01));
    HWY_ASSERT_VEC_EQ(d, v11, Max128(d, v11, v10));

    // Also check 128-bit blocks are independent
    for (size_t rep = 0; rep < AdjustedReps(1000); ++rep) {
      for (size_t i = 0; i < N; ++i) {
        a_lanes[i] = Random64(&rng);
        b_lanes[i] = Random64(&rng);
      }
      const V a = Load(d, a_lanes.get());
      const V b = Load(d, b_lanes.get());
      for (size_t i = 0; i < N; i += 2) {
        const bool lt = a_lanes[i + 1] == b_lanes[i + 1]
                            ? (a_lanes[i] < b_lanes[i])
                            : (a_lanes[i + 1] < b_lanes[i + 1]);
        min_lanes[i + 0] = lt ? a_lanes[i + 0] : b_lanes[i + 0];
        min_lanes[i + 1] = lt ? a_lanes[i + 1] : b_lanes[i + 1];
        max_lanes[i + 0] = lt ? b_lanes[i + 0] : a_lanes[i + 0];
        max_lanes[i + 1] = lt ? b_lanes[i + 1] : a_lanes[i + 1];
      }
      HWY_ASSERT_VEC_EQ(d, min_lanes.get(), Min128(d, a, b));
      HWY_ASSERT_VEC_EQ(d, max_lanes.get(), Max128(d, a, b));
    }
  }
};

HWY_NOINLINE void TestAllMinMax128() {
  ForGEVectors<128, TestMinMax128>()(uint64_t());
}


struct TestSumOfLanes {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto in_lanes = AllocateAligned<T>(N);

    // Lane i = bit i, higher lanes 0
    double sum = 0.0;
    // Avoid setting sign bit and cap at double precision
    constexpr size_t kBits = HWY_MIN(sizeof(T) * 8 - 1, 51);
    for (size_t i = 0; i < N; ++i) {
      in_lanes[i] = i < kBits ? static_cast<T>(1ull << i) : 0;
      sum += static_cast<double>(in_lanes[i]);
    }
    HWY_ASSERT_VEC_EQ(d, Set(d, T(sum)),
                      SumOfLanes(d, Load(d, in_lanes.get())));

    // Lane i = i (iota) to include upper lanes
    sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
      sum += static_cast<double>(i);
    }
    HWY_ASSERT_VEC_EQ(d, Set(d, T(sum)), SumOfLanes(d, Iota(d, 0)));
  }
};

HWY_NOINLINE void TestAllSumOfLanes() {
  ForUIF3264(ForPartialVectors<TestSumOfLanes>());
}

struct TestMinOfLanes {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto in_lanes = AllocateAligned<T>(N);

    // Lane i = bit i, higher lanes = 2 (not the minimum)
    T min = HighestValue<T>();
    // Avoid setting sign bit and cap at double precision
    constexpr size_t kBits = HWY_MIN(sizeof(T) * 8 - 1, 51);
    for (size_t i = 0; i < N; ++i) {
      in_lanes[i] = i < kBits ? static_cast<T>(1ull << i) : 2;
      min = HWY_MIN(min, in_lanes[i]);
    }
    HWY_ASSERT_VEC_EQ(d, Set(d, min), MinOfLanes(d, Load(d, in_lanes.get())));

    // Lane i = N - i to include upper lanes
    min = HighestValue<T>();
    for (size_t i = 0; i < N; ++i) {
      in_lanes[i] = static_cast<T>(N - i);  // no 8-bit T so no wraparound
      min = HWY_MIN(min, in_lanes[i]);
    }
    HWY_ASSERT_VEC_EQ(d, Set(d, min), MinOfLanes(d, Load(d, in_lanes.get())));
  }
};

struct TestMaxOfLanes {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto in_lanes = AllocateAligned<T>(N);

    T max = LowestValue<T>();
    // Avoid setting sign bit and cap at double precision
    constexpr size_t kBits = HWY_MIN(sizeof(T) * 8 - 1, 51);
    for (size_t i = 0; i < N; ++i) {
      in_lanes[i] = i < kBits ? static_cast<T>(1ull << i) : 0;
      max = HWY_MAX(max, in_lanes[i]);
    }
    HWY_ASSERT_VEC_EQ(d, Set(d, max), MaxOfLanes(d, Load(d, in_lanes.get())));

    // Lane i = i to include upper lanes
    max = LowestValue<T>();
    for (size_t i = 0; i < N; ++i) {
      in_lanes[i] = static_cast<T>(i);  // no 8-bit T so no wraparound
      max = HWY_MAX(max, in_lanes[i]);
    }
    HWY_ASSERT_VEC_EQ(d, Set(d, max), MaxOfLanes(d, Load(d, in_lanes.get())));
  }
};

HWY_NOINLINE void TestAllMinMaxOfLanes() {
  const ForPartialVectors<TestMinOfLanes> test_min;
  const ForPartialVectors<TestMaxOfLanes> test_max;
  ForUIF3264(test_min);
  ForUIF3264(test_max);
  test_min(uint16_t());
  test_max(uint16_t());
  test_min(int16_t());
  test_max(int16_t());
}

struct TestSumsOf8 {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    RandomState rng;

    const size_t N = Lanes(d);
    if (N < 8) return;
    const Repartition<uint64_t, D> du64;

    auto in_lanes = AllocateAligned<T>(N);
    auto sum_lanes = AllocateAligned<uint64_t>(N / 8);

    for (size_t rep = 0; rep < 100; ++rep) {
      for (size_t i = 0; i < N; ++i) {
        in_lanes[i] = Random64(&rng) & 0xFF;
      }

      for (size_t idx_sum = 0; idx_sum < N / 8; ++idx_sum) {
        uint64_t sum = 0;
        for (size_t i = 0; i < 8; ++i) {
          sum += in_lanes[idx_sum * 8 + i];
        }
        sum_lanes[idx_sum] = sum;
      }

      const Vec<D> in = Load(d, in_lanes.get());
      HWY_ASSERT_VEC_EQ(du64, sum_lanes.get(), SumsOf8(in));
    }
  }
};

HWY_NOINLINE void TestAllSumsOf8() {
  ForGEVectors<64, TestSumsOf8>()(uint8_t());
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
HWY_BEFORE_TEST(HwyArithmeticTest);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllPlusMinus);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllSaturatingArithmetic);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllMinMax);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllMinMax128);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllAverage);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllAbs);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllSumOfLanes);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllMinMaxOfLanes);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllSumsOf8);
HWY_EXPORT_AND_TEST_P(HwyArithmeticTest, TestAllNeg);
}  // namespace hwy

#endif
