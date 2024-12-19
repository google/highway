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

#include <stddef.h>
#include <stdint.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/minmax_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

struct TestUnsignedMinMax {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    // Leave headroom such that v1 < v2 even after wraparound.
    const auto mod = And(Iota(d, 0), Set(d, LimitsMax<T>() >> 1));
    const auto v1 = Add(mod, Set(d, static_cast<T>(1)));
    const auto v2 = Add(mod, Set(d, static_cast<T>(2)));
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
    const auto mod =
        And(Iota(d, 0), Set(d, ConvertScalarTo<T>(LimitsMax<T>() >> 1)));
    const auto v1 = Add(mod, Set(d, ConvertScalarTo<T>(1)));
    const auto v2 = Add(mod, Set(d, ConvertScalarTo<T>(2)));
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
    const auto v_neg = Iota(d, -ConvertScalarTo<T>(Lanes(d)));
    HWY_ASSERT_VEC_EQ(d, v1, Min(v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2, Max(v1, v2));
    HWY_ASSERT_VEC_EQ(d, v_neg, Min(v1, v_neg));
    HWY_ASSERT_VEC_EQ(d, v1, Max(v1, v_neg));

    const auto v0 = Zero(d);
    const auto vmin = Set(d, ConvertScalarTo<T>(-1E30));
    const auto vmax = Set(d, ConvertScalarTo<T>(1E30));
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

template <class D>
static HWY_NOINLINE Vec<D> Make128(D d, uint64_t hi, uint64_t lo) {
  alignas(16) uint64_t in[2];
  in[0] = lo;
  in[1] = hi;
  return LoadDup128(d, in);
}

struct TestMinMax128 {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    using V = Vec<D>;
    const size_t N = Lanes(d);
    auto a_lanes = AllocateAligned<T>(N);
    auto b_lanes = AllocateAligned<T>(N);
    auto min_lanes = AllocateAligned<T>(N);
    auto max_lanes = AllocateAligned<T>(N);
    HWY_ASSERT(a_lanes && b_lanes && min_lanes && max_lanes);
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

struct TestMinMax128Upper {
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
    HWY_ASSERT_VEC_EQ(d, v00, Min128Upper(d, v00, v00));
    HWY_ASSERT_VEC_EQ(d, v01, Min128Upper(d, v01, v01));
    HWY_ASSERT_VEC_EQ(d, v10, Min128Upper(d, v10, v10));
    HWY_ASSERT_VEC_EQ(d, v11, Min128Upper(d, v11, v11));
    HWY_ASSERT_VEC_EQ(d, v00, Max128Upper(d, v00, v00));
    HWY_ASSERT_VEC_EQ(d, v01, Max128Upper(d, v01, v01));
    HWY_ASSERT_VEC_EQ(d, v10, Max128Upper(d, v10, v10));
    HWY_ASSERT_VEC_EQ(d, v11, Max128Upper(d, v11, v11));

    // Equivalent but not equal (chooses second arg)
    HWY_ASSERT_VEC_EQ(d, v01, Min128Upper(d, v00, v01));
    HWY_ASSERT_VEC_EQ(d, v11, Min128Upper(d, v10, v11));
    HWY_ASSERT_VEC_EQ(d, v00, Min128Upper(d, v01, v00));
    HWY_ASSERT_VEC_EQ(d, v10, Min128Upper(d, v11, v10));
    HWY_ASSERT_VEC_EQ(d, v00, Max128Upper(d, v01, v00));
    HWY_ASSERT_VEC_EQ(d, v10, Max128Upper(d, v11, v10));
    HWY_ASSERT_VEC_EQ(d, v01, Max128Upper(d, v00, v01));
    HWY_ASSERT_VEC_EQ(d, v11, Max128Upper(d, v10, v11));

    // First arg less
    HWY_ASSERT_VEC_EQ(d, v01, Min128Upper(d, v01, v10));
    HWY_ASSERT_VEC_EQ(d, v10, Max128Upper(d, v01, v10));

    // Second arg less
    HWY_ASSERT_VEC_EQ(d, v01, Min128Upper(d, v10, v01));
    HWY_ASSERT_VEC_EQ(d, v10, Max128Upper(d, v10, v01));

    // Also check 128-bit blocks are independent
    for (size_t rep = 0; rep < AdjustedReps(1000); ++rep) {
      for (size_t i = 0; i < N; ++i) {
        a_lanes[i] = Random64(&rng);
        b_lanes[i] = Random64(&rng);
      }
      const V a = Load(d, a_lanes.get());
      const V b = Load(d, b_lanes.get());
      for (size_t i = 0; i < N; i += 2) {
        const bool lt = a_lanes[i + 1] < b_lanes[i + 1];
        min_lanes[i + 0] = lt ? a_lanes[i + 0] : b_lanes[i + 0];
        min_lanes[i + 1] = lt ? a_lanes[i + 1] : b_lanes[i + 1];
        max_lanes[i + 0] = lt ? b_lanes[i + 0] : a_lanes[i + 0];
        max_lanes[i + 1] = lt ? b_lanes[i + 1] : a_lanes[i + 1];
      }
      HWY_ASSERT_VEC_EQ(d, min_lanes.get(), Min128Upper(d, a, b));
      HWY_ASSERT_VEC_EQ(d, max_lanes.get(), Max128Upper(d, a, b));
    }
  }
};

HWY_NOINLINE void TestAllMinMax128Upper() {
  ForGEVectors<128, TestMinMax128Upper>()(uint64_t());
}

struct TestMinMaxMagnitude {
  template <class T>
  static constexpr MakeSigned<T> MaxPosIotaVal(hwy::FloatTag /*type_tag*/) {
    return static_cast<MakeSigned<T>>(MantissaMask<T>() + 1);
  }
  template <class T>
  static constexpr MakeSigned<T> MaxPosIotaVal(hwy::NonFloatTag /*type_tag*/) {
    return static_cast<MakeSigned<T>>(((LimitsMax<MakeSigned<T>>()) >> 1) + 1);
  }

  template <class D>
  HWY_NOINLINE static void VerifyMinMaxMagnitude(
      D d, const TFromD<D>* HWY_RESTRICT in1_lanes,
      const TFromD<D>* HWY_RESTRICT in2_lanes, const int line) {
    using T = TFromD<D>;
    using TAbs = If<IsFloat<T>() || IsSpecialFloat<T>(), T, MakeUnsigned<T>>;

    const char* file = __FILE__;
    const size_t N = Lanes(d);
    auto expected_min_mag = AllocateAligned<T>(N);
    auto expected_max_mag = AllocateAligned<T>(N);
    HWY_ASSERT(expected_min_mag && expected_max_mag);

    for (size_t i = 0; i < N; i++) {
      const T val1 = in1_lanes[i];
      const T val2 = in2_lanes[i];
      const TAbs abs_val1 = static_cast<TAbs>(ScalarAbs(val1));
      const TAbs abs_val2 = static_cast<TAbs>(ScalarAbs(val2));
      if (abs_val1 < abs_val2 || (abs_val1 == abs_val2 && val1 < val2)) {
        expected_min_mag[i] = val1;
        expected_max_mag[i] = val2;
      } else {
        expected_min_mag[i] = val2;
        expected_max_mag[i] = val1;
      }
    }

    const auto in1 = Load(d, in1_lanes);
    const auto in2 = Load(d, in2_lanes);
    AssertVecEqual(d, expected_min_mag.get(), MinMagnitude(in1, in2), file,
                   line);
    AssertVecEqual(d, expected_min_mag.get(), MinMagnitude(in2, in1), file,
                   line);
    AssertVecEqual(d, expected_max_mag.get(), MaxMagnitude(in1, in2), file,
                   line);
    AssertVecEqual(d, expected_max_mag.get(), MaxMagnitude(in2, in1), file,
                   line);
  }

  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    using TI = MakeSigned<T>;
    using TU = MakeUnsigned<T>;
    constexpr TI kMaxPosIotaVal = MaxPosIotaVal<T>(hwy::IsFloatTag<T>());
    static_assert(kMaxPosIotaVal > 0, "kMaxPosIotaVal > 0 must be true");

    constexpr size_t kPositiveIotaMask = static_cast<size_t>(
        static_cast<TU>(kMaxPosIotaVal - 1) & (HWY_MAX_LANES_D(D) - 1));

    const size_t N = Lanes(d);
    auto in1_lanes = AllocateAligned<T>(N);
    auto in2_lanes = AllocateAligned<T>(N);
    auto in3_lanes = AllocateAligned<T>(N);
    auto in4_lanes = AllocateAligned<T>(N);
    HWY_ASSERT(in1_lanes && in2_lanes && in3_lanes && in4_lanes);

    for (size_t i = 0; i < N; i++) {
      const TI x1 = static_cast<TI>((i & kPositiveIotaMask) + 1);
      const TI x2 = static_cast<TI>(kMaxPosIotaVal - x1);
      const TI x3 = static_cast<TI>(-x1);
      const TI x4 = static_cast<TI>(-x2);

      in1_lanes[i] = ConvertScalarTo<T>(x1);
      in2_lanes[i] = ConvertScalarTo<T>(x2);
      in3_lanes[i] = ConvertScalarTo<T>(x3);
      in4_lanes[i] = ConvertScalarTo<T>(x4);
    }

    VerifyMinMaxMagnitude(d, in1_lanes.get(), in2_lanes.get(), __LINE__);
    VerifyMinMaxMagnitude(d, in1_lanes.get(), in3_lanes.get(), __LINE__);
    VerifyMinMaxMagnitude(d, in1_lanes.get(), in4_lanes.get(), __LINE__);
    VerifyMinMaxMagnitude(d, in2_lanes.get(), in3_lanes.get(), __LINE__);
    VerifyMinMaxMagnitude(d, in2_lanes.get(), in4_lanes.get(), __LINE__);
    VerifyMinMaxMagnitude(d, in3_lanes.get(), in4_lanes.get(), __LINE__);

    in2_lanes[0] = HighestValue<T>();
    in4_lanes[0] = LowestValue<T>();

    VerifyMinMaxMagnitude(d, in1_lanes.get(), in2_lanes.get(), __LINE__);
    VerifyMinMaxMagnitude(d, in1_lanes.get(), in4_lanes.get(), __LINE__);
    VerifyMinMaxMagnitude(d, in2_lanes.get(), in3_lanes.get(), __LINE__);
    VerifyMinMaxMagnitude(d, in2_lanes.get(), in4_lanes.get(), __LINE__);
    VerifyMinMaxMagnitude(d, in3_lanes.get(), in4_lanes.get(), __LINE__);
  }
};

HWY_NOINLINE void TestAllMinMaxMagnitude() {
  ForAllTypes(ForPartialVectors<TestMinMaxMagnitude>());
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(HwyMinMaxTest);
HWY_EXPORT_AND_TEST_P(HwyMinMaxTest, TestAllMinMax);
HWY_EXPORT_AND_TEST_P(HwyMinMaxTest, TestAllMinMax128);
HWY_EXPORT_AND_TEST_P(HwyMinMaxTest, TestAllMinMax128Upper);
HWY_EXPORT_AND_TEST_P(HwyMinMaxTest, TestAllMinMaxMagnitude);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
