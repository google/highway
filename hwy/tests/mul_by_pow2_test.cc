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

#include <cmath>  // std::floor

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/mul_by_pow2_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

#if HWY_TARGET <= HWY_NEON_WITHOUT_AES && HWY_ARCH_ARM_V7
template <class D>
static HWY_INLINE void ZeroOutDenormalValues(D d, TFromD<D>* HWY_RESTRICT ptr) {
  using T = TFromD<D>;
  using TU = MakeUnsigned<T>;

  const T kMinNormalVal =
      BitCastScalar<T>(static_cast<TU>(TU{1} << MantissaBits<T>()));

  const auto v = Load(d, ptr);
  Store(IfThenZeroElse(Lt(Abs(v), Set(d, kMinNormalVal)), v), d, ptr);
}
#endif

struct TestMulByPow2 {
  template <class D>
  static HWY_INLINE void VerifyMulByPow2(D d, const Vec<D> v,
                                         const Vec<RebindToSigned<D>> exp,
                                         const int line) {
    using TF = TFromD<D>;
    using TI = MakeSigned<TF>;

    const RebindToSigned<decltype(d)> di;

    constexpr TI kMaxClampedExp =
        static_cast<TI>(LimitsMax<int>() & LimitsMax<TI>());
    static_assert(kMaxClampedExp > 0, "kMaxClampedExp > 0 must be true");

    Vec<RebindToSigned<D>> clamped_exp = exp;
    HWY_IF_CONSTEXPR(kMaxClampedExp < LimitsMax<TI>()) {
      constexpr TI kMinClampedExp = static_cast<TI>(~kMaxClampedExp);
      static_assert(kMinClampedExp < 0, "kMinClampedExp < 0 must be true");

      clamped_exp = Min(Max(clamped_exp, Set(di, kMinClampedExp)),
                        Set(di, kMaxClampedExp));
    }

    const size_t N = Lanes(d);
    auto v_lanes = AllocateAligned<TF>(N);
    auto expected = AllocateAligned<TF>(N);
    auto clamped_exp_lanes = AllocateAligned<TI>(N);
    HWY_ASSERT(v_lanes && expected && clamped_exp_lanes);

    Store(v, d, v_lanes.get());
    Store(clamped_exp, di, clamped_exp_lanes.get());

    for (size_t i = 0; i < N; i++) {
      expected[i] = ConvertScalarTo<TF>(
          std::ldexp(ConvertScalarTo<double>(v_lanes[i]),
                     static_cast<int>(clamped_exp_lanes[i])));
    }

#if HWY_TARGET <= HWY_NEON_WITHOUT_AES && HWY_ARCH_ARM_V7
    ZeroOutDenormalValues(d, expected.get());
#endif

    const Vec<D> actual = MulByPow2(v, exp);
    AssertVecEqual(d, expected.get(), actual, __FILE__, line);
  }

  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    using TI = MakeSigned<T>;
    const RebindToSigned<decltype(d)> di;
    const RebindToUnsigned<decltype(d)> du;

    constexpr int kNumOfMantBits = MantissaBits<T>();
    constexpr TI kExpBias = static_cast<TI>(MaxExponentField<T>() >> 1);
    static_assert(kExpBias > 0, "kExpBias > 0 must be true");

    const auto v0 = Zero(d);
    const auto v1 = Set(d, ConvertScalarTo<T>(1.0));
    const auto v_neg_1 = Set(d, ConvertScalarTo<T>(-1.0));

    const auto zero_int_val = Zero(di);
    const auto min_int_val = Set(di, LimitsMin<TI>());
    const auto max_int_val = Set(di, LimitsMax<TI>());

    const auto huge_exp = Set(di, static_cast<TI>(kExpBias + 2));
    const auto tiny_exp =
        Set(di, static_cast<TI>(-kExpBias - kNumOfMantBits - 2));

    HWY_ASSERT_VEC_EQ(d, v0, MulByPow2(v0, zero_int_val));
    HWY_ASSERT_VEC_EQ(d, v0, MulByPow2(v0, tiny_exp));
    HWY_ASSERT_VEC_EQ(d, v0, MulByPow2(v0, min_int_val));
    HWY_ASSERT_VEC_EQ(d, v0, MulByPow2(v0, huge_exp));
    HWY_ASSERT_VEC_EQ(d, v0, MulByPow2(v0, max_int_val));

    const auto vnan = NaN(d);
    const auto vinf = Inf(d);
    const auto v_neg_inf = Neg(vinf);

    const auto all_true = MaskTrue(d);

    HWY_ASSERT_MASK_EQ(d, all_true, IsNaN(MulByPow2(vnan, zero_int_val)));
    HWY_ASSERT_MASK_EQ(d, all_true, IsNaN(MulByPow2(vnan, tiny_exp)));
    HWY_ASSERT_MASK_EQ(d, all_true, IsNaN(MulByPow2(vnan, min_int_val)));
    HWY_ASSERT_MASK_EQ(d, all_true, IsNaN(MulByPow2(vnan, huge_exp)));
    HWY_ASSERT_MASK_EQ(d, all_true, IsNaN(MulByPow2(vnan, max_int_val)));

    HWY_ASSERT_VEC_EQ(d, v1, MulByPow2(v1, zero_int_val));
    HWY_ASSERT_VEC_EQ(d, v0, MulByPow2(v1, tiny_exp));
    HWY_ASSERT_VEC_EQ(d, v0, MulByPow2(v1, min_int_val));
    HWY_ASSERT_VEC_EQ(d, vinf, MulByPow2(v1, huge_exp));
    HWY_ASSERT_VEC_EQ(d, vinf, MulByPow2(v1, max_int_val));

    HWY_ASSERT_VEC_EQ(d, v_neg_1, MulByPow2(v_neg_1, zero_int_val));
    HWY_ASSERT_VEC_EQ(d, v0, MulByPow2(v_neg_1, tiny_exp));
    HWY_ASSERT_VEC_EQ(d, v0, MulByPow2(v_neg_1, min_int_val));
    HWY_ASSERT_VEC_EQ(d, v_neg_inf, MulByPow2(v_neg_1, huge_exp));
    HWY_ASSERT_VEC_EQ(d, v_neg_inf, MulByPow2(v_neg_1, max_int_val));

    HWY_ASSERT_VEC_EQ(d, vinf, MulByPow2(vinf, zero_int_val));
    HWY_ASSERT_VEC_EQ(d, vinf, MulByPow2(vinf, tiny_exp));
    HWY_ASSERT_VEC_EQ(d, vinf, MulByPow2(vinf, min_int_val));
    HWY_ASSERT_VEC_EQ(d, vinf, MulByPow2(vinf, huge_exp));
    HWY_ASSERT_VEC_EQ(d, vinf, MulByPow2(vinf, max_int_val));

    HWY_ASSERT_VEC_EQ(d, v_neg_inf, MulByPow2(v_neg_inf, zero_int_val));
    HWY_ASSERT_VEC_EQ(d, v_neg_inf, MulByPow2(v_neg_inf, tiny_exp));
    HWY_ASSERT_VEC_EQ(d, v_neg_inf, MulByPow2(v_neg_inf, min_int_val));
    HWY_ASSERT_VEC_EQ(d, v_neg_inf, MulByPow2(v_neg_inf, huge_exp));
    HWY_ASSERT_VEC_EQ(d, v_neg_inf, MulByPow2(v_neg_inf, max_int_val));

    const auto v_iota = PositiveIota(d);
    HWY_ASSERT_VEC_EQ(
        d, Mul(v_iota, Set(d, ConvertScalarTo<T>(1.7294921875))),
        MulByPow2(Mul(v_iota, Set(d, ConvertScalarTo<T>(13.8359375))),
                  Set(di, static_cast<TI>(-3))));
    HWY_ASSERT_VEC_EQ(d, Add(v_iota, v_iota),
                      MulByPow2(v_iota, Set(di, static_cast<TI>(1))));

    const size_t N = Lanes(d);
    auto in1_lanes = AllocateAligned<T>(N);
    auto in2_lanes = AllocateAligned<TI>(N);
    HWY_ASSERT(in1_lanes && in2_lanes);

    RandomState rng;
    for (size_t rep = 0; rep < AdjustedReps(100); ++rep) {
      for (size_t i = 0; i < N; i++) {
        in1_lanes[i] = RandomFiniteValue<T>(&rng);
        in2_lanes[i] = RandomFiniteValue<TI>(&rng);
      }

      const auto in1 = Load(d, in1_lanes.get());
      const auto in2 = Load(di, in2_lanes.get());
      VerifyMulByPow2(d, in1, in2, __LINE__);

      const auto in1_exp =
          Add(BitCast(di, ShiftRight<kNumOfMantBits>(BitCast(du, Abs(in1)))),
              Set(di, static_cast<TI>(-kExpBias)));

      HWY_ASSERT_VEC_EQ(d, CopySignToAbs(vinf, in1),
                        MulByPow2(in1, Sub(huge_exp, in1_exp)));
      HWY_ASSERT_VEC_EQ(d, v0, MulByPow2(in1, Sub(tiny_exp, in1_exp)));
    }
  }
};

HWY_NOINLINE void TestAllMulByPow2() {
  ForFloatTypes(ForPartialVectors<TestMulByPow2>());
}

struct TestMulByFloorPow2 {
  template <class D>
  static HWY_INLINE void VerifyMulByFloorPow2(D d, const Vec<D> v,
                                              const Vec<D> exp,
                                              const int line) {
    using T = TFromD<D>;

    constexpr double kMinInt = static_cast<double>(LimitsMin<int>());
    constexpr double kMaxInt = static_cast<double>(LimitsMax<int>());

    const size_t N = Lanes(d);
    auto v_lanes = AllocateAligned<T>(N);
    auto exp_lanes = AllocateAligned<T>(N);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(v_lanes && exp_lanes && expected);

    Store(v, d, v_lanes.get());
    Store(exp, d, exp_lanes.get());

    const double kF64NegInf =
        BitCastScalar<double>(static_cast<uint64_t>(~MantissaMask<double>()));

    for (size_t i = 0; i < N; i++) {
      const double f64_val = ConvertScalarTo<double>(v_lanes[i]);
      const double floor_f64_exp =
          std::floor(ConvertScalarTo<double>(exp_lanes[i]));

      if (ScalarIsFinite(floor_f64_exp)) {
        expected[i] = ConvertScalarTo<T>(std::ldexp(
            f64_val, static_cast<int>(
                         HWY_MAX(HWY_MIN(floor_f64_exp, kMaxInt), kMinInt))));
      } else {
        expected[i] = ConvertScalarTo<T>(
            f64_val * ((floor_f64_exp == kF64NegInf) ? 0.0 : floor_f64_exp));
      }
    }

#if HWY_TARGET <= HWY_NEON_WITHOUT_AES && HWY_ARCH_ARM_V7
    ZeroOutDenormalValues(d, expected.get());
#endif

    AssertVecEqual(d, expected.get(), MulByFloorPow2(v, exp), __FILE__, line);
    AssertVecEqual(d, expected.get(), MulByFloorPow2(v, Floor(exp)), __FILE__,
                   line);
  }

  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    using TI = MakeSigned<T>;

    const RebindToSigned<decltype(d)> di;
    const RebindToUnsigned<decltype(d)> du;

    const auto v0 = Zero(d);
    const auto v1 = Set(d, ConvertScalarTo<T>(1.0));
    const auto v_iota = PositiveIota(d);
    const auto vnan = NaN(d);
    const auto vinf = Inf(d);
    const auto v_neg_inf = Neg(vinf);

    const auto all_true = MaskTrue(d);

    HWY_ASSERT_MASK_EQ(d, all_true, IsNaN(MulByFloorPow2(v0, vnan)));
    HWY_ASSERT_MASK_EQ(d, all_true, IsNaN(MulByFloorPow2(v_iota, vnan)));
    HWY_ASSERT_MASK_EQ(d, all_true, IsNaN(MulByFloorPow2(v0, vinf)));
    HWY_ASSERT_MASK_EQ(d, all_true,
                       IsNaN(MulByFloorPow2(v_neg_inf, v_neg_inf)));
    HWY_ASSERT_MASK_EQ(d, all_true, IsNaN(MulByFloorPow2(vinf, v_neg_inf)));
    HWY_ASSERT_MASK_EQ(d, all_true, IsNaN(MulByFloorPow2(vnan, v0)));
    HWY_ASSERT_MASK_EQ(d, all_true, IsNaN(MulByFloorPow2(vnan, v_iota)));

    const auto v_seq2 = Mul(v_iota, Set(d, ConvertScalarTo<T>(1.7294921875)));
    const auto v_seq3 =
        Mul(v_iota, Set(d, ConvertScalarTo<T>(-28.968994140625)));

    VerifyMulByFloorPow2(d, v_iota, v_seq2, __LINE__);
    VerifyMulByFloorPow2(d, v_seq2, v_iota, __LINE__);
    VerifyMulByFloorPow2(d, v_iota, v_seq3, __LINE__);
    VerifyMulByFloorPow2(d, v_seq3, v_iota, __LINE__);
    VerifyMulByFloorPow2(d, v1, v_seq2, __LINE__);
    VerifyMulByFloorPow2(d, v_seq2, v1, __LINE__);

    constexpr int kNumOfMantBits = MantissaBits<T>();
    constexpr TI kExpBias = static_cast<TI>(MaxExponentField<T>() >> 1);
    static_assert(kExpBias > 0, "kExpBias > 0 must be true");

    const auto huge_exp = Set(d, ConvertScalarTo<T>(kExpBias + 2));
    const auto tiny_exp =
        Set(d, ConvertScalarTo<T>(-kExpBias - kNumOfMantBits - 2));

    const size_t N = Lanes(d);
    auto in1_lanes = AllocateAligned<T>(N);
    auto in2_lanes = AllocateAligned<T>(N);
    HWY_ASSERT(in1_lanes && in2_lanes);

    RandomState rng;
    for (size_t rep = 0; rep < AdjustedReps(100); ++rep) {
      for (size_t i = 0; i < N; i++) {
        in1_lanes[i] = RandomFiniteValue<T>(&rng);
        in2_lanes[i] = RandomFiniteValue<T>(&rng);
      }

      const auto in1 = Load(d, in1_lanes.get());
      const auto in2 = Load(d, in2_lanes.get());
      VerifyMulByFloorPow2(d, in1, in2, __LINE__);

      const auto in1_exp = ConvertTo(
          d, Add(BitCast(di, ShiftRight<kNumOfMantBits>(BitCast(du, Abs(in1)))),
                 Set(di, static_cast<TI>(-kExpBias))));

      HWY_ASSERT_VEC_EQ(d, CopySignToAbs(vinf, in1),
                        MulByFloorPow2(in1, Sub(huge_exp, in1_exp)));
      HWY_ASSERT_VEC_EQ(d, v0, MulByFloorPow2(in1, Sub(tiny_exp, in1_exp)));
    }
  }
};

HWY_NOINLINE void TestAllMulByFloorPow2() {
  ForFloatTypes(ForPartialVectors<TestMulByFloorPow2>());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
HWY_BEFORE_TEST(HwyMulByPow2Test);
HWY_EXPORT_AND_TEST_P(HwyMulByPow2Test, TestAllMulByPow2);
HWY_EXPORT_AND_TEST_P(HwyMulByPow2Test, TestAllMulByFloorPow2);
HWY_AFTER_TEST();
}  // namespace hwy

#endif
