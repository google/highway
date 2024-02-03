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
#define HWY_TARGET_INCLUDE "tests/mul_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

template <size_t kBits>
constexpr uint64_t FirstBits() {
  return (1ull << kBits) - 1;
}
template <>
constexpr uint64_t FirstBits<64>() {
  return ~uint64_t{0};
}

struct TestUnsignedMul {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const Vec<D> v0 = Zero(d);
    const Vec<D> v1 = Set(d, static_cast<T>(1));
    const Vec<D> vi = Iota(d, 1);
    const Vec<D> vj = Iota(d, 3);
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(expected);

    HWY_ASSERT_VEC_EQ(d, v0, Mul(v0, v0));
    HWY_ASSERT_VEC_EQ(d, v1, Mul(v1, v1));
    HWY_ASSERT_VEC_EQ(d, vi, Mul(v1, vi));
    HWY_ASSERT_VEC_EQ(d, vi, Mul(vi, v1));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = ConvertScalarTo<T>((1 + i) * (1 + i));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), Mul(vi, vi));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = ConvertScalarTo<T>((1 + i) * (3 + i));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), Mul(vi, vj));

    const T max = LimitsMax<T>();
    const auto vmax = Set(d, max);
    HWY_ASSERT_VEC_EQ(d, vmax, Mul(vmax, v1));
    HWY_ASSERT_VEC_EQ(d, vmax, Mul(v1, vmax));

    constexpr uint64_t kMask = FirstBits<sizeof(T) * 8>();
    const T max2 = (static_cast<uint64_t>(max) * max) & kMask;
    HWY_ASSERT_VEC_EQ(d, Set(d, max2), Mul(vmax, vmax));
  }
};

struct TestSignedMul {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);

    const Vec<D> v0 = Zero(d);
    const Vec<D> v1 = Set(d, static_cast<T>(1));
    const Vec<D> vi = Iota(d, 1);
    // i8 is not supported, so T is large enough to avoid wraparound.
    const Vec<D> vn = Iota(d, -static_cast<T>(N));
    HWY_ASSERT_VEC_EQ(d, v0, Mul(v0, v0));
    HWY_ASSERT_VEC_EQ(d, v1, Mul(v1, v1));
    HWY_ASSERT_VEC_EQ(d, vi, Mul(v1, vi));
    HWY_ASSERT_VEC_EQ(d, vi, Mul(vi, v1));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = ConvertScalarTo<T>((1 + i) * (1 + i));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), Mul(vi, vi));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>((-static_cast<T>(N) + static_cast<T>(i)) *
                                   static_cast<T>(1 + i));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), Mul(vn, vi));
    HWY_ASSERT_VEC_EQ(d, expected.get(), Mul(vi, vn));
  }
};

struct TestMulOverflow {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto vMax = Set(d, LimitsMax<T>());
    HWY_ASSERT_VEC_EQ(d, Mul(vMax, vMax), Mul(vMax, vMax));
  }
};

struct TestDivOverflow {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const Vec<D> vZero = Set(d, ConvertScalarTo<T>(0));
    const Vec<D> v1 = Set(d, ConvertScalarTo<T>(1));
    HWY_ASSERT_VEC_EQ(d, Div(v1, vZero), Div(v1, vZero));
  }
};

HWY_NOINLINE void TestAllMul() {
  ForUnsignedTypes(ForPartialVectors<TestUnsignedMul>());
  ForSignedTypes(ForPartialVectors<TestSignedMul>());

  ForSignedTypes(ForPartialVectors<TestMulOverflow>());

  ForFloatTypes(ForPartialVectors<TestDivOverflow>());
}

struct TestMulHigh {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    using Wide = MakeWide<T>;
    const size_t N = Lanes(d);
    auto in_lanes = AllocateAligned<T>(N);
    auto expected_lanes = AllocateAligned<T>(N);

    const Vec<D> vi = Iota(d, 1);
    // no i8 supported, so no wraparound
    const Vec<D> vni = Iota(d, ConvertScalarTo(~N + 1));

    const Vec<D> v0 = Zero(d);
    HWY_ASSERT_VEC_EQ(d, v0, MulHigh(v0, v0));
    HWY_ASSERT_VEC_EQ(d, v0, MulHigh(v0, vi));
    HWY_ASSERT_VEC_EQ(d, v0, MulHigh(vi, v0));

    // Large positive squared
    for (size_t i = 0; i < N; ++i) {
      in_lanes[i] = static_cast<T>(LimitsMax<T>() >> i);
      expected_lanes[i] =
          static_cast<T>((Wide(in_lanes[i]) * in_lanes[i]) >> 16);
    }
    Vec<D> v = Load(d, in_lanes.get());
    HWY_ASSERT_VEC_EQ(d, expected_lanes.get(), MulHigh(v, v));

    // Large positive * small positive
    for (size_t i = 0; i < N; ++i) {
      expected_lanes[i] =
          static_cast<T>((Wide(in_lanes[i]) * static_cast<T>(1 + i)) >> 16);
    }
    HWY_ASSERT_VEC_EQ(d, expected_lanes.get(), MulHigh(v, vi));
    HWY_ASSERT_VEC_EQ(d, expected_lanes.get(), MulHigh(vi, v));

    // Large positive * small negative
    for (size_t i = 0; i < N; ++i) {
      const T neg = static_cast<T>(static_cast<T>(i) - static_cast<T>(N));
      expected_lanes[i] =
          static_cast<T>((static_cast<Wide>(in_lanes[i]) * neg) >> 16);
    }
    HWY_ASSERT_VEC_EQ(d, expected_lanes.get(), MulHigh(v, vni));
    HWY_ASSERT_VEC_EQ(d, expected_lanes.get(), MulHigh(vni, v));
  }
};

HWY_NOINLINE void TestAllMulHigh() {
  ForPartialVectors<TestMulHigh> test;
  test(int16_t());
  test(uint16_t());
}

struct TestMulFixedPoint15 {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    HWY_ASSERT_VEC_EQ(d, v0, MulFixedPoint15(v0, v0));
    HWY_ASSERT_VEC_EQ(d, v0, MulFixedPoint15(v0, v0));

    const size_t N = Lanes(d);
    auto in1 = AllocateAligned<T>(N);
    auto in2 = AllocateAligned<T>(N);
    auto expected = AllocateAligned<T>(N);

    // Random inputs in each lane
    RandomState rng;
    for (size_t rep = 0; rep < AdjustedReps(10000); ++rep) {
      for (size_t i = 0; i < N; ++i) {
        in1[i] = ConvertScalarTo<T>(Random64(&rng) & 0xFFFF);
        in2[i] = ConvertScalarTo<T>(Random64(&rng) & 0xFFFF);
      }

      for (size_t i = 0; i < N; ++i) {
        // There are three ways to compute the results. x86 and Arm are defined
        // using 32-bit multiplication results:
        const int arm =
            static_cast<int32_t>(2u * static_cast<uint32_t>(in1[i] * in2[i]) +
                                 0x8000u) >>
            16;
        const int x86 = (((in1[i] * in2[i]) >> 14) + 1) >> 1;
        // On other platforms, split the result into upper and lower 16 bits.
        const auto v1 = Set(d, in1[i]);
        const auto v2 = Set(d, in2[i]);
        const int hi = GetLane(MulHigh(v1, v2));
        const int lo = GetLane(Mul(v1, v2)) & 0xFFFF;
        const int split = 2 * hi + ((lo + 0x4000) >> 15);
        expected[i] = ConvertScalarTo<T>(arm);
        if (in1[i] != -32768 || in2[i] != -32768) {
          HWY_ASSERT_EQ(arm, x86);
          HWY_ASSERT_EQ(arm, split);
        }
      }

      const auto a = Load(d, in1.get());
      const auto b = Load(d, in2.get());
      HWY_ASSERT_VEC_EQ(d, expected.get(), MulFixedPoint15(a, b));
    }
  }
};

HWY_NOINLINE void TestAllMulFixedPoint15() {
  ForPartialVectors<TestMulFixedPoint15>()(int16_t());
}

struct TestMulEven {
  template <class D, HWY_IF_SIGNED_D(D)>
  HWY_INLINE void DoTestNegMulEven(D /*d*/, Vec<D> v) {
    using T = TFromD<D>;
    using Wide = MakeWide<T>;
    const Repartition<Wide, D> d2;

    const auto v_squared = MulEven(v, v);
    const auto neg_v_squared = Neg(v_squared);
    const auto neg_v = Neg(v);
    HWY_ASSERT_VEC_EQ(d2, v_squared, MulEven(neg_v, neg_v));
    HWY_ASSERT_VEC_EQ(d2, neg_v_squared, MulEven(neg_v, v));
    HWY_ASSERT_VEC_EQ(d2, neg_v_squared, MulEven(v, neg_v));
  }
  template <class D, HWY_IF_UNSIGNED_D(D)>
  HWY_INLINE void DoTestNegMulEven(D /*d*/, Vec<D> /*v*/) {}

  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    using Wide = MakeWide<T>;
    const Repartition<Wide, D> d2;
    const auto v0 = Zero(d);
    HWY_ASSERT_VEC_EQ(d2, Zero(d2), MulEven(v0, v0));

    constexpr size_t kShiftAmtMask = sizeof(T) * 8 - 1;
    const size_t N = Lanes(d);
    auto in_lanes = AllocateAligned<T>(N);
    auto expected = AllocateAligned<Wide>(Lanes(d2));
    for (size_t i = 0; i < N; i += 2) {
      in_lanes[i + 0] =
          ConvertScalarTo<T>(LimitsMax<T>() >> (i & kShiftAmtMask));
      if (N != 1) {
        in_lanes[i + 1] = 1;  // unused
      }
      expected[i / 2] =
          static_cast<Wide>(Wide(in_lanes[i + 0]) * in_lanes[i + 0]);
    }

    const auto v = Load(d, in_lanes.get());
    HWY_ASSERT_VEC_EQ(d2, expected.get(), MulEven(v, v));

    DoTestNegMulEven(d, v);
  }
};

struct TestMulOdd {
  template <class D, HWY_IF_SIGNED_D(D)>
  HWY_INLINE void DoTestNegMulOdd(D d, Vec<D> v) {
    using T = TFromD<D>;
    using Wide = MakeWide<T>;
    const Repartition<Wide, D> d2;

    const auto v_squared = MulOdd(v, v);
    const auto neg_v_squared = Neg(v_squared);
    const auto neg_v = Neg(v);
    HWY_ASSERT_VEC_EQ(d2, v_squared, MulOdd(neg_v, neg_v));
    HWY_ASSERT_VEC_EQ(d2, neg_v_squared, MulOdd(neg_v, v));
    HWY_ASSERT_VEC_EQ(d2, neg_v_squared, MulOdd(v, neg_v));
    HWY_ASSERT_VEC_EQ(d2, neg_v_squared, MulEven(DupOdd(v), DupOdd(neg_v)));
    HWY_ASSERT_VEC_EQ(d2, neg_v_squared,
                      MulEven(Reverse2(d, v), Reverse2(d, neg_v)));
  }
  template <class D, HWY_IF_UNSIGNED_D(D)>
  HWY_INLINE void DoTestNegMulOdd(D /*d*/, Vec<D> /*v*/) {}

  template <typename T, class D, HWY_IF_LANES_GT_D(D, 1)>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
#if HWY_TARGET != HWY_SCALAR
    const size_t N = Lanes(d);
    if (N < 2) return;

    using Wide = MakeWide<T>;
    const Repartition<Wide, D> d2;
    const auto v0 = Zero(d);
    HWY_ASSERT_VEC_EQ(d2, Zero(d2), MulOdd(v0, v0));

    constexpr size_t kShiftAmtMask = sizeof(T) * 8 - 1;
    auto in_lanes = AllocateAligned<T>(N);
    auto expected = AllocateAligned<Wide>(Lanes(d2));
    for (size_t i = 0; i < N; i += 2) {
      in_lanes[i + 0] = 1;  // unused
      in_lanes[i + 1] =
          ConvertScalarTo<T>(LimitsMax<T>() >> (i & kShiftAmtMask));
      expected[i / 2] =
          static_cast<Wide>(Wide(in_lanes[i + 1]) * in_lanes[i + 1]);
    }

    const auto v = Load(d, in_lanes.get());
    HWY_ASSERT_VEC_EQ(d2, expected.get(), MulOdd(v, v));

    const auto v_dupodd = DupOdd(v);
    HWY_ASSERT_VEC_EQ(d2, expected.get(), MulEven(v_dupodd, v_dupodd));
    HWY_ASSERT_VEC_EQ(d2, expected.get(), MulOdd(v_dupodd, v_dupodd));
    HWY_ASSERT_VEC_EQ(d2, expected.get(), MulOdd(v_dupodd, v));
    HWY_ASSERT_VEC_EQ(d2, expected.get(), MulOdd(v, v_dupodd));

    const auto v_reverse2 = Reverse2(d, v);
    HWY_ASSERT_VEC_EQ(d2, expected.get(), MulEven(v_reverse2, v_reverse2));

    DoTestNegMulOdd(d, v);
#else
    (void)d;
#endif
  }
  template <typename T, class D, HWY_IF_LANES_LE_D(D, 1)>
  HWY_INLINE void operator()(T /*unused*/, D /*d*/) {}
};

#if HWY_HAVE_INTEGER64 && HWY_TARGET != HWY_SCALAR
struct TestMulEvenOdd64 {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v0 = Zero(d);
    HWY_ASSERT_VEC_EQ(d, Zero(d), MulEven(v0, v0));
    HWY_ASSERT_VEC_EQ(d, Zero(d), MulOdd(v0, v0));

    const size_t N = Lanes(d);
    if (N == 1) return;

    auto in1 = AllocateAligned<T>(N);
    auto in2 = AllocateAligned<T>(N);
    auto expected_even = AllocateAligned<T>(N);
    auto expected_odd = AllocateAligned<T>(N);

    // Random inputs in each lane
    RandomState rng;
    for (size_t rep = 0; rep < AdjustedReps(1000); ++rep) {
      for (size_t i = 0; i < N; ++i) {
        in1[i] = Random64(&rng);
        in2[i] = Random64(&rng);
      }

      for (size_t i = 0; i < N; i += 2) {
        expected_even[i] = Mul128(in1[i], in2[i], &expected_even[i + 1]);
        expected_odd[i] = Mul128(in1[i + 1], in2[i + 1], &expected_odd[i + 1]);
      }

      const auto a = Load(d, in1.get());
      const auto b = Load(d, in2.get());
      HWY_ASSERT_VEC_EQ(d, expected_even.get(), MulEven(a, b));
      HWY_ASSERT_VEC_EQ(d, expected_odd.get(), MulOdd(a, b));
    }
  }
};
#endif  // HWY_HAVE_INTEGER64 && HWY_TARGET != HWY_SCALAR

HWY_NOINLINE void TestAllMulEven() {
  ForUI8(ForGEVectors<16, TestMulEven>());
  ForUI16(ForGEVectors<32, TestMulEven>());

#if HWY_HAVE_INTEGER64
  ForUI32(ForGEVectors<64, TestMulEven>());
#if HWY_TARGET != HWY_SCALAR
  ForGEVectors<128, TestMulEvenOdd64>()(uint64_t());
#endif  // HWY_TARGET != HWY_SCALAR
#endif  // HWY_HAVE_INTEGER64
}

HWY_NOINLINE void TestAllMulOdd() {
  ForUI8(ForGEVectors<16, TestMulOdd>());
  ForUI16(ForGEVectors<32, TestMulOdd>());
#if HWY_HAVE_INTEGER64
  ForUI32(ForGEVectors<64, TestMulOdd>());
#endif

  // uint64_t MulOdd is already tested in TestMulEvenOdd64
}

#ifndef HWY_NATIVE_FMA
#error "Bug in set_macros-inl.h, did not set HWY_NATIVE_FMA"
#endif

struct TestMulAdd {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const Vec<D> k0 = Zero(d);
    const Vec<D> v1 = Iota(d, 1);
    const Vec<D> v2 = Iota(d, 2);

    // Unlike RebindToSigned, we want to leave floating-point unchanged.
    // This allows Neg for unsigned types.
    const Rebind<If<IsFloat<T>(), T, MakeSigned<T>>, D> dif;
    const Vec<D> neg_v2 = BitCast(d, Neg(BitCast(dif, v2)));

    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT_VEC_EQ(d, k0, MulAdd(k0, k0, k0));
    HWY_ASSERT_VEC_EQ(d, v2, MulAdd(k0, v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2, MulAdd(v1, k0, v2));
    HWY_ASSERT_VEC_EQ(d, k0, NegMulAdd(k0, k0, k0));
    HWY_ASSERT_VEC_EQ(d, v2, NegMulAdd(k0, v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2, NegMulAdd(v1, k0, v2));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = ConvertScalarTo<T>((i + 1) * (i + 2));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulAdd(v2, v1, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulAdd(v1, v2, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulAdd(neg_v2, v1, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulAdd(v1, neg_v2, k0));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = ConvertScalarTo<T>((i + 2) * (i + 2) + (i + 1));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulAdd(v2, v2, v1));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulAdd(neg_v2, v2, v1));

    for (size_t i = 0; i < N; ++i) {
      const T nm = ConvertScalarTo<T>(-static_cast<int>(i + 2));
      const T f = ConvertScalarTo<T>(i + 2);
      const T a = ConvertScalarTo<T>(i + 1);
      expected[i] = ConvertScalarTo<T>(nm * f + a);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulAdd(v2, v2, v1));
  }
};

struct TestMulSub {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const Vec<D> k0 = Zero(d);
    const Vec<D> kNeg0 = Set(d, ConvertScalarTo<T>(-0.0));
    const Vec<D> v1 = Iota(d, 1);
    const Vec<D> v2 = Iota(d, 2);
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);

    // Unlike RebindToSigned, we want to leave floating-point unchanged.
    // This allows Neg for unsigned types.
    const Rebind<If<IsFloat<T>(), T, MakeSigned<T>>, D> dif;

    HWY_ASSERT_VEC_EQ(d, k0, MulSub(k0, k0, k0));
    HWY_ASSERT_VEC_EQ(d, kNeg0, NegMulSub(k0, k0, k0));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = ConvertScalarTo<T>(-static_cast<int>(i + 2));
    }
    const auto neg_k0 = BitCast(d, Neg(BitCast(dif, k0)));
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulSub(k0, v1, v2));
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulSub(v1, k0, v2));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulSub(neg_k0, v1, v2));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulSub(v1, neg_k0, v2));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = ConvertScalarTo<T>((i + 1) * (i + 2));
    }
    const auto neg_v1 = BitCast(d, Neg(BitCast(dif, v1)));
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulSub(v1, v2, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulSub(v2, v1, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulSub(neg_v1, v2, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulSub(v2, neg_v1, k0));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = ConvertScalarTo<T>((i + 2) * (i + 2) - (1 + i));
    }
    const auto neg_v2 = BitCast(d, Neg(BitCast(dif, v2)));
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulSub(v2, v2, v1));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulSub(neg_v2, v2, v1));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulSub(v2, neg_v2, v1));
  }
};

struct TestMulAddSub {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const Vec<D> k0 = Zero(d);
    const Vec<D> v1 = Iota(d, 1);
    const Vec<D> v2 = Iota(d, 2);

    // Unlike RebindToSigned, we want to leave floating-point unchanged.
    // This allows Neg for unsigned types.
    const Rebind<If<IsFloat<T>(), T, MakeSigned<T>>, D> dif;
    const Vec<D> neg_v2 = BitCast(d, Neg(BitCast(dif, v2)));

    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(expected);

    HWY_ASSERT_VEC_EQ(d, k0, MulAddSub(k0, k0, k0));

    const auto v2_negated_if_even = OddEven(v2, neg_v2);
    HWY_ASSERT_VEC_EQ(d, v2_negated_if_even, MulAddSub(k0, v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2_negated_if_even, MulAddSub(v1, k0, v2));

    for (size_t i = 0; i < N; ++i) {
      expected[i] =
          ConvertScalarTo<T>(((i & 1) == 0) ? ((i + 2) * (i + 2) - (i + 1))
                                            : ((i + 2) * (i + 2) + (i + 1)));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulAddSub(v2, v2, v1));
  }
};

HWY_NOINLINE void TestAllMulAdd() {
  ForAllTypes(ForPartialVectors<TestMulAdd>());
  ForAllTypes(ForPartialVectors<TestMulSub>());
  ForAllTypes(ForPartialVectors<TestMulAddSub>());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
HWY_BEFORE_TEST(HwyMulTest);
HWY_EXPORT_AND_TEST_P(HwyMulTest, TestAllMul);
HWY_EXPORT_AND_TEST_P(HwyMulTest, TestAllMulHigh);
HWY_EXPORT_AND_TEST_P(HwyMulTest, TestAllMulFixedPoint15);
HWY_EXPORT_AND_TEST_P(HwyMulTest, TestAllMulEven);
HWY_EXPORT_AND_TEST_P(HwyMulTest, TestAllMulOdd);
HWY_EXPORT_AND_TEST_P(HwyMulTest, TestAllMulAdd);

}  // namespace hwy

#endif
