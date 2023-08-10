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
    const auto v0 = Zero(d);
    const auto v1 = Set(d, T(1));
    const auto vi = Iota(d, 1);
    const auto vj = Iota(d, 3);
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(expected);

    HWY_ASSERT_VEC_EQ(d, v0, Mul(v0, v0));
    HWY_ASSERT_VEC_EQ(d, v1, Mul(v1, v1));
    HWY_ASSERT_VEC_EQ(d, vi, Mul(v1, vi));
    HWY_ASSERT_VEC_EQ(d, vi, Mul(vi, v1));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>((1 + i) * (1 + i));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), Mul(vi, vi));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>((1 + i) * (3 + i));
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

    const auto v0 = Zero(d);
    const auto v1 = Set(d, T(1));
    const auto vi = Iota(d, 1);
    const auto vn = Iota(d, -T(N));  // no i8 supported, so no wraparound
    HWY_ASSERT_VEC_EQ(d, v0, Mul(v0, v0));
    HWY_ASSERT_VEC_EQ(d, v1, Mul(v1, v1));
    HWY_ASSERT_VEC_EQ(d, vi, Mul(v1, vi));
    HWY_ASSERT_VEC_EQ(d, vi, Mul(vi, v1));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>((1 + i) * (1 + i));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), Mul(vi, vi));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>((-T(N) + T(i)) * T(1u + i));
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
    const auto vZero = Set(d, T(0));
    const auto v1 = Set(d, T(1));
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

    const auto vi = Iota(d, 1);
    // no i8 supported, so no wraparound
    const auto vni = Iota(d, T(static_cast<T>(~N + 1)));

    const auto v0 = Zero(d);
    HWY_ASSERT_VEC_EQ(d, v0, MulHigh(v0, v0));
    HWY_ASSERT_VEC_EQ(d, v0, MulHigh(v0, vi));
    HWY_ASSERT_VEC_EQ(d, v0, MulHigh(vi, v0));

    // Large positive squared
    for (size_t i = 0; i < N; ++i) {
      in_lanes[i] = T(LimitsMax<T>() >> i);
      expected_lanes[i] = T((Wide(in_lanes[i]) * in_lanes[i]) >> 16);
    }
    auto v = Load(d, in_lanes.get());
    HWY_ASSERT_VEC_EQ(d, expected_lanes.get(), MulHigh(v, v));

    // Large positive * small positive
    for (size_t i = 0; i < N; ++i) {
      expected_lanes[i] = T((Wide(in_lanes[i]) * T(1u + i)) >> 16);
    }
    HWY_ASSERT_VEC_EQ(d, expected_lanes.get(), MulHigh(v, vi));
    HWY_ASSERT_VEC_EQ(d, expected_lanes.get(), MulHigh(vi, v));

    // Large positive * small negative
    for (size_t i = 0; i < N; ++i) {
      expected_lanes[i] = T((Wide(in_lanes[i]) * T(i - N)) >> 16);
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
        in1[i] = static_cast<T>(Random64(&rng) & 0xFFFF);
        in2[i] = static_cast<T>(Random64(&rng) & 0xFFFF);
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
        expected[i] = static_cast<T>(arm);
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
      in_lanes[i + 0] = LimitsMax<T>() >> (i & kShiftAmtMask);
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
      in_lanes[i + 1] = LimitsMax<T>() >> (i & kShiftAmtMask);
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
    const Rebind<If<IsFloat<T>(), T, MakeSigned<T>>, D> di;
    using TI = TFromD<decltype(di)>;
    const Vec<D> neg_v2 = BitCast(d, Neg(BitCast(di, v2)));

    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT_VEC_EQ(d, k0, MulAdd(k0, k0, k0));
    HWY_ASSERT_VEC_EQ(d, v2, MulAdd(k0, v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2, MulAdd(v1, k0, v2));
    HWY_ASSERT_VEC_EQ(d, k0, NegMulAdd(k0, k0, k0));
    HWY_ASSERT_VEC_EQ(d, v2, NegMulAdd(k0, v1, v2));
    HWY_ASSERT_VEC_EQ(d, v2, NegMulAdd(v1, k0, v2));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>((i + 1) * (i + 2));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulAdd(v2, v1, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulAdd(v1, v2, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulAdd(neg_v2, v1, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulAdd(v1, neg_v2, k0));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>((i + 2) * (i + 2) + (i + 1));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulAdd(v2, v2, v1));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulAdd(neg_v2, v2, v1));

    for (size_t i = 0; i < N; ++i) {
      const T nm = static_cast<T>(-(static_cast<TI>(i) + TI{2}));
      const T f = static_cast<T>(i + 2);
      const T a = static_cast<T>(i + 1);
      expected[i] = static_cast<T>(nm * f + a);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulAdd(v2, v2, v1));
  }
};

struct TestMulSub {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const Vec<D> k0 = Zero(d);
    const Vec<D> kNeg0 = Set(d, T(-0.0));
    const Vec<D> v1 = Iota(d, 1);
    const Vec<D> v2 = Iota(d, 2);
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);

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
      expected[i] = static_cast<T>((i + 1) * (i + 2));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulSub(v1, v2, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulSub(v2, v1, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulSub(Neg(v1), v2, k0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulSub(v2, Neg(v1), k0));

    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>((i + 2) * (i + 2) - (1 + i));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), MulSub(v2, v2, v1));
    HWY_ASSERT_VEC_EQ(d, expected.get(), NegMulSub(Neg(v2), v2, v1));
  }
};

HWY_NOINLINE void TestAllMulAdd() {
  ForAllTypes(ForPartialVectors<TestMulAdd>());
  ForFloatTypes(ForPartialVectors<TestMulSub>());
}

struct TestWidenMulPairwiseAdd {
  // Must be inlined on aarch64 for bf16, else clang crashes.
  template <typename TN, class DN>
  HWY_INLINE void operator()(TN /*unused*/, DN dn) {
    using TW = MakeWide<TN>;
    const RepartitionToWide<DN> dw;
    using VW = Vec<decltype(dw)>;
    using VN = Vec<decltype(dn)>;
    const size_t NN = Lanes(dn);

    const VW f0 = Zero(dw);
    const VW f1 = Set(dw, TW{1});
    const VN bf0 = Zero(dn);
    // Cannot Set() bfloat16_t directly.
    const VN bf1 = ReorderDemote2To(dn, f1, f1);

    // Any input zero => both outputs zero
    HWY_ASSERT_VEC_EQ(dw, f0, WidenMulPairwiseAdd(dw, bf0, bf0));
    HWY_ASSERT_VEC_EQ(dw, f0, WidenMulPairwiseAdd(dw, bf0, bf1));
    HWY_ASSERT_VEC_EQ(dw, f0, WidenMulPairwiseAdd(dw, bf1, bf0));

    // delta[p] := p all others zero.
    auto delta_w = AllocateAligned<TW>(NN);
    for (size_t p = 0; p < NN; ++p) {
      // Workaround for incorrect Clang wasm codegen: re-initialize the entire
      // array rather than zero-initialize once and then set lane p to p.
      for (size_t i = 0; i < NN; ++i) {
        delta_w[i] = static_cast<TW>((i == p) ? p : 0);
      }
      const VW delta0 = Load(dw, delta_w.get() + 0);
      const VW delta1 = Load(dw, delta_w.get() + NN / 2);
      const VN delta = OrderedDemote2To(dn, delta0, delta1);

      const VW expected = InsertLane(f0, p / 2, static_cast<TW>(p));
      {
        const VW actual = WidenMulPairwiseAdd(dw, delta, bf1);
        HWY_ASSERT_VEC_EQ(dw, expected, actual);
      }
      // Swapped arg order
      {
        const VW actual = WidenMulPairwiseAdd(dw, bf1, delta);
        HWY_ASSERT_VEC_EQ(dw, expected, actual);
      }
    }
  }
};

HWY_NOINLINE void TestAllWidenMulPairwiseAdd() {
  ForShrinkableVectors<TestWidenMulPairwiseAdd>()(bfloat16_t());
  ForShrinkableVectors<TestWidenMulPairwiseAdd>()(int16_t());
  ForShrinkableVectors<TestWidenMulPairwiseAdd>()(uint16_t());
}

struct TestSatWidenMulPairwiseAdd {
  template <typename TN, class DN>
  HWY_NOINLINE void operator()(TN /*unused*/, DN dn) {
    static_assert(IsSame<TN, int8_t>(), "TN should be int8_t");

    using TN_U = MakeUnsigned<TN>;
    using TW = MakeWide<TN>;
    const RepartitionToWide<DN> dw;
    using VW = Vec<decltype(dw)>;
    using VN = Vec<decltype(dn)>;
    const size_t NN = Lanes(dn);
    const size_t NW = Lanes(dw);
    HWY_ASSERT(NN == NW * 2);

    const RebindToUnsigned<decltype(dn)> dn_u;

    const VW f0 = Zero(dw);
    const VN nf0 = Zero(dn);
    const VN nf1 = Set(dn, TN{1});

    // Any input zero => both outputs zero
    HWY_ASSERT_VEC_EQ(dw, f0,
                      SatWidenMulPairwiseAdd(dw, BitCast(dn_u, nf0), nf0));
    HWY_ASSERT_VEC_EQ(dw, f0,
                      SatWidenMulPairwiseAdd(dw, BitCast(dn_u, nf0), nf1));
    HWY_ASSERT_VEC_EQ(dw, f0,
                      SatWidenMulPairwiseAdd(dw, BitCast(dn_u, nf1), nf0));

    // delta[p] := p all others zero.
    auto delta_w = AllocateAligned<TW>(NN);
    HWY_ASSERT(delta_w);

    auto expected = AllocateAligned<TW>(NW);
    HWY_ASSERT(expected);
    Store(f0, dw, expected.get());

    for (size_t p = 0; p < NN; ++p) {
      // Workaround for incorrect Clang wasm codegen: re-initialize the entire
      // array rather than zero-initialize once and then set lane p to p.

      const TN pn = static_cast<TN>(p);
      const TN_U pn_u = static_cast<TN_U>(pn);
      for (size_t i = 0; i < NN; ++i) {
        delta_w[i] = static_cast<TW>((i == p) ? pn : 0);
      }
      const VW delta0 = Load(dw, delta_w.get() + 0);
      const VW delta1 = Load(dw, delta_w.get() + NN / 2);
      const VN delta = OrderedDemote2To(dn, delta0, delta1);

      expected[p / 2] = static_cast<TW>(pn_u);
      const VW actual_1 = SatWidenMulPairwiseAdd(dw, BitCast(dn_u, delta), nf1);
      HWY_ASSERT_VEC_EQ(dw, expected.get(), actual_1);

      // Swapped arg order
      expected[p / 2] = static_cast<TW>(pn);
      const VW actual_2 = SatWidenMulPairwiseAdd(dw, BitCast(dn_u, nf1), delta);
      HWY_ASSERT_VEC_EQ(dw, expected.get(), actual_2);

      expected[p / 2] = TW{0};
    }

    const auto vn_signed_min = Set(dn, LimitsMin<TN>());
    const auto vn_signed_max = Set(dn, LimitsMax<TN>());
    const auto vn_unsigned_max = Set(dn_u, LimitsMax<TN_U>());
    const auto vw_signed_min = Set(dw, LimitsMin<TW>());
    const auto vw_signed_max = Set(dw, LimitsMax<TW>());
    const auto vw_neg_tn_unsigned_max =
        Set(dw, static_cast<TW>(-static_cast<TW>(LimitsMax<TN_U>())));

    HWY_ASSERT_VEC_EQ(
        dw, vw_signed_max,
        SatWidenMulPairwiseAdd(dw, vn_unsigned_max, vn_signed_max));
    HWY_ASSERT_VEC_EQ(
        dw, vw_signed_min,
        SatWidenMulPairwiseAdd(dw, vn_unsigned_max, vn_signed_min));
    HWY_ASSERT_VEC_EQ(dw, vw_neg_tn_unsigned_max,
                      SatWidenMulPairwiseAdd(
                          dw, vn_unsigned_max,
                          InterleaveLower(dn, vn_signed_max, vn_signed_min)));
    HWY_ASSERT_VEC_EQ(dw, vw_neg_tn_unsigned_max,
                      SatWidenMulPairwiseAdd(
                          dw, vn_unsigned_max,
                          InterleaveLower(dn, vn_signed_min, vn_signed_max)));

    constexpr TN kSignedMax = LimitsMax<TN>();
    constexpr TN kZeroIotaRepl = static_cast<TN>(LimitsMax<TN>() - 16);

    auto in_a = AllocateAligned<TN>(NN);
    auto in_b = AllocateAligned<TN>(NN);
    auto in_neg_b = AllocateAligned<TN>(NN);
    HWY_ASSERT(in_a && in_b && in_neg_b);

    for (size_t i = 0; i < NN; i++) {
      const auto val = ((i + 1) & kSignedMax);
      const auto a_val = static_cast<TN>((val != 0) ? val : kZeroIotaRepl);
      const auto b_val = static_cast<TN>((a_val & 63) + 20);
      in_a[i] = a_val;
      in_b[i] = static_cast<TN>(b_val);
      in_neg_b[i] = static_cast<TN>(-b_val);
    }

    for (size_t i = 0; i < NW; i++) {
      const TW a0 = static_cast<TW>(in_a[2 * i]);
      const TW a1 = static_cast<TW>(in_a[2 * i + 1]);
      expected[i] = static_cast<TW>(a0 * a0 + a1 * a1);
    }

    auto vn_a = Load(dn, in_a.get());
    HWY_ASSERT_VEC_EQ(dw, expected.get(),
                      SatWidenMulPairwiseAdd(dw, BitCast(dn_u, vn_a), vn_a));

    for (size_t i = 0; i < NW; i++) {
      expected[i] = -expected[i];
    }

    HWY_ASSERT_VEC_EQ(
        dw, expected.get(),
        SatWidenMulPairwiseAdd(dw, BitCast(dn_u, vn_a), Neg(vn_a)));

    auto vn_b = Load(dn, in_b.get());

    HWY_ASSERT_VEC_EQ(
        dw, vw_signed_max,
        SatWidenMulPairwiseAdd(
            dw, InterleaveLower(dn_u, BitCast(dn_u, vn_b), vn_unsigned_max),
            InterleaveLower(dn, vn_b, vn_signed_max)));
    HWY_ASSERT_VEC_EQ(
        dw, vw_signed_max,
        SatWidenMulPairwiseAdd(
            dw, InterleaveUpper(dn_u, BitCast(dn_u, vn_b), vn_unsigned_max),
            InterleaveUpper(dn, vn_b, vn_signed_max)));
    HWY_ASSERT_VEC_EQ(
        dw, vw_signed_max,
        SatWidenMulPairwiseAdd(
            dw, InterleaveLower(dn_u, vn_unsigned_max, BitCast(dn_u, vn_b)),
            InterleaveLower(dn, vn_signed_max, vn_b)));
    HWY_ASSERT_VEC_EQ(
        dw, vw_signed_max,
        SatWidenMulPairwiseAdd(
            dw, InterleaveUpper(dn_u, vn_unsigned_max, BitCast(dn_u, vn_b)),
            InterleaveUpper(dn, vn_signed_max, vn_b)));

    const auto vn_neg_b = Load(dn, in_neg_b.get());
    HWY_ASSERT_VEC_EQ(
        dw, vw_signed_min,
        SatWidenMulPairwiseAdd(
            dw, InterleaveLower(dn_u, BitCast(dn_u, vn_b), vn_unsigned_max),
            InterleaveLower(dn, vn_neg_b, vn_signed_min)));
    HWY_ASSERT_VEC_EQ(
        dw, vw_signed_min,
        SatWidenMulPairwiseAdd(
            dw, InterleaveUpper(dn_u, BitCast(dn_u, vn_b), vn_unsigned_max),
            InterleaveUpper(dn, vn_neg_b, vn_signed_min)));
    HWY_ASSERT_VEC_EQ(
        dw, vw_signed_min,
        SatWidenMulPairwiseAdd(
            dw, InterleaveLower(dn_u, vn_unsigned_max, BitCast(dn_u, vn_b)),
            InterleaveLower(dn, vn_signed_min, vn_neg_b)));
    HWY_ASSERT_VEC_EQ(
        dw, vw_signed_min,
        SatWidenMulPairwiseAdd(
            dw, InterleaveUpper(dn_u, vn_unsigned_max, BitCast(dn_u, vn_b)),
            InterleaveUpper(dn, vn_signed_min, vn_neg_b)));

    constexpr size_t kMaxLanesPerNBlock = 16 / sizeof(TN);
    constexpr size_t kMaxLanesPerWBlock = 16 / sizeof(TW);

    for (size_t i = 0; i < NW; i++) {
      const size_t blk_idx = i / kMaxLanesPerWBlock;
      const TW b = static_cast<TW>(
          in_b[blk_idx * kMaxLanesPerNBlock + (i & (kMaxLanesPerWBlock - 1))]);
      expected[i] =
          static_cast<TW>(b * b + static_cast<TW>(LimitsMax<TN_U>()) *
                                      static_cast<TW>(LimitsMin<TN>()));
    }
    HWY_ASSERT_VEC_EQ(
        dw, expected.get(),
        SatWidenMulPairwiseAdd(
            dw, InterleaveLower(dn_u, vn_unsigned_max, BitCast(dn_u, vn_b)),
            InterleaveLower(dn, vn_signed_min, vn_b)));
    HWY_ASSERT_VEC_EQ(
        dw, expected.get(),
        SatWidenMulPairwiseAdd(
            dw, InterleaveLower(dn_u, BitCast(dn_u, vn_b), vn_unsigned_max),
            InterleaveLower(dn, vn_b, vn_signed_min)));
  }
};

HWY_NOINLINE void TestAllSatWidenMulPairwiseAdd() {
  ForShrinkableVectors<TestSatWidenMulPairwiseAdd>()(int8_t());
}

struct TestReorderWidenMulAccumulate {
  // Must be inlined on aarch64 for bf16, else clang crashes.
  template <typename TN, class DN>
  HWY_INLINE void operator()(TN /*unused*/, DN dn) {
    using TW = MakeWide<TN>;
    const RepartitionToWide<DN> dw;
    const Half<DN> dnh;
    using VW = Vec<decltype(dw)>;
    using VN = Vec<decltype(dn)>;
    const size_t NN = Lanes(dn);

    const VW f0 = Zero(dw);
    const VW f1 = Set(dw, TW{1});
    const VN bf0 = Zero(dn);
    // Cannot Set() bfloat16_t directly.
    const VN bf1 = ReorderDemote2To(dn, f1, f1);

    // Any input zero => both outputs zero
    VW sum1 = f0;
    HWY_ASSERT_VEC_EQ(dw, f0,
                      ReorderWidenMulAccumulate(dw, bf0, bf0, f0, sum1));
    HWY_ASSERT_VEC_EQ(dw, f0, sum1);
    HWY_ASSERT_VEC_EQ(dw, f0,
                      ReorderWidenMulAccumulate(dw, bf0, bf1, f0, sum1));
    HWY_ASSERT_VEC_EQ(dw, f0, sum1);
    HWY_ASSERT_VEC_EQ(dw, f0,
                      ReorderWidenMulAccumulate(dw, bf1, bf0, f0, sum1));
    HWY_ASSERT_VEC_EQ(dw, f0, sum1);

    // delta[p] := 1, all others zero. For each p: Dot(delta, all-ones) == 1.
    auto delta_w = AllocateAligned<TW>(NN);
    for (size_t p = 0; p < NN; ++p) {
      // Workaround for incorrect Clang wasm codegen: re-initialize the entire
      // array rather than zero-initialize once and then toggle lane p.
      for (size_t i = 0; i < NN; ++i) {
        delta_w[i] = static_cast<TW>(i == p);
      }
      const VW delta0 = Load(dw, delta_w.get());
      const VW delta1 = Load(dw, delta_w.get() + NN / 2);
      const VN delta = ReorderDemote2To(dn, delta0, delta1);

      {
        sum1 = f0;
        const VW sum0 = ReorderWidenMulAccumulate(dw, delta, bf1, f0, sum1);
        HWY_ASSERT_EQ(TW{1}, ReduceSum(dw, Add(sum0, sum1)));
      }
      // Swapped arg order
      {
        sum1 = f0;
        const VW sum0 = ReorderWidenMulAccumulate(dw, bf1, delta, f0, sum1);
        HWY_ASSERT_EQ(TW{1}, ReduceSum(dw, Add(sum0, sum1)));
      }
      // Start with nonzero sum0 or sum1
      {
        VW sum0 = PromoteTo(dw, LowerHalf(dnh, delta));
        sum1 = PromoteTo(dw, UpperHalf(dnh, delta));
        sum0 = ReorderWidenMulAccumulate(dw, delta, bf1, sum0, sum1);
        HWY_ASSERT_EQ(TW{2}, ReduceSum(dw, Add(sum0, sum1)));
      }
      // Start with nonzero sum0 or sum1, and swap arg order
      {
        VW sum0 = PromoteTo(dw, LowerHalf(dnh, delta));
        sum1 = PromoteTo(dw, UpperHalf(dnh, delta));
        sum0 = ReorderWidenMulAccumulate(dw, bf1, delta, sum0, sum1);
        HWY_ASSERT_EQ(TW{2}, ReduceSum(dw, Add(sum0, sum1)));
      }
    }
  }
};

HWY_NOINLINE void TestAllReorderWidenMulAccumulate() {
  ForShrinkableVectors<TestReorderWidenMulAccumulate>()(bfloat16_t());
  ForShrinkableVectors<TestReorderWidenMulAccumulate>()(int16_t());
  ForShrinkableVectors<TestReorderWidenMulAccumulate>()(uint16_t());
}

struct TestRearrangeToOddPlusEven {
  // Must be inlined on aarch64 for bf16, else clang crashes.
  template <typename TN, class DN>
  HWY_INLINE void operator()(TN /*unused*/, DN dn) {
    using TW = MakeWide<TN>;
    const RepartitionToWide<DN> dw;
    using VW = Vec<decltype(dw)>;
    using VN = Vec<decltype(dn)>;
    const size_t NW = Lanes(dw);

    const auto expected = AllocateAligned<TW>(NW);
    for (size_t iw = 0; iw < NW; ++iw) {
      const size_t in = iw * 2;  // even, odd is +1
      const size_t a0 = 1 + in;
      const size_t b0 = 1 + 2 * NW - a0;
      const size_t a1 = a0 + 1;
      const size_t b1 = b0 - 1;
      expected[iw] = static_cast<TW>(a0 * b0 + a1 * b1);
    }

    const VW up0 = Iota(dw, TW{1});
    const VW up1 = Iota(dw, static_cast<TW>(1 + NW));
    // We will compute i * (N-i) to avoid per-lane overflow.
    const VW down0 = Reverse(dw, up1);
    const VW down1 = Reverse(dw, up0);

    const VN a = OrderedDemote2To(dn, up0, up1);
    const VN b = OrderedDemote2To(dn, down0, down1);

    VW sum0 = Zero(dw);
    VW sum1 = Zero(dw);
    sum0 = ReorderWidenMulAccumulate(dw, a, b, sum0, sum1);
    const VW sum_odd_even = RearrangeToOddPlusEven(sum0, sum1);
    HWY_ASSERT_VEC_EQ(dw, expected.get(), sum_odd_even);
  }
};

HWY_NOINLINE void TestAllRearrangeToOddPlusEven() {
// For reasons unknown, <128 bit crashes aarch64 clang.
#if HWY_ARCH_ARM_A64 && HWY_COMPILER_CLANG
  ForGEVectors<128, TestRearrangeToOddPlusEven>()(bfloat16_t());
#else
  ForShrinkableVectors<TestRearrangeToOddPlusEven>()(bfloat16_t());
#endif
  ForShrinkableVectors<TestRearrangeToOddPlusEven>()(int16_t());
  ForShrinkableVectors<TestRearrangeToOddPlusEven>()(uint16_t());
}

template <bool MixedSignedness>
struct TestSumOfMulQuadAccumulate {
  template <class DW2, class TN1, class TN2>
  static HWY_INLINE void TestConsecutiveSeqMulQuadAccum(DW2 dw2, TN1 a0,
                                                        TN2 b0) {
    using TW2 = TFromD<DW2>;
    const Repartition<TN1, DW2> dn1;
    const Repartition<TN2, DW2> dn2;

    const auto vn_iota0_mod4 = And(Iota(dn1, TN1{0}), Set(dn1, TN1{3}));

    const auto va = Add(vn_iota0_mod4, Set(dn1, a0));
    const auto vb = Add(BitCast(dn2, vn_iota0_mod4), Set(dn2, b0));
    const auto expected =
        Set(dw2,
            static_cast<TW2>((TW2{4} * static_cast<TW2>(a0) * b0) +
                             (TW2{6} * (static_cast<TW2>(a0) + b0)) + TW2{17}));

    HWY_ASSERT_VEC_EQ(dw2, expected,
                      SumOfMulQuadAccumulate(dw2, va, vb, Set(dw2, TW2{3})));
  }

  template <typename TN2, class DN2>
  HWY_INLINE void operator()(TN2 /*unused*/, DN2 dn2) {
    static_assert(!MixedSignedness || IsSigned<TN2>(),
                  "TN2 must be signed if MixedSignedness is true");
    using TN1 = If<MixedSignedness, MakeUnsigned<TN2>, TN2>;
    using TW2 = MakeWide<MakeWide<TN2>>;

    const Rebind<TN1, DN2> dn1;
    const Repartition<TW2, DN2> dw2;

    const auto vn1_k1 = Set(dn1, TN1{1});
    const auto vn2_k1 = BitCast(dn2, vn1_k1);
    const auto vn1_k4 = Set(dn1, TN1{4});
    const auto vn2_k4 = BitCast(dn2, vn1_k4);

    const auto vw2_k0 = Zero(dw2);
    const auto vw2_k1 = Set(dw2, TW2{1});
    const auto vw2_k4 = Set(dw2, TW2{4});
    const auto vw2_k5 = Set(dw2, TW2{5});
    const auto vw2_k21 = Set(dw2, TW2{21});

    HWY_ASSERT_VEC_EQ(dw2, vw2_k4,
                      SumOfMulQuadAccumulate(dw2, vn1_k1, vn2_k1, vw2_k0));
    HWY_ASSERT_VEC_EQ(dw2, vw2_k5,
                      SumOfMulQuadAccumulate(dw2, vn1_k1, vn2_k1, vw2_k1));
    HWY_ASSERT_VEC_EQ(dw2, vw2_k21,
                      SumOfMulQuadAccumulate(dw2, vn1_k1, vn2_k4, vw2_k5));
    HWY_ASSERT_VEC_EQ(dw2, vw2_k21,
                      SumOfMulQuadAccumulate(dw2, vn1_k4, vn2_k1, vw2_k5));

    constexpr TN1 kTN1ValWithMaxMag =
        static_cast<TN1>(IsSigned<TN1>() ? LimitsMin<TN1>() : LimitsMax<TN1>());
    constexpr TN2 kTN2ValWithMaxMag =
        static_cast<TN2>(IsSigned<TN2>() ? LimitsMin<TN2>() : LimitsMax<TN2>());
    HWY_ASSERT_VEC_EQ(
        dw2,
        Set(dw2, static_cast<TW2>(static_cast<TW2>(kTN1ValWithMaxMag) *
                                  kTN2ValWithMaxMag * TW2{4})),
        SumOfMulQuadAccumulate(dw2, Set(dn1, kTN1ValWithMaxMag),
                               Set(dn2, kTN2ValWithMaxMag), vw2_k0));

    TestConsecutiveSeqMulQuadAccum(dw2, static_cast<TN1>(27),
                                   static_cast<TN2>(34));
    TestConsecutiveSeqMulQuadAccum(dw2, static_cast<TN1>(13),
                                   static_cast<TN2>(-5));
    TestConsecutiveSeqMulQuadAccum(dw2, static_cast<TN1>(-29),
                                   static_cast<TN2>(2));
    TestConsecutiveSeqMulQuadAccum(dw2, static_cast<TN1>(-14),
                                   static_cast<TN2>(-35));
    TestConsecutiveSeqMulQuadAccum(dw2, static_cast<TN1>(LimitsMin<TN1>() + 5),
                                   static_cast<TN2>(LimitsMax<TN2>() - 4));
    TestConsecutiveSeqMulQuadAccum(dw2, static_cast<TN1>(LimitsMax<TN1>() - 4),
                                   static_cast<TN2>(LimitsMin<TN2>() + 11));
  }
};

HWY_NOINLINE void TestAllSumOfMulQuadAccumulate() {
  ForShrinkableVectors<TestSumOfMulQuadAccumulate<false>, 2>()(int8_t());
  ForShrinkableVectors<TestSumOfMulQuadAccumulate<false>, 2>()(uint8_t());
  ForShrinkableVectors<TestSumOfMulQuadAccumulate<true>, 2>()(int8_t());
#if HWY_HAVE_INTEGER64
  ForShrinkableVectors<TestSumOfMulQuadAccumulate<false>, 2>()(int16_t());
  ForShrinkableVectors<TestSumOfMulQuadAccumulate<false>, 2>()(uint16_t());
#endif
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
HWY_EXPORT_AND_TEST_P(HwyMulTest, TestAllWidenMulPairwiseAdd);
HWY_EXPORT_AND_TEST_P(HwyMulTest, TestAllSatWidenMulPairwiseAdd);
HWY_EXPORT_AND_TEST_P(HwyMulTest, TestAllReorderWidenMulAccumulate);
HWY_EXPORT_AND_TEST_P(HwyMulTest, TestAllRearrangeToOddPlusEven);
HWY_EXPORT_AND_TEST_P(HwyMulTest, TestAllSumOfMulQuadAccumulate);

}  // namespace hwy

#endif
