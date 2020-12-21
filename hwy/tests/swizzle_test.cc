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
#define HWY_TARGET_INCLUDE "tests/swizzle_test.cc"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

struct TestLowerHalf {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const Half<D> d2;

    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);
    std::fill(lanes.get(), lanes.get() + N, 0);
    const auto v = Iota(d, 1);
    Store(LowerHalf(v), d2, lanes.get());
    size_t i = 0;
    for (; i < Lanes(d2); ++i) {
      HWY_ASSERT_EQ(T(1 + i), lanes[i]);
    }
    // Other half remains unchanged
    for (; i < N; ++i) {
      HWY_ASSERT_EQ(T(0), lanes[i]);
    }
  }
};

struct TestLowerQuarter {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const Half<Half<D>> d4;

    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);
    std::fill(lanes.get(), lanes.get() + N, 0);
    const auto v = Iota(d, 1);
    const auto lo = LowerHalf(LowerHalf(v));
    Store(lo, d4, lanes.get());
    size_t i = 0;
    for (; i < Lanes(d4); ++i) {
      HWY_ASSERT_EQ(T(i + 1), lanes[i]);
    }
    // Upper 3/4 remain unchanged
    for (; i < N; ++i) {
      HWY_ASSERT_EQ(T(0), lanes[i]);
    }
  }
};

HWY_NOINLINE void TestAllLowerHalf() {
  constexpr size_t kDiv = 1;
  ForAllTypes(ForPartialVectors<TestLowerHalf, kDiv, /*kMinLanes=*/2>());
  ForAllTypes(ForPartialVectors<TestLowerQuarter, kDiv, /*kMinLanes=*/4>());
}

struct TestUpperHalf {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Scalar does not define UpperHalf.
#if HWY_TARGET != HWY_SCALAR
    const Half<D> d2;

    const auto v = Iota(d, 1);
    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);
    std::fill(lanes.get(), lanes.get() + N, 0);

    Store(UpperHalf(v), d2, lanes.get());
    size_t i = 0;
    for (; i < Lanes(d2); ++i) {
      HWY_ASSERT_EQ(T(Lanes(d2) + 1 + i), lanes[i]);
    }
    // Other half remains unchanged
    for (; i < N; ++i) {
      HWY_ASSERT_EQ(T(0), lanes[i]);
    }
#else
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllUpperHalf() {
  ForAllTypes(ForGE128Vectors<TestUpperHalf>());
}

struct TestZeroExtendVector {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
#if HWY_CAP_GE256
    const Twice<D> d2;

    const auto v = Iota(d, 1);
    const size_t N2 = Lanes(d2);
    auto lanes = AllocateAligned<T>(N2);
    Store(v, d, &lanes[0]);
    Store(v, d, &lanes[N2 / 2]);

    const auto ext = ZeroExtendVector(v);
    Store(ext, d2, lanes.get());

    size_t i = 0;
    // Lower half is unchanged
    for (; i < N2 / 2; ++i) {
      HWY_ASSERT_EQ(T(1 + i), lanes[i]);
    }
    // Upper half is zero
    for (; i < N2; ++i) {
      HWY_ASSERT_EQ(T(0), lanes[i]);
    }
#else
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllZeroExtendVector() {
  ForAllTypes(ForExtendableVectors<TestZeroExtendVector>());
}

struct TestCombine {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
#if HWY_CAP_GE256
    const Twice<D> d2;
    const size_t N2 = Lanes(d2);
    auto lanes = AllocateAligned<T>(N2);

    const auto lo = Iota(d, 1);
    const auto hi = Iota(d, N2 / 2 + 1);
    const auto combined = Combine(hi, lo);
    Store(combined, d2, lanes.get());

    const auto expected = Iota(d2, 1);
    HWY_ASSERT_VEC_EQ(d2, expected, combined);
#else
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllCombine() {
  ForAllTypes(ForExtendableVectors<TestCombine>());
}

struct TestShiftBytes {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Scalar does not define Shift*Bytes.
#if HWY_TARGET != HWY_SCALAR || HWY_IDE
    const Repartition<uint8_t, D> du8;
    const size_t N8 = Lanes(du8);

    // Zero remains zero
    const auto v0 = Zero(d);
    HWY_ASSERT_VEC_EQ(d, v0, ShiftLeftBytes<1>(v0));
    HWY_ASSERT_VEC_EQ(d, v0, ShiftRightBytes<1>(v0));

    // Zero after shifting out the high/low byte

    auto bytes = AllocateAligned<uint8_t>(N8);
    std::fill(bytes.get(), bytes.get() + N8, 0);

    bytes[N8 - 1] = 0x7F;
    const auto vhi = BitCast(d, Load(du8, bytes.get()));
    bytes[N8 - 1] = 0;
    bytes[0] = 0x7F;
    const auto vlo = BitCast(d, Load(du8, bytes.get()));
    HWY_ASSERT(AllTrue(ShiftLeftBytes<1>(vhi) == v0));
    HWY_ASSERT(AllTrue(ShiftRightBytes<1>(vlo) == v0));

    const size_t N = Lanes(d);
    auto in = AllocateAligned<T>(N);
    const uint8_t* in_bytes = reinterpret_cast<const uint8_t*>(in.get());
    const auto v = BitCast(d, Iota(du8, 1));
    Store(v, d, in.get());

    auto shifted = AllocateAligned<T>(N);
    const uint8_t* shifted_bytes =
        reinterpret_cast<const uint8_t*>(shifted.get());

    const size_t kBlockSize = HWY_MIN(N8, 16);
    Store(ShiftLeftBytes<1>(v), d, shifted.get());
    for (size_t block = 0; block < N8; block += kBlockSize) {
      HWY_ASSERT_EQ(uint8_t(0), shifted_bytes[block]);
      HWY_ASSERT(BytesEqual(in_bytes + block, shifted_bytes + block + 1,
                            kBlockSize - 1));
    }

    Store(ShiftRightBytes<1>(v), d, shifted.get());
    for (size_t block = 0; block < N8; block += kBlockSize) {
      HWY_ASSERT_EQ(uint8_t(0), shifted_bytes[block + kBlockSize - 1]);
      HWY_ASSERT(BytesEqual(in_bytes + block + 1, shifted_bytes + block,
                            kBlockSize - 1));
    }
#else
    (void)d;
#endif  // #if HWY_TARGET != HWY_SCALAR
  }
};

HWY_NOINLINE void TestAllShiftBytes() {
  ForIntegerTypes(ForGE128Vectors<TestShiftBytes>());
}

struct TestShiftLanes {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Scalar does not define Shift*Lanes.
#if HWY_TARGET != HWY_SCALAR || HWY_IDE
    const auto v = Iota(d, T(1));
    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);
    {
      const auto left0 = ShiftLeftLanes<0>(v);
      const auto right0 = ShiftRightLanes<0>(v);
      HWY_ASSERT_VEC_EQ(d, left0, v);
      HWY_ASSERT_VEC_EQ(d, right0, v);
    }

    constexpr size_t kLanesPerBlock = 16 / sizeof(T);

    {
      const auto left1 = ShiftLeftLanes<1>(v);
      Store(left1, d, lanes.get());
      for (size_t i = 0; i < Lanes(d); ++i) {
        const T expected = (i % kLanesPerBlock) == 0 ? T(0) : T(i);
        HWY_ASSERT_EQ(expected, lanes[i]);
      }
    }
    {
      const auto right1 = ShiftRightLanes<1>(v);
      Store(right1, d, lanes.get());
      for (size_t i = 0; i < Lanes(d); ++i) {
        const T expected =
            (i % kLanesPerBlock) == (kLanesPerBlock - 1) ? T(0) : T(2 + i);
        HWY_ASSERT_EQ(expected, lanes[i]);
      }
    }
#else
    (void)d;
#endif  // #if HWY_TARGET != HWY_SCALAR
  }
};

HWY_NOINLINE void TestAllShiftLanes() {
  ForAllTypes(ForGE128Vectors<TestShiftLanes>());
}

template <typename D, int kLane>
struct TestBroadcastR {
  HWY_NOINLINE void operator()() const {
    using T = typename D::T;
    const D d;
    const size_t N = Lanes(d);
    auto in_lanes = AllocateAligned<T>(N);
    std::fill(in_lanes.get(), in_lanes.get() + N, 0);
    const size_t blockN = HWY_MIN(N * sizeof(T), 16) / sizeof(T);
    // Need to set within each 128-bit block
    for (size_t block = 0; block < N; block += blockN) {
      in_lanes[block + kLane] = static_cast<T>(block + 1);
    }
    const auto in = Load(d, in_lanes.get());
    auto out_lanes = AllocateAligned<T>(N);
    Store(Broadcast<kLane>(in), d, out_lanes.get());
    for (size_t block = 0; block < N; block += blockN) {
      for (size_t i = 0; i < blockN; ++i) {
        HWY_ASSERT_EQ(T(block + 1), out_lanes[block + i]);
      }
    }

    TestBroadcastR<D, kLane - 1>()();
  }
};

template <class D>
struct TestBroadcastR<D, -1> {
  void operator()() const {}
};

struct TestBroadcast {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    TestBroadcastR<D, HWY_MIN(MaxLanes(d), 16 / sizeof(T)) - 1>()();
  }
};

HWY_NOINLINE void TestAllBroadcast() {
  const ForPartialVectors<TestBroadcast> test;
  // No u8.
  test(uint16_t());
  test(uint32_t());
#if HWY_CAP_INTEGER64
  test(uint64_t());
#endif

  // No i8.
  test(int16_t());
  test(int32_t());
#if HWY_CAP_INTEGER64
  test(int64_t());
#endif

  ForFloatTypes(test);
}

struct TestTableLookupBytes {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Avoid "Do not know how to split the result of this operator"
#if !defined(HWY_DISABLE_BROKEN_AVX3_TESTS) || HWY_TARGET != HWY_AVX3
    RandomState rng{1234};
    const Repartition<uint8_t, D> du8;
    const size_t N = Lanes(d);
    const size_t N8 = Lanes(du8);
    auto in_bytes = AllocateAligned<uint8_t>(N8);
    for (size_t i = 0; i < N8; ++i) {
      in_bytes[i] = Random32(&rng) & 0xFF;
    }
    const auto in = BitCast(d, Load(du8, in_bytes.get()));
    HWY_ALIGN const uint8_t index_bytes[kTestMaxVectorSize] = {
        // Same index as source, multiple outputs from same input,
        // unused input (9), ascending/descending and nonconsecutive neighbors.
        0,  2,  1, 2, 15, 12, 13, 14, 6,  7,  8,  5,  4,  3,  10, 11,
        11, 10, 3, 4, 5,  8,  7,  6,  14, 13, 12, 15, 2,  1,  2,  0,
        4,  3,  2, 2, 5,  6,  7,  7,  15, 15, 15, 15, 15, 15, 0,  1};
    const auto indices = Load(du8, index_bytes);
    auto out_lanes = AllocateAligned<T>(N);
    Store(TableLookupBytes(in, indices), d, out_lanes.get());
    const uint8_t* out_bytes =
        reinterpret_cast<const uint8_t*>(out_lanes.get());

    for (size_t block = 0; block + 16 <= N8; block += 16) {
      for (size_t i = 0; i < 16; ++i) {
        const uint8_t index = index_bytes[block + i];
        const uint8_t expected = in_bytes[(block + index) % N8];
        HWY_ASSERT_EQ(expected, out_bytes[block + i]);
      }
    }
#else  // HWY_DISABLE_BROKEN_AVX3_TESTS
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllTableLookupBytes() {
  // Not supported by HWY_SCALAR (its vector size is always less than 16 bytes)
  ForIntegerTypes(ForGE128Vectors<TestTableLookupBytes>());
}
struct TestTableLookupLanes {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Avoid "Do not know how to split the result of this operator"
#if (!defined(HWY_DISABLE_BROKEN_AVX3_TESTS) || HWY_TARGET != HWY_AVX3) && \
    HWY_CAP_GE256
    {
      const int32_t N = Lanes(d);
      // Too many permutations to test exhaustively; choose one with repeated
      // and cross-block indices and ensure indices do not exceed #lanes.
      HWY_ALIGN int32_t idx[kTestMaxVectorSize / sizeof(int32_t)] = {
          1,      3,      2,      2,      8 % N, 1,     7,     6,
          15 % N, 14 % N, 14 % N, 15 % N, 4,     9 % N, 8 % N, 5};
      const auto v = Iota(d, 1);
      auto expected = AllocateAligned<T>(static_cast<size_t>(N));
      for (int32_t i = 0; i < N; ++i) {
        expected[i] = idx[i] + 1;  // == v[idx[i]]
      }

      const auto opaque = SetTableIndices(d, idx);
      const auto actual = TableLookupLanes(v, opaque);
      HWY_ASSERT_VEC_EQ(d, expected.get(), actual);
    }
#elif !HWY_CAP_GE256 && HWY_TARGET != HWY_SCALAR  // 128-bit
    // Test all possible permutations.
    const size_t N = Lanes(d);
    auto idx = AllocateAligned<int32_t>(N);
    const auto v = Iota(d, 1);
    auto expected = AllocateAligned<T>(N);

    for (int32_t i0 = 0; i0 < static_cast<int32_t>(N); ++i0) {
      idx[0] = i0;
      for (int32_t i1 = 0; i1 < static_cast<int32_t>(N); ++i1) {
        idx[1] = i1;
        for (int32_t i2 = 0; i2 < static_cast<int32_t>(N); ++i2) {
          idx[2] = i2;
          for (int32_t i3 = 0; i3 < static_cast<int32_t>(N); ++i3) {
            idx[3] = i3;

            for (size_t i = 0; i < N; ++i) {
              expected[i] = idx[i] + 1;  // == v[idx[i]]
            }

            const auto opaque = SetTableIndices(d, idx.get());
            const auto actual = TableLookupLanes(v, opaque);
            HWY_ASSERT_VEC_EQ(d, expected.get(), actual);
          }
        }
      }
    }
#else  // HWY_DISABLE_BROKEN_AVX3_TESTS
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllTableLookupLanes() {
  const ForFullVectors<TestTableLookupLanes> test;
  test(uint32_t());
  test(int32_t());
  test(float());
}

struct TestInterleave {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto even_lanes = AllocateAligned<T>(N);
    auto odd_lanes = AllocateAligned<T>(N);
    for (size_t i = 0; i < N; ++i) {
      even_lanes[i] = static_cast<T>(2 * i + 0);
      odd_lanes[i] = static_cast<T>(2 * i + 1);
    }
    const auto even = Load(d, even_lanes.get());
    const auto odd = Load(d, odd_lanes.get());

    auto lo_lanes = AllocateAligned<T>(N);
    auto hi_lanes = AllocateAligned<T>(N);
    Store(InterleaveLower(even, odd), d, lo_lanes.get());
    Store(InterleaveUpper(even, odd), d, hi_lanes.get());

    const size_t blockN = 16 / sizeof(T);
    for (size_t i = 0; i < Lanes(d); ++i) {
      const size_t block = i / blockN;
      const size_t lo = (i % blockN) + block * 2 * blockN;
      HWY_ASSERT_EQ(T(lo), lo_lanes[i]);
      HWY_ASSERT_EQ(T(lo + blockN), hi_lanes[i]);
    }
  }
};

HWY_NOINLINE void TestAllInterleave() {
  // Not supported by HWY_SCALAR: Interleave(f32, f32) would return f32x2.
  ForAllTypes(ForGE128Vectors<TestInterleave>());
}

template <typename T>
struct MakeWideSigned {
  using type = typename TypesOfSize<2 * sizeof(T)>::Signed;
};

template <typename T>
struct MakeWideUnsigned {
  using type = typename TypesOfSize<2 * sizeof(T)>::Unsigned;
};

template <template <class> class MakeWide>
struct TestZipLower {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    using WideT = typename MakeWide<T>::type;
    static_assert(sizeof(T) * 2 == sizeof(WideT), "Must be double-width");
    static_assert(IsSigned<T>() == IsSigned<WideT>(), "Must have same sign");
    const Repartition<WideT, D> dw;
    const size_t N = Lanes(d);
    auto even_lanes = AllocateAligned<T>(N);
    auto odd_lanes = AllocateAligned<T>(N);
    for (size_t i = 0; i < N; ++i) {
      even_lanes[i] = static_cast<T>(2 * i + 0);
      odd_lanes[i] = static_cast<T>(2 * i + 1);
    }
    const auto even = Load(d, even_lanes.get());
    const auto odd = Load(d, odd_lanes.get());

    auto lo_lanes = AllocateAligned<WideT>(Lanes(dw));
    Store(ZipLower(even, odd), dw, lo_lanes.get());

    const WideT blockN = static_cast<WideT>(16 / sizeof(WideT));
    for (size_t i = 0; i < Lanes(dw); ++i) {
      const size_t block = i / blockN;
      // Value of least-significant lane in lo-vector.
      const WideT lo =
          static_cast<WideT>(2 * (i % blockN) + 4 * block * blockN);
      const WideT kBits = static_cast<WideT>(sizeof(T) * 8);
      const WideT expected_lo =
          static_cast<WideT>((static_cast<WideT>(lo + 1) << kBits) + lo);
      HWY_ASSERT_EQ(expected_lo, lo_lanes[i]);
    }
  }
};

template <template <class> class MakeWide>
struct TestZipUpper {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    using WideT = typename MakeWide<T>::type;
    static_assert(sizeof(T) * 2 == sizeof(WideT), "Must be double-width");
    static_assert(IsSigned<T>() == IsSigned<WideT>(), "Must have same sign");
    const size_t N = Lanes(d);
    auto even_lanes = AllocateAligned<T>(N);
    auto odd_lanes = AllocateAligned<T>(N);
    for (size_t i = 0; i < Lanes(d); ++i) {
      even_lanes[i] = static_cast<T>(2 * i + 0);
      odd_lanes[i] = static_cast<T>(2 * i + 1);
    }
    const auto even = Load(d, even_lanes.get());
    const auto odd = Load(d, odd_lanes.get());

    const Repartition<WideT, D> dw;
    auto hi_lanes = AllocateAligned<WideT>(Lanes(dw));
    Store(ZipUpper(even, odd), dw, hi_lanes.get());

    constexpr WideT blockN = static_cast<WideT>(16 / sizeof(WideT));
    for (size_t i = 0; i < Lanes(dw); ++i) {
      const size_t block = i / blockN;
      const WideT lo =
          static_cast<WideT>(2 * (i % blockN) + 4 * block * blockN);
      const WideT kBits = static_cast<WideT>(sizeof(T) * 8);
      const WideT expected_hi = static_cast<WideT>(
          (static_cast<WideT>(lo + 2 * blockN + 1) << kBits) + lo + 2 * blockN);
      HWY_ASSERT_EQ(expected_hi, hi_lanes[i]);
    }
  }
};

HWY_NOINLINE void TestAllZip() {
  const ForPartialVectors<TestZipLower<MakeWideUnsigned>, 2> lower_unsigned;
  lower_unsigned(uint8_t());
  lower_unsigned(uint16_t());
#if HWY_CAP_INTEGER64
  lower_unsigned(uint32_t());  // generates u64
#endif

  const ForPartialVectors<TestZipLower<MakeWideSigned>, 2> lower_signed;
  lower_signed(int8_t());
  lower_signed(int16_t());
#if HWY_CAP_INTEGER64
  lower_signed(int32_t());  // generates i64
#endif

  const ForGE128Vectors<TestZipUpper<MakeWideUnsigned>> upper_unsigned;
  upper_unsigned(uint8_t());
  upper_unsigned(uint16_t());
#if HWY_CAP_INTEGER64
  upper_unsigned(uint32_t());  // generates u64
#endif

  const ForGE128Vectors<TestZipUpper<MakeWideSigned>> upper_signed;
  upper_signed(int8_t());
  upper_signed(int16_t());
#if HWY_CAP_INTEGER64
  upper_signed(int32_t());  // generates i64
#endif

  // No float - concatenating f32 does not result in a f64
}

class TestSpecialShuffle32 {
 public:
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v = Iota(d, 0);

#define VERIFY_LANES_32(d, v, i3, i2, i1, i0) \
  VerifyLanes32((d), (v), (i3), (i2), (i1), (i0), __FILE__, __LINE__)

    VERIFY_LANES_32(d, Shuffle2301(v), 2, 3, 0, 1);
    VERIFY_LANES_32(d, Shuffle1032(v), 1, 0, 3, 2);
    VERIFY_LANES_32(d, Shuffle0321(v), 0, 3, 2, 1);
    VERIFY_LANES_32(d, Shuffle2103(v), 2, 1, 0, 3);
    VERIFY_LANES_32(d, Shuffle0123(v), 0, 1, 2, 3);

#undef VERIFY_LANES_32
  }

 private:
  template <class D, class V>
  HWY_NOINLINE void VerifyLanes32(D d, V v, const int i3, const int i2,
                                  const int i1, const int i0,
                                  const char* filename, const int line) {
    using T = typename D::T;
    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);
    Store(v, d, lanes.get());
    const std::string name = TypeName(lanes[0], N);
    constexpr size_t kBlockN = 16 / sizeof(T);
    for (int block = 0; block < static_cast<int>(N); block += kBlockN) {
      AssertEqual(T(block + i3), lanes[block + 3], name, filename, line);
      AssertEqual(T(block + i2), lanes[block + 2], name, filename, line);
      AssertEqual(T(block + i1), lanes[block + 1], name, filename, line);
      AssertEqual(T(block + i0), lanes[block + 0], name, filename, line);
    }
  }
};

class TestSpecialShuffle64 {
 public:
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v = Iota(d, 0);
    VerifyLanes64(d, Shuffle01(v), 0, 1, __FILE__, __LINE__);
  }

 private:
  template <class D, class V>
  HWY_NOINLINE void VerifyLanes64(D d, V v, const int i1, const int i0,
                                  const char* filename, const int line) {
    using T = typename D::T;
    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);
    Store(v, d, lanes.get());
    const std::string name = TypeName(lanes[0], N);
    constexpr size_t kBlockN = 16 / sizeof(T);
    for (int block = 0; block < static_cast<int>(N); block += kBlockN) {
      AssertEqual(T(block + i1), lanes[block + 1], name, filename, line);
      AssertEqual(T(block + i0), lanes[block + 0], name, filename, line);
    }
  }
};

HWY_NOINLINE void TestAllSpecialShuffles() {
  const ForGE128Vectors<TestSpecialShuffle32> test32;
  test32(uint32_t());
  test32(int32_t());
  test32(float());

#if HWY_CAP_INTEGER64
  const ForGE128Vectors<TestSpecialShuffle64> test64;
  test64(uint64_t());
  test64(int64_t());
#endif

#if HWY_CAP_FLOAT64
  const ForGE128Vectors<TestSpecialShuffle64> test_d;
  test_d(double());
#endif
}

template <int kBytes>
struct TestCombineShiftRightBytesR {
  template <class T, class D>
  HWY_NOINLINE void operator()(T t, D d) {
// Scalar does not define CombineShiftRightBytes.
#if HWY_TARGET != HWY_SCALAR || HWY_IDE
    const Repartition<uint8_t, D> d8;
    const size_t N8 = Lanes(d8);
    const auto lo = BitCast(d, Iota(d8, 1));
    const auto hi = BitCast(d, Iota(d8, 1 + N8));

    auto lanes = AllocateAligned<T>(Lanes(d));

    Store(CombineShiftRightBytes<kBytes>(hi, lo), d, lanes.get());
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(lanes.get());

    const size_t kBlockSize = 16;
    for (size_t i = 0; i < N8; ++i) {
      const size_t block = i / kBlockSize;
      const size_t lane = i % kBlockSize;
      const size_t first_lo = block * kBlockSize;
      const size_t idx = lane + kBytes;
      const size_t offset = (idx < kBlockSize) ? 0 : N8 - kBlockSize;
      const bool at_end = idx >= 2 * kBlockSize;
      const uint8_t expected =
          at_end ? 0 : static_cast<uint8_t>(first_lo + idx + 1 + offset);
      HWY_ASSERT_EQ(expected, bytes[i]);
    }

    TestCombineShiftRightBytesR<kBytes - 1>()(t, d);
#else
    (void)t;
    (void)d;
#endif  // #if HWY_TARGET != HWY_SCALAR
  }
};

template <int kLanes>
struct TestCombineShiftRightLanesR {
  template <class T, class D>
  HWY_NOINLINE void operator()(T t, D d) {
// Scalar does not define CombineShiftRightBytes (needed for *Lanes).
#if HWY_TARGET != HWY_SCALAR || HWY_IDE
    const Repartition<uint8_t, D> d8;
    const size_t N8 = Lanes(d8);
    const auto lo = BitCast(d, Iota(d8, 1));
    const auto hi = BitCast(d, Iota(d8, 1 + N8));

    auto lanes = AllocateAligned<T>(Lanes(d));

    Store(CombineShiftRightLanes<kLanes>(hi, lo), d, lanes.get());
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(lanes.get());

    const size_t kBlockSize = 16;
    for (size_t i = 0; i < N8; ++i) {
      const size_t block = i / kBlockSize;
      const size_t lane = i % kBlockSize;
      const size_t first_lo = block * kBlockSize;
      const size_t idx = lane + kLanes * sizeof(T);
      const size_t offset = (idx < kBlockSize) ? 0 : N8 - kBlockSize;
      const bool at_end = idx >= 2 * kBlockSize;
      const uint8_t expected =
          at_end ? 0 : static_cast<uint8_t>(first_lo + idx + 1 + offset);
      HWY_ASSERT_EQ(expected, bytes[i]);
    }

    TestCombineShiftRightBytesR<kLanes - 1>()(t, d);
#else
    (void)t;
    (void)d;
#endif  // #if HWY_TARGET != HWY_SCALAR
  }
};

template <>
struct TestCombineShiftRightBytesR<0> {
  template <class T, class D>
  void operator()(T /*unused*/, D /*unused*/) {}
};

template <>
struct TestCombineShiftRightLanesR<0> {
  template <class T, class D>
  void operator()(T /*unused*/, D /*unused*/) {}
};

struct TestCombineShiftRight {
  template <class T, class D>
  HWY_NOINLINE void operator()(T t, D d) {
    TestCombineShiftRightBytesR<15>()(t, d);
    TestCombineShiftRightLanesR<16 / sizeof(T) - 1>()(t, d);
  }
};

HWY_NOINLINE void TestAllCombineShiftRight() {
  ForAllTypes(ForGE128Vectors<TestCombineShiftRight>());
}

struct TestConcatHalves {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Construct inputs such that interleaved halves == iota.
    const auto expected = Iota(d, 1);

    const size_t N = Lanes(d);
    auto lo = AllocateAligned<T>(N);
    auto hi = AllocateAligned<T>(N);
    size_t i;
    for (i = 0; i < N / 2; ++i) {
      lo[i] = static_cast<T>(1 + i);
      hi[i] = static_cast<T>(lo[i] + T(N) / 2);
    }
    for (; i < N; ++i) {
      lo[i] = hi[i] = 0;
    }
    HWY_ASSERT_VEC_EQ(d, expected,
                      ConcatLowerLower(Load(d, hi.get()), Load(d, lo.get())));

    // Same for high blocks.
    for (i = 0; i < N / 2; ++i) {
      lo[i] = hi[i] = 0;
    }
    for (; i < N; ++i) {
      lo[i] = static_cast<T>(1 + i - N / 2);
      hi[i] = static_cast<T>(lo[i] + T(N) / 2);
    }
    HWY_ASSERT_VEC_EQ(d, expected,
                      ConcatUpperUpper(Load(d, hi.get()), Load(d, lo.get())));
  }
};

HWY_NOINLINE void TestAllConcatHalves() {
  ForAllTypes(ForGE128Vectors<TestConcatHalves>());
}

struct TestConcatLowerUpper {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    // Middle part of Iota(1) == Iota(1 + N / 2).
    const auto lo = Iota(d, 1);
    const auto hi = Iota(d, 1 + N);
    HWY_ASSERT_VEC_EQ(d, Iota(d, 1 + N / 2), ConcatLowerUpper(hi, lo));
  }
};

HWY_NOINLINE void TestAllConcatLowerUpper() {
  ForAllTypes(ForGE128Vectors<TestConcatLowerUpper>());
}

struct TestConcatUpperLower {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    const auto lo = Iota(d, 1);
    const auto hi = Iota(d, 1 + N);
    auto expected = AllocateAligned<T>(N);
    size_t i = 0;
    for (; i < N / 2; ++i) {
      expected[i] = static_cast<T>(1 + i);
    }
    for (; i < N; ++i) {
      expected[i] = static_cast<T>(1 + i + N);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), ConcatUpperLower(hi, lo));
  }
};

HWY_NOINLINE void TestAllConcatUpperLower() {
  ForAllTypes(ForGE128Vectors<TestConcatUpperLower>());
}

struct TestOddEven {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    const auto even = Iota(d, 1);
    const auto odd = Iota(d, 1 + N);
    auto expected = AllocateAligned<T>(N);
    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>(1 + i + ((i & 1) ? N : 0));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), OddEven(odd, even));
  }
};

HWY_NOINLINE void TestAllOddEven() {
  ForAllTypes(ForGE128Vectors<TestOddEven>());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {

class HwySwizzleTest : public hwy::TestWithParamTarget {};

HWY_TARGET_INSTANTIATE_TEST_SUITE_P(HwySwizzleTest);

HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllLowerHalf);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllUpperHalf);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllZeroExtendVector);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllCombine);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllShiftBytes);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllShiftLanes);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllBroadcast);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllTableLookupBytes);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllTableLookupLanes);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllInterleave);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllZip);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllSpecialShuffles);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllCombineShiftRight);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllConcatHalves);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllConcatLowerUpper);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllConcatUpperLower);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllOddEven);

}  // namespace hwy
#endif
