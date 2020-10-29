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
#define HWY_TARGET_INCLUDE "tests/swizzle_test.cc"
#include "hwy/foreach_target.h"
// ^ must come before highway.h and any *-inl.h.

#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

struct TestLowerHalfT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    constexpr size_t N2 = (MaxLanes(d) + 1) / 2;
    const Simd<T, N2> d2;

    HWY_ALIGN T lanes[MaxLanes(d)] = {0};
    const auto v = Iota(d, 1);
    Store(LowerHalf(v), d2, lanes);
    size_t i = 0;
    for (; i < N2; ++i) {
      HWY_ASSERT_EQ(T(1 + i), lanes[i]);
    }
    // Other half remains unchanged
    for (; i < Lanes(d); ++i) {
      HWY_ASSERT_EQ(T(0), lanes[i]);
    }
  }
};

struct TestLowerQuarterT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    constexpr size_t N4 = (MaxLanes(d) + 3) / 4;
    const HWY_CAPPED(T, N4) d4;

    HWY_ALIGN T lanes[MaxLanes(d)] = {0};
    const auto v = Iota(d, 1);
    const auto lo = LowerHalf(LowerHalf(v));
    Store(lo, d4, lanes);
    size_t i = 0;
    for (; i < N4; ++i) {
      HWY_ASSERT_EQ(T(i + 1), lanes[i]);
    }
    // Upper 3/4 remain unchanged
    for (; i < Lanes(d); ++i) {
      HWY_ASSERT_EQ(T(0), lanes[i]);
    }
  }
};

HWY_NOINLINE void TestLowerHalf() {
  ForAllTypes(
      ForPartialVectors<TestLowerHalfT, /*kDivLanes=*/1, /*kMinLanes=*/2>());
  ForAllTypes(
      ForPartialVectors<TestLowerQuarterT, /*kDivLanes=*/1, /*kMinLanes=*/4>());
}

struct TestUpperHalfT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Scalar does not define UpperHalf.
#if HWY_TARGET != HWY_SCALAR
    constexpr size_t N2 = (MaxLanes(d) + 1) / 2;
    const Simd<T, N2> d2;

    const auto v = Iota(d, 1);
    HWY_ALIGN T lanes[MaxLanes(d)] = {0};

    Store(UpperHalf(v), d2, lanes);
    size_t i = 0;
    for (; i < N2; ++i) {
      HWY_ASSERT_EQ(T(N2 + 1 + i), lanes[i]);
    }
    // Other half remains unchanged
    for (; i < Lanes(d); ++i) {
      HWY_ASSERT_EQ(T(0), lanes[i]);
    }
#else
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestUpperHalf() {
  ForAllTypes(ForGE128Vectors<TestUpperHalfT>());
}

struct TestZeroExtendVectorT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
#if HWY_CAP_GE256
    constexpr size_t N2 = MaxLanes(d) * 2;
    const Simd<T, N2> d2;

    const auto v = Iota(d, 1);
    HWY_ALIGN T lanes[N2];
    Store(v, d, lanes);
    Store(v, d, lanes + N2 / 2);

    const auto ext = ZeroExtendVector(v);
    Store(ext, d2, lanes);

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

HWY_NOINLINE void TestZeroExtendVector() {
  ForAllTypes(ForExtendableVectors<TestZeroExtendVectorT>());
}

struct TestCombineT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
#if HWY_CAP_GE256
    constexpr size_t N2 = MaxLanes(d) * 2;
    const Simd<T, N2> d2;

    const auto lo = Iota(d, 1);
    const auto hi = Iota(d, N2 / 2 + 1);
    HWY_ALIGN T lanes[N2];
    const auto combined = Combine(hi, lo);
    Store(combined, d2, lanes);

    const auto expected = Iota(d2, 1);
    HWY_ASSERT_VEC_EQ(d2, expected, combined);
#else
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestCombine() {
  ForAllTypes(ForExtendableVectors<TestCombineT>());
}

struct TestShiftBytesT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Scalar does not define Shift*Bytes.
#if HWY_TARGET != HWY_SCALAR || HWY_IDE
    constexpr size_t kN = MaxLanes(d);
    const Simd<uint8_t, kN * sizeof(T)> du8;
    const size_t N8 = Lanes(du8);

    // Zero remains zero
    const auto v0 = Zero(d);
    HWY_ASSERT_VEC_EQ(d, v0, ShiftLeftBytes<1>(v0));
    HWY_ASSERT_VEC_EQ(d, v0, ShiftRightBytes<1>(v0));

    // Zero after shifting out the high/low byte
    HWY_ALIGN uint8_t bytes[MaxLanes(du8)] = {0};
    bytes[N8 - 1] = 0x7F;
    const auto vhi = BitCast(d, Load(du8, bytes));
    bytes[N8 - 1] = 0;
    bytes[0] = 0x7F;
    const auto vlo = BitCast(d, Load(du8, bytes));
    HWY_ASSERT(AllTrue(ShiftLeftBytes<1>(vhi) == v0));
    HWY_ASSERT(AllTrue(ShiftRightBytes<1>(vlo) == v0));

    HWY_ALIGN T in[kN];
    const uint8_t* in_bytes = reinterpret_cast<const uint8_t*>(in);
    const auto v = BitCast(d, Iota(du8, 1));
    Store(v, d, in);

    HWY_ALIGN T shifted[kN];
    const uint8_t* shifted_bytes = reinterpret_cast<const uint8_t*>(shifted);

    const size_t kBlockSize = HWY_MIN(N8, 16);
    Store(ShiftLeftBytes<1>(v), d, shifted);
    for (size_t block = 0; block < N8; block += kBlockSize) {
      HWY_ASSERT_EQ(uint8_t(0), shifted_bytes[block]);
      HWY_ASSERT(BytesEqual(in_bytes + block, shifted_bytes + block + 1,
                            kBlockSize - 1));
    }

    Store(ShiftRightBytes<1>(v), d, shifted);
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

HWY_NOINLINE void TestShiftBytes() {
  ForIntegerTypes(ForGE128Vectors<TestShiftBytesT>());
}

struct TestShiftLanesT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Scalar does not define Shift*Lanes.
#if HWY_TARGET != HWY_SCALAR || HWY_IDE
    const auto v = Iota(d, T(1));
    HWY_ALIGN T lanes[MaxLanes(d)];
    {
      const auto left0 = ShiftLeftLanes<0>(v);
      const auto right0 = ShiftRightLanes<0>(v);
      HWY_ASSERT_VEC_EQ(d, left0, v);
      HWY_ASSERT_VEC_EQ(d, right0, v);
    }

    constexpr size_t kLanesPerBlock = 16 / sizeof(T);

    {
      const auto left1 = ShiftLeftLanes<1>(v);
      Store(left1, d, lanes);
      for (size_t i = 0; i < Lanes(d); ++i) {
        const T expected = (i % kLanesPerBlock) == 0 ? T(0) : T(i);
        HWY_ASSERT_EQ(expected, lanes[i]);
      }
    }
    {
      const auto right1 = ShiftRightLanes<1>(v);
      Store(right1, d, lanes);
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

HWY_NOINLINE void TestShiftLanes() {
  ForAllTypes(ForGE128Vectors<TestShiftLanesT>());
}

template <typename D, int kLane>
struct TestBroadcastR {
  HWY_NOINLINE void operator()() const {
    using T = typename D::T;
    const D d;
    constexpr size_t kN = MaxLanes(d);
    const size_t N = Lanes(d);
    HWY_ALIGN T in_lanes[kN] = {};
    constexpr size_t kBlockN = HWY_MIN(kN * sizeof(T), 16) / sizeof(T);
    // Need to set within each 128-bit block
    for (size_t block = 0; block < N; block += kBlockN) {
      in_lanes[block + kLane] = static_cast<T>(block + 1);
    }
    const auto in = Load(d, in_lanes);
    HWY_ALIGN T out_lanes[kN];
    Store(Broadcast<kLane>(in), d, out_lanes);
    for (size_t block = 0; block < N; block += kBlockN) {
      for (size_t i = 0; i < kBlockN; ++i) {
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

struct TestBroadcastT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    TestBroadcastR<D, HWY_MIN(MaxLanes(d), 16 / sizeof(T)) - 1>()();
  }
};

HWY_NOINLINE void TestBroadcast() {
  const ForPartialVectors<TestBroadcastT> test;
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

struct TestPermuteT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
#if HWY_CAP_GE256
    {
      const int32_t N = Lanes(d);
      // Too many permutations to test exhaustively; choose one with repeated
      // and cross-block indices and ensure indices do not exceed #lanes.
      HWY_ALIGN int32_t idx[kTestMaxVectorSize / sizeof(int32_t)] = {
          1,      3,      2,      2,      8 % N, 1,     7,     6,
          15 % N, 14 % N, 14 % N, 15 % N, 4,     9 % N, 8 % N, 5};
      const auto v = Iota(d, 1);
      HWY_ALIGN T expected_lanes[MaxLanes(d)];
      for (int32_t i = 0; i < N; ++i) {
        expected_lanes[i] = idx[i] + 1;  // == v[idx[i]]
      }

      const auto opaque = SetTableIndices(d, idx);
      const auto actual = TableLookupLanes(v, opaque);
      HWY_ASSERT_VEC_EQ(d, expected_lanes, actual);
    }
#elif HWY_TARGET == HWY_SCALAR
    (void)d;
#else  // 128-bit
    // Test all possible permutations.
    HWY_ALIGN int32_t idx[MaxLanes(d)];
    const auto v = Iota(d, 1);
    HWY_ALIGN T expected_lanes[MaxLanes(d)];

    const int32_t N = static_cast<int32_t>(Lanes(d));
    for (int32_t i0 = 0; i0 < N; ++i0) {
      idx[0] = i0;
      for (int32_t i1 = 0; i1 < N; ++i1) {
        idx[1] = i1;
        for (int32_t i2 = 0; i2 < N; ++i2) {
          idx[2] = i2;
          for (int32_t i3 = 0; i3 < N; ++i3) {
            idx[3] = i3;

            for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
              expected_lanes[i] = idx[i] + 1;  // == v[idx[i]]
            }

            const auto opaque = SetTableIndices(d, idx);
            const auto actual = TableLookupLanes(v, opaque);
            HWY_ASSERT_VEC_EQ(d, expected_lanes, actual);
          }
        }
      }
    }
#endif
  }
};

HWY_NOINLINE void TestPermute() {
  const ForFullVectors<TestPermuteT> test;
  test(uint32_t());
  test(int32_t());
  test(float());
}

struct TestInterleaveT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    constexpr size_t kN = MaxLanes(d);
    HWY_ALIGN T even_lanes[kN];
    HWY_ALIGN T odd_lanes[kN];
    for (size_t i = 0; i < Lanes(d); ++i) {
      even_lanes[i] = static_cast<T>(2 * i + 0);
      odd_lanes[i] = static_cast<T>(2 * i + 1);
    }
    const auto even = Load(d, even_lanes);
    const auto odd = Load(d, odd_lanes);

    HWY_ALIGN T lo_lanes[kN];
    HWY_ALIGN T hi_lanes[kN];
    Store(InterleaveLower(even, odd), d, lo_lanes);
    Store(InterleaveUpper(even, odd), d, hi_lanes);

    constexpr size_t kBlockN = 16 / sizeof(T);
    for (size_t i = 0; i < Lanes(d); ++i) {
      const size_t block = i / kBlockN;
      const size_t lo = (i % kBlockN) + block * 2 * kBlockN;
      HWY_ASSERT_EQ(T(lo), lo_lanes[i]);
      HWY_ASSERT_EQ(T(lo + kBlockN), hi_lanes[i]);
    }
  }
};

HWY_NOINLINE void TestInterleave() {
  // Not supported by HWY_SCALAR: Interleave(f32, f32) would return f32x2.
  ForAllTypes(ForGE128Vectors<TestInterleaveT>());
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
struct TestZipLowerT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    using WideT = typename MakeWide<T>::type;
    static_assert(sizeof(T) * 2 == sizeof(WideT), "Must be double-width");
    static_assert(IsSigned<T>() == IsSigned<WideT>(), "Must have same sign");
    const Simd<WideT, (MaxLanes(d) + 1) / 2> dw;
    HWY_ALIGN T even_lanes[MaxLanes(d)];
    HWY_ALIGN T odd_lanes[MaxLanes(d)];
    for (size_t i = 0; i < Lanes(d); ++i) {
      even_lanes[i] = static_cast<T>(2 * i + 0);
      odd_lanes[i] = static_cast<T>(2 * i + 1);
    }
    const auto even = Load(d, even_lanes);
    const auto odd = Load(d, odd_lanes);

    HWY_ALIGN WideT lo_lanes[MaxLanes(dw)];
    Store(ZipLower(even, odd), dw, lo_lanes);

    constexpr WideT kBlockN = static_cast<WideT>(16 / sizeof(WideT));
    for (size_t i = 0; i < Lanes(dw); ++i) {
      const size_t block = i / kBlockN;
      // Value of least-significant lane in lo-vector.
      const WideT lo =
          static_cast<WideT>(2 * (i % kBlockN) + 4 * block * kBlockN);
      const WideT kBits = static_cast<WideT>(sizeof(T) * 8);
      const WideT expected_lo =
          static_cast<WideT>((static_cast<WideT>(lo + 1) << kBits) + lo);
      HWY_ASSERT_EQ(expected_lo, lo_lanes[i]);
    }
  }
};

template <template <class> class MakeWide>
struct TestZipUpperT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    using WideT = typename MakeWide<T>::type;
    static_assert(sizeof(T) * 2 == sizeof(WideT), "Must be double-width");
    static_assert(IsSigned<T>() == IsSigned<WideT>(), "Must have same sign");
    constexpr size_t kN = MaxLanes(d);
    HWY_ALIGN T even_lanes[kN];
    HWY_ALIGN T odd_lanes[kN];
    for (size_t i = 0; i < Lanes(d); ++i) {
      even_lanes[i] = static_cast<T>(2 * i + 0);
      odd_lanes[i] = static_cast<T>(2 * i + 1);
    }
    const auto even = Load(d, even_lanes);
    const auto odd = Load(d, odd_lanes);

    const Simd<WideT, (kN + 1) / 2> dw;
    HWY_ALIGN WideT hi_lanes[MaxLanes(dw)];
    Store(ZipUpper(even, odd), dw, hi_lanes);

    constexpr WideT kBlockN = static_cast<WideT>(16 / sizeof(WideT));
    for (size_t i = 0; i < Lanes(dw); ++i) {
      const size_t block = i / kBlockN;
      const WideT lo =
          static_cast<WideT>(2 * (i % kBlockN) + 4 * block * kBlockN);
      const WideT kBits = static_cast<WideT>(sizeof(T) * 8);
      const WideT expected_hi = static_cast<WideT>(
          (static_cast<WideT>(lo + 2 * kBlockN + 1) << kBits) + lo +
          2 * kBlockN);
      HWY_ASSERT_EQ(expected_hi, hi_lanes[i]);
    }
  }
};

HWY_NOINLINE void TestZip() {
  const ForPartialVectors<TestZipLowerT<MakeWideUnsigned>, 2> lower_unsigned;
  lower_unsigned(uint8_t());
  lower_unsigned(uint16_t());
#if HWY_CAP_INTEGER64
  lower_unsigned(uint32_t());  // generates u64
#endif

  const ForPartialVectors<TestZipLowerT<MakeWideSigned>, 2> lower_signed;
  lower_signed(int8_t());
  lower_signed(int16_t());
#if HWY_CAP_INTEGER64
  lower_signed(int32_t());  // generates i64
#endif

  const ForGE128Vectors<TestZipUpperT<MakeWideUnsigned>> upper_unsigned;
  upper_unsigned(uint8_t());
  upper_unsigned(uint16_t());
#if HWY_CAP_INTEGER64
  upper_unsigned(uint32_t());  // generates u64
#endif

  const ForGE128Vectors<TestZipUpperT<MakeWideSigned>> upper_signed;
  upper_signed(int8_t());
  upper_signed(int16_t());
#if HWY_CAP_INTEGER64
  upper_signed(int32_t());  // generates i64
#endif

  // No float - concatenating f32 does not result in a f64
}

struct TestShuffleT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    RandomState rng{1234};
    const Simd<uint8_t, MaxLanes(d) * sizeof(T)> du8;
    HWY_ALIGN uint8_t in_bytes[MaxLanes(du8)];
    for (uint8_t& in_byte : in_bytes) {
      in_byte = Random32(&rng) & 0xFF;
    }
    const auto in = BitCast(d, Load(du8, in_bytes));
    HWY_ALIGN const uint8_t index_bytes[kTestMaxVectorSize] = {
        // Same index as source, multiple outputs from same input,
        // unused input (9), ascending/descending and nonconsecutive neighbors.
        0,  2,  1, 2, 15, 12, 13, 14, 6,  7,  8,  5,  4,  3,  10, 11,
        11, 10, 3, 4, 5,  8,  7,  6,  14, 13, 12, 15, 2,  1,  2,  0,
        4,  3,  2, 2, 5,  6,  7,  7,  15, 15, 15, 15, 15, 15, 0,  1};
    const auto indices = Load(du8, index_bytes);
    HWY_ALIGN T out_lanes[MaxLanes(d)];
    Store(TableLookupBytes(in, indices), d, out_lanes);
    const uint8_t* out_bytes = reinterpret_cast<const uint8_t*>(out_lanes);

    for (size_t block = 0; block + 16 <= Lanes(du8); block += 16) {
      for (size_t i = 0; i < 16; ++i) {
        const uint8_t index = index_bytes[block + i];
        const uint8_t expected = in_bytes[(block + index) % MaxLanes(du8)];
        HWY_ASSERT_EQ(expected, out_bytes[block + i]);
      }
    }
  }
};

HWY_NOINLINE void TestShuffle() {
  // Not supported by HWY_SCALAR (its vector size is always less than 16 bytes)
  ForIntegerTypes(ForGE128Vectors<TestShuffleT>());
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
    HWY_ALIGN T lanes[MaxLanes(d)];
    Store(v, d, lanes);
    const std::string name = TypeName(lanes[0], Lanes(d));
    constexpr size_t kBlockN = 16 / sizeof(T);
    for (int block = 0; block < static_cast<int>(Lanes(d)); block += kBlockN) {
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
    HWY_ALIGN T lanes[MaxLanes(d)];
    Store(v, d, lanes);
    const std::string name = TypeName(lanes[0], Lanes(d));
    constexpr size_t kBlockN = 16 / sizeof(T);
    for (int block = 0; block < static_cast<int>(Lanes(d)); block += kBlockN) {
      AssertEqual(T(block + i1), lanes[block + 1], name, filename, line);
      AssertEqual(T(block + i0), lanes[block + 0], name, filename, line);
    }
  }
};

HWY_NOINLINE void TestSpecialShuffles() {
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
struct TestCombineShiftRightR {
  template <class T, class D>
  HWY_NOINLINE void operator()(T t, D d) {
// Scalar does not define CombineShiftRightBytes.
#if HWY_TARGET != HWY_SCALAR || HWY_IDE
    const Simd<uint8_t, MaxLanes(d) * sizeof(T)> d8;
    const size_t N8 = Lanes(d8);
    const auto lo = BitCast(d, Iota(d8, 1));
    const auto hi = BitCast(d, Iota(d8, 1 + N8));

    HWY_ALIGN T lanes[MaxLanes(d)];
    Store(CombineShiftRightBytes<kBytes>(hi, lo), d, lanes);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(lanes);

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

    TestCombineShiftRightR<kBytes - 1>()(t, d);
#else
    (void)t;
    (void)d;
#endif  // #if HWY_TARGET != HWY_SCALAR
  }
};

template <>
struct TestCombineShiftRightR<0> {
  template <class T, class D>
  void operator()(T /*unused*/, D /*unused*/) {}
};

struct TestCombineShiftRightT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T t, D d) {
    TestCombineShiftRightR<15>()(t, d);
  }
};

HWY_NOINLINE void TestCombineShiftRight() {
  ForIntegerTypes(ForGE128Vectors<TestCombineShiftRightT>());
}

struct TestConcatHalvesT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Construct inputs such that interleaved halves == iota.
    const auto expected = Iota(d, 1);

    HWY_ALIGN T lo[MaxLanes(d)];
    HWY_ALIGN T hi[MaxLanes(d)];
    const size_t N = Lanes(d);
    size_t i;
    for (i = 0; i < N / 2; ++i) {
      lo[i] = static_cast<T>(1 + i);
      hi[i] = static_cast<T>(lo[i] + T(N) / 2);
    }
    for (; i < N; ++i) {
      lo[i] = hi[i] = 0;
    }
    HWY_ASSERT_VEC_EQ(d, expected, ConcatLowerLower(Load(d, hi), Load(d, lo)));

    // Same for high blocks.
    for (i = 0; i < N / 2; ++i) {
      lo[i] = hi[i] = 0;
    }
    for (; i < N; ++i) {
      lo[i] = static_cast<T>(1 + i - N / 2);
      hi[i] = static_cast<T>(lo[i] + T(N) / 2);
    }
  }
};

HWY_NOINLINE void TestConcatHalves() {
  ForAllTypes(ForGE128Vectors<TestConcatHalvesT>());
}

struct TestConcatLowerUpperT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    // Middle part of Iota(1) == Iota(1 + N / 2).
    const auto lo = Iota(d, 1);
    const auto hi = Iota(d, 1 + N);
    HWY_ASSERT_VEC_EQ(d, Iota(d, 1 + N / 2), ConcatLowerUpper(hi, lo));
  }
};

HWY_NOINLINE void TestConcatLowerUpper() {
  ForAllTypes(ForGE128Vectors<TestConcatLowerUpperT>());
}

struct TestConcatUpperLowerT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    const auto lo = Iota(d, 1);
    const auto hi = Iota(d, 1 + N);
    T expected[MaxLanes(d)];
    size_t i = 0;
    for (; i < N / 2; ++i) {
      expected[i] = static_cast<T>(1 + i);
    }
    for (; i < N; ++i) {
      expected[i] = static_cast<T>(1 + i + N);
    }
    HWY_ASSERT_VEC_EQ(d, expected, ConcatUpperLower(hi, lo));
  }
};

HWY_NOINLINE void TestConcatUpperLower() {
  ForAllTypes(ForGE128Vectors<TestConcatUpperLowerT>());
}

struct TestOddEvenT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    const auto even = Iota(d, 1);
    const auto odd = Iota(d, 1 + N);
    T expected[MaxLanes(d)];
    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>(1 + i + ((i & 1) ? N : 0));
    }
    HWY_ASSERT_VEC_EQ(d, expected, OddEven(odd, even));
  }
};

HWY_NOINLINE void TestOddEven() {
  ForAllTypes(ForGE128Vectors<TestOddEvenT>());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {

class HwySwizzleTest : public hwy::TestWithParamTarget {};

HWY_TARGET_INSTANTIATE_TEST_SUITE_P(HwySwizzleTest);

HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestLowerHalf);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestUpperHalf);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestZeroExtendVector);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestCombine);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestShiftBytes);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestShiftLanes);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestBroadcast);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestPermute);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestInterleave);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestZip);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestShuffle);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestSpecialShuffles);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestCombineShiftRight);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestConcatHalves);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestConcatLowerUpper);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestConcatUpperLower);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestOddEven);

}  // namespace hwy
#endif
