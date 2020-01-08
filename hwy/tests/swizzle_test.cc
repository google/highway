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

#ifndef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/swizzle_test.cc"

#include "hwy/tests/test_util.h"
struct SwizzleTest {
  HWY_DECLARE(void, ())
};
TEST(HwySwizzleTest, Run) { hwy::RunTests<SwizzleTest>(); }

#endif  // HWY_TARGET_INCLUDE
#include "hwy/tests/test_target_util.h"

namespace hwy {
namespace HWY_NAMESPACE {
namespace {

constexpr HWY_FULL(uint8_t) du8;
constexpr HWY_FULL(uint16_t) du16;
constexpr HWY_FULL(uint32_t) du32;
constexpr HWY_FULL(uint64_t) du64;
constexpr HWY_FULL(int8_t) di8;
constexpr HWY_FULL(int16_t) di16;
constexpr HWY_FULL(int32_t) di32;
constexpr HWY_FULL(int64_t) di64;
constexpr HWY_FULL(float) df;
constexpr HWY_FULL(double) dd;

template <class D>
HWY_NOINLINE HWY_ATTR void TestShiftBytesT(D d) {
#if HWY_BITS != 0 || HWY_IDE
  using T = typename D::T;
  // Zero remains zero
  const auto v0 = Zero(d);
  HWY_ASSERT_VEC_EQ(d, v0, ShiftLeftBytes<1>(v0));
  HWY_ASSERT_VEC_EQ(d, v0, ShiftRightBytes<1>(v0));

  // Zero after shifting out the high/low byte
  HWY_ALIGN uint8_t bytes[du8.N] = {0};
  bytes[du8.N - 1] = 0x7F;
  const auto vhi = BitCast(d, Load(du8, bytes));
  bytes[du8.N - 1] = 0;
  bytes[0] = 0x7F;
  const auto vlo = BitCast(d, Load(du8, bytes));
  HWY_ASSERT_EQ(true, ext::AllTrue(ShiftLeftBytes<1>(vhi) == v0));
  HWY_ASSERT_EQ(true, ext::AllTrue(ShiftRightBytes<1>(vlo) == v0));

  HWY_ALIGN T in[d.N];
  const uint8_t* in_bytes = reinterpret_cast<const uint8_t*>(in);
  const auto v = BitCast(d, Iota(du8, 1));
  Store(v, d, in);

  // Shifting by one lane is the same as shifting #bytes
  HWY_ASSERT_VEC_EQ(d, ShiftLeftLanes<1>(v), ShiftLeftBytes<sizeof(T)>(v));
  HWY_ASSERT_VEC_EQ(d, ShiftRightLanes<1>(v), ShiftRightBytes<sizeof(T)>(v));
  // Two lanes, we can only try to shift lanes if there are at least two
  // lanes.
  if (D::N > 2) {
    // If there are only two lanes we just define the one lane version to
    // avoid running into static_asserts.
    const size_t shift_bytes = D::N > 2 ? 2 * sizeof(T) : sizeof(T);
    const size_t shift_lanes = D::N > 2 ? 2 : 1;
    HWY_ASSERT_VEC_EQ(d, ShiftLeftLanes<shift_lanes>(v),
                      ShiftLeftBytes<shift_bytes>(v));
    HWY_ASSERT_VEC_EQ(d, ShiftRightLanes<shift_lanes>(v),
                      ShiftRightBytes<shift_bytes>(v));
  }

  HWY_ALIGN T shifted[d.N];
  const uint8_t* shifted_bytes = reinterpret_cast<const uint8_t*>(shifted);

  const size_t kBlockSize = HWY_MIN(du8.N, 16);
  Store(ShiftLeftBytes<1>(v), d, shifted);
  for (size_t block = 0; block < du8.N; block += kBlockSize) {
    HWY_ASSERT_EQ(uint8_t(0), shifted_bytes[block]);
    HWY_ASSERT_EQ(true, BytesEqual(in_bytes + block, shifted_bytes + block + 1,
                                   kBlockSize - 1));
  }

  Store(ShiftRightBytes<1>(v), d, shifted);
  for (size_t block = 0; block < du8.N; block += kBlockSize) {
    HWY_ASSERT_EQ(uint8_t(0), shifted_bytes[block + kBlockSize - 1]);
    HWY_ASSERT_EQ(true, BytesEqual(in_bytes + block + 1, shifted_bytes + block,
                                   kBlockSize - 1));
  }
#else
  (void)d;
#endif
}

template <typename D, int kLane>
struct TestBroadcastR {
  HWY_NOINLINE HWY_ATTR void operator()() const {
    using T = typename D::T;
    const D d;
    HWY_ALIGN T in_lanes[d.N] = {};
    constexpr size_t kBlockN = HWY_MIN(d.N * sizeof(T), 16) / sizeof(T);
    // Need to set within each 128-bit block
    for (size_t block = 0; block < d.N; block += kBlockN) {
      in_lanes[block + kLane] = block + 1;
    }
    const auto in = Load(d, in_lanes);
    HWY_ALIGN T out_lanes[d.N];
    Store(Broadcast<kLane>(in), d, out_lanes);
    for (size_t block = 0; block < d.N; block += kBlockN) {
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

template <class D>
HWY_NOINLINE HWY_ATTR void TestBroadcastT(D d) {
  using T = typename D::T;
  TestBroadcastR<D, HWY_MIN(d.N, 16 / sizeof(T)) - 1>()();
}

HWY_NOINLINE HWY_ATTR void TestBroadcast() {
  // No u8.
  TestBroadcastT(du16);
  TestBroadcastT(du32);
#if HWY_HAS_INT64
  TestBroadcastT(du64);
#endif
  // No i8.
  TestBroadcastT(di16);
#if HWY_HAS_INT64
  TestBroadcastT(di64);
#endif
  HWY_FOREACH_F(TestBroadcastT);
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestPermuteT(D d) {
#if HWY_BITS >= 256 || HWY_IDE
  {
    using T = typename D::T;
    // Too many permutations to test exhaustively; choose one with repeated
    // and cross-block indices and ensure indices do not exceed #lanes.
    HWY_ALIGN int32_t idx[kTestMaxVectorSize / sizeof(int32_t)] = {
        1,        3,        2,        2,        8 % d.N, 1,       7,       6,
        15 % d.N, 14 % d.N, 14 % d.N, 15 % d.N, 4,       9 % d.N, 8 % d.N, 5};
    const auto v = Iota(d, 1);
    HWY_ALIGN T expected_lanes[d.N];
    for (size_t i = 0; i < d.N; ++i) {
      expected_lanes[i] = idx[i] + 1;  // == v[idx[i]]
    }

    const auto opaque = SetTableIndices(d, idx);
    const auto actual = TableLookupLanes(v, opaque);
    HWY_ASSERT_VEC_EQ(d, expected_lanes, actual);
  }
#endif
#if HWY_BITS == 128 || HWY_IDE
  using T = typename D::T;
  // Test all possible permutations.
  HWY_ALIGN int32_t idx[d.N];
  const auto v = Iota(d, 1);
  HWY_ALIGN T expected_lanes[d.N];

  const int32_t N = static_cast<int32_t>(d.N);
  for (int32_t i0 = 0; i0 < N; ++i0) {
    idx[0] = i0;
    for (int32_t i1 = 0; i1 < N; ++i1) {
      idx[1] = i1;
      for (int32_t i2 = 0; i2 < N; ++i2) {
        idx[2] = i2;
        for (int32_t i3 = 0; i3 < N; ++i3) {
          idx[3] = i3;

          for (size_t i = 0; i < d.N; ++i) {
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
#if HWY_BITS == 0
  (void)d;
#endif
}

HWY_NOINLINE HWY_ATTR void TestPermute() {
  // Only uif32.
  TestPermuteT(du32);
  TestPermuteT(di32);
  TestPermuteT(df);
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestInterleave(D d) {
// Not supported by scalar.h: zip(f32, f32) would need to return f32x2.
#if HWY_BITS != 0 || HWY_IDE
  using T = typename D::T;
  HWY_ALIGN T even_lanes[d.N];
  HWY_ALIGN T odd_lanes[d.N];
  for (size_t i = 0; i < d.N; ++i) {
    even_lanes[i] = 2 * i + 0;
    odd_lanes[i] = 2 * i + 1;
  }
  const auto even = Load(d, even_lanes);
  const auto odd = Load(d, odd_lanes);

  HWY_ALIGN T lo_lanes[d.N];
  HWY_ALIGN T hi_lanes[d.N];
  Store(InterleaveLo(even, odd), d, lo_lanes);
  Store(InterleaveHi(even, odd), d, hi_lanes);

  constexpr size_t kBlockN = 16 / sizeof(T);
  for (size_t i = 0; i < d.N; ++i) {
    const size_t block = i / kBlockN;
    const size_t lo = (i % kBlockN) + block * 2 * kBlockN;
    HWY_ASSERT_EQ(T(lo), lo_lanes[i]);
    HWY_ASSERT_EQ(T(lo + kBlockN), hi_lanes[i]);
  }
#else
  (void)d;
#endif
}

template <typename T, typename WideT>
HWY_NOINLINE HWY_ATTR void TestZipT() {
  static_assert(sizeof(T) * 2 == sizeof(WideT), "Expected 2x size");
  static_assert(IsSigned<T>() == IsSigned<WideT>(), "Should have same sign");
  const HWY_FULL(T) d;
  const HWY_FULL(WideT) dw;
  HWY_ALIGN T even_lanes[d.N];
  HWY_ALIGN T odd_lanes[d.N];
  for (size_t i = 0; i < d.N; ++i) {
    even_lanes[i] = static_cast<T>(2 * i + 0);
    odd_lanes[i] = static_cast<T>(2 * i + 1);
  }
  const auto even = Load(d, even_lanes);
  const auto odd = Load(d, odd_lanes);

  HWY_ALIGN WideT lo_lanes[dw.N];
  Store(ZipLo(even, odd), dw, lo_lanes);
#if HWY_BITS != 0
  HWY_ALIGN WideT hi_lanes[dw.N];
  Store(ZipHi(even, odd), dw, hi_lanes);
#endif

  constexpr WideT kBlockN = static_cast<WideT>(16 / sizeof(WideT));
  for (size_t i = 0; i < dw.N; ++i) {
    const size_t block = i / kBlockN;
    // Value of least-significant lane in lo-vector.
    const WideT lo =
        static_cast<WideT>(2 * (i % kBlockN) + 4 * block * kBlockN);
    const WideT kBits = static_cast<WideT>(sizeof(T) * 8);
    const WideT expected_lo =
        static_cast<WideT>((static_cast<WideT>(lo + 1) << kBits) + lo);

    HWY_ASSERT_EQ(expected_lo, lo_lanes[i]);
#if HWY_BITS != 0
    const WideT expected_hi = static_cast<WideT>(
        (static_cast<WideT>(lo + 2 * kBlockN + 1) << kBits) + lo + 2 * kBlockN);
    HWY_ASSERT_EQ(expected_hi, hi_lanes[i]);
#endif
  }
}

HWY_NOINLINE HWY_ATTR void TestZip() {
  TestZipT<uint8_t, uint16_t>();
  TestZipT<uint16_t, uint32_t>();
#if HWY_HAS_INT64
  TestZipT<uint32_t, uint64_t>();
#endif
  // No 64-bit nor float.
  TestZipT<int8_t, int16_t>();
  TestZipT<int16_t, int32_t>();
#if HWY_HAS_INT64
  TestZipT<int32_t, int64_t>();
#endif
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestShuffleT(D d) {
// Not supported by scalar.h (its vector size is always less than 16 bytes)
#if HWY_BITS != 0 || HWY_IDE
  using T = typename D::T;
  RandomState rng{1234};
  constexpr size_t N8 = HWY_FULL(uint8_t)::N;
  HWY_ALIGN uint8_t in_bytes[N8];
  for (size_t i = 0; i < N8; ++i) {
    in_bytes[i] = Random32(&rng) & 0xFF;
  }
  const auto in = Load(du8, in_bytes);
  HWY_ALIGN const uint8_t index_bytes[kTestMaxVectorSize] = {
      // Same index as source, multiple outputs from same input,
      // unused input (9), ascending/descending and nonconsecutive neighbors.
      0,  2,  1, 2, 15, 12, 13, 14, 6,  7,  8,  5,  4,  3,  10, 11,
      11, 10, 3, 4, 5,  8,  7,  6,  14, 13, 12, 15, 2,  1,  2,  0,
      4,  3,  2, 2, 5,  6,  7,  7,  15, 15, 15, 15, 15, 15, 0,  1};
  const auto indices = Load(du8, index_bytes);
  HWY_ALIGN T out_lanes[d.N];
  Store(TableLookupBytes(BitCast(d, in), indices), d, out_lanes);
  const uint8_t* out_bytes = reinterpret_cast<const uint8_t*>(out_lanes);

  for (size_t block = 0; block < N8; block += 16) {
    for (size_t i = 0; i < 16; ++i) {
      const uint8_t expected = in_bytes[block + index_bytes[block + i]];
      HWY_ASSERT_EQ(expected, out_bytes[block + i]);
    }
  }
#else
  (void)d;
#endif
}

template <class D, int kBytes>
struct TestCombineShiftRightR {
  HWY_NOINLINE HWY_ATTR void operator()() const {
#if HWY_BITS != 0 || HWY_IDE
    using T = typename D::T;
    const D d;
    const HWY_FULL(uint8_t) d8;
    const auto lo = BitCast(d, Iota(d8, 1));
    const auto hi = BitCast(d, Iota(d8, 1 + d8.N));

    HWY_ALIGN T lanes[D::N];
    Store(CombineShiftRightBytes<kBytes>(hi, lo), d, lanes);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(lanes);

    const size_t kBlockSize = 16;
    for (size_t i = 0; i < d8.N; ++i) {
      const size_t block = i / kBlockSize;
      const size_t lane = i % kBlockSize;
      const size_t first_lo = block * kBlockSize;
      const size_t idx = lane + kBytes;
      const size_t offset = (idx < kBlockSize) ? 0 : d8.N - kBlockSize;
      const bool at_end = idx >= 2 * kBlockSize;
      const uint8_t expected = at_end ? 0 : (first_lo + idx + 1 + offset);
      HWY_ASSERT_EQ(expected, bytes[i]);
    }

    TestCombineShiftRightR<D, kBytes - 1>()();
#endif
  }
};

template <class D>
struct TestCombineShiftRightR<D, 0> {
  HWY_ATTR void operator()() const {}
};

template <class D>
HWY_NOINLINE HWY_ATTR void TestCombineShiftRightT(D /*d*/) {
  TestCombineShiftRightR<D, 15>()();
}

#if HWY_BITS != 0 || HWY_IDE

template <class D, class V>
HWY_NOINLINE HWY_ATTR void VerifyLanes32(D d, V v, const int i3, const int i2,
                                         const int i1, const int i0,
                                         const char* filename, const int line) {
  using T = typename D::T;
  HWY_ALIGN T lanes[d.N];
  Store(v, d, lanes);
  constexpr size_t kBlockN = 16 / sizeof(T);
  for (size_t block = 0; block < d.N; block += kBlockN) {
    AssertEqual(T(block + i3), lanes[block + 3], filename, line);
    AssertEqual(T(block + i2), lanes[block + 2], filename, line);
    AssertEqual(T(block + i1), lanes[block + 1], filename, line);
    AssertEqual(T(block + i0), lanes[block + 0], filename, line);
  }
}

template <class D, class V>
HWY_NOINLINE HWY_ATTR void VerifyLanes64(D d, V v, const int i1, const int i0,
                                         const char* filename, const int line) {
  using T = typename D::T;
  HWY_ALIGN T lanes[d.N];
  Store(v, d, lanes);
  constexpr size_t kBlockN = 16 / sizeof(T);
  for (size_t block = 0; block < d.N; block += kBlockN) {
    AssertEqual(T(block + i1), lanes[block + 1], filename, line);
    AssertEqual(T(block + i0), lanes[block + 0], filename, line);
  }
}

#define VERIFY_LANES_32(d, v, i3, i2, i1, i0) \
  VerifyLanes32((d), (v), (i3), (i2), (i1), (i0), __FILE__, __LINE__)

#define VERIFY_LANES_64(d, v, i1, i0) \
  VerifyLanes64((d), (v), (i1), (i0), __FILE__, __LINE__)

template <class D>
HWY_NOINLINE HWY_ATTR void TestSpecialShuffle32(D d) {
  const auto v = Iota(d, 0);
  VERIFY_LANES_32(d, Shuffle2301(v), 2, 3, 0, 1);
  VERIFY_LANES_32(d, Shuffle1032(v), 1, 0, 3, 2);
  VERIFY_LANES_32(d, Shuffle0321(v), 0, 3, 2, 1);
  VERIFY_LANES_32(d, Shuffle2103(v), 2, 1, 0, 3);
  VERIFY_LANES_32(d, Shuffle0123(v), 0, 1, 2, 3);
}

#if HWY_HAS_INT64
template <class D>
HWY_NOINLINE HWY_ATTR void TestSpecialShuffle64(D d) {
  const auto v = Iota(d, 0);
  VERIFY_LANES_64(d, Shuffle01(v), 0, 1);
}
#endif

#endif

HWY_NOINLINE HWY_ATTR void TestSpecialShuffles() {
#if HWY_BITS != 0 || HWY_IDE
  TestSpecialShuffle32(di32);
#if HWY_HAS_INT64
  TestSpecialShuffle64(di64);
#endif
  // Can't use HWY_FOREACH_F, function depends on lane type
  TestSpecialShuffle32(df);
#if HWY_HAS_DOUBLE
  TestSpecialShuffle64(dd);
#endif
#endif
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestConcatHalves(D d) {
#if HWY_BITS != 0 || HWY_IDE
  using T = typename D::T;
  // Construct inputs such that interleaved halves == iota.
  const auto expected = Iota(d, 1);

  HWY_ALIGN T lo[d.N];
  HWY_ALIGN T hi[d.N];
  size_t i;
  for (i = 0; i < d.N / 2; ++i) {
    lo[i] = 1 + i;
    hi[i] = lo[i] + d.N / 2;
  }
  for (; i < d.N; ++i) {
    lo[i] = hi[i] = 0;
  }
  HWY_ASSERT_VEC_EQ(d, expected, ConcatLoLo(Load(d, hi), Load(d, lo)));

  // Same for high blocks.
  for (i = 0; i < d.N / 2; ++i) {
    lo[i] = hi[i] = 0;
  }
  for (; i < d.N; ++i) {
    lo[i] = 1 + i - d.N / 2;
    hi[i] = lo[i] + d.N / 2;
  }
  HWY_ASSERT_VEC_EQ(d, expected, ConcatHiHi(Load(d, hi), Load(d, lo)));
#else
  (void)d;
#endif
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestConcatLoHi(D d) {
#if HWY_BITS != 0 || HWY_IDE
  // Middle part of Iota(1) == Iota(1 + d.N / 2).
  const auto lo = Iota(d, 1);
  const auto hi = Iota(d, 1 + d.N);
  HWY_ASSERT_VEC_EQ(d, Iota(d, 1 + d.N / 2), ConcatLoHi(hi, lo));
#else
  (void)d;
#endif
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestConcatHiLo(D d) {
#if HWY_BITS != 0 || HWY_IDE
  using T = typename D::T;
  const auto lo = Iota(d, 1);
  const auto hi = Iota(d, 1 + d.N);
  T expected[d.N];
  size_t i = 0;
  for (; i < d.N / 2; ++i) {
    expected[i] = 1 + i;
  }
  for (; i < d.N; ++i) {
    expected[i] = 1 + i + d.N;
  }
  HWY_ASSERT_VEC_EQ(d, expected, ConcatHiLo(hi, lo));
#else
  (void)d;
#endif
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestOddEven(D d) {
#if HWY_BITS != 0 || HWY_IDE
  using T = typename D::T;
  const auto even = Iota(d, 1);
  const auto odd = Iota(d, 1 + d.N);
  T expected[d.N];
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = 1 + i + ((i & 1) ? d.N : 0);
  }
  HWY_ASSERT_VEC_EQ(d, expected, OddEven(odd, even));
#else
  (void)d;
#endif
}

HWY_NOINLINE HWY_ATTR void TestSwizzle() {
  (void)dd;
  (void)di64;
  (void)du64;

  HWY_FOREACH_UI(TestShiftBytesT);
  TestBroadcast();
  HWY_FOREACH_UIF(TestInterleave);
  TestPermute();
  TestZip();
  HWY_FOREACH_UI(TestShuffleT);

  HWY_FOREACH_UI(TestCombineShiftRightT);
  TestSpecialShuffles();
  HWY_FOREACH_UIF(TestConcatHalves);
  HWY_FOREACH_UIF(TestConcatLoHi);
  HWY_FOREACH_UIF(TestConcatHiLo);
  HWY_FOREACH_UIF(TestOddEven);
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy

// Instantiate for the current target.
void SwizzleTest::HWY_FUNC() { hwy::HWY_NAMESPACE::TestSwizzle(); }
