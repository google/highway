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

#include "highway/swizzle_test.h"
#include "highway/test_target_util.h"

namespace jxl {
namespace SIMD_NAMESPACE {
namespace {

struct TestShiftBytesT {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
#if SIMD_BITS != 0
    const SIMD_FULL(uint8_t) d8;

    // Zero remains zero
    const auto v0 = setzero(d);
    SIMD_ASSERT_VEC_EQ(d, v0, shift_left_bytes<1>(v0));
    SIMD_ASSERT_VEC_EQ(d, v0, shift_right_bytes<1>(v0));

    // Zero after shifting out the high/low byte
    SIMD_ALIGN uint8_t bytes[d8.N] = {0};
    bytes[d8.N - 1] = 0x7F;
    const auto vhi = bit_cast(d, load(d8, bytes));
    bytes[d8.N - 1] = 0;
    bytes[0] = 0x7F;
    const auto vlo = bit_cast(d, load(d8, bytes));
    ASSERT_EQ(true, ext::all_zero(shift_left_bytes<1>(vhi)));
    ASSERT_EQ(true, ext::all_zero(shift_right_bytes<1>(vlo)));

    SIMD_ALIGN T in[d.N];
    const uint8_t* in_bytes = reinterpret_cast<const uint8_t*>(in);
    const auto v = bit_cast(d, iota(d8, 1));
    store(v, d, in);

    // Shifting by one lane is the same as shifting #bytes
    SIMD_ASSERT_VEC_EQ(d, shift_left_lanes<1>(v),
                       shift_left_bytes<sizeof(T)>(v));
    SIMD_ASSERT_VEC_EQ(d, shift_right_lanes<1>(v),
                       shift_right_bytes<sizeof(T)>(v));
    // Two lanes, we can only try to shift lanes if there are at least two
    // lanes.
    if (D::N > 2) {
      // If there are only two lanes we just define the one lane version to
      // avoid running into static_asserts.
      const size_t shift_bytes = D::N > 2 ? 2 * sizeof(T) : sizeof(T);
      const size_t shift_lanes = D::N > 2 ? 2 : 1;
      SIMD_ASSERT_VEC_EQ(d, shift_left_lanes<shift_lanes>(v),
                         shift_left_bytes<shift_bytes>(v));
      SIMD_ASSERT_VEC_EQ(d, shift_right_lanes<shift_lanes>(v),
                         shift_right_bytes<shift_bytes>(v));
    }

    SIMD_ALIGN T shifted[d.N];
    const uint8_t* shifted_bytes = reinterpret_cast<const uint8_t*>(shifted);

    const size_t kBlockSize = SIMD_MIN(d8.N, 16);
    store(shift_left_bytes<1>(v), d, shifted);
    for (size_t block = 0; block < d8.N; block += kBlockSize) {
      ASSERT_EQ(uint8_t(0), shifted_bytes[block]);
      ASSERT_EQ(true, BytesEqual(in_bytes + block, shifted_bytes + block + 1,
                                 kBlockSize - 1));
    }

    store(shift_right_bytes<1>(v), d, shifted);
    for (size_t block = 0; block < d8.N; block += kBlockSize) {
      ASSERT_EQ(uint8_t(0), shifted_bytes[block + kBlockSize - 1]);
      ASSERT_EQ(true, BytesEqual(in_bytes + block + 1, shifted_bytes + block,
                                 kBlockSize - 1));
    }
#endif
  }
};

SIMD_ATTR void TestShiftBytes() {
  ForeachUnsignedLaneType<TestShiftBytesT>();
  ForeachSignedLaneType<TestShiftBytesT>();
  // No float.
}

template <typename T, int kLane>
struct TestBroadcastR {
  SIMD_ATTR void operator()() const {
    const SIMD_FULL(T) d;
    SIMD_ALIGN T in_lanes[d.N] = {};
    constexpr size_t kVecN = SIMD_FULL(T)::N;
    constexpr size_t kBlockN = SIMD_MIN(kVecN * sizeof(T), 16) / sizeof(T);
    // Need to set within each 128-bit block
    for (size_t block = 0; block < d.N; block += kBlockN) {
      in_lanes[block + kLane] = block + 1;
    }
    const auto in = load(d, in_lanes);
    SIMD_ALIGN T out_lanes[d.N];
    store(broadcast<kLane>(in), d, out_lanes);
    for (size_t block = 0; block < d.N; block += kBlockN) {
      for (size_t i = 0; i < kBlockN; ++i) {
        SIMD_ASSERT_EQ(T(block + 1), out_lanes[block + i]);
      }
    }

    TestBroadcastR<T, kLane - 1>()();
  }
};

template <typename T>
struct TestBroadcastR<T, -1> {
  void operator()() const {}
};

template <typename T>
SIMD_ATTR void TestBroadcastT() {
  constexpr size_t kVecN = SIMD_FULL(T)::N;
  TestBroadcastR<T, SIMD_MIN(kVecN, 16 / sizeof(T)) - 1>()();
}

SIMD_ATTR void TestBroadcast() {
  // No u8.
  TestBroadcastT<uint16_t>();
  TestBroadcastT<uint32_t>();
  TestBroadcastT<uint64_t>();
  // No i8.
  TestBroadcastT<int16_t>();
  TestBroadcastT<int64_t>();
  TestBroadcastT<float>();
  TestBroadcastT<double>();
}

struct TestPermuteT {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
#if SIMD_BITS >= 256
    // Too many permutations to test exhaustively; choose one with repeated and
    // cross-block indices and ensure indices do not exceed #lanes.
    SIMD_ALIGN int32_t idx[kTestMaxVectorSize / sizeof(int32_t)] = {
        1,        3,        2,        2,        8 % d.N, 1,       7,       6,
        15 % d.N, 14 % d.N, 14 % d.N, 15 % d.N, 4,       9 % d.N, 8 % d.N, 5};
    const auto v = iota(d, 1);
    SIMD_ALIGN T expected_lanes[d.N];
    for (size_t i = 0; i < d.N; ++i) {
      expected_lanes[i] = idx[i] + 1;  // == v[idx[i]]
    }

    const auto opaque = set_table_indices(d, idx);
    const auto actual = table_lookup_lanes(v, opaque);
    SIMD_ASSERT_VEC_EQ(d, expected_lanes, actual);
#elif SIMD_BITS == 128
    // Test all possible permutations.
    SIMD_ALIGN int32_t idx[d.N];
    const auto v = iota(d, 1);
    SIMD_ALIGN T expected_lanes[d.N];

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

            const auto opaque = set_table_indices(d, idx);
            const auto actual = table_lookup_lanes(v, opaque);
            SIMD_ASSERT_VEC_EQ(d, expected_lanes, actual);
          }
        }
      }
    }
#endif
  }
};

SIMD_ATTR void TestPermute() {
  // Only uif32.
  Call<TestPermuteT, uint32_t>();
  Call<TestPermuteT, int32_t>();
  Call<TestPermuteT, float>();
}

struct TestInterleave {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
// Not supported by scalar.h: zip(f32, f32) would need to return f32x2.
#if SIMD_BITS != 0
    SIMD_ALIGN T even_lanes[d.N];
    SIMD_ALIGN T odd_lanes[d.N];
    for (size_t i = 0; i < d.N; ++i) {
      even_lanes[i] = 2 * i + 0;
      odd_lanes[i] = 2 * i + 1;
    }
    const auto even = load(d, even_lanes);
    const auto odd = load(d, odd_lanes);

    SIMD_ALIGN T lo_lanes[d.N];
    SIMD_ALIGN T hi_lanes[d.N];
    store(interleave_lo(even, odd), d, lo_lanes);
    store(interleave_hi(even, odd), d, hi_lanes);

    constexpr size_t kBlockN = 16 / sizeof(T);
    for (size_t i = 0; i < d.N; ++i) {
      const size_t block = i / kBlockN;
      const size_t lo = (i % kBlockN) + block * 2 * kBlockN;
      ASSERT_EQ(T(lo), lo_lanes[i]);
      ASSERT_EQ(T(lo + kBlockN), hi_lanes[i]);
    }
#endif
  }
};

template <typename T, typename WideT>
struct TestZipT {
  SIMD_ATTR void operator()() const {
    const SIMD_FULL(T) d;
    const SIMD_FULL(WideT) dw;
    SIMD_ALIGN T even_lanes[d.N];
    SIMD_ALIGN T odd_lanes[d.N];
    for (size_t i = 0; i < d.N; ++i) {
      even_lanes[i] = 2 * i + 0;
      odd_lanes[i] = 2 * i + 1;
    }
    const auto even = load(d, even_lanes);
    const auto odd = load(d, odd_lanes);

    SIMD_ALIGN WideT lo_lanes[dw.N];
    SIMD_ALIGN WideT hi_lanes[dw.N];
    store(zip_lo(even, odd), dw, lo_lanes);
    store(zip_hi(even, odd), dw, hi_lanes);

    constexpr size_t kBlockN = 16 / sizeof(WideT);
    for (size_t i = 0; i < dw.N; ++i) {
      const size_t block = i / kBlockN;
      const size_t lo = (i % kBlockN) + block * 2 * kBlockN;
      const size_t bits = sizeof(T) * 8;
      const size_t expected_lo = ((lo + 1) << bits) + lo;
      const size_t expected_hi = ((lo + kBlockN + 1) << bits) + lo + kBlockN;
      SIMD_ASSERT_EQ(T(expected_lo), lo_lanes[i]);
      SIMD_ASSERT_EQ(T(expected_hi), hi_lanes[i]);
    }
  }
};

SIMD_ATTR void TestZip() {
  TestZipT<uint8_t, uint16_t>();
  TestZipT<uint16_t, uint32_t>();
  TestZipT<uint32_t, uint64_t>();
  // No 64-bit nor float.
  TestZipT<int8_t, int16_t>();
  TestZipT<int16_t, int32_t>();
  TestZipT<int32_t, int64_t>();
}

struct TestShuffleT {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
// Not supported by scalar.h (its vector size is always less than 16 bytes)
#if SIMD_BITS != 0
    RandomState rng{1234};
    const SIMD_FULL(uint8_t) d8;
    constexpr size_t N8 = SIMD_FULL(uint8_t)::N;
    SIMD_ALIGN uint8_t in_bytes[N8];
    for (size_t i = 0; i < N8; ++i) {
      in_bytes[i] = Random32(&rng) & 0xFF;
    }
    const auto in = load(d8, in_bytes);
    SIMD_ALIGN const uint8_t index_bytes[kTestMaxVectorSize] = {
        // Same index as source, multiple outputs from same input,
        // unused input (9), ascending/descending and nonconsecutive neighbors.
        0,  2,  1, 2, 15, 12, 13, 14, 6,  7,  8,  5,  4,  3,  10, 11,
        11, 10, 3, 4, 5,  8,  7,  6,  14, 13, 12, 15, 2,  1,  2,  0,
        4,  3,  2, 2, 5,  6,  7,  7,  15, 15, 15, 15, 15, 15, 0,  1};
    const auto indices = load(d8, index_bytes);
    SIMD_ALIGN T out_lanes[d.N];
    store(table_lookup_bytes(bit_cast(d, in), indices), d, out_lanes);
    const uint8_t* out_bytes = reinterpret_cast<const uint8_t*>(out_lanes);

    for (size_t block = 0; block < N8; block += 16) {
      for (size_t i = 0; i < 16; ++i) {
        const uint8_t expected = in_bytes[block + index_bytes[block + i]];
        ASSERT_EQ(expected, out_bytes[block + i]);
      }
    }
#endif
  }
};

SIMD_ATTR void TestShuffle() {
  ForeachUnsignedLaneType<TestShuffleT>();
  ForeachSignedLaneType<TestShuffleT>();
  // No float.
}

template <typename T, class D, int kBytes>
struct TestExtractR {
  SIMD_ATTR void operator()() const {
#if SIMD_BITS != 0
    const D d;
    const SIMD_FULL(uint8_t) d8;
    const auto lo = bit_cast(d, iota(d8, 1));
    const auto hi = bit_cast(d, iota(d8, 1 + d8.N));

    SIMD_ALIGN T lanes[D::N];
    store(combine_shift_right_bytes<kBytes>(hi, lo), d, lanes);
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
      ASSERT_EQ(expected, bytes[i]);
    }

    TestExtractR<T, D, kBytes - 1>()();
#endif
  }
};

template <typename T, class D>
struct TestExtractR<T, D, 0> {
  SIMD_ATTR void operator()() const {}
};

struct TestExtractT {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    TestExtractR<T, D, 15>()();
  }
};

SIMD_ATTR void TestExtract() {
  ForeachUnsignedLaneType<TestExtractT>();
  ForeachSignedLaneType<TestExtractT>();
  // No float.
}

#if SIMD_BITS != 0

#define VERIFY_LANES_32(d, v, i3, i2, i1, i0)        \
  do {                                               \
    SCOPED_TRACE("On a call to VerifyLanes32:");     \
    VerifyLanes32((d), (v), (i3), (i2), (i1), (i0)); \
  } while (0)
template <class D, class V>
SIMD_ATTR void VerifyLanes32(D d, V v, const int i3, const int i2, const int i1,
                             const int i0) {
  using T = typename D::T;
  SIMD_ALIGN T lanes[d.N];
  store(v, d, lanes);
  constexpr size_t kBlockN = 16 / sizeof(T);
  for (size_t block = 0; block < d.N; block += kBlockN) {
    ASSERT_EQ(T(block + i3), lanes[block + 3]);
    ASSERT_EQ(T(block + i2), lanes[block + 2]);
    ASSERT_EQ(T(block + i1), lanes[block + 1]);
    ASSERT_EQ(T(block + i0), lanes[block + 0]);
  }
}

#define VERIFY_LANES_64(d, v, i1, i0)            \
  do {                                           \
    SCOPED_TRACE("On a call to VerifyLanes64:"); \
    VerifyLanes64((d), (v), (i1), (i0));         \
  } while (0)
template <class D, class V>
SIMD_ATTR void VerifyLanes64(D d, V v, const int i1, const int i0) {
  using T = typename D::T;
  SIMD_ALIGN T lanes[d.N];
  store(v, d, lanes);
  constexpr size_t kBlockN = 16 / sizeof(T);
  for (size_t block = 0; block < d.N; block += kBlockN) {
    ASSERT_EQ(T(block + i1), lanes[block + 1]);
    ASSERT_EQ(T(block + i0), lanes[block + 0]);
  }
}

struct TestSpecialShuffle32 {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v = iota(d, 0);
    VERIFY_LANES_32(d, shuffle_1032(v), 1, 0, 3, 2);
    VERIFY_LANES_32(d, shuffle_0321(v), 0, 3, 2, 1);
    VERIFY_LANES_32(d, shuffle_2103(v), 2, 1, 0, 3);
    VERIFY_LANES_32(d, shuffle_0123(v), 0, 1, 2, 3);
  }
};

struct TestSpecialShuffle64 {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v = iota(d, 0);
    VERIFY_LANES_64(d, shuffle_01(v), 0, 1);
  }
};

#endif

SIMD_ATTR void TestSpecialShuffles() {
#if SIMD_BITS != 0
  Call<TestSpecialShuffle32, int32_t>();
  Call<TestSpecialShuffle64, int64_t>();
  Call<TestSpecialShuffle32, float>();
  Call<TestSpecialShuffle64, double>();
#endif
}

struct TestConcatHalves {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
#if SIMD_BITS != 0
    // Construct inputs such that interleaved halves == iota.
    const auto expected = iota(d, 1);

    SIMD_ALIGN T lo[d.N];
    SIMD_ALIGN T hi[d.N];
    size_t i;
    for (i = 0; i < d.N / 2; ++i) {
      lo[i] = 1 + i;
      hi[i] = lo[i] + d.N / 2;
    }
    for (; i < d.N; ++i) {
      lo[i] = hi[i] = 0;
    }
    SIMD_ASSERT_VEC_EQ(d, expected, concat_lo_lo(load(d, hi), load(d, lo)));

    // Same for high blocks.
    for (i = 0; i < d.N / 2; ++i) {
      lo[i] = hi[i] = 0;
    }
    for (; i < d.N; ++i) {
      lo[i] = 1 + i - d.N / 2;
      hi[i] = lo[i] + d.N / 2;
    }
    SIMD_ASSERT_VEC_EQ(d, expected, concat_hi_hi(load(d, hi), load(d, lo)));
#endif
  }
};

struct TestConcatLoHi {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
#if SIMD_BITS != 0
    // Middle part of iota(1) == iota(1 + d.N / 2).
    const auto lo = iota(d, 1);
    const auto hi = iota(d, 1 + d.N);
    SIMD_ASSERT_VEC_EQ(d, iota(d, 1 + d.N / 2), concat_lo_hi(hi, lo));
#endif
  }
};

struct TestConcatHiLo {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
#if SIMD_BITS != 0
    const auto lo = iota(d, 1);
    const auto hi = iota(d, 1 + d.N);
    T expected[d.N];
    size_t i = 0;
    for (; i < d.N / 2; ++i) {
      expected[i] = 1 + i;
    }
    for (; i < d.N; ++i) {
      expected[i] = 1 + i + d.N;
    }
    SIMD_ASSERT_VEC_EQ(d, expected, concat_hi_lo(hi, lo));
#endif
  }
};

struct TestOddEven {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
#if SIMD_BITS != 0
    const auto even = iota(d, 1);
    const auto odd = iota(d, 1 + d.N);
    T expected[d.N];
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = 1 + i + ((i & 1) ? d.N : 0);
    }
    SIMD_ASSERT_VEC_EQ(d, expected, odd_even(odd, even));
#endif
  }
};

template <unsigned idx_ref>
SIMD_ATTR void TestMPSAD_T() {
#if SIMD_BITS != 0
  const SIMD_CAPPED(uint8_t, 16) d8;
  const SIMD_CAPPED(uint16_t, 8) d16;

  SIMD_ALIGN const uint8_t window[16] = {2,   1,   3, 0, 100, 200, 102, 99,
                                         255, 254, 0, 1, 8,   8,   128, 128};
  SIMD_ALIGN const uint8_t all_ref[16] = {1, 3, 5,  5,  2,  10, 10, 14,
                                          9, 9, 13, 17, 18, 16, 14, 0};
  // Sum of absolute distance at indices w and r ((0..3, relative to idx_ref).
  const auto S = [window, all_ref](size_t w, size_t r) {
    return std::abs(window[w] - all_ref[idx_ref * 4 + r]);
  };

  uint16_t expected_sad[8];
  for (size_t i = 0; i < 8; ++i) {
    int sad_sum = 0;
    for (size_t q = 0; q < 4; ++q) {
      sad_sum += S(i + q, q);
    }
    expected_sad[i] = static_cast<uint16_t>(sad_sum);
  }

  const auto sad = ext::mpsadbw<idx_ref>(load(d8, window), load(d8, all_ref));
  SIMD_ASSERT_VEC_EQ(d16, expected_sad, sad);

// AVX2 mpsadbw2 = two mpsadbw; test both halves independently
#if SIMD_BITS > 128
  const SIMD_CAPPED(uint8_t, d8.N * 2) d8x2;
  const SIMD_CAPPED(uint16_t, d16.N * 2) d16x2;
  const auto window_x2 = load_dup128(d8x2, window);
  const auto ref_x2 = load_dup128(d8x2, all_ref);
  const auto sad_lo = ext::mpsadbw2<3 - idx_ref, idx_ref>(window_x2, ref_x2);
  SIMD_ALIGN uint16_t sad2[d16x2.N];
  store(sad_lo, d16x2, sad2);
  for (size_t i = 0; i < d16.N; ++i) {
    SIMD_ASSERT_EQ(expected_sad[i], sad2[i]);
  }

  const auto sad_hi = ext::mpsadbw2<idx_ref, 3 - idx_ref>(window_x2, ref_x2);
  store(sad_hi, d16x2, sad2);
  for (size_t i = 0; i < d16.N; ++i) {
    SIMD_ASSERT_EQ(expected_sad[i], sad2[i + d16.N]);
  }
#endif
#endif
}

SIMD_ATTR void TestMPSAD() {
  // For each idx_ref (= index of 32-bit lane)
  TestMPSAD_T<0>();
  TestMPSAD_T<1>();
  TestMPSAD_T<2>();
  TestMPSAD_T<3>();
}

SIMD_ATTR void TestSwizzle() {
  TestShiftBytes();
  TestBroadcast();
  ForeachLaneType<TestInterleave>();
  TestPermute();
  TestZip();
  TestShuffle();
  TestExtract();
  TestSpecialShuffles();
  ForeachLaneType<TestConcatHalves>();
  ForeachLaneType<TestConcatLoHi>();
  ForeachLaneType<TestConcatHiLo>();
  ForeachLaneType<TestOddEven>();
  TestMPSAD();
}

}  // namespace
}  // namespace SIMD_NAMESPACE  // NOLINT(google-readability-namespace-comments)
}  // namespace jxl

// Instantiate for the current target.
template <>
void SwizzleTest::operator()<jxl::SIMD_TARGET>() {
  jxl::SIMD_NAMESPACE::TestSwizzle();
}
