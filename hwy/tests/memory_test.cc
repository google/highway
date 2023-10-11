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

// Ensure incompabilities with Windows macros (e.g. #define StoreFence) are
// detected. Must come before Highway headers.
#include "hwy/base.h"
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

#include <algorithm>  // std::fill

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/memory_test.cc"
#include "hwy/cache_control.h"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

struct TestLoadStore {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    const auto hi = Iota(d, static_cast<T>(1 + N));
    const auto lo = Iota(d, 1);
    auto lanes = AllocateAligned<T>(2 * N);
    auto lanes2 = AllocateAligned<T>(2 * N);
    auto lanes3 = AllocateAligned<T>(N);
    HWY_ASSERT(lanes && lanes2 && lanes3);

    Store(hi, d, &lanes[N]);
    Store(lo, d, &lanes[0]);

    // Aligned load
    const auto lo2 = Load(d, &lanes[0]);
    HWY_ASSERT_VEC_EQ(d, lo2, lo);

    // Aligned store
    Store(lo2, d, &lanes2[0]);
    Store(hi, d, &lanes2[N]);
    for (size_t i = 0; i < 2 * N; ++i) {
      HWY_ASSERT_EQ(lanes[i], lanes2[i]);
    }

    // Unaligned load
    const auto vu = LoadU(d, &lanes[1]);
    Store(vu, d, lanes3.get());
    for (size_t i = 0; i < N; ++i) {
      HWY_ASSERT_EQ(T(i + 2), lanes3[i]);
    }

    // Unaligned store
    StoreU(lo2, d, &lanes2[N / 2]);
    size_t i = 0;
    for (; i < N / 2; ++i) {
      HWY_ASSERT_EQ(lanes[i], lanes2[i]);
    }
    for (; i < 3 * N / 2; ++i) {
      HWY_ASSERT_EQ(T(i - N / 2 + 1), lanes2[i]);
    }
    // Subsequent values remain unchanged.
    for (; i < 2 * N; ++i) {
      HWY_ASSERT_EQ(T(i + 1), lanes2[i]);
    }
  }
};

HWY_NOINLINE void TestAllLoadStore() {
  ForAllTypes(ForPartialVectors<TestLoadStore>());
}

struct TestSafeCopyN {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    const auto v = Iota(d, 1);
    auto from = AllocateAligned<T>(N + 2);
    auto to = AllocateAligned<T>(N + 2);
    HWY_ASSERT(from && to);
    Store(v, d, from.get());

    // 0: nothing changes
    to[0] = T();
    SafeCopyN(0, d, from.get(), to.get());
    HWY_ASSERT_EQ(T(), to[0]);

    // 1: only first changes
    to[1] = T();
    SafeCopyN(1, d, from.get(), to.get());
    HWY_ASSERT_EQ(static_cast<T>(1), to[0]);
    HWY_ASSERT_EQ(T(), to[1]);

    // N-1: last does not change
    to[N - 1] = T();
    SafeCopyN(N - 1, d, from.get(), to.get());
    HWY_ASSERT_EQ(T(), to[N - 1]);
    // Also check preceding lanes
    to[N - 1] = static_cast<T>(N);
    HWY_ASSERT_VEC_EQ(d, to.get(), v);

    // N: all change
    to[N] = T();
    SafeCopyN(N, d, from.get(), to.get());
    HWY_ASSERT_VEC_EQ(d, to.get(), v);
    HWY_ASSERT_EQ(T(), to[N]);

    // N+1: subsequent lane does not change if using masked store
    to[N + 1] = T();
    SafeCopyN(N + 1, d, from.get(), to.get());
    HWY_ASSERT_VEC_EQ(d, to.get(), v);
#if !HWY_MEM_OPS_MIGHT_FAULT
    HWY_ASSERT_EQ(T(), to[N + 1]);
#endif
  }
};

HWY_NOINLINE void TestAllSafeCopyN() {
  ForAllTypes(ForPartialVectors<TestSafeCopyN>());
}

struct TestLoadDup128 {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Scalar does not define LoadDup128.
#if HWY_TARGET != HWY_SCALAR || HWY_IDE
    constexpr size_t N128 = 16 / sizeof(T);
    alignas(16) T lanes[N128];
    for (size_t i = 0; i < N128; ++i) {
      lanes[i] = static_cast<T>(1 + i);
    }

    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(expected);
    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>(i % N128 + 1);
    }

    HWY_ASSERT_VEC_EQ(d, expected.get(), LoadDup128(d, lanes));
#else
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllLoadDup128() {
  ForAllTypes(ForGEVectors<128, TestLoadDup128>());
}

struct TestStream {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v = Iota(d, T(1));
    const size_t affected_bytes =
        (Lanes(d) * sizeof(T) + HWY_STREAM_MULTIPLE - 1) &
        ~size_t(HWY_STREAM_MULTIPLE - 1);
    const size_t affected_lanes = affected_bytes / sizeof(T);
    auto out = AllocateAligned<T>(2 * affected_lanes);
    HWY_ASSERT(out);
    std::fill(out.get(), out.get() + 2 * affected_lanes, T(0));

    Stream(v, d, out.get());
    FlushStream();
    const auto actual = Load(d, out.get());
    HWY_ASSERT_VEC_EQ(d, v, actual);
    // Ensure Stream didn't modify more memory than expected
    for (size_t i = affected_lanes; i < 2 * affected_lanes; ++i) {
      HWY_ASSERT_EQ(T(0), out[i]);
    }
  }
};

HWY_NOINLINE void TestAllStream() {
  const ForPartialVectors<TestStream> test;
  // No u8,u16.
  test(uint32_t());
  test(uint64_t());
  // No i8,i16.
  test(int32_t());
  test(int64_t());
  ForFloatTypes(test);
}

// Assumes little-endian byte order!
struct TestScatter {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    using Offset = MakeSigned<T>;
    const Rebind<Offset, D> d_offsets;

    const size_t N = Lanes(d);
    const size_t range = 4 * N;                  // number of items to scatter
    const size_t max_bytes = range * sizeof(T);  // upper bound on offset

    RandomState rng;

    auto values = AllocateAligned<T>(range);
    auto offsets = AllocateAligned<Offset>(N);  // or indices
    // Scatter into these regions, ensure vector results match scalar
    auto expected = AllocateAligned<T>(range);
    auto actual = AllocateAligned<T>(range);
    HWY_ASSERT(values && offsets && expected && actual);

    // Data to be scattered
    uint8_t *bytes = reinterpret_cast<uint8_t*>(values.get());
    for (size_t i = 0; i < max_bytes; ++i) {
      bytes[i] = static_cast<uint8_t>(Random32(&rng) & 0xFF);
    }
    const auto data = Load(d, values.get());

    for (size_t rep = 0; rep < 100; ++rep) {
      // Byte offsets
      std::fill(expected.get(), expected.get() + range, T(0));
      std::fill(actual.get(), actual.get() + range, T(0));
      for (size_t i = 0; i < N; ++i) {
        // Must be aligned
        offsets[i] = static_cast<Offset>((Random32(&rng) % range) * sizeof(T));
        CopyBytes<sizeof(T)>(
            values.get() + i,
            reinterpret_cast<uint8_t*>(expected.get()) + offsets[i]);
      }
      const auto voffsets = Load(d_offsets, offsets.get());
      ScatterOffset(data, d, actual.get(), voffsets);
      if (!BytesEqual(expected.get(), actual.get(), max_bytes)) {
        Print(d, "Data", data);
        Print(d_offsets, "Offsets", voffsets);
        HWY_ASSERT(false);
      }

      // Indices
      std::fill(expected.get(), expected.get() + range, T(0));
      std::fill(actual.get(), actual.get() + range, T(0));
      for (size_t i = 0; i < N; ++i) {
        offsets[i] = static_cast<Offset>(Random32(&rng) % range);
        CopyBytes<sizeof(T)>(values.get() + i ,
                             &expected[size_t(offsets[i])]);
      }
      const auto vindices = Load(d_offsets, offsets.get());
      ScatterIndex(data, d, actual.get(), vindices);
      if (!BytesEqual(expected.get(), actual.get(), max_bytes)) {
        Print(d, "Data", data);
        Print(d_offsets, "Indices", vindices);
        HWY_ASSERT(false);
      }
    }
  }
};

HWY_NOINLINE void TestAllScatter() {
  ForUIF3264(ForPartialVectors<TestScatter>());
}

struct TestGather {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    using Offset = MakeSigned<T>;

    const size_t N = Lanes(d);
    const size_t range = 4 * N;                  // number of items to gather
    const size_t max_bytes = range * sizeof(T);  // upper bound on offset

    RandomState rng;
    auto values = AllocateAligned<T>(range);
    auto expected = AllocateAligned<T>(N);
    auto offsets = AllocateAligned<Offset>(N);
    auto indices = AllocateAligned<Offset>(N);
    HWY_ASSERT(values && expected && offsets && indices);

    // Data to be gathered from
    uint8_t *bytes = reinterpret_cast<uint8_t*>(values.get());
    for (size_t i = 0; i < max_bytes; ++i) {
      bytes[i] = static_cast<uint8_t>(Random32(&rng) & 0xFF);
    }

    for (size_t rep = 0; rep < 100; ++rep) {
      // Offsets
      for (size_t i = 0; i < N; ++i) {
        // Must be aligned
        offsets[i] = static_cast<Offset>((Random32(&rng) % range) * sizeof(T));
        CopyBytes<sizeof(T)>(bytes + offsets[i], &expected[i]);
      }

      const Rebind<Offset, D> d_offset;
      const T* base = values.get();
      auto actual = GatherOffset(d, base, Load(d_offset, offsets.get()));
      HWY_ASSERT_VEC_EQ(d, expected.get(), actual);

      // Indices
      for (size_t i = 0; i < N; ++i) {
        indices[i] =
            static_cast<Offset>(Random32(&rng) % (max_bytes / sizeof(T)));
        CopyBytes<sizeof(T)>(base + indices[i], &expected[i]);
      }
      actual = GatherIndex(d, base, Load(d_offset, indices.get()));
      HWY_ASSERT_VEC_EQ(d, expected.get(), actual);
    }
  }
};

HWY_NOINLINE void TestAllGather() {
  ForUIF3264(ForPartialVectors<TestGather>());
}

HWY_NOINLINE void TestAllCache() {
  LoadFence();
  FlushStream();
  int test = 0;
  Prefetch(&test);
  FlushCacheline(&test);
  Pause();
}

namespace detail {
template <int kNo, class T, HWY_IF_NOT_FLOAT_NOR_SPECIAL(T)>
HWY_INLINE T GenerateOtherValue(size_t val) {
  const T conv_val = static_cast<T>(val);
  return (conv_val == static_cast<T>(kNo)) ? static_cast<T>(-17) : conv_val;
}
template <int kNo, class T, HWY_IF_FLOAT3264(T)>
HWY_INLINE T GenerateOtherValue(size_t val) {
  const T flt_val = static_cast<T>(val);
  return (flt_val == static_cast<T>(kNo) ? static_cast<T>(0.5426808228865735)
                                         : flt_val);
}
template <int kNo, class T, HWY_IF_BF16(T)>
HWY_INLINE T GenerateOtherValue(size_t val) {
  return BF16FromF32(GenerateOtherValue<kNo, float>(val));
}
template <int kNo, class T, HWY_IF_F16(T)>
HWY_INLINE T GenerateOtherValue(size_t val) {
  return F16FromF32(GenerateOtherValue<kNo, float>(val));
}

}  // namespace detail

struct TestLoadN {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    constexpr size_t kMaxLanesPerBlock = 16 / sizeof(T);
    const size_t lpb = HWY_MIN(N, kMaxLanesPerBlock);
    HWY_ASSERT(lpb >= 1);
    HWY_ASSERT(N <= (static_cast<size_t>(~size_t(0)) / 4));

    const size_t load_buf_len = (3 * N) + 4;

    auto load_buf = AllocateAligned<T>(load_buf_len);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(load_buf && expected);

    for (size_t i = 0; i < load_buf_len; i++) {
      load_buf[i] = detail::GenerateOtherValue<0, T>(i + 1);
    }

    std::fill(expected.get(), expected.get() + N, static_cast<T>(0));
    HWY_ASSERT_VEC_EQ(d, expected.get(), LoadN(d, load_buf.get(), 0));

    for (size_t i = 0; i <= lpb; i++) {
      CopyBytes(load_buf.get(), expected.get(), i * sizeof(T));
      const auto actual_1 = LoadN(d, load_buf.get(), i);
      HWY_ASSERT_VEC_EQ(d, expected.get(), actual_1);

      CopyBytes(load_buf.get() + 3, expected.get(), i * sizeof(T));
      const auto actual_2 = LoadN(d, load_buf.get() + 3, i);
      HWY_ASSERT_VEC_EQ(d, expected.get(), actual_2);
    }

    const size_t lplb = HWY_MAX(N / 4, lpb);
    for (size_t i = HWY_MAX(lpb * 2, lplb); i <= N * 2; i += lplb) {
      const size_t max_num_of_lanes_to_load = i + (11 & (lpb - 1));
      const size_t expected_num_of_lanes_loaded =
          HWY_MIN(max_num_of_lanes_to_load, N);

      CopyBytes(load_buf.get(), expected.get(),
                expected_num_of_lanes_loaded * sizeof(T));
      const auto actual_1 = LoadN(d, load_buf.get(), max_num_of_lanes_to_load);
      HWY_ASSERT_VEC_EQ(d, expected.get(), actual_1);

      CopyBytes(load_buf.get() + 3, expected.get(),
                expected_num_of_lanes_loaded * sizeof(T));
      const auto actual_2 =
          LoadN(d, load_buf.get() + 3, max_num_of_lanes_to_load);
      HWY_ASSERT_VEC_EQ(d, expected.get(), actual_2);
    }

    load_buf[0] = detail::GenerateOtherValue<0, T>(0);
    CopyBytes(load_buf.get(), expected.get(), N * sizeof(T));
    HWY_ASSERT_VEC_EQ(d, expected.get(), LoadN(d, load_buf.get(), N));
  }
};

HWY_NOINLINE void TestAllLoadN() {
  ForAllTypes(ForPartialVectors<TestLoadN>());
}

struct TestLoadNOr {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    constexpr int kNo = 2;
    const size_t N = Lanes(d);
    constexpr size_t kMaxLanesPerBlock = 16 / sizeof(T);
    const size_t lpb = HWY_MIN(N, kMaxLanesPerBlock);
    HWY_ASSERT(lpb >= 1);
    HWY_ASSERT(N <= (static_cast<size_t>(~size_t(0)) / 4));

    const size_t load_buf_len = (3 * N) + 4;

    auto load_buf = AllocateAligned<T>(load_buf_len);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(load_buf && expected);

    for (size_t i = 0; i < load_buf_len; i++) {
      load_buf[i] = detail::GenerateOtherValue<kNo, T>(i + 1);
    }
    const Vec<D> no = Set(d, static_cast<T>(kNo));

    std::fill(expected.get(), expected.get() + N, static_cast<T>(kNo));
    HWY_ASSERT_VEC_EQ(d, expected.get(), LoadNOr(no, d, load_buf.get(), 0));

    for (size_t i = 0; i <= lpb; i++) {
      CopyBytes(load_buf.get(), expected.get(), i * sizeof(T));
      const auto actual_1 = LoadNOr(no, d, load_buf.get(), i);
      HWY_ASSERT_VEC_EQ(d, expected.get(), actual_1);

      CopyBytes(load_buf.get() + 3, expected.get(), i * sizeof(T));
      const auto actual_2 = LoadNOr(no, d, load_buf.get() + 3, i);
      HWY_ASSERT_VEC_EQ(d, expected.get(), actual_2);
    }

    const size_t lplb = HWY_MAX(N / 4, lpb);
    for (size_t i = HWY_MAX(lpb * 2, lplb); i <= N * 2; i += lplb) {
      const size_t max_num_of_lanes_to_load = i + (11 & (lpb - 1));
      const size_t expected_num_of_lanes_loaded =
          HWY_MIN(max_num_of_lanes_to_load, N);

      CopyBytes(load_buf.get(), expected.get(),
                expected_num_of_lanes_loaded * sizeof(T));
      const auto actual_1 =
          LoadNOr(no, d, load_buf.get(), max_num_of_lanes_to_load);
      HWY_ASSERT_VEC_EQ(d, expected.get(), actual_1);

      CopyBytes(load_buf.get() + 3, expected.get(),
                expected_num_of_lanes_loaded * sizeof(T));
      const auto actual_2 =
          LoadNOr(no, d, load_buf.get() + 3, max_num_of_lanes_to_load);
      HWY_ASSERT_VEC_EQ(d, expected.get(), actual_2);
    }

    load_buf[0] = detail::GenerateOtherValue<kNo, T>(kNo);
    CopyBytes(load_buf.get(), expected.get(), N * sizeof(T));
    HWY_ASSERT_VEC_EQ(d, expected.get(), LoadNOr(no, d, load_buf.get(), N));
  }
};

HWY_NOINLINE void TestAllLoadNOr() {
  ForAllTypes(ForPartialVectors<TestLoadNOr>());
}

class TestStoreN {
 private:
  template <class T, HWY_IF_FLOAT_OR_SPECIAL(T)>
  static HWY_INLINE T NegativeFillValue() {
    return LowestValue<T>();
  }

  template <class T, HWY_IF_NOT_FLOAT_NOR_SPECIAL(T)>
  static HWY_INLINE T NegativeFillValue() {
    return static_cast<T>(-1);
  }

  template <class D, HWY_IF_NOT_FLOAT_NOR_SPECIAL_D(D)>
  static HWY_INLINE Vec<D> PositiveIota(D d, size_t start) {
    using T = TFromD<D>;
    using TI = MakeSigned<T>;

    constexpr T kSignedMax = static_cast<T>(LimitsMax<TI>());
    constexpr T kZeroReplVal = static_cast<T>(kSignedMax - 16);

    const auto v = Iota(d, static_cast<T>(start));
    return Or(v, IfThenElseZero(Eq(v, Zero(d)), Set(d, kZeroReplVal)));
  }

  template <class D, HWY_IF_FLOAT3264_D(D)>
  static HWY_INLINE Vec<D> PositiveIota(D d, size_t start) {
    return Max(Abs(Iota(d, static_cast<TFromD<D>>(start))),
               Set(d, static_cast<TFromD<D>>(0.5426808228865735)));
  }

  template <class DF32>
  static HWY_INLINE Vec<DF32> MaskedF32PositiveIota(DF32 df32, size_t start,
                                                    int32_t mask) {
    const RebindToSigned<decltype(df32)> di32;
    return ConvertTo(
        df32, And(Iota(di32, static_cast<int32_t>(start)), Set(di32, mask)));
  }

  template <class D, HWY_IF_F16_D(D),
            HWY_IF_V_SIZE_LE_D(DFromV<Vec<D>>, HWY_MAX_BYTES / 2)>
  static HWY_INLINE Vec<D> PositiveIota(D d, size_t start) {
    const Rebind<float, decltype(d)> df32;
    const RebindToSigned<decltype(d)> di16;
    const auto vf32_iota = MaskedF32PositiveIota(df32, start, int32_t{0x07FF});
    return BitCast(d, Max(BitCast(di16, DemoteTo(d, vf32_iota)),
                          Set(di16, int16_t{0x3857})));
  }

  template <class D, HWY_IF_BF16_D(D),
            HWY_IF_V_SIZE_LE_D(DFromV<Vec<D>>, HWY_MAX_BYTES / 2)>
  static HWY_INLINE Vec<D> PositiveIota(D d, size_t start) {
    const Rebind<float, decltype(d)> df32;
    const RebindToSigned<decltype(d)> di16;
    const auto vf32_iota = MaskedF32PositiveIota(df32, start, int32_t{0x00FF});
    return BitCast(d, Max(BitCast(di16, DemoteTo(d, vf32_iota)),
                          Set(di16, int16_t{0x3F0B})));
  }

#if HWY_TARGET != HWY_SCALAR
  template <class D, HWY_IF_F16_D(D),
            HWY_IF_V_SIZE_GT_D(DFromV<Vec<D>>, HWY_MAX_BYTES / 2)>
  static HWY_INLINE Vec<D> PositiveIota(D d, size_t start) {
    const Repartition<float, decltype(d)> df32;
    const RebindToSigned<decltype(d)> di16;
    const Half<decltype(d)> dh;

    const size_t N_f32 = Lanes(df32);
    const auto lo_vf32_iota =
        MaskedF32PositiveIota(df32, start, int32_t{0x07FF});
    const auto hi_vf32_iota =
        MaskedF32PositiveIota(df32, start + N_f32, int32_t{0x07FF});

    const auto vf16_iota =
        Combine(d, DemoteTo(dh, hi_vf32_iota), DemoteTo(dh, lo_vf32_iota));
    return BitCast(d,
                   Max(BitCast(di16, vf16_iota), Set(di16, int16_t{0x3857})));
  }

  template <class D, HWY_IF_BF16_D(D),
            HWY_IF_V_SIZE_GT_D(DFromV<Vec<D>>, HWY_MAX_BYTES / 2)>
  static HWY_INLINE Vec<D> PositiveIota(D d, size_t start) {
    const Repartition<float, decltype(d)> df32;
    const RebindToSigned<decltype(d)> di16;
    const Half<decltype(d)> dh;

    const size_t N_f32 = Lanes(df32);
    const auto lo_vf32_iota =
        MaskedF32PositiveIota(df32, start, int32_t{0x07FF});
    const auto hi_vf32_iota =
        MaskedF32PositiveIota(df32, start + N_f32, int32_t{0x07FF});

    const auto vbf16_iota = OrderedDemote2To(d, lo_vf32_iota, hi_vf32_iota);
    return BitCast(d,
                   Max(BitCast(di16, vbf16_iota), Set(di16, int16_t{0x3F0B})));
  }
#endif

 public:
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    constexpr size_t kMaxLanesPerBlock = 16 / sizeof(T);
    const size_t lpb = HWY_MIN(N, kMaxLanesPerBlock);
    HWY_ASSERT(lpb >= 1);

    const size_t full_dvec_N = Lanes(DFromV<Vec<D>>());
    HWY_ASSERT(N <= full_dvec_N);
    HWY_ASSERT(full_dvec_N <= (static_cast<size_t>(~size_t(0)) / 8));

    const size_t store_n_buf_offset = HWY_MAX(kMaxLanesPerBlock, full_dvec_N);
    const size_t store_buf_N = store_n_buf_offset + 3 * full_dvec_N + 4;
    auto expected = AllocateAligned<T>(store_buf_N);
    auto actual = AllocateAligned<T>(store_buf_N);
    HWY_ASSERT(expected && actual);

    const T neg_fill_val = NegativeFillValue<T>();
    std::fill(expected.get(), expected.get() + store_buf_N, neg_fill_val);
    std::fill(actual.get(), actual.get() + store_buf_N, neg_fill_val);

    const auto v_neg_fill_val = Set(d, neg_fill_val);

    for (size_t i = 0; i <= lpb; i++) {
      const auto v = PositiveIota(d, i + 1);
      const auto v_expected = IfThenElse(FirstN(d, i), v, v_neg_fill_val);

      Store(v_expected, d, expected.get() + store_n_buf_offset);
      Store(v_neg_fill_val, d, actual.get() + store_n_buf_offset);
      StoreN(v, d, actual.get() + store_n_buf_offset, i);

      HWY_ASSERT_ARRAY_EQ(expected.get(), actual.get(), store_buf_N);

      StoreU(v_expected, d, expected.get() + store_n_buf_offset + 3);
      StoreU(v_neg_fill_val, d, actual.get() + store_n_buf_offset + 3);
      StoreN(v, d, actual.get() + store_n_buf_offset + 3, i);
      HWY_ASSERT_ARRAY_EQ(expected.get(), actual.get(), store_buf_N);
    }

    const size_t lplb = HWY_MAX(N / 4, lpb);
    for (size_t i = HWY_MAX(lpb * 2, lplb); i <= N * 2; i += lplb) {
      const size_t max_num_of_lanes_to_store = i + (11 & (lpb - 1));
      const size_t expected_num_of_lanes_written =
          HWY_MIN(max_num_of_lanes_to_store, N);

      const auto v = PositiveIota(d, max_num_of_lanes_to_store + 1);
      const auto v_expected = IfThenElse(
          FirstN(d, expected_num_of_lanes_written), v, v_neg_fill_val);

      Store(v_expected, d, expected.get() + store_n_buf_offset);
      Store(v_neg_fill_val, d, actual.get() + store_n_buf_offset);
      StoreN(v, d, actual.get() + store_n_buf_offset,
             max_num_of_lanes_to_store);

      HWY_ASSERT_ARRAY_EQ(expected.get(), actual.get(), store_buf_N);

      StoreU(v_expected, d, expected.get() + store_n_buf_offset + 3);
      StoreU(v_neg_fill_val, d, actual.get() + store_n_buf_offset + 3);
      StoreN(v, d, actual.get() + store_n_buf_offset + 3,
             max_num_of_lanes_to_store);
      HWY_ASSERT_ARRAY_EQ(expected.get(), actual.get(), store_buf_N);
    }
  }
};

HWY_NOINLINE void TestAllStoreN() {
  ForAllTypes(ForPartialVectors<TestStoreN>());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
HWY_BEFORE_TEST(HwyMemoryTest);
HWY_EXPORT_AND_TEST_P(HwyMemoryTest, TestAllLoadStore);
HWY_EXPORT_AND_TEST_P(HwyMemoryTest, TestAllSafeCopyN);
HWY_EXPORT_AND_TEST_P(HwyMemoryTest, TestAllLoadDup128);
HWY_EXPORT_AND_TEST_P(HwyMemoryTest, TestAllStream);
HWY_EXPORT_AND_TEST_P(HwyMemoryTest, TestAllScatter);
HWY_EXPORT_AND_TEST_P(HwyMemoryTest, TestAllGather);
HWY_EXPORT_AND_TEST_P(HwyMemoryTest, TestAllCache);
HWY_EXPORT_AND_TEST_P(HwyMemoryTest, TestAllLoadN);
HWY_EXPORT_AND_TEST_P(HwyMemoryTest, TestAllLoadNOr);
HWY_EXPORT_AND_TEST_P(HwyMemoryTest, TestAllStoreN);
}  // namespace hwy

#endif
