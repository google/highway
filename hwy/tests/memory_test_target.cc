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

#include "highway/cache_control.h"
#include "highway/memory_test.h"
#include "highway/test_target_util.h"

namespace jxl {
namespace SIMD_NAMESPACE {
namespace {

struct TestLoadStore {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto hi = iota(d, 1 + d.N);
    const auto lo = iota(d, 1);
    SIMD_ALIGN T lanes[2 * d.N];
    store(hi, d, lanes + d.N);
    store(lo, d, lanes);

    // Aligned load
    const auto lo2 = load(d, lanes);
    SIMD_ASSERT_VEC_EQ(d, lo2, lo);

    // Aligned store
    SIMD_ALIGN T lanes2[2 * d.N];
    store(lo2, d, lanes2);
    store(hi, d, lanes2 + d.N);
    for (size_t i = 0; i < 2 * d.N; ++i) {
      SIMD_ASSERT_EQ(lanes[i], lanes2[i]);
    }

    // Unaligned load
    const auto vu = load_u(d, lanes + 1);
    SIMD_ALIGN T lanes3[d.N];
    store(vu, d, lanes3);
    for (size_t i = 0; i < d.N; ++i) {
      SIMD_ASSERT_EQ(T(i + 2), lanes3[i]);
    }

    // Unaligned store
    store_u(lo2, d, lanes2 + d.N / 2);
    size_t i = 0;
    for (; i < d.N / 2; ++i) {
      SIMD_ASSERT_EQ(lanes[i], lanes2[i]);
    }
    for (; i < 3 * d.N / 2; ++i) {
      SIMD_ASSERT_EQ(T(i - d.N / 2 + 1), lanes2[i]);
    }
    // Subsequent values remain unchanged.
    for (; i < 2 * d.N; ++i) {
      SIMD_ASSERT_EQ(T(i + 1), lanes2[i]);
    }
  }
};

struct TestLoadDup128 {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
#if SIMD_BITS != 0
    constexpr size_t N128 = 16 / sizeof(T);
    alignas(16) T lanes[N128];
    for (size_t i = 0; i < N128; ++i) {
      lanes[i] = 1 + i;
    }
    const auto v = load_dup128(d, lanes);
    SIMD_ALIGN T out[d.N];
    store(v, d, out);
    for (size_t i = 0; i < d.N; ++i) {
      ASSERT_EQ(T(i % N128 + 1), out[i]);
    }
#endif
  }
};

struct TestStreamT {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v = iota(d, 0);
    SIMD_ALIGN T out[d.N];
    stream(v, d, out);
    store_fence();
    for (size_t i = 0; i < d.N; ++i) {
      SIMD_ASSERT_EQ(T(i), out[i]);
    }
  }
};

#if SIMD_HAS_GATHER

template <typename Offset, int kShift>
struct TestGatherT {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    static_assert(sizeof(T) == (1 << kShift), "Incorrect kShift");

    // Base points to middle; |max_offset| + sizeof(T) <= kNumBytes / 2.
    constexpr size_t kNumBytes = kMaxVectorSize * 2;
    uint8_t bytes[kNumBytes];
    for (size_t i = 0; i < kNumBytes; ++i) {
      bytes[i] = i + 1;
    }
    const uint8_t* middle = bytes + kNumBytes / 2;

    // Offsets: combinations of aligned, repeated, negative.
    SIMD_ALIGN Offset offset_lanes[SIMD_MAX(d.N, 16)] = {
        2, 12, 4, 4, -16, -16, -21, -20, 8, 8, 8, -13, -13, -20, 20, 3};

    SIMD_ALIGN T expected[d.N];
    for (size_t i = 0; i < d.N; ++i) {
      CopyBytes<sizeof(T)>(middle + offset_lanes[i], &expected[i]);
    }

    const auto offsets = load(SIMD_FULL(Offset)(), offset_lanes);
    auto actual =
        ext::gather_offset(d, reinterpret_cast<const T*>(middle), offsets);
    SIMD_ASSERT_VEC_EQ(d, expected, actual);

    // Indices
    SIMD_ALIGN const Offset index_lanes[SIMD_MAX(d.N, 16)] = {
        1, -2, 0, 1, 3, -2, -1, 2, 4, -3, 5, -5, 0, 2, -4, 0};
    for (size_t i = 0; i < d.N; ++i) {
      CopyBytes<sizeof(T)>(
          middle + index_lanes[i] * static_cast<Offset>(sizeof(T)),
          &expected[i]);
    }
    const auto indices = load(SIMD_FULL(Offset)(), index_lanes);
    actual = ext::gather_index(d, reinterpret_cast<const T*>(middle), indices);
    SIMD_ASSERT_VEC_EQ(d, expected, actual);
  }
};

template <typename Offset, int kShift>
struct TestFloatGatherT {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    static_assert(sizeof(T) == (1 << kShift), "Incorrect kShift");

    constexpr size_t kNumValues = 16;
    // Base points to middle; |max_index| < kNumValues / 2.
    SIMD_ALIGN const T values[SIMD_MAX(d.N, kNumValues)] = {
        T(100.0), T(110.0), T(111.0), T(128.0), T(1024.0), T(-1.0),
        T(-2.0),  T(-3.0),  T(0.25),  T(0.5),   T(0.75),   T(1.25),
        T(1.5),   T(1.75),  T(-0.25), T(-0.5)};
    const T* middle = values + kNumValues / 2;

    // Indices: combinations of aligned, repeated, negative.
    SIMD_ALIGN const Offset index_lanes[SIMD_MAX(d.N, 16)] = {1, -6, 0,  1,
                                                              3, -6, -1, 7};
    SIMD_ALIGN T expected[d.N];
    for (size_t i = 0; i < d.N; ++i) {
      CopyBytes<sizeof(T)>(middle + index_lanes[i], &expected[i]);
    }
    const auto indices = load(SIMD_FULL(Offset)(), index_lanes);
    auto actual = ext::gather_index(d, middle, indices);
    SIMD_ASSERT_VEC_EQ(d, expected, actual);

    // Offsets: same as index * sizeof(T).
    const auto offsets = shift_left<kShift>(indices);
    actual = ext::gather_offset(d, middle, offsets);
    SIMD_ASSERT_VEC_EQ(d, expected, actual);
  }
};

#endif

SIMD_ATTR void TestStream() {
  // No u8,u16.
  Call<TestStreamT, uint32_t>();
  Call<TestStreamT, uint64_t>();
  // No i8,i16.
  Call<TestStreamT, int32_t>();
  Call<TestStreamT, int64_t>();
  Call<TestStreamT, float>();
  Call<TestStreamT, double>();
}

SIMD_ATTR void TestGather() {
#if SIMD_HAS_GATHER
  // No u8,u16.
  Call<TestGatherT<int32_t, 2>, uint32_t>();
  Call<TestGatherT<int64_t, 3>, uint64_t>();
  // No i8,i16.
  Call<TestGatherT<int32_t, 2>, int32_t>();
  Call<TestGatherT<int64_t, 3>, int64_t>();

  Call<TestFloatGatherT<int32_t, 2>, float>();
  Call<TestFloatGatherT<int64_t, 3>, double>();
#endif
}

SIMD_ATTR void TestMemory() {
  ForeachLaneType<TestLoadStore>();
  ForeachLaneType<TestLoadDup128>();
  TestStream();
  TestGather();
}

}  // namespace
}  // namespace SIMD_NAMESPACE  // NOLINT(google-readability-namespace-comments)
}  // namespace jxl

// Instantiate for the current target.
template <>
void MemoryTest::operator()<jxl::SIMD_TARGET>() {
  jxl::SIMD_NAMESPACE::TestMemory();
}
