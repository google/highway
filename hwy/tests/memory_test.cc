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
#define HWY_TARGET_INCLUDE "tests/memory_test.cc"

#include "hwy/cache_control.h"
#include "hwy/tests/test_util.h"
struct MemoryTest {
  HWY_DECLARE(void, ())
};
TEST(HwyMemoryTest, Run) { hwy::RunTests<MemoryTest>(); }

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
HWY_NOINLINE HWY_ATTR void TestLoadStore(D d) {
  using T = typename D::T;
  const auto hi = Iota(d, 1 + d.N);
  const auto lo = Iota(d, 1);
  HWY_ALIGN T lanes[2 * d.N];
  Store(hi, d, lanes + d.N);
  Store(lo, d, lanes);

  // Aligned load
  const auto lo2 = Load(d, lanes);
  HWY_ASSERT_VEC_EQ(d, lo2, lo);

  // Aligned store
  HWY_ALIGN T lanes2[2 * d.N];
  Store(lo2, d, lanes2);
  Store(hi, d, lanes2 + d.N);
  for (size_t i = 0; i < 2 * d.N; ++i) {
    HWY_ASSERT_EQ(lanes[i], lanes2[i]);
  }

  // Unaligned load
  const auto vu = LoadU(d, lanes + 1);
  HWY_ALIGN T lanes3[d.N];
  Store(vu, d, lanes3);
  for (size_t i = 0; i < d.N; ++i) {
    HWY_ASSERT_EQ(T(i + 2), lanes3[i]);
  }

  // Unaligned store
  StoreU(lo2, d, lanes2 + d.N / 2);
  size_t i = 0;
  for (; i < d.N / 2; ++i) {
    HWY_ASSERT_EQ(lanes[i], lanes2[i]);
  }
  for (; i < 3 * d.N / 2; ++i) {
    HWY_ASSERT_EQ(T(i - d.N / 2 + 1), lanes2[i]);
  }
  // Subsequent values remain unchanged.
  for (; i < 2 * d.N; ++i) {
    HWY_ASSERT_EQ(T(i + 1), lanes2[i]);
  }
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestLoadDup128(D d) {
#if HWY_BITS != 0 || HWY_IDE
  using T = typename D::T;
  constexpr size_t N128 = 16 / sizeof(T);
  alignas(16) T lanes[N128];
  for (size_t i = 0; i < N128; ++i) {
    lanes[i] = static_cast<T>(1 + i);
  }
  const auto v = LoadDup128(d, lanes);
  HWY_ALIGN T out[d.N];
  Store(v, d, out);
  for (size_t i = 0; i < d.N; ++i) {
    HWY_ASSERT_EQ(T(i % N128 + 1), out[i]);
  }
#else
  (void)d;
#endif
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestStreamT(D d) {
  using T = typename D::T;
  const auto v = Iota(d, 0);
  HWY_ALIGN T out[d.N];
  Stream(v, d, out);
  StoreFence();
  for (size_t i = 0; i < d.N; ++i) {
    HWY_ASSERT_EQ(T(i), out[i]);
  }
}

#if HWY_HAS_GATHER || HWY_IDE

// kShift must be log2(sizeof(T)).
template <typename Offset, int kShift, class D>
HWY_NOINLINE HWY_ATTR void TestGatherT(D d) {
  using T = typename D::T;
  static_assert(sizeof(T) == (1 << kShift), "Incorrect kShift");

  // Base points to middle; |max_offset| + sizeof(T) <= kNumBytes / 2.
  constexpr size_t kNumBytes = kMaxVectorSize * 2;
  uint8_t bytes[kNumBytes];
  for (size_t i = 0; i < kNumBytes; ++i) {
    bytes[i] = i + 1;
  }
  const uint8_t* middle = bytes + kNumBytes / 2;

  // Offsets: combinations of aligned, repeated, negative.
  HWY_ALIGN Offset offset_lanes[HWY_MAX(d.N, 16)] = {
      2, 12, 4, 4, -16, -16, -21, -20, 8, 8, 8, -13, -13, -20, 20, 3};

  HWY_ALIGN T expected[d.N];
  for (size_t i = 0; i < d.N; ++i) {
    CopyBytes<sizeof(T)>(middle + offset_lanes[i], &expected[i]);
  }

  const auto offsets = Load(HWY_FULL(Offset)(), offset_lanes);
  auto actual =
      ext::GatherOffset(d, reinterpret_cast<const T*>(middle), offsets);
  HWY_ASSERT_VEC_EQ(d, expected, actual);

  // Indices
  HWY_ALIGN const Offset index_lanes[HWY_MAX(d.N, 16)] = {
      1, -2, 0, 1, 3, -2, -1, 2, 4, -3, 5, -5, 0, 2, -4, 0};
  for (size_t i = 0; i < d.N; ++i) {
    CopyBytes<sizeof(T)>(
        middle + index_lanes[i] * static_cast<Offset>(sizeof(T)), &expected[i]);
  }
  const auto indices = Load(HWY_FULL(Offset)(), index_lanes);
  actual = ext::GatherIndex(d, reinterpret_cast<const T*>(middle), indices);
  HWY_ASSERT_VEC_EQ(d, expected, actual);
}

template <typename Offset, int kShift, class D>
HWY_NOINLINE HWY_ATTR void TestFloatGatherT(D d) {
  using T = typename D::T;
  static_assert(sizeof(T) == (1 << kShift), "Incorrect kShift");

  constexpr size_t kNumValues = 16;
  // Base points to middle; |max_index| < kNumValues / 2.
  HWY_ALIGN const T values[HWY_MAX(d.N, kNumValues)] = {
      T(100.0), T(110.0), T(111.0), T(128.0), T(1024.0), T(-1.0),
      T(-2.0),  T(-3.0),  T(0.25),  T(0.5),   T(0.75),   T(1.25),
      T(1.5),   T(1.75),  T(-0.25), T(-0.5)};
  const T* middle = values + kNumValues / 2;

  // Indices: combinations of aligned, repeated, negative.
  HWY_ALIGN const Offset index_lanes[HWY_MAX(d.N, 16)] = {1, -6, 0,  1,
                                                          3, -6, -1, 7};
  HWY_ALIGN T expected[d.N];
  for (size_t i = 0; i < d.N; ++i) {
    CopyBytes<sizeof(T)>(middle + index_lanes[i], &expected[i]);
  }
  const auto indices = Load(HWY_FULL(Offset)(), index_lanes);
  auto actual = ext::GatherIndex(d, middle, indices);
  HWY_ASSERT_VEC_EQ(d, expected, actual);

  // Offsets: same as index * sizeof(T).
  const auto offsets = hwy::ShiftLeft<kShift>(indices);
  actual = ext::GatherOffset(d, middle, offsets);
  HWY_ASSERT_VEC_EQ(d, expected, actual);
}

#endif

HWY_NOINLINE HWY_ATTR void TestStream() {
  // No u8,u16.
  TestStreamT(du32);
  TestStreamT(du64);
  // No i8,i16.
  TestStreamT(di32);
  TestStreamT(di64);
  HWY_FOREACH_F(TestStreamT);
}

HWY_NOINLINE HWY_ATTR void TestGather() {
#if HWY_HAS_GATHER || HWY_IDE
  // No u8,u16.
  TestGatherT<int32_t, 2>(du32);
  TestGatherT<int64_t, 3>(du64);
  // No i8,i16.
  TestGatherT<int32_t, 2>(di32);
  TestGatherT<int64_t, 3>(di64);

  // Can't use HWY_FOREACH because kShift depends on type.
  TestFloatGatherT<int32_t, 2>(df);
#if HWY_HAS_DOUBLE
  TestFloatGatherT<int64_t, 3>(dd);
#endif
#endif
}

HWY_NOINLINE HWY_ATTR void TestMemory() {
  (void)dd;
  (void)di64;
  (void)du64;

  HWY_FOREACH_UIF(TestLoadStore);
  HWY_FOREACH_UIF(TestLoadDup128);
  TestStream();
  TestGather();
  // Test that these functions compile.
  LoadFence();
  StoreFence();
  int test = 0;
  Prefetch(&test);
  FlushCacheline(&test);
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy

// Instantiate for the current target.
void MemoryTest::HWY_FUNC() { hwy::HWY_NAMESPACE::TestMemory(); }
