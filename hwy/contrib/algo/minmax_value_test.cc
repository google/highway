// Copyright 2026 Google LLC
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

#include <stdio.h>

#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/algo/minmax_value_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/algo/minmax-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

template <typename T>
T Random(RandomState& rng) {
  const int32_t bits = static_cast<int32_t>(Random32(&rng)) & 1023;
  double val = (bits - 512) / 64.0;
  if (!hwy::IsSigned<T>() && val < 0.0) {
    val = -val;
  }
  return ConvertScalarTo<T>(val);
}

template <typename T>
T ScalarMin(const T* in, size_t count) {
  T result = hwy::PositiveInfOrHighestValue<T>();
  for (size_t i = 0; i < count; ++i) {
    if (in[i] < result) {
      result = in[i];
    }
  }
  return result;
}

template <typename T>
T ScalarMax(const T* in, size_t count) {
  T result = hwy::NegativeInfOrLowestValue<T>();
  for (size_t i = 0; i < count; ++i) {
    if (in[i] > result) {
      result = in[i];
    }
  }
  return result;
}

template <class Test>
struct ForeachCountAndMisalign {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) const {
    RandomState rng;
    const size_t N = Lanes(d);
    const size_t misalignments[3] = {0, N / 4, 3 * N / 5};

    std::vector<size_t> counts(AdjustedReps(512));
    for (size_t& count : counts) {
      count = static_cast<size_t>(rng()) % (16 * N + 1);
    }
    counts[0] = 0;

    for (size_t count : counts) {
      for (size_t m : misalignments) {
        Test()(d, count, m, rng);
      }
    }
  }
};

struct TestMinValue {
  template <class D>
  void operator()(D d, size_t count, size_t misalign, RandomState& rng) {
    using T = TFromD<D>;
    AlignedFreeUniquePtr<T[]> storage =
        AllocateAligned<T>(HWY_MAX(1, misalign + count));
    HWY_ASSERT(storage);
    T* in = storage.get() + misalign;
    for (size_t i = 0; i < count; ++i) {
      in[i] = Random<T>(rng);
    }

    const T expected = ScalarMin(in, count);
    const T actual = MinValue(d, in, count);

    if (!IsEqual(expected, actual)) {
      fprintf(stderr, "%s count %d misalign %d: MinValue expected %f got %f\n",
              hwy::TypeName(T(), Lanes(d)).c_str(), static_cast<int>(count),
              static_cast<int>(misalign), ConvertScalarTo<double>(expected),
              ConvertScalarTo<double>(actual));
      HWY_ASSERT(false);
    }
  }
};

struct TestMaxValue {
  template <class D>
  void operator()(D d, size_t count, size_t misalign, RandomState& rng) {
    using T = TFromD<D>;
    AlignedFreeUniquePtr<T[]> storage =
        AllocateAligned<T>(HWY_MAX(1, misalign + count));
    HWY_ASSERT(storage);
    T* in = storage.get() + misalign;
    for (size_t i = 0; i < count; ++i) {
      in[i] = Random<T>(rng);
    }

    const T expected = ScalarMax(in, count);
    const T actual = MaxValue(d, in, count);

    if (!IsEqual(expected, actual)) {
      fprintf(stderr, "%s count %d misalign %d: MaxValue expected %f got %f\n",
              hwy::TypeName(T(), Lanes(d)).c_str(), static_cast<int>(count),
              static_cast<int>(misalign), ConvertScalarTo<double>(expected),
              ConvertScalarTo<double>(actual));
      HWY_ASSERT(false);
    }
  }
};

void TestAllMinValue() {
  ForAllTypes(ForPartialVectors<ForeachCountAndMisalign<TestMinValue>>());
}

void TestAllMaxValue() {
  ForAllTypes(ForPartialVectors<ForeachCountAndMisalign<TestMaxValue>>());
}

struct TestReduce {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) const {
    const size_t N = Lanes(d);
    AlignedFreeUniquePtr<T[]> storage = AllocateAligned<T>(8 * N);
    HWY_ASSERT(storage);
    T* in = storage.get();
    for (size_t i = 0; i < 8 * N; ++i) {
      in[i] = static_cast<T>((i % 13) + 1);
    }

    auto x0 = Load(d, in + 0 * N);
    auto x1 = Load(d, in + 1 * N);
    auto x2 = Load(d, in + 2 * N);
    auto x3 = Load(d, in + 3 * N);
    auto x4 = Load(d, in + 4 * N);
    auto x5 = Load(d, in + 5 * N);
    auto x6 = Load(d, in + 6 * N);
    auto x7 = Load(d, in + 7 * N);

    T min_s[8];
    T max_s[8];
    T sum_s[8];
    for (int v = 0; v < 8; ++v) {
      min_s[v] = hwy::PositiveInfOrHighestValue<T>();
      max_s[v] = hwy::NegativeInfOrLowestValue<T>();
      sum_s[v] = 0;
      for (size_t i = 0; i < N; ++i) {
        T val = in[v * N + i];
        if (val < min_s[v]) min_s[v] = val;
        if (val > max_s[v]) max_s[v] = val;
        sum_s[v] = AddWithWraparound(sum_s[v], val);
      }
    }

    using D2 = CappedTag<T, 2>;
    const D2 d2;
    if (Lanes(d2) == 2) {
      HWY_ALIGN T actual_min[2];
      HWY_ALIGN T actual_max[2];
      HWY_ALIGN T actual_sum[2];
      auto p_min = Reduce2Min(d, x0, x1);
      actual_min[0] = p_min.first;
      actual_min[1] = p_min.second;
      auto p_max = Reduce2Max(d, x0, x1);
      actual_max[0] = p_max.first;
      actual_max[1] = p_max.second;
      auto p_sum = Reduce2Sum(d, x0, x1);
      actual_sum[0] = p_sum.first;
      actual_sum[1] = p_sum.second;
      for (int i = 0; i < 2; ++i) {
        HWY_ASSERT_EQ(min_s[i], actual_min[i]);
        HWY_ASSERT_EQ(max_s[i], actual_max[i]);
        HWY_ASSERT_EQ(sum_s[i], actual_sum[i]);
      }
    }

    using D4 = CappedTag<T, 4>;
    const D4 d4;
    if (Lanes(d4) == 4) {
      HWY_ALIGN T actual_min[4];
      HWY_ALIGN T actual_max[4];
      HWY_ALIGN T actual_sum[4];
      Store(Reduce4Min(d, x0, x1, x2, x3), d4, actual_min);
      Store(Reduce4Max(d, x0, x1, x2, x3), d4, actual_max);
      Store(Reduce4Sum(d, x0, x1, x2, x3), d4, actual_sum);
      for (int i = 0; i < 4; ++i) {
        HWY_ASSERT_EQ(min_s[i], actual_min[i]);
        HWY_ASSERT_EQ(max_s[i], actual_max[i]);
        HWY_ASSERT_EQ(sum_s[i], actual_sum[i]);
      }
    }

    using D8 = CappedTag<T, 8>;
    const D8 d8;
    if (Lanes(d8) == 8) {
      HWY_ALIGN T actual_min[8];
      HWY_ALIGN T actual_max[8];
      HWY_ALIGN T actual_sum[8];
      Store(Reduce8Min(d, x0, x1, x2, x3, x4, x5, x6, x7), d8, actual_min);
      Store(Reduce8Max(d, x0, x1, x2, x3, x4, x5, x6, x7), d8, actual_max);
      Store(Reduce8Sum(d, x0, x1, x2, x3, x4, x5, x6, x7), d8, actual_sum);
      for (int i = 0; i < 8; ++i) {
        HWY_ASSERT_EQ(min_s[i], actual_min[i]);
        HWY_ASSERT_EQ(max_s[i], actual_max[i]);
        HWY_ASSERT_EQ(sum_s[i], actual_sum[i]);
      }
    }
  }
};

void TestAllReduce() {
  ForAllTypes(ForPartialVectors<TestReduce>());
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(MinMaxTest);
HWY_EXPORT_AND_TEST_P(MinMaxTest, TestAllMinValue);
HWY_EXPORT_AND_TEST_P(MinMaxTest, TestAllMaxValue);
HWY_EXPORT_AND_TEST_P(MinMaxTest, TestAllReduce);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
