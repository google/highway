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

#include <algorithm>  // std::sort
#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/algo/is_sorted_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/algo/is_sorted-inl.h"
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

// Scalar references matching std::is_sorted with the default and `greater`
// comparators, respectively.
template <typename T>
bool ScalarIsSortedAsc(const T* in, size_t count) {
  for (size_t i = 1; i < count; ++i) {
    if (in[i] < in[i - 1]) return false;
  }
  return true;
}

template <typename T>
bool ScalarIsSortedDesc(const T* in, size_t count) {
  for (size_t i = 1; i < count; ++i) {
    if (in[i - 1] < in[i]) return false;
  }
  return true;
}

// Comparator for the IsSorted overload: checks non-increasing order.
struct GreaterComp {
  template <class D, class V>
  Mask<D> operator()(D /*d*/, V a, V b) const {
    return Gt(a, b);
  }
};

template <class Test>
struct ForeachCountAndMisalign {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) const {
    RandomState rng;
    const size_t N = Lanes(d);
    const size_t misalignments[3] = {0, N / 4, 3 * N / 5};

    // Trivial and boundary cases are always covered, plus random lengths.
    std::vector<size_t> counts = {0, 1, 2, N, N + 1, 2 * N, 2 * N + 1};
    const size_t num_random = AdjustedReps(512);
    counts.reserve(counts.size() + num_random);
    for (size_t k = 0; k < num_random; ++k) {
      counts.push_back(static_cast<size_t>(rng()) % (16 * N + 1));
    }

    for (size_t count : counts) {
      for (size_t m : misalignments) {
        Test()(d, count, m, rng);
      }
    }
  }
};

template <class D, typename T = TFromD<D>>
void Check(D d, size_t count, size_t misalign, bool expected, bool actual,
           const char* label) {
  if (expected != actual) {
    fprintf(stderr,
            "%s count %d misalign %d [%s]: IsSorted expected %d got %d\n",
            hwy::TypeName(T(), Lanes(d)).c_str(), static_cast<int>(count),
            static_cast<int>(misalign), label, static_cast<int>(expected),
            static_cast<int>(actual));
    HWY_ASSERT(false);
  }
}

struct TestIsSorted {
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

    // Random data: usually unsorted, occasionally sorted by chance.
    Check(d, count, misalign, ScalarIsSortedAsc(in, count),
          IsSorted(d, in, count), "random");

    // Sorted ascending: must report sorted.
    std::sort(in, in + count, [](const T& a, const T& b) { return a < b; });
    Check(d, count, misalign, ScalarIsSortedAsc(in, count),
          IsSorted(d, in, count), "sorted");

    // All-equal: sorted (no strict descent), exercises tie handling.
    for (size_t i = 0; i < count; ++i) {
      in[i] = ConvertScalarTo<T>(1.0);
    }
    Check(d, count, misalign, true, IsSorted(d, in, count), "all-equal");

    // Single descent: must report unsorted. Deterministically cover the
    // block-boundary and last-pair positions, plus one random position.
    if (count >= 2) {
      const size_t N = Lanes(d);
      const size_t positions[8] = {0,
                                   N - 1,
                                   N,
                                   2 * N - 1,
                                   2 * N,
                                   count / 2,
                                   count - 2,
                                   static_cast<size_t>(rng()) % (count - 1)};
      for (size_t pos : positions) {
        if (pos > count - 2) continue;
        in[pos] = ConvertScalarTo<T>(2.0);
        Check(d, count, misalign, false, IsSorted(d, in, count), "descent");
        in[pos] = ConvertScalarTo<T>(1.0);  // restore all-equal
      }
    }
  }
};

struct TestIsSortedComp {
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

    Check(d, count, misalign, ScalarIsSortedDesc(in, count),
          IsSorted(d, in, count, GreaterComp()), "random-desc");

    // Sorted descending: must report sorted under the Gt comparator.
    std::sort(in, in + count, [](const T& a, const T& b) { return b < a; });
    Check(d, count, misalign, ScalarIsSortedDesc(in, count),
          IsSorted(d, in, count, GreaterComp()), "sorted-desc");

    // All-equal: sorted under any strict comparator.
    for (size_t i = 0; i < count; ++i) {
      in[i] = ConvertScalarTo<T>(1.0);
    }
    Check(d, count, misalign, true, IsSorted(d, in, count, GreaterComp()),
          "all-equal-desc");

    // Single ascent: must report unsorted under the Gt comparator.
    // Deterministically cover block-boundary and last-pair positions.
    if (count >= 2) {
      const size_t N = Lanes(d);
      const size_t positions[8] = {0,
                                   N - 1,
                                   N,
                                   2 * N - 1,
                                   2 * N,
                                   count / 2,
                                   count - 2,
                                   static_cast<size_t>(rng()) % (count - 1)};
      for (size_t pos : positions) {
        if (pos > count - 2) continue;
        in[pos] = ConvertScalarTo<T>(0.0);
        Check(d, count, misalign, false, IsSorted(d, in, count, GreaterComp()),
              "ascent-desc");
        in[pos] = ConvertScalarTo<T>(1.0);  // restore all-equal
      }
    }
  }
};

void TestAllIsSorted() {
  ForAllTypes(ForPartialVectors<ForeachCountAndMisalign<TestIsSorted>>());
}

void TestAllIsSortedComp() {
  ForAllTypes(ForPartialVectors<ForeachCountAndMisalign<TestIsSortedComp>>());
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(IsSortedTest);
HWY_EXPORT_AND_TEST_P(IsSortedTest, TestAllIsSorted);
HWY_EXPORT_AND_TEST_P(IsSortedTest, TestAllIsSortedComp);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
