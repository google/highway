// Copyright 2021 Google LLC
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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/sort/sort_test.cc"
#include "hwy/foreach_target.h"

#include "hwy/contrib/sort/sort-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

#if HWY_TARGET != HWY_SCALAR && HWY_ARCH_X86

class TestRanges {
  constexpr size_t K() { return 32; }

  template <SortOrder kOrder, class D>
  void Validate(D d, const TFromD<D>* in, const TFromD<D>* out) {
    const size_t N = Lanes(d);
    // For each range:
    for (size_t i = 0; i < 8 * N; i += K()) {
      // Ensure it matches the sort order
      for (size_t j = 0; j < K() - 1; ++j) {
        if (!Compare<kOrder>(out[i + j], out[i + j + 1])) {
          printf("range=%zu lane=%zu N=%zu %.0f %.0f\n\n", i, j, N,
                 static_cast<float>(out[i + j + 0]),
                 static_cast<float>(out[i + j + 1]));
          for (size_t k = 0; k < K(); ++k) {
            printf("%.0f\n", static_cast<float>(out[i + k]));
          }

          printf("\n\nin was:\n");
          for (size_t k = 0; k < K(); ++k) {
            printf("%.0f\n", static_cast<float>(in[i + k]));
          }
          fflush(stdout);
          HWY_ABORT("Sort is incorrect");
        }
      }
    }

    // Also verify sums match (detects duplicated/lost values)
    double expected_sum = 0.0;
    double actual_sum = 0.0;
    for (size_t i = 0; i < 8 * N; ++i) {
      expected_sum += in[i];
      actual_sum += out[i];
    }
    if (expected_sum != actual_sum) {
      for (size_t i = 0; i < 8 * N; ++i) {
        printf("%.0f  %.0f\n", static_cast<float>(in[i]),
               static_cast<float>(out[i]));
      }
      HWY_ABORT("Mismatch");
    }
  }

  template <SortOrder kOrder, class D>
  void TestOrder(D d, RandomState& rng) {
    using T = TFromD<D>;
    const size_t N = Lanes(d);
    HWY_ASSERT((N % 4) == 0);
    auto in = AllocateAligned<T>(8 * N);
    auto inout = AllocateAligned<T>(8 * N);

    // For each range, try all 0/1 combinations and set any other lanes to
    // random inputs.
    for (size_t range = 0; range < 8 * N; range += K()) {
      // First set all to random, will later overwrite those for `range`
      for (size_t i = 0; i < 8 * N; ++i) {
        in[i] = static_cast<T>(Random32(&rng) & 0xFF);
      }

      // Now try all combinations of {0,1} for lanes in the range. This is
      // sufficient to establish correctness (arbitrary inputs could be
      // mapped to 0/1 with a comparison predicate). Need to stop after
      // 20 bits so tests run quickly enough.
      const size_t max_bits = AdjustedReps(1ull << HWY_MIN(K(), 20));
      for (size_t bits = 0; bits < max_bits; ++bits) {
        for (size_t i = 0; i < K(); ++i) {
          in[range + i] = (bits >> i) & 1;
        }

        for (size_t i = 0; i < 8 * N; ++i) {
          inout[i] = in[i];
        }
        Sort8Vectors<kOrder>(d, inout.get());
        Validate<kOrder>(d, in.get(), inout.get());
      }
    }
  }

 public:
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    RandomState rng;
    TestOrder<SortOrder::kAscending>(d, rng);
    TestOrder<SortOrder::kDescending>(d, rng);
  }
};

void TestAllRanges() {
  TestRanges test;
  // TODO(janwas): ScalableTag
  test(int32_t(), CappedTag<int32_t, 4>());
  test(uint32_t(), CappedTag<uint32_t, 4>());
}

#else
void TestAllRanges() {}
#endif  // HWY_TARGET != HWY_SCALAR && HWY_ARCH_X86

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
HWY_BEFORE_TEST(SortTest);
HWY_EXPORT_AND_TEST_P(SortTest, TestAllRanges);
}  // namespace hwy

// Ought not to be necessary, but without this, no tests run on RVV.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif
