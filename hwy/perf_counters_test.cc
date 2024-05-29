// Copyright 2024 Google LLC
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

#include "hwy/perf_counters.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "hwy/nanobenchmark.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

namespace hwy {
namespace {

TEST(NanobenchmarkTest, RunTest) {
  RandomState rng;
  platform::PMU pmu;
  if (pmu.Start()) {
    const size_t iters = (hwy::Unpredictable1() * 1000) + (rng() & 1);
    uint64_t r = rng();
    fprintf(stderr, "r: %zu\n", r);
    for (size_t i = 0; i < iters; ++i) {
      if (PopCount(rng()) < 36) {
        r -= rng() & 0xF;
      } else {
        // Entirely different operation to ensure there is a branch.
        r >>= 1;
      }
    }

    platform::PerfCounters counters;
    const double min_coverage = pmu.Stop(counters);
    fprintf(stderr, "r: %d, coverage %f\n", static_cast<int>(r), min_coverage);
    if (min_coverage != 0.0) {
#if HWY_CXX_LANG >= 201402L
      counters.ForEach(&counters,
                       [](auto& val, auto& /*val2*/, const char* name) {
                         fprintf(stderr, "%-20s: %.3E\n", name, val);
                       });
#endif

      HWY_ASSERT(counters.ref_cycle > 1000);
      HWY_ASSERT(counters.instruction > 1000);
      HWY_ASSERT(counters.branch > 1000);
      HWY_ASSERT(counters.branch_mispred > 200);
    }
  }
}

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
