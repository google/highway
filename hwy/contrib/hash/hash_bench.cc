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

#include <stdint.h>
#include <stdio.h>

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

#include "hwy/nanobenchmark.h"
#include "hwy/per_target.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/hash/hash_bench.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/hash/hash-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
#if HWY_TARGET != HWY_SCALAR

template <class IHash>
HWY_NOINLINE void TestLatency(const IHash& hash) {
  FuncInput input = static_cast<FuncInput>(Unpredictable1());
  Params params = DefaultBenchmarkParams();
  params.verbose = false;
  Result results[1];

  const size_t num_results = MeasureClosure(
      [&hash](FuncInput func_input) {
        return hash(static_cast<uint32_t>(func_input));
      },
      &input, 1, results, params);
  if (num_results == 1) {
    const double ns =
        results[0].ticks / platform::InvariantTicksPerSecond() * 1E9;
    printf("%12s: %6.2f ns = %4.1f GB/s; measurement MAD=%4.2f%%\n",
           hash.Name(), ns, static_cast<double>(VectorBytes()) / ns,
           results[0].variability * 100.0);
  } else {
    HWY_WARN("Measurement failed.");
  }
}

template <class IHash>
HWY_NOINLINE void TestThroughput(const IHash& hash) {
  const size_t kNumU32 = 4096;
  HWY_ALIGN_MAX uint32_t inout[kNumU32];
  for (size_t i = 0; i < kNumU32; ++i) {
    inout[i] = static_cast<uint32_t>(Unpredictable1());
  }

  FuncInput input = static_cast<FuncInput>(Unpredictable1());
  Result results[1];
  Params params = DefaultBenchmarkParams();
  params.verbose = false;

  const size_t num_results = MeasureClosure(
      [&](FuncInput func_input) {
        HashArray(hash, inout, kNumU32);
        return inout[func_input];
      },
      &input, 1, results, params);
  if (num_results == 1) {
    const double ns =
        results[0].ticks / platform::InvariantTicksPerSecond() * 1E9;
    printf("%12s: %7.2f ns = %4.1f GB/s; measurement MAD=%4.2f%%\n",
           hash.Name(), ns, kNumU32 * sizeof(uint32_t) / ns,
           results[0].variability * 100.0);
  } else {
    HWY_WARN("Measurement failed.");
  }
}

HWY_NOINLINE void TestAllLatency() {
  AesCtrEngine engine(/*deterministic=*/true);
  ForeachHash(engine, 0, [](const auto& hash) { TestLatency(hash); });
}

HWY_NOINLINE void TestAllThroughput() {
  AesCtrEngine engine(/*deterministic=*/true);
  ForeachHash(engine, 0, [](const auto& hash) { TestThroughput(hash); });
}

#else   // HWY_TARGET == HWY_SCALAR
void TestAllLatency() {}
void TestAllThroughput() {}
#endif  // HWY_TARGET != HWY_SCALAR

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(HashBench);
HWY_EXPORT_AND_TEST_BEST_P(HashBench, TestAllLatency);
HWY_EXPORT_AND_TEST_BEST_P(HashBench, TestAllThroughput);
HWY_AFTER_TEST();
}  // namespace hwy
#endif  // HWY_ONCE
