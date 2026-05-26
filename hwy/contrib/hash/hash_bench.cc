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

template <class IHash>
HWY_NOINLINE void TestLatency(const IHash& hash) {
  FuncInput input = Unpredictable1();
  Params params = DefaultBenchmarkParams();
  params.verbose = false;
  Result results[1];

  const size_t num_results = MeasureClosure(
      [&hash](FuncInput input) { return hash(static_cast<uint32_t>(input)); },
      &input, 1, results, params);
  if (num_results == 1) {
    const double ns =
        results[0].ticks / platform::InvariantTicksPerSecond() * 1E9;
    printf("%12s: %6.2f ns = %4.1f GB/s; measurement MAD=%4.2f%%\n",
           hash.Name(), ns, VectorBytes() / ns, results[0].variability * 100.0);
  } else {
    HWY_WARN("Measurement failed.");
  }
}

// Each Hash* only provides TwoVec. This adapter avoids duplicating a loop for
// each hash function.
template <class Hash>
static void HashArray(const Hash& hash, uint32_t* HWY_RESTRICT inout,
                      size_t count) {
  const ScalableTag<uint32_t> du32;
  using VU32 = Vec<decltype(du32)>;
  HWY_LANES_CONSTEXPR size_t N = Lanes(du32);

  HWY_DASSERT(count % (4 * N) == 0);
  for (size_t i = 0; i < count; i += 4 * N) {
    VU32 v0 = Load(du32, inout + i + 0 * N);
    VU32 v1 = Load(du32, inout + i + 1 * N);
    VU32 v2 = Load(du32, inout + i + 2 * N);
    VU32 v3 = Load(du32, inout + i + 3 * N);
    hash.TwoVec(du32, v0, v1);
    hash.TwoVec(du32, v2, v3);
    Store(v0, du32, inout + i + 0 * N);
    Store(v1, du32, inout + i + 1 * N);
    Store(v2, du32, inout + i + 2 * N);
    Store(v3, du32, inout + i + 3 * N);
  }
}

template <class IHash>
HWY_NOINLINE void TestThroughput(const IHash& hash) {
  const size_t kNumU32 = 4096;
  HWY_ALIGN_MAX uint32_t inout[kNumU32];
  for (size_t i = 0; i < kNumU32; ++i) {
    inout[i] = static_cast<uint32_t>(Unpredictable1());
  }

  FuncInput input = Unpredictable1();
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
