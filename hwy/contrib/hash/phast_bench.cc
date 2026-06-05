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

#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/contrib/thread_pool/topology.h"
#include "hwy/nanobenchmark.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/hash/phast_bench.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/hash/phast-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
#if (HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128) || HWY_IDE

HWY_NOINLINE void TestLatency(const Phast& phast) {
  if (phast.IsEmpty()) {
    HWY_WARN("Phast build failed, skipping latency test.\n");
    return;
  }

  FuncInput input = Unpredictable1();
  Params params = DefaultBenchmarkParams();
  params.verbose = false;
  Result results[1];

  const size_t num_results = MeasureClosure(
      [&phast](FuncInput func_input) {
        return phast(static_cast<uint32_t>(func_input));
      },
      &input, 1, results, params);
  if (num_results == 1) {
    const double ns =
        results[0].ticks / platform::InvariantTicksPerSecond() * 1E9;
    printf("Query latency: %6.2f ns; measurement MAD=%4.2f%%\n", ns,
           results[0].variability * 100.0);
  } else {
    HWY_WARN("Measurement failed.");
  }
}

HWY_NOINLINE void TestThroughput(const Phast& phast,
                                 const AlignedVector<uint32_t>& keys) {
  if (phast.IsEmpty()) {
    HWY_WARN("Phast build failed, skipping throughput test.\n");
    return;
  }

  AlignedVector<uint32_t> indices(keys.size());

  FuncInput input = Unpredictable1();
  Result results[1];
  Params params = DefaultBenchmarkParams();
  params.verbose = false;

  const size_t num_results = MeasureClosure(
      [&](FuncInput func_input) {
        phast.QueryBatch(keys.data(), keys.size(), indices.data());
        return indices[func_input];
      },
      &input, 1, results, params);
  if (num_results == 1) {
    const double ns =
        results[0].ticks / platform::InvariantTicksPerSecond() * 1E9;
    printf(
        "Query batch throughput: %7.2f ns = %4.1f MB/s; measurement "
        "MAD=%4.2f%%\n",
        ns, static_cast<double>(keys.size() * sizeof(uint32_t)) / ns * 1E3,
        results[0].variability * 100.0);
  } else {
    HWY_WARN("Measurement failed.");
  }
}

HWY_NOINLINE AlignedVector<uint32_t> GenerateKeys(size_t num_keys) {
  // Must be distinct, hence do not use FillRandom().
  AlignedVector<uint32_t> keys;
  keys.reserve(num_keys);
  AesCtrEngine engine(/*deterministic=*/true);
  Triple32 permutation(engine, Unpredictable1());
  for (size_t i = 0; i < num_keys; ++i) {
    keys.push_back(permutation(i));
  }
  return keys;
}

static ThreadPool MakePool() {
  static Topology topology;
  if (topology.packages.empty()) return ThreadPool(ThreadPool::MaxThreads());
  // Minus one because these are in addition to the main thread.
  return ThreadPool(topology.packages[0].cores.size() - 1);
}

HWY_NOINLINE Phast MakePhast(const AlignedVector<uint32_t>& keys) {
  const size_t num_keys = keys.size();
  const uint32_t slice_length = num_keys > 256 * 1024   ? 4096
                                : num_keys >= 10 * 1000 ? 512
                                                        : 256;
  const uint32_t headroom_percent = num_keys > 256 * 1024 ? 2 : 5;
  PhastConfig config(num_keys, 2, slice_length, headroom_percent);
  ThreadPool pool = MakePool();
  return BuildPhast(keys.data(), config, pool);
}

HWY_NOINLINE void TestAllLatency() {
  const AlignedVector<uint32_t> keys = GenerateKeys(1000);
  TestLatency(MakePhast(keys));
}
HWY_NOINLINE void TestAllThroughput() {
  const size_t num_keys = HWY_IS_DEBUG_BUILD ? 10 * 1000 : 1000 * 1000;
  const AlignedVector<uint32_t> keys = GenerateKeys(num_keys);
  TestThroughput(MakePhast(keys), keys);
}

#else   // HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128
void TestAllLatency() {}
void TestAllThroughput() {}
#endif  // HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(PhastBench);
HWY_EXPORT_AND_TEST_BEST_P(PhastBench, TestAllLatency);
HWY_EXPORT_AND_TEST_BEST_P(PhastBench, TestAllThroughput);
HWY_AFTER_TEST();
}  // namespace hwy
#endif  // HWY_ONCE
