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

// Tests for PHAST perfect hash.

#include <stdint.h>
#include <stdio.h>

#include <algorithm>  // std::unique
#include <vector>

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

#include "hwy/contrib/sort/vqsort.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/contrib/thread_pool/topology.h"
#include "hwy/nanobenchmark.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/hash/phast_test.cc"  // NOLINT
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

// --------------------------------------------------------------------------
// AesCtrEngine requires AES-NI, which is not available on HWY_SCALAR.
#if HWY_TARGET == HWY_SCALAR
HWY_NOINLINE void TestAllRoundtrip() {}
HWY_NOINLINE void TestQueryConsistency() {}
HWY_NOINLINE void TestHeadroomSweep() {}
#else

static ThreadPool MakePool() {
  static Topology topology;
  if (topology.packages.empty()) return ThreadPool(ThreadPool::MaxThreads());
  // Minus one because these are in addition to the main thread.
  return ThreadPool(topology.packages[0].cores.size() - 1);
}

// --------------------------------------------------------------------------
// Test: round-trip, coverage, collision

void CheckUniqueAndRange(uint32_t* indices, uint32_t num_indices,
                         uint32_t num_slots) {
  VQSort(indices, num_indices, SortAscending());
  uint32_t* end = std::unique(indices, indices + num_indices);
  HWY_ASSERT_M(end == indices + num_indices, "Collision detected");

  for (uint32_t i = 0; i < num_indices; ++i) {
    HWY_ASSERT_M(indices[i] < num_slots, "Index out of range");
  }
}

void TestRoundTrip(uint32_t num_keys, uint32_t keys_per_bucket,
                   uint32_t slice_length) {
  // Generate distinct keys: Triple32(iota).
  AlignedVector<uint32_t> keys(num_keys);
  AesCtrEngine engine(/*deterministic=*/true);
  Triple32 permutation(engine, 0);
  for (uint32_t i = 0; i < num_keys; ++i) {
    keys[i] = permutation(i);
  }

  const double t0 = platform::Now();
  PhastConfig config(num_keys, keys_per_bucket, slice_length);
  ThreadPool pool = MakePool();
  Phast phast = BuildPhast(keys.data(), config, pool);
  const double elapsed = platform::Now() - t0;
  HWY_ASSERT_M(phast.Config().num_keys != 0, "Build failed");
  printf("  Build(%u keys, lambda=%u, L=%u): %.2f ms, num_slots=%u\n", num_keys,
         keys_per_bucket, slice_length, elapsed * 1e3,
         phast.Config().num_slots);

  // Check that all keys map to distinct indices in [0, num_slots).
  const uint32_t num_slots = phast.Config().num_slots;
  AlignedVector<uint32_t> indices(num_keys);
  phast.QueryBatch(keys.data(), num_keys, indices.data());
  CheckUniqueAndRange(indices.data(), num_keys, num_slots);

  // Report space usage.
  const double bits_per_key =
      static_cast<double>(phast.Config().num_buckets * 8) /
      static_cast<double>(num_keys);
  printf("  Space: %.2f bits/key\n", bits_per_key);
}

HWY_NOINLINE void TestAllRoundtrip() {
  printf("=== TestSmall ===\n");
  TestRoundTrip(/*num_keys=*/100, /*keys_per_bucket=*/3,
                /*slice_length=*/2048);
  printf("=== TestMedium ===\n");
  TestRoundTrip(/*num_keys=*/10000, /*keys_per_bucket=*/2,
                /*slice_length=*/4096);
}

// --------------------------------------------------------------------------
// Test: queries are repeatable.

HWY_NOINLINE void TestQueryConsistency() {
  printf("=== TestQueryConsistency ===\n");
  const uint32_t num_keys = 5000;
  AlignedVector<uint32_t> keys(num_keys);
  for (uint32_t i = 0; i < num_keys; ++i) {
    keys[i] = i * 37 + 1;  // Distinct, non-sequential.
  }

  PhastConfig config(num_keys);
  ThreadPool pool = MakePool();
  Phast phast = BuildPhast(keys.data(), config, pool);

  // Query each key twice and verify same result.
  for (uint32_t i = 0; i < num_keys; ++i) {
    const uint32_t idx1 = phast(keys[i]);
    const uint32_t idx2 = phast(keys[i]);
    HWY_ASSERT_M(idx1 == idx2, "Query not deterministic");
  }
  printf("  OK: %u queries consistent\n", num_keys);
}

// --------------------------------------------------------------------------
// Headroom sweep: find minimum viable overprovisioning

HWY_NOINLINE void TestHeadroomSweep() {
  printf("\n=== Headroom Sweep ===\n");
  printf("bytes   h%% kpb L  ");

  // Pre-generate keys once at max size.
  const size_t kMaxN = 1000 * 1000;
  AesCtrEngine engine(/*deterministic=*/true);
  AlignedVector<uint32_t> all_keys = FillRandom<uint32_t>(kMaxN, engine, 0);
  VQSort(all_keys.data(), kMaxN, SortAscending());
  // Remove duplicates.
  uint32_t* unique = std::unique(all_keys.data(), all_keys.data() + kMaxN);
  const size_t num_unique = unique - all_keys.data();

  const size_t num_keys[] = {num_unique};
  for (size_t s : num_keys) {
    printf("  n=%-6zu            ", s);
  }
  printf("\n");

  // Generate configs to check.
  AlignedVector<PhastConfig> configs;
  for (uint32_t keys_per_bucket : {2}) {
    for (uint32_t slice_length : {8192, 4096}) {
      for (uint32_t headroom : {4}) {
        for (size_t n : num_keys) {
          constexpr uint32_t kMaxRetries = 300;
          configs.emplace_back(n, keys_per_bucket, slice_length, headroom,
                               kMaxRetries);
        }
      }
    }
  }

  ThreadPool pool = MakePool();
  for (PhastConfig& in_config : configs) {
    PhastStats stats;
    Phast phast = BuildPhast(all_keys.data(), in_config, pool, &stats);
    const PhastConfig& config = phast.Config();
    HWY_ASSERT(stats.success == (config.num_keys != 0));

    if (stats.success) {
      printf("%s round %zu worker %zu\n", config.ToString().c_str(),
             stats.round, stats.worker);

      // Verify correctness.
      AlignedVector<uint32_t> indices(config.num_keys);
      phast.QueryBatch(all_keys.data(), config.num_keys, indices.data());
      CheckUniqueAndRange(indices.data(), config.num_keys, config.num_slots);
    } else {
      printf("%s failed; rank: %s\n", config.ToString().c_str(),
             stats.s_rank.ToString().c_str());
    }
  }

  PROFILER_PRINT_RESULTS();
}

#endif  // HWY_TARGET != HWY_SCALAR

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(PhastTest);
HWY_EXPORT_AND_TEST_BEST_P(PhastTest, TestAllRoundtrip);
HWY_EXPORT_AND_TEST_BEST_P(PhastTest, TestQueryConsistency);
HWY_EXPORT_AND_TEST_BEST_P(PhastTest, TestHeadroomSweep);
HWY_AFTER_TEST();
}  // namespace hwy
#endif  // HWY_ONCE
