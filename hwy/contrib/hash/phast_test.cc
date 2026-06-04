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

// Mutates input.
void CheckUniqueAndRange(uint32_t* indices, uint32_t num_indices,
                         uint32_t num_slots) {
  VQSort(indices, num_indices, SortAscending());
  uint32_t* end = std::unique(indices, indices + num_indices);
  HWY_ASSERT_M(end == indices + num_indices, "Collision detected");

  for (uint32_t i = 0; i < num_indices; ++i) {
    HWY_ASSERT_M(indices[i] < num_slots, "Index out of range");
  }
}

void TestRoundTrip(const uint32_t num_keys, const uint32_t keys_per_bucket,
                   const uint32_t slice_length) {
  // Generate distinct keys: Triple32(iota).
  AlignedVector<uint32_t> keys(num_keys);
  AesCtrEngine engine(/*deterministic=*/true);
  Triple32 permutation(engine, 0);
  for (uint32_t i = 0; i < num_keys; ++i) {
    keys[i] = permutation(i);
  }

  const double t0 = platform::Now();
  const uint32_t headroom_percent = 6;  // small num_keys -> higher headroom
  const uint32_t max_retries = 1000;
  const PhastConfig config(num_keys, keys_per_bucket, slice_length,
                           headroom_percent, max_retries);
  ThreadPool pool = MakePool();
  const Phast phast = BuildPhast(keys.data(), config, pool);
  const double elapsed = platform::Now() - t0;
  HWY_ASSERT_M(!phast.IsEmpty(), "Build failed");
  fprintf(stderr, "  Build(%u keys, lambda=%u, L=%u): %.2f ms\n", num_keys,
          keys_per_bucket, slice_length, elapsed * 1e3);

  // Check that all keys map to distinct indices in [0, num_slots).
  AlignedVector<uint32_t> indices(num_keys);
  phast.QueryBatch(keys.data(), num_keys, indices.data());
  CheckUniqueAndRange(indices.data(), num_keys, phast.Config().NumSlots());

  // Report space usage.
  const double bits_per_key =
      static_cast<double>(phast.Config().ExtraBytes() * 8) /
      static_cast<double>(num_keys);
  fprintf(stderr, "  Space: %.2f bits/key\n", bits_per_key);
}

HWY_NOINLINE void TestAllRoundtrip() {
  fprintf(stderr, "=== TestSmall ===\n");
  TestRoundTrip(/*num_keys=*/10 * 1000, /*keys_per_bucket=*/3,
                /*slice_length=*/1024);
  fprintf(stderr, "=== TestMedium ===\n");
  TestRoundTrip(/*num_keys=*/100 * 1000, /*keys_per_bucket=*/2,
                /*slice_length=*/2048);
}

// --------------------------------------------------------------------------
// Test: queries are repeatable.

HWY_NOINLINE void TestQueryConsistency() {
  fprintf(stderr, "=== TestQueryConsistency ===\n");
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
  fprintf(stderr, "  OK: %u queries consistent\n", num_keys);
}

// --------------------------------------------------------------------------
// Headroom sweep: find minimum viable overprovisioning

HWY_NOINLINE void TestHeadroomSweep() {
  fprintf(stderr, "bytes   h%% kpb L\n");

  // Pre-generate keys once at max size.
  const size_t kMaxN = 1000 * 1000;
  AesCtrEngine engine(/*deterministic=*/true);
  AlignedVector<uint32_t> all_keys = FillRandom<uint32_t>(kMaxN, engine, 0);
  VQSort(all_keys.data(), kMaxN, SortAscending());
  // Remove duplicates.
  uint32_t* unique = std::unique(all_keys.data(), all_keys.data() + kMaxN);
  const size_t num_keys = unique - all_keys.data();

  // Generate configs to check.
  AlignedVector<PhastConfig> configs;
  for (uint32_t keys_per_bucket : {2}) {
    for (uint32_t slice_length : {4096, 8192}) {
      // Test the greedy path by including 4 (no Cuckoo swaps required).
      for (uint32_t headroom : {2, 4}) {
        constexpr uint32_t kMaxRetries = 300;
        configs.emplace_back(num_keys, keys_per_bucket, slice_length, headroom,
                             kMaxRetries);
      }
    }
  }

  ThreadPool pool = MakePool();
  for (PhastConfig& in_config : configs) {
    PhastStats stats;
    Phast phast = BuildPhast(all_keys.data(), in_config, pool, &stats);
    HWY_ASSERT(stats.success == !phast.IsEmpty());
    const PhastConfig& config = phast.Config();

    char config_str[100];
    if (stats.success) {
      config.ToString(config_str, sizeof(config_str));
      fprintf(stderr, "%s round %zu worker %zu\n", config_str, stats.round,
              stats.worker);

      // Verify correctness.
      AlignedVector<uint32_t> indices(num_keys);
      phast.QueryBatch(all_keys.data(), num_keys, indices.data());
      CheckUniqueAndRange(indices.data(), num_keys, config.NumSlots());
    } else {
      in_config.ToString(config_str, sizeof(config_str));
      fprintf(stderr, "%s failed; rank: %s\n", config_str,
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
