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

// Tests for static cuckoo hashing.

#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <chrono>  // NOLINT
#include <set>

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

#include "hwy/contrib/sort/vqsort.h"
#include "hwy/nanobenchmark.h"
#include "hwy/timer.h"
#include "third_party/ortools/ortools/graph/min_cost_flow.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/hash/cuckoo_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/hash/cuckoo-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

// --------------------------------------------------------------------------
#if (HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128) && !HWY_IDE
HWY_NOINLINE void TestAllBuildSmall() {}
HWY_NOINLINE void TestAllBuildMedium() {}
HWY_NOINLINE void TestAllQueryCorrectness() {}
HWY_NOINLINE void TestAllBatchQuery() {}
HWY_NOINLINE void TestAllEpsilonSweep() {}
HWY_NOINLINE void TestAllOptimizedBuild() {}
HWY_NOINLINE void TestAllMinCostFlowComparison() {}
#else

// --------------------------------------------------------------------------
// Helper: generate distinct keys using a permutation.

static AlignedVector<uint32_t> GenerateKeys(uint32_t num_keys,
                                            uint64_t seed = 0) {
  if (num_keys >= 1000000) {
    fprintf(stderr, "GenerateKeys(%u) starting...\n", num_keys);
  }
  AlignedVector<uint32_t> keys(num_keys);
  AesCtrEngine engine(/*deterministic=*/true);
  Triple32 perm(engine, seed);
  for (uint32_t i = 0; i < num_keys; ++i) {
    keys[i] = perm(i);
    // Ensure no key equals the sentinel value.
    if (keys[i] == CuckooConfig::kEmpty) keys[i] = 0;
  }
  if (num_keys >= 1000000) {
    fprintf(stderr, "GenerateKeys(%u) finished.\n", num_keys);
  }
  return keys;
}

// --------------------------------------------------------------------------
// Test: small build (100 keys)

namespace {
void TestKeys(uint32_t num_keys) {
  auto keys = GenerateKeys(num_keys);

  CuckooBuildStats stats;
  const double t0 = platform::Now();
  CuckooTable table =
      CuckooBuild(keys.data(), num_keys, /*epsilon=*/0.25,
                  /*max_attempts=*/100, /*optimize_primary=*/false, &stats);
  const double elapsed = platform::Now() - t0;

  HWY_ASSERT_M(stats.success, "Build failed for 100 keys");
  HWY_ASSERT_M(!table.IsEmpty(), "Table should not be empty");

  fprintf(stderr, "  Build(%u keys): %.2f ms, attempt=%u, primary=%u/%u\n",
          num_keys, elapsed * 1e3, stats.global_seed, stats.num_primary,
          num_keys);
}
}  // namespace

HWY_NOINLINE void TestAllBuildSmall() {
  fprintf(stderr, "=== TestBuildSmall ===\n");
  TestKeys(100);
}

// --------------------------------------------------------------------------
// Test: medium build (10000 keys)

HWY_NOINLINE void TestAllBuildMedium() {
  fprintf(stderr, "=== TestBuildMedium ===\n");
  TestKeys(10000);
}

// --------------------------------------------------------------------------
// Test: query correctness — every inserted key is found

HWY_NOINLINE void TestAllQueryCorrectness() {
  fprintf(stderr, "=== TestQueryCorrectness ===\n");
  const uint32_t num_keys = 5000;
  auto keys = GenerateKeys(num_keys);

  CuckooTable table = CuckooBuild(keys.data(), num_keys, /*epsilon=*/0.25);
  HWY_ASSERT_M(!table.IsEmpty(), "Build failed");

  // Every inserted key must be found.
  for (uint32_t i = 0; i < num_keys; ++i) {
    HWY_ASSERT_M(table.QueryOne(keys[i]), "QueryOne missed a key");
  }

  // A key that was not inserted should (very likely) not be found.
  // Use a different permutation to generate non-member keys.
  auto non_keys = GenerateKeys(1000, /*seed=*/999);
  for (uint32_t i = 0; i < 1000; ++i) {
    // Skip if this key happens to be in the original set.
    bool is_member = false;
    for (uint32_t j = 0; j < num_keys; ++j) {
      if (non_keys[i] == keys[j]) {
        is_member = true;
        break;
      }
    }
    HWY_ASSERT(is_member || !table.QueryOne(non_keys[i]));
  }
}

// --------------------------------------------------------------------------
// Test: batch query matches single query

HWY_NOINLINE void TestAllBatchQuery() {
  fprintf(stderr, "=== TestBatchQuery ===\n");
  const uint32_t num_keys = 2000;
  auto keys = GenerateKeys(num_keys);

  CuckooTable table = CuckooBuild(keys.data(), num_keys, /*epsilon=*/0.25);
  HWY_ASSERT_M(!table.IsEmpty(), "Build failed");

  const ScalableTag<uint32_t> du32;
  HWY_LANES_CONSTEXPR size_t N = Lanes(du32);

  // All member keys must be found.
  size_t i = 0;
  for (; i + N <= num_keys; i += N) {
    auto not_found = table.QueryBatch(du32, keys.data() + i);
    HWY_ASSERT_M(AllFalse(du32, not_found), "Batch query missed a key");
    not_found = table.QueryBatchSmallEpsilon(du32, keys.data() + i);
    HWY_ASSERT_M(AllFalse(du32, not_found),
                 "Batch query missed a key (small epsilon)");
  }

  for (; i < num_keys; ++i) {
    HWY_ASSERT_M(table.QueryOne(keys[i]), "Batch query missed a key (tail)");
  }

  // Non-member keys must not produce false positives.
  AlignedVector<uint32_t> non_keys(RoundUpTo(N, N));
  for (size_t j = 0; j < non_keys.size(); ++j) {
    non_keys[j] = num_keys + 1000 + static_cast<uint32_t>(j);
  }
  const auto not_found = table.QueryBatchSmallEpsilon(du32, non_keys.data());
  HWY_ASSERT_M(AllTrue(du32, not_found),
               "QueryBatchSmallEpsilon false positive on non-key");

  fprintf(stderr, "  OK: batch query matches single query for %u keys\n",
          num_keys);
}

// --------------------------------------------------------------------------
// Test: epsilon sweep

HWY_NOINLINE void TestAllEpsilonSweep() {
  fprintf(stderr, "=== TestEpsilonSweep ===\n");
  const uint32_t num_keys = HWY_IS_DEBUG_BUILD ? 1000 : 5000;

  auto keys = GenerateKeys(num_keys);

  for (double eps : {0.05, 0.10, 0.25, 0.50}) {
    CuckooBuildStats stats;
    CuckooTable table = CuckooBuild(keys.data(), num_keys, eps,
                                    /*max_attempts=*/200,
                                    /*optimize_primary=*/false, &stats);
    (void)table;

    if (stats.success) {
      fprintf(stderr, "  eps=%.2f: OK, attempt=%u, primary=%u/%u (%.1f%%)\n",
              eps, stats.global_seed, stats.num_primary, num_keys,
              100.0 * stats.num_primary / num_keys);
    } else {
      fprintf(stderr, "  eps=%.2f: FAILED after %u attempts\n", eps,
              stats.attempts);
    }
  }
}

// --------------------------------------------------------------------------
// Test: maximize primary placement build vs non-maximized secondary placement
// build

HWY_NOINLINE void TestAllOptimizedBuild() {
  fprintf(stderr, "=== TestOptimizedBuild ===\n");
  const uint32_t num_keys = AdjustedReps(10'000);
  auto keys = GenerateKeys(num_keys);

  for (double eps : {0.05, 0.10, 0.25, 0.50}) {
    CuckooBuildStats stats_basic, stats_opt;

    CuckooTable table_basic =
        CuckooBuild(keys.data(), num_keys, eps, /*max_attempts=*/200,
                    /*optimize_primary=*/false, &stats_basic);
    CuckooTable table_opt =
        CuckooBuild(keys.data(), num_keys, eps, /*max_attempts=*/200,
                    /*optimize_primary=*/true, &stats_opt);

    if (stats_basic.success && stats_opt.success) {
      // Optimized should have at least as many keys in primary.
      HWY_ASSERT(stats_opt.num_primary >= stats_basic.num_primary);

      // Verify query correctness of optimized table.
      for (uint32_t i = 0; i < num_keys; ++i) {
        HWY_ASSERT_M(table_opt.QueryOne(keys[i]), "Optimized table lost a key");
      }

      fprintf(stderr,
              "  eps=%.2f: basic=%u/%u (%.1f%%), optimized=%u/%u (%.1f%%)\n",
              eps, stats_basic.num_primary, num_keys,
              100.0 * stats_basic.num_primary / num_keys, stats_opt.num_primary,
              num_keys, 100.0 * stats_opt.num_primary / num_keys);
    }
  }
}

// copybara:strip_begin
// --------------------------------------------------------------------------
// Test: comparison against SimpleMinCostFlow

HWY_NOINLINE void TestAllMinCostFlowComparison() {
  fprintf(stderr, "=== TestMinCostFlowComparison ===\n");
  const uint32_t key_counts[] = {10'000, 100'000, 500'000};
  const double epsilons[] = {0.01};
  const bool verify_min_cost_flow = true;

  for (uint32_t num_keys : key_counts) {
    auto keys = GenerateKeys(num_keys);
    for (double eps : epsilons) {
      CuckooBuildStats stats;
      stats.collect_path_cost_stats = true;
      auto t_cuckoo_start = platform::Now();
      CuckooTable table =
          CuckooBuild(keys.data(), num_keys, eps, /*max_attempts=*/200,
                      /*optimize_primary=*/true, &stats);
      (void)table;
      auto t_cuckoo_end = platform::Now();
      double cuckoo_ms = (t_cuckoo_end - t_cuckoo_start) * 1000;
      fprintf(stderr, "  CuckooBuild full runtime: %.2f ms\n", cuckoo_ms);

      if (!stats.success) {
        fprintf(stderr, "  keys=%u, eps=%.2f: matching failed\n", num_keys,
                eps);
        continue;
      }

      uint32_t optimal_primary = stats.num_primary;
      if (verify_min_cost_flow) {
        auto t0 = std::chrono::high_resolution_clock::now();
        CuckooConfig config(num_keys, eps);
        const uint32_t num_buckets = config.NumBuckets();

        AesCtrEngine engine(/*deterministic=*/true);
        Triple32 h1(engine, stats.global_seed * 2);
        Triple32 h2(engine, stats.global_seed * 2 + 1);

        ::operations_research::SimpleMinCostFlow min_cost_flow;

        const int32_t source = 0;
        const int32_t sink = 1;
        const int32_t key_base = 2;
        const int32_t bucket_base = 2 + static_cast<int32_t>(num_keys);

        min_cost_flow.SetNodeSupply(source, num_keys);
        min_cost_flow.SetNodeSupply(sink, -static_cast<int64_t>(num_keys));

        for (uint32_t i = 0; i < num_keys; ++i) {
          const int32_t key_node = key_base + static_cast<int32_t>(i);
          min_cost_flow.AddArcWithCapacityAndUnitCost(source, key_node, 1, 0);

          const uint32_t b1 = h1(keys[i]) % num_buckets;
          const uint32_t b2 = h2(keys[i]) % num_buckets;

          min_cost_flow.AddArcWithCapacityAndUnitCost(
              key_node, bucket_base + static_cast<int32_t>(b1), 1, 0);
          min_cost_flow.AddArcWithCapacityAndUnitCost(
              key_node, bucket_base + static_cast<int32_t>(b2), 1, 1);
        }

        for (uint32_t b = 0; b < num_buckets; ++b) {
          min_cost_flow.AddArcWithCapacityAndUnitCost(
              bucket_base + static_cast<int32_t>(b), sink,
              CuckooConfig::kBucketSize, 0);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        HWY_ASSERT(min_cost_flow.Solve() ==
                   ::operations_research::SimpleMinCostFlow::OPTIMAL);
        auto t2 = std::chrono::high_resolution_clock::now();

        double setup_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        double solve_ms =
            std::chrono::duration<double, std::milli>(t2 - t1).count();
        fprintf(stderr,
                "  SimpleMinCostFlow setup: %.2f ms, Solve: %.2f ms (Total: "
                "%.2f ms)\n",
                setup_ms, solve_ms, setup_ms + solve_ms);

        const uint32_t optimal_secondary =
            static_cast<uint32_t>(min_cost_flow.OptimalCost());
        optimal_primary = num_keys - optimal_secondary;
      }

      fprintf(stderr,
              "  keys=%u, eps=%.2f: CuckooBuilder primary=%u, "
              "OR-Tools optimal=%u, greedy unassigned=%u\n",
              num_keys, eps, stats.num_primary, optimal_primary,
              stats.num_unmatched_after_greedy);

      fprintf(stderr, "    Paths per path cost: ");
      for (size_t g = 0; g < stats.paths_per_path_cost.size(); ++g) {
        if (stats.paths_per_path_cost[g] > 0) {
          fprintf(stderr, "[g=%zu: %u] ", g, stats.paths_per_path_cost[g]);
        }
      }
      fprintf(stderr, "\n");

      if (verify_min_cost_flow) {
        HWY_ASSERT_M(stats.num_primary == optimal_primary,
                     "CuckooBuilder did not reach min-cost optimal solution");
      }
    }
  }
}
// copybara:strip_end

#endif  // HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(CuckooTest);
HWY_EXPORT_AND_TEST_BEST_P(CuckooTest, TestAllBuildSmall);
HWY_EXPORT_AND_TEST_BEST_P(CuckooTest, TestAllBuildMedium);
HWY_EXPORT_AND_TEST_BEST_P(CuckooTest, TestAllQueryCorrectness);
HWY_EXPORT_AND_TEST_BEST_P(CuckooTest, TestAllBatchQuery);
HWY_EXPORT_AND_TEST_BEST_P(CuckooTest, TestAllEpsilonSweep);
HWY_EXPORT_AND_TEST_BEST_P(CuckooTest, TestAllOptimizedBuild);
HWY_EXPORT_AND_TEST_BEST_P(CuckooTest, TestAllMinCostFlowComparison);
HWY_AFTER_TEST();
}  // namespace hwy
#endif  // HWY_ONCE
