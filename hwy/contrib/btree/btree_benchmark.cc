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

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/btree/btree_benchmark.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/btree/btree-inl.h"
#include "hwy/nanobenchmark.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

#define HWY_HAVE_TCMALLOC 0
#if HWY_HAVE_TCMALLOC
#include // Placeholder for tcmalloc, do not remove
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <map>
#include <set>
#include <utility>
#include <vector>

#include "third_party/absl/container/btree_map.h"
#include "third_party/absl/container/btree_set.h"
#include "third_party/absl/random/random.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

static size_t AllocatedBefore() {
#if HWY_HAVE_TCMALLOC
  return tcmalloc::MallocExtension::GetNumericProperty(
             "generic.current_allocated_bytes")
      .value_or(0);
#else
  return 0;
#endif
}

static size_t GetAllocatedBytes(size_t before, size_t guessed) {
#if HWY_HAVE_TCMALLOC
  const size_t after = tcmalloc::MallocExtension::GetNumericProperty(
                           "generic.current_allocated_bytes")
                           .value_or(0);
  const size_t allocated = (after > before) ? (after - before) : 0;
  return (allocated != 0) ? allocated : guessed;
#else
  return guessed;
#endif
}

template <typename KeyT>
void RunBenchmarkSuite(size_t num_keys) {
  printf("\n===============================================================\n");
  printf("  B-Tree Benchmark Suite (N = %zu keys, %s)\n", num_keys,
         hwy::TargetName(HWY_TARGET));
  printf("===============================================================\n");

  absl::BitGen bitgen;
  std::vector<KeyT> keys;
  keys.reserve(num_keys);
  for (size_t i = 0; i < num_keys; ++i) {
    keys.push_back(static_cast<KeyT>((i + 1) * 10));
  }

  // 1. Build Containers (measured via hwy::platform::Now() & TCMalloc)
  printf("Building containers...\n");

  const size_t std_before = AllocatedBefore();
  const double start_std = hwy::platform::Now();
  std::set<KeyT> std_tree(keys.begin(), keys.end());
  const double end_std = hwy::platform::Now();
  const size_t std_bytes = GetAllocatedBytes(std_before, 0);

  const size_t absl_before = AllocatedBefore();
  const double start_absl = hwy::platform::Now();
  absl::btree_set<KeyT> absl_tree(keys.begin(), keys.end());
  const double end_absl = hwy::platform::Now();
  const size_t absl_bytes = GetAllocatedBytes(absl_before, 0);

  const size_t hwy_before = AllocatedBefore();
  const double start_hwy = hwy::platform::Now();
  auto hwy_tree = BTreeSet<KeyT>::Build(keys.data(), keys.size());
  const double end_hwy = hwy::platform::Now();
  const size_t hwy_bytes =
      GetAllocatedBytes(hwy_before, hwy_tree.AllocatedBytes());

  const double hwy_build_ms = (end_hwy - start_hwy) * 1000.0;
  const double absl_build_ms = (end_absl - start_absl) * 1000.0;
  const double std_build_ms = (end_std - start_std) * 1000.0;

  printf("Build Time:\n");
  printf("  std::set        : %8.2f ms\n", std_build_ms);
  printf("  absl::btree_set : %8.2f ms\n", absl_build_ms);
  printf("  hwy::BTreeSet   : %8.2f ms (%.1fx faster than absl)\n",
         hwy_build_ms, absl_build_ms / (hwy_build_ms + 1e-6));

  // Memory Footprint (Measured directly via TCMalloc Heap Interception)
  printf("\nMemory Footprint (TCMalloc Measured):\n");
  printf("  std::set        : %6.2f MB (%5.1f B/key)\n",
         std_bytes / (1024.0 * 1024.0),
         static_cast<double>(std_bytes) / num_keys);
  printf("  absl::btree_set : %6.2f MB (%5.1f B/key)\n",
         absl_bytes / (1024.0 * 1024.0),
         static_cast<double>(absl_bytes) / num_keys);
  printf("  hwy::BTreeSet   : %6.2f MB (%5.1f B/key)\n",
         hwy_bytes / (1024.0 * 1024.0),
         static_cast<double>(hwy_bytes) / num_keys);

  // 2. Generate Random Query Keys
  constexpr size_t kNumQueries = 1000000;
  std::vector<KeyT> queries;
  queries.reserve(kNumQueries);
  for (size_t i = 0; i < kNumQueries; ++i) {
    queries.push_back(
        static_cast<KeyT>(absl::Uniform<KeyT>(bitgen, 0, (num_keys + 1) * 10)));
  }

  // 3. Point Lookups (Contains / Find) via hwy::platform::Now()
  uint64_t hwy_hits = 0, absl_hits = 0, std_hits = 0;

  const double t0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumQueries; ++i) {
    std_hits += (std_tree.find(queries[i]) != std_tree.end());
  }
  hwy::PreventElision(std_hits);
  const double t1 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    absl_hits += (absl_tree.find(queries[i]) != absl_tree.end());
  }
  hwy::PreventElision(absl_hits);
  const double t2 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    hwy_hits += hwy_tree.Contains(queries[i]);
  }
  hwy::PreventElision(hwy_hits);
  const double t3 = hwy::platform::Now();

  const double std_lookup_ns = (t1 - t0) * 1e9 / kNumQueries;
  const double absl_lookup_ns = (t2 - t1) * 1e9 / kNumQueries;
  const double hwy_lookup_ns = (t3 - t2) * 1e9 / kNumQueries;

  printf("\nPoint Lookup Latency (1M queries):\n");
  printf("  std::set        : %6.2f ns/op (%6.2f Mops/s)\n", std_lookup_ns,
         1000.0 / std_lookup_ns);
  printf("  absl::btree_set : %6.2f ns/op (%6.2f Mops/s)\n", absl_lookup_ns,
         1000.0 / absl_lookup_ns);
  printf("  hwy::BTreeSet   : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
         hwy_lookup_ns, 1000.0 / hwy_lookup_ns, absl_lookup_ns / hwy_lookup_ns);

  // 4. Ordered Range Queries (LowerBound)
  uint64_t hwy_lb_sum = 0, absl_lb_sum = 0, std_lb_sum = 0;

  const double r0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = std_tree.lower_bound(queries[i]);
    if (it != std_tree.end()) std_lb_sum += *it;
  }
  hwy::PreventElision(std_lb_sum);
  const double r1 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = absl_tree.lower_bound(queries[i]);
    if (it != absl_tree.end()) absl_lb_sum += *it;
  }
  hwy::PreventElision(absl_lb_sum);
  const double r2 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    const KeyT* ptr = hwy_tree.LowerBound(queries[i]);
    if (ptr != nullptr) hwy_lb_sum += *ptr;
  }
  hwy::PreventElision(hwy_lb_sum);
  const double r3 = hwy::platform::Now();

  const double std_lb_ns = (r1 - r0) * 1e9 / kNumQueries;
  const double absl_lb_ns = (r2 - r1) * 1e9 / kNumQueries;
  const double hwy_lb_ns = (r3 - r2) * 1e9 / kNumQueries;

  printf("\nLowerBound Range Query Latency (1M queries):\n");
  printf("  std::set        : %6.2f ns/op (%6.2f Mops/s)\n", std_lb_ns,
         1000.0 / std_lb_ns);
  printf("  absl::btree_set : %6.2f ns/op (%6.2f Mops/s)\n", absl_lb_ns,
         1000.0 / absl_lb_ns);
  printf("  hwy::BTreeSet   : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
         hwy_lb_ns, 1000.0 / hwy_lb_ns, absl_lb_ns / hwy_lb_ns);

  HWY_ASSERT(hwy_hits == absl_hits);
  HWY_ASSERT(hwy_lb_sum == absl_lb_sum);
}

template <typename KeyT, typename ValueT>
void RunMapBenchmarkSuite(size_t num_keys) {
  printf("\n===============================================================\n");
  printf("  B-Tree Map Benchmark Suite (N = %zu keys, %s)\n", num_keys,
         hwy::TargetName(HWY_TARGET));
  printf("===============================================================\n");

  absl::BitGen bitgen;
  std::vector<KeyT> keys;
  std::vector<ValueT> vals;
  std::vector<std::pair<KeyT, ValueT>> kv_pairs;
  keys.reserve(num_keys);
  vals.reserve(num_keys);
  kv_pairs.reserve(num_keys);
  for (size_t i = 0; i < num_keys; ++i) {
    KeyT k = static_cast<KeyT>((i + 1) * 10);
    ValueT v = static_cast<ValueT>((i + 1) * 100);
    keys.push_back(k);
    vals.push_back(v);
    kv_pairs.push_back({k, v});
  }

  // 1. Build Containers
  printf("Building map containers...\n");

  const size_t std_before = AllocatedBefore();
  const double start_std = hwy::platform::Now();
  std::map<KeyT, ValueT> std_map(kv_pairs.begin(), kv_pairs.end());
  const double end_std = hwy::platform::Now();
  const size_t std_bytes = GetAllocatedBytes(std_before, 0);

  const size_t absl_before = AllocatedBefore();
  const double start_absl = hwy::platform::Now();
  absl::btree_map<KeyT, ValueT> absl_map(kv_pairs.begin(), kv_pairs.end());
  const double end_absl = hwy::platform::Now();
  const size_t absl_bytes = GetAllocatedBytes(absl_before, 0);

  const size_t hwy_before = AllocatedBefore();
  const double start_hwy = hwy::platform::Now();
  auto hwy_map =
      BTreeMap<KeyT, ValueT>::Build(keys.data(), vals.data(), keys.size());
  const double end_hwy = hwy::platform::Now();
  const size_t hwy_bytes =
      GetAllocatedBytes(hwy_before, hwy_map.AllocatedBytes());

  const double hwy_build_ms = (end_hwy - start_hwy) * 1000.0;
  const double absl_build_ms = (end_absl - start_absl) * 1000.0;
  const double std_build_ms = (end_std - start_std) * 1000.0;

  printf("Build Time:\n");
  printf("  std::map        : %8.2f ms\n", std_build_ms);
  printf("  absl::btree_map : %8.2f ms\n", absl_build_ms);
  printf("  hwy::BTreeMap   : %8.2f ms (%.1fx faster than absl)\n",
         hwy_build_ms, absl_build_ms / (hwy_build_ms + 1e-6));

  // Memory Footprint
  printf("\nMemory Footprint (TCMalloc Measured):\n");
  printf("  std::map        : %6.2f MB (%5.1f B/pair)\n",
         std_bytes / (1024.0 * 1024.0),
         static_cast<double>(std_bytes) / num_keys);
  printf("  absl::btree_map : %6.2f MB (%5.1f B/pair)\n",
         absl_bytes / (1024.0 * 1024.0),
         static_cast<double>(absl_bytes) / num_keys);
  printf("  hwy::BTreeMap   : %6.2f MB (%5.1f B/pair)\n",
         hwy_bytes / (1024.0 * 1024.0),
         static_cast<double>(hwy_bytes) / num_keys);

  // 2. Generate Random Query Keys
  constexpr size_t kNumQueries = 1000000;
  std::vector<KeyT> queries;
  queries.reserve(kNumQueries);
  for (size_t i = 0; i < kNumQueries; ++i) {
    queries.push_back(static_cast<KeyT>(
        absl::Uniform<uint64_t>(bitgen, 0, (num_keys + 1) * 10)));
  }

  // 3. Point Lookups (FindValue / Find)
  uint64_t hwy_hits = 0, absl_hits = 0, std_hits = 0;

  const double t0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = std_map.find(queries[i]);
    if (it != std_map.end()) std_hits += static_cast<uint64_t>(it->second);
  }
  hwy::PreventElision(std_hits);
  const double t1 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = absl_map.find(queries[i]);
    if (it != absl_map.end()) absl_hits += static_cast<uint64_t>(it->second);
  }
  hwy::PreventElision(absl_hits);
  const double t2 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    const ValueT* ptr = hwy_map.FindValue(queries[i]);
    if (ptr != nullptr) hwy_hits += static_cast<uint64_t>(*ptr);
  }
  hwy::PreventElision(hwy_hits);
  const double t3 = hwy::platform::Now();

  const double std_lookup_ns = (t1 - t0) * 1e9 / kNumQueries;
  const double absl_lookup_ns = (t2 - t1) * 1e9 / kNumQueries;
  const double hwy_lookup_ns = (t3 - t2) * 1e9 / kNumQueries;

  printf("\nPoint Lookup Latency (1M queries):\n");
  printf("  std::map        : %6.2f ns/op (%6.2f Mops/s)\n", std_lookup_ns,
         1000.0 / std_lookup_ns);
  printf("  absl::btree_map : %6.2f ns/op (%6.2f Mops/s)\n", absl_lookup_ns,
         1000.0 / absl_lookup_ns);
  printf("  hwy::BTreeMap   : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
         hwy_lookup_ns, 1000.0 / hwy_lookup_ns, absl_lookup_ns / hwy_lookup_ns);

  // 4. Ordered Range Queries (LowerBound)
  uint64_t hwy_lb_sum = 0, absl_lb_sum = 0, std_lb_sum = 0;

  const double r0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = std_map.lower_bound(queries[i]);
    if (it != std_map.end()) std_lb_sum += static_cast<uint64_t>(it->second);
  }
  hwy::PreventElision(std_lb_sum);
  const double r1 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = absl_map.lower_bound(queries[i]);
    if (it != absl_map.end()) absl_lb_sum += static_cast<uint64_t>(it->second);
  }
  hwy::PreventElision(absl_lb_sum);
  const double r2 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = hwy_map.lower_bound(queries[i]);
    if (it != hwy_map.end()) hwy_lb_sum += static_cast<uint64_t>(it->second);
  }
  hwy::PreventElision(hwy_lb_sum);
  const double r3 = hwy::platform::Now();

  const double std_lb_ns = (r1 - r0) * 1e9 / kNumQueries;
  const double absl_lb_ns = (r2 - r1) * 1e9 / kNumQueries;
  const double hwy_lb_ns = (r3 - r2) * 1e9 / kNumQueries;

  printf("\nLowerBound Range Query Latency (1M queries):\n");
  printf("  std::map        : %6.2f ns/op (%6.2f Mops/s)\n", std_lb_ns,
         1000.0 / std_lb_ns);
  printf("  absl::btree_map : %6.2f ns/op (%6.2f Mops/s)\n", absl_lb_ns,
         1000.0 / absl_lb_ns);
  printf("  hwy::BTreeMap   : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
         hwy_lb_ns, 1000.0 / hwy_lb_ns, absl_lb_ns / hwy_lb_ns);

  HWY_ASSERT(hwy_hits == absl_hits);
  HWY_ASSERT(hwy_lb_sum == absl_lb_sum);
}

HWY_NOINLINE void BenchmarkAll() {
  printf("\n###############################################################\n");
  printf("  32-bit Key Set Benchmarks (BTreeSet<uint32_t>)\n");
  printf("###############################################################\n");
  RunBenchmarkSuite<uint32_t>(10000);    // 10K keys (L1/L2 Cache)
  RunBenchmarkSuite<uint32_t>(100000);   // 100K keys (L3 Cache)
  RunBenchmarkSuite<uint32_t>(1000000);  // 1M keys (RAM)

  printf("\n###############################################################\n");
  printf("  64-bit Key Set Benchmarks (BTreeSet<uint64_t>)\n");
  printf("###############################################################\n");
  RunBenchmarkSuite<uint64_t>(10000);    // 10K keys (L1/L2 Cache)
  RunBenchmarkSuite<uint64_t>(100000);   // 100K keys (L3 Cache)
  RunBenchmarkSuite<uint64_t>(1000000);  // 1M keys (RAM)

  printf("\n###############################################################\n");
  printf("  32-bit Key Map Benchmarks (BTreeMap<uint32_t, uint64_t>)\n");
  printf("###############################################################\n");
  RunMapBenchmarkSuite<uint32_t, uint64_t>(10000);    // 10K keys (L1/L2 Cache)
  RunMapBenchmarkSuite<uint32_t, uint64_t>(100000);   // 100K keys (L3 Cache)
  RunMapBenchmarkSuite<uint32_t, uint64_t>(1000000);  // 1M keys (RAM)

  printf("\n###############################################################\n");
  printf("  64-bit Key Map Benchmarks (BTreeMap<uint64_t, double>)\n");
  printf("###############################################################\n");
  RunMapBenchmarkSuite<uint64_t, double>(10000);    // 10K keys (L1/L2 Cache)
  RunMapBenchmarkSuite<uint64_t, double>(100000);   // 100K keys (L3 Cache)
  RunMapBenchmarkSuite<uint64_t, double>(1000000);  // 1M keys (RAM)
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
namespace {
HWY_EXPORT(BenchmarkAll);
}  // namespace
}  // namespace hwy

int main(int argc, char** argv) {
  using namespace hwy;  // NOLINT
  HWY_DYNAMIC_DISPATCH(BenchmarkAll)();
  return 0;
}

#endif  // HWY_ONCE
