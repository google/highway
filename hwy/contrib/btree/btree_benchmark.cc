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
#include "hwy/contrib/btree/btree-inl.h"
#include "hwy/contrib/btree/compact_btree-inl.h"
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

#define HWY_HAVE_ABSL 0
#if HWY_HAVE_ABSL
#include "third_party/absl/container/btree_map.h"
#include "third_party/absl/container/btree_set.h"
#include "third_party/absl/random/random.h"
#endif

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

  const size_t compact_before = AllocatedBefore();
  const double start_compact = hwy::platform::Now();
  auto compact_tree = CompactBTreeSet<KeyT>::Build(keys.data(), keys.size());
  const double end_compact = hwy::platform::Now();
  const size_t compact_bytes =
      GetAllocatedBytes(compact_before, compact_tree.AllocatedBytes());

  const double hwy_build_ms = (end_hwy - start_hwy) * 1000.0;
  const double compact_build_ms = (end_compact - start_compact) * 1000.0;
  const double absl_build_ms = (end_absl - start_absl) * 1000.0;
  const double std_build_ms = (end_std - start_std) * 1000.0;

  printf("Build Time:\n");
  printf("  std::set             : %8.2f ms\n", std_build_ms);
  printf("  absl::btree_set      : %8.2f ms\n", absl_build_ms);
  printf("  hwy::BTreeSet        : %8.2f ms (%.1fx faster than absl)\n",
         hwy_build_ms, absl_build_ms / (hwy_build_ms + 1e-6));
  printf("  hwy::CompactBTreeSet : %8.2f ms (%.1fx faster than absl)\n",
         compact_build_ms, absl_build_ms / (compact_build_ms + 1e-6));

  // Memory Footprint (Measured directly via TCMalloc Heap Interception)
  printf(
      "\nMemory Footprint (100%% Fill Bulk-Loaded State, TCMalloc "
      "Measured):\n");
  printf("  std::set             : %6.2f MB (%5.1f B/key)\n",
         std_bytes / (1024.0 * 1024.0),
         static_cast<double>(std_bytes) / num_keys);
  printf("  absl::btree_set      : %6.2f MB (%5.1f B/key)\n",
         absl_bytes / (1024.0 * 1024.0),
         static_cast<double>(absl_bytes) / num_keys);
  printf("  hwy::BTreeSet        : %6.2f MB (%5.1f B/key)\n",
         hwy_bytes / (1024.0 * 1024.0),
         static_cast<double>(hwy_bytes) / num_keys);
  printf(
      "  hwy::CompactBTreeSet : %6.2f MB (%5.1f B/key) -> %.1f%% smaller than "
      "absl!\n",
      compact_bytes / (1024.0 * 1024.0),
      static_cast<double>(compact_bytes) / num_keys,
      100.0 * (1.0 - static_cast<double>(compact_bytes) / absl_bytes));

  // 2. Generate Random Query Keys
  constexpr size_t kNumQueries = 1000000;
  std::vector<KeyT> queries;
  queries.reserve(kNumQueries);
  for (size_t i = 0; i < kNumQueries; ++i) {
    queries.push_back(
        static_cast<KeyT>(absl::Uniform<KeyT>(bitgen, 0, (num_keys + 1) * 10)));
  }

  // 3. Point Lookups (Contains / Find) via hwy::platform::Now()
  uint64_t hwy_hits = 0, compact_hits = 0, absl_hits = 0, std_hits = 0;

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

  for (size_t i = 0; i < kNumQueries; ++i) {
    compact_hits += compact_tree.contains(queries[i]);
  }
  hwy::PreventElision(compact_hits);
  const double t4 = hwy::platform::Now();

  const double std_lookup_ns = (t1 - t0) * 1e9 / kNumQueries;
  const double absl_lookup_ns = (t2 - t1) * 1e9 / kNumQueries;
  const double hwy_lookup_ns = (t3 - t2) * 1e9 / kNumQueries;
  const double compact_lookup_ns = (t4 - t3) * 1e9 / kNumQueries;

  printf("\nPoint Lookup Latency (1M queries on 100%% Bulk-Loaded Tree):\n");
  printf("  std::set             : %6.2f ns/op (%6.2f Mops/s)\n", std_lookup_ns,
         1000.0 / std_lookup_ns);
  printf("  absl::btree_set      : %6.2f ns/op (%6.2f Mops/s)\n",
         absl_lookup_ns, 1000.0 / absl_lookup_ns);
  printf(
      "  hwy::BTreeSet        : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
      hwy_lookup_ns, 1000.0 / hwy_lookup_ns, absl_lookup_ns / hwy_lookup_ns);
  printf(
      "  hwy::CompactBTreeSet : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
      compact_lookup_ns, 1000.0 / compact_lookup_ns,
      absl_lookup_ns / compact_lookup_ns);

  // 4. Ordered Range Queries (LowerBound)
  uint64_t hwy_lb_sum = 0, compact_lb_sum = 0, absl_lb_sum = 0, std_lb_sum = 0;

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

  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = compact_tree.lower_bound(queries[i]);
    if (it != compact_tree.end()) compact_lb_sum += *it;
  }
  hwy::PreventElision(compact_lb_sum);
  const double r4 = hwy::platform::Now();

  const double std_lb_ns = (r1 - r0) * 1e9 / kNumQueries;
  const double absl_lb_ns = (r2 - r1) * 1e9 / kNumQueries;
  const double hwy_lb_ns = (r3 - r2) * 1e9 / kNumQueries;
  const double compact_lb_ns = (r4 - r3) * 1e9 / kNumQueries;

  printf(
      "\nLowerBound Range Query Latency (1M queries on 100%% Bulk-Loaded "
      "Tree):\n");
  printf("  std::set             : %6.2f ns/op (%6.2f Mops/s)\n", std_lb_ns,
         1000.0 / std_lb_ns);
  printf("  absl::btree_set      : %6.2f ns/op (%6.2f Mops/s)\n", absl_lb_ns,
         1000.0 / absl_lb_ns);
  printf(
      "  hwy::BTreeSet        : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
      hwy_lb_ns, 1000.0 / hwy_lb_ns, absl_lb_ns / hwy_lb_ns);
  printf(
      "  hwy::CompactBTreeSet : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
      compact_lb_ns, 1000.0 / compact_lb_ns, absl_lb_ns / compact_lb_ns);

  // 4b. Ordered Range Queries (UpperBound)
  uint64_t hwy_ub_sum = 0, compact_ub_sum = 0, absl_ub_sum = 0, std_ub_sum = 0;

  const double u0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = std_tree.upper_bound(queries[i]);
    if (it != std_tree.end()) std_ub_sum += *it;
  }
  hwy::PreventElision(std_ub_sum);
  const double u1 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = absl_tree.upper_bound(queries[i]);
    if (it != absl_tree.end()) absl_ub_sum += *it;
  }
  hwy::PreventElision(absl_ub_sum);
  const double u2 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = hwy_tree.upper_bound(queries[i]);
    if (it != hwy_tree.end()) hwy_ub_sum += *it;
  }
  hwy::PreventElision(hwy_ub_sum);
  const double u3 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = compact_tree.upper_bound(queries[i]);
    if (it != compact_tree.end()) compact_ub_sum += *it;
  }
  hwy::PreventElision(compact_ub_sum);
  const double u4 = hwy::platform::Now();

  const double std_ub_ns = (u1 - u0) * 1e9 / kNumQueries;
  const double absl_ub_ns = (u2 - u1) * 1e9 / kNumQueries;
  const double hwy_ub_ns = (u3 - u2) * 1e9 / kNumQueries;
  const double compact_ub_ns = (u4 - u3) * 1e9 / kNumQueries;

  printf(
      "\nUpperBound Range Query Latency (1M queries on 100%% Bulk-Loaded "
      "Tree):\n");
  printf("  std::set             : %6.2f ns/op (%6.2f Mops/s)\n", std_ub_ns,
         1000.0 / std_ub_ns);
  printf("  absl::btree_set      : %6.2f ns/op (%6.2f Mops/s)\n", absl_ub_ns,
         1000.0 / absl_ub_ns);
  printf(
      "  hwy::BTreeSet        : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
      hwy_ub_ns, 1000.0 / hwy_ub_ns, absl_ub_ns / hwy_ub_ns);
  printf(
      "  hwy::CompactBTreeSet : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
      compact_ub_ns, 1000.0 / compact_ub_ns, absl_ub_ns / compact_ub_ns);

  // 5. Batch Point Lookups (ContainsBatch - 8-way pipelined prefetch)
  auto batch_found = std::make_unique<bool[]>(kNumQueries);
  const double b0 = hwy::platform::Now();
  hwy_tree.ContainsBatch(queries.data(), kNumQueries, batch_found.get());
  const double b1 = hwy::platform::Now();

  uint64_t batch_hits = 0;
  for (size_t i = 0; i < kNumQueries; ++i) {
    batch_hits += batch_found[i];
  }
  hwy::PreventElision(batch_hits);

  auto compact_batch_found = std::make_unique<bool[]>(kNumQueries);
  const double cb0 = hwy::platform::Now();
  compact_tree.ContainsBatch(queries.data(), kNumQueries,
                             compact_batch_found.get());
  const double cb1 = hwy::platform::Now();

  uint64_t compact_batch_hits = 0;
  for (size_t i = 0; i < kNumQueries; ++i) {
    compact_batch_hits += compact_batch_found[i];
  }
  hwy::PreventElision(compact_batch_hits);

  const double hwy_batch_lookup_ns = (b1 - b0) * 1e9 / kNumQueries;
  const double compact_batch_lookup_ns = (cb1 - cb0) * 1e9 / kNumQueries;

  printf(
      "\nBatch Point Lookup (1M queries on 100%% Bulk-Loaded Tree, 8-way "
      "pipelined prefetch):\n");
  printf("  hwy::BTreeSet (Serial)        : %6.2f ns/op (%6.2f Mops/s)\n",
         hwy_lookup_ns, 1000.0 / hwy_lookup_ns);
  printf(
      "  hwy::BTreeSet (Batch)         : %6.2f ns/op (%6.2f Mops/s) -> "
      "%.2fx vs Serial (%.2fx vs absl)\n",
      hwy_batch_lookup_ns, 1000.0 / hwy_batch_lookup_ns,
      hwy_lookup_ns / hwy_batch_lookup_ns,
      absl_lookup_ns / hwy_batch_lookup_ns);
  printf("  hwy::CompactBTreeSet (Serial) : %6.2f ns/op (%6.2f Mops/s)\n",
         compact_lookup_ns, 1000.0 / compact_lookup_ns);
  printf(
      "  hwy::CompactBTreeSet (Batch)  : %6.2f ns/op (%6.2f Mops/s) -> "
      "%.2fx vs Serial (%.2fx vs absl)\n",
      compact_batch_lookup_ns, 1000.0 / compact_batch_lookup_ns,
      compact_lookup_ns / compact_batch_lookup_ns,
      absl_lookup_ns / compact_batch_lookup_ns);

  // 6. Batch LowerBound Queries (LowerBoundBatch - 8-way pipelined prefetch)
  auto batch_lb_ptrs = std::make_unique<const KeyT*[]>(kNumQueries);
  const double blb0 = hwy::platform::Now();
  hwy_tree.LowerBoundBatch(queries.data(), kNumQueries, batch_lb_ptrs.get());
  const double blb1 = hwy::platform::Now();

  uint64_t batch_lb_sum = 0;
  for (size_t i = 0; i < kNumQueries; ++i) {
    if (batch_lb_ptrs[i] != nullptr) batch_lb_sum += *batch_lb_ptrs[i];
  }
  hwy::PreventElision(batch_lb_sum);

  std::vector<typename CompactBTreeSet<KeyT>::const_iterator> compact_batch_lb(
      kNumQueries);
  const double cblb0 = hwy::platform::Now();
  compact_tree.LowerBoundBatch(queries.data(), kNumQueries,
                               compact_batch_lb.data());
  const double cblb1 = hwy::platform::Now();

  uint64_t compact_batch_lb_sum = 0;
  for (size_t i = 0; i < kNumQueries; ++i) {
    if (compact_batch_lb[i] != compact_tree.end()) {
      compact_batch_lb_sum += *compact_batch_lb[i];
    }
  }
  hwy::PreventElision(compact_batch_lb_sum);

  const double hwy_batch_lb_ns = (blb1 - blb0) * 1e9 / kNumQueries;
  const double compact_batch_lb_ns = (cblb1 - cblb0) * 1e9 / kNumQueries;
  printf(
      "\nBatch LowerBound Query (1M queries on 100%% Bulk-Loaded Tree, 8-way "
      "pipelined prefetch):\n");
  printf("  hwy::BTreeSet (Serial)        : %6.2f ns/op (%6.2f Mops/s)\n",
         hwy_lb_ns, 1000.0 / hwy_lb_ns);
  printf(
      "  hwy::BTreeSet (Batch)         : %6.2f ns/op (%6.2f Mops/s) -> %.2fx "
      "vs "
      "Serial (%.2fx vs absl)\n",
      hwy_batch_lb_ns, 1000.0 / hwy_batch_lb_ns, hwy_lb_ns / hwy_batch_lb_ns,
      absl_lb_ns / hwy_batch_lb_ns);
  printf("  hwy::CompactBTreeSet (Serial) : %6.2f ns/op (%6.2f Mops/s)\n",
         compact_lb_ns, 1000.0 / compact_lb_ns);
  printf(
      "  hwy::CompactBTreeSet (Batch)  : %6.2f ns/op (%6.2f Mops/s) -> %.2fx "
      "vs "
      "Serial (%.2fx vs absl)\n",
      compact_batch_lb_ns, 1000.0 / compact_batch_lb_ns,
      compact_lb_ns / compact_batch_lb_ns, absl_lb_ns / compact_batch_lb_ns);

  // 7. Dynamic Random Insertions & 8. Dynamic Deletions on Empty Tree
  const size_t kNumMutations = std::min(num_keys, static_cast<size_t>(100000));
  const size_t kNumErases = kNumMutations / 2;
  std::vector<KeyT> mutation_keys;
  mutation_keys.reserve(kNumMutations);
  for (size_t i = 0; i < kNumMutations; ++i) {
    mutation_keys.push_back(static_cast<KeyT>(
        absl::Uniform<uint64_t>(bitgen, 0, (num_keys + 1) * 20)));
  }

  // --- std::set ---
  const size_t std_dyn_before = AllocatedBefore();
  std::set<KeyT> std_dyn_set;
  const double mi_std_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumMutations; ++i) {
    std_dyn_set.insert(mutation_keys[i]);
  }
  const double mi_std_1 = hwy::platform::Now();
  const size_t std_dyn_bytes = GetAllocatedBytes(std_dyn_before, 0);

  const double me_std_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumErases; ++i) {
    std_dyn_set.erase(mutation_keys[i]);
  }
  const double me_std_1 = hwy::platform::Now();
  const size_t std_del_bytes = GetAllocatedBytes(std_dyn_before, 0);

  // --- absl::btree_set ---
  const size_t absl_dyn_before = AllocatedBefore();
  absl::btree_set<KeyT> absl_dyn_set;
  const double mi_absl_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumMutations; ++i) {
    absl_dyn_set.insert(mutation_keys[i]);
  }
  const double mi_absl_1 = hwy::platform::Now();
  const size_t absl_dyn_bytes = GetAllocatedBytes(absl_dyn_before, 0);

  const double me_absl_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumErases; ++i) {
    absl_dyn_set.erase(mutation_keys[i]);
  }
  const double me_absl_1 = hwy::platform::Now();
  const size_t absl_del_bytes = GetAllocatedBytes(absl_dyn_before, 0);

  // --- hwy::BTreeSet ---
  const size_t hwy_dyn_before = AllocatedBefore();
  BTreeSet<KeyT> hwy_dyn_set;
  const double mi_hwy_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumMutations; ++i) {
    hwy_dyn_set.insert(mutation_keys[i]);
  }
  const double mi_hwy_1 = hwy::platform::Now();
  const size_t hwy_dyn_bytes =
      GetAllocatedBytes(hwy_dyn_before, hwy_dyn_set.AllocatedBytes());

  const double me_hwy_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumErases; ++i) {
    hwy_dyn_set.erase(mutation_keys[i]);
  }
  const double me_hwy_1 = hwy::platform::Now();
  const size_t hwy_del_bytes =
      GetAllocatedBytes(hwy_dyn_before, hwy_dyn_set.AllocatedBytes());

  // --- hwy::CompactBTreeSet ---
  const size_t compact_dyn_before = AllocatedBefore();
  CompactBTreeSet<KeyT> compact_dyn_set;
  const double mi_compact_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumMutations; ++i) {
    compact_dyn_set.insert(mutation_keys[i]);
  }
  const double mi_compact_1 = hwy::platform::Now();
  const size_t compact_dyn_bytes =
      GetAllocatedBytes(compact_dyn_before, compact_dyn_set.AllocatedBytes());

  const double me_compact_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumErases; ++i) {
    compact_dyn_set.erase(mutation_keys[i]);
  }
  const double me_compact_1 = hwy::platform::Now();
  const size_t compact_del_bytes =
      GetAllocatedBytes(compact_dyn_before, compact_dyn_set.AllocatedBytes());

  const double std_ins_ns = (mi_std_1 - mi_std_0) * 1e9 / kNumMutations;
  const double absl_ins_ns = (mi_absl_1 - mi_absl_0) * 1e9 / kNumMutations;
  const double hwy_ins_ns = (mi_hwy_1 - mi_hwy_0) * 1e9 / kNumMutations;
  const double compact_ins_ns =
      (mi_compact_1 - mi_compact_0) * 1e9 / kNumMutations;

  printf(
      "\nDynamic Insertions into Empty Tree (%zu random keys -> Pure Dynamic "
      "Steady State):\n",
      kNumMutations);
  printf("  std::set             : %6.2f ns/op (%5.1f B/key, %5.2f MB)\n",
         std_ins_ns,
         static_cast<double>(std_dyn_bytes) / (std_dyn_set.size() + kNumErases),
         std_dyn_bytes / (1024.0 * 1024.0));
  printf(
      "  absl::btree_set      : %6.2f ns/op (%5.1f B/key, %5.2f MB)\n",
      absl_ins_ns,
      static_cast<double>(absl_dyn_bytes) / (absl_dyn_set.size() + kNumErases),
      absl_dyn_bytes / (1024.0 * 1024.0));
  printf(
      "  hwy::BTreeSet        : %6.2f ns/op (%5.1f B/key, %5.2f MB) -> "
      "%.2fx speedup vs absl\n",
      hwy_ins_ns,
      static_cast<double>(hwy_dyn_bytes) / (hwy_dyn_set.size() + kNumErases),
      hwy_dyn_bytes / (1024.0 * 1024.0), absl_ins_ns / hwy_ins_ns);
  printf(
      "  hwy::CompactBTreeSet : %6.2f ns/op (%5.1f B/key, %5.2f MB) -> "
      "%.2fx vs absl (%.1f%% smaller!)\n",
      compact_ins_ns,
      static_cast<double>(compact_dyn_bytes) /
          (compact_dyn_set.size() + kNumErases),
      compact_dyn_bytes / (1024.0 * 1024.0), absl_ins_ns / compact_ins_ns,
      100.0 * (1.0 - static_cast<double>(compact_dyn_bytes) / absl_dyn_bytes));

  const double std_erase_ns = (me_std_1 - me_std_0) * 1e9 / kNumErases;
  const double absl_erase_ns = (me_absl_1 - me_absl_0) * 1e9 / kNumErases;
  const double hwy_erase_ns = (me_hwy_1 - me_hwy_0) * 1e9 / kNumErases;
  const double compact_erase_ns =
      (me_compact_1 - me_compact_0) * 1e9 / kNumErases;

  printf(
      "\nDynamic Deletions Latency & Memory on Steady-State Tree (%zu random "
      "keys -> remaining keys: %zu after 50%% deletions):\n",
      kNumErases, std_dyn_set.size());
  printf("  std::set             : %6.2f ns/op (%5.1f B/key, %5.2f MB)\n",
         std_erase_ns, static_cast<double>(std_del_bytes) / std_dyn_set.size(),
         std_del_bytes / (1024.0 * 1024.0));
  printf("  absl::btree_set      : %6.2f ns/op (%5.1f B/key, %5.2f MB)\n",
         absl_erase_ns,
         static_cast<double>(absl_del_bytes) / absl_dyn_set.size(),
         absl_del_bytes / (1024.0 * 1024.0));
  printf(
      "  hwy::BTreeSet        : %6.2f ns/op (%5.1f B/key, %5.2f MB) -> "
      "%.2fx speedup vs absl\n",
      hwy_erase_ns, static_cast<double>(hwy_del_bytes) / hwy_dyn_set.size(),
      hwy_del_bytes / (1024.0 * 1024.0), absl_erase_ns / hwy_erase_ns);
  if (compact_del_bytes <= absl_del_bytes) {
    printf(
        "  hwy::CompactBTreeSet : %6.2f ns/op (%5.1f B/key, %5.2f MB) -> "
        "%.2fx vs absl (%.1f%% smaller!)\n",
        compact_erase_ns,
        static_cast<double>(compact_del_bytes) / compact_dyn_set.size(),
        compact_del_bytes / (1024.0 * 1024.0), absl_erase_ns / compact_erase_ns,
        100.0 *
            (1.0 - static_cast<double>(compact_del_bytes) / absl_del_bytes));
  } else {
    printf(
        "  hwy::CompactBTreeSet : %6.2f ns/op (%5.1f B/key, %5.2f MB) -> "
        "%.2fx vs absl (%.1f%% larger)\n",
        compact_erase_ns,
        static_cast<double>(compact_del_bytes) / compact_dyn_set.size(),
        compact_del_bytes / (1024.0 * 1024.0), absl_erase_ns / compact_erase_ns,
        100.0 *
            (static_cast<double>(compact_del_bytes) / absl_del_bytes - 1.0));
  }

  // 9. Incremental Insertions & 10. Incremental Deletions on Pre-Built Tree
  const size_t kNumIncremental =
      std::min(num_keys / 10, static_cast<size_t>(10000));
  if (kNumIncremental > 0) {
    std::vector<KeyT> inc_keys;
    inc_keys.reserve(kNumIncremental);
    for (size_t i = 0; i < kNumIncremental; ++i) {
      inc_keys.push_back(static_cast<KeyT>(
          absl::Uniform<uint64_t>(bitgen, 0, (num_keys + 1) * 10)));
    }

    // --- std::set ---
    const size_t std_inc_before = AllocatedBefore();
    std::set<KeyT> std_prebuilt(keys.begin(), keys.end());
    const double inc_std_0 = hwy::platform::Now();
    for (size_t i = 0; i < kNumIncremental; ++i) {
      std_prebuilt.insert(inc_keys[i]);
    }
    const double inc_std_1 = hwy::platform::Now();
    const size_t std_inc_bytes = GetAllocatedBytes(std_inc_before, 0);

    const double dec_std_0 = hwy::platform::Now();
    for (size_t i = 0; i < kNumIncremental; ++i) {
      std_prebuilt.erase(inc_keys[i]);
    }
    const double dec_std_1 = hwy::platform::Now();
    const size_t std_inc_del_bytes = GetAllocatedBytes(std_inc_before, 0);

    // --- absl::btree_set ---
    const size_t absl_inc_before = AllocatedBefore();
    absl::btree_set<KeyT> absl_prebuilt(keys.begin(), keys.end());
    const double inc_absl_0 = hwy::platform::Now();
    for (size_t i = 0; i < kNumIncremental; ++i) {
      absl_prebuilt.insert(inc_keys[i]);
    }
    const double inc_absl_1 = hwy::platform::Now();
    const size_t absl_inc_bytes = GetAllocatedBytes(absl_inc_before, 0);

    const double dec_absl_0 = hwy::platform::Now();
    for (size_t i = 0; i < kNumIncremental; ++i) {
      absl_prebuilt.erase(inc_keys[i]);
    }
    const double dec_absl_1 = hwy::platform::Now();
    const size_t absl_inc_del_bytes = GetAllocatedBytes(absl_inc_before, 0);

    // --- hwy::BTreeSet ---
    const size_t hwy_inc_before = AllocatedBefore();
    auto hwy_prebuilt =
        BTreeSet<KeyT>::Build(keys.data(), keys.size(), /*fill_ratio=*/0.75f);
    const double inc_hwy_0 = hwy::platform::Now();
    for (size_t i = 0; i < kNumIncremental; ++i) {
      hwy_prebuilt.insert(inc_keys[i]);
    }
    const double inc_hwy_1 = hwy::platform::Now();
    const size_t hwy_inc_bytes =
        GetAllocatedBytes(hwy_inc_before, hwy_prebuilt.AllocatedBytes());

    const double dec_hwy_0 = hwy::platform::Now();
    for (size_t i = 0; i < kNumIncremental; ++i) {
      hwy_prebuilt.erase(inc_keys[i]);
    }
    const double dec_hwy_1 = hwy::platform::Now();
    const size_t hwy_inc_del_bytes =
        GetAllocatedBytes(hwy_inc_before, hwy_prebuilt.AllocatedBytes());

    // --- hwy::CompactBTreeSet ---
    const size_t compact_inc_before = AllocatedBefore();
    auto compact_prebuilt = CompactBTreeSet<KeyT>::Build(
        keys.data(), keys.size(), /*fill_ratio=*/0.75f);
    const double inc_compact_0 = hwy::platform::Now();
    for (size_t i = 0; i < kNumIncremental; ++i) {
      compact_prebuilt.insert(inc_keys[i]);
    }
    const double inc_compact_1 = hwy::platform::Now();
    const size_t compact_inc_bytes = GetAllocatedBytes(
        compact_inc_before, compact_prebuilt.AllocatedBytes());

    const double dec_compact_0 = hwy::platform::Now();
    for (size_t i = 0; i < kNumIncremental; ++i) {
      compact_prebuilt.erase(inc_keys[i]);
    }
    const double dec_compact_1 = hwy::platform::Now();
    const size_t compact_inc_del_bytes = GetAllocatedBytes(
        compact_inc_before, compact_prebuilt.AllocatedBytes());

    const double std_inc_ns = (inc_std_1 - inc_std_0) * 1e9 / kNumIncremental;
    const double absl_inc_ns =
        (inc_absl_1 - inc_absl_0) * 1e9 / kNumIncremental;
    const double hwy_inc_ns = (inc_hwy_1 - inc_hwy_0) * 1e9 / kNumIncremental;
    const double compact_inc_ns =
        (inc_compact_1 - inc_compact_0) * 1e9 / kNumIncremental;

    printf(
        "\nIncremental Insertions Latency & Memory (Pre-Built Tree N = %zu + "
        "%zu keys, 75%% initial fill):\n",
        num_keys, kNumIncremental);
    printf("  std::set             : %6.2f ns/op (%5.1f B/key, %5.2f MB)\n",
           std_inc_ns,
           static_cast<double>(std_inc_bytes) /
               (std_prebuilt.size() + kNumIncremental),
           std_inc_bytes / (1024.0 * 1024.0));
    printf("  absl::btree_set      : %6.2f ns/op (%5.1f B/key, %5.2f MB)\n",
           absl_inc_ns,
           static_cast<double>(absl_inc_bytes) /
               (absl_prebuilt.size() + kNumIncremental),
           absl_inc_bytes / (1024.0 * 1024.0));
    printf(
        "  hwy::BTreeSet        : %6.2f ns/op (%5.1f B/key, %5.2f MB) -> "
        "%.2fx speedup vs absl\n",
        hwy_inc_ns,
        static_cast<double>(hwy_inc_bytes) /
            (hwy_prebuilt.size() + kNumIncremental),
        hwy_inc_bytes / (1024.0 * 1024.0), absl_inc_ns / hwy_inc_ns);
    if (compact_inc_bytes <= absl_inc_bytes) {
      printf(
          "  hwy::CompactBTreeSet : %6.2f ns/op (%5.1f B/key, %5.2f MB) -> "
          "%.2fx vs absl (%.1f%% smaller!)\n",
          compact_inc_ns,
          static_cast<double>(compact_inc_bytes) /
              (compact_prebuilt.size() + kNumIncremental),
          compact_inc_bytes / (1024.0 * 1024.0), absl_inc_ns / compact_inc_ns,
          100.0 *
              (1.0 - static_cast<double>(compact_inc_bytes) / absl_inc_bytes));
    } else {
      printf(
          "  hwy::CompactBTreeSet : %6.2f ns/op (%5.1f B/key, %5.2f MB) -> "
          "%.2fx vs absl (%.1f%% larger)\n",
          compact_inc_ns,
          static_cast<double>(compact_inc_bytes) /
              (compact_prebuilt.size() + kNumIncremental),
          compact_inc_bytes / (1024.0 * 1024.0), absl_inc_ns / compact_inc_ns,
          100.0 *
              (static_cast<double>(compact_inc_bytes) / absl_inc_bytes - 1.0));
    }

    const double std_dec_ns = (dec_std_1 - dec_std_0) * 1e9 / kNumIncremental;
    const double absl_dec_ns =
        (dec_absl_1 - dec_absl_0) * 1e9 / kNumIncremental;
    const double hwy_dec_ns = (dec_hwy_1 - dec_hwy_0) * 1e9 / kNumIncremental;
    const double compact_dec_ns =
        (dec_compact_1 - dec_compact_0) * 1e9 / kNumIncremental;

    printf(
        "\nIncremental Deletions Latency & Memory on Pre-Built Tree (%zu "
        "random keys on %zu-key tree -> remaining keys: %zu):\n",
        kNumIncremental, num_keys, std_prebuilt.size());
    printf("  std::set             : %6.2f ns/op (%5.1f B/key, %5.2f MB)\n",
           std_dec_ns,
           static_cast<double>(std_inc_del_bytes) / std_prebuilt.size(),
           std_inc_del_bytes / (1024.0 * 1024.0));
    printf("  absl::btree_set      : %6.2f ns/op (%5.1f B/key, %5.2f MB)\n",
           absl_dec_ns,
           static_cast<double>(absl_inc_del_bytes) / absl_prebuilt.size(),
           absl_inc_del_bytes / (1024.0 * 1024.0));
    printf(
        "  hwy::BTreeSet        : %6.2f ns/op (%5.1f B/key, %5.2f MB) -> "
        "%.2fx speedup vs absl\n",
        hwy_dec_ns,
        static_cast<double>(hwy_inc_del_bytes) / hwy_prebuilt.size(),
        hwy_inc_del_bytes / (1024.0 * 1024.0), absl_dec_ns / hwy_dec_ns);
    if (compact_inc_del_bytes <= absl_inc_del_bytes) {
      printf(
          "  hwy::CompactBTreeSet : %6.2f ns/op (%5.1f B/key, %5.2f MB) -> "
          "%.2fx vs absl (%.1f%% smaller!)\n",
          compact_dec_ns,
          static_cast<double>(compact_inc_del_bytes) / compact_prebuilt.size(),
          compact_inc_del_bytes / (1024.0 * 1024.0),
          absl_dec_ns / compact_dec_ns,
          100.0 * (1.0 - static_cast<double>(compact_inc_del_bytes) /
                             absl_inc_del_bytes));
    } else {
      printf(
          "  hwy::CompactBTreeSet : %6.2f ns/op (%5.1f B/key, %5.2f MB) -> "
          "%.2fx vs absl (%.1f%% larger)\n",
          compact_dec_ns,
          static_cast<double>(compact_inc_del_bytes) / compact_prebuilt.size(),
          compact_inc_del_bytes / (1024.0 * 1024.0),
          absl_dec_ns / compact_dec_ns,
          100.0 *
              (static_cast<double>(compact_inc_del_bytes) / absl_inc_del_bytes -
               1.0));
    }
  }

  HWY_ASSERT(hwy_hits == absl_hits);
  HWY_ASSERT(compact_hits == absl_hits);
  HWY_ASSERT(batch_hits == absl_hits);
  HWY_ASSERT(compact_batch_hits == absl_hits);
  HWY_ASSERT(hwy_lb_sum == absl_lb_sum);
  HWY_ASSERT(compact_lb_sum == absl_lb_sum);
  HWY_ASSERT(batch_lb_sum == absl_lb_sum);
  HWY_ASSERT(compact_batch_lb_sum == absl_lb_sum);
}

template <typename KeyT>
void RunWorstCaseBenchmarkSuite(size_t num_keys) {
  printf("\n===============================================================\n");
  printf("  Worst-Case B-Tree Set Benchmark (%zu-bit, N = %zu keys, %s)\n",
         sizeof(KeyT) * 8, num_keys, hwy::TargetName(HWY_TARGET));
  printf("  Key Distribution: Uncompressible Uniform %zu-bit Random Keys\n",
         sizeof(KeyT) * 8);
  printf("  Query Pattern   : 100%% Lookup Misses (Disjoint Range)\n");
  printf("===============================================================\n");

  absl::BitGen bitgen;
  std::vector<KeyT> keys;
  keys.reserve(num_keys);
  for (size_t i = 0; i < num_keys; ++i) {
    keys.push_back(static_cast<KeyT>(
        absl::Uniform<KeyT>(bitgen, 0, std::numeric_limits<KeyT>::max() / 2) *
        2));
  }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  num_keys = keys.size();

  // 1. Build Containers
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
  auto hwy_tree = BTreeSet<KeyT>::Build(keys.data(), keys.size(), 1.0f);
  const double end_hwy = hwy::platform::Now();
  const size_t hwy_bytes =
      GetAllocatedBytes(hwy_before, hwy_tree.AllocatedBytes());

  const size_t compact_before = AllocatedBefore();
  const double start_compact = hwy::platform::Now();
  auto compact_tree =
      CompactBTreeSet<KeyT>::Build(keys.data(), keys.size(), 1.0f);
  const double end_compact = hwy::platform::Now();
  const size_t compact_bytes =
      GetAllocatedBytes(compact_before, compact_tree.AllocatedBytes());

  printf("Worst-Case Build Time:\n");
  printf("  std::set             : %8.2f ms\n", (end_std - start_std) * 1000.0);
  printf("  absl::btree_set      : %8.2f ms\n",
         (end_absl - start_absl) * 1000.0);
  printf("  hwy::BTreeSet        : %8.2f ms (%.1fx faster than absl)\n",
         (end_hwy - start_hwy) * 1000.0,
         (end_absl - start_absl) / (end_hwy - start_hwy));
  printf("  hwy::CompactBTreeSet : %8.2f ms (%.1fx faster than absl)\n",
         (end_compact - start_compact) * 1000.0,
         (end_absl - start_absl) / (end_compact - start_compact));

  printf("\nWorst-Case Memory Footprint (Uncompressible Raw Mode):\n");
  printf("  std::set             : %5.2f MB (%5.1f B/key)\n",
         std_bytes / (1024.0 * 1024.0),
         static_cast<double>(std_bytes) / num_keys);
  printf("  absl::btree_set      : %5.2f MB (%5.1f B/key)\n",
         absl_bytes / (1024.0 * 1024.0),
         static_cast<double>(absl_bytes) / num_keys);
  printf("  hwy::BTreeSet        : %5.2f MB (%5.1f B/key)\n",
         hwy_bytes / (1024.0 * 1024.0),
         static_cast<double>(hwy_bytes) / num_keys);
  printf("  hwy::CompactBTreeSet : %5.2f MB (%5.1f B/key)\n",
         compact_bytes / (1024.0 * 1024.0),
         static_cast<double>(compact_bytes) / num_keys);

  // 2. Worst-case 100% Miss Point Lookups (Odd keys vs Even set)
  constexpr size_t kNumQueries = 1000000;
  std::vector<KeyT> miss_queries;
  miss_queries.reserve(kNumQueries);
  for (size_t i = 0; i < kNumQueries; ++i) {
    miss_queries.push_back(static_cast<KeyT>(
        absl::Uniform<KeyT>(bitgen, 0, std::numeric_limits<KeyT>::max() / 2) *
            2 +
        1));
  }

  uint64_t std_hits = 0, absl_hits = 0, hwy_hits = 0, compact_hits = 0;
  const double t0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumQueries; ++i) {
    std_hits += (std_tree.find(miss_queries[i]) != std_tree.end());
  }
  hwy::PreventElision(std_hits);
  const double t1 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    absl_hits += (absl_tree.find(miss_queries[i]) != absl_tree.end());
  }
  hwy::PreventElision(absl_hits);
  const double t2 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    hwy_hits += hwy_tree.Contains(miss_queries[i]);
  }
  hwy::PreventElision(hwy_hits);
  const double t3 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    compact_hits += compact_tree.contains(miss_queries[i]);
  }
  hwy::PreventElision(compact_hits);
  const double t4 = hwy::platform::Now();

  const double std_miss_ns = (t1 - t0) * 1e9 / kNumQueries;
  const double absl_miss_ns = (t2 - t1) * 1e9 / kNumQueries;
  const double hwy_miss_ns = (t3 - t2) * 1e9 / kNumQueries;
  const double compact_miss_ns = (t4 - t3) * 1e9 / kNumQueries;

  printf("\nWorst-Case Point Lookup Miss Latency (100%% Key Misses):\n");
  printf("  std::set             : %6.2f ns/op (%6.2f Mops/s)\n", std_miss_ns,
         1000.0 / std_miss_ns);
  printf("  absl::btree_set      : %6.2f ns/op (%6.2f Mops/s)\n", absl_miss_ns,
         1000.0 / absl_miss_ns);
  printf(
      "  hwy::BTreeSet        : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
      hwy_miss_ns, 1000.0 / hwy_miss_ns, absl_miss_ns / hwy_miss_ns);
  printf(
      "  hwy::CompactBTreeSet : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
      compact_miss_ns, 1000.0 / compact_miss_ns,
      absl_miss_ns / compact_miss_ns);

  HWY_ASSERT(std_hits == 0);
  HWY_ASSERT(absl_hits == 0);
  HWY_ASSERT(hwy_hits == 0);
  HWY_ASSERT(compact_hits == 0);
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

  const size_t compact_before = AllocatedBefore();
  const double start_compact = hwy::platform::Now();
  auto compact_map = CompactBTreeMap<KeyT, ValueT>::Build(
      keys.data(), vals.data(), keys.size());
  const double end_compact = hwy::platform::Now();
  const size_t compact_bytes =
      GetAllocatedBytes(compact_before, compact_map.AllocatedBytes());

  const double hwy_build_ms = (end_hwy - start_hwy) * 1000.0;
  const double compact_build_ms = (end_compact - start_compact) * 1000.0;
  const double absl_build_ms = (end_absl - start_absl) * 1000.0;
  const double std_build_ms = (end_std - start_std) * 1000.0;

  printf("Build Time:\n");
  printf("  std::map             : %8.2f ms\n", std_build_ms);
  printf("  absl::btree_map      : %8.2f ms\n", absl_build_ms);
  printf("  hwy::BTreeMap        : %8.2f ms (%.1fx faster than absl)\n",
         hwy_build_ms, absl_build_ms / (hwy_build_ms + 1e-6));
  printf("  hwy::CompactBTreeMap : %8.2f ms (%.1fx faster than absl)\n",
         compact_build_ms, absl_build_ms / (compact_build_ms + 1e-6));

  // Memory Footprint
  printf(
      "\nMemory Footprint (100%% Fill Bulk-Loaded State, TCMalloc "
      "Measured):\n");
  printf("  std::map             : %6.2f MB (%5.1f B/pair)\n",
         std_bytes / (1024.0 * 1024.0),
         static_cast<double>(std_bytes) / num_keys);
  printf("  absl::btree_map      : %6.2f MB (%5.1f B/pair)\n",
         absl_bytes / (1024.0 * 1024.0),
         static_cast<double>(absl_bytes) / num_keys);
  printf("  hwy::BTreeMap        : %6.2f MB (%5.1f B/pair)\n",
         hwy_bytes / (1024.0 * 1024.0),
         static_cast<double>(hwy_bytes) / num_keys);
  printf(
      "  hwy::CompactBTreeMap : %6.2f MB (%5.1f B/pair) -> %.1f%% smaller than "
      "absl!\n",
      compact_bytes / (1024.0 * 1024.0),
      static_cast<double>(compact_bytes) / num_keys,
      100.0 * (1.0 - static_cast<double>(compact_bytes) / absl_bytes));

  // 2. Generate Random Query Keys
  constexpr size_t kNumQueries = 1000000;
  std::vector<KeyT> queries;
  queries.reserve(kNumQueries);
  for (size_t i = 0; i < kNumQueries; ++i) {
    queries.push_back(static_cast<KeyT>(
        absl::Uniform<uint64_t>(bitgen, 0, (num_keys + 1) * 10)));
  }

  // 3. Point Lookups (FindValue / Find)
  uint64_t hwy_hits = 0, compact_hits = 0, absl_hits = 0, std_hits = 0;

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

  for (size_t i = 0; i < kNumQueries; ++i) {
    const ValueT* ptr = compact_map.FindValue(queries[i]);
    if (ptr != nullptr) compact_hits += static_cast<uint64_t>(*ptr);
  }
  hwy::PreventElision(compact_hits);
  const double t4 = hwy::platform::Now();

  const double std_lookup_ns = (t1 - t0) * 1e9 / kNumQueries;
  const double absl_lookup_ns = (t2 - t1) * 1e9 / kNumQueries;
  const double hwy_lookup_ns = (t3 - t2) * 1e9 / kNumQueries;
  const double compact_lookup_ns = (t4 - t3) * 1e9 / kNumQueries;

  printf("\nPoint Lookup Latency (1M queries on 100%% Bulk-Loaded Map):\n");
  printf("  std::map             : %6.2f ns/op (%6.2f Mops/s)\n", std_lookup_ns,
         1000.0 / std_lookup_ns);
  printf("  absl::btree_map      : %6.2f ns/op (%6.2f Mops/s)\n",
         absl_lookup_ns, 1000.0 / absl_lookup_ns);
  printf(
      "  hwy::BTreeMap        : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
      hwy_lookup_ns, 1000.0 / hwy_lookup_ns, absl_lookup_ns / hwy_lookup_ns);
  printf(
      "  hwy::CompactBTreeMap : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
      compact_lookup_ns, 1000.0 / compact_lookup_ns,
      absl_lookup_ns / compact_lookup_ns);

  // 4. Ordered Range Queries (LowerBound)
  uint64_t hwy_lb_sum = 0, compact_lb_sum = 0, absl_lb_sum = 0, std_lb_sum = 0;

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

  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = compact_map.lower_bound(queries[i]);
    if (it != compact_map.end()) {
      compact_lb_sum += static_cast<uint64_t>(it->second);
    }
  }
  hwy::PreventElision(compact_lb_sum);
  const double r4 = hwy::platform::Now();

  const double std_lb_ns = (r1 - r0) * 1e9 / kNumQueries;
  const double absl_lb_ns = (r2 - r1) * 1e9 / kNumQueries;
  const double hwy_lb_ns = (r3 - r2) * 1e9 / kNumQueries;
  const double compact_lb_ns = (r4 - r3) * 1e9 / kNumQueries;

  printf(
      "\nLowerBound Range Query Latency (1M queries on 100%% Bulk-Loaded "
      "Map):\n");
  printf("  std::map             : %6.2f ns/op (%6.2f Mops/s)\n", std_lb_ns,
         1000.0 / std_lb_ns);
  printf("  absl::btree_map      : %6.2f ns/op (%6.2f Mops/s)\n", absl_lb_ns,
         1000.0 / absl_lb_ns);
  printf(
      "  hwy::BTreeMap        : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
      hwy_lb_ns, 1000.0 / hwy_lb_ns, absl_lb_ns / hwy_lb_ns);
  printf(
      "  hwy::CompactBTreeMap : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
      compact_lb_ns, 1000.0 / compact_lb_ns, absl_lb_ns / compact_lb_ns);

  // 4b. Ordered Range Queries (UpperBound)
  uint64_t hwy_ub_sum = 0, compact_ub_sum = 0, absl_ub_sum = 0, std_ub_sum = 0;

  const double u0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = std_map.upper_bound(queries[i]);
    if (it != std_map.end()) std_ub_sum += static_cast<uint64_t>(it->second);
  }
  hwy::PreventElision(std_ub_sum);
  const double u1 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = absl_map.upper_bound(queries[i]);
    if (it != absl_map.end()) absl_ub_sum += static_cast<uint64_t>(it->second);
  }
  hwy::PreventElision(absl_ub_sum);
  const double u2 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = hwy_map.upper_bound(queries[i]);
    if (it != hwy_map.end()) hwy_ub_sum += static_cast<uint64_t>(it->second);
  }
  hwy::PreventElision(hwy_ub_sum);
  const double u3 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = compact_map.upper_bound(queries[i]);
    if (it != compact_map.end()) {
      compact_ub_sum += static_cast<uint64_t>(it->second);
    }
  }
  hwy::PreventElision(compact_ub_sum);
  const double u4 = hwy::platform::Now();

  const double std_ub_ns = (u1 - u0) * 1e9 / kNumQueries;
  const double absl_ub_ns = (u2 - u1) * 1e9 / kNumQueries;
  const double hwy_ub_ns = (u3 - u2) * 1e9 / kNumQueries;
  const double compact_ub_ns = (u4 - u3) * 1e9 / kNumQueries;

  printf(
      "\nUpperBound Range Query Latency (1M queries on 100%% Bulk-Loaded "
      "Map):\n");
  printf("  std::map             : %6.2f ns/op (%6.2f Mops/s)\n", std_ub_ns,
         1000.0 / std_ub_ns);
  printf("  absl::btree_map      : %6.2f ns/op (%6.2f Mops/s)\n", absl_ub_ns,
         1000.0 / absl_ub_ns);
  printf(
      "  hwy::BTreeMap        : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
      hwy_ub_ns, 1000.0 / hwy_ub_ns, absl_ub_ns / hwy_ub_ns);
  printf(
      "  hwy::CompactBTreeMap : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
      compact_ub_ns, 1000.0 / compact_ub_ns, absl_ub_ns / compact_ub_ns);

  // 5. Batch Value Lookups (FindValueBatch - 8-way pipelined prefetch)
  std::vector<const ValueT*> batch_vals(kNumQueries);
  const double mb0 = hwy::platform::Now();
  hwy_map.FindValueBatch(queries.data(), kNumQueries, batch_vals.data());
  const double mb1 = hwy::platform::Now();

  uint64_t batch_hits = 0;
  for (size_t i = 0; i < kNumQueries; ++i) {
    if (batch_vals[i] != nullptr) {
      batch_hits += static_cast<uint64_t>(*batch_vals[i]);
    }
  }
  hwy::PreventElision(batch_hits);

  std::vector<ValueT> compact_batch_vals(kNumQueries);
  std::unique_ptr<bool[]> compact_batch_found(new bool[kNumQueries]);
  const double cmb0 = hwy::platform::Now();
  compact_map.LookupBatch(queries.data(), kNumQueries,
                          compact_batch_vals.data(), compact_batch_found.get());
  const double cmb1 = hwy::platform::Now();

  uint64_t compact_batch_hits = 0;
  for (size_t i = 0; i < kNumQueries; ++i) {
    if (compact_batch_found[i]) {
      compact_batch_hits += static_cast<uint64_t>(compact_batch_vals[i]);
    }
  }
  hwy::PreventElision(compact_batch_hits);

  const double hwy_batch_lookup_ns = (mb1 - mb0) * 1e9 / kNumQueries;
  const double compact_batch_lookup_ns = (cmb1 - cmb0) * 1e9 / kNumQueries;
  printf(
      "\nBatch Value Lookup (1M queries on 100%% Bulk-Loaded Map, 8-way "
      "pipelined prefetch):\n");
  printf("  hwy::BTreeMap (Serial)        : %6.2f ns/op (%6.2f Mops/s)\n",
         hwy_lookup_ns, 1000.0 / hwy_lookup_ns);
  printf(
      "  hwy::BTreeMap (Batch)         : %6.2f ns/op (%6.2f Mops/s) -> %.2fx "
      "vs Serial (%.2fx vs absl)\n",
      hwy_batch_lookup_ns, 1000.0 / hwy_batch_lookup_ns,
      hwy_lookup_ns / hwy_batch_lookup_ns,
      absl_lookup_ns / hwy_batch_lookup_ns);
  printf("  hwy::CompactBTreeMap (Serial) : %6.2f ns/op (%6.2f Mops/s)\n",
         compact_lookup_ns, 1000.0 / compact_lookup_ns);
  printf(
      "  hwy::CompactBTreeMap (Batch)  : %6.2f ns/op (%6.2f Mops/s) -> %.2fx "
      "vs Serial (%.2fx vs absl)\n",
      compact_batch_lookup_ns, 1000.0 / compact_batch_lookup_ns,
      compact_lookup_ns / compact_batch_lookup_ns,
      absl_lookup_ns / compact_batch_lookup_ns);

  // 6. Batch LowerBound Queries (LowerBoundBatch - 8-way pipelined prefetch)
  std::vector<typename BTreeMap<KeyT, ValueT>::const_iterator> batch_iters(
      kNumQueries);
  const double mblb0 = hwy::platform::Now();
  hwy_map.LowerBoundBatch(queries.data(), kNumQueries, batch_iters.data());
  const double mblb1 = hwy::platform::Now();

  uint64_t batch_lb_sum = 0;
  for (size_t i = 0; i < kNumQueries; ++i) {
    if (batch_iters[i] != hwy_map.end()) {
      batch_lb_sum += static_cast<uint64_t>(batch_iters[i]->second);
    }
  }
  hwy::PreventElision(batch_lb_sum);

  std::vector<typename CompactBTreeMap<KeyT, ValueT>::const_iterator>
      compact_batch_iters(kNumQueries);
  const double cmblb0 = hwy::platform::Now();
  compact_map.LowerBoundBatch(queries.data(), kNumQueries,
                              compact_batch_iters.data());
  const double cmblb1 = hwy::platform::Now();

  uint64_t compact_batch_lb_sum = 0;
  for (size_t i = 0; i < kNumQueries; ++i) {
    if (compact_batch_iters[i] != compact_map.end()) {
      compact_batch_lb_sum +=
          static_cast<uint64_t>(compact_batch_iters[i]->second);
    }
  }
  hwy::PreventElision(compact_batch_lb_sum);

  const double hwy_batch_lb_ns = (mblb1 - mblb0) * 1e9 / kNumQueries;
  const double compact_batch_lb_ns = (cmblb1 - cmblb0) * 1e9 / kNumQueries;
  printf(
      "\nBatch LowerBound Query (1M queries on 100%% Bulk-Loaded Map, 8-way "
      "pipelined prefetch):\n");
  printf("  hwy::BTreeMap (Serial)        : %6.2f ns/op (%6.2f Mops/s)\n",
         hwy_lb_ns, 1000.0 / hwy_lb_ns);
  printf(
      "  hwy::BTreeMap (Batch)         : %6.2f ns/op (%6.2f Mops/s) -> %.2fx "
      "vs Serial (%.2fx vs absl)\n",
      hwy_batch_lb_ns, 1000.0 / hwy_batch_lb_ns, hwy_lb_ns / hwy_batch_lb_ns,
      absl_lb_ns / hwy_batch_lb_ns);
  printf("  hwy::CompactBTreeMap (Serial) : %6.2f ns/op (%6.2f Mops/s)\n",
         compact_lb_ns, 1000.0 / compact_lb_ns);
  printf(
      "  hwy::CompactBTreeMap (Batch)  : %6.2f ns/op (%6.2f Mops/s) -> %.2fx "
      "vs Serial (%.2fx vs absl)\n",
      compact_batch_lb_ns, 1000.0 / compact_batch_lb_ns,
      compact_lb_ns / compact_batch_lb_ns, absl_lb_ns / compact_batch_lb_ns);

  // 7. Dynamic Random Insertions & 8. Dynamic Deletions on Empty Map
  const size_t kNumMutations = std::min(num_keys, static_cast<size_t>(100000));
  const size_t kNumErases = kNumMutations / 2;
  std::vector<KeyT> mut_keys;
  std::vector<ValueT> mut_vals;
  mut_keys.reserve(kNumMutations);
  mut_vals.reserve(kNumMutations);
  for (size_t i = 0; i < kNumMutations; ++i) {
    KeyT k = static_cast<KeyT>(
        absl::Uniform<uint64_t>(bitgen, 0, (num_keys + 1) * 20));
    ValueT v = static_cast<ValueT>(k * 3 + 7);
    mut_keys.push_back(k);
    mut_vals.push_back(v);
  }

  // --- std::map ---
  const size_t std_dyn_map_before = AllocatedBefore();
  std::map<KeyT, ValueT> std_dyn_map;
  const double mi_std_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumMutations; ++i) {
    std_dyn_map[mut_keys[i]] = mut_vals[i];
  }
  const double mi_std_1 = hwy::platform::Now();
  const size_t std_dyn_map_bytes = GetAllocatedBytes(std_dyn_map_before, 0);

  const double me_std_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumErases; ++i) {
    std_dyn_map.erase(mut_keys[i]);
  }
  const double me_std_1 = hwy::platform::Now();
  const size_t std_dyn_map_del_bytes = GetAllocatedBytes(std_dyn_map_before, 0);

  // --- absl::btree_map ---
  const size_t absl_dyn_map_before = AllocatedBefore();
  absl::btree_map<KeyT, ValueT> absl_dyn_map;
  const double mi_absl_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumMutations; ++i) {
    absl_dyn_map[mut_keys[i]] = mut_vals[i];
  }
  const double mi_absl_1 = hwy::platform::Now();
  const size_t absl_dyn_map_bytes = GetAllocatedBytes(absl_dyn_map_before, 0);

  const double me_absl_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumErases; ++i) {
    absl_dyn_map.erase(mut_keys[i]);
  }
  const double me_absl_1 = hwy::platform::Now();
  const size_t absl_dyn_map_del_bytes =
      GetAllocatedBytes(absl_dyn_map_before, 0);

  // --- hwy::BTreeMap ---
  const size_t hwy_dyn_map_before = AllocatedBefore();
  BTreeMap<KeyT, ValueT> hwy_dyn_map;
  const double mi_hwy_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumMutations; ++i) {
    hwy_dyn_map.insert(mut_keys[i], mut_vals[i]);
  }
  const double mi_hwy_1 = hwy::platform::Now();
  const size_t hwy_dyn_map_bytes =
      GetAllocatedBytes(hwy_dyn_map_before, hwy_dyn_map.AllocatedBytes());

  const double me_hwy_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumErases; ++i) {
    hwy_dyn_map.erase(mut_keys[i]);
  }
  const double me_hwy_1 = hwy::platform::Now();
  const size_t hwy_dyn_map_del_bytes =
      GetAllocatedBytes(hwy_dyn_map_before, hwy_dyn_map.AllocatedBytes());

  // --- hwy::CompactBTreeMap ---
  const size_t compact_dyn_map_before = AllocatedBefore();
  CompactBTreeMap<KeyT, ValueT> compact_dyn_map;
  const double mi_compact_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumMutations; ++i) {
    compact_dyn_map.insert(mut_keys[i], mut_vals[i]);
  }
  const double mi_compact_1 = hwy::platform::Now();
  const size_t compact_dyn_map_bytes = GetAllocatedBytes(
      compact_dyn_map_before, compact_dyn_map.AllocatedBytes());

  const double me_compact_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumErases; ++i) {
    compact_dyn_map.erase(mut_keys[i]);
  }
  const double me_compact_1 = hwy::platform::Now();
  const size_t compact_dyn_map_del_bytes = GetAllocatedBytes(
      compact_dyn_map_before, compact_dyn_map.AllocatedBytes());

  const double std_ins_ns = (mi_std_1 - mi_std_0) * 1e9 / kNumMutations;
  const double absl_ins_ns = (mi_absl_1 - mi_absl_0) * 1e9 / kNumMutations;
  const double hwy_ins_ns = (mi_hwy_1 - mi_hwy_0) * 1e9 / kNumMutations;
  const double compact_ins_ns =
      (mi_compact_1 - mi_compact_0) * 1e9 / kNumMutations;

  printf(
      "\nDynamic Insertions into Empty Map (%zu random pairs -> Pure Dynamic "
      "Steady State):\n",
      kNumMutations);
  printf("  std::map             : %6.2f ns/op (%5.1f B/pair, %5.2f MB)\n",
         std_ins_ns,
         static_cast<double>(std_dyn_map_bytes) /
             (std_dyn_map.size() + kNumErases),
         std_dyn_map_bytes / (1024.0 * 1024.0));
  printf("  absl::btree_map      : %6.2f ns/op (%5.1f B/pair, %5.2f MB)\n",
         absl_ins_ns,
         static_cast<double>(absl_dyn_map_bytes) /
             (absl_dyn_map.size() + kNumErases),
         absl_dyn_map_bytes / (1024.0 * 1024.0));
  printf(
      "  hwy::BTreeMap        : %6.2f ns/op (%5.1f B/pair, %5.2f MB) -> %.2fx "
      "speedup vs absl\n",
      hwy_ins_ns,
      static_cast<double>(hwy_dyn_map_bytes) /
          (hwy_dyn_map.size() + kNumErases),
      hwy_dyn_map_bytes / (1024.0 * 1024.0), absl_ins_ns / hwy_ins_ns);
  printf(
      "  hwy::CompactBTreeMap : %6.2f ns/op (%5.1f B/pair, %5.2f MB) -> %.2fx "
      "speedup vs absl\n",
      compact_ins_ns,
      static_cast<double>(compact_dyn_map_bytes) /
          (compact_dyn_map.size() + kNumErases),
      compact_dyn_map_bytes / (1024.0 * 1024.0), absl_ins_ns / compact_ins_ns);

  const double std_erase_ns = (me_std_1 - me_std_0) * 1e9 / kNumErases;
  const double absl_erase_ns = (me_absl_1 - me_absl_0) * 1e9 / kNumErases;
  const double hwy_erase_ns = (me_hwy_1 - me_hwy_0) * 1e9 / kNumErases;
  const double compact_erase_ns =
      (me_compact_1 - me_compact_0) * 1e9 / kNumErases;

  printf(
      "\nDynamic Deletions Latency & Memory on Steady-State Map (%zu random "
      "pairs -> remaining pairs: %zu after 50%% deletions):\n",
      kNumErases, std_dyn_map.size());
  printf("  std::map             : %6.2f ns/op (%5.1f B/pair, %5.2f MB)\n",
         std_erase_ns,
         static_cast<double>(std_dyn_map_del_bytes) / std_dyn_map.size(),
         std_dyn_map_del_bytes / (1024.0 * 1024.0));
  printf("  absl::btree_map      : %6.2f ns/op (%5.1f B/pair, %5.2f MB)\n",
         absl_erase_ns,
         static_cast<double>(absl_dyn_map_del_bytes) / absl_dyn_map.size(),
         absl_dyn_map_del_bytes / (1024.0 * 1024.0));
  printf(
      "  hwy::BTreeMap        : %6.2f ns/op (%5.1f B/pair, %5.2f MB) -> %.2fx "
      "speedup vs absl\n",
      hwy_erase_ns,
      static_cast<double>(hwy_dyn_map_del_bytes) / hwy_dyn_map.size(),
      hwy_dyn_map_del_bytes / (1024.0 * 1024.0), absl_erase_ns / hwy_erase_ns);
  if (compact_dyn_map_del_bytes <= absl_dyn_map_del_bytes) {
    printf(
        "  hwy::CompactBTreeMap : %6.2f ns/op (%5.1f B/pair, %5.2f MB) -> "
        "%.2fx vs absl (%.1f%% smaller!)\n",
        compact_erase_ns,
        static_cast<double>(compact_dyn_map_del_bytes) / compact_dyn_map.size(),
        compact_dyn_map_del_bytes / (1024.0 * 1024.0),
        absl_erase_ns / compact_erase_ns,
        100.0 * (1.0 - static_cast<double>(compact_dyn_map_del_bytes) /
                           absl_dyn_map_del_bytes));
  } else {
    printf(
        "  hwy::CompactBTreeMap : %6.2f ns/op (%5.1f B/pair, %5.2f MB) -> "
        "%.2fx vs absl (%.1f%% larger)\n",
        compact_erase_ns,
        static_cast<double>(compact_dyn_map_del_bytes) / compact_dyn_map.size(),
        compact_dyn_map_del_bytes / (1024.0 * 1024.0),
        absl_erase_ns / compact_erase_ns,
        100.0 * (static_cast<double>(compact_dyn_map_del_bytes) /
                     absl_dyn_map_del_bytes -
                 1.0));
  }

  // 9. Incremental Insertions & 10. Incremental Deletions on Pre-Built Map
  const size_t kNumIncremental =
      std::min(num_keys / 10, static_cast<size_t>(10000));
  if (kNumIncremental > 0) {
    std::vector<KeyT> inc_keys;
    std::vector<ValueT> inc_vals;
    inc_keys.reserve(kNumIncremental);
    inc_vals.reserve(kNumIncremental);
    for (size_t i = 0; i < kNumIncremental; ++i) {
      KeyT k = static_cast<KeyT>(
          absl::Uniform<uint64_t>(bitgen, 0, (num_keys + 1) * 10));
      ValueT v = static_cast<ValueT>(k * 5 + 13);
      inc_keys.push_back(k);
      inc_vals.push_back(v);
    }

    // --- std::map ---
    const size_t std_inc_map_before = AllocatedBefore();
    std::map<KeyT, ValueT> std_prebuilt_map(kv_pairs.begin(), kv_pairs.end());
    const double inc_std_0 = hwy::platform::Now();
    for (size_t i = 0; i < kNumIncremental; ++i) {
      std_prebuilt_map[inc_keys[i]] = inc_vals[i];
    }
    const double inc_std_1 = hwy::platform::Now();
    const size_t std_inc_map_bytes = GetAllocatedBytes(std_inc_map_before, 0);

    const double dec_std_0 = hwy::platform::Now();
    for (size_t i = 0; i < kNumIncremental; ++i) {
      std_prebuilt_map.erase(inc_keys[i]);
    }
    const double dec_std_1 = hwy::platform::Now();
    const size_t std_inc_map_del_bytes =
        GetAllocatedBytes(std_inc_map_before, 0);

    // --- absl::btree_map ---
    const size_t absl_inc_map_before = AllocatedBefore();
    absl::btree_map<KeyT, ValueT> absl_prebuilt_map(kv_pairs.begin(),
                                                    kv_pairs.end());
    const double inc_absl_0 = hwy::platform::Now();
    for (size_t i = 0; i < kNumIncremental; ++i) {
      absl_prebuilt_map[inc_keys[i]] = inc_vals[i];
    }
    const double inc_absl_1 = hwy::platform::Now();
    const size_t absl_inc_map_bytes = GetAllocatedBytes(absl_inc_map_before, 0);

    const double dec_absl_0 = hwy::platform::Now();
    for (size_t i = 0; i < kNumIncremental; ++i) {
      absl_prebuilt_map.erase(inc_keys[i]);
    }
    const double dec_absl_1 = hwy::platform::Now();
    const size_t absl_inc_map_del_bytes =
        GetAllocatedBytes(absl_inc_map_before, 0);

    // --- hwy::BTreeMap ---
    const size_t hwy_inc_map_before = AllocatedBefore();
    auto hwy_prebuilt_map = BTreeMap<KeyT, ValueT>::Build(
        keys.data(), vals.data(), keys.size(), /*fill_ratio=*/0.75f);
    const double inc_hwy_0 = hwy::platform::Now();
    for (size_t i = 0; i < kNumIncremental; ++i) {
      hwy_prebuilt_map.insert(inc_keys[i], inc_vals[i]);
    }
    const double inc_hwy_1 = hwy::platform::Now();
    const size_t hwy_inc_map_bytes = GetAllocatedBytes(
        hwy_inc_map_before, hwy_prebuilt_map.AllocatedBytes());

    const double dec_hwy_0 = hwy::platform::Now();
    for (size_t i = 0; i < kNumIncremental; ++i) {
      hwy_prebuilt_map.erase(inc_keys[i]);
    }
    const double dec_hwy_1 = hwy::platform::Now();
    const size_t hwy_inc_map_del_bytes = GetAllocatedBytes(
        hwy_inc_map_before, hwy_prebuilt_map.AllocatedBytes());

    // --- hwy::CompactBTreeMap ---
    const size_t compact_inc_map_before = AllocatedBefore();
    auto compact_prebuilt_map = CompactBTreeMap<KeyT, ValueT>::Build(
        keys.data(), vals.data(), keys.size(), /*fill_ratio=*/0.75);
    const double inc_compact_0 = hwy::platform::Now();
    for (size_t i = 0; i < kNumIncremental; ++i) {
      compact_prebuilt_map.insert_or_assign(inc_keys[i], inc_vals[i]);
    }
    const double inc_compact_1 = hwy::platform::Now();
    const size_t compact_inc_map_bytes = GetAllocatedBytes(
        compact_inc_map_before, compact_prebuilt_map.AllocatedBytes());

    const double dec_compact_0 = hwy::platform::Now();
    for (size_t i = 0; i < kNumIncremental; ++i) {
      compact_prebuilt_map.erase(inc_keys[i]);
    }
    const double dec_compact_1 = hwy::platform::Now();
    const size_t compact_inc_map_del_bytes = GetAllocatedBytes(
        compact_inc_map_before, compact_prebuilt_map.AllocatedBytes());

    const double std_inc_ns = (inc_std_1 - inc_std_0) * 1e9 / kNumIncremental;
    const double absl_inc_ns =
        (inc_absl_1 - inc_absl_0) * 1e9 / kNumIncremental;
    const double hwy_inc_ns = (inc_hwy_1 - inc_hwy_0) * 1e9 / kNumIncremental;
    const double compact_inc_ns =
        (inc_compact_1 - inc_compact_0) * 1e9 / kNumIncremental;

    printf(
        "\nIncremental Insertions Latency & Memory (Pre-Built Map N = %zu + "
        "%zu pairs, 75%% initial fill):\n",
        num_keys, kNumIncremental);
    printf("  std::map             : %6.2f ns/op (%5.1f B/pair, %5.2f MB)\n",
           std_inc_ns,
           static_cast<double>(std_inc_map_bytes) /
               (std_prebuilt_map.size() + kNumIncremental),
           std_inc_map_bytes / (1024.0 * 1024.0));
    printf("  absl::btree_map      : %6.2f ns/op (%5.1f B/pair, %5.2f MB)\n",
           absl_inc_ns,
           static_cast<double>(absl_inc_map_bytes) /
               (absl_prebuilt_map.size() + kNumIncremental),
           absl_inc_map_bytes / (1024.0 * 1024.0));
    printf(
        "  hwy::BTreeMap        : %6.2f ns/op (%5.1f B/pair, %5.2f MB) -> "
        "%.2fx speedup vs absl\n",
        hwy_inc_ns,
        static_cast<double>(hwy_inc_map_bytes) /
            (hwy_prebuilt_map.size() + kNumIncremental),
        hwy_inc_map_bytes / (1024.0 * 1024.0), absl_inc_ns / hwy_inc_ns);
    if (compact_inc_map_bytes <= absl_inc_map_bytes) {
      printf(
          "  hwy::CompactBTreeMap : %6.2f ns/op (%5.1f B/pair, %5.2f MB) -> "
          "%.2fx vs absl (%.1f%% smaller!)\n",
          compact_inc_ns,
          static_cast<double>(compact_inc_map_bytes) /
              (compact_prebuilt_map.size() + kNumIncremental),
          compact_inc_map_bytes / (1024.0 * 1024.0),
          absl_inc_ns / compact_inc_ns,
          100.0 * (1.0 - static_cast<double>(compact_inc_map_bytes) /
                             absl_inc_map_bytes));
    } else {
      printf(
          "  hwy::CompactBTreeMap : %6.2f ns/op (%5.1f B/pair, %5.2f MB) -> "
          "%.2fx vs absl (%.1f%% larger)\n",
          compact_inc_ns,
          static_cast<double>(compact_inc_map_bytes) /
              (compact_prebuilt_map.size() + kNumIncremental),
          compact_inc_map_bytes / (1024.0 * 1024.0),
          absl_inc_ns / compact_inc_ns,
          100.0 *
              (static_cast<double>(compact_inc_map_bytes) / absl_inc_map_bytes -
               1.0));
    }

    const double std_dec_ns = (dec_std_1 - dec_std_0) * 1e9 / kNumIncremental;
    const double absl_dec_ns =
        (dec_absl_1 - dec_absl_0) * 1e9 / kNumIncremental;
    const double hwy_dec_ns = (dec_hwy_1 - dec_hwy_0) * 1e9 / kNumIncremental;
    const double compact_dec_ns =
        (dec_compact_1 - dec_compact_0) * 1e9 / kNumIncremental;

    printf(
        "\nIncremental Deletions Latency & Memory on Pre-Built Map (%zu "
        "random pairs on %zu-pair map -> remaining pairs: %zu):\n",
        kNumIncremental, num_keys, std_prebuilt_map.size());
    printf("  std::map             : %6.2f ns/op (%5.1f B/pair, %5.2f MB)\n",
           std_dec_ns,
           static_cast<double>(std_inc_map_del_bytes) / std_prebuilt_map.size(),
           std_inc_map_del_bytes / (1024.0 * 1024.0));
    printf(
        "  absl::btree_map      : %6.2f ns/op (%5.1f B/pair, %5.2f MB)\n",
        absl_dec_ns,
        static_cast<double>(absl_inc_map_del_bytes) / absl_prebuilt_map.size(),
        absl_inc_map_del_bytes / (1024.0 * 1024.0));
    printf(
        "  hwy::BTreeMap        : %6.2f ns/op (%5.1f B/pair, %5.2f MB) -> "
        "%.2fx speedup vs absl\n",
        hwy_dec_ns,
        static_cast<double>(hwy_inc_map_del_bytes) / hwy_prebuilt_map.size(),
        hwy_inc_map_del_bytes / (1024.0 * 1024.0), absl_dec_ns / hwy_dec_ns);
    if (compact_inc_map_del_bytes <= absl_inc_map_del_bytes) {
      printf(
          "  hwy::CompactBTreeMap : %6.2f ns/op (%5.1f B/pair, %5.2f MB) -> "
          "%.2fx vs absl (%.1f%% smaller!)\n",
          compact_dec_ns,
          static_cast<double>(compact_inc_map_del_bytes) /
              compact_prebuilt_map.size(),
          compact_inc_map_del_bytes / (1024.0 * 1024.0),
          absl_dec_ns / compact_dec_ns,
          100.0 * (1.0 - static_cast<double>(compact_inc_map_del_bytes) /
                             absl_inc_map_del_bytes));
    } else {
      printf(
          "  hwy::CompactBTreeMap : %6.2f ns/op (%5.1f B/pair, %5.2f MB) -> "
          "%.2fx vs absl (%.1f%% larger)\n",
          compact_dec_ns,
          static_cast<double>(compact_inc_map_del_bytes) /
              compact_prebuilt_map.size(),
          compact_inc_map_del_bytes / (1024.0 * 1024.0),
          absl_dec_ns / compact_dec_ns,
          100.0 * (static_cast<double>(compact_inc_map_del_bytes) /
                       absl_inc_map_del_bytes -
                   1.0));
    }
  }

  HWY_ASSERT(hwy_hits == absl_hits);
  HWY_ASSERT(compact_hits == absl_hits);
  HWY_ASSERT(batch_hits == absl_hits);
  HWY_ASSERT(compact_batch_hits == absl_hits);
  HWY_ASSERT(hwy_lb_sum == absl_lb_sum);
  HWY_ASSERT(compact_lb_sum == absl_lb_sum);
  HWY_ASSERT(batch_lb_sum == absl_lb_sum);
  HWY_ASSERT(compact_batch_lb_sum == absl_lb_sum);
  HWY_ASSERT(hwy_dyn_map.size() == absl_dyn_map.size());
  HWY_ASSERT(compact_dyn_map.size() == absl_dyn_map.size());
}

template <typename KeyT, typename ValueT>
void RunWorstCaseMapBenchmarkSuite(size_t num_keys) {
  printf("\n===============================================================\n");
  printf(
      "  Worst-Case B-Tree Map Benchmark (%zu-bit Key, %zu-bit Value, N = %zu "
      "pairs, %s)\n",
      sizeof(KeyT) * 8, sizeof(ValueT) * 8, num_keys,
      hwy::TargetName(HWY_TARGET));
  printf("  Key Distribution: Uncompressible Uniform %zu-bit Random Keys\n",
         sizeof(KeyT) * 8);
  printf("  Query Pattern   : 100%% Lookup Misses (Disjoint Range)\n");
  printf("===============================================================\n");

  absl::BitGen bitgen;
  std::vector<KeyT> keys;
  keys.reserve(num_keys);
  for (size_t i = 0; i < num_keys; ++i) {
    keys.push_back(static_cast<KeyT>(
        absl::Uniform<KeyT>(bitgen, 0, std::numeric_limits<KeyT>::max() / 2) *
        2));
  }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  num_keys = keys.size();

  std::vector<ValueT> vals;
  std::vector<std::pair<KeyT, ValueT>> kv_pairs;
  vals.reserve(num_keys);
  kv_pairs.reserve(num_keys);
  for (size_t i = 0; i < num_keys; ++i) {
    ValueT v = static_cast<ValueT>(keys[i] * 3 + 7);
    vals.push_back(v);
    kv_pairs.push_back({keys[i], v});
  }

  // 1. Build Containers
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
  auto hwy_map = BTreeMap<KeyT, ValueT>::Build(keys.data(), vals.data(),
                                               keys.size(), 1.0f);
  const double end_hwy = hwy::platform::Now();
  const size_t hwy_bytes =
      GetAllocatedBytes(hwy_before, hwy_map.AllocatedBytes());

  const size_t compact_before = AllocatedBefore();
  const double start_compact = hwy::platform::Now();
  auto compact_map = CompactBTreeMap<KeyT, ValueT>::Build(
      keys.data(), vals.data(), keys.size(), 1.0);
  const double end_compact = hwy::platform::Now();
  const size_t compact_bytes =
      GetAllocatedBytes(compact_before, compact_map.AllocatedBytes());

  printf("Worst-Case Build Time:\n");
  printf("  std::map             : %8.2f ms\n", (end_std - start_std) * 1000.0);
  printf("  absl::btree_map      : %8.2f ms\n",
         (end_absl - start_absl) * 1000.0);
  printf("  hwy::BTreeMap        : %8.2f ms (%.1fx faster than absl)\n",
         (end_hwy - start_hwy) * 1000.0,
         (end_absl - start_absl) / (end_hwy - start_hwy + 1e-6));
  printf("  hwy::CompactBTreeMap : %8.2f ms (%.1fx faster than absl)\n",
         (end_compact - start_compact) * 1000.0,
         (end_absl - start_absl) / (end_compact - start_compact + 1e-6));

  printf("\nWorst-Case Memory Footprint (Uncompressible Raw Mode):\n");
  printf("  std::map             : %5.2f MB (%5.1f B/pair)\n",
         std_bytes / (1024.0 * 1024.0),
         static_cast<double>(std_bytes) / num_keys);
  printf("  absl::btree_map      : %5.2f MB (%5.1f B/pair)\n",
         absl_bytes / (1024.0 * 1024.0),
         static_cast<double>(absl_bytes) / num_keys);
  printf("  hwy::BTreeMap        : %5.2f MB (%5.1f B/pair)\n",
         hwy_bytes / (1024.0 * 1024.0),
         static_cast<double>(hwy_bytes) / num_keys);
  if (compact_bytes <= absl_bytes) {
    printf(
        "  hwy::CompactBTreeMap : %5.2f MB (%5.1f B/pair) -> %.1f%% smaller "
        "than absl!\n",
        compact_bytes / (1024.0 * 1024.0),
        static_cast<double>(compact_bytes) / num_keys,
        100.0 * (1.0 - static_cast<double>(compact_bytes) / absl_bytes));
  } else {
    printf(
        "  hwy::CompactBTreeMap : %5.2f MB (%5.1f B/pair) -> %.1f%% larger "
        "than absl\n",
        compact_bytes / (1024.0 * 1024.0),
        static_cast<double>(compact_bytes) / num_keys,
        100.0 * (static_cast<double>(compact_bytes) / absl_bytes - 1.0));
  }

  // 2. Worst-case 100% Miss Point Lookups (Odd keys vs Even keys in map)
  constexpr size_t kNumQueries = 1000000;
  std::vector<KeyT> miss_queries;
  miss_queries.reserve(kNumQueries);
  for (size_t i = 0; i < kNumQueries; ++i) {
    miss_queries.push_back(static_cast<KeyT>(
        absl::Uniform<KeyT>(bitgen, 0, std::numeric_limits<KeyT>::max() / 2) *
            2 +
        1));
  }

  uint64_t std_hits = 0, absl_hits = 0, hwy_hits = 0, compact_hits = 0;
  const double t0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = std_map.find(miss_queries[i]);
    if (it != std_map.end()) std_hits++;
  }
  hwy::PreventElision(std_hits);
  const double t1 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    auto it = absl_map.find(miss_queries[i]);
    if (it != absl_map.end()) absl_hits++;
  }
  hwy::PreventElision(absl_hits);
  const double t2 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    const ValueT* ptr = hwy_map.FindValue(miss_queries[i]);
    if (ptr != nullptr) hwy_hits++;
  }
  hwy::PreventElision(hwy_hits);
  const double t3 = hwy::platform::Now();

  for (size_t i = 0; i < kNumQueries; ++i) {
    const ValueT* ptr = compact_map.FindValue(miss_queries[i]);
    if (ptr != nullptr) compact_hits++;
  }
  hwy::PreventElision(compact_hits);
  const double t4 = hwy::platform::Now();

  const double std_miss_ns = (t1 - t0) * 1e9 / kNumQueries;
  const double absl_miss_ns = (t2 - t1) * 1e9 / kNumQueries;
  const double hwy_miss_ns = (t3 - t2) * 1e9 / kNumQueries;
  const double compact_miss_ns = (t4 - t3) * 1e9 / kNumQueries;

  printf("\nWorst-Case Point Lookup Miss Latency (100%% Key Misses):\n");
  printf("  std::map             : %6.2f ns/op (%6.2f Mops/s)\n", std_miss_ns,
         1000.0 / std_miss_ns);
  printf("  absl::btree_map      : %6.2f ns/op (%6.2f Mops/s)\n", absl_miss_ns,
         1000.0 / absl_miss_ns);
  printf(
      "  hwy::BTreeMap        : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
      hwy_miss_ns, 1000.0 / hwy_miss_ns, absl_miss_ns / hwy_miss_ns);
  printf(
      "  hwy::CompactBTreeMap : %6.2f ns/op (%6.2f Mops/s) -> %.2fx speedup!\n",
      compact_miss_ns, 1000.0 / compact_miss_ns,
      absl_miss_ns / compact_miss_ns);

  HWY_ASSERT(std_hits == 0);
  HWY_ASSERT(absl_hits == 0);
  HWY_ASSERT(hwy_hits == 0);
  HWY_ASSERT(compact_hits == 0);
}

HWY_NOINLINE void BenchmarkAll() {
  printf("\n###############################################################\n");
  printf("  32-bit Key Set Benchmarks (BTreeSet<uint32_t>)\n");
  printf("###############################################################\n");
  RunBenchmarkSuite<uint32_t>(10000);    // 10K keys (L1/L2 Cache)
  RunBenchmarkSuite<uint32_t>(100000);   // 100K keys (L3 Cache)
  RunBenchmarkSuite<uint32_t>(1000000);  // 1M keys (RAM)

  printf("\n###############################################################\n");
  printf("  Worst-Case Uncompressible 32-bit Set Benchmarks\n");
  printf("###############################################################\n");
  // 100K uncompressible 32-bit keys
  RunWorstCaseBenchmarkSuite<uint32_t>(100000);

  printf("\n###############################################################\n");
  printf("  64-bit Key Set Benchmarks (BTreeSet<uint64_t>)\n");
  printf("###############################################################\n");
  RunBenchmarkSuite<uint64_t>(10000);    // 10K keys (L1/L2 Cache)
  RunBenchmarkSuite<uint64_t>(100000);   // 100K keys (L3 Cache)
  RunBenchmarkSuite<uint64_t>(1000000);  // 1M keys (RAM)

  printf("\n###############################################################\n");
  printf("  Worst-Case Uncompressible 64-bit Set Benchmarks\n");
  printf("###############################################################\n");
  // 100K uncompressible 64-bit keys
  RunWorstCaseBenchmarkSuite<uint64_t>(100000);

  printf("\n###############################################################\n");
  printf("  32-bit Key Map Benchmarks (BTreeMap<uint32_t, uint64_t>)\n");
  printf("###############################################################\n");
  RunMapBenchmarkSuite<uint32_t, uint64_t>(10000);    // 10K keys (L1/L2 Cache)
  RunMapBenchmarkSuite<uint32_t, uint64_t>(100000);   // 100K keys (L3 Cache)
  RunMapBenchmarkSuite<uint32_t, uint64_t>(1000000);  // 1M keys (RAM)

  printf("\n###############################################################\n");
  printf("  Worst-Case Uncompressible 32-bit Map Benchmarks\n");
  printf("###############################################################\n");
  // 100K uncompressible 32-bit map pairs
  RunWorstCaseMapBenchmarkSuite<uint32_t, uint64_t>(100000);

  printf("\n###############################################################\n");
  printf("  64-bit Key Map Benchmarks (BTreeMap<uint64_t, double>)\n");
  printf("###############################################################\n");
  RunMapBenchmarkSuite<uint64_t, double>(10000);    // 10K keys (L1/L2 Cache)
  RunMapBenchmarkSuite<uint64_t, double>(100000);   // 100K keys (L3 Cache)
  RunMapBenchmarkSuite<uint64_t, double>(1000000);  // 1M keys (RAM)

  printf("\n###############################################################\n");
  printf("  Worst-Case Uncompressible 64-bit Map Benchmarks\n");
  printf("###############################################################\n");
  // 100K uncompressible 64-bit map pairs
  RunWorstCaseMapBenchmarkSuite<uint64_t, double>(100000);
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
