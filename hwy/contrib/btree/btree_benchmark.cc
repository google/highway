// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
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

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <type_traits>
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

// Set to true to sweep through multiple scale tiers
HWY_INLINE_VAR constexpr bool kSweepSizes = false;

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
  printf(
      "\n======================================================================"
      "==================================================\n");
  printf(
      "  Latency: hwy::btree_set<%s> (N = %zu keys, %s) vs absl::btree_set vs "
      "std::set\n",
      sizeof(KeyT) == 4 ? "uint32_t" : "uint64_t", num_keys,
      hwy::TargetName(HWY_TARGET));
  printf(
      "========================================================================"
      "================================================\n");
  printf("%-17s %11s %10s %10s %10s %10s %12s %12s %18s\n", "Container",
         "Build (ms)", "Find (ns)", "LB (ns)", "UB (ns)", "Iter Walk",
         "Batch Find", "Batch LB", "Dyn Ins/Del (ns)");
  printf(
      "------------------------------------------------------------------------"
      "------------------------------------------------\n");

  absl::BitGen bitgen;
  std::vector<KeyT> keys;
  keys.reserve(num_keys);
  for (size_t i = 0; i < num_keys; ++i) {
    keys.push_back(static_cast<KeyT>((i + 1) * 10));
  }
  std::shuffle(keys.begin(), keys.end(), bitgen);

  // 1. Build Containers
  const double start_std = hwy::platform::Now();
  std::set<KeyT> std_tree(keys.begin(), keys.end());
  const double end_std = hwy::platform::Now();

  const double start_absl = hwy::platform::Now();
  absl::btree_set<KeyT> absl_tree(keys.begin(), keys.end());
  const double end_absl = hwy::platform::Now();

  const double start_hwy = hwy::platform::Now();
  auto hwy_tree = BTreeSet<KeyT>::Build(keys.data(), keys.size());
  const double end_hwy = hwy::platform::Now();

  const double std_build_ms = (end_std - start_std) * 1000.0;
  const double absl_build_ms = (end_absl - start_absl) * 1000.0;
  const double hwy_build_ms = (end_hwy - start_hwy) * 1000.0;

  // 2. Generate Random Query Keys
  constexpr size_t kNumQueries = 1000000;
  std::vector<KeyT> queries;
  queries.reserve(kNumQueries);
  for (size_t i = 0; i < kNumQueries; ++i) {
    queries.push_back(
        static_cast<KeyT>(absl::Uniform<KeyT>(bitgen, 0, (num_keys + 1) * 10)));
  }

  // 3. Point Lookups (contains / find)
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
    hwy_hits += hwy_tree.contains(queries[i]);
  }
  hwy::PreventElision(hwy_hits);
  const double t3 = hwy::platform::Now();

  const double std_find_ns = (t1 - t0) * 1e9 / kNumQueries;
  const double absl_find_ns = (t2 - t1) * 1e9 / kNumQueries;
  const double hwy_find_ns = (t3 - t2) * 1e9 / kNumQueries;

  // 4. Ordered Range Queries (lower_bound)
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
    auto it = hwy_tree.lower_bound(queries[i]);
    if (it != hwy_tree.end()) hwy_lb_sum += *it;
  }
  hwy::PreventElision(hwy_lb_sum);
  const double r3 = hwy::platform::Now();

  const double std_lb_ns = (r1 - r0) * 1e9 / kNumQueries;
  const double absl_lb_ns = (r2 - r1) * 1e9 / kNumQueries;
  const double hwy_lb_ns = (r3 - r2) * 1e9 / kNumQueries;

  // 4b. Ordered Range Queries (upper_bound)
  uint64_t hwy_ub_sum = 0, absl_ub_sum = 0, std_ub_sum = 0;

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

  const double std_ub_ns = (u1 - u0) * 1e9 / kNumQueries;
  const double absl_ub_ns = (u2 - u1) * 1e9 / kNumQueries;
  const double hwy_ub_ns = (u3 - u2) * 1e9 / kNumQueries;

  // 4c. Full Ordered Iteration (begin -> end)
  const size_t kIterPasses =
      std::max<size_t>(10, (10 * kNumQueries) / num_keys);
  const size_t kTotalIterSteps = kIterPasses * num_keys;
  uint64_t hwy_iter_sum = 0, absl_iter_sum = 0, std_iter_sum = 0;

  const double it0 = hwy::platform::Now();
  for (size_t p = 0; p < kIterPasses; ++p) {
    for (KeyT k : std_tree) {
      std_iter_sum += k;
    }
  }
  hwy::PreventElision(std_iter_sum);
  const double it1 = hwy::platform::Now();

  for (size_t p = 0; p < kIterPasses; ++p) {
    for (KeyT k : absl_tree) {
      absl_iter_sum += k;
    }
  }
  hwy::PreventElision(absl_iter_sum);
  const double it2 = hwy::platform::Now();

  for (size_t p = 0; p < kIterPasses; ++p) {
    for (KeyT k : hwy_tree) {
      hwy_iter_sum += k;
    }
  }
  hwy::PreventElision(hwy_iter_sum);
  const double it3 = hwy::platform::Now();

  const double std_iter_ns = (it1 - it0) * 1e9 / kTotalIterSteps;
  const double absl_iter_ns = (it2 - it1) * 1e9 / kTotalIterSteps;
  const double hwy_iter_ns = (it3 - it2) * 1e9 / kTotalIterSteps;

  // 5. Batch Point Lookups (ContainsBatch)
  auto batch_found = std::make_unique<bool[]>(kNumQueries);
  const double b0 = hwy::platform::Now();
  hwy_tree.ContainsBatch(queries.data(), kNumQueries, batch_found.get());
  const double b1 = hwy::platform::Now();

  uint64_t batch_hits = 0;
  for (size_t i = 0; i < kNumQueries; ++i) {
    batch_hits += batch_found[i];
  }
  hwy::PreventElision(batch_hits);
  const double hwy_batch_find_ns = (b1 - b0) * 1e9 / kNumQueries;

  // 6. Batch LowerBound Queries (LowerBoundBatch)
  std::vector<typename BTreeSet<KeyT>::const_iterator> batch_lb_results(
      kNumQueries);
  const double blb0 = hwy::platform::Now();
  hwy_tree.LowerBoundBatch(queries.data(), kNumQueries,
                           batch_lb_results.data());
  const double blb1 = hwy::platform::Now();

  uint64_t batch_lb_sum = 0;
  for (size_t i = 0; i < kNumQueries; ++i) {
    if (batch_lb_results[i] != hwy_tree.end()) {
      batch_lb_sum += *batch_lb_results[i];
    }
  }
  hwy::PreventElision(batch_lb_sum);
  const double hwy_batch_lb_ns = (blb1 - blb0) * 1e9 / kNumQueries;

  // 7. Dynamic Insertions & Deletions on Empty Tree
  const size_t kNumMutations = std::min(num_keys, static_cast<size_t>(100000));
  const size_t kNumErases = kNumMutations / 2;
  std::vector<KeyT> mutation_keys;
  mutation_keys.reserve(kNumMutations);
  for (size_t i = 0; i < kNumMutations; ++i) {
    mutation_keys.push_back(static_cast<KeyT>(
        absl::Uniform<uint64_t>(bitgen, 0, (num_keys + 1) * 20)));
  }

  std::set<KeyT> std_dyn_set;
  const double mi_std_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumMutations; ++i) {
    std_dyn_set.insert(mutation_keys[i]);
  }
  const double mi_std_1 = hwy::platform::Now();

  const double me_std_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumErases; ++i) {
    std_dyn_set.erase(mutation_keys[i]);
  }
  const double me_std_1 = hwy::platform::Now();

  absl::btree_set<KeyT> absl_dyn_set;
  const double mi_absl_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumMutations; ++i) {
    absl_dyn_set.insert(mutation_keys[i]);
  }
  const double mi_absl_1 = hwy::platform::Now();
  const size_t dyn_unique_keys = absl_dyn_set.size();

  const double me_absl_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumErases; ++i) {
    absl_dyn_set.erase(mutation_keys[i]);
  }
  const double me_absl_1 = hwy::platform::Now();

  BTreeSet<KeyT> hwy_dyn_set;
  const double mi_hwy_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumMutations; ++i) {
    hwy_dyn_set.insert(mutation_keys[i]);
  }
  const double mi_hwy_1 = hwy::platform::Now();
  HWY_ASSERT(hwy_dyn_set.size() == dyn_unique_keys);

  const double me_hwy_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumErases; ++i) {
    hwy_dyn_set.erase(mutation_keys[i]);
  }
  const double me_hwy_1 = hwy::platform::Now();

  const double std_ins_ns = (mi_std_1 - mi_std_0) * 1e9 / kNumMutations;
  const double absl_ins_ns = (mi_absl_1 - mi_absl_0) * 1e9 / kNumMutations;
  const double hwy_ins_ns = (mi_hwy_1 - mi_hwy_0) * 1e9 / kNumMutations;

  const double std_erase_ns = (me_std_1 - me_std_0) * 1e9 / kNumErases;
  const double absl_erase_ns = (me_absl_1 - me_absl_0) * 1e9 / kNumErases;
  const double hwy_erase_ns = (me_hwy_1 - me_hwy_0) * 1e9 / kNumErases;

  // Print Formatted Rows
  char std_ins_del[32], absl_ins_del[32], hwy_ins_del[32];
  snprintf(std_ins_del, sizeof(std_ins_del), "%5.1f / %5.1f", std_ins_ns,
           std_erase_ns);
  snprintf(absl_ins_del, sizeof(absl_ins_del), "%5.1f / %5.1f", absl_ins_ns,
           absl_erase_ns);
  snprintf(hwy_ins_del, sizeof(hwy_ins_del), "%5.1f / %5.1f", hwy_ins_ns,
           hwy_erase_ns);

  printf(
      "%-17s %8.1f ms %7.1f ns %7.1f ns %7.1f ns %7.2f ns %9.1f ns "
      "%9.1f ns %18s\n",
      "std::set", std_build_ms, std_find_ns, std_lb_ns, std_ub_ns, std_iter_ns,
      std_find_ns, std_lb_ns, std_ins_del);
  printf(
      "%-17s %8.1f ms %7.1f ns %7.1f ns %7.1f ns %7.2f ns %9.1f ns "
      "%9.1f ns %18s\n",
      "absl::btree_set", absl_build_ms, absl_find_ns, absl_lb_ns, absl_ub_ns,
      absl_iter_ns, absl_find_ns, absl_lb_ns, absl_ins_del);
  printf(
      "%-17s %8.1f ms %7.1f ns %7.1f ns %7.1f ns %7.2f ns %9.1f ns "
      "%9.1f ns %18s\n",
      "hwy::btree_set", hwy_build_ms, hwy_find_ns, hwy_lb_ns, hwy_ub_ns,
      hwy_iter_ns, hwy_batch_find_ns, hwy_batch_lb_ns, hwy_ins_del);

  char build_sp[32], find_sp[32], lb_sp[32], ub_sp[32], iter_sp[32],
      bfind_sp[32], blb_sp[32], ins_del_sp[32];
  snprintf(build_sp, sizeof(build_sp), "(%.1fx)",
           absl_build_ms / (hwy_build_ms + 1e-6));
  snprintf(find_sp, sizeof(find_sp), "(%.1fx)",
           absl_find_ns / (hwy_find_ns + 1e-6));
  snprintf(lb_sp, sizeof(lb_sp), "(%.1fx)", absl_lb_ns / (hwy_lb_ns + 1e-6));
  snprintf(ub_sp, sizeof(ub_sp), "(%.1fx)", absl_ub_ns / (hwy_ub_ns + 1e-6));
  snprintf(iter_sp, sizeof(iter_sp), "(%.1fx)",
           absl_iter_ns / (hwy_iter_ns + 1e-6));
  snprintf(bfind_sp, sizeof(bfind_sp), "(%.1fx)",
           absl_find_ns / (hwy_batch_find_ns + 1e-6));
  snprintf(blb_sp, sizeof(blb_sp), "(%.1fx)",
           absl_lb_ns / (hwy_batch_lb_ns + 1e-6));
  snprintf(ins_del_sp, sizeof(ins_del_sp), "(%.1fx / %.1fx)",
           absl_ins_ns / (hwy_ins_ns + 1e-6),
           absl_erase_ns / (hwy_erase_ns + 1e-6));

  printf("%-17s %11s %10s %10s %10s %10s %12s %12s %18s\n", "  faster vs absl",
         build_sp, find_sp, lb_sp, ub_sp, iter_sp, bfind_sp, blb_sp,
         ins_del_sp);
  printf(
      "========================================================================"
      "================================================\n");

  HWY_ASSERT(std_hits == absl_hits);
  HWY_ASSERT(hwy_hits == absl_hits);
  HWY_ASSERT(batch_hits == absl_hits);
  HWY_ASSERT(std_lb_sum == absl_lb_sum);
  HWY_ASSERT(hwy_lb_sum == absl_lb_sum);
  HWY_ASSERT(batch_lb_sum == absl_lb_sum);
  HWY_ASSERT(std_iter_sum == absl_iter_sum);
  HWY_ASSERT(hwy_iter_sum == absl_iter_sum);
}

template <typename KeyT, typename ValueT>
void RunMapBenchmarkSuite(size_t num_keys) {
  printf(
      "\n======================================================================"
      "==================================================\n");
  printf(
      "  Latency: hwy::btree_map<%s, %s> (N = %zu pairs, %s) vs "
      "absl::btree_map vs std::map\n",
      sizeof(KeyT) == 4 ? "uint32_t" : "uint64_t",
      sizeof(ValueT) == 8
          ? (std::is_floating_point_v<ValueT> ? "double" : "uint64_t")
          : "uint32_t",
      num_keys, hwy::TargetName(HWY_TARGET));
  printf(
      "========================================================================"
      "================================================\n");
  printf("%-17s %11s %10s %10s %10s %10s %12s %12s %18s\n", "Container",
         "Build (ms)", "Find (ns)", "LB (ns)", "UB (ns)", "Iter Walk",
         "Batch Find", "Batch LB", "Dyn Ins/Del (ns)");
  printf(
      "------------------------------------------------------------------------"
      "------------------------------------------------\n");

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
    kv_pairs.push_back({k, v});
  }
  std::shuffle(kv_pairs.begin(), kv_pairs.end(), bitgen);
  for (const auto& [k, v] : kv_pairs) {
    keys.push_back(k);
    vals.push_back(v);
  }

  // 1. Build Containers
  const double start_std = hwy::platform::Now();
  std::map<KeyT, ValueT> std_map(kv_pairs.begin(), kv_pairs.end());
  const double end_std = hwy::platform::Now();

  const double start_absl = hwy::platform::Now();
  absl::btree_map<KeyT, ValueT> absl_map(kv_pairs.begin(), kv_pairs.end());
  const double end_absl = hwy::platform::Now();

  const double start_hwy = hwy::platform::Now();
  auto hwy_map =
      BTreeMap<KeyT, ValueT>::Build(keys.data(), vals.data(), keys.size());
  const double end_hwy = hwy::platform::Now();

  const double std_build_ms = (end_std - start_std) * 1000.0;
  const double absl_build_ms = (end_absl - start_absl) * 1000.0;
  const double hwy_build_ms = (end_hwy - start_hwy) * 1000.0;

  // 2. Generate Random Query Keys
  constexpr size_t kNumQueries = 1000000;
  std::vector<KeyT> queries;
  queries.reserve(kNumQueries);
  for (size_t i = 0; i < kNumQueries; ++i) {
    queries.push_back(static_cast<KeyT>(
        absl::Uniform<uint64_t>(bitgen, 0, (num_keys + 1) * 10)));
  }

  // 3. Point Lookups (FindValue / find)
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

  const double std_find_ns = (t1 - t0) * 1e9 / kNumQueries;
  const double absl_find_ns = (t2 - t1) * 1e9 / kNumQueries;
  const double hwy_find_ns = (t3 - t2) * 1e9 / kNumQueries;

  // 4. Ordered Range Queries (lower_bound)
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

  // 4b. Ordered Range Queries (upper_bound)
  uint64_t hwy_ub_sum = 0, absl_ub_sum = 0, std_ub_sum = 0;

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

  const double std_ub_ns = (u1 - u0) * 1e9 / kNumQueries;
  const double absl_ub_ns = (u2 - u1) * 1e9 / kNumQueries;
  const double hwy_ub_ns = (u3 - u2) * 1e9 / kNumQueries;

  // 4c. Full Ordered Iteration (begin -> end)
  const size_t kIterPasses =
      std::max<size_t>(10, (10 * kNumQueries) / num_keys);
  const size_t kTotalIterSteps = kIterPasses * num_keys;
  uint64_t hwy_iter_sum = 0, absl_iter_sum = 0, std_iter_sum = 0;

  const double it0 = hwy::platform::Now();
  for (size_t p = 0; p < kIterPasses; ++p) {
    for (const auto& kv : std_map) {
      std_iter_sum +=
          static_cast<uint64_t>(kv.first) + static_cast<uint64_t>(kv.second);
    }
  }
  hwy::PreventElision(std_iter_sum);
  const double it1 = hwy::platform::Now();

  for (size_t p = 0; p < kIterPasses; ++p) {
    for (const auto& kv : absl_map) {
      absl_iter_sum +=
          static_cast<uint64_t>(kv.first) + static_cast<uint64_t>(kv.second);
    }
  }
  hwy::PreventElision(absl_iter_sum);
  const double it2 = hwy::platform::Now();

  for (size_t p = 0; p < kIterPasses; ++p) {
    for (const auto& kv : hwy_map) {
      hwy_iter_sum +=
          static_cast<uint64_t>(kv.first) + static_cast<uint64_t>(kv.second);
    }
  }
  hwy::PreventElision(hwy_iter_sum);
  const double it3 = hwy::platform::Now();

  const double std_iter_ns = (it1 - it0) * 1e9 / kTotalIterSteps;
  const double absl_iter_ns = (it2 - it1) * 1e9 / kTotalIterSteps;
  const double hwy_iter_ns = (it3 - it2) * 1e9 / kTotalIterSteps;

  // 5. Batch Value Lookups (LookupBatch)
  auto batch_found = std::make_unique<bool[]>(kNumQueries);
  std::vector<ValueT> batch_values(kNumQueries);
  const double b0 = hwy::platform::Now();
  hwy_map.LookupBatch(queries.data(), kNumQueries, batch_values.data(),
                      batch_found.get());
  const double b1 = hwy::platform::Now();

  uint64_t batch_val_sum = 0;
  for (size_t i = 0; i < kNumQueries; ++i) {
    if (batch_found[i]) batch_val_sum += static_cast<uint64_t>(batch_values[i]);
  }
  hwy::PreventElision(batch_val_sum);
  const double hwy_batch_find_ns = (b1 - b0) * 1e9 / kNumQueries;

  // 6. Batch LowerBound Queries (LowerBoundBatch)
  std::vector<typename BTreeMap<KeyT, ValueT>::const_iterator> batch_lb_results(
      kNumQueries);
  const double blb0 = hwy::platform::Now();
  hwy_map.LowerBoundBatch(queries.data(), kNumQueries, batch_lb_results.data());
  const double blb1 = hwy::platform::Now();

  uint64_t batch_lb_sum = 0;
  for (size_t i = 0; i < kNumQueries; ++i) {
    if (batch_lb_results[i] != hwy_map.end()) {
      batch_lb_sum += static_cast<uint64_t>(batch_lb_results[i]->second);
    }
  }
  hwy::PreventElision(batch_lb_sum);
  const double hwy_batch_lb_ns = (blb1 - blb0) * 1e9 / kNumQueries;

  // 7. Dynamic Insertions & Deletions on Empty Map
  const size_t kNumMutations = std::min(num_keys, static_cast<size_t>(100000));
  const size_t kNumErases = kNumMutations / 2;
  std::vector<std::pair<KeyT, ValueT>> mutation_pairs;
  mutation_pairs.reserve(kNumMutations);
  for (size_t i = 0; i < kNumMutations; ++i) {
    mutation_pairs.push_back({static_cast<KeyT>(absl::Uniform<uint64_t>(
                                  bitgen, 0, (num_keys + 1) * 20)),
                              static_cast<ValueT>(absl::Uniform<uint64_t>(
                                  bitgen, 1, (num_keys + 1) * 200))});
  }

  std::map<KeyT, ValueT> std_dyn_map;
  const double mi_std_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumMutations; ++i) {
    std_dyn_map.insert(mutation_pairs[i]);
  }
  const double mi_std_1 = hwy::platform::Now();

  const double me_std_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumErases; ++i) {
    std_dyn_map.erase(mutation_pairs[i].first);
  }
  const double me_std_1 = hwy::platform::Now();

  absl::btree_map<KeyT, ValueT> absl_dyn_map;
  const double mi_absl_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumMutations; ++i) {
    absl_dyn_map.insert(mutation_pairs[i]);
  }
  const double mi_absl_1 = hwy::platform::Now();
  const size_t dyn_unique_pairs = absl_dyn_map.size();

  const double me_absl_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumErases; ++i) {
    absl_dyn_map.erase(mutation_pairs[i].first);
  }
  const double me_absl_1 = hwy::platform::Now();

  BTreeMap<KeyT, ValueT> hwy_dyn_map;
  const double mi_hwy_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumMutations; ++i) {
    hwy_dyn_map.insert(mutation_pairs[i]);
  }
  const double mi_hwy_1 = hwy::platform::Now();
  HWY_ASSERT(hwy_dyn_map.size() == dyn_unique_pairs);

  const double me_hwy_0 = hwy::platform::Now();
  for (size_t i = 0; i < kNumErases; ++i) {
    hwy_dyn_map.erase(mutation_pairs[i].first);
  }
  const double me_hwy_1 = hwy::platform::Now();

  const double std_ins_ns = (mi_std_1 - mi_std_0) * 1e9 / kNumMutations;
  const double absl_ins_ns = (mi_absl_1 - mi_absl_0) * 1e9 / kNumMutations;
  const double hwy_ins_ns = (mi_hwy_1 - mi_hwy_0) * 1e9 / kNumMutations;

  const double std_erase_ns = (me_std_1 - me_std_0) * 1e9 / kNumErases;
  const double absl_erase_ns = (me_absl_1 - me_absl_0) * 1e9 / kNumErases;
  const double hwy_erase_ns = (me_hwy_1 - me_hwy_0) * 1e9 / kNumErases;

  char std_ins_del[32], absl_ins_del[32], hwy_ins_del[32];
  snprintf(std_ins_del, sizeof(std_ins_del), "%5.1f / %5.1f", std_ins_ns,
           std_erase_ns);
  snprintf(absl_ins_del, sizeof(absl_ins_del), "%5.1f / %5.1f", absl_ins_ns,
           absl_erase_ns);
  snprintf(hwy_ins_del, sizeof(hwy_ins_del), "%5.1f / %5.1f", hwy_ins_ns,
           hwy_erase_ns);

  printf(
      "%-17s %8.1f ms %7.1f ns %7.1f ns %7.1f ns %7.2f ns %9.1f ns "
      "%9.1f ns %18s\n",
      "std::map", std_build_ms, std_find_ns, std_lb_ns, std_ub_ns, std_iter_ns,
      std_find_ns, std_lb_ns, std_ins_del);
  printf(
      "%-17s %8.1f ms %7.1f ns %7.1f ns %7.1f ns %7.2f ns %9.1f ns "
      "%9.1f ns %18s\n",
      "absl::btree_map", absl_build_ms, absl_find_ns, absl_lb_ns, absl_ub_ns,
      absl_iter_ns, absl_find_ns, absl_lb_ns, absl_ins_del);
  printf(
      "%-17s %8.1f ms %7.1f ns %7.1f ns %7.1f ns %7.2f ns %9.1f ns "
      "%9.1f ns %18s\n",
      "hwy::btree_map", hwy_build_ms, hwy_find_ns, hwy_lb_ns, hwy_ub_ns,
      hwy_iter_ns, hwy_batch_find_ns, hwy_batch_lb_ns, hwy_ins_del);

  char build_sp[32], find_sp[32], lb_sp[32], ub_sp[32], iter_sp[32],
      bfind_sp[32], blb_sp[32], ins_del_sp[32];
  snprintf(build_sp, sizeof(build_sp), "(%.1fx)",
           absl_build_ms / (hwy_build_ms + 1e-6));
  snprintf(find_sp, sizeof(find_sp), "(%.1fx)",
           absl_find_ns / (hwy_find_ns + 1e-6));
  snprintf(lb_sp, sizeof(lb_sp), "(%.1fx)", absl_lb_ns / (hwy_lb_ns + 1e-6));
  snprintf(ub_sp, sizeof(ub_sp), "(%.1fx)", absl_ub_ns / (hwy_ub_ns + 1e-6));
  snprintf(iter_sp, sizeof(iter_sp), "(%.1fx)",
           absl_iter_ns / (hwy_iter_ns + 1e-6));
  snprintf(bfind_sp, sizeof(bfind_sp), "(%.1fx)",
           absl_find_ns / (hwy_batch_find_ns + 1e-6));
  snprintf(blb_sp, sizeof(blb_sp), "(%.1fx)",
           absl_lb_ns / (hwy_batch_lb_ns + 1e-6));
  snprintf(ins_del_sp, sizeof(ins_del_sp), "(%.1fx / %.1fx)",
           absl_ins_ns / (hwy_ins_ns + 1e-6),
           absl_erase_ns / (hwy_erase_ns + 1e-6));

  printf("%-17s %11s %10s %10s %10s %10s %12s %12s %18s\n", "  faster vs absl",
         build_sp, find_sp, lb_sp, ub_sp, iter_sp, bfind_sp, blb_sp,
         ins_del_sp);
  printf(
      "========================================================================"
      "================================================\n");

  HWY_ASSERT(std_hits == absl_hits);
  HWY_ASSERT(hwy_hits == absl_hits);
  HWY_ASSERT(std_lb_sum == absl_lb_sum);
  HWY_ASSERT(hwy_lb_sum == absl_lb_sum);
  HWY_ASSERT(batch_lb_sum == absl_lb_sum);
  HWY_ASSERT(std_iter_sum == absl_iter_sum);
  HWY_ASSERT(hwy_iter_sum == absl_iter_sum);
}

// ---------------------------------------------------------------------------
// Memory benchmarks (std vs absl vs hwy)
// ---------------------------------------------------------------------------

// Adapters so one memory suite drives both sets and maps.
template <typename KeyT>
struct SetMemOps {
  using Key = KeyT;
  using Elem = KeyT;
  using Hwy = BTreeSet<KeyT>;
  using Absl = absl::btree_set<KeyT>;
  using Std = std::set<KeyT>;
  static constexpr const char* kHwyName = "hwy::btree_set";
  static constexpr const char* kAbslName = "absl::btree_set";
  static constexpr const char* kStdName = "std::set";
  static constexpr const char* kUnit = "key";

  static Elem MakeElem(KeyT k) { return k; }
  static Hwy Build(const std::vector<KeyT>& keys) {
    return Hwy::Build(keys.data(), keys.size());
  }
};

template <typename KeyT, typename ValueT>
struct MapMemOps {
  using Key = KeyT;
  using Elem = std::pair<KeyT, ValueT>;
  using Hwy = BTreeMap<KeyT, ValueT>;
  using Absl = absl::btree_map<KeyT, ValueT>;
  using Std = std::map<KeyT, ValueT>;
  static constexpr const char* kHwyName = "hwy::btree_map";
  static constexpr const char* kAbslName = "absl::btree_map";
  static constexpr const char* kStdName = "std::map";
  static constexpr const char* kUnit = "pair";

  static Elem MakeElem(KeyT k) { return {k, static_cast<ValueT>(k)}; }
  static Hwy Build(const std::vector<KeyT>& keys) {
    std::vector<ValueT> vals(keys.begin(), keys.end());  // Freed on return.
    return Hwy::Build(keys.data(), vals.data(), keys.size());
  }
};

// Fallback if tcmalloc is unavailable: hwy trees report their own size.
template <class Tree>
auto GuessedBytes(const Tree& tree, int) -> decltype(tree.AllocatedBytes()) {
  return tree.AllocatedBytes();
}
template <class Tree>
size_t GuessedBytes(const Tree&, long) {  // std / absl: unknown.
  return 0;
}

template <class Tree>
double BytesPerElem(size_t before, const Tree& tree) {
  const size_t bytes = GetAllocatedBytes(before, GuessedBytes(tree, 0));
  return static_cast<double>(bytes) / static_cast<double>(tree.size());
}

struct MemPair {
  double hwy;
  double absl;
  double std;
};

// hwy Build() vs std / absl range constructors, all from the same unsorted
// keys.
template <class Ops>
MemPair BulkBuildMemory(const std::vector<typename Ops::Key>& keys) {
  std::vector<typename Ops::Elem> elems;
  elems.reserve(keys.size());
  for (auto k : keys) elems.push_back(Ops::MakeElem(k));

  MemPair m;
  size_t before = AllocatedBefore();
  const typename Ops::Hwy hwy_tree = Ops::Build(keys);
  m.hwy = BytesPerElem(before, hwy_tree);

  before = AllocatedBefore();
  const typename Ops::Absl absl_tree(elems.begin(), elems.end());
  m.absl = BytesPerElem(before, absl_tree);

  before = AllocatedBefore();
  const typename Ops::Std std_tree(elems.begin(), elems.end());
  m.std = BytesPerElem(before, std_tree);
  return m;
}

// insert()s `keys` one by one into an empty Tree, then erase()s the first
// `num_erase` of them. Returns bytes per live element after each phase.
template <class Ops, class Tree>
std::pair<double, double> InsertEraseMemory(
    const std::vector<typename Ops::Key>& keys, size_t num_erase) {
  Tree tree;
  const size_t before = AllocatedBefore();
  for (auto k : keys) tree.insert(Ops::MakeElem(k));
  HWY_ASSERT(tree.size() == keys.size());
  const double after_insert = BytesPerElem(before, tree);
  for (size_t i = 0; i < num_erase; ++i) tree.erase(keys[i]);
  HWY_ASSERT(tree.size() == keys.size() - num_erase);
  const double after_erase = num_erase ? BytesPerElem(before, tree) : 0.0;
  return {after_insert, after_erase};
}

template <class Ops>
MemPair InsertMemory(const std::vector<typename Ops::Key>& keys) {
  return {InsertEraseMemory<Ops, typename Ops::Hwy>(keys, 0).first,
          InsertEraseMemory<Ops, typename Ops::Absl>(keys, 0).first,
          InsertEraseMemory<Ops, typename Ops::Std>(keys, 0).first};
}

template <class Ops>
void RunMemorySuite(const char* name, size_t n) {
  using KeyT = typename Ops::Key;
  absl::BitGen bitgen;

  // Compressible: 10, 20, ..., 10*N in random order.
  std::vector<KeyT> keys(n);
  for (size_t i = 0; i < n; ++i) keys[i] = static_cast<KeyT>((i + 1) * 10);
  std::shuffle(keys.begin(), keys.end(), bitgen);

  // Uncompressible: unique random even keys over the whole key range, in
  // random order. Neighbors are ~2^bits / N apart, so leaves stay raw.
  std::vector<KeyT> worst(n);
  for (auto& k : worst) {
    k = static_cast<KeyT>(
        absl::Uniform<KeyT>(bitgen, 0, std::numeric_limits<KeyT>::max() / 2) *
        2);
  }
  std::sort(worst.begin(), worst.end());
  worst.erase(std::unique(worst.begin(), worst.end()), worst.end());
  std::shuffle(worst.begin(), worst.end(), bitgen);

  const auto hwy_mut = InsertEraseMemory<Ops, typename Ops::Hwy>(keys, n / 2);
  const auto absl_mut = InsertEraseMemory<Ops, typename Ops::Absl>(keys, n / 2);
  const auto std_mut = InsertEraseMemory<Ops, typename Ops::Std>(keys, n / 2);
  std::vector<KeyT> sorted = keys;
  std::sort(sorted.begin(), sorted.end());
  const MemPair asc = InsertMemory<Ops>(sorted);
  std::reverse(sorted.begin(), sorted.end());
  const MemPair desc = InsertMemory<Ops>(sorted);

  const MemPair cols[] = {
      BulkBuildMemory<Ops>(keys),
      {hwy_mut.first, absl_mut.first, std_mut.first},
      {hwy_mut.second, absl_mut.second, std_mut.second},
      asc,
      desc,
      BulkBuildMemory<Ops>(worst),
      InsertMemory<Ops>(worst),
  };

  printf(
      "\n======================================================================"
      "==================================================\n");
  printf("  Memory: %s (N = %zu %ss, bytes per %s) vs %s vs %s\n", name, n,
         Ops::kUnit, Ops::kUnit, Ops::kAbslName, Ops::kStdName);
  printf(
      "========================================================================"
      "================================================\n");
  printf("%-18s %13s %13s %13s %13s %13s %13s %13s\n", "Container",
         "Bulk Build", "Rand Insert", "Rand Erase", "Asc Insert", "Desc Insert",
         "Worst Build", "Worst Insert");
  printf(
      "------------------------------------------------------------------------"
      "------------------------------------------------\n");
  printf("%-18s", Ops::kStdName);
  for (const MemPair& c : cols) printf(" %11.2f B", c.std);
  printf("\n%-18s", Ops::kAbslName);
  for (const MemPair& c : cols) printf(" %11.2f B", c.absl);
  printf("\n%-18s", Ops::kHwyName);
  for (const MemPair& c : cols) printf(" %11.2f B", c.hwy);
  printf("\n%-18s", "  smaller vs absl");
  for (const MemPair& c : cols) {
    char ratio[32];
    snprintf(ratio, sizeof(ratio), "(%.2fx)", c.absl / (c.hwy + 1e-6));
    printf(" %13s", ratio);
  }
  printf(
      "\n====================================================================="
      "===================================================\n");
}

static void PrintBenchmarkLegend() {
  static const char* const kLines[] = {
      "Latency tables (std vs absl vs hwy; faster vs absl = absl time / hwy "
      "time)",
      "  * Build (ms)        : Bulk-construction latency from unsorted arrays "
      "(dense keys with uniform step delta = 10, fill_ratio = 1.0).",
      "  * Find (ns)         : Serial point lookup latency on the bulk-loaded "
      "tree (1M random queries sampled from [0, 10*N], ~10% hit rate).",
      "  * LB / UB (ns)      : Serial lower_bound / upper_bound range search "
      "latency on the bulk-loaded tree (1M random queries).",
      "  * Iter Walk (ns)    : Per-element latency during full ordered "
      "begin()-to-end() traversal of the bulk-loaded tree.",
      "  * Batch Find (ns)   : Time per query when querying a batch of 1M "
      "keys (sequential loop for std/absl, native 8-way pipelined SIMD batch "
      "for hwy).",
      "  * Batch LB (ns)     : Time per range query when querying a batch of "
      "1M keys (sequential loop for std/absl, native 8-way pipelined SIMD "
      "batch for hwy).",
      "  * Dyn Ins/Del (ns)  : Mutation latency starting from an empty tree "
      "(100K random insertions sampled from [0, 20*N], followed by 50K random "
      "erases).",
      "",
      "Memory tables (std vs absl vs hwy; heap bytes measured via TCMalloc, "
      "divided by the live keys / pairs; smaller vs absl = absl / hwy)",
      "  * Compressible keys : 10, 20, ..., 10*N (unique) in random order.",
      "  * Uncompressible    : N unique random even keys spread over the "
      "whole key range, in random order (forces raw, uncompressed leaves).",
      "  * Bulk Build        : Compressible keys; hwy Build() vs std / absl "
      "range constructors.",
      "  * Rand Insert       : Compressible keys insert()ed one by one into an "
      "empty tree.",
      "  * Rand Erase        : The Rand Insert tree after erase()ing a random "
      "half of its keys (divided by the N/2 remaining).",
      "  * Asc / Desc Insert : Compressible keys insert()ed one by one in "
      "ascending / descending order into an empty tree.",
      "  * Worst Build       : Uncompressible keys; hwy Build() vs std / "
      "absl range constructors.",
      "  * Worst Insert      : Uncompressible keys insert()ed one by one into "
      "an empty tree.",
  };
  printf(
      "\n======================================================================"
      "==================================================\n");
  printf("  Benchmark Setup\n");
  printf(
      "========================================================================"
      "================================================\n");
  for (const char* line : kLines) printf("  %s\n", line);
  printf(
      "========================================================================"
      "================================================\n\n");
}

static void RunAll(size_t n) {
  RunBenchmarkSuite<uint32_t>(n);
  RunBenchmarkSuite<uint64_t>(n);
  RunMapBenchmarkSuite<uint32_t, uint64_t>(n);
  RunMapBenchmarkSuite<uint64_t, double>(n);

  RunMemorySuite<SetMemOps<uint32_t>>("hwy::btree_set<uint32_t>", n);
  RunMemorySuite<SetMemOps<uint64_t>>("hwy::btree_set<uint64_t>", n);
  RunMemorySuite<MapMemOps<uint32_t, uint64_t>>(
      "hwy::btree_map<uint32_t, uint64_t>", n);
  RunMemorySuite<MapMemOps<uint64_t, double>>(
      "hwy::btree_map<uint64_t, double>", n);
}

HWY_NOINLINE void BenchmarkAll() {
  if constexpr (kSweepSizes) {
    for (size_t n : {10000, 20000, 30000, 50000, 70000, 100000, 200000, 300000,
                     500000, 700000, 1000000, 2000000, 4000000}) {
      RunAll(n);
    }
  } else {
    // Standard representative L3-scale (100K keys)
    RunAll(100000);
  }

  PrintBenchmarkLegend();
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
