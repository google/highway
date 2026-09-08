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

// Set to true to sweep through multiple scale tiers (10K L2, 100K L3, 1M RAM).
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
  printf("  BTreeSet<%s> (N = %zu keys, %s) vs absl::btree_set vs std::set\n",
         sizeof(KeyT) == 4 ? "uint32_t" : "uint64_t", num_keys,
         hwy::TargetName(HWY_TARGET));
  printf(
      "========================================================================"
      "================================================\n");
  printf("%-18s %14s %12s %12s %12s %12s %14s %14s %18s\n", "Container",
         "Memory (B/k)", "Build (ms)", "Find (ns)", "LB (ns)", "UB (ns)",
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
  auto hwy_tree = BTreeSet<KeyT>::Build(keys.data(), keys.size());
  const double end_hwy = hwy::platform::Now();
  const size_t hwy_bytes =
      GetAllocatedBytes(hwy_before, hwy_tree.AllocatedBytes());

  const double std_build_ms = (end_std - start_std) * 1000.0;
  const double absl_build_ms = (end_absl - start_absl) * 1000.0;
  const double hwy_build_ms = (end_hwy - start_hwy) * 1000.0;

  const double std_bk = static_cast<double>(std_bytes) / num_keys;
  const double absl_bk = static_cast<double>(absl_bytes) / num_keys;
  const double hwy_bk = static_cast<double>(hwy_bytes) / num_keys;

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
      "%-18s %12.1f B %10.1f ms %10.1f ns %10.1f ns %10.1f ns %12.1f ns "
      "%12.1f ns %18s\n",
      "std::set", std_bk, std_build_ms, std_find_ns, std_lb_ns, std_ub_ns,
      std_find_ns, std_lb_ns, std_ins_del);
  printf(
      "%-18s %12.1f B %10.1f ms %10.1f ns %10.1f ns %10.1f ns %12.1f ns "
      "%12.1f ns %18s\n",
      "absl::btree_set", absl_bk, absl_build_ms, absl_find_ns, absl_lb_ns,
      absl_ub_ns, absl_find_ns, absl_lb_ns, absl_ins_del);
  printf(
      "%-18s %12.1f B %10.1f ms %10.1f ns %10.1f ns %10.1f ns %12.1f ns "
      "%12.1f ns %18s\n",
      "hwy::BTreeSet", hwy_bk, hwy_build_ms, hwy_find_ns, hwy_lb_ns, hwy_ub_ns,
      hwy_batch_find_ns, hwy_batch_lb_ns, hwy_ins_del);

  char save_str[32], build_sp[32], find_sp[32], lb_sp[32], ub_sp[32],
      bfind_sp[32], blb_sp[32], ins_del_sp[32];
  snprintf(save_str, sizeof(save_str), "(%.1fx smaller)",
           absl_bk / (hwy_bk + 1e-6));
  snprintf(build_sp, sizeof(build_sp), "(%.1fx)",
           absl_build_ms / (hwy_build_ms + 1e-6));
  snprintf(find_sp, sizeof(find_sp), "(%.1fx)",
           absl_find_ns / (hwy_find_ns + 1e-6));
  snprintf(lb_sp, sizeof(lb_sp), "(%.1fx)", absl_lb_ns / (hwy_lb_ns + 1e-6));
  snprintf(ub_sp, sizeof(ub_sp), "(%.1fx)", absl_ub_ns / (hwy_ub_ns + 1e-6));
  snprintf(bfind_sp, sizeof(bfind_sp), "(%.1fx)",
           absl_find_ns / (hwy_batch_find_ns + 1e-6));
  snprintf(blb_sp, sizeof(blb_sp), "(%.1fx)",
           absl_lb_ns / (hwy_batch_lb_ns + 1e-6));
  snprintf(ins_del_sp, sizeof(ins_del_sp), "(%.1fx / %.1fx)",
           absl_ins_ns / (hwy_ins_ns + 1e-6),
           absl_erase_ns / (hwy_erase_ns + 1e-6));

  printf("%-18s %14s %12s %12s %12s %12s %14s %14s %18s\n", "  vs absl",
         save_str, build_sp, find_sp, lb_sp, ub_sp, bfind_sp, blb_sp,
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
}

template <typename KeyT, typename ValueT>
void RunMapBenchmarkSuite(size_t num_keys) {
  printf(
      "\n======================================================================"
      "==================================================\n");
  printf(
      "  BTreeMap<%s, %s> (N = %zu pairs, %s) vs absl::btree_map vs std::map\n",
      sizeof(KeyT) == 4 ? "uint32_t" : "uint64_t",
      sizeof(ValueT) == 8
          ? (std::is_floating_point_v<ValueT> ? "double" : "uint64_t")
          : "uint32_t",
      num_keys, hwy::TargetName(HWY_TARGET));
  printf(
      "========================================================================"
      "================================================\n");
  printf("%-18s %14s %12s %12s %12s %12s %14s %14s %18s\n", "Container",
         "Memory (B/p)", "Build (ms)", "Find (ns)", "LB (ns)", "UB (ns)",
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
    keys.push_back(k);
    vals.push_back(v);
    kv_pairs.push_back({k, v});
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
  auto hwy_map =
      BTreeMap<KeyT, ValueT>::Build(keys.data(), vals.data(), keys.size());
  const double end_hwy = hwy::platform::Now();
  const size_t hwy_bytes =
      GetAllocatedBytes(hwy_before, hwy_map.AllocatedBytes());

  const double std_build_ms = (end_std - start_std) * 1000.0;
  const double absl_build_ms = (end_absl - start_absl) * 1000.0;
  const double hwy_build_ms = (end_hwy - start_hwy) * 1000.0;

  const double std_bp = static_cast<double>(std_bytes) / num_keys;
  const double absl_bp = static_cast<double>(absl_bytes) / num_keys;
  const double hwy_bp = static_cast<double>(hwy_bytes) / num_keys;

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
      "%-18s %12.1f B %10.1f ms %10.1f ns %10.1f ns %10.1f ns %12.1f ns "
      "%12.1f ns %18s\n",
      "std::map", std_bp, std_build_ms, std_find_ns, std_lb_ns, std_ub_ns,
      std_find_ns, std_lb_ns, std_ins_del);
  printf(
      "%-18s %12.1f B %10.1f ms %10.1f ns %10.1f ns %10.1f ns %12.1f ns "
      "%12.1f ns %18s\n",
      "absl::btree_map", absl_bp, absl_build_ms, absl_find_ns, absl_lb_ns,
      absl_ub_ns, absl_find_ns, absl_lb_ns, absl_ins_del);
  printf(
      "%-18s %12.1f B %10.1f ms %10.1f ns %10.1f ns %10.1f ns %12.1f ns "
      "%12.1f ns %18s\n",
      "hwy::BTreeMap", hwy_bp, hwy_build_ms, hwy_find_ns, hwy_lb_ns, hwy_ub_ns,
      hwy_batch_find_ns, hwy_batch_lb_ns, hwy_ins_del);

  char save_str[32], build_sp[32], find_sp[32], lb_sp[32], ub_sp[32],
      bfind_sp[32], blb_sp[32], ins_del_sp[32];
  snprintf(save_str, sizeof(save_str), "(%.1fx smaller)",
           absl_bp / (hwy_bp + 1e-6));
  snprintf(build_sp, sizeof(build_sp), "(%.1fx)",
           absl_build_ms / (hwy_build_ms + 1e-6));
  snprintf(find_sp, sizeof(find_sp), "(%.1fx)",
           absl_find_ns / (hwy_find_ns + 1e-6));
  snprintf(lb_sp, sizeof(lb_sp), "(%.1fx)", absl_lb_ns / (hwy_lb_ns + 1e-6));
  snprintf(ub_sp, sizeof(ub_sp), "(%.1fx)", absl_ub_ns / (hwy_ub_ns + 1e-6));
  snprintf(bfind_sp, sizeof(bfind_sp), "(%.1fx)",
           absl_find_ns / (hwy_batch_find_ns + 1e-6));
  snprintf(blb_sp, sizeof(blb_sp), "(%.1fx)",
           absl_lb_ns / (hwy_batch_lb_ns + 1e-6));
  snprintf(ins_del_sp, sizeof(ins_del_sp), "(%.1fx / %.1fx)",
           absl_ins_ns / (hwy_ins_ns + 1e-6),
           absl_erase_ns / (hwy_erase_ns + 1e-6));

  printf("%-18s %14s %12s %12s %12s %12s %14s %14s %18s\n", "  vs absl",
         save_str, build_sp, find_sp, lb_sp, ub_sp, bfind_sp, blb_sp,
         ins_del_sp);
  printf(
      "========================================================================"
      "================================================\n");

  HWY_ASSERT(std_hits == absl_hits);
  HWY_ASSERT(hwy_hits == absl_hits);
  HWY_ASSERT(std_lb_sum == absl_lb_sum);
  HWY_ASSERT(hwy_lb_sum == absl_lb_sum);
  HWY_ASSERT(batch_lb_sum == absl_lb_sum);
}

template <typename KeyT>
void RunWorstCaseMemoryComparison(size_t num_keys) {
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

  const size_t absl_before = AllocatedBefore();
  absl::btree_set<KeyT> absl_tree(keys.begin(), keys.end());
  const size_t absl_bytes = GetAllocatedBytes(absl_before, 0);

  const size_t hwy_before = AllocatedBefore();
  auto hwy_tree = BTreeSet<KeyT>::Build(keys.data(), keys.size(), 1.0f);
  const size_t hwy_bytes =
      GetAllocatedBytes(hwy_before, hwy_tree.AllocatedBytes());

  const double absl_bk = static_cast<double>(absl_bytes) / num_keys;
  const double hwy_bk = static_cast<double>(hwy_bytes) / num_keys;

  printf(
      "  Worst-Case Uncompressible Memory (N = %zu keys): absl = %.1f B/k | "
      "hwy = %.1f B/k\n\n",
      num_keys, absl_bk, hwy_bk);
}

template <typename KeyT, typename ValueT>
void RunWorstCaseMapMemoryComparison(size_t num_keys) {
  absl::BitGen bitgen;
  std::map<KeyT, ValueT> ref_map;
  while (ref_map.size() < num_keys) {
    KeyT k = static_cast<KeyT>(
        absl::Uniform<KeyT>(bitgen, 0, std::numeric_limits<KeyT>::max() / 2) *
        2);
    ValueT v =
        static_cast<ValueT>(absl::Uniform<uint64_t>(bitgen, 1, 1000000000));
    ref_map[k] = v;
  }
  num_keys = ref_map.size();

  std::vector<KeyT> keys;
  std::vector<ValueT> vals;
  keys.reserve(num_keys);
  vals.reserve(num_keys);
  for (const auto& [k, v] : ref_map) {
    keys.push_back(k);
    vals.push_back(v);
  }

  const size_t absl_before = AllocatedBefore();
  absl::btree_map<KeyT, ValueT> absl_map(ref_map.begin(), ref_map.end());
  const size_t absl_bytes = GetAllocatedBytes(absl_before, 0);

  const size_t hwy_before = AllocatedBefore();
  auto hwy_map = BTreeMap<KeyT, ValueT>::Build(keys.data(), vals.data(),
                                               keys.size(), 1.0f);
  const size_t hwy_bytes =
      GetAllocatedBytes(hwy_before, hwy_map.AllocatedBytes());

  const double absl_bp = static_cast<double>(absl_bytes) / num_keys;
  const double hwy_bp = static_cast<double>(hwy_bytes) / num_keys;

  printf(
      "  Worst-Case Uncompressible Memory (N = %zu pairs): absl = %.1f B/p | "
      "hwy = %.1f B/p\n\n",
      num_keys, absl_bp, hwy_bp);
}

static void PrintBenchmarkLegend() {
  printf(
      "========================================================================"
      "================================================\n");
  printf("  Benchmark Setup\n");
  printf(
      "========================================================================"
      "================================================\n");
  printf(
      "  * Memory (B/k, B/p) : Total heap allocation measured via TCMalloc "
      "(bytes per key / pair) on the bulk-loaded tree.\n");
  printf(
      "  * Build (ms)        : Bulk-construction latency from sorted arrays "
      "(dense keys with uniform step delta = 10, fill_ratio = 1.0).\n");
  printf(
      "  * Find (ns)         : Serial point lookup latency on the bulk-loaded "
      "tree (1M random queries sampled from [0, 10*N], ~10%% hit rate).\n");
  printf(
      "  * LB / UB (ns)      : Serial lower_bound / upper_bound range search "
      "latency on the bulk-loaded tree (1M random queries).\n");
  printf(
      "  * Batch Find (ns)   : Time per query when querying a batch of 1M "
      "keys (sequential loop for std/absl, native 8-way pipelined SIMD batch "
      "for hwy).\n");
  printf(
      "  * Batch LB (ns)     : Time per range query when querying a batch of "
      "1M keys (sequential loop for std/absl, native 8-way pipelined SIMD "
      "batch for hwy).\n");
  printf(
      "  * Dyn Ins/Del (ns)  : Mutation latency starting from an empty tree "
      "(100K random insertions sampled from [0, 20*N], followed by 50K random "
      "erases).\n");
  printf(
      "  * Worst-Case Memory : Heap space on completely uncompressible uniform "
      "random keys with large spread (forcing raw uncompressed mode).\n");
  printf(
      "========================================================================"
      "================================================\n\n");
}

HWY_NOINLINE void BenchmarkAll() {
  if constexpr (kSweepSizes) {
    for (size_t n : {10000, 100000, 1000000}) {
      RunBenchmarkSuite<uint32_t>(n);
      RunWorstCaseMemoryComparison<uint32_t>(n);

      RunBenchmarkSuite<uint64_t>(n);
      RunWorstCaseMemoryComparison<uint64_t>(n);

      RunMapBenchmarkSuite<uint32_t, uint64_t>(n);
      RunWorstCaseMapMemoryComparison<uint32_t, uint64_t>(n);

      RunMapBenchmarkSuite<uint64_t, double>(n);
      RunWorstCaseMapMemoryComparison<uint64_t, double>(n);
    }
  } else {
    // Standard representative L3-scale (100K keys)
    constexpr size_t kDefaultN = 100000;
    RunBenchmarkSuite<uint32_t>(kDefaultN);
    RunWorstCaseMemoryComparison<uint32_t>(kDefaultN);

    RunBenchmarkSuite<uint64_t>(kDefaultN);
    RunWorstCaseMemoryComparison<uint64_t>(kDefaultN);

    RunMapBenchmarkSuite<uint32_t, uint64_t>(kDefaultN);
    RunWorstCaseMapMemoryComparison<uint32_t, uint64_t>(kDefaultN);

    RunMapBenchmarkSuite<uint64_t, double>(kDefaultN);
    RunWorstCaseMapMemoryComparison<uint64_t, double>(kDefaultN);
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
