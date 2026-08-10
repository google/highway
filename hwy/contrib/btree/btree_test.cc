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

#include <algorithm>
#include <iterator>
#include <limits>
#include <set>
#include <utility>
#include <vector>

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

#include "third_party/absl/container/btree_map.h"
#include "third_party/absl/container/btree_set.h"
#include "third_party/absl/random/random.h"
#include "hwy/base.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/btree/btree_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/btree/btree-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

#if (HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128) && !HWY_IDE
HWY_NOINLINE void TestAll() {}
#else

template <typename KeyT>
void TestEmptyTree() {
  auto tree = BTreeSet<KeyT>::Build(nullptr, 0);
  HWY_ASSERT(tree.empty());
  HWY_ASSERT_EQ(tree.size(), size_t{0});
  HWY_ASSERT_EQ(tree.height(), uint16_t{0});
  HWY_ASSERT(!tree.Contains(10));
  HWY_ASSERT(tree.find(10) == tree.end());
  HWY_ASSERT(tree.lower_bound(10) == tree.end());
  HWY_ASSERT(tree.upper_bound(10) == tree.end());
  HWY_ASSERT(tree.begin() == tree.end());
}

template <typename KeyT>
void TestSingleLeaf() {
  std::vector<KeyT> keys = {10, 20, 30, 40, 50};
  auto tree = BTreeSet<KeyT>::Build(keys.data(), keys.size());

  HWY_ASSERT_EQ(tree.size(), size_t{5});
  HWY_ASSERT_EQ(tree.height(), uint16_t{0});
  for (KeyT k : keys) {
    HWY_ASSERT(tree.Contains(k));
    HWY_ASSERT(tree.find(k) != tree.end());
    HWY_ASSERT_EQ(*tree.find(k), k);
  }
  HWY_ASSERT(!tree.Contains(5));
  HWY_ASSERT(tree.find(5) == tree.end());
  HWY_ASSERT(!tree.Contains(25));
  HWY_ASSERT(!tree.Contains(60));

  // LowerBound tests
  HWY_ASSERT_EQ(*tree.lower_bound(5), 10);
  HWY_ASSERT_EQ(*tree.lower_bound(10), 10);
  HWY_ASSERT_EQ(*tree.lower_bound(15), 20);
  HWY_ASSERT_EQ(*tree.lower_bound(50), 50);
  HWY_ASSERT(tree.lower_bound(55) == tree.end());

  // UpperBound tests
  HWY_ASSERT_EQ(*tree.upper_bound(5), 10);
  HWY_ASSERT_EQ(*tree.upper_bound(10), 20);
  HWY_ASSERT_EQ(*tree.upper_bound(49), 50);
  HWY_ASSERT(tree.upper_bound(50) == tree.end());

  // Traversal test
  std::vector<KeyT> traversed(tree.begin(), tree.end());
  HWY_ASSERT(traversed == keys);
}

template <typename KeyT>
void TestMultiLevelTree(size_t num_keys) {
  std::vector<KeyT> keys;
  keys.reserve(num_keys);
  for (size_t i = 0; i < num_keys; ++i) {
    keys.push_back(static_cast<KeyT>((i + 1) * 10));
  }

  auto tree = BTreeSet<KeyT>::Build(keys.data(), keys.size());
  HWY_ASSERT_EQ(tree.size(), num_keys);
  HWY_ASSERT(tree.AllocatedBytes() > 0);

  // Point lookups
  for (size_t i = 0; i < num_keys; ++i) {
    HWY_ASSERT_M(tree.Contains(keys[i]), "Key missing");
    HWY_ASSERT(!tree.Contains(keys[i] + 5));
  }

  // Boundary lookups
  HWY_ASSERT_EQ(*tree.lower_bound(0), keys.front());
  HWY_ASSERT_EQ(*tree.lower_bound(keys.front()), keys.front());
  HWY_ASSERT_EQ(*tree.lower_bound(keys.back()), keys.back());
  HWY_ASSERT(tree.lower_bound(keys.back() + 1) == tree.end());

  HWY_ASSERT_EQ(*tree.upper_bound(0), keys.front());
  HWY_ASSERT_EQ(*tree.upper_bound(keys.front()), keys[1]);
  HWY_ASSERT(tree.upper_bound(keys.back()) == tree.end());

  // Full Traversal
  size_t idx = 0;
  for (auto it = tree.begin(); it != tree.end(); ++it, ++idx) {
    HWY_ASSERT_EQ(*it, keys[idx]);
  }
  HWY_ASSERT_EQ(idx, num_keys);
}

template <typename KeyT>
void TestSignedKeys() {
  std::vector<KeyT> keys = {-500, -200, -100, -50, 0, 50, 100, 200, 500};
  auto tree = BTreeSet<KeyT>::Build(keys.data(), keys.size());

  HWY_ASSERT_EQ(tree.size(), keys.size());
  for (KeyT k : keys) {
    HWY_ASSERT(tree.Contains(k));
    HWY_ASSERT(tree.find(k) != tree.end());
    HWY_ASSERT_EQ(*tree.find(k), k);
  }
  HWY_ASSERT(!tree.Contains(-600));
  HWY_ASSERT(!tree.Contains(-150));
  HWY_ASSERT(!tree.Contains(600));

  HWY_ASSERT_EQ(*tree.lower_bound(-600), -500);
  HWY_ASSERT_EQ(*tree.lower_bound(-500), -500);
  HWY_ASSERT_EQ(*tree.lower_bound(-150), -100);
  HWY_ASSERT_EQ(*tree.lower_bound(0), 0);
  HWY_ASSERT_EQ(*tree.lower_bound(200), 200);
  HWY_ASSERT(tree.lower_bound(501) == tree.end());

  std::vector<KeyT> traversed(tree.begin(), tree.end());
  HWY_ASSERT(traversed == keys);
}

template <typename KeyT>
void TestMoveSemantics() {
  std::vector<KeyT> keys = {10, 20, 30, 40, 50, 60, 70, 80};
  auto tree1 = BTreeSet<KeyT>::Build(keys.data(), keys.size());
  HWY_ASSERT_EQ(tree1.size(), keys.size());
  HWY_ASSERT(!tree1.empty());

  // Move construction
  BTreeSet<KeyT> tree2 = std::move(tree1);
  HWY_ASSERT_EQ(tree2.size(), keys.size());
  HWY_ASSERT(!tree2.empty());
  for (KeyT k : keys) {
    HWY_ASSERT(tree2.Contains(k));
  }

  // Moved-from tree1 must be safely empty
  HWY_ASSERT(tree1.empty());
  HWY_ASSERT_EQ(tree1.size(), size_t{0});
  HWY_ASSERT(!tree1.Contains(10));
  HWY_ASSERT(tree1.find(10) == tree1.end());

  // Move assignment
  BTreeSet<KeyT> tree3;
  tree3 = std::move(tree2);
  HWY_ASSERT_EQ(tree3.size(), keys.size());
  HWY_ASSERT(!tree3.empty());
  HWY_ASSERT(tree2.empty());
  HWY_ASSERT_EQ(tree2.size(), size_t{0});
}

template <typename KeyT>
void TestRandomizedComparisonAgainstStdSet(size_t num_keys,
                                           size_t num_queries) {
  absl::BitGen bitgen;
  std::set<KeyT> reference_set;
  while (reference_set.size() < num_keys) {
    reference_set.insert(
        static_cast<KeyT>(absl::Uniform<KeyT>(bitgen, 1, 10000000)));
  }

  std::vector<KeyT> sorted_keys(reference_set.begin(), reference_set.end());
  auto tree = BTreeSet<KeyT>::Build(sorted_keys.data(), sorted_keys.size());

  // 1. Full In-Order Traversal Check vs std::set
  HWY_ASSERT(std::equal(tree.begin(), tree.end(), reference_set.begin(),
                        reference_set.end()));

  // 2. Random Query Checks (Find, Contains, LowerBound, Range Scans)
  for (size_t q = 0; q < num_queries; ++q) {
    KeyT query_key =
        static_cast<KeyT>(absl::Uniform<KeyT>(bitgen, 0, 10000050));

    // A. Contains & Find check
    bool expected_contains =
        (reference_set.find(query_key) != reference_set.end());
    HWY_ASSERT_EQ(tree.Contains(query_key), expected_contains);
    HWY_ASSERT_EQ(tree.find(query_key) != tree.end(), expected_contains);

    // B. LowerBound check
    auto ref_lb = reference_set.lower_bound(query_key);
    auto tree_lb = tree.lower_bound(query_key);
    if (ref_lb == reference_set.end()) {
      HWY_ASSERT(tree_lb == tree.end());
    } else {
      HWY_ASSERT(tree_lb != tree.end());
      HWY_ASSERT_EQ(*tree_lb, *ref_lb);
    }

    // C. UpperBound check
    auto ref_ub = reference_set.upper_bound(query_key);
    auto tree_ub = tree.upper_bound(query_key);
    if (ref_ub == reference_set.end()) {
      HWY_ASSERT(tree_ub == tree.end());
    } else {
      HWY_ASSERT(tree_ub != tree.end());
      HWY_ASSERT_EQ(*tree_ub, *ref_ub);
    }

    // D. Range Traversal check: [query_key, query_key + 50000]
    KeyT range_end_key = query_key + static_cast<KeyT>(50000);
    auto ref_range_it = reference_set.lower_bound(query_key);
    auto ref_range_end = reference_set.upper_bound(range_end_key);

    auto tree_range_it = tree.lower_bound(query_key);
    auto tree_range_end = tree.upper_bound(range_end_key);

    while (ref_range_it != ref_range_end) {
      HWY_ASSERT(tree_range_it != tree.end());
      HWY_ASSERT(tree_range_it != tree_range_end);
      HWY_ASSERT_EQ(*tree_range_it, *ref_range_it);
      ++ref_range_it;
      ++tree_range_it;
    }
    HWY_ASSERT(tree_range_it == tree_range_end);
  }
}

template <typename KeyT, typename ValueT>
void TestMapEmpty() {
  auto map = BTreeMap<KeyT, ValueT>::Build(nullptr, nullptr, 0);
  HWY_ASSERT(map.empty());
  HWY_ASSERT_EQ(map.size(), size_t{0});
  HWY_ASSERT_EQ(map.height(), uint16_t{0});
  HWY_ASSERT(!map.Contains(10));
  HWY_ASSERT(map.find(10) == map.end());
  HWY_ASSERT(map.FindValue(10) == nullptr);
  HWY_ASSERT(map.lower_bound(10) == map.end());
  HWY_ASSERT(map.upper_bound(10) == map.end());
  HWY_ASSERT(map.begin() == map.end());
}

template <typename KeyT, typename ValueT>
void TestMapSingleLeaf() {
  std::vector<KeyT> keys = {10, 20, 30, 40, 50};
  std::vector<ValueT> vals = {100, 200, 300, 400, 500};
  auto map =
      BTreeMap<KeyT, ValueT>::Build(keys.data(), vals.data(), keys.size());

  HWY_ASSERT_EQ(map.size(), size_t{5});
  HWY_ASSERT_EQ(map.height(), uint16_t{0});

  for (size_t i = 0; i < keys.size(); ++i) {
    HWY_ASSERT(map.Contains(keys[i]));
    auto it = map.find(keys[i]);
    HWY_ASSERT(it != map.end());
    HWY_ASSERT_EQ(it->first, keys[i]);
    HWY_ASSERT_EQ(it->second, vals[i]);
    HWY_ASSERT(map.FindValue(keys[i]) != nullptr);
    HWY_ASSERT_EQ(*map.FindValue(keys[i]), vals[i]);
  }

  HWY_ASSERT(!map.Contains(5));
  HWY_ASSERT(map.find(5) == map.end());
  HWY_ASSERT(map.FindValue(5) == nullptr);

  // LowerBound tests
  HWY_ASSERT_EQ(map.lower_bound(5)->first, 10);
  HWY_ASSERT_EQ(map.lower_bound(5)->second, 100);
  HWY_ASSERT_EQ(map.lower_bound(25)->first, 30);
  HWY_ASSERT_EQ(map.lower_bound(25)->second, 300);
  HWY_ASSERT(map.lower_bound(55) == map.end());

  // Forward Traversal
  size_t idx = 0;
  for (auto it = map.begin(); it != map.end(); ++it, ++idx) {
    HWY_ASSERT_EQ(it->first, keys[idx]);
    HWY_ASSERT_EQ(it->second, vals[idx]);
  }
  HWY_ASSERT_EQ(idx, size_t{5});

  // Backward Traversal (operator--)
  auto it = map.end();
  while (idx > 0) {
    --idx;
    --it;
    HWY_ASSERT_EQ(it->first, keys[idx]);
    HWY_ASSERT_EQ(it->second, vals[idx]);
  }
  HWY_ASSERT(it == map.begin());
}

template <typename KeyT, typename ValueT>
void TestMapMultiLevel(size_t num_keys) {
  std::vector<KeyT> keys;
  std::vector<ValueT> vals;
  keys.reserve(num_keys);
  vals.reserve(num_keys);
  for (size_t i = 0; i < num_keys; ++i) {
    keys.push_back(static_cast<KeyT>((i + 1) * 10));
    vals.push_back(static_cast<ValueT>((i + 1) * 100));
  }

  auto map =
      BTreeMap<KeyT, ValueT>::Build(keys.data(), vals.data(), keys.size());
  HWY_ASSERT_EQ(map.size(), num_keys);
  HWY_ASSERT(map.AllocatedBytes() > 0);

  // Point lookups
  for (size_t i = 0; i < num_keys; ++i) {
    HWY_ASSERT(map.Contains(keys[i]));
    const ValueT* val_ptr = map.FindValue(keys[i]);
    HWY_ASSERT(val_ptr != nullptr);
    HWY_ASSERT_EQ(*val_ptr, vals[i]);
    HWY_ASSERT(!map.Contains(keys[i] + 5));
    HWY_ASSERT(map.FindValue(keys[i] + 5) == nullptr);
  }

  // Forward traversal
  size_t idx = 0;
  for (auto it = map.begin(); it != map.end(); ++it, ++idx) {
    HWY_ASSERT_EQ(it->first, keys[idx]);
    HWY_ASSERT_EQ(it->second, vals[idx]);
  }
  HWY_ASSERT_EQ(idx, num_keys);

  // Backward traversal
  auto back_it = map.end();
  while (idx > 0) {
    --idx;
    --back_it;
    HWY_ASSERT_EQ(back_it->first, keys[idx]);
    HWY_ASSERT_EQ(back_it->second, vals[idx]);
  }
  HWY_ASSERT(back_it == map.begin());
}

template <typename KeyT, typename ValueT>
void TestMapMoveSemantics() {
  std::vector<KeyT> keys = {10, 20, 30, 40};
  std::vector<ValueT> vals = {100, 200, 300, 400};
  auto map1 =
      BTreeMap<KeyT, ValueT>::Build(keys.data(), vals.data(), keys.size());
  HWY_ASSERT_EQ(map1.size(), keys.size());

  // Move constructor
  BTreeMap<KeyT, ValueT> map2 = std::move(map1);
  HWY_ASSERT_EQ(map2.size(), keys.size());
  HWY_ASSERT(map1.empty());
  HWY_ASSERT_EQ(map1.size(), size_t{0});
  HWY_ASSERT(!map1.Contains(10));

  // Move assignment
  BTreeMap<KeyT, ValueT> map3;
  map3 = std::move(map2);
  HWY_ASSERT_EQ(map3.size(), keys.size());
  HWY_ASSERT(map2.empty());
  HWY_ASSERT_EQ(map2.size(), size_t{0});
}

template <typename KeyT, typename ValueT>
void TestMapRandomizedComparisonAgainstAbsl(size_t num_keys,
                                            size_t num_queries) {
  absl::BitGen bitgen;
  absl::btree_map<KeyT, ValueT> ref_map;
  while (ref_map.size() < num_keys) {
    KeyT k = static_cast<KeyT>(absl::Uniform<uint64_t>(bitgen, 1, 10000000));
    ValueT v =
        static_cast<ValueT>(absl::Uniform<uint64_t>(bitgen, 1, 10000000));
    ref_map[k] = v;
  }

  std::vector<KeyT> sorted_keys;
  std::vector<ValueT> vals;
  sorted_keys.reserve(ref_map.size());
  vals.reserve(ref_map.size());
  for (const auto& [k, v] : ref_map) {
    sorted_keys.push_back(k);
    vals.push_back(v);
  }

  auto map = BTreeMap<KeyT, ValueT>::Build(sorted_keys.data(), vals.data(),
                                           sorted_keys.size());

  // 1. Forward Traversal Check vs absl::btree_map
  auto ref_it = ref_map.begin();
  auto map_it = map.begin();
  while (ref_it != ref_map.end()) {
    HWY_ASSERT(map_it != map.end());
    HWY_ASSERT_EQ(map_it->first, ref_it->first);
    HWY_ASSERT_EQ(map_it->second, ref_it->second);
    ++ref_it;
    ++map_it;
  }
  HWY_ASSERT(map_it == map.end());

  // 2. Random Query Checks (Find, Contains, LowerBound, UpperBound)
  for (size_t q = 0; q < num_queries; ++q) {
    KeyT query_key =
        static_cast<KeyT>(absl::Uniform<uint64_t>(bitgen, 0, 10000050));

    // A. Contains & Find check
    auto ref_find = ref_map.find(query_key);
    auto map_find = map.find(query_key);
    if (ref_find == ref_map.end()) {
      HWY_ASSERT(!map.Contains(query_key));
      HWY_ASSERT(map_find == map.end());
      HWY_ASSERT(map.FindValue(query_key) == nullptr);
    } else {
      HWY_ASSERT(map.Contains(query_key));
      HWY_ASSERT(map_find != map.end());
      HWY_ASSERT_EQ(map_find->first, ref_find->first);
      HWY_ASSERT_EQ(map_find->second, ref_find->second);
      HWY_ASSERT(map.FindValue(query_key) != nullptr);
      HWY_ASSERT_EQ(*map.FindValue(query_key), ref_find->second);
    }

    // B. LowerBound check
    auto ref_lb = ref_map.lower_bound(query_key);
    auto map_lb = map.lower_bound(query_key);
    if (ref_lb == ref_map.end()) {
      HWY_ASSERT(map_lb == map.end());
    } else {
      HWY_ASSERT(map_lb != map.end());
      HWY_ASSERT_EQ(map_lb->first, ref_lb->first);
      HWY_ASSERT_EQ(map_lb->second, ref_lb->second);
    }

    // C. UpperBound check
    auto ref_ub = ref_map.upper_bound(query_key);
    auto map_ub = map.upper_bound(query_key);
    if (ref_ub == ref_map.end()) {
      HWY_ASSERT(map_ub == map.end());
    } else {
      HWY_ASSERT(map_ub != map.end());
      HWY_ASSERT_EQ(map_ub->first, ref_ub->first);
      HWY_ASSERT_EQ(map_ub->second, ref_ub->second);
    }

    // D. Sub-Range scan check [lo, hi)
    KeyT lo = query_key;
    KeyT hi = static_cast<KeyT>(lo + absl::Uniform<uint64_t>(bitgen, 0, 50000));
    auto map_range_it = map.lower_bound(lo);
    auto map_range_end = map.upper_bound(hi);
    auto ref_range_it = ref_map.lower_bound(lo);
    auto ref_range_end = ref_map.upper_bound(hi);

    while (ref_range_it != ref_range_end) {
      HWY_ASSERT(map_range_it != map.end());
      HWY_ASSERT(map_range_it != map_range_end);
      HWY_ASSERT_EQ(map_range_it->first, ref_range_it->first);
      HWY_ASSERT_EQ(map_range_it->second, ref_range_it->second);
      ++ref_range_it;
      ++map_range_it;
    }
    HWY_ASSERT(map_range_it == map_range_end);
  }
}

template <typename KeyT>
void TestBatchQueries(size_t num_keys, size_t num_queries) {
  // 1. Empty tree test
  {
    auto empty_tree = BTreeSet<KeyT>::Build(nullptr, 0);
    std::vector<KeyT> q = {10, 20, 30, 40, 50};
    auto found = std::make_unique<bool[]>(q.size());
    std::fill_n(found.get(), q.size(), true);
    std::vector<const KeyT*> ptrs(q.size(), reinterpret_cast<const KeyT*>(0x1));
    std::vector<typename BTreeSet<KeyT>::const_iterator> iters(q.size());

    empty_tree.ContainsBatch(q.data(), q.size(), found.get());
    empty_tree.FindBatch(q.data(), q.size(), ptrs.data());
    empty_tree.LowerBoundBatch(q.data(), q.size(), ptrs.data());
    empty_tree.LowerBoundBatch(q.data(), q.size(), iters.data());

    for (size_t i = 0; i < q.size(); ++i) {
      HWY_ASSERT(!found[i]);
      HWY_ASSERT(ptrs[i] == nullptr);
      HWY_ASSERT(iters[i] == empty_tree.end());
    }
  }

  // 2. Multi-level tree randomized batch verification
  absl::BitGen bitgen;
  std::set<KeyT> ref_set;
  while (ref_set.size() < num_keys) {
    ref_set.insert(
        static_cast<KeyT>(absl::Uniform<uint64_t>(bitgen, 1, 10000000)));
  }

  std::vector<KeyT> sorted_keys(ref_set.begin(), ref_set.end());
  auto tree = BTreeSet<KeyT>::Build(sorted_keys.data(), sorted_keys.size());

  std::vector<KeyT> queries;
  queries.reserve(num_queries);
  for (size_t i = 0; i < num_queries; ++i) {
    queries.push_back(
        static_cast<KeyT>(absl::Uniform<uint64_t>(bitgen, 0, 10000050)));
  }

  auto batch_found = std::make_unique<bool[]>(num_queries);
  std::vector<const KeyT*> batch_find_ptrs(num_queries);
  std::vector<const KeyT*> batch_lb_ptrs(num_queries);
  std::vector<typename BTreeSet<KeyT>::const_iterator> batch_lb_iters(
      num_queries);

  tree.ContainsBatch(queries.data(), queries.size(), batch_found.get());
  tree.FindBatch(queries.data(), queries.size(), batch_find_ptrs.data());
  tree.LowerBoundBatch(queries.data(), queries.size(), batch_lb_ptrs.data());
  tree.LowerBoundBatch(queries.data(), queries.size(), batch_lb_iters.data());

  for (size_t i = 0; i < num_queries; ++i) {
    KeyT qk = queries[i];
    bool expected_found = tree.Contains(qk);
    HWY_ASSERT_EQ(batch_found[i], expected_found);

    auto serial_find = tree.find(qk);
    if (serial_find == tree.end()) {
      HWY_ASSERT(batch_find_ptrs[i] == nullptr);
    } else {
      HWY_ASSERT(batch_find_ptrs[i] != nullptr);
      HWY_ASSERT_EQ(*batch_find_ptrs[i], qk);
    }

    auto serial_lb = tree.lower_bound(qk);
    if (serial_lb == tree.end()) {
      HWY_ASSERT(batch_lb_ptrs[i] == nullptr);
      HWY_ASSERT(batch_lb_iters[i] == tree.end());
    } else {
      HWY_ASSERT(batch_lb_ptrs[i] != nullptr);
      HWY_ASSERT_EQ(*batch_lb_ptrs[i], *serial_lb);
      HWY_ASSERT(batch_lb_iters[i] != tree.end());
      HWY_ASSERT_EQ(*batch_lb_iters[i], *serial_lb);
    }
  }
}

template <typename KeyT, typename ValueT>
void TestMapBatchQueries(size_t num_keys, size_t num_queries) {
  // 1. Empty map test
  {
    auto empty_map = BTreeMap<KeyT, ValueT>::Build(nullptr, nullptr, 0);
    std::vector<KeyT> q = {10, 20, 30, 40, 50};
    auto found = std::make_unique<bool[]>(q.size());
    std::fill_n(found.get(), q.size(), true);
    std::vector<const ValueT*> vals(q.size(),
                                    reinterpret_cast<const ValueT*>(0x1));
    std::vector<typename BTreeMap<KeyT, ValueT>::const_iterator> iters(
        q.size());

    empty_map.ContainsBatch(q.data(), q.size(), found.get());
    empty_map.FindValueBatch(q.data(), q.size(), vals.data());
    empty_map.LowerBoundBatch(q.data(), q.size(), iters.data());

    for (size_t i = 0; i < q.size(); ++i) {
      HWY_ASSERT(!found[i]);
      HWY_ASSERT(vals[i] == nullptr);
      HWY_ASSERT(iters[i] == empty_map.end());
    }
  }

  // 2. Multi-level map randomized batch verification
  absl::BitGen bitgen;
  absl::btree_map<KeyT, ValueT> ref_map;
  while (ref_map.size() < num_keys) {
    KeyT k = static_cast<KeyT>(absl::Uniform<uint64_t>(bitgen, 1, 10000000));
    ValueT v =
        static_cast<ValueT>(absl::Uniform<uint64_t>(bitgen, 1, 10000000));
    ref_map[k] = v;
  }

  std::vector<KeyT> sorted_keys;
  std::vector<ValueT> sorted_vals;
  sorted_keys.reserve(ref_map.size());
  sorted_vals.reserve(ref_map.size());
  for (const auto& [k, v] : ref_map) {
    sorted_keys.push_back(k);
    sorted_vals.push_back(v);
  }

  auto map = BTreeMap<KeyT, ValueT>::Build(
      sorted_keys.data(), sorted_vals.data(), sorted_keys.size());

  std::vector<KeyT> queries;
  queries.reserve(num_queries);
  for (size_t i = 0; i < num_queries; ++i) {
    queries.push_back(
        static_cast<KeyT>(absl::Uniform<uint64_t>(bitgen, 0, 10000050)));
  }

  auto batch_found = std::make_unique<bool[]>(num_queries);
  std::vector<const ValueT*> batch_vals(num_queries);
  std::vector<typename BTreeMap<KeyT, ValueT>::const_iterator> batch_lb_iters(
      num_queries);

  map.ContainsBatch(queries.data(), queries.size(), batch_found.get());
  map.FindValueBatch(queries.data(), queries.size(), batch_vals.data());
  map.LowerBoundBatch(queries.data(), queries.size(), batch_lb_iters.data());

  for (size_t i = 0; i < num_queries; ++i) {
    KeyT qk = queries[i];
    bool expected_found = map.Contains(qk);
    HWY_ASSERT_EQ(batch_found[i], expected_found);

    const ValueT* serial_val = map.FindValue(qk);
    if (serial_val == nullptr) {
      HWY_ASSERT(batch_vals[i] == nullptr);
    } else {
      HWY_ASSERT(batch_vals[i] != nullptr);
      HWY_ASSERT_EQ(*batch_vals[i], *serial_val);
    }

    auto serial_lb = map.lower_bound(qk);
    if (serial_lb == map.end()) {
      HWY_ASSERT(batch_lb_iters[i] == map.end());
    } else {
      HWY_ASSERT(batch_lb_iters[i] != map.end());
      HWY_ASSERT_EQ(batch_lb_iters[i]->first, serial_lb->first);
      HWY_ASSERT_EQ(batch_lb_iters[i]->second, serial_lb->second);
    }
  }
}

template <typename KeyT>
void TestSetDynamicInsertAndErase(size_t num_insertions) {
  absl::BitGen rng;
  std::set<KeyT> reference_set;
  BTreeSet<KeyT> dynamic_tree;

  // 1. Dynamic sequential and random insertions
  for (size_t i = 0; i < num_insertions; ++i) {
    KeyT key = static_cast<KeyT>(absl::Uniform<uint32_t>(rng, 0, 1000000));
    auto [ref_it, ref_inserted] = reference_set.insert(key);
    auto [tree_it, tree_inserted] = dynamic_tree.insert(key);

    HWY_ASSERT_EQ(tree_inserted, ref_inserted);
    HWY_ASSERT_EQ(*tree_it, key);
    HWY_ASSERT_EQ(dynamic_tree.size(), reference_set.size());
  }

  // 2. Validate all elements match reference set
  std::vector<KeyT> tree_elements(dynamic_tree.begin(), dynamic_tree.end());
  std::vector<KeyT> ref_elements(reference_set.begin(), reference_set.end());
  HWY_ASSERT_EQ(tree_elements.size(), ref_elements.size());
  for (size_t i = 0; i < ref_elements.size(); ++i) {
    HWY_ASSERT_EQ(tree_elements[i], ref_elements[i]);
    HWY_ASSERT(dynamic_tree.Contains(ref_elements[i]));
  }

  // 3. Dynamic deletions
  std::vector<KeyT> keys_to_erase = ref_elements;
  std::shuffle(keys_to_erase.begin(), keys_to_erase.end(), rng);
  const size_t erase_count = keys_to_erase.size() / 2;

  for (size_t i = 0; i < erase_count; ++i) {
    KeyT key = keys_to_erase[i];
    size_t ref_erased = reference_set.erase(key);
    size_t tree_erased = dynamic_tree.erase(key);
    HWY_ASSERT_EQ(tree_erased, ref_erased);
    HWY_ASSERT(!dynamic_tree.Contains(key));
    HWY_ASSERT_EQ(dynamic_tree.size(), reference_set.size());
  }

  // 4. Verify remaining elements
  std::vector<KeyT> remaining_tree(dynamic_tree.begin(), dynamic_tree.end());
  std::vector<KeyT> remaining_ref(reference_set.begin(), reference_set.end());
  HWY_ASSERT_EQ(remaining_tree.size(), remaining_ref.size());
  for (size_t i = 0; i < remaining_ref.size(); ++i) {
    HWY_ASSERT_EQ(remaining_tree[i], remaining_ref[i]);
  }
}

template <typename KeyT, typename ValueT>
void TestMapDynamicInsertAndErase(size_t num_insertions) {
  absl::BitGen rng;
  absl::btree_map<KeyT, ValueT> reference_map;
  BTreeMap<KeyT, ValueT> dynamic_map;

  for (size_t i = 0; i < num_insertions; ++i) {
    KeyT key = static_cast<KeyT>(absl::Uniform<uint32_t>(rng, 0, 1000000));
    ValueT val = static_cast<ValueT>(key * 3 + 7);

    auto [ref_it, ref_inserted] = reference_map.insert({key, val});
    auto [tree_it, tree_inserted] = dynamic_map.insert(key, val);

    HWY_ASSERT_EQ(tree_inserted, ref_inserted);
    HWY_ASSERT_EQ(tree_it->first, key);
    HWY_ASSERT_EQ(tree_it->second, ref_it->second);
    HWY_ASSERT_EQ(dynamic_map.size(), reference_map.size());
  }

  // Verify operator[]
  for (const auto& [k, v] : reference_map) {
    HWY_ASSERT(dynamic_map.Contains(k));
    const ValueT* val_ptr = dynamic_map.FindValue(k);
    HWY_ASSERT(val_ptr != nullptr);
    HWY_ASSERT_EQ(*val_ptr, v);
  }

  // Test erase
  std::vector<KeyT> keys_to_erase;
  for (const auto& [k, v] : reference_map) keys_to_erase.push_back(k);
  std::shuffle(keys_to_erase.begin(), keys_to_erase.end(), rng);
  const size_t erase_count = keys_to_erase.size() / 2;

  for (size_t i = 0; i < erase_count; ++i) {
    KeyT key = keys_to_erase[i];
    size_t ref_erased = reference_map.erase(key);
    size_t tree_erased = dynamic_map.erase(key);
    HWY_ASSERT_EQ(tree_erased, ref_erased);
    HWY_ASSERT(!dynamic_map.Contains(key));
    HWY_ASSERT_EQ(dynamic_map.size(), reference_map.size());
  }
}

template <typename KeyT>
void TestSlackFillRatio() {
  std::vector<KeyT> keys = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  auto tree = BTreeSet<KeyT>::Build(keys.data(), keys.size(),
                                    /*fill_ratio=*/0.75f);
  HWY_ASSERT_EQ(tree.size(), keys.size());

  // Insert intermediate keys without split
  tree.insert(15);
  tree.insert(25);
  tree.insert(35);
  HWY_ASSERT_EQ(tree.size(), 13);
  HWY_ASSERT(tree.Contains(15));
  HWY_ASSERT(tree.Contains(25));
  HWY_ASSERT(tree.Contains(35));
}

template <typename KeyT>
void TestSetSTLInterfaceAndReverseIterators() {
  using SetType = BTreeSet<KeyT>;
  static_assert(std::is_same_v<typename SetType::key_type, KeyT>);
  static_assert(std::is_same_v<typename SetType::value_type, KeyT>);
  static_assert(std::is_same_v<typename SetType::size_type, size_t>);
  static_assert(
      std::is_same_v<typename SetType::difference_type, std::ptrdiff_t>);
  static_assert(std::is_same_v<typename SetType::reference, const KeyT&>);
  static_assert(std::is_same_v<typename SetType::const_reference, const KeyT&>);
  static_assert(std::is_same_v<typename SetType::pointer, const KeyT*>);
  static_assert(std::is_same_v<typename SetType::const_pointer, const KeyT*>);

  // 1. Empty tree
  SetType empty_tree;
  HWY_ASSERT(empty_tree.rbegin() == empty_tree.rend());
  HWY_ASSERT(empty_tree.crbegin() == empty_tree.crend());
  HWY_ASSERT(!empty_tree.contains(10));
  HWY_ASSERT_EQ(empty_tree.count(10), size_t{0});
  auto empty_range = empty_tree.equal_range(10);
  HWY_ASSERT(empty_range.first == empty_tree.end());
  HWY_ASSERT(empty_range.second == empty_tree.end());

  // 2. Populated tree with 1,000 keys
  std::vector<KeyT> keys;
  const size_t N = 1000;
  keys.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    keys.push_back(static_cast<KeyT>((i + 1) * 10));
  }

  auto tree = SetType::Build(keys.data(), keys.size());

  // contains and count
  for (KeyT k : keys) {
    HWY_ASSERT(tree.contains(k));
    HWY_ASSERT_EQ(tree.count(k), size_t{1});
    HWY_ASSERT(!tree.contains(static_cast<KeyT>(k + 1)));
    HWY_ASSERT_EQ(tree.count(static_cast<KeyT>(k + 1)), size_t{0});
  }

  // equal_range
  for (size_t i = 0; i < keys.size(); ++i) {
    KeyT k = keys[i];
    auto range = tree.equal_range(k);
    HWY_ASSERT(range.first != tree.end());
    HWY_ASSERT_EQ(*range.first, k);
    if (i + 1 < keys.size()) {
      HWY_ASSERT(range.second != tree.end());
      HWY_ASSERT_EQ(*range.second, keys[i + 1]);
    } else {
      HWY_ASSERT(range.second == tree.end());
    }

    // Between keys: equal_range(k + 5)
    auto mid_range = tree.equal_range(static_cast<KeyT>(k + 5));
    if (i + 1 < keys.size()) {
      HWY_ASSERT(mid_range.first == mid_range.second);
      HWY_ASSERT_EQ(*mid_range.first, keys[i + 1]);
    }
  }

  // Reverse iteration
  std::vector<KeyT> rev_traversed(tree.rbegin(), tree.rend());
  std::vector<KeyT> expected_rev = keys;
  std::reverse(expected_rev.begin(), expected_rev.end());
  HWY_ASSERT(rev_traversed == expected_rev);

  std::vector<KeyT> crev_traversed(tree.crbegin(), tree.crend());
  HWY_ASSERT(crev_traversed == expected_rev);

  // Reverse iteration after dynamic mutations
  tree.insert(5);
  tree.insert(15);
  tree.erase(keys[0]);      // Erase 10
  expected_rev.pop_back();  // Removed 10
  expected_rev.push_back(15);
  expected_rev.push_back(5);

  std::vector<KeyT> dynamic_rev(tree.rbegin(), tree.rend());
  HWY_ASSERT(dynamic_rev == expected_rev);
}

template <typename KeyT, typename ValueT>
void TestMapSTLInterfaceAndReverseIterators() {
  using MapType = BTreeMap<KeyT, ValueT>;
  static_assert(std::is_same_v<typename MapType::key_type, KeyT>);
  static_assert(std::is_same_v<typename MapType::mapped_type, ValueT>);
  static_assert(std::is_same_v<typename MapType::value_type,
                               std::pair<const KeyT, ValueT> >);
  static_assert(std::is_same_v<typename MapType::size_type, size_t>);
  static_assert(
      std::is_same_v<typename MapType::difference_type, std::ptrdiff_t>);

  // 1. Empty map
  MapType empty_map;
  HWY_ASSERT(empty_map.rbegin() == empty_map.rend());
  HWY_ASSERT(empty_map.crbegin() == empty_map.crend());
  HWY_ASSERT(!empty_map.contains(10));
  HWY_ASSERT_EQ(empty_map.count(10), size_t{0});
  auto empty_range = empty_map.equal_range(10);
  HWY_ASSERT(empty_range.first == empty_map.end());
  HWY_ASSERT(empty_range.second == empty_map.end());

  // 2. Populated map with 1,000 pairs
  std::vector<KeyT> keys;
  std::vector<ValueT> vals;
  const size_t N = 1000;
  keys.reserve(N);
  vals.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    keys.push_back(static_cast<KeyT>((i + 1) * 10));
    vals.push_back(static_cast<ValueT>((i + 1) * 100));
  }

  auto map = MapType::Build(keys.data(), vals.data(), keys.size());

  // contains and count
  for (size_t i = 0; i < keys.size(); ++i) {
    KeyT k = keys[i];
    HWY_ASSERT(map.contains(k));
    HWY_ASSERT_EQ(map.count(k), size_t{1});
    HWY_ASSERT(!map.contains(static_cast<KeyT>(k + 1)));
    HWY_ASSERT_EQ(map.count(static_cast<KeyT>(k + 1)), size_t{0});
  }

  // equal_range
  for (size_t i = 0; i < keys.size(); ++i) {
    KeyT k = keys[i];
    auto range = map.equal_range(k);
    HWY_ASSERT(range.first != map.end());
    HWY_ASSERT_EQ(range.first->first, k);
    HWY_ASSERT_EQ(range.first->second, vals[i]);
    if (i + 1 < keys.size()) {
      HWY_ASSERT(range.second != map.end());
      HWY_ASSERT_EQ(range.second->first, keys[i + 1]);
    } else {
      HWY_ASSERT(range.second == map.end());
    }
  }

  // Reverse iteration
  std::vector<KeyT> rev_keys;
  std::vector<ValueT> rev_vals;
  for (auto it = map.rbegin(); it != map.rend(); ++it) {
    rev_keys.push_back(it->first);
    rev_vals.push_back(it->second);
  }

  std::vector<KeyT> expected_keys = keys;
  std::vector<ValueT> expected_vals = vals;
  std::reverse(expected_keys.begin(), expected_keys.end());
  std::reverse(expected_vals.begin(), expected_vals.end());
  HWY_ASSERT(rev_keys == expected_keys);
  HWY_ASSERT(rev_vals == expected_vals);

  // Const reverse iteration with crbegin/crend
  std::vector<KeyT> crev_keys;
  for (auto it = map.crbegin(); it != map.crend(); ++it) {
    crev_keys.push_back((*it).first);
  }
  HWY_ASSERT(crev_keys == expected_keys);

  // Reverse iteration after dynamic mutations
  map.insert(5, static_cast<ValueT>(50));
  map.insert(15, static_cast<ValueT>(150));
  map.erase(keys[0]);        // Erase 10
  expected_keys.pop_back();  // Removed 10
  expected_keys.push_back(15);
  expected_keys.push_back(5);

  std::vector<KeyT> dynamic_rev_keys;
  for (auto it = map.rbegin(); it != map.rend(); ++it) {
    dynamic_rev_keys.push_back(it->first);
  }
  HWY_ASSERT(dynamic_rev_keys == expected_keys);
}

template <typename KeyT>
void TestSetCopySemantics() {
  // 1. Copy empty tree
  BTreeSet<KeyT> empty_tree;
  BTreeSet<KeyT> empty_copy(empty_tree);
  HWY_ASSERT(empty_copy.empty());
  HWY_ASSERT_EQ(empty_copy.size(), size_t{0});

  BTreeSet<KeyT> empty_assigned;
  empty_assigned = empty_tree;
  HWY_ASSERT(empty_assigned.empty());

  // 2. Copy populated tree
  std::vector<KeyT> keys;
  const size_t N = 1000;
  keys.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    keys.push_back(static_cast<KeyT>((i + 1) * 10));
  }
  auto original = BTreeSet<KeyT>::Build(keys.data(), keys.size());

  // Copy constructor
  BTreeSet<KeyT> copy(original);
  HWY_ASSERT_EQ(copy.size(), original.size());
  HWY_ASSERT_EQ(copy.height(), original.height());
  for (KeyT k : keys) {
    HWY_ASSERT(copy.contains(k));
  }

  // Pointer independence: mutate copy, ensure original is unmodified
  copy.insert(5);
  copy.erase(keys[0]);
  HWY_ASSERT_EQ(copy.size(), keys.size());
  HWY_ASSERT(copy.contains(5));
  HWY_ASSERT(!copy.contains(keys[0]));

  HWY_ASSERT_EQ(original.size(), keys.size());
  HWY_ASSERT(!original.contains(5));
  HWY_ASSERT(original.contains(keys[0]));

  // Copy assignment: self-assignment
  original = original;
  HWY_ASSERT_EQ(original.size(), keys.size());
  HWY_ASSERT(original.contains(keys[0]));

  // Copy assignment: overwrite smaller tree with larger tree
  BTreeSet<KeyT> small_tree;
  small_tree.insert(100);
  small_tree = original;
  HWY_ASSERT_EQ(small_tree.size(), original.size());
  for (KeyT k : keys) {
    HWY_ASSERT(small_tree.contains(k));
  }

  // Copy assignment: overwrite larger tree with smaller tree
  std::vector<KeyT> tiny_keys = {1, 2, 3};
  auto tiny_tree = BTreeSet<KeyT>::Build(tiny_keys.data(), tiny_keys.size());
  small_tree = tiny_tree;
  HWY_ASSERT_EQ(small_tree.size(), size_t{3});
  for (KeyT k : tiny_keys) {
    HWY_ASSERT(small_tree.contains(k));
  }
}

template <typename KeyT, typename ValueT>
void TestMapCopySemantics() {
  // 1. Copy empty map
  BTreeMap<KeyT, ValueT> empty_map;
  BTreeMap<KeyT, ValueT> empty_copy(empty_map);
  HWY_ASSERT(empty_copy.empty());
  HWY_ASSERT_EQ(empty_copy.size(), size_t{0});

  BTreeMap<KeyT, ValueT> empty_assigned;
  empty_assigned = empty_map;
  HWY_ASSERT(empty_assigned.empty());

  // 2. Copy populated map
  std::vector<KeyT> keys;
  std::vector<ValueT> vals;
  const size_t N = 1000;
  keys.reserve(N);
  vals.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    keys.push_back(static_cast<KeyT>((i + 1) * 10));
    vals.push_back(static_cast<ValueT>((i + 1) * 100));
  }
  auto original =
      BTreeMap<KeyT, ValueT>::Build(keys.data(), vals.data(), keys.size());

  // Copy constructor
  BTreeMap<KeyT, ValueT> copy(original);
  HWY_ASSERT_EQ(copy.size(), original.size());
  HWY_ASSERT_EQ(copy.height(), original.height());
  for (size_t i = 0; i < keys.size(); ++i) {
    const ValueT* v = copy.FindValue(keys[i]);
    HWY_ASSERT(v != nullptr);
    HWY_ASSERT_EQ(*v, vals[i]);
  }

  // Pointer independence: mutate copy, ensure original is unmodified
  copy.insert(5, static_cast<ValueT>(50));
  copy.erase(keys[0]);
  HWY_ASSERT_EQ(copy.size(), keys.size());
  HWY_ASSERT(copy.contains(5));
  HWY_ASSERT(!copy.contains(keys[0]));

  HWY_ASSERT_EQ(original.size(), keys.size());
  HWY_ASSERT(!original.contains(5));
  HWY_ASSERT(original.contains(keys[0]));

  // Copy assignment: self-assignment
  original = original;
  HWY_ASSERT_EQ(original.size(), keys.size());
  HWY_ASSERT(original.contains(keys[0]));

  // Copy assignment: overwrite smaller map with larger map
  BTreeMap<KeyT, ValueT> small_map;
  small_map.insert(100, static_cast<ValueT>(1000));
  small_map = original;
  HWY_ASSERT_EQ(small_map.size(), original.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    const ValueT* v = small_map.FindValue(keys[i]);
    HWY_ASSERT(v != nullptr);
    HWY_ASSERT_EQ(*v, vals[i]);
  }
}

void TestAll() {
  fprintf(stderr, "Running Set 32-bit tests...\n");
  TestEmptyTree<uint32_t>();
  TestSingleLeaf<uint32_t>();
  TestMultiLevelTree<uint32_t>(100);
  TestMultiLevelTree<uint32_t>(10000);
  TestMultiLevelTree<uint32_t>(100000);
  TestMoveSemantics<uint32_t>();
  TestSetCopySemantics<uint32_t>();
  TestRandomizedComparisonAgainstStdSet<uint32_t>(10000, 2000);
  TestBatchQueries<uint32_t>(10000, 2500);
  TestSetDynamicInsertAndErase<uint32_t>(5000);
  TestSlackFillRatio<uint32_t>();
  TestSetSTLInterfaceAndReverseIterators<uint32_t>();

  fprintf(stderr, "Running Set signed 32-bit tests...\n");
  TestSignedKeys<int32_t>();
  TestMoveSemantics<int32_t>();
  TestSetCopySemantics<int32_t>();
  TestRandomizedComparisonAgainstStdSet<int32_t>(10000, 2000);
  TestBatchQueries<int32_t>(10000, 2500);
  TestSetSTLInterfaceAndReverseIterators<int32_t>();

  fprintf(stderr, "Running Set 64-bit tests...\n");
  TestEmptyTree<uint64_t>();
  TestSingleLeaf<uint64_t>();
  TestMultiLevelTree<uint64_t>(100);
  TestMultiLevelTree<uint64_t>(10000);
  TestMoveSemantics<uint64_t>();
  TestSetCopySemantics<uint64_t>();
  TestRandomizedComparisonAgainstStdSet<uint64_t>(10000, 2000);
  TestBatchQueries<uint64_t>(10000, 2500);
  TestSetDynamicInsertAndErase<uint64_t>(5000);
  TestSlackFillRatio<uint64_t>();
  TestSetSTLInterfaceAndReverseIterators<uint64_t>();

  fprintf(stderr, "Running Set signed 64-bit tests...\n");
  TestSignedKeys<int64_t>();
  TestMoveSemantics<int64_t>();
  TestSetCopySemantics<int64_t>();
  TestRandomizedComparisonAgainstStdSet<int64_t>(10000, 2000);
  TestBatchQueries<int64_t>(10000, 2500);
  TestSetSTLInterfaceAndReverseIterators<int64_t>();

  fprintf(stderr, "Running Map uint32_t -> uint64_t tests...\n");
  TestMapEmpty<uint32_t, uint64_t>();
  TestMapSingleLeaf<uint32_t, uint64_t>();
  TestMapMultiLevel<uint32_t, uint64_t>(100);
  TestMapMultiLevel<uint32_t, uint64_t>(10000);
  TestMapMoveSemantics<uint32_t, uint64_t>();
  TestMapCopySemantics<uint32_t, uint64_t>();
  TestMapRandomizedComparisonAgainstAbsl<uint32_t, uint64_t>(10000, 2000);
  TestMapBatchQueries<uint32_t, uint64_t>(10000, 2500);
  TestMapDynamicInsertAndErase<uint32_t, uint64_t>(5000);
  TestMapSTLInterfaceAndReverseIterators<uint32_t, uint64_t>();

  fprintf(stderr, "Running Map uint64_t -> double tests...\n");
  TestMapEmpty<uint64_t, double>();
  TestMapSingleLeaf<uint64_t, double>();
  TestMapMultiLevel<uint64_t, double>(100);
  TestMapMultiLevel<uint64_t, double>(10000);
  TestMapMoveSemantics<uint64_t, double>();
  TestMapCopySemantics<uint64_t, double>();
  TestMapRandomizedComparisonAgainstAbsl<uint64_t, double>(10000, 2000);
  TestMapBatchQueries<uint64_t, double>(10000, 2500);
  TestMapDynamicInsertAndErase<uint64_t, double>(5000);
  TestMapSTLInterfaceAndReverseIterators<uint64_t, double>();
  fprintf(stderr, "All tests passed!\n");
}

#endif  // (HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128) && !HWY_IDE

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
HWY_BEFORE_TEST(BTreeTest);
HWY_EXPORT_AND_TEST_P(BTreeTest, TestAll);
HWY_AFTER_TEST();
}  // namespace hwy
HWY_TEST_MAIN();

#endif  // HWY_ONCE
