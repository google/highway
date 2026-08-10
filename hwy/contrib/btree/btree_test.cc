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

  // 2. Random Query Checks (Find, Contains, LowerBound, UpperBound, Range
  // Scans)
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

void TestAll() {
  fprintf(stderr, "Running Set 32-bit tests...\n");
  TestEmptyTree<uint32_t>();
  TestSingleLeaf<uint32_t>();
  TestMultiLevelTree<uint32_t>(100);
  TestMultiLevelTree<uint32_t>(10000);
  TestMultiLevelTree<uint32_t>(100000);
  TestMoveSemantics<uint32_t>();
  TestRandomizedComparisonAgainstStdSet<uint32_t>(10000, 2000);

  fprintf(stderr, "Running Set signed 32-bit tests...\n");
  TestSignedKeys<int32_t>();
  TestMoveSemantics<int32_t>();
  TestRandomizedComparisonAgainstStdSet<int32_t>(10000, 2000);

  fprintf(stderr, "Running Set 64-bit tests...\n");
  TestEmptyTree<uint64_t>();
  TestSingleLeaf<uint64_t>();
  TestMultiLevelTree<uint64_t>(100);
  TestMultiLevelTree<uint64_t>(10000);
  TestMoveSemantics<uint64_t>();
  TestRandomizedComparisonAgainstStdSet<uint64_t>(10000, 2000);

  fprintf(stderr, "Running Set signed 64-bit tests...\n");
  TestSignedKeys<int64_t>();
  TestMoveSemantics<int64_t>();
  TestRandomizedComparisonAgainstStdSet<int64_t>(10000, 2000);

  fprintf(stderr, "Running Map uint32_t -> uint64_t tests...\n");
  TestMapEmpty<uint32_t, uint64_t>();
  TestMapSingleLeaf<uint32_t, uint64_t>();
  TestMapMultiLevel<uint32_t, uint64_t>(100);
  TestMapMultiLevel<uint32_t, uint64_t>(10000);
  TestMapMoveSemantics<uint32_t, uint64_t>();
  TestMapRandomizedComparisonAgainstAbsl<uint32_t, uint64_t>(10000, 2000);

  fprintf(stderr, "Running Map uint64_t -> double tests...\n");
  TestMapEmpty<uint64_t, double>();
  TestMapSingleLeaf<uint64_t, double>();
  TestMapMultiLevel<uint64_t, double>(100);
  TestMapMultiLevel<uint64_t, double>(10000);
  TestMapMoveSemantics<uint64_t, double>();
  TestMapRandomizedComparisonAgainstAbsl<uint64_t, double>(10000, 2000);
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
HWY_EXPORT_AND_TEST_BEST_P(BTreeTest, TestAll);
HWY_AFTER_TEST();
}  // namespace hwy
HWY_TEST_MAIN();

#endif  // HWY_ONCE
