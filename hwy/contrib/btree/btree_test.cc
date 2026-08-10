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
