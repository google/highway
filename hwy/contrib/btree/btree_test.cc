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

#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

#define HWY_HAVE_ABSL 0
#if HWY_HAVE_ABSL
#include "third_party/absl/container/btree_map.h"
#include "third_party/absl/container/btree_set.h"
#include "third_party/absl/random/random.h"
#endif
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
  HWY_ASSERT(!tree.contains(10));
  HWY_ASSERT(tree.find(10) == tree.end());
  HWY_ASSERT(tree.lower_bound(10) == tree.end());
  HWY_ASSERT(tree.begin() == tree.end());
}

template <typename KeyT>
void TestSingleLeaf() {
  std::vector<KeyT> keys = {10, 20, 30, 40, 50};
  auto tree = BTreeSet<KeyT>::Build(keys.data(), keys.size());
  HWY_ASSERT(!tree.empty());
  HWY_ASSERT_EQ(tree.size(), size_t{5});
  HWY_ASSERT_EQ(tree.height(), uint16_t{0});

  for (KeyT k : keys) {
    HWY_ASSERT(tree.contains(k));
    auto it = tree.find(k);
    HWY_ASSERT(it != tree.end());
    HWY_ASSERT_EQ(*it, k);
  }
  HWY_ASSERT(!tree.contains(15));
  HWY_ASSERT(tree.find(15) == tree.end());

  auto lb = tree.lower_bound(25);
  HWY_ASSERT(lb != tree.end());
  HWY_ASSERT_EQ(*lb, 30);

  size_t idx = 0;
  for (auto it = tree.begin(); it != tree.end(); ++it) {
    HWY_ASSERT_EQ(*it, keys[idx++]);
  }
  HWY_ASSERT_EQ(idx, size_t{5});
}

template <typename KeyT>
void TestRandomizedComparisonAgainstAbsl(size_t n, size_t num_queries) {
  absl::BitGen bitgen;
  std::set<KeyT> unique_keys;
  for (size_t i = 0; i < n; ++i) {
    unique_keys.insert(
        static_cast<KeyT>(absl::Uniform<KeyT>(bitgen, 0, n * 50)));
  }
  std::vector<KeyT> sorted_keys(unique_keys.begin(), unique_keys.end());
  absl::btree_set<KeyT> absl_tree(sorted_keys.begin(), sorted_keys.end());
  auto tree = BTreeSet<KeyT>::Build(sorted_keys.data(), sorted_keys.size());

  HWY_ASSERT_EQ(tree.size(), absl_tree.size());

  for (size_t q = 0; q < num_queries; ++q) {
    KeyT query_key = static_cast<KeyT>(absl::Uniform<KeyT>(bitgen, 0, n * 60));
    bool absl_contains = absl_tree.contains(query_key);
    bool hwy_contains = tree.contains(query_key);
    HWY_ASSERT_EQ(hwy_contains, absl_contains);

    auto hwy_it = tree.find(query_key);
    if (absl_contains) {
      HWY_ASSERT(hwy_it != tree.end());
      HWY_ASSERT_EQ(*hwy_it, query_key);
    } else {
      HWY_ASSERT(hwy_it == tree.end());
    }

    auto absl_lb = absl_tree.lower_bound(query_key);
    auto hwy_lb = tree.lower_bound(query_key);
    if (absl_lb == absl_tree.end()) {
      HWY_ASSERT(hwy_lb == tree.end());
    } else {
      HWY_ASSERT(hwy_lb != tree.end());
      HWY_ASSERT_EQ(*hwy_lb, *absl_lb);
    }
  }

  auto absl_it = absl_tree.begin();
  auto hwy_it = tree.begin();
  while (absl_it != absl_tree.end()) {
    HWY_ASSERT(hwy_it != tree.end());
    HWY_ASSERT_EQ(*hwy_it, *absl_it);
    ++absl_it;
    ++hwy_it;
  }
  HWY_ASSERT(hwy_it == tree.end());
}

template <typename KeyT>
void TestBatchQueries(size_t n, size_t num_queries) {
  absl::BitGen bitgen;
  std::set<KeyT> unique_keys;
  for (size_t i = 0; i < n; ++i) {
    unique_keys.insert(
        static_cast<KeyT>(absl::Uniform<KeyT>(bitgen, 0, n * 50)));
  }
  std::vector<KeyT> sorted_keys(unique_keys.begin(), unique_keys.end());
  absl::btree_set<KeyT> absl_tree(sorted_keys.begin(), sorted_keys.end());
  auto tree = BTreeSet<KeyT>::Build(sorted_keys.data(), sorted_keys.size());

  std::vector<KeyT> queries(num_queries);
  for (size_t i = 0; i < num_queries; ++i) {
    queries[i] = static_cast<KeyT>(absl::Uniform<KeyT>(bitgen, 0, n * 60));
  }

  std::vector<uint8_t> batch_found(num_queries);
  tree.ContainsBatch(queries.data(), num_queries,
                     reinterpret_cast<bool*>(batch_found.data()));
  for (size_t i = 0; i < num_queries; ++i) {
    const bool absl_found = (absl_tree.find(queries[i]) != absl_tree.end());
    HWY_ASSERT_EQ(static_cast<bool>(batch_found[i]), absl_found);
  }

  std::vector<typename BTreeSet<KeyT>::const_iterator> batch_results(
      num_queries);
  tree.LowerBoundBatch(queries.data(), num_queries, batch_results.data());

  for (size_t i = 0; i < num_queries; ++i) {
    auto absl_lb = absl_tree.lower_bound(queries[i]);
    if (absl_lb == absl_tree.end()) {
      HWY_ASSERT(batch_results[i] == tree.end());
    } else {
      HWY_ASSERT(batch_results[i] != tree.end());
      HWY_ASSERT_EQ(*batch_results[i], *absl_lb);
    }
  }
}

template <typename KeyT>
void TestDiverseBitModes() {
  std::vector<KeyT> dense_keys;
  for (size_t i = 0; i < 500; ++i) {
    dense_keys.push_back(static_cast<KeyT>(i * 2 + 10));
  }
  auto dense_tree = BTreeSet<KeyT>::Build(dense_keys.data(), dense_keys.size());
  for (KeyT k : dense_keys) {
    HWY_ASSERT(dense_tree.contains(k));
    HWY_ASSERT_EQ(*dense_tree.lower_bound(k), k);
  }

  std::vector<KeyT> sparse_keys;
  for (size_t i = 0; i < 200; ++i) {
    if constexpr (sizeof(KeyT) == 4) {
      sparse_keys.push_back(static_cast<KeyT>(i * 10000000U + 500));
    } else {
      sparse_keys.push_back(
          static_cast<KeyT>(static_cast<uint64_t>(i) * 10000000000ULL + 500));
    }
  }
  auto sparse_tree =
      BTreeSet<KeyT>::Build(sparse_keys.data(), sparse_keys.size());
  for (KeyT k : sparse_keys) {
    HWY_ASSERT(sparse_tree.contains(k));
    HWY_ASSERT_EQ(*sparse_tree.lower_bound(k), k);
  }
}

template <typename KeyT>
void TestDynamicInsertAndErase(size_t num_mutations) {
  absl::BitGen bitgen;
  absl::btree_set<KeyT> reference_set;
  BTreeSet<KeyT> tree;

  std::vector<KeyT> inserted_keys;
  inserted_keys.reserve(num_mutations);

  // 1. Dynamic Insertions from Empty Tree
  for (size_t i = 0; i < num_mutations; ++i) {
    KeyT k = static_cast<KeyT>(absl::Uniform<uint64_t>(bitgen, 1, 50000000));
    auto ref_res = reference_set.insert(k);
    auto res = tree.insert(k);

    HWY_ASSERT_EQ(res.second, ref_res.second);
    HWY_ASSERT_EQ(*res.first, k);
    HWY_ASSERT_EQ(tree.size(), reference_set.size());
    if (ref_res.second) {
      inserted_keys.push_back(k);
    }
  }

  // 2. Full In-Order Traversal Verification vs std::set
  HWY_ASSERT(std::equal(tree.begin(), tree.end(), reference_set.begin(),
                        reference_set.end()));

  // 3. Verification of Contains & LowerBound across all inserted keys
  for (KeyT k : inserted_keys) {
    HWY_ASSERT(tree.contains(k));
    auto it = tree.find(k);
    HWY_ASSERT(it != tree.end());
    HWY_ASSERT_EQ(*it, k);
    HWY_ASSERT_EQ(*tree.lower_bound(k), k);
  }

  // 4. Random Query Verification
  for (size_t i = 0; i < 2000; ++i) {
    KeyT q = static_cast<KeyT>(absl::Uniform<uint64_t>(bitgen, 0, 50000050));
    bool expected_contains = (reference_set.find(q) != reference_set.end());
    HWY_ASSERT_EQ(tree.contains(q), expected_contains);
    HWY_ASSERT_EQ(tree.find(q) != tree.end(), expected_contains);

    auto ref_lb = reference_set.lower_bound(q);
    auto lb = tree.lower_bound(q);
    if (ref_lb == reference_set.end()) {
      HWY_ASSERT(lb == tree.end());
    } else {
      HWY_ASSERT(lb != tree.end());
      HWY_ASSERT_EQ(*lb, *ref_lb);
    }
  }

  // 5. Dynamic Deletions (Erase Half the Keys)
  std::shuffle(inserted_keys.begin(), inserted_keys.end(), bitgen);
  size_t to_delete = inserted_keys.size() / 2;
  for (size_t i = 0; i < to_delete; ++i) {
    KeyT k = inserted_keys[i];
    size_t ref_erased = reference_set.erase(k);
    size_t erased = tree.erase(k);

    HWY_ASSERT_EQ(erased, ref_erased);
    HWY_ASSERT_EQ(tree.size(), reference_set.size());
    HWY_ASSERT(!tree.contains(k));
  }

  // 6. In-Order Traversal Check After Deletions
  HWY_ASSERT(std::equal(tree.begin(), tree.end(), reference_set.begin(),
                        reference_set.end()));

  // 7. Verify Non-Deleted Keys
  for (size_t i = to_delete; i < inserted_keys.size(); ++i) {
    KeyT k = inserted_keys[i];
    HWY_ASSERT(tree.contains(k));
    HWY_ASSERT_EQ(*tree.lower_bound(k), k);
  }

  // 8. Dynamic Re-insertions
  for (size_t i = 0; i < to_delete / 2; ++i) {
    KeyT k = inserted_keys[i];
    reference_set.insert(k);
    auto res = tree.insert(k);
    HWY_ASSERT(res.second);
    HWY_ASSERT_EQ(tree.size(), reference_set.size());
    HWY_ASSERT(tree.contains(k));
  }
}

template <typename KeyT, typename ValueT>
void TestMapEmpty() {
  BTreeMap<KeyT, ValueT> map;
  HWY_ASSERT(map.empty());
  HWY_ASSERT_EQ(map.size(), size_t{0});
  HWY_ASSERT(map.begin() == map.end());
  HWY_ASSERT(!map.contains(10));
  HWY_ASSERT(map.find(10) == map.end());
  HWY_ASSERT(map.lower_bound(10) == map.end());
}

template <typename KeyT, typename ValueT>
void TestMapSingleLeaf() {
  std::vector<KeyT> keys = {10, 20, 30, 40, 50};
  std::vector<ValueT> vals = {100, 200, 300, 400, 500};
  auto map =
      BTreeMap<KeyT, ValueT>::Build(keys.data(), vals.data(), keys.size());

  HWY_ASSERT(!map.empty());
  HWY_ASSERT_EQ(map.size(), size_t{5});
  HWY_ASSERT_EQ(map.height(), 0);

  for (size_t i = 0; i < keys.size(); ++i) {
    HWY_ASSERT(map.contains(keys[i]));
    auto it = map.find(keys[i]);
    HWY_ASSERT(it != map.end());
    HWY_ASSERT_EQ(it->first, keys[i]);
    HWY_ASSERT_EQ(it->second, vals[i]);
    HWY_ASSERT_EQ(map[keys[i]], vals[i]);
    HWY_ASSERT_EQ(map.at(keys[i]), vals[i]);
  }

  HWY_ASSERT(!map.contains(5));
  HWY_ASSERT(!map.contains(25));
  HWY_ASSERT(!map.contains(55));

  HWY_ASSERT_EQ(map.lower_bound(5)->first, 10);
  HWY_ASSERT_EQ(map.lower_bound(10)->first, 10);
  HWY_ASSERT_EQ(map.lower_bound(25)->first, 30);
  HWY_ASSERT_EQ(map.lower_bound(50)->first, 50);
  HWY_ASSERT(map.lower_bound(55) == map.end());
}

template <typename KeyT, typename ValueT>
void TestMapRandomizedComparisonAgainstAbsl(size_t num_keys,
                                            size_t num_queries) {
  absl::BitGen bitgen;
  std::map<KeyT, ValueT> ref_map;
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

  HWY_ASSERT_EQ(map.size(), ref_map.size());

  // Traversal check
  size_t idx = 0;
  for (auto it = map.begin(); it != map.end(); ++it, ++idx) {
    HWY_ASSERT_EQ(it->first, sorted_keys[idx]);
    HWY_ASSERT_EQ(it->second, sorted_vals[idx]);
  }
  HWY_ASSERT_EQ(idx, sorted_keys.size());

  // Random Queries
  for (size_t q = 0; q < num_queries; ++q) {
    KeyT query_key =
        static_cast<KeyT>(absl::Uniform<uint64_t>(bitgen, 0, 10000050));

    auto ref_it = ref_map.find(query_key);
    bool expected_contains = (ref_it != ref_map.end());
    HWY_ASSERT_EQ(map.contains(query_key), expected_contains);

    auto it = map.find(query_key);
    if (expected_contains) {
      HWY_ASSERT(it != map.end());
      HWY_ASSERT_EQ(it->first, ref_it->first);
      HWY_ASSERT_EQ(it->second, ref_it->second);
    } else {
      HWY_ASSERT(it == map.end());
    }

    auto ref_lb = ref_map.lower_bound(query_key);
    auto lb = map.lower_bound(query_key);
    if (ref_lb == ref_map.end()) {
      HWY_ASSERT(lb == map.end());
    } else {
      HWY_ASSERT(lb != map.end());
      HWY_ASSERT_EQ(lb->first, ref_lb->first);
      HWY_ASSERT_EQ(lb->second, ref_lb->second);
    }
  }
}

template <typename KeyT, typename ValueT>
void TestMapBatchQueries(size_t num_keys, size_t num_queries) {
  absl::BitGen bitgen;
  std::map<KeyT, ValueT> ref_map;
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

  std::vector<KeyT> queries(num_queries);
  for (size_t i = 0; i < num_queries; ++i) {
    queries[i] =
        static_cast<KeyT>(absl::Uniform<uint64_t>(bitgen, 0, 10000050));
  }

  std::unique_ptr<bool[]> out_found(new bool[num_queries]);
  map.ContainsBatch(queries.data(), num_queries, out_found.get());
  for (size_t i = 0; i < num_queries; ++i) {
    bool expected = (ref_map.find(queries[i]) != ref_map.end());
    HWY_ASSERT_EQ(out_found[i], expected);
  }

  std::vector<ValueT> out_values(num_queries);
  map.LookupBatch(queries.data(), num_queries, out_values.data(),
                  out_found.get());
  for (size_t i = 0; i < num_queries; ++i) {
    auto it = ref_map.find(queries[i]);
    if (it != ref_map.end()) {
      HWY_ASSERT(out_found[i]);
      HWY_ASSERT_EQ(out_values[i], it->second);
    } else {
      HWY_ASSERT(!out_found[i]);
    }
  }
}

template <typename KeyT, typename ValueT>
void TestMapDynamicInsertAndErase(size_t num_mutations) {
  absl::BitGen bitgen;
  std::map<KeyT, ValueT> reference_map;
  BTreeMap<KeyT, ValueT> map;

  std::vector<KeyT> inserted_keys;
  inserted_keys.reserve(num_mutations);

  for (size_t i = 0; i < num_mutations; ++i) {
    KeyT k = static_cast<KeyT>(absl::Uniform<uint64_t>(bitgen, 1, 50000000));
    ValueT v =
        static_cast<ValueT>(absl::Uniform<uint64_t>(bitgen, 1, 50000000));

    auto ref_res = reference_map.insert({k, v});
    auto res = map.insert({k, v});

    HWY_ASSERT_EQ(res.second, ref_res.second);
    HWY_ASSERT_EQ(res.first->first, ref_res.first->first);
    HWY_ASSERT_EQ(res.first->second, ref_res.first->second);
    HWY_ASSERT_EQ(map.size(), reference_map.size());
    if (ref_res.second) {
      inserted_keys.push_back(k);
    }
  }

  for (KeyT k : inserted_keys) {
    HWY_ASSERT(map.contains(k));
    auto it = map.find(k);
    HWY_ASSERT(it != map.end());
    HWY_ASSERT_EQ(it->first, k);
    HWY_ASSERT_EQ(it->second, reference_map[k]);
  }

  std::shuffle(inserted_keys.begin(), inserted_keys.end(), bitgen);
  size_t to_delete = inserted_keys.size() / 2;
  for (size_t i = 0; i < to_delete; ++i) {
    KeyT k = inserted_keys[i];
    size_t ref_erased = reference_map.erase(k);
    size_t erased = map.erase(k);

    HWY_ASSERT_EQ(erased, ref_erased);
    HWY_ASSERT_EQ(map.size(), reference_map.size());
    HWY_ASSERT(!map.contains(k));
  }

  for (size_t i = to_delete; i < inserted_keys.size(); ++i) {
    KeyT k = inserted_keys[i];
    HWY_ASSERT(map.contains(k));
    HWY_ASSERT_EQ(map.find(k)->second, reference_map[k]);
  }
}

void TestAll() {
  fprintf(stderr, "Running BTreeSet uint32_t tests...\n");
  TestEmptyTree<uint32_t>();
  TestSingleLeaf<uint32_t>();
  TestRandomizedComparisonAgainstAbsl<uint32_t>(5000, 2000);
  TestBatchQueries<uint32_t>(5000, 1000);
  TestDiverseBitModes<uint32_t>();
  TestDynamicInsertAndErase<uint32_t>(5000);

  fprintf(stderr, "Running BTreeSet uint64_t tests...\n");
  TestEmptyTree<uint64_t>();
  TestSingleLeaf<uint64_t>();
  TestRandomizedComparisonAgainstAbsl<uint64_t>(5000, 2000);
  TestBatchQueries<uint64_t>(5000, 1000);
  TestDiverseBitModes<uint64_t>();
  TestDynamicInsertAndErase<uint64_t>(5000);

  fprintf(stderr, "Running BTreeMap uint32_t -> uint64_t tests...\n");
  TestMapEmpty<uint32_t, uint64_t>();
  TestMapSingleLeaf<uint32_t, uint64_t>();
  TestMapRandomizedComparisonAgainstAbsl<uint32_t, uint64_t>(5000, 2000);
  TestMapBatchQueries<uint32_t, uint64_t>(5000, 1000);
  TestMapDynamicInsertAndErase<uint32_t, uint64_t>(5000);

  fprintf(stderr, "Running BTreeMap uint64_t -> uint64_t tests...\n");
  TestMapEmpty<uint64_t, uint64_t>();
  TestMapSingleLeaf<uint64_t, uint64_t>();
  TestMapRandomizedComparisonAgainstAbsl<uint64_t, uint64_t>(5000, 2000);
  TestMapBatchQueries<uint64_t, uint64_t>(5000, 1000);
  TestMapDynamicInsertAndErase<uint64_t, uint64_t>(5000);

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
