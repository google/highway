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
#include <map>
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
#endif
#include "hwy/base.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/btree/btree_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/btree/btree-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

#if (HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128) && !HWY_IDE
HWY_NOINLINE void TestAll() {}
#else

// =============================================================================
// Unified Differential Verification Harness (similar to absl:: tests)
// =============================================================================

template <typename T, typename = void>
struct IsPairLike : std::false_type {};

template <typename T>
struct IsPairLike<T, std::void_t<decltype(std::declval<T>().first),
                                 decltype(std::declval<T>().second)> >
    : std::true_type {};

template <typename T, typename U>
void VerifyEqualValues(const T& a, const U& b) {
  if constexpr (IsPairLike<T>::value) {
    HWY_ASSERT_EQ(a.first, b.first);
    HWY_ASSERT_EQ(a.second, b.second);
  } else {
    HWY_ASSERT_EQ(a, b);
  }
}

template <typename Iter1, typename Iter2>
void VerifyEqualElements(Iter1 it1, Iter2 it2) {
  VerifyEqualValues(*it1, *it2);
}

template <typename ValueT>
struct ValueGenerator {
  uint64_t max_val;
  explicit ValueGenerator(uint64_t m) : max_val(m) {}

  ValueT operator()(uint64_t i) const {
    if constexpr (hwy::IsSameEither<ValueT, float, double>()) {
      return static_cast<ValueT>(i) * static_cast<ValueT>(0.5);
    } else {
      return static_cast<ValueT>(i);
    }
  }
};

template <typename K, typename V>
struct ValueGenerator<std::pair<K, V> > {
  uint64_t max_val;
  explicit ValueGenerator(uint64_t m) : max_val(m) {}

  std::pair<K, V> operator()(uint64_t i) const {
    K k = static_cast<K>(i);
    V v;
    if constexpr (hwy::IsSameEither<V, float, double>()) {
      v = static_cast<V>(i) * static_cast<V>(1.5);
    } else {
      v = static_cast<V>(i * 10 + 7);
    }
    return {k, v};
  }
};

template <typename V>
std::vector<V> GenerateValuesWithSeed(size_t n, uint64_t max_val,
                                      uint32_t seed) {
  if (n == 0) return {};
  hn::AesCtrEngine engine(/*deterministic=*/true);
  hn::RngStream rng(engine, seed);
  std::set<uint64_t> unique_nums;
  std::vector<uint64_t> nums;
  nums.reserve(n);
  while (nums.size() < n) {
    uint64_t val = (rng() % (max_val + 1)) + 1;
    if (unique_nums.insert(val).second) {
      nums.push_back(val);
    }
  }
  ValueGenerator<V> gen(max_val);
  std::vector<V> res;
  res.reserve(n);
  for (uint64_t x : nums) {
    res.push_back(gen(x));
  }
  return res;
}

template <typename T>
struct KeyExtractor {
  static const T& Get(const T& v) { return v; }
};

template <typename K, typename V>
struct KeyExtractor<std::pair<K, V> > {
  static const K& Get(const std::pair<K, V>& p) { return p.first; }
};

template <typename K, typename V>
struct KeyExtractor<std::pair<const K, V> > {
  static const K& Get(const std::pair<const K, V>& p) { return p.first; }
};

template <typename ValueT>
auto ExtractKey(const ValueT& v) {
  return KeyExtractor<ValueT>::Get(v);
}

template <typename ValueT>
struct ValueComparator {
  bool operator()(const ValueT& a, const ValueT& b) const { return a < b; }
};

template <typename K, typename V>
struct ValueComparator<std::pair<K, V> > {
  bool operator()(const std::pair<K, V>& a, const std::pair<K, V>& b) const {
    return a.first < b.first;
  }
};

template <typename K, typename V>
struct ValueComparator<std::pair<const K, V> > {
  bool operator()(const std::pair<const K, V>& a,
                  const std::pair<const K, V>& b) const {
    return a.first < b.first;
  }
};

template <typename ValueT>
struct ValueEquality {
  bool operator()(const ValueT& a, const ValueT& b) const { return a == b; }
};

template <typename K, typename V>
struct ValueEquality<std::pair<K, V> > {
  bool operator()(const std::pair<K, V>& a, const std::pair<K, V>& b) const {
    return a.first == b.first;
  }
};

template <typename K, typename V>
struct ValueEquality<std::pair<const K, V> > {
  bool operator()(const std::pair<const K, V>& a,
                  const std::pair<const K, V>& b) const {
    return a.first == b.first;
  }
};

template <typename ValueT>
void SortUniqueValues(std::vector<ValueT>& values) {
  std::sort(values.begin(), values.end(), ValueComparator<ValueT>{});
  values.erase(
      std::unique(values.begin(), values.end(), ValueEquality<ValueT>{}),
      values.end());
}

template <typename TreeT, typename ValueT>
TreeT BuildTreeFromValues(const std::vector<ValueT>& values,
                          float fill_ratio = 1.0f) {
  if constexpr (TreeT::kIsMap) {
    std::vector<typename TreeT::key_type> keys;
    std::vector<typename TreeT::mapped_type> vals;
    keys.reserve(values.size());
    vals.reserve(values.size());
    for (const auto& item : values) {
      keys.push_back(item.first);
      vals.push_back(item.second);
    }
    return TreeT::Build(keys.data(), vals.data(), keys.size(), fill_ratio);
  } else {
    return TreeT::Build(values.data(), values.size(), fill_ratio);
  }
}

// -----------------------------------------------------------------------------
// BTreeChecker: Continuous Invariant and Differential Verification
// -----------------------------------------------------------------------------

template <typename TreeT, typename StdRefT>
class BTreeChecker {
 public:
  using key_type = typename TreeT::key_type;
  using value_type = typename TreeT::value_type;
  using iterator = typename TreeT::iterator;
  using const_iterator = typename TreeT::const_iterator;
  using reverse_iterator = typename TreeT::reverse_iterator;
  using const_reverse_iterator = typename TreeT::const_reverse_iterator;
  static constexpr bool kIsMap = TreeT::kIsMap;

  BTreeChecker() = default;

  std::pair<iterator, bool> insert(const value_type& v) {
    const size_t prev_size = tree_.size();
    auto ref_res = ref_.insert(v);
    auto tree_res = tree_.insert(v);

    HWY_ASSERT_EQ(tree_res.second, ref_res.second);
    HWY_ASSERT_EQ(tree_.size(), ref_.size());
    HWY_ASSERT_EQ(tree_.size(), prev_size + (tree_res.second ? 1 : 0));
    if (tree_res.second) {
      VerifyEqualElements(tree_res.first, ref_res.first);
    }
    return tree_res;
  }

  size_t erase(key_type key) {
    const size_t prev_size = tree_.size();
    size_t ref_erased = ref_.erase(key);
    size_t tree_erased = tree_.erase(key);

    HWY_ASSERT_EQ(tree_erased, ref_erased);
    HWY_ASSERT_EQ(tree_.size(), ref_.size());
    HWY_ASSERT_EQ(tree_.size(), prev_size - tree_erased);
    HWY_ASSERT(!tree_.contains(key));
    return tree_erased;
  }

  void CheckLookup(key_type key) const {
    const bool expected_contains = (ref_.find(key) != ref_.end());
    HWY_ASSERT_EQ(tree_.contains(key), expected_contains);

    auto tree_find = tree_.find(key);
    auto ref_find = ref_.find(key);
    if (expected_contains) {
      HWY_ASSERT(tree_find != tree_.end());
      VerifyEqualElements(tree_find, ref_find);
    } else {
      HWY_ASSERT(tree_find == tree_.end());
    }

    auto tree_lb = tree_.lower_bound(key);
    auto ref_lb = ref_.lower_bound(key);
    if (ref_lb == ref_.end()) {
      HWY_ASSERT(tree_lb == tree_.end());
    } else {
      HWY_ASSERT(tree_lb != tree_.end());
      VerifyEqualElements(tree_lb, ref_lb);
    }

    auto tree_ub = tree_.upper_bound(key);
    auto ref_ub = ref_.upper_bound(key);
    if (ref_ub == ref_.end()) {
      HWY_ASSERT(tree_ub == tree_.end());
    } else {
      HWY_ASSERT(tree_ub != tree_.end());
      VerifyEqualElements(tree_ub, ref_ub);
    }

    if constexpr (kIsMap) {
      const auto* vp = tree_.FindValue(key);
      if (expected_contains) {
        HWY_ASSERT(vp != nullptr);
        HWY_ASSERT_EQ(*vp, ref_find->second);
        HWY_ASSERT_EQ(tree_.at(key), ref_find->second);
      } else {
        HWY_ASSERT(vp == nullptr);
      }
    }
  }

  void verify() const {
    HWY_ASSERT_EQ(tree_.size(), ref_.size());
    HWY_ASSERT_EQ(tree_.empty(), ref_.empty());

    // 1. Forward iteration
    auto ref_it = ref_.begin();
    auto tree_it = tree_.begin();
    size_t count = 0;
    for (; tree_it != tree_.end(); ++tree_it, ++ref_it, ++count) {
      HWY_ASSERT(ref_it != ref_.end());
      VerifyEqualElements(tree_it, ref_it);
    }
    HWY_ASSERT(ref_it == ref_.end());
    HWY_ASSERT_EQ(count, tree_.size());

    // 2. Reverse iteration
    auto ref_rit = ref_.rbegin();
    auto tree_rit = tree_.rbegin();
    count = 0;
    for (; tree_rit != tree_.rend(); ++tree_rit, ++ref_rit, ++count) {
      HWY_ASSERT(ref_rit != ref_.rend());
      VerifyEqualElements(tree_rit, ref_rit);
    }
    HWY_ASSERT(ref_rit == ref_.rend());
    HWY_ASSERT_EQ(count, tree_.size());

    // 3. Backward decrementing with --it from end() to begin()
    if (!tree_.empty()) {
      auto b_tree_it = tree_.end();
      auto b_ref_it = ref_.end();
      --b_tree_it;
      --b_ref_it;
      while (true) {
        VerifyEqualElements(b_tree_it, b_ref_it);
        if (b_tree_it == tree_.begin()) {
          HWY_ASSERT(b_ref_it == ref_.begin());
          break;
        }
        --b_tree_it;
        --b_ref_it;
      }
    }
  }

  void clear() {
    tree_.clear();
    ref_.clear();
    verify();
  }

  size_t size() const { return tree_.size(); }
  bool empty() const { return tree_.empty(); }

  TreeT& tree() { return tree_; }
  const TreeT& tree() const { return tree_; }
  const StdRefT& ref() const { return ref_; }

 private:
  TreeT tree_;
  StdRefT ref_;
};

// -----------------------------------------------------------------------------
// Test Suites
// -----------------------------------------------------------------------------

template <typename TreeT, typename StdRefT>
void DoFullContainerTest(const std::vector<typename TreeT::value_type>& values,
                         uint32_t seed) {
  using key_type = typename TreeT::key_type;
  static constexpr bool kIsMap = TreeT::kIsMap;

  BTreeChecker<TreeT, StdRefT> checker;
  checker.verify();
  HWY_ASSERT(checker.empty());
  HWY_ASSERT_EQ(checker.size(), size_t{0});

  // 1. Insert elements and verify invariant at each step
  std::vector<key_type> inserted_keys;
  inserted_keys.reserve(values.size());
  for (const auto& val : values) {
    auto res = checker.insert(val);
    key_type k = ExtractKey(val);
    if (res.second) {
      inserted_keys.push_back(k);
    }
    checker.CheckLookup(k);
  }
  checker.verify();

  // 2. Lookups on all inserted keys and random queries
  for (key_type k : inserted_keys) {
    checker.CheckLookup(k);
  }
  hn::AesCtrEngine engine(/*deterministic=*/true);
  hn::RngStream q_rng(engine, seed + 42);
  for (size_t i = 0; i < 500; ++i) {
    key_type q =
        static_cast<key_type>((q_rng() % (values.size() * 50 + 10)) + 1);
    checker.CheckLookup(q);
  }

  // 3. Move construction & assignment verification
  {
    TreeT moved_tree(std::move(checker.tree()));
    HWY_ASSERT(checker.tree().empty());
    HWY_ASSERT_EQ(moved_tree.size(), checker.ref().size());
    for (key_type k : inserted_keys) {
      HWY_ASSERT(moved_tree.contains(k));
    }
    checker.tree() = std::move(moved_tree);
    HWY_ASSERT(moved_tree.empty());
    HWY_ASSERT_EQ(checker.tree().size(), checker.ref().size());
  }

  // 4. Deletions (Erase half the inserted elements)
  hn::RngStream shuf_rng(engine, seed + 84);
  std::shuffle(inserted_keys.begin(), inserted_keys.end(), shuf_rng);
  size_t to_delete = inserted_keys.size() / 2;
  for (size_t i = 0; i < to_delete; ++i) {
    key_type k = inserted_keys[i];
    checker.erase(k);
    checker.CheckLookup(k);
  }
  checker.verify();

  // Non-deleted keys check
  for (size_t i = to_delete; i < inserted_keys.size(); ++i) {
    key_type k = inserted_keys[i];
    checker.CheckLookup(k);
  }

  // 5. Re-insertion of deleted elements
  for (size_t i = 0; i < to_delete; ++i) {
    key_type k = inserted_keys[i];
    if constexpr (kIsMap) {
      checker.insert({k, static_cast<typename TreeT::mapped_type>(k * 10 + 3)});
    } else {
      checker.insert(k);
    }
    checker.CheckLookup(k);
  }
  checker.verify();

  // 6. Clear
  checker.clear();
}

template <typename TreeT, typename StdRefT>
void DoBulkBuildAndBatchTest(size_t n, uint32_t seed) {
  using key_type = typename TreeT::key_type;
  using value_type = typename TreeT::value_type;
  static constexpr bool kIsMap = TreeT::kIsMap;

  auto values = GenerateValuesWithSeed<value_type>(n, n * 50, seed);
  SortUniqueValues(values);
  n = values.size();

  StdRefT ref(values.begin(), values.end());

  for (float fill_ratio : {0.5f, 0.75f, 1.0f}) {
    TreeT tree = BuildTreeFromValues<TreeT>(values, fill_ratio);
    HWY_ASSERT_EQ(tree.size(), ref.size());
    HWY_ASSERT_EQ(tree.empty(), ref.empty());

    // 1. In-order forward traversal
    auto ref_it = ref.begin();
    auto tree_it = tree.begin();
    for (; tree_it != tree.end(); ++tree_it, ++ref_it) {
      VerifyEqualElements(tree_it, ref_it);
    }
    HWY_ASSERT(ref_it == ref.end());

    // 2. Reverse traversal
    auto ref_rit = ref.rbegin();
    auto tree_rit = tree.rbegin();
    for (; tree_rit != tree.rend(); ++tree_rit, ++ref_rit) {
      VerifyEqualElements(tree_rit, ref_rit);
    }
    HWY_ASSERT(ref_rit == ref.rend());

    // 3. Backward traversal with --it
    if (!tree.empty()) {
      auto b_tree_it = tree.end();
      auto b_ref_it = ref.end();
      --b_tree_it;
      --b_ref_it;
      while (true) {
        VerifyEqualElements(b_tree_it, b_ref_it);
        if (b_tree_it == tree.begin()) break;
        --b_tree_it;
        --b_ref_it;
      }
    }

    // 4. Batch query sweeps across multiple batch sizes
    hn::AesCtrEngine engine(/*deterministic=*/true);
    hn::RngStream q_rng(engine, seed + 100);
    for (size_t batch_sz :
         {size_t{0}, size_t{1}, size_t{7}, size_t{8}, size_t{9}, size_t{15},
          size_t{16}, size_t{17}, size_t{64}, size_t{256}}) {
      std::vector<key_type> queries(batch_sz);
      for (size_t i = 0; i < batch_sz; ++i) {
        queries[i] = static_cast<key_type>((q_rng() % (n * 60 + 10)) + 1);
      }

      std::vector<uint8_t> found(batch_sz);
      tree.ContainsBatch(queries.data(), batch_sz,
                         reinterpret_cast<bool*>(found.data()));
      for (size_t i = 0; i < batch_sz; ++i) {
        bool expected = (ref.find(queries[i]) != ref.end());
        HWY_ASSERT_EQ(static_cast<bool>(found[i]), expected);
      }

      std::vector<typename TreeT::const_iterator> batch_lb(batch_sz);
      tree.LowerBoundBatch(queries.data(), batch_sz, batch_lb.data());
      for (size_t i = 0; i < batch_sz; ++i) {
        auto ref_lb = ref.lower_bound(queries[i]);
        if (ref_lb == ref.end()) {
          HWY_ASSERT(batch_lb[i] == tree.end());
        } else {
          HWY_ASSERT(batch_lb[i] != tree.end());
          VerifyEqualElements(batch_lb[i], ref_lb);
        }
      }

      std::vector<typename TreeT::const_iterator> batch_find(batch_sz);
      tree.FindBatch(queries.data(), batch_sz, batch_find.data());
      for (size_t i = 0; i < batch_sz; ++i) {
        auto ref_find = ref.find(queries[i]);
        if (ref_find == ref.end()) {
          HWY_ASSERT(batch_find[i] == tree.end());
        } else {
          HWY_ASSERT(batch_find[i] != tree.end());
          VerifyEqualElements(batch_find[i], ref_find);
        }
      }

      if constexpr (kIsMap) {
        using mapped_type = typename TreeT::mapped_type;
        std::vector<mapped_type> out_values(batch_sz);
        std::vector<uint8_t> out_found(batch_sz);
        tree.LookupBatch(queries.data(), batch_sz, out_values.data(),
                         reinterpret_cast<bool*>(out_found.data()));
        for (size_t i = 0; i < batch_sz; ++i) {
          auto ref_it_val = ref.find(queries[i]);
          if (ref_it_val != ref.end()) {
            HWY_ASSERT(out_found[i]);
            HWY_ASSERT_EQ(out_values[i], ref_it_val->second);
          } else {
            HWY_ASSERT(!out_found[i]);
          }
        }
      }
    }
  }
}

template <typename TreeT, typename StdRefT>
void DoBoundarySizeSweep() {
  using value_type = typename TreeT::value_type;
  for (size_t n : {size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{5},
                   size_t{16}, size_t{31}, size_t{32}, size_t{63}, size_t{64},
                   size_t{127}, size_t{128}, size_t{243}, size_t{244},
                   size_t{245}, size_t{488}, size_t{489}, size_t{500}}) {
    auto values = GenerateValuesWithSeed<value_type>(
        n, std::max<uint64_t>(100, n * 20), /*seed=*/12345 + n);
    DoFullContainerTest<TreeT, StdRefT>(values, /*seed=*/12345 + n);
    DoBulkBuildAndBatchTest<TreeT, StdRefT>(n, /*seed=*/54321 + n);
  }
}

template <typename TreeT>
void DoDiverseBitModesTest() {
  using key_type = typename TreeT::key_type;
  using value_type = typename TreeT::value_type;

  // Dense 8-bit mode
  {
    std::vector<value_type> dense_vals;
    for (size_t i = 0; i < 500; ++i) {
      if constexpr (TreeT::kIsMap) {
        dense_vals.push_back({static_cast<key_type>(i * 2 + 10),
                              static_cast<typename TreeT::mapped_type>(i)});
      } else {
        dense_vals.push_back(static_cast<key_type>(i * 2 + 10));
      }
    }
    TreeT tree = BuildTreeFromValues<TreeT>(dense_vals);
    for (const auto& v : dense_vals) {
      HWY_ASSERT(tree.contains(ExtractKey(v)));
    }
  }

  // Medium 16-bit mode
  {
    std::vector<value_type> med_vals;
    for (size_t i = 0; i < 500; ++i) {
      if constexpr (TreeT::kIsMap) {
        med_vals.push_back({static_cast<key_type>(i * 100 + 10),
                            static_cast<typename TreeT::mapped_type>(i)});
      } else {
        med_vals.push_back(static_cast<key_type>(i * 100 + 10));
      }
    }
    TreeT tree = BuildTreeFromValues<TreeT>(med_vals);
    for (const auto& v : med_vals) {
      HWY_ASSERT(tree.contains(ExtractKey(v)));
    }
  }

  // Sparse 32-bit / raw 64-bit mode
  {
    std::vector<value_type> sparse_vals;
    for (size_t i = 0; i < 200; ++i) {
      key_type k = (sizeof(key_type) == 4)
                       ? static_cast<key_type>(i * 10000000U + 500)
                       : static_cast<key_type>(
                             static_cast<uint64_t>(i) * 10000000000ULL + 500);
      if constexpr (TreeT::kIsMap) {
        sparse_vals.push_back(
            {k, static_cast<typename TreeT::mapped_type>(i * 3 + 1)});
      } else {
        sparse_vals.push_back(k);
      }
    }
    TreeT tree = BuildTreeFromValues<TreeT>(sparse_vals);
    for (const auto& v : sparse_vals) {
      HWY_ASSERT(tree.contains(ExtractKey(v)));
    }
  }
}

template <typename TreeT, typename StdRefT>
void RunFullTestSuite() {
  DoBoundarySizeSweep<TreeT, StdRefT>();
  DoDiverseBitModesTest<TreeT>();

  // Multi-level scale (5,000 elements)
  auto large_values = GenerateValuesWithSeed<typename TreeT::value_type>(
      5000, 5000 * 50, /*seed=*/98765);
  DoFullContainerTest<TreeT, StdRefT>(large_values, /*seed=*/98765);
  DoBulkBuildAndBatchTest<TreeT, StdRefT>(5000, /*seed=*/56789);
}

void TestAll() {
  fprintf(stderr, "Running BTreeSet uint32_t tests...\n");
  RunFullTestSuite<BTreeSet<uint32_t>, std::set<uint32_t> >();

  fprintf(stderr, "Running BTreeSet uint64_t tests...\n");
  RunFullTestSuite<BTreeSet<uint64_t>, std::set<uint64_t> >();

  fprintf(stderr, "Running BTreeSet int32_t signed tests...\n");
  RunFullTestSuite<BTreeSet<int32_t>, std::set<int32_t> >();

  fprintf(stderr, "Running BTreeSet int64_t signed tests...\n");
  RunFullTestSuite<BTreeSet<int64_t>, std::set<int64_t> >();

  fprintf(stderr, "Running BTreeMap uint32_t -> uint64_t tests...\n");
  RunFullTestSuite<BTreeMap<uint32_t, uint64_t>,
                   std::map<uint32_t, uint64_t> >();

  fprintf(stderr, "Running BTreeMap uint64_t -> uint64_t tests...\n");
  RunFullTestSuite<BTreeMap<uint64_t, uint64_t>,
                   std::map<uint64_t, uint64_t> >();

  fprintf(stderr, "Running BTreeMap uint64_t -> double tests...\n");
  RunFullTestSuite<BTreeMap<uint64_t, double>, std::map<uint64_t, double> >();

  fprintf(stderr, "Running BTreeMap uint32_t -> float tests...\n");
  RunFullTestSuite<BTreeMap<uint32_t, float>, std::map<uint32_t, float> >();

  fprintf(stderr, "All unified BTree tests passed successfully!\n");
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
