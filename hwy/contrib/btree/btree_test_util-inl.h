// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

#include "hwy/tests/test_util-inl.h"

// Per-target include guard
// clang-format off
#if defined(HIGHWAY_HWY_CONTRIB_BTREE_BTREE_TEST_UTIL_INL_H_) == defined(HWY_TARGET_TOGGLE)  // NOLINT
// clang-format on
#ifdef HIGHWAY_HWY_CONTRIB_BTREE_BTREE_TEST_UTIL_INL_H_
#undef HIGHWAY_HWY_CONTRIB_BTREE_BTREE_TEST_UTIL_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_BTREE_BTREE_TEST_UTIL_INL_H_
#endif

#include "hwy/contrib/random/random-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

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
    } else if constexpr (IsSigned<ValueT>()) {
      // Shift range from [1, max_val] to [-max_val/2, +max_val/2] to test
      // negative keys.
      return static_cast<ValueT>(i) - static_cast<ValueT>(max_val / 2);
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
    K k;
    if constexpr (IsSigned<K>()) {
      // Shift range from [1, max_val] to [-max_val/2, +max_val/2] to test
      // negative keys.
      k = static_cast<K>(i) - static_cast<K>(max_val / 2);
    } else {
      k = static_cast<K>(i);
    }
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
  AesCtrEngine engine(/*deterministic=*/true);
  RngStream rng(engine, seed);
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
  AesCtrEngine engine2(/*deterministic=*/true);
  RngStream q_rng(engine2, seed + 42);
  const size_t kNumRandomQueries = AdjustedReps(500);
  for (size_t i = 0; i < kNumRandomQueries; ++i) {
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
  AesCtrEngine engine3(/*deterministic=*/true);
  RngStream shuf_rng(engine3, seed + 84);
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
    AesCtrEngine engine4(/*deterministic=*/true);
    RngStream q_rng(engine4, seed + 100);
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
    const size_t kDenseCount = AdjustedReps(500);
    std::vector<value_type> dense_vals;
    dense_vals.reserve(kDenseCount);
    for (size_t i = 0; i < kDenseCount; ++i) {
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
    const size_t kMedCount = AdjustedReps(500);
    std::vector<value_type> med_vals;
    med_vals.reserve(kMedCount);
    for (size_t i = 0; i < kMedCount; ++i) {
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
    const size_t kSparseCount = AdjustedReps(200);
    std::vector<value_type> sparse_vals;
    sparse_vals.reserve(kSparseCount);
    for (size_t i = 0; i < kSparseCount; ++i) {
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

template <typename TreeT>
void DoTypedefsAndObserversTest() {
  using key_type = typename TreeT::key_type;
  using value_type = typename TreeT::value_type;

  // 1. Static compile-time concept assertions
  static_assert(requires { typename TreeT::key_compare; });
  static_assert(requires { typename TreeT::value_compare; });
  static_assert(requires { typename TreeT::reference; });
  static_assert(requires { typename TreeT::const_reference; });
  static_assert(requires { typename TreeT::pointer; });
  static_assert(requires { typename TreeT::const_pointer; });
  static_assert(requires { typename TreeT::allocator_type; });
  static_assert(
      std::is_same_v<typename TreeT::key_compare, std::less<key_type> >);

  // 2. Runtime comparator observer verification (matching absl::btree tests)
  TreeT tree;
  auto comp = tree.key_comp();
  HWY_ASSERT(comp(static_cast<key_type>(1), static_cast<key_type>(2)));
  HWY_ASSERT(!comp(static_cast<key_type>(2), static_cast<key_type>(2)));
  HWY_ASSERT(!comp(static_cast<key_type>(2), static_cast<key_type>(1)));

  auto val_comp = tree.value_comp();
  if constexpr (TreeT::kIsMap) {
    using mapped_type = typename TreeT::mapped_type;
    value_type v1{static_cast<key_type>(1), mapped_type{10}};
    value_type v2{static_cast<key_type>(2), mapped_type{20}};
    HWY_ASSERT(val_comp(v1, v2));
    HWY_ASSERT(!val_comp(v2, v2));
    HWY_ASSERT(!val_comp(v2, v1));
  } else {
    HWY_ASSERT(val_comp(static_cast<key_type>(1), static_cast<key_type>(2)));
    HWY_ASSERT(!val_comp(static_cast<key_type>(2), static_cast<key_type>(2)));
    HWY_ASSERT(!val_comp(static_cast<key_type>(2), static_cast<key_type>(1)));
  }

  // 3. Constructor with comparator argument
  typename TreeT::key_compare kc;
  TreeT tree_with_kc(kc);
  HWY_ASSERT(tree_with_kc.empty());

  // 4. Allocator getter
  auto alloc = tree.get_allocator();
  (void)alloc;
}

template <typename TreeT, typename StdRefT>
void DoCopyAndSwapTest() {
  using key_type = typename TreeT::key_type;
  using value_type = typename TreeT::value_type;

  // 1. Copy empty tree
  {
    TreeT empty_tree;
    TreeT copy_empty(empty_tree);
    HWY_ASSERT(copy_empty.empty());
    HWY_ASSERT_EQ(copy_empty.size(), size_t{0});
  }

  // 2. Copy populated tree (Set and Map)
  {
    const size_t kPopulatedCount = AdjustedReps(200);
    auto vals = GenerateValuesWithSeed<value_type>(kPopulatedCount, 4000, 777);
    TreeT orig;
    for (const auto& v : vals) {
      if constexpr (TreeT::kIsMap) {
        orig.insert(v.first, v.second);
      } else {
        orig.insert(v);
      }
    }

    // Copy Constructor
    TreeT copied(orig);
    HWY_ASSERT_EQ(copied.size(), orig.size());
    for (const auto& v : vals) {
      HWY_ASSERT(copied.contains(ExtractKey(v)));
    }

    // Element-by-element iterator equivalence check
    auto it_orig = orig.cbegin();
    auto it_copy = copied.cbegin();
    for (; it_orig != orig.cend() && it_copy != copied.cend();
         ++it_orig, ++it_copy) {
      VerifyEqualElements(it_copy, it_orig);
    }
    HWY_ASSERT(it_orig == orig.cend() && it_copy == copied.cend());

    // Mutating copy does not affect original
    key_type non_exist = static_cast<key_type>(999999);
    if constexpr (TreeT::kIsMap) {
      copied.insert(non_exist, typename TreeT::mapped_type{1});
    } else {
      copied.insert(non_exist);
    }
    HWY_ASSERT_EQ(copied.size(), orig.size() + 1);
    HWY_ASSERT(!orig.contains(non_exist));

    // Copy Assignment Operator into empty tree
    TreeT assigned;
    assigned = orig;
    HWY_ASSERT_EQ(assigned.size(), orig.size());
    for (const auto& v : vals) {
      HWY_ASSERT(assigned.contains(ExtractKey(v)));
    }

    // Copy Assignment Operator overwriting already-populated tree
    TreeT overwriting_tree;
    if constexpr (TreeT::kIsMap) {
      overwriting_tree.insert(static_cast<key_type>(12345),
                              typename TreeT::mapped_type{999});
    } else {
      overwriting_tree.insert(static_cast<key_type>(12345));
    }
    overwriting_tree = orig;
    HWY_ASSERT_EQ(overwriting_tree.size(), orig.size());
    for (const auto& v : vals) {
      HWY_ASSERT(overwriting_tree.contains(ExtractKey(v)));
    }

    // Self-assignment (nothing changes)
    assigned = assigned;
    HWY_ASSERT_EQ(assigned.size(), orig.size());

    // Member and ADL swap
    TreeT a, b;
    if constexpr (TreeT::kIsMap) {
      a.insert(static_cast<key_type>(10), typename TreeT::mapped_type{100});
      b.insert(static_cast<key_type>(20), typename TreeT::mapped_type{200});
    } else {
      a.insert(static_cast<key_type>(10));
      b.insert(static_cast<key_type>(20));
    }
    a.swap(b);
    HWY_ASSERT(a.contains(static_cast<key_type>(20)));
    HWY_ASSERT(b.contains(static_cast<key_type>(10)));

    using std::swap;
    swap(a, b);
    HWY_ASSERT(a.contains(static_cast<key_type>(10)));
    HWY_ASSERT(b.contains(static_cast<key_type>(20)));

    // Initializer list assignment
    if constexpr (!TreeT::kIsMap) {
      a = {static_cast<key_type>(1), static_cast<key_type>(2),
           static_cast<key_type>(3)};
      HWY_ASSERT_EQ(a.size(), size_t{3});
      HWY_ASSERT(a.contains(static_cast<key_type>(1)));
      HWY_ASSERT(a.contains(static_cast<key_type>(2)));
      HWY_ASSERT(a.contains(static_cast<key_type>(3)));
    }
  }

  // 3. Multi-level tree scale copy test
  {
    const size_t kLargeScale = AdjustedReps(5000);
    auto large_vals =
        GenerateValuesWithSeed<value_type>(kLargeScale, 100000, 888);
    TreeT large_orig;
    for (const auto& v : large_vals) {
      if constexpr (TreeT::kIsMap) {
        large_orig.insert(v.first, v.second);
      } else {
        large_orig.insert(v);
      }
    }

    TreeT large_copy(large_orig);
    HWY_ASSERT_EQ(large_copy.size(), large_orig.size());
    HWY_ASSERT_EQ(large_copy.height(), large_orig.height());
    for (const auto& v : large_vals) {
      HWY_ASSERT(large_copy.contains(ExtractKey(v)));
    }
  }
}

// Tests boundary edge cases (min, min+1, -1, 0, 1, max-1, max) and exercises
// multi-level tree splits to ensure internal node sentinel padding (0xFFFFFFFF)
// and signed key traversals/queries behave correctly across all boundaries.
template <typename TreeT, typename StdRefT>
void DoExtremeBoundariesTest() {
  using key_type = typename TreeT::key_type;
  using value_type = typename TreeT::value_type;
  static constexpr bool kIsMap = TreeT::kIsMap;

  std::vector<key_type> special_keys = {
      std::numeric_limits<key_type>::min(),
      static_cast<key_type>(std::numeric_limits<key_type>::min() + 1),
      static_cast<key_type>(-1),
      static_cast<key_type>(0),
      static_cast<key_type>(1),
      static_cast<key_type>(std::numeric_limits<key_type>::max() - 1),
      std::numeric_limits<key_type>::max(),
  };

  std::vector<value_type> vals;
  std::set<key_type> seen;
  for (key_type k : special_keys) {
    if (seen.insert(k).second) {
      if constexpr (kIsMap) {
        vals.push_back({k, static_cast<typename TreeT::mapped_type>(
                               static_cast<uint64_t>(k) & 0xFF)});
      } else {
        vals.push_back(k);
      }
    }
  }

  // Add 600 intermediate keys to ensure tree height >= 1 and internal node
  // splits.
  for (int64_t i = -300; i <= 300; ++i) {
    key_type k = static_cast<key_type>(i * 1000);
    if (seen.insert(k).second) {
      if constexpr (kIsMap) {
        vals.push_back({k, static_cast<typename TreeT::mapped_type>(
                               static_cast<uint64_t>(k) & 0xFF)});
      } else {
        vals.push_back(k);
      }
    }
  }

  DoFullContainerTest<TreeT, StdRefT>(vals, /*seed=*/77777);
}

template <typename TreeT, typename StdRefT>
void RunFullTestSuite() {
  DoTypedefsAndObserversTest<TreeT>();
  DoCopyAndSwapTest<TreeT, StdRefT>();
  DoBoundarySizeSweep<TreeT, StdRefT>();
  DoDiverseBitModesTest<TreeT>();
  DoExtremeBoundariesTest<TreeT, StdRefT>();

  // Multi-level scale (AdjustedReps ensures ASan/MSan/QEMU builds don't
  // timeout)
  const size_t kLargeScale = AdjustedReps(5000);
  auto large_values = GenerateValuesWithSeed<typename TreeT::value_type>(
      kLargeScale, kLargeScale * 50, /*seed=*/98765);
  DoFullContainerTest<TreeT, StdRefT>(large_values, /*seed=*/98765);
  DoBulkBuildAndBatchTest<TreeT, StdRefT>(kLargeScale, /*seed=*/56789);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_BTREE_BTREE_TEST_UTIL_INL_H_
