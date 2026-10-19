// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

#include "hwy/contrib/btree/btree_nodes.h"
#include "hwy/tests/test_util-inl.h"

#ifndef HWY_BTREE_SWEEP_ALL
#define HWY_BTREE_SWEEP_ALL 0
#endif

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
                                 decltype(std::declval<T>().second)>>
    : std::true_type {};

template <typename KeyT>
typename KeyCodec<KeyT>::LaneType KeyToRaw(KeyT k) {
  using LaneKeyT = typename KeyCodec<KeyT>::LaneType;
  if constexpr (IsPointer<KeyT>()) {
    return static_cast<LaneKeyT>(reinterpret_cast<uintptr_t>(k));
  } else {
    return static_cast<LaneKeyT>(k);
  }
}

template <typename KeyT>
KeyT KeyFromRaw(typename KeyCodec<KeyT>::LaneType val) {
  if constexpr (IsPointer<KeyT>()) {
    return reinterpret_cast<KeyT>(static_cast<uintptr_t>(val));
  } else {
    return static_cast<KeyT>(val);
  }
}

template <typename T, typename U>
void VerifyEqualValues(const T& a, const U& b) {
  if constexpr (IsPairLike<T>::value) {
    VerifyEqualValues(a.first, b.first);
    VerifyEqualValues(a.second, b.second);
  } else if constexpr (IsPointer<T>() || IsEnum<T>()) {
    HWY_ASSERT_EQ(KeyToRaw(a), KeyToRaw(b));
  } else {
    HWY_ASSERT_EQ(a, b);
  }
}

template <typename Iter1, typename Iter2>
void VerifyEqualElements(Iter1 it1, Iter2 it2) {
  VerifyEqualValues(*it1, *it2);
}

template <typename KeyT>
KeyT KeyFromRawInt(uint64_t val) {
  using LaneKeyT = typename KeyCodec<KeyT>::LaneType;
  return KeyFromRaw<KeyT>(static_cast<LaneKeyT>(val));
}

template <typename KeyT>
uint64_t KeyToRawU64(KeyT k) {
  return static_cast<uint64_t>(KeyToRaw(k));
}

template <typename ValueT>
struct ValueGenerator {
  uint64_t max_val;
  explicit ValueGenerator(uint64_t m) : max_val(m) {}

  ValueT operator()(uint64_t i) const {
    if constexpr (hwy::IsSameEither<ValueT, float, double>()) {
      return static_cast<ValueT>(i) * static_cast<ValueT>(0.5);
    } else if constexpr (KeyCodec<ValueT>::kIsSigned) {
      // Shift range from [1, max_val] to [-max_val/2, +max_val/2] to test
      // negative keys.
      using LaneT = typename KeyCodec<ValueT>::LaneType;
      return KeyFromRaw<ValueT>(static_cast<LaneT>(i) -
                                static_cast<LaneT>(max_val / 2));
    } else {
      return KeyFromRawInt<ValueT>(i);
    }
  }
};

template <typename K, typename V>
struct ValueGenerator<std::pair<K, V>> {
  uint64_t max_val;
  explicit ValueGenerator(uint64_t m) : max_val(m) {}

  std::pair<K, V> operator()(uint64_t i) const {
    K k = ValueGenerator<K>(max_val)(i);
    V v;
    if constexpr (hwy::IsSameEither<V, float, double>()) {
      v = static_cast<V>(i) * static_cast<V>(1.5);
    } else if constexpr (IsPointer<V>() || IsEnum<V>()) {
      v = KeyFromRawInt<V>(i * 10 + 7);
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
struct KeyExtractor<std::pair<K, V>> {
  static const K& Get(const std::pair<K, V>& p) { return p.first; }
};

template <typename K, typename V>
struct KeyExtractor<std::pair<const K, V>> {
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
struct ValueComparator<std::pair<K, V>> {
  bool operator()(const std::pair<K, V>& a, const std::pair<K, V>& b) const {
    return a.first < b.first;
  }
};

template <typename K, typename V>
struct ValueComparator<std::pair<const K, V>> {
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
struct ValueEquality<std::pair<K, V>> {
  bool operator()(const std::pair<K, V>& a, const std::pair<K, V>& b) const {
    return a.first == b.first;
  }
};

template <typename K, typename V>
struct ValueEquality<std::pair<const K, V>> {
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

template <typename LeafNodeT>
void VerifyLeafNode(const LeafNodeT* leaf) {
  using StorageKeyT = typename LeafNodeT::StorageKeyT;
  HWY_ASSERT(leaf != nullptr);
  const size_t count = leaf->NumKeys();

  if (count == 0) {
    HWY_ASSERT_EQ(leaf->base_key, StorageKeyT{0});
    for (size_t b = 0; b < LeafNodeT::kDataBytes; ++b) {
      HWY_ASSERT_EQ(leaf->KeyData()[b], uint8_t{0xFF});
    }
    return;
  }

  const uint8_t mode = leaf->BitMode();
  if (mode == kMode8Bit) {
    HWY_ASSERT(count <= HWY_MIN(LeafNodeT::kMax8, size_t{256}));
    const auto* offsets = HWY_RCAST_ALIGNED(const uint8_t*, leaf->KeyData());
    HWY_ASSERT_EQ(offsets[0], uint8_t{0});
    for (size_t k = 1; k < count; ++k) {
      HWY_ASSERT(offsets[k - 1] < offsets[k]);
    }
    HWY_ASSERT(leaf->base_key <=
               LimitsMax<StorageKeyT>() -
                   static_cast<StorageKeyT>(offsets[count - 1]));
    for (size_t k = count; k < LeafNodeT::kMax8; ++k) {
      HWY_ASSERT_EQ(offsets[k], uint8_t{0xFF});
    }
  } else if (mode == kMode16Bit) {
    HWY_ASSERT(count <= LeafNodeT::kMax16);
    const auto* offsets = HWY_RCAST_ALIGNED(const uint16_t*, leaf->KeyData());
    HWY_ASSERT_EQ(offsets[0], uint16_t{0});
    for (size_t k = 1; k < count; ++k) {
      HWY_ASSERT(offsets[k - 1] < offsets[k]);
    }
    HWY_ASSERT(leaf->base_key <=
               LimitsMax<StorageKeyT>() -
                   static_cast<StorageKeyT>(offsets[count - 1]));
    for (size_t k = count; k < LeafNodeT::kMax16; ++k) {
      HWY_ASSERT_EQ(offsets[k], uint16_t{0xFFFF});
    }
  } else if (mode == kMode32Bit) {
    HWY_ASSERT(count <= LeafNodeT::kMax32);
    if constexpr (sizeof(StorageKeyT) == 4) {
      const auto* raw_keys =
          HWY_RCAST_ALIGNED(const uint32_t*, leaf->KeyData());
      HWY_ASSERT_EQ(raw_keys[0], static_cast<uint32_t>(leaf->base_key));
      for (size_t k = 1; k < count; ++k) {
        HWY_ASSERT(raw_keys[k - 1] < raw_keys[k]);
      }
      for (size_t k = count; k < LeafNodeT::kMax32; ++k) {
        HWY_ASSERT_EQ(raw_keys[k], uint32_t{0xFFFFFFFFU});
      }
    } else {
      const auto* offsets = HWY_RCAST_ALIGNED(const uint32_t*, leaf->KeyData());
      HWY_ASSERT_EQ(offsets[0], uint32_t{0});
      for (size_t k = 1; k < count; ++k) {
        HWY_ASSERT(offsets[k - 1] < offsets[k]);
      }
      HWY_ASSERT(leaf->base_key <=
                 LimitsMax<StorageKeyT>() -
                     static_cast<StorageKeyT>(offsets[count - 1]));
      for (size_t k = count; k < LeafNodeT::kMax32; ++k) {
        HWY_ASSERT_EQ(offsets[k], uint32_t{0xFFFFFFFFU});
      }
    }
  } else {
    HWY_ASSERT_EQ(mode, uint8_t{kModeRaw64});
    HWY_ASSERT_EQ(sizeof(StorageKeyT), size_t{8});
    HWY_ASSERT(count <= LeafNodeT::kMax64);
    const auto* raw_keys = HWY_RCAST_ALIGNED(const uint64_t*, leaf->KeyData());
    HWY_ASSERT_EQ(raw_keys[0], static_cast<uint64_t>(leaf->base_key));
    for (size_t k = 1; k < count; ++k) {
      HWY_ASSERT(raw_keys[k - 1] < raw_keys[k]);
    }
    for (size_t k = count; k < LeafNodeT::kMax64; ++k) {
      HWY_ASSERT_EQ(raw_keys[k], uint64_t{0xFFFFFFFFFFFFFFFFULL});
    }
  }
}

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

  static void VerifyPhysicalTree(const TreeT& tree_) {
    using StorageKeyT = typename KeyCodec<key_type>::StorageKey;
    using InternalNodeT = InternalNode<StorageKeyT>;
    const auto* st = tree_.state();

    if (tree_.empty()) {
      HWY_ASSERT(st->root_ == nullptr);
      HWY_ASSERT(st->first_leaf_ == nullptr);
      HWY_ASSERT(st->last_leaf_ == nullptr);
      HWY_ASSERT_EQ(st->tree_height_, size_t{0});
      HWY_ASSERT_EQ(st->num_leaves_, size_t{0});
      HWY_ASSERT_EQ(st->num_internals_, size_t{0});
      HWY_ASSERT(tree_.begin() == tree_.end());
      return;
    }

    HWY_ASSERT(st->root_ != nullptr);
    HWY_ASSERT(st->first_leaf_ != nullptr);
    HWY_ASSERT(st->last_leaf_ != nullptr);
    HWY_ASSERT(st->first_leaf_->Prev() == nullptr);
    HWY_ASSERT(st->last_leaf_->Next() == nullptr);

    struct NodeBounds {
      const void* node;
      bool has_lo;
      StorageKeyT lo;
      bool has_hi;
      StorageKeyT hi;
    };

    std::vector<NodeBounds> level{{st->root_, false, 0, false, 0}};
    size_t visited_internals = 0;

    for (size_t h = st->tree_height_; h > 0; --h) {
      std::vector<NodeBounds> next;
      for (const auto& item : level) {
        HWY_ASSERT(item.node != nullptr);
        const auto* internal = static_cast<const InternalNodeT*>(item.node);
        visited_internals++;

        if (h == st->tree_height_) {
          HWY_ASSERT(internal->num_keys >= 1 &&
                     internal->num_keys <= InternalNodeT::kCapacity);
        } else {
          HWY_ASSERT(internal->num_keys >= InternalNodeT::kCapacity / 2 &&
                     internal->num_keys <= InternalNodeT::kCapacity);
        }

        for (size_t k = 0; k < internal->num_keys; ++k) {
          if (k > 0) {
            HWY_ASSERT(internal->keys[k - 1] < internal->keys[k]);
          }
          if (item.has_lo) {
            HWY_ASSERT(internal->keys[k] >= item.lo);
          }
          if (item.has_hi) {
            HWY_ASSERT(internal->keys[k] < item.hi);
          }
        }

        // Verify unused key and child slots hold sentinels (MAX and nullptr)
        for (size_t k = internal->num_keys; k < InternalNodeT::kCapacity; ++k) {
          HWY_ASSERT_EQ(internal->keys[k],
                        std::numeric_limits<StorageKeyT>::max());
        }
        for (size_t c = internal->num_keys + 1; c < InternalNodeT::kMaxChildren;
             ++c) {
          HWY_ASSERT(internal->children[c] == nullptr);
        }

        for (size_t c = 0; c <= internal->num_keys; ++c) {
          HWY_ASSERT(internal->children[c] != nullptr);
          const bool c_has_lo = (c > 0) ? true : item.has_lo;
          const StorageKeyT c_lo = (c > 0) ? internal->keys[c - 1] : item.lo;
          const bool c_has_hi = (c < internal->num_keys) ? true : item.has_hi;
          const StorageKeyT c_hi =
              (c < internal->num_keys) ? internal->keys[c] : item.hi;
          next.push_back(
              {internal->children[c], c_has_lo, c_lo, c_has_hi, c_hi});
        }
      }
      level = std::move(next);
    }

    HWY_ASSERT_EQ(visited_internals, st->num_internals_);
    HWY_ASSERT_EQ(level.size(), st->num_leaves_);

    using LeafPtrT = decltype(st->first_leaf_);
    HWY_ASSERT(static_cast<LeafPtrT>(const_cast<void*>(level.front().node)) ==
               st->first_leaf_);
    HWY_ASSERT(static_cast<LeafPtrT>(const_cast<void*>(level.back().node)) ==
               st->last_leaf_);

    size_t total_leaf_keys = 0;
    for (size_t i = 0; i < level.size(); ++i) {
      const auto* leaf =
          static_cast<const std::remove_pointer_t<LeafPtrT>*>(level[i].node);
      HWY_ASSERT(leaf->NumKeys() >= 1);
      VerifyLeafNode(leaf);
      total_leaf_keys += leaf->NumKeys();

      if (i == 0) {
        HWY_ASSERT(leaf->Prev() == nullptr);
      } else {
        HWY_ASSERT(leaf->Prev() ==
                   static_cast<LeafPtrT>(const_cast<void*>(level[i - 1].node)));
      }
      if (i + 1 == level.size()) {
        HWY_ASSERT(leaf->Next() == nullptr);
      } else {
        HWY_ASSERT(leaf->Next() ==
                   static_cast<LeafPtrT>(const_cast<void*>(level[i + 1].node)));
      }

      for (size_t k = 0; k < leaf->NumKeys(); ++k) {
        const StorageKeyT sk = GetLeafKey(leaf, k);
        if (k > 0) {
          HWY_ASSERT(GetLeafKey(leaf, k - 1) < sk);
        }
        if (level[i].has_lo) {
          HWY_ASSERT(sk >= level[i].lo);
        }
        if (level[i].has_hi) {
          HWY_ASSERT(sk < level[i].hi);
        }
      }
    }
    HWY_ASSERT_EQ(total_leaf_keys, st->num_elements_);
  }

  void verify() const {
    HWY_ASSERT_EQ(tree_.size(), ref_.size());
    HWY_ASSERT_EQ(tree_.empty(), ref_.empty());
    VerifyPhysicalTree(tree_);

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
  if (!inserted_keys.empty()) {
    key_type existing_k = inserted_keys[0];
    if constexpr (kIsMap) {
      using mapped_type = typename TreeT::mapped_type;
      auto [it, inserted] =
          checker.tree().emplace(existing_k, static_cast<mapped_type>(999));
      HWY_ASSERT(!inserted);
      HWY_ASSERT(it != checker.tree().end());
    } else {
      auto [it, inserted] = checker.tree().emplace(existing_k);
      HWY_ASSERT(!inserted);
      HWY_ASSERT(it != checker.tree().end());
    }
    checker.verify();
  }

  // Preserve original insertion order (sorted, rsorted, or random)
  const std::vector<key_type> initial_order_keys = inserted_keys;
  std::vector<key_type> sorted_keys = inserted_keys;
  std::sort(sorted_keys.begin(), sorted_keys.end());

  // 2. Lookups on all inserted keys and random queries
  for (key_type k : inserted_keys) {
    checker.CheckLookup(k);
  }
  AesCtrEngine engine2(/*deterministic=*/true);
  RngStream q_rng(engine2, seed + 42);
  const size_t kNumRandomQueries =
      HWY_BTREE_SWEEP_ALL ? size_t{40000} : AdjustedReps(500);
  for (size_t i = 0; i < kNumRandomQueries; ++i) {
    key_type q =
        KeyFromRawInt<key_type>((q_rng() % (values.size() * 50 + 10)) + 1);
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

  // 4. Deletions (Erase half the inserted elements in random order)
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
  auto reinsert_key = [&](key_type k) {
    if constexpr (kIsMap) {
      // Compute in uint64_t: extreme signed keys would overflow in key_type.
      const uint64_t v = KeyToRawU64(k) * 10 + 3;
      checker.insert({k, static_cast<typename TreeT::mapped_type>(v)});
    } else {
      checker.insert(k);
    }
  };
  for (size_t i = 0; i < to_delete; ++i) {
    reinsert_key(inserted_keys[i]);
    checker.CheckLookup(inserted_keys[i]);
  }
  checker.verify();

  // 6. Sub-range deletions (similar to absl::btree_test.cc: first half, second
  // half, and middle quarter)
  const size_t n_keys = sorted_keys.size();
  // 6a. First half [0, N/2)
  for (size_t i = 0; i < n_keys / 2; ++i) checker.erase(sorted_keys[i]);
  checker.verify();
  for (size_t i = 0; i < n_keys / 2; ++i) reinsert_key(sorted_keys[i]);
  checker.verify();

  // 6b. Second half [N/2, N)
  for (size_t i = n_keys / 2; i < n_keys; ++i) checker.erase(sorted_keys[i]);
  checker.verify();
  for (size_t i = n_keys / 2; i < n_keys; ++i) reinsert_key(sorted_keys[i]);
  checker.verify();

  // 6c. Middle quarter [N/4, N/2)
  for (size_t i = n_keys / 4; i < n_keys / 2; ++i) {
    checker.erase(sorted_keys[i]);
  }
  checker.verify();
  for (size_t i = n_keys / 4; i < n_keys / 2; ++i) {
    reinsert_key(sorted_keys[i]);
  }
  checker.verify();

  // 7. Full 100% drain in the input's order (sorted = front-to-back, rsorted =
  // back-to-front, random = random)
  const size_t split_60 = (initial_order_keys.size() * 3) / 5;
  for (size_t i = 0; i < split_60; ++i) {
    checker.erase(initial_order_keys[i]);
    HWY_ASSERT_EQ(checker.erase(initial_order_keys[i]), size_t{0});
  }
  checker.verify();

  for (size_t i = split_60; i < initial_order_keys.size(); ++i) {
    checker.erase(initial_order_keys[i]);
    HWY_ASSERT_EQ(checker.erase(initial_order_keys[i]), size_t{0});
  }
  checker.verify();
  HWY_ASSERT(checker.empty());
  HWY_ASSERT_EQ(checker.tree().height(), size_t{0});
  HWY_ASSERT(checker.tree().begin() == checker.tree().end());

  // 8. Clear
  checker.clear();
}

template <typename TreeT, typename StdRefT>
void DoBulkBuildAndBatchTest(size_t n, uint32_t seed) {
  using key_type = typename TreeT::key_type;
  using value_type = typename TreeT::value_type;
  static constexpr bool kIsMap = TreeT::kIsMap;

  auto values = GenerateValuesWithSeed<value_type>(n, n * 50, seed);
  auto unsorted_values = values;
  SortUniqueValues(values);
  n = values.size();

  StdRefT ref(values.begin(), values.end());

  // 0.0 and 5.0 exercise the clamp to [0.1, 1.0].
  for (float fill_ratio : {0.5f, 0.75f, 1.0f, 0.0f, 5.0f}) {
    TreeT tree = BuildTreeFromValues<TreeT>(values, fill_ratio);
    HWY_ASSERT_EQ(tree.size(), ref.size());
    HWY_ASSERT_EQ(tree.empty(), ref.empty());
    BTreeChecker<TreeT, StdRefT>::VerifyPhysicalTree(tree);

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
    // Mixed reverse_iterator / const_reverse_iterator comparisons.
    HWY_ASSERT(tree_rit == tree.crend());
    HWY_ASSERT(tree.crend() == tree_rit);
    HWY_ASSERT(!(tree_rit != tree.crend()));
    if (!tree.empty()) {
      HWY_ASSERT(tree.rbegin() != tree.crend());
      HWY_ASSERT(tree.crbegin() != tree.rend());
    }

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
        queries[i] = KeyFromRawInt<key_type>((q_rng() % (n * 60 + 10)) + 1);
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

  // Verify Build with unsorted unique inputs and unsorted duplicate inputs
  // (ensuring first-occurrence wins for Maps, matching
  // std::map/absl::btree_map).
  {
    auto unsorted_with_dups = unsorted_values;
    for (size_t i = 0; i < unsorted_values.size(); i += 2) {
      if constexpr (kIsMap) {
        using mapped_type = typename TreeT::mapped_type;
        unsorted_with_dups.push_back(
            {unsorted_values[i].first,
             static_cast<mapped_type>(unsorted_values[i].second +
                                      static_cast<mapped_type>(77))});
      } else {
        unsorted_with_dups.push_back(unsorted_values[i]);
      }
    }

    for (const auto& input_vec : {unsorted_values, unsorted_with_dups}) {
      StdRefT dup_ref(input_vec.begin(), input_vec.end());
      TreeT built = BuildTreeFromValues<TreeT>(input_vec, 1.0f);
      HWY_ASSERT_EQ(built.size(), dup_ref.size());
      BTreeChecker<TreeT, StdRefT>::VerifyPhysicalTree(built);
      auto r_it = dup_ref.begin();
      for (auto t_it = built.begin(); t_it != built.end(); ++t_it, ++r_it) {
        VerifyEqualElements(t_it, r_it);
      }
      HWY_ASSERT(r_it == dup_ref.end());
    }
  }
}

template <typename TreeT, typename StdRefT>
void DoBoundarySizeSweep() {
  using value_type = typename TreeT::value_type;
  auto run_for_size = [](size_t n) {
    auto values = GenerateValuesWithSeed<value_type>(
        n, std::max<uint64_t>(100, n * 20), /*seed=*/12345 + n);
    DoFullContainerTest<TreeT, StdRefT>(values, /*seed=*/12345 + n);
    DoBulkBuildAndBatchTest<TreeT, StdRefT>(n, /*seed=*/54321 + n);
  };

  if constexpr (HWY_BTREE_SWEEP_ALL) {
    for (size_t n = 0; n <= 800; ++n) {
      run_for_size(n);
    }
  } else {
    for (size_t n : {size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{5},
                     size_t{16}, size_t{31}, size_t{32}, size_t{63}, size_t{64},
                     size_t{127}, size_t{128}, size_t{243}, size_t{244},
                     size_t{245}, size_t{488}, size_t{489}, size_t{500}}) {
      run_for_size(n);
    }
  }
}

template <typename TreeT>
void DoDiverseBitModesTest() {
  using key_type = typename TreeT::key_type;
  using value_type = typename TreeT::value_type;

  // Dense 8-bit mode
  {
    const size_t kDenseCount =
        HWY_BTREE_SWEEP_ALL ? size_t{20000} : AdjustedReps(500);
    std::vector<value_type> dense_vals;
    dense_vals.reserve(kDenseCount);
    for (size_t i = 0; i < kDenseCount; ++i) {
      if constexpr (TreeT::kIsMap) {
        dense_vals.push_back({KeyFromRawInt<key_type>(i * 2 + 10),
                              static_cast<typename TreeT::mapped_type>(i)});
      } else {
        dense_vals.push_back(KeyFromRawInt<key_type>(i * 2 + 10));
      }
    }
    TreeT tree = BuildTreeFromValues<TreeT>(dense_vals);
    for (const auto& v : dense_vals) {
      HWY_ASSERT(tree.contains(ExtractKey(v)));
    }
  }

  // Medium 16-bit mode
  {
    const size_t kMedCount =
        HWY_BTREE_SWEEP_ALL ? size_t{20000} : AdjustedReps(500);
    std::vector<value_type> med_vals;
    med_vals.reserve(kMedCount);
    for (size_t i = 0; i < kMedCount; ++i) {
      if constexpr (TreeT::kIsMap) {
        med_vals.push_back({KeyFromRawInt<key_type>(i * 100 + 10),
                            static_cast<typename TreeT::mapped_type>(i)});
      } else {
        med_vals.push_back(KeyFromRawInt<key_type>(i * 100 + 10));
      }
    }
    TreeT tree = BuildTreeFromValues<TreeT>(med_vals);
    for (const auto& v : med_vals) {
      HWY_ASSERT(tree.contains(ExtractKey(v)));
    }
  }

  // Sparse 32-bit / raw 64-bit mode
  {
    const size_t kSparseCount =
        HWY_BTREE_SWEEP_ALL ? size_t{20000} : AdjustedReps(200);
    std::vector<value_type> sparse_vals;
    sparse_vals.reserve(kSparseCount);
    const uint64_t kSparseStep =
        (sizeof(key_type) == 4) ? 100000ULL : 10000000000ULL;
    for (size_t i = 0; i < kSparseCount; ++i) {
      key_type k =
          KeyFromRawInt<key_type>(static_cast<uint64_t>(i) * kSparseStep + 500);
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

  // Mode-downgrade outlier split: pack Leaf 0 with 244 dense keys in
  // 8-bit/16-bit mode and Leaf 1 with a far key, then insert outlier keys
  // with huge deltas into both Leaf 0 (right-half downgrade) and Leaf 1
  // (left-half downgrade).
  {
    std::vector<value_type> mixed_vals;
    mixed_vals.reserve(490);
    // 244 dense keys in [10000 .. 10243]
    for (size_t i = 0; i < 244; ++i) {
      key_type k = KeyFromRawInt<key_type>(10000 + i);
      if constexpr (TreeT::kIsMap) {
        mixed_vals.push_back(
            {k, static_cast<typename TreeT::mapped_type>(i + 1)});
      } else {
        mixed_vals.push_back(k);
      }
    }
    // 244 dense keys in [100000000 .. 100000243] to form a second dense leaf
    const uint64_t far_base =
        (sizeof(key_type) == 4) ? 100000000ULL : 10000000000ULL;
    for (size_t i = 0; i < 244; ++i) {
      key_type k = KeyFromRawInt<key_type>(far_base + i);
      if constexpr (TreeT::kIsMap) {
        mixed_vals.push_back(
            {k, static_cast<typename TreeT::mapped_type>(i + 1000)});
      } else {
        mixed_vals.push_back(k);
      }
    }
    // Outlier 1: lands at the end of Leaf 0 (forces right-half mode downgrade)
    // Outlier 2: lands at the start of Leaf 1 (forces left-half mode downgrade)
    const key_type outlier_right = KeyFromRawInt<key_type>(
        (sizeof(key_type) == 4) ? 50000000ULL : 5000000000ULL);
    const key_type outlier_left = KeyFromRawInt<key_type>(
        (sizeof(key_type) == 4) ? 90000000ULL : 9000000000ULL);
    for (key_type k : {outlier_right, outlier_left}) {
      if constexpr (TreeT::kIsMap) {
        mixed_vals.push_back(
            {k, static_cast<typename TreeT::mapped_type>(777)});
      } else {
        mixed_vals.push_back(k);
      }
    }

    TreeT tree;
    for (const auto& v : mixed_vals) {
      tree.insert(v);
      BTreeChecker<TreeT, TreeT>::VerifyPhysicalTree(tree);
    }
    for (const auto& v : mixed_vals) {
      HWY_ASSERT(tree.contains(ExtractKey(v)));
    }
  }
}

template <typename TreeT, bool kIsMap = TreeT::kIsMap>
struct TreeMappedOrByte {
  using type = uint8_t;
};

template <typename TreeT>
struct TreeMappedOrByte<TreeT, true> {
  using type = typename TreeT::mapped_type;
};

// Sweeps every leaf compression BitMode and NumKeys (1..256) on a 1-leaf tree
// (tree_height_ == 0), testing FindLeafSlot (lower_bound / upper_bound),
// LeafContains (contains / find), TryFastEraseFromLeaf (erase), and
// TryFastInsertIntoLeaf (insert, including slot == 0 base_key rebase).
template <typename TreeT>
void DoExhaustiveLeafKernelSweepTest() {
  using KeyT = typename TreeT::key_type;
  using StorageKeyT = typename KeyCodec<KeyT>::StorageKey;
  static constexpr bool kIsMap = TreeT::kIsMap;
  using ValT = typename TreeMappedOrByte<TreeT>::type;
  using LeafNodeT = std::remove_pointer_t<
      decltype(std::declval<const TreeT>().state()->first_leaf_)>;

  constexpr StorageKeyT kBaseKey =
      (sizeof(StorageKeyT) == 8) ? static_cast<StorageKeyT>(0x100020000ULL)
                                 : static_cast<StorageKeyT>(70000U);

  struct ModeSpec {
    uint8_t mode;
    size_t capacity;
    size_t max_count;
    bool is_raw;
    uint64_t max_delta;
    StorageKeyT normal_span;
    StorageKeyT sentinel_span;
  };

  std::vector<ModeSpec> specs = {
      {kMode8Bit, LeafNodeT::kMax8, HWY_MIN(LeafNodeT::kMax8, size_t{256}),
       false, 255ULL, StorageKeyT{200}, StorageKeyT{255}},
      {kMode16Bit, LeafNodeT::kMax16, LeafNodeT::kMax16, false, 65535ULL,
       StorageKeyT{1000}, StorageKeyT{65535}},
  };
  if constexpr (sizeof(StorageKeyT) == 4) {
    specs.push_back(
        {kMode32Bit, LeafNodeT::kMax32, LeafNodeT::kMax32, true,
         LimitsMax<StorageKeyT>(), StorageKeyT{100000},
         static_cast<StorageKeyT>(LimitsMax<StorageKeyT>() - kBaseKey)});
  } else {
    specs.push_back({kMode32Bit, LeafNodeT::kMax32, LeafNodeT::kMax32, false,
                     0xFFFFFFFFULL, StorageKeyT{100000},
                     static_cast<StorageKeyT>(0xFFFFFFFFULL)});
    specs.push_back(
        {kModeRaw64, LeafNodeT::kMax64, LeafNodeT::kMax64, true,
         LimitsMax<uint64_t>(), static_cast<StorageKeyT>(0x100000100ULL),
         static_cast<StorageKeyT>(LimitsMax<StorageKeyT>() - kBaseKey)});
  }

  const auto is_critical_count = [](size_t count, size_t max_count) {
    if (count <= 18 || count + 1 >= max_count) return true;
    switch (count) {
      case 30:
      case 31:
      case 32:
      case 33:
      case 53:
      case 54:
      case 60:
      case 61:
      case 63:
      case 64:
      case 65:
      case 121:
      case 122:
      case 123:
      case 127:
      case 128:
      case 129:
      case 243:
      case 244:
      case 246:
      case 255:
      case 256:
        return true;
      default:
        return false;
    }
  };

  StorageKeyT keys[256];
  KeyT user_keys[256];
  ValT user_vals[256];
  std::vector<StorageKeyT> queries;
  queries.reserve(64);

  const auto build_leaf_tree = [&](const ModeSpec& spec,
                                   size_t count) -> TreeT {
    if (count == 1 && spec.mode != kMode8Bit) {
      // Build a 2-key leaf in spec.mode, then fast-erase slot 1 so count == 1
      // stays in spec.mode.
      KeyT tmp_uk[2] = {user_keys[0],
                        KeyCodec<KeyT>::FromStorage(static_cast<StorageKeyT>(
                            kBaseKey + spec.normal_span))};
      TreeT t;
      if constexpr (kIsMap) {
        ValT tmp_uv[2] = {user_vals[0], static_cast<ValT>(20)};
        t = TreeT::Build(tmp_uk, tmp_uv, 2);
      } else {
        t = TreeT::Build(tmp_uk, 2);
      }
      HWY_ASSERT_EQ(t.state()->first_leaf_->BitMode(), spec.mode);
      HWY_ASSERT_EQ(t.erase(tmp_uk[1]), size_t{1});
      return t;
    }
    if constexpr (kIsMap) {
      return TreeT::Build(user_keys, user_vals, count);
    } else {
      return TreeT::Build(user_keys, count);
    }
  };

  for (const ModeSpec& spec : specs) {
    for (size_t count = 1; count <= spec.max_count; ++count) {
      const size_t num_spans = (count == 1) ? 1 : 2;
      for (size_t span_idx = 0; span_idx < num_spans; ++span_idx) {
        const StorageKeyT target_span =
            (span_idx == 0) ? spec.normal_span : spec.sentinel_span;

        if (count == 1) {
          keys[0] = kBaseKey;
          user_keys[0] = KeyCodec<KeyT>::FromStorage(keys[0]);
          user_vals[0] = static_cast<ValT>(7);
        } else {
          const StorageKeyT actual_span =
              HWY_MAX(target_span, static_cast<StorageKeyT>(count - 1));
          const StorageKeyT max_key =
              static_cast<StorageKeyT>(kBaseKey + actual_span);
          keys[0] = kBaseKey;
          keys[count - 1] = max_key;
          if (count >= 3) {
            keys[1] = static_cast<StorageKeyT>(kBaseKey + 1);
            const StorageKeyT rem_span =
                static_cast<StorageKeyT>(max_key - (kBaseKey + 1));
            const StorageKeyT step =
                static_cast<StorageKeyT>(rem_span / (count - 2));
            for (size_t i = 2; i + 1 < count; ++i) {
              keys[i] = static_cast<StorageKeyT>(
                  (kBaseKey + 1) + step * static_cast<StorageKeyT>(i - 1));
            }
          }
          for (size_t i = 0; i < count; ++i) {
            user_keys[i] = KeyCodec<KeyT>::FromStorage(keys[i]);
            user_vals[i] = static_cast<ValT>(i * 13 + 7);
          }
        }

        TreeT tree = build_leaf_tree(spec, count);
        HWY_ASSERT_EQ(tree.state()->tree_height_, size_t{0});
        HWY_ASSERT_EQ(tree.state()->num_leaves_, size_t{1});
        const auto* leaf = tree.state()->first_leaf_;
        HWY_ASSERT_EQ(leaf->BitMode(), spec.mode);
        HWY_ASSERT_EQ(leaf->NumKeys(), count);
        VerifyLeafNode(leaf);

        // 1. Read kernels: FindLeafSlot (lower_bound / upper_bound) and
        // LeafContains (both fast contains path and find slot path).
        queries.clear();
        queries.push_back(StorageKeyT{0});
        queries.push_back(StorageKeyT{1});
        queries.push_back(static_cast<StorageKeyT>(kBaseKey - 1));
        queries.push_back(kBaseKey);
        queries.push_back(static_cast<StorageKeyT>(kBaseKey + 1));
        queries.push_back(static_cast<StorageKeyT>(keys[count - 1] - 1));
        queries.push_back(keys[count - 1]);
        if (keys[count - 1] < LimitsMax<StorageKeyT>()) {
          queries.push_back(static_cast<StorageKeyT>(keys[count - 1] + 1));
        }
        queries.push_back(
            static_cast<StorageKeyT>(LimitsMax<StorageKeyT>() - 1));
        queries.push_back(LimitsMax<StorageKeyT>());

        constexpr uint64_t kSentinelDeltas[] = {
            254ULL,   255ULL,        256ULL,        65534ULL,      65535ULL,
            65536ULL, 0xFFFFFFFEULL, 0xFFFFFFFFULL, 0x100000000ULL};
        for (uint64_t d : kSentinelDeltas) {
          if (d <= static_cast<uint64_t>(LimitsMax<StorageKeyT>() - kBaseKey)) {
            queries.push_back(static_cast<StorageKeyT>(
                kBaseKey + static_cast<StorageKeyT>(d)));
          }
        }

        const size_t probe_slots[] = {
            0, (count > 1) ? size_t{1} : size_t{0}, count / 2,
            (count >= 2) ? (count - 2) : size_t{0}, count - 1};
        for (size_t s : probe_slots) {
          const StorageKeyT k = keys[s];
          queries.push_back(static_cast<StorageKeyT>(k - 1));
          queries.push_back(k);
          if (k < LimitsMax<StorageKeyT>()) {
            queries.push_back(static_cast<StorageKeyT>(k + 1));
          }
        }

        for (StorageKeyT q : queries) {
          const KeyT uq = KeyCodec<KeyT>::FromStorage(q);
          const size_t expected_lb = static_cast<size_t>(
              std::lower_bound(keys, keys + count, q) - keys);
          const size_t expected_ub = static_cast<size_t>(
              std::upper_bound(keys, keys + count, q) - keys);
          const bool expected_found =
              (expected_lb < count && keys[expected_lb] == q);

          const auto lb_it = tree.lower_bound(uq);
          if (expected_lb < count) {
            HWY_ASSERT(lb_it != tree.end());
            HWY_ASSERT_EQ(lb_it.slot(), expected_lb);
          } else {
            HWY_ASSERT(lb_it == tree.end());
          }

          const auto ub_it = tree.upper_bound(uq);
          if (expected_ub < count) {
            HWY_ASSERT(ub_it != tree.end());
            HWY_ASSERT_EQ(ub_it.slot(), expected_ub);
          } else {
            HWY_ASSERT(ub_it == tree.end());
          }

          HWY_ASSERT_EQ(tree.contains(uq), expected_found);

          const auto find_it = tree.find(uq);
          if (expected_found) {
            HWY_ASSERT(find_it != tree.end());
            HWY_ASSERT_EQ(find_it.slot(), expected_lb);
            if constexpr (kIsMap) {
              HWY_ASSERT_EQ(find_it->second, user_vals[expected_lb]);
            }
          } else {
            HWY_ASSERT(find_it == tree.end());
          }
        }

        const bool sweep_all_slots =
            HWY_BTREE_SWEEP_ALL || is_critical_count(count, spec.max_count);

        // 2. Erase kernel: TryFastEraseFromLeaf via
        // tree.erase(user_keys[slot]).
        for (size_t slot = 0; slot < count; ++slot) {
          if (!sweep_all_slots && slot != 0 && slot != 1 && slot != count / 2 &&
              slot + 2 != count && slot + 1 != count) {
            continue;
          }
          TreeT scratch = build_leaf_tree(spec, count);
          HWY_ASSERT_EQ(scratch.erase(user_keys[slot]), size_t{1});
          HWY_ASSERT_EQ(scratch.size(), count - 1);
          if (count == 1) {
            HWY_ASSERT(scratch.empty());
            HWY_ASSERT(scratch.state()->root_ == nullptr);
            HWY_ASSERT_EQ(scratch.state()->num_leaves_, size_t{0});
          } else {
            HWY_ASSERT_EQ(scratch.state()->tree_height_, size_t{0});
            HWY_ASSERT_EQ(scratch.state()->num_leaves_, size_t{1});
            const auto* s_leaf = scratch.state()->first_leaf_;
            VerifyLeafNode(s_leaf);
            HWY_ASSERT_EQ(s_leaf->BitMode(), spec.mode);
            HWY_ASSERT_EQ(s_leaf->NumKeys(), count - 1);
            for (size_t j = 0; j < count - 1; ++j) {
              const size_t src = (j < slot) ? j : (j + 1);
              HWY_ASSERT_EQ(GetLeafKey(s_leaf, j), keys[src]);
            }
            if constexpr (kIsMap) {
              size_t j = 0;
              for (auto it = scratch.begin(); it != scratch.end(); ++it, ++j) {
                const size_t src = (j < slot) ? j : (j + 1);
                HWY_ASSERT_EQ(it->second, user_vals[src]);
              }
              HWY_ASSERT_EQ(j, count - 1);
            }
          }
        }

        // 3. Insert kernel at slot 0 (new_key < base_key, SIMD offset rebase).
        std::vector<StorageKeyT> slot0_keys = {
            static_cast<StorageKeyT>(kBaseKey - 1),
            static_cast<StorageKeyT>(kBaseKey - 7)};
        if (!spec.is_raw && static_cast<uint64_t>(keys[count - 1] - kBaseKey) <
                                spec.max_delta) {
          const StorageKeyT exact_fit_base = static_cast<StorageKeyT>(
              keys[count - 1] - static_cast<StorageKeyT>(spec.max_delta));
          if (exact_fit_base > 0 && exact_fit_base < kBaseKey) {
            slot0_keys.push_back(exact_fit_base);
            slot0_keys.push_back(static_cast<StorageKeyT>(exact_fit_base - 1));
          }
        }
        for (StorageKeyT new_key : slot0_keys) {
          const uint64_t new_span =
              static_cast<uint64_t>(keys[count - 1] - new_key);
          const bool expected_fits =
              (count < spec.capacity) &&
              (spec.is_raw || new_span <= spec.max_delta);
          const KeyT new_uk = KeyCodec<KeyT>::FromStorage(new_key);
          const ValT new_uv = static_cast<ValT>(999);
          TreeT scratch = build_leaf_tree(spec, count);
          bool inserted = false;
          size_t ret_slot = ~size_t{0};
          if constexpr (kIsMap) {
            auto [it, ok] = scratch.insert(new_uk, new_uv);
            inserted = ok;
            ret_slot = it.slot();
          } else {
            auto [it, ok] = scratch.insert(new_uk);
            inserted = ok;
            ret_slot = it.slot();
          }
          HWY_ASSERT(inserted);
          HWY_ASSERT_EQ(scratch.size(), count + 1);
          if (expected_fits) {
            HWY_ASSERT_EQ(scratch.state()->tree_height_, size_t{0});
            HWY_ASSERT_EQ(scratch.state()->num_leaves_, size_t{1});
            const auto* s_leaf = scratch.state()->first_leaf_;
            VerifyLeafNode(s_leaf);
            HWY_ASSERT_EQ(s_leaf->BitMode(), spec.mode);
            HWY_ASSERT_EQ(s_leaf->NumKeys(), count + 1);
            HWY_ASSERT_EQ(s_leaf->base_key, new_key);
            HWY_ASSERT_EQ(ret_slot, size_t{0});
            HWY_ASSERT_EQ(GetLeafKey(s_leaf, 0), new_key);
            for (size_t j = 0; j < count; ++j) {
              HWY_ASSERT_EQ(GetLeafKey(s_leaf, j + 1), keys[j]);
            }
            if constexpr (kIsMap) {
              auto it = scratch.begin();
              HWY_ASSERT_EQ(it->second, new_uv);
              ++it;
              for (size_t j = 0; j < count; ++j, ++it) {
                HWY_ASSERT_EQ(it->second, user_vals[j]);
              }
            }
          } else {
            BTreeChecker<TreeT, TreeT>::VerifyPhysicalTree(scratch);
            HWY_ASSERT(scratch.state()->num_leaves_ == 2 ||
                       scratch.state()->first_leaf_->BitMode() > spec.mode);
            HWY_ASSERT(scratch.contains(new_uk));
          }
        }

        // 4. Insert kernel at slots 1..count (new_key > base_key).
        for (size_t slot = 1; slot <= count; ++slot) {
          if (!sweep_all_slots && slot != 1 && slot != count / 2 &&
              slot + 1 != count && slot != count) {
            continue;
          }
          StorageKeyT new_key = 0;
          if (slot < count) {
            if (keys[slot - 1] + 1 >= keys[slot]) continue;
            new_key = static_cast<StorageKeyT>(keys[slot - 1] + 1);
          } else {
            if (keys[count - 1] == LimitsMax<StorageKeyT>()) continue;
            new_key = static_cast<StorageKeyT>(keys[count - 1] + 1);
          }

          const uint64_t delta = static_cast<uint64_t>(new_key - kBaseKey);
          const bool expected_fits = (count < spec.capacity) &&
                                     (spec.is_raw || delta <= spec.max_delta);
          const KeyT new_uk = KeyCodec<KeyT>::FromStorage(new_key);
          const ValT new_uv = static_cast<ValT>(slot * 17 + 3);
          TreeT scratch = build_leaf_tree(spec, count);
          bool inserted = false;
          size_t ret_slot = ~size_t{0};
          if constexpr (kIsMap) {
            auto [it, ok] = scratch.insert(new_uk, new_uv);
            inserted = ok;
            ret_slot = it.slot();
          } else {
            auto [it, ok] = scratch.insert(new_uk);
            inserted = ok;
            ret_slot = it.slot();
          }
          HWY_ASSERT(inserted);
          HWY_ASSERT_EQ(scratch.size(), count + 1);
          if (expected_fits) {
            HWY_ASSERT_EQ(scratch.state()->tree_height_, size_t{0});
            HWY_ASSERT_EQ(scratch.state()->num_leaves_, size_t{1});
            const auto* s_leaf = scratch.state()->first_leaf_;
            VerifyLeafNode(s_leaf);
            HWY_ASSERT_EQ(s_leaf->BitMode(), spec.mode);
            HWY_ASSERT_EQ(s_leaf->NumKeys(), count + 1);
            HWY_ASSERT_EQ(ret_slot, slot);
            for (size_t j = 0; j < count + 1; ++j) {
              const StorageKeyT expected_k =
                  (j < slot) ? keys[j] : (j == slot ? new_key : keys[j - 1]);
              HWY_ASSERT_EQ(GetLeafKey(s_leaf, j), expected_k);
            }
            if constexpr (kIsMap) {
              auto it = scratch.begin();
              for (size_t j = 0; j < count + 1; ++j, ++it) {
                const ValT expected_v =
                    (j < slot) ? user_vals[j]
                               : (j == slot ? new_uv : user_vals[j - 1]);
                HWY_ASSERT_EQ(it->second, expected_v);
              }
            }
          } else {
            BTreeChecker<TreeT, TreeT>::VerifyPhysicalTree(scratch);
            HWY_ASSERT(scratch.state()->num_leaves_ == 2 ||
                       scratch.state()->first_leaf_->BitMode() > spec.mode);
            HWY_ASSERT(scratch.contains(new_uk));
          }
        }
      }
    }
  }
}

// Sibling rebalancing on insert (Tier 2.5): when a leaf is full, keys are
// redistributed with an adjacent sibling instead of allocating a new leaf.
// The observable difference between "rebalanced" and "split" is num_leaves_,
// so every scenario below asserts on it, then re-validates the physical tree.
template <typename TreeT>
void DoSiblingRebalanceTest() {
  using key_type = typename TreeT::key_type;
  using value_type = typename TreeT::value_type;
  using LaneKeyT = typename KeyCodec<key_type>::LaneType;
  using StorageKeyT = typename KeyCodec<key_type>::StorageKey;
  using InternalNodeT = InternalNode<StorageKeyT>;
  using Checker = BTreeChecker<TreeT, TreeT>;
  using LeafPtrT = decltype(std::declval<TreeT>().state()->first_leaf_);
  using LeafT = std::remove_pointer_t<LeafPtrT>;

  const auto make_value = [](LaneKeyT raw_k) -> value_type {
    const key_type k = KeyFromRaw<key_type>(raw_k);
    if constexpr (TreeT::kIsMap) {
      return {k, static_cast<typename TreeT::mapped_type>(raw_k)};
    } else {
      return k;
    }
  };
  const auto leaf_key = [](const LeafT* leaf, size_t slot) -> LaneKeyT {
    return KeyToRaw<key_type>(
        KeyCodec<key_type>::FromStorage(GetLeafKey(leaf, slot)));
  };
  const auto nth_leaf = [](const TreeT& tree, size_t n) -> LeafPtrT {
    LeafPtrT leaf = tree.state()->first_leaf_;
    for (size_t i = 0; i < n; ++i) leaf = leaf->Next();
    return leaf;
  };
  // Separator between two adjacent leaves must satisfy
  // left.max < sep <= right.min (any such value routes correctly).
  const auto check_sep = [](StorageKeyT sep, const LeafT* left,
                            const LeafT* right) {
    HWY_ASSERT(GetLeafKey(left, left->NumKeys() - 1) < sep);
    HWY_ASSERT(sep <= right->base_key);
  };

  // Full 16-bit leaf capacity (the common mode); keys with step 200 force
  // 16-bit mode after the 3rd key and keep a full leaf's span < 65536.
  constexpr size_t kCap16 = LeafT::kMax16;
  constexpr LaneKeyT kStep = 200;
  constexpr LaneKeyT kBase = 1000;

  // ---------------------------------------------------------------------------
  // Scenario 1: rebalance with the right sibling, then with the left sibling,
  // then the append position bias. Start from 4 leaves under one root:
  // ascending appends use boundary-biased splits, so L0..L2 are full and L3
  // holds whatever we add after the 4th leaf appears.
  // ---------------------------------------------------------------------------
  {
    TreeT tree;
    LaneKeyT next_key = kBase;
    while (tree.state()->num_leaves_ < 4) {
      tree.insert(make_value(next_key));
      next_key += kStep;
    }
    HWY_ASSERT_EQ(tree.state()->tree_height_, size_t{1});
    for (size_t i = 0; i < 3; ++i) {
      HWY_ASSERT_EQ(nth_leaf(tree, i)->NumKeys(), kCap16);
    }
    // Grow L3 to about half capacity.
    while (nth_leaf(tree, 3)->NumKeys() < kCap16 / 2) {
      tree.insert(make_value(next_key));
      next_key += kStep;
    }
    auto* root = static_cast<const InternalNodeT*>(tree.state()->root_);

    // 1a. Insert into the middle of the full L2. L1 (left) is full, so the
    // rebalance must use L3 (right): no new leaf, both at ~50/50, and the
    // separator L2|L3 is valid for the new contents.
    {
      LeafPtrT l2 = nth_leaf(tree, 2);
      LeafPtrT l3 = nth_leaf(tree, 3);
      const size_t total = l2->NumKeys() + l3->NumKeys() + 1;
      const LaneKeyT probe = leaf_key(l2, 3) + kStep / 2;
      HWY_ASSERT(tree.insert(make_value(probe)).second);
      HWY_ASSERT_EQ(tree.state()->num_leaves_, size_t{4});
      HWY_ASSERT_EQ(l2->NumKeys(), total / 2);
      HWY_ASSERT_EQ(l3->NumKeys(), total - total / 2);
      check_sep(root->keys[2], l2, l3);
      HWY_ASSERT(tree.contains(KeyFromRaw<key_type>(probe)));
      Checker::VerifyPhysicalTree(tree);
    }

    // 1b. Fill L3 back up via its gaps, then insert once more into a middle
    // gap: L3 is full and L2 (left) now has room, so the rebalance must use
    // the left sibling.
    {
      LeafPtrT l2 = nth_leaf(tree, 2);
      LeafPtrT l3 = nth_leaf(tree, 3);
      const LaneKeyT lo = leaf_key(l3, 0);
      size_t gap = 0;
      while (l3->NumKeys() < kCap16) {
        const LaneKeyT k = lo + kStep * static_cast<LaneKeyT>(gap) + kStep / 2;
        HWY_ASSERT(tree.insert(make_value(k)).second);
        ++gap;
      }
      HWY_ASSERT_EQ(tree.state()->num_leaves_, size_t{4});
      const size_t total = l2->NumKeys() + l3->NumKeys() + 1;
      const LaneKeyT probe = leaf_key(l3, l3->NumKeys() - 2) + kStep / 2;
      HWY_ASSERT(tree.insert(make_value(probe)).second);
      HWY_ASSERT_EQ(tree.state()->num_leaves_, size_t{4});
      HWY_ASSERT_EQ(l2->NumKeys(), total / 2);
      HWY_ASSERT_EQ(l3->NumKeys(), total - total / 2);
      check_sep(root->keys[2], l2, l3);
      HWY_ASSERT(tree.contains(KeyFromRaw<key_type>(probe)));
      Checker::VerifyPhysicalTree(tree);
    }

    // 1c. Position bias: thin L2 a little (staying above the erase-merge
    // threshold), fill L3 by appending at the end, then append once more. The
    // new key is the maximum of L2 ∪ L3, so instead of a 50/50 cut the left
    // sibling L2 is packed (strictly more than half) and the free space stays
    // in L3 where the ascending run continues.
    {
      LeafPtrT l2 = nth_leaf(tree, 2);
      LeafPtrT l3 = nth_leaf(tree, 3);
      while (l2->NumKeys() > kCap16 * 3 / 5) {
        HWY_ASSERT_EQ(tree.erase(KeyFromRaw<key_type>(leaf_key(l2, 1))),
                      size_t{1});
      }
      HWY_ASSERT_EQ(tree.state()->num_leaves_, size_t{4});
      while (l3->NumKeys() < kCap16) {
        HWY_ASSERT(tree.insert(make_value(next_key)).second);
        next_key += kStep;
      }
      const size_t l2_before = l2->NumKeys();
      const size_t total = l2_before + l3->NumKeys() + 1;
      HWY_ASSERT(tree.insert(make_value(next_key)).second);
      HWY_ASSERT_EQ(tree.state()->num_leaves_, size_t{4});
      HWY_ASSERT(l2->NumKeys() > total / 2);
      HWY_ASSERT(l2->NumKeys() > l2_before);
      HWY_ASSERT(l2->NumKeys() <= kCap16);
      HWY_ASSERT_EQ(l2->NumKeys() + l3->NumKeys(), total);
      check_sep(root->keys[2], l2, l3);
      HWY_ASSERT(tree.contains(KeyFromRaw<key_type>(next_key)));
      Checker::VerifyPhysicalTree(tree);
    }
  }

  // ---------------------------------------------------------------------------
  // Scenario 2: the O(1) pre-check passes but the exact post-check must
  // reject. A = dense 8-bit leaf (plenty of raw 8-bit slots), B = full 16-bit
  // leaf far away. Pre-check only looks at A's current-mode capacity and
  // passes; after decompression, A's half would also contain B's first keys,
  // whose distance forces 32-bit (or raw 64-bit) mode with a much smaller
  // capacity, so the rebalance is rejected and B must split instead.
  // ---------------------------------------------------------------------------
  {
    constexpr size_t kDense = kCap16 - 6;  // > kMax32, fits one 8-bit leaf
    static_assert(kDense <= HWY_MIN(LeafT::kMax8, size_t{256}));
    static_assert(kDense + 1 > LeafT::kMax32);
    constexpr LaneKeyT kFar = 200000;

    TreeT tree;
    for (size_t i = 0; i < kDense; ++i) {
      tree.insert(make_value(static_cast<LaneKeyT>(kBase + i)));
    }
    HWY_ASSERT_EQ(tree.state()->num_leaves_, size_t{1});
    // First far key cannot fit A in any mode -> boundary-biased root split:
    // A keeps all dense keys, B starts with one key.
    for (size_t i = 0; i < kCap16; ++i) {
      tree.insert(make_value(kFar + kStep * static_cast<LaneKeyT>(i)));
    }
    HWY_ASSERT_EQ(tree.state()->num_leaves_, size_t{2});
    LeafPtrT a = nth_leaf(tree, 0);
    LeafPtrT b = nth_leaf(tree, 1);
    HWY_ASSERT_EQ(a->NumKeys(), kDense);
    HWY_ASSERT_EQ(a->BitMode(), static_cast<uint8_t>(kMode8Bit));
    HWY_ASSERT_EQ(b->NumKeys(), kCap16);
    HWY_ASSERT_EQ(b->BitMode(), static_cast<uint8_t>(kMode16Bit));

    // Insert into the middle of full B: B has no right sibling, A is too far
    // away to share keys -> split (num_leaves_ 2 -> 3), no corruption.
    const LaneKeyT probe =
        kFar + kStep * static_cast<LaneKeyT>(kCap16 / 2) + kStep / 2;
    HWY_ASSERT(tree.insert(make_value(probe)).second);
    HWY_ASSERT_EQ(tree.state()->num_leaves_, size_t{3});
    HWY_ASSERT_EQ(a->NumKeys(), kDense);  // A untouched
    Checker::VerifyPhysicalTree(tree);
    for (size_t i = 0; i < kDense; ++i) {
      HWY_ASSERT(
          tree.contains(KeyFromRaw<key_type>(static_cast<LaneKeyT>(kBase + i))));
    }
    for (size_t i = 0; i < kCap16; ++i) {
      HWY_ASSERT(tree.contains(
          KeyFromRaw<key_type>(kFar + kStep * static_cast<LaneKeyT>(i))));
    }
    HWY_ASSERT(tree.contains(KeyFromRaw<key_type>(probe)));
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
      std::is_same_v<typename TreeT::key_compare, std::less<key_type>>);

  // 2. Runtime comparator observer verification (similar to absl::btree tests)
  TreeT tree;
  auto comp = tree.key_comp();
  const key_type k1 = KeyFromRawInt<key_type>(1);
  const key_type k2 = KeyFromRawInt<key_type>(2);
  HWY_ASSERT(comp(k1, k2));
  HWY_ASSERT(!comp(k2, k2));
  HWY_ASSERT(!comp(k2, k1));

  auto val_comp = tree.value_comp();
  if constexpr (TreeT::kIsMap) {
    using mapped_type = typename TreeT::mapped_type;
    value_type v1{k1, mapped_type{10}};
    value_type v2{k2, mapped_type{20}};
    HWY_ASSERT(val_comp(v1, v2));
    HWY_ASSERT(!val_comp(v2, v2));
    HWY_ASSERT(!val_comp(v2, v1));
  } else {
    HWY_ASSERT(val_comp(k1, k2));
    HWY_ASSERT(!val_comp(k2, k2));
    HWY_ASSERT(!val_comp(k2, k1));
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
    key_type non_exist = KeyFromRawInt<key_type>(999999);
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
      overwriting_tree.insert(KeyFromRawInt<key_type>(12345),
                              typename TreeT::mapped_type{999});
    } else {
      overwriting_tree.insert(KeyFromRawInt<key_type>(12345));
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
    const key_type k10 = KeyFromRawInt<key_type>(10);
    const key_type k20 = KeyFromRawInt<key_type>(20);
    if constexpr (TreeT::kIsMap) {
      a.insert(k10, typename TreeT::mapped_type{100});
      b.insert(k20, typename TreeT::mapped_type{200});
    } else {
      a.insert(k10);
      b.insert(k20);
    }
    a.swap(b);
    HWY_ASSERT(a.contains(k20));
    HWY_ASSERT(b.contains(k10));

    using std::swap;
    swap(a, b);
    HWY_ASSERT(a.contains(k10));
    HWY_ASSERT(b.contains(k20));

    // Initializer list assignment
    if constexpr (!TreeT::kIsMap) {
      const key_type k1 = KeyFromRawInt<key_type>(1);
      const key_type k2 = KeyFromRawInt<key_type>(2);
      const key_type k3 = KeyFromRawInt<key_type>(3);
      a = {k1, k2, k3};
      HWY_ASSERT_EQ(a.size(), size_t{3});
      HWY_ASSERT(a.contains(k1));
      HWY_ASSERT(a.contains(k2));
      HWY_ASSERT(a.contains(k3));
    }
  }

  // 3. Multi-level tree scale copy test
  {
    const size_t kLargeScale =
        HWY_BTREE_SWEEP_ALL ? size_t{200000} : AdjustedReps(5000);
    auto large_vals =
        GenerateValuesWithSeed<value_type>(kLargeScale, kLargeScale * 20, 888);
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
  using LaneKeyT = typename KeyCodec<key_type>::LaneType;
  static constexpr bool kIsMap = TreeT::kIsMap;

  std::vector<key_type> special_keys = {
      KeyFromRaw<key_type>(std::numeric_limits<LaneKeyT>::min()),
      KeyFromRaw<key_type>(
          static_cast<LaneKeyT>(std::numeric_limits<LaneKeyT>::min() + 1)),
      KeyFromRaw<key_type>(static_cast<LaneKeyT>(-1)),
      KeyFromRaw<key_type>(static_cast<LaneKeyT>(0)),
      KeyFromRaw<key_type>(static_cast<LaneKeyT>(1)),
      KeyFromRaw<key_type>(
          static_cast<LaneKeyT>(std::numeric_limits<LaneKeyT>::max() - 1)),
      KeyFromRaw<key_type>(std::numeric_limits<LaneKeyT>::max()),
  };

  std::vector<value_type> vals;
  std::set<key_type> seen;
  for (key_type k : special_keys) {
    if (seen.insert(k).second) {
      if constexpr (kIsMap) {
        vals.push_back({k, static_cast<typename TreeT::mapped_type>(
                               KeyToRawU64(k) & 0xFF)});
      } else {
        vals.push_back(k);
      }
    }
  }

  // Add 600 intermediate keys to ensure tree height >= 1 and internal node
  // splits.
  for (int64_t i = -300; i <= 300; ++i) {
    key_type k = KeyFromRaw<key_type>(static_cast<LaneKeyT>(i * 1000));
    if (seen.insert(k).second) {
      if constexpr (kIsMap) {
        vals.push_back({k, static_cast<typename TreeT::mapped_type>(
                               KeyToRawU64(k) & 0xFF)});
      } else {
        vals.push_back(k);
      }
    }
  }

  DoFullContainerTest<TreeT, StdRefT>(vals, /*seed=*/77777);
}

// Verifies erase(iterator): the returned iterator must designate the same
// element as the std:: reference's erase(iterator), across leaf merges, leaf
// frees, and first_leaf_/last_leaf_ updates. Runs on a dense (8-bit delta)
// and a sparse (wide delta / raw) key distribution.
template <typename TreeT, typename StdRefT>
void DoEraseIteratorTest() {
  using value_type = typename TreeT::value_type;
  using key_type = typename TreeT::key_type;
  using Checker = BTreeChecker<TreeT, StdRefT>;

  // Map iterators yield proxy references (not std::pair), so extract the key
  // structurally rather than via KeyExtractor.
  const auto key_of = [](const auto& v) -> key_type {
    if constexpr (IsPairLike<std::decay_t<decltype(v)>>::value) {
      return v.first;
    } else {
      return v;
    }
  };

  const size_t n = HWY_BTREE_SWEEP_ALL ? size_t{100000} : AdjustedReps(4000);
  const uint64_t sparse_max =
      (sizeof(key_type) == 4) ? std::min<uint64_t>(n * 100000ULL, 2000000000ULL)
                              : n * 100000ULL;
  const std::vector<std::vector<value_type>> datasets = {
      GenerateValuesWithSeed<value_type>(n, n + n / 4, /*seed=*/4242),
      GenerateValuesWithSeed<value_type>(n, sparse_max, /*seed=*/4243),
  };

  for (const auto& values : datasets) {
    // 1. Single forward pass, erasing every element whose key hash hits a
    //    predicate. Compare the returned iterator to std::'s at every step.
    {
      TreeT tree;
      StdRefT ref;
      for (const auto& v : values) {
        tree.insert(v);
        ref.insert(v);
      }
      Checker::VerifyPhysicalTree(tree);

      auto it = tree.begin();
      auto ref_it = ref.begin();
      size_t erased = 0;
      while (it != tree.end()) {
        HWY_ASSERT(ref_it != ref.end());
        VerifyEqualElements(it, ref_it);
        const uint64_t k = KeyToRawU64(key_of(*it));
        if ((k * 0x9E3779B97F4A7C15ULL) >> 62 != 0) {  // erase ~75%
          it = tree.erase(it);
          ref_it = ref.erase(ref_it);
          ++erased;
          HWY_ASSERT_EQ(tree.size(), ref.size());
          if (it == tree.end()) {
            HWY_ASSERT(ref_it == ref.end());
          } else {
            HWY_ASSERT(ref_it != ref.end());
            VerifyEqualElements(it, ref_it);
          }
          if ((erased & 255) == 0) {
            Checker::VerifyPhysicalTree(tree);
          }
        } else {
          ++it;
          ++ref_it;
        }
      }
      HWY_ASSERT(ref_it == ref.end());
      HWY_ASSERT_EQ(tree.size(), ref.size());
      Checker::VerifyPhysicalTree(tree);
      // Remaining elements must match exactly.
      auto t2 = tree.begin();
      for (auto r2 = ref.begin(); r2 != ref.end(); ++r2, ++t2) {
        HWY_ASSERT(t2 != tree.end());
        VerifyEqualElements(t2, r2);
      }
      HWY_ASSERT(t2 == tree.end());
    }

    // 2. Erase from the front until empty: returned iterator is the new
    //    begin(); exercises leaf frees and first_leaf_ maintenance.
    {
      TreeT tree;
      StdRefT ref;
      for (const auto& v : values) {
        tree.insert(v);
        ref.insert(v);
      }
      size_t steps = 0;
      while (!tree.empty()) {
        auto it = tree.erase(tree.begin());
        ref.erase(ref.begin());
        HWY_ASSERT(it == tree.begin());
        HWY_ASSERT_EQ(tree.size(), ref.size());
        if (tree.empty()) {
          HWY_ASSERT(it == tree.end());
        } else {
          VerifyEqualElements(it, ref.begin());
        }
        if ((++steps & 511) == 0) {
          Checker::VerifyPhysicalTree(tree);
        }
      }
      HWY_ASSERT(tree.begin() == tree.end());
      HWY_ASSERT(tree.state()->root_ == nullptr);
    }

    // 3. Erase from the back until empty: returned iterator is always end();
    //    exercises last_leaf_ maintenance (and thus --end()).
    {
      TreeT tree;
      StdRefT ref;
      for (const auto& v : values) {
        tree.insert(v);
        ref.insert(v);
      }
      size_t steps = 0;
      while (!tree.empty()) {
        auto last = tree.end();
        --last;
        auto it = tree.erase(last);
        ref.erase(std::prev(ref.end()));
        HWY_ASSERT(it == tree.end());
        HWY_ASSERT_EQ(tree.size(), ref.size());
        if (!tree.empty()) {
          auto fresh_last = tree.end();
          --fresh_last;
          VerifyEqualElements(fresh_last, std::prev(ref.end()));
        }
        if ((++steps & 511) == 0) {
          Checker::VerifyPhysicalTree(tree);
        }
      }
      HWY_ASSERT(tree.state()->root_ == nullptr);
    }

    // 4. Erase the last key of every leaf: the successor always lives in the
    //    next leaf, so the re-descent must cross leaf boundaries correctly.
    {
      TreeT tree;
      StdRefT ref;
      for (const auto& v : values) {
        tree.insert(v);
        ref.insert(v);
      }
      // Collect last-key-of-leaf positions first (erasing mutates leaves).
      std::vector<key_type> targets;
      for (auto it = tree.begin(); it != tree.end(); ++it) {
        if (it.slot() + 1 == it.leaf()->NumKeys()) {
          targets.push_back(key_of(*it));
        }
      }
      HWY_ASSERT(targets.size() >= 2);  // multi-leaf tree
      for (const key_type k : targets) {
        auto pos = tree.find(k);
        HWY_ASSERT(pos != tree.end());
        auto ref_pos = ref.find(k);
        auto it = tree.erase(pos);
        auto ref_it = ref.erase(ref_pos);
        if (ref_it == ref.end()) {
          HWY_ASSERT(it == tree.end());
        } else {
          HWY_ASSERT(it != tree.end());
          VerifyEqualElements(it, ref_it);
        }
      }
      HWY_ASSERT_EQ(tree.size(), ref.size());
      Checker::VerifyPhysicalTree(tree);
    }
  }
}

// Engine-only: moves must transfer contents through an adapter's external
// state, matching copy-assignment and swap. Not part of RunFullTestSuite
// because the dynamic-dispatch wrappers (hwy::BTreeSet / hwy::BTreeMap) have no
// adapter constructor; btree_test.cc calls this directly.
template <typename TreeT>
void DoAdapterMoveTest() {
  using key_type = typename TreeT::key_type;
  using State = typename TreeT::State;
  constexpr size_t kN = 100;

  const auto insert_range = [](TreeT& tree, size_t first, size_t count) {
    for (size_t i = first; i < first + count; ++i) {
      const key_type k = KeyFromRawInt<key_type>(i);
      if constexpr (TreeT::kIsMap) {
        tree.insert(k, typename TreeT::mapped_type{});
      } else {
        tree.insert(k);
      }
    }
  };
  const auto contains_range = [](const TreeT& tree, size_t first,
                                 size_t count) {
    for (size_t i = first; i < first + count; ++i) {
      HWY_ASSERT(tree.contains(KeyFromRawInt<key_type>(i)));
    }
  };

  // Move-assign into an adapter writes through to the external state.
  {
    State ext;
    TreeT src;
    insert_range(src, 0, kN);
    {
      TreeT adapter(&ext);
      adapter = std::move(src);
    }  // adapter is non-owning: must not free ext's nodes.
    HWY_ASSERT(src.empty());
    HWY_ASSERT_EQ(size_t{kN}, ext.num_elements_);
    TreeT check(&ext);
    contains_range(check, 0, kN);
    check.clear();
  }

  // Move-assign into an adapter that already holds nodes frees them first.
  {
    State ext;
    {
      TreeT pre(&ext);
      insert_range(pre, 1000, kN);
    }
    TreeT src;
    insert_range(src, 0, kN);
    {
      TreeT adapter(&ext);
      adapter = std::move(src);
    }
    TreeT check(&ext);
    HWY_ASSERT_EQ(size_t{kN}, check.size());
    contains_range(check, 0, kN);
    HWY_ASSERT(!check.contains(KeyFromRawInt<key_type>(1000)));
    check.clear();
  }

  // Move-construct from an adapter yields an owning tree and empties the
  // external state.
  {
    State ext;
    {
      TreeT pre(&ext);
      insert_range(pre, 0, kN);
    }
    {
      TreeT adapter(&ext);
      TreeT moved(std::move(adapter));
      HWY_ASSERT(moved.state() != &ext);
      HWY_ASSERT_EQ(size_t{kN}, moved.size());
      HWY_ASSERT(adapter.empty());
      HWY_ASSERT_EQ(size_t{0}, ext.num_elements_);
      HWY_ASSERT(ext.root_ == nullptr);
      contains_range(moved, 0, kN);
    }  // moved owns the nodes and frees them here.
  }

  // Move-assign from an adapter into an owning tree empties the external
  // state.
  {
    State ext;
    {
      TreeT pre(&ext);
      insert_range(pre, 0, kN);
    }
    TreeT dst;
    insert_range(dst, 7777, 1);
    {
      TreeT adapter(&ext);
      dst = std::move(adapter);
    }
    HWY_ASSERT_EQ(size_t{0}, ext.num_elements_);
    HWY_ASSERT_EQ(size_t{kN}, dst.size());
    HWY_ASSERT(!dst.contains(KeyFromRawInt<key_type>(7777)));
  }

  // Moving or copying between two adapters of the same external state is a
  // no-op (no rebuild: the root pointer is unchanged).
  {
    State ext;
    {
      TreeT pre(&ext);
      insert_range(pre, 0, kN);
    }
    {
      TreeT a(&ext);
      TreeT b(&ext);
      const void* root_before = ext.root_;
      a = std::move(b);
      HWY_ASSERT(ext.root_ == root_before);
      HWY_ASSERT_EQ(size_t{kN}, ext.num_elements_);
      a = b;
      HWY_ASSERT(ext.root_ == root_before);
      HWY_ASSERT_EQ(size_t{kN}, a.size());
    }
    TreeT cleanup(&ext);
    cleanup.clear();
  }
}

template <typename TreeT, typename StdRefT>
void RunFullTestSuite() {
  DoTypedefsAndObserversTest<TreeT>();
  DoCopyAndSwapTest<TreeT, StdRefT>();
  DoBoundarySizeSweep<TreeT, StdRefT>();
  DoDiverseBitModesTest<TreeT>();
  DoExhaustiveLeafKernelSweepTest<TreeT>();
  DoSiblingRebalanceTest<TreeT>();
  DoExtremeBoundariesTest<TreeT, StdRefT>();
  DoEraseIteratorTest<TreeT, StdRefT>();

  // Multi-level scale across sorted, rsorted, and random orderings (matching
  // absl::btree_test.cc's BtreeTest()). AdjustedReps ensures ASan/MSan/QEMU
  // builds don't timeout; HWY_BTREE_SWEEP_ALL runs a ladder of scales to hit
  // every tree height (Height 1, 2, 3, and 4) across all test phases.
  const std::vector<size_t> scales =
      HWY_BTREE_SWEEP_ALL ? std::vector<size_t>{2500, 15000, 60000, 320000}
                          : std::vector<size_t>{AdjustedReps(5000)};

  uint32_t seed_offset = 0;
  for (size_t scale : scales) {
    auto random_values = GenerateValuesWithSeed<typename TreeT::value_type>(
        scale, scale * 50, /*seed=*/98765 + seed_offset);

    auto sorted_values = random_values;
    std::sort(sorted_values.begin(), sorted_values.end(),
              ValueComparator<typename TreeT::value_type>{});
    DoFullContainerTest<TreeT, StdRefT>(sorted_values,
                                        /*seed=*/98765 + seed_offset);

    auto rsorted_values = sorted_values;
    std::reverse(rsorted_values.begin(), rsorted_values.end());
    DoFullContainerTest<TreeT, StdRefT>(rsorted_values,
                                        /*seed=*/98766 + seed_offset);

    DoFullContainerTest<TreeT, StdRefT>(random_values,
                                        /*seed=*/98767 + seed_offset);
    DoBulkBuildAndBatchTest<TreeT, StdRefT>(scale,
                                            /*seed=*/56789 + seed_offset);
    seed_offset += 100;
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_BTREE_BTREE_TEST_UTIL_INL_H_
