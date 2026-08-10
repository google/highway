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

// Highway SIMD B-Tree
//
// Co-designs in-memory search trees for SIMD vector registers and 64-byte L1
// cache lines. Uses branchless SIMD comparison (hn::Lt / hn::Le) + CountTrue
// to navigate tree nodes in ~2 clock cycles without binary search loops.

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

#include "hwy/base.h"
#include "hwy/cache_control.h"

#if defined(HIGHWAY_HWY_CONTRIB_BTREE_BTREE_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_BTREE_BTREE_INL_H_
#undef HIGHWAY_HWY_CONTRIB_BTREE_BTREE_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_BTREE_BTREE_INL_H_
#endif

#include "hwy/highway.h"

static_assert(HWY_CXX_LANG >= 201703L, "requires C++17 or later.");

#if HWY_TARGET != HWY_SCALAR
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename KeyT>
struct BTreeTraits {
  // 64 bytes per cache line / sizeof(KeyT)
  static constexpr size_t kKeysPerNode = 64 / sizeof(KeyT);
  static constexpr size_t kMaxChildren = kKeysPerNode + 1;
};

// -----------------------------------------------------------------------------
// Helper: SIMD Key Array Initialization
// -----------------------------------------------------------------------------

template <typename KeyT>
HWY_INLINE void InitKeysWithSentinel(KeyT* HWY_RESTRICT keys) {
  constexpr size_t kTotalKeys = BTreeTraits<KeyT>::kKeysPerNode;
  const hn::CappedTag<KeyT, kTotalKeys> d;
  const size_t N = hn::Lanes(d);
  const KeyT sentinel = std::numeric_limits<KeyT>::max();
  const auto v_sentinel = hn::Set(d, sentinel);

  for (size_t i = 0; i < kTotalKeys; i += N) {
    hn::StoreU(v_sentinel, d, keys + i);
  }
}

// -----------------------------------------------------------------------------
// Node Definitions (Structure-of-Arrays aligned to 64-byte cache lines)
// -----------------------------------------------------------------------------

template <typename KeyT>
struct LeafNode {
  static constexpr size_t kCapacity = BTreeTraits<KeyT>::kKeysPerNode;

  KeyT keys[kCapacity];
  LeafNode* next = nullptr;
  LeafNode* prev = nullptr;
  uint16_t num_keys = 0;
  uint16_t level = 0;  // Level 0 for leaves

  LeafNode() { InitKeysWithSentinel(keys); }
};

template <typename KeyT>
struct InternalNode {
  static constexpr size_t kCapacity = BTreeTraits<KeyT>::kKeysPerNode;
  static constexpr size_t kMaxChildren = BTreeTraits<KeyT>::kMaxChildren;

  KeyT keys[kCapacity];
  void* children[kMaxChildren] = {};  // Zero-initialized to nullptr
  uint16_t num_keys = 0;
  uint16_t level = 1;  // level >= 1 for internal nodes

  InternalNode() { InitKeysWithSentinel(keys); }
};

// -----------------------------------------------------------------------------
// SIMD Search Primitives
// -----------------------------------------------------------------------------

// Returns child index in [0, num_keys] for internal node navigation by
// counting the number of separator keys <= target.
// children[0] holds keys < keys[0].
// children[i] holds keys >= keys[i-1] and < keys[i].
template <typename KeyT>
HWY_INLINE size_t FindChild(const InternalNode<KeyT>* HWY_RESTRICT node,
                            KeyT target) {
  constexpr size_t kTotalKeys = BTreeTraits<KeyT>::kKeysPerNode;
  const hn::CappedTag<KeyT, kTotalKeys> d;
  const size_t N = hn::Lanes(d);
  const auto v_target = hn::Set(d, target);

  size_t count = 0;
  for (size_t i = 0; i < kTotalKeys; i += N) {
    const auto v_keys = hn::LoadU(d, node->keys + i);
    const auto mask = hn::Le(v_keys, v_target);
    count += hn::CountTrue(d, mask);
  }
  return count;
}

// Searches leaf node. Returns index of first key >= target by counting the
// number of keys < target (lower_bound index in [0, num_keys]).
template <typename KeyT>
HWY_INLINE size_t FindLeafSlot(const LeafNode<KeyT>* HWY_RESTRICT leaf,
                               KeyT target) {
  constexpr size_t kTotalKeys = BTreeTraits<KeyT>::kKeysPerNode;
  const hn::CappedTag<KeyT, kTotalKeys> d;
  const size_t N = hn::Lanes(d);
  const auto v_target = hn::Set(d, target);

  size_t count = 0;
  for (size_t i = 0; i < kTotalKeys; i += N) {
    const auto v_keys = hn::LoadU(d, leaf->keys + i);
    const auto mask = hn::Lt(v_keys, v_target);
    count += hn::CountTrue(d, mask);
  }
  return count;
}

// -----------------------------------------------------------------------------
// BTreeSet (Static Bulk-Loaded SIMD B-Tree)
// -----------------------------------------------------------------------------

template <typename KeyT>
class BTreeSet {
 public:
  using Traits = BTreeTraits<KeyT>;
  using key_type = KeyT;
  using value_type = KeyT;
  using size_type = size_t;

  // Bidirectional Range Iterator
  class const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = KeyT;
    using difference_type = std::ptrdiff_t;
    using pointer = const KeyT*;
    using reference = const KeyT&;

    const_iterator() = default;
    const_iterator(const LeafNode<KeyT>* leaf, size_t slot,
                   const LeafNode<KeyT>* last_leaf = nullptr)
        : leaf_(leaf), slot_(slot), last_leaf_(last_leaf) {}

    reference operator*() const { return leaf_->keys[slot_]; }
    pointer operator->() const { return &leaf_->keys[slot_]; }

    const_iterator& operator++() {
      if (leaf_ == nullptr) return *this;
      ++slot_;
      if (slot_ >= leaf_->num_keys) {
        last_leaf_ = leaf_;
        leaf_ = leaf_->next;
        slot_ = 0;
      }
      return *this;
    }

    const_iterator operator++(int) {
      const_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    const_iterator& operator--() {
      if (leaf_ == nullptr) {
        if (last_leaf_ != nullptr && last_leaf_->num_keys > 0) {
          leaf_ = last_leaf_;
          slot_ = last_leaf_->num_keys - 1;
        }
        return *this;
      }
      if (slot_ == 0) {
        if (leaf_->prev != nullptr) {
          leaf_ = leaf_->prev;
          slot_ = leaf_->num_keys > 0 ? leaf_->num_keys - 1 : 0;
        }
      } else {
        --slot_;
      }
      return *this;
    }

    const_iterator operator--(int) {
      const_iterator tmp = *this;
      --(*this);
      return tmp;
    }

    bool operator==(const const_iterator& other) const {
      if (leaf_ == nullptr && other.leaf_ == nullptr) return true;
      return leaf_ == other.leaf_ && slot_ == other.slot_;
    }

    bool operator!=(const const_iterator& other) const {
      return !(*this == other);
    }

    const LeafNode<KeyT>* leaf() const { return leaf_; }
    size_t slot() const { return slot_; }

   private:
    const LeafNode<KeyT>* leaf_ = nullptr;
    size_t slot_ = 0;
    const LeafNode<KeyT>* last_leaf_ = nullptr;
  };

  using iterator = const_iterator;

  BTreeSet() = default;
  BTreeSet(BTreeSet&& other) noexcept
      : root_(other.root_),
        first_leaf_(other.first_leaf_),
        last_leaf_(other.last_leaf_),
        tree_height_(other.tree_height_),
        num_elements_(other.num_elements_),
        leaf_storage_(std::move(other.leaf_storage_)),
        internal_storage_(std::move(other.internal_storage_)) {
    other.root_ = nullptr;
    other.first_leaf_ = nullptr;
    other.last_leaf_ = nullptr;
    other.tree_height_ = 0;
    other.num_elements_ = 0;
  }

  BTreeSet& operator=(BTreeSet&& other) noexcept {
    if (this != &other) {
      root_ = other.root_;
      first_leaf_ = other.first_leaf_;
      last_leaf_ = other.last_leaf_;
      tree_height_ = other.tree_height_;
      num_elements_ = other.num_elements_;
      leaf_storage_ = std::move(other.leaf_storage_);
      internal_storage_ = std::move(other.internal_storage_);

      other.root_ = nullptr;
      other.first_leaf_ = nullptr;
      other.last_leaf_ = nullptr;
      other.tree_height_ = 0;
      other.num_elements_ = 0;
    }
    return *this;
  }

  // Builds a BTreeSet in O(N) time from a sorted, unique sequence of keys.
  static BTreeSet Build(const KeyT* HWY_RESTRICT sorted_keys, size_t num_keys) {
    BTreeSet tree;
    tree.num_elements_ = num_keys;
    if (num_keys == 0) {
      tree.leaf_storage_.resize(1);
      tree.first_leaf_ = &tree.leaf_storage_[0];
      tree.root_ = &tree.leaf_storage_[0];
      tree.tree_height_ = 0;
      return tree;
    }

    // Step 1: Build leaf level in contiguous storage
    const size_t keys_per_leaf = Traits::kKeysPerNode;
    const size_t num_leaves = (num_keys + keys_per_leaf - 1) / keys_per_leaf;
    tree.leaf_storage_.resize(num_leaves);

    std::vector<void*> current_level_ptrs;
    current_level_ptrs.reserve(num_leaves);
    std::vector<KeyT> delimiters;
    if (num_leaves > 1) {
      delimiters.reserve(num_leaves - 1);
    }

    LeafNode<KeyT>* prev_leaf = nullptr;
    for (size_t i = 0; i < num_leaves; ++i) {
      LeafNode<KeyT>& leaf = tree.leaf_storage_[i];
      const size_t start_idx = i * keys_per_leaf;
      const size_t count = std::min(keys_per_leaf, num_keys - start_idx);
      std::copy_n(sorted_keys + start_idx, count, leaf.keys);
      leaf.num_keys = static_cast<uint16_t>(count);

      if (prev_leaf != nullptr) {
        prev_leaf->next = &leaf;
        leaf.prev = prev_leaf;
        // Separator key for parent navigation
        delimiters.push_back(leaf.keys[0]);
      } else {
        tree.first_leaf_ = &leaf;
      }

      current_level_ptrs.push_back(&leaf);
      prev_leaf = &leaf;
    }
    tree.last_leaf_ = prev_leaf;

    // If only 1 leaf, root is the leaf (height = 0)
    if (num_leaves == 1) {
      tree.root_ = tree.first_leaf_;
      tree.tree_height_ = 0;
      return tree;
    }

    // Step 2: Build internal levels bottom-up in contiguous storage
    const size_t max_children = Traits::kMaxChildren;
    size_t total_internals = 0;
    size_t cur_children = num_leaves;
    while (cur_children > 1) {
      size_t cur_internals = (cur_children + max_children - 1) / max_children;
      total_internals += cur_internals;
      cur_children = cur_internals;
    }
    tree.internal_storage_.resize(total_internals);

    uint16_t current_level = 1;
    size_t internal_idx = 0;

    while (current_level_ptrs.size() > 1) {
      const size_t num_children = current_level_ptrs.size();
      const size_t num_internals =
          (num_children + max_children - 1) / max_children;

      std::vector<void*> next_level_ptrs;
      next_level_ptrs.reserve(num_internals);
      std::vector<KeyT> next_delimiters;
      if (num_internals > 1) {
        next_delimiters.reserve(num_internals - 1);
      }

      for (size_t i = 0; i < num_internals; ++i) {
        InternalNode<KeyT>& internal = tree.internal_storage_[internal_idx++];
        internal.level = current_level;

        const size_t child_start = i * max_children;
        const size_t child_count =
            std::min(max_children, num_children - child_start);

        for (size_t c = 0; c < child_count; ++c) {
          internal.children[c] = current_level_ptrs[child_start + c];
        }

        // Fill keys from delimiters
        const size_t key_count = child_count - 1;
        internal.num_keys = static_cast<uint16_t>(key_count);
        for (size_t k = 0; k < key_count; ++k) {
          internal.keys[k] = delimiters[child_start + k];
        }

        if (i > 0) {
          // Promote boundary key to next level
          next_delimiters.push_back(delimiters[child_start - 1]);
        }

        next_level_ptrs.push_back(&internal);
      }

      current_level_ptrs = std::move(next_level_ptrs);
      delimiters = std::move(next_delimiters);
      current_level++;
    }

    tree.root_ = current_level_ptrs[0];
    tree.tree_height_ = current_level - 1;
    return tree;
  }

  // Iterators
  const_iterator begin() const {
    if (num_elements_ == 0 || first_leaf_ == nullptr) return end();
    return const_iterator(first_leaf_, 0, last_leaf_);
  }
  const_iterator end() const { return const_iterator(nullptr, 0, last_leaf_); }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

  // Point lookup: returns true if key exists in O(log16 N) time.
  bool Contains(KeyT target) const {
    if (num_elements_ == 0) return false;
    const LeafNode<KeyT>* leaf = NavigateToLeaf(target);
    const size_t slot = FindLeafSlot(leaf, target);
    // could have made a helper function using hn::Eq() but latency should be
    // the same as FindLeadSlot, so lets just reuse
    return slot < leaf->num_keys && leaf->keys[slot] == target;
  }

  const_iterator find(KeyT target) const {
    if (num_elements_ == 0) return end();
    const LeafNode<KeyT>* leaf = NavigateToLeaf(target);
    const size_t slot = FindLeafSlot(leaf, target);
    if (slot < leaf->num_keys && leaf->keys[slot] == target) {
      return const_iterator(leaf, slot, last_leaf_);
    }
    return end();
  }

  // Ordered Range Query: returns iterator to first key >= target.
  const_iterator lower_bound(KeyT target) const {
    if (num_elements_ == 0) return end();
    const LeafNode<KeyT>* leaf = NavigateToLeaf(target);
    const size_t slot = FindLeafSlot(leaf, target);

    const bool inside = (slot < leaf->num_keys);
    const LeafNode<KeyT>* res_leaf = inside ? leaf : leaf->next;
    const size_t res_slot = inside ? slot : 0;

    return const_iterator(res_leaf, res_slot, last_leaf_);
  }

  // Returns iterator to first key > target.
  const_iterator upper_bound(KeyT target) const {
    if (target == std::numeric_limits<KeyT>::max()) {
      return end();
    }
    return lower_bound(static_cast<KeyT>(target + 1));
  }

  // Pointer-based legacy convenience APIs
  const KeyT* LowerBound(KeyT target) const {
    auto it = lower_bound(target);
    return it != end() ? &(*it) : nullptr;
  }

  const KeyT* UpperBound(KeyT target) const {
    auto it = upper_bound(target);
    return it != end() ? &(*it) : nullptr;
  }

  size_t size() const { return num_elements_; }
  bool empty() const { return num_elements_ == 0; }
  uint16_t height() const { return tree_height_; }

  // Memory footprint in bytes
  size_t AllocatedBytes() const {
    return leaf_storage_.capacity() * sizeof(LeafNode<KeyT>) +
           internal_storage_.capacity() * sizeof(InternalNode<KeyT>);
  }

 private:
  const LeafNode<KeyT>* NavigateToLeaf(KeyT target) const {
    if (tree_height_ == 0) {
      return static_cast<const LeafNode<KeyT>*>(root_);
    }
    const void* curr = root_;
    for (uint16_t h = tree_height_; h > 1; --h) {
      const auto* internal = static_cast<const InternalNode<KeyT>*>(curr);
      curr = internal->children[FindChild(internal, target)];
    }
    const auto* parent_of_leaf = static_cast<const InternalNode<KeyT>*>(curr);
    return static_cast<const LeafNode<KeyT>*>(
        parent_of_leaf->children[FindChild(parent_of_leaf, target)]);
  }

  void* root_ = nullptr;
  LeafNode<KeyT>* first_leaf_ = nullptr;
  LeafNode<KeyT>* last_leaf_ = nullptr;
  uint16_t tree_height_ = 0;
  size_t num_elements_ = 0;

  std::vector<LeafNode<KeyT>> leaf_storage_;
  std::vector<InternalNode<KeyT>> internal_storage_;
};

// -----------------------------------------------------------------------------
// Map Node Definitions (Structure-of-Arrays aligned to 64-byte cache lines)
// -----------------------------------------------------------------------------

template <typename KeyT, typename ValueT>
struct MapLeafNode {
  static constexpr size_t kCapacity = BTreeTraits<KeyT>::kKeysPerNode;

  KeyT keys[kCapacity];
  ValueT values[kCapacity];
  MapLeafNode* next = nullptr;
  MapLeafNode* prev = nullptr;
  uint16_t num_keys = 0;
  uint16_t level = 0;  // Level 0 for leaves

  MapLeafNode() { InitKeysWithSentinel(keys); }
};

// Searches map leaf node. Returns index of first key >= target by counting the
// number of keys < target (lower_bound index in [0, num_keys]).
template <typename KeyT, typename ValueT>
HWY_INLINE size_t
FindLeafSlot(const MapLeafNode<KeyT, ValueT>* HWY_RESTRICT leaf, KeyT target) {
  constexpr size_t kTotalKeys = BTreeTraits<KeyT>::kKeysPerNode;
  const hn::CappedTag<KeyT, kTotalKeys> d;
  const size_t N = hn::Lanes(d);
  const auto v_target = hn::Set(d, target);

  size_t count = 0;
  for (size_t i = 0; i < kTotalKeys; i += N) {
    const auto v_keys = hn::LoadU(d, leaf->keys + i);
    const auto mask = hn::Lt(v_keys, v_target);
    count += hn::CountTrue(d, mask);
  }
  return count;
}

// -----------------------------------------------------------------------------
// BTreeMap (Static Bulk-Loaded SIMD Key-Value B-Tree)
// -----------------------------------------------------------------------------

template <typename KeyT, typename ValueT>
class BTreeMap {
 public:
  using Traits = BTreeTraits<KeyT>;
  using key_type = KeyT;
  using mapped_type = ValueT;
  using value_type = std::pair<const KeyT, ValueT>;
  using size_type = size_t;

  // Bidirectional Range Iterator
  class const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = std::pair<const KeyT&, const ValueT&>;
    using difference_type = std::ptrdiff_t;

    struct KeyValueRef {
      const KeyT& first;
      const ValueT& second;
    };

    struct ArrowProxy {
      KeyValueRef ref;
      const KeyValueRef* operator->() const { return &ref; }
    };

    const_iterator() = default;
    const_iterator(const MapLeafNode<KeyT, ValueT>* leaf, size_t slot,
                   const MapLeafNode<KeyT, ValueT>* last_leaf = nullptr)
        : leaf_(leaf), slot_(slot), last_leaf_(last_leaf) {}

    KeyValueRef operator*() const {
      return {leaf_->keys[slot_], leaf_->values[slot_]};
    }

    ArrowProxy operator->() const {
      return ArrowProxy{KeyValueRef{leaf_->keys[slot_], leaf_->values[slot_]}};
    }

    const_iterator& operator++() {
      if (leaf_ == nullptr) return *this;
      ++slot_;
      if (slot_ >= leaf_->num_keys) {
        last_leaf_ = leaf_;
        leaf_ = leaf_->next;
        slot_ = 0;
      }
      return *this;
    }

    const_iterator operator++(int) {
      const_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    const_iterator& operator--() {
      if (leaf_ == nullptr) {
        if (last_leaf_ != nullptr && last_leaf_->num_keys > 0) {
          leaf_ = last_leaf_;
          slot_ = last_leaf_->num_keys - 1;
        }
        return *this;
      }
      if (slot_ == 0) {
        if (leaf_->prev != nullptr) {
          leaf_ = leaf_->prev;
          slot_ = leaf_->num_keys > 0 ? leaf_->num_keys - 1 : 0;
        }
      } else {
        --slot_;
      }
      return *this;
    }

    const_iterator operator--(int) {
      const_iterator tmp = *this;
      --(*this);
      return tmp;
    }

    bool operator==(const const_iterator& other) const {
      if (leaf_ == nullptr && other.leaf_ == nullptr) return true;
      return leaf_ == other.leaf_ && slot_ == other.slot_;
    }

    bool operator!=(const const_iterator& other) const {
      return !(*this == other);
    }

    const MapLeafNode<KeyT, ValueT>* leaf() const { return leaf_; }
    size_t slot() const { return slot_; }

   private:
    const MapLeafNode<KeyT, ValueT>* leaf_ = nullptr;
    size_t slot_ = 0;
    const MapLeafNode<KeyT, ValueT>* last_leaf_ = nullptr;
  };

  using iterator = const_iterator;

  BTreeMap() = default;
  BTreeMap(BTreeMap&& other) noexcept
      : root_(other.root_),
        first_leaf_(other.first_leaf_),
        last_leaf_(other.last_leaf_),
        tree_height_(other.tree_height_),
        num_elements_(other.num_elements_),
        leaf_storage_(std::move(other.leaf_storage_)),
        internal_storage_(std::move(other.internal_storage_)) {
    other.root_ = nullptr;
    other.first_leaf_ = nullptr;
    other.last_leaf_ = nullptr;
    other.tree_height_ = 0;
    other.num_elements_ = 0;
  }

  BTreeMap& operator=(BTreeMap&& other) noexcept {
    if (this != &other) {
      root_ = other.root_;
      first_leaf_ = other.first_leaf_;
      last_leaf_ = other.last_leaf_;
      tree_height_ = other.tree_height_;
      num_elements_ = other.num_elements_;
      leaf_storage_ = std::move(other.leaf_storage_);
      internal_storage_ = std::move(other.internal_storage_);

      other.root_ = nullptr;
      other.first_leaf_ = nullptr;
      other.last_leaf_ = nullptr;
      other.tree_height_ = 0;
      other.num_elements_ = 0;
    }
    return *this;
  }

  // Builds a BTreeMap in O(N) time from parallel sorted keys and values.
  static BTreeMap Build(const KeyT* HWY_RESTRICT sorted_keys,
                        const ValueT* HWY_RESTRICT values, size_t num_keys) {
    BTreeMap map;
    map.num_elements_ = num_keys;
    if (num_keys == 0) {
      map.leaf_storage_.resize(1);
      map.first_leaf_ = &map.leaf_storage_[0];
      map.last_leaf_ = &map.leaf_storage_[0];
      map.root_ = &map.leaf_storage_[0];
      map.tree_height_ = 0;
      return map;
    }

    // Step 1: Build leaf level in contiguous storage
    const size_t keys_per_leaf = Traits::kKeysPerNode;
    const size_t num_leaves = (num_keys + keys_per_leaf - 1) / keys_per_leaf;
    map.leaf_storage_.resize(num_leaves);

    std::vector<void*> current_level_ptrs;
    current_level_ptrs.reserve(num_leaves);
    std::vector<KeyT> delimiters;
    if (num_leaves > 1) {
      delimiters.reserve(num_leaves - 1);
    }

    MapLeafNode<KeyT, ValueT>* prev_leaf = nullptr;
    for (size_t i = 0; i < num_leaves; ++i) {
      MapLeafNode<KeyT, ValueT>& leaf = map.leaf_storage_[i];
      const size_t start_idx = i * keys_per_leaf;
      const size_t count = std::min(keys_per_leaf, num_keys - start_idx);
      std::copy_n(sorted_keys + start_idx, count, leaf.keys);
      if (values != nullptr) {
        std::copy_n(values + start_idx, count, leaf.values);
      }
      leaf.num_keys = static_cast<uint16_t>(count);

      if (prev_leaf != nullptr) {
        prev_leaf->next = &leaf;
        leaf.prev = prev_leaf;
        delimiters.push_back(leaf.keys[0]);
      } else {
        map.first_leaf_ = &leaf;
      }

      current_level_ptrs.push_back(&leaf);
      prev_leaf = &leaf;
    }
    map.last_leaf_ = prev_leaf;

    // If only 1 leaf, root is the leaf (height = 0)
    if (num_leaves == 1) {
      map.root_ = map.first_leaf_;
      map.tree_height_ = 0;
      return map;
    }

    // Step 2: Build internal levels bottom-up in contiguous storage
    const size_t max_children = Traits::kMaxChildren;
    size_t total_internals = 0;
    size_t cur_children = num_leaves;
    while (cur_children > 1) {
      size_t cur_internals = (cur_children + max_children - 1) / max_children;
      total_internals += cur_internals;
      cur_children = cur_internals;
    }
    map.internal_storage_.resize(total_internals);

    uint16_t current_level = 1;
    size_t internal_idx = 0;

    while (current_level_ptrs.size() > 1) {
      const size_t num_children = current_level_ptrs.size();
      const size_t num_internals =
          (num_children + max_children - 1) / max_children;

      std::vector<void*> next_level_ptrs;
      next_level_ptrs.reserve(num_internals);
      std::vector<KeyT> next_delimiters;
      if (num_internals > 1) {
        next_delimiters.reserve(num_internals - 1);
      }

      for (size_t i = 0; i < num_internals; ++i) {
        InternalNode<KeyT>& internal = map.internal_storage_[internal_idx++];
        internal.level = current_level;

        const size_t child_start = i * max_children;
        const size_t child_count =
            std::min(max_children, num_children - child_start);

        for (size_t c = 0; c < child_count; ++c) {
          internal.children[c] = current_level_ptrs[child_start + c];
        }

        const size_t key_count = child_count - 1;
        internal.num_keys = static_cast<uint16_t>(key_count);
        for (size_t k = 0; k < key_count; ++k) {
          internal.keys[k] = delimiters[child_start + k];
        }

        if (i > 0) {
          next_delimiters.push_back(delimiters[child_start - 1]);
        }

        next_level_ptrs.push_back(&internal);
      }

      current_level_ptrs = std::move(next_level_ptrs);
      delimiters = std::move(next_delimiters);
      current_level++;
    }

    map.root_ = current_level_ptrs[0];
    map.tree_height_ = current_level - 1;
    return map;
  }

  // Builds a BTreeMap in O(N) time from sorted std::pair array.
  static BTreeMap Build(const std::pair<KeyT, ValueT>* HWY_RESTRICT kv_pairs,
                        size_t num_keys) {
    std::vector<KeyT> keys;
    std::vector<ValueT> vals;
    keys.reserve(num_keys);
    vals.reserve(num_keys);
    for (size_t i = 0; i < num_keys; ++i) {
      keys.push_back(kv_pairs[i].first);
      vals.push_back(kv_pairs[i].second);
    }
    return Build(keys.data(), vals.data(), num_keys);
  }

  // Iterators
  const_iterator begin() const {
    if (num_elements_ == 0 || first_leaf_ == nullptr) return end();
    return const_iterator(first_leaf_, 0, last_leaf_);
  }
  const_iterator end() const { return const_iterator(nullptr, 0, last_leaf_); }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

  // Point lookup: returns true if key exists in O(log16 N) time.
  bool Contains(KeyT target) const {
    if (num_elements_ == 0) return false;
    const auto* leaf = NavigateToLeaf(target);
    const size_t slot = FindLeafSlot(leaf, target);
    return slot < leaf->num_keys && leaf->keys[slot] == target;
  }

  const_iterator find(KeyT target) const {
    if (num_elements_ == 0) return end();
    const auto* leaf = NavigateToLeaf(target);
    const size_t slot = FindLeafSlot(leaf, target);
    if (slot < leaf->num_keys && leaf->keys[slot] == target) {
      return const_iterator(leaf, slot, last_leaf_);
    }
    return end();
  }

  // Fast direct value lookup (returns nullptr if key not found)
  const ValueT* FindValue(KeyT target) const {
    if (num_elements_ == 0) return nullptr;
    const auto* leaf = NavigateToLeaf(target);
    const size_t slot = FindLeafSlot(leaf, target);
    if (slot < leaf->num_keys && leaf->keys[slot] == target) {
      return &leaf->values[slot];
    }
    return nullptr;
  }

  ValueT* FindValue(KeyT target) {
    return const_cast<ValueT*>(
        static_cast<const BTreeMap*>(this)->FindValue(target));
  }

  // Ordered Range Query: returns iterator to first key >= target.
  const_iterator lower_bound(KeyT target) const {
    if (num_elements_ == 0) return end();
    const auto* leaf = NavigateToLeaf(target);
    const size_t slot = FindLeafSlot(leaf, target);

    const bool inside = (slot < leaf->num_keys);
    const auto* res_leaf = inside ? leaf : leaf->next;
    const size_t res_slot = inside ? slot : 0;

    return const_iterator(res_leaf, res_slot, last_leaf_);
  }

  // Returns iterator to first key > target.
  const_iterator upper_bound(KeyT target) const {
    if (target == std::numeric_limits<KeyT>::max()) {
      return end();
    }
    return lower_bound(static_cast<KeyT>(target + 1));
  }

  size_t size() const { return num_elements_; }
  bool empty() const { return num_elements_ == 0; }
  uint16_t height() const { return tree_height_; }

  // Memory footprint in bytes
  size_t AllocatedBytes() const {
    return leaf_storage_.capacity() * sizeof(MapLeafNode<KeyT, ValueT>) +
           internal_storage_.capacity() * sizeof(InternalNode<KeyT>);
  }

 private:
  const MapLeafNode<KeyT, ValueT>* NavigateToLeaf(KeyT target) const {
    if (tree_height_ == 0) {
      return static_cast<const MapLeafNode<KeyT, ValueT>*>(root_);
    }
    const void* curr = root_;
    for (uint16_t h = tree_height_; h > 1; --h) {
      const auto* internal = static_cast<const InternalNode<KeyT>*>(curr);
      curr = internal->children[FindChild(internal, target)];
    }
    const auto* parent_of_leaf = static_cast<const InternalNode<KeyT>*>(curr);
    return static_cast<const MapLeafNode<KeyT, ValueT>*>(
        parent_of_leaf->children[FindChild(parent_of_leaf, target)]);
  }

  void* root_ = nullptr;
  MapLeafNode<KeyT, ValueT>* first_leaf_ = nullptr;
  MapLeafNode<KeyT, ValueT>* last_leaf_ = nullptr;
  uint16_t tree_height_ = 0;
  size_t num_elements_ = 0;

  std::vector<MapLeafNode<KeyT, ValueT>> leaf_storage_;
  std::vector<InternalNode<KeyT>> internal_storage_;
};

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
using HWY_NAMESPACE::BTreeMap;
using HWY_NAMESPACE::BTreeSet;
}  // namespace hwy
#endif

#endif  // HWY_TARGET != HWY_SCALAR
#endif  // HIGHWAY_HWY_CONTRIB_BTREE_BTREE_INL_H_
