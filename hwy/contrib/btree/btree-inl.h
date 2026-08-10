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
#include <string.h>

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
HWY_INLINE void FillSentinel(KeyT* HWY_RESTRICT keys, size_t count) {
  std::fill_n(keys, count, std::numeric_limits<KeyT>::max());
}

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
// NodePool Allocator (Chunked 64-byte aligned pages with intrusive free-list)
// -----------------------------------------------------------------------------

template <typename NodeT, size_t kNodesPerPage = 256>
class NodePool {
 public:
  NodePool() = default;
  ~NodePool() = default;

  NodePool(NodePool&& other) noexcept
      : pages_(std::move(other.pages_)),
        free_list_(other.free_list_),
        next_in_page_(other.next_in_page_) {
    other.free_list_ = nullptr;
    other.next_in_page_ = 0;
  }

  NodePool& operator=(NodePool&& other) noexcept {
    if (this != &other) {
      pages_ = std::move(other.pages_);
      free_list_ = other.free_list_;
      next_in_page_ = other.next_in_page_;
      other.free_list_ = nullptr;
      other.next_in_page_ = 0;
    }
    return *this;
  }

  NodePool(const NodePool&) = delete;
  NodePool& operator=(const NodePool&) = delete;

  NodeT* Allocate() {
    if (free_list_ != nullptr) {
      NodeT* node = free_list_;
      NodeT* next_free = nullptr;
      memcpy(&next_free, free_list_, sizeof(NodeT*));
      free_list_ = next_free;
      return new (node) NodeT();
    }
    if (pages_.empty() || next_in_page_ == kNodesPerPage) {
      pages_.push_back(std::make_unique<Page>());
      next_in_page_ = 0;
    }
    return new (&pages_.back()->nodes[next_in_page_++]) NodeT();
  }

  void Deallocate(NodeT* node) {
    if (node == nullptr) return;
    node->~NodeT();
    memcpy(node, &free_list_, sizeof(NodeT*));
    free_list_ = node;
  }

  void Clear() {
    pages_.clear();
    free_list_ = nullptr;
    next_in_page_ = 0;
  }

  size_t AllocatedBytes() const { return pages_.size() * sizeof(Page); }

 private:
  struct alignas(64) Page {
    NodeT nodes[kNodesPerPage];
  };
  std::vector<std::unique_ptr<Page>> pages_;
  NodeT* free_list_ = nullptr;
  size_t next_in_page_ = 0;
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
      while (leaf_ != nullptr && slot_ >= leaf_->num_keys) {
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
        const auto* curr = last_leaf_;
        while (curr != nullptr && curr->num_keys == 0) {
          curr = curr->prev;
        }
        if (curr != nullptr && curr->num_keys > 0) {
          leaf_ = curr;
          slot_ = curr->num_keys - 1;
        }
        return *this;
      }
      if (slot_ == 0) {
        const auto* curr = leaf_->prev;
        while (curr != nullptr && curr->num_keys == 0) {
          curr = curr->prev;
        }
        if (curr != nullptr) {
          leaf_ = curr;
          slot_ = curr->num_keys > 0 ? curr->num_keys - 1 : 0;
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
        leaf_pool_(std::move(other.leaf_pool_)),
        internal_pool_(std::move(other.internal_pool_)) {
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
      leaf_pool_ = std::move(other.leaf_pool_);
      internal_pool_ = std::move(other.internal_pool_);

      other.root_ = nullptr;
      other.first_leaf_ = nullptr;
      other.last_leaf_ = nullptr;
      other.tree_height_ = 0;
      other.num_elements_ = 0;
    }
    return *this;
  }

  // Builds a BTreeSet in O(N) time from a sorted, unique sequence of keys.
  static BTreeSet Build(const KeyT* HWY_RESTRICT sorted_keys, size_t num_keys,
                        float fill_ratio = 1.0f) {
    BTreeSet tree;
    tree.num_elements_ = num_keys;
    if (num_keys == 0) {
      tree.first_leaf_ = tree.leaf_pool_.Allocate();
      tree.last_leaf_ = tree.first_leaf_;
      tree.root_ = tree.first_leaf_;
      tree.tree_height_ = 0;
      return tree;
    }

    // Step 1: Build leaf level in pool
    const size_t keys_per_leaf = std::clamp<size_t>(
        static_cast<size_t>(Traits::kKeysPerNode * fill_ratio), 2,
        Traits::kKeysPerNode);
    const size_t num_leaves = (num_keys + keys_per_leaf - 1) / keys_per_leaf;

    std::vector<void*> current_level_ptrs;
    current_level_ptrs.reserve(num_leaves);
    std::vector<KeyT> delimiters;
    if (num_leaves > 1) {
      delimiters.reserve(num_leaves - 1);
    }

    LeafNode<KeyT>* prev_leaf = nullptr;
    for (size_t i = 0; i < num_leaves; ++i) {
      LeafNode<KeyT>* leaf = tree.leaf_pool_.Allocate();
      const size_t start_idx = i * keys_per_leaf;
      const size_t count = std::min(keys_per_leaf, num_keys - start_idx);
      std::copy_n(sorted_keys + start_idx, count, leaf->keys);
      leaf->num_keys = static_cast<uint16_t>(count);

      if (prev_leaf != nullptr) {
        prev_leaf->next = leaf;
        leaf->prev = prev_leaf;
        // Separator key for parent navigation
        delimiters.push_back(leaf->keys[0]);
      } else {
        tree.first_leaf_ = leaf;
      }

      current_level_ptrs.push_back(leaf);
      prev_leaf = leaf;
    }
    tree.last_leaf_ = prev_leaf;

    // If only 1 leaf, root is the leaf (height = 0)
    if (num_leaves == 1) {
      tree.root_ = tree.first_leaf_;
      tree.tree_height_ = 0;
      return tree;
    }

    // Step 2: Build internal levels bottom-up in pool
    const size_t max_children = Traits::kMaxChildren;
    uint16_t current_level = 1;

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
        InternalNode<KeyT>* internal = tree.internal_pool_.Allocate();
        internal->level = current_level;

        const size_t child_start = i * max_children;
        const size_t child_count =
            std::min(max_children, num_children - child_start);

        for (size_t c = 0; c < child_count; ++c) {
          internal->children[c] = current_level_ptrs[child_start + c];
        }

        // Fill keys from delimiters
        const size_t key_count = child_count - 1;
        internal->num_keys = static_cast<uint16_t>(key_count);
        for (size_t k = 0; k < key_count; ++k) {
          internal->keys[k] = delimiters[child_start + k];
        }

        if (i > 0) {
          // Promote boundary key to next level
          next_delimiters.push_back(delimiters[child_start - 1]);
        }

        next_level_ptrs.push_back(internal);
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
    const auto* curr = first_leaf_;
    while (curr != nullptr && curr->num_keys == 0) {
      curr = curr->next;
    }
    if (curr == nullptr) return end();
    return const_iterator(curr, 0, last_leaf_);
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

  // ---------------------------------------------------------------------------
  // Vectorized Pipelined Batch Query APIs
  // ---------------------------------------------------------------------------

  // Batch point lookup: fills out_found[i] with true if queries[i] is present.
  void ContainsBatch(const KeyT* HWY_RESTRICT queries, size_t num_queries,
                     bool* HWY_RESTRICT out_found) const {
    if (num_elements_ == 0) {
      std::fill_n(out_found, num_queries, false);
      return;
    }
    static constexpr size_t kBatch = 8;
    size_t i = 0;
    const LeafNode<KeyT>* leaves[kBatch];

    for (; i + kBatch <= num_queries; i += kBatch) {
      NavigateBatchToLeaves(queries + i, leaves);
      for (size_t k = 0; k < kBatch; ++k) {
        const size_t slot = FindLeafSlot(leaves[k], queries[i + k]);
        out_found[i + k] = (slot < leaves[k]->num_keys &&
                            leaves[k]->keys[slot] == queries[i + k]);
      }
    }

    for (; i < num_queries; ++i) {
      out_found[i] = Contains(queries[i]);
    }
  }

  // Batch find: fills out_ptrs[i] with pointer to key in tree or nullptr.
  void FindBatch(const KeyT* HWY_RESTRICT queries, size_t num_queries,
                 const KeyT** HWY_RESTRICT out_ptrs) const {
    if (num_elements_ == 0) {
      std::fill_n(out_ptrs, num_queries, nullptr);
      return;
    }
    static constexpr size_t kBatch = 8;
    size_t i = 0;
    const LeafNode<KeyT>* leaves[kBatch];

    for (; i + kBatch <= num_queries; i += kBatch) {
      NavigateBatchToLeaves(queries + i, leaves);
      for (size_t k = 0; k < kBatch; ++k) {
        const size_t slot = FindLeafSlot(leaves[k], queries[i + k]);
        const bool found = (slot < leaves[k]->num_keys &&
                            leaves[k]->keys[slot] == queries[i + k]);
        out_ptrs[i + k] = found ? &leaves[k]->keys[slot] : nullptr;
      }
    }

    for (; i < num_queries; ++i) {
      auto it = find(queries[i]);
      out_ptrs[i] = (it != end()) ? &(*it) : nullptr;
    }
  }

  // Batch lower_bound pointers: fills out_ptrs[i] with pointer to first key >=
  // queries[i] (or nullptr).
  void LowerBoundBatch(const KeyT* HWY_RESTRICT queries, size_t num_queries,
                       const KeyT** HWY_RESTRICT out_ptrs) const {
    if (num_elements_ == 0) {
      std::fill_n(out_ptrs, num_queries, nullptr);
      return;
    }
    static constexpr size_t kBatch = 8;
    size_t i = 0;
    const LeafNode<KeyT>* leaves[kBatch];

    for (; i + kBatch <= num_queries; i += kBatch) {
      NavigateBatchToLeaves(queries + i, leaves);
      for (size_t k = 0; k < kBatch; ++k) {
        const size_t slot = FindLeafSlot(leaves[k], queries[i + k]);
        if (slot < leaves[k]->num_keys) {
          out_ptrs[i + k] = &leaves[k]->keys[slot];
        } else if (leaves[k]->next != nullptr &&
                   leaves[k]->next->num_keys > 0) {
          out_ptrs[i + k] = &leaves[k]->next->keys[0];
        } else {
          out_ptrs[i + k] = nullptr;
        }
      }
    }

    for (; i < num_queries; ++i) {
      out_ptrs[i] = LowerBound(queries[i]);
    }
  }

  // Batch lower_bound iterators: fills out_iters[i] with const_iterator for
  // each query.
  void LowerBoundBatch(const KeyT* HWY_RESTRICT queries, size_t num_queries,
                       const_iterator* HWY_RESTRICT out_iters) const {
    if (num_elements_ == 0) {
      std::fill_n(out_iters, num_queries, end());
      return;
    }
    static constexpr size_t kBatch = 8;
    size_t i = 0;
    const LeafNode<KeyT>* leaves[kBatch];

    for (; i + kBatch <= num_queries; i += kBatch) {
      NavigateBatchToLeaves(queries + i, leaves);
      for (size_t k = 0; k < kBatch; ++k) {
        const size_t slot = FindLeafSlot(leaves[k], queries[i + k]);
        const bool inside = (slot < leaves[k]->num_keys);
        const LeafNode<KeyT>* res_leaf = inside ? leaves[k] : leaves[k]->next;
        const size_t res_slot = inside ? slot : 0;
        out_iters[i + k] = const_iterator(res_leaf, res_slot, last_leaf_);
      }
    }

    for (; i < num_queries; ++i) {
      out_iters[i] = lower_bound(queries[i]);
    }
  }

  size_t size() const { return num_elements_; }
  bool empty() const { return num_elements_ == 0; }
  uint16_t height() const { return tree_height_; }

  // Memory footprint in bytes
  size_t AllocatedBytes() const {
    return leaf_pool_.AllocatedBytes() + internal_pool_.AllocatedBytes();
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

  HWY_INLINE void NavigateBatchToLeaves(const KeyT* HWY_RESTRICT queries,
                                        const LeafNode<KeyT>** HWY_RESTRICT
                                            out_leaves) const {
    static constexpr size_t kBatch = 8;
    if (tree_height_ == 0) {
      const auto* leaf = static_cast<const LeafNode<KeyT>*>(root_);
      for (size_t k = 0; k < kBatch; ++k) {
        out_leaves[k] = leaf;
      }
      return;
    }

    const void* curr[kBatch];
    for (size_t k = 0; k < kBatch; ++k) {
      curr[k] = root_;
    }

    for (uint16_t h = tree_height_; h > 1; --h) {
      for (size_t k = 0; k < kBatch; ++k) {
        const auto* internal = static_cast<const InternalNode<KeyT>*>(curr[k]);
        size_t child_idx = FindChild(internal, queries[k]);
        const void* next_node = internal->children[child_idx];
        curr[k] = next_node;
        hwy::Prefetch(next_node);
      }
    }

    for (size_t k = 0; k < kBatch; ++k) {
      const auto* parent = static_cast<const InternalNode<KeyT>*>(curr[k]);
      size_t child_idx = FindChild(parent, queries[k]);
      const auto* leaf =
          static_cast<const LeafNode<KeyT>*>(parent->children[child_idx]);
      out_leaves[k] = leaf;
      hwy::Prefetch(leaf);
    }
  }

 public:
  // Dynamic Insertion
  std::pair<const_iterator, bool> insert(KeyT key) {
    if (root_ == nullptr) {
      first_leaf_ = last_leaf_ = leaf_pool_.Allocate();
      first_leaf_->keys[0] = key;
      first_leaf_->num_keys = 1;
      root_ = first_leaf_;
      tree_height_ = 0;
      num_elements_ = 1;
      return {const_iterator(first_leaf_, 0, last_leaf_), true};
    }

    if (tree_height_ == 0) {
      auto* leaf = static_cast<LeafNode<KeyT>*>(root_);
      size_t slot = FindLeafSlot(leaf, key);
      if (slot < leaf->num_keys && leaf->keys[slot] == key) {
        return {const_iterator(leaf, slot, last_leaf_), false};
      }
      if (leaf->num_keys < LeafNode<KeyT>::kCapacity) {
        for (size_t i = leaf->num_keys; i > slot; --i) {
          leaf->keys[i] = leaf->keys[i - 1];
        }
        leaf->keys[slot] = key;
        leaf->num_keys++;
        num_elements_++;
        return {const_iterator(leaf, slot, last_leaf_), true};
      }

      // Root leaf split
      LeafNode<KeyT>* new_leaf = leaf_pool_.Allocate();
      constexpr size_t kSplit = LeafNode<KeyT>::kCapacity / 2;
      const size_t right_count = leaf->num_keys - kSplit;
      std::copy_n(leaf->keys + kSplit, right_count, new_leaf->keys);
      new_leaf->num_keys = static_cast<uint16_t>(right_count);
      leaf->num_keys = static_cast<uint16_t>(kSplit);
      FillSentinel(leaf->keys + kSplit, LeafNode<KeyT>::kCapacity - kSplit);

      new_leaf->next = leaf->next;
      new_leaf->prev = leaf;
      if (leaf->next != nullptr) leaf->next->prev = new_leaf;
      leaf->next = new_leaf;
      last_leaf_ = new_leaf;

      if (key >= new_leaf->keys[0]) {
        size_t s = FindLeafSlot(new_leaf, key);
        for (size_t i = new_leaf->num_keys; i > s; --i) {
          new_leaf->keys[i] = new_leaf->keys[i - 1];
        }
        new_leaf->keys[s] = key;
        new_leaf->num_keys++;
      } else {
        size_t s = FindLeafSlot(leaf, key);
        for (size_t i = leaf->num_keys; i > s; --i) {
          leaf->keys[i] = leaf->keys[i - 1];
        }
        leaf->keys[s] = key;
        leaf->num_keys++;
      }

      InternalNode<KeyT>* new_root = internal_pool_.Allocate();
      new_root->level = 1;
      new_root->keys[0] = new_leaf->keys[0];
      new_root->children[0] = leaf;
      new_root->children[1] = new_leaf;
      new_root->num_keys = 1;
      root_ = new_root;
      tree_height_ = 1;
      num_elements_++;
      return {find(key), true};
    }

    // General case: tree_height_ >= 1
    InternalNode<KeyT>* path[32];
    size_t child_indices[32];
    void* curr = root_;
    for (uint16_t lvl = tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<InternalNode<KeyT>*>(curr);
      path[lvl] = internal;
      size_t c_idx = FindChild(internal, key);
      child_indices[lvl] = c_idx;
      curr = internal->children[c_idx];
    }

    auto* leaf = static_cast<LeafNode<KeyT>*>(curr);
    size_t slot = FindLeafSlot(leaf, key);
    if (slot < leaf->num_keys && leaf->keys[slot] == key) {
      return {const_iterator(leaf, slot, last_leaf_), false};
    }

    if (leaf->num_keys < LeafNode<KeyT>::kCapacity) {
      for (size_t i = leaf->num_keys; i > slot; --i) {
        leaf->keys[i] = leaf->keys[i - 1];
      }
      leaf->keys[slot] = key;
      leaf->num_keys++;
      num_elements_++;
      return {const_iterator(leaf, slot, last_leaf_), true};
    }

    // Leaf is full -> Split leaf
    LeafNode<KeyT>* new_leaf = leaf_pool_.Allocate();
    constexpr size_t kSplit = LeafNode<KeyT>::kCapacity / 2;
    const size_t right_count = leaf->num_keys - kSplit;
    std::copy_n(leaf->keys + kSplit, right_count, new_leaf->keys);
    new_leaf->num_keys = static_cast<uint16_t>(right_count);
    leaf->num_keys = static_cast<uint16_t>(kSplit);
    FillSentinel(leaf->keys + kSplit, LeafNode<KeyT>::kCapacity - kSplit);

    new_leaf->next = leaf->next;
    new_leaf->prev = leaf;
    if (leaf->next != nullptr) {
      leaf->next->prev = new_leaf;
    } else {
      last_leaf_ = new_leaf;
    }
    leaf->next = new_leaf;

    if (key >= new_leaf->keys[0]) {
      size_t s = FindLeafSlot(new_leaf, key);
      for (size_t i = new_leaf->num_keys; i > s; --i) {
        new_leaf->keys[i] = new_leaf->keys[i - 1];
      }
      new_leaf->keys[s] = key;
      new_leaf->num_keys++;
    } else {
      size_t s = FindLeafSlot(leaf, key);
      for (size_t i = leaf->num_keys; i > s; --i) {
        leaf->keys[i] = leaf->keys[i - 1];
      }
      leaf->keys[s] = key;
      leaf->num_keys++;
    }
    num_elements_++;

    KeyT promo_key = new_leaf->keys[0];
    void* promo_child = new_leaf;

    // Propagate splits up internal levels
    for (uint16_t lvl = 1; lvl <= tree_height_; ++lvl) {
      auto* parent = path[lvl];
      size_t c_idx = child_indices[lvl];

      if (parent->num_keys < Traits::kKeysPerNode) {
        for (size_t i = parent->num_keys; i > c_idx; --i) {
          parent->keys[i] = parent->keys[i - 1];
          parent->children[i + 1] = parent->children[i];
        }
        parent->keys[c_idx] = promo_key;
        parent->children[c_idx + 1] = promo_child;
        parent->num_keys++;
        return {find(key), true};
      }

      // Internal node split
      InternalNode<KeyT>* new_internal = internal_pool_.Allocate();
      new_internal->level = parent->level;

      constexpr size_t kTotalK = Traits::kKeysPerNode + 1;
      KeyT temp_keys[kTotalK];
      void* temp_children[kTotalK + 1];

      for (size_t i = 0; i < c_idx; ++i) {
        temp_keys[i] = parent->keys[i];
        temp_children[i] = parent->children[i];
      }
      temp_children[c_idx] = parent->children[c_idx];
      temp_keys[c_idx] = promo_key;
      temp_children[c_idx + 1] = promo_child;
      for (size_t i = c_idx; i < parent->num_keys; ++i) {
        temp_keys[i + 1] = parent->keys[i];
        temp_children[i + 2] = parent->children[i + 1];
      }

      constexpr size_t kMid = kTotalK / 2;
      promo_key = temp_keys[kMid];
      promo_child = new_internal;

      std::copy_n(temp_keys, kMid, parent->keys);
      std::copy_n(temp_children, kMid + 1, parent->children);
      parent->num_keys = static_cast<uint16_t>(kMid);
      FillSentinel(parent->keys + kMid, Traits::kKeysPerNode - kMid);

      const size_t right_k = kTotalK - kMid - 1;
      std::copy_n(temp_keys + kMid + 1, right_k, new_internal->keys);
      std::copy_n(temp_children + kMid + 1, right_k + 1,
                  new_internal->children);
      new_internal->num_keys = static_cast<uint16_t>(right_k);
    }

    // Root split
    InternalNode<KeyT>* new_root = internal_pool_.Allocate();
    new_root->level = tree_height_ + 1;
    new_root->keys[0] = promo_key;
    new_root->children[0] = root_;
    new_root->children[1] = promo_child;
    new_root->num_keys = 1;
    root_ = new_root;
    tree_height_++;

    return {find(key), true};
  }

  // Dynamic Deletion
  size_t erase(KeyT key) {
    if (root_ == nullptr || num_elements_ == 0) return 0;
    if (tree_height_ == 0) {
      auto* leaf = static_cast<LeafNode<KeyT>*>(root_);
      size_t slot = FindLeafSlot(leaf, key);
      if (slot >= leaf->num_keys || leaf->keys[slot] != key) return 0;
      for (size_t i = slot; i + 1 < leaf->num_keys; ++i) {
        leaf->keys[i] = leaf->keys[i + 1];
      }
      leaf->keys[leaf->num_keys - 1] = std::numeric_limits<KeyT>::max();
      leaf->num_keys--;
      num_elements_--;
      return 1;
    }

    void* curr = root_;
    for (uint16_t lvl = tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<InternalNode<KeyT>*>(curr);
      size_t c_idx = FindChild(internal, key);
      curr = internal->children[c_idx];
    }
    auto* leaf = static_cast<LeafNode<KeyT>*>(curr);
    size_t slot = FindLeafSlot(leaf, key);
    if (slot >= leaf->num_keys || leaf->keys[slot] != key) return 0;

    for (size_t i = slot; i + 1 < leaf->num_keys; ++i) {
      leaf->keys[i] = leaf->keys[i + 1];
    }
    leaf->keys[leaf->num_keys - 1] = std::numeric_limits<KeyT>::max();
    leaf->num_keys--;
    num_elements_--;
    if (leaf->num_keys == 0) {
      if (leaf->prev != nullptr) leaf->prev->next = leaf->next;
      if (leaf->next != nullptr) leaf->next->prev = leaf->prev;
      if (leaf == first_leaf_) first_leaf_ = leaf->next;
      if (leaf == last_leaf_) last_leaf_ = leaf->prev;
      leaf_pool_.Deallocate(leaf);
    }
    return 1;
  }

 private:
  void* root_ = nullptr;
  LeafNode<KeyT>* first_leaf_ = nullptr;
  LeafNode<KeyT>* last_leaf_ = nullptr;
  uint16_t tree_height_ = 0;
  size_t num_elements_ = 0;

  NodePool<LeafNode<KeyT>> leaf_pool_;
  NodePool<InternalNode<KeyT>> internal_pool_;
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
      while (leaf_ != nullptr && slot_ >= leaf_->num_keys) {
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
        const auto* curr = last_leaf_;
        while (curr != nullptr && curr->num_keys == 0) {
          curr = curr->prev;
        }
        if (curr != nullptr && curr->num_keys > 0) {
          leaf_ = curr;
          slot_ = curr->num_keys - 1;
        }
        return *this;
      }
      if (slot_ == 0) {
        const auto* curr = leaf_->prev;
        while (curr != nullptr && curr->num_keys == 0) {
          curr = curr->prev;
        }
        if (curr != nullptr) {
          leaf_ = curr;
          slot_ = curr->num_keys > 0 ? curr->num_keys - 1 : 0;
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
        leaf_pool_(std::move(other.leaf_pool_)),
        internal_pool_(std::move(other.internal_pool_)) {
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
      leaf_pool_ = std::move(other.leaf_pool_);
      internal_pool_ = std::move(other.internal_pool_);

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
                        const ValueT* HWY_RESTRICT values, size_t num_keys,
                        float fill_ratio = 1.0f) {
    BTreeMap map;
    map.num_elements_ = num_keys;
    if (num_keys == 0) {
      map.first_leaf_ = map.leaf_pool_.Allocate();
      map.last_leaf_ = map.first_leaf_;
      map.root_ = map.first_leaf_;
      map.tree_height_ = 0;
      return map;
    }

    // Step 1: Build leaf level in pool
    const size_t keys_per_leaf = std::clamp<size_t>(
        static_cast<size_t>(Traits::kKeysPerNode * fill_ratio), 2,
        Traits::kKeysPerNode);
    const size_t num_leaves = (num_keys + keys_per_leaf - 1) / keys_per_leaf;

    std::vector<void*> current_level_ptrs;
    current_level_ptrs.reserve(num_leaves);
    std::vector<KeyT> delimiters;
    if (num_leaves > 1) {
      delimiters.reserve(num_leaves - 1);
    }

    MapLeafNode<KeyT, ValueT>* prev_leaf = nullptr;
    for (size_t i = 0; i < num_leaves; ++i) {
      MapLeafNode<KeyT, ValueT>* leaf = map.leaf_pool_.Allocate();
      const size_t start_idx = i * keys_per_leaf;
      const size_t count = std::min(keys_per_leaf, num_keys - start_idx);
      std::copy_n(sorted_keys + start_idx, count, leaf->keys);
      if (values != nullptr) {
        std::copy_n(values + start_idx, count, leaf->values);
      }
      leaf->num_keys = static_cast<uint16_t>(count);

      if (prev_leaf != nullptr) {
        prev_leaf->next = leaf;
        leaf->prev = prev_leaf;
        delimiters.push_back(leaf->keys[0]);
      } else {
        map.first_leaf_ = leaf;
      }

      current_level_ptrs.push_back(leaf);
      prev_leaf = leaf;
    }
    map.last_leaf_ = prev_leaf;

    // If only 1 leaf, root is the leaf (height = 0)
    if (num_leaves == 1) {
      map.root_ = map.first_leaf_;
      map.tree_height_ = 0;
      return map;
    }

    // Step 2: Build internal levels bottom-up in pool
    const size_t max_children = Traits::kMaxChildren;
    uint16_t current_level = 1;

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
        InternalNode<KeyT>* internal = map.internal_pool_.Allocate();
        internal->level = current_level;

        const size_t child_start = i * max_children;
        const size_t child_count =
            std::min(max_children, num_children - child_start);

        for (size_t c = 0; c < child_count; ++c) {
          internal->children[c] = current_level_ptrs[child_start + c];
        }

        const size_t key_count = child_count - 1;
        internal->num_keys = static_cast<uint16_t>(key_count);
        for (size_t k = 0; k < key_count; ++k) {
          internal->keys[k] = delimiters[child_start + k];
        }

        if (i > 0) {
          next_delimiters.push_back(delimiters[child_start - 1]);
        }

        next_level_ptrs.push_back(internal);
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
                        size_t num_keys, float fill_ratio = 1.0f) {
    std::vector<KeyT> keys;
    std::vector<ValueT> vals;
    keys.reserve(num_keys);
    vals.reserve(num_keys);
    for (size_t i = 0; i < num_keys; ++i) {
      keys.push_back(kv_pairs[i].first);
      vals.push_back(kv_pairs[i].second);
    }
    return Build(keys.data(), vals.data(), num_keys, fill_ratio);
  }

  // Iterators
  const_iterator begin() const {
    if (num_elements_ == 0 || first_leaf_ == nullptr) return end();
    const auto* curr = first_leaf_;
    while (curr != nullptr && curr->num_keys == 0) {
      curr = curr->next;
    }
    if (curr == nullptr) return end();
    return const_iterator(curr, 0, last_leaf_);
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

  // ---------------------------------------------------------------------------
  // Vectorized Pipelined Batch Query APIs
  // ---------------------------------------------------------------------------

  // Batch point lookup: fills out_found[i] with true if queries[i] is present.
  void ContainsBatch(const KeyT* HWY_RESTRICT queries, size_t num_queries,
                     bool* HWY_RESTRICT out_found) const {
    if (num_elements_ == 0) {
      std::fill_n(out_found, num_queries, false);
      return;
    }
    static constexpr size_t kBatch = 8;
    size_t i = 0;
    const MapLeafNode<KeyT, ValueT>* leaves[kBatch];

    for (; i + kBatch <= num_queries; i += kBatch) {
      NavigateBatchToLeaves(queries + i, leaves);
      for (size_t k = 0; k < kBatch; ++k) {
        const size_t slot = FindLeafSlot(leaves[k], queries[i + k]);
        out_found[i + k] = (slot < leaves[k]->num_keys &&
                            leaves[k]->keys[slot] == queries[i + k]);
      }
    }

    for (; i < num_queries; ++i) {
      out_found[i] = Contains(queries[i]);
    }
  }

  // Batch value lookup: fills out_vals[i] with pointer to ValueT (or nullptr).
  void FindValueBatch(const KeyT* HWY_RESTRICT queries, size_t num_queries,
                      const ValueT** HWY_RESTRICT out_vals) const {
    if (num_elements_ == 0) {
      std::fill_n(out_vals, num_queries, nullptr);
      return;
    }
    static constexpr size_t kBatch = 8;
    size_t i = 0;
    const MapLeafNode<KeyT, ValueT>* leaves[kBatch];

    for (; i + kBatch <= num_queries; i += kBatch) {
      NavigateBatchToLeaves(queries + i, leaves);
      for (size_t k = 0; k < kBatch; ++k) {
        const size_t slot = FindLeafSlot(leaves[k], queries[i + k]);
        const bool found = (slot < leaves[k]->num_keys &&
                            leaves[k]->keys[slot] == queries[i + k]);
        out_vals[i + k] = found ? &leaves[k]->values[slot] : nullptr;
      }
    }

    for (; i < num_queries; ++i) {
      out_vals[i] = FindValue(queries[i]);
    }
  }

  // Batch lower_bound iterators: fills out_iters[i] with const_iterator.
  void LowerBoundBatch(const KeyT* HWY_RESTRICT queries, size_t num_queries,
                       const_iterator* HWY_RESTRICT out_iters) const {
    if (num_elements_ == 0) {
      std::fill_n(out_iters, num_queries, end());
      return;
    }
    static constexpr size_t kBatch = 8;
    size_t i = 0;
    const MapLeafNode<KeyT, ValueT>* leaves[kBatch];

    for (; i + kBatch <= num_queries; i += kBatch) {
      NavigateBatchToLeaves(queries + i, leaves);
      for (size_t k = 0; k < kBatch; ++k) {
        const size_t slot = FindLeafSlot(leaves[k], queries[i + k]);
        const bool inside = (slot < leaves[k]->num_keys);
        const auto* res_leaf = inside ? leaves[k] : leaves[k]->next;
        const size_t res_slot = inside ? slot : 0;
        out_iters[i + k] = const_iterator(res_leaf, res_slot, last_leaf_);
      }
    }

    for (; i < num_queries; ++i) {
      out_iters[i] = lower_bound(queries[i]);
    }
  }

  size_t size() const { return num_elements_; }
  bool empty() const { return num_elements_ == 0; }
  uint16_t height() const { return tree_height_; }

  // Memory footprint in bytes
  size_t AllocatedBytes() const {
    return leaf_pool_.AllocatedBytes() + internal_pool_.AllocatedBytes();
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

  HWY_INLINE void NavigateBatchToLeaves(
      const KeyT* HWY_RESTRICT queries,
      const MapLeafNode<KeyT, ValueT>** HWY_RESTRICT out_leaves) const {
    static constexpr size_t kBatch = 8;
    if (tree_height_ == 0) {
      const auto* leaf = static_cast<const MapLeafNode<KeyT, ValueT>*>(root_);
      for (size_t k = 0; k < kBatch; ++k) {
        out_leaves[k] = leaf;
      }
      return;
    }

    const void* curr[kBatch];
    for (size_t k = 0; k < kBatch; ++k) {
      curr[k] = root_;
    }

    for (uint16_t h = tree_height_; h > 1; --h) {
      for (size_t k = 0; k < kBatch; ++k) {
        const auto* internal = static_cast<const InternalNode<KeyT>*>(curr[k]);
        size_t child_idx = FindChild(internal, queries[k]);
        const void* next_node = internal->children[child_idx];
        curr[k] = next_node;
        hwy::Prefetch(next_node);
      }
    }

    for (size_t k = 0; k < kBatch; ++k) {
      const auto* parent = static_cast<const InternalNode<KeyT>*>(curr[k]);
      size_t child_idx = FindChild(parent, queries[k]);
      const auto* leaf = static_cast<const MapLeafNode<KeyT, ValueT>*>(
          parent->children[child_idx]);
      out_leaves[k] = leaf;
      hwy::Prefetch(leaf);
    }
  }

 public:
  // Dynamic Insertion
  std::pair<const_iterator, bool> insert(KeyT key, ValueT value) {
    if (root_ == nullptr) {
      first_leaf_ = last_leaf_ = leaf_pool_.Allocate();
      first_leaf_->keys[0] = key;
      first_leaf_->values[0] = value;
      first_leaf_->num_keys = 1;
      root_ = first_leaf_;
      tree_height_ = 0;
      num_elements_ = 1;
      return {const_iterator(first_leaf_, 0, last_leaf_), true};
    }

    if (tree_height_ == 0) {
      auto* leaf = static_cast<MapLeafNode<KeyT, ValueT>*>(root_);
      size_t slot = FindLeafSlot(leaf, key);
      if (slot < leaf->num_keys && leaf->keys[slot] == key) {
        return {const_iterator(leaf, slot, last_leaf_), false};
      }
      if (leaf->num_keys < MapLeafNode<KeyT, ValueT>::kCapacity) {
        for (size_t i = leaf->num_keys; i > slot; --i) {
          leaf->keys[i] = leaf->keys[i - 1];
          leaf->values[i] = leaf->values[i - 1];
        }
        leaf->keys[slot] = key;
        leaf->values[slot] = value;
        leaf->num_keys++;
        num_elements_++;
        return {const_iterator(leaf, slot, last_leaf_), true};
      }

      // Root leaf split
      MapLeafNode<KeyT, ValueT>* new_leaf = leaf_pool_.Allocate();
      constexpr size_t kSplit = MapLeafNode<KeyT, ValueT>::kCapacity / 2;
      const size_t right_count = leaf->num_keys - kSplit;
      std::copy_n(leaf->keys + kSplit, right_count, new_leaf->keys);
      std::copy_n(leaf->values + kSplit, right_count, new_leaf->values);
      new_leaf->num_keys = static_cast<uint16_t>(right_count);
      leaf->num_keys = static_cast<uint16_t>(kSplit);
      FillSentinel(leaf->keys + kSplit,
                   MapLeafNode<KeyT, ValueT>::kCapacity - kSplit);

      new_leaf->next = leaf->next;
      new_leaf->prev = leaf;
      if (leaf->next != nullptr) leaf->next->prev = new_leaf;
      leaf->next = new_leaf;
      last_leaf_ = new_leaf;

      if (key >= new_leaf->keys[0]) {
        size_t s = FindLeafSlot(new_leaf, key);
        for (size_t i = new_leaf->num_keys; i > s; --i) {
          new_leaf->keys[i] = new_leaf->keys[i - 1];
          new_leaf->values[i] = new_leaf->values[i - 1];
        }
        new_leaf->keys[s] = key;
        new_leaf->values[s] = value;
        new_leaf->num_keys++;
      } else {
        size_t s = FindLeafSlot(leaf, key);
        for (size_t i = leaf->num_keys; i > s; --i) {
          leaf->keys[i] = leaf->keys[i - 1];
          leaf->values[i] = leaf->values[i - 1];
        }
        leaf->keys[s] = key;
        leaf->values[s] = value;
        leaf->num_keys++;
      }

      InternalNode<KeyT>* new_root = internal_pool_.Allocate();
      new_root->level = 1;
      new_root->keys[0] = new_leaf->keys[0];
      new_root->children[0] = leaf;
      new_root->children[1] = new_leaf;
      new_root->num_keys = 1;
      root_ = new_root;
      tree_height_ = 1;
      num_elements_++;
      return {find(key), true};
    }

    // General case: tree_height_ >= 1
    InternalNode<KeyT>* path[32];
    size_t child_indices[32];
    void* curr = root_;
    for (uint16_t lvl = tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<InternalNode<KeyT>*>(curr);
      path[lvl] = internal;
      size_t c_idx = FindChild(internal, key);
      child_indices[lvl] = c_idx;
      curr = internal->children[c_idx];
    }

    auto* leaf = static_cast<MapLeafNode<KeyT, ValueT>*>(curr);
    size_t slot = FindLeafSlot(leaf, key);
    if (slot < leaf->num_keys && leaf->keys[slot] == key) {
      return {const_iterator(leaf, slot, last_leaf_), false};
    }

    if (leaf->num_keys < MapLeafNode<KeyT, ValueT>::kCapacity) {
      for (size_t i = leaf->num_keys; i > slot; --i) {
        leaf->keys[i] = leaf->keys[i - 1];
        leaf->values[i] = leaf->values[i - 1];
      }
      leaf->keys[slot] = key;
      leaf->values[slot] = value;
      leaf->num_keys++;
      num_elements_++;
      return {const_iterator(leaf, slot, last_leaf_), true};
    }

    // Leaf split
    MapLeafNode<KeyT, ValueT>* new_leaf = leaf_pool_.Allocate();
    constexpr size_t kSplit = MapLeafNode<KeyT, ValueT>::kCapacity / 2;
    const size_t right_count = leaf->num_keys - kSplit;
    std::copy_n(leaf->keys + kSplit, right_count, new_leaf->keys);
    std::copy_n(leaf->values + kSplit, right_count, new_leaf->values);
    new_leaf->num_keys = static_cast<uint16_t>(right_count);
    leaf->num_keys = static_cast<uint16_t>(kSplit);
    FillSentinel(leaf->keys + kSplit,
                 MapLeafNode<KeyT, ValueT>::kCapacity - kSplit);

    new_leaf->next = leaf->next;
    new_leaf->prev = leaf;
    if (leaf->next != nullptr) {
      leaf->next->prev = new_leaf;
    } else {
      last_leaf_ = new_leaf;
    }
    leaf->next = new_leaf;

    if (key >= new_leaf->keys[0]) {
      size_t s = FindLeafSlot(new_leaf, key);
      for (size_t i = new_leaf->num_keys; i > s; --i) {
        new_leaf->keys[i] = new_leaf->keys[i - 1];
        new_leaf->values[i] = new_leaf->values[i - 1];
      }
      new_leaf->keys[s] = key;
      new_leaf->values[s] = value;
      new_leaf->num_keys++;
    } else {
      size_t s = FindLeafSlot(leaf, key);
      for (size_t i = leaf->num_keys; i > s; --i) {
        leaf->keys[i] = leaf->keys[i - 1];
        leaf->values[i] = leaf->values[i - 1];
      }
      leaf->keys[s] = key;
      leaf->values[s] = value;
      leaf->num_keys++;
    }
    num_elements_++;

    KeyT promo_key = new_leaf->keys[0];
    void* promo_child = new_leaf;

    for (uint16_t lvl = 1; lvl <= tree_height_; ++lvl) {
      auto* parent = path[lvl];
      size_t c_idx = child_indices[lvl];

      if (parent->num_keys < Traits::kKeysPerNode) {
        for (size_t i = parent->num_keys; i > c_idx; --i) {
          parent->keys[i] = parent->keys[i - 1];
          parent->children[i + 1] = parent->children[i];
        }
        parent->keys[c_idx] = promo_key;
        parent->children[c_idx + 1] = promo_child;
        parent->num_keys++;
        return {find(key), true};
      }

      // Split internal node
      InternalNode<KeyT>* new_internal = internal_pool_.Allocate();
      new_internal->level = parent->level;

      constexpr size_t kTotalK = Traits::kKeysPerNode + 1;
      KeyT temp_keys[kTotalK];
      void* temp_children[kTotalK + 1];

      for (size_t i = 0; i < c_idx; ++i) {
        temp_keys[i] = parent->keys[i];
        temp_children[i] = parent->children[i];
      }
      temp_children[c_idx] = parent->children[c_idx];
      temp_keys[c_idx] = promo_key;
      temp_children[c_idx + 1] = promo_child;
      for (size_t i = c_idx; i < parent->num_keys; ++i) {
        temp_keys[i + 1] = parent->keys[i];
        temp_children[i + 2] = parent->children[i + 1];
      }

      constexpr size_t kMid = kTotalK / 2;
      promo_key = temp_keys[kMid];
      promo_child = new_internal;

      std::copy_n(temp_keys, kMid, parent->keys);
      std::copy_n(temp_children, kMid + 1, parent->children);
      parent->num_keys = static_cast<uint16_t>(kMid);
      FillSentinel(parent->keys + kMid, Traits::kKeysPerNode - kMid);

      const size_t right_k = kTotalK - kMid - 1;
      std::copy_n(temp_keys + kMid + 1, right_k, new_internal->keys);
      std::copy_n(temp_children + kMid + 1, right_k + 1,
                  new_internal->children);
      new_internal->num_keys = static_cast<uint16_t>(right_k);
    }

    // Root split
    InternalNode<KeyT>* new_root = internal_pool_.Allocate();
    new_root->level = tree_height_ + 1;
    new_root->keys[0] = promo_key;
    new_root->children[0] = root_;
    new_root->children[1] = promo_child;
    new_root->num_keys = 1;
    root_ = new_root;
    tree_height_++;

    return {find(key), true};
  }

  std::pair<const_iterator, bool> insert(const std::pair<KeyT, ValueT>& p) {
    return insert(p.first, p.second);
  }

  ValueT& operator[](KeyT key) {
    auto res = insert(key, ValueT{});
    auto* leaf = const_cast<MapLeafNode<KeyT, ValueT>*>(res.first.leaf());
    return leaf->values[res.first.slot()];
  }

  size_t erase(KeyT key) {
    if (root_ == nullptr || num_elements_ == 0) return 0;
    if (tree_height_ == 0) {
      auto* leaf = static_cast<MapLeafNode<KeyT, ValueT>*>(root_);
      size_t slot = FindLeafSlot(leaf, key);
      if (slot >= leaf->num_keys || leaf->keys[slot] != key) return 0;
      for (size_t i = slot; i + 1 < leaf->num_keys; ++i) {
        leaf->keys[i] = leaf->keys[i + 1];
        leaf->values[i] = leaf->values[i + 1];
      }
      leaf->keys[leaf->num_keys - 1] = std::numeric_limits<KeyT>::max();
      leaf->num_keys--;
      num_elements_--;
      return 1;
    }

    void* curr = root_;
    for (uint16_t lvl = tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<InternalNode<KeyT>*>(curr);
      size_t c_idx = FindChild(internal, key);
      curr = internal->children[c_idx];
    }
    auto* leaf = static_cast<MapLeafNode<KeyT, ValueT>*>(curr);
    size_t slot = FindLeafSlot(leaf, key);
    if (slot >= leaf->num_keys || leaf->keys[slot] != key) return 0;

    for (size_t i = slot; i + 1 < leaf->num_keys; ++i) {
      leaf->keys[i] = leaf->keys[i + 1];
      leaf->values[i] = leaf->values[i + 1];
    }
    leaf->keys[leaf->num_keys - 1] = std::numeric_limits<KeyT>::max();
    leaf->num_keys--;
    num_elements_--;
    if (leaf->num_keys == 0) {
      if (leaf->prev != nullptr) leaf->prev->next = leaf->next;
      if (leaf->next != nullptr) leaf->next->prev = leaf->prev;
      if (leaf == first_leaf_) first_leaf_ = leaf->next;
      if (leaf == last_leaf_) last_leaf_ = leaf->prev;
      leaf_pool_.Deallocate(leaf);
    }
    return 1;
  }

 private:
  void* root_ = nullptr;
  MapLeafNode<KeyT, ValueT>* first_leaf_ = nullptr;
  MapLeafNode<KeyT, ValueT>* last_leaf_ = nullptr;
  uint16_t tree_height_ = 0;
  size_t num_elements_ = 0;

  NodePool<MapLeafNode<KeyT, ValueT>> leaf_pool_;
  NodePool<InternalNode<KeyT>> internal_pool_;
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
