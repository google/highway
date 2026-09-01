// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Interface to Highway BTreeSet with dynamic dispatch.
//
// Methods in this header incur a small dynamic dispatch overhead per call.
// To amortize this overhead, callers should prefer using the batch query APIs
// (e.g. ContainsBatch, FindBatch, LowerBoundBatch). For maximum performance/
// direct inlining without dynamic dispatch overhead, users should prefer
// including "third_party/highway/hwy/contrib/btree/btree-inl.h" directly.

#ifndef HIGHWAY_HWY_CONTRIB_BTREE_BTREE_SET_H_
#define HIGHWAY_HWY_CONTRIB_BTREE_BTREE_SET_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <utility>

#include "hwy/base.h"
#include "hwy/contrib/btree/btree_nodes.h"

namespace hwy {

template <typename KeyT>
class BTreeSet {
 public:
  using key_type = KeyT;
  using value_type = KeyT;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using reference = KeyT&;
  using const_reference = const KeyT&;
  using pointer = KeyT*;
  using const_pointer = const KeyT*;
  using key_compare = std::less<KeyT>;
  using allocator_type = std::allocator<KeyT>;

  static constexpr bool kIsMap = false;

  BTreeSet() = default;
  explicit BTreeSet(const key_compare& comp,
                    const allocator_type& alloc = allocator_type()) {}
  explicit BTreeSet(const allocator_type& alloc) {}
  ~BTreeSet();

  BTreeSet(const BTreeSet& other);
  BTreeSet& operator=(const BTreeSet& other);
  BTreeSet(BTreeSet&& other) noexcept;
  BTreeSet& operator=(BTreeSet&& other) noexcept;
  BTreeSet& operator=(std::initializer_list<value_type> ilist) {
    clear();
    for (auto v : ilist) insert(v);
    return *this;
  }

  static BTreeSet Build(const KeyT* sorted_keys, size_t num_keys,
                        float fill_ratio = 1.0f);

  void clear();

  // ---------------------------------------------------------------------------
  // Bidirectional Iterators
  // ---------------------------------------------------------------------------
  struct SetConstRef {
    KeyT val;
    KeyT operator*() const { return val; }
    const KeyT* operator->() const { return &val; }
  };

  class const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = KeyT;
    using difference_type = std::ptrdiff_t;
    using reference = KeyT;
    using pointer = SetConstRef;

    const_iterator() = default;
    const_iterator(const LeafNode<KeyT>* leaf, size_t slot,
                   const LeafNode<KeyT>* last_leaf = nullptr)
        : leaf_(leaf), slot_(slot), last_leaf_(last_leaf) {}

    auto operator*() const { return GetLeafKey(leaf_, slot_); }
    auto operator->() const { return SetConstRef{GetLeafKey(leaf_, slot_)}; }

    const_iterator& operator++() {
      if (HWY_UNLIKELY(leaf_ == nullptr)) return *this;
      slot_++;
      if (HWY_UNLIKELY(slot_ >= leaf_->NumKeys())) {
        last_leaf_ = leaf_;
        leaf_ = leaf_->Next();
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
      if (HWY_UNLIKELY(leaf_ == nullptr)) {
        leaf_ = last_leaf_;
        slot_ = (leaf_ != nullptr && leaf_->NumKeys() > 0)
                    ? leaf_->NumKeys() - 1
                    : 0;
        return *this;
      }
      if (HWY_UNLIKELY(slot_ == 0)) {
        leaf_ = leaf_->Prev();
        slot_ = (leaf_ != nullptr && leaf_->NumKeys() > 0)
                    ? leaf_->NumKeys() - 1
                    : 0;
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

    const LeafNode<KeyT>* node() const { return leaf_; }
    size_t slot() const { return slot_; }

   protected:
    const LeafNode<KeyT>* leaf_ = nullptr;
    size_t slot_ = 0;
    const LeafNode<KeyT>* last_leaf_ = nullptr;
  };

  class iterator : public const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = KeyT;
    using difference_type = std::ptrdiff_t;
    using reference = KeyT;
    using pointer = SetConstRef;

    iterator() = default;
    iterator(LeafNode<KeyT>* leaf, size_t slot,
             LeafNode<KeyT>* last_leaf = nullptr)
        : const_iterator(leaf, slot, last_leaf) {}

    auto operator*() const { return GetLeafKey(this->leaf_, this->slot_); }
    auto operator->() const {
      return SetConstRef{GetLeafKey(this->leaf_, this->slot_)};
    }

    iterator& operator++() {
      if (HWY_UNLIKELY(this->leaf_ == nullptr)) return *this;
      this->slot_++;
      if (HWY_UNLIKELY(this->slot_ >= this->leaf_->NumKeys())) {
        this->last_leaf_ = this->leaf_;
        this->leaf_ = this->leaf_->Next();
        this->slot_ = 0;
      }
      return *this;
    }

    iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    iterator& operator--() {
      if (HWY_UNLIKELY(this->leaf_ == nullptr)) {
        this->leaf_ = this->last_leaf_;
        this->slot_ = (this->leaf_ != nullptr && this->leaf_->NumKeys() > 0)
                          ? this->leaf_->NumKeys() - 1
                          : 0;
        return *this;
      }
      if (HWY_UNLIKELY(this->slot_ == 0)) {
        this->leaf_ = this->leaf_->Prev();
        this->slot_ = (this->leaf_ != nullptr && this->leaf_->NumKeys() > 0)
                          ? this->leaf_->NumKeys() - 1
                          : 0;
      } else {
        --this->slot_;
      }
      return *this;
    }

    iterator operator--(int) {
      iterator tmp = *this;
      --(*this);
      return tmp;
    }

    operator const_iterator() const {
      return const_iterator(this->leaf_, this->slot_, this->last_leaf_);
    }

    LeafNode<KeyT>* node() const {
      return const_cast<LeafNode<KeyT>*>(this->leaf_);
    }
  };

  class const_reverse_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = KeyT;
    using difference_type = std::ptrdiff_t;
    using reference = KeyT;
    using pointer = SetConstRef;

    const_reverse_iterator() = default;
    explicit const_reverse_iterator(const_iterator it) : current_(it) {}

    auto operator*() const {
      auto tmp = current_;
      --tmp;
      return *tmp;
    }

    auto operator->() const {
      auto tmp = current_;
      --tmp;
      return tmp.operator->();
    }

    const_reverse_iterator& operator++() {
      --current_;
      return *this;
    }
    const_reverse_iterator operator++(int) {
      const_reverse_iterator tmp = *this;
      --current_;
      return tmp;
    }
    const_reverse_iterator& operator--() {
      ++current_;
      return *this;
    }
    const_reverse_iterator operator--(int) {
      const_reverse_iterator tmp = *this;
      ++current_;
      return tmp;
    }

    bool operator==(const const_reverse_iterator& other) const {
      return current_ == other.current_;
    }
    bool operator!=(const const_reverse_iterator& other) const {
      return current_ != other.current_;
    }

    const_iterator base() const { return current_; }

   private:
    const_iterator current_;
  };

  class reverse_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = KeyT;
    using difference_type = std::ptrdiff_t;
    using reference = KeyT;
    using pointer = SetConstRef;

    reverse_iterator() = default;
    explicit reverse_iterator(iterator it) : current_(it) {}

    auto operator*() const {
      auto tmp = current_;
      --tmp;
      return *tmp;
    }

    auto operator->() const {
      auto tmp = current_;
      --tmp;
      return tmp.operator->();
    }

    reverse_iterator& operator++() {
      --current_;
      return *this;
    }
    reverse_iterator operator++(int) {
      reverse_iterator tmp = *this;
      --current_;
      return tmp;
    }
    reverse_iterator& operator--() {
      ++current_;
      return *this;
    }
    reverse_iterator operator--(int) {
      reverse_iterator tmp = *this;
      ++current_;
      return tmp;
    }

    bool operator==(const reverse_iterator& other) const {
      return current_ == other.current_;
    }
    bool operator!=(const reverse_iterator& other) const {
      return current_ != other.current_;
    }

    operator const_reverse_iterator() const {
      return const_reverse_iterator(current_);
    }

    iterator base() const { return current_; }

   private:
    iterator current_;
  };

  iterator begin() {
    return iterator(state_.first_leaf_, 0, state_.last_leaf_);
  }
  const_iterator cbegin() const { return begin(); }
  const_iterator begin() const {
    return const_iterator(state_.first_leaf_, 0, state_.last_leaf_);
  }
  iterator end() { return iterator(nullptr, 0, state_.last_leaf_); }
  const_iterator end() const {
    return const_iterator(nullptr, 0, state_.last_leaf_);
  }
  const_iterator cend() const { return end(); }

  reverse_iterator rbegin() { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator crbegin() const {
    return const_reverse_iterator(end());
  }

  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }
  const_reverse_iterator crend() const {
    return const_reverse_iterator(begin());
  }

  // ---------------------------------------------------------------------------
  // Status & Accessors
  // ---------------------------------------------------------------------------
  size_t size() const { return state_.num_elements_; }
  size_t height() const { return state_.tree_height_; }
  bool empty() const { return state_.num_elements_ == 0; }

  struct value_compare {
    bool operator()(const KeyT& lhs, const KeyT& rhs) const {
      return lhs < rhs;
    }
  };

  value_compare value_comp() const { return value_compare(); }
  value_compare key_comp() const { return value_compare(); }
  allocator_type get_allocator() const { return allocator_type(); }

  void swap(BTreeSet& other) noexcept { std::swap(state_, other.state_); }

  // ---------------------------------------------------------------------------
  // Range Queries & Mutation
  // ---------------------------------------------------------------------------
  const_iterator lower_bound(KeyT key) const;
  const_iterator upper_bound(KeyT key) const;
  const_iterator find(KeyT key) const;

  iterator lower_bound(KeyT key) {
    auto it = static_cast<const BTreeSet*>(this)->lower_bound(key);
    return iterator(const_cast<LeafNode<KeyT>*>(it.node()), it.slot(),
                    state_.last_leaf_);
  }
  iterator upper_bound(KeyT key) {
    auto it = static_cast<const BTreeSet*>(this)->upper_bound(key);
    return iterator(const_cast<LeafNode<KeyT>*>(it.node()), it.slot(),
                    state_.last_leaf_);
  }
  iterator find(KeyT key) {
    auto it = static_cast<const BTreeSet*>(this)->find(key);
    return iterator(const_cast<LeafNode<KeyT>*>(it.node()), it.slot(),
                    state_.last_leaf_);
  }

  std::pair<iterator, bool> insert(KeyT key);
  size_t erase(KeyT key);

  const LeafNode<KeyT>* last_leaf() const { return state_.last_leaf_; }
  LeafNode<KeyT>* last_leaf() { return state_.last_leaf_; }

  // Batch Query (Amortizes dynamic dispatch overhead)
  bool contains(KeyT key) const { return Contains(key); }
  bool Contains(KeyT key) const;
  void ContainsBatch(const KeyT* keys, size_t count, bool* out) const;
  void FindBatch(const KeyT* keys, size_t count, const_iterator* out) const;
  void LowerBoundBatch(const KeyT* keys, size_t count,
                       const_iterator* out) const;

 private:
  BTreeState<KeyT> state_;
};

template <>
HWY_CONTRIB_DLLEXPORT BTreeSet<uint32_t>::~BTreeSet();

template <>
HWY_CONTRIB_DLLEXPORT BTreeSet<uint32_t>::BTreeSet(const BTreeSet& other);

template <>
HWY_CONTRIB_DLLEXPORT BTreeSet<uint32_t>& BTreeSet<uint32_t>::operator=(
    const BTreeSet& other);

template <>
HWY_CONTRIB_DLLEXPORT BTreeSet<uint32_t>::BTreeSet(BTreeSet&& other) noexcept;

template <>
HWY_CONTRIB_DLLEXPORT BTreeSet<uint32_t>& BTreeSet<uint32_t>::operator=(
    BTreeSet&& other) noexcept;

template <>
HWY_CONTRIB_DLLEXPORT void BTreeSet<uint32_t>::clear();

template <>
HWY_CONTRIB_DLLEXPORT BTreeSet<uint32_t> BTreeSet<uint32_t>::Build(
    const uint32_t* sorted_keys, size_t num_keys, float fill_ratio);

template <>
HWY_CONTRIB_DLLEXPORT BTreeSet<uint32_t>::const_iterator
BTreeSet<uint32_t>::lower_bound(uint32_t key) const;

template <>
HWY_CONTRIB_DLLEXPORT BTreeSet<uint32_t>::const_iterator
BTreeSet<uint32_t>::upper_bound(uint32_t key) const;

template <>
HWY_CONTRIB_DLLEXPORT BTreeSet<uint32_t>::const_iterator
BTreeSet<uint32_t>::find(uint32_t key) const;

template <>
HWY_CONTRIB_DLLEXPORT std::pair<BTreeSet<uint32_t>::iterator, bool>
BTreeSet<uint32_t>::insert(uint32_t key);

template <>
HWY_CONTRIB_DLLEXPORT size_t BTreeSet<uint32_t>::erase(uint32_t key);

template <>
HWY_CONTRIB_DLLEXPORT bool BTreeSet<uint32_t>::Contains(uint32_t key) const;

template <>
HWY_CONTRIB_DLLEXPORT void BTreeSet<uint32_t>::ContainsBatch(
    const uint32_t* keys, size_t count, bool* out) const;

template <>
HWY_CONTRIB_DLLEXPORT void BTreeSet<uint32_t>::FindBatch(
    const uint32_t* keys, size_t count, const_iterator* out) const;

template <>
HWY_CONTRIB_DLLEXPORT void BTreeSet<uint32_t>::LowerBoundBatch(
    const uint32_t* keys, size_t count, const_iterator* out) const;

extern template class HWY_CONTRIB_DLLEXPORT BTreeSet<uint32_t>;
static_assert(sizeof(BTreeSet<uint32_t>) == sizeof(BTreeState<uint32_t>),
              "BTreeSet must have the exact same size as BTreeState");

template <typename KeyT>
void swap(BTreeSet<KeyT>& a, BTreeSet<KeyT>& b) noexcept {
  a.swap(b);
}

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_BTREE_BTREE_SET_H_
