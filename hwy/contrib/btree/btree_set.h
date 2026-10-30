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

// Keys are type-erased via KeyCodec<KeyT>::StorageKey so that any supported
// 32/64-bit integer, pointer (T*), or 32/64-bit enum key maps to one of the
// two pre-compiled SIMD dispatch instantiations (uint32_t, uint64_t).
namespace detail {

template <typename StorageKeyT>
struct SetDispatch;

#define HWY_BTREE_SET_DISPATCH_DECLARE(StorageKeyT)                            \
  template <>                                                                  \
  struct HWY_CONTRIB_DLLEXPORT SetDispatch<StorageKeyT> {                      \
    using Leaf = LeafNode<StorageKeyT>;                                        \
    using State = BTreeState<StorageKeyT, Leaf>;                               \
                                                                               \
    static void Clear(State* state);                                           \
    static void CopyConstruct(State* dst, const State* src);                   \
    static void CopyAssign(State* dst, const State* src);                      \
    static void MoveConstruct(State* dst, State* src);                         \
    static void MoveAssign(State* dst, State* src);                            \
    static void Build(const StorageKeyT* keys, size_t count, float fill,       \
                      StorageKeyT sign_mask, State* out_state);                \
    static bool Contains(const State* state, StorageKeyT key);                 \
    static std::pair<const Leaf*, size_t> LowerBound(const State* state,       \
                                                     StorageKeyT key);         \
    static std::pair<const Leaf*, size_t> UpperBound(const State* state,       \
                                                     StorageKeyT key);         \
    static std::pair<const Leaf*, size_t> Find(const State* state,             \
                                               StorageKeyT key);               \
    static std::pair<std::pair<Leaf*, size_t>, bool> Insert(State* state,      \
                                                            StorageKeyT key);  \
    static size_t Erase(State* state, StorageKeyT key);                        \
    static std::pair<Leaf*, size_t> EraseIter(State* state, const Leaf* leaf,  \
                                              size_t slot);                    \
    static void ContainsBatch(const State* state, const StorageKeyT* keys,     \
                              size_t count, StorageKeyT sign_mask, bool* out); \
    static void FindBatch(const State* state, const StorageKeyT* keys,         \
                          size_t count, StorageKeyT sign_mask, void* out);     \
    static void LowerBoundBatch(const State* state, const StorageKeyT* keys,   \
                                size_t count, StorageKeyT sign_mask,           \
                                void* out);                                    \
  }

HWY_BTREE_SET_DISPATCH_DECLARE(uint32_t);
HWY_BTREE_SET_DISPATCH_DECLARE(uint64_t);

#undef HWY_BTREE_SET_DISPATCH_DECLARE

}  // namespace detail

// SIMD-accelerated ordered set of 32/64-bit integer, pointer, or enum keys with
// an API modelled on std::set / absl::btree_set. Keys are stored
// delta-compressed in 512-byte leaves; SIMD kernels are selected at runtime via
// dynamic dispatch. Keys are always ordered by their numeric, address, or
// underlying enum value; custom operator< overloads and key_compare functors
// are ignored.
//
// Iterator invalidation: as with absl::btree_set, if an insertion or erasure
// occurs (insert, emplace, erase, clear, Build, swap, assignment), ALL
// outstanding iterators, pointers and references may be invalidated,
// including end(). To continue iterating after a mutation, use the iterator
// returned by insert() or erase(iterator):
//   for (auto it = s.begin(); it != s.end();) {
//     if (ShouldErase(*it)) it = s.erase(it); else ++it;
//   }
template <typename KeyT>
class BTreeSet {
 public:
  using key_type = KeyT;
  using value_type = KeyT;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using key_compare = std::less<KeyT>;
  using allocator_type = std::allocator<KeyT>;

  using StorageKeyT = typename KeyCodec<KeyT>::StorageKey;
  using LeafT = LeafNode<StorageKeyT>;
  static constexpr StorageKeyT kSignMask = KeyCodec<KeyT>::kSignMask;

  static constexpr bool kIsMap = false;

  BTreeSet() = default;
  explicit BTreeSet(const key_compare& comp,
                    const allocator_type& alloc = allocator_type()) {}
  explicit BTreeSet(const allocator_type& alloc) {}
  ~BTreeSet() { clear(); }

  BTreeSet(const BTreeSet& other) {
    Dispatch::CopyConstruct(&state_, &other.state_);
  }
  BTreeSet& operator=(const BTreeSet& other) {
    if (this != &other) {
      Dispatch::CopyAssign(&state_, &other.state_);
    }
    return *this;
  }
  BTreeSet(BTreeSet&& other) noexcept {
    Dispatch::MoveConstruct(&state_, &other.state_);
  }
  BTreeSet& operator=(BTreeSet&& other) noexcept {
    if (this != &other) {
      Dispatch::MoveAssign(&state_, &other.state_);
    }
    return *this;
  }
  BTreeSet& operator=(std::initializer_list<value_type> ilist) {
    clear();
    for (auto v : ilist) insert(v);
    return *this;
  }

  // Bulk-builds a set from `num_keys` keys in O(N) time if strictly ascending,
  // or O(N log N) time via VQSort if unsorted or duplicate keys are present.
  // Reinterpreting `keys` as `const StorageKeyT*` is safe because it crosses an
  // opaque function-pointer dispatch table into `btree-inl.h`, which reads
  // elements via `HWY_RCAST_ALIGNED` / `BitCastScalar` and applies `kSignMask`.
  // Assigning the result to an existing set invalidates all of its iterators.
  static BTreeSet Build(const KeyT* keys, size_t num_keys,
                        float fill_ratio = 1.0f) {
    BTreeSet tree;
    Dispatch::Build(reinterpret_cast<const StorageKeyT*>(keys), num_keys,
                    fill_ratio, kSignMask, &tree.state_);
    return tree;
  }

  // Removes all elements. Invalidates all iterators, pointers and references.
  void clear() { Dispatch::Clear(&state_); }

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
    const_iterator(const LeafT* leaf, size_t slot,
                   const LeafT* last_leaf = nullptr)
        : leaf_(leaf), slot_(slot), last_leaf_(last_leaf) {}

    auto operator*() const {
      return KeyCodec<KeyT>::FromStorage(GetLeafKey(leaf_, slot_));
    }
    auto operator->() const { return SetConstRef{operator*()}; }

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
        slot_ = (leaf_ != nullptr) ? leaf_->NumKeys() - 1 : 0;
        return *this;
      }
      if (HWY_UNLIKELY(slot_ == 0)) {
        leaf_ = leaf_->Prev();
        slot_ = (leaf_ != nullptr) ? leaf_->NumKeys() - 1 : 0;
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

    const LeafT* node() const { return leaf_; }
    const LeafT* leaf() const { return leaf_; }
    size_t slot() const { return slot_; }

   protected:
    const LeafT* leaf_ = nullptr;
    size_t slot_ = 0;
    const LeafT* last_leaf_ = nullptr;
  };

  class iterator : public const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = KeyT;
    using difference_type = std::ptrdiff_t;
    using reference = KeyT;
    using pointer = SetConstRef;

    iterator() = default;
    iterator(LeafT* leaf, size_t slot, LeafT* last_leaf = nullptr)
        : const_iterator(leaf, slot, last_leaf) {}

    auto operator*() const {
      return KeyCodec<KeyT>::FromStorage(GetLeafKey(this->leaf_, this->slot_));
    }
    auto operator->() const { return SetConstRef{operator*()}; }

    iterator& operator++() {
      const_iterator::operator++();
      return *this;
    }

    iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    iterator& operator--() {
      const_iterator::operator--();
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

    LeafT* node() const { return const_cast<LeafT*>(this->leaf_); }
    LeafT* leaf() const { return const_cast<LeafT*>(this->leaf_); }
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
    // Mixed comparisons (rit == crit). Needed before C++20, where the
    // implicit object parameter does not undergo user-defined conversions.
    bool operator==(const const_reverse_iterator& other) const {
      return current_ == other.base();
    }
    bool operator!=(const const_reverse_iterator& other) const {
      return current_ != other.base();
    }

    operator const_reverse_iterator() const {
      return const_reverse_iterator(current_);
    }

    iterator base() const { return current_; }

   private:
    iterator current_;
  };

  // Keys are decompressed on the fly, so dereferencing yields a value, not an
  // lvalue; mirror the iterator typedefs (as btree_map.h does).
  using reference = typename iterator::reference;              // KeyT
  using const_reference = typename const_iterator::reference;  // KeyT
  using pointer = typename iterator::pointer;                  // SetConstRef
  using const_pointer = typename const_iterator::pointer;      // SetConstRef

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
  const_iterator lower_bound(KeyT key) const {
    auto res = Dispatch::LowerBound(&state_, KeyCodec<KeyT>::ToStorage(key));
    return const_iterator(res.first, res.second, state_.last_leaf_);
  }
  const_iterator upper_bound(KeyT key) const {
    auto res = Dispatch::UpperBound(&state_, KeyCodec<KeyT>::ToStorage(key));
    return const_iterator(res.first, res.second, state_.last_leaf_);
  }
  const_iterator find(KeyT key) const {
    auto res = Dispatch::Find(&state_, KeyCodec<KeyT>::ToStorage(key));
    return const_iterator(res.first, res.second, state_.last_leaf_);
  }

  iterator lower_bound(KeyT key) {
    auto it = static_cast<const BTreeSet*>(this)->lower_bound(key);
    return iterator(const_cast<LeafT*>(it.node()), it.slot(),
                    state_.last_leaf_);
  }
  iterator upper_bound(KeyT key) {
    auto it = static_cast<const BTreeSet*>(this)->upper_bound(key);
    return iterator(const_cast<LeafT*>(it.node()), it.slot(),
                    state_.last_leaf_);
  }
  iterator find(KeyT key) {
    auto it = static_cast<const BTreeSet*>(this)->find(key);
    return iterator(const_cast<LeafT*>(it.node()), it.slot(),
                    state_.last_leaf_);
  }

  // Inserts `key`. Returns (iterator to the element, whether it was inserted).
  // If an insertion occurs, all other iterators are invalidated.
  std::pair<iterator, bool> insert(KeyT key) {
    auto res = Dispatch::Insert(&state_, KeyCodec<KeyT>::ToStorage(key));
    return {iterator(res.first.first, res.first.second, state_.last_leaf_),
            res.second};
  }
  template <typename... Args>
  std::pair<iterator, bool> emplace(Args&&... args) {
    return insert(KeyT(std::forward<Args>(args)...));
  }
  // Erases `key` if present. Returns the number of elements erased (0 or 1).
  // If an erasure occurs, all iterators are invalidated.
  size_t erase(KeyT key) {
    return Dispatch::Erase(&state_, KeyCodec<KeyT>::ToStorage(key));
  }
  // Erases the element at `pos` (must not be end()). Returns an iterator to the
  // following element, or end(). All other iterators are invalidated.
  iterator erase(const_iterator pos) {
    auto res = Dispatch::EraseIter(&state_, pos.leaf(), pos.slot());
    return iterator(res.first, res.second, state_.last_leaf_);
  }

  const LeafT* last_leaf() const { return state_.last_leaf_; }
  LeafT* last_leaf() { return state_.last_leaf_; }
  const BTreeState<StorageKeyT, LeafT>* state() const { return &state_; }

  // Batch Query (Amortizes dynamic dispatch overhead)
  bool contains(KeyT key) const { return Contains(key); }
  bool Contains(KeyT key) const {
    return Dispatch::Contains(&state_, KeyCodec<KeyT>::ToStorage(key));
  }
  void ContainsBatch(const KeyT* keys, size_t count, bool* out) const {
    Dispatch::ContainsBatch(&state_, reinterpret_cast<const StorageKeyT*>(keys),
                            count, kSignMask, out);
  }
  void FindBatch(const KeyT* keys, size_t count, const_iterator* out) const {
    static_assert(sizeof(const_iterator) == 3 * sizeof(void*));
    static_assert(IsTriviallyCopyable<const_iterator>());
    Dispatch::FindBatch(&state_, reinterpret_cast<const StorageKeyT*>(keys),
                        count, kSignMask, reinterpret_cast<void*>(out));
  }
  void LowerBoundBatch(const KeyT* keys, size_t count,
                       const_iterator* out) const {
    static_assert(sizeof(const_iterator) == 3 * sizeof(void*));
    static_assert(IsTriviallyCopyable<const_iterator>());
    Dispatch::LowerBoundBatch(&state_,
                              reinterpret_cast<const StorageKeyT*>(keys), count,
                              kSignMask, reinterpret_cast<void*>(out));
  }

 private:
  using Dispatch = detail::SetDispatch<StorageKeyT>;

  BTreeState<StorageKeyT, LeafT> state_;
};

static_assert(sizeof(BTreeSet<uint32_t>) == sizeof(BTreeState<uint32_t>),
              "BTreeSet must have the exact same size as BTreeState");
static_assert(sizeof(BTreeSet<int32_t>) == sizeof(BTreeState<int32_t>),
              "BTreeSet must have the exact same size as BTreeState");
static_assert(sizeof(BTreeSet<uint64_t>) == sizeof(BTreeState<uint64_t>),
              "BTreeSet must have the exact same size as BTreeState");
static_assert(sizeof(BTreeSet<int64_t>) == sizeof(BTreeState<int64_t>),
              "BTreeSet must have the exact same size as BTreeState");

template <typename KeyT>
void swap(BTreeSet<KeyT>& a, BTreeSet<KeyT>& b) noexcept {
  a.swap(b);
}

template <typename KeyT>
using btree_set = BTreeSet<KeyT>;

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_BTREE_BTREE_SET_H_
