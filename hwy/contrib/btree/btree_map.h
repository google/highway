// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Interface to Highway BTreeMap with dynamic dispatch.
//
// Methods in this header incur a small dynamic dispatch overhead per call.
// To amortize this overhead, callers should prefer using the batch query APIs
// (e.g. ContainsBatch, FindBatch, LowerBoundBatch, LookupBatch). For maximum
// performance / direct inlining without dynamic dispatch overhead, users should
// prefer including "third_party/highway/hwy/contrib/btree/btree-inl.h"
// directly.

#ifndef HIGHWAY_HWY_CONTRIB_BTREE_BTREE_MAP_H_
#define HIGHWAY_HWY_CONTRIB_BTREE_BTREE_MAP_H_

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

// Values are not inspected by SIMD search kernels, so we partially type-erase
// any 32-bit or 64-bit trivially copyable ValueT to uint32_t/uint64_t storage
// via BitCastScalar/pointer casts to reduce pre-compiled instantiations.
//
// Note: For other value sizes, use btree-inl.h directly.
namespace detail {

template <typename KeyT, typename StorageValueT>
struct MapDispatch;

#define HWY_BTREE_MAP_DISPATCH_DECLARE(KeyT, StorageValueT)                   \
  template <>                                                                 \
  struct HWY_CONTRIB_DLLEXPORT MapDispatch<KeyT, StorageValueT> {             \
    using StorageKeyT = typename KeyCodec<KeyT>::StorageKey;                  \
    using Leaf = MapLeafNode<StorageKeyT, StorageValueT>;                     \
    using State = BTreeState<KeyT, Leaf>;                                     \
                                                                              \
    static void Clear(State* state);                                          \
    static void CopyConstruct(State* dst, const State* src);                  \
    static void CopyAssign(State* dst, const State* src);                     \
    static void MoveConstruct(State* dst, State* src);                        \
    static void MoveAssign(State* dst, State* src);                           \
    static void Build(const KeyT* keys, const void* values, size_t count,     \
                      float fill, State* out_state);                          \
    static bool Contains(const State* state, KeyT key);                       \
    static const StorageValueT* FindValue(const State* state, KeyT key);      \
    static std::pair<const Leaf*, size_t> LowerBound(const State* state,      \
                                                     KeyT key);               \
    static std::pair<const Leaf*, size_t> UpperBound(const State* state,      \
                                                     KeyT key);               \
    static std::pair<const Leaf*, size_t> Find(const State* state, KeyT key); \
    static std::pair<std::pair<Leaf*, size_t>, bool> Insert(                  \
        State* state, KeyT key, StorageValueT value, bool assign_if_exists);  \
    static size_t Erase(State* state, KeyT key);                              \
    static void ContainsBatch(const State* state, const KeyT* keys,           \
                              size_t count, bool* out);                       \
    static void FindBatch(const State* state, const KeyT* keys, size_t count, \
                          void* out);                                         \
    static void LowerBoundBatch(const State* state, const KeyT* keys,         \
                                size_t count, void* out);                     \
    static void LookupBatch(const State* state, const KeyT* keys,             \
                            size_t count, void* out_values, bool* out_found); \
  };

#define HWY_BTREE_MAP_DISPATCH_DECLARE_VALUES(KeyT) \
  HWY_BTREE_MAP_DISPATCH_DECLARE(KeyT, uint32_t)    \
  HWY_BTREE_MAP_DISPATCH_DECLARE(KeyT, uint64_t)

HWY_BTREE_MAP_DISPATCH_DECLARE_VALUES(uint32_t);
HWY_BTREE_MAP_DISPATCH_DECLARE_VALUES(int32_t);
HWY_BTREE_MAP_DISPATCH_DECLARE_VALUES(uint64_t);
HWY_BTREE_MAP_DISPATCH_DECLARE_VALUES(int64_t);

#undef HWY_BTREE_MAP_DISPATCH_DECLARE_VALUES
#undef HWY_BTREE_MAP_DISPATCH_DECLARE

}  // namespace detail

// Type-erasure traits: maps any trivially copyable 32-bit type to uint32_t
// storage, and any trivially copyable 64-bit type to uint64_t storage.
// This allows the compiled dispatch table to only instantiate two value sizes
// while supporting floats, doubles, integers, pointers, custom structs etc.
template <typename ValueT>
struct ValueStorageTraits {
  static_assert(std::is_trivially_copyable_v<ValueT>,
                "Highway BTreeMap only supports trivially copyable values.");
  static_assert(sizeof(ValueT) == 4 || sizeof(ValueT) == 8,
                "Highway BTreeMap only supports 32-bit and 64-bit values.");

  using Type = std::conditional_t<sizeof(ValueT) == 4, uint32_t, uint64_t>;
};

template <typename KeyT, typename ValueT>
class BTreeMap {
 public:
  using key_type = KeyT;
  using mapped_type = ValueT;
  using value_type = std::pair<KeyT, ValueT>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using key_compare = std::less<KeyT>;
  using allocator_type = std::allocator<value_type>;

  using StorageValueT = typename ValueStorageTraits<ValueT>::Type;
  using StorageKeyT = typename KeyCodec<KeyT>::StorageKey;
  using LeafT = MapLeafNode<StorageKeyT, StorageValueT>;
  using Dispatch = detail::MapDispatch<KeyT, StorageValueT>;

  static constexpr bool kIsMap = true;

  BTreeMap() = default;
  explicit BTreeMap(const key_compare& /*comp*/,
                    const allocator_type& /*alloc*/ = allocator_type()) {}
  explicit BTreeMap(const allocator_type& /*alloc*/) {}
  ~BTreeMap() { clear(); }

  BTreeMap(const BTreeMap& other) {
    Dispatch::CopyConstruct(&state_, &other.state_);
  }
  BTreeMap& operator=(const BTreeMap& other) {
    if (this != &other) {
      Dispatch::CopyAssign(&state_, &other.state_);
    }
    return *this;
  }
  BTreeMap(BTreeMap&& other) noexcept {
    Dispatch::MoveConstruct(&state_, &other.state_);
  }
  BTreeMap& operator=(BTreeMap&& other) noexcept {
    if (this != &other) {
      Dispatch::MoveAssign(&state_, &other.state_);
    }
    return *this;
  }
  BTreeMap& operator=(std::initializer_list<value_type> ilist) {
    clear();
    for (const auto& v : ilist) insert(v.first, v.second);
    return *this;
  }

  // Bulk-builds a tree from sorted keys and values. The contiguous values array
  // is passed as const void* across dynamic dispatch to avoid strict-aliasing
  // issues across different types sharing the same 32-bit or 64-bit size.
  static BTreeMap Build(const KeyT* sorted_keys, const ValueT* sorted_values,
                        size_t num_keys, float fill_ratio = 1.0f) {
    BTreeMap map;
    Dispatch::Build(sorted_keys, static_cast<const void*>(sorted_values),
                    num_keys, fill_ratio, &map.state_);
    return map;
  }

  void clear() { Dispatch::Clear(&state_); }

  // ---------------------------------------------------------------------------
  // Bidirectional Iterators & Value Proxies
  // ---------------------------------------------------------------------------
  struct MapConstRef {
    KeyT first;
    const mapped_type& second;
  };

  struct MapConstArrowProxy {
    MapConstRef ref;
    const MapConstRef* operator->() const { return &ref; }
  };

  struct MapMutRef {
    KeyT first;
    mapped_type& second;
  };

  struct MapMutArrowProxy {
    MapMutRef ref;
    MapMutRef* operator->() { return &ref; }
  };

  class const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = std::pair<KeyT, ValueT>;
    using difference_type = std::ptrdiff_t;
    using reference = MapConstRef;
    using pointer = MapConstArrowProxy;

    const_iterator() = default;
    const_iterator(const LeafT* leaf, size_t slot,
                   const LeafT* last_leaf = nullptr)
        : leaf_(leaf), slot_(slot), last_leaf_(last_leaf) {}

    // Reinterprets the aligned StorageValueT* array in the leaf node as
    // const mapped_type* with zero copy overhead.
    auto operator*() const {
      const auto* vals = reinterpret_cast<const mapped_type*>(leaf_->Values());
      return MapConstRef{KeyCodec<KeyT>::FromStorage(GetLeafKey(leaf_, slot_)),
                         vals[slot_]};
    }
    auto operator->() const { return MapConstArrowProxy{operator*()}; }

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
    using value_type = std::pair<KeyT, ValueT>;
    using difference_type = std::ptrdiff_t;
    using reference = MapMutRef;
    using pointer = MapMutArrowProxy;

    iterator() = default;
    iterator(LeafT* leaf, size_t slot, LeafT* last_leaf = nullptr)
        : const_iterator(leaf, slot, last_leaf) {}

    // Reinterprets the aligned StorageValueT* array in the leaf node as
    // mapped_type*, providing a mutable reference that allows in-place
    // mutation.
    auto operator*() const {
      auto* vals = reinterpret_cast<mapped_type*>(
          const_cast<LeafT*>(this->leaf_)->Values());
      return MapMutRef{
          KeyCodec<KeyT>::FromStorage(GetLeafKey(this->leaf_, this->slot_)),
          vals[this->slot_]};
    }
    auto operator->() const { return MapMutArrowProxy{operator*()}; }

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

    LeafT* node() const { return const_cast<LeafT*>(this->leaf_); }
    LeafT* leaf() const { return const_cast<LeafT*>(this->leaf_); }
  };

  class const_reverse_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = std::pair<KeyT, ValueT>;
    using difference_type = std::ptrdiff_t;
    using reference = MapConstRef;
    using pointer = MapConstArrowProxy;

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
      return !(*this == other);
    }

    const_iterator base() const { return current_; }

   private:
    const_iterator current_;
  };

  class reverse_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = std::pair<KeyT, ValueT>;
    using difference_type = std::ptrdiff_t;
    using reference = MapMutRef;
    using pointer = MapMutArrowProxy;

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
      return !(*this == other);
    }

    operator const_reverse_iterator() const {
      return const_reverse_iterator(current_);
    }

    iterator base() const { return current_; }

   private:
    iterator current_;
  };

  using reference = typename iterator::reference;
  using const_reference = typename const_iterator::reference;
  using pointer = typename iterator::pointer;
  using const_pointer = typename const_iterator::pointer;

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
    bool operator()(const value_type& lhs, const value_type& rhs) const {
      return lhs.first < rhs.first;
    }
  };

  value_compare value_comp() const { return value_compare(); }
  key_compare key_comp() const { return key_compare(); }
  allocator_type get_allocator() const { return allocator_type(); }

  void swap(BTreeMap& other) noexcept { std::swap(state_, other.state_); }

  // ---------------------------------------------------------------------------
  // Element Access
  // ---------------------------------------------------------------------------
  // Returns a mutable reference to the value associated with key, inserting a
  // default-constructed value if the key does not exist. Reinterprets the
  // leaf's StorageValueT* array as ValueT* with zero overhead.
  ValueT& operator[](KeyT key) {
    auto res = insert(key, ValueT{});
    auto* vals = reinterpret_cast<ValueT*>(res.first.leaf()->Values());
    return vals[res.first.slot()];
  }

  const ValueT& at(KeyT key) const {
    auto it = find(key);
    if (HWY_UNLIKELY(it == end())) {
      HWY_ABORT("BTreeMap::at: key not found");
    }
    const auto* vals = reinterpret_cast<const ValueT*>(it.leaf()->Values());
    return vals[it.slot()];
  }

  ValueT& at(KeyT key) {
    auto it = find(key);
    if (HWY_UNLIKELY(it == end())) {
      HWY_ABORT("BTreeMap::at: key not found");
    }
    auto* vals = reinterpret_cast<ValueT*>(it.leaf()->Values());
    return vals[it.slot()];
  }

  // ---------------------------------------------------------------------------
  // Range Queries & Mutation
  // ---------------------------------------------------------------------------
  const_iterator lower_bound(KeyT key) const {
    auto res = Dispatch::LowerBound(&state_, key);
    return const_iterator(res.first, res.second, state_.last_leaf_);
  }
  const_iterator upper_bound(KeyT key) const {
    auto res = Dispatch::UpperBound(&state_, key);
    return const_iterator(res.first, res.second, state_.last_leaf_);
  }
  const_iterator find(KeyT key) const {
    auto res = Dispatch::Find(&state_, key);
    return const_iterator(res.first, res.second, state_.last_leaf_);
  }

  iterator lower_bound(KeyT key) {
    auto it = static_cast<const BTreeMap*>(this)->lower_bound(key);
    return iterator(const_cast<LeafT*>(it.node()), it.slot(),
                    state_.last_leaf_);
  }
  iterator upper_bound(KeyT key) {
    auto it = static_cast<const BTreeMap*>(this)->upper_bound(key);
    return iterator(const_cast<LeafT*>(it.node()), it.slot(),
                    state_.last_leaf_);
  }
  iterator find(KeyT key) {
    auto it = static_cast<const BTreeMap*>(this)->find(key);
    return iterator(const_cast<LeafT*>(it.node()), it.slot(),
                    state_.last_leaf_);
  }

  // Direct pointer lookup avoiding iterator construction. Reinterprets the
  // aligned 32-bit or 64-bit storage address to the user's ValueT*.
  const ValueT* FindValue(KeyT key) const {
    const StorageValueT* p = Dispatch::FindValue(&state_, key);
    return reinterpret_cast<const ValueT*>(p);
  }
  ValueT* FindValue(KeyT key) {
    return const_cast<ValueT*>(
        static_cast<const BTreeMap*>(this)->FindValue(key));
  }

  // Inserts key and value. Converts value to StorageValueT via BitCastScalar,
  // preventing strict aliasing issues while compiling down to a register move.
  std::pair<iterator, bool> insert(KeyT key, const ValueT& value) {
    StorageValueT s_val = hwy::BitCastScalar<StorageValueT>(value);
    auto res =
        Dispatch::Insert(&state_, key, s_val, /*assign_if_exists=*/false);
    return {iterator(res.first.first, res.first.second, state_.last_leaf_),
            res.second};
  }
  std::pair<iterator, bool> insert(const value_type& kv) {
    return insert(kv.first, kv.second);
  }
  std::pair<iterator, bool> insert_or_assign(KeyT key, const ValueT& value) {
    StorageValueT s_val = hwy::BitCastScalar<StorageValueT>(value);
    auto res = Dispatch::Insert(&state_, key, s_val, /*assign_if_exists=*/true);
    return {iterator(res.first.first, res.first.second, state_.last_leaf_),
            res.second};
  }
  size_t erase(KeyT key) { return Dispatch::Erase(&state_, key); }

  const LeafT* last_leaf() const { return state_.last_leaf_; }
  LeafT* last_leaf() { return state_.last_leaf_; }

  // Batch Query (Amortizes dynamic dispatch overhead)
  bool contains(KeyT key) const { return Contains(key); }
  bool Contains(KeyT key) const { return Dispatch::Contains(&state_, key); }
  void ContainsBatch(const KeyT* keys, size_t count, bool* out) const {
    Dispatch::ContainsBatch(&state_, keys, count, out);
  }
  void FindBatch(const KeyT* keys, size_t count, const_iterator* out) const {
    static_assert(sizeof(const_iterator) == 3 * sizeof(void*));
    Dispatch::FindBatch(&state_, keys, count, reinterpret_cast<void*>(out));
  }
  void LowerBoundBatch(const KeyT* keys, size_t count,
                       const_iterator* out) const {
    static_assert(sizeof(const_iterator) == 3 * sizeof(void*));
    Dispatch::LowerBoundBatch(&state_, keys, count,
                              reinterpret_cast<void*>(out));
  }
  // Bulk-lookups values directly into the caller's ValueT* output buffer.
  // Passed as void* across dynamic dispatch to avoid strict-aliasing issues
  // across different types sharing the same 32-bit or 64-bit size.
  void LookupBatch(const KeyT* keys, size_t count, ValueT* out_values,
                   bool* out_found) const {
    Dispatch::LookupBatch(&state_, keys, count, static_cast<void*>(out_values),
                          out_found);
  }

 private:
  BTreeState<KeyT, LeafT> state_;
};

template <typename KeyT, typename ValueT>
void swap(BTreeMap<KeyT, ValueT>& a, BTreeMap<KeyT, ValueT>& b) noexcept {
  a.swap(b);
}

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_BTREE_BTREE_MAP_H_
