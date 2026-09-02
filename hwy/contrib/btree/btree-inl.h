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

// Related Work:
//   Kim et al., "FAST: Fast Architecture Sensitive Tree Search on Modern CPUs
//   and GPUs", ACM SIGMOD 2010 (Best Paper Award).
//   https://dl.acm.org/doi/10.1145/1807167.1807206

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "hwy/base.h"
#include "hwy/cache_control.h"
#include "hwy/contrib/btree/btree_nodes.h"

#if defined(HIGHWAY_HWY_CONTRIB_BTREE_BTREE_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_BTREE_BTREE_INL_H_
#undef HIGHWAY_HWY_CONTRIB_BTREE_BTREE_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_BTREE_BTREE_INL_H_
#endif

#include "hwy/highway.h"

static_assert(HWY_CXX_LANG >= 201703L, "requires C++17 or later.");

// Check that Highway target is not scalar
#if HWY_TARGET != HWY_SCALAR

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// -----------------------------------------------------------------------------
// Traits Definitions
// -----------------------------------------------------------------------------

template <typename KeyT>
struct SetTraits {
  using key_type = KeyT;
  using storage_key_type = typename KeyCodec<KeyT>::StorageKey;
  using value_type = KeyT;
  using mapped_type = void;
  static constexpr bool kIsMap = false;

  using Leaf = LeafNode<storage_key_type>;

  template <typename OffsetT>
  static constexpr size_t MaxKeys() {
    return Leaf::kDataBytes / sizeof(OffsetT);
  }
};

// Traits specialization for BTreeMap.
template <typename KeyT, typename ValueT>
struct MapTraits {
  using key_type = KeyT;
  using storage_key_type = typename KeyCodec<KeyT>::StorageKey;
  using value_type = std::pair<KeyT, ValueT>;
  using mapped_type = ValueT;
  static constexpr bool kIsMap = true;

  using Leaf = MapLeafNode<storage_key_type, ValueT>;

  template <typename OffsetT>
  static constexpr size_t MaxKeys() {
    return Leaf::template ComputeMaxPairs<OffsetT>();
  }
};

// -----------------------------------------------------------------------------
// Key Decompression & Slot Search Primitives
// -----------------------------------------------------------------------------

// Given an array of compressed offsets, returns
// the number of elements strictly less than target_val (kLowerBound) or
// less than or equal to target_val (kUpperBound).
template <BoundMode kBound = BoundMode::kLowerBound, typename OffsetT,
          size_t kTotal>
HWY_INLINE size_t ScanOffsets(const void* HWY_RESTRICT data,
                              OffsetT target_val) {
  const auto* offsets = static_cast<const OffsetT*>(data);
  const ScalableTag<OffsetT> d;
  const size_t N = Lanes(d);
  const auto v_target = Set(d, target_val);
  static_assert(kTotal <= 512 / sizeof(OffsetT));

  if constexpr (HWY_NATIVE_MASK) {
    // CountTrue is inexpensive with native mask registers and avoids a
    // horizontal vector reduction.
    size_t count = 0;
    size_t i = 0;
    for (; i + N <= kTotal; i += N) {
      const auto v = Load(d, offsets + i);
      if constexpr (kBound == BoundMode::kLowerBound) {
        count += CountTrue(d, Lt(v, v_target));
      } else {
        count += CountTrue(d, Le(v, v_target));
      }
    }

    if (i < kTotal) {
      const size_t remaining = kTotal - i;
      const auto v = LoadU(d, offsets + i);
      const auto mask = FirstN(d, remaining);
      if constexpr (kBound == BoundMode::kLowerBound) {
        count += CountTrue(d, MaskedLt(mask, v, v_target));
      } else {
        count += CountTrue(d, MaskedLe(mask, v, v_target));
      }
    }

    return count;
  } else {
    const auto is_before = [&](const auto v) HWY_ATTR {
      if constexpr (kBound == BoundMode::kLowerBound) {
        return Lt(v, v_target);
      } else {
        return Le(v, v_target);
      }
    };

    // Without native mask registers, accumulate comparison masks in vectors
    // and reduce once to avoid repeated vector-to-scalar transfers. The loop
    // is 2x unrolled to shorten the accumulator dependency chain.
    auto counts0 = Zero(d);
    auto counts1 = Zero(d);
    size_t i = 0;
    if (kTotal >= 2 * N) {
      for (; i <= kTotal - 2 * N; i += 2 * N) {
        counts0 = Sub(counts0, VecFromMask(d, is_before(Load(d, offsets + i))));
        counts1 =
            Sub(counts1, VecFromMask(d, is_before(Load(d, offsets + i + N))));
      }
    }

    for (; i + N <= kTotal; i += N) {
      counts0 = Sub(counts0, VecFromMask(d, is_before(Load(d, offsets + i))));
    }

    if (i < kTotal) {
      const size_t remaining = kTotal - i;
      const auto v = LoadU(d, offsets + i);
      const auto mask = FirstN(d, remaining);
      counts0 = Sub(counts0, VecFromMask(d, And(mask, is_before(v))));
    }

    const auto counts = Add(counts0, counts1);
    if constexpr (sizeof(OffsetT) == 1) {
      const Repartition<uint64_t, decltype(d)> d64;
      return static_cast<size_t>(ReduceSum(d64, SumsOf8(counts)));
    } else {
      return static_cast<size_t>(ReduceSum(d, counts));
    }
  }
}

// Returns true if target_val exists in the compressed offsets array using pure
// vector SIMD equality without scalar ALU popcnt bottlenecks.
template <typename OffsetT, size_t kTotal>
HWY_INLINE bool HasOffset(const void* HWY_RESTRICT data, OffsetT target_val) {
  const auto* offsets = static_cast<const OffsetT*>(data);
  const ScalableTag<OffsetT> d;
  const size_t N = Lanes(d);
  const auto v_target = Set(d, target_val);
  auto any_match = MaskFalse(d);
  size_t i = 0;
  for (; i + N <= kTotal; i += N) {
    const auto v = Load(d, offsets + i);
    any_match = Or(any_match, Eq(v, v_target));
  }

  if (i < kTotal) {
    const size_t remaining = kTotal - i;
    const auto v = LoadU(d, offsets + i);
    const auto mask = FirstN(d, remaining);
    const auto tail_match = MaskedEq(mask, v, v_target);
    any_match = Or(any_match, tail_match);
  }

  return !AllFalse(d, any_match);
}

// Finds the lower_bound or upper_bound slot index (0..num_keys) for target
// within a leaf node.
template <BoundMode kBound = BoundMode::kLowerBound, typename LeafNode,
          typename KeyT>
HWY_INLINE size_t FindLeafSlot(const LeafNode* HWY_RESTRICT leaf, KeyT target) {
  using Node = LeafNode;
  if constexpr (kBound == BoundMode::kLowerBound) {
    if (HWY_UNLIKELY(target <= leaf->base_key)) return 0;
  } else {
    if (HWY_UNLIKELY(target < leaf->base_key)) return 0;
  }

  const uint64_t delta = static_cast<uint64_t>(target - leaf->base_key);
  const uint8_t mode = leaf->BitMode();

  if (HWY_LIKELY(mode == kMode16Bit)) {
    if (HWY_UNLIKELY(delta > 65535)) return leaf->NumKeys();
    return ScanOffsets<kBound, uint16_t, Node::kMax16>(
        leaf->KeyData(), static_cast<uint16_t>(delta));
  } else if (mode == kMode8Bit) {
    if (HWY_UNLIKELY(delta > 255)) return leaf->NumKeys();
    return ScanOffsets<kBound, uint8_t, Node::kMax8>(
        leaf->KeyData(), static_cast<uint8_t>(delta));
  } else if (mode == kMode32Bit) {
    if constexpr (sizeof(KeyT) == 4) {
      return ScanOffsets<kBound, uint32_t, Node::kMax32>(
          leaf->KeyData(), static_cast<uint32_t>(target));
    } else {
      if (HWY_UNLIKELY(delta > 0xFFFFFFFFULL)) return leaf->NumKeys();
      return ScanOffsets<kBound, uint32_t, Node::kMax32>(
          leaf->KeyData(), static_cast<uint32_t>(delta));
    }
  } else {
    return ScanOffsets<kBound, uint64_t, Node::kMax64>(
        leaf->KeyData(), static_cast<uint64_t>(target));
  }
}

// Returns true if target exists in the leaf. If found, optionally writes the
// slot index.
template <typename LeafNode, typename KeyT>
HWY_INLINE bool LeafContains(const LeafNode* HWY_RESTRICT leaf, KeyT target,
                             size_t* HWY_RESTRICT out_slot = nullptr) {
  using Node = LeafNode;

  // If caller requested the exact slot index (for insert/erase/find),
  // compute the lower_bound slot via Lt comparisons.
  if (out_slot != nullptr) {
    const size_t slot = FindLeafSlot(leaf, target);
    *out_slot = slot;
    if (slot >= leaf->NumKeys()) return false;
    return GetLeafKey(leaf, slot) == target;
  }

  // Fast-path point lookup via SIMD Eq comparisons (for
  // contains/ContainsBatch).
  if (HWY_UNLIKELY(target < leaf->base_key)) return false;
  if (HWY_UNLIKELY(target == leaf->base_key)) return true;

  const uint64_t delta = static_cast<uint64_t>(target - leaf->base_key);
  const uint8_t mode = leaf->BitMode();

  if (HWY_LIKELY(mode == kMode16Bit)) {
    if (HWY_UNLIKELY(delta > 65535)) return false;
    if (HWY_UNLIKELY(delta == 65535)) {
      return GetLeafKey(leaf, leaf->NumKeys() - 1) == target;
    }
    return HasOffset<uint16_t, Node::kMax16>(leaf->KeyData(),
                                             static_cast<uint16_t>(delta));
  } else if (mode == kMode8Bit) {
    if (HWY_UNLIKELY(delta > 255)) return false;
    if (HWY_UNLIKELY(delta == 255)) {
      return GetLeafKey(leaf, leaf->NumKeys() - 1) == target;
    }
    return HasOffset<uint8_t, Node::kMax8>(leaf->KeyData(),
                                           static_cast<uint8_t>(delta));
  } else if (mode == kMode32Bit) {
    if constexpr (sizeof(KeyT) == 4) {
      if (HWY_UNLIKELY(target == 0xFFFFFFFF)) {
        return GetLeafKey(leaf, leaf->NumKeys() - 1) == target;
      }
      return HasOffset<uint32_t, Node::kMax32>(leaf->KeyData(),
                                               static_cast<uint32_t>(target));
    } else {
      if (HWY_UNLIKELY(delta > 0xFFFFFFFFULL)) return false;
      if (HWY_UNLIKELY(delta == 0xFFFFFFFFULL)) {
        return GetLeafKey(leaf, leaf->NumKeys() - 1) == target;
      }
      return HasOffset<uint32_t, Node::kMax32>(leaf->KeyData(),
                                               static_cast<uint32_t>(delta));
    }
  } else {
    if (HWY_UNLIKELY(target == 0xFFFFFFFFFFFFFFFFULL)) {
      return GetLeafKey(leaf, leaf->NumKeys() - 1) == target;
    }
    return HasOffset<uint64_t, Node::kMax64>(leaf->KeyData(),
                                             static_cast<uint64_t>(target));
  }
}

// Scans an internal node's separator keys and returns the child pointer index
// to descend.
template <typename KeyT>
HWY_INLINE size_t FindChild(const InternalNode<KeyT>* HWY_RESTRICT internal,
                            typename InternalNode<KeyT>::StorageKeyT target) {
  using StorageKeyT = typename InternalNode<KeyT>::StorageKeyT;
  constexpr size_t kCapacity = InternalNode<KeyT>::kCapacity;
  const CappedTag<StorageKeyT, kCapacity> d;
  const size_t N = Lanes(d);
  const auto v_target = Set(d, target);

  size_t count = 0;
  for (size_t i = 0; i < kCapacity; i += N) {
    const auto v_keys = Load(d, internal->keys + i);
    count += CountTrue(d, Le(v_keys, v_target));
  }
  return std::min<size_t>(count, internal->num_keys);
}

// -----------------------------------------------------------------------------
// Leaf-Level Dynamic Mutation & Compression Helpers
// -----------------------------------------------------------------------------

// Decompresses all keys from a leaf into the out_keys destination array.
template <typename LeafNode, typename KeyT>
HWY_INLINE void DecompressLeaf(const LeafNode* leaf, KeyT* out_keys) {
  const size_t count = leaf->NumKeys();
  for (size_t i = 0; i < count; ++i) {
    out_keys[i] = GetLeafKey(leaf, i);
  }
}

// Decompresses all keys and values from a map leaf into destination arrays.
template <typename KeyT, typename ValueT>
HWY_INLINE void DecompressLeaf(const MapLeafNode<KeyT, ValueT>* leaf,
                               KeyT* out_keys, ValueT* out_values) {
  const size_t count = leaf->NumKeys();
  const ValueT* vals = leaf->Values();
  for (size_t i = 0; i < count; ++i) {
    out_keys[i] = GetLeafKey(leaf, i);
    out_values[i] = vals[i];
  }
}

// Encodes sorted keys as delta offsets from base_key into the destination
// buffer.
template <typename OffsetT, OffsetT kSentinel, typename KeyT>
HWY_INLINE void StoreCompressedOffsets(void* dst_data, const KeyT* keys,
                                       size_t count, KeyT base_key,
                                       size_t capacity) {
  auto* dst = HWY_RCAST_ALIGNED(OffsetT*, dst_data);
  for (size_t k = 0; k < count; ++k) {
    dst[k] = static_cast<OffsetT>(keys[k] - base_key);
  }
  std::fill_n(dst + count, capacity - count, kSentinel);
}

// Encodes a sorted key array into a leaf node using the narrowest viable
// compression mode (Set).
template <typename KeyT>
HWY_INLINE void CompressIntoLeaf(LeafNode<KeyT>* leaf, const KeyT* keys,
                                 size_t count) {
  if (count == 0) {
    leaf->SetNumKeys(0);
    leaf->base_key = 0;
    std::memset(leaf->data, 0xFF, LeafNode<KeyT>::kDataBytes);
    return;
  }

  leaf->base_key = keys[0];
  constexpr size_t kMax8 = LeafNode<KeyT>::kMax8;
  constexpr size_t kMax16 = LeafNode<KeyT>::kMax16;
  constexpr size_t kMax32 = LeafNode<KeyT>::kMax32;
  constexpr size_t kMax64 = LeafNode<KeyT>::kMax64;

  const uint64_t max_delta =
      (count > 1) ? static_cast<uint64_t>(keys[count - 1] - keys[0]) : 0;

  if (count <= kMax8 && max_delta <= 255) {
    leaf->SetBitMode(kMode8Bit);
    StoreCompressedOffsets<uint8_t, 0xFF>(leaf->data, keys, count,
                                          leaf->base_key, kMax8);
  } else if (count <= kMax16 && max_delta <= 65535) {
    leaf->SetBitMode(kMode16Bit);
    StoreCompressedOffsets<uint16_t, 0xFFFF>(leaf->data, keys, count,
                                             leaf->base_key, kMax16);
  } else if (count <= kMax32 &&
             (sizeof(KeyT) == 4 || max_delta <= 0xFFFFFFFFULL)) {
    leaf->SetBitMode(kMode32Bit);
    if constexpr (sizeof(KeyT) == 4) {
      auto* dst = HWY_RCAST_ALIGNED(uint32_t*, leaf->data);
      for (size_t k = 0; k < count; ++k) {
        dst[k] = static_cast<uint32_t>(keys[k]);
      }
      std::fill_n(dst + count, kMax32 - count, 0xFFFFFFFF);
    } else {
      StoreCompressedOffsets<uint32_t, 0xFFFFFFFF>(leaf->data, keys, count,
                                                   leaf->base_key, kMax32);
    }
  } else {
    leaf->SetBitMode(kModeRaw64);
    auto* dst = HWY_RCAST_ALIGNED(uint64_t*, leaf->data);
    for (size_t k = 0; k < count; ++k) {
      dst[k] = static_cast<uint64_t>(keys[k]);
    }
    std::fill_n(dst + count, kMax64 - count, 0xFFFFFFFFFFFFFFFFULL);
  }
  leaf->SetNumKeys(static_cast<uint16_t>(count));
}

// Encodes sorted keys and values into a map leaf using the narrowest viable
// compression mode (Map).
template <typename KeyT, typename ValueT>
HWY_INLINE void CompressIntoLeaf(MapLeafNode<KeyT, ValueT>* leaf,
                                 const KeyT* keys, const ValueT* values,
                                 size_t count) {
  if (count == 0) {
    leaf->SetNumKeys(0);
    leaf->base_key = 0;
    std::memset(leaf->payload, 0xFF, MapLeafNode<KeyT, ValueT>::kPayloadBytes);
    return;
  }

  leaf->base_key = keys[0];
  constexpr size_t kMax8 = MapLeafNode<KeyT, ValueT>::kMax8;
  constexpr size_t kMax16 = MapLeafNode<KeyT, ValueT>::kMax16;
  constexpr size_t kMax32 = MapLeafNode<KeyT, ValueT>::kMax32;
  constexpr size_t kMax64 = MapLeafNode<KeyT, ValueT>::kMax64;

  const uint64_t max_delta =
      (count > 1) ? static_cast<uint64_t>(keys[count - 1] - keys[0]) : 0;

  if (count <= kMax8 && max_delta <= 255) {
    leaf->SetBitMode(kMode8Bit);
    StoreCompressedOffsets<uint8_t, 0xFF>(leaf->KeyData(), keys, count,
                                          leaf->base_key, kMax8);
  } else if (count <= kMax16 && max_delta <= 65535) {
    leaf->SetBitMode(kMode16Bit);
    StoreCompressedOffsets<uint16_t, 0xFFFF>(leaf->KeyData(), keys, count,
                                             leaf->base_key, kMax16);
  } else if (count <= kMax32 &&
             (sizeof(KeyT) == 4 || max_delta <= 0xFFFFFFFFULL)) {
    leaf->SetBitMode(kMode32Bit);
    if constexpr (sizeof(KeyT) == 4) {
      auto* dst = HWY_RCAST_ALIGNED(uint32_t*, leaf->KeyData());
      for (size_t k = 0; k < count; ++k) {
        dst[k] = static_cast<uint32_t>(keys[k]);
      }
      std::fill_n(dst + count, kMax32 - count, 0xFFFFFFFF);
    } else {
      StoreCompressedOffsets<uint32_t, 0xFFFFFFFF>(leaf->KeyData(), keys, count,
                                                   leaf->base_key, kMax32);
    }
  } else {
    leaf->SetBitMode(kModeRaw64);
    auto* dst = HWY_RCAST_ALIGNED(uint64_t*, leaf->KeyData());
    for (size_t k = 0; k < count; ++k) {
      dst[k] = static_cast<uint64_t>(keys[k]);
    }
    std::fill_n(dst + count, kMax64 - count, 0xFFFFFFFFFFFFFFFFULL);
  }

  ValueT* vals = leaf->Values();
  std::memcpy(vals, values, count * sizeof(ValueT));
  leaf->SetNumKeys(static_cast<uint16_t>(count));
}

// Returns true if a new key can fit into the leaf (potentially upgrading its
// compression mode).
template <typename LeafNode, typename KeyT>
HWY_INLINE bool CanLeafFitInsert(const LeafNode* leaf, KeyT new_key) {
  const size_t cur_num = leaf->NumKeys();
  if (HWY_UNLIKELY(cur_num == 0)) return true;
  const size_t new_count = cur_num + 1;
  const KeyT min_k = std::min(leaf->base_key, new_key);
  const KeyT max_existing =
      (cur_num > 0) ? GetLeafKey(leaf, cur_num - 1) : leaf->base_key;
  const KeyT max_k = std::max(max_existing, new_key);
  const uint64_t max_delta = static_cast<uint64_t>(max_k - min_k);

  constexpr size_t kMax8 = LeafNode::kMax8;
  constexpr size_t kMax16 = LeafNode::kMax16;
  constexpr size_t kMax32 = LeafNode::kMax32;
  constexpr size_t kMax64 = LeafNode::kMax64;

  if (new_count <= kMax8 && max_delta <= 255) return true;
  if (new_count <= kMax16 && max_delta <= 65535) return true;
  if constexpr (sizeof(KeyT) == 4) return new_count <= kMax32;
  if (new_count <= kMax32 && max_delta <= 0xFFFFFFFFULL) return true;
  return new_count <= kMax64;
}

// Decompresses leaf keys, inserts a new key in sorted order, and returns the
// new count.
template <typename KeyT>
HWY_INLINE size_t DecompressAndInsertKey(const LeafNode<KeyT>* leaf,
                                         KeyT new_key, KeyT* out_keys) {
  DecompressLeaf(leaf, out_keys);
  const size_t count = leaf->NumKeys();
  size_t slot = 0;
  while (slot < count && out_keys[slot] < new_key) {
    slot++;
  }
  if (count > slot) {
    std::memmove(out_keys + slot + 1, out_keys + slot,
                 (count - slot) * sizeof(KeyT));
  }
  out_keys[slot] = new_key;
  return count + 1;
}

// Decompresses all existing pairs from map leaf and inserts (new_key, new_val)
// in sorted order.
template <typename KeyT, typename ValueT>
HWY_INLINE size_t DecompressAndInsertMapPair(
    const MapLeafNode<KeyT, ValueT>* leaf, KeyT new_key,
    const ValueT& new_value, KeyT* out_keys, ValueT* out_values) {
  DecompressLeaf(leaf, out_keys, out_values);
  const size_t count = leaf->NumKeys();
  size_t slot = 0;
  while (slot < count && out_keys[slot] < new_key) {
    slot++;
  }
  std::memmove(out_keys + slot + 1, out_keys + slot,
               (count - slot) * sizeof(KeyT));
  std::memmove(out_values + slot + 1, out_values + slot,
               (count - slot) * sizeof(ValueT));
  out_keys[slot] = new_key;
  out_values[slot] = new_value;
  return count + 1;
}

// Decompresses a leaf, inserts the new key in sorted order, and recompresses
// the leaf (Set).
template <typename KeyT>
HWY_INLINE void InsertIntoLeaf(LeafNode<KeyT>* leaf, KeyT new_key) {
  // Temporary scratch buffer on the stack to hold decompressed keys.
  KeyT temp[512];

  // Decompress existing keys and insert new_key in sorted order.
  const size_t new_count = DecompressAndInsertKey(leaf, new_key, temp);

  // Recompress the sorted array back into the leaf node.
  CompressIntoLeaf(leaf, temp, new_count);
}

// Decompresses a leaf, inserts the new key-value pair in sorted order, and
// recompresses the leaf (Map).
template <typename KeyT, typename ValueT>
HWY_INLINE void InsertIntoLeaf(MapLeafNode<KeyT, ValueT>* leaf, KeyT new_key,
                               const ValueT& new_value) {
  // Temporary scratch buffers on the stack to hold decompressed keys and
  // values.
  KeyT temp_keys[512];
  ValueT temp_values[512];

  // Decompress existing pairs and insert (new_key, new_val) in sorted order.
  const size_t new_count = DecompressAndInsertMapPair(leaf, new_key, new_value,
                                                      temp_keys, temp_values);

  // Recompress the sorted arrays back into the leaf node.
  CompressIntoLeaf(leaf, temp_keys, temp_values, new_count);
}

// In-place fast path for inserting a key into a compressed offset leaf without
// recompression (Set).
template <typename Traits, typename OffsetT, uint64_t kMaxDelta, typename KeyT>
HWY_INLINE bool TryFastInsertOffset(LeafNode<KeyT>* leaf, KeyT new_key,
                                    size_t slot) {
  constexpr size_t kCapacity = LeafNode<KeyT>::kDataBytes / sizeof(OffsetT);
  const size_t count = leaf->NumKeys();
  if (HWY_UNLIKELY(count >= kCapacity)) return false;

  auto* offsets = HWY_RCAST_ALIGNED(OffsetT*, leaf->data);

  // new_key >= base_key.
  // base_key stays unchanged; compute positive offset delta from base_key.
  if (HWY_LIKELY(new_key >= leaf->base_key)) {
    const uint64_t delta = static_cast<uint64_t>(new_key - leaf->base_key);
    if (HWY_LIKELY(delta <= kMaxDelta)) {
      std::memmove(offsets + slot + 1, offsets + slot,
                   (kCapacity - 1 - slot) * sizeof(OffsetT));
      offsets[slot] = static_cast<OffsetT>(delta);
      leaf->SetNumKeys(count + 1);
      return true;
    }
  } else {
    // new_key < base_key (inserted at slot 0).
    // new_key becomes the new base_key (offset 0), and all existing offsets
    // are shifted up by (old_base_key - new_key).
    const KeyT max_existing = GetLeafKey(leaf, count - 1);
    const uint64_t new_span = static_cast<uint64_t>(max_existing - new_key);
    if (HWY_LIKELY(new_span <= kMaxDelta)) {
      const OffsetT shift = static_cast<OffsetT>(leaf->base_key - new_key);
      // Constant-size move for the full payload without libc
      // length branching.
      std::memmove(offsets + 1, offsets, (kCapacity - 1) * sizeof(OffsetT));
      offsets[0] = 0;

      const ScalableTag<OffsetT> d;
      const size_t N = Lanes(d);
      const auto v_shift = Set(d, shift);
      size_t i = 1;
      for (; i + N <= count + 1; i += N) {
        const auto v = LoadU(d, offsets + i);
        StoreU(Add(v, v_shift), d, offsets + i);
      }
      if (i <= count) {
        const size_t remaining = count + 1 - i;
        const auto v = LoadN(d, offsets + i, remaining);
        StoreN(Add(v, v_shift), d, offsets + i, remaining);
      }
      leaf->base_key = new_key;
      leaf->SetNumKeys(count + 1);
      return true;
    }
  }
  return false;
}

// In-place fast path for inserting a key-value pair into a compressed map leaf
// without recompression (Map).
template <typename Traits, typename OffsetT, uint64_t kMaxDelta, typename KeyT,
          typename ValueT>
HWY_INLINE bool TryFastInsertOffset(MapLeafNode<KeyT, ValueT>* leaf,
                                    KeyT new_key, const ValueT& new_value,
                                    size_t slot) {
  const size_t count = leaf->NumKeys();
  const size_t kCapacity = Traits::template MaxKeys<OffsetT>();
  if (HWY_UNLIKELY(count >= kCapacity)) return false;

  auto* offsets = HWY_RCAST_ALIGNED(OffsetT*, leaf->payload);
  ValueT* vals = leaf->Values();

  // Branch 1: new_key >= base_key.
  // base_key stays unchanged; compute positive offset delta and shift values.
  if (HWY_LIKELY(new_key >= leaf->base_key)) {
    const uint64_t delta = static_cast<uint64_t>(new_key - leaf->base_key);
    if (HWY_LIKELY(delta <= kMaxDelta)) {
      std::memmove(offsets + slot + 1, offsets + slot,
                   (count - slot) * sizeof(OffsetT));
      std::memmove(vals + slot + 1, vals + slot,
                   (count - slot) * sizeof(ValueT));
      offsets[slot] = static_cast<OffsetT>(delta);
      vals[slot] = new_value;
      leaf->SetNumKeys(count + 1);
      return true;
    }
  } else {
    // Branch 2: new_key < base_key (inserted at slot 0).
    // new_key becomes the new base_key (offset 0), and all existing offsets
    // and values are shifted up by (old_base_key - new_key).
    const KeyT max_existing = GetLeafKey(leaf, count - 1);
    const uint64_t new_span = static_cast<uint64_t>(max_existing - new_key);
    if (HWY_LIKELY(new_span <= kMaxDelta)) {
      const OffsetT shift = static_cast<OffsetT>(leaf->base_key - new_key);
      std::memmove(offsets + 1, offsets, count * sizeof(OffsetT));
      std::memmove(vals + 1, vals, count * sizeof(ValueT));
      offsets[0] = 0;
      vals[0] = new_value;

      const ScalableTag<OffsetT> d;
      const size_t N = Lanes(d);
      const auto v_shift = Set(d, shift);
      size_t i = 1;
      for (; i + N <= count + 1; i += N) {
        const auto v = LoadU(d, offsets + i);
        StoreU(Add(v, v_shift), d, offsets + i);
      }
      if (i <= count) {
        const size_t remaining = count + 1 - i;
        const auto v = LoadN(d, offsets + i, remaining);
        StoreN(Add(v, v_shift), d, offsets + i, remaining);
      }
      leaf->base_key = new_key;
      leaf->SetNumKeys(count + 1);
      return true;
    }
  }
  return false;
}

// Fast-path dispatcher that attempts in-place insertion into a leaf without
// full recompression (Set).
template <typename Traits, typename KeyT>
HWY_INLINE bool TryFastInsertIntoLeaf(LeafNode<KeyT>* leaf, KeyT new_key,
                                      size_t slot) {
  using Leaf = LeafNode<KeyT>;
  const size_t count = leaf->NumKeys();
  if (HWY_UNLIKELY(count == 0)) {
    CompressIntoLeaf(leaf, &new_key, 1);
    return true;
  }

  const uint8_t mode = leaf->BitMode();
  if (HWY_LIKELY(mode == kMode16Bit)) {
    return TryFastInsertOffset<Traits, uint16_t, 65535>(leaf, new_key, slot);
  } else if (mode == kMode8Bit) {
    return TryFastInsertOffset<Traits, uint8_t, 255>(leaf, new_key, slot);
  } else if (mode == kMode32Bit) {
    if (HWY_UNLIKELY(count >= Leaf::kMax32)) return false;

    if constexpr (sizeof(KeyT) == 4) {
      auto* raw_keys = HWY_RCAST_ALIGNED(uint32_t*, leaf->data);
      std::memmove(raw_keys + slot + 1, raw_keys + slot,
                   (Leaf::kMax32 - 1 - slot) * sizeof(uint32_t));
      raw_keys[slot] = static_cast<uint32_t>(new_key);
      if (HWY_UNLIKELY(slot == 0)) leaf->base_key = new_key;
      leaf->SetNumKeys(count + 1);
      return true;
    } else {
      return TryFastInsertOffset<Traits, uint32_t, 0xFFFFFFFFULL>(leaf, new_key,
                                                                  slot);
    }
  } else {
    if (HWY_UNLIKELY(count >= Leaf::kMax64)) return false;
    auto* raw_keys = HWY_RCAST_ALIGNED(uint64_t*, leaf->data);
    std::memmove(raw_keys + slot + 1, raw_keys + slot,
                 (Leaf::kMax64 - 1 - slot) * sizeof(uint64_t));
    raw_keys[slot] = static_cast<uint64_t>(new_key);
    if (HWY_UNLIKELY(slot == 0)) leaf->base_key = new_key;
    leaf->SetNumKeys(count + 1);
    return true;
  }
}

// Fast-path dispatcher that attempts in-place insertion into a map leaf across
// all bit modes (Map).
template <typename Traits, typename KeyT, typename ValueT>
HWY_INLINE bool TryFastInsertIntoLeaf(MapLeafNode<KeyT, ValueT>* leaf,
                                      KeyT new_key, const ValueT& value,
                                      size_t slot) {
  using Leaf = MapLeafNode<KeyT, ValueT>;
  const size_t count = leaf->NumKeys();
  if (count == 0) {
    CompressIntoLeaf(leaf, &new_key, &value, 1);
    return true;
  }

  const uint8_t mode = leaf->BitMode();
  if (HWY_LIKELY(mode == kMode16Bit)) {
    return TryFastInsertOffset<Traits, uint16_t, 65535>(leaf, new_key, value,
                                                        slot);
  } else if (mode == kMode8Bit) {
    return TryFastInsertOffset<Traits, uint8_t, 255>(leaf, new_key, value,
                                                     slot);
  } else if (mode == kMode32Bit) {
    if (HWY_UNLIKELY(count >= Leaf::kMax32)) return false;
    ValueT* vals = leaf->Values();

    if constexpr (sizeof(KeyT) == 4) {
      auto* raw_keys = HWY_RCAST_ALIGNED(uint32_t*, leaf->payload);
      std::memmove(raw_keys + slot + 1, raw_keys + slot,
                   (count - slot) * sizeof(uint32_t));
      std::memmove(vals + slot + 1, vals + slot,
                   (count - slot) * sizeof(ValueT));
      raw_keys[slot] = static_cast<uint32_t>(new_key);
      vals[slot] = value;
      if (HWY_UNLIKELY(slot == 0)) leaf->base_key = new_key;
      leaf->SetNumKeys(count + 1);
      return true;
    } else {
      return TryFastInsertOffset<Traits, uint32_t, 0xFFFFFFFFULL>(leaf, new_key,
                                                                  value, slot);
    }
  } else {
    if (HWY_UNLIKELY(count >= Leaf::kMax64)) return false;
    ValueT* vals = leaf->Values();
    auto* raw_keys = HWY_RCAST_ALIGNED(uint64_t*, leaf->payload);
    std::memmove(raw_keys + slot + 1, raw_keys + slot,
                 (count - slot) * sizeof(uint64_t));
    std::memmove(vals + slot + 1, vals + slot, (count - slot) * sizeof(ValueT));
    raw_keys[slot] = static_cast<uint64_t>(new_key);
    vals[slot] = value;
    if (HWY_UNLIKELY(slot == 0)) leaf->base_key = new_key;
    leaf->SetNumKeys(count + 1);
    return true;
  }
}

// In-place fast path for erasing a key at slot from a compressed offset leaf
// (Set).
template <typename OffsetT, OffsetT kSentinel, typename KeyT>
HWY_INLINE bool TryFastEraseOffset(LeafNode<KeyT>* leaf, size_t slot) {
  constexpr size_t kCapacity = LeafNode<KeyT>::kDataBytes / sizeof(OffsetT);
  const size_t count = leaf->NumKeys();
  auto* offsets = HWY_RCAST_ALIGNED(OffsetT*, leaf->data);
  // If erasing slot 0 (the base_key), shift all offsets and advance base_key.
  if (HWY_UNLIKELY(slot == 0)) {
    const OffsetT shift = offsets[1];
    for (size_t i = 1; i < count; ++i) {
      offsets[i - 1] = offsets[i] - shift;
    }
    offsets[count - 1] = kSentinel;
    leaf->base_key += shift;
  } else {
    // If erasing slot > 0, shift subsequent offsets left.
    std::memmove(offsets + slot, offsets + slot + 1,
                 (kCapacity - 1 - slot) * sizeof(OffsetT));
    offsets[count - 1] = kSentinel;
  }
  leaf->SetNumKeys(count - 1);
  return true;
}

// In-place fast path for erasing a key-value pair from a compressed map leaf
// (Map).
template <typename OffsetT, OffsetT kSentinel, typename KeyT, typename ValueT>
HWY_INLINE bool TryFastEraseOffset(MapLeafNode<KeyT, ValueT>* leaf,
                                   size_t slot) {
  const size_t count = leaf->NumKeys();
  auto* offsets = HWY_RCAST_ALIGNED(OffsetT*, leaf->payload);
  ValueT* vals = leaf->Values();
  // If erasing slot 0 (the base_key), shift all offsets and advance base_key.
  if (HWY_UNLIKELY(slot == 0)) {
    const OffsetT shift = offsets[1];
    for (size_t i = 1; i < count; ++i) {
      offsets[i - 1] = offsets[i] - shift;
    }
    std::memmove(vals, vals + 1, (count - 1) * sizeof(ValueT));
    offsets[count - 1] = kSentinel;
    leaf->base_key += shift;
  } else {
    // If erasing slot > 0, shift subsequent offsets and values left.
    std::memmove(offsets + slot, offsets + slot + 1,
                 (count - 1 - slot) * sizeof(OffsetT));
    std::memmove(vals + slot, vals + slot + 1,
                 (count - 1 - slot) * sizeof(ValueT));
    offsets[count - 1] = kSentinel;
  }
  leaf->SetNumKeys(count - 1);
  return true;
}

// Fast-path dispatcher that erases a key from a leaf in-place across all bit
// modes (Set).
template <typename KeyT>
HWY_INLINE bool TryFastEraseFromLeaf(LeafNode<KeyT>* leaf, size_t slot) {
  using Leaf = LeafNode<KeyT>;
  const size_t count = leaf->NumKeys();
  if (HWY_UNLIKELY(count <= 1)) {
    leaf->SetNumKeys(0);
    leaf->base_key = 0;
    std::memset(leaf->data, 0xFF, Leaf::kDataBytes);
    return true;
  }

  const uint8_t mode = leaf->BitMode();
  if (HWY_LIKELY(mode == kMode16Bit)) {
    return TryFastEraseOffset<uint16_t, 0xFFFF>(leaf, slot);
  } else if (mode == kMode8Bit) {
    return TryFastEraseOffset<uint8_t, 0xFF>(leaf, slot);
  } else if (mode == kMode32Bit) {
    if constexpr (sizeof(KeyT) == 4) {
      auto* raw_keys = HWY_RCAST_ALIGNED(uint32_t*, leaf->data);
      std::memmove(raw_keys + slot, raw_keys + slot + 1,
                   (Leaf::kMax32 - 1 - slot) * sizeof(uint32_t));
      raw_keys[count - 1] = 0xFFFFFFFF;
      if (HWY_UNLIKELY(slot == 0 && count > 1)) {
        leaf->base_key = static_cast<KeyT>(raw_keys[0]);
      }
      leaf->SetNumKeys(count - 1);
      return true;
    } else {
      return TryFastEraseOffset<uint32_t, 0xFFFFFFFF>(leaf, slot);
    }
  } else {
    auto* raw_keys = HWY_RCAST_ALIGNED(uint64_t*, leaf->data);
    std::memmove(raw_keys + slot, raw_keys + slot + 1,
                 (Leaf::kMax64 - 1 - slot) * sizeof(uint64_t));
    raw_keys[count - 1] = 0xFFFFFFFFFFFFFFFFULL;
    if (HWY_UNLIKELY(slot == 0 && count > 1)) {
      leaf->base_key = static_cast<KeyT>(raw_keys[0]);
    }
    leaf->SetNumKeys(count - 1);
    return true;
  }
}

// Fast-path dispatcher that erases a key from a map leaf in-place across all
// bit modes (Map).
template <typename KeyT, typename ValueT>
HWY_INLINE bool TryFastEraseFromLeaf(MapLeafNode<KeyT, ValueT>* leaf,
                                     size_t slot) {
  using Leaf = MapLeafNode<KeyT, ValueT>;
  const size_t count = leaf->NumKeys();
  if (HWY_UNLIKELY(count <= 1)) {
    leaf->SetNumKeys(0);
    leaf->base_key = 0;
    std::memset(leaf->payload, 0xFF, Leaf::kPayloadBytes);
    return true;
  }

  const uint8_t mode = leaf->BitMode();
  if (HWY_LIKELY(mode == kMode16Bit)) {
    return TryFastEraseOffset<uint16_t, 0xFFFF>(leaf, slot);
  } else if (mode == kMode8Bit) {
    return TryFastEraseOffset<uint8_t, 0xFF>(leaf, slot);
  } else if (mode == kMode32Bit) {
    ValueT* vals = leaf->Values();
    if constexpr (sizeof(KeyT) == 4) {
      auto* raw_keys = HWY_RCAST_ALIGNED(uint32_t*, leaf->payload);
      std::memmove(raw_keys + slot, raw_keys + slot + 1,
                   (count - 1 - slot) * sizeof(uint32_t));
      std::memmove(vals + slot, vals + slot + 1,
                   (count - 1 - slot) * sizeof(ValueT));
      raw_keys[count - 1] = 0xFFFFFFFF;
      if (HWY_UNLIKELY(slot == 0 && count > 1)) {
        leaf->base_key = static_cast<KeyT>(raw_keys[0]);
      }
      leaf->SetNumKeys(count - 1);
      return true;
    } else {
      return TryFastEraseOffset<uint32_t, 0xFFFFFFFF>(leaf, slot);
    }
  } else {
    ValueT* vals = leaf->Values();
    auto* raw_keys = HWY_RCAST_ALIGNED(uint64_t*, leaf->payload);
    std::memmove(raw_keys + slot, raw_keys + slot + 1,
                 (count - 1 - slot) * sizeof(uint64_t));
    std::memmove(vals + slot, vals + slot + 1,
                 (count - 1 - slot) * sizeof(ValueT));
    raw_keys[count - 1] = 0xFFFFFFFFFFFFFFFFULL;
    if (HWY_UNLIKELY(slot == 0 && count > 1)) {
      leaf->base_key = static_cast<KeyT>(raw_keys[0]);
    }
    leaf->SetNumKeys(count - 1);
    return true;
  }
}

// Splits a full leaf plus a new key into two balanced leaves and returns the
// separator key (Set).
template <typename KeyT>
HWY_INLINE void SplitLeafNode(LeafNode<KeyT>* leaf, LeafNode<KeyT>* new_leaf,
                              KeyT new_key, KeyT* out_promo_key) {
  // Stack storage to avoid heap allocation.
  KeyT temp[512];

  // Decompress all existing keys from leaf and insert new_key in sorted
  // order.
  const size_t total = DecompressAndInsertKey(leaf, new_key, temp);

  // Position-biased split (similar to absl::btree):
  // When appending at the end of the rightmost leaf (ascending sequence),
  // bias the split so the left leaf stays full (100% fill factor).
  // When prepending at the start of the leftmost leaf (descending sequence),
  // bias the split so the right leaf stays full.
  // Otherwise, split 50/50 for balanced tree depth under random workloads.
  size_t split_point = total / 2;
  if (leaf->Next() == nullptr && new_key == temp[total - 1]) {
    split_point = total - 1;
  } else if (leaf->Prev() == nullptr && new_key == temp[0]) {
    split_point = 1;
  }

  // Recompress left half (temp[0..split_point-1]) back into original leaf.
  CompressIntoLeaf(leaf, temp, split_point);

  // Recompress right half (temp[split_point..total-1]) into the new_leaf.
  CompressIntoLeaf(new_leaf, temp + split_point, total - split_point);

  // The first key of right half becomes the separator key for parent
  // routing.
  *out_promo_key = new_leaf->base_key;
}

// Splits a full leaf plus a new key/value pair into two balanced leaves and
// returns the separator key (Map).
template <typename KeyT, typename ValueT>
HWY_INLINE void SplitLeafNode(MapLeafNode<KeyT, ValueT>* leaf,
                              MapLeafNode<KeyT, ValueT>* new_leaf, KeyT new_key,
                              const ValueT& new_value, KeyT* out_promo_key) {
  // Stack storage to avoid heap allocation.
  KeyT temp_keys[512];
  ValueT temp_values[512];

  // Decompress all existing pairs from leaf and insert (new_key, new_value) in
  // sorted order.
  const size_t total = DecompressAndInsertMapPair(leaf, new_key, new_value,
                                                  temp_keys, temp_values);

  // Position-biased split (similar to absl::btree):
  // When appending at the end of the rightmost leaf (ascending sequence),
  // bias the split so the left leaf stays full (100% fill factor).
  // When prepending at the start of the leftmost leaf (descending sequence),
  // bias the split so the right leaf stays full.
  // Otherwise, split 50/50 for balanced tree depth under random workloads.
  size_t split_point = total / 2;
  if (leaf->Next() == nullptr && new_key == temp_keys[total - 1]) {
    split_point = total - 1;
  } else if (leaf->Prev() == nullptr && new_key == temp_keys[0]) {
    split_point = 1;
  }

  // Recompress left half back into original leaf.
  CompressIntoLeaf(leaf, temp_keys, temp_values, split_point);

  // Recompress right half into the new_leaf.
  CompressIntoLeaf(new_leaf, temp_keys + split_point, temp_values + split_point,
                   total - split_point);

  // The first key of right half becomes the separator key for parent routing.
  *out_promo_key = temp_keys[split_point];
}

// Returns true if two adjacent leaves can merge without exceeding leaf
// capacity.
template <typename LeafNode>
HWY_INLINE bool CanMergeLeaves(const LeafNode* leaf,
                               const LeafNode* next_leaf) {
  using Node = LeafNode;
  using KeyT = decltype(leaf->base_key);
  const size_t total_keys = leaf->NumKeys() + next_leaf->NumKeys();
  if (total_keys > Node::kMax16) return false;
  if (leaf->NumKeys() == 0 || next_leaf->NumKeys() == 0) return true;

  const KeyT max_key = GetLeafKey(next_leaf, next_leaf->NumKeys() - 1);
  const uint64_t spread = static_cast<uint64_t>(max_key - leaf->base_key);

  if (spread <= 65535) return true;
  if constexpr (sizeof(KeyT) == 4) {
    return (total_keys <= Node::kMax32);
  } else {
    if (spread <= 0xFFFFFFFFULL) {
      return (total_keys <= Node::kMax32);
    }
    return (total_keys <= Node::kMax64);
  }
}

// Merges next_leaf into leaf, updates doubly linked pointers, and frees
// next_leaf (Set).
template <typename KeyT>
HWY_INLINE void MergeLeaves(LeafNode<KeyT>* leaf, LeafNode<KeyT>* next_leaf,
                            LeafNode<KeyT>*& last_leaf) {
  const size_t leaf_keys = leaf->NumKeys();
  const size_t next_keys = next_leaf->NumKeys();
  if (next_keys > 0) {
    KeyT temp_keys[512];
    DecompressLeaf(leaf, temp_keys);
    DecompressLeaf(next_leaf, temp_keys + leaf_keys);
    const size_t total_keys = leaf_keys + next_keys;
    CompressIntoLeaf(leaf, temp_keys, total_keys);
  }

  LeafNode<KeyT>* next_next = next_leaf->Next();
  leaf->SetNext(next_next);
  if (next_next != nullptr) {
    next_next->SetPrev(leaf);
  } else {
    last_leaf = leaf;
  }

  delete next_leaf;
}

// Merges next_leaf into leaf, updates doubly linked pointers, and frees
// next_leaf (Map).
template <typename KeyT, typename ValueT>
HWY_INLINE void MergeLeaves(MapLeafNode<KeyT, ValueT>* leaf,
                            MapLeafNode<KeyT, ValueT>* next_leaf,
                            MapLeafNode<KeyT, ValueT>*& last_leaf) {
  const size_t leaf_keys = leaf->NumKeys();
  const size_t next_keys = next_leaf->NumKeys();
  if (next_keys > 0) {
    KeyT temp_keys[512];
    ValueT temp_values[512];
    DecompressLeaf(leaf, temp_keys, temp_values);
    DecompressLeaf(next_leaf, temp_keys + leaf_keys, temp_values + leaf_keys);
    const size_t total_keys = leaf_keys + next_keys;
    CompressIntoLeaf(leaf, temp_keys, temp_values, total_keys);
  }

  MapLeafNode<KeyT, ValueT>* next_next = next_leaf->Next();
  leaf->SetNext(next_next);
  if (next_next != nullptr) {
    next_next->SetPrev(leaf);
  } else {
    last_leaf = leaf;
  }

  delete next_leaf;
}

// -----------------------------------------------------------------------------
// Unified BTree Engine
// -----------------------------------------------------------------------------

// Supports two operational modes:
// 1. Standalone Mode (Owning): Default-constructed BTree instances own their
//    own state (owned_state_) and deallocate nodes upon destruction.
// 2. Adapter Mode (Non-owning): Constructed with an external BTreeState pointer
//    (e.g., from hwy::BTreeSet). Operates directly on the external state
//    without copying data, and leaves node deallocation to the external owner.
template <typename Traits>
class BTree {
 public:
  using KeyT = typename Traits::key_type;
  using StorageKeyT = typename Traits::storage_key_type;
  using Leaf = typename Traits::Leaf;
  using Internal = InternalNode<StorageKeyT>;
  using State = BTreeState<KeyT, Leaf>;

  State* state() { return state_; }
  const State* state() const { return state_; }
  using key_type = KeyT;
  using value_type = typename Traits::value_type;
  using mapped_type = typename Traits::mapped_type;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  static constexpr bool kIsMap = Traits::kIsMap;

  // ---------------------------------------------------------------------------
  // Bidirectional Iterators
  // ---------------------------------------------------------------------------
  // Iterators traverse leaf linked-list nodes in ascending sorted order.
  // Because keys are compressed dynamically on the fly, operator* and
  // operator-> return lightweight proxy objects without heap allocation.

  struct SetConstRef {
    KeyT val;
    KeyT operator*() const { return val; }
    const KeyT* operator->() const { return &val; }
  };

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
    using value_type = typename Traits::value_type;
    using difference_type = std::ptrdiff_t;
    using reference = hwy::If<Traits::kIsMap, MapConstRef, KeyT>;
    using pointer = hwy::If<Traits::kIsMap, MapConstArrowProxy, SetConstRef>;

    const_iterator() = default;
    const_iterator(const Leaf* leaf, size_t slot,
                   const Leaf* last_leaf = nullptr)
        : leaf_(leaf), slot_(slot), last_leaf_(last_leaf) {}

    auto operator*() const {
      if constexpr (Traits::kIsMap) {
        return MapConstRef{
            KeyCodec<KeyT>::FromStorage(GetLeafKey(leaf_, slot_)),
            leaf_->Values()[slot_]};
      } else {
        return KeyCodec<KeyT>::FromStorage(GetLeafKey(leaf_, slot_));
      }
    }

    auto operator->() const {
      if constexpr (Traits::kIsMap) {
        return MapConstArrowProxy{operator*()};
      } else {
        return SetConstRef{
            KeyCodec<KeyT>::FromStorage(GetLeafKey(leaf_, slot_))};
      }
    }

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

    const Leaf* node() const { return leaf_; }
    const Leaf* leaf() const { return leaf_; }
    size_t slot() const { return slot_; }

   protected:
    const Leaf* leaf_ = nullptr;
    size_t slot_ = 0;
    const Leaf* last_leaf_ = nullptr;
  };

  class iterator : public const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = typename Traits::value_type;
    using difference_type = std::ptrdiff_t;
    using reference = hwy::If<Traits::kIsMap, MapMutRef, KeyT>;
    using pointer = hwy::If<Traits::kIsMap, MapMutArrowProxy, SetConstRef>;

    iterator() = default;
    iterator(Leaf* leaf, size_t slot, Leaf* last_leaf = nullptr)
        : const_iterator(leaf, slot, last_leaf) {}

    auto operator*() const {
      if constexpr (Traits::kIsMap) {
        return MapMutRef{
            KeyCodec<KeyT>::FromStorage(GetLeafKey(this->leaf_, this->slot_)),
            const_cast<Leaf*>(this->leaf_)->Values()[this->slot_]};
      } else {
        return KeyCodec<KeyT>::FromStorage(
            GetLeafKey(this->leaf_, this->slot_));
      }
    }

    auto operator->() const {
      if constexpr (Traits::kIsMap) {
        return MapMutArrowProxy{operator*()};
      } else {
        return SetConstRef{
            KeyCodec<KeyT>::FromStorage(GetLeafKey(this->leaf_, this->slot_))};
      }
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

    Leaf* leaf() const { return const_cast<Leaf*>(this->leaf_); }
  };

  class const_reverse_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = typename Traits::value_type;
    using difference_type = std::ptrdiff_t;
    using reference = hwy::If<Traits::kIsMap, MapConstRef, KeyT>;
    using pointer = hwy::If<Traits::kIsMap, MapConstArrowProxy, SetConstRef>;

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
    using value_type = typename Traits::value_type;
    using difference_type = std::ptrdiff_t;
    using reference = hwy::If<Traits::kIsMap, MapMutRef, KeyT>;
    using pointer = hwy::If<Traits::kIsMap, MapMutArrowProxy, SetConstRef>;

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

  using key_compare = std::less<KeyT>;

  class value_compare {
   public:
    bool operator()(const value_type& lhs, const value_type& rhs) const {
      if constexpr (kIsMap) {
        return key_compare()(lhs.first, rhs.first);
      } else {
        return key_compare()(lhs, rhs);
      }
    }
  };

  using reference = typename iterator::reference;
  using const_reference = typename const_iterator::reference;
  using pointer = typename iterator::pointer;
  using const_pointer = typename const_iterator::pointer;
  using allocator_type = std::allocator<value_type>;

  // Standalone constructor: owns its own state.
  BTree() : state_(&owned_state_) {}
  explicit BTree(const key_compare& /*comp*/) : BTree() {}

  // Adapter constructor: operates directly on an external BTreeState (e.g.
  // from hwy::BTreeSet) without copying nodes or taking ownership.
  explicit BTree(State* external_state) : state_(external_state) {}

  ~BTree() {
    // Only deallocate nodes if this instance owns the state. In adapter mode,
    // the external owner (e.g. BTreeSet) manages the node lifetimes.
    if (state_ == &owned_state_) {
      clear();
    }
  }

  BTree(BTree&& other) noexcept {
    if (other.state_ == &other.owned_state_) {
      owned_state_ = other.owned_state_;
      other.owned_state_ = State{};
      state_ = &owned_state_;
    } else {
      state_ = other.state_;
    }
    other.state_ = &other.owned_state_;
  }

  BTree& operator=(BTree&& other) noexcept {
    if (this != &other) {
      clear();
      if (other.state_ == &other.owned_state_) {
        owned_state_ = other.owned_state_;
        other.owned_state_ = State{};
        state_ = &owned_state_;
      } else {
        state_ = other.state_;
      }
      other.state_ = &other.owned_state_;
    }
    return *this;
  }

  BTree(const BTree& other) : state_(&owned_state_) {
    if (other.empty()) return;
    if constexpr (Traits::kIsMap) {
      std::vector<KeyT> keys;
      std::vector<mapped_type> values;
      keys.reserve(other.size());
      values.reserve(other.size());
      for (auto it = other.begin(); it != other.end(); ++it) {
        auto pair = *it;
        keys.push_back(pair.first);
        values.push_back(pair.second);
      }
      *this = Build(keys.data(), values.data(), keys.size());
    } else {
      std::vector<KeyT> keys;
      keys.reserve(other.size());
      for (auto it = other.begin(); it != other.end(); ++it) {
        keys.push_back(*it);
      }
      *this = Build(keys.data(), keys.size());
    }
  }

  BTree& operator=(const BTree& other) {
    if (this != &other) {
      BTree temp(other);
      swap(temp);
    }
    return *this;
  }

  BTree& operator=(std::initializer_list<value_type> ilist) {
    clear();
    for (const auto& item : ilist) {
      insert(item);
    }
    return *this;
  }

  void swap(BTree& other) noexcept {
    using std::swap;
    swap(*state_, *other.state_);
  }

  friend void swap(BTree& a, BTree& b) noexcept { a.swap(b); }

  // ---------------------------------------------------------------------------
  // Bulk Construction from Sorted Keys / Key-Value Pairs
  // ---------------------------------------------------------------------------

  // Constructs a BTreeSet from an array of pre-sorted, unique keys in
  // O(N) time.
  //
  // Example usage:
  //   std::vector<uint32_t> sorted_keys = {10, 20, 30, 40, 50};
  //   auto tree = BTreeSet<uint32_t>::Build(sorted_keys.data(),
  //                                               sorted_keys.size());
  //   bool found = tree.contains(30);
  //
  // Parameters:
  // - keys: Pointer to an array of strictly ascending keys.
  // - num_keys: Total number of keys in the array.
  // - fill_ratio: Target fill factor per leaf node (between 0.1 and 1.0).
  //     * 1.0f (default): Maximum memory density and fastest throughput for
  //       read-heavy or static lookup workloads.
  //     * 0.75f - 0.85f: Leaves headroom in leaves to accommodate subsequent
  //       dynamic insertions without immediate splits.
  static BTree Build(const KeyT* sorted_keys, size_t num_keys,
                     float fill_ratio = 1.0f) {
    static_assert(!Traits::kIsMap, "Build with keys only is for Sets");
    if constexpr (IsSigned<KeyT>()) {
      // Allocate temporary contiguous storage buffer to encode signed keys
      // upfront. Although heap-allocated, this buffer is transient (freed
      // immediately upon Build return), keeping the builder algorithm in
      // BuildInternal uniform and simple.
      std::vector<StorageKeyT> ukeys(num_keys);
      for (size_t i = 0; i < num_keys; ++i) {
        ukeys[i] = KeyCodec<KeyT>::ToStorage(sorted_keys[i]);
      }
      return BuildInternal(ukeys.data(),
                           static_cast<const mapped_type*>(nullptr), num_keys,
                           fill_ratio);
    } else {
      return BuildInternal(sorted_keys,
                           static_cast<const mapped_type*>(nullptr), num_keys,
                           fill_ratio);
    }
  }

  // Constructs a BTreeMap from pre-sorted arrays of unique keys and
  // values in O(N) time.
  //
  // Example usage:
  //   std::vector<uint32_t> sorted_keys = {10, 20, 30, 40, 50};
  //   std::vector<uint64_t> sorted_vals = {100, 200, 300, 400, 500};
  //   auto map = BTreeMap<uint32_t, uint64_t>::Build(
  //       sorted_keys.data(), sorted_vals.data(), sorted_keys.size());
  //   bool found = map.contains(30);
  //
  // Parameters:
  // - sorted_keys: Pointer to strictly ascending keys.
  // - sorted_values: Pointer to corresponding values.
  // - num_keys: Number of key-value pairs.
  // - fill_ratio: Target fill factor per leaf node (between 0.1 and 1.0).
  template <typename V = mapped_type,
            typename = std::enable_if_t<Traits::kIsMap && !std::is_void_v<V>>>
  static BTree Build(const KeyT* sorted_keys, const V* sorted_values,
                     size_t num_keys, float fill_ratio = 1.0f) {
    if constexpr (IsSigned<KeyT>()) {
      // Allocate temporary contiguous storage buffer to encode signed keys
      // upfront. Although heap-allocated, this buffer is transient (freed
      // immediately upon Build return), keeping the builder algorithm in
      // BuildInternal uniform and simple.
      std::vector<StorageKeyT> ukeys(num_keys);
      for (size_t i = 0; i < num_keys; ++i) {
        ukeys[i] = KeyCodec<KeyT>::ToStorage(sorted_keys[i]);
      }
      return BuildInternal(ukeys.data(), sorted_values, num_keys, fill_ratio);
    } else {
      return BuildInternal(sorted_keys, sorted_values, num_keys, fill_ratio);
    }
  }

  // ---------------------------------------------------------------------------
  // Point Lookups
  // ---------------------------------------------------------------------------

  // Returns true if key is present in the tree.
  bool contains(KeyT key) const {
    return ContainsInternal(KeyCodec<KeyT>::ToStorage(key));
  }

  // Returns const_iterator to key if found, or end() otherwise.
  const_iterator find(KeyT key) const {
    return FindInternal(KeyCodec<KeyT>::ToStorage(key));
  }

  // Returns iterator to key if found, or end() otherwise.
  iterator find(KeyT key) {
    return FindInternal(KeyCodec<KeyT>::ToStorage(key));
  }

  // Returns const_iterator to the first key >= target, or end() if all keys <
  // target.
  const_iterator lower_bound(KeyT target) const {
    return LowerBoundInternal(KeyCodec<KeyT>::ToStorage(target));
  }

  // Returns iterator to the first key >= target, or end() if all keys < target.
  iterator lower_bound(KeyT target) {
    return LowerBoundInternal(KeyCodec<KeyT>::ToStorage(target));
  }

  // Returns an iterator to the first key that is > target, or end() if none.
  HWY_INLINE const_iterator upper_bound(KeyT target) const {
    return UpperBoundInternal(KeyCodec<KeyT>::ToStorage(target));
  }

  // Returns an iterator to the first key that is > target, or end() if none.
  HWY_INLINE iterator upper_bound(KeyT target) {
    return UpperBoundInternal(KeyCodec<KeyT>::ToStorage(target));
  }

  // ---------------------------------------------------------------------------
  // Batch Lookups (8-Way Pipelined SIMD Prefetching)
  // ---------------------------------------------------------------------------

  // Executes multiple contains queries with 8-way software pipelining.
  template <typename FoundT>
  void ContainsBatch(const KeyT* HWY_RESTRICT queries, size_t count,
                     FoundT* HWY_RESTRICT out_found) const {
    if (count == 0) return;
    if (HWY_UNLIKELY(state_->root_ == nullptr || state_->num_elements_ == 0)) {
      std::fill_n(out_found, count, static_cast<FoundT>(0));
      return;
    }

    constexpr size_t kBatchSize = 8;
    size_t i = 0;
    for (; i + kBatchSize <= count; i += kBatchSize) {
      // Stack scratchpad holding pre-converted storage keys for this 8-way
      // micro-batch.
      StorageKeyT q[kBatchSize];
      void* curr[kBatchSize];
      for (size_t b = 0; b < kBatchSize; ++b) {
        q[b] = KeyCodec<KeyT>::ToStorage(queries[i + b]);
        curr[b] = state_->root_;
      }

      for (uint16_t lvl = state_->tree_height_; lvl > 0; --lvl) {
        for (size_t b = 0; b < kBatchSize; ++b) {
          auto* internal = static_cast<Internal*>(curr[b]);
          size_t child_idx = FindChild(internal, q[b]);
          curr[b] = internal->children[child_idx];
          hwy::Prefetch(curr[b]);
        }
      }

      for (size_t b = 0; b < kBatchSize; ++b) {
        out_found[i + b] = static_cast<FoundT>(
            LeafContains(static_cast<Leaf*>(curr[b]), q[b]));
      }
    }

    for (; i < count; ++i) {
      out_found[i] = static_cast<FoundT>(contains(queries[i]));
    }
  }

  // Executes multiple map value lookups with 8-way pipelined prefetching.
  template <typename V = mapped_type,
            typename = std::enable_if_t<Traits::kIsMap && !std::is_void_v<V>>>
  void LookupBatch(const KeyT* HWY_RESTRICT queries, size_t count,
                   V* HWY_RESTRICT out_values,
                   bool* HWY_RESTRICT out_found) const {
    if (count == 0) return;
    if (HWY_UNLIKELY(state_->root_ == nullptr || state_->num_elements_ == 0)) {
      std::fill_n(out_found, count, false);
      return;
    }

    constexpr size_t kBatchSize = 8;
    size_t i = 0;
    for (; i + kBatchSize <= count; i += kBatchSize) {
      // Stack scratchpad holding pre-converted storage keys for this 8-way
      // micro-batch.
      StorageKeyT q[kBatchSize];
      void* curr[kBatchSize];
      for (size_t b = 0; b < kBatchSize; ++b) {
        q[b] = KeyCodec<KeyT>::ToStorage(queries[i + b]);
        curr[b] = state_->root_;
      }

      for (uint16_t lvl = state_->tree_height_; lvl > 0; --lvl) {
        for (size_t b = 0; b < kBatchSize; ++b) {
          auto* internal = static_cast<Internal*>(curr[b]);
          size_t child_idx = FindChild(internal, q[b]);
          curr[b] = internal->children[child_idx];
          hwy::Prefetch(curr[b]);
        }
      }

      for (size_t b = 0; b < kBatchSize; ++b) {
        auto* leaf = static_cast<Leaf*>(curr[b]);
        size_t slot = 0;
        if (LeafContains(leaf, q[b], &slot)) {
          out_found[i + b] = true;
          out_values[i + b] = leaf->Values()[slot];
        } else {
          out_found[i + b] = false;
        }
      }
    }

    for (; i < count; ++i) {
      const V* val_ptr = FindValue(queries[i]);
      if (val_ptr != nullptr) {
        out_found[i] = true;
        out_values[i] = *val_ptr;
      } else {
        out_found[i] = false;
      }
    }
  }

  // Executes multiple find queries with 8-way pipelined prefetching.
  void FindBatch(const KeyT* HWY_RESTRICT targets, size_t num_queries,
                 const_iterator* HWY_RESTRICT results) const {
    if (HWY_UNLIKELY(state_->root_ == nullptr)) {
      for (size_t k = 0; k < num_queries; ++k) results[k] = end();
      return;
    }

    constexpr size_t kBatchSize = 8;
    size_t i = 0;
    for (; i + kBatchSize <= num_queries; i += kBatchSize) {
      // Stack scratchpad holding pre-converted storage keys for this 8-way
      // micro-batch.
      StorageKeyT q[kBatchSize];
      void* curr[kBatchSize];
      for (size_t b = 0; b < kBatchSize; ++b) {
        q[b] = KeyCodec<KeyT>::ToStorage(targets[i + b]);
        curr[b] = state_->root_;
      }

      for (uint16_t lvl = state_->tree_height_; lvl > 0; --lvl) {
        for (size_t b = 0; b < kBatchSize; ++b) {
          auto* internal = static_cast<Internal*>(curr[b]);
          size_t child_idx = FindChild(internal, q[b]);
          curr[b] = internal->children[child_idx];
          hwy::Prefetch(curr[b]);
        }
      }

      for (size_t b = 0; b < kBatchSize; ++b) {
        auto* leaf = static_cast<Leaf*>(curr[b]);
        size_t slot = 0;
        if (LeafContains(leaf, q[b], &slot)) {
          results[i + b] = const_iterator(leaf, slot, state_->last_leaf_);
        } else {
          results[i + b] = end();
        }
      }
    }

    for (; i < num_queries; ++i) {
      results[i] = find(targets[i]);
    }
  }

  // Executes multiple lower_bound queries with 8-way pipelined prefetching.
  void LowerBoundBatch(const KeyT* HWY_RESTRICT targets, size_t num_queries,
                       const_iterator* HWY_RESTRICT results) const {
    if (HWY_UNLIKELY(state_->root_ == nullptr)) {
      for (size_t k = 0; k < num_queries; ++k) results[k] = end();
      return;
    }

    constexpr size_t kBatchSize = 8;
    size_t i = 0;
    for (; i + kBatchSize <= num_queries; i += kBatchSize) {
      // Stack scratchpad holding pre-converted storage keys for this 8-way
      // micro-batch.
      StorageKeyT q[kBatchSize];
      void* curr[kBatchSize];
      for (size_t b = 0; b < kBatchSize; ++b) {
        q[b] = KeyCodec<KeyT>::ToStorage(targets[i + b]);
        curr[b] = state_->root_;
      }

      for (uint16_t lvl = state_->tree_height_; lvl > 0; --lvl) {
        for (size_t b = 0; b < kBatchSize; ++b) {
          auto* internal = static_cast<Internal*>(curr[b]);
          size_t child_idx = FindChild(internal, q[b]);
          curr[b] = internal->children[child_idx];
          hwy::Prefetch(curr[b]);
        }
      }

      for (size_t b = 0; b < kBatchSize; ++b) {
        auto* leaf = static_cast<Leaf*>(curr[b]);
        size_t slot = FindLeafSlot(leaf, q[b]);
        if (HWY_LIKELY(slot < leaf->NumKeys())) {
          results[i + b] = const_iterator(leaf, slot, state_->last_leaf_);
        } else if (HWY_LIKELY(leaf->Next() != nullptr &&
                              leaf->Next()->NumKeys() > 0)) {
          results[i + b] = const_iterator(leaf->Next(), 0, state_->last_leaf_);
        } else {
          results[i + b] = end();
        }
      }
    }

    for (; i < num_queries; ++i) {
      results[i] = lower_bound(targets[i]);
    }
  }

  // ---------------------------------------------------------------------------
  // Dynamic Mutations (Insertions & Deletions)
  // ---------------------------------------------------------------------------

  // Inserts a key into the Set. Returns pair of (iterator, bool_inserted).
  template <bool IsMap = Traits::kIsMap, typename = std::enable_if_t<!IsMap>>
  std::pair<iterator, bool> insert(KeyT key) {
    return InsertSetInternal(KeyCodec<KeyT>::ToStorage(key));
  }

  // Inserts a key-value pair into the Map. Returns pair of (iterator,
  // bool_inserted).
  template <typename V = mapped_type,
            typename = std::enable_if_t<Traits::kIsMap && !std::is_void_v<V>>>
  std::pair<iterator, bool> insert(const std::pair<KeyT, V>& kv) {
    return InsertMapInternal(KeyCodec<KeyT>::ToStorage(kv.first), kv.second,
                             /*assign_if_exists=*/false);
  }

  // Inserts a key and value into the Map. Returns pair of (iterator,
  // bool_inserted).
  template <typename V = mapped_type,
            typename = std::enable_if_t<Traits::kIsMap && !std::is_void_v<V>>>
  std::pair<iterator, bool> insert(KeyT key, const V& value) {
    return InsertMapInternal(KeyCodec<KeyT>::ToStorage(key), value,
                             /*assign_if_exists=*/false);
  }

  // Inserts a key-value pair or updates existing key's value.
  template <typename V = mapped_type,
            typename = std::enable_if_t<Traits::kIsMap && !std::is_void_v<V>>>
  std::pair<iterator, bool> insert_or_assign(KeyT key, const V& value) {
    return InsertMapInternal(KeyCodec<KeyT>::ToStorage(key), value,
                             /*assign_if_exists=*/true);
  }

  // Emplaces a key-value pair into the Map.
  template <typename V = mapped_type, typename... Args,
            typename = std::enable_if_t<Traits::kIsMap && !std::is_void_v<V>>>
  std::pair<iterator, bool> emplace(KeyT key, Args&&... args) {
    return insert_or_assign(key, V(std::forward<Args>(args)...));
  }

  // Emplaces a key into the Set.
  template <typename... Args, bool IsMap = Traits::kIsMap,
            typename = std::enable_if_t<!IsMap>>
  std::pair<iterator, bool> emplace(Args&&... args) {
    return insert(KeyT(std::forward<Args>(args)...));
  }

  // Subscript operator for Map: inserts default value if key not present.
  template <typename V = mapped_type,
            typename = std::enable_if_t<Traits::kIsMap && !std::is_void_v<V>>>
  V& operator[](KeyT key) {
    auto res = insert(key, V{});
    return res.first.leaf()->Values()[res.first.slot()];
  }

  // Bounds-checked value accessor (const).
  template <typename V = mapped_type,
            typename = std::enable_if_t<Traits::kIsMap && !std::is_void_v<V>>>
  const V& at(KeyT key) const {
    auto it = find(key);
    if (it == end()) {
      HWY_ABORT("BTreeMap::at: key not found");
    }
    return it.leaf()->Values()[it.slot()];
  }

  // Bounds-checked value accessor (mutable).
  template <typename V = mapped_type,
            typename = std::enable_if_t<Traits::kIsMap && !std::is_void_v<V>>>
  V& at(KeyT key) {
    auto it = find(key);
    if (it == end()) {
      HWY_ABORT("BTreeMap::at: key not found");
    }
    return it.leaf()->Values()[it.slot()];
  }

  // Fast direct value pointer lookup without iterator construction (const).
  template <typename V = mapped_type,
            typename = std::enable_if_t<Traits::kIsMap && !std::is_void_v<V>>>
  const V* FindValue(KeyT key) const {
    return FindValueInternal(KeyCodec<KeyT>::ToStorage(key));
  }

  // Fast direct value pointer lookup without iterator construction (mutable).
  template <typename V = mapped_type,
            typename = std::enable_if_t<Traits::kIsMap && !std::is_void_v<V>>>
  V* FindValue(KeyT key) {
    return FindValueInternal(KeyCodec<KeyT>::ToStorage(key));
  }

  // Erases a key from the tree. Returns 1 if erased, 0 if not found.
  size_t erase(KeyT key) {
    return EraseInternal(KeyCodec<KeyT>::ToStorage(key));
  }

  // ---------------------------------------------------------------------------
  // Capacity & Iteration
  // ---------------------------------------------------------------------------

  void clear() {
    DestroySubtree(state_->root_, state_->tree_height_);
    state_->root_ = nullptr;
    state_->first_leaf_ = nullptr;
    state_->last_leaf_ = nullptr;
    state_->tree_height_ = 0;
    state_->num_elements_ = 0;
    state_->num_leaves_ = 0;
    state_->num_internals_ = 0;
  }

  size_t size() const { return state_->num_elements_; }
  bool empty() const { return state_->num_elements_ == 0; }
  size_t height() const { return state_->tree_height_; }
  size_t AllocatedBytes() const {
    return state_->num_leaves_ * sizeof(Leaf) +
           state_->num_internals_ * sizeof(Internal);
  }

  iterator begin() {
    return iterator(state_->first_leaf_, 0, state_->last_leaf_);
  }
  iterator end() { return iterator(nullptr, 0, state_->last_leaf_); }
  const_iterator begin() const {
    return const_iterator(state_->first_leaf_, 0, state_->last_leaf_);
  }
  const_iterator end() const {
    return const_iterator(nullptr, 0, state_->last_leaf_);
  }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

  reverse_iterator rbegin() { return reverse_iterator(end()); }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }
  const_reverse_iterator crbegin() const {
    return const_reverse_iterator(cend());
  }
  const_reverse_iterator crend() const {
    return const_reverse_iterator(cbegin());
  }

  // ---------------------------------------------------------------------------
  // Observers
  // ---------------------------------------------------------------------------

  key_compare key_comp() const noexcept { return key_compare(); }
  value_compare value_comp() const noexcept { return value_compare(); }
  allocator_type get_allocator() const noexcept { return allocator_type(); }

 private:
  bool ContainsInternal(StorageKeyT key) const {
    if (HWY_UNLIKELY(state_->root_ == nullptr)) return false;
    void* curr = state_->root_;
    for (uint16_t lvl = state_->tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      size_t child_idx = FindChild(internal, key);
      curr = internal->children[child_idx];
    }
    return LeafContains(static_cast<Leaf*>(curr), key);
  }

  const_iterator FindInternal(StorageKeyT key) const {
    if (HWY_UNLIKELY(state_->root_ == nullptr)) return end();
    void* curr = state_->root_;
    for (uint16_t lvl = state_->tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      size_t child_idx = FindChild(internal, key);
      curr = internal->children[child_idx];
    }
    auto* leaf = static_cast<Leaf*>(curr);
    size_t slot = 0;
    if (LeafContains(leaf, key, &slot)) {
      return const_iterator(leaf, slot, state_->last_leaf_);
    }
    return end();
  }

  iterator FindInternal(StorageKeyT key) {
    if (HWY_UNLIKELY(state_->root_ == nullptr)) return end();
    void* curr = state_->root_;
    for (uint16_t lvl = state_->tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      size_t child_idx = FindChild(internal, key);
      curr = internal->children[child_idx];
    }
    auto* leaf = static_cast<Leaf*>(curr);
    size_t slot = 0;
    if (LeafContains(leaf, key, &slot)) {
      return iterator(leaf, slot, state_->last_leaf_);
    }
    return end();
  }

  const_iterator LowerBoundInternal(StorageKeyT target) const {
    if (HWY_UNLIKELY(state_->root_ == nullptr)) return end();
    void* curr = state_->root_;
    for (uint16_t lvl = state_->tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      size_t child_idx = FindChild(internal, target);
      curr = internal->children[child_idx];
    }
    auto* leaf = static_cast<Leaf*>(curr);
    size_t slot = FindLeafSlot(leaf, target);
    if (HWY_LIKELY(slot < leaf->NumKeys())) {
      return const_iterator(leaf, slot, state_->last_leaf_);
    }
    if (HWY_LIKELY(leaf->Next() != nullptr && leaf->Next()->NumKeys() > 0)) {
      return const_iterator(leaf->Next(), 0, state_->last_leaf_);
    }
    return end();
  }

  iterator LowerBoundInternal(StorageKeyT target) {
    if (HWY_UNLIKELY(state_->root_ == nullptr)) return end();
    void* curr = state_->root_;
    for (uint16_t lvl = state_->tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      size_t child_idx = FindChild(internal, target);
      curr = internal->children[child_idx];
    }
    auto* leaf = static_cast<Leaf*>(curr);
    size_t slot = FindLeafSlot(leaf, target);
    if (HWY_LIKELY(slot < leaf->NumKeys())) {
      return iterator(leaf, slot, state_->last_leaf_);
    }
    if (HWY_LIKELY(leaf->Next() != nullptr && leaf->Next()->NumKeys() > 0)) {
      return iterator(leaf->Next(), 0, state_->last_leaf_);
    }
    return end();
  }

  const_iterator UpperBoundInternal(StorageKeyT target) const {
    if (HWY_UNLIKELY(state_->root_ == nullptr)) return end();
    void* curr = state_->root_;
    for (uint16_t lvl = state_->tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      size_t child_idx = FindChild(internal, target);
      curr = internal->children[child_idx];
    }
    auto* leaf = static_cast<Leaf*>(curr);
    size_t slot = FindLeafSlot<BoundMode::kUpperBound>(leaf, target);
    if (slot < leaf->NumKeys()) {
      return const_iterator(leaf, slot, state_->last_leaf_);
    }
    if (leaf->Next() != nullptr && leaf->Next()->NumKeys() > 0) {
      return const_iterator(leaf->Next(), 0, state_->last_leaf_);
    }
    return end();
  }

  iterator UpperBoundInternal(StorageKeyT target) {
    if (HWY_UNLIKELY(state_->root_ == nullptr)) return end();
    void* curr = state_->root_;
    for (uint16_t lvl = state_->tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      size_t child_idx = FindChild(internal, target);
      curr = internal->children[child_idx];
    }
    auto* leaf = static_cast<Leaf*>(curr);
    size_t slot = FindLeafSlot<BoundMode::kUpperBound>(leaf, target);
    if (slot < leaf->NumKeys()) {
      return iterator(leaf, slot, state_->last_leaf_);
    }
    if (leaf->Next() != nullptr && leaf->Next()->NumKeys() > 0) {
      return iterator(leaf->Next(), 0, state_->last_leaf_);
    }
    return end();
  }

  template <typename V = mapped_type,
            typename = std::enable_if_t<Traits::kIsMap && !std::is_void_v<V>>>
  const V* FindValueInternal(StorageKeyT key) const {
    if (HWY_UNLIKELY(state_->root_ == nullptr)) return nullptr;
    void* curr = state_->root_;
    for (uint16_t lvl = state_->tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      size_t child_idx = FindChild(internal, key);
      curr = internal->children[child_idx];
    }
    auto* leaf = static_cast<Leaf*>(curr);
    size_t slot = 0;
    if (LeafContains(leaf, key, &slot)) {
      return leaf->Values() + slot;
    }
    return nullptr;
  }

  template <typename V = mapped_type,
            typename = std::enable_if_t<Traits::kIsMap && !std::is_void_v<V>>>
  V* FindValueInternal(StorageKeyT key) {
    if (HWY_UNLIKELY(state_->root_ == nullptr)) return nullptr;
    void* curr = state_->root_;
    for (uint16_t lvl = state_->tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      size_t child_idx = FindChild(internal, key);
      curr = internal->children[child_idx];
    }
    auto* leaf = static_cast<Leaf*>(curr);
    size_t slot = 0;
    if (LeafContains(leaf, key, &slot)) {
      return leaf->Values() + slot;
    }
    return nullptr;
  }

  size_t EraseInternal(StorageKeyT key) {
    if (HWY_UNLIKELY(state_->root_ == nullptr || state_->num_elements_ == 0)) {
      return 0;
    }

    // Handle single-node tree (height == 0)
    if (state_->tree_height_ == 0) {
      auto* leaf = static_cast<Leaf*>(state_->root_);
      size_t slot = 0;
      // Check if key exists in leaf
      if (!LeafContains(leaf, key, &slot)) return 0;

      // In-place fast erase from leaf
      TryFastEraseFromLeaf(leaf, slot);
      state_->num_elements_--;
      if (HWY_UNLIKELY(leaf->NumKeys() == 0)) {
        delete leaf;
        state_->root_ = nullptr;
        state_->first_leaf_ = nullptr;
        state_->last_leaf_ = nullptr;
        state_->num_leaves_ = 0;
      }
      return 1;
    }

    // Multi-level tree: Record descent path from root to target leaf
    // (ancestors are saved on stack to propagate parent splits without
    // recursion).
    Internal* path[kMaxTreeHeight];
    size_t child_indices[kMaxTreeHeight];
    void* curr = state_->root_;
    for (uint16_t lvl = state_->tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      path[lvl] = internal;
      size_t child_idx = FindChild(internal, key);
      child_indices[lvl] = child_idx;
      curr = internal->children[child_idx];
    }

    auto* leaf = static_cast<Leaf*>(curr);
    size_t slot = 0;
    // Check if key exists in leaf
    if (!LeafContains(leaf, key, &slot)) return 0;

    // In-place fast erase from leaf
    TryFastEraseFromLeaf(leaf, slot);
    state_->num_elements_--;

    // Underflow Handling: If leaf has <= Leaf::kMax16 / 2 keys, attempt merge
    // with adjacent siblings
    Internal* parent = path[1];
    size_t c_idx = child_indices[1];

    if (HWY_UNLIKELY(leaf->NumKeys() <= Leaf::kMax16 / 2)) {
      // Determine merge index: right sibling (c_idx) or left sibling (c_idx -
      // 1)
      const size_t merge_idx =
          (c_idx + 1 <= parent->num_keys)
              ? c_idx
              : (c_idx > 0 ? c_idx - 1 : static_cast<size_t>(-1));

      if (merge_idx != static_cast<size_t>(-1)) {
        auto* l_leaf = static_cast<Leaf*>(parent->children[merge_idx]);
        auto* r_leaf = static_cast<Leaf*>(parent->children[merge_idx + 1]);

        if (CanMergeLeaves(l_leaf, r_leaf)) {
          MergeLeaves(l_leaf, r_leaf, state_->last_leaf_);
          state_->num_leaves_--;

          // Remove separator key and child pointer from parent
          std::memmove(
              parent->keys + merge_idx, parent->keys + merge_idx + 1,
              (parent->num_keys - 1 - merge_idx) * sizeof(StorageKeyT));
          std::memmove(parent->children + merge_idx + 1,
                       parent->children + merge_idx + 2,
                       (parent->num_keys - 1 - merge_idx) * sizeof(void*));
          parent->num_keys--;
          parent->keys[parent->num_keys] =
              std::numeric_limits<StorageKeyT>::max();
          parent->children[parent->num_keys + 1] = nullptr;

          // If root internal node becomes empty, shrink tree height to 0
          if (parent->num_keys == 0 && parent == state_->root_ &&
              state_->tree_height_ == 1) {
            delete parent;
            state_->num_internals_--;
            state_->root_ = l_leaf;
            state_->tree_height_ = 0;
          }
        }
      }
    }

    return 1;
  }
  std::pair<iterator, bool> InsertSetInternal(StorageKeyT key) {
    // Handle empty tree initialization
    if (HWY_UNLIKELY(state_->root_ == nullptr)) {
      state_->first_leaf_ = state_->last_leaf_ = new Leaf();
      state_->num_leaves_ = 1;
      CompressIntoLeaf(state_->first_leaf_, &key, 1);
      state_->root_ = state_->first_leaf_;
      state_->tree_height_ = 0;
      state_->num_elements_ = 1;
      return {iterator(state_->first_leaf_, 0, state_->last_leaf_), true};
    }

    // Root leaf handling (height == 0)
    if (state_->tree_height_ == 0) {
      auto* leaf = static_cast<Leaf*>(state_->root_);
      size_t slot = 0;
      // Check for duplicate key (set semantics)
      if (LeafContains(leaf, key, &slot)) {
        return {iterator(leaf, slot, state_->last_leaf_), false};
      }

      // Tier 1: Fast-path in-place insert without recompression
      if (HWY_LIKELY(TryFastInsertIntoLeaf<Traits>(leaf, key, slot))) {
        state_->num_elements_++;
        return {iterator(leaf, slot, state_->last_leaf_), true};
      }

      // Tier 2: Slow-path mode widening (upgrades compression mode if leaf has
      // capacity)
      if (CanLeafFitInsert(leaf, key)) {
        InsertIntoLeaf(leaf, key);
        state_->num_elements_++;
        return {FindInternal(key), true};
      }

      // Tier 3: Root leaf split (allocates new leaf and creates root internal
      // node)
      auto* new_leaf = new Leaf();
      state_->num_leaves_++;
      StorageKeyT promo_key = 0;
      SplitLeafNode(leaf, new_leaf, key, &promo_key);

      new_leaf->SetNext(leaf->Next());
      new_leaf->SetPrev(leaf);
      if (leaf->Next() != nullptr) {
        leaf->Next()->SetPrev(new_leaf);
      }
      leaf->SetNext(new_leaf);
      state_->last_leaf_ = new_leaf;

      auto* new_root = new Internal();
      state_->num_internals_++;
      new_root->keys[0] = promo_key;
      new_root->children[0] = leaf;
      new_root->children[1] = new_leaf;
      new_root->num_keys = 1;
      state_->root_ = new_root;
      state_->tree_height_ = 1;
      state_->num_elements_++;

      return {FindInternal(key), true};
    }

    // General case (height >= 1): Record descent path from root to target
    // leaf (ancestors are saved on stack to propagate parent splits without
    // recursion).
    Internal* path[kMaxTreeHeight];
    size_t child_indices[kMaxTreeHeight];
    void* curr = state_->root_;
    for (uint16_t lvl = state_->tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      path[lvl] = internal;
      size_t child_idx = FindChild(internal, key);
      child_indices[lvl] = child_idx;
      curr = internal->children[child_idx];
    }

    auto* leaf = static_cast<Leaf*>(curr);
    size_t slot = 0;
    // Check for duplicate key (set semantics)
    if (LeafContains(leaf, key, &slot)) {
      return {iterator(leaf, slot, state_->last_leaf_), false};
    }

    // Tier 1: Fast-path in-place insert without recompression
    if (HWY_LIKELY(TryFastInsertIntoLeaf<Traits>(leaf, key, slot))) {
      state_->num_elements_++;
      return {iterator(leaf, slot, state_->last_leaf_), true};
    }

    // Tier 2: Slow-path mode widening (upgrades compression mode if leaf has
    // capacity)
    if (CanLeafFitInsert(leaf, key)) {
      InsertIntoLeaf(leaf, key);
      state_->num_elements_++;
      return {FindInternal(key), true};
    }

    // Tier 3: Leaf split (leaf is full; allocates new leaf and divides keys
    // 50/50)
    auto* new_leaf = new Leaf();
    state_->num_leaves_++;
    StorageKeyT promo_key = 0;
    SplitLeafNode(leaf, new_leaf, key, &promo_key);

    new_leaf->SetNext(leaf->Next());
    new_leaf->SetPrev(leaf);
    if (leaf->Next() != nullptr) {
      leaf->Next()->SetPrev(new_leaf);
    } else {
      state_->last_leaf_ = new_leaf;
    }
    leaf->SetNext(new_leaf);
    state_->num_elements_++;

    // Propagate separator keys and splits up ancestor internal levels
    void* promo_child = new_leaf;
    for (uint16_t lvl = 1; lvl <= state_->tree_height_; ++lvl) {
      Internal* parent = path[lvl];
      // Case A: Parent has room (num_keys < 16).
      // Shift keys and children right of c_idx to insert the new entry.
      if (HWY_LIKELY(parent->num_keys < Internal::kCapacity)) {
        size_t c_idx = child_indices[lvl];
        for (size_t k = parent->num_keys; k > c_idx; --k) {
          parent->keys[k] = parent->keys[k - 1];
          parent->children[k + 1] = parent->children[k];
        }
        parent->keys[c_idx] = promo_key;
        parent->children[c_idx + 1] = promo_child;
        parent->num_keys++;
        return {FindInternal(key), true};
      }

      // Case B: Parent is full (16 keys, 17 children) -> Internal node split!
      auto* new_internal = new Internal();
      state_->num_internals_++;

      // Assemble all 17 keys and 18 children in sorted order on the stack.
      constexpr size_t kTotalK = Internal::kCapacity + 1;
      StorageKeyT temp_keys[kTotalK];
      void* temp_children[kTotalK + 1];
      size_t c_idx = child_indices[lvl];

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

      // Promote the middle key (index 8) to the next ancestor level.
      constexpr size_t kMid = kTotalK / 2;
      promo_key = temp_keys[kMid];
      promo_child = new_internal;

      // Left node (parent) keeps 8 keys and 9 children.
      std::copy_n(temp_keys, kMid, parent->keys);
      std::copy_n(temp_children, kMid + 1, parent->children);
      parent->num_keys = static_cast<uint8_t>(kMid);
      std::fill_n(parent->keys + kMid, Internal::kCapacity - kMid,
                  std::numeric_limits<StorageKeyT>::max());

      // Right node (new_internal) gets 8 keys and 9 children.
      const size_t right_k = kTotalK - kMid - 1;
      std::copy_n(temp_keys + kMid + 1, right_k, new_internal->keys);
      std::copy_n(temp_children + kMid + 1, right_k + 1,
                  new_internal->children);
      new_internal->num_keys = static_cast<uint8_t>(right_k);
      std::fill_n(new_internal->keys + right_k, Internal::kCapacity - right_k,
                  std::numeric_limits<StorageKeyT>::max());
    }

    // Root split (grows tree height by 1)
    auto* new_root = new Internal();
    state_->num_internals_++;
    new_root->keys[0] = promo_key;
    new_root->children[0] = state_->root_;
    new_root->children[1] = promo_child;
    new_root->num_keys = 1;
    state_->root_ = new_root;
    state_->tree_height_++;

    return {FindInternal(key), true};
  }

  template <typename V>
  std::pair<iterator, bool> InsertMapInternal(StorageKeyT key, const V& value,
                                              bool assign_if_exists) {
    // Handle empty tree initialization
    if (HWY_UNLIKELY(state_->root_ == nullptr)) {
      state_->first_leaf_ = state_->last_leaf_ = new Leaf();
      state_->num_leaves_ = 1;
      CompressIntoLeaf(state_->first_leaf_, &key, &value, 1);
      state_->root_ = state_->first_leaf_;
      state_->tree_height_ = 0;
      state_->num_elements_ = 1;
      return {iterator(state_->first_leaf_, 0, state_->last_leaf_), true};
    }

    // Root leaf handling (height == 0)
    if (state_->tree_height_ == 0) {
      auto* leaf = static_cast<Leaf*>(state_->root_);
      size_t slot = 0;
      // Check for existing key (map assign semantics)
      if (LeafContains(leaf, key, &slot)) {
        if (assign_if_exists) {
          leaf->Values()[slot] = value;
        }
        return {iterator(leaf, slot, state_->last_leaf_), false};
      }

      // Tier 1: Fast-path in-place insert without recompression
      if (HWY_LIKELY(TryFastInsertIntoLeaf<Traits>(leaf, key, value, slot))) {
        state_->num_elements_++;
        return {iterator(leaf, slot, state_->last_leaf_), true};
      }

      // Tier 2: Slow-path mode widening (upgrades compression mode if leaf has
      // capacity)
      if (CanLeafFitInsert(leaf, key)) {
        InsertIntoLeaf(leaf, key, value);
        state_->num_elements_++;
        return {FindInternal(key), true};
      }

      // Tier 3: Root leaf split (allocates new leaf and creates root internal
      // node)
      auto* new_leaf = new Leaf();
      state_->num_leaves_++;
      StorageKeyT promo_key = 0;
      SplitLeafNode(leaf, new_leaf, key, value, &promo_key);

      new_leaf->SetNext(leaf->Next());
      new_leaf->SetPrev(leaf);
      if (leaf->Next() != nullptr) {
        leaf->Next()->SetPrev(new_leaf);
      }
      leaf->SetNext(new_leaf);
      state_->last_leaf_ = new_leaf;

      auto* new_root = new Internal();
      state_->num_internals_++;
      new_root->keys[0] = promo_key;
      new_root->children[0] = leaf;
      new_root->children[1] = new_leaf;
      new_root->num_keys = 1;
      state_->root_ = new_root;
      state_->tree_height_ = 1;
      state_->num_elements_++;

      return {FindInternal(key), true};
    }

    // General case (height >= 1): Record descent path from root to target
    // leaf (ancestors are saved on stack to propagate parent splits without
    // recursion).
    Internal* path[kMaxTreeHeight];
    size_t child_indices[kMaxTreeHeight];
    void* curr = state_->root_;
    for (uint16_t lvl = state_->tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      path[lvl] = internal;
      size_t child_idx = FindChild(internal, key);
      child_indices[lvl] = child_idx;
      curr = internal->children[child_idx];
    }

    auto* leaf = static_cast<Leaf*>(curr);
    size_t slot = 0;
    // Check for existing key (map assign semantics)
    if (LeafContains(leaf, key, &slot)) {
      if (assign_if_exists) {
        leaf->Values()[slot] = value;
      }
      return {iterator(leaf, slot, state_->last_leaf_), false};
    }

    // Tier 1: Fast-path in-place insert without recompression
    if (HWY_LIKELY(TryFastInsertIntoLeaf<Traits>(leaf, key, value, slot))) {
      state_->num_elements_++;
      return {iterator(leaf, slot, state_->last_leaf_), true};
    }

    // Tier 2: Slow-path mode widening (upgrades compression mode if leaf has
    // capacity)
    if (CanLeafFitInsert(leaf, key)) {
      InsertIntoLeaf(leaf, key, value);
      state_->num_elements_++;
      return {FindInternal(key), true};
    }

    // Tier 3: Leaf split (leaf is full; allocates new leaf and divides
    // keys/values 50/50)
    auto* new_leaf = new Leaf();
    state_->num_leaves_++;
    StorageKeyT promo_key = 0;
    SplitLeafNode(leaf, new_leaf, key, value, &promo_key);

    new_leaf->SetNext(leaf->Next());
    new_leaf->SetPrev(leaf);
    if (leaf->Next() != nullptr) {
      leaf->Next()->SetPrev(new_leaf);
    } else {
      state_->last_leaf_ = new_leaf;
    }
    leaf->SetNext(new_leaf);
    state_->num_elements_++;

    // Propagate separator keys and splits up ancestor internal levels
    void* promo_child = new_leaf;
    for (uint16_t lvl = 1; lvl <= state_->tree_height_; ++lvl) {
      Internal* parent = path[lvl];
      // Case A: Parent has room (num_keys < 16).
      // Shift keys and children right of c_idx to insert the new entry.
      if (HWY_LIKELY(parent->num_keys < Internal::kCapacity)) {
        size_t c_idx = child_indices[lvl];
        for (size_t k = parent->num_keys; k > c_idx; --k) {
          parent->keys[k] = parent->keys[k - 1];
          parent->children[k + 1] = parent->children[k];
        }
        parent->keys[c_idx] = promo_key;
        parent->children[c_idx + 1] = promo_child;
        parent->num_keys++;
        return {FindInternal(key), true};
      }

      // Case B: Parent is full (16 keys, 17 children) -> Internal node split!
      auto* new_internal = new Internal();
      state_->num_internals_++;

      // Assemble all 17 keys and 18 children in sorted order on the stack.
      constexpr size_t kTotalK = Internal::kCapacity + 1;
      StorageKeyT temp_keys[kTotalK];
      void* temp_children[kTotalK + 1];
      size_t c_idx = child_indices[lvl];

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

      // Promote the middle key (index 8) to the next ancestor level.
      constexpr size_t kMid = kTotalK / 2;
      promo_key = temp_keys[kMid];
      promo_child = new_internal;

      // Left node (parent) keeps 8 keys and 9 children.
      std::copy_n(temp_keys, kMid, parent->keys);
      std::copy_n(temp_children, kMid + 1, parent->children);
      parent->num_keys = static_cast<uint8_t>(kMid);
      std::fill_n(parent->keys + kMid, Internal::kCapacity - kMid,
                  std::numeric_limits<StorageKeyT>::max());

      // Right node (new_internal) gets 8 keys and 9 children.
      const size_t right_k = kTotalK - kMid - 1;
      std::copy_n(temp_keys + kMid + 1, right_k, new_internal->keys);
      std::copy_n(temp_children + kMid + 1, right_k + 1,
                  new_internal->children);
      new_internal->num_keys = static_cast<uint8_t>(right_k);
      std::fill_n(new_internal->keys + right_k, Internal::kCapacity - right_k,
                  std::numeric_limits<StorageKeyT>::max());
    }

    // Root split (grows tree height by 1)
    auto* new_root = new Internal();
    state_->num_internals_++;
    new_root->keys[0] = promo_key;
    new_root->children[0] = state_->root_;
    new_root->children[1] = promo_child;
    new_root->num_keys = 1;
    state_->root_ = new_root;
    state_->tree_height_++;

    return {FindInternal(key), true};
  }

  template <typename V>
  static BTree BuildInternal(const StorageKeyT* sorted_keys,
                             const V* sorted_values, size_t num_keys,
                             float fill_ratio) {
    BTree tree;
    if (num_keys == 0) return tree;

    tree.state_->num_elements_ = num_keys;
    fill_ratio = std::clamp(fill_ratio, 0.1f, 1.0f);

    const size_t max_keys_8 =
        std::max<size_t>(1, static_cast<size_t>(Leaf::kMax8 * fill_ratio));
    const size_t max_keys_16 =
        std::max<size_t>(1, static_cast<size_t>(Leaf::kMax16 * fill_ratio));
    const size_t max_keys_32 =
        std::max<size_t>(1, static_cast<size_t>(Leaf::kMax32 * fill_ratio));
    const size_t max_keys_64 =
        std::max<size_t>(1, static_cast<size_t>(Leaf::kMax64 * fill_ratio));

    // Pointers to nodes at the current level being linked into parent internal
    // nodes.
    std::vector<void*> current_level_ptrs;
    // Separator keys between adjacent children, used to populate internal node
    // keys.
    std::vector<StorageKeyT> separators;
    Leaf* prev_leaf = nullptr;

    // Build compressed leaf level
    for (size_t key_idx = 0; key_idx < num_keys;) {
      auto* leaf = new Leaf();
      tree.state_->num_leaves_++;
      if (tree.state_->first_leaf_ == nullptr) tree.state_->first_leaf_ = leaf;
      if (prev_leaf != nullptr) {
        prev_leaf->SetNext(leaf);
        leaf->SetPrev(prev_leaf);
      }

      leaf->base_key = sorted_keys[key_idx];
      const size_t remaining = num_keys - key_idx;
      size_t count = 0;

      // Select the narrowest compression mode that can accommodate the key
      // range. Tries 8-bit, then 16-bit, then 32-bit, and falls back to raw
      // 64-bit keys.
      const size_t try8 = std::min(remaining, max_keys_8);
      if (try8 > 0 && static_cast<uint64_t>(sorted_keys[key_idx + try8 - 1] -
                                            leaf->base_key) <= 255) {
        leaf->SetBitMode(kMode8Bit);
        count = try8;
        auto* offsets = HWY_RCAST_ALIGNED(uint8_t*, leaf->KeyData());
        for (size_t k = 0; k < count; ++k) {
          offsets[k] =
              static_cast<uint8_t>(sorted_keys[key_idx + k] - leaf->base_key);
        }
        // Pad unused trailing slots with sentinels for SIMD scanning.
        std::fill_n(offsets + count, Leaf::kMax8 - count, 0xFF);
      } else if (const size_t try16 = std::min(remaining, max_keys_16);
                 try16 > 0 &&
                 static_cast<uint64_t>(sorted_keys[key_idx + try16 - 1] -
                                       leaf->base_key) <= 65535) {
        leaf->SetBitMode(kMode16Bit);
        count = try16;
        auto* offsets = HWY_RCAST_ALIGNED(uint16_t*, leaf->KeyData());
        for (size_t k = 0; k < count; ++k) {
          offsets[k] =
              static_cast<uint16_t>(sorted_keys[key_idx + k] - leaf->base_key);
        }
        // Pad unused trailing slots with sentinels for SIMD scanning.
        std::fill_n(offsets + count, Leaf::kMax16 - count, 0xFFFF);
      } else if (const size_t try32 = std::min(remaining, max_keys_32);
                 try32 > 0 &&
                 (sizeof(StorageKeyT) == 4 ||
                  static_cast<uint64_t>(sorted_keys[key_idx + try32 - 1] -
                                        leaf->base_key) <= 0xFFFFFFFFULL)) {
        leaf->SetBitMode(kMode32Bit);
        count = try32;
        if constexpr (sizeof(StorageKeyT) == 4) {
          auto* raw_keys = HWY_RCAST_ALIGNED(uint32_t*, leaf->KeyData());
          for (size_t k = 0; k < count; ++k) {
            raw_keys[k] = static_cast<uint32_t>(sorted_keys[key_idx + k]);
          }
          std::fill_n(raw_keys + count, Leaf::kMax32 - count, 0xFFFFFFFF);
        } else {
          auto* offsets = HWY_RCAST_ALIGNED(uint32_t*, leaf->KeyData());
          for (size_t k = 0; k < count; ++k) {
            offsets[k] = static_cast<uint32_t>(sorted_keys[key_idx + k] -
                                               leaf->base_key);
          }
          // Pad unused trailing slots with sentinels for SIMD scanning.
          std::fill_n(offsets + count, Leaf::kMax32 - count, 0xFFFFFFFF);
        }
      } else {
        leaf->SetBitMode(kModeRaw64);
        count = std::min(remaining, max_keys_64);
        auto* raw_keys = HWY_RCAST_ALIGNED(uint64_t*, leaf->KeyData());
        for (size_t k = 0; k < count; ++k) {
          raw_keys[k] = static_cast<uint64_t>(sorted_keys[key_idx + k]);
        }
        std::fill_n(raw_keys + count, Leaf::kMax64 - count,
                    0xFFFFFFFFFFFFFFFFULL);
      }

      if constexpr (Traits::kIsMap) {
        auto* vals = leaf->Values();
        for (size_t k = 0; k < count; ++k) {
          vals[k] = sorted_values[key_idx + k];
        }
      }

      leaf->SetNumKeys(static_cast<uint16_t>(count));
      current_level_ptrs.push_back(leaf);

      if (key_idx > 0) {
        separators.push_back(leaf->base_key);
      }

      key_idx += count;
      prev_leaf = leaf;
    }

    tree.state_->last_leaf_ = prev_leaf;

    // If single leaf, root is the leaf
    if (current_level_ptrs.size() == 1) {
      tree.state_->root_ = current_level_ptrs[0];
      tree.state_->tree_height_ = 0;
      return tree;
    }

    // Build internal levels bottom-up until a single root node remains.
    size_t current_height = 0;
    while (current_level_ptrs.size() > 1) {
      current_height++;
      std::vector<void*> next_level_ptrs;
      std::vector<StorageKeyT> next_separators;

      constexpr size_t kBranching = Internal::kMaxChildren;
      size_t num_nodes = current_level_ptrs.size();

      for (size_t idx = 0; idx < num_nodes;) {
        auto* internal = new Internal();
        tree.state_->num_internals_++;

        size_t children_count = std::min(kBranching, num_nodes - idx);
        // Link child node pointers into this internal node.
        for (size_t c = 0; c < children_count; ++c) {
          internal->children[c] = current_level_ptrs[idx + c];
          // Set separator keys (N children are separated by N - 1 keys).
          if (c > 0) {
            internal->keys[c - 1] = separators[idx + c - 1];
          }
        }
        internal->num_keys = static_cast<uint8_t>(children_count - 1);
        next_level_ptrs.push_back(internal);

        // Propagate the delimiter separating this internal node from the
        // previous one.
        if (idx > 0) {
          next_separators.push_back(separators[idx - 1]);
        }

        idx += children_count;
      }

      // Advance up to the next internal level.
      current_level_ptrs = std::move(next_level_ptrs);
      separators = std::move(next_separators);
    }

    tree.state_->root_ = current_level_ptrs[0];
    tree.state_->tree_height_ = current_height;
    return tree;
  }

  static void DestroySubtree(void* node, size_t height) {
    if (node == nullptr) return;
    if (height == 0) {
      delete static_cast<Leaf*>(node);
      return;
    }
    auto* internal = static_cast<Internal*>(node);
    for (size_t i = 0; i <= internal->num_keys; ++i) {
      DestroySubtree(internal->children[i], height - 1);
    }
    delete internal;
  }

 private:
  // Internal state storage used when BTree operates in standalone mode.
  State owned_state_;
  // Pointer to the active BTreeState. Points to &owned_state_ in standalone
  // mode, or to an externally owned state (e.g. from BTreeSet) in adapter mode.
  State* state_ = &owned_state_;
};

// -----------------------------------------------------------------------------
// Public Type Aliases
// -----------------------------------------------------------------------------

template <typename KeyT>
using BTreeSet = BTree<SetTraits<KeyT>>;

template <typename KeyT, typename ValueT>
using BTreeMap = BTree<MapTraits<KeyT, ValueT>>;

template <typename KeyT>
using btree_set = BTree<SetTraits<KeyT>>;

template <typename KeyT, typename ValueT>
using btree_map = BTree<MapTraits<KeyT, ValueT>>;

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HWY_TARGET != HWY_SCALAR
#endif  // HIGHWAY_HWY_CONTRIB_BTREE_BTREE_INL_H_
