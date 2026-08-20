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
#include <utility>
#include <vector>

#include "hwy/base.h"
#include "hwy/cache_control.h"

#if defined(HIGHWAY_HWY_CONTRIB_BTREE_COMPACT_BTREE_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_BTREE_COMPACT_BTREE_INL_H_
#undef HIGHWAY_HWY_CONTRIB_BTREE_COMPACT_BTREE_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_BTREE_COMPACT_BTREE_INL_H_
#endif

#include "hwy/highway.h"

static_assert(HWY_CXX_LANG >= 201703L, "requires C++17 or later.");

// Check that Highway target is not scalar
#if HWY_TARGET != HWY_SCALAR

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// -----------------------------------------------------------------------------
// Bit-Mode Compression Enums
// -----------------------------------------------------------------------------

// Controls how keys within a leaf node are compressed relative to base_key.
// The mode is chosen dynamically per leaf based on the spread (max_key -
// min_key).
enum CompactBitMode : uint8_t {
  kMode8Bit = 0,   // 8-bit unsigned offsets (holds up to 192 keys)
  kMode16Bit = 1,  // 16-bit unsigned offsets (holds up to 96 keys)
  kMode32Bit = 2,  // 32-bit offsets/keys (holds up to 48 keys)
  kModeRaw64 = 3,  // 64-bit raw uncompressed keys (holds up to 24 keys)
};

// -----------------------------------------------------------------------------
// Compact Node Definitions
// -----------------------------------------------------------------------------

// Leaf node storing compressed key offsets.
// Total node size is 256 bytes (4 cache lines):
// 1. Matches TCMalloc's 256-byte size-class bin with zero internal
// fragmentation.
// 2. Aligned to 64-byte cache lines: metadata/pointers occupy the first 64
// bytes,
//    and key data occupies the remaining 192 bytes (3 cache lines).
template <typename KeyT>
struct alignas(64) CompactLeafNode {
  static constexpr size_t kDataBytes = 192;  // 3 cache lines of key payload
  static constexpr size_t kMax8 = kDataBytes / sizeof(uint8_t);
  static constexpr size_t kMax16 = kDataBytes / sizeof(uint16_t);
  static constexpr size_t kMax32 = kDataBytes / sizeof(uint32_t);
  static constexpr size_t kMax64 = kDataBytes / sizeof(uint64_t);

  CompactLeafNode* next = nullptr;
  CompactLeafNode* prev = nullptr;
  KeyT base_key = 0;
  uint16_t num_keys = 0;
  uint8_t bit_mode = kMode16Bit;
  // Pad metadata to 64 bytes so data[] starts aligned to a 64-byte cache line.
  uint8_t padding[sizeof(KeyT) == 8 ? 37 : 41] = {};

  uint8_t data[kDataBytes];

  CompactLeafNode() {
    // Fill with 0xFF so unused slots hold UINT_MAX and are ignored by SIMD Lt
    // comparisons.
    std::memset(data, 0xFF, kDataBytes);
  }
};

static_assert(sizeof(CompactLeafNode<uint32_t>) == 256,
              "CompactLeafNode<uint32_t> must be exactly 256 bytes");
static_assert(sizeof(CompactLeafNode<uint64_t>) == 256,
              "CompactLeafNode<uint64_t> must be exactly 256 bytes");

// -----------------------------------------------------------------------------
// Internal Node
// -----------------------------------------------------------------------------

// Internal node storing separator keys and child pointers.
// Padded to 256 bytes (4 cache lines) to match TCMalloc's 256-byte size class.
template <typename KeyT>
struct alignas(64) CompactInternalNode {
  static constexpr size_t kCapacity = 16;
  static constexpr size_t kMaxChildren = 17;

  KeyT keys[kCapacity];
  void* children[kMaxChildren];
  uint8_t num_keys = 0;
  // Pad struct to 256 bytes.
  uint8_t padding[sizeof(KeyT) == 8 ? 23 : 55] = {};

  CompactInternalNode() {
    // Unused key slots hold the maximum value so SIMD comparisons ignore them.
    std::fill_n(keys, kCapacity, std::numeric_limits<KeyT>::max());
    std::fill_n(children, kMaxChildren, nullptr);
  }
};

static_assert(sizeof(CompactInternalNode<uint32_t>) == 256,
              "CompactInternalNode<uint32_t> must be exactly 256 bytes");
static_assert(sizeof(CompactInternalNode<uint64_t>) == 320,
              "CompactInternalNode<uint64_t> must be exactly 320 bytes");

// -----------------------------------------------------------------------------
// Key Decompression & Slot Search Primitives
// -----------------------------------------------------------------------------

// Decompresses and returns the raw key stored at the given slot index in a
// leaf.
template <typename KeyT>
HWY_INLINE KeyT GetCompactKey(const CompactLeafNode<KeyT>* HWY_RESTRICT leaf,
                              size_t slot) {
  if (slot == 0) return leaf->base_key;

  // Reinterprets the raw byte buffer leaf->data as the appropriate integer
  // array (uint8_t, uint16_t, uint32_t, or uint64_t) and adds base_key.
  if (HWY_LIKELY(leaf->bit_mode == kMode16Bit)) {
    const auto* offsets = HWY_RCAST_ALIGNED(const uint16_t*, leaf->data);
    return leaf->base_key + static_cast<KeyT>(offsets[slot]);
  } else if (leaf->bit_mode == kMode8Bit) {
    const auto* offsets = HWY_RCAST_ALIGNED(const uint8_t*, leaf->data);
    return leaf->base_key + static_cast<KeyT>(offsets[slot]);
  } else if (leaf->bit_mode == kMode32Bit) {
    if constexpr (sizeof(KeyT) == 4) {
      const auto* raw_keys = HWY_RCAST_ALIGNED(const uint32_t*, leaf->data);
      return static_cast<KeyT>(raw_keys[slot]);
    } else {
      const auto* offsets = HWY_RCAST_ALIGNED(const uint32_t*, leaf->data);
      return leaf->base_key + static_cast<KeyT>(offsets[slot]);
    }
  } else {
    const auto* raw_keys = HWY_RCAST_ALIGNED(const uint64_t*, leaf->data);
    return static_cast<KeyT>(raw_keys[slot]);
  }
}

// Given an array of compressed offsets, returns
// the number of elements strictly less than target_val (the lower_bound slot).
template <typename OffsetT, size_t kTotal>
HWY_INLINE size_t ScanCompactOffsets(const void* HWY_RESTRICT data,
                                     OffsetT target_val) {
  const auto* offsets = static_cast<const OffsetT*>(data);
  const hn::CappedTag<OffsetT, kTotal> d;
  const size_t N = hn::Lanes(d);
  const auto v_target = hn::Set(d, target_val);
  size_t count = 0;
  for (size_t i = 0; i < kTotal; i += N) {
    const auto v = hn::Load(d, offsets + i);
    count += hn::CountTrue(d, hn::Lt(v, v_target));
  }
  return count;
}

// Returns true if target_val exists in the compressed offsets array using pure
// vector SIMD equality without scalar ALU popcnt bottlenecks.
template <typename OffsetT, size_t kTotal>
HWY_INLINE bool HasCompactOffset(const void* HWY_RESTRICT data,
                                 OffsetT target_val) {
  const auto* offsets = static_cast<const OffsetT*>(data);
  const hn::CappedTag<OffsetT, kTotal> d;
  const size_t N = hn::Lanes(d);
  const auto v_target = hn::Set(d, target_val);
  auto any_match = hn::MaskFalse(d);
  for (size_t i = 0; i < kTotal; i += N) {
    const auto v = hn::Load(d, offsets + i);
    any_match = hn::Or(any_match, hn::Eq(v, v_target));
  }
  return !hn::AllFalse(d, any_match);
}

// Finds the lower_bound slot index (0..num_keys) for target within a leaf node.
template <typename KeyT>
HWY_INLINE size_t FindCompactLeafSlot(
    const CompactLeafNode<KeyT>* HWY_RESTRICT leaf, KeyT target) {
  using Node = CompactLeafNode<KeyT>;
  if (target <= leaf->base_key) return 0;

  const uint64_t delta = static_cast<uint64_t>(target - leaf->base_key);

  if (HWY_LIKELY(leaf->bit_mode == kMode16Bit)) {
    if (delta > 65535) return leaf->num_keys;
    return ScanCompactOffsets<uint16_t, Node::kMax16>(
        leaf->data, static_cast<uint16_t>(delta));
  } else if (leaf->bit_mode == kMode8Bit) {
    if (delta > 255) return leaf->num_keys;
    return ScanCompactOffsets<uint8_t, Node::kMax8>(
        leaf->data, static_cast<uint8_t>(delta));
  } else if (leaf->bit_mode == kMode32Bit) {
    if constexpr (sizeof(KeyT) == 4) {
      return ScanCompactOffsets<uint32_t, Node::kMax32>(
          leaf->data, static_cast<uint32_t>(target));
    } else {
      if (delta > 0xFFFFFFFFULL) return leaf->num_keys;
      return ScanCompactOffsets<uint32_t, Node::kMax32>(
          leaf->data, static_cast<uint32_t>(delta));
    }
  } else {
    return ScanCompactOffsets<uint64_t, Node::kMax64>(
        leaf->data, static_cast<uint64_t>(target));
  }
}

// Returns true if target exists in the leaf. If found, optionally writes the
// slot index.
template <typename KeyT>
HWY_INLINE bool CompactLeafContains(
    const CompactLeafNode<KeyT>* HWY_RESTRICT leaf, KeyT target,
    size_t* HWY_RESTRICT out_slot = nullptr) {
  using Node = CompactLeafNode<KeyT>;

  // If caller requested the exact slot index (for insert/erase/find),
  // compute the lower_bound slot via Lt comparisons.
  if (out_slot != nullptr) {
    const size_t slot = FindCompactLeafSlot(leaf, target);
    *out_slot = slot;
    if (slot >= leaf->num_keys) return false;
    return GetCompactKey(leaf, slot) == target;
  }

  // Fast-path point lookup via SIMD Eq comparisons (for
  // contains/ContainsBatch).
  if (target < leaf->base_key) return false;
  if (target == leaf->base_key) return true;

  const uint64_t delta = static_cast<uint64_t>(target - leaf->base_key);

  if (HWY_LIKELY(leaf->bit_mode == kMode16Bit)) {
    if (delta > 65535) return false;
    if (delta == 65535) {
      return GetCompactKey(leaf, leaf->num_keys - 1) == target;
    }
    return HasCompactOffset<uint16_t, Node::kMax16>(
        leaf->data, static_cast<uint16_t>(delta));
  } else if (leaf->bit_mode == kMode8Bit) {
    if (delta > 255) return false;
    if (delta == 255) {
      return GetCompactKey(leaf, leaf->num_keys - 1) == target;
    }
    return HasCompactOffset<uint8_t, Node::kMax8>(leaf->data,
                                                  static_cast<uint8_t>(delta));
  } else if (leaf->bit_mode == kMode32Bit) {
    if constexpr (sizeof(KeyT) == 4) {
      if (target == 0xFFFFFFFF) {
        return GetCompactKey(leaf, leaf->num_keys - 1) == target;
      }
      return HasCompactOffset<uint32_t, Node::kMax32>(
          leaf->data, static_cast<uint32_t>(target));
    } else {
      if (delta > 0xFFFFFFFFULL) return false;
      if (delta == 0xFFFFFFFFULL) {
        return GetCompactKey(leaf, leaf->num_keys - 1) == target;
      }
      return HasCompactOffset<uint32_t, Node::kMax32>(
          leaf->data, static_cast<uint32_t>(delta));
    }
  } else {
    if (target == 0xFFFFFFFFFFFFFFFFULL) {
      return GetCompactKey(leaf, leaf->num_keys - 1) == target;
    }
    return HasCompactOffset<uint64_t, Node::kMax64>(
        leaf->data, static_cast<uint64_t>(target));
  }
}

// Scans an internal node's separator keys and returns the child pointer index
// to descend.
template <typename KeyT>
HWY_INLINE size_t FindCompactChild(
    const CompactInternalNode<KeyT>* HWY_RESTRICT internal, KeyT target) {
  constexpr size_t kCapacity = CompactInternalNode<KeyT>::kCapacity;
  const hn::CappedTag<KeyT, kCapacity> d;
  const size_t N = hn::Lanes(d);
  const auto v_target = hn::Set(d, target);

  size_t count = 0;
  for (size_t i = 0; i < kCapacity; i += N) {
    const auto v_keys = hn::Load(d, internal->keys + i);
    count += hn::CountTrue(d, hn::Le(v_keys, v_target));
  }
  return std::min<size_t>(count, internal->num_keys);
}

// -----------------------------------------------------------------------------
// Leaf-Level Dynamic Mutation & Compression Helpers
// -----------------------------------------------------------------------------

// Decompresses all keys from a leaf into the out_keys destination array.
template <typename KeyT>
HWY_INLINE void DecompressLeaf(const CompactLeafNode<KeyT>* leaf,
                               KeyT* out_keys) {
  for (size_t i = 0; i < leaf->num_keys; ++i) {
    out_keys[i] = GetCompactKey(leaf, i);
  }
}

// Encodes sorted keys as delta offsets from base_key into the destination
// buffer.
template <typename OffsetT, OffsetT kSentinel, typename KeyT>
HWY_INLINE void StoreCompressedOffsets(void* dst_data, const KeyT* keys,
                                       size_t count, KeyT base_key) {
  constexpr size_t kCapacity =
      CompactLeafNode<KeyT>::kDataBytes / sizeof(OffsetT);
  auto* dst = HWY_RCAST_ALIGNED(OffsetT*, dst_data);
  for (size_t k = 0; k < count; ++k) {
    dst[k] = static_cast<OffsetT>(keys[k] - base_key);
  }
  std::fill_n(dst + count, kCapacity - count, kSentinel);
}

// Encodes a sorted key array into a leaf node using the narrowest viable
// compression mode.
template <typename KeyT>
HWY_INLINE void CompressIntoLeaf(CompactLeafNode<KeyT>* leaf, const KeyT* keys,
                                 size_t count) {
  if (count == 0) {
    leaf->num_keys = 0;
    leaf->base_key = 0;
    std::memset(leaf->data, 0xFF, CompactLeafNode<KeyT>::kDataBytes);
    return;
  }

  leaf->base_key = keys[0];

  constexpr size_t kMax8 = CompactLeafNode<KeyT>::kDataBytes;
  constexpr size_t kMax16 = CompactLeafNode<KeyT>::kDataBytes / 2;
  constexpr size_t kMax32 = CompactLeafNode<KeyT>::kDataBytes / 4;
  constexpr size_t kMax64 = CompactLeafNode<KeyT>::kDataBytes / 8;

  const uint64_t max_delta =
      (count > 1) ? static_cast<uint64_t>(keys[count - 1] - keys[0]) : 0;

  if (count <= kMax8 && max_delta <= 255) {
    leaf->bit_mode = kMode8Bit;
    StoreCompressedOffsets<uint8_t, 0xFF>(leaf->data, keys, count,
                                          leaf->base_key);
  } else if (count <= kMax16 && max_delta <= 65535) {
    leaf->bit_mode = kMode16Bit;
    StoreCompressedOffsets<uint16_t, 0xFFFF>(leaf->data, keys, count,
                                             leaf->base_key);
  } else if (count <= kMax32 &&
             (sizeof(KeyT) == 4 || max_delta <= 0xFFFFFFFFULL)) {
    leaf->bit_mode = kMode32Bit;
    if constexpr (sizeof(KeyT) == 4) {
      auto* dst = HWY_RCAST_ALIGNED(uint32_t*, leaf->data);
      for (size_t k = 0; k < count; ++k) {
        dst[k] = static_cast<uint32_t>(keys[k]);
      }
      std::fill_n(dst + count, kMax32 - count, 0xFFFFFFFF);
    } else {
      StoreCompressedOffsets<uint32_t, 0xFFFFFFFF>(leaf->data, keys, count,
                                                   leaf->base_key);
    }
  } else {
    leaf->bit_mode = kModeRaw64;
    auto* dst = HWY_RCAST_ALIGNED(uint64_t*, leaf->data);
    for (size_t k = 0; k < count; ++k) {
      dst[k] = static_cast<uint64_t>(keys[k]);
    }
    std::fill_n(dst + count, kMax64 - count, 0xFFFFFFFFFFFFFFFFULL);
  }
  leaf->num_keys = static_cast<uint8_t>(count);
}

// Returns true if a new key can fit into the leaf (potentially upgrading its
// compression mode).
template <typename KeyT>
HWY_INLINE bool CanLeafFitInsert(const CompactLeafNode<KeyT>* leaf,
                                 KeyT new_key) {
  if (leaf->num_keys == 0) return true;
  const size_t new_count = leaf->num_keys + 1;
  const KeyT min_k = std::min(leaf->base_key, new_key);
  const KeyT max_existing = (leaf->num_keys > 0)
                                ? GetCompactKey(leaf, leaf->num_keys - 1)
                                : leaf->base_key;
  const KeyT max_k = std::max(max_existing, new_key);
  const uint64_t max_delta = static_cast<uint64_t>(max_k - min_k);

  constexpr size_t kMax8 = CompactLeafNode<KeyT>::kDataBytes;
  constexpr size_t kMax16 = CompactLeafNode<KeyT>::kDataBytes / 2;
  constexpr size_t kMax32 = CompactLeafNode<KeyT>::kDataBytes / 4;
  constexpr size_t kMax64 = CompactLeafNode<KeyT>::kDataBytes / 8;

  if (new_count <= kMax8 && max_delta <= 255) return true;
  if (new_count <= kMax16 && max_delta <= 65535) return true;
  if constexpr (sizeof(KeyT) == 4) return new_count <= kMax32;
  if (new_count <= kMax32 && max_delta <= 0xFFFFFFFFULL) return true;
  return new_count <= kMax64;
}

// Decompresses leaf keys, inserts a new key in sorted order, and returns the
// new count.
template <typename KeyT>
HWY_INLINE size_t DecompressAndInsertKey(const CompactLeafNode<KeyT>* leaf,
                                         KeyT new_key, KeyT* out_keys) {
  DecompressLeaf(leaf, out_keys);
  const size_t count = leaf->num_keys;
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

// Decompresses a leaf, inserts the new key in sorted order, and recompresses
// the leaf.
template <typename KeyT>
HWY_INLINE void InsertIntoLeaf(CompactLeafNode<KeyT>* leaf, KeyT new_key) {
  // Step 1: Temporary scratch buffer on the stack to hold decompressed keys.
  KeyT temp[256];

  // Step 2: Decompress existing keys and insert new_key in sorted order.
  const size_t new_count = DecompressAndInsertKey(leaf, new_key, temp);

  // Step 3: Recompress the sorted array back into the leaf node.
  CompressIntoLeaf(leaf, temp, new_count);
}

// In-place fast path for inserting a key into a compressed offset leaf without
// recompression.
template <typename OffsetT, uint64_t kMaxDelta, typename KeyT>
HWY_INLINE bool TryFastInsertOffset(CompactLeafNode<KeyT>* leaf, KeyT new_key,
                                    size_t slot) {
  constexpr size_t kCapacity =
      CompactLeafNode<KeyT>::kDataBytes / sizeof(OffsetT);
  if (leaf->num_keys >= kCapacity) return false;

  auto* offsets = HWY_RCAST_ALIGNED(OffsetT*, leaf->data);

  // new_key >= base_key.
  // base_key stays unchanged; compute positive offset delta from base_key.
  if (new_key >= leaf->base_key) {
    const uint64_t delta = static_cast<uint64_t>(new_key - leaf->base_key);
    if (delta <= kMaxDelta) {
      std::memmove(offsets + slot + 1, offsets + slot,
                   (kCapacity - 1 - slot) * sizeof(OffsetT));
      offsets[slot] = static_cast<OffsetT>(delta);
      leaf->num_keys++;
      return true;
    }
  } else {
    // new_key < base_key (inserted at slot 0).
    // new_key becomes the new base_key (offset 0), and all existing offsets
    // are shifted up by (old_base_key - new_key).
    const KeyT max_existing = GetCompactKey(leaf, leaf->num_keys - 1);
    const uint64_t new_span = static_cast<uint64_t>(max_existing - new_key);
    if (new_span <= kMaxDelta) {
      const OffsetT shift = static_cast<OffsetT>(leaf->base_key - new_key);
      // Constant-size move for the full 192-byte buffer payload without libc
      // length branching.
      std::memmove(offsets + 1, offsets, (kCapacity - 1) * sizeof(OffsetT));
      offsets[0] = 0;
      for (size_t i = 1; i <= leaf->num_keys; ++i) {
        offsets[i] += shift;
      }
      leaf->base_key = new_key;
      leaf->num_keys++;
      return true;
    }
  }
  return false;
}

// Fast-path dispatcher that attempts in-place insertion into a leaf without
// full recompression.
template <typename KeyT>
HWY_INLINE bool TryFastInsertIntoLeaf(CompactLeafNode<KeyT>* leaf, KeyT new_key,
                                      size_t slot) {
  if (leaf->num_keys == 0) {
    CompressIntoLeaf(leaf, &new_key, 1);
    return true;
  }

  if (HWY_LIKELY(leaf->bit_mode == kMode16Bit)) {
    return TryFastInsertOffset<uint16_t, 65535>(leaf, new_key, slot);
  } else if (leaf->bit_mode == kMode8Bit) {
    return TryFastInsertOffset<uint8_t, 255>(leaf, new_key, slot);
  } else if (leaf->bit_mode == kMode32Bit) {
    constexpr size_t kCapacity = CompactLeafNode<KeyT>::kDataBytes / 4;
    if (leaf->num_keys >= kCapacity) return false;

    if constexpr (sizeof(KeyT) == 4) {
      auto* raw_keys = HWY_RCAST_ALIGNED(uint32_t*, leaf->data);
      std::memmove(raw_keys + slot + 1, raw_keys + slot,
                   (kCapacity - 1 - slot) * sizeof(uint32_t));
      raw_keys[slot] = static_cast<uint32_t>(new_key);
      if (slot == 0) leaf->base_key = new_key;
      leaf->num_keys++;
      return true;
    } else {
      return TryFastInsertOffset<uint32_t, 0xFFFFFFFFULL>(leaf, new_key, slot);
    }
  } else {
    constexpr size_t kCapacity = CompactLeafNode<KeyT>::kDataBytes / 8;
    if (leaf->num_keys >= kCapacity) return false;
    auto* raw_keys = HWY_RCAST_ALIGNED(uint64_t*, leaf->data);
    std::memmove(raw_keys + slot + 1, raw_keys + slot,
                 (kCapacity - 1 - slot) * sizeof(uint64_t));
    raw_keys[slot] = static_cast<uint64_t>(new_key);
    if (slot == 0) leaf->base_key = new_key;
    leaf->num_keys++;
    return true;
  }

  return false;
}

// In-place fast path for erasing a key at slot from a compressed offset leaf.
template <typename OffsetT, OffsetT kSentinel, typename KeyT>
HWY_INLINE bool TryFastEraseOffset(CompactLeafNode<KeyT>* leaf, size_t slot) {
  constexpr size_t kCapacity =
      CompactLeafNode<KeyT>::kDataBytes / sizeof(OffsetT);
  auto* offsets = HWY_RCAST_ALIGNED(OffsetT*, leaf->data);
  // If erasing slot 0 (the base_key), shift all offsets and advance base_key.
  if (slot == 0) {
    const OffsetT shift = offsets[1];
    for (size_t i = 1; i < leaf->num_keys; ++i) {
      offsets[i - 1] = offsets[i] - shift;
    }
    offsets[leaf->num_keys - 1] = kSentinel;
    leaf->base_key += shift;
  } else {
    // If erasing slot > 0, shift subsequent offsets left.
    std::memmove(offsets + slot, offsets + slot + 1,
                 (kCapacity - 1 - slot) * sizeof(OffsetT));
    offsets[leaf->num_keys - 1] = kSentinel;
  }
  leaf->num_keys--;
  return true;
}

// Fast-path dispatcher that erases a key from a leaf in-place across all bit
// modes.
template <typename KeyT>
HWY_INLINE bool TryFastEraseFromLeaf(CompactLeafNode<KeyT>* leaf, size_t slot) {
  if (leaf->num_keys <= 1) {
    leaf->num_keys = 0;
    leaf->base_key = 0;
    std::memset(leaf->data, 0xFF, CompactLeafNode<KeyT>::kDataBytes);
    return true;
  }

  if (HWY_LIKELY(leaf->bit_mode == kMode16Bit)) {
    return TryFastEraseOffset<uint16_t, 0xFFFF>(leaf, slot);
  } else if (leaf->bit_mode == kMode8Bit) {
    return TryFastEraseOffset<uint8_t, 0xFF>(leaf, slot);
  } else if (leaf->bit_mode == kMode32Bit) {
    if constexpr (sizeof(KeyT) == 4) {
      constexpr size_t kCapacity = CompactLeafNode<KeyT>::kDataBytes / 4;
      auto* raw_keys = HWY_RCAST_ALIGNED(uint32_t*, leaf->data);
      std::memmove(raw_keys + slot, raw_keys + slot + 1,
                   (kCapacity - 1 - slot) * sizeof(uint32_t));
      raw_keys[leaf->num_keys - 1] = 0xFFFFFFFF;
      if (slot == 0 && leaf->num_keys > 1) {
        leaf->base_key = static_cast<KeyT>(raw_keys[0]);
      }
      leaf->num_keys--;
      return true;
    } else {
      return TryFastEraseOffset<uint32_t, 0xFFFFFFFF>(leaf, slot);
    }
  } else {
    constexpr size_t kCapacity = CompactLeafNode<KeyT>::kDataBytes / 8;
    auto* raw_keys = HWY_RCAST_ALIGNED(uint64_t*, leaf->data);
    std::memmove(raw_keys + slot, raw_keys + slot + 1,
                 (kCapacity - 1 - slot) * sizeof(uint64_t));
    raw_keys[leaf->num_keys - 1] = 0xFFFFFFFFFFFFFFFFULL;
    if (slot == 0 && leaf->num_keys > 1) {
      leaf->base_key = static_cast<KeyT>(raw_keys[0]);
    }
    leaf->num_keys--;
    return true;
  }
}

// Splits a full leaf plus a new key into two balanced leaves and returns the
// separator key.
template <typename KeyT>
HWY_INLINE void SplitLeafNode(CompactLeafNode<KeyT>* leaf,
                              CompactLeafNode<KeyT>* new_leaf, KeyT new_key,
                              KeyT* out_promo_key) {
  // Stack storage to avoid heap allocation.
  KeyT temp[257];

  // Decompress all existing keys from leaf and insert new_key in sorted
  // order.
  const size_t total = DecompressAndInsertKey(leaf, new_key, temp);

  // Find the exact midpoint (e.g., 97 total keys / 2 = 48).
  const size_t mid = total / 2;

  // Recompress left half (temp[0..47]) back into original leaf.
  CompressIntoLeaf(leaf, temp, mid);

  // Recompress right half (temp[48..96]) into the new_leaf.
  CompressIntoLeaf(new_leaf, temp + mid, total - mid);

  // The first key of right half becomes the separator key for parent
  // routing.
  *out_promo_key = new_leaf->base_key;
}

// Returns true if two adjacent leaves can merge without exceeding leaf
// capacity.
template <typename KeyT>
HWY_INLINE bool CanMergeCompactLeaves(const CompactLeafNode<KeyT>* leaf,
                                      const CompactLeafNode<KeyT>* next_leaf) {
  using Node = CompactLeafNode<KeyT>;
  if (leaf->num_keys + next_leaf->num_keys > Node::kMax16) return false;
  if (leaf->num_keys == 0 || next_leaf->num_keys == 0) return true;

  const KeyT max_key = GetCompactKey(next_leaf, next_leaf->num_keys - 1);
  const uint64_t spread = static_cast<uint64_t>(max_key - leaf->base_key);

  if (spread <= 65535) return true;
  if constexpr (sizeof(KeyT) == 4) {
    return (leaf->num_keys + next_leaf->num_keys <= Node::kMax32);
  } else {
    if (spread <= 0xFFFFFFFFULL) {
      return (leaf->num_keys + next_leaf->num_keys <= Node::kMax32);
    }
    return (leaf->num_keys + next_leaf->num_keys <= Node::kMax64);
  }
}

// Merges next_leaf into leaf, updates doubly linked pointers, and frees
// next_leaf.
template <typename KeyT>
HWY_INLINE void MergeCompactLeaves(CompactLeafNode<KeyT>* leaf,
                                   CompactLeafNode<KeyT>* next_leaf,
                                   CompactLeafNode<KeyT>*& last_leaf) {
  if (next_leaf->num_keys > 0) {
    KeyT temp_keys[256];
    DecompressLeaf(leaf, temp_keys);
    DecompressLeaf(next_leaf, temp_keys + leaf->num_keys);
    const size_t total_keys = leaf->num_keys + next_leaf->num_keys;
    CompressIntoLeaf(leaf, temp_keys, total_keys);
  }

  CompactLeafNode<KeyT>* next_next = next_leaf->next;
  leaf->next = next_next;
  if (next_next != nullptr) {
    next_next->prev = leaf;
  } else {
    last_leaf = leaf;
  }

  delete next_leaf;
}

// -----------------------------------------------------------------------------
// CompactBTreeSet
// -----------------------------------------------------------------------------

template <typename KeyT>
class CompactBTreeSet {
 public:
  using Leaf = CompactLeafNode<KeyT>;
  using Internal = CompactInternalNode<KeyT>;

  // Forward iterator providing on-the-fly key decompression as it traverses
  // across doubly linked leaf nodes.
  class const_iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = KeyT;
    using difference_type = std::ptrdiff_t;
    using pointer = const KeyT*;
    using reference = KeyT;

    const_iterator() = default;
    const_iterator(const Leaf* leaf, size_t slot,
                   const Leaf* last_leaf = nullptr)
        : leaf_(leaf), slot_(slot), last_leaf_(last_leaf) {}

    reference operator*() const { return GetCompactKey(leaf_, slot_); }

    const_iterator& operator++() {
      slot_++;
      if (slot_ >= leaf_->num_keys) {
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

    bool operator==(const const_iterator& other) const {
      return leaf_ == other.leaf_ && slot_ == other.slot_;
    }
    bool operator!=(const const_iterator& other) const {
      return !(*this == other);
    }

   private:
    const Leaf* leaf_ = nullptr;
    size_t slot_ = 0;
    const Leaf* last_leaf_ = nullptr;
  };

  using iterator = const_iterator;

  CompactBTreeSet() = default;
  ~CompactBTreeSet() { clear(); }

  CompactBTreeSet(CompactBTreeSet&& other) noexcept
      : root_(other.root_),
        first_leaf_(other.first_leaf_),
        last_leaf_(other.last_leaf_),
        tree_height_(other.tree_height_),
        num_elements_(other.num_elements_),
        num_leaves_(other.num_leaves_),
        num_internals_(other.num_internals_) {
    other.root_ = nullptr;
    other.first_leaf_ = nullptr;
    other.last_leaf_ = nullptr;
    other.tree_height_ = 0;
    other.num_elements_ = 0;
    other.num_leaves_ = 0;
    other.num_internals_ = 0;
  }

  CompactBTreeSet& operator=(CompactBTreeSet&& other) noexcept {
    if (this != &other) {
      clear();
      root_ = other.root_;
      first_leaf_ = other.first_leaf_;
      last_leaf_ = other.last_leaf_;
      tree_height_ = other.tree_height_;
      num_elements_ = other.num_elements_;
      num_leaves_ = other.num_leaves_;
      num_internals_ = other.num_internals_;
      other.root_ = nullptr;
      other.first_leaf_ = nullptr;
      other.last_leaf_ = nullptr;
      other.tree_height_ = 0;
      other.num_elements_ = 0;
      other.num_leaves_ = 0;
      other.num_internals_ = 0;
    }
    return *this;
  }

  CompactBTreeSet(const CompactBTreeSet&) = delete;
  CompactBTreeSet& operator=(const CompactBTreeSet&) = delete;

  // ---------------------------------------------------------------------------
  // Bulk Construction from Sorted Keys
  // ---------------------------------------------------------------------------

  // Constructs a CompactBTreeSet from an array of pre-sorted, unique keys in
  // O(N) time.
  //
  // Example usage:
  //   std::vector<uint32_t> sorted_keys = {10, 20, 30, 40, 50};
  //   auto tree = CompactBTreeSet<uint32_t>::Build(sorted_keys.data(),
  //   sorted_keys.size());
  //   bool found = tree.Contains(30);
  //
  // Parameters:
  // - keys: Pointer to an array of strictly ascending keys.
  // - num_keys: Total number of keys in the array.
  // - fill_ratio: Target fill factor per leaf node (between 0.0 and 1.0).
  //     * 1.0f (default): Maximum memory density and fastest throughput for
  //     read-heavy
  //       or static lookup workloads.
  //     * 0.75f - 0.85f: Leaves headroom in leaves to accommodate subsequent
  //     dynamic
  //       insertions without triggering immediate node splits.
  //
  // Returns: A fully constructed CompactBTreeSet.
  static CompactBTreeSet Build(const KeyT* keys, size_t num_keys,
                               float fill_ratio = 1.0f) {
    CompactBTreeSet tree;
    if (num_keys == 0) return tree;

    tree.num_elements_ = num_keys;
    // Pointers to nodes at the current level being linked into parent internal
    // nodes.
    std::vector<void*> current_level_ptrs;
    // Separator keys between adjacent children, used to populate internal node
    // keys.
    std::vector<KeyT> delimiters;

    size_t key_idx = 0;
    Leaf* prev_leaf = nullptr;

    const size_t max_keys_8 = std::clamp<size_t>(
        static_cast<size_t>(Leaf::kMax8 * fill_ratio), 2, Leaf::kMax8);
    const size_t max_keys_16 = std::clamp<size_t>(
        static_cast<size_t>(Leaf::kMax16 * fill_ratio), 2, Leaf::kMax16);
    const size_t max_keys_32 = std::clamp<size_t>(
        static_cast<size_t>(Leaf::kMax32 * fill_ratio), 2, Leaf::kMax32);
    const size_t max_keys_64 = std::clamp<size_t>(
        static_cast<size_t>(Leaf::kMax64 * fill_ratio), 2, Leaf::kMax64);

    // Build compressed leaf level
    while (key_idx < num_keys) {
      auto* leaf = new Leaf();
      tree.num_leaves_++;
      if (tree.first_leaf_ == nullptr) tree.first_leaf_ = leaf;
      if (prev_leaf != nullptr) {
        prev_leaf->next = leaf;
        leaf->prev = prev_leaf;
      }

      leaf->base_key = keys[key_idx];
      const size_t remaining = num_keys - key_idx;
      size_t count = 0;

      // Select the narrowest compression mode that can accommodate the key
      // range. Tries 8-bit, then 16-bit, then 32-bit, and falls back to raw
      // 64-bit keys.
      const size_t try8 = std::min(remaining, max_keys_8);
      if (try8 > 0 && static_cast<uint64_t>(keys[key_idx + try8 - 1] -
                                            leaf->base_key) <= 255) {
        leaf->bit_mode = kMode8Bit;
        count = try8;
        auto* offsets = HWY_RCAST_ALIGNED(uint8_t*, leaf->data);
        for (size_t k = 0; k < count; ++k) {
          offsets[k] = static_cast<uint8_t>(keys[key_idx + k] - leaf->base_key);
        }
        // Pad unused trailing slots with sentinels for SIMD scanning.
        std::fill_n(offsets + count, Leaf::kMax8 - count, 0xFF);
      } else if (const size_t try16 = std::min(remaining, max_keys_16);
                 try16 > 0 && static_cast<uint64_t>(keys[key_idx + try16 - 1] -
                                                    leaf->base_key) <= 65535) {
        leaf->bit_mode = kMode16Bit;
        count = try16;
        auto* offsets = HWY_RCAST_ALIGNED(uint16_t*, leaf->data);
        for (size_t k = 0; k < count; ++k) {
          offsets[k] =
              static_cast<uint16_t>(keys[key_idx + k] - leaf->base_key);
        }
        std::fill_n(offsets + count, Leaf::kMax16 - count, 0xFFFF);
      } else if (sizeof(KeyT) == 4 ||
                 (std::min(remaining, max_keys_32) > 0 &&
                  static_cast<uint64_t>(
                      keys[key_idx + std::min(remaining, max_keys_32) - 1] -
                      leaf->base_key) <= 0xFFFFFFFFULL)) {
        leaf->bit_mode = kMode32Bit;
        count = std::min(remaining, max_keys_32);
        if constexpr (sizeof(KeyT) == 4) {
          auto* raw_keys = HWY_RCAST_ALIGNED(uint32_t*, leaf->data);
          for (size_t k = 0; k < count; ++k) {
            raw_keys[k] = static_cast<uint32_t>(keys[key_idx + k]);
          }
          std::fill_n(raw_keys + count, Leaf::kMax32 - count, 0xFFFFFFFF);
        } else {
          auto* offsets = HWY_RCAST_ALIGNED(uint32_t*, leaf->data);
          for (size_t k = 0; k < count; ++k) {
            offsets[k] =
                static_cast<uint32_t>(keys[key_idx + k] - leaf->base_key);
          }
          std::fill_n(offsets + count, Leaf::kMax32 - count, 0xFFFFFFFF);
        }
      } else {
        leaf->bit_mode = kModeRaw64;
        count = std::min(remaining, max_keys_64);
        auto* raw_keys = HWY_RCAST_ALIGNED(uint64_t*, leaf->data);
        for (size_t k = 0; k < count; ++k) {
          raw_keys[k] = static_cast<uint64_t>(keys[key_idx + k]);
        }
        std::fill_n(raw_keys + count, Leaf::kMax64 - count,
                    0xFFFFFFFFFFFFFFFFULL);
      }

      leaf->num_keys = static_cast<uint8_t>(count);
      current_level_ptrs.push_back(leaf);

      if (key_idx > 0) {
        delimiters.push_back(leaf->base_key);
      }

      key_idx += count;
      prev_leaf = leaf;
    }
    tree.last_leaf_ = prev_leaf;

    // If single leaf, root is the leaf
    if (current_level_ptrs.size() == 1) {
      tree.root_ = current_level_ptrs[0];
      tree.tree_height_ = 0;
      return tree;
    }

    // Build internal levels bottom-up until a single root node remains.
    uint16_t level_height = 0;
    while (current_level_ptrs.size() > 1) {
      level_height++;
      std::vector<void*> next_level_ptrs;
      std::vector<KeyT> next_delimiters;

      constexpr size_t max_children = Internal::kMaxChildren;
      const size_t num_children = current_level_ptrs.size();
      const size_t num_internals =
          (num_children + max_children - 1) / max_children;
      next_level_ptrs.reserve(num_internals);
      if (num_internals > 1) {
        next_delimiters.reserve(num_internals - 1);
      }
      for (size_t i = 0; i < num_internals; ++i) {
        auto* internal = new Internal();
        tree.num_internals_++;
        const size_t child_start = i * max_children;
        const size_t child_count =
            std::min(max_children, num_children - child_start);

        // Link child node pointers into this internal node.
        for (size_t c = 0; c < child_count; ++c) {
          internal->children[c] = current_level_ptrs[child_start + c];
        }

        // Set separator keys (N children are separated by N - 1 keys).
        const size_t key_count = child_count - 1;
        internal->num_keys = static_cast<uint8_t>(key_count);
        for (size_t k = 0; k < key_count; ++k) {
          internal->keys[k] = delimiters[child_start + k];
        }

        // Propagate the delimiter separating this internal node from the
        // previous one.
        if (i > 0) {
          next_delimiters.push_back(delimiters[child_start - 1]);
        }

        next_level_ptrs.push_back(internal);
      }

      // Advance up to the next internal level.
      current_level_ptrs = std::move(next_level_ptrs);
      delimiters = std::move(next_delimiters);
    }

    tree.root_ = current_level_ptrs[0];
    tree.tree_height_ = level_height;
    return tree;
  }

  // ---------------------------------------------------------------------------
  // Iterators & Accessors
  // ---------------------------------------------------------------------------
  const_iterator begin() const {
    if (first_leaf_ == nullptr || num_elements_ == 0) return end();
    return const_iterator(first_leaf_, 0, last_leaf_);
  }

  const_iterator end() const { return const_iterator(nullptr, 0, last_leaf_); }

  // ---------------------------------------------------------------------------
  // Point Lookup & Range Queries
  // ---------------------------------------------------------------------------

  // Returns true if key exists in the tree, false otherwise.
  HWY_INLINE bool contains(KeyT key) const {
    if (root_ == nullptr || num_elements_ == 0) return false;
    void* curr = root_;
    for (uint16_t lvl = tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      curr = internal->children[FindCompactChild(internal, key)];
    }
    return CompactLeafContains(static_cast<Leaf*>(curr), key);
  }

  // Returns an iterator pointing to the key if found, or end() if not present.
  HWY_INLINE const_iterator find(KeyT key) const {
    if (root_ == nullptr || num_elements_ == 0) return end();
    void* curr = root_;
    for (uint16_t lvl = tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      curr = internal->children[FindCompactChild(internal, key)];
    }
    auto* leaf = static_cast<Leaf*>(curr);
    size_t slot = 0;
    if (CompactLeafContains(leaf, key, &slot)) {
      return const_iterator(leaf, slot, last_leaf_);
    }
    return end();
  }

  // Returns an iterator to the first key that is >= target, or end() if none.
  HWY_INLINE const_iterator lower_bound(KeyT target) const {
    if (root_ == nullptr || num_elements_ == 0) return end();
    void* curr = root_;
    for (uint16_t lvl = tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      size_t c_idx = FindCompactChild(internal, target);
      curr = internal->children[c_idx];
    }
    auto* leaf = static_cast<Leaf*>(curr);
    size_t slot = FindCompactLeafSlot(leaf, target);
    if (slot < leaf->num_keys) {
      return const_iterator(leaf, slot, last_leaf_);
    }
    if (leaf->next != nullptr) {
      return const_iterator(leaf->next, 0, last_leaf_);
    }
    return end();
  }

  // ---------------------------------------------------------------------------
  // 8-Way Pipelined Batch Queries with Software Prefetching
  // ---------------------------------------------------------------------------

  // Checks existence for a batch of keys in parallel with software prefetching.
  void ContainsBatch(const KeyT* queries, size_t count, bool* out_found) const {
    if (count == 0) return;
    if (root_ == nullptr || num_elements_ == 0) {
      std::fill_n(out_found, count, false);
      return;
    }

    constexpr size_t kBatchSize = 8;
    size_t i = 0;

    for (; i + kBatchSize <= count; i += kBatchSize) {
      void* curr[kBatchSize];
      for (size_t b = 0; b < kBatchSize; ++b) curr[b] = root_;

      for (uint16_t lvl = tree_height_; lvl > 0; --lvl) {
        for (size_t b = 0; b < kBatchSize; ++b) {
          auto* internal = static_cast<Internal*>(curr[b]);
          size_t c_idx = FindCompactChild(internal, queries[i + b]);
          void* next_child = internal->children[c_idx];
          hwy::Prefetch(next_child);
          curr[b] = next_child;
        }
      }

      for (size_t b = 0; b < kBatchSize; ++b) {
        out_found[i + b] =
            CompactLeafContains(static_cast<Leaf*>(curr[b]), queries[i + b]);
      }
    }

    for (; i < count; ++i) {
      out_found[i] = contains(queries[i]);
    }
  }

  // Computes lower_bound for a batch of keys in parallel with prefetching.
  void LowerBoundBatch(const KeyT* targets, size_t count,
                       const_iterator* results) const {
    if (count == 0) return;
    if (root_ == nullptr || num_elements_ == 0) {
      for (size_t i = 0; i < count; ++i) results[i] = end();
      return;
    }

    constexpr size_t kBatchSize = 8;
    size_t i = 0;

    for (; i + kBatchSize <= count; i += kBatchSize) {
      void* curr[kBatchSize];
      for (size_t b = 0; b < kBatchSize; ++b) curr[b] = root_;

      for (uint16_t lvl = tree_height_; lvl > 0; --lvl) {
        for (size_t b = 0; b < kBatchSize; ++b) {
          auto* internal = static_cast<Internal*>(curr[b]);
          size_t c_idx = FindCompactChild(internal, targets[i + b]);
          void* next_child = internal->children[c_idx];
          hwy::Prefetch(next_child);
          curr[b] = next_child;
        }
      }

      for (size_t b = 0; b < kBatchSize; ++b) {
        auto* leaf = static_cast<Leaf*>(curr[b]);
        size_t slot = FindCompactLeafSlot(leaf, targets[i + b]);
        if (slot < leaf->num_keys) {
          results[i + b] = const_iterator(leaf, slot, last_leaf_);
        } else if (leaf->next != nullptr) {
          results[i + b] = const_iterator(leaf->next, 0, last_leaf_);
        } else {
          results[i + b] = end();
        }
      }
    }

    // Remainder
    for (; i < count; ++i) {
      results[i] = lower_bound(targets[i]);
    }
  }

  // ---------------------------------------------------------------------------
  // Dynamic Mutations (Insertions & Deletions)
  // ---------------------------------------------------------------------------
  // Inserts a key into the set if not already present. Returns pair of
  // (iterator, bool_inserted).
  std::pair<const_iterator, bool> insert(KeyT key) {
    // Handle empty tree initialization
    if (root_ == nullptr) {
      first_leaf_ = last_leaf_ = new Leaf();
      num_leaves_ = 1;
      CompressIntoLeaf(first_leaf_, &key, 1);
      root_ = first_leaf_;
      tree_height_ = 0;
      num_elements_ = 1;
      return {const_iterator(first_leaf_, 0, last_leaf_), true};
    }

    // Handle single-node tree (height == 0)
    if (tree_height_ == 0) {
      auto* leaf = static_cast<Leaf*>(root_);
      size_t slot = 0;
      if (CompactLeafContains(leaf, key, &slot)) {
        return {const_iterator(leaf, slot, last_leaf_), false};
      }

      if (HWY_LIKELY(TryFastInsertIntoLeaf(leaf, key, slot))) {
        num_elements_++;
        return {const_iterator(leaf, slot, last_leaf_), true};
      }

      if (CanLeafFitInsert(leaf, key)) {
        InsertIntoLeaf(leaf, key);
        num_elements_++;
        return {find(key), true};
      }

      // Root leaf split
      auto* new_leaf = new Leaf();
      num_leaves_++;
      KeyT promo_key = 0;
      SplitLeafNode(leaf, new_leaf, key, &promo_key);

      new_leaf->next = leaf->next;
      new_leaf->prev = leaf;
      if (leaf->next != nullptr) {
        leaf->next->prev = new_leaf;
      }
      leaf->next = new_leaf;
      last_leaf_ = new_leaf;

      auto* new_root = new Internal();
      num_internals_++;
      new_root->keys[0] = promo_key;
      new_root->children[0] = leaf;
      new_root->children[1] = new_leaf;
      new_root->num_keys = 1;
      root_ = new_root;
      tree_height_ = 1;
      num_elements_++;
      return {find(key), true};
    }

    // General case (height >= 1): Record descent path from root to target
    // leaf (ancestors are saved on stack to propagate parent splits without
    // recursion).
    Internal* path[32];
    size_t child_indices[32];
    void* curr = root_;
    for (uint16_t lvl = tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      path[lvl] = internal;
      size_t c_idx = FindCompactChild(internal, key);
      child_indices[lvl] = c_idx;
      curr = internal->children[c_idx];
    }

    auto* leaf = static_cast<Leaf*>(curr);

    // Check for duplicate key (set semantics)
    size_t slot = 0;
    if (CompactLeafContains(leaf, key, &slot)) {
      return {const_iterator(leaf, slot, last_leaf_), false};
    }

    // Tier 1: Fast-path in-place insert without recompression
    if (HWY_LIKELY(TryFastInsertIntoLeaf(leaf, key, slot))) {
      num_elements_++;
      return {const_iterator(leaf, slot, last_leaf_), true};
    }

    // Tier 2: Slow-path mode widening (upgrades compression mode if leaf has
    // capacity)
    if (CanLeafFitInsert(leaf, key)) {
      InsertIntoLeaf(leaf, key);
      num_elements_++;
      return {find(key), true};
    }

    // Tier 3: Leaf split (leaf is full; allocates new leaf and divides keys
    // 50/50)
    auto* new_leaf = new Leaf();
    num_leaves_++;
    KeyT promo_key = 0;
    SplitLeafNode(leaf, new_leaf, key, &promo_key);

    new_leaf->next = leaf->next;
    new_leaf->prev = leaf;
    if (leaf->next != nullptr) {
      leaf->next->prev = new_leaf;
    } else {
      last_leaf_ = new_leaf;
    }
    leaf->next = new_leaf;
    num_elements_++;

    void* promo_child = new_leaf;

    // Propagate separator keys and splits up ancestor internal levels
    for (uint16_t lvl = 1; lvl <= tree_height_; ++lvl) {
      auto* parent = path[lvl];
      size_t c_idx = child_indices[lvl];

      // Case A: Parent has room (num_keys < 16).
      // Shift keys and children right of c_idx to insert the new entry.
      if (parent->num_keys < Internal::kCapacity) {
        for (size_t i = parent->num_keys; i > c_idx; --i) {
          parent->keys[i] = parent->keys[i - 1];
          parent->children[i + 1] = parent->children[i];
        }
        parent->keys[c_idx] = promo_key;
        parent->children[c_idx + 1] = promo_child;
        parent->num_keys++;
        return {find(key), true};
      }

      // Case B: Parent is full (16 keys, 17 children) -> Internal node split!
      auto* new_internal = new Internal();
      num_internals_++;

      // Assemble all 17 keys and 18 children in sorted order on the stack.
      constexpr size_t kTotalK = Internal::kCapacity + 1;
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

      // Promote the middle key (index 8) to the next ancestor level.
      constexpr size_t kMid = kTotalK / 2;
      promo_key = temp_keys[kMid];
      promo_child = new_internal;

      // Left node (parent) keeps 8 keys and 9 children.
      std::copy_n(temp_keys, kMid, parent->keys);
      std::copy_n(temp_children, kMid + 1, parent->children);
      parent->num_keys = static_cast<uint8_t>(kMid);
      std::fill_n(parent->keys + kMid, Internal::kCapacity - kMid,
                  std::numeric_limits<KeyT>::max());

      // Right node (new_internal) gets 8 keys and 9 children.
      const size_t right_k = kTotalK - kMid - 1;
      std::copy_n(temp_keys + kMid + 1, right_k, new_internal->keys);
      std::copy_n(temp_children + kMid + 1, right_k + 1,
                  new_internal->children);
      new_internal->num_keys = static_cast<uint8_t>(right_k);
      std::fill_n(new_internal->keys + right_k, Internal::kCapacity - right_k,
                  std::numeric_limits<KeyT>::max());
    }

    // Root split (grows tree height by 1)
    auto* new_root = new Internal();
    num_internals_++;
    new_root->keys[0] = promo_key;
    new_root->children[0] = root_;
    new_root->children[1] = promo_child;
    new_root->num_keys = 1;
    root_ = new_root;
    tree_height_++;

    return {find(key), true};
  }

  template <typename... Args>
  std::pair<const_iterator, bool> emplace(Args&&... args) {
    return insert(KeyT(std::forward<Args>(args)...));
  }

  // Erases a key from the set. Returns 1 if erased, 0 if not found.
  size_t erase(KeyT key) {
    if (root_ == nullptr || num_elements_ == 0) return 0;

    // Handle single-node tree (height == 0)
    if (tree_height_ == 0) {
      auto* leaf = static_cast<Leaf*>(root_);
      size_t slot = 0;
      if (!CompactLeafContains(leaf, key, &slot)) return 0;

      TryFastEraseFromLeaf(leaf, slot);
      num_elements_--;
      if (leaf->num_keys == 0) {
        delete leaf;
        root_ = nullptr;
        first_leaf_ = nullptr;
        last_leaf_ = nullptr;
        num_leaves_ = 0;
      }
      return 1;
    }

    // Multi-level tree: Record descent path from root to target leaf
    Internal* path[32];
    size_t child_indices[32];
    void* curr = root_;
    for (uint16_t lvl = tree_height_; lvl > 0; --lvl) {
      auto* internal = static_cast<Internal*>(curr);
      path[lvl] = internal;
      size_t c_idx = FindCompactChild(internal, key);
      child_indices[lvl] = c_idx;
      curr = internal->children[c_idx];
    }
    auto* leaf = static_cast<Leaf*>(curr);

    // Check if key exists in leaf
    size_t slot = 0;
    if (!CompactLeafContains(leaf, key, &slot)) return 0;

    // In-place fast erase from leaf
    TryFastEraseFromLeaf(leaf, slot);
    num_elements_--;

    // Underflow Handling: If leaf has <= 40 keys, attempt merge with
    // adjacent siblings
    Internal* parent = path[1];
    size_t c_idx = child_indices[1];
    if (leaf->num_keys <= 40) {
      // Determine merge index: right sibling (c_idx) or left sibling (c_idx -
      // 1)
      const size_t merge_idx =
          (c_idx + 1 <= parent->num_keys)
              ? c_idx
              : (c_idx > 0 ? c_idx - 1 : static_cast<size_t>(-1));
      if (merge_idx != static_cast<size_t>(-1)) {
        auto* left = static_cast<Leaf*>(parent->children[merge_idx]);
        auto* right = static_cast<Leaf*>(parent->children[merge_idx + 1]);
        if (CanMergeCompactLeaves(left, right)) {
          MergeCompactLeaves(left, right, last_leaf_);
          num_leaves_--;

          // Remove separator key and child pointer from parent
          std::memmove(parent->keys + merge_idx, parent->keys + merge_idx + 1,
                       (parent->num_keys - 1 - merge_idx) * sizeof(KeyT));
          std::memmove(parent->children + merge_idx + 1,
                       parent->children + merge_idx + 2,
                       (parent->num_keys - 1 - merge_idx) * sizeof(void*));
          parent->num_keys--;
          parent->keys[parent->num_keys] = std::numeric_limits<KeyT>::max();
          parent->children[parent->num_keys + 1] = nullptr;

          // If root internal node becomes empty, shrink tree height to 0
          if (parent->num_keys == 0 && parent == root_ && tree_height_ == 1) {
            delete parent;
            num_internals_--;
            root_ = left;
            tree_height_ = 0;
          }
        }
      }
    }
    return 1;
  }

  // Recursively deletes all nodes in the subtree.
  static void DestroySubtree(void* node, size_t height) {
    if (node == nullptr) return;
    if (height == 0) {
      delete static_cast<Leaf*>(node);
    } else {
      auto* internal = static_cast<Internal*>(node);
      for (size_t i = 0; i <= internal->num_keys; ++i) {
        if (internal->children[i] != nullptr) {
          DestroySubtree(internal->children[i], height - 1);
        }
      }
      delete internal;
    }
  }

  // Erases all elements and frees all tree nodes.
  void clear() {
    DestroySubtree(root_, tree_height_);
    root_ = nullptr;
    first_leaf_ = nullptr;
    last_leaf_ = nullptr;
    tree_height_ = 0;
    num_elements_ = 0;
    num_leaves_ = 0;
    num_internals_ = 0;
  }

  // Returns the number of elements in the set.
  size_t size() const { return num_elements_; }

  // Returns true if the set contains no elements.
  bool empty() const { return num_elements_ == 0; }

  // Returns the height of the tree (0 for leaf-only root).
  size_t height() const { return tree_height_; }

  // Returns the total heap memory allocated for leaf and internal nodes.
  size_t AllocatedBytes() const {
    return num_leaves_ * sizeof(Leaf) + num_internals_ * sizeof(Internal);
  }

 private:
  void* root_ = nullptr;
  Leaf* first_leaf_ = nullptr;
  Leaf* last_leaf_ = nullptr;
  size_t tree_height_ = 0;
  size_t num_elements_ = 0;
  size_t num_leaves_ = 0;
  size_t num_internals_ = 0;
};

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
using HWY_NAMESPACE::CompactBTreeSet;
}  // namespace hwy
#endif

#endif  // HWY_TARGET != HWY_SCALAR
#endif  // HIGHWAY_HWY_CONTRIB_BTREE_COMPACT_BTREE_INL_H_
