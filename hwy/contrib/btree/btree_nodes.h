// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef HIGHWAY_HWY_CONTRIB_BTREE_BTREE_NODES_H_
#define HIGHWAY_HWY_CONTRIB_BTREE_BTREE_NODES_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "hwy/base.h"

namespace hwy {

// -----------------------------------------------------------------------------
// Enums
// -----------------------------------------------------------------------------

// Controls how keys within a leaf node are compressed relative to base_key.
// The mode is chosen dynamically per leaf based on the spread (max_key -
// min_key).
enum CompactBitMode : uint8_t {
  kMode8Bit = 0,   // 8-bit unsigned offsets (holds up to 492/488 keys)
  kMode16Bit = 1,  // 16-bit unsigned offsets (holds up to 246/244 keys)
  kMode32Bit = 2,  // 32-bit offsets/keys (holds up to 123/122 keys)
  kModeRaw64 = 3,  // 64-bit raw uncompressed keys (holds up to 61 keys)
};

// Controls whether leaf slot scanning computes lower_bound (< target) or
// upper_bound (<= target).
enum class BoundMode : uint8_t {
  kLowerBound = 0,  // Finds first slot where key >= target (strict Lt scan)
  kUpperBound = 1,  // Finds first slot where key > target (Le scan)
};

// -----------------------------------------------------------------------------
// Key Codec
// -----------------------------------------------------------------------------

// Maps signed integer keys (int32_t, int64_t) to/from unsigned integer storage
// keys (uint32_t, uint64_t) via an order-preserving MSB inversion (XOR).
// For unsigned keys, this is a zero-cost compile-time identity.
template <typename KeyT>
struct KeyCodec {
  static_assert(std::is_integral_v<KeyT>,
                "Highway BTree only supports integral keys.");
  static_assert(sizeof(KeyT) == 4 || sizeof(KeyT) == 8,
                "Highway BTree only supports 32-bit and 64-bit keys.");

  using StorageKey = MakeUnsigned<KeyT>;
  // Mask for the Most Significant Bit (sign bit: 0x80000000 for 32-bit,
  // 0x8000000000000000 for 64-bit). Inverting this bit bijectively maps signed
  // two's complement integers [kMinSigned, kMaxSigned] to unsigned integers
  // [0, kMaxUnsigned] while strictly preserving relative order
  static constexpr StorageKey kSignBit = SignMask<KeyT>();

  static HWY_INLINE StorageKey ToStorage(KeyT key) {
    if constexpr (IsSigned<KeyT>()) {
      return static_cast<StorageKey>(key) ^ kSignBit;
    } else {
      return key;
    }
  }

  static HWY_INLINE KeyT FromStorage(StorageKey key) {
    if constexpr (IsSigned<KeyT>()) {
      return static_cast<KeyT>(key ^ kSignBit);
    } else {
      return key;
    }
  }
};

// -----------------------------------------------------------------------------
// Node Definitions
// -----------------------------------------------------------------------------

// Leaf node storing compressed key offsets (Set).
// Total node size is 512 bytes (8 cache lines):
// Matches TCMalloc's 512-byte size-class bin with zero internal fragmentation.
// 492-byte (32-bit) / 488-byte (64-bit) payload placed at offset 0.
// Metadata placed after payload (base_key, next_tagged, prev_tagged).
// Aligned to kNodeBytes (512 bytes), guaranteeing 9 (log2(512)) low zero bits
// in both next_tagged and prev_tagged pointers.
template <typename KeyT>
struct alignas(512) LeafNode {
  using StorageKeyT = typename KeyCodec<KeyT>::StorageKey;

  static constexpr size_t kNodeBytes = 512;
  // Bitmask for tagged pointer low bits (e.g. 512 - 1 = 0x1FF, gives 9 bits
  // for storing num_keys up to 511 without separate metadata overhead).
  static constexpr uintptr_t kNumKeysMask = kNodeBytes - 1;

  static constexpr size_t kDataBytes = (sizeof(StorageKeyT) == 4) ? 492 : 488;
  static constexpr size_t kMax8 = kDataBytes / sizeof(uint8_t);
  static constexpr size_t kMax16 = kDataBytes / sizeof(uint16_t);
  static constexpr size_t kMax32 = kDataBytes / sizeof(uint32_t);
  static constexpr size_t kMax64 = kDataBytes / sizeof(uint64_t);

  // Payload placed at offset 0
  uint8_t data[kDataBytes];

  // Metadata header placed at offset 492 (32-bit) / 488 (64-bit):
  StorageKeyT base_key = 0;
  uintptr_t next_tagged = 0;  // Low 2 bits: bit_mode (0..3)
  uintptr_t prev_tagged = 0;  // Low bits: num_keys

  LeafNode() {
    // Fill with 0xFF so unused slots hold UINT_MAX and are ignored by SIMD Lt
    // comparisons.
    std::memset(data, 0xFF, kDataBytes);
  }

  // Uniform key buffer accessor across Set and Map leaf node layouts.
  const uint8_t* KeyData() const { return data; }
  uint8_t* KeyData() { return data; }

  // Bit 0..1 of next_tagged: bit_mode (2 bits, values 0..3)
  HWY_INLINE uint8_t BitMode() const {
    return static_cast<uint8_t>(next_tagged & 0x03);
  }

  HWY_INLINE void SetBitMode(uint8_t mode) {
    next_tagged = (next_tagged & ~uintptr_t{0x03}) | (mode & 0x03);
  }

  // num_keys stored in low bits of prev_tagged.
  // Single load and mask without cross-word assembly.
  HWY_INLINE uint16_t NumKeys() const {
    return static_cast<uint16_t>(prev_tagged & kNumKeysMask);
  }

  HWY_INLINE void SetNumKeys(uint16_t n) {
    prev_tagged = (prev_tagged & ~kNumKeysMask) | (n & kNumKeysMask);
  }

  // Mask out low tag bits (kNumKeysMask) to extract aligned node pointer.
  HWY_INLINE LeafNode* Next() const {
    return reinterpret_cast<LeafNode*>(next_tagged & ~kNumKeysMask);
  }

  HWY_INLINE void SetNext(LeafNode* ptr) {
    const uintptr_t tag = next_tagged & uintptr_t{0x03};
    next_tagged = (reinterpret_cast<uintptr_t>(ptr) & ~kNumKeysMask) | tag;
  }

  HWY_INLINE LeafNode* Prev() const {
    return reinterpret_cast<LeafNode*>(prev_tagged & ~kNumKeysMask);
  }

  HWY_INLINE void SetPrev(LeafNode* ptr) {
    const uintptr_t tag = prev_tagged & kNumKeysMask;
    prev_tagged = (reinterpret_cast<uintptr_t>(ptr) & ~kNumKeysMask) | tag;
  }
};

static_assert(sizeof(LeafNode<uint32_t>) == LeafNode<uint32_t>::kNodeBytes);
static_assert(sizeof(LeafNode<int32_t>) == LeafNode<int32_t>::kNodeBytes);
static_assert(sizeof(LeafNode<uint64_t>) == LeafNode<uint64_t>::kNodeBytes);
static_assert(sizeof(LeafNode<int64_t>) == LeafNode<int64_t>::kNodeBytes);
static_assert(alignof(LeafNode<uint32_t>) == LeafNode<uint32_t>::kNodeBytes);
static_assert(alignof(LeafNode<int32_t>) == LeafNode<int32_t>::kNodeBytes);
static_assert(alignof(LeafNode<uint64_t>) == LeafNode<uint64_t>::kNodeBytes);
static_assert(alignof(LeafNode<int64_t>) == LeafNode<int64_t>::kNodeBytes);

// Leaf node storing structure-of-arrays compressed keys and values (Map).
// Aligned to 512 bytes: payload placed at offset 0.
// Metadata placed at tail (base_key, next_tagged, prev_tagged).
// Aligned to kNodeBytes (512 bytes), guaranteeing 9 (log2(512)) low zero bits
// in both next_tagged and prev_tagged pointers.
template <typename KeyT, typename ValueT>
struct alignas(512) MapLeafNode {
  using StorageKeyT = typename KeyCodec<KeyT>::StorageKey;
  using mapped_type = ValueT;

  static_assert(std::is_trivially_copyable_v<ValueT>,
                "ValueT must be trivially copyable");

  static constexpr size_t kNodeBytes = 512;
  // Bitmask for tagged pointer low bits (e.g. 512 - 1 = 0x1FF, gives 9 bits
  // for storing num_keys up to 511 without separate metadata overhead).
  static constexpr uintptr_t kNumKeysMask = kNodeBytes - 1;
  static constexpr size_t kDataBytes = (sizeof(StorageKeyT) == 4) ? 492 : 488;
  static constexpr size_t kPayloadBytes = kDataBytes;

  template <typename OffsetT>
  static constexpr size_t ValuesOffset(size_t max_pairs) {
    const size_t raw_offset = max_pairs * sizeof(OffsetT);
    const size_t align = alignof(ValueT);
    return (raw_offset + align - 1) & ~(align - 1);
  }

  template <typename OffsetT>
  static constexpr size_t ComputeMaxPairs() {
    size_t n = kPayloadBytes / (sizeof(OffsetT) + sizeof(ValueT));
    while (n > 0 &&
           (ValuesOffset<OffsetT>(n) + n * sizeof(ValueT) > kPayloadBytes)) {
      --n;
    }
    return n;
  }

  static constexpr size_t kMax8 = ComputeMaxPairs<uint8_t>();
  static constexpr size_t kMax16 = ComputeMaxPairs<uint16_t>();
  static constexpr size_t kMax32 = ComputeMaxPairs<uint32_t>();
  static constexpr size_t kMax64 = ComputeMaxPairs<uint64_t>();

  uint8_t payload[kPayloadBytes];

  StorageKeyT base_key = 0;
  uintptr_t next_tagged = 0;  // Low 2 bits: bit_mode (0..3)
  uintptr_t prev_tagged = 0;  // Low 9 bits: num_keys (0..511)

  MapLeafNode() { std::memset(payload, 0xFF, kPayloadBytes); }

  // Uniform key buffer accessor across Set and Map leaf node layouts.
  const uint8_t* KeyData() const { return payload; }
  uint8_t* KeyData() { return payload; }

  HWY_INLINE uint8_t BitMode() const {
    return static_cast<uint8_t>(next_tagged & 0x03);
  }

  HWY_INLINE void SetBitMode(uint8_t mode) {
    next_tagged = (next_tagged & ~uintptr_t{0x03}) | (mode & 0x03);
  }

  HWY_INLINE uint16_t NumKeys() const {
    return static_cast<uint16_t>(prev_tagged & kNumKeysMask);
  }

  HWY_INLINE void SetNumKeys(uint16_t count) {
    prev_tagged = (prev_tagged & ~kNumKeysMask) |
                  (static_cast<uintptr_t>(count) & kNumKeysMask);
  }

  HWY_INLINE MapLeafNode* Next() const {
    return reinterpret_cast<MapLeafNode*>(next_tagged & ~kNumKeysMask);
  }

  HWY_INLINE void SetNext(MapLeafNode* ptr) {
    const uintptr_t tag = next_tagged & uintptr_t{0x03};
    next_tagged = (reinterpret_cast<uintptr_t>(ptr) & ~kNumKeysMask) | tag;
  }

  HWY_INLINE MapLeafNode* Prev() const {
    return reinterpret_cast<MapLeafNode*>(prev_tagged & ~kNumKeysMask);
  }

  HWY_INLINE void SetPrev(MapLeafNode* ptr) {
    const uintptr_t tag = prev_tagged & kNumKeysMask;
    prev_tagged = (reinterpret_cast<uintptr_t>(ptr) & ~kNumKeysMask) | tag;
  }

  const ValueT* Values() const {
    const uint8_t mode = BitMode();
    if (HWY_LIKELY(mode == kMode16Bit)) {
      return HWY_RCAST_ALIGNED(const ValueT*,
                               payload + ValuesOffset<uint16_t>(kMax16));
    } else if (mode == kMode8Bit) {
      return HWY_RCAST_ALIGNED(const ValueT*,
                               payload + ValuesOffset<uint8_t>(kMax8));
    } else if (mode == kMode32Bit) {
      return HWY_RCAST_ALIGNED(const ValueT*,
                               payload + ValuesOffset<uint32_t>(kMax32));
    } else {
      return HWY_RCAST_ALIGNED(const ValueT*,
                               payload + ValuesOffset<uint64_t>(kMax64));
    }
  }

  ValueT* Values() {
    return const_cast<ValueT*>(const_cast<const MapLeafNode*>(this)->Values());
  }
};

static_assert(sizeof(MapLeafNode<uint32_t, uint32_t>) ==
              MapLeafNode<uint32_t, uint32_t>::kNodeBytes);
static_assert(sizeof(MapLeafNode<int32_t, uint32_t>) ==
              MapLeafNode<int32_t, uint32_t>::kNodeBytes);
static_assert(sizeof(MapLeafNode<uint32_t, uint64_t>) ==
              MapLeafNode<uint32_t, uint64_t>::kNodeBytes);
static_assert(sizeof(MapLeafNode<int32_t, uint64_t>) ==
              MapLeafNode<int32_t, uint64_t>::kNodeBytes);
static_assert(sizeof(MapLeafNode<uint64_t, uint64_t>) ==
              MapLeafNode<uint64_t, uint64_t>::kNodeBytes);
static_assert(sizeof(MapLeafNode<int64_t, uint64_t>) ==
              MapLeafNode<int64_t, uint64_t>::kNodeBytes);
static_assert(sizeof(MapLeafNode<uint64_t, uint32_t>) ==
              MapLeafNode<uint64_t, uint32_t>::kNodeBytes);
static_assert(sizeof(MapLeafNode<int64_t, uint32_t>) ==
              MapLeafNode<int64_t, uint32_t>::kNodeBytes);

// -----------------------------------------------------------------------------
// Internal Node
// -----------------------------------------------------------------------------

// Internal node storing separator keys and child pointers.
// Padded to 256 bytes (32-bit) / 320 bytes (64-bit) to match TCMalloc's size
// classes.
template <typename KeyT>
struct alignas(64) InternalNode {
  using StorageKeyT = typename KeyCodec<KeyT>::StorageKey;
  static constexpr size_t kCapacity = 16;
  static constexpr size_t kMaxChildren = 17;

  StorageKeyT keys[kCapacity];
  void* children[kMaxChildren];
  uint8_t num_keys = 0;
  // Pad struct to 256 bytes (32-bit) / 320 bytes (64-bit).
  uint8_t padding[sizeof(StorageKeyT) == 8 ? 23 : 55] = {};

  InternalNode() {
    // Unused key slots hold the maximum value so SIMD comparisons ignore them.
    std::fill_n(keys, kCapacity, std::numeric_limits<StorageKeyT>::max());
    std::fill_n(children, kMaxChildren, nullptr);
  }
};

static_assert(sizeof(InternalNode<uint32_t>) == 256,
              "InternalNode<uint32_t> must be exactly 256 bytes");
static_assert(sizeof(InternalNode<int32_t>) == 256,
              "InternalNode<int32_t> must be exactly 256 bytes");
static_assert(sizeof(InternalNode<uint64_t>) == 320,
              "InternalNode<uint64_t> must be exactly 320 bytes");
static_assert(sizeof(InternalNode<int64_t>) == 320,
              "InternalNode<int64_t> must be exactly 320 bytes");

// Maximum possible B-Tree height on 64-bit architectures.
// Internal nodes have capacity for 16 keys (17 children) and split 50/50,
// guaranteeing a minimum branching factor of B >= 8 (at least 8 children per
// non-root internal node).
// Thus, a tree of height H stores at least 32 * 8^(H-1) elements.
// For H = 32, minimum capacity is 32 * 8^31 = 2^98 elements, which exceeds
// the entire 64-bit addressable memory space (2^64 bytes).
// In physical 64-bit RAM, the tree height can never exceed ~20 levels, so a
// fixed stack array of 32 elements is guaranteed safe against overflow.
static constexpr size_t kMaxTreeHeight = 32;

// -----------------------------------------------------------------------------
// BTreeState: Shared layout for public container and internal SIMD engine.
template <typename KeyT,
          typename LeafT = LeafNode<typename KeyCodec<KeyT>::StorageKey>>
struct BTreeState {
  static_assert(std::is_integral_v<KeyT>,
                "Highway BTree only supports integral keys.");
  static_assert(sizeof(KeyT) == 4 || sizeof(KeyT) == 8,
                "Highway BTree only supports 32-bit and 64-bit keys.");

  void* root_ = nullptr;
  LeafT* first_leaf_ = nullptr;
  LeafT* last_leaf_ = nullptr;
  size_t tree_height_ = 0;
  size_t num_elements_ = 0;
  size_t num_leaves_ = 0;
  size_t num_internals_ = 0;
};

// -----------------------------------------------------------------------------
// Key Decompression Primitives
// -----------------------------------------------------------------------------

// Decompresses and returns the raw key stored at the given slot index in a
// leaf. Reinterprets the raw byte buffer leaf->KeyData() as the appropriate
// integer array (uint8_t, uint16_t, uint32_t, or uint64_t) and adds base_key.
template <typename LeafNode>
HWY_INLINE auto GetLeafKey(const LeafNode* HWY_RESTRICT leaf, size_t slot) {
  using KeyT = decltype(leaf->base_key);
  if (HWY_UNLIKELY(slot == 0)) return leaf->base_key;

  const uint8_t mode = leaf->BitMode();
  if (HWY_LIKELY(mode == kMode16Bit)) {
    const auto* offsets = HWY_RCAST_ALIGNED(const uint16_t*, leaf->KeyData());
    return leaf->base_key + static_cast<KeyT>(offsets[slot]);
  } else if (mode == kMode8Bit) {
    const auto* offsets = HWY_RCAST_ALIGNED(const uint8_t*, leaf->KeyData());
    return leaf->base_key + static_cast<KeyT>(offsets[slot]);
  } else if (mode == kMode32Bit) {
    if constexpr (sizeof(KeyT) == 4) {
      const auto* raw_keys =
          HWY_RCAST_ALIGNED(const uint32_t*, leaf->KeyData());
      return static_cast<KeyT>(raw_keys[slot]);
    } else {
      const auto* offsets = HWY_RCAST_ALIGNED(const uint32_t*, leaf->KeyData());
      return leaf->base_key + static_cast<KeyT>(offsets[slot]);
    }
  } else {
    const auto* raw_keys = HWY_RCAST_ALIGNED(const uint64_t*, leaf->KeyData());
    return static_cast<KeyT>(raw_keys[slot]);
  }
}

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_BTREE_BTREE_NODES_H_
