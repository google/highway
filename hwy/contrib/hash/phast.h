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

// Non-SIMD declarations for PHAST: PhastConfig, PackedSeeds, PhastData,
// PhastStats, and BuildPhast. SIMD query methods are in phast-inl.h.

#ifndef HIGHWAY_HWY_CONTRIB_HASH_PHAST_H_
#define HIGHWAY_HWY_CONTRIB_HASH_PHAST_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>  // snprintf

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/stats.h"

namespace hwy {

// --------------------------------------------------------------------------
// Configuration

#pragma pack(push, 1)  // prevents false sharing
class PhastConfig {
 public:
  PhastConfig() = default;
  explicit PhastConfig(size_t num_keys, uint32_t keys_per_bucket = 2,
                       uint32_t slice_length = 4096,
                       uint32_t headroom_percent = 3,
                       uint32_t max_retries = 400)
      : num_keys_(static_cast<uint32_t>(num_keys)),
        slice_mask_(slice_length - 1),  // pow2 for fast modulo
        headroom_percent_(headroom_percent),
        max_retries_(max_retries) {
    num_slots_ =
        static_cast<uint32_t>(num_keys * (100 + headroom_percent) / 100 + 1);
    // Prevent underflow in num_slice_offsets_.
    num_slots_ = HWY_MAX(num_slots_, SliceLength());
    // Number of starting positions for overlapping slices.
    num_slice_offsets_ = num_slots_ - SliceLength() + 1;

    const uint32_t min_buckets = HWY_MAX(num_keys_ / keys_per_bucket, 1u);
    // Round up to power of 2 for fast modulo, and precompute mask.
    bucket_mask_ = (1u << hwy::CeilLog2(min_buckets)) - 1;
  }

  void ToString(char* buf, size_t buf_size) const {
    const double bits_per_key =
        static_cast<double>(ExtraBytes()) * 8 / num_keys_;
    snprintf(
        buf, buf_size,
        "Overhead=%zuK (%.1f bits/key), buckets=%uK, slice=%2uK, headroom=%u%%",
        ExtraBytes() / 1024, bits_per_key, NumBuckets() / 1024,
        SliceLength() / 1024, headroom_percent_);
  }

  uint32_t NumKeys() const { return num_keys_; }
  uint32_t NumSlots() const { return num_slots_; }
  uint32_t NumBuckets() const { return bucket_mask_ + 1; }
  uint32_t BucketMask() const { return bucket_mask_; }
  uint32_t SliceMask() const { return slice_mask_; }
  uint32_t NumSliceOffsets() const { return num_slice_offsets_; }
  uint32_t MaxRetries() const { return max_retries_; }
  uint32_t GlobalSeed() const { return global_seed_; }
  void SetGlobalSeed(uint32_t seed) { global_seed_ = seed; }

  size_t ExtraBytes(size_t payload_bytes = sizeof(uint32_t)) const {
    return (num_slots_ - num_keys_) * payload_bytes + NumBuckets();
  }

 private:
  uint32_t SliceLength() const { return slice_mask_ + 1; }

  uint32_t num_keys_ = 0;
  uint32_t num_slots_ = 0;  // includes headroom
  uint32_t bucket_mask_ = 0;
  uint32_t slice_mask_ = 0;
  uint32_t num_slice_offsets_ = 0;  // num_slots - slice_length + 1
  uint32_t global_seed_ = 0;       // set by the builder
  uint32_t headroom_percent_ = 0;
  uint32_t max_retries_ = 0;
};
#pragma pack(pop)
static_assert(sizeof(PhastConfig) == 32, "Wrong size of PhastConfig");

// --------------------------------------------------------------------------
// PackedSeeds

// The only storage is 8-bit seeds, one per bucket. We pack them into u32 to
// enable Gather.
class PackedSeeds {
 public:
  PackedSeeds() = default;
  explicit PackedSeeds(size_t num_buckets) {
    bits_.resize(hwy::DivCeil(num_buckets, sizeof(uint32_t)));
  }

  // For moving into PhastData.
  PackedSeeds(PackedSeeds&&) = default;
  PackedSeeds& operator=(PackedSeeds&&) = default;

  void Reset() {
    HWY_ASSERT(!bits_.empty());  // Must not call after moving from this.
    ZeroBytes(bits_.data(), bits_.size() * sizeof(bits_[0]));
  }

  const uint32_t* Data() const { return bits_.data(); }

  HWY_INLINE uint32_t Get(uint32_t bucket_idx) const {
    const uint32_t bit_idx = (bucket_idx & 3) * 8;
    return (bits_[bucket_idx >> 2] >> bit_idx) & 0xFF;
  }

  // Not named Set() due to conflict with hn::Set.
  HWY_INLINE void SetSeed(uint32_t bucket_idx, uint32_t seed) {
    const uint32_t bit_idx = (bucket_idx & 3) * 8;
    bits_[bucket_idx >> 2] |= seed << bit_idx;
  }

  void Clear(uint32_t bucket_idx) {
    const uint32_t bit_idx = (bucket_idx & 3) * 8;
    bits_[bucket_idx >> 2] &= ~(0xFFu << bit_idx);
  }

 private:
  AlignedVector<uint32_t> bits_;
};

// --------------------------------------------------------------------------
// PhastData: build result, returned by BuildPhast.

struct PhastData {
  bool IsEmpty() const { return config.NumKeys() == 0; }

  PhastConfig config;
  PackedSeeds seeds_packed;
};

// --------------------------------------------------------------------------

struct PhastStats {
  bool success;
  // Only populated if success:
  size_t round = ~size_t{0};
  size_t worker = ~size_t{0};

  // Only populated if !success:
  Stats s_rank;
};

// Builds from a set of distinct keys. Returns PhastData with IsEmpty() if
// max_retries is exceeded. Uses thread pool for parallel global_seed search.
HWY_CONTRIB_DLLEXPORT PhastData BuildPhast(const uint32_t* keys,
                                           PhastConfig config,
                                           ThreadPool& pool,
                                           PhastStats* stats = nullptr);

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_HASH_PHAST_H_
