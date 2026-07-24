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

// Non-SIMD declarations. The query-time SIMD Phast class is in phast-inl.h.
// Only PhastData and BuildPhast are public.

#ifndef HIGHWAY_HWY_CONTRIB_HASH_PHAST_H_
#define HIGHWAY_HWY_CONTRIB_HASH_PHAST_H_

#include <stddef.h>
#include <stdint.h>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace hwy {

// Placement parameters used by Phast::PosFromHashAndSeed and the builder.
struct PhastPlacement {
  PhastPlacement() = default;
  explicit PhastPlacement(size_t num_slots, size_t slice_length)
      : num_slice_offsets(static_cast<uint32_t>(num_slots - slice_length + 1)),
        slice_mask(static_cast<uint32_t>(slice_length - 1)) {
    HWY_DASSERT(num_slots >= slice_length);  // else underflow
    HWY_DASSERT(num_slots <= 0xFFFFFFFFu);
    HWY_DASSERT(slice_length <= 0xFFFFFFFFu);
  }

  // Number of starting positions for overlapping slices.
  uint32_t num_slice_offsets = 0;
  uint32_t slice_mask = 0;
};

// Hyperparameters for the builder. We store multiple in a vector, hence keep
// this separate from seed storage.
struct PhastConfig {
  PhastConfig() = default;
  PhastConfig(size_t num_keys_in, size_t num_slots_in, size_t num_buckets,
              uint32_t hash_key_in, const PhastPlacement& placement_in)
      : num_slots(num_slots_in),
        hash_key(hash_key_in),
        bucket_mask(static_cast<uint32_t>(num_buckets - 1)),
        placement(placement_in) {
    HWY_DASSERT(num_slots >= num_keys_in);
    HWY_DASSERT(num_buckets <= 0xFFFFFFFFu);
    (void)num_keys_in;
  }

  size_t NumBuckets() const { return bucket_mask + 1; }
  size_t AllocatedBytes(size_t payload_bytes) const {
    return num_slots * payload_bytes +
           RoundUpTo(NumBuckets(), sizeof(uint32_t));
  }

  size_t num_slots = 0;  // 0 if build failed
  uint32_t hash_key = 0;
  uint32_t bucket_mask = 0;

  PhastPlacement placement;
};

// The only storage is 8-bit seeds, one per bucket. We pack them into u32 to
// enable Gather.
class PhastSeeds {
 public:
  PhastSeeds() = default;
  explicit PhastSeeds(size_t num_buckets)
      : words_(hwy::DivCeil(num_buckets, sizeof(uint32_t))) {}

  // For moving into PhastData.
  PhastSeeds(PhastSeeds&&) = default;
  PhastSeeds& operator=(PhastSeeds&&) = default;

  // For GatherSeeds() in phast-inl.h.
  const uint32_t* Data() const { return words_.data(); }

  HWY_INLINE uint32_t Get(uint32_t bucket_idx) const {
    const uint32_t bit_idx = (bucket_idx & 3) * 8;
    return (words_[bucket_idx >> 2] >> bit_idx) & 0xFF;
  }

  // Not named Set() due to conflict with hn::Set.
  HWY_INLINE void SetSeed(uint32_t bucket_idx, uint32_t seed) {
    const uint32_t bit_idx = (bucket_idx & 3) * 8;
    words_[bucket_idx >> 2] |= seed << bit_idx;
  }

  void Clear(uint32_t bucket_idx) {
    const uint32_t bit_idx = (bucket_idx & 3) * 8;
    words_[bucket_idx >> 2] &= ~(0xFFu << bit_idx);
  }

 private:
  AlignedVector<uint32_t> words_;
};

// Build result returned by BuildPhast.
struct PhastData {
  size_t NumSlots() const { return config.num_slots; }
  size_t AllocatedBytes(size_t payload_bytes) const {
    return config.AllocatedBytes(payload_bytes);
  }

  PhastConfig config;
  PhastSeeds seeds;

  // Telemetry indicating which config/retry succeeded.
  size_t config_idx = 0;
  size_t attempt_idx = 0;
};

// Builds from a set of distinct keys. Uses thread pool for parallel attempts.
// Takes about 1 second for 1M keys. `payload_bytes` is the number of bytes
// required per slot, i.e. potential index returned from queries. This allows us
// to optimize memory usage.
HWY_CONTRIB_DLLEXPORT PhastData BuildPhast(Span<const uint32_t> keys,
                                           size_t payload_bytes,
                                           ThreadPool& pool);

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_HASH_PHAST_H_
