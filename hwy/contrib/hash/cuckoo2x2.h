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

// Non-SIMD declarations. The query-time SIMD `Cuckoo2x2` class is in
// cuckoo2x2-inl.h. Only Cuckoo2x2Data and BuildCuckoo2x2 are public.

#ifndef HIGHWAY_HWY_CONTRIB_HASH_CUCKOO2X2_H_
#define HIGHWAY_HWY_CONTRIB_HASH_CUCKOO2X2_H_

#include <stddef.h>
#include <stdint.h>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace hwy {

// Hyperparameters for the builder.
struct Cuckoo2x2Config {
  Cuckoo2x2Config() = default;
  Cuckoo2x2Config(size_t num_buckets, uint32_t hash_key_in)
      : bucket_mask(static_cast<uint32_t>(num_buckets - 1)),
        hash_key(hash_key_in) {
    HWY_DASSERT(num_buckets > 0 && num_buckets <= size_t{0xFFFFFFFF});
    HWY_DASSERT((num_buckets & (num_buckets - 1)) == 0);  // Power of 2.
  }

  size_t NumBuckets() const { return bucket_mask + 1; }

  uint32_t bucket_mask = 0;  // num_buckets - 1, for AND.
  uint32_t hash_key = 0;     // Seed for Triple32 hash1.
};

// Build result returned by BuildCuckoo2x2.
struct Cuckoo2x2Data {
  size_t NumBuckets() const { return config.NumBuckets(); }
  size_t AllocatedBytes() const { return entries.size() * sizeof(entries[0]); }

  Cuckoo2x2Config config;
  // Each entry stores 0..2 u16 values: 14-bit fingerprint + 2-bit tag
  // (00=empty, 01=hash1, 10=hash2). Empty entries are 0 (from ZeroBytes).
  AlignedVector<uint32_t> entries;  // [num_buckets]

  // Telemetry indicating which config/retry succeeded.
  size_t config_idx = 0;
  size_t attempt_idx = 0;
};

// Builds from a set of distinct keys. Uses thread pool for parallel attempts.
// Two-choice hashing: each key has two candidate buckets via independent hash
// functions. Builder assigns each key to the less-loaded bucket, requiring max
// occupancy <= 2 per bucket.
HWY_CONTRIB_DLLEXPORT Cuckoo2x2Data BuildCuckoo2x2(Span<const uint32_t> keys,
                                                   ThreadPool& pool);

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_HASH_CUCKOO2X2_H_
