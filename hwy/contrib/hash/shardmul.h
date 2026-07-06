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

// Non-SIMD declarations. The query-time SIMD `ShardMul` is in shardmul-inl.h.

#ifndef HIGHWAY_HWY_CONTRIB_HASH_SHARDMUL_H_
#define HIGHWAY_HWY_CONTRIB_HASH_SHARDMUL_H_

#include <stddef.h>
#include <stdint.h>

#include <array>

#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"               // HWY_CONTRIB_DLLEXPORT
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace hwy {

// Build result returned by BuildShardMul.
struct HWY_ALIGN_MAX ShardMulData {
  ShardMulData() = default;  // returned on build failure
  // Pass in Feistel keys because AesCtrEngine is SIMD-dependent.
  explicit ShardMulData(std::array<uint32_t, 4> keys_in) : keys(keys_in) {}

  uint32_t table[16] = {};            // u16x2 multipliers for MulHigh, aligned.
  std::array<uint32_t, 4> keys = {};  // for Feistel rounds
};

// `ShardMul` constructed from the returned `ShardMulData` reports `IsEmpty()`
// if construction fails (likely too many keys, the test verifies up to 1M).
// Otherwise, it produces distinct u32 outputs for all `keys`, which must be
// distinct. Uses `pool` to parallelize across buckets.
HWY_CONTRIB_DLLEXPORT ShardMulData BuildShardMul(Span<const uint64_t> keys,
                                                 ThreadPool& pool);

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_HASH_SHARDMUL_H_
