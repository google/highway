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

// ShardMul builder: constructs the u64->u32 reducer from distinct u64 keys.

#include "hwy/contrib/hash/shardmul.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <array>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/sort/vqsort.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/hash/shardmul.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/algo/find-inl.h"
#include "hwy/contrib/hash/shardmul-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
#if HWY_TARGET != HWY_SCALAR

HWY_INLINE_VAR constexpr size_t kBuckets = 16;

// AesCtrEngine is inside HWY_NAMESPACE, hence not passed to the ShardMulData
// constructor directly, which is declared in a normal header.
std::array<uint32_t, 4> MakeFeistelKeys(AesCtrEngine& engine, uint64_t seed) {
  return {static_cast<uint32_t>(RngStream(engine, 4 * seed + 0)()),
          static_cast<uint32_t>(RngStream(engine, 4 * seed + 1)()),
          static_cast<uint32_t>(RngStream(engine, 4 * seed + 2)()),
          static_cast<uint32_t>(RngStream(engine, 4 * seed + 3)())};
}

uint32_t MakePair(uint32_t mul0, uint32_t mul1) {
  HWY_ASSERT(mul0 < 0x10000u && mul1 < 0x10000u);
  HWY_ASSERT(mul0 != mul1);
  return (mul1 << 16) | mul0;
}

ShardMulData BuildShardMulImpl(Span<const uint64_t> keys, ThreadPool& pool) {
  ScalableTag<uint32_t> du32;
  RepartitionToWide<decltype(du32)> du64;
  using VU64 = Vec<decltype(du64)>;
  HWY_LANES_CONSTEXPR size_t NU32 = Lanes(du32);
  HWY_LANES_CONSTEXPR size_t NU64 = Lanes(du64);
  HWY_ASSERT(NU64 * 2 == NU32);

  // Seeds have minor influence, but it is important to find good multipliers.
  constexpr size_t kMaxSeeds = 8;
  constexpr size_t kMulAttempts = 4000;

  AesCtrEngine engine(/*deterministic=*/true);

  AlignedVector<uint64_t> keys_per_bucket[kBuckets];
  // Per-bucket outputs for uniqueness check.
  AlignedVector<uint32_t> bucket_out[kBuckets];
  // Prevents false sharing.
  constexpr size_t kU32PerLine = HWY_ALIGNMENT / sizeof(uint32_t);
  alignas(HWY_ALIGNMENT) uint32_t muls_per_bucket[kBuckets * kU32PerLine];

  for (uint64_t seed = 0; seed < kMaxSeeds; ++seed) {
    // Temporary ShardMul with all-zero table to compute bucket assignments.
    ShardMul scatter_eval{ShardMulData(MakeFeistelKeys(engine, seed))};

    // Scatter keys into buckets.
    for (auto& my_keys : keys_per_bucket) {
      my_keys.clear();
    }
    for (const uint64_t key : keys) {
      uint32_t LL, RR;
      scatter_eval.Feistel(key, LL, RR);
      const size_t bucket_idx = scatter_eval.BucketIndex(LL);
      keys_per_bucket[bucket_idx].push_back(key);
    }
    size_t actual_bucket_sizes[kBuckets];
    // Resize output scratch per bucket, pad arrays, and reset muls.
    for (size_t bucket_idx = 0; bucket_idx < kBuckets; ++bucket_idx) {
      actual_bucket_sizes[bucket_idx] = keys_per_bucket[bucket_idx].size();
      const size_t padded = RoundUpTo(actual_bucket_sizes[bucket_idx], NU32);
      keys_per_bucket[bucket_idx].resize(padded);
      bucket_out[bucket_idx].resize(padded);
      muls_per_bucket[bucket_idx * kU32PerLine] = 0;
    }

    // Parallel search for each bucket: find (mul0, mul1) with zero collisions.
    pool.Run(0, kBuckets, [&](uint64_t bucket_idx, size_t /*worker*/) {
      const AlignedVector<uint64_t>& my_keys = keys_per_bucket[bucket_idx];
      const size_t keys_in_bucket = actual_bucket_sizes[bucket_idx];
      // Even if we have a single key, still choose a mul so that we
      // better handle out-of-distribution keys.

      uint32_t* HWY_RESTRICT out = bucket_out[bucket_idx].data();
      // The seed offset is necessary to decouple from the Feistel round
      // constants.
      const uint64_t rng_seed = seed * kBuckets * kMulAttempts +
                                bucket_idx * kMulAttempts + 0x9E3779B9u;
      RngStream rng(engine, rng_seed);

      for (size_t att = 0; att < kMulAttempts; ++att) {
        uint32_t mul0, mul1;
        // MulHigh results are no greater than the multiplier, hence ensure all
        // multipliers are at least 0x8000. The mul-shift universal hash family
        // also requires an odd multiplier.
        do {
          mul0 = static_cast<uint32_t>((rng() & 0xFFFFu) | 0x8001u);
        } while (PopCount(mul0) < 6 || PopCount(mul0) > 13);
        do {
          mul1 = static_cast<uint32_t>((rng() & 0xFFFFu) | 0x8001u);
        } while (PopCount(mul1) < 6 || PopCount(mul1) > 13 || mul1 == mul0);

        // Build a ShardMul with only this bucket's muls set.
        ShardMulData candidate(MakeFeistelKeys(engine, seed));
        // Must set all multipliers because TwoVec verifies !IsEmpty().
        for (uint32_t& mul_pair : candidate.table) {
          mul_pair = MakePair(mul0, mul1);
        }
        ShardMul eval{candidate};

        // Evaluate keys in this bucket. Both keys and outputs are padded,
        // hence we can skip checking remainders.
        for (size_t pos = 0; pos < keys_in_bucket; pos += NU32) {
          const VU64 keys0 = Load(du64, &my_keys[pos]);
          const VU64 keys1 = Load(du64, &my_keys[pos + NU64]);
          Store(eval.TwoVec(du32, keys0, keys1), du32, out + pos);
        }

        // Done if all outputs are distinct.
        VQSort(out, keys_in_bucket, SortAscending());
        if (Unique(du32, out, keys_in_bucket) == keys_in_bucket) {
          muls_per_bucket[bucket_idx * kU32PerLine] = MakePair(mul0, mul1);
          return;
        }
      }
      // Failed after kMulAttempts: leave multipliers 0 to signal failure.
    });

    // How many buckets failed to find 0-collision multipliers?
    size_t failed_buckets = 0;
    for (size_t bucket_idx = 0; bucket_idx < kBuckets; ++bucket_idx) {
      failed_buckets += (muls_per_bucket[bucket_idx * kU32PerLine] == 0);
    }
    if (failed_buckets > 0) {
      HWY_WARN("  seed %3zu: %zu buckets failed\n", static_cast<size_t>(seed),
               failed_buckets);
      continue;  // next seed
    }

    // Success, populate and return ShardMulData.
    ShardMulData data(MakeFeistelKeys(engine, seed));
    for (size_t bucket_idx = 0; bucket_idx < kBuckets; ++bucket_idx) {
      data.table[bucket_idx] = muls_per_bucket[bucket_idx * kU32PerLine];
    }
    return data;
  }
  return ShardMulData();  // failure
}

#else   // HWY_TARGET == HWY_SCALAR
static ShardMulData BuildShardMulImpl(Span<const uint64_t>, ThreadPool&) {
  return ShardMulData();
}
#endif  // HWY_TARGET != HWY_SCALAR

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_EXPORT(BuildShardMulImpl);

HWY_CONTRIB_DLLEXPORT ShardMulData BuildShardMul(Span<const uint64_t> keys,
                                                 ThreadPool& pool) {
  return HWY_DYNAMIC_DISPATCH(BuildShardMulImpl)(keys, pool);
}

}  // namespace hwy
#endif  // HWY_ONCE
