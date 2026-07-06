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
#include <atomic>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"

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
#include "hwy/contrib/sort/vqsort-inl.h"
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

constexpr size_t kNumFeistelCandidates = 9;
constexpr size_t kMaxAttemptsPerBucket = 150'000;

// Try kNumFeistelCandidates Feistel keys, return the seed (index) whose
// bucketing has the smallest max:min ratio. Returns kNumFeistelCandidates on
// failure (no seed has all buckets non-empty).
size_t ChooseBestSeed(Span<const uint64_t> keys, AesCtrEngine& engine) {
  PROFILER_ZONE("build.ChooseBestSeed");
  size_t best_seed = kNumFeistelCandidates;  // invalid sentinel
  float best_ratio = 1e30f;

  for (size_t seed = 0; seed < kNumFeistelCandidates; ++seed) {
    ShardMul scatter_eval{ShardMulData(MakeFeistelKeys(engine, seed))};

    size_t bucket_sizes[kBuckets] = {};
    for (const uint64_t key : keys) {
      uint32_t LL, RR;
      scatter_eval.Feistel(key, LL, RR);
      bucket_sizes[scatter_eval.BucketIndex(LL)]++;
    }

    size_t min_size = bucket_sizes[0], max_size = bucket_sizes[0];
    for (size_t b = 1; b < kBuckets; ++b) {
      min_size = HWY_MIN(min_size, bucket_sizes[b]);
      max_size = HWY_MAX(max_size, bucket_sizes[b]);
    }

    if (min_size == 0) continue;  // skip seeds with empty buckets

    const float ratio =
        static_cast<float>(max_size) / static_cast<float>(min_size);
    if (ratio < best_ratio) {
      best_ratio = ratio;
      best_seed = seed;
    }
  }

  return best_seed;
}

// With the chosen Feistel seed, scatter keys into buckets and store LL/RR
// (interleaved: NU32 of LL, then NU32 of RR per chunk).
// Populates `feistel_per_bucket` and `actual_bucket_sizes`. Negligible cost:
// only a few ms for 1M keys.
void ScatterFeistel(Span<const uint64_t> keys, AesCtrEngine& engine,
                    size_t best_seed,
                    AlignedVector<uint32_t> (&feistel_per_bucket)[kBuckets],
                    size_t (&actual_bucket_sizes)[kBuckets]) {
  ScalableTag<uint32_t> du32;
  RepartitionToWide<decltype(du32)> du64;
  using VU32 = Vec<decltype(du32)>;
  HWY_LANES_CONSTEXPR size_t NU32 = Lanes(du32);
  HWY_LANES_CONSTEXPR size_t NU64 = Lanes(du64);

  ShardMul scatter_eval{ShardMulData(MakeFeistelKeys(engine, best_seed))};

  // First pass: count bucket sizes.
  for (size_t b = 0; b < kBuckets; ++b) actual_bucket_sizes[b] = 0;
  for (const uint64_t key : keys) {
    uint32_t LL, RR;
    scatter_eval.Feistel(key, LL, RR);
    actual_bucket_sizes[scatter_eval.BucketIndex(LL)]++;
  }

  // Temporary: gather keys per bucket for SIMD Feistel.
  AlignedVector<uint64_t> keys_per_bucket[kBuckets];
  for (size_t b = 0; b < kBuckets; ++b) {
    keys_per_bucket[b].reserve(RoundUpTo(actual_bucket_sizes[b], NU32));
  }
  for (const uint64_t key : keys) {
    uint32_t LL, RR;
    scatter_eval.Feistel(key, LL, RR);
    keys_per_bucket[scatter_eval.BucketIndex(LL)].push_back(key);
  }

  // Compute and store LL/RR per bucket via SIMD Feistel.
  for (size_t b = 0; b < kBuckets; ++b) {
    const size_t padded = RoundUpTo(actual_bucket_sizes[b], NU32);
    keys_per_bucket[b].resize(padded);  // pad for SIMD
    feistel_per_bucket[b].resize(padded * 2);

    for (size_t pos = 0; pos < padded; pos += NU32) {
      VU32 LL, RR;
      scatter_eval.Feistel(du32, Load(du64, &keys_per_bucket[b][pos]),
                           Load(du64, &keys_per_bucket[b][pos + NU64]), LL, RR);
      Store(LL, du32, &feistel_per_bucket[b][pos * 2]);
      Store(RR, du32, &feistel_per_bucket[b][pos * 2 + NU32]);
    }
    // Release original keys for this bucket.
    keys_per_bucket[b] = AlignedVector<uint64_t>();
  }
}

// Parallel search for collision-free multipliers per bucket. Returns the number
// of buckets that failed; on success (0), populates `muls_out` and
// `attempt_at_success`.
size_t FindMuls(AesCtrEngine& engine, size_t best_seed,
                const AlignedVector<uint32_t> (&feistel_per_bucket)[kBuckets],
                const size_t (&actual_bucket_sizes)[kBuckets], ThreadPool& pool,
                uint32_t (&muls_out)[kBuckets],
                uint32_t (&attempt_at_success)[kBuckets]) {
  ScalableTag<uint32_t> du32;
  using VU32 = Vec<decltype(du32)>;
  HWY_LANES_CONSTEXPR size_t NU32 = Lanes(du32);

  // Per-bucket results, spaced to avoid false sharing. Atomic with relaxed
  // ordering because multiple tasks may race to read/write the same bucket;
  // it does not matter which mul "wins".
  const auto kRel = std::memory_order_relaxed;
  constexpr size_t kU32PerLine = HWY_ALIGNMENT / sizeof(std::atomic<uint32_t>);
  alignas(HWY_ALIGNMENT) std::atomic<uint32_t>
      muls_per_bucket[kBuckets * kU32PerLine];
  std::atomic<uint32_t> attempts[kBuckets];
  for (size_t b = 0; b < kBuckets; ++b) {
    muls_per_bucket[b * kU32PerLine].store(0, kRel);
    attempts[b].store(0, kRel);
  }

  // Task decomposition: 4 tasks per worker, RoundUpToPow2 total.
  // task_idx % 16 = bucket_idx, upper bits = batch index.
  // Each task loops kAttemptsPerTask times internally to reduce pool.Run
  // dispatch overhead.
  constexpr size_t kAttemptsPerTask = 16;
  const size_t total_tasks =
      HWY_MAX(kBuckets, RoundUpToPow2(4 * pool.NumWorkers()));
  const size_t batches_per_run = total_tasks / kBuckets;
  const size_t attempts_per_run = batches_per_run * kAttemptsPerTask;
  const size_t num_outer = DivCeil(kMaxAttemptsPerBucket, attempts_per_run);

  // Per-worker scratch buffer for sorting output.
  size_t max_bucket_size = 0;
  for (size_t b = 0; b < kBuckets; ++b) {
    max_bucket_size = HWY_MAX(max_bucket_size, actual_bucket_sizes[b]);
  }
  const size_t scratch_padded = RoundUpTo(max_bucket_size, NU32);
  const size_t num_workers = pool.NumWorkers();
  AlignedVector<AlignedVector<uint32_t>> scratch(num_workers);
  for (size_t w = 0; w < num_workers; ++w) {
    scratch[w].resize(scratch_padded);
  }

  Profiler& profiler = Profiler::Get();
  const profiler::ZoneHandle z_mult = profiler.AddZone("build.mult");
  const profiler::ZoneHandle z_sort = profiler.AddZone("build.sort");

  for (size_t outer = 0; outer < num_outer; ++outer) {
    // Early termination: check if all buckets are done.
    size_t remaining = 0;
    for (size_t b = 0; b < kBuckets; ++b) {
      remaining += (muls_per_bucket[b * kU32PerLine].load(kRel) == 0);
    }
    if (remaining == 0) break;

    pool.Run(0, total_tasks, [&](uint64_t task_idx, size_t worker) {
      const size_t bucket_idx = static_cast<size_t>(task_idx) & (kBuckets - 1);
      const size_t local_batch =
          static_cast<size_t>(task_idx) >> CeilLog2(kBuckets);
      const size_t first_attempt =
          outer * attempts_per_run + local_batch * kAttemptsPerTask;

      // Early-out if bucket already solved.
      if (muls_per_bucket[bucket_idx * kU32PerLine].load(kRel) != 0) return;

      const size_t keys_in_bucket = actual_bucket_sizes[bucket_idx];
      const size_t padded = RoundUpTo(keys_in_bucket, NU32);
      // No early out if empty. Still choose a mul so that we better handle
      // out-of-distribution keys that land in this empty bucket.

      uint32_t* HWY_RESTRICT out = scratch[worker].data();
      const uint32_t* HWY_RESTRICT feistel_data =
          feistel_per_bucket[bucket_idx].data();

      for (size_t inner = 0; inner < kAttemptsPerTask; ++inner) {
        const size_t attempt = first_attempt + inner;
        if (attempt >= kMaxAttemptsPerBucket) return;

        // Re-check: another task for the same bucket may have succeeded.
        if (muls_per_bucket[bucket_idx * kU32PerLine].load(kRel) != 0) return;

        // Generate multiplier pair with attempt as seed (negligible cost).
        const uint64_t rng_seed = best_seed * kBuckets * kMaxAttemptsPerBucket +
                                  bucket_idx * kMaxAttemptsPerBucket + attempt +
                                  0x9E3779B9u;
        RngStream rng(engine, rng_seed);
        const uint32_t muls = static_cast<uint32_t>(rng() & 0xFFFFFFFFu);
        // MulHigh results are no greater than the multiplier, hence ensure
        // all multipliers are at least 0x8000. The mul-shift universal hash
        // family also requires an odd multiplier.
        const uint32_t mul0 = (muls & 0xFFFFu) | 0x8001u;
        uint32_t mul1 = (muls >> 16) | 0x8001u;
        while (mul1 == mul0) {
          mul1 = static_cast<uint32_t>((rng() & 0xFFFFu) | 0x8001u);
        }
        const VU32 muls_broadcast = Set(du32, MakePair(mul0, mul1));

        {
          PROFILER_ZONE3(profiler, worker, z_mult);
          // Compute MulAndXor for all keys in this bucket.
          for (size_t pos = 0; pos < padded; pos += NU32) {
            const VU32 LL = Load(du32, &feistel_data[pos * 2]);
            const VU32 RR = Load(du32, &feistel_data[pos * 2 + NU32]);
            Store(ShardMul::MulAndXor(du32, LL, RR, muls_broadcast), du32,
                  out + pos);
          }
        }

        // Check uniqueness: sort the output, then scan. Starting with a small
        // sample of 128 does not help. This is MUCH faster than unordered_set.
        // Most of the time is spent in VQSort; AllUnique is negligible.
        {
          PROFILER_ZONE3(profiler, worker, z_sort);
          VQSortStatic(out, keys_in_bucket, SortAscending());
        }
        if (!AllUnique(du32, out, keys_in_bucket)) continue;

        muls_per_bucket[bucket_idx * kU32PerLine].store(GetLane(muls_broadcast),
                                                        kRel);
        attempts[bucket_idx].store(static_cast<uint32_t>(attempt + 1), kRel);
        return;  // success, done with this task
      }
    });
  }

  // Copy results out and count failures.
  size_t failed_buckets = 0;
  for (size_t b = 0; b < kBuckets; ++b) {
    muls_out[b] = muls_per_bucket[b * kU32PerLine].load(kRel);
    attempt_at_success[b] = attempts[b].load(kRel);
    failed_buckets += (muls_out[b] == 0);
  }
  return failed_buckets;
}

ShardMulData BuildShardMulImpl(Span<const uint64_t> keys, ThreadPool& pool) {
  AesCtrEngine engine(/*deterministic=*/true);

  // Pick the Feistel key with the most balanced bucketing.
  const size_t best_seed = ChooseBestSeed(keys, engine);
  if (best_seed >= kNumFeistelCandidates) {
    HWY_WARN("No Feistel seed produces all non-empty buckets\n");
    // continue, best_seed is still usable.
  }

  // Scatter keys and store LL/RR per bucket.
  AlignedVector<uint32_t> feistel_per_bucket[kBuckets];
  size_t actual_bucket_sizes[kBuckets];
  ScatterFeistel(keys, engine, best_seed, feistel_per_bucket,
                 actual_bucket_sizes);

  // Find collision-free multipliers.
  uint32_t muls[kBuckets];
  uint32_t attempt_at_success[kBuckets];
  const size_t failed =
      FindMuls(engine, best_seed, feistel_per_bucket, actual_bucket_sizes, pool,
               muls, attempt_at_success);
  if (failed > 0) {
    HWY_WARN("  %zu buckets failed after %zu attempts each\n", failed,
             kMaxAttemptsPerBucket);
    return ShardMulData();
  }

  ShardMulData data(MakeFeistelKeys(engine, best_seed));
  for (size_t b = 0; b < kBuckets; ++b) {
    data.table[b] = muls[b];
    data.s_bucket_reps.Notify(static_cast<float>(attempt_at_success[b]));
  }
  return data;
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
