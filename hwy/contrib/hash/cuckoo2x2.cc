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

// `Cuckoo2x2` builder: two-choice hashing. Requires distinct keys.

#include "hwy/contrib/hash/cuckoo2x2.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>  // std::sort, std::unique
#include <utility>    // std::move

#include "hwy/aligned_allocator.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/hash/cuckoo2x2.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/hash/hash-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/contrib/sort/vqsort-inl.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
#if HWY_TARGET != HWY_SCALAR

// --------------------------------------------------------------------------
// Enumerate feasible configs.

HWY_INLINE_VAR constexpr size_t kMaxAttempts = 512;

static AlignedVector<Cuckoo2x2Config> EnumerateConfigs(
    size_t num_keys, size_t num_workers, size_t& reps_per_config) {
  // num_buckets = 60% of num_keys often works, but must be rounded up to a
  // power of two. Start at 25% to ensure the rounding up is less than that,
  // then also try the next two powers of two.
  const size_t base = RoundUpToPow2(num_keys / 4);
  const size_t kMultipliers[] = {1, 2, 4};
  const size_t num_mul = sizeof(kMultipliers) / sizeof(kMultipliers[0]);

  reps_per_config = HWY_MIN(kMaxAttempts, RoundUpToPow2(num_workers / num_mul));
  reps_per_config = HWY_MAX(reps_per_config, size_t{1});

  AlignedVector<Cuckoo2x2Config> configs;
  configs.reserve(num_mul * reps_per_config);

  AesCtrEngine engine(/*deterministic=*/true);

  for (size_t mul_idx = 0; mul_idx < num_mul; ++mul_idx) {
    size_t num_buckets = base * kMultipliers[mul_idx];
    // Ensure >= 256K buckets so that 18 bucket bits + 14 fingerprint bits = 32,
    // guaranteeing zero false positives via bijection.
    num_buckets = HWY_MAX(num_buckets, uint32_t{262144});

    const size_t seed_base = mul_idx * kMaxAttempts;
    for (size_t rep = 0; rep < reps_per_config; ++rep) {
      const uint32_t hash_key =
          static_cast<uint32_t>(RngStream(engine, seed_base + rep)());
      configs.push_back(Cuckoo2x2Config(num_buckets, hash_key));
    }
  }

  std::sort(configs.begin(), configs.end(),
            [](const Cuckoo2x2Config& a, const Cuckoo2x2Config& b) {
              return a.NumBuckets() < b.NumBuckets();
            });
  return configs;
}

// --------------------------------------------------------------------------
// Per-worker builder, reused across MaybeBuild() calls.

class PerWorkerBuilder {
 public:
  explicit PerWorkerBuilder(size_t num_keys)
      : hashes1_(num_keys), hashes2_(num_keys), choice_(num_keys) {}

  uint32_t* MutableHashes() { return hashes1_.data(); }

  void MaybeBuild(Span<const uint32_t> keys, const Cuckoo2x2Config& config,
                  uint32_t hash_key1) {
    PROFILER_FUNC;
    config_ = config;
    config_.hash_key = hash_key1;
    hash1_ = Triple32(hash_key1);

    const size_t num_keys = hashes1_.size();
    HWY_ASSERT(num_keys == keys.size());
    const size_t num_buckets = config_.NumBuckets();

    // Ensure buffers are large enough (grow only, never shrink).
    if (count_.size() < num_buckets) {
      count_.resize(num_buckets);
    }
    if (entries_.size() < num_buckets) {
      entries_.resize(num_buckets);
    }
    ZeroBytes(count_.data(), num_buckets * sizeof(count_[0]));
    ZeroBytes(entries_.data(), num_buckets * sizeof(entries_[0]));

    // Vectorized hashing via Triple32 and WeakOneMul.
    HashArray(hash1_, keys.data(), hashes1_.data(), num_keys);
    HashArray(WeakOneMul(), keys.data(), hashes2_.data(), num_keys);

    // Cuckoo insertion with displacement; fail if cycle detected.
    if (!CuckooAssign(num_keys)) {
      succeeded_ = false;
      return;
    }

    // Populate fingerprint entries.
    PopulateEntries(num_keys, num_buckets);
    succeeded_ = true;
  }

  bool Succeeded() const { return succeeded_; }

  Cuckoo2x2Data Take(size_t config_idx, size_t attempt_idx) {
    HWY_ASSERT(Succeeded());
    Cuckoo2x2Data data;
    data.config = config_;
    data.entries = std::move(entries_);
    data.config_idx = config_idx;
    data.attempt_idx = attempt_idx;
    return data;
  }

 private:
  size_t Bucket(size_t key_idx, uint8_t which) const {
    const uint32_t h = (which == 0) ? hashes1_[key_idx] : hashes2_[key_idx];
    return h & config_.bucket_mask;
  }

  // Cuckoo insertion with displacement. Returns true if all keys placed
  // with max 2 per bucket. For bucket_size=2, supports load factor ~84%.
  bool CuckooAssign(size_t num_keys) {
    const size_t num_buckets = config_.NumBuckets();
    static constexpr int kMaxDisplacements = 500;

    // Track which key indices occupy each bucket (at most 2 per bucket).
    if (bucket_keys_.size() < num_buckets * 2) {
      bucket_keys_.resize(num_buckets * 2);
    }
    // Sentinel: ~0u means empty slot.
    for (size_t i = 0; i < num_buckets * 2; ++i) {
      bucket_keys_[i] = ~0u;
    }

    for (size_t i = 0; i < num_keys; ++i) {
      size_t ki = i;
      uint8_t which = 0;  // Start with hash1.

      // Try hash1 bucket first; if full, try hash2.
      const size_t b1 = Bucket(ki, 0);
      const size_t b2 = Bucket(ki, 1);
      if (count_[b1] <= count_[b2] && count_[b1] < 2) {
        choice_[ki] = 0;
        bucket_keys_[b1 * 2 + count_[b1]] = static_cast<uint32_t>(ki);
        ++count_[b1];
        continue;
      }
      if (count_[b2] < 2) {
        choice_[ki] = 1;
        bucket_keys_[b2 * 2 + count_[b2]] = static_cast<uint32_t>(ki);
        ++count_[b2];
        continue;
      }
      if (count_[b1] < 2) {
        choice_[ki] = 0;
        bucket_keys_[b1 * 2 + count_[b1]] = static_cast<uint32_t>(ki);
        ++count_[b1];
        continue;
      }

      // Both buckets full — displace a chain of residents.
      which = 0;
      bool placed = false;
      for (int d = 0; d < kMaxDisplacements; ++d) {
        const size_t b = Bucket(ki, which);
        if (count_[b] < 2) {
          // Place here.
          choice_[ki] = which;
          bucket_keys_[b * 2 + count_[b]] = static_cast<uint32_t>(ki);
          ++count_[b];
          placed = true;
          break;
        }
        // Evict the first resident of this bucket.
        const size_t evicted = bucket_keys_[b * 2];
        // Shift second resident to first slot.
        bucket_keys_[b * 2] = bucket_keys_[b * 2 + 1];
        // Place new key in second slot.
        bucket_keys_[b * 2 + 1] = static_cast<uint32_t>(ki);
        choice_[ki] = which;

        // The evicted key must go to its OTHER bucket.
        ki = evicted;
        which = static_cast<uint8_t>(1 - choice_[evicted]);
      }
      if (!placed) return false;  // Cycle detected.
    }
    return true;
  }

  void PopulateEntries(size_t num_keys, size_t num_buckets) {
    ZeroBytes(count_.data(), num_buckets * sizeof(count_[0]));

    for (size_t i = 0; i < num_keys; ++i) {
      const uint32_t h = (choice_[i] == 0) ? hashes1_[i] : hashes2_[i];
      const size_t b = h & config_.bucket_mask;
      // 14-bit fingerprint (h >> 18) + 2-bit tag in bits 14-15:
      //   00 = empty, 01 = hash1, 10 = hash2.
      // 18 bucket bits + 14 fingerprint bits = 32, so we cover all 32 hash
      // bits and thus rule out any false positives. The tag prevents one hash
      // from colliding with the other, and also matching and empty bucket.
      const uint32_t fp = (h >> 18) | (choice_[i] == 0 ? 0x4000u : 0x8000u);
      const uint8_t slot = count_[b]++;
      entries_[b] |= (fp << (slot * 16));
    }

    // Duplicate fp for single-occupancy buckets. Empty buckets stay 0
    // (tag=00), which can never match any query (tag=01 or 10).
    for (size_t b = 0; b < num_buckets; ++b) {
      if (count_[b] == 1) {
        entries_[b] |= (entries_[b] << 16);
      }
    }
  }

  AlignedVector<uint32_t> hashes1_;      // hash1[num_keys]
  AlignedVector<uint32_t> hashes2_;      // hash2[num_keys]
  AlignedVector<uint8_t> choice_;        // 0=bucket1, 1=bucket2
  AlignedVector<uint8_t> count_;         // [num_buckets]
  AlignedVector<uint32_t> entries_;      // [num_buckets]
  AlignedVector<uint32_t> bucket_keys_;  // [num_buckets * 2], key indices

  Cuckoo2x2Config config_;
  Triple32 hash1_;
  bool succeeded_ = false;
};

// --------------------------------------------------------------------------
// Parallel builder driver.

static Cuckoo2x2Data BuildCuckoo2x2Impl(Span<const uint32_t> keys,
                                        ThreadPool& pool) {
  const size_t num_keys = keys.size();
  const size_t num_workers = pool.NumWorkers();

  AlignedVector<PerWorkerBuilder> per_worker;
  per_worker.reserve(num_workers);
  for (size_t i = 0; i < num_workers; ++i) {
    per_worker.emplace_back(num_keys);
  }

  // Ensure keys are distinct.
  AesCtrEngine engine(/*deterministic=*/true);
  HashArray(Triple32(engine, 0), keys.data(), per_worker[0].MutableHashes(),
            num_keys);
  VQSortStatic(per_worker[0].MutableHashes(), num_keys, SortAscending());
  uint32_t* end = per_worker[0].MutableHashes() + num_keys;
  HWY_ASSERT_M(end == std::unique(per_worker[0].MutableHashes(), end),
               "Collision detected");

  size_t reps_per_config;
  AlignedVector<Cuckoo2x2Config> configs =
      EnumerateConfigs(num_keys, num_workers, reps_per_config);
  const Divisor div_reps(static_cast<uint32_t>(reps_per_config));
  const size_t outer_reps = div_reps.Divide(kMaxAttempts);

  for (size_t config_idx = 0; config_idx < configs.size();
       config_idx += num_workers) {
    const size_t num_tasks = HWY_MIN(num_workers, configs.size() - config_idx);
    for (size_t outer_rep = 0; outer_rep < outer_reps; ++outer_rep) {
      pool.Run(0, num_tasks, [&](uint64_t task_idx, size_t worker) {
        const Cuckoo2x2Config& config =
            configs[config_idx + static_cast<size_t>(task_idx)];
        const size_t rep =
            outer_rep * reps_per_config + static_cast<size_t>(task_idx);
        const uint32_t hash_key1 =
            static_cast<uint32_t>(RngStream(engine, rep)());
        per_worker[worker].MaybeBuild(keys, config, hash_key1);
      });

      const size_t attempt_idx = outer_rep * reps_per_config;
      for (size_t worker = 0; worker < num_tasks; ++worker) {
        if (per_worker[worker].Succeeded()) {
          return per_worker[worker].Take(
              config_idx + div_reps.Divide(static_cast<uint32_t>(worker)),
              attempt_idx + div_reps.Remainder(static_cast<uint32_t>(worker)));
        }
      }
    }
  }

  return Cuckoo2x2Data();  // All configs exhausted.
}

#else   // HWY_TARGET == HWY_SCALAR
static Cuckoo2x2Data BuildCuckoo2x2Impl(Span<const uint32_t>, ThreadPool&) {
  return Cuckoo2x2Data();
}
#endif  // HWY_TARGET != HWY_SCALAR
}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_EXPORT(BuildCuckoo2x2Impl);

HWY_CONTRIB_DLLEXPORT Cuckoo2x2Data BuildCuckoo2x2(Span<const uint32_t> keys,
                                                   ThreadPool& pool) {
  return HWY_DYNAMIC_DISPATCH(BuildCuckoo2x2Impl)(keys, pool);
}

}  // namespace hwy
#endif  // HWY_ONCE
