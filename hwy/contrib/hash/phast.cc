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

// PHAST builder: constructs the perfect hash function from distinct u32 keys.

#include "hwy/contrib/hash/phast.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>  // std::sort, std::unique
#include <utility>    // std::move
#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/hash/phast.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/algo/find-inl.h"
#include "hwy/contrib/hash/hash-inl.h"
#include "hwy/contrib/hash/phast-inl.h"
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

HWY_INLINE_VAR constexpr size_t kMaxAttempts = 256;  // pow2 for faster mul/div.

PhastConfig MakeConfig(size_t num_keys, int headroom_percent,
                       size_t keys_per_bucket, size_t slice_length,
                       uint32_t seed) {
  size_t num_slots =
      num_keys * static_cast<size_t>(100 + headroom_percent) / 100 + 1;
  // Prevents underflow in PhastPlacement::num_slice_offsets.
  num_slots = HWY_MAX(num_slots, slice_length);

  // Round up to power of 2 for fast modulo.
  const size_t num_buckets = RoundUpToPow2(num_keys / keys_per_bucket);

  return PhastConfig(num_keys, num_slots, num_buckets, seed,
                     PhastPlacement(num_slots, slice_length));
}

size_t MinSliceLength(size_t num_keys) {
  HWY_ASSERT(num_keys >= 1);
  // Based on suggestions from the PHAST paper, section 5.1.
  if (num_keys < 64) return RoundDownToPow2(num_keys);
  if (num_keys < 1'300) return 64;
  if (num_keys < 9'500) return 128;
  if (num_keys < 12'000) return 256;
  if (num_keys < 140'000) return 512;
  return 2048;
}

// Enumerates feasible configs and sorts by extra_bytes ascending.
static AlignedVector<PhastConfig> EnumerateConfigs(size_t num_keys,
                                                   size_t num_workers,
                                                   size_t& reps_per_config) {
  const size_t min_slice_length = MinSliceLength(num_keys);
  HWY_ASSERT(min_slice_length < num_keys);

  const std::vector<int> kHeadroomPercents = {1, 2, 3, 4, 10};
  const std::vector<size_t> kSliceShifts = {0, 1};
  // If num_keys is less than 30% above a power of 2, kpb=3 is preferred
  // because kpb=2 would nearly double the bucket count (wasteful).
  // If more than 70% above, kpb=2 and kpb=3 yield the same bucket count
  // due to rounding up to pow2. Otherwise try both.
  const size_t prev_pow2 = RoundDownToPow2(num_keys);
  const double ratio =
      static_cast<double>(num_keys) / static_cast<double>(prev_pow2);
  std::vector<size_t> kpb_values;
  if (ratio < 1.3) {
    kpb_values = {3};  // kpb=2 wastes bucket space
  } else if (ratio > 1.7) {
    kpb_values = {2};  // kpb=2 and 3 give same bucket count
  } else {
    kpb_values = {3, 2};  // try both
  }

  // Replicate configs (with different hash_key) to keep most workers busy.
  const size_t permutations =
      kHeadroomPercents.size() * kSliceShifts.size() * kpb_values.size();
  reps_per_config =
      HWY_MIN(kMaxAttempts, RoundUpToPow2(num_workers / permutations));
  HWY_ASSERT(reps_per_config != 0);

  AlignedVector<PhastConfig> configs;
  configs.reserve(permutations * reps_per_config);

  size_t permutation_idx = 0;
  for (size_t kpb : kpb_values) {
    for (int h : kHeadroomPercents) {
      for (size_t slice_shift : kSliceShifts) {
        const size_t slice_length = min_slice_length << slice_shift;

        // Different seed for each config to increase diversity.
        const size_t seed_base = (permutation_idx++) * kMaxAttempts;
        for (size_t rep = 0; rep < reps_per_config; ++rep) {
          const uint32_t seed = static_cast<uint32_t>(seed_base + rep);
          configs.push_back(MakeConfig(num_keys, h, kpb, slice_length, seed));
        }
      }
    }
  }
  HWY_ASSERT(configs.size() == permutations * reps_per_config);

  // Should already be in order, but just in case.
  std::sort(configs.begin(), configs.end(),
            [](const PhastConfig& a, const PhastConfig& b) {
              return a.extra_bytes < b.extra_bytes;
            });
  return configs;
}

// --------------------------------------------------------------------------

// Tracks which slots are occupied during building; holds one bit per slot.
class Occupancy {
 public:
  Occupancy() = default;
  explicit Occupancy(size_t num_slots)
      : bits_(DivCeil(num_slots, size_t{32})) {}

  bool IsOccupied(uint32_t pos) const {
    return (bits_[pos >> 5] >> (pos & 31)) & 1;
  }

  // Returns unoccupied slots, AND mask (avoids checking positions where other
  // keys are already occupied).
  template <class DU32, class MU32 = Mask<DU32>, class VU32 = Vec<DU32>>
  MU32 Unoccupied(DU32 du32, MU32 mask, VU32 pos) const {
    const RebindToSigned<DU32> di32;
    const VU32 word_idx = ShiftRight<5>(pos);
    const VU32 bit_idx = And(pos, Set(du32, 31));
    const VU32 bit_mask = Shl(Set(du32, 1), bit_idx);
    const VU32 words =
        MaskedGatherIndex(mask, du32, bits_.data(), BitCast(di32, word_idx));
    // Unoccupied where the bit is NOT set.
    return MaskedEq(mask, And(words, bit_mask), Zero(du32));
  }

  void SetOccupied(uint32_t pos) {
    HWY_DASSERT(!IsOccupied(pos));
    bits_[pos >> 5] |= 1u << (pos & 31);
  }

  void ClearOccupied(uint32_t pos) {
    HWY_DASSERT(IsOccupied(pos));
    bits_[pos >> 5] &= ~(1u << (pos & 31));
  }

  // Returns occupancy in the 32-position neighborhood of `pos`. This steers
  // placements toward sparse neighborhoods. 32-bit words are required for
  // SIMD, otherwise we could use 64. One popcount outperforms 3x64
  // neighborhoods. First-viable and max_pos tiebreakers were worse.
  uint32_t NeighborDensity(uint32_t pos) const {
    return static_cast<uint32_t>(hwy::PopCount(bits_[pos >> 5]));
  }

 private:
  AlignedVector<uint32_t> bits_;
};

// --------------------------------------------------------------------------

// Reused for multiple MaybeBuild() calls per worker to reduce allocation cost.
class PerWorkerBuilder {
  static constexpr size_t kMaxHashesPerBucket = 32;

 public:
  explicit PerWorkerBuilder(size_t num_keys)
      : hash_for_key_idx_(num_keys),
        all_pos_(size_t{kMaxHashesPerBucket} * 256),
        key_idx_for_pos_(num_keys) {}

  // This can be used for validating keys are unique without another copy.
  uint32_t* MutableHashes() { return hash_for_key_idx_.data(); }

  // Attempts to build with a given config and hash_key, which overrides
  // config_.hash_key. Call Succeeded() to check success.
  void MaybeBuild(const uint32_t* keys, const PhastConfig& config,
                  uint32_t hash_key) {
    PROFILER_FUNC;
    HWY_ALIGN uint32_t seed_candidates[256];

    config_ = config;
    config_.hash_key = hash_key;
    const size_t num_buckets = config.NumBuckets();
    hash_ = Triple32(hash_key);
    occupancy_ = Occupancy(config.num_slots);
    seeds_ = PhastSeeds(num_buckets);
    committed_seeds_.resize(num_buckets);

    num_hashes_for_bucket_.resize(num_buckets);
    total_hashes_before_bucket_.resize(num_buckets + 1);
    bucket_idx_largest_first_.resize(num_buckets);

    const size_t num_keys = hash_for_key_idx_.size();
    HashArray(hash_, keys, MutableHashes(), num_keys);  // Negligible time.
    PopulateBuckets(num_keys);

    // Process all non-empty buckets in descending size order.
    const size_t num_ge1_buckets = num_buckets - num_buckets_with_size_[0];
    for (uint32_t rank = 0; rank < num_ge1_buckets; ++rank) {
      const uint32_t seed =
          TryPlaceBucket(rank, /*excluded_seed=*/~0u, seed_candidates);
      if (HWY_LIKELY(seed != ~0u)) {  // Successfully placed bucket.
        committed_seeds_[rank] = static_cast<uint8_t>(seed);
        continue;
      }

      // Greedy placement failed; try single-bucket cuckoo swaps.
      if (HWY_UNLIKELY(!TryCuckooSwap(rank, seed_candidates))) {
        succeeded_ = false;  // Failed. rank indicates how close we got.
        return;
      }
    }

    succeeded_ = true;
  }

  // Call after MaybeBuild().
  bool Succeeded() const { return succeeded_; }

  // Only call if Succeeded().
  PhastData Take(size_t config_idx, size_t attempt_idx) {
    HWY_ASSERT(Succeeded());
    HWY_ASSERT(config_.hash_key == hash_.Key());
    return {config_, std::move(seeds_), config_idx, attempt_idx};
  }

 private:
  // When greedy placement fails at `rank`, try single-bucket cuckoo swaps.
  // Finds buckets that occupy the positions the failing bucket needs, then
  // swaps one at a time: undo the blocker, place the failing bucket (now
  // unblocked), and place the blocker again with a different seed.
  //
  // At 98% occupancy and two keys per bucket, assuming uncorrelated hashes
  // (Hash16 is reasonably good), the expected probability of placing both is
  // 4E-4, but can vary across buckets. Given 255 seeds/trials, the chance of
  // any succeeding is then ~10%: 1 - (1 - p)^255. It is critical to undo/swap
  // one at a time, because all succeeding has probability 0.1^N -> 0, whereas
  // trying individually gives 1 - 0.9^N -> 1.
  bool TryCuckooSwap(const uint32_t rank,
                     uint32_t* HWY_RESTRICT seed_candidates) {
    PROFILER_FUNC;
    constexpr uint32_t kMaxSwaps = 20;
    constexpr uint32_t kScanWindow = 50000;
    const uint32_t scan_start = (rank > kScanWindow) ? rank - kScanWindow : 0;

    // Compute the failing bucket's candidate positions. WARNING: another call
    // to ComputeBucketPos overwrites all_pos_, but ComputeOnePos does not.
    const size_t fail_b_size = ComputeBucketPos(rank);
    const size_t num_pos_to_find = fail_b_size * 256;

    // Scan recent buckets for those whose committed positions match any
    // target position (i.e., they block a candidate seed).
    const ScalableTag<uint32_t> du32;
    uint32_t overlap_ranks[kMaxSwaps];
    uint8_t original_seeds[kMaxSwaps];
    uint32_t num_overlap = 0;
    for (uint32_t scan_pos = rank;
         scan_pos > scan_start && num_overlap < kMaxSwaps; --scan_pos) {
      const uint32_t scan_bucket_idx = bucket_idx_largest_first_[scan_pos - 1];
      const uint8_t scan_seed = committed_seeds_[scan_pos - 1];
      const size_t scan_begin = total_hashes_before_bucket_[scan_bucket_idx];
      const size_t scan_size = num_hashes_for_bucket_[scan_bucket_idx];

      // Check if this bucket's committed positions match any target.
      // Uses scalar ComputeOnePos - cheaper, and avoids overwriting all_pos_.
      bool matches = false;
      for (size_t idx = 0; idx < scan_size && !matches; ++idx) {
        const uint32_t pos = ComputeOnePos(scan_begin + idx, scan_seed);
        matches = Find(du32, pos, all_pos_.data(), num_pos_to_find) !=
                  num_pos_to_find;
      }
      if (matches) {
        overlap_ranks[num_overlap] = scan_pos - 1;
        original_seeds[num_overlap] = scan_seed;
        ++num_overlap;
      }
    }

    // Try swapping each blocker individually.
    for (uint32_t i = 0; i < num_overlap; ++i) {
      const uint32_t blocker_rank = overlap_ranks[i];
      const uint8_t blocker_seed = original_seeds[i];

      UndoBucket(blocker_rank, blocker_seed);

      // Place the failing bucket (gets first pick of freed space).
      // NOTE: this overwrites all_pos_.
      const uint32_t failed_bucket_seed =
          TryPlaceBucket(rank, /*excluded_seed=*/~0u, seed_candidates);
      if (failed_bucket_seed != ~0u) {
        committed_seeds_[rank] = static_cast<uint8_t>(failed_bucket_seed);

        // Place the blocker again with a different seed.
        const uint32_t new_blocker_seed =
            TryPlaceBucket(blocker_rank, blocker_seed, seed_candidates);
        if (new_blocker_seed != ~0u) {
          committed_seeds_[blocker_rank] =
              static_cast<uint8_t>(new_blocker_seed);
          return true;  // Swap succeeded.
        }

        // Unable to place blocker; undo the failing bucket.
        UndoBucket(rank, failed_bucket_seed);
      }

      // Restore the blocker to its original state.
      ForcePlaceBucket(blocker_rank, blocker_seed);
      committed_seeds_[blocker_rank] = blocker_seed;
    }

    return false;
  }

  // Computes the position for a single hash+seed pair (scalar).
  // Much cheaper than ComputeBucketPos, which computes all 256 seeds.
  uint32_t ComputeOnePos(size_t b_pos, uint32_t seed) const {
    const uint32_t key_idx = key_idx_for_pos_[b_pos];
    const uint32_t hash = hash_for_key_idx_[key_idx];
    return Phast::PosFromHashAndSeed(config_.placement, hash, seed);
  }

  // Computes positions for all keys in a bucket, storing at indices
  // [0, b_size) * 256 in all_pos_. Returns b_size.
  size_t ComputeBucketPos(size_t rank) {
    const uint32_t bucket_idx = bucket_idx_largest_first_[rank];
    const size_t b_begin = total_hashes_before_bucket_[bucket_idx];
    const size_t b_size = num_hashes_for_bucket_[bucket_idx];

    const ScalableTag<uint32_t> du32;
    HWY_LANES_CONSTEXPR size_t NU32 = Lanes(du32);
    using VU32 = Vec<decltype(du32)>;

    for (size_t idx = 0; idx < b_size; ++idx) {
      const uint32_t key_idx = key_idx_for_pos_[b_begin + idx];
      const VU32 hash = Set(du32, hash_for_key_idx_[key_idx]);
      for (size_t seed = 0; seed < 256; seed += 2 * NU32) {
        const VU32 s0 = Iota(du32, seed);
        const VU32 s1 = Iota(du32, seed + NU32);
        VU32 pos0, pos1;
        Phast::PosFromHashAndSeed(config_.placement, du32, hash, hash, s0, s1,
                                  pos0, pos1);
        StoreU(pos0, du32, &all_pos_[idx * 256 + seed]);
        StoreU(pos1, du32, &all_pos_[idx * 256 + seed + NU32]);
      }
    }
    return b_size;
  }

  // Returns the committed seed (0-255) on success, or ~0u on failure.
  // If excluded_seed < 256, that seed is skipped.
  uint32_t TryPlaceBucket(const uint32_t rank, const uint32_t excluded_seed,
                          uint32_t* HWY_RESTRICT seed_candidates) {
    const size_t b_size = ComputeBucketPos(rank);
    const size_t num_candidates = WriteSeedCandidates(b_size, seed_candidates);

    uint32_t best_seed = 0;
    uint32_t best_cost = ~0u;

    for (size_t cand_idx = 0; cand_idx < num_candidates; ++cand_idx) {
      const uint32_t seed = seed_candidates[cand_idx];
      if (seed == excluded_seed) continue;

      if (b_size >= 2) {
        // Pairwise collision check.
        bool has_dup = false;
        for (size_t i = 0; i < b_size && !has_dup; ++i) {
          for (size_t j = i + 1; j < b_size && !has_dup; ++j) {
            has_dup = (all_pos_[i * 256 + seed] == all_pos_[j * 256 + seed]);
          }
        }
        if (has_dup) continue;

        // Density cost: sum of occupied neighbors.
        uint32_t cost = 0;
        for (size_t idx = 0; idx < b_size; ++idx) {
          cost += occupancy_.NeighborDensity(all_pos_[idx * 256 + seed]);
        }

        if (cost < best_cost) {
          best_cost = cost;
          best_seed = seed;
          if (b_size <= 2 && cost == 0) break;  // Good enough.
        }
      } else {
        // b_size == 1: first unoccupied candidate wins.
        best_cost = 0;
        best_seed = seed;
        break;
      }
    }

    if (best_cost == ~0u) return ~0u;

    // Commit the best seed's positions.
    for (size_t idx = 0; idx < b_size; ++idx) {
      occupancy_.SetOccupied(all_pos_[idx * 256 + best_seed]);
    }
    seeds_.SetSeed(bucket_idx_largest_first_[rank], best_seed);
    return best_seed;
  }

  // Undoes the placement of the bucket at the given rank.
  void UndoBucket(uint32_t rank, uint32_t seed) {
    const uint32_t bucket_idx = bucket_idx_largest_first_[rank];
    const size_t b_begin = total_hashes_before_bucket_[bucket_idx];
    const size_t b_size = num_hashes_for_bucket_[bucket_idx];
    HWY_DASSERT(b_size > 0 && seed < 256);

    for (size_t idx = 0; idx < b_size; ++idx) {
      occupancy_.ClearOccupied(ComputeOnePos(b_begin + idx, seed));
    }
    seeds_.Clear(bucket_idx);
  }

  // Places a bucket with a known-valid seed (no candidate search).
  // Used to restore original state after a failed backtrack attempt.
  void ForcePlaceBucket(uint32_t rank, uint32_t seed) {
    const uint32_t bucket_idx = bucket_idx_largest_first_[rank];
    const size_t b_begin = total_hashes_before_bucket_[bucket_idx];
    const size_t b_size = num_hashes_for_bucket_[bucket_idx];
    HWY_DASSERT(b_size > 0 && seed < 256);

    for (size_t idx = 0; idx < b_size; ++idx) {
      occupancy_.SetOccupied(ComputeOnePos(b_begin + idx, seed));
    }
    seeds_.SetSeed(bucket_idx, seed);
  }

  size_t BucketIdxFromKeyIdx(size_t key_idx) const {
    return hash_for_key_idx_[key_idx] & config_.bucket_mask;
  }

  void PopulateBuckets(size_t num_keys) {
    PROFILER_FUNC;
    // O(N) scan: increment counts for each bucket.
    const size_t num_buckets = num_hashes_for_bucket_.size();
    ZeroBytes(num_hashes_for_bucket_.data(),
              num_buckets * sizeof(num_hashes_for_bucket_[0]));
    for (size_t key_idx = 0; key_idx < num_keys; ++key_idx) {
      const size_t bucket_idx = BucketIdxFromKeyIdx(key_idx);
      num_hashes_for_bucket_[bucket_idx]++;
      HWY_ASSERT(num_hashes_for_bucket_[bucket_idx] < kMaxHashesPerBucket);
    }

    // Histogram of bucket sizes for counting sort.
    size_t max_hashes_per_bucket = 0;
    ZeroBytes(num_buckets_with_size_, sizeof(num_buckets_with_size_));
    for (uint32_t bucket_idx = 0; bucket_idx < num_buckets; ++bucket_idx) {
      const size_t hashes_in_bucket = num_hashes_for_bucket_[bucket_idx];
      max_hashes_per_bucket = HWY_MAX(max_hashes_per_bucket, hashes_in_bucket);
      num_buckets_with_size_[hashes_in_bucket]++;
    }

    // Compute total_hashes_before_bucket_ as exclusive prefix sum of
    // num_hashes_for_bucket_.
    total_hashes_before_bucket_[0] = 0;
    for (uint32_t bucket_idx = 0; bucket_idx < num_buckets; ++bucket_idx) {
      total_hashes_before_bucket_[bucket_idx + 1] =
          total_hashes_before_bucket_[bucket_idx] +
          num_hashes_for_bucket_[bucket_idx];
    }

    // Store key_idx grouped by bucket_idx.
    AlignedVector<uint32_t> write_pos(num_buckets);  // write cursor
    CopyBytes(total_hashes_before_bucket_.data(), write_pos.data(),
              num_buckets * sizeof(uint32_t));
    for (uint32_t key_idx = 0; key_idx < num_keys; ++key_idx) {
      const size_t bucket_idx = BucketIdxFromKeyIdx(key_idx);
      key_idx_for_pos_[write_pos[bucket_idx]++] = key_idx;
    }

    // Compute suffix sums: order_begin[s] = number of buckets with size > s.
    uint32_t order_begin[kMaxHashesPerBucket + 1];
    order_begin[max_hashes_per_bucket] = 0;
    for (size_t size = max_hashes_per_bucket; size > 0; --size) {
      order_begin[size - 1] = order_begin[size] + num_buckets_with_size_[size];
    }

    // Store bucket_idx in descending-size order.
    for (uint32_t bucket_idx = 0; bucket_idx < num_buckets; ++bucket_idx) {
      const size_t size = num_hashes_for_bucket_[bucket_idx];
      bucket_idx_largest_first_[order_begin[size]++] = bucket_idx;
    }
  }

  // Writes seed candidates for which slots are unoccupied for all bucket keys
  // with relative index [0, b_size).
  size_t WriteSeedCandidates(size_t b_size,
                             uint32_t* HWY_RESTRICT seed_candidates) const {
    const ScalableTag<uint32_t> du32;
    HWY_LANES_CONSTEXPR size_t NU32 = Lanes(du32);
    using VU32 = Vec<decltype(du32)>;
    using MU32 = Mask<decltype(du32)>;

    size_t num_candidates = 0;
    for (size_t seed = 0; seed < 256; seed += 2 * NU32) {
      const VU32 s0 = Iota(du32, seed);
      const VU32 s1 = Iota(du32, seed + NU32);
      MU32 avail0 = SetMask(du32, true);
      MU32 avail1 = SetMask(du32, true);
      for (size_t idx = 0; idx < b_size; ++idx) {
        const VU32 pos0 = LoadU(du32, &all_pos_[idx * 256 + seed]);
        const VU32 pos1 = LoadU(du32, &all_pos_[idx * 256 + seed + NU32]);
        avail0 = occupancy_.Unoccupied(du32, avail0, pos0);
        avail1 = occupancy_.Unoccupied(du32, avail1, pos1);
      }
      num_candidates +=
          CompressStore(s0, avail0, du32, seed_candidates + num_candidates);
      num_candidates +=
          CompressStore(s1, avail1, du32, seed_candidates + num_candidates);
    }

    return num_candidates;
  }

  AlignedVector<uint32_t> hash_for_key_idx_;  // [num_keys]
  // Indexed by [local_idx * 256 + seed], computed per bucket.
  AlignedVector<uint32_t> all_pos_;  // [kMaxHashesPerBucket*256]

  // Access via key_idx_for_pos_[total_hashes_before_bucket_[bucket_idx] + i]
  // for i-th key in bucket_idx.
  AlignedVector<uint32_t> key_idx_for_pos_;  // [num_keys]

  // Set by MaybeBuild():

  PhastConfig config_;
  Triple32 hash_;
  Occupancy occupancy_;
  PhastSeeds seeds_;
  AlignedVector<uint8_t> committed_seeds_;  // seed per rank, for backtracking

  AlignedVector<uint8_t> num_hashes_for_bucket_;        // [num_buckets]
  AlignedVector<uint32_t> total_hashes_before_bucket_;  // [num_buckets + 1]
  uint32_t num_buckets_with_size_[kMaxHashesPerBucket + 1];

  // Bucket indices sorted by decreasing bucket size.
  AlignedVector<uint32_t> bucket_idx_largest_first_;  // [num_buckets]

  bool succeeded_ = false;
};

// --------------------------------------------------------------------------

// Enumerates configs, tries each with parallel workers, returns the best.
static PhastData BuildPhastImpl(const uint32_t* keys, size_t num_keys,
                                ThreadPool& pool) {
  const size_t num_workers = pool.NumWorkers();

  // Allocate per-worker builders once; reused across all configs.
  AlignedVector<PerWorkerBuilder> per_worker;
  per_worker.reserve(num_workers);
  for (size_t i = 0; i < num_workers; ++i) {
    per_worker.emplace_back(num_keys);
  }

  // Ensure keys are distinct, otherwise building would fail. Sort the hashes
  // because `keys` are immutable, and the hash function is a permutation,
  // hence hash collisions imply duplicate keys.
  AesCtrEngine engine(/*deterministic=*/true);
  HashArray(Triple32(engine, 0), keys, per_worker[0].MutableHashes(), num_keys);
  VQSortStatic(per_worker[0].MutableHashes(), num_keys, SortAscending());
  uint32_t* end = per_worker[0].MutableHashes() + num_keys;
  HWY_ASSERT_M(end == std::unique(per_worker[0].MutableHashes(), end),
               "Collision detected");

  // Enumerate configs sorted by extra_bytes ascending.
  size_t reps_per_config;
  AlignedVector<PhastConfig> configs =
      EnumerateConfigs(num_keys, num_workers, reps_per_config);
  const Divisor div_reps(static_cast<uint32_t>(reps_per_config));
  const size_t outer_reps = div_reps.Divide(kMaxAttempts);

  // Attempt configs in batches of up to num_workers.
  for (size_t config_idx = 0; config_idx < configs.size();
       config_idx += num_workers) {
    const size_t num_tasks = HWY_MIN(num_workers, configs.size() - config_idx);
    for (size_t outer_rep = 0; outer_rep < outer_reps; ++outer_rep) {
      pool.Run(0, num_tasks, [&](uint64_t task_idx, size_t worker) {
        const PhastConfig& config =
            configs[config_idx + static_cast<size_t>(task_idx)];
        HWY_ASSERT(task_idx == worker);  // one task per worker
        // config.hash_key = permutation_idx * kMaxAttempts + rep, where
        // rep < reps_per_config. Adding outer_rep * reps_per_config makes for a
        // contiguous range < kMaxAttempts.
        const uint64_t seed = config.hash_key + outer_rep * reps_per_config;
        const uint32_t hash_key =
            static_cast<uint32_t>(RngStream(engine, seed)());
        per_worker[worker].MaybeBuild(keys, config, hash_key);
      });

      // Tasks are sorted by size, hence the first to succeed is the best.
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

  return PhastData();
}

#else   // HWY_TARGET == HWY_SCALAR
static PhastData BuildPhastImpl(const uint32_t*, size_t, ThreadPool&) {
  return PhastData();
}
#endif  // HWY_TARGET != HWY_SCALAR
}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_EXPORT(BuildPhastImpl);

HWY_CONTRIB_DLLEXPORT PhastData BuildPhast(const uint32_t* keys,
                                           size_t num_keys, ThreadPool& pool) {
  return HWY_DYNAMIC_DISPATCH(BuildPhastImpl)(keys, num_keys, pool);
}

}  // namespace hwy
#endif  // HWY_ONCE
