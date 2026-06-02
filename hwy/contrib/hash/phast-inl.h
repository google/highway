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

// PHAST: near-minimal perfect hashing for u32 keys.
//
// This variant does no bumping, which increases storage requirements: roughly
// 5 bits/key (2 keys per bucket and 4% headroom). The main design goal is
// high-throughput branchless and vectorized queries: 5-9 GB/s on AMD Milan and
// Turin. This will be memory bandwidth-bound unless cache-resident.
//
// Compared with Jenkins' https://burtleburtle.net/bob/hash/perfect.html,
// we use 3x as much space for ~2x query speedups (single seed lookup vs. two
// dependent lookups).
//
// For background on PHAST, see https://arxiv.org/abs/2504.17918.
// Queries first hash keys using a 32-bit permutation, which guarantees there
// are no collisions. Each hash selects a length 4K slice of the slots using
// Lemire's division-free modulo. The lower bits of the hash select a "bucket",
// i.e. an 8-bit seed value used to perturb the placement within the slice,
// which also depends on the upper hash bits.
//
// The builder chooses seeds for buckets in descending size order. It tries all
// possible 8-bit values, prioritizing those in sparse regions. We perform
// several retries in parallel with different global seeds, which are XORed
// with the u32 key being hashed.

#if defined(HIGHWAY_HWY_CONTRIB_HASH_PHAST_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_HASH_PHAST_INL_H_
#undef HIGHWAY_HWY_CONTRIB_HASH_PHAST_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_HASH_PHAST_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>  // snprintf

#include <algorithm>  // std::unique
#include <string>
#include <utility>

#include "hwy/aligned_allocator.h"
#include "hwy/contrib/hash/hash-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/contrib/sort/vqsort-inl.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/profiler.h"
#include "hwy/stats.h"

HWY_BEFORE_NAMESPACE();

namespace hwy {
namespace HWY_NAMESPACE {

// --------------------------------------------------------------------------
// Configuration

#pragma pack(push, 1)  // prevents false sharing
struct PhastConfig {
  PhastConfig() = default;
  PhastConfig(size_t num_keys_val, uint32_t keys_per_bucket_val = 2,
              uint32_t slice_length_val = 4096,
              uint32_t headroom_percent_val = 4, uint32_t max_retries_val = 300)
      : num_keys(static_cast<uint32_t>(num_keys_val)),
        slice_length(slice_length_val),
        keys_per_bucket(keys_per_bucket_val),
        headroom_percent(headroom_percent_val),
        max_retries(max_retries_val) {
    const uint32_t min_by_load = static_cast<uint32_t>(
        num_keys_val * (100 + headroom_percent_val) / 100 + 1);
    num_slots = HWY_MAX(min_by_load, num_keys + slice_length);
    // Number of valid starting positions for overlapping slices.
    num_slice_offsets = num_slots - slice_length + 1;

    // Power-of-two bucket count.
    num_buckets = 1u << hwy::CeilLog2(HWY_MAX(num_keys / keys_per_bucket, 1u));
  }

  std::string ToString() const {
    char buf[256];
    snprintf(
        buf, sizeof(buf),
        "Overhead=%zuK, buckets=%uK, slice=%2uK, key/bucket=%u, headroom=%u%%",
        ExtraBytes() / 1024, num_buckets / 1024, slice_length / 1024,
        keys_per_bucket, headroom_percent);
    return buf;
  }

  uint32_t BucketMask() const { return num_buckets - 1; }

  uint32_t num_keys = 0;
  uint32_t num_slots = 0;          // m: allocated space, including slack
  uint32_t num_buckets = 0;        // B: power of 2, ~num_keys / keys_per_bucket
  uint32_t slice_length = 0;       // L: power of 2
  uint32_t num_slice_offsets = 0;  // num_slots - slice_length + 1
  uint32_t keys_per_bucket = 0;    // lambda
  uint32_t headroom_percent = 0;
  uint32_t max_retries = 0;

  size_t ExtraBytes(size_t payload_bytes = sizeof(uint32_t)) const {
    return (num_slots - num_keys) * payload_bytes + num_buckets;
  }
  uint32_t SliceMask() const { return slice_length - 1; }
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

  // For moving into Phast.
  PackedSeeds(PackedSeeds&&) = default;
  PackedSeeds& operator=(PackedSeeds&&) = default;

  void Reset() {
    HWY_ASSERT(!bits_.empty());  // Must not call after moving from this.
    ZeroBytes(bits_.data(), bits_.size() * sizeof(bits_[0]));
  }

  template <class DU32, class VU32 = Vec<DU32>>
  HWY_INLINE VU32 Get(DU32 du32, VU32 bucket_idx) const {
    const RebindToSigned<DU32> di32;
    const VU32 word_idx = ShiftRight<2>(bucket_idx);
    const VU32 lane_idx = And(bucket_idx, Set(du32, 3));
    const VU32 words = GatherIndex(du32, bits_.data(), BitCast(di32, word_idx));
    const VU32 shift = ShiftLeft<3>(lane_idx);
    return And(Shr(words, shift), Set(du32, 0xFFu));
  }

  HWY_INLINE uint32_t Get(uint32_t bucket_idx) const {
    return (bits_[bucket_idx >> 2] >> ((bucket_idx & 3) * 8)) & 0xFF;
  }

  HWY_INLINE void SetSeed(uint32_t bucket_idx, uint32_t seed) {
    bits_[bucket_idx >> 2] |= (seed << ((bucket_idx & 3) * 8));
  }

 private:
  AlignedVector<uint32_t> bits_;
};

// --------------------------------------------------------------------------
// Placement function, used by query and build.

// 16-bit bijective hash. Avoids slower 32-bit mul.
template <class DU16, class VU16 = Vec<DU16>>
static HWY_INLINE VU16 Hash16(DU16 du16, VU16 x) {
  x = Xor(x, ShiftRight<8>(x));
  x = Mul(x, Set(du16, uint16_t{0x88B5u}));
  x = Xor(x, ShiftRight<7>(x));
  x = Mul(x, Set(du16, uint16_t{0xDB2Du}));
  x = Xor(x, ShiftRight<9>(x));
  return x;
}

// For 2*N seeds and hashes (to enable a 16-bit hash), returns a placement
// within the slice (caller must take modulo slice_length).
template <class DU32, class VU32 = Vec<DU32>>
static HWY_INLINE void Placement(DU32 du32, const VU32 hash0, const VU32 hash1,
                                 const VU32 seed0, const VU32 seed1,
                                 VU32& HWY_RESTRICT ofs0,
                                 VU32& HWY_RESTRICT ofs1) {
  const RepartitionToNarrow<DU32> du16;
  using VU16 = Vec<decltype(du16)>;

  // Odd/Even packing allows PromoteEvenTo.
  const VU16 seeds =
      OddEven(BitCast(du16, ShiftLeft<16>(seed1)), BitCast(du16, seed0));
  // Use upper 16 bits of hashes because the lower ~19 (for 1M keys and kpb=2)
  // already selected the bucket and thus seed. Odd lanes have hash1 >> 16,
  // even lanes have hash0 >> 16.
  const VU16 hashes =
      OddEven(BitCast(du16, hash1), BitCast(du16, ShiftRight<16>(hash0)));
  // XOR is weaker than ADD. Mul together does not work.
  const VU16 combined = Add(hashes, seeds);
  const VU16 hashed = Hash16(du16, combined);
  // Split back to u32: even u16 lanes -> ofs0, odd u16 lanes -> ofs1.
  ofs0 = PromoteEvenTo(du32, hashed);
  ofs1 = PromoteOddTo(du32, hashed);
}

// Variant for single query vector. About 6% lower latency.
template <class DU32, class VU32 = Vec<DU32>>
static HWY_INLINE VU32 Placement1(DU32 du32, const VU32 hash0,
                                  const VU32 seed0) {
  const RepartitionToNarrow<DU32> du16;
  using VU16 = Vec<decltype(du16)>;

  // Only even u16 lanes are used.
  const VU16 seeds = BitCast(du16, seed0);
  // Use upper 16 bits of hashes, see above.
  const VU16 hashes = BitCast(du16, ShiftRight<16>(hash0));
  // XOR is weaker than ADD. Mul together does not work.
  const VU16 combined = Add(hashes, seeds);
  const VU16 hashed = Hash16(du16, combined);
  // Caller will AND, no need to mask here.
  return BitCast(du32, hashed);
}

// --------------------------------------------------------------------------
// Phast: query-time structure

class Phast {
 public:
  Phast() = default;
  Phast(PhastConfig config, const Triple32 hash, PackedSeeds&& seeds_packed)
      : config_(config), hash_(hash), seeds_packed_(std::move(seeds_packed)) {}
  Phast(Phast&& other) = default;
  Phast& operator=(Phast&& other) = default;

  const PhastConfig& Config() const { return config_; }

  // Maps a single key to an index in [0, num_slots). 22 cycle latency on Milan.
  HWY_INLINE uint32_t operator()(uint32_t key) const {
    const ScalableTag<uint32_t> du32;
    return GetLane(Query1(du32, Set(du32, key)));
  }

  // Same, for a batch of keys. Considerably higher throughput than repeated
  // single queries: 9 GB/s on Turin.
  void QueryBatch(const uint32_t* HWY_RESTRICT keys, size_t num_keys,
                  uint32_t* HWY_RESTRICT indices) const {
    const ScalableTag<uint32_t> du32;
    using VU32 = Vec<decltype(du32)>;
    HWY_LANES_CONSTEXPR size_t N = Lanes(du32);

    size_t i = 0;
    if (HWY_LIKELY(num_keys >= 2 * N)) {
      for (; i <= num_keys - 2 * N; i += 2 * N) {
        VU32 v0 = Load(du32, keys + i + 0 * N);
        VU32 v1 = Load(du32, keys + i + 1 * N);
        VU32 idx0, idx1;
        Query2(du32, v0, v1, idx0, idx1);
        Store(idx0, du32, indices + i + 0 * N);
        Store(idx1, du32, indices + i + 1 * N);
      }
    }
    if (HWY_UNLIKELY(i != num_keys)) {
      const size_t remaining = num_keys - i;
      HWY_DASSERT(remaining < 2 * N);
      const size_t remaining1 = remaining <= N ? 0 : remaining - N;
      VU32 v0 = LoadN(du32, keys + i + 0 * N, remaining);
      VU32 v1 = LoadN(du32, keys + i + 1 * N, remaining1);
      VU32 idx0, idx1;
      Query2(du32, v0, v1, idx0, idx1);
      StoreN(idx0, du32, indices + i + 0 * N, remaining);
      StoreN(idx1, du32, indices + i + 1 * N, remaining1);
    }
  }

  // Used by the builder, which tries known seeds, and Query2.
  template <class DU32, class VU32 = Vec<DU32>>
  static HWY_INLINE void QueryWithSeeds(DU32 du32, const VU32 hash0,
                                        const VU32 hash1, const VU32 seed0,
                                        const VU32 seed1,
                                        const uint32_t num_slice_offsets,
                                        const uint32_t slice_mask, VU32& idx0,
                                        VU32& idx1) {
    // Compute slice offsets via Lemire modulo.
    const VU32 vnum_slice_offsets = Set(du32, num_slice_offsets);
    const VU32 slice_offset0 = MulHigh(hash0, vnum_slice_offsets);
    const VU32 slice_offset1 = MulHigh(hash1, vnum_slice_offsets);

    // Compute placements within slices.
    const VU32 vslice_mask = Set(du32, slice_mask);
    VU32 ofs0, ofs1;
    Placement(du32, hash0, hash1, seed0, seed1, ofs0, ofs1);
    idx0 = Add(slice_offset0, And(ofs0, vslice_mask));
    idx1 = Add(slice_offset1, And(ofs1, vslice_mask));
  }

 private:
  template <class DU32, class VU32 = Vec<DU32>>
  HWY_INLINE VU32 Query1(DU32 du32, const VU32 key) const {
    const VU32 hash = hash_.OneVec(du32, key);

    // Load 8-bit seeds from bucket.
    const VU32 bucket_mask = Set(du32, config_.BucketMask());
    const VU32 seed = seeds_packed_.Get(du32, And(hash, bucket_mask));

    // Compute slice offsets via Lemire modulo.
    const VU32 num_slice_offsets = Set(du32, config_.num_slice_offsets);
    const VU32 slice_offset = MulHigh(hash, num_slice_offsets);

    // Compute placements within slices.
    const VU32 slice_mask = Set(du32, config_.SliceMask());
    const VU32 ofs = Placement1(du32, hash, seed);
    return Add(slice_offset, And(ofs, slice_mask));
  }

  // Two vectors are slightly faster because they utilize all 16-bit lanes for
  // the second hash.
  template <class DU32, class VU32 = Vec<DU32>>
  HWY_INLINE void Query2(DU32 du32, VU32 key0, VU32 key1, VU32& idx0,
                         VU32& idx1) const {
    hash_.TwoVec(du32, key0, key1);

    // Load 8-bit seeds from bucket.
    const VU32 bucket_mask = Set(du32, config_.BucketMask());
    const VU32 seed0 = seeds_packed_.Get(du32, And(key0, bucket_mask));
    const VU32 seed1 = seeds_packed_.Get(du32, And(key1, bucket_mask));

    QueryWithSeeds(du32, key0, key1, seed0, seed1, config_.num_slice_offsets,
                   config_.SliceMask(), idx0, idx1);
  }

  PhastConfig config_ = {};
  Triple32 hash_;
  PackedSeeds seeds_packed_;
};

// --------------------------------------------------------------------------
// Occupancy

// Tracks which slots are occupied; one bit per slot, supports Gather.
class Occupancy {
 public:
  explicit Occupancy(size_t num_slots)
      : bits_(DivCeil(num_slots, size_t{32})) {}

  void Reset() { ZeroBytes(bits_.data(), bits_.size() * sizeof(bits_[0])); }

  bool IsOccupied(uint32_t pos) const {
    return (bits_[pos >> 5] >> (pos & 31)) & 1;
  }

  template <class DU32, class VU32 = Vec<DU32>>
  Mask<DU32> Unoccupied(DU32 du32, VU32 pos) const {
    const RebindToSigned<DU32> di32;
    const VU32 word_idx = ShiftRight<5>(pos);
    const VU32 bit_idx = And(pos, Set(du32, 31));
    const VU32 bit_mask = Shl(Set(du32, 1), bit_idx);
    const VU32 words = GatherIndex(du32, bits_.data(), BitCast(di32, word_idx));
    // Unoccupied where the bit is NOT set.
    return Eq(And(words, bit_mask), Zero(du32));
  }

  void SetOccupied(uint32_t pos) {
    HWY_DASSERT(!IsOccupied(pos));
    bits_[pos >> 5] |= 1u << (pos & 31);
  }

  // Returns occupancy in the 32-position neighborhood of `pos`. This steers
  // placements toward sparse neighborhoods. 32-bit words are required for SIMD,
  // otherwise we could use 64. One popcount outperforms 3x64 neighborhoods.
  // First-viable and max_pos tiebreakers were worse.
  uint32_t NeighborDensity(uint32_t pos) const {
    return hwy::PopCount(bits_[pos >> 5]);
  }

 private:
  AlignedVector<uint32_t> bits_;
};

// --------------------------------------------------------------------------
// PhastBuilder

// Per-thread.
class PhastBuilder {
 public:
  PhastBuilder(PhastConfig config)
      : config_(config),
        num_keys_(config.num_keys),  // for brevity
        num_buckets_(config.num_buckets),
        engine_(/*deterministic=*/true),
        occupancy_(config.num_slots),
        seeds_packed_(config.num_buckets) {
    hash_for_key_idx_.resize(num_keys_);
    all_pos_.resize(size_t{kMaxHashesPerBucket} * 256);
    num_hashes_for_bucket_.resize(num_buckets_);
    key_idx_for_pos_.resize(num_keys_);
    total_hashes_before_bucket_.resize(num_buckets_ + 1);
    write_pos_.resize(num_buckets_);
    bucket_idx_largest_first_.resize(num_buckets_);
  }

  // This can be used for validating keys are unique without another copy.
  uint32_t* MutableHashes() { return hash_for_key_idx_.data(); }

  // Attempts to build with a given hash. After this, RankOr0() is 0 on
  // success, or the rank of the first bucket that failed to be seeded, which is
  // a measure of progress or how close we came to success.
  // The caller parallelizes; each thread tries one global seed.
  void MaybeBuild(const uint32_t* keys, uint32_t global_seed) {
    const size_t num_keys = config_.num_keys;
    HWY_ALIGN uint32_t seed_candidates[256];

    seeds_packed_.Reset();
    occupancy_.Reset();

    hash_ = Triple32(engine_, global_seed);
    HashArray(hash_, keys, MutableHashes(), num_keys);
    const size_t num_buckets = PopulateBuckets(num_keys);

    // We handle buckets with >2, 2 and 1 key separately; the latter are fairly
    // common given our keys_per_bucket of 2..3, which is further rounded down
    // by rounding number of buckets up to a power of 2.
    const size_t num_le1_buckets =
        num_buckets_with_size_[0] + num_buckets_with_size_[1];
    const size_t num_ge1_buckets = num_buckets - num_buckets_with_size_[0];
    const size_t num_ge2_buckets = num_buckets - num_le1_buckets;
    const size_t num_gt2_buckets = num_ge2_buckets - num_buckets_with_size_[2];

    // Best-seed: try all 256 candidates, pick the one whose positions land in
    // the least-occupied 32-bit words (maximizes future freedom). Lowest
    // sum_pos and max_pos heuristics performed worse. Using density as a
    // second bucket sort key also did not help. We do limit the search to
    // "candidate" seeds where all keys' slots are unoccupied.

    // General case of buckets with > 2 keys.
    {
      PROFILER_ZONE("Main builder size > 2");
      for (uint32_t rank = 0; rank < num_gt2_buckets; ++rank) {
        const uint32_t b = bucket_idx_largest_first_[rank];
        const size_t b_begin = total_hashes_before_bucket_[b];
        const size_t b_size = num_hashes_for_bucket_[b];
        HWY_DASSERT(b_size > 2);

        ComputeBucketPos(b_begin, b_size);
        const size_t num_candidates =
            WriteSeedCandidates(b_size, seed_candidates);

        uint32_t best_seed = 0;
        uint32_t best_cost = ~0u;
        for (size_t cand_idx = 0; cand_idx < num_candidates; ++cand_idx) {
          const uint32_t seed = seed_candidates[cand_idx];

          // Pairwise collision check: faster than sort+unique for small
          // buckets.
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
          }
        }  // foreach candidate seed

        if (HWY_UNLIKELY(best_cost == ~0u)) {
          rank_or_0_ = rank + 1;
          return;
        }

        // Commit the best seed's positions.
        for (size_t idx = 0; idx < b_size; ++idx) {
          occupancy_.SetOccupied(all_pos_[idx * 256 + best_seed]);
        }
        seeds_packed_.SetSeed(b, best_seed);
      }  // foreach rank
    }

    {
      PROFILER_ZONE("Main builder size = 2");
      for (uint32_t rank = num_gt2_buckets; rank < num_ge2_buckets; ++rank) {
        const uint32_t b = bucket_idx_largest_first_[rank];
        const size_t b_begin = total_hashes_before_bucket_[b];
        HWY_DASSERT(num_hashes_for_bucket_[b] == 2);

        ComputeBucketPos(b_begin, 2);
        const size_t num_candidates = WriteSeedCandidates(2, seed_candidates);

        uint32_t best_seed = 0;
        uint32_t best_cost = ~0u;
        // Could vectorize/fuse with WriteSeedCandidates, but this is already
        // fast compared to the general case.
        for (size_t cand_idx = 0; cand_idx < num_candidates; ++cand_idx) {
          const uint32_t seed = seed_candidates[cand_idx];
          const uint32_t pos0 = all_pos_[seed];
          const uint32_t pos1 = all_pos_[256 + seed];
          if (pos0 == pos1) continue;

          const uint32_t cost = occupancy_.NeighborDensity(pos0) +
                                occupancy_.NeighborDensity(pos1);
          if (cost < best_cost) {
            best_cost = cost;
            best_seed = seed;
            if (cost == 0) break;
          }
        }

        if (HWY_UNLIKELY(best_cost == ~0u)) {
          rank_or_0_ = rank + 1;
          return;
        }

        // Commit the best seed's positions.
        occupancy_.SetOccupied(all_pos_[best_seed]);
        occupancy_.SetOccupied(all_pos_[256 + best_seed]);
        seeds_packed_.SetSeed(b, best_seed);
      }  // foreach rank
    }

    {
      PROFILER_ZONE("Main builder: size 1 buckets");
      for (uint32_t rank = num_ge2_buckets; rank < num_ge1_buckets; ++rank) {
        const uint32_t b = bucket_idx_largest_first_[rank];
        const size_t b_begin =
            static_cast<size_t>(total_hashes_before_bucket_[b]);
        HWY_DASSERT(num_hashes_for_bucket_[b] == 1);

        ComputeBucketPos(b_begin, 1);
        // Could terminate early at the first candidate, but this is already
        // fast even compared to the size=2 case.
        const size_t num_candidates = WriteSeedCandidates(1, seed_candidates);
        if (HWY_UNLIKELY(num_candidates == 0)) {  // All occupied - fail.
          rank_or_0_ = rank + 1;
          return;
        }

        // Candidates are unoccupied by definition, just use the first one.
        const uint32_t best_seed = seed_candidates[0];
        occupancy_.SetOccupied(all_pos_[best_seed]);
        seeds_packed_.SetSeed(b, best_seed);
      }  // foreach rank
    }

    rank_or_0_ = 0;  // Success.
  }

  // Must not call before MaybeBuild().
  uint32_t RankOr0() const { return rank_or_0_; }

  Phast Take() { return Phast(config_, hash_, std::move(seeds_packed_)); }

 private:
  // Computes positions for all keys in a bucket, storing at local indices
  // [0, b_size) * 256 in all_pos_.
  void ComputeBucketPos(size_t b_begin, size_t b_size) {
    const ScalableTag<uint32_t> du32;
    HWY_LANES_CONSTEXPR size_t NU32 = Lanes(du32);
    using VU32 = Vec<decltype(du32)>;

    for (size_t idx = 0; idx < b_size; ++idx) {
      const uint32_t key_idx = key_idx_for_pos_[b_begin + idx];
      const VU32 c = Set(du32, hash_for_key_idx_[key_idx]);
      for (uint32_t seed = 0; seed < 256; seed += 2 * NU32) {
        const VU32 s0 = Iota(du32, seed);
        const VU32 s1 = Iota(du32, seed + NU32);
        VU32 pos0, pos1;
        Phast::QueryWithSeeds(du32, c, c, s0, s1, config_.num_slice_offsets,
                              config_.SliceMask(), pos0, pos1);
        StoreU(pos0, du32, &all_pos_[idx * 256 + seed]);
        StoreU(pos1, du32, &all_pos_[idx * 256 + seed + NU32]);
      }
    }
  }

  size_t BucketIdxFromKeyIdx(size_t key_idx) const {
    return hash_for_key_idx_[key_idx] & config_.BucketMask();
  }

  static constexpr size_t kMaxHashesPerBucket = 32;

  size_t PopulateBuckets(size_t num_keys) {
    PROFILER_FUNC;
    const size_t num_buckets = config_.num_buckets;

    // O(N) scan: increment counts for each bucket.
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
    CopyBytes(total_hashes_before_bucket_.data(), write_pos_.data(),
              num_buckets * sizeof(uint32_t));
    for (uint32_t key_idx = 0; key_idx < num_keys; ++key_idx) {
      const uint32_t bucket_idx = BucketIdxFromKeyIdx(key_idx);
      key_idx_for_pos_[write_pos_[bucket_idx]++] = key_idx;
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

    return num_buckets;
  }

  // Writes seeds where ALL keys at local indices [0, b_size) are unoccupied.
  size_t WriteSeedCandidates(uint32_t b_size,
                             uint32_t* HWY_RESTRICT seed_candidates) const {
    const ScalableTag<uint32_t> du32;
    HWY_LANES_CONSTEXPR size_t NU32 = Lanes(du32);
    using VU32 = Vec<decltype(du32)>;
    using MU32 = Mask<decltype(du32)>;

    size_t num_candidates = 0;
    for (uint32_t seed = 0; seed < 256; seed += 2 * NU32) {
      const VU32 s0 = Iota(du32, seed);
      const VU32 s1 = Iota(du32, seed + NU32);
      MU32 avail0 = SetMask(du32, true);
      MU32 avail1 = SetMask(du32, true);
      for (size_t idx = 0; idx < b_size; ++idx) {
        const VU32 pos0 = LoadU(du32, &all_pos_[idx * 256 + seed]);
        const VU32 pos1 = LoadU(du32, &all_pos_[idx * 256 + seed + NU32]);
        avail0 = And(avail0, occupancy_.Unoccupied(du32, pos0));
        avail1 = And(avail1, occupancy_.Unoccupied(du32, pos1));
      }
      num_candidates +=
          CompressStore(s0, avail0, du32, seed_candidates + num_candidates);
      num_candidates +=
          CompressStore(s1, avail1, du32, seed_candidates + num_candidates);
    }

    return num_candidates;
  }

  PhastConfig config_;
  size_t num_keys_;
  size_t num_buckets_;
  AesCtrEngine engine_;
  Triple32 hash_;

  AlignedVector<uint32_t> hash_for_key_idx_;  // [num_keys]
  // Indexed by [local_idx * 256 + seed], computed per bucket.
  AlignedVector<uint32_t> all_pos_;  // [kMaxHashesPerBucket*256]

  AlignedVector<uint8_t> num_hashes_for_bucket_;        // [num_buckets]
  AlignedVector<uint32_t> total_hashes_before_bucket_;  // [num_buckets + 1]
  AlignedVector<uint32_t> write_pos_;                   // write cursor, copied
  uint32_t num_buckets_with_size_[kMaxHashesPerBucket + 1];

  // Access via key_idx_for_pos_[total_hashes_before_bucket_[bucket_idx] + i]
  // for i-th key in bucket_idx.
  AlignedVector<uint32_t> key_idx_for_pos_;  // [num_keys]
  // Bucket indices sorted by decreasing bucket size.
  AlignedVector<uint32_t> bucket_idx_largest_first_;  // [num_buckets]

  Occupancy occupancy_;
  PackedSeeds seeds_packed_;

  uint32_t rank_or_0_;  // uninitialized until MaybeBuild() is called.
};

struct PhastStats {
  bool success;
  // Only populated if success:
  size_t round = ~size_t{0};
  size_t worker = ~size_t{0};

  // Only populated if !success:
  Stats s_rank;
};

// Builds from a set of distinct keys. Returns empty Phast (num_keys == 0) if
// max_retries is exceeded. Checks global_seed in parallel, one per worker.
static HWY_MAYBE_UNUSED Phast BuildPhast(const uint32_t* keys,
                                         PhastConfig config, ThreadPool& pool,
                                         PhastStats* stats = nullptr) {
  const size_t num_workers = pool.NumWorkers();
  AlignedVector<PhastBuilder> per_worker;
  per_worker.reserve(num_workers);
  for (size_t i = 0; i < num_workers; ++i) {
    per_worker.emplace_back(config);
  }

  // Ensure keys are distinct, otherwise, building would fail. Sort the hashes
  // because `keys` are immutable, and the hash function is a permutation,
  // hence hash collisions imply duplicate keys.
  AesCtrEngine engine(/*deterministic=*/true);
  HashArray(Triple32(engine, 0), keys, per_worker[0].MutableHashes(),
            config.num_keys);
  VQSortStatic(per_worker[0].MutableHashes(), config.num_keys, SortAscending());
  uint32_t* end = per_worker[0].MutableHashes() + config.num_keys;
  HWY_ASSERT_M(end == std::unique(per_worker[0].MutableHashes(), end),
               "Collision detected");

  Stats s_rank;
  // One seed per worker, then check if any succeeded, then repeat.
  const size_t max_rounds = DivCeil(config.max_retries, num_workers);
  for (size_t round = 0; round < max_rounds; ++round) {
    pool.Run(0, num_workers, [&](uint64_t task_idx, size_t worker) {
      HWY_ASSERT(task_idx == worker);  // one task per worker
      PhastBuilder& builder = per_worker[worker];
      const uint32_t global_seed =
          static_cast<uint32_t>(round * num_workers + task_idx);
      builder.MaybeBuild(keys, global_seed);
    });

    for (size_t worker = 0; worker < num_workers; ++worker) {
      uint32_t rank = per_worker[worker].RankOr0();
      if (rank == 0) {  // Success
        if (stats != nullptr) {
          stats->success = true;
          stats->round = round;
          stats->worker = worker;
        }
        return per_worker[worker].Take();
      }
      s_rank.Notify(rank);
    }
  }

  if (stats != nullptr) {
    stats->success = false;
    stats->s_rank = s_rank;
  }
  return Phast();
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_HASH_PHAST_INL_H_
