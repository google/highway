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
// This variant avoids bumping, hence is branchless. The main design goal is
// high-throughput SIMD queries: 2.8 and 7.8 GB/s on AMD Milan and Turin for 1M
// queries. This is actually memory-bound, not compute-bound. The tradeoff is
// higher storage requirements: 4.8 bits/key (2 keys per bucket, plus only 2%
// headroom, i.e. extra slots).
//
// Jenkins' https://burtleburtle.net/bob/hash/perfect.html manages perfect
// hashing (i.e. 0% headroom) in 3-8 bits/key, but also requires two
// dependent lookups, vs. our single seed lookup in an L2-resident array.
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
// with the u32 key being hashed. On failure, similar to Cuckoo hashing, we
// redo prior buckets that overlap with the failed bucket. This halves
// required headroom from 4% to 2% for 1M keys.

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
#include <utility>    // std::move

#include "hwy/aligned_allocator.h"
#include "hwy/contrib/algo/find-inl.h"
#include "hwy/contrib/hash/hash-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/contrib/sort/vqsort-inl.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/profiler.h"
#include "hwy/stats.h"

#if HWY_TARGET != HWY_SCALAR
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

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
        keys_per_bucket_(keys_per_bucket),
        headroom_percent_(headroom_percent),
        max_retries_(max_retries) {
    num_slots_ =
        static_cast<uint32_t>(num_keys * (100 + headroom_percent) / 100 + 1);
    // Prevent underflow in num_slice_offsets_.
    num_slots_ = HWY_MAX(num_slots_, SliceLength());
    // Number of starting positions for overlapping slices.
    num_slice_offsets_ = num_slots_ - SliceLength() + 1;

    const uint32_t min_buckets = HWY_MAX(num_keys_ / keys_per_bucket_, 1u);
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
  uint32_t keys_per_bucket_ = 0;    // aka lambda
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
// Phast: query-time structure

class Phast {
 public:
  Phast() = default;
  Phast(PhastConfig config, const Triple32 hash, PackedSeeds&& seeds_packed)
      : config_(config), hash_(hash), seeds_packed_(std::move(seeds_packed)) {}
  Phast(Phast&& other) = default;
  Phast& operator=(Phast&& other) = default;

  const PhastConfig& Config() const { return config_; }
  bool IsEmpty() const { return config_.NumKeys() == 0; }

  // Maps a single key to an index in [0, num_slots). 13-17 cycle latency on
  // Turin and Milan.
  HWY_INLINE uint32_t operator()(uint32_t key) const {
    const uint32_t hash = hash_(key);
    const uint32_t bucket_idx = hash & config_.BucketMask();
    const uint32_t seed = seeds_packed_.Get(bucket_idx);
    return QueryWithSeeds(hash, seed, config_);
  }

  // Same, for a batch of keys. Considerably higher throughput than repeated
  // single queries: 7.8 GB/s on Turin for 1M keys.
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

  // For internal use by the builder and Query2:

  static HWY_INLINE uint32_t QueryWithSeeds(const uint32_t hash,
                                            const uint32_t seed,
                                            const PhastConfig& config) {
    const uint32_t slice_offset = LemireMod(hash, config.NumSliceOffsets());
    const uint32_t within_slice = Placement(hash, seed) & config.SliceMask();
    return slice_offset + within_slice;
  }

  template <class DU32, class VU32 = Vec<DU32>>
  static HWY_INLINE void QueryWithSeeds(DU32 du32, const VU32 hash0,
                                        const VU32 hash1, const VU32 seed0,
                                        const VU32 seed1,
                                        const PhastConfig& config, VU32& idx0,
                                        VU32& idx1) {
    // Compute slice offsets via Lemire modulo.
    const VU32 num_slice_offsets = Set(du32, config.NumSliceOffsets());
    const VU32 slice_offset0 = MulHigh(hash0, num_slice_offsets);
    const VU32 slice_offset1 = MulHigh(hash1, num_slice_offsets);

    // Compute placements within slices.
    const VU32 slice_mask = Set(du32, config.SliceMask());
    VU32 ofs0, ofs1;
    Placement(du32, hash0, hash1, seed0, seed1, ofs0, ofs1);
    idx0 = Add(slice_offset0, And(ofs0, slice_mask));
    idx1 = Add(slice_offset1, And(ofs1, slice_mask));
  }

 private:
  // 16-bit bijective hash. Avoids slower 32-bit mul and processes twice as many
  // elements per vector. Discovered via hash_prospector16. Leads to higher
  // builder success rates than the prior XMXMX approach. Choosing multipliers
  // based on `x` is barely better and not worth the complexity/cost.
  template <class DU16, class VU16 = Vec<DU16>>
  static HWY_INLINE VU16 Hash16(DU16 du16, VU16 x) {
    x = Xor(x, ShiftRight<8>(x));
    x = Add(x, Mul(Set(du16, 0xca32), Mul(x, x)));
    x = Xor(x, ShiftRight<12>(x));
    x = Mul(x, Set(du16, 0x3929));
    return x;
  }

  // Scalar version for builder.
  static HWY_INLINE uint16_t Hash16(uint16_t x) {
    x = static_cast<uint16_t>(x ^ (x >> 8));
    const uint16_t x2 = static_cast<uint16_t>(x * x);
    x = static_cast<uint16_t>(x + static_cast<uint16_t>(x2 * 0xca32));
    x = static_cast<uint16_t>(x ^ (x >> 12));
    x = static_cast<uint16_t>(x * 0x3929u);
    return x;
  }

  // For 2*N seeds and hashes (to enable a 16-bit hash), returns a placement
  // within the slice (caller must take modulo slice_length).
  template <class DU32, class VU32 = Vec<DU32>>
  static HWY_INLINE void Placement(DU32 du32, const VU32 hash0,
                                   const VU32 hash1, const VU32 seed0,
                                   const VU32 seed1, VU32& HWY_RESTRICT ofs0,
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

  // Scalar version for builder.
  static HWY_INLINE uint32_t Placement(const uint32_t hash,
                                       const uint32_t seed) {
    const uint16_t combined = static_cast<uint16_t>((hash >> 16) + seed);
    return Hash16(combined);  // caller will AND
  }

  // Two vectors have higher throughput because they utilize all 16-bit lanes
  // for Hash16. Called by QueryBatch.
  template <class DU32, class VU32 = Vec<DU32>>
  HWY_INLINE void Query2(DU32 du32, VU32 key0, VU32 key1, VU32& idx0,
                         VU32& idx1) const {
    hash_.TwoVec(du32, key0, key1);

    // Load 8-bit seeds from bucket.
    const VU32 bucket_mask = Set(du32, config_.BucketMask());
    const VU32 seed0 = seeds_packed_.Get(du32, And(key0, bucket_mask));
    const VU32 seed1 = seeds_packed_.Get(du32, And(key1, bucket_mask));

    QueryWithSeeds(du32, key0, key1, seed0, seed1, config_, idx0, idx1);
  }

  PhastConfig config_ = {};
  Triple32 hash_;
  PackedSeeds seeds_packed_;
};

// --------------------------------------------------------------------------

// Tracks which slots are occupied during building; holds one bit per slot.
class Occupancy {
 public:
  explicit Occupancy(size_t num_slots)
      : bits_(DivCeil(num_slots, size_t{32})) {}

  void Reset() { ZeroBytes(bits_.data(), bits_.size() * sizeof(bits_[0])); }

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
        num_keys_(config.NumKeys()),  // for brevity
        num_buckets_(config.NumBuckets()),
        engine_(/*deterministic=*/true),
        occupancy_(config.NumSlots()),
        seeds_packed_(config.NumBuckets()) {
    hash_for_key_idx_.resize(num_keys_);
    all_pos_.resize(size_t{kMaxHashesPerBucket} * 256);
    num_hashes_for_bucket_.resize(num_buckets_);
    key_idx_for_pos_.resize(num_keys_);
    total_hashes_before_bucket_.resize(num_buckets_ + 1);
    write_pos_.resize(num_buckets_);
    bucket_idx_largest_first_.resize(num_buckets_);
    committed_seeds_.resize(num_buckets_);
  }

  // This can be used for validating keys are unique without another copy.
  uint32_t* MutableHashes() { return hash_for_key_idx_.data(); }

  // Attempts to build with a given hash. After this, RankOr0() is 0 on
  // success, or the rank of the first bucket that failed to be seeded, which is
  // a measure of progress or how close we came to success. On failure, first
  // try increasing the config's max retries, then increase headroom by several
  // percent. You may also try tuning the constants in TryCuckooSwap. The
  // caller parallelizes; each thread tries one global seed. Expect 1-2 sec for
  // 1M keys.
  void MaybeBuild(const uint32_t* keys, const uint32_t global_seed) {
    PROFILER_FUNC;
    HWY_ALIGN uint32_t seed_candidates[256];

    seeds_packed_.Reset();
    occupancy_.Reset();

    hash_ = Triple32(engine_, global_seed);
    {
      PROFILER_ZONE("HashArray");
      HashArray(hash_, keys, MutableHashes(), num_keys_);
    }
    const size_t num_buckets = PopulateBuckets(num_keys_);

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
        rank_or_0_ = rank + 1;  // Failed, report how close we came.
        return;
      }
    }

    rank_or_0_ = 0;  // Success.
  }

  // Must not call before MaybeBuild().
  uint32_t RankOr0() const { return rank_or_0_; }

  Phast Take() { return Phast(config_, hash_, std::move(seeds_packed_)); }

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
    return Phast::QueryWithSeeds(hash, seed, config_);
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
      for (uint32_t seed = 0; seed < 256; seed += 2 * NU32) {
        const VU32 s0 = Iota(du32, seed);
        const VU32 s1 = Iota(du32, seed + NU32);
        VU32 pos0, pos1;
        Phast::QueryWithSeeds(du32, hash, hash, s0, s1, config_, pos0, pos1);
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
    seeds_packed_.SetSeed(bucket_idx_largest_first_[rank], best_seed);
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
    seeds_packed_.Clear(bucket_idx);
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
    seeds_packed_.SetSeed(bucket_idx, seed);
  }

  size_t BucketIdxFromKeyIdx(size_t key_idx) const {
    return hash_for_key_idx_[key_idx] & config_.BucketMask();
  }

  static constexpr size_t kMaxHashesPerBucket = 16;

  size_t PopulateBuckets(size_t num_keys) {
    PROFILER_FUNC;
    const size_t num_buckets = config_.NumBuckets();

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

  // Writes seed candidates for which slots are unoccupied for all bucket keys
  // with relative index [0, b_size).
  size_t WriteSeedCandidates(size_t b_size,
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
  AlignedVector<uint8_t> committed_seeds_;  // seed per rank, for backtracking

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

// Builds from a set of distinct keys. Returns Phast with IsEmpty() if
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
  const size_t num_keys = config.NumKeys();
  HashArray(Triple32(engine, 0), keys, per_worker[0].MutableHashes(), num_keys);
  VQSortStatic(per_worker[0].MutableHashes(), num_keys, SortAscending());
  uint32_t* end = per_worker[0].MutableHashes() + num_keys;
  HWY_ASSERT_M(end == std::unique(per_worker[0].MutableHashes(), end),
               "Collision detected");

  Stats s_rank;
  // One seed per worker, then check if any succeeded, then repeat.
  const size_t max_rounds = DivCeil(config.MaxRetries(), num_workers);
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
#endif  // HWY_TARGET != HWY_SCALAR

#endif  // HIGHWAY_HWY_CONTRIB_HASH_PHAST_INL_H_
