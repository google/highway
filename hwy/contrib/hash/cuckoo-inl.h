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

// Static cuckoo hashing for uint32_t keys.
//
// Keys are assigned to 16-slot buckets (64 bytes = one cache line) via two
// hash functions (primary and secondary). Build-time placement uses
// Hopcroft-Karp maximum bipartite matching. Queries use SIMD to compare all
// 16 slots in a bucket simultaneously.
//
// Usage:
//   1. Call CuckooBuild() with your keys and desired epsilon.
//   2. The returned CuckooTable supports QueryOne() and QueryBatch().

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <deque>
#include <utility>
#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/cache_control.h"
#include "hwy/timer.h"

#if defined(HIGHWAY_HWY_CONTRIB_HASH_CUCKOO_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_HASH_CUCKOO_INL_H_
#undef HIGHWAY_HWY_CONTRIB_HASH_CUCKOO_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_HASH_CUCKOO_INL_H_
#endif

#include "hwy/contrib/algo/find-inl.h"
#include "hwy/contrib/hash/hash-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/highway.h"

#if HWY_TARGET != HWY_SCALAR
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

template <typename V>
HWY_INLINE auto GetRaw(V& v, int) -> decltype(v.raw)& {
  return v.raw;
}

template <typename V>
HWY_INLINE auto GetRaw(const V& v, int) -> const decltype(v.raw)& {
  return v.raw;
}

template <typename V>
HWY_INLINE V& GetRaw(V& v, ...) {
  return v;
}

template <typename V>
HWY_INLINE const V& GetRaw(const V& v, ...) {
  return v;
}

// --------------------------------------------------------------------------
// CuckooConfig

class CuckooConfig {
 public:
  CuckooConfig() = default;
  explicit CuckooConfig(uint32_t num_keys, double epsilon = 0.25)
      : num_keys_(num_keys), epsilon_(epsilon) {
    // Compute number of slots, rounded up to a multiple of kBucketSize.
    const uint32_t raw_slots =
        static_cast<uint32_t>(num_keys * (1.0 + epsilon)) + 1;
    const uint32_t min_buckets = DivCeil(raw_slots, kBucketSize);
    num_buckets_ = 1u << hwy::CeilLog2(HWY_MAX(min_buckets, 1u));
    num_slots_ = num_buckets_ * kBucketSize;
    bucket_mask_ = num_buckets_ - 1;
    fprintf(stderr, "Cuckoo hashing num buckets = %d\n", num_buckets_);
  }

  uint32_t NumKeys() const { return num_keys_; }
  uint32_t NumSlots() const { return num_slots_; }
  uint32_t NumBuckets() const { return num_buckets_; }
  uint32_t BucketMask() const { return bucket_mask_; }
  double Epsilon() const { return epsilon_; }

  static constexpr uint32_t kBucketSize = 16;  // slots per bucket
  static constexpr uint32_t kBucketBytes = kBucketSize * sizeof(uint32_t);
  // Sentinel value for empty slots. We reserve this value and keys must not
  // equal it.
  static constexpr uint32_t kEmpty = 0xFFFFFFFFu;
  static constexpr uint32_t kUnmatched = 0xFFFFFFFFu;

 private:
  uint32_t num_keys_ = 0;
  uint32_t num_slots_ = 0;
  uint32_t num_buckets_ = 0;
  uint32_t bucket_mask_ = 0;
  double epsilon_ = 0.0;
};

// --------------------------------------------------------------------------
// CuckooTable: query-time structure

class CuckooTable {
 public:
  CuckooTable() = default;
  CuckooTable(CuckooConfig config, Triple32 hash_primary,
              Triple32 hash_secondary, AlignedVector<uint32_t>&& slots,
              uint32_t num_primary)
      : config_(config),
        hash_primary_(hash_primary),
        hash_secondary_(hash_secondary),
        slots_(std::move(slots)),
        num_primary_(num_primary) {}

  CuckooTable(CuckooTable&&) = default;
  CuckooTable& operator=(CuckooTable&&) = default;

  bool IsEmpty() const { return config_.NumKeys() == 0; }
  const CuckooConfig& Config() const { return config_; }

  size_t AllocatedBytes() const { return slots_.size() * sizeof(uint32_t); }

  // Returns the number of keys placed in their primary bucket.
  uint32_t NumPrimary() const { return num_primary_; }

  // Query a single bucket (16 slots) for `key`.
  HWY_INLINE bool QueryBucket(uint32_t key, uint32_t b) const {
    const CappedTag<uint32_t, 16> d;
    HWY_LANES_CONSTEXPR size_t N = Lanes(d);
    const auto vkey = Set(d, key);
    const uint32_t* base = slots_.data() + b;
    auto any_eq = MaskFalse(d);
    for (size_t i = 0; i < 16; i += N) {
      any_eq = Or(any_eq, Eq(vkey, Load(d, base + i)));
    }
    return !AllFalse(d, any_eq);
  }

  // Query a single key. Returns true if found.
  HWY_INLINE bool QueryOne(uint32_t key) const {
    if (QueryBucket(key, PrimaryBucketOffset(key))) return true;
    return QueryBucket(key, SecondaryBucketOffset(key));
  }

  // SIMD set membership for N u32 keys, checking both primary and secondary
  // buckets. For small epsilon (precomputes both bucket offsets upfront).
  // Returns "not-found" mask (true = key NOT in set), matching Cuckoo2x2.
  // `keys` must point to N aligned uint32_t values.
  template <class DU32, class MU32 = Mask<DU32>>
  HWY_INLINE MU32 QueryBatchSmallEpsilon(
      DU32 du32, const uint32_t* HWY_RESTRICT keys) const {
    using VU32 = Vec<DU32>;
    HWY_LANES_CONSTEXPR size_t N = Lanes(du32);
    const uint32_t bucket_mask = config_.BucketMask();
    const VU32 vmask = Set(du32, bucket_mask);
    const uint32_t* base_slots = slots_.data();

    const VU32 vkeys = Load(du32, keys);

    HWY_ALIGN uint32_t pri_offsets[MaxLanes(du32)];
    HWY_ALIGN uint32_t sec_offsets[MaxLanes(du32)];

    // Compute primary bucket offsets for all N lanes.
    VU32 h_pri = hash_primary_.OneVec(du32, vkeys);
    const VU32 b_pri = ShiftLeft<4>(And(h_pri, vmask));
    Store(b_pri, du32, pri_offsets);

    // Compute secondary bucket offsets for all N lanes.
    VU32 h_sec = hash_secondary_.OneVec(du32, vkeys);
    const VU32 b_sec = ShiftLeft<4>(And(h_sec, vmask));
    Store(b_sec, du32, sec_offsets);

    if constexpr (kPrefetchMode == PrefetchMode::kGather) {
      const RebindToSigned<decltype(du32)> di32;
      VU32 g0 = GatherIndex(du32, base_slots, BitCast(di32, b_pri));
#if defined(HWY_X86_GCC_INLINE_ASM_VEC_CONSTRAINT)
      asm volatile(
          ""
          : "+" HWY_X86_GCC_INLINE_ASM_VEC_CONSTRAINT(GetRaw(g0, 0))::"memory");
#elif HWY_COMPILER_GCC || HWY_COMPILER_CLANG
      asm volatile("" : : "g"(GetRaw(g0, 0)) : "memory");
#endif
    }
    if constexpr (kPrefetchMode == PrefetchMode::kPrefetch) {
      for (size_t lane = 0; lane < N; ++lane) {
        hwy::Prefetch(base_slots + pri_offsets[lane]);
      }
    }

    // Pass 1: check all primary buckets.
    HWY_ALIGN uint32_t found_arr[MaxLanes(du32)];
    for (size_t lane = 0; lane < N; ++lane) {
      found_arr[lane] =
          QueryBucket(keys[lane], pri_offsets[lane]) ? 1u : 0u;
    }

    // Pass 2: check secondary buckets only for lanes that missed.
    const VU32 found = Load(du32, found_arr);
    const VU32 vzero = Zero(du32);
    const auto miss = Eq(found, vzero);
    HWY_ALIGN uint32_t miss_idx[MaxLanes(du32)];
    const size_t num_miss = CompressStore(Iota(du32, 0), miss, du32, miss_idx);
    for (size_t j = 0; j < num_miss; ++j) {
      const uint32_t lane = miss_idx[j];
      found_arr[lane] =
          QueryBucket(keys[lane], sec_offsets[lane]) ? 1u : 0u;
    }

    // Return not-found mask: true = key NOT in set.
    const VU32 result = Load(du32, found_arr);
    return Eq(result, vzero);
  }

  // SIMD set membership for N u32 keys. Returns "not-found" mask
  // (true = key NOT in set), matching Cuckoo2x2::operator().
  // `keys` must point to N aligned uint32_t values.
  template <class DU32, class MU32 = Mask<DU32>>
  HWY_INLINE MU32 QueryBatch(DU32 du32,
                              const uint32_t* HWY_RESTRICT keys) const {
    using VU32 = Vec<DU32>;
    HWY_LANES_CONSTEXPR size_t N = Lanes(du32);
    const uint32_t bucket_mask = config_.BucketMask();
    const VU32 vmask = Set(du32, bucket_mask);
    const uint32_t* base_slots = slots_.data();

    const VU32 vkeys = Load(du32, keys);

    HWY_ALIGN uint32_t offsets[MaxLanes(du32)];

    VU32 h0 = hash_primary_.OneVec(du32, vkeys);
    VU32 b0 = ShiftLeft<4>(And(h0, vmask));
    Store(b0, du32, offsets);

    if constexpr (kPrefetchMode == PrefetchMode::kGather) {
      const RebindToSigned<decltype(du32)> di32;
      VU32 g0 = GatherIndex(du32, base_slots, BitCast(di32, b0));
#if defined(HWY_X86_GCC_INLINE_ASM_VEC_CONSTRAINT)
      asm volatile(
          ""
          : "+" HWY_X86_GCC_INLINE_ASM_VEC_CONSTRAINT(GetRaw(g0, 0))::"memory");
#elif HWY_COMPILER_GCC || HWY_COMPILER_CLANG
      asm volatile("" : : "g"(GetRaw(g0, 0)) : "memory");
#endif
    }
    if constexpr (kPrefetchMode == PrefetchMode::kPrefetch) {
      for (size_t lane = 0; lane < N; ++lane) {
        hwy::Prefetch(base_slots + offsets[lane]);
      }
    }

    // Pass 1: check all primary buckets.
    HWY_ALIGN uint32_t found_arr[MaxLanes(du32)];
    for (size_t lane = 0; lane < N; ++lane) {
      found_arr[lane] = QueryBucket(keys[lane], offsets[lane]) ? 1u : 0u;
    }

    const VU32 found0 = Load(du32, found_arr);

    // Pass 2: only check secondary for lanes that missed in primary.
    const VU32 vzero = Zero(du32);
    const auto miss0 = Eq(found0, vzero);
    HWY_ALIGN uint32_t miss_idx[MaxLanes(du32)];
    const size_t num_miss = CompressStore(Iota(du32, 0), miss0, du32, miss_idx);
    for (size_t j = 0; j < num_miss; ++j) {
      const uint32_t lane = miss_idx[j];
      const uint32_t key = keys[lane];
      const uint32_t b2 = SecondaryBucketOffset(key);
      found_arr[lane] = QueryBucket(key, b2) ? 1u : 0u;
    }

    // Return not-found mask: true = key NOT in set.
    const VU32 result = Load(du32, found_arr);
    return Eq(result, vzero);
  }

  // Access to slot array for testing.
  const uint32_t* Slots() const { return slots_.data(); }

 private:
  enum class PrefetchMode { kGather, kPrefetch };
  static constexpr PrefetchMode kPrefetchMode = PrefetchMode::kPrefetch;

  // Returns slot offset of the primary bucket for `key`.
  HWY_INLINE uint32_t PrimaryBucketOffset(uint32_t key) const {
    const uint32_t hash = hash_primary_(key);
    const uint32_t bucket_idx = hash & config_.BucketMask();
    return bucket_idx * kBucketSize;
  }

  // Returns slot offset of the secondary bucket for `key`.
  HWY_INLINE uint32_t SecondaryBucketOffset(uint32_t key) const {
    const uint32_t hash = hash_secondary_(key);
    const uint32_t bucket_idx = hash & config_.BucketMask();
    return bucket_idx * kBucketSize;
  }

  static constexpr auto kBucketSize = CuckooConfig::kBucketSize;

  CuckooConfig config_;
  Triple32 hash_primary_;
  Triple32 hash_secondary_;
  AlignedVector<uint32_t> slots_;
  uint32_t num_primary_ = 0;
};

// --------------------------------------------------------------------------
// Build statistics and metrics

struct CuckooBuildStats {
  bool success = false;
  uint32_t num_primary = 0;                 // keys in primary bucket
  uint32_t global_seed = 0;                 // seed that succeeded
  uint32_t attempts = 0;                    // number of attempts tried
  uint32_t num_unmatched_after_greedy = 0;  // keys with both buckets full

  // If collect_path_cost_stats is true, populates paths_per_path_cost where
  // paths_per_path_cost[g] is the number of augmenting paths found at cost g.
  bool collect_path_cost_stats = false;
  std::vector<uint32_t> paths_per_path_cost;
};

// --------------------------------------------------------------------------
// CuckooBuilder: Hopcroft-Karp bipartite matching

class CuckooBuilder {
 public:
  explicit CuckooBuilder(CuckooConfig config)
      : config_(config),
        num_keys_(config.NumKeys()),
        num_slots_(config.NumSlots()),
        num_buckets_(config.NumBuckets()) {}

  // Attempts to build with a given pair of hash functions.
  // Returns true on success (all keys matched).
  bool Build(const uint32_t* keys, Triple32 hash_primary,
             Triple32 hash_secondary, bool optimize_primary = false,
             CuckooBuildStats* stats = nullptr) {
    if (num_keys_ >= 1000000) {
      fprintf(stderr,
              "  CuckooBuilder::Build starting (optimize_primary=%d)...\n",
              optimize_primary);
    }
    auto t_build_start = platform::Now();
    hash_primary_ = hash_primary;
    hash_secondary_ = hash_secondary;

    // Compute bucket assignments for each key.
    primary_bucket_.resize(num_keys_);
    secondary_bucket_.resize(num_keys_);
    const uint32_t bucket_mask = config_.BucketMask();
    for (uint32_t i = 0; i < num_keys_; ++i) {
      primary_bucket_[i] = hash_primary_(keys[i]) & bucket_mask;
      secondary_bucket_[i] = hash_secondary_(keys[i]) & bucket_mask;
    }

    // Initialize matching arrays.
    match_key_to_slot_.assign(num_keys_, kUnmatched);
    match_slot_to_key_.assign(num_slots_, kUnmatched);

    // Track the next free slot in each bucket for O(1) greedy assignment.
    bucket_fill_.assign(num_buckets_, 0);

    // --- Greedy phase 1: assign keys to primary buckets ---
    uint32_t matching_size = 0;
    for (uint32_t k = 0; k < num_keys_; ++k) {
      const uint32_t b = primary_bucket_[k];
      if (bucket_fill_[b] < kBucketSize) {
        const uint32_t slot = b * kBucketSize + bucket_fill_[b];
        match_key_to_slot_[k] = slot;
        match_slot_to_key_[slot] = k;
        ++bucket_fill_[b];
        ++matching_size;
      }
    }
    const uint32_t phase1_matches = matching_size;

    // --- Greedy phase 2: assign remaining keys to secondary buckets ---
    uint32_t phase2_matches = 0;
    for (uint32_t k = 0; k < num_keys_; ++k) {
      if (match_key_to_slot_[k] != kUnmatched) continue;  // already placed
      const uint32_t b = secondary_bucket_[k];
      if (bucket_fill_[b] < kBucketSize) {
        const uint32_t slot = b * kBucketSize + bucket_fill_[b];
        match_key_to_slot_[k] = slot;
        match_slot_to_key_[slot] = k;
        ++bucket_fill_[b];
        ++matching_size;
        ++phase2_matches;
      }
    }

    if (matching_size == num_keys_) {
      if (stats) stats->num_unmatched_after_greedy = 0;
      if (num_keys_ >= 1000000) {
        auto t_greedy_end = platform::Now();
        double greedy_ms = (t_greedy_end - t_build_start) * 1000;
        fprintf(stderr,
                "  Greedy phase finished in %.2f ms: Phase 1 matched=%u, "
                "Phase 2 matched=%u, Total matched=%u/%u, unmatched=0\n",
                greedy_ms, phase1_matches, phase2_matches, matching_size,
                num_keys_);
      }
      return true;
    }

    std::vector<uint32_t> unmatched_keys;
    unmatched_keys.reserve(num_keys_ - matching_size);
    for (uint32_t k = 0; k < num_keys_; ++k) {
      if (match_key_to_slot_[k] == kUnmatched) {
        unmatched_keys.push_back(k);
      }
    }
    if (stats) {
      stats->num_unmatched_after_greedy =
          static_cast<uint32_t>(unmatched_keys.size());
    }

    if (num_keys_ >= 1000000) {
      auto t_greedy_end = platform::Now();
      double greedy_ms = (t_greedy_end - t_build_start) * 1000;
      fprintf(stderr,
              "  Greedy phase finished in %.2f ms: Phase 1 matched=%u, "
              "Phase 2 matched=%u, Total matched=%u/%u, unmatched=%zu\n",
              greedy_ms, phase1_matches, phase2_matches, matching_size,
              num_keys_, unmatched_keys.size());
    }

    dist_.resize(num_keys_);
    min_cost_dist_.resize(num_keys_);

    if (optimize_primary) {
      // Phase 3: (Min-Cost): Algorithm 1 from Section 3.1 of
      // Dietzfelbinger et al. "Cuckoo Hashing with Pages". This algorithm
      // gradually finds augmenting paths of increasing cost (variable
      // cur_path_cost).
      BuildCuckooGraphRepresentation();

      const int32_t kInfCost = 1000000;
      layer_L_.assign(num_keys_, -1);
      layer_R_.assign(num_slots_, -1);
      cost_L_.assign(num_keys_, kInfCost);
      cost_R_.assign(num_slots_, kInfCost);
      visited_R_.assign(num_slots_, 0);

      dirty_L_.clear();
      dirty_R_.clear();
      dirty_vis_R_.clear();

      // Cost of augmenting paths, starting with cost 1 since after phase 1 and
      // 2, paths of cost 0 do not exist. Note that although cost of path always
      // non-negative, we use int32_t for consistency when comparing it with
      // partial path cost that can be negative.
      int32_t cur_path_cost = 1;
      uint32_t phase3_matches = 0;
      if (stats && stats->collect_path_cost_stats) {
        stats->paths_per_path_cost.clear();
      }
      auto t_phase3_start = platform::Now();

      while (matching_size < num_keys_ && cur_path_cost <= num_keys_) {
        std::vector<uint32_t> unmatched_L;
        unmatched_L.reserve(num_keys_ - matching_size);
        for (uint32_t i = 0; i < num_keys_; ++i) {
          if (match_key_to_slot_[i] == kUnmatched) {
            unmatched_L.push_back(i);
          }
        }
        if (unmatched_L.empty()) break;

        if (num_keys_ >= 1000000) {
          fprintf(stderr, "\n  cur_path_cost=%d, unmatched=%zu\n",
                  cur_path_cost, unmatched_L.size());
        }

        bool found_paths_cur_path_cost = true;
        uint32_t round_for_cur_path_cost = 0;

        while (found_paths_cur_path_cost) {
          found_paths_cur_path_cost = false;
          ++round_for_cur_path_cost;

          std::vector<uint32_t> R0;
          if (!BFS_MinCost(cur_path_cost, unmatched_L, R0)) {
            break;
          }

          auto t_dfs_start = platform::Now();
          for (uint32_t idx : dirty_vis_R_) visited_R_[idx] = 0;
          dirty_vis_R_.clear();

          uint32_t paths_found = 0;

          for (uint32_t r : R0) {
            if (!visited_R_[r]) {
              if (DFS_MinCost(r)) {
                found_paths_cur_path_cost = true;
                ++matching_size;
                ++paths_found;
                ++phase3_matches;
              }
            }
          }

          for (uint32_t idx : dirty_R_) {
            if (visited_R_[idx]) dirty_vis_R_.push_back(idx);
          }
          for (uint32_t r : R0) {
            if (visited_R_[r]) dirty_vis_R_.push_back(r);
          }
          auto t_dfs_end = platform::Now();

          if (num_keys_ >= 1000000) {
            double dfs_ms = (t_dfs_end - t_dfs_start) * 1000;
            fprintf(stderr,
                    "    round %u: |R0|=%zu, paths=%u, matched=%u/%u, DFS=%.2f "
                    "ms\n",
                    round_for_cur_path_cost, R0.size(), paths_found,
                    matching_size, num_keys_, dfs_ms);
          }

          if (stats && stats->collect_path_cost_stats && paths_found > 0) {
            if (stats->paths_per_path_cost.size() <=
                static_cast<size_t>(cur_path_cost)) {
              stats->paths_per_path_cost.resize(
                  static_cast<size_t>(cur_path_cost) + 1, 0);
            }
            stats->paths_per_path_cost[static_cast<size_t>(cur_path_cost)] +=
                paths_found;
          }

          std::vector<uint32_t> new_unmatched;
          new_unmatched.reserve(unmatched_L.size() - paths_found);
          for (uint32_t i : unmatched_L) {
            if (match_key_to_slot_[i] == kUnmatched) {
              new_unmatched.push_back(i);
            }
          }
          unmatched_L = std::move(new_unmatched);
        }

        ++cur_path_cost;
      }
      MaybeLogPhase3Stats(t_phase3_start, phase3_matches);
      return matching_size == num_keys_;
    } else {
      // --- Phase 3: Hopcroft-Karp for remaining unmatched keys ---
      uint32_t phase3_matches = 0;
      auto t_phase3_start = platform::Now();
      while (BFS(unmatched_keys)) {
        std::vector<uint32_t> next_unmatched;
        next_unmatched.reserve(unmatched_keys.size());
        for (uint32_t k : unmatched_keys) {
          if (DFS(k)) {
            ++matching_size;
            ++phase3_matches;
          } else {
            next_unmatched.push_back(k);
          }
        }
        unmatched_keys = std::move(next_unmatched);
      }
      MaybeLogPhase3Stats(t_phase3_start, phase3_matches);
      return matching_size == num_keys_;
    }
  }

  // Build the CuckooTable from a successful matching.
  CuckooTable Take(const uint32_t* keys) {
    AlignedVector<uint32_t> slots(num_slots_);
    // Fill with sentinel.
    for (uint32_t i = 0; i < num_slots_; ++i) {
      slots[i] = kEmpty;
    }

    uint32_t num_primary = 0;
    for (uint32_t k = 0; k < num_keys_; ++k) {
      const uint32_t slot = match_key_to_slot_[k];
      HWY_ASSERT(slot != kUnmatched);
      HWY_ASSERT(slots[slot] == kEmpty);
      slots[slot] = keys[k];

      // Check if this key ended up in its primary bucket.
      const uint32_t bucket_of_slot = slot / kBucketSize;
      if (bucket_of_slot == primary_bucket_[k]) {
        ++num_primary;
      }
    }

    CuckooTable table(config_, hash_primary_, hash_secondary_, std::move(slots),
                      num_primary);
    return table;
  }

 private:
  void MaybeLogPhase3Stats(double t_phase3_start, uint32_t phase3_matches) {
    if (num_keys_ < 1000000) return;
    auto t_phase3_end = platform::Now();
    double phase_ms = (t_phase3_end - t_phase3_start) * 1000;
    fprintf(stderr, "  Phase 3 completed in %.2f ms: matched=%u\n", phase_ms,
            phase3_matches);
  }

  void BuildCuckooGraphRepresentation() {
    std::vector<uint32_t> prim_count(num_buckets_, 0);
    std::vector<uint32_t> sec_count(num_buckets_, 0);
    for (uint32_t i = 0; i < num_keys_; ++i) {
      prim_count[primary_bucket_[i]]++;
      if (primary_bucket_[i] != secondary_bucket_[i]) {
        sec_count[secondary_bucket_[i]]++;
      }
    }

    prim_offset_.assign(num_buckets_ + 1, 0);
    sec_offset_.assign(num_buckets_ + 1, 0);
    for (uint32_t b = 0; b < num_buckets_; ++b) {
      prim_offset_[b + 1] = prim_offset_[b] + prim_count[b];
      sec_offset_[b + 1] = sec_offset_[b] + sec_count[b];
    }

    prim_keys_.assign(prim_offset_[num_buckets_], 0);
    sec_keys_.assign(sec_offset_[num_buckets_], 0);

    prim_count.assign(num_buckets_, 0);
    sec_count.assign(num_buckets_, 0);
    for (uint32_t i = 0; i < num_keys_; ++i) {
      uint32_t b = primary_bucket_[i];
      prim_keys_[prim_offset_[b] + prim_count[b]++] = i;
      if (primary_bucket_[i] != secondary_bucket_[i]) {
        b = secondary_bucket_[i];
        sec_keys_[sec_offset_[b] + sec_count[b]++] = i;
      }
    }
  }

  // BFS phase of Algorithm 1 (Section 3.1) with cost relaxation from paper
  // Dietzfelbinger et al. "Cuckoo Hashing with Pages".
  // cur_path_cost is the cost of the augmenting paths we are searching for.
  bool BFS_MinCost(int32_t cur_path_cost,
                   const std::vector<uint32_t>& unmatched_L,
                   std::vector<uint32_t>& R0) {
    auto t0 = platform::Now();
    const int32_t kInfCost = 1000000;

    for (uint32_t idx : dirty_L_) {
      layer_L_[idx] = -1;
      cost_L_[idx] = kInfCost;
    }
    for (uint32_t idx : dirty_R_) {
      layer_R_[idx] = -1;
      cost_R_[idx] = kInfCost;
    }
    dirty_L_.clear();
    dirty_R_.clear();

    std::deque<int64_t> queue;
    for (uint32_t i : unmatched_L) {
      if (match_key_to_slot_[i] == kUnmatched) {
        layer_L_[i] = 0;
        cost_L_[i] = 0;
        dirty_L_.push_back(i);
        queue.push_back(static_cast<int64_t>(i));
      }
    }

    R0.clear();
    int32_t target_layer = -1;

    while (!queue.empty()) {
      int64_t val = queue.front();
      queue.pop_front();

      if (val >= 0) {
        // L node (key)
        uint32_t i = static_cast<uint32_t>(val);
        int32_t cur_layer = layer_L_[i];
        int32_t cur_cost = cost_L_[i];
        if (target_layer >= 0 && cur_layer + 1 > target_layer) continue;

        // Primary edges (cost 0)
        uint32_t base_p = primary_bucket_[i] * kBucketSize;
        for (uint32_t j = 0; j < kBucketSize; ++j) {
          uint32_t r = base_p + j;
          if (match_key_to_slot_[i] == r) continue;
          int32_t new_cost = cur_cost;  // primary edge cost = 0
          int32_t new_layer = cur_layer + 1;

          if (layer_R_[r] == -1 || new_cost < cost_R_[r]) {
            if (layer_R_[r] == -1) dirty_R_.push_back(r);
            layer_R_[r] = new_layer;
            cost_R_[r] = new_cost;
            if (match_slot_to_key_[r] == kUnmatched) {
              if (new_cost == cur_path_cost) {
                if (target_layer == -1) target_layer = new_layer;
                if (new_layer == target_layer) R0.push_back(r);
              }
            } else {
              queue.push_back(-static_cast<int64_t>(r) - 1);
            }
          }
        }

        // Secondary edges (cost 1)
        if (primary_bucket_[i] != secondary_bucket_[i]) {
          uint32_t base_s = secondary_bucket_[i] * kBucketSize;
          for (uint32_t j = 0; j < kBucketSize; ++j) {
            uint32_t r = base_s + j;
            if (match_key_to_slot_[i] == r) continue;
            int32_t new_cost = cur_cost + 1;  // secondary edge cost = 1
            int32_t new_layer = cur_layer + 1;

            if (layer_R_[r] == -1 || new_cost < cost_R_[r]) {
              if (layer_R_[r] == -1) dirty_R_.push_back(r);
              layer_R_[r] = new_layer;
              cost_R_[r] = new_cost;
              if (match_slot_to_key_[r] == kUnmatched) {
                if (new_cost == cur_path_cost) {
                  if (target_layer == -1) target_layer = new_layer;
                  if (new_layer == target_layer) R0.push_back(r);
                }
              } else {
                queue.push_back(-static_cast<int64_t>(r) - 1);
              }
            }
          }
        }
      } else {
        // R node (slot) — follow matched edge backward to L
        uint32_t r = static_cast<uint32_t>(-(val + 1));
        int32_t cur_layer_r = layer_R_[r];
        int32_t cur_cost_r = cost_R_[r];
        if (target_layer >= 0 && cur_layer_r + 1 > target_layer) continue;

        uint32_t matched_left = match_slot_to_key_[r];
        if (matched_left == kUnmatched) continue;

        // Backward matched edge cost = -original_cost
        uint32_t bucket_r = r / kBucketSize;
        int32_t orig_cost =
            (bucket_r == secondary_bucket_[matched_left] &&
             primary_bucket_[matched_left] != secondary_bucket_[matched_left])
                ? 1
                : 0;
        int32_t back_cost = -orig_cost;

        int32_t new_cost = cur_cost_r + back_cost;
        if (new_cost < 0) continue;
        int32_t new_layer = cur_layer_r + 1;

        if (layer_L_[matched_left] == -1 || new_cost < cost_L_[matched_left]) {
          if (layer_L_[matched_left] == -1) dirty_L_.push_back(matched_left);
          layer_L_[matched_left] = new_layer;
          cost_L_[matched_left] = new_cost;
          queue.push_back(static_cast<int64_t>(matched_left));
        }
      }
    }

    if (num_keys_ >= 1000000) {
      auto t1 = platform::Now();
      double bfs_ms = (t1 - t0) * 1000;
      fprintf(stderr,
              "    BFS_MinCost(cur_path_cost=%d): |R0|=%zu in %.2f ms\n",
              cur_path_cost, R0.size(), bfs_ms);
    }

    return !R0.empty();
  }

  // DFS phase of Algorithm 1 (Section 3.1): backward DFS from target right node
  // r.
  bool DFS_MinCost(uint32_t r) {
    visited_R_[r] = 1;
    int32_t r_layer = layer_R_[r];
    int32_t r_cost = cost_R_[r];
    uint32_t bkt = r / kBucketSize;

    // Try primary keys in this bucket (edge cost 0)
    for (uint32_t idx = prim_offset_[bkt]; idx < prim_offset_[bkt + 1]; ++idx) {
      uint32_t i = prim_keys_[idx];
      if (layer_L_[i] != r_layer - 1) continue;
      if (match_key_to_slot_[i] == r) continue;  // matched edge, skip
      if (cost_L_[i] + 0 != r_cost) continue;

      if (match_key_to_slot_[i] == kUnmatched) {
        // Found unmatched left node — augmenting path complete
        match_key_to_slot_[i] = r;
        match_slot_to_key_[r] = i;
        return true;
      } else {
        uint32_t old_r = match_key_to_slot_[i];
        if (!visited_R_[old_r] && layer_R_[old_r] != -1) {
          if (DFS_MinCost(old_r)) {
            match_key_to_slot_[i] = r;
            match_slot_to_key_[r] = i;
            return true;
          }
        }
      }
    }

    // Try secondary keys (edge cost 1)
    for (uint32_t idx = sec_offset_[bkt]; idx < sec_offset_[bkt + 1]; ++idx) {
      uint32_t i = sec_keys_[idx];
      if (layer_L_[i] != r_layer - 1) continue;
      if (match_key_to_slot_[i] == r) continue;
      if (cost_L_[i] + 1 != r_cost) continue;

      if (match_key_to_slot_[i] == kUnmatched) {
        match_key_to_slot_[i] = r;
        match_slot_to_key_[r] = i;
        return true;
      } else {
        uint32_t old_r = match_key_to_slot_[i];
        if (!visited_R_[old_r] && layer_R_[old_r] != -1) {
          if (DFS_MinCost(old_r)) {
            match_key_to_slot_[i] = r;
            match_slot_to_key_[r] = i;
            return true;
          }
        }
      }
    }

    return false;
  }

  // BFS phase of Hopcroft-Karp.
  // Builds a layered graph from all free (unmatched) keys. Stops at the
  // first layer k where free slots are reached: processes all keys at layer
  // k but does not enqueue keys at layer k+1. This ensures the DFS finds a
  // maximal set of vertex-disjoint shortest augmenting paths.
  // Returns true if at least one augmenting path exists.
  bool BFS(const std::vector<uint32_t>& unmatched_keys) {
    std::vector<uint32_t> queue;
    queue.reserve(num_keys_);

    dist_.assign(num_keys_, kUnmatched);
    for (uint32_t k : unmatched_keys) {
      dist_[k] = 0;
      queue.push_back(k);
    }

    bool found = false;
    size_t head = 0;

    while (head < queue.size()) {
      const uint32_t k = queue[head++];

      // Enumerate candidate slots for key k.
      const uint32_t b1 = primary_bucket_[k] * kBucketSize;
      const uint32_t b2 = secondary_bucket_[k] * kBucketSize;

      for (int pass = 0; pass < 2; ++pass) {
        const uint32_t base = (pass == 0) ? b1 : b2;
        for (uint32_t s = base; s < base + kBucketSize; ++s) {
          const uint32_t other_key = match_slot_to_key_[s];
          if (other_key == kUnmatched) {
            // Free slot reachable → augmenting path exists.
            found = true;
          } else if (dist_[other_key] == kUnmatched && !found) {
            // Matched slot → continue BFS through the matched key.
            // Only enqueue if we haven't reached the terminal layer yet.
            dist_[other_key] = dist_[k] + 1;
            queue.push_back(other_key);
          }
        }
      }
    }

    return found;
  }

  // DFS phase of Hopcroft-Karp.
  // Attempts to find an augmenting path from key k along BFS layers.
  bool DFS(uint32_t k) {
    const uint32_t b1 = primary_bucket_[k] * kBucketSize;
    const uint32_t b2 = secondary_bucket_[k] * kBucketSize;

    for (int pass = 0; pass < 2; ++pass) {
      const uint32_t base = (pass == 0) ? b1 : b2;
      for (uint32_t s = base; s < base + kBucketSize; ++s) {
        const uint32_t other_key = match_slot_to_key_[s];
        if (other_key == kUnmatched) {
          // Found a free slot: augment the matching.
          match_key_to_slot_[k] = s;
          match_slot_to_key_[s] = k;
          return true;
        }
        if (dist_[other_key] == dist_[k] + 1) {
          if (DFS(other_key)) {
            // Augment: reassign k to slot s.
            match_key_to_slot_[k] = s;
            match_slot_to_key_[s] = k;
            return true;
          }
        }
      }
    }

    // No augmenting path found from k in this phase.
    dist_[k] = kUnmatched;
    return false;
  }

  static constexpr auto kBucketSize = CuckooConfig::kBucketSize;
  static constexpr auto kUnmatched = CuckooConfig::kUnmatched;
  static constexpr auto kEmpty = CuckooConfig::kEmpty;

  CuckooConfig config_;
  uint32_t num_keys_;
  uint32_t num_slots_;
  uint32_t num_buckets_;

  Triple32 hash_primary_;
  Triple32 hash_secondary_;

  std::vector<uint32_t> primary_bucket_;    // [num_keys] → bucket index
  std::vector<uint32_t> secondary_bucket_;  // [num_keys] → bucket index

  std::vector<uint32_t> match_key_to_slot_;  // [num_keys] → slot index
  std::vector<uint32_t> match_slot_to_key_;  // [num_slots] → key index
  std::vector<uint32_t> dist_;               // [num_keys] BFS distance
  std::vector<int32_t> min_cost_dist_;       // [num_keys] Min-cost distance
  std::vector<uint32_t> bucket_fill_;        // [num_buckets] slots used

  // Phase 3 optimal Hopcroft-Karp data structures
  std::vector<int32_t> layer_L_;    // [num_keys_]
  std::vector<int32_t> layer_R_;    // [num_slots_]
  std::vector<int32_t> cost_L_;     // [num_keys_]
  std::vector<int32_t> cost_R_;     // [num_slots_]
  std::vector<uint8_t> visited_R_;  // [num_slots_]

  std::vector<uint32_t> dirty_L_;
  std::vector<uint32_t> dirty_R_;
  std::vector<uint32_t> dirty_vis_R_;

  // Cuckoo graph representation:
  // prim_offset_ and sec_offset_ store cumulative key counts for each bucket.
  // prim_keys_ and sec_keys_ store contiguous list of key IDs mapping to each
  // bucket.
  std::vector<uint32_t> prim_offset_;  // [num_buckets_ + 1]
  std::vector<uint32_t> prim_keys_;    // [num_keys_]
  std::vector<uint32_t> sec_offset_;   // [num_buckets_ + 1]
  std::vector<uint32_t> sec_keys_;     // [num_keys_]
};

// --------------------------------------------------------------------------
// Top-level build function

// Builds a CuckooTable from distinct uint32_t keys.
// Tries multiple hash function seeds. Returns an empty table on failure.
// If optimize_primary is true, runs a min-cost optimization phase after
// finding a valid matching, moving keys from secondary to primary buckets
// via displacement chains (Section 3.1 of arXiv:1104.5111).
//
// Note about epsilon: epsilon is the load factor, i.e. the ratio of the number
// of keys to the number of slots. The table has num_keys / epsilon slots.
// We allow limited list of epsilons.
static HWY_MAYBE_UNUSED CuckooTable
CuckooBuild(const uint32_t* keys, uint32_t num_keys, double epsilon = 0.25,
            uint32_t max_attempts = 100, bool optimize_primary = false,
            CuckooBuildStats* stats = nullptr) {
  constexpr double kEpsilons[] = {0.01, 0.05, 0.10, 0.25, 0.50};
  bool found_epsilon = false;
  for (double e : kEpsilons) {
    if (epsilon == e) {
      found_epsilon = true;
      break;
    }
  }
  if (!found_epsilon) {
    fprintf(stderr, "Unsupported epsilon: %f\n", epsilon);
    return CuckooTable();
  }

  CuckooConfig config(num_keys, epsilon);
  CuckooBuilder builder(config);

  AesCtrEngine engine(/*deterministic=*/true);

  for (uint32_t attempt = 0; attempt < max_attempts; ++attempt) {
    if (num_keys >= 1000000) {
      fprintf(stderr, "CuckooBuild attempt %u for %u keys...\n", attempt,
              num_keys);
    }
    // Use different seeds for primary and secondary hash functions.
    Triple32 h1(engine, attempt * 2);
    Triple32 h2(engine, attempt * 2 + 1);

    if (builder.Build(keys, h1, h2, optimize_primary, stats)) {
      if (stats) {
        stats->success = true;
        stats->global_seed = attempt;
        stats->attempts = attempt + 1;
      }
      CuckooTable table = builder.Take(keys);
      if (stats) {
        stats->num_primary = table.NumPrimary();
      }
      return table;
    }
  }

  // All attempts failed.
  if (stats) {
    stats->success = false;
    stats->attempts = max_attempts;
  }
  return CuckooTable();
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
#endif  // HWY_TARGET != HWY_SCALAR

#endif  // HIGHWAY_HWY_CONTRIB_HASH_CUCKOO_INL_H_
