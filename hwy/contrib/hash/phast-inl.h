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

#include <utility>  // std::move

#include "hwy/contrib/hash/hash-inl.h"
#include "hwy/contrib/hash/phast.h"
#include "hwy/highway.h"

#if HWY_TARGET != HWY_SCALAR
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// --------------------------------------------------------------------------
// Phast: query-time structure

class Phast {
 public:
  Phast() = default;
  explicit Phast(PhastData&& data)
      : config_(data.config), seeds_packed_(std::move(data.seeds_packed)) {
    AesCtrEngine engine(/*deterministic=*/true);
    hash_ = Triple32(engine, data.config.GlobalSeed());
  }

  Phast(Phast&&) = default;
  Phast& operator=(Phast&&) = default;

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

  // Two vectors have higher throughput because they utilize all 16-bit lanes
  // for Hash16. Can be called directly, or via QueryBatch.
  template <class DU32, class VU32 = Vec<DU32>>
  HWY_INLINE void Query2(DU32 du32, VU32 key0, VU32 key1, VU32& idx0,
                         VU32& idx1) const {
    hash_.TwoVec(du32, key0, key1);

    const VU32 bucket_mask = Set(du32, config_.BucketMask());
    const VU32 seed0 = GatherSeeds(du32, And(key0, bucket_mask));
    const VU32 seed1 = GatherSeeds(du32, And(key1, bucket_mask));

    QueryWithSeeds(du32, key0, key1, seed0, seed1, config_, idx0, idx1);
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

  // For internal use by Query2:

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
  // SIMD Gather from packed seeds. Uses TableLookupBytes to select one byte
  // per u32 lane, avoiding a variable shift.
  template <class DU32, class VU32 = Vec<DU32>>
  HWY_INLINE VU32 GatherSeeds(DU32 du32, VU32 bucket_idx) const {
    const RebindToSigned<DU32> di32;
    const Repartition<uint8_t, DU32> du8;
    using VU8 = Vec<decltype(du8)>;
    const VU32 word_idx = ShiftRight<2>(bucket_idx);
    const VU32 words =
        GatherIndex(du32, seeds_packed_.Data(), BitCast(di32, word_idx));
    // 0-3 in the low byte of each u32 lane; upper bytes are 0.
    const VU8 byte_idx = BitCast(du8, And(bucket_idx, Set(du32, 3)));
    // Add 4*iota to byte_idx, setting the starting offset for each u32 lane.
    // Setting the upper 3 bytes >= 0x80 ensures they are zeroed.
    const VU8 kBase = Dup128VecFromValues(du8, 0, 0x80, 0x80, 0x80, 4, 0x80,
                                          0x80, 0x80, 8, 0x80, 0x80, 0x80, 12,
                                          0x80, 0x80, 0x80);
    const VU8 indices = Or(byte_idx, kBase);
    return BitCast(du32, TableLookupBytesOr0(words, indices));
  }

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
    const uint16_t x2 = static_cast<uint16_t>(uint32_t{x} * x);
    x = static_cast<uint16_t>(x + static_cast<uint16_t>(uint32_t{x2} * 0xca32));
    x = static_cast<uint16_t>(x ^ (x >> 12));
    x = static_cast<uint16_t>(uint32_t{x} * 0x3929u);
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

  PhastConfig config_ = {};
  Triple32 hash_;
  PackedSeeds seeds_packed_;
};

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
#endif  // HWY_TARGET != HWY_SCALAR

#endif  // HIGHWAY_HWY_CONTRIB_HASH_PHAST_INL_H_
