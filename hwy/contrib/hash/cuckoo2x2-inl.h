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

// Cuckoo2x2: SIMD set membership via two-choice hashing with two entries per
// bucket. The main novelty is that we prevent false positives with only a
// 14-bit fingerprint plus a 2-bit tag. This reduces lookups to two gathers plus
// a few SIMD instructions. The tradeoff is that we require at least 256K
// buckets to cover/constrain the other 18 hash bits.
//
// For individual membership queries in small sets, it is twice as fast as
// absl::flat_hash_set. When used for batched hash set membership queries
// (1M keys, AVX2), it is twice as fast as `PHAST` and uses 90% as much
// memory. From 2^16 keys, it is about three times as fast as
// absl::flat_hash_set and from 2^18 keys, requires only 40% as much memory.
//
// Each key is hashed with two independent functions. The builder assigns each
// key to the less-loaded bucket, ensuring max 2 keys per bucket. At query
// time, we check both candidate buckets and OR the results.

#include <stddef.h>
#include <stdint.h>

#include <utility>  // std::move

#include "hwy/aligned_allocator.h"

#if defined(HIGHWAY_HWY_CONTRIB_HASH_CUCKOO2X2_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_HASH_CUCKOO2X2_INL_H_
#undef HIGHWAY_HWY_CONTRIB_HASH_CUCKOO2X2_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_HASH_CUCKOO2X2_INL_H_
#endif

#include "hwy/contrib/hash/cuckoo2x2.h"
#include "hwy/contrib/hash/hash-inl.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"

static_assert(HWY_CXX_LANG >= 201703L, "requires C++17 or later.");

#if HWY_TARGET != HWY_SCALAR
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// --------------------------------------------------------------------------
// Query-time structure. Owns/wraps Cuckoo2x2Data.

class Cuckoo2x2 {
 public:
  Cuckoo2x2() = default;
  explicit Cuckoo2x2(Cuckoo2x2Data&& data)
      : hash1_(data.config.hash_key), data_(std::move(data)) {
    HWY_ASSERT_M(data_.NumBuckets() > 0, "Build failed");
  }

  Cuckoo2x2(Cuckoo2x2&&) = default;
  Cuckoo2x2& operator=(Cuckoo2x2&&) = default;

  const Cuckoo2x2Data& Data() const { return data_; }

  // Scalar single-key lookup. 11 cycles on AMD Milan.
  HWY_INLINE bool Contains(uint32_t key) const {
    const uint32_t mask = data_.config.bucket_mask;
    const uint32_t h1 = hash1_(key);
    // Golden ratio; mul by odd is a bijection.
    const uint32_t h2 = key * 0x9E3779B9u;
    const uint32_t b1 = h1 & mask;
    const uint32_t b2 = h2 & mask;
    // 14-bit fp (h >> 18) + 2-bit tag: 01=hash1, 10=hash2.
    const uint16_t fp1 = static_cast<uint16_t>((h1 >> 18) | 0x4000);
    const uint16_t fp2 = static_cast<uint16_t>((h2 >> 18) | 0x8000);
    const uint32_t e1 = data_.entries[b1];
    const uint32_t e2 = data_.entries[b2];
    return fp1 == static_cast<uint16_t>(e1 & 0xFFFF) ||
           fp1 == static_cast<uint16_t>(e1 >> 16) ||
           fp2 == static_cast<uint16_t>(e2 & 0xFFFF) ||
           fp2 == static_cast<uint16_t>(e2 >> 16);
  }

  // SIMD set membership for N u32 keys. Returns "not-found" mask
  // (true = key NOT in set) to avoid Ne/Not overhead. Up to 59 GB/s throughput
  // on all cores of an AMD Milan.
  template <class DU32, class VU32 = Vec<DU32>, class MU32 = Mask<DU32>>
  HWY_INLINE MU32 operator()(DU32 du32, VU32 key) const {
    const RepartitionToNarrow<DU32> du16;
    const RebindToSigned<DU32> di32;
    using VU16 = Vec<decltype(du16)>;
    using VI32 = Vec<decltype(di32)>;

    // Hash1 uses Triple32, Hash2 uses golden ratio.
    const VU32 h1 = hash1_.OneVec(du32, key);
    const VU32 h2 = Mul(key, Set(du32, 0x9E3779B9u));

    // Bucket indices via AND mask. This is critical for correctness, because
    // our false-positive elimination method is to check all bits which do not
    // influence/govern the choice of bucket. Lemire's modulo is a function of
    // ALL hash bits, hence would not work here.
    const VU32 bucket_mask = Set(du32, data_.config.bucket_mask);
    const VI32 bucket_idx1 = BitCast(di32, And(h1, bucket_mask));
    const VI32 bucket_idx2 = BitCast(di32, And(h2, bucket_mask));

    // 14-bit fingerprint (bits 18-31 of hash) + 2-bit tag (bits 14-15):
    //   01 = hash1, 10 = hash2. ShiftRight<2> extracts bits 18-31
    //   into bits 0-13 with bits 14-15 = 0, then Or sets the tag.
    //   Empty entries (tag=00) can never match (tag always 01 or 10).
    // Note that we have >= 256K buckets, hence bits 0-17 govern the choice of
    // bucket and do not have to be verified.
    const VU16 fp1 =
        Or(ShiftRight<2>(DupOdd(BitCast(du16, h1))), Set(du16, 0x4000));
    const VU16 fp2 =
        Or(ShiftRight<2>(DupOdd(BitCast(du16, h2))), Set(du16, 0x8000));

    const uint32_t* base = data_.entries.data();

#if HWY_TARGET == HWY_AVX2
    // Unconditional independent gathers hide latency and are considerably
    // faster than *dependent* MaskedGatherIndex on AVX2. We still use a
    // masked gather because GatherIndex creates a mask internally, so it helps
    // to only create it once.
    const Mask<DU32> mask = SetMask(du32, true);
    const VU32 e1 = MaskedGatherIndex(mask, du32, base, bucket_idx1);
    const VU32 e2 = MaskedGatherIndex(mask, du32, base, bucket_idx2);
#else
    const VU32 e1 = GatherIndex(du32, base, bucket_idx1);
    const VU32 e2 = GatherIndex(du32, base, bucket_idx2);
#endif

    const VU16 eq1 = VecFromMask(du16, Eq(BitCast(du16, e1), fp1));
    const VU16 eq2 = VecFromMask(du16, Eq(BitCast(du16, e2), fp2));

    const VU32 zero = Zero(du32);
    // Or is "found in either bucket". If that is 0, return "not found".
    return Eq(BitCast(du32, Or(eq1, eq2)), zero);
  }

 private:
  Triple32 hash1_;
  Cuckoo2x2Data data_;
};

inline Cuckoo2x2 MakeCuckoo2x2(Span<const uint32_t> keys, ThreadPool& pool) {
  return Cuckoo2x2(BuildCuckoo2x2(keys, pool));
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
#endif  // HWY_TARGET != HWY_SCALAR

#endif  // HIGHWAY_HWY_CONTRIB_HASH_CUCKOO2X2_INL_H_
