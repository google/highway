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

// ShardMul: u64 to u32 reducer, collision-free for a known set of keys,
// designed for high SIMD throughput without Gather. Uses Feistel-based mixing
// to handle structured/related keys. To break the 2^16 birthday bound on the
// number of keys, we require a 'steerable' hash function family. Jenkins'
// perfect hash requires per-element table lookups and thus Gather. We instead
// use Dietzfelbinger's mul-shift universal hashing scheme, with 16-bit
// multipliers because these are faster than 32-bit on some SIMD platforms.
// The per-bucket multipliers fit in a few registers and can be retrieved using
// shuffles rather than Gather. MulHigh instructions even avoid the shift.
// This approach is also inspired by PHAST, which trades extra effort during an
// (offline) build step for more compact storage.

#include <stddef.h>
#include <stdint.h>

#include "hwy/aligned_allocator.h"
#include "hwy/contrib/hash/shardmul.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

#if defined(HIGHWAY_HWY_CONTRIB_HASH_SHARDMUL_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_HASH_SHARDMUL_INL_H_
#undef HIGHWAY_HWY_CONTRIB_HASH_SHARDMUL_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_HASH_SHARDMUL_INL_H_
#endif

#include "hwy/contrib/hash/hash-inl.h"
#include "hwy/highway.h"

static_assert(HWY_CXX_LANG >= 201703L, "requires C++17 or later.");

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
#if HWY_TARGET != HWY_SCALAR

class HWY_ALIGN_MAX ShardMul {
 public:
  explicit ShardMul(const ShardMulData& data)
      : f0_(data.keys[0]),
        f1_(data.keys[1]),
        f2_(data.keys[2]),
        f3_(data.keys[3]) {
    // Not CopySameSize because array arguments decay to pointers!
    CopyBytes<sizeof(table_)>(data.table, table_);
  }

  // This is the only way to check that build failed. During construction, we
  // have a partially-initialized instance where this returns false, hence we do
  // not check this in the constructor. The builder also relies on this
  // implementation; it sets the first table entry to bypass this check.
  bool IsEmpty() const {
    if (table_[0] == 0) return true;
    // Ensure all table entries are non-zero.
    for (size_t i = 0; i < 16; ++i) {
      if (table_[i] == 0) HWY_ABORT("Invalid table, entry %zu is 0", i);
    }
    return false;
  }

  // ---------------------------------------------------------------------------
  // Internal, for use by the builder.

  // Unlike operator() and TwoVec(), these are safe to
  // call even with a partially-initialized instance where IsEmpty().
  HWY_INLINE void Feistel(const uint64_t x, uint32_t& LL, uint32_t& RR) const {
    LL = static_cast<uint32_t>(x & 0xFFFFFFFFu);
    RR = static_cast<uint32_t>(x >> 32);
    LL ^= f0_(RR);
    RR ^= f1_(LL);
    LL ^= f2_(RR);
    RR ^= f3_(LL);
  }

  template <class DU32, class DU64 = Rebind<uint64_t, DU32>,
            class VU32 = Vec<DU32>, class VU64 = Vec<DU64>, HWY_IF_U32_D(DU32)>
  HWY_INLINE void Feistel(DU32 du32, const VU64 key, VU32& LL, VU32& RR) const {
    // Split u64 key into two u32 half-vectors.
    LL = TruncateTo(du32, key);
    RR = TruncateTo(du32, ShiftRight<32>(key));

    // 4-round Feistel to ensure the two halves are independent (Luby-Rackoff).
    // Even with a stronger round function, three rounds are insufficient
    // because LL and RR still share several terms.
    LL = Xor(LL, f0_.OneVec(du32, RR));
    RR = Xor(RR, f1_.OneVec(du32, LL));
    LL = Xor(LL, f2_.OneVec(du32, RR));
    RR = Xor(RR, f3_.OneVec(du32, LL));
  }

  template <class DU32, class DU64 = RepartitionToWide<DU32>,
            class VU32 = Vec<DU32>, class VU64 = Vec<DU64>, HWY_IF_U32_D(DU32)>
  HWY_INLINE void Feistel(DU32 du32, const VU64 key0, const VU64 key1, VU32& LL,
                          VU32& RR) const {
    // Split u64 keys into two u32 halves.
    const VU32 lo = BitCast(du32, key0);
    const VU32 hi = BitCast(du32, key1);
#if HWY_IS_BIG_ENDIAN
    LL = ConcatOdd(du32, hi, lo);
    RR = ConcatEven(du32, hi, lo);
#else
    LL = ConcatEven(du32, hi, lo);
    RR = ConcatOdd(du32, hi, lo);
#endif

    // 4-round Feistel to ensure the two halves are independent (Luby-Rackoff).
    // Even with a stronger round function, three rounds are insufficient
    // because LL and RR still share several terms.
    LL = Xor(LL, f0_.OneVec(du32, RR));
    RR = Xor(RR, f1_.OneVec(du32, LL));
    LL = Xor(LL, f2_.OneVec(du32, RR));
    RR = Xor(RR, f3_.OneVec(du32, LL));
  }

  uint32_t BucketIndex(const uint32_t LL) const { return LL >> 28; }
  uint32_t LookupMul(const uint32_t bucket) const { return table_[bucket]; }

  // The MulHigh + XorAndNot portion of TwoVec, hoisted so that the builder can
  // call it directly with pre-stored LL/RR and broadcast multipliers.
  // `muls` is a u32 vector where each element holds a packed u16x2 pair.
  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  static HWY_INLINE VU32 MulAndXor(DU32 du32, VU32 LL, VU32 RR, VU32 muls) {
    const RepartitionToNarrow<DU32> du16;
    // Multiply pairs of u16; the upper 16 bits are better-mixed, hence MulHigh.
    // Note that the result is no greater than the multiplier, which the builder
    // ensures is at least 0x8001.
    const VU32 result =
        BitCast(du32, MulHigh(BitCast(du16, RR), BitCast(du16, muls)));
    // It is crucial to inject the bucket index into the output. Partitioning
    // the output range prevents cross-bucket collisions, hence allows
    // building buckets independently and in parallel. The bucket index resides
    // in the upper bits because they are lower quality (MSB might not be set by
    // MulHigh), and downstream users may also be more responsive to MSB. It is
    // sufficient to clear MSBs so that the XOR by LL sets the bucket index.
    const VU32 bucket_idx_mask = Set(du32, 0xF0000000u);
    return XorAndNot(LL, bucket_idx_mask, result);
  }

  // ---------------------------------------------------------------------------
  // Public API:

  // Maps a single u64 key to a u32 value. Collision-free for the key set used
  // during construction. See TwoVec for detailed comments. Measured latency is
  // 16 cycles on Zen 4.
  HWY_INLINE uint32_t operator()(uint64_t x) const {
    HWY_DASSERT(!IsEmpty());

    uint32_t LL, RR;
    Feistel(x, LL, RR);
    const uint32_t bucket = BucketIndex(LL);

    // Lookup per-bucket multiplier pair.
    const uint32_t muls = table_[bucket];
    const uint32_t mul0 = muls & 0xFFFF;
    const uint32_t mul1 = muls >> 16;

    // MulHigh with independent multipliers per u16 in each u32.
    const uint32_t x0 = RR & 0xFFFF;
    const uint32_t x1 = RR >> 16;
    const uint32_t r0 = (x0 * mul0) >> 16;
    const uint32_t r1 = (x1 * mul1) >> 16;

    // Combine and XOR with unused Feistel bits; passes through bucket index.
    const uint32_t out = ((r1 & 0x0FFF) << 16) | r0;
    return out ^ LL;
  }

  // SIMD: returns full u32 vector from two u64 vectors. Collision-free for the
  // key set used during construction. Measured throughput is 34 GB/s on one
  // Zen 4 core.
  template <class DU32, class DU64 = RepartitionToWide<DU32>,
            class VU32 = Vec<DU32>, class VU64 = Vec<DU64>, HWY_IF_U32_D(DU32)>
  HWY_INLINE VU32 TwoVec(DU32 du32, VU64 key0, VU64 key1) const {
    HWY_DASSERT(!IsEmpty());

    VU32 LL, RR;
    Feistel(du32, key0, key1, LL, RR);
    return ResultFromFeistel(du32, LL, RR);
  }

  // Variant with one u64 vector input and one u32 half-vector output.
  template <class DU32, class DU64 = Rebind<uint64_t, DU32>,
            class VU32 = Vec<DU32>, class VU64 = Vec<DU64>, HWY_IF_U32_D(DU32)>
  HWY_INLINE VU32 OneVec(DU32 du32, VU64 key) const {
    HWY_DASSERT(!IsEmpty());

    VU32 LL, RR;
    Feistel(du32, key, LL, RR);
    return ResultFromFeistel(du32, LL, RR);
  }

 private:
  // Returns pairs of u16 per-bucket multipliers retrieved from the table. We
  // use u32 lookups rather than a single u16 Lookup16 because independent
  // multipliers are helpful for reducing collisions for structured keys, and
  // AVX2 anyway only supports u32 lookups.
  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  HWY_INLINE VU32 LookupMuls(DU32 du32, VU32 bucket) const {
    const RebindToSigned<DU32> di32;
    if constexpr (CanLookup16(du32)) {  // AVX2/AVX3
      return Lookup16(du32, table_, BitCast(di32, bucket));
    } else {
      HWY_DASSERT(AllTrue(du32, Lt(bucket, Set(du32, 16))));
      Vec<decltype(di32)> tbl_idx = And(BitCast(di32, bucket), Set(di32, 7));
      const VU32 muls0 = Lookup8(du32, table_ + 0, tbl_idx);
      const VU32 muls1 = Lookup8(du32, table_ + 8, tbl_idx);
      return IfThenElse(Lt(bucket, Set(du32, 8)), muls0, muls1);
    }
  }

  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  HWY_INLINE VU32 ResultFromFeistel(DU32 du32, VU32 LL, VU32 RR) const {
    // Both LL and RR are independent and well-mixed. Use LL as the bucket
    // selector because it is ready earlier, so lookups can overlap with the
    // computation of RR. Use the upper bits so we can pass them through via
    // XOR - see comment below.
    const VU32 bucket = ShiftRight<28>(LL);  // 0..15

    // Per-bucket multiplier pair defines the universal hash family member.
    return MulAndXor(du32, LL, RR, LookupMuls(du32, bucket));
  }

  uint32_t table_[16];  // First, for alignment.
  // The Luby-Rackoff result requires independent round functions; this indeed
  // speeds up construction.
  WeakTwoMul f0_;
  WeakTwoMul f1_;
  WeakTwoMul f2_;
  WeakTwoMul f3_;
};

// Call this overload when it's OK to use the full range of u32 outputs.
inline ShardMul MakeShardMul(Span<const uint64_t> keys, ThreadPool& pool) {
  return ShardMul(BuildShardMul(keys, Span<const uint32_t>(), pool));
}

// Call this overload when we want the generated outputs NOT to overlap with
// `extra_outputs`.
inline ShardMul MakeShardMul(Span<const uint64_t> keys,
                             Span<const uint32_t> extra_outputs,
                             ThreadPool& pool) {
  return ShardMul(BuildShardMul(keys, extra_outputs, pool));
}

#endif  // HWY_TARGET != HWY_SCALAR
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_HASH_SHARDMUL_INL_H_
