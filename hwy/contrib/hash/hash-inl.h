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

// Vectorized, lane-independent hash functions for u32 and u64. This space does
// not seem to have been well-explored yet. Most SIMD hashes (including our own
// HighwayHash) absorb register-sized chunks of strings into a large state.
// We instead compute N hashes in parallel.
//
// Many recent hash functions also rely on operations unavailable or expensive
// on SIMD. For example, rapidhash et al. use u64 x u64 = u128 multiplications
// and others use CRC. Both lack native SIMD instructions and are expensive to
// emulate. Intel is speculated to implement their u32 x u32 = u64 and u32 x u32
// = u32 instructions using two invocations of 16-bit multipliers, which
// explains their higher latency.
// [https://fgiesen.wordpress.com/2024/10/26/why-those-particular-integer-multiplies/]
// 64-bit multiplications are also emulated (three Mul or two MulEven) on x86.
//
// However, for both u32 and u64, multiplications turn out to be clear winners
// in mixing quality per throughput, even on Intel. Surprisingly, they even
// outperform (in terms of throughput) Feistel networks, which operate on u32
// halves. This is likely because multiple inputs (TwoVec) allow hiding the
// multiplication latency. By contrast, Feistel typically involves three or
// even four rounds, each involving two or three u32 multiplications, which
// more than outweighs the advantage of operating on twice as many lanes.

// All our hash functions are also BIJECTIONS (1-1 mappings), constructed by
// chaining reversible operations. This is to allow hashing ~1M u32 keys
// without collisions, which simplifies PHAST perfect hashing. Normally, any
// randomly chosen non-bijective hash function can expect collisions after
// sqrt(2^32) keys. By contrast, bijections rule out any collisions, and are not
// necessarily harder to compute. They also enable an additional space-saving
// optimization in Cuckoo hashing. Rather than storing the original key to
// detect collisions, we observe that a hash collision places both keys in the
// same bucket. This already constrains/verifies the hash bits which became the
// bucket index. It suffices to store and verify only the remaining hash bits.
// Thus we can reconstruct the hash; we could also invert the bijection to
// retrieve the original key, but can instead just compare the hash bits with
// the stored fingerprint to ensure the key is known and not a false positive.
// This is similar to Cuckoo filters, but exact (zero false positives).
//
// Finally, we also support MASKED hash functions, or bijections on [0, 2^k).
// This allows extending to 64-bit keys the above trick which is used in the
// 32-bit cuckoo2x2-inl.h. With typical numbers of buckets in [2^16, 2^20],
// we could support single-u64 buckets (enables Gather) consisting of two slots,
// each with one 30-bit fingerprint plus a 2-bit tag. Together, the bucket bits
// plus fingerprints cover/constrain k=46..50 bit keys. Alternatively, we could
// have single-slot buckets, but with a payload for a hash map. For example,
// k=60 requires 40..44 bit fingerprints, leaving 20..24 bits for a payload.
// In both cases, both the hash input and output should be less than 2^k. Prior
// art includes Wang's masked integer hash, which is essentially also xor-folds
// and multiplies by odd. That has much worse mixing because it is implemented
// with shift/add rather than actual multiplications with high-popcount
// constants. However, we can adapt Wang's technique of masking after each
// multiply to ensure a bijection. The xor-fold is upper-triangular and is
// already invertible without masking.

#if defined(HIGHWAY_HWY_CONTRIB_HASH_HASH_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_HASH_HASH_INL_H_
#undef HIGHWAY_HWY_CONTRIB_HASH_HASH_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_HASH_HASH_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>

#include "hwy/aligned_allocator.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/highway.h"

#if HWY_TARGET != HWY_SCALAR
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// Helper functions for masked hash functions.
namespace detail {

template <size_t kBits, class V>
HWY_INLINE HWY_MUST_USE_RESULT V MaybeMask(const V x, const V mask) {
  if constexpr (kBits != sizeof(TFromV<V>) * 8) {
    return And(x, mask);
  } else {
    (void)mask;
    return x;
  }
}

template <size_t kBits, typename T, HWY_IF_INTEGER(T)>
HWY_INLINE HWY_MUST_USE_RESULT T MaybeMask1(const T x, const T mask) {
  if constexpr (kBits != sizeof(T) * 8) {
    return x & mask;
  } else {
    (void)mask;
    return x;
  }
}

template <size_t kBits, class V, hwy::EnableIf<!hwy::IsInteger<V>()>* = nullptr>
HWY_INLINE void MaybeCompare(const V x, const V mask) {
  if constexpr (kBits != sizeof(TFromV<V>) * 8) {
    HWY_DASSERT(AllTrue(DFromV<V>(), Le(x, mask)));
  } else {
    (void)x;
    (void)mask;
  }
}

template <size_t kBits, typename T, HWY_IF_INTEGER(T)>
HWY_INLINE void MaybeCompare1(const T x, const T mask) {
  if constexpr (kBits != sizeof(T) * 8) {
    HWY_DASSERT(x <= mask);
  } else {
    (void)x;
    (void)mask;
  }
}

}  // namespace detail

// ----------------------------------------------------------------------------
// 32-bit hashes
// Each class provides a scalar operator() and vectorized OneVec/TwoVec.

// Two-multiply + xor-fold by Chris Wellons and TheIronBorn:
// https://github.com/skeeto/hash-prospector/issues/19
// Faster than Triple32 but insufficient to pass SMHasher, especially for
// cyclic keys.
template <size_t kBits>
class MaskedWeakTwoMul {
  static_assert(kBits <= 32);

 public:
  using LaneType = uint32_t;

  static constexpr const char* Name() {
    return kBits == 32 ? "WeakTwoMul" : "MaskedWeakTwoMul";
  }

  static constexpr uint32_t kMask =
      kBits == 32 ? ~uint32_t{0} : (uint32_t{1} << kBits) - 1;

  MaskedWeakTwoMul() = default;
  explicit MaskedWeakTwoMul(uint32_t key) : key_(key) {}
  MaskedWeakTwoMul(AesCtrEngine& engine, uint64_t seed)
      : key_(detail::MaybeMask1<kBits>(
            static_cast<uint32_t>(RngStream(engine, seed)()), kMask)) {}

  uint32_t operator()(uint32_t x) const {
    detail::MaybeCompare1<kBits>(x, kMask);
    x ^= key_;

    x ^= x >> 16;
    x *= 0x21F0AAADu;
    x = detail::MaybeMask1<kBits>(x, kMask);

    x ^= x >> 15;
    x *= 0xF35A2D97u;
    x = detail::MaybeMask1<kBits>(x, kMask);

    x ^= x >> 15;
    detail::MaybeCompare1<kBits>(x, kMask);
    return x;
  }

  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  HWY_INLINE HWY_MUST_USE_RESULT VU32 OneVec(DU32 du32, const VU32 in) const {
    const VU32 mask = Set(du32, kMask);
    detail::MaybeCompare<kBits>(in, mask);
    VU32 hash = Xor(in, Set(du32, key_));

    hash = Xor(hash, ShiftRight<16>(hash));
    hash = Mul(hash, Set(du32, 0x21F0AAADu));
    hash = detail::MaybeMask<kBits>(hash, mask);

    hash = Xor(hash, ShiftRight<15>(hash));
    hash = Mul(hash, Set(du32, 0xF35A2D97u));
    hash = detail::MaybeMask<kBits>(hash, mask);

    hash = Xor(hash, ShiftRight<15>(hash));
    detail::MaybeCompare<kBits>(hash, mask);
    return hash;
  }

  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  HWY_INLINE void TwoVec(DU32 du32, VU32& inout0, VU32& inout1) const {
    inout0 = OneVec(du32, inout0);
    inout1 = OneVec(du32, inout1);
  }

 private:
  uint32_t key_ = 0;
};

using WeakTwoMul = MaskedWeakTwoMul<32>;  // no masking

// Good quality and reasonable speed; safe default for u32.
template <size_t kBits>
class MaskedTriple32 {
  static_assert(kBits <= 32);

 public:
  using LaneType = uint32_t;

  static constexpr const char* Name() {
    return kBits == 32 ? "Triple32" : "MaskedTriple32";
  }

  static constexpr uint32_t kMask =
      kBits == 32 ? ~uint32_t{0} : (uint32_t{1} << kBits) - 1;

  MaskedTriple32() = default;
  explicit MaskedTriple32(uint32_t key) : key_(key) {}
  MaskedTriple32(AesCtrEngine& engine, uint64_t seed)
      : key_(detail::MaybeMask1<kBits>(
            static_cast<uint32_t>(RngStream(engine, seed)()), kMask)) {}

  uint32_t Key() const { return key_; }

  uint32_t operator()(uint32_t x) const {
    detail::MaybeCompare1<kBits>(x, kMask);
    x ^= key_;

    x ^= x >> 17;
    x *= 0xED5AD4BBu;
    x = detail::MaybeMask1<kBits>(x, kMask);

    x ^= x >> 11;
    x *= 0xAC4C1B51u;
    x = detail::MaybeMask1<kBits>(x, kMask);

    x ^= x >> 15;
    x *= 0x31848BABu;
    x = detail::MaybeMask1<kBits>(x, kMask);

    x ^= x >> 14;
    detail::MaybeCompare1<kBits>(x, kMask);
    return x;
  }

  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  HWY_INLINE HWY_MUST_USE_RESULT VU32 OneVec(DU32 du32, const VU32 in) const {
    const VU32 mask = Set(du32, kMask);
    detail::MaybeCompare<kBits>(in, mask);
    VU32 hash = Xor(in, Set(du32, key_));

    hash = Xor(hash, ShiftRight<17>(hash));
    hash = Mul(hash, Set(du32, 0xED5AD4BBu));
    hash = detail::MaybeMask<kBits>(hash, mask);

    hash = Xor(hash, ShiftRight<11>(hash));
    hash = Mul(hash, Set(du32, 0xAC4C1B51u));
    hash = detail::MaybeMask<kBits>(hash, mask);

    hash = Xor(hash, ShiftRight<15>(hash));
    hash = Mul(hash, Set(du32, 0x31848BABu));
    hash = detail::MaybeMask<kBits>(hash, mask);

    hash = Xor(hash, ShiftRight<14>(hash));
    detail::MaybeCompare<kBits>(hash, mask);
    return hash;
  }

  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  HWY_INLINE void TwoVec(DU32 du32, VU32& inout0, VU32& inout1) const {
    inout0 = OneVec(du32, inout0);
    inout1 = OneVec(du32, inout1);
  }

 private:
  uint32_t key_ = 0;
};

using Triple32 = MaskedTriple32<32>;  // no masking

// ----------------------------------------------------------------------------
// 64-bit hashes

// Moremur mixer by Pelle Evensen (= Murmur with much better constants).
// Considerably faster than Feistel: 64-bit multiplies are still reasonably
// affordable despite requiring multiple instructions on x86, and provide
// excellent mixing.
template <size_t kBits>
class MaskedMoremur {
  static_assert(kBits <= 64);

 public:
  using LaneType = uint64_t;

  static constexpr const char* Name() {
    return kBits == 64 ? "Moremur" : "MaskedMoremur";
  }

  static constexpr uint64_t kMask =
      kBits == 64 ? ~uint64_t{0} : (uint64_t{1} << kBits) - 1;

  MaskedMoremur() = default;
  MaskedMoremur(AesCtrEngine& engine, uint64_t seed)
      : key_(detail::MaybeMask1<kBits>(RngStream(engine, seed)(), kMask)) {}

  uint64_t operator()(uint64_t x) const {
    detail::MaybeCompare1<kBits>(x, kMask);
    x ^= key_;

    x ^= x >> 27;
    x *= 0x3C79AC492BA7B653ULL;
    x = detail::MaybeMask1<kBits>(x, kMask);

    x ^= x >> 33;
    x *= 0x1C69B3F74AC4AE35ULL;
    x = detail::MaybeMask1<kBits>(x, kMask);

    x ^= x >> 27;
    detail::MaybeCompare1<kBits>(x, kMask);
    return x;
  }

  template <class DU64, class VU64 = Vec<DU64>, HWY_IF_U64_D(DU64)>
  HWY_INLINE HWY_MUST_USE_RESULT VU64 OneVec(DU64 du64, const VU64 in) const {
    const VU64 mask = Set(du64, kMask);
    detail::MaybeCompare<kBits>(in, mask);
    VU64 hash = Xor(in, Set(du64, key_));

    hash = Xor(hash, ShiftRight<27>(hash));
    hash = Mul(hash, Set(du64, uint64_t{0x3C79AC492BA7B653u}));
    hash = detail::MaybeMask<kBits>(hash, mask);

    hash = Xor(hash, ShiftRight<33>(hash));
    hash = Mul(hash, Set(du64, uint64_t{0x1C69B3F74AC4AE35u}));
    hash = detail::MaybeMask<kBits>(hash, mask);

    hash = Xor(hash, ShiftRight<27>(hash));
    detail::MaybeCompare<kBits>(hash, mask);
    return hash;
  }

  template <class DU64, class VU64 = Vec<DU64>, HWY_IF_U64_D(DU64)>
  HWY_INLINE void TwoVec(DU64 du64, VU64& inout0, VU64& inout1) const {
    inout0 = OneVec(du64, inout0);
    inout1 = OneVec(du64, inout1);
  }

  uint64_t Key() const { return key_; }

 private:
  uint64_t key_ = 0;
};

using Moremur = MaskedMoremur<64>;  // no masking

// ----------------------------------------------------------------------------
#if 0  // obsolete - use one of the above instead

// Fastest: just one multiply. Lower bits are not well-mixed.
class WeakOneMul {
 public:
  using LaneType = uint32_t;
  static constexpr const char* Name() { return "WeakOneMul"; }

  WeakOneMul() = default;

  uint32_t operator()(uint32_t x) const { return x * 0x9E3779B9u; }

  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  HWY_INLINE HWY_MUST_USE_RESULT VU32 OneVec(DU32 du32, const VU32 in) const {
    return Mul(in, Set(du32, 0x9E3779B9u));
  }

  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  HWY_INLINE void TwoVec(DU32 du32, VU32& inout0, VU32& inout1) const {
    inout0 = OneVec(du32, inout0);
    inout1 = OneVec(du32, inout1);
  }
};

// Round-reduced ARX cipher. Considerably slower than Triple32 on AVX2 due to
// the rotates.
class Speck32 {
 public:
  using LaneType = uint32_t;
  static constexpr const char* Name() { return "Speck32"; }

  Speck32(AesCtrEngine& engine, uint64_t seed)
      : keys_(FillRandom<uint16_t>(16, engine, seed)) {}

  uint32_t operator()(uint32_t x) const {
    const ScalableTag<uint32_t> du32;
    auto inout0 = Set(du32, x);
    auto inout1 = inout0;
    TwoVec(du32, inout0, inout1);
    return GetLane(inout0);
  }

  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  HWY_INLINE void TwoVec(DU32 du32, VU32& inout0, VU32& inout1) const {
    const RepartitionToNarrow<DU32> du16;
    using VU16 = Vec<decltype(du16)>;

    // Split each u32 into its even (lower) and odd (upper) u16.
    const VU16 lo = BitCast(du16, inout0);
    const VU16 hi = BitCast(du16, inout1);
    VU16 x0 = ConcatEven(du16, hi, lo);
    VU16 x1 = ConcatOdd(du16, hi, lo);

    // 8 unrolled rounds, one key each.
    Round(du16, x0, x1, Set(du16, keys_[0]));
    Round(du16, x0, x1, Set(du16, keys_[1]));
    Round(du16, x0, x1, Set(du16, keys_[2]));
    Round(du16, x0, x1, Set(du16, keys_[3]));
    Round(du16, x0, x1, Set(du16, keys_[4]));
    Round(du16, x0, x1, Set(du16, keys_[5]));
    Round(du16, x0, x1, Set(du16, keys_[6]));
    Round(du16, x0, x1, Set(du16, keys_[7]));

    // Re-interleave x0 and x1 back into u32.
    inout0 = BitCast(du32, InterleaveWholeLower(du16, x0, x1));
    inout1 = BitCast(du32, InterleaveWholeUpper(du16, x0, x1));
  }

  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  HWY_INLINE HWY_MUST_USE_RESULT VU32 OneVec(DU32 du32, const VU32 in) const {
    const Rebind<uint16_t, DU32> du16;
    using VU16 = Vec<decltype(du16)>;

    // Split each u32 into its even (lower) and odd (upper) u16.
    VU16 x0 = TruncateTo(du16, in);
    VU16 x1 = TruncateTo(du16, ShiftRight<16>(in));

    // 8 unrolled rounds, one key each.
    Round(du16, x0, x1, Set(du16, keys_[0]));
    Round(du16, x0, x1, Set(du16, keys_[1]));
    Round(du16, x0, x1, Set(du16, keys_[2]));
    Round(du16, x0, x1, Set(du16, keys_[3]));
    Round(du16, x0, x1, Set(du16, keys_[4]));
    Round(du16, x0, x1, Set(du16, keys_[5]));
    Round(du16, x0, x1, Set(du16, keys_[6]));
    Round(du16, x0, x1, Set(du16, keys_[7]));

    // Re-interleave x0 and x1 back into u32.
    const Twice<decltype(du16)> du16t;
    const VU16 lo = InterleaveWholeLower(du16, x0, x1);
    const VU16 hi = InterleaveWholeUpper(du16, x0, x1);
    return BitCast(du32, Combine(du16t, lo, hi));
  }

 private:
  // One round of Speck32: mix data (x0, x1) using k0.
  template <class DU16, class VU16 = Vec<DU16>, HWY_IF_U16_D(DU16)>
  static HWY_INLINE void Round(DU16, VU16& x0, VU16& x1, const VU16 k0) {
    x0 = RotateRight<7>(x0);
    x0 = Add(x0, x1);
    x0 = Xor(x0, k0);
    x1 = RotateLeft<2>(x1);
    x1 = Xor(x1, x0);
  }

  AlignedVector<uint16_t> keys_;
};

// Lai-Massey diffuses faster than Feistel because it updates both halves
// concurrently, but this is also a weakness in that input differentials
// partially cancel, leading to collisions in DiffDist. By contrast, Feistel
// updates one half at a time and has more nonlinear depth.
class WeakLaiMassey3Mul2 {
 public:
  using LaneType = uint32_t;
  static constexpr const char* Name() { return "WeakLaiMassey3Mul2"; }

  WeakLaiMassey3Mul2(AesCtrEngine& engine, uint64_t seed)
      : keys_(FillRandom<uint16_t>(6, engine, seed)) {}

  uint32_t operator()(uint32_t inout) const {
    ScalableTag<uint32_t> du32;
    auto inout0 = Set(du32, inout);
    auto inout1 = inout0;
    TwoVec(du32, inout0, inout1);
    return GetLane(inout1);
  }

  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  HWY_INLINE void TwoVec(DU32 du32, VU32& inout0, VU32& inout1) const {
    const RepartitionToNarrow<DU32> du16;
    using VU16 = Vec<decltype(du16)>;

    // Lai-Massey turns any F into a bijection. Split each u32 into its even
    // (lower) and odd (upper) u16, as in Randen's GFN.
    const VU16 lo = BitCast(du16, inout0);
    const VU16 hi = BitCast(du16, inout1);
    VU16 LL = ConcatEven(du16, hi, lo);
    VU16 RR = ConcatOdd(du16, hi, lo);

    // Must apply the same function to all lanes, hence broadcast.
    const VU16 kKey0 = Set(du16, keys_[0]);
    const VU16 kKey1 = Set(du16, keys_[1]);
    const VU16 kKey2 = Set(du16, keys_[2]);
    const VU16 kKey3 = Set(du16, keys_[3]);
    const VU16 kKey4 = Set(du16, keys_[4]);
    const VU16 kKey5 = Set(du16, keys_[5]);
    const VU16 kMul0 = Set(du16, 0xA3D3u);
    const VU16 kMul1 = Set(du16, 0x4B2Du);

    // Alternate keys for at least some variation.
    F(du16, kKey0, kKey1, kMul0, kMul1, LL, RR);
    F(du16, kKey2, kKey3, kMul0, kMul1, LL, RR);
    F(du16, kKey4, kKey5, kMul0, kMul1, LL, RR);

    // Re-interleave LL and RR back into u32.
    inout0 = BitCast(du32, InterleaveWholeLower(du16, LL, RR));
    inout1 = BitCast(du32, InterleaveWholeUpper(du16, LL, RR));
  }

 private:
  template <class DU16, class VU16 = Vec<DU16>, HWY_IF_U16_D(DU16)>
  static HWY_INLINE void F(DU16 du16, const VU16 kKey0, const VU16 kKey1,
                           const VU16 kMul0, const VU16 kMul1, VU16& LL,
                           VU16& RR) {
    VU16 T = Xor3(LL, RR, kKey0);
    T = Xor(T, ShiftRight<8>(T));
    T = Mul(T, kMul0);
    T = Xor(T, ShiftRight<7>(T));
    T = Mul(T, kMul1);
    T = Xor3(T, ShiftRight<9>(T), kKey1);
    LL = Xor(LL, T);
    RR = Xor(RR, T);
    RR = RotateRight<7>(RR);  // near orthomorphism
  }

  AlignedVector<uint16_t> keys_;
};

// Adapted from the Murmur string hash. Obsoleted by Triple32.
class Murmur3 {
 public:
  using LaneType = uint32_t;
  static constexpr const char* Name() { return "Murmur3"; }

  Murmur3(AesCtrEngine& engine, uint64_t seed)
      : keys_(FillRandom<uint32_t>(1, engine, seed)) {}

  uint32_t operator()(uint32_t x) const {
    ScalableTag<uint32_t> du32;
    return GetLane(OneVec(du32, Set(du32, x)));
  }

  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  HWY_INLINE void TwoVec(DU32 du32, VU32& inout0, VU32& inout1) const {
    inout0 = OneVec(du32, inout0);
    inout1 = OneVec(du32, inout1);
  }

 private:
  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  HWY_INLINE HWY_MUST_USE_RESULT VU32 OneVec(DU32 du32, const VU32 in) const {
    const VU32 c1 = Set(du32, 0xcc9e2d51);
    const VU32 c2 = Set(du32, 0x1b873593);
    const VU32 m = Set(du32, 5);
    const VU32 n = Set(du32, 0xe6546b64);

    VU32 hash = Set(du32, keys_[0]);

    VU32 k = in;
    k = Mul(k, c1);
    k = RotateLeft<15>(k);
    k = Mul(k, c2);

    hash ^= k;
    hash = RotateLeft<13>(hash) * m + n;

    // No XOR by length here.

    hash = Xor(hash, ShiftRight<16>(hash));
    hash = Mul(hash, Set(du32, 0x85EBCA6Bu));
    hash = Xor(hash, ShiftRight<13>(hash));
    hash = Mul(hash, Set(du32, 0x85EBCA6Bu));
    hash = Xor(hash, ShiftRight<16>(hash));

    return hash;
  }

  AlignedVector<uint32_t> keys_;
};

// https://github.com/gzm55/hash-garage/tree/master; fails DiffDist.
class WeakNMHash {
 public:
  using LaneType = uint32_t;
  static constexpr const char* Name() { return "WeakNMHash"; }

  WeakNMHash(AesCtrEngine& engine, uint64_t seed)
      : keys_(FillRandom<uint32_t>(1, engine, seed)) {}

  uint32_t operator()(uint32_t x) const {
    ScalableTag<uint32_t> du32;
    return GetLane(OneVec(du32, Set(du32, x)));
  }

  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  HWY_INLINE void TwoVec(DU32 du32, VU32& inout0, VU32& inout1) const {
    inout0 = OneVec(du32, inout0);
    inout1 = OneVec(du32, inout1);
  }

 private:
  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  HWY_INLINE HWY_MUST_USE_RESULT VU32 OneVec(DU32 du32, const VU32 in) const {
    VU32 hash = Xor(in, Set(du32, keys_[0]));
    hash = Mul(hash, Set(du32, 0xBDAB1EA9u));
    hash = Xor(hash, ShiftRight<18>(hash));
    hash = Mul(hash, Set(du32, 0xA7896A1Bu));
    hash = Xor(hash, ShiftRight<12>(hash));
    hash = Mul(hash, Set(du32, 0x83796A2Du));
    hash = Xor(hash, ShiftRight<16>(hash));
    return hash;
  }

  AlignedVector<uint32_t> keys_;
};

// Xmrx: xor-shift, multiply, xor-rotate-rotate. Excellent throughput on Zen5,
// but fails hash_eval TwoBytes, Sparse, Avalanche, RotIota.
// https://jonkagstrom.com/bit-mixer-construction/
class WeakXmrx {
 public:
  using LaneType = uint64_t;
  static constexpr const char* Name() { return "Xmrx"; }

  WeakXmrx(AesCtrEngine& engine, uint64_t seed)
      : key_(RngStream(engine, seed)()) {}

  uint64_t operator()(uint64_t x) const {
    x ^= key_;
    x ^= x >> 32;
    x *= 0xFF51AFD7ED558CCDULL;
    x ^= (x >> 47 | x << 17) ^ (x >> 23 | x << 41);
    return x;
  }

  template <class DU64, class VU64 = Vec<DU64>, HWY_IF_U64_D(DU64)>
  HWY_INLINE HWY_MUST_USE_RESULT VU64 OneVec(DU64 du64, const VU64 in) const {
    VU64 hash = Xor(in, Set(du64, key_));
    hash = Xor(hash, ShiftRight<32>(hash));
    hash = Mul(hash, Set(du64, uint64_t{0xFF51AFD7ED558CCDu}));
    hash = Xor3(hash, RotateRight<47>(hash), RotateRight<23>(hash));
    return hash;
  }

  template <class DU64, class VU64 = Vec<DU64>, HWY_IF_U64_D(DU64)>
  HWY_INLINE void TwoVec(DU64 du64, VU64& inout0, VU64& inout1) const {
    inout0 = OneVec(du64, inout0);
    inout1 = OneVec(du64, inout1);
  }

 private:
  uint64_t key_;
};

// Feistel using four rounds of WeakTwoMul. Splits each u64 into two u32 halves,
// Slightly faster than Feistel3Mul3 on Zen5, with ~same hash_eval score, but
// considerably slower than Moremur.
class Feistel4Mul2 {
 public:
  using LaneType = uint64_t;
  static constexpr const char* Name() { return "Feistel4Mul2"; }

  Feistel4Mul2(AesCtrEngine& engine, uint64_t seed)
      : f0_(engine, 2 * seed + 0), f1_(engine, 2 * seed + 1) {}

  uint64_t operator()(uint64_t x) const {
    ScalableTag<uint64_t> du64;
    return GetLane(OneVec(du64, Set(du64, x)));
  }

  template <class DU64, class VU64 = Vec<DU64>, HWY_IF_U64_D(DU64)>
  HWY_INLINE HWY_MUST_USE_RESULT VU64 OneVec(DU64 du64, const VU64 in) const {
    const Rebind<uint32_t, DU64> du32;
    using VU32 = Vec<decltype(du32)>;

    // Split each u64 into lower and upper u32.
    VU32 LL = TruncateTo(du32, in);
    VU32 RR = TruncateTo(du32, ShiftRight<32>(in));

    // 4 Feistel rounds, alternating between f0_ and f1_.
    LL = Xor(LL, f0_.OneVec(du32, RR));
    RR = Xor(RR, f1_.OneVec(du32, LL));
    LL = Xor(LL, f0_.OneVec(du32, RR));
    RR = Xor(RR, f1_.OneVec(du32, LL));

    // Re-interleave LL and RR back into u64.
    const Twice<decltype(du32)> du32t;
    const VU32 lo = InterleaveWholeLower(du32, LL, RR);
    const VU32 hi = InterleaveWholeUpper(du32, LL, RR);
    return BitCast(du64, Combine(du32t, hi, lo));
  }

  template <class DU64, class VU64 = Vec<DU64>, HWY_IF_U64_D(DU64)>
  HWY_INLINE void TwoVec(DU64 du64, VU64& inout0, VU64& inout1) const {
    const RepartitionToNarrow<DU64> du32;
    using VU32 = Vec<decltype(du32)>;

    // Split each u64 into lower and upper u32.
    const VU32 lo = BitCast(du32, inout0);
    const VU32 hi = BitCast(du32, inout1);
    VU32 LL = ConcatEven(du32, hi, lo);
    VU32 RR = ConcatOdd(du32, hi, lo);

    // 4 Feistel rounds, alternating between f0_ and f1_.
    LL = Xor(LL, f0_.OneVec(du32, RR));
    RR = Xor(RR, f1_.OneVec(du32, LL));
    LL = Xor(LL, f0_.OneVec(du32, RR));
    RR = Xor(RR, f1_.OneVec(du32, LL));

    // Re-interleave LL and RR back into u64.
    inout0 = BitCast(du64, InterleaveWholeLower(du32, LL, RR));
    inout1 = BitCast(du64, InterleaveWholeUpper(du32, LL, RR));
  }

 private:
  WeakTwoMul f0_;
  WeakTwoMul f1_;
};

// Feistel using three rounds of Triple32. Splits each u64 into two u32 halves.
// The round pattern (LL, RR, LL) ensures the lower 32 bits are mixed twice,
// which is important because they are used as bucket indices.
class Feistel3Mul3 {
 public:
  using LaneType = uint64_t;
  static constexpr const char* Name() { return "Feistel3Mul3"; }

  Feistel3Mul3(AesCtrEngine& engine, uint64_t seed)
      : f0_(engine, 3 * seed + 0),
        f1_(engine, 3 * seed + 1),
        f2_(engine, 3 * seed + 2) {}

  uint64_t operator()(uint64_t x) const {
    ScalableTag<uint64_t> du64;
    return GetLane(OneVec(du64, Set(du64, x)));
  }

  template <class DU64, class VU64 = Vec<DU64>, HWY_IF_U64_D(DU64)>
  HWY_INLINE HWY_MUST_USE_RESULT VU64 OneVec(DU64 du64, const VU64 in) const {
    const Rebind<uint32_t, DU64> du32;
    using VU32 = Vec<decltype(du32)>;

    // Split each u64 into lower and upper u32.
    VU32 LL = TruncateTo(du32, in);
    VU32 RR = TruncateTo(du32, ShiftRight<32>(in));

    // 3 Feistel rounds: lower bits (LL) are mixed in rounds 1 and 3.
    LL = Xor(LL, f0_.OneVec(du32, RR));
    RR = Xor(RR, f1_.OneVec(du32, LL));
    LL = Xor(LL, f2_.OneVec(du32, RR));

    // Re-interleave LL and RR back into u64.
    const Twice<decltype(du32)> du32t;
    const VU32 lo = InterleaveWholeLower(du32, LL, RR);
    const VU32 hi = InterleaveWholeUpper(du32, LL, RR);
    return BitCast(du64, Combine(du32t, hi, lo));
  }

  template <class DU64, class VU64 = Vec<DU64>, HWY_IF_U64_D(DU64)>
  HWY_INLINE void TwoVec(DU64 du64, VU64& inout0, VU64& inout1) const {
    const RepartitionToNarrow<DU64> du32;
    using VU32 = Vec<decltype(du32)>;

    // Split each u64 into lower and upper u32.
    const VU32 lo = BitCast(du32, inout0);
    const VU32 hi = BitCast(du32, inout1);
    VU32 LL = ConcatEven(du32, hi, lo);
    VU32 RR = ConcatOdd(du32, hi, lo);

    // 3 Feistel rounds: lower bits (LL) are mixed in rounds 1 and 3.
    LL = Xor(LL, f0_.OneVec(du32, RR));
    RR = Xor(RR, f1_.OneVec(du32, LL));
    LL = Xor(LL, f2_.OneVec(du32, RR));

    // Re-interleave LL and RR back into u64.
    inout0 = BitCast(du64, InterleaveWholeLower(du32, LL, RR));
    inout1 = BitCast(du64, InterleaveWholeUpper(du32, LL, RR));
  }

 private:
  Triple32 f0_;
  Triple32 f1_;
  Triple32 f2_;
};

// Nasam by Pelle Evensen. Slightly better hash_eval scores, but 1.1-1.2x
// and 1.2-1.3x more expensive for scalar and SIMD, and rotates are slower on
// AVX2 and interfere with masking.
class Nasam {
 public:
  using LaneType = uint64_t;
  static constexpr const char* Name() { return "Nasam"; }

  Nasam(AesCtrEngine& engine, uint64_t seed)
      : key_(RngStream(engine, seed)()) {}

  uint64_t operator()(uint64_t x) const {
    x ^= key_;
    x ^= (x >> 25 | x << 39) ^ (x >> 47 | x << 17);
    x *= 0x9E6C63D0676A9A99ULL;
    x ^= (x >> 23) ^ (x >> 51);
    x *= 0x9E6D62D06F6A9A9BULL;
    x ^= (x >> 23) ^ (x >> 51);
    return x;
  }

  template <class DU64, class VU64 = Vec<DU64>, HWY_IF_U64_D(DU64)>
  HWY_INLINE HWY_MUST_USE_RESULT VU64 OneVec(DU64 du64, const VU64 in) const {
    VU64 hash = Xor(in, Set(du64, key_));
    hash = Xor3(hash, RotateRight<25>(hash), RotateRight<47>(hash));
    hash = Mul(hash, Set(du64, uint64_t{0x9E6C63D0676A9A99u}));
    hash = Xor3(hash, ShiftRight<23>(hash), ShiftRight<51>(hash));
    hash = Mul(hash, Set(du64, uint64_t{0x9E6D62D06F6A9A9Bu}));
    hash = Xor3(hash, ShiftRight<23>(hash), ShiftRight<51>(hash));
    return hash;
  }

  template <class DU64, class VU64 = Vec<DU64>, HWY_IF_U64_D(DU64)>
  HWY_INLINE void TwoVec(DU64 du64, VU64& inout0, VU64& inout1) const {
    inout0 = OneVec(du64, inout0);
    inout1 = OneVec(du64, inout1);
  }

 private:
  uint64_t key_;
};

#endif  // obsolete

// In-place version.
template <class Hash, typename T = typename Hash::LaneType>
static HWY_MAYBE_UNUSED void HashArray(const Hash& hash, T* HWY_RESTRICT inout,
                                       size_t count) {
  const ScalableTag<T> d;
  using V = Vec<decltype(d)>;
  HWY_LANES_CONSTEXPR size_t N = Lanes(d);

  size_t i = 0;
  if (HWY_LIKELY(count >= 4 * N)) {
    for (; i <= count - 4 * N; i += 4 * N) {
      V v0 = Load(d, inout + i + 0 * N);
      V v1 = Load(d, inout + i + 1 * N);
      V v2 = Load(d, inout + i + 2 * N);
      V v3 = Load(d, inout + i + 3 * N);
      hash.TwoVec(d, v0, v1);
      hash.TwoVec(d, v2, v3);
      Store(v0, d, inout + i + 0 * N);
      Store(v1, d, inout + i + 1 * N);
      Store(v2, d, inout + i + 2 * N);
      Store(v3, d, inout + i + 3 * N);
    }
  }
  size_t remaining = count - i;
  for (; remaining >= N; i += N, remaining -= N) {
    V v0 = Load(d, inout + i);
    v0 = hash.OneVec(d, v0);
    Store(v0, d, inout + i);
  }
  {
    V v0 = LoadN(d, inout + i, remaining);
    v0 = hash.OneVec(d, v0);
    StoreN(v0, d, inout + i, remaining);
  }
}

// Same, but separate input and output arrays.
template <class Hash, typename T = typename Hash::LaneType>
static HWY_MAYBE_UNUSED void HashArray(const Hash& hash,
                                       const T* HWY_RESTRICT in,
                                       T* HWY_RESTRICT out, size_t count) {
  const ScalableTag<T> d;
  using V = Vec<decltype(d)>;
  HWY_LANES_CONSTEXPR size_t N = Lanes(d);

  size_t i = 0;
  if (HWY_LIKELY(count >= 4 * N)) {
    for (; i <= count - 4 * N; i += 4 * N) {
      V v0 = Load(d, in + i + 0 * N);
      V v1 = Load(d, in + i + 1 * N);
      V v2 = Load(d, in + i + 2 * N);
      V v3 = Load(d, in + i + 3 * N);
      hash.TwoVec(d, v0, v1);
      hash.TwoVec(d, v2, v3);
      Store(v0, d, out + i + 0 * N);
      Store(v1, d, out + i + 1 * N);
      Store(v2, d, out + i + 2 * N);
      Store(v3, d, out + i + 3 * N);
    }
  }
  size_t remaining = count - i;
  for (; remaining >= N; i += N, remaining -= N) {
    V v0 = Load(d, in + i);
    v0 = hash.OneVec(d, v0);
    Store(v0, d, out + i);
  }
  {
    V v0 = LoadN(d, in + i, remaining);
    v0 = hash.OneVec(d, v0);
    StoreN(v0, d, out + i, remaining);
  }
}

template <class Func>
void ForeachHash(AesCtrEngine& engine, uint64_t seed, const Func& func) {
  func(Triple32(engine, seed));
  // func(WeakTwoMul(engine, seed));

  // func(Speck32(engine, seed));
  // func(WeakLaiMassey3Mul2(engine, seed));
  // func(Murmur3(engine, seed));
  // func(WeakOneMul(engine, seed));
  // func(WeakNMHash(engine, seed));
}

template <class Func>
void ForeachHash64(AesCtrEngine& engine, uint64_t seed, const Func& func) {
  func(Moremur(engine, seed));
  // func(Nasam(engine, seed));
  // func(Feistel4Mul2(engine, seed));
  // func(Feistel3Mul3(engine, seed));
  // func(WeakXmrx(engine, seed));
}

// Returns vector filled with a bijection of a counter. This is not the same as
// a permutation of [0, count), but no values repeat.
template <typename T>
AlignedVector<T> FillRandomDistinct(size_t count, uint32_t key) {
  Triple32 permutation(key);
  AlignedVector<T> v;
  v.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    v.push_back(permutation(static_cast<uint32_t>(i)));
  }
  return v;
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
#endif  // HWY_TARGET != HWY_SCALAR

#endif  // HIGHWAY_HWY_CONTRIB_HASH_HASH_INL_H_
