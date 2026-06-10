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

// Vectorized hash functions, initially 32-bit because this doubles throughput
// compared to 64-bit. To allow hashing millions of keys, which is very large
// relative to the sqrt(2^32) birthday bound, we only implement permutations
// (bijections) using only reversible operations. This rules out any collisions.
//
// This space does not seem to have been well-explored yet. Most hashes using
// SIMD, including our own HighwayHash, absorb register-sized chunks of strings
// into a large state. Unpublished internal experiments in 2012 used PMADDWD,
// but also for string hashing. We instead compute N hashes in parallel.
//
// Many recent hash functions rely on operations unavailable or expensive on
// SIMD. For example, rapidhash et al. use u64 x u64 = u128 multiplications and
// others use CRC. Both often lack native SIMD instructions and are expensive to
// emulate. Even 32-bit multiplications might be slower than expected. Intel is
// speculated to implement their u32 x u32 = u64 and u32 x u32 = u32
// instructions using two invocations of 16-bit multipliers, which explains
// their higher latency.
// [https://fgiesen.wordpress.com/2024/10/26/why-those-particular-integer-multiplies/]
//
// However, 32-bit multiplications (Triple32) turn out to be clear winners in
// mixing quality per throughput, even on Intel.

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

// Each class provides an operator() and TwoVec. Both are used by hash_bench
// and hash_eval.

// Theoretically sound, but always at least 2x slower than Triple32.
class Feistel4Mul2 {
 public:
  static constexpr const char* Name() { return "Feistel4Mul2"; }

  Feistel4Mul2(AesCtrEngine& engine, uint64_t seed)
      : keys_(FillRandom<uint16_t>(2, engine, seed)) {}

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

    // Feistel turns any F into a bijection. Split each u32 into its even
    // (lower) and odd (upper) u16. Randen also splits, but into 128-bit blocks.
    const VU16 lo = BitCast(du16, inout0);
    const VU16 hi = BitCast(du16, inout1);
    VU16 LL = ConcatEven(du16, hi, lo);
    VU16 RR = ConcatOdd(du16, hi, lo);

    // Feistel must apply the same function to all lanes, hence broadcast.
    const VU16 kKey0 = Set(du16, keys_[0]);
    const VU16 kKey1 = Set(du16, keys_[1]);
    const VU16 kMul0 = Set(du16, 0xA3D3u);
    const VU16 kMul1 = Set(du16, 0x4B2Du);

    // Alternate keys for at least some variation.
    LL = FeistelMul(du16, RR, LL, kKey0, kMul0, kMul1);
    RR = FeistelMul(du16, LL, RR, kKey1, kMul0, kMul1);
    LL = FeistelMul(du16, RR, LL, kKey0, kMul0, kMul1);
    RR = FeistelMul(du16, LL, RR, kKey1, kMul0, kMul1);

    // Re-interleave LL and RR back into u32.
    inout0 = BitCast(du32, InterleaveWholeLower(du16, LL, RR));
    inout1 = BitCast(du32, InterleaveWholeUpper(du16, LL, RR));
  }

 private:
  template <class DU16, class VU16 = Vec<DU16>, HWY_IF_U16_D(DU16)>
  static HWY_INLINE VU16 FeistelMul(DU16, VU16 x, const VU16 other,
                                    const VU16 kKey, const VU16 kMul0,
                                    const VU16 kMul1) {
    x = Xor(x, ShiftRight<8>(x));
    x = Mul(x, kMul0);
    x = Xor(x, ShiftRight<7>(x));
    x = Mul(x, kMul1);
    x = Xor(x, ShiftRight<9>(x));
    return Xor3(x, kKey, other);
  }

  AlignedVector<uint16_t> keys_;
};

// Low bias, fastest.
class Triple32 {
 public:
  static constexpr const char* Name() { return "Triple32"; }

  Triple32() = default;
  explicit Triple32(uint32_t key) : key_(key) {}
  Triple32(AesCtrEngine& engine, uint64_t seed)
      : key_(static_cast<uint32_t>(RngStream(engine, seed)())) {}

  uint32_t Key() const { return key_; }

  uint32_t operator()(uint32_t x) const {
    ScalableTag<uint32_t> du32;
    return GetLane(OneVec(du32, Set(du32, x)));
  }

  // Used by Phast.
  template <class DU32, class VU32 = Vec<DU32>, HWY_IF_U32_D(DU32)>
  HWY_INLINE VU32 OneVec(DU32 du32, const VU32 in) const {
    VU32 hash = Xor(in, Set(du32, key_));
    hash = Xor(hash, ShiftRight<17>(hash));
    hash = Mul(hash, Set(du32, 0xED5AD4BBu));
    hash = Xor(hash, ShiftRight<11>(hash));
    hash = Mul(hash, Set(du32, 0xAC4C1B51u));
    hash = Xor(hash, ShiftRight<15>(hash));
    hash = Mul(hash, Set(du32, 0x31848BABu));
    hash = Xor(hash, ShiftRight<14>(hash));
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

// Round-reduced ARX cipher. Considerably slower than Triple32 on AVX2 due to
// the rotates, but slightly faster than Feistel4Mul2 on Turin.
class Speck32 {
 public:
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

#if 0  // obsolete - use one of the above instead

// Lai-Massey diffuses faster than Feistel because it updates both halves
// concurrently, but this is also a weakness in that input differentials
// partially cancel, leading to collisions in DiffDist. By contrast, Feistel
// updates one half at a time and has more nonlinear depth.
class WeakLaiMassey3Mul2 {
 public:
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
  HWY_INLINE VU32 OneVec(DU32 du32, const VU32 in) const {
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

// Better constants found by TheIronBorn. Super-fast but insufficient to pass
// SMHasher, especially for cyclic keys.
class WeakTwoMul {
 public:
  static constexpr const char* Name() { return "WeakTwoMul"; }

  WeakTwoMul(AesCtrEngine& engine, uint64_t seed)
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
  HWY_INLINE VU32 OneVec(DU32 du32, const VU32 in) const {
    VU32 hash = Xor(in, Set(du32, keys_[0]));
    hash = Xor(hash, ShiftRight<16>(hash));
    hash = Mul(hash, Set(du32, 0x21F0AAADu));
    hash = Xor(hash, ShiftRight<15>(hash));
    hash = Mul(hash, Set(du32, 0xF35A2D97u));
    hash = Xor(hash, ShiftRight<15>(hash));
    return hash;
  }

  AlignedVector<uint32_t> keys_;
};

// https://github.com/gzm55/hash-garage/tree/master; fails DiffDist.
class WeakNMHash {
 public:
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
  HWY_INLINE VU32 OneVec(DU32 du32, const VU32 in) const {
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

#endif  // obsolete

// In-place version.
template <class Hash>
static void HashArray(const Hash& hash, uint32_t* HWY_RESTRICT inout,
                      size_t count) {
  const ScalableTag<uint32_t> du32;
  using VU32 = Vec<decltype(du32)>;
  HWY_LANES_CONSTEXPR size_t N = Lanes(du32);

  size_t i = 0;
  if (HWY_LIKELY(count >= 4 * N)) {
    for (; i <= count - 4 * N; i += 4 * N) {
      VU32 v0 = Load(du32, inout + i + 0 * N);
      VU32 v1 = Load(du32, inout + i + 1 * N);
      VU32 v2 = Load(du32, inout + i + 2 * N);
      VU32 v3 = Load(du32, inout + i + 3 * N);
      hash.TwoVec(du32, v0, v1);
      hash.TwoVec(du32, v2, v3);
      Store(v0, du32, inout + i + 0 * N);
      Store(v1, du32, inout + i + 1 * N);
      Store(v2, du32, inout + i + 2 * N);
      Store(v3, du32, inout + i + 3 * N);
    }
  }
  for (; i < count; ++i) {
    inout[i] = hash(inout[i]);
  }
}

// Same, but separate input and output arrays.
template <class Hash>
static void HashArray(const Hash& hash, const uint32_t* HWY_RESTRICT in,
                      uint32_t* HWY_RESTRICT out, size_t count) {
  const ScalableTag<uint32_t> du32;
  using VU32 = Vec<decltype(du32)>;
  HWY_LANES_CONSTEXPR size_t N = Lanes(du32);

  size_t i = 0;
  if (HWY_LIKELY(count >= 4 * N)) {
    for (; i <= count - 4 * N; i += 4 * N) {
      VU32 v0 = Load(du32, in + i + 0 * N);
      VU32 v1 = Load(du32, in + i + 1 * N);
      VU32 v2 = Load(du32, in + i + 2 * N);
      VU32 v3 = Load(du32, in + i + 3 * N);
      hash.TwoVec(du32, v0, v1);
      hash.TwoVec(du32, v2, v3);
      Store(v0, du32, out + i + 0 * N);
      Store(v1, du32, out + i + 1 * N);
      Store(v2, du32, out + i + 2 * N);
      Store(v3, du32, out + i + 3 * N);
    }
  }
  for (; i < count; ++i) {
    out[i] = hash(in[i]);
  }
}

template <class Func>
void ForeachHash(AesCtrEngine& engine, uint64_t seed, const Func& func) {
  func(Feistel4Mul2(engine, seed));
  func(Triple32(engine, seed));
  func(Speck32(engine, seed));
  // func(WeakLaiMassey3Mul2(engine, seed));
  // func(Murmur3(engine, seed));
  // func(WeakTwoMul(engine, seed));
  // func(WeakNMHash(engine, seed));
}

// Returns vector filled with a bijection of a counter. This is not the same as
// a permutation of [0, count), but no values repeat.
template <typename T>
AlignedVector<T> FillRandomDistinct(size_t count, uint32_t key) {
  Triple32 permutation(key);
  AlignedVector<T> v;
  v.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    v.push_back(permutation(i));
  }
  return v;
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
#endif  // HWY_TARGET != HWY_SCALAR

#endif  // HIGHWAY_HWY_CONTRIB_HASH_HASH_INL_H_
