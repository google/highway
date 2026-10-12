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

// Portable HighwayHash: a fast, keyed, strong (SIMD) hash. This is a
// re-implementation of https://github.com/google/highwayhash expressed via
// Highway ops, so a single source runs on every Highway target (x86, NEON,
// SVE, RVV, PPC, WASM, LoongArch, ...) and returns the frozen golden values.
//
// The 256-bit internal state is four u64 lanes for each of {v0, v1, mul0,
// mul1}. Using CappedTag<uint64_t, 4> with a loop over Lanes(d) processes all
// four lanes at once on >= 256-bit targets while naturally running two 128-bit
// iterations on 128-bit targets.
//
// The state is stored as plain uint64_t arrays; each round loads them into
// vectors, mixes, and stores them back. On fixed-size targets the compiler
// keeps them in registers (SROA), so the loads/stores fold away - see the
// objdump in PR #3334. The scalable targets (SVE, RVV) cannot hold a vector in
// a struct member or array, which is why the array form is used rather than
// vector members. Only HWY_SCALAR (1 lane) is excluded.
//
// Reference: J. Alakuijala, B. Cox, J. Wassenberg, "Fast keyed hash/pseudo-
// random function using SIMD multiply and permute",
// https://arxiv.org/abs/1612.06257

// Include guard (still compiled once per target)
#if defined(HIGHWAY_HWY_CONTRIB_HASH_HIGHWAYHASH_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_HASH_HIGHWAYHASH_INL_H_
#undef HIGHWAY_HWY_CONTRIB_HASH_HIGHWAYHASH_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_HASH_HIGHWAYHASH_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace highwayhash {

// 32-byte packet; the hash absorbs the message in units of this size.
static constexpr size_t kPacketSize = 32;

// ZipperMerge and ModularReduction operate on 16-byte blocks, so at least 128
// bits are required. HWY_SCALAR is 1 lane and cannot provide that.
#if HWY_TARGET != HWY_SCALAR

namespace detail {

using D64 = CappedTag<uint64_t, 4>;
using D32 = Repartition<uint32_t, D64>;
using D8 = Repartition<uint8_t, D64>;
using V64 = VFromD<D64>;
using V8 = VFromD<D8>;

// Swap the two 32-bit halves of each u64 lane (== rotate the u64 by 32).
HWY_INLINE V64 Rotate64By32(V64 v) {
  const D32 d32;
  return BitCast(D64(), Reverse2(d32, BitCast(d32, v)));
}

// Fixed 16-byte permutation. See ZipperMerge in the reference: it scatters the
// well-mixed low bytes of a 32x32 product across the lane and its neighbour,
// keeping the least-mixed bytes in the upper 32 bits (unused by the next mul).
HWY_INLINE V64 ZipperMerge(V64 v) {
  const D8 d8;
  // Reference masks lo=0x000F010E05020C03, hi=0x070806090D0A040B, applied by
  // pshufb; those are the little-endian index bytes below.
  const V8 idx = Dup128VecFromValues(d8, 3, 12, 2, 5, 14, 1, 15, 0, 11, 4, 10,
                                     13, 9, 6, 8, 7);
#if HWY_IS_BIG_ENDIAN
  v = ReverseLaneBytes(v);
#endif
  V64 r = BitCast(D64(), TableLookupBytes(BitCast(d8, v), idx));
#if HWY_IS_BIG_ENDIAN
  r = ReverseLaneBytes(r);
#endif
  return r;
}

// (u32)a[i] * (a2[i] >> 32) for each u64 lane, via the "even 32-bit lanes"
// multiply. BitCast(d32, a) even lanes are the low 32 bits; ShiftRight<32>
// moves the high 32 bits into that position.
HWY_INLINE V64 MulLoHi(V64 lo_src, V64 hi_src) {
  const D32 d32;
#if HWY_IS_BIG_ENDIAN
  return MulOdd(BitCast(d32, lo_src), BitCast(d32, ShiftRight<32>(hi_src)));
#else
  return MulEven(BitCast(d32, lo_src), BitCast(d32, ShiftRight<32>(hi_src)));
#endif
}

// The 256-bit state, as four 4-lane arrays (v0/v1/mul0/mul1).
// Plain arrays so State is a value type on every target, including SVE/RVV
// where a vector cannot be a struct member. Each round loads them into vectors
// and stores them back; on fixed-size targets the compiler keeps them in
// registers.
struct State {
  HWY_ALIGN uint64_t v0[4];
  HWY_ALIGN uint64_t v1[4];
  HWY_ALIGN uint64_t mul0[4];
  HWY_ALIGN uint64_t mul1[4];

  // Core round for lanes [i, i + Lanes(D64())).
  HWY_INLINE void Update(V64 pkt, size_t i) {
    const D64 d;
    V64 v0_i = Load(d, v0 + i);
    V64 v1_i = Load(d, v1 + i);
    V64 mul0_i = Load(d, mul0 + i);
    V64 mul1_i = Load(d, mul1 + i);

    v1_i = Add(Add(v1_i, pkt), mul0_i);
    mul0_i = Xor(mul0_i, MulLoHi(v1_i, v0_i));
    v0_i = Add(v0_i, mul1_i);
    mul1_i = Xor(mul1_i, MulLoHi(v0_i, v1_i));
    v0_i = Add(v0_i, ZipperMerge(v1_i));
    v1_i = Add(v1_i, ZipperMerge(v0_i));

    Store(v0_i, d, v0 + i);
    Store(v1_i, d, v1 + i);
    Store(mul0_i, d, mul0 + i);
    Store(mul1_i, d, mul1 + i);
  }

  HWY_INLINE void UpdatePacket(const uint8_t* HWY_RESTRICT packet) {
    const D64 d;
    const D8 d8;
    const size_t N = Lanes(d);
    for (size_t i = 0; i < 4; i += N) {
      // Load as bytes (no aliasing/alignment assumptions on the caller's
      // buffer), then reinterpret as u64 lanes.
      V64 pkt = BitCast(d, LoadU(d8, packet + i * sizeof(uint64_t)));
#if !HWY_IS_LITTLE_ENDIAN
      // HighwayHash is defined on little-endian lane values.
      pkt = ReverseLaneBytes(pkt);
#endif
      Update(pkt, i);
    }
  }

  // Absorb the trailing 1..31 bytes with HighwayHash's frozen length padding.
  HWY_INLINE void UpdateRemainder(const uint8_t* HWY_RESTRICT bytes,
                                  const size_t size_mod32) {
    const D64 d;
    const D32 d32;
    const size_t N = Lanes(d);

    // Length padding: inject size_mod32 into every u64 lane as (s<<32)|s.
    const V64 mod32_pair =
        BitCast(d, Set(d32, static_cast<uint32_t>(size_mod32)));
    for (size_t i = 0; i < 4; i += N) {
      Store(Add(Load(d, v0 + i), mod32_pair), d, v0 + i);
      // Boost the avalanche of the length: rotate v1's u32 halves left by s.
      const V64 v1_i = Load(d, v1 + i);
      Store(BitCast(d, RotateLeftSame(BitCast(d32, v1_i),
                                      static_cast<int>(size_mod32))),
            d, v1 + i);
    }

    // Build the padded 32-byte packet exactly as the reference does.
    HWY_ALIGN uint8_t packet[kPacketSize];
    ZeroBytes(packet, kPacketSize);
    const size_t aligned = size_mod32 & ~size_t{3};
    CopyBytes(bytes, packet, aligned);
    const size_t mod4 = size_mod32 & 3;

    if (size_mod32 & 16) {
      // 16..31 bytes: place the last 4 bytes (all valid) at packet[28..31].
      CopyBytes(bytes + size_mod32 - 4, packet + kPacketSize - 4, size_t{4});
    } else if (mod4 != 0) {
      // <16 bytes: frozen "unordered" 3-byte pack at packet[16..18].
      const uint8_t* r = bytes + aligned;
      packet[16] = r[0];
      packet[17] = r[mod4 >> 1];
      packet[18] = r[mod4 - 1];
    }
    UpdatePacket(packet);
  }

  HWY_INLINE void PermuteAndUpdate() {
    // The reference permutes v0 by lanes {2,3,0,1} and rotates each u64 by 32.
    // For N == 4, ReverseBlocks swaps the two 128-bit halves; for N == 2,
    // ReverseBlocks is a no-op and i ^ (4 - N) == i ^ 2 swaps the two halves.
    const D64 d;
    const size_t N = Lanes(d);
    HWY_ALIGN uint64_t permuted[4];
    for (size_t i = 0; i < 4; i += N) {
      const V64 v = ReverseBlocks(d, Load(d, v0 + (i ^ (4 - N))));
      Store(Rotate64By32(v), d, permuted + i);
    }
    for (size_t i = 0; i < 4; i += N) {
      Update(Load(d, permuted + i), i);
    }
  }

  // ---- Finalizers -----------------------------------------------------------

  HWY_INLINE uint64_t Finalize64() {
    for (int r = 0; r < 4; ++r) PermuteAndUpdate();
    const D64 d;
    return GetLane(
        Add(Add(Load(d, v0), Load(d, mul0)), Add(Load(d, v1), Load(d, mul1))));
  }

  HWY_INLINE void Finalize128(uint64_t* HWY_RESTRICT out) {
    for (int r = 0; r < 6; ++r) PermuteAndUpdate();
    const Full128<uint64_t> d128;
    const VFromD<Full128<uint64_t>> hash =
        Add(Add(Load(d128, v0 + 0), Load(d128, mul0 + 0)),
            Add(Load(d128, v1 + 2), Load(d128, mul1 + 2)));
    StoreU(hash, d128, out);
  }

  // Modular reduction by the irreducible polynomial x^128 + x^2 + x, over each
  // 256-bit pair of 128-bit blocks (hi:lo) = (sum1 : sum0). See Lemire
  // 1503.03465.
  HWY_INLINE V64 ModularReduction(V64 sum1, V64 sum0) {
    const D64 d;
    const V64 sign_bit128 = Dup128VecFromValues(d, 0ull, 0x8000000000000000ull);

    const V64 top_bits2 = ShiftRight<62>(sum1);
    const V64 top_bits1 = ShiftRight<63>(sum1);
    const V64 shifted1_unmasked = Add(sum1, sum1);  // per-lane << 1
    const V64 shifted2 = Add(shifted1_unmasked, shifted1_unmasked);  // << 2
    const V64 shifted1 = AndNot(sign_bit128, shifted1_unmasked);  // clear b127
    const V64 new_low_bits2 = ShiftLeftBytes<8>(d, top_bits2);
    const V64 new_low_bits1 = ShiftLeftBytes<8>(d, top_bits1);

    V64 out = sum0;
    out = Xor(out, shifted2);
    out = Xor(out, new_low_bits2);
    out = Xor(out, shifted1);
    out = Xor(out, new_low_bits1);
    return out;
  }

  HWY_INLINE void Finalize256(uint64_t* HWY_RESTRICT out) {
    for (int r = 0; r < 10; ++r) PermuteAndUpdate();
    const D64 d;
    const size_t N = Lanes(d);
    for (size_t i = 0; i < 4; i += N) {
      const V64 sum0 = Add(Load(d, v0 + i), Load(d, mul0 + i));
      const V64 sum1 = Add(Load(d, v1 + i), Load(d, mul1 + i));
      StoreU(ModularReduction(sum1, sum0), d, out + i);
    }
  }
};

HWY_INLINE State ResetState(const uint64_t* HWY_RESTRICT key) {
  // "Nothing up my sleeve" constants (fractional digits of pi).
  HWY_ALIGN static constexpr uint64_t kInit0[4] = {
      0xdbe6d5d5fe4cce2full, 0xa4093822299f31d0ull, 0x13198a2e03707344ull,
      0x243f6a8885a308d3ull};
  HWY_ALIGN static constexpr uint64_t kInit1[4] = {
      0x3bd39e10cb0ef593ull, 0xc0acf169b5f18a8cull, 0xbe5466cf34e90c6cull,
      0x452821e638d01377ull};
  const D64 d;
  const size_t N = Lanes(d);
  State state;
  for (size_t i = 0; i < 4; i += N) {
    const V64 key_i = LoadU(d, key + i);
    const V64 init0 = Load(d, kInit0 + i);
    const V64 init1 = Load(d, kInit1 + i);
    Store(Xor(key_i, init0), d, state.v0 + i);
    Store(Xor(Rotate64By32(key_i), init1), d, state.v1 + i);
    Store(init0, d, state.mul0 + i);
    Store(init1, d, state.mul1 + i);
  }
  return state;
}

// Absorb full packets then the remainder.
HWY_INLINE void Absorb(State* HWY_RESTRICT state,
                       const uint8_t* HWY_RESTRICT bytes, size_t size) {
  const size_t remainder = size & (kPacketSize - 1);
  const size_t truncated = size - remainder;
  for (size_t i = 0; i < truncated; i += kPacketSize) {
    state->UpdatePacket(bytes + i);
  }
  if (remainder != 0) {
    state->UpdateRemainder(bytes + truncated, remainder);
  }
}

}  // namespace detail

// ---- One-shot public API ---------------------------------------------------

HWY_INLINE uint64_t HighwayHash64(const uint64_t key[4],
                                  const uint8_t* HWY_RESTRICT bytes,
                                  size_t size) {
  detail::State state = detail::ResetState(key);
  detail::Absorb(&state, bytes, size);
  return state.Finalize64();
}

HWY_INLINE void HighwayHash128(const uint64_t key[4],
                               const uint8_t* HWY_RESTRICT bytes, size_t size,
                               uint64_t out[2]) {
  detail::State state = detail::ResetState(key);
  detail::Absorb(&state, bytes, size);
  state.Finalize128(out);
}

HWY_INLINE void HighwayHash256(const uint64_t key[4],
                               const uint8_t* HWY_RESTRICT bytes, size_t size,
                               uint64_t out[4]) {
  detail::State state = detail::ResetState(key);
  detail::Absorb(&state, bytes, size);
  state.Finalize256(out);
}

#else  // HWY_SCALAR: no 16-byte block, not supported.

HWY_INLINE uint64_t HighwayHash64(const uint64_t[4], const uint8_t*, size_t) {
  HWY_DASSERT(0);
  return 0;
}
HWY_INLINE void HighwayHash128(const uint64_t[4], const uint8_t*, size_t,
                               uint64_t out[2]) {
  HWY_DASSERT(0);
  out[0] = out[1] = 0;
}
HWY_INLINE void HighwayHash256(const uint64_t[4], const uint8_t*, size_t,
                               uint64_t out[4]) {
  HWY_DASSERT(0);
  out[0] = out[1] = out[2] = out[3] = 0;
}

#endif  // HWY_TARGET != HWY_SCALAR

}  // namespace highwayhash
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // toggle guard
