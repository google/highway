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
// The 256-bit internal state is four u64 lanes {v0, v1, mul0, mul1}, mixed in
// 128-bit halves (matching the reference SSE4.1 code), so this needs only
// Vec128 and works even on 128-bit-only targets. It does not yet use wider
// vectors when available; a 256-bit specialization is a possible follow-up.
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

// HighwayHash mixes its 256-bit state in 128-bit halves, so it needs a
// fixed-size 128-bit vector, which the 1-lane HWY_SCALAR target lacks.
#if HWY_TARGET != HWY_SCALAR

namespace detail {

using D64 = Full128<uint64_t>;
using D32 = Full128<uint32_t>;
using D8 = Full128<uint8_t>;
using V64 = Vec128<uint64_t>;

// "Nothing up my sleeve" constants (fractional digits of pi), as 128-bit
// halves: L = logical lanes {0,1}, H = logical lanes {2,3}.
HWY_INLINE V64 Init0L(D64 d) {
  return Dup128VecFromValues(d, 0xdbe6d5d5fe4cce2full, 0xa4093822299f31d0ull);
}
HWY_INLINE V64 Init0H(D64 d) {
  return Dup128VecFromValues(d, 0x13198a2e03707344ull, 0x243f6a8885a308d3ull);
}
HWY_INLINE V64 Init1L(D64 d) {
  return Dup128VecFromValues(d, 0x3bd39e10cb0ef593ull, 0xc0acf169b5f18a8cull);
}
HWY_INLINE V64 Init1H(D64 d) {
  return Dup128VecFromValues(d, 0xbe5466cf34e90c6cull, 0x452821e638d01377ull);
}

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
  const Vec128<uint8_t> idx = Dup128VecFromValues(d8, 3, 12, 2, 5, 14, 1, 15, 0,
                                                  11, 4, 10, 13, 9, 6, 8, 7);
  return BitCast(D64(), TableLookupBytes(BitCast(d8, v), idx));
}

// (u32)a[i] * (a2[i] >> 32) for each u64 lane, via the "even 32-bit lanes"
// multiply. BitCast(d32, a) even lanes are the low 32 bits; ShiftRight<32>
// moves the high 32 bits into that position.
HWY_INLINE V64 MulLoHi(V64 lo_src, V64 hi_src) {
  const D32 d32;
  return MulEven(BitCast(d32, lo_src), BitCast(d32, ShiftRight<32>(hi_src)));
}

// Aggregate (no user constructor) so the target-specific Vec128 default ctor is
// never called from an unattributed implicit constructor (breaks the GCC NEON
// build). Instances are created via ResetState() below.
struct State {
  V64 v0L, v0H, v1L, v1H, mul0L, mul0H, mul1L, mul1H;

  // Core round. packetL/packetH are logical lanes {0,1}/{2,3}.
  HWY_INLINE void Update(V64 packetL, V64 packetH) {
    v1L = Add(Add(v1L, packetL), mul0L);
    v1H = Add(Add(v1H, packetH), mul0H);

    mul0L = Xor(mul0L, MulLoHi(v1L, v0L));
    mul0H = Xor(mul0H, MulLoHi(v1H, v0H));

    v0L = Add(v0L, mul1L);
    v0H = Add(v0H, mul1H);

    mul1L = Xor(mul1L, MulLoHi(v0L, v1L));
    mul1H = Xor(mul1H, MulLoHi(v0H, v1H));

    v0L = Add(v0L, ZipperMerge(v1L));
    v0H = Add(v0H, ZipperMerge(v1H));
    v1L = Add(v1L, ZipperMerge(v0L));
    v1H = Add(v1H, ZipperMerge(v0H));
  }

  HWY_INLINE void UpdatePacket(const uint8_t* HWY_RESTRICT packet) {
    const D64 d;
    const D8 d8;
    // Load as bytes (no aliasing/alignment assumptions on the caller's buffer),
    // then reinterpret as u64 lanes.
    V64 lo = BitCast(d, LoadU(d8, packet + 0));
    V64 hi = BitCast(d, LoadU(d8, packet + 16));
#if !HWY_IS_LITTLE_ENDIAN
    // HighwayHash is defined on little-endian lane values.
    lo = ReverseLaneBytes(lo);
    hi = ReverseLaneBytes(hi);
#endif
    Update(lo, hi);
  }

  // Absorb the trailing 1..31 bytes with HighwayHash's frozen length padding.
  HWY_INLINE void UpdateRemainder(const uint8_t* HWY_RESTRICT bytes,
                                  const size_t size_mod32) {
    const D64 d;
    const D32 d32;

    // Length padding: inject size_mod32 into every u64 lane as (s<<32)|s.
    const V64 mod32_pair =
        BitCast(d, Set(d32, static_cast<uint32_t>(size_mod32)));
    v0L = Add(v0L, mod32_pair);
    v0H = Add(v0H, mod32_pair);
    // Boost the avalanche of the length: rotate v1's u32 halves left by s.
    v1L = BitCast(
        d, RotateLeftSame(BitCast(d32, v1L), static_cast<int>(size_mod32)));
    v1H = BitCast(
        d, RotateLeftSame(BitCast(d32, v1H), static_cast<int>(size_mod32)));

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
    // The reference permutes v0 by lanes {2,3,0,1} and rotates each u64 by 32;
    // with explicit L/H halves that is: swap L<->H and Rotate64By32 each.
    Update(Rotate64By32(v0H), Rotate64By32(v0L));
  }

  // ---- Finalizers -----------------------------------------------------------

  HWY_INLINE uint64_t Finalize64() {
    for (int i = 0; i < 4; ++i) PermuteAndUpdate();
    return GetLane(Add(Add(v0L, mul0L), Add(v1L, mul1L)));
  }

  HWY_INLINE void Finalize128(uint64_t* HWY_RESTRICT out) {
    for (int i = 0; i < 6; ++i) PermuteAndUpdate();
    const V64 hash = Add(Add(v0L, mul0L), Add(v1H, mul1H));
    StoreU(hash, D64(), out);
  }

  // Modular reduction by the irreducible polynomial x^128 + x^2 + x, over the
  // 256-bit value (hi:lo) = (sum1 : sum0). See Lemire 1503.03465.
  HWY_INLINE V64 ModularReduction(V64 sum1, V64 sum0) {
    const D64 d;
    const D32 d32;
    // 0x80000000 in u32 lane 3 == bit 127.
    const V64 sign_bit128 =
        BitCast(d, Dup128VecFromValues(d32, 0u, 0u, 0u, 0x80000000u));

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
    for (int i = 0; i < 10; ++i) PermuteAndUpdate();
    const V64 hashL = ModularReduction(Add(v1L, mul1L), Add(v0L, mul0L));
    const V64 hashH = ModularReduction(Add(v1H, mul1H), Add(v0H, mul0H));
    StoreU(hashL, D64(), out + 0);
    StoreU(hashH, D64(), out + 2);
  }
};

HWY_INLINE State ResetState(const uint64_t* HWY_RESTRICT key) {
  const D64 d;
  const V64 keyL = LoadU(d, key + 0);
  const V64 keyH = LoadU(d, key + 2);
  return State{Xor(keyL, Init0L(d)),
               Xor(keyH, Init0H(d)),
               Xor(Rotate64By32(keyL), Init1L(d)),
               Xor(Rotate64By32(keyH), Init1H(d)),
               Init0L(d),
               Init0H(d),
               Init1L(d),
               Init1H(d)};
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

#else  // HWY_TARGET == HWY_SCALAR: no 128-bit vector, not supported.

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
