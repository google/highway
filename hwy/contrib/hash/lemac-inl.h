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

// LeMac: a fast AES-based keyed hash with 128-bit output, here intended for
// detecting data corruption, including tree hashing of large inputs: hash each
// chunk, then the concatenation of the chunk hashes.
//
// Reference: A. Bariant, J. Baudrin, G. Leurent, C. Pernot, L. Perrin,
// T. Peyrin, "Fast AES-Based Universal Hash Functions and MACs: Featuring LeMac
// and PetitMac", ToSC 2024(2), and its corrigendum in ToSC 2025(1). This is
// the corrected LeMac, not LeMac-0. With kStripes = 1, the results match the
// authors' reference implementation and test vectors at
// https://github.com/AugustinBariant/Implementations_LeMac_PetitMac.
//
// Its universal hash updates nine 128-bit state words, each with one AES round
// per 64-byte step, plus four words of delayed input. The authors' MILP model
// shows that every differential trail activates at least 26 S-boxes, i.e. has
// probability at most 2^-156, hence they conjecture a collision probability of
// about 2^-128 for any two distinct inputs. The eight AES rounds of each step
// are independent, which is enough to saturate a 128-bit AES unit with latency
// 4 and a throughput of two per cycle.
//
// LeMac-x2 (kStripes = 2, the default) runs two independent instances of the
// universal hash, with distinct initial states: 16-byte block b of the input
// is absorbed by stripe b % 2. Thus a 256-bit vector holds the same state word
// of both stripes, and 256-bit VAES processes twice as many bytes per cycle.
// The finalization covers all 9 * kStripes state words: it XORs their AES
// encryptions with ten subkeys from a sliding window (a sum hash), and combines
// the result with a 128-bit nonce via EWCDM, i.e. the hash is
// AES_k3(sum ^ nonce ^ AES_k2(nonce)). As in the reference, the initial state,
// subkeys, k2 and k3 are AES-128 encryptions of consecutive counters under the
// key; the second byte of the counter block is kStripes - 1 so that keys for
// different kStripes are independent.
//
// For tree hashing, initialize a LeMacKey once and share it: it is immutable,
// occupies about 1.2 KiB and its construction costs about 50 AES-128
// encryptions. Pass a distinct nonce to each hash, e.g. the chunk index, total
// size and tree level, so that reordered or duplicated chunks are detected.
// This is much cheaper than a key per chunk. Use LeMacHash for contiguous
// inputs, and LeMacState with LeMacUpdate/LeMacFinalize for inputs that arrive
// in pieces. LeMacKey and LeMacState are plain data in lemac.h, whose layout
// and contents do not depend on the target, hence non-SIMD code can use them.
//
// For detecting accidental corruption, the key can be public, and the
// reduced-round LeMacFingerprint variant (LeMacFingerprintKey) is faster for
// small inputs: it uses the same universal hash and key derivation (10-round
// AES-128), but 4-round AES in the sum hash (4 vs 10 subkeys per state word,
// hence a smaller key: 752 vs 1216 bytes for x2). Any difference in a state
// word then still activates at least 25 S-boxes (wide trail theorem), and a
// collision of the sum requires differences in at least two words. Instead of
// EWCDM, the fingerprint is sum ^ AES4_k2(nonce): the final encryption would be
// a permutation and thus not affect collisions, and the nonce encryption alone,
// without the extra XOR of the nonce, ensures that distinct nonces never
// collide for equal inputs.
//
// WARNING: anyone who knows the key can construct distinct inputs with the same
// hash, for either variant: the additional rounds of LeMacOriginal only matter
// for secret keys, i.e. when used as a MAC. Hence with a public key, use either
// variant only where no adversary chooses the inputs, e.g. for detecting
// accidental corruption, or deduplication of trusted data. A random per-file
// key, which may be stored in the file header, prevents precomputed collisions
// and also protects against adversaries who supply inputs before the key is
// chosen and never learn it. For integrity or deduplication of untrusted data,
// keep the key secret and use LeMacOriginal, or use a cryptographic hash.
//
// LeMac is fast where Highway's AESRound is native, i.e. x86 starting with
// HWY_SSE4, Arm NEON (with the crypto extension), SVE2 with AES, and PPC8+.
// On other targets, AESRound is emulated and much slower. On Zen4, throughput
// for 256 KiB inputs is about 68 and 136 GB/s for x1 and x2 (HWY_AVX3_DL),
// and about 46 GB/s on Zen3 (HWY_AVX2) for both x1 and x2.

#include <stddef.h>
#include <stdint.h>

#include "hwy/contrib/hash/lemac.h"

// Include guard (still compiled once per target)
#if defined(HIGHWAY_HWY_CONTRIB_HASH_LEMAC_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_HASH_LEMAC_INL_H_
#undef HIGHWAY_HWY_CONTRIB_HASH_LEMAC_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_HASH_LEMAC_INL_H_
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// AESRound requires vectors of at least 128 bits. On HWY_EMU128, emulated AES
// would be slow (to compile and run), hence LeMac is also not available there.
#if HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128

// Default vectors for LeMacUpdate and LeMacFinalize: they hold the same state
// word of one or more stripes. On x86 without VAES (AVX2 and AVX3), Highway
// emulates 256-bit AESRound via 128-bit halves. Processing the stripes in
// separate passes of 128-bit vectors, each of which requires only 13 vector
// registers, is then twice as fast on Zen3. The result does not depend on the
// vectors; others are useful for testing.
template <size_t kStripes>
using LeMacDefaultD =
    CappedTag<uint8_t, (HWY_ARCH_X86 && HWY_TARGET > HWY_AVX3_DL)
                           ? LeMac::kBlockBytes
                           : LeMac::kBlockBytes * kStripes>;

namespace detail {

// LeMacKey and LeMacState only guarantee HWY_ALIGN_MAX, which may be less than
// a vector, hence all their loads and stores are unaligned.

using LeMacD16 = CappedTag<uint8_t, LeMac::kBlockBytes>;

// Returns the AES encryption of `x` with `kRoundKeys` expanded `round_keys`
// (11 for AES-128, or 5 for 4-round AES in LeMacFingerprint).
template <size_t kRoundKeys, class D, class V = VFromD<D>>
HWY_INLINE V LeMacEncrypt(
    D d, const uint8_t (*HWY_RESTRICT round_keys)[LeMac::kBlockBytes], V x) {
  x = Xor(x, LoadU(d, round_keys[0]));
  for (size_t r = 1; r < kRoundKeys - 1; ++r) {
    x = AESRound(x, LoadU(d, round_keys[r]));
  }
  return AESLastRound(x, LoadU(d, round_keys[kRoundKeys - 1]));
}

// Returns the AES-128 round key after `prev`.
template <uint8_t kRcon, class D, class V = VFromD<D>>
HWY_INLINE V LeMacNextRoundKey(D d, V prev) {
  const Repartition<uint32_t, D> du32;
  // Word 3 of AESKeyGenAssist is RotWord(SubWord(word 3 of prev)) ^ kRcon.
  const V assist =
      BitCast(d, Broadcast<3>(BitCast(du32, AESKeyGenAssist<kRcon>(prev))));
  // Prefix-XOR of the four words of `prev`.
  V words = Xor(prev, ShiftLeftBytes<4>(d, prev));
  words = Xor(words, ShiftLeftBytes<8>(d, words));
  return Xor(words, assist);
}

// AES-128 key schedule for `kRoundKeys` (5 or 11) round keys.
template <size_t kRoundKeys, class D, class V = VFromD<D>>
HWY_INLINE void LeMacExpandKey(
    D d, V key, uint8_t (*HWY_RESTRICT round_keys)[LeMac::kBlockBytes]) {
  static_assert(kRoundKeys == 5 || kRoundKeys == 11, "4 or 10 AES rounds");
  StoreU(key, d, round_keys[0]);
  key = LeMacNextRoundKey<0x01>(d, key);
  StoreU(key, d, round_keys[1]);
  key = LeMacNextRoundKey<0x02>(d, key);
  StoreU(key, d, round_keys[2]);
  key = LeMacNextRoundKey<0x04>(d, key);
  StoreU(key, d, round_keys[3]);
  key = LeMacNextRoundKey<0x08>(d, key);
  StoreU(key, d, round_keys[4]);
  HWY_IF_CONSTEXPR(kRoundKeys > 5) {
    key = LeMacNextRoundKey<0x10>(d, key);
    StoreU(key, d, round_keys[5]);
    key = LeMacNextRoundKey<0x20>(d, key);
    StoreU(key, d, round_keys[6]);
    key = LeMacNextRoundKey<0x40>(d, key);
    StoreU(key, d, round_keys[7]);
    key = LeMacNextRoundKey<0x80>(d, key);
    StoreU(key, d, round_keys[8]);
    key = LeMacNextRoundKey<0x1B>(d, key);
    StoreU(key, d, round_keys[9]);
    key = LeMacNextRoundKey<0x36>(d, key);
    StoreU(key, d, round_keys[10]);
  }
}

// Returns the AES-128 encryption of `{counter, (kStripes - 1) + tag, 0, ..}`,
// where tag is 16 for LeMacFingerprint, so that keys for different `kStripes`
// and `Variant` are independent. Always 10 rounds: fewer would be
// distinguishable from random for such structured plaintexts (e.g. integral
// or impossible-differential properties), and this is a one-time cost.
template <size_t kStripes, class Variant, class D>
HWY_INLINE VFromD<D> LeMacEncryptCounter(
    D d, const uint8_t (*HWY_RESTRICT round_keys)[LeMac::kBlockBytes],
    size_t counter) {
  HWY_DASSERT(counter < 256);
  constexpr uint8_t kTag = static_cast<uint8_t>(
      (kStripes - 1) + (IsSame<Variant, LeMacFingerprint>() ? 16 : 0));
  HWY_ALIGN uint8_t block[LeMac::kBlockBytes] = {static_cast<uint8_t>(counter),
                                                 kTag};
  return LeMacEncrypt<LeMac::kDeriveRoundKeys>(d, round_keys, Load(d, block));
}

// One step of the universal hash, for each stripe in the vectors. `m0` to
// `m3` are the four input blocks. Matches ROUND in the reference lemac.c.
template <class V>
HWY_INLINE void LeMacStep(const V m0, const V m1, const V m2, const V m3, V& s0,
                          V& s1, V& s2, V& s3, V& s4, V& s5, V& s6, V& s7,
                          V& s8, V& rr, V& r0, V& r1, V& r2) {
  const V prev_s8 = s8;
  s8 = AESRound(s7, m3);
  s7 = AESRound(s6, m1);
  s6 = AESRound(s5, m1);
  s5 = AESRound(s4, m0);
  s4 = AESRound(s3, m0);
  s3 = AESRound(s2, Xor(r1, r2));
  s2 = AESRound(s1, m3);
  s1 = AESRound(s0, m3);
  s0 = Xor3(s0, prev_s8, m2);
  r2 = r1;
  r1 = r0;
  r0 = Xor(rr, m1);
  rr = m2;
}

// Whether vectors of `D` might hold fewer than all stripes, in which case we
// make one pass per group of stripes. This is the case on x86 without VAES,
// Arm and RVV. For scalable vectors, which might also hold all stripes,
// multiple passes are still correct, and only slightly more expensive.
template <class D, size_t kStripes>
constexpr bool LeMacMultiPass() {
  return HWY_MAX_LANES_D(D) < LeMacState<kStripes>::kVecBytes ||
         HWY_MIN_BYTES < LeMacState<kStripes>::kVecBytes;
}

enum class LeMacPass { kUpdate, kFinalize };

// One pass of LeMacAbsorb for the vectors at byte offset `i` within each state
// word of `in`. Absorbs `num_steps` steps from `bytes`. If `kPass == kUpdate`,
// `num_steps` must be even, and stores the state to `out` (which may alias
// `in`). Otherwise, also absorbs the padded step `last` and kDrainSteps steps
// of zero input, and writes the sum hash of the state words with the subkeys
// of `key` to `sums`.
template <LeMacPass kPass, class D, size_t kStripes, class Key>
HWY_INLINE void LeMacPassAt(D d, size_t i, const LeMacState<kStripes>& in,
                            const uint8_t* HWY_RESTRICT bytes, size_t num_steps,
                            const uint8_t* HWY_RESTRICT last, const Key* key,
                            LeMacState<kStripes>* out,
                            uint8_t* HWY_RESTRICT sums) {
  using V = VFromD<D>;
  constexpr size_t kVecBytes = LeMacState<kStripes>::kVecBytes;
  constexpr size_t kStepBytes = LeMacState<kStripes>::kStepBytes;
  V s0 = LoadU(d, in.state[0] + i);
  V s1 = LoadU(d, in.state[1] + i);
  V s2 = LoadU(d, in.state[2] + i);
  V s3 = LoadU(d, in.state[3] + i);
  V s4 = LoadU(d, in.state[4] + i);
  V s5 = LoadU(d, in.state[5] + i);
  V s6 = LoadU(d, in.state[6] + i);
  V s7 = LoadU(d, in.state[7] + i);
  V s8 = LoadU(d, in.state[8] + i);
  V rr = LoadU(d, in.delay[0] + i);
  V r0 = LoadU(d, in.delay[1] + i);
  V r1 = LoadU(d, in.delay[2] + i);
  V r2 = LoadU(d, in.delay[3] + i);
  // Two steps per iteration avoid most register moves for rotating the
  // state, which is 1.36x as fast on Zen3.
  size_t step = 0;
  for (; step + 2 <= num_steps; step += 2) {
    const uint8_t* HWY_RESTRICT pos = bytes + step * kStepBytes + i;
    LeMacStep(LoadU(d, pos), LoadU(d, pos + kVecBytes),
              LoadU(d, pos + 2 * kVecBytes), LoadU(d, pos + 3 * kVecBytes), s0,
              s1, s2, s3, s4, s5, s6, s7, s8, rr, r0, r1, r2);
    pos += kStepBytes;
    LeMacStep(LoadU(d, pos), LoadU(d, pos + kVecBytes),
              LoadU(d, pos + 2 * kVecBytes), LoadU(d, pos + 3 * kVecBytes), s0,
              s1, s2, s3, s4, s5, s6, s7, s8, rr, r0, r1, r2);
  }
  HWY_IF_CONSTEXPR(kPass == LeMacPass::kUpdate) {
    StoreU(s0, d, out->state[0] + i);
    StoreU(s1, d, out->state[1] + i);
    StoreU(s2, d, out->state[2] + i);
    StoreU(s3, d, out->state[3] + i);
    StoreU(s4, d, out->state[4] + i);
    StoreU(s5, d, out->state[5] + i);
    StoreU(s6, d, out->state[6] + i);
    StoreU(s7, d, out->state[7] + i);
    StoreU(s8, d, out->state[8] + i);
    StoreU(rr, d, out->delay[0] + i);
    StoreU(r0, d, out->delay[1] + i);
    StoreU(r1, d, out->delay[2] + i);
    StoreU(r2, d, out->delay[3] + i);
  }
  HWY_IF_CONSTEXPR(kPass == LeMacPass::kFinalize) {
    if (step != num_steps) {  // Odd number of steps.
      const uint8_t* HWY_RESTRICT pos = bytes + step * kStepBytes + i;
      LeMacStep(LoadU(d, pos), LoadU(d, pos + kVecBytes),
                LoadU(d, pos + 2 * kVecBytes), LoadU(d, pos + 3 * kVecBytes),
                s0, s1, s2, s3, s4, s5, s6, s7, s8, rr, r0, r1, r2);
    }
    const uint8_t* HWY_RESTRICT pos = last + i;
    LeMacStep(Load(d, pos), Load(d, pos + kVecBytes),
              Load(d, pos + 2 * kVecBytes), Load(d, pos + 3 * kVecBytes), s0,
              s1, s2, s3, s4, s5, s6, s7, s8, rr, r0, r1, r2);
    const V zero = Zero(d);
    for (size_t k = 0; k < LeMac::kDrainSteps; ++k) {
      LeMacStep(zero, zero, zero, zero, s0, s1, s2, s3, s4, s5, s6, s7, s8, rr,
                r0, r1, r2);
    }
    // Sum hash while `s0`..`s8` are already in registers: XOR of the
    // encryptions of each state word with their subkeys and a final round
    // (including MixColumns) with a zero key.
    const uint8_t (*HWY_RESTRICT fin)[kVecBytes] = key->fin;
    s0 = Xor(s0, LoadU(d, fin[0] + i));
    s1 = Xor(s1, LoadU(d, fin[1] + i));
    s2 = Xor(s2, LoadU(d, fin[2] + i));
    s3 = Xor(s3, LoadU(d, fin[3] + i));
    s4 = Xor(s4, LoadU(d, fin[4] + i));
    s5 = Xor(s5, LoadU(d, fin[5] + i));
    s6 = Xor(s6, LoadU(d, fin[6] + i));
    s7 = Xor(s7, LoadU(d, fin[7] + i));
    s8 = Xor(s8, LoadU(d, fin[8] + i));
    for (size_t k = 1; k < Key::kFinKeys; ++k) {
      s0 = AESRound(s0, LoadU(d, fin[0 + k] + i));
      s1 = AESRound(s1, LoadU(d, fin[1 + k] + i));
      s2 = AESRound(s2, LoadU(d, fin[2 + k] + i));
      s3 = AESRound(s3, LoadU(d, fin[3 + k] + i));
      s4 = AESRound(s4, LoadU(d, fin[4 + k] + i));
      s5 = AESRound(s5, LoadU(d, fin[5 + k] + i));
      s6 = AESRound(s6, LoadU(d, fin[6 + k] + i));
      s7 = AESRound(s7, LoadU(d, fin[7 + k] + i));
      s8 = AESRound(s8, LoadU(d, fin[8 + k] + i));
    }
    s0 = AESRound(s0, zero);
    s1 = AESRound(s1, zero);
    s2 = AESRound(s2, zero);
    s3 = AESRound(s3, zero);
    s4 = AESRound(s4, zero);
    s5 = AESRound(s5, zero);
    s6 = AESRound(s6, zero);
    s7 = AESRound(s7, zero);
    s8 = AESRound(s8, zero);
    s0 = Xor3(s0, s1, s2);
    s3 = Xor3(s3, s4, s5);
    s6 = Xor3(s6, s7, s8);
    Store(Xor3(s0, s3, s6), d, sums + i);
  }
}

// Not inlined because the compiler would otherwise keep the state of all
// passes in registers, which causes spills in the hot loop on AVX2. Also used
// for a single pass, because inlining its large body (especially where AES is
// emulated) into every LeMacUpdate/LeMacFinalize call site greatly increases
// code size and compile time.
template <LeMacPass kPass, class D, size_t kStripes, class Key>
HWY_NOINLINE void LeMacCallPass(D d, size_t i, const LeMacState<kStripes>& in,
                                const uint8_t* HWY_RESTRICT bytes,
                                size_t num_steps,
                                const uint8_t* HWY_RESTRICT last,
                                const Key* key, LeMacState<kStripes>* out,
                                uint8_t* HWY_RESTRICT sums) {
  LeMacPassAt<kPass>(d, i, in, bytes, num_steps, last, key, out, sums);
}

// See LeMacPassAt. If vectors hold fewer than all stripes, makes one pass per
// group of stripes, which is possible because they are independent.
template <LeMacPass kPass, class D, size_t kStripes, class Key>
HWY_INLINE void LeMacAbsorb(D d, const LeMacState<kStripes>& in,
                            const uint8_t* HWY_RESTRICT bytes, size_t num_steps,
                            const uint8_t* HWY_RESTRICT last, const Key* key,
                            LeMacState<kStripes>* out,
                            uint8_t* HWY_RESTRICT sums) {
  constexpr size_t kVecBytes = LeMacState<kStripes>::kVecBytes;
  static_assert(IsSame<TFromD<D>, uint8_t>(), "D must be for uint8_t");
  static_assert(LeMac::kBlockBytes <= HWY_MAX_LANES_D(D) &&
                    HWY_MAX_LANES_D(D) <= kVecBytes,
                "Vectors must hold one or more, but not more than kStripes");
  const size_t N = Lanes(d);
  HWY_DASSERT(N >= LeMac::kBlockBytes);
  HWY_IF_CONSTEXPR(LeMacMultiPass<D, kStripes>()) {
    for (size_t i = 0; i < kVecBytes; i += N) {
      LeMacCallPass<kPass>(d, i, in, bytes, num_steps, last, key, out, sums);
    }
  }
  HWY_IF_CONSTEXPR(!LeMacMultiPass<D, kStripes>()) {
    HWY_DASSERT(N == kVecBytes);
    LeMacCallPass<kPass>(d, 0, in, bytes, num_steps, last, key, out, sums);
  }
}

// Absorbs `size` bytes, a nonzero multiple of kMinUpdateBytes, into `state`.
template <class D, size_t kStripes>
HWY_INLINE void LeMacUpdateBlocks(D d, LeMacState<kStripes>& state,
                                  const uint8_t* HWY_RESTRICT bytes,
                                  size_t size) {
  using State = LeMacState<kStripes>;
  // The update does not use the key.
  const LeMacKey<kStripes>* no_key = nullptr;
  HWY_DASSERT(size != 0 && size % State::kMinUpdateBytes == 0);
  HWY_IF_CONSTEXPR(LeMacMultiPass<D, kStripes>()) {
    // Each pass of LeMacAbsorb reads all cache lines of its input, hence we
    // limit the size of passes such that later passes hit in L1. Note that
    // passes are faster on x86 than holding all stripes in registers.
    for (size_t pos = 0; pos < size; pos += LeMac::kPassBytes) {
      const size_t pass = HWY_MIN(size - pos, LeMac::kPassBytes);
      LeMacAbsorb<LeMacPass::kUpdate>(d, state, bytes + pos,
                                      pass / State::kStepBytes, nullptr, no_key,
                                      &state, nullptr);
    }
  }
  HWY_IF_CONSTEXPR(!LeMacMultiPass<D, kStripes>()) {
    // All stripes are in registers: a single pass over all input.
    LeMacAbsorb<LeMacPass::kUpdate>(d, state, bytes, size / State::kStepBytes,
                                    nullptr, no_key, &state, nullptr);
  }
}

// Writes to `out` the hash of the input absorbed into the state and delay words
// of `state` (ignoring its buffered bytes, which the caller handles) followed
// by `bytes[0, size)`, with `nonce`.
template <class D, size_t kStripes, class Variant>
HWY_INLINE void LeMacFinalizeBlocks(D d, const LeMacKey<kStripes, Variant>& key,
                                    const LeMacState<kStripes>& state,
                                    const uint8_t* HWY_RESTRICT bytes,
                                    size_t size,
                                    const uint8_t nonce[LeMac::kNonceBytes],
                                    uint8_t out[LeMac::kHashBytes]) {
  using Key = LeMacKey<kStripes, Variant>;
  using State = LeMacState<kStripes>;
  constexpr size_t kStepBytes = State::kStepBytes;
  const size_t num_steps = size / kStepBytes;
  const size_t remainder = size - num_steps * kStepBytes;
  // 10* padding to a whole step, also if `size` is a multiple of steps.
  HWY_ALIGN uint8_t last[kStepBytes];
  ZeroBytes(last, kStepBytes);
  if (remainder != 0) {
    CopyBytes(bytes + num_steps * kStepBytes, last, remainder);
  }
  last[remainder] = 1;
  HWY_ALIGN uint8_t sums[State::kVecBytes];
  State* no_out = nullptr;
  LeMacAbsorb<LeMacPass::kFinalize>(d, state, bytes, num_steps, last, &key,
                                    no_out, sums);

  // XOR the sums of all stripes, then combine with the nonce.
  const LeMacD16 d16;
  using V16 = VFromD<LeMacD16>;
  V16 total = Load(d16, sums);
  for (size_t s = 1; s < kStripes; ++s) {
    total = Xor(total, Load(d16, sums + s * LeMac::kBlockBytes));
  }
  const V16 n = LoadU(d16, nonce);
  const V16 enc_n = LeMacEncrypt<Key::kRoundKeys>(d16, key.enc_keys, n);
  HWY_IF_CONSTEXPR(IsSame<Variant, LeMacOriginal>()) {
    // EWCDM.
    total = Xor3(total, n, enc_n);
    StoreU(LeMacEncrypt<Key::kRoundKeys>(d16, key.enc_keys + Key::kRoundKeys,
                                         total),
           d16, out);
  }
  HWY_IF_CONSTEXPR(IsSame<Variant, LeMacFingerprint>()) {
    StoreU(Xor(total, enc_n), d16, out);
  }
}

}  // namespace detail

// Initializes `key` from the 128-bit `raw_key`. The result is the same for
// every target.
template <size_t kStripes, class Variant>
HWY_INLINE void LeMacInitKey(const uint8_t raw_key[LeMac::kKeyBytes],
                             LeMacKey<kStripes, Variant>& key) {
  using Key = LeMacKey<kStripes, Variant>;
  constexpr size_t kWords = LeMac::kStateWords * kStripes;
  const detail::LeMacD16 d;
  HWY_ALIGN uint8_t round_keys[LeMac::kDeriveRoundKeys][LeMac::kBlockBytes];
  detail::LeMacExpandKey<LeMac::kDeriveRoundKeys>(d, LoadU(d, raw_key),
                                                  round_keys);
  // Counters [0, kWords) are for the initial state, then kWords + kFinKeys - 1
  // subkeys, then k2 and (for LeMacOriginal) k3. Interleaved by stripe, see
  // LeMacState.
  for (size_t s = 0; s < kStripes; ++s) {
    const size_t offset = s * LeMac::kBlockBytes;
    for (size_t j = 0; j < LeMac::kStateWords; ++j) {
      const size_t counter = s * LeMac::kStateWords + j;
      StoreU(detail::LeMacEncryptCounter<kStripes, Variant>(d, round_keys,
                                                            counter),
             d, key.init[j] + offset);
    }
    // State word j of stripe s uses subkeys [9 s + j, 9 s + j + kFinKeys).
    for (size_t t = 0; t < Key::kFinRows; ++t) {
      const size_t counter = kWords + s * LeMac::kStateWords + t;
      StoreU(detail::LeMacEncryptCounter<kStripes, Variant>(d, round_keys,
                                                            counter),
             d, key.fin[t] + offset);
    }
  }
  const size_t k2 = kWords + kWords + Key::kFinKeys - 1;
  for (size_t e = 0; e < Key::kEncryptions; ++e) {
    detail::LeMacExpandKey<Key::kRoundKeys>(
        d,
        detail::LeMacEncryptCounter<kStripes, Variant>(d, round_keys, k2 + e),
        key.enc_keys + e * Key::kRoundKeys);
  }
}

// Returns the key material for the 128-bit `raw_key`.
template <size_t kStripes = 2, class Variant = LeMacOriginal>
HWY_INLINE LeMacKey<kStripes, Variant> LeMacMakeKey(
    const uint8_t raw_key[LeMac::kKeyBytes]) {
  LeMacKey<kStripes, Variant> key;
  LeMacInitKey(raw_key, key);
  return key;
}

// Returns a LeMacFingerprintKey for a fixed public key, the first 16 bytes of
// the fractional part of Pi.
template <size_t kStripes = 2>
HWY_INLINE LeMacFingerprintKey<kStripes> LeMacMakeFingerprintKey() {
  static constexpr uint8_t kPi[LeMac::kKeyBytes] = {
      0x24, 0x3F, 0x6A, 0x88, 0x85, 0xA3, 0x08, 0xD3,
      0x13, 0x19, 0x8A, 0x2E, 0x03, 0x70, 0x73, 0x44};
  return LeMacMakeKey<kStripes, LeMacFingerprint>(kPi);
}

// Absorbs `size` bytes (any number) into `state`, which must have been Reset.
// Bytes beyond a multiple of LeMacState::kMinUpdateBytes are buffered in
// `state`; multiples of LeMac::kPassBytes avoid copying and amortize loading
// and storing the state. `d` describes the vectors with which to absorb.
template <class D, size_t kStripes>
HWY_INLINE void LeMacUpdate(D d, LeMacState<kStripes>& state,
                            const uint8_t* HWY_RESTRICT bytes, size_t size) {
  constexpr size_t kMin = LeMacState<kStripes>::kMinUpdateBytes;
  if (HWY_UNLIKELY(size == 0)) return;
  if (state.filled != 0) {
    const size_t take = HWY_MIN(size, kMin - state.filled);
    CopyBytes(bytes, state.buf + state.filled, take);
    state.filled += static_cast<uint32_t>(take);
    if (state.filled != kMin) return;
    detail::LeMacUpdateBlocks(d, state, state.buf, kMin);
    state.filled = 0;
    bytes += take;
    size -= take;
  }
  const size_t whole = size & ~(kMin - 1);
  if (whole != 0) detail::LeMacUpdateBlocks(d, state, bytes, whole);
  state.filled = static_cast<uint32_t>(size - whole);
  if (state.filled != 0) CopyBytes(bytes + whole, state.buf, state.filled);
}

template <size_t kStripes>
HWY_INLINE void LeMacUpdate(LeMacState<kStripes>& state,
                            const uint8_t* HWY_RESTRICT bytes, size_t size) {
  LeMacUpdate(LeMacDefaultD<kStripes>(), state, bytes, size);
}

// Writes to `out` the hash, with `key` (the same as passed to Reset) and
// `nonce`, of all bytes passed to LeMacUpdate since Reset followed by
// `bytes[0, size)` (any number including zero). Does not modify `state`, hence
// the same prefix can be finalized with several suffixes or nonces. The
// `nonce` is for domain separation, e.g. the position within a tree.
template <class D, size_t kStripes, class Variant>
HWY_INLINE void LeMacFinalize(D d, const LeMacKey<kStripes, Variant>& key,
                              const LeMacState<kStripes>& state,
                              const uint8_t* HWY_RESTRICT bytes, size_t size,
                              const uint8_t nonce[LeMac::kNonceBytes],
                              uint8_t out[LeMac::kHashBytes]) {
  using State = LeMacState<kStripes>;
  constexpr size_t kMin = State::kMinUpdateBytes;
  if (HWY_LIKELY(state.filled == 0)) {
    detail::LeMacFinalizeBlocks(d, key, state, bytes, size, nonce, out);
  } else if (state.filled + size < kMin) {
    // Concatenate the buffered bytes and `bytes`, which do not change `state`.
    HWY_ALIGN uint8_t buf[kMin];
    CopyBytes(state.buf, buf, state.filled);
    if (size != 0) CopyBytes(bytes, buf + state.filled, size);
    detail::LeMacFinalizeBlocks(d, key, state, buf, state.filled + size, nonce,
                                out);
  } else {
    // At least one unit: absorb it into a copy of the state.
    State copy = state;
    const size_t take = kMin - state.filled;
    LeMacUpdate(d, copy, bytes, take);
    detail::LeMacFinalizeBlocks(d, key, copy, bytes + take, size - take, nonce,
                                out);
  }
}

template <size_t kStripes, class Variant>
HWY_INLINE void LeMacFinalize(const LeMacKey<kStripes, Variant>& key,
                              const LeMacState<kStripes>& state,
                              const uint8_t* HWY_RESTRICT bytes, size_t size,
                              const uint8_t nonce[LeMac::kNonceBytes],
                              uint8_t out[LeMac::kHashBytes]) {
  LeMacFinalize(LeMacDefaultD<kStripes>(), key, state, bytes, size, nonce, out);
}

// Writes the LeMac or fingerprint (depending on `Variant`) of `size` bytes to
// `out`. The `nonce` is for domain separation, e.g. the position within a tree.
template <size_t kStripes, class Variant>
HWY_INLINE void LeMacHash(const LeMacKey<kStripes, Variant>& key,
                          const uint8_t nonce[LeMac::kNonceBytes],
                          const uint8_t* HWY_RESTRICT bytes, size_t size,
                          uint8_t out[LeMac::kHashBytes]) {
  const LeMacDefaultD<kStripes> d;
  LeMacState<kStripes> state;
  state.Reset(key);
  const size_t whole = size & ~(LeMac::kPassBytes - 1);
  if (whole != 0) detail::LeMacUpdateBlocks(d, state, bytes, whole);
  detail::LeMacFinalizeBlocks(d, key, state, bytes + whole, size - whole, nonce,
                              out);
}

#endif  // HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_HASH_LEMAC_INL_H_
