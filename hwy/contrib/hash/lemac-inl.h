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
// For tree hashing, construct a LeMacKey once and share it: it is immutable,
// occupies about 1.2 KiB and its construction costs about 50 AES-128
// encryptions. Pass a distinct nonce to each hash, e.g. the chunk index, total
// size and tree level, so that reordered or duplicated chunks are detected.
// This is much cheaper than a key per chunk. Use LeMacHash for contiguous
// inputs, and LeMacHasher for inputs that arrive in multiples of 8 KiB.
//
// For detecting accidental corruption, the key can be public, and the
// reduced-round LeMacFingerprint variant (LeMacFingerprintKey and
// LeMacFingerprintHasher) is faster for small inputs: it uses the same
// universal hash and key derivation (10-round AES-128), but 4-round AES in the
// sum hash (4 vs 10 subkeys per state word, hence a smaller key: 752 vs 1216
// bytes for x2). Any difference in a state word then still activates at least
// 25 S-boxes (wide trail theorem), and a collision of the sum requires
// differences in at least two words. Instead of EWCDM, the fingerprint is
// sum ^ AES4_k2(nonce): the final encryption would be a permutation and thus
// not affect collisions, and the nonce encryption alone, without the extra XOR
// of the nonce, ensures that distinct nonces never collide for equal inputs.
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

// Include guard (still compiled once per target)
#if defined(HIGHWAY_HWY_CONTRIB_HASH_LEMAC_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_HASH_LEMAC_INL_H_
#undef HIGHWAY_HWY_CONTRIB_HASH_LEMAC_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_HASH_LEMAC_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// AESRound requires vectors of at least 128 bits.
#if HWY_TARGET != HWY_SCALAR

// Tag classes for the `Variant` template parameter of LeMacKey and LeMacHasher:
// full LeMac specification, or reduced finalization for non-cryptographic
// corruption detection.
struct LeMacOriginal {};
struct LeMacFingerprint {};

template <size_t kStripes, class Variant>
class LeMacKey;
template <size_t kStripes, class D, class Variant>
class LeMacHasher;

// Sizes of the inputs and output, plus implementation details shared by
// LeMacKey and LeMacHasher.
class LeMac {
 public:
  static constexpr size_t kKeyBytes = 16;
  static constexpr size_t kNonceBytes = 16;
  static constexpr size_t kHashBytes = 16;

  // One AES block, which is also the size of a state word and an input block.
  static constexpr size_t kBlockBytes = 16;

  // Default vectors for LeMacHasher: they hold the same state word of one or
  // more stripes. On x86 without VAES (AVX2 and AVX3), Highway emulates 256-bit
  // AESRound via 128-bit halves. Processing the stripes in separate passes of
  // 128-bit vectors, each of which requires only 13 vector registers, is then
  // twice as fast on Zen3.
  template <size_t kStripes>
  using DefaultD = CappedTag<uint8_t, (HWY_ARCH_X86 && HWY_TARGET > HWY_AVX3_DL)
                                          ? kBlockBytes
                                          : kBlockBytes * kStripes>;

 private:
  template <size_t, class>
  friend class LeMacKey;
  template <size_t, class, class>
  friend class LeMacHasher;

  // Per stripe: state words updated with AES rounds, and words of delayed
  // input (RR, R0, R1, R2 in the reference).
  static constexpr size_t kStateWords = 9;
  static constexpr size_t kDelayWords = 4;
  // Input blocks absorbed per stripe and step.
  static constexpr size_t kBlocksPerStep = 4;
  // Steps of zero input after the padding, which are required for the last
  // input blocks to pass through the delay words and enough S-boxes.
  static constexpr size_t kDrainSteps = 4;
  // Upper bound on the input bytes per pass, see LeMacHasher::Absorb.
  // Larger passes amortize loading/storing the state; 8 KiB is about 4% faster
  // than 2 KiB on Zen3, and still fits in L1 so that later passes hit.
  static constexpr size_t kPassBytes = 8192;
  // AES-128 round keys for deriving the key material from the key.
  static constexpr size_t kDeriveRoundKeys = 11;

  using D16 = CappedTag<uint8_t, kBlockBytes>;

  // Returns the AES encryption of `x` with `kRoundKeys` expanded `round_keys`
  // (11 for AES-128, or 5 for 4-round AES in LeMacFingerprint).
  template <size_t kRoundKeys, class D, class V = VFromD<D>>
  static HWY_INLINE V
  Encrypt(D d, const uint8_t (*HWY_RESTRICT round_keys)[kBlockBytes], V x) {
    x = Xor(x, Load(d, round_keys[0]));
    for (size_t r = 1; r < kRoundKeys - 1; ++r) {
      x = AESRound(x, Load(d, round_keys[r]));
    }
    return AESLastRound(x, Load(d, round_keys[kRoundKeys - 1]));
  }

  // Returns the AES-128 round key after `prev`.
  template <uint8_t kRcon, class D, class V = VFromD<D>>
  static HWY_INLINE V NextRoundKey(D d, V prev) {
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
  static HWY_INLINE void ExpandKey(
      D d, V key, uint8_t (*HWY_RESTRICT round_keys)[kBlockBytes]) {
    static_assert(kRoundKeys == 5 || kRoundKeys == 11, "4 or 10 AES rounds");
    Store(key, d, round_keys[0]);
    key = NextRoundKey<0x01>(d, key);
    Store(key, d, round_keys[1]);
    key = NextRoundKey<0x02>(d, key);
    Store(key, d, round_keys[2]);
    key = NextRoundKey<0x04>(d, key);
    Store(key, d, round_keys[3]);
    key = NextRoundKey<0x08>(d, key);
    Store(key, d, round_keys[4]);
    HWY_IF_CONSTEXPR(kRoundKeys > 5) {
      key = NextRoundKey<0x10>(d, key);
      Store(key, d, round_keys[5]);
      key = NextRoundKey<0x20>(d, key);
      Store(key, d, round_keys[6]);
      key = NextRoundKey<0x40>(d, key);
      Store(key, d, round_keys[7]);
      key = NextRoundKey<0x80>(d, key);
      Store(key, d, round_keys[8]);
      key = NextRoundKey<0x1B>(d, key);
      Store(key, d, round_keys[9]);
      key = NextRoundKey<0x36>(d, key);
      Store(key, d, round_keys[10]);
    }
  }

  // Returns the AES-128 encryption of `{counter, (kStripes - 1) + tag, 0, ..}`,
  // where tag is 16 for LeMacFingerprint, so that keys for different `kStripes`
  // and `Variant` are independent. Always 10 rounds: fewer would be
  // distinguishable from random for such structured plaintexts (e.g. integral
  // or impossible-differential properties), and this is a one-time cost.
  template <size_t kStripes, class Variant, class D>
  static HWY_INLINE VFromD<D> EncryptCounter(
      D d, const uint8_t (*HWY_RESTRICT round_keys)[kBlockBytes],
      size_t counter) {
    HWY_DASSERT(counter < 256);
    constexpr uint8_t kTag = static_cast<uint8_t>(
        (kStripes - 1) + (IsSame<Variant, LeMacFingerprint>() ? 16 : 0));
    HWY_ALIGN uint8_t block[kBlockBytes] = {static_cast<uint8_t>(counter),
                                            kTag};
    return Encrypt<kDeriveRoundKeys>(d, round_keys, Load(d, block));
  }

  // One step of the universal hash, for each stripe in the vectors. `m0` to
  // `m3` are the four input blocks. Matches ROUND in the reference lemac.c.
  template <class V>
  static HWY_INLINE void Step(const V m0, const V m1, const V m2, const V m3,
                              V& s0, V& s1, V& s2, V& s3, V& s4, V& s5, V& s6,
                              V& s7, V& s8, V& rr, V& r0, V& r1, V& r2) {
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
};

// Key material derived from a 128-bit key. Immutable and thread-safe; for
// efficiency, construct once and share. `Variant = LeMacOriginal` is the full
// LeMac specification; `Variant = LeMacFingerprint` uses fewer finalization
// rounds for non-cryptographic corruption detection.
template <size_t kStripes = 2, class Variant = LeMacOriginal>
class HWY_ALIGN LeMacKey {
  static_assert(kStripes == 1 || kStripes == 2, "Only LeMac-x1 or -x2");
  static_assert(IsSame<Variant, LeMacOriginal>() ||
                    IsSame<Variant, LeMacFingerprint>(),
                "Variant must be LeMacOriginal or LeMacFingerprint");
  static constexpr size_t kVecBytes = LeMac::kBlockBytes * kStripes;
  static constexpr size_t kWords = LeMac::kStateWords * kStripes;

 public:
  // Default constructor for LeMacFingerprintKey using the first 16 bytes of the
  // fractional part of Pi.
  template <class V = Variant,
            hwy::EnableIf<IsSame<V, LeMacFingerprint>()>* = nullptr>
  LeMacKey() {
    static constexpr uint8_t kPi[LeMac::kKeyBytes] = {
        0x24, 0x3F, 0x6A, 0x88, 0x85, 0xA3, 0x08, 0xD3,
        0x13, 0x19, 0x8A, 0x2E, 0x03, 0x70, 0x73, 0x44};
    Init(kPi);
  }

  explicit LeMacKey(const uint8_t key[LeMac::kKeyBytes]) { Init(key); }

 private:
  template <size_t, class, class>
  friend class LeMacHasher;

  // The finalization encrypts each state word with this many subkeys, plus a
  // final round with zero key (10 rounds for LeMacOriginal, 4 for
  // LeMacFingerprint).
  static constexpr size_t kFinKeys = IsSame<Variant, LeMacOriginal>() ? 10 : 4;
  // Round keys for the nonce encryption (10 or 4 AES rounds) and, only for
  // LeMacOriginal, the final encryption.
  static constexpr size_t kRoundKeys =
      IsSame<Variant, LeMacOriginal>() ? 11 : 5;
  static constexpr size_t kEncryptions =
      IsSame<Variant, LeMacOriginal>() ? 2 : 1;
  static constexpr size_t kFinRows = LeMac::kStateWords + kFinKeys - 1;

  void Init(const uint8_t key[LeMac::kKeyBytes]) {
    const LeMac::D16 d;
    HWY_ALIGN uint8_t round_keys[LeMac::kDeriveRoundKeys][LeMac::kBlockBytes];
    LeMac::ExpandKey<LeMac::kDeriveRoundKeys>(d, LoadU(d, key), round_keys);
    // Counters [0, kWords) are for the initial state, then kWords + kFinKeys
    // - 1 subkeys, then k2 and (for LeMacOriginal) k3. Interleaved by stripe,
    // see LeMacHasher.
    for (size_t s = 0; s < kStripes; ++s) {
      const size_t offset = s * LeMac::kBlockBytes;
      for (size_t j = 0; j < LeMac::kStateWords; ++j) {
        const size_t counter = s * LeMac::kStateWords + j;
        Store(LeMac::EncryptCounter<kStripes, Variant>(d, round_keys, counter),
              d, init_[j] + offset);
      }
      // State word j of stripe s uses subkeys [9 s + j, 9 s + j + kFinKeys).
      for (size_t t = 0; t < kFinRows; ++t) {
        const size_t counter = kWords + s * LeMac::kStateWords + t;
        Store(LeMac::EncryptCounter<kStripes, Variant>(d, round_keys, counter),
              d, fin_[t] + offset);
      }
    }
    const size_t k2 = kWords + kWords + kFinKeys - 1;
    for (size_t e = 0; e < kEncryptions; ++e) {
      LeMac::ExpandKey<kRoundKeys>(
          d, LeMac::EncryptCounter<kStripes, Variant>(d, round_keys, k2 + e),
          enc_keys_ + e * kRoundKeys);
    }
  }

  // Initial value of state word j for each stripe.
  HWY_ALIGN uint8_t init_[LeMac::kStateWords][kVecBytes];
  // Subkey t of each stripe: state word j uses rows [j, j + kFinKeys).
  HWY_ALIGN uint8_t fin_[kFinRows][kVecBytes];
  // Round keys for encrypting the nonce (k2), followed by those of the final
  // encryption (k3) for LeMacOriginal.
  HWY_ALIGN uint8_t enc_keys_[kEncryptions * kRoundKeys][LeMac::kBlockBytes];
};

template <size_t kStripes = 2>
using LeMacFingerprintKey = LeMacKey<kStripes, LeMacFingerprint>;

// Hashes inputs that arrive in several pieces: call Update any number of times
// with whole multiples of kUpdateBytes, then Finalize with the remaining bytes.
// Call Reset before reusing the LeMacHasher. Each thread requires its own
// LeMacHasher. `D` describes the vectors with which to absorb the input. The
// result does not depend on it; the default is the most efficient for the
// current target, and others are useful for testing.
template <size_t kStripes = 2, class D = LeMac::DefaultD<kStripes>,
          class Variant = LeMacOriginal>
class HWY_ALIGN LeMacHasher {
  using Key = LeMacKey<kStripes, Variant>;
  static constexpr size_t kVecBytes = LeMac::kBlockBytes * kStripes;
  // Each step absorbs four input blocks per stripe.
  static constexpr size_t kStepBytes = LeMac::kBlocksPerStep * kVecBytes;
  using V = VFromD<D>;
  static_assert(IsSame<TFromD<D>, uint8_t>(), "D must be for uint8_t");
  static_assert(LeMac::kBlockBytes <= HWY_MAX_LANES_D(D) &&
                    HWY_MAX_LANES_D(D) <= kVecBytes,
                "Vectors must hold one or more, but not more than kStripes");
  // Whether vectors might hold fewer than all stripes, in which case we make
  // one pass per group of stripes. This is the case on x86 without VAES, Arm
  // and RVV. For scalable vectors, which might also hold all stripes, multiple
  // passes are still correct, and only slightly more expensive.
  static constexpr bool kMultiPass =
      HWY_MAX_LANES_D(D) < kVecBytes || HWY_MIN_BYTES < kVecBytes;
  enum PassMode { kUpdate, kFinalize };

 public:
  // The size passed to Update must be a multiple of this. Chunks of large
  // inputs are typically powers of two, and thus also multiples. This is also
  // the size of a pass if vectors hold fewer than all stripes, see Absorb.
  static constexpr size_t kUpdateBytes = LeMac::kPassBytes;
  static_assert(kUpdateBytes % (2 * kStepBytes) == 0, "Must be even steps");
  static_assert((kUpdateBytes & (kUpdateBytes - 1)) == 0, "Power of two");

  // `key` must outlive this LeMacHasher.
  explicit LeMacHasher(const Key& key) : key_(&key) { Reset(); }

  void Reset() {
    CopyBytes(key_->init_, state_, sizeof(state_));
    ZeroBytes(delay_, sizeof(delay_));
  }

  // Absorbs `size` bytes, which must be a nonzero multiple of kUpdateBytes.
  void Update(const uint8_t* HWY_RESTRICT bytes, size_t size) {
    HWY_DASSERT(size != 0 && size % kUpdateBytes == 0);
    HWY_IF_CONSTEXPR(kMultiPass) {
      // Each pass of Absorb reads all cache lines of its input, hence we limit
      // the size of passes such that later passes hit in L1. Note that passes
      // are faster on x86 than holding all stripes in registers.
      for (size_t pos = 0; pos < size; pos += kUpdateBytes) {
        Absorb<kUpdate>(bytes + pos, kUpdateBytes / kStepBytes, nullptr,
                        nullptr);
      }
    }
    HWY_IF_CONSTEXPR(!kMultiPass) {
      // All stripes are in registers: a single pass over all input.
      Absorb<kUpdate>(bytes, size / kStepBytes, nullptr, nullptr);
    }
  }

  // Absorbs the final `size` bytes (any number including zero, but typically
  // fewer than kUpdateBytes), then writes the hash of all bytes passed to
  // Update and Finalize since Reset to `out`. The `nonce` is for domain
  // separation, e.g. the position within a tree.
  void Finalize(const uint8_t* HWY_RESTRICT bytes, size_t size,
                const uint8_t nonce[LeMac::kNonceBytes],
                uint8_t out[LeMac::kHashBytes]) {
    const size_t num_steps = size / kStepBytes;
    const size_t remainder = size - num_steps * kStepBytes;
    // 10* padding to a whole step, also if `size` is a multiple of steps.
    HWY_ALIGN uint8_t last[kStepBytes];
    ZeroBytes(last, kStepBytes);
    if (remainder != 0) {
      CopyBytes(bytes + num_steps * kStepBytes, last, remainder);
    }
    last[remainder] = 1;
    HWY_ALIGN uint8_t sums[kVecBytes];
    Absorb<kFinalize>(bytes, num_steps, last, sums);

    // XOR the sums of all stripes, then combine with the nonce.
    const LeMac::D16 d16;
    using V16 = VFromD<LeMac::D16>;
    V16 total = Load(d16, sums);
    for (size_t s = 1; s < kStripes; ++s) {
      total = Xor(total, Load(d16, sums + s * LeMac::kBlockBytes));
    }
    const V16 n = LoadU(d16, nonce);
    const V16 enc_n = LeMac::Encrypt<Key::kRoundKeys>(d16, key_->enc_keys_, n);
    HWY_IF_CONSTEXPR(IsSame<Variant, LeMacOriginal>()) {
      // EWCDM.
      total = Xor3(total, n, enc_n);
      StoreU(LeMac::Encrypt<Key::kRoundKeys>(
                 d16, key_->enc_keys_ + Key::kRoundKeys, total),
             d16, out);
    }
    HWY_IF_CONSTEXPR(IsSame<Variant, LeMacFingerprint>()) {
      StoreU(Xor(total, enc_n), d16, out);
    }
  }

 private:
  // Absorbs `num_steps` steps from `bytes`. If `kPass == kFinalize`, then also
  // absorbs the padded step `last` and kDrainSteps steps of zero input, and
  // writes the sum hash of the state words to `sums`; otherwise, `num_steps`
  // must be even. If vectors hold fewer than all stripes, makes one pass per
  // group of stripes, which is possible because they are independent.
  template <PassMode kPass>
  HWY_INLINE void Absorb(const uint8_t* HWY_RESTRICT bytes, size_t num_steps,
                         const uint8_t* HWY_RESTRICT last,
                         uint8_t* HWY_RESTRICT sums) {
    const size_t N = Lanes(D());
    HWY_DASSERT(N >= LeMac::kBlockBytes);
    HWY_IF_CONSTEXPR(kMultiPass) {
      for (size_t i = 0; i < kVecBytes; i += N) {
        OutlinedPass<kPass>(i, bytes, num_steps, last, sums);
      }
    }
    HWY_IF_CONSTEXPR(!kMultiPass) {
      HWY_DASSERT(N == kVecBytes);
      Pass<kPass>(0, bytes, num_steps, last, sums);
    }
  }

  // Not inlined because the compiler would otherwise keep the state of all
  // passes in registers, which causes spills in the hot loop on AVX2.
  template <PassMode kPass>
  HWY_NOINLINE void OutlinedPass(size_t i, const uint8_t* HWY_RESTRICT bytes,
                                 size_t num_steps,
                                 const uint8_t* HWY_RESTRICT last,
                                 uint8_t* HWY_RESTRICT sums) {
    Pass<kPass>(i, bytes, num_steps, last, sums);
  }

  // One pass of Absorb for the vectors at byte offset `i` within each state
  // word.
  template <PassMode kPass>
  HWY_INLINE void Pass(size_t i, const uint8_t* HWY_RESTRICT bytes,
                       size_t num_steps, const uint8_t* HWY_RESTRICT last,
                       uint8_t* HWY_RESTRICT sums) {
    const D d;
    V s0 = Load(d, state_[0] + i);
    V s1 = Load(d, state_[1] + i);
    V s2 = Load(d, state_[2] + i);
    V s3 = Load(d, state_[3] + i);
    V s4 = Load(d, state_[4] + i);
    V s5 = Load(d, state_[5] + i);
    V s6 = Load(d, state_[6] + i);
    V s7 = Load(d, state_[7] + i);
    V s8 = Load(d, state_[8] + i);
    V rr = Load(d, delay_[0] + i);
    V r0 = Load(d, delay_[1] + i);
    V r1 = Load(d, delay_[2] + i);
    V r2 = Load(d, delay_[3] + i);
    // Two steps per iteration avoid most register moves for rotating the
    // state, which is 1.36x as fast on Zen3.
    size_t step = 0;
    for (; step + 2 <= num_steps; step += 2) {
      const uint8_t* HWY_RESTRICT in = bytes + step * kStepBytes + i;
      LeMac::Step(LoadU(d, in), LoadU(d, in + kVecBytes),
                  LoadU(d, in + 2 * kVecBytes), LoadU(d, in + 3 * kVecBytes),
                  s0, s1, s2, s3, s4, s5, s6, s7, s8, rr, r0, r1, r2);
      in += kStepBytes;
      LeMac::Step(LoadU(d, in), LoadU(d, in + kVecBytes),
                  LoadU(d, in + 2 * kVecBytes), LoadU(d, in + 3 * kVecBytes),
                  s0, s1, s2, s3, s4, s5, s6, s7, s8, rr, r0, r1, r2);
    }
    HWY_IF_CONSTEXPR(kPass == kUpdate) {
      Store(s0, d, state_[0] + i);
      Store(s1, d, state_[1] + i);
      Store(s2, d, state_[2] + i);
      Store(s3, d, state_[3] + i);
      Store(s4, d, state_[4] + i);
      Store(s5, d, state_[5] + i);
      Store(s6, d, state_[6] + i);
      Store(s7, d, state_[7] + i);
      Store(s8, d, state_[8] + i);
      Store(rr, d, delay_[0] + i);
      Store(r0, d, delay_[1] + i);
      Store(r1, d, delay_[2] + i);
      Store(r2, d, delay_[3] + i);
    }
    HWY_IF_CONSTEXPR(kPass == kFinalize) {
      if (step != num_steps) {  // Odd number of steps.
        const uint8_t* HWY_RESTRICT in = bytes + step * kStepBytes + i;
        LeMac::Step(LoadU(d, in), LoadU(d, in + kVecBytes),
                    LoadU(d, in + 2 * kVecBytes), LoadU(d, in + 3 * kVecBytes),
                    s0, s1, s2, s3, s4, s5, s6, s7, s8, rr, r0, r1, r2);
      }
      const uint8_t* HWY_RESTRICT in = last + i;
      LeMac::Step(Load(d, in), Load(d, in + kVecBytes),
                  Load(d, in + 2 * kVecBytes), Load(d, in + 3 * kVecBytes), s0,
                  s1, s2, s3, s4, s5, s6, s7, s8, rr, r0, r1, r2);
      const V zero = Zero(d);
      for (size_t k = 0; k < LeMac::kDrainSteps; ++k) {
        LeMac::Step(zero, zero, zero, zero, s0, s1, s2, s3, s4, s5, s6, s7, s8,
                    rr, r0, r1, r2);
      }
      // Sum hash while `s0`..`s8` are already in registers: XOR of the
      // encryptions of each state word with their subkeys and a final round
      // (including MixColumns) with a zero key.
      const auto fin = key_->fin_;
      s0 = Xor(s0, Load(d, fin[0] + i));
      s1 = Xor(s1, Load(d, fin[1] + i));
      s2 = Xor(s2, Load(d, fin[2] + i));
      s3 = Xor(s3, Load(d, fin[3] + i));
      s4 = Xor(s4, Load(d, fin[4] + i));
      s5 = Xor(s5, Load(d, fin[5] + i));
      s6 = Xor(s6, Load(d, fin[6] + i));
      s7 = Xor(s7, Load(d, fin[7] + i));
      s8 = Xor(s8, Load(d, fin[8] + i));
      for (size_t k = 1; k < Key::kFinKeys; ++k) {
        s0 = AESRound(s0, Load(d, fin[0 + k] + i));
        s1 = AESRound(s1, Load(d, fin[1 + k] + i));
        s2 = AESRound(s2, Load(d, fin[2 + k] + i));
        s3 = AESRound(s3, Load(d, fin[3 + k] + i));
        s4 = AESRound(s4, Load(d, fin[4 + k] + i));
        s5 = AESRound(s5, Load(d, fin[5 + k] + i));
        s6 = AESRound(s6, Load(d, fin[6 + k] + i));
        s7 = AESRound(s7, Load(d, fin[7 + k] + i));
        s8 = AESRound(s8, Load(d, fin[8 + k] + i));
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

  // Word j of each stripe, interleaved such that block b of a step is absorbed
  // by stripe b % kStripes.
  HWY_ALIGN uint8_t state_[LeMac::kStateWords][kVecBytes];
  HWY_ALIGN uint8_t delay_[LeMac::kDelayWords][kVecBytes];
  const Key* key_;
};

template <size_t kStripes = 2, class D = LeMac::DefaultD<kStripes>>
using LeMacFingerprintHasher = LeMacHasher<kStripes, D, LeMacFingerprint>;

// Writes the LeMac or fingerprint (depending on `Variant`) of `size` bytes to
// `out`. The `nonce` is for domain separation, e.g. the position within a tree.
template <size_t kStripes, class Variant>
HWY_INLINE void LeMacHash(const LeMacKey<kStripes, Variant>& key,
                          const uint8_t nonce[LeMac::kNonceBytes],
                          const uint8_t* HWY_RESTRICT bytes, size_t size,
                          uint8_t out[LeMac::kHashBytes]) {
  using Hasher = LeMacHasher<kStripes, LeMac::DefaultD<kStripes>, Variant>;
  Hasher hasher(key);
  const size_t whole = size & ~(Hasher::kUpdateBytes - 1);
  if (whole != 0) hasher.Update(bytes, whole);
  hasher.Finalize(bytes + whole, size - whole, nonce, out);
}

#endif  // HWY_TARGET != HWY_SCALAR

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_HASH_LEMAC_INL_H_
