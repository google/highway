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

// Non-SIMD part of LeMac (see lemac-inl.h): sizes, the key material and the
// incremental state. Both are plain data whose layout and contents do not
// depend on the target, hence they can be stored in non-SIMD headers, copied,
// and used by any target. Constructing the key and absorbing input require
// AES, see LeMacInitKey, LeMacUpdate and LeMacFinalize in lemac-inl.h.

#ifndef HIGHWAY_HWY_CONTRIB_HASH_LEMAC_H_
#define HIGHWAY_HWY_CONTRIB_HASH_LEMAC_H_

#include <stddef.h>
#include <stdint.h>

#include "hwy/base.h"

namespace hwy {

// Tag classes for the `Variant` template parameter of LeMacKey: full LeMac
// specification, or reduced finalization for non-cryptographic corruption
// detection.
struct LeMacOriginal {};
struct LeMacFingerprint {};

// Sizes of the inputs and output, and of the internal state.
struct LeMac {
  static constexpr size_t kKeyBytes = 16;
  static constexpr size_t kNonceBytes = 16;
  static constexpr size_t kHashBytes = 16;

  // One AES block, which is also the size of a state word and an input block.
  static constexpr size_t kBlockBytes = 16;
  // Per stripe: state words updated with AES rounds, and words of delayed
  // input (RR, R0, R1, R2 in the reference).
  static constexpr size_t kStateWords = 9;
  static constexpr size_t kDelayWords = 4;
  // Input blocks absorbed per stripe and step.
  static constexpr size_t kBlocksPerStep = 4;
  // Steps of zero input after the padding, which are required for the last
  // input blocks to pass through the delay words and enough S-boxes.
  static constexpr size_t kDrainSteps = 4;
  // Upper bound on the input bytes per pass, see LeMacUpdate. Larger passes
  // amortize loading/storing the state; 8 KiB is about 4% faster than 2 KiB on
  // Zen3, and still fits in L1 so that later passes hit.
  static constexpr size_t kPassBytes = 8192;
  // AES-128 round keys for deriving the key material from the key.
  static constexpr size_t kDeriveRoundKeys = 11;
};

// Key material derived from a 128-bit key by LeMacInitKey (lemac-inl.h).
// Immutable after that, hence thread-safe; for efficiency, initialize once and
// share. `Variant = LeMacOriginal` is the full LeMac specification;
// `Variant = LeMacFingerprint` uses fewer finalization rounds for
// non-cryptographic corruption detection.
template <size_t kStripes = 2, class Variant = LeMacOriginal>
class HWY_ALIGN_MAX LeMacKey {
  static_assert(kStripes == 1 || kStripes == 2, "Only LeMac-x1 or -x2");
  static_assert(IsSame<Variant, LeMacOriginal>() ||
                    IsSame<Variant, LeMacFingerprint>(),
                "Variant must be LeMacOriginal or LeMacFingerprint");

 public:
  static constexpr size_t kVecBytes = LeMac::kBlockBytes * kStripes;
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

  // Initial value of state word j for each stripe, interleaved as in
  // LeMacState.
  uint8_t init[LeMac::kStateWords][kVecBytes];
  // Subkey t of each stripe: state word j uses rows [j, j + kFinKeys).
  uint8_t fin[kFinRows][kVecBytes];
  // Round keys for encrypting the nonce (k2), followed by those of the final
  // encryption (k3) for LeMacOriginal.
  uint8_t enc_keys[kEncryptions * kRoundKeys][LeMac::kBlockBytes];
};

template <size_t kStripes = 2>
using LeMacFingerprintKey = LeMacKey<kStripes, LeMacFingerprint>;

// State for hashing inputs that arrive in several pieces: call Reset, then
// LeMacUpdate (lemac-inl.h) any number of times with any sizes, then
// LeMacFinalize with any remaining bytes. Each thread requires its own state.
template <size_t kStripes = 2>
class HWY_ALIGN_MAX LeMacState {
  static_assert(kStripes == 1 || kStripes == 2, "Only LeMac-x1 or -x2");

 public:
  static constexpr size_t kVecBytes = LeMac::kBlockBytes * kStripes;
  // Each step absorbs four input blocks per stripe.
  static constexpr size_t kStepBytes = LeMac::kBlocksPerStep * kVecBytes;
  // Unit of absorption (one pair of steps); smaller remainders are buffered.
  static constexpr size_t kMinUpdateBytes = 2 * kStepBytes;
  static_assert(LeMac::kPassBytes % kMinUpdateBytes == 0, "Even steps");

  // Prepares for hashing a new input with `key`.
  template <class Variant>
  void Reset(const LeMacKey<kStripes, Variant>& key) {
    CopyBytes(key.init, state, sizeof(state));
    ZeroBytes(delay, sizeof(delay));
    filled = 0;
  }

  // Word j of each stripe, interleaved such that block b of a step is absorbed
  // by stripe b % kStripes.
  uint8_t state[LeMac::kStateWords][kVecBytes];
  uint8_t delay[LeMac::kDelayWords][kVecBytes];
  // The first `filled` (< kMinUpdateBytes) bytes are not yet absorbed.
  uint8_t buf[kMinUpdateBytes];
  uint32_t filled = 0;
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_HASH_LEMAC_H_
