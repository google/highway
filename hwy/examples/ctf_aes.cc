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

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <iostream>
#include <numeric>

#include "hwy/timer.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/examples/ctf_aes.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep

#include "hwy/highway.h"

/*
Highway SIMD Tutorial: AES Cryptography Brute-force Challenge (CTR Mode)

THE TASK:
A secret 16-byte message (the "flag") has been encrypted using a 6-digit password
(range 0-999,999) in a simplified AES-CTR (Counter) mode.

- WE KNOW: The decrypted message is guaranteed to start with the prefix "GG EZ ".
- WE DO NOT KNOW: The rest of the decrypted message.
- THE GOAL: Brute-force all 1,000,000 candidate passwords in parallel using SIMD,
  decrypting the ciphertext block, and verifying if the result starts with
  the known prefix "GG EZ ". Once a match is found, extract the complete flag.

How the Encryption and Decryption works:
In standard AES-CTR (Counter) mode, the block cipher is used as a stream cipher.
Instead of encrypting the plaintext directly, we encrypt a sequence of counters
and XOR the resulting keystream with the plaintext.

1. Encryption:
   For a 16-byte block (Plaintext):
   - We have a 16-byte Counter: {1, 2, 3, ..., 16}.
   - We derive a 16-byte Key from a secret password.
   - We generate the Keystream by running one round of AES on the Counter:
     Keystream = AESRound(Counter, Key)
   - We XOR the Plaintext with the Keystream to get the Ciphertext:
     Ciphertext = Plaintext ^ Keystream

   ASCII Art Diagram:
   Plaintext  -----> [ XOR ] -----------------------------> Ciphertext
                        ^
                        |
   Key -------> [ AESRound ] <--- Counter (Fixed)

2. Decryption:
   Since XOR is self-inverse, decryption is identical to encryption!
   - We generate the exact same Keystream using the same Counter and Key:
     Keystream = AESRound(Counter, Key)
   - We XOR the Ciphertext with the Keystream to recover the Plaintext:
     Plaintext = Ciphertext ^ Keystream

This example demonstrates 128-bit fixed-width SIMD (processing 1 block at a time)
using FixedTag. It brute-forces candidate passwords by changing the key
until the decrypted block starts with the known prefix "GG EZ ". Note that
on platforms with wider registers, one can implement this more efficiently by
comparing M blocks simultaneously, but this was omitted due to its higher
complexity.

Password: 765432
Decrypted Flag: {ʇɟq‾pɯ1s} ZƎ ⅁⅁ (Intentonally upside down)

*/

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
namespace hn = hwy::HWY_NAMESPACE;

HWY_ALIGN constexpr uint8_t kPrefix[16] = {
    'G', 'G', ' ', 'E', 'Z', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// 128-bit fixed-width brute-force solver (processes 1 password at a time).
uint32_t Solve(const uint8_t* ciphertext, const uint8_t* counter,
               uint32_t max_pass, uint8_t* decrypted_flag) {
  static_assert(HWY_IS_LITTLE_ENDIAN, "Only little-endian supported");
  // FixedTag<uint8_t, 16> locks the vector width to exactly 128-bit (16 bytes)
  using D = hn::FixedTag<uint8_t, 16>;
  const D d;
  using V = hn::Vec<D>;
  using M_Type = hn::Mask<D>;

  // Load ciphertext and counter blocks
  V v_cipher = hn::Load(d, ciphertext);
  V v_ctr = hn::Load(d, counter);

  // Target prefix to match: "GG EZ"
  V v_prefix = hn::Load(d, kPrefix);

  // Generate mask for the first 5 bytes "GG EZ " (using FirstN)
  M_Type m_ggez = hn::FirstN(d, 5);

  // Buffer to build candidate keys
  HWY_ALIGN uint8_t key_bytes[16] = {0};

  for (uint32_t p = 0; p < max_pass; ++p) {
    CopyBytes(&p, key_bytes, 3);

    V v_key = hn::Load(d, key_bytes);

    // Keystream = AESRound(Counter, Key)
    V v_keystream = hn::AESRound(v_ctr, v_key);

    // Decrypted = Ciphertext ^ Keystream
    V v_dec = hn::Xor(v_cipher, v_keystream);

    // Compare decrypted text with "GG EZ" prefix under mask (using MaskedEq)
    M_Type mask_ggez_match = hn::MaskedEq(m_ggez, v_dec, v_prefix);

    // Check if all 5 lanes of the prefix matched
    if (hn::CountTrue(d, mask_ggez_match) == 5) {
      hn::Store(v_dec, d, decrypted_flag);
      return p;
    }
  }
  return static_cast<uint32_t>(-1);
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_EXPORT(Solve);

// Hardcoded ciphertext generated by gen_ciphertext using CTR mode
// Plaintext: "GG EZ {s1md_bft}"
// Hardcoded ciphertext generated by gen_ciphertext using CTR mode
// Password: 765432
static constexpr uint8_t kCiphertext[16] = {0x17, 0xff, 0x2b, 0x0a, 0xa1, 0xed,
                                            0x96, 0x20, 0x9e, 0x9c, 0x68, 0xec,
                                            0x75, 0xf5, 0x5c, 0x4a};

static uint32_t RunSolver(uint8_t* decrypted_flag) {
  // Password is 6 digits, so max is 999,999
  const uint32_t max_pass = 1000000;

  // Initialize 16-byte Counter block {1, 2, 3, ..., 16} using std::iota
  HWY_ALIGN uint8_t kCounter[16];
  std::iota(kCounter, kCounter + 16, 1);

  return HWY_DYNAMIC_DISPATCH(Solve)(kCiphertext, kCounter, max_pass,
                                     decrypted_flag);
}

}  // namespace hwy

int main() {
  HWY_ALIGN uint8_t decrypted_flag[17] = "[not found]";

  std::cout << "Starting CTF AES Brute-force Challenge (CTR Mode)...\n";
  std::cout << "Target Ciphertext: ";
  for (int i = 0; i < 16; ++i) printf("%02x", hwy::kCiphertext[i]);
  std::cout << "\n\n";

  const double t_start = hwy::platform::Now();
  uint32_t found_pass = hwy::RunSolver(decrypted_flag);
  const double t_end = hwy::platform::Now();
  std::cout << "[SIMD] Found password: "
            << static_cast<int32_t>(found_pass) << "\n\n";
  std::cout << "Decrypted Flag:\n" << decrypted_flag << "\n\n";

  const double duration = (t_end - t_start) * 1000.0;  // in ms
  std::cout << "SIMD solver took: " << duration << " ms\n";

  return found_pass != static_cast<uint32_t>(-1) ? 0 : 1;
}
#endif  // HWY_ONCE
