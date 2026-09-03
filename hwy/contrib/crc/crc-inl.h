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

// Portable SIMD CRC-64/XZ (resolves the "Add CRC example" op_wishlist item).
//
// The bulk of the message is reduced with carryless-multiply folding (Highway's
// CLMulLower/CLMulUpper), which every target provides - natively via
// PCLMULQDQ / PMULL / vpmsumd, or through the portable emulation. It uses only
// a fixed 16-byte block held in named locals, so it works on fixed and scalable
// SIMD alike; only HWY_SCALAR (1 lane, no 16-byte block) falls back to the
// bit-at-a-time path. A single source runs everywhere and returns the standard
// check value.
//
// Reference: V. Gopal et al., "Fast CRC Computation for Generic Polynomials
// Using PCLMULQDQ Instruction" (Intel, 2009).

// Include guard (still compiled once per target)
#if defined(HIGHWAY_HWY_CONTRIB_CRC_CRC_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_CRC_CRC_INL_H_
#undef HIGHWAY_HWY_CONTRIB_CRC_CRC_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_CRC_CRC_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace crc {

// CRC-64/XZ (a.k.a. CRC-64/GO-ECMA): reflected, poly 0x42F0E1EBA9EA3693,
// init and xor-out all-ones. check("123456789") == 0x995DC9BBDF1939FA.
namespace detail {

// Reflected polynomial (bit i holds the coefficient of x^(63-i)); the x^64 term
// is implicit.
constexpr uint64_t kPolyReflected = 0xC96C5795D7870F42ull;

// Reflected form of x^(n + 63) mod P: start from the reflected x^63 (bit 0) and
// apply one reflected CRC step (multiply the remainder by x, reduce) n times.
constexpr uint64_t RefXn63(unsigned n) {
  uint64_t r = 1ull;
  for (unsigned i = 0; i < n; ++i) {
    r = (r >> 1) ^ (kPolyReflected & (uint64_t{0} - (r & 1)));
  }
  return r;
}

// One byte of the reflected CRC, LSB-first.
HWY_INLINE uint64_t Byte(uint64_t crc, uint8_t byte) {
  crc ^= byte;
  for (int i = 0; i < 8; ++i) {
    crc = (crc >> 1) ^ (kPolyReflected & (uint64_t{0} - (crc & 1)));
  }
  return crc;
}

// Bit-at-a-time reflected CRC on a "raw" state (no init / xor-out applied).
HWY_INLINE uint64_t Bitwise(uint64_t crc, const uint8_t* data, size_t size) {
  for (size_t i = 0; i < size; ++i) crc = Byte(crc, data[i]);
  return crc;
}

#if HWY_TARGET != HWY_SCALAR

// Folds >= 16 bytes with CLMUL, then finishes with Bitwise(). `state` is the
// raw running CRC (prev ^ xorout); it is folded in at the position of data[0].
HWY_INLINE uint64_t Fold(uint64_t state, const uint8_t* data, size_t size) {
  const Full128<uint64_t> d64;
  const Full128<uint8_t> d8;
  using V = VFromD<decltype(d64)>;

  // Fold-by-16. data[0..7] land in the low lane and are the more significant
  // half of the reflected polynomial, so the low lane is multiplied by
  // x^192 mod P and the high lane by x^128 mod P. CLMul on bit-reflected inputs
  // yields reflect(a*b*x), so the exponents drop by one to 191 and 127
  // (RefXn63 offsets 128 and 64).
  const V k = Dup128VecFromValues(d64, RefXn63(128), RefXn63(64));

  V acc = BitCast(d64, LoadU(d8, data));
  acc = Xor(acc, Dup128VecFromValues(d64, state, 0));

  size_t pos = 16;
  for (; pos + 16 <= size; pos += 16) {
    const V next = BitCast(d64, LoadU(d8, data + pos));
    acc = Xor3(CLMulLower(acc, k), CLMulUpper(acc, k), next);
  }

  // acc (reflected, 128-bit) is now congruent to the processed prefix mod P.
  // Bitwise() over its 16 bytes yields prefix * x^64 mod P == the raw CRC.
  uint8_t bytes[16];
  StoreU(BitCast(d8, acc), d8, bytes);
  uint64_t crc = Bitwise(0, bytes, 16);
  return Bitwise(crc, data + pos, size - pos);
}

#endif  // HWY_TARGET != HWY_SCALAR

}  // namespace detail

// Streaming CRC-64/XZ. Pass 0 for `prev` to start; feed the previous return
// value back to continue over more data. Bit-identical on every target.
HWY_INLINE uint64_t Crc64Xz(const uint8_t* data, size_t size,
                            uint64_t prev = 0) {
  uint64_t state = ~prev;  // undo xor-out (init == xor-out == ~0)

#if HWY_TARGET != HWY_SCALAR
  if (size >= 16) {
    state = detail::Fold(state, data, size);
  } else {
    state = detail::Bitwise(state, data, size);
  }
#else
  state = detail::Bitwise(state, data, size);
#endif

  return ~state;
}

}  // namespace crc
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // include guard
