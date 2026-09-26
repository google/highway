// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause
//
// See the LICENSE file in the project root for the full license text.

#ifndef HIGHWAY_HWY_CONTRIB_DISTRIBUTIONS_UNIFORM_H_
#define HIGHWAY_HWY_CONTRIB_DISTRIBUTIONS_UNIFORM_H_

#include <stddef.h>
#include <stdint.h>

#include "hwy/base.h"

namespace hwy {

// Uniform double samples in [0, 1), using the high 53 bits of a full-range u64
// bit generator. No particular PRNG, seed representation or state is required.
struct Uniform {
  static HWY_CXX14_CONSTEXPR double FromBits(uint64_t bits) noexcept {
    return static_cast<double>(bits >> 11) *
           0.00000000000000011102230246251565404236316680908203125;
  }

  template <class BitGenerator>
  double operator()(BitGenerator& bits) const {
    return FromBits(bits());
  }

  template <class BitGenerator>
  void Fill(BitGenerator& bits, double* out, size_t count) const {
    for (size_t i = 0; i < count; ++i) out[i] = (*this)(bits);
  }
};

// Uniform float samples on a 23-bit grid in [-1, 1). Consumes one full-range
// u64 per result, using its low 23 bits as the mantissa.
struct NormalizedUniform {
  static float FromBits(uint64_t bits) noexcept {
    const uint32_t exponent = hwy::BitCastScalar<uint32_t>(1.0f);
    const uint32_t mantissa =
        static_cast<uint32_t>(bits) & hwy::MantissaMask<float>();
    const float f12 = hwy::BitCastScalar<float>(exponent | mantissa);
    return (2.0f * (f12 - 1.0f)) - 1.0f;
  }

  template <class BitGenerator>
  float operator()(BitGenerator& bits) const {
    return FromBits(bits());
  }

  template <class BitGenerator>
  void Fill(BitGenerator& bits, float* out, size_t count) const {
    for (size_t i = 0; i < count; ++i) out[i] = (*this)(bits);
  }
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_DISTRIBUTIONS_UNIFORM_H_
