// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause
//
// See the LICENSE file in the project root for the full license text.

#ifndef HIGHWAY_HWY_GENERATOR_H_
#define HIGHWAY_HWY_GENERATOR_H_

#include <stddef.h>

#include <utility>

namespace hwy {

// Composes a bit generator with caller-supplied distributions. This header is
// target-independent and does not include any concrete backend or distribution.
// BitGenerator owns its state; any resources it borrows must outlive Generator.
template <class BitGenerator>
class Generator {
 public:
  explicit Generator(BitGenerator bit_generator)
      : bit_generator_(std::move(bit_generator)) {}

  // Returns the backend's native result (a scalar or SIMD vector).
  auto operator()() -> decltype(std::declval<BitGenerator&>()()) {
    return bit_generator_();
  }

  // A distribution consumes random bits through operator()(BitGenerator&).
  // Passing an lvalue preserves any state cached by the distribution.
  template <class Distribution>
  auto Sample(Distribution&& distribution)
      -> decltype(std::forward<Distribution>(distribution)(
          std::declval<BitGenerator&>())) {
    return std::forward<Distribution>(distribution)(bit_generator_);
  }

  // Distributions implement Fill(BitGenerator&, T*, size_t). This lets SIMD
  // distributions fuse conversion and stores with a backend's batch loop.
  template <class Distribution, typename T>
  void Fill(Distribution&& distribution, T* out, size_t count) {
    std::forward<Distribution>(distribution).Fill(bit_generator_, out, count);
  }

  BitGenerator& GetBitGenerator() { return bit_generator_; }
  const BitGenerator& GetBitGenerator() const { return bit_generator_; }

 private:
  BitGenerator bit_generator_;
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_GENERATOR_H_
