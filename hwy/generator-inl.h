// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause
//
// See the LICENSE file in the project root for the full license text.

#if defined(HIGHWAY_HWY_GENERATOR_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_GENERATOR_INL_H_
#undef HIGHWAY_HWY_GENERATOR_INL_H_
#else
#define HIGHWAY_HWY_GENERATOR_INL_H_
#endif

#include <stddef.h>

#include <utility>

#include "hwy/generator.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// Composes scalar or SIMD backends with distributions in this target's scope.
// Native vector returns require the same target attributes as the backend.
template <class BitGenerator>
class Generator {
 public:
  explicit Generator(BitGenerator bit_generator)
      : generator_(std::move(bit_generator)) {}

  HWY_INLINE auto operator()() -> decltype(std::declval<BitGenerator&>()()) {
    return generator_.GetBitGenerator()();
  }

  template <class Distribution>
  HWY_INLINE auto Sample(Distribution&& distribution)
      -> decltype(std::forward<Distribution>(distribution)(
          std::declval<BitGenerator&>())) {
    return std::forward<Distribution>(distribution)(
        generator_.GetBitGenerator());
  }

  template <class Distribution, typename T>
  HWY_INLINE void Fill(Distribution&& distribution, T* out, size_t count) {
    std::forward<Distribution>(distribution)
        .Fill(generator_.GetBitGenerator(), out, count);
  }

  BitGenerator& GetBitGenerator() { return generator_.GetBitGenerator(); }
  const BitGenerator& GetBitGenerator() const {
    return generator_.GetBitGenerator();
  }

 private:
  hwy::Generator<BitGenerator> generator_;
};

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_GENERATOR_INL_H_
