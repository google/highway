// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause
//
// See the LICENSE file in the project root for the full license text.

#ifndef HIGHWAY_HWY_CONTRIB_RANDOM_CACHED_H_
#define HIGHWAY_HWY_CONTRIB_RANDOM_CACHED_H_

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <limits>
#include <utility>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

namespace hwy {

// Adapts any backend with FillBits(uint64_t*, size_t) to a scalar URBG. Refills
// preserve the backend's batch consumption policy, including discarded lanes.
// Copying this adapter (when supported by BitGenerator) also copies its cache
// and current position. It does not expose a backend-only state snapshot.
template <class BitGenerator, size_t kSize = 1024>
class BufferedBitGenerator {
 public:
  using result_type = uint64_t;

  static constexpr result_type(min)() {
    return (std::numeric_limits<result_type>::min)();
  }
  static constexpr result_type(max)() {
    return (std::numeric_limits<result_type>::max)();
  }

  explicit BufferedBitGenerator(BitGenerator bit_generator)
      : bit_generator_(std::move(bit_generator)), index_(0) {
    bit_generator_.FillBits(cache_.data(), kSize);
  }

  result_type operator()() {
    if (HWY_UNLIKELY(index_ == kSize)) {
      bit_generator_.FillBits(cache_.data(), kSize);
      index_ = 0;
    }
    return cache_[index_++];
  }

  void FillBits(uint64_t* out, size_t count) {
    for (size_t i = 0; i < count; ++i) out[i] = (*this)();
  }

 private:
  static_assert(kSize != 0 && (kSize & (kSize - 1)) == 0,
                "only power of 2 are supported");

  BitGenerator bit_generator_;
  alignas(HWY_ALIGNMENT) std::array<result_type, kSize> cache_;
  size_t index_;
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_RANDOM_CACHED_H_
