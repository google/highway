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

#include <utility>  // std::swap

// Per-target include guard
#if defined(HIGHWAY_HWY_CONTRIB_ALGO_SHUFFLE_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_ALGO_SHUFFLE_INL_H_
#undef HIGHWAY_HWY_CONTRIB_ALGO_SHUFFLE_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_ALGO_SHUFFLE_INL_H_
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace detail {

// Returns a position in [0, i] from 64 random bits, for i past the u32 path.
HWY_INLINE uint64_t ShuffleIndex64(uint64_t bits, uint64_t i) {
  uint64_t upper;
  Mul128(bits, i + 1, &upper);
  return upper;
}

// Random bits drawn from a UniformRandomBitGenerator, in order of decreasing
// swap position, as a sequential loop would.
template <class URBG>
class ShuffleDrawnBits {
  using Result = typename URBG::result_type;

 public:
  explicit ShuffleDrawnBits(URBG& g) : g_(g) {
    static_assert(
        static_cast<uint64_t>((URBG::max)() - (URBG::min)()) >= 0xFFFFFFFFu,
        "ShuffleSpan needs a generator with at least 32 bits");
  }

  uint64_t Bits64() {
    if constexpr (sizeof(Result) >= sizeof(uint64_t)) {
      if ((URBG::max)() - (URBG::min)() == ~Result{0}) {
        return static_cast<uint64_t>(g_() - (URBG::min)());
      }
    }
    const uint64_t upper = Bits32();
    return (upper << 32) | Bits32();
  }

  uint32_t Bits32() { return static_cast<uint32_t>(g_() - (URBG::min)()); }

  template <class DU32, class VU32 = Vec<DU32>>
  VU32 Bits(DU32 du32, uint32_t* buf) {
    for (size_t k = Lanes(du32); k-- != 0;) {
      buf[k] = Bits32();
    }
    return Load(du32, buf);
  }

 private:
  URBG& g_;
};

// Fisher-Yates: position i swaps with a random position in [0, i], from the
// top down. Positions are generated a vector at a time, then swapped in order.
template <class D, typename T, class Bits>
void ShuffleSpanImpl(D /*d*/, T* HWY_RESTRICT inout, size_t count, Bits& bits) {
  if (count < 2) return;
  size_t i = count - 1;

  // Keeps i + 1 within u32 below. Only reachable with 64-bit size_t.
  for (; i >= size_t{0xFFFFFFFFu}; --i) {
    std::swap(inout[i], inout[ShuffleIndex64(bits.Bits64(), i)]);
  }

  // Positions are u32 whatever T is; capping keeps the buffer on the stack.
  const CappedTag<uint32_t, HWY_MIN(HWY_MAX_LANES_D(D), 64)> du32;
  const size_t N = Lanes(du32);
  const Vec<decltype(du32)> k1 = Set(du32, 1u);
  HWY_ALIGN uint32_t js[MaxLanes(du32)];
  // Each batch handles [lo, i]; position 0 never needs a swap.
  while (i >= N) {
    const uint32_t lo = static_cast<uint32_t>(i - (N - 1));
    const Vec<decltype(du32)> positions = Iota(du32, lo);
    const Vec<decltype(du32)> rand = bits.Bits(du32, js);
    Store(MulHigh(rand, Add(positions, k1)), du32, js);
    for (size_t k = N; k-- != 0;) {
      std::swap(inout[lo + k], inout[js[k]]);
    }
    i -= N;
  }

  for (; i != 0; --i) {
    const uint32_t i32 = static_cast<uint32_t>(i);
    std::swap(inout[i], inout[LemireMod(bits.Bits32(), i32 + 1)]);
  }
}

}  // namespace detail

// Randomly permutes `inout[0, count)`, like std::shuffle, with random bits
// drawn from `g`, a UniformRandomBitGenerator with at least a 32-bit range,
// such as RngStream or std::mt19937. Draws are consumed in the same order as a
// sequential loop, so the permutation is the same on every target. Positions
// come from Lemire's multiply-shift, whose bias is negligible for
// count << 2^32.
template <class D, class URBG, typename T = TFromD<D>>
void ShuffleSpan(D d, T* HWY_RESTRICT inout, size_t count, URBG&& g) {
  detail::ShuffleDrawnBits<RemoveCvRef<URBG>> bits(g);
  detail::ShuffleSpanImpl(d, inout, count, bits);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_ALGO_SHUFFLE_INL_H_
