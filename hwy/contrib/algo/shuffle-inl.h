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

// Triple32 from hash/hash-inl.h. That header cannot be included here because
// it depends on random-inl.h and thus VQSort, which includes contrib/algo.
HWY_INLINE uint32_t ShuffleHash32(uint32_t key, uint32_t x) {
  x ^= key;
  x ^= x >> 17;
  x *= 0xED5AD4BBu;
  x ^= x >> 11;
  x *= 0xAC4C1B51u;
  x ^= x >> 15;
  x *= 0x31848BABu;
  x ^= x >> 14;
  return x;
}

template <class DU32, class VU32 = Vec<DU32>>
HWY_INLINE VU32 ShuffleHash32(DU32 du32, uint32_t key, VU32 x) {
  x = Xor(x, Set(du32, key));
  x = Xor(x, ShiftRight<17>(x));
  x = Mul(x, Set(du32, 0xED5AD4BBu));
  x = Xor(x, ShiftRight<11>(x));
  x = Mul(x, Set(du32, 0xAC4C1B51u));
  x = Xor(x, ShiftRight<15>(x));
  x = Mul(x, Set(du32, 0x31848BABu));
  return Xor(x, ShiftRight<14>(x));
}

// Returns a position in [0, i] from 64 random bits, for i past the u32 path.
HWY_INLINE uint64_t ShuffleIndex64(uint64_t bits, uint64_t i) {
  uint64_t upper;
  Mul128(bits, i + 1, &upper);
  return upper;
}

// Random bits that depend only on the seed and the swap position.
class ShuffleSeedBits {
 public:
  explicit ShuffleSeedBits(uint64_t seed)
      : seed_(seed), key_(static_cast<uint32_t>(seed ^ (seed >> 32))) {}

  // SplitMix64 finalizer.
  uint64_t Bits64(uint64_t i) const {
    uint64_t x = i ^ seed_;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
  }

  uint32_t Bits32(uint32_t i) const { return ShuffleHash32(key_, i); }

  template <class DU32, class VU32 = Vec<DU32>>
  VU32 Bits(DU32 du32, VU32 positions, uint32_t* /*buf*/) const {
    return ShuffleHash32(du32, key_, positions);
  }

 private:
  uint64_t seed_;
  uint32_t key_;
};

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

  uint64_t Bits64(uint64_t /*i*/) {
    if constexpr (sizeof(Result) >= sizeof(uint64_t)) {
      if ((URBG::max)() - (URBG::min)() == ~Result{0}) {
        return static_cast<uint64_t>(g_() - (URBG::min)());
      }
    }
    const uint64_t upper = Bits32(0);
    return (upper << 32) | Bits32(0);
  }

  uint32_t Bits32(uint32_t /*i*/) {
    return static_cast<uint32_t>(g_() - (URBG::min)());
  }

  template <class DU32, class VU32 = Vec<DU32>>
  VU32 Bits(DU32 du32, VU32 /*positions*/, uint32_t* buf) {
    for (size_t k = Lanes(du32); k-- != 0;) {
      buf[k] = Bits32(0);
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
    std::swap(inout[i], inout[ShuffleIndex64(bits.Bits64(i), i)]);
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
    const Vec<decltype(du32)> rand = bits.Bits(du32, positions, js);
    Store(MulHigh(rand, Add(positions, k1)), du32, js);
    for (size_t k = N; k-- != 0;) {
      std::swap(inout[lo + k], inout[js[k]]);
    }
    i -= N;
  }

  for (; i != 0; --i) {
    const uint32_t i32 = static_cast<uint32_t>(i);
    std::swap(inout[i], inout[LemireMod(bits.Bits32(i32), i32 + 1)]);
  }
}

}  // namespace detail

// Randomly permutes `inout[0, count)`, like std::shuffle. The permutation
// depends only on `seed`, so it is the same on every target. Positions come
// from Lemire's multiply-shift, whose bias is negligible for count << 2^32.
template <class D, typename T = TFromD<D>>
void ShuffleSpan(D d, T* HWY_RESTRICT inout, size_t count, uint64_t seed) {
  detail::ShuffleSeedBits bits(seed);
  detail::ShuffleSpanImpl(d, inout, count, bits);
}

// As above, but with random bits drawn from `g`, a UniformRandomBitGenerator
// with at least a 32-bit range, such as RngStream or std::mt19937.
template <class D, class URBG, typename T = TFromD<D>,
          hwy::EnableIf<!hwy::IsInteger<URBG>()>* = nullptr>
void ShuffleSpan(D d, T* HWY_RESTRICT inout, size_t count, URBG&& g) {
  detail::ShuffleDrawnBits<RemoveCvRef<URBG>> bits(g);
  detail::ShuffleSpanImpl(d, inout, count, bits);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_ALGO_SHUFFLE_INL_H_
