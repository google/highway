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

#include <type_traits>  // std::is_trivially_copyable
#include <utility>      // std::swap

// Per-target include guard
#if defined(HIGHWAY_HWY_CONTRIB_ALGO_SHUFFLE_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_ALGO_SHUFFLE_INL_H_
#undef HIGHWAY_HWY_CONTRIB_ALGO_SHUFFLE_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_ALGO_SHUFFLE_INL_H_
#endif

#include "hwy/contrib/hash/hash-inl.h"
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

// Random bits drawn from a UniformRandomBitGenerator, in the order the callers
// below ask for them.
template <class URBG>
class ShuffleDrawnBits {
  using Result = typename URBG::result_type;

 public:
  explicit ShuffleDrawnBits(URBG& g) : g_(g) {
    // Draws must take 2^k values, else their low 32 bits are not uniform.
    constexpr uint64_t kRange =
        static_cast<uint64_t>((URBG::max)() - (URBG::min)());
    static_assert(kRange >= 0xFFFFFFFFu && (kRange & (kRange + 1)) == 0,
                  "ShuffleSpan needs a generator of 32 or more random bits");
  }

  uint64_t Bits64() {
    HWY_IF_CONSTEXPR(sizeof(Result) >= sizeof(uint64_t)) {
      if ((URBG::max)() - (URBG::min)() == ~Result{0}) {
        return static_cast<uint64_t>(g_() - (URBG::min)());
      }
    }
    const uint64_t upper = Bits32();
    return (upper << 32) | Bits32();
  }

 private:
  uint32_t Bits32() { return static_cast<uint32_t>(g_() - (URBG::min)()); }

  URBG& g_;
};

// Positions per key; a constant, so every target draws the same.
constexpr size_t kShuffleBlock = 32;

// Positions are u32 whatever T is, in D's vector size. At most one block, so
// the buffers stay on the stack.
template <class D>
using ShuffleTagU32 =
    CappedTag<uint32_t,
              HWY_MIN(MaxLanes(Repartition<uint32_t, D>()), kShuffleBlock)>;

// Odd, so positions get distinct hash inputs, and large, so that the carries in
// offset + position * kShuffleStep depend on every bit of the offset.
constexpr uint32_t kShuffleStep = 0x9E3779B9u;

// 64 random bits per position from a 64-bit key (2^63 outcomes per block). With
// a 32-bit key, a block would repeat after about 2^16 shuffles.
class ShuffleHash {
 public:
  explicit ShuffleHash(uint64_t key)
      : offset_(static_cast<uint32_t>(key >> 32)),
        upper_(static_cast<uint32_t>(key)),
        lower_(offset_) {}

  // For positions first + [0, Lanes); `steps` is Iota(du32, 0) * kShuffleStep.
  template <class DU32>
  HWY_INLINE Vec<DU32> Upper(DU32 du32, uint32_t first, Vec<DU32> steps) const {
    const uint32_t start = offset_ + first * kShuffleStep;
    return upper_.OneVec(du32, Add(Set(du32, start), steps));
  }

  template <class DU32>
  HWY_INLINE Vec<DU32> Lower(DU32 du32, Vec<DU32> upper) const {
    return lower_.OneVec(du32, upper);
  }

 private:
  uint32_t offset_;
  Triple32 upper_;
  Triple32 lower_;
};

// (upper:lower * range) >> 64, Lemire's multiply-shift of 64 random bits. Calls
// hash.Lower only if the low half of upper * range is within range of 2^32.
template <class DU32, class VU32 = Vec<DU32>, class Hash>
HWY_INLINE VU32 ShuffleTarget(DU32 du32, VU32 upper, VU32 range,
                              const Hash& hash) {
  const VU32 high = MulHigh(upper, range);
  const VU32 mid = Mul(upper, range);
  if (HWY_LIKELY(AllFalse(du32, Lt(Add(mid, range), mid)))) return high;
  const VU32 sum = Add(mid, MulHigh(hash.Lower(du32, upper), range));
  return Sub(high, VecFromMask(du32, Lt(sum, mid)));
}

// Fisher-Yates from the top down: position i swaps with a position in [0, i]
// from ShuffleHash of i, with a new key for each block of kShuffleBlock
// positions. The last block takes 32 bits per position from `bits` instead.
template <class D, typename T, class Bits>
void ShuffleSpanImpl(D /*d*/, T* HWY_RESTRICT inout, size_t count, Bits& bits) {
  if (count < 2) return;
  size_t i = count - 1;

  // Keeps i + 1 within u32 below. Only reachable with 64-bit size_t.
  for (; i >= size_t{0xFFFFFFFFu}; --i) {
    std::swap(inout[i], inout[ShuffleIndex64(bits.Bits64(), i)]);
  }

  const ShuffleTagU32<D> du32;
  using VU32 = Vec<decltype(du32)>;
  const size_t N = Lanes(du32);
  const VU32 k1 = Set(du32, 1u);
  const VU32 steps = Mul(Iota(du32, 0u), Set(du32, kShuffleStep));
  HWY_ALIGN uint32_t js[kShuffleBlock + MaxLanes(du32)];
  while (i >= kShuffleBlock) {
    const size_t lo = i / kShuffleBlock * kShuffleBlock;
    const ShuffleHash hash(bits.Bits64());
    for (size_t k = 0; k <= i - lo; k += N) {
      const uint32_t first = static_cast<uint32_t>(lo + k);
      const VU32 upper = hash.Upper(du32, first, steps);
      const VU32 range = Add(Iota(du32, first), k1);
      Store(ShuffleTarget(du32, upper, range, hash), du32, js + k);
    }
    for (size_t q = i; q >= lo; --q) {
      std::swap(inout[q], inout[js[q - lo]]);
    }
    i = lo - 1;
  }

  // One key would allow at most 2^64 of the 32! (~2^118) orderings, so each
  // position takes 32 bits: the upper, then the lower half of a draw.
  for (; i >= 2; i -= 2) {
    const uint64_t r = bits.Bits64();
    const uint32_t upper = static_cast<uint32_t>(r >> 32);
    const uint32_t lower = static_cast<uint32_t>(r);
    const uint32_t i32 = static_cast<uint32_t>(i);
    std::swap(inout[i], inout[MulHigh32(upper, i32 + 1)]);
    std::swap(inout[i - 1], inout[MulHigh32(lower, i32)]);
  }
  if (i == 1) {
    const uint32_t upper = static_cast<uint32_t>(bits.Bits64() >> 32);
    std::swap(inout[1], inout[MulHigh32(upper, 2u)]);
  }
}

// Constants, so the bucket count and thus the permutation match on every
// machine. The cap bounds the stack; larger arrays get larger buckets.
constexpr size_t kShuffleBucketMinBytes = size_t{64} << 20;
constexpr size_t kShuffleBucketBytes = size_t{1} << 20;
constexpr size_t kShuffleMaxLog2Buckets = 10;

// Returns log2 of the bucket count, or 0 if ShuffleSpanImpl runs directly.
// Buckets are filled by copying bytes, hence only for trivially copyable T.
template <typename T>
constexpr size_t ShuffleLog2Buckets(size_t count) {
  if (!std::is_trivially_copyable<T>::value ||
      sizeof(T) > kShuffleBucketBytes ||
      count < DivCeil(kShuffleBucketMinBytes, sizeof(T))) {
    return 0;
  }
  size_t log2 = 1;
  while (log2 < kShuffleMaxLog2Buckets &&
         ((kShuffleBucketBytes / sizeof(T)) << log2) < count) {
    ++log2;
  }
  return log2;
}

// The scattered copy, then one 64-bit key per block.
template <typename T>
constexpr size_t ShuffleBucketBufNum(size_t count) {
  return count +
         DivCeil(DivCeil(count, kShuffleBlock) * sizeof(uint64_t), sizeof(T));
}

// Bucket of each position in [lo, lo + kShuffleBlock): the upper bits of its
// ShuffleHash under `key`. `steps` is as for ShuffleHash::Upper.
template <class DU32>
HWY_INLINE void ShuffleBucketIds(DU32 du32, uint64_t key, size_t lo, int shift,
                                 Vec<DU32> steps, uint32_t* HWY_RESTRICT ids) {
  const ShuffleHash hash(key);
  for (size_t k = 0; k < kShuffleBlock; k += Lanes(du32)) {
    const uint32_t first = static_cast<uint32_t>(lo + k);
    Store(ShiftRightSame(hash.Upper(du32, first, steps), shift), du32, ids + k);
  }
}

// Rao-Sandelius: scatters into 2^log2_buckets buckets keyed per block as in
// ShuffleSpanImpl, then shuffles each. `buf`: ShuffleBucketBufNum<T>(count).
template <class D, typename T, class Bits>
void ShuffleBuckets(D d, T* HWY_RESTRICT inout, size_t count, Bits& bits,
                    size_t log2_buckets, T* HWY_RESTRICT buf) {
  HWY_DASSERT(1 <= log2_buckets && log2_buckets <= kShuffleMaxLog2Buckets);
  HWY_DASSERT(buf != nullptr);
  const size_t num_buckets = size_t{1} << log2_buckets;
  const int shift = static_cast<int>(32 - log2_buckets);
  const ShuffleTagU32<D> du32;
  const Vec<decltype(du32)> steps =
      Mul(Iota(du32, 0u), Set(du32, kShuffleStep));
  HWY_ALIGN uint32_t ids[kShuffleBlock + MaxLanes(du32)];
  // Keys are kept so the scatter can recompute each block's buckets.
  uint8_t* keys = reinterpret_cast<uint8_t*>(buf + count);
  size_t ends[size_t{1} << kShuffleMaxLog2Buckets];
  ZeroBytes(ends, num_buckets * sizeof(size_t));

  for (size_t lo = 0; lo < count; lo += kShuffleBlock) {
    const uint64_t key = bits.Bits64();
    CopyBytes<8>(&key, keys + lo / kShuffleBlock * 8);
    ShuffleBucketIds(du32, key, lo, shift, steps, ids);
    const size_t num = HWY_MIN(kShuffleBlock, count - lo);
    for (size_t k = 0; k < num; ++k) ++ends[ids[k]];
  }
  size_t begin = 0;
  for (size_t b = 0; b < num_buckets; ++b) {  // sizes to starts
    const size_t size = ends[b];
    ends[b] = begin;
    begin += size;
  }
  for (size_t lo = 0; lo < count; lo += kShuffleBlock) {
    uint64_t key;
    CopyBytes<8>(keys + lo / kShuffleBlock * 8, &key);
    ShuffleBucketIds(du32, key, lo, shift, steps, ids);
    const size_t num = HWY_MIN(kShuffleBlock, count - lo);
    for (size_t k = 0; k < num; ++k) {
      // Bytes, so this also compiles for T that cannot be copied.
      CopyBytes<sizeof(T)>(static_cast<const void*>(inout + lo + k),
                           static_cast<void*>(buf + ends[ids[k]]++));
    }
  }
  begin = 0;
  for (size_t b = 0; b < num_buckets; ++b) {
    ShuffleSpanImpl(d, buf + begin, ends[b] - begin, bits);
    begin = ends[b];
  }
  CopyBytes(buf, inout, count * sizeof(T));
}

}  // namespace detail

// Returns how many elements the `buf` passed to ShuffleSpan must hold. This is
// zero unless `count` elements of T are at least 64 MiB and trivially copyable.
template <typename T>
constexpr size_t ShuffleSpanBufNum(size_t count) {
  return detail::ShuffleLog2Buckets<T>(count) == 0
             ? 0
             : detail::ShuffleBucketBufNum<T>(count);
}

// Randomly permutes `inout[0, count)` in place, like std::shuffle, with random
// bits drawn from `g`, a UniformRandomBitGenerator of 32 or more random bits,
// such as RngStream. Each block of 32 positions hashes the position under a new
// 64-bit key from `g`, except the last 32 positions, which take 32 bits each
// from `g`. The permutation thus depends only on `g` and `count`, and is the
// same on every target.
// Positions come from Lemire's multiply-shift of 64 random bits (32 for the
// last 32 positions), so the bias is at most 2^-27 + count / 2^64.
template <class D, class URBG, typename T = TFromD<D>>
void ShuffleSpan(D d, T* HWY_RESTRICT inout, size_t count, URBG&& g) {
  detail::ShuffleDrawnBits<RemoveCvRef<URBG>> bits(g);
  detail::ShuffleSpanImpl(d, inout, count, bits);
}

// As above, but faster for large arrays: if ShuffleSpanBufNum<T>(count) is
// nonzero (from 64 MiB of trivially copyable T), the elements are first
// scattered into 64 to 1024 random buckets in `buf`, which must hold that many
// elements, and each bucket is shuffled in cache. The permutation then also
// depends on sizeof(T), and differs from the one above.
template <class D, class URBG, typename T = TFromD<D>>
void ShuffleSpan(D d, T* HWY_RESTRICT inout, size_t count, URBG&& g,
                 T* HWY_RESTRICT buf) {
  detail::ShuffleDrawnBits<RemoveCvRef<URBG>> bits(g);
  const size_t log2_buckets = detail::ShuffleLog2Buckets<T>(count);
  if (log2_buckets == 0) {
    detail::ShuffleSpanImpl(d, inout, count, bits);
  } else {
    detail::ShuffleBuckets(d, inout, count, bits, log2_buckets, buf);
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_ALGO_SHUFFLE_INL_H_
