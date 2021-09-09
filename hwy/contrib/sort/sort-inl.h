// Copyright 2021 Google LLC
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

// Per-target include guard

#if defined(HIGHWAY_HWY_CONTRIB_SORT_SORT_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_SORT_SORT_INL_H_
#undef HIGHWAY_HWY_CONTRIB_SORT_SORT_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_SORT_SORT_INL_H_
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

#if HWY_TARGET != HWY_SCALAR && HWY_ARCH_X86

#define HWY_SORT_VERIFY 1

enum class SortOrder { kAscending, kDescending };

template <SortOrder kOrder, typename T>
bool Compare(T a, T b) {
  if (kOrder == SortOrder::kAscending) return a <= b;
  return a >= b;
}

namespace detail {

// TODO(janwas): move into op
template <class D, class V>
HWY_INLINE V InterleaveEven(D /* tag */, V hi, V lo) {
  hi = ShiftLeftLanes<1>(hi);
  return OddEven(hi, lo);
}

template <class D, class V>
HWY_INLINE V InterleaveOdd(D /* tag */, V hi, V lo) {
  lo = ShiftRightLanes<1>(lo);
  return OddEven(hi, lo);
}

HWY_API Vec128<uint32_t> Shuffle3120(Vec128<uint32_t> lo, Vec128<uint32_t> hi) {
  const Full128<uint32_t> d;
  const RebindToFloat<decltype(d)> df;
  constexpr int kShuffle = _MM_SHUFFLE(3, 1, 2, 0);
  return BitCast(d, Vec128<float>{_mm_shuffle_ps(
                        BitCast(df, lo).raw, BitCast(df, hi).raw, kShuffle)});
}
HWY_API Vec128<int32_t> Shuffle3120(Vec128<int32_t> lo, Vec128<int32_t> hi) {
  const Full128<int32_t> d;
  const RebindToFloat<decltype(d)> df;
  constexpr int kShuffle = _MM_SHUFFLE(3, 1, 2, 0);
  return BitCast(d, Vec128<float>{_mm_shuffle_ps(
                        BitCast(df, lo).raw, BitCast(df, hi).raw, kShuffle)});
}

HWY_API Vec128<uint32_t> Shuffle2031(Vec128<uint32_t> lo, Vec128<uint32_t> hi) {
  const Full128<uint32_t> d;
  const RebindToFloat<decltype(d)> df;
  constexpr int kShuffle = _MM_SHUFFLE(2, 0, 3, 1);
  return BitCast(d, Vec128<float>{_mm_shuffle_ps(
                        BitCast(df, lo).raw, BitCast(df, hi).raw, kShuffle)});
}
HWY_API Vec128<int32_t> Shuffle2031(Vec128<int32_t> lo, Vec128<int32_t> hi) {
  const Full128<int32_t> d;
  const RebindToFloat<decltype(d)> df;
  constexpr int kShuffle = _MM_SHUFFLE(2, 0, 3, 1);
  return BitCast(d, Vec128<float>{_mm_shuffle_ps(
                        BitCast(df, lo).raw, BitCast(df, hi).raw, kShuffle)});
}

// For each lane i: replaces a[i] with the first and b[i] with the second
// according to kOrder.
// Corresponds to a conditional swap, which is one "node" of a sorting network.
// Min/Max are cheaper than compare + blend at least for integers.
template <SortOrder kOrder, class V>
HWY_INLINE void SortLanesIn2Vectors(V& a, V& b) {
  V temp = a;
  a = (kOrder == SortOrder::kAscending) ? Min(a, b) : Max(a, b);
  b = (kOrder == SortOrder::kAscending) ? Max(temp, b) : Min(temp, b);
}

// For the last layer of bitonic merge. Conditionally swaps lane 1 with lane 0
// and 2 with 3 within each quartet.
template <SortOrder kOrder, class D>
HWY_INLINE void SortAdjacentLanes(D d, Vec<D>& v0, Vec<D>& v1) {
  const Vec<D> tmp = v0;
  v0 = Shuffle3120(v0, v1);   // 7520
  v1 = Shuffle2031(tmp, v1);  // 6431
  SortLanesIn2Vectors<kOrder>(v0, v1);

  tmp = v0;
  v0 = InterleaveLower(d, v0, v1);
  v1 = InterleaveUpper(d, tmp, v1);
}

// (Not available for u32 because u64 comparisons are not in the API.)
template <SortOrder kOrder, size_t N, class V = Vec<Simd<int32_t, N>>>
HWY_INLINE void SortAdjacentLanes(Simd<int32_t, N> d, V& v0, V& v1) {
  const RepartitionToWide<decltype(d)> dw;
  const auto wide0 = BitCast(dw, v0);
  const auto wide1 = BitCast(dw, v1);
  const auto swap0 = BitCast(dw, Shuffle2301(v0));
  const auto swap1 = BitCast(dw, Shuffle2301(v1));
  const auto mask0 =
      (kOrder == SortOrder::kAscending) ? Gt(swap0, wide0) : Lt(swap0, wide0);
  const auto mask1 =
      (kOrder == SortOrder::kAscending) ? Gt(swap1, wide1) : Lt(swap1, wide1);
  v0 = BitCast(d, IfThenElse(mask0, swap0, wide0));
  v1 = BitCast(d, IfThenElse(mask1, swap1, wide1));
}

// For each lane: sorts the four values in the that lane of the four vectors.
template <SortOrder kOrder, class D, class V = Vec<D>>
HWY_INLINE void SortLanesIn4Vectors(D d, const TFromD<D>* in, V& v0, V& v1,
                                    V& v2, V& v3) {
  const size_t N = Lanes(d);

  // Bitonic and odd-even sorters both have 5 nodes. This one is from
  // http://users.telenet.be/bertdobbelaere/SorterHunter/sorting_networks.html

  // layer 1
  v0 = Load(d, in + 0 * N);
  v2 = Load(d, in + 2 * N);
  SortLanesIn2Vectors<kOrder>(v0, v2);
  v1 = Load(d, in + 1 * N);
  v3 = Load(d, in + 3 * N);
  SortLanesIn2Vectors<kOrder>(v1, v3);

  // layer 2
  SortLanesIn2Vectors<kOrder>(v0, v1);
  SortLanesIn2Vectors<kOrder>(v2, v3);

  // layer 3
  SortLanesIn2Vectors<kOrder>(v1, v2);
}

// Inputs are the result of SortLanesIn4Vectors (row-major). Their columns are
// sorted, and the output vectors consist of sorted quartets (128-bit blocks).
template <class D, class V = Vec<D>>
HWY_INLINE void Transpose4x4(D d, V& v0, V& v1, V& v2, V& v3) {
  const RepartitionToWide<decltype(d)> dw;

  // Input: first number is reg, second is lane (0 is lowest)
  // 03 02 01 00  |
  // 13 12 11 10  | columns are sorted
  // 23 22 21 20  | (in this order)
  // 33 32 31 30  V
  const V t0 = InterleaveLower(d, v0, v1);  // 11 01 10 00
  const V t1 = InterleaveLower(d, v2, v3);  // 31 21 30 20
  const V t2 = InterleaveUpper(d, v0, v1);  // 13 03 12 02
  const V t3 = InterleaveUpper(d, v2, v3);  // 33 23 32 22

  // 30 20 10 00
  v0 = BitCast(d, InterleaveLower(BitCast(dw, t0), BitCast(dw, t1)));
  // 31 21 11 01
  v1 = BitCast(d, InterleaveUpper(BitCast(dw, t0), BitCast(dw, t1)));
  // 32 22 12 02
  v2 = BitCast(d, InterleaveLower(BitCast(dw, t2), BitCast(dw, t3)));
  // 33 23 13 03 --> sorted in descending order (03=smallest in lane 0).
  v3 = BitCast(d, InterleaveUpper(BitCast(dw, t2), BitCast(dw, t3)));
}

#if HWY_SORT_VERIFY

template <typename T>
HWY_INLINE bool IsBitonic(const T* p, size_t N) {
  bool is_asc = true;
  bool is_desc = true;
  bool is_zero = true;
  for (size_t i = 0; i < N / 2 - 1; ++i) {
    is_asc &= (p[i] <= p[i + 1]);
    is_desc &= (p[i] >= p[i + 1]);
  }
  for (size_t i = 0; i < N; ++i) {
    is_zero &= (p[i] == 0);
  }

  bool is_asc2 = true;
  bool is_desc2 = true;
  for (size_t i = N / 2; i < N - 1; ++i) {
    is_asc2 &= (p[i] <= p[i + 1]);
    is_desc2 &= (p[i] >= p[i + 1]);
  }

  if (is_zero) return true;
  if (is_asc && is_desc2) return true;
  if (is_desc && is_asc2) return true;
  return false;
}

template <typename T>
HWY_INLINE void CheckBitonic(const T* p, size_t N, int line, int caller) {
  if (IsBitonic(p, N)) return;
  for (size_t i = 0; i < N; ++i) {
    printf("%.0f\n", static_cast<float>(p[i]));
  }
  printf("caller %d\n", caller);
  hwy::Abort("", line, "not bitonic");
}

template <SortOrder kOrder, typename T>
HWY_INLINE void CheckSorted(const T* input, const T* p, size_t N, int line,
                            int caller) {
  for (size_t i = 0; i < N - 1; ++i) {
    if (!Compare<kOrder>(p[i], p[i + 1])) {
      printf("N=%zu order=%d\n", N, static_cast<int>(kOrder));
      for (size_t i = 0; i < N; ++i) {
        printf("%.0f  %.0f\n", static_cast<float>(input[i]),
               static_cast<float>(p[i]));
      }
      printf("caller %d\n", caller);
      hwy::Abort("", line, "not sorted");
    }
  }
}

#endif

// Precondition: v0 and v1 are already sorted according to kOrder.
// Postcondition: concatenate(v0, v1) is sorted and v0 is the lower half.
template <SortOrder kOrder, class D, class V = Vec<D>>
HWY_INLINE void SortedMergePlus4(D d, V& v0, V& v1, int caller) {
#if HWY_SORT_VERIFY
  TFromD<D> input[8];
  TFromD<D> lanes[8];
  Store(v0, d, input + 0 * 4);
  Store(v1, d, input + 1 * 4);
  CheckSorted<kOrder>(input, input, 4, __LINE__, caller);
  CheckSorted<kOrder>(input + 4, input + 4, 4, __LINE__, caller);
#endif

  // See figure 5 from https://www.vldb.org/pvldb/vol8/p1274-inoue.pdf.
  // This requires 8 min/max vs 6 for bitonic merge (see Figure 2 in
  // https://dl.acm.org/doi/10.14778/1454159.1454171), but is faster overall
  // because it needs less shuffling, and does not need a bitonic input.
  SortLanesIn2Vectors<kOrder>(v0, v1);
  v0 = Shuffle0321(v0);
  SortLanesIn2Vectors<kOrder>(v0, v1);
  v0 = Shuffle0321(v0);
  SortLanesIn2Vectors<kOrder>(v0, v1);
  v0 = Shuffle0321(v0);
  SortLanesIn2Vectors<kOrder>(v0, v1);
  v0 = Shuffle0321(v0);

#if HWY_SORT_VERIFY
  Store(v0, d, lanes + 0 * 4);
  Store(v1, d, lanes + 1 * 4);
  CheckSorted<kOrder>(input, lanes, 8, __LINE__, caller);
#endif
}

// 14 ops. TODO(janwas): faster final layer?
template <SortOrder kOrder, class D, class V = Vec<D>>
HWY_INLINE void BitonicMerge4Plus4(D d, V& v0, V& v1, int caller) {
#if HWY_SORT_VERIFY
  TFromD<D> input[8];
  TFromD<D> lanes[8];
  Store(v0, d, lanes + 0 * 4);
  Store(v1, d, lanes + 1 * 4);
  Store(v0, d, input + 0 * 4);
  Store(v1, d, input + 1 * 4);
  if (caller == -1) CheckBitonic(lanes, 8, __LINE__, __LINE__);
#endif

  // Layer 1
  SortLanesIn2Vectors<kOrder>(v0, v1);

  // Layer 2
  const RepartitionToWide<decltype(d)> dw;
  V L0 = BitCast(d, InterleaveLower(dw, BitCast(dw, v0), BitCast(dw, v1)));
  V H0 = BitCast(d, InterleaveUpper(dw, BitCast(dw, v0), BitCast(dw, v1)));
  SortLanesIn2Vectors<kOrder>(L0, H0);

  // Layer 3
  V tmp = L0;
  L0 = InterleaveEven(d, H0, L0);
  H0 = InterleaveOdd(d, H0, tmp);
  SortLanesIn2Vectors<kOrder>(L0, H0);

  v0 = InterleaveLower(d, L0, H0);
  v1 = InterleaveUpper(d, L0, H0);

#if HWY_SORT_VERIFY
  Store(v0, d, lanes + 0 * 4);
  Store(v1, d, lanes + 1 * 4);
  CheckSorted<kOrder>(input, lanes, 8, __LINE__, caller);
#endif
}

// 32 ops, more efficient than three 4+4 merges (36 ops).
template <SortOrder kOrder, class D, class V = Vec<D>>
HWY_INLINE void BitonicMergeSorted8Plus8(D d, V& v0, V& v1, V& v2, V& v3,
                                         int caller) {
#if HWY_SORT_VERIFY
  TFromD<D> input[16];
  TFromD<D> lanes[16];
  Store(v0, d, input + 0 * 4);
  Store(v1, d, input + 1 * 4);
  Store(v2, d, input + 2 * 4);
  Store(v3, d, input + 3 * 4);
  Store(v0, d, lanes + 0 * 4);
  Store(v1, d, lanes + 1 * 4);
  Store(v2, d, lanes + 2 * 4);
  Store(v3, d, lanes + 3 * 4);
  if (caller == -1) CheckBitonic(lanes, 16, __LINE__, __LINE__);
#endif

  // Layer 1: lane stride 8
  SortLanesIn2Vectors<kOrder>(v0, v2);
  SortLanesIn2Vectors<kOrder>(v1, v3);

  // Layers 2 to 4
  // Inputs are not fully sorted, so cannot use SortedMergePlus4.
  BitonicMerge4Plus4<kOrder>(d, v0, v1, __LINE__);
  BitonicMerge4Plus4<kOrder>(d, v2, v3, __LINE__);

#if HWY_SORT_VERIFY
  Store(v0, d, lanes + 0 * 4);
  Store(v1, d, lanes + 1 * 4);
  Store(v2, d, lanes + 2 * 4);
  Store(v3, d, lanes + 3 * 4);
  CheckSorted<kOrder>(input, lanes, 16, __LINE__, caller);
#endif
}

// Bitonic merge, quartets in concatenate(v0, v1, v2, v3) and concatenate(v4,
// v5, v6, v7) must be sorted in opposite orders.
template <SortOrder kOrder, class D, class V = Vec<D>>
HWY_INLINE void BitonicMergeSorted16Plus16(D d, V& v0, V& v1, V& v2, V& v3,
                                           V& v4, V& v5, V& v6, V& v7) {
#if HWY_SORT_VERIFY
  TFromD<D> input[32];
  TFromD<D> lanes[32];
  Store(v0, d, input + 0 * 4);
  Store(v1, d, input + 1 * 4);
  Store(v2, d, input + 2 * 4);
  Store(v3, d, input + 3 * 4);
  Store(v4, d, input + 4 * 4);
  Store(v5, d, input + 5 * 4);
  Store(v6, d, input + 6 * 4);
  Store(v7, d, input + 7 * 4);
  Store(v0, d, lanes + 0 * 4);
  Store(v1, d, lanes + 1 * 4);
  Store(v2, d, lanes + 2 * 4);
  Store(v3, d, lanes + 3 * 4);
  Store(v4, d, lanes + 4 * 4);
  Store(v5, d, lanes + 5 * 4);
  Store(v6, d, lanes + 6 * 4);
  Store(v7, d, lanes + 7 * 4);
  CheckBitonic(lanes, 32, __LINE__, __LINE__);
#endif

  // Layer 1
  SortLanesIn2Vectors<kOrder>(v0, v4);
  SortLanesIn2Vectors<kOrder>(v1, v5);
  SortLanesIn2Vectors<kOrder>(v2, v6);
  SortLanesIn2Vectors<kOrder>(v3, v7);

  // Layers 2 to 5
  BitonicMergeSorted8Plus8<kOrder>(d, v0, v1, v2, v3, __LINE__);
  BitonicMergeSorted8Plus8<kOrder>(d, v4, v5, v6, v7, __LINE__);

#if HWY_SORT_VERIFY
  Store(v0, d, lanes + 0 * 4);
  Store(v1, d, lanes + 1 * 4);
  Store(v2, d, lanes + 2 * 4);
  Store(v3, d, lanes + 3 * 4);
  Store(v4, d, lanes + 4 * 4);
  Store(v5, d, lanes + 5 * 4);
  Store(v6, d, lanes + 6 * 4);
  Store(v7, d, lanes + 7 * 4);
  CheckSorted<kOrder>(input, lanes, 32, __LINE__, __LINE__);
#endif
}

}  // namespace detail

template <SortOrder kOrder, class D>
HWY_API void Sort8Vectors(D d, TFromD<D>* inout) {
  const size_t N = Lanes(d);

  Vec<D> v0, v1, v2, v3;
  detail::SortLanesIn4Vectors<kOrder>(d, inout, v0, v1, v2, v3);
  detail::Transpose4x4(d, v0, v1, v2, v3);
  detail::SortedMergePlus4<kOrder>(d, v0, v1, -1);
  detail::SortedMergePlus4<kOrder>(d, v2, v3, -1);

  // Bitonic merges require one input to be in reverse order.
  constexpr SortOrder kReverse = (kOrder == SortOrder::kAscending)
                                     ? SortOrder::kDescending
                                     : SortOrder::kAscending;

  Vec<D> v4, v5, v6, v7;
  detail::SortLanesIn4Vectors<kReverse>(d, inout + 4 * N, v4, v5, v6, v7);
  detail::Transpose4x4(d, v4, v5, v6, v7);
  detail::SortedMergePlus4<kReverse>(d, v4, v5, -1);
  detail::SortedMergePlus4<kReverse>(d, v6, v7, -1);

  detail::BitonicMergeSorted8Plus8<kOrder>(d, v0, v1, v4, v5, -1);
  detail::BitonicMergeSorted8Plus8<kReverse>(d, v2, v3, v6, v7, -1);

  detail::BitonicMergeSorted16Plus16<kOrder>(d, v0, v1, v4, v5, v2, v3, v6, v7);

  // TODO(janwas): in-reg transpose of 128-bit blocks?
  // TODO(janwas): or multi-way merge of all the 128-bit blocks?

#if HWY_MAX_BYTES == 16 || 1
  // Definitely only a single quartet per register - store octets.
  Store(v0, d, inout + 0 * N);
  Store(v1, d, inout + 1 * N);
  Store(v4, d, inout + 2 * N);
  Store(v5, d, inout + 3 * N);
  Store(v2, d, inout + 4 * N);
  Store(v3, d, inout + 5 * N);
  Store(v6, d, inout + 6 * N);
  Store(v7, d, inout + 7 * N);
#else
  alignas(64) TFromD<D> tmp[8 * 16];  // TODO(janwas): allocate
  Store(v0, d, tmp + 0 * N);
  Store(v1, d, tmp + 1 * N);
  Store(v4, d, tmp + 2 * N);
  Store(v5, d, tmp + 3 * N);
  Store(v2, d, tmp + 4 * N);
  Store(v3, d, tmp + 5 * N);
  Store(v6, d, tmp + 6 * N);
  Store(v7, d, tmp + 7 * N);

  const CappedTag<TFromD<D>, 4> d4;
  for (size_t pair = 0; pair < 8 * N; pair += 2 * N) {
    for (size_t i = 0; i < N; i += 4) {
      Store(Load(d4, tmp + pair + i + 0), d4, inout);
      inout += 4;
      Store(Load(d4, tmp + pair + i + N), d4, inout);
      inout += 4;
    }
  }
#endif  // HWY_MAX_BYTES == 16
}

#endif  // HWY_TARGET != HWY_SCALAR

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_SORT_SORT_INL_H_
