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

#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

#if HWY_TARGET != HWY_SCALAR && HWY_ARCH_X86

#define HWY_SORT_VERIFY 1

enum class SortOrder { kAscending, kDescending };

template <typename T>
bool Compare(T a, T b, SortOrder kOrder) {
  if (kOrder == SortOrder::kAscending) return a <= b;
  return a >= b;
}

namespace detail {

// TODO(janwas): move into op
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

#if HWY_TARGET <= HWY_AVX2
HWY_API Vec256<uint32_t> Shuffle3120(Vec256<uint32_t> lo, Vec256<uint32_t> hi) {
  const Full256<uint32_t> d;
  const RebindToFloat<decltype(d)> df;
  constexpr int kShuffle = _MM_SHUFFLE(3, 1, 2, 0);
  return BitCast(d, Vec256<float>{_mm256_shuffle_ps(
                        BitCast(df, lo).raw, BitCast(df, hi).raw, kShuffle)});
}
HWY_API Vec256<int32_t> Shuffle3120(Vec256<int32_t> lo, Vec256<int32_t> hi) {
  const Full256<int32_t> d;
  const RebindToFloat<decltype(d)> df;
  constexpr int kShuffle = _MM_SHUFFLE(3, 1, 2, 0);
  return BitCast(d, Vec256<float>{_mm256_shuffle_ps(
                        BitCast(df, lo).raw, BitCast(df, hi).raw, kShuffle)});
}

HWY_API Vec256<uint32_t> Shuffle2031(Vec256<uint32_t> lo, Vec256<uint32_t> hi) {
  const Full256<uint32_t> d;
  const RebindToFloat<decltype(d)> df;
  constexpr int kShuffle = _MM_SHUFFLE(2, 0, 3, 1);
  return BitCast(d, Vec256<float>{_mm256_shuffle_ps(
                        BitCast(df, lo).raw, BitCast(df, hi).raw, kShuffle)});
}
HWY_API Vec256<int32_t> Shuffle2031(Vec256<int32_t> lo, Vec256<int32_t> hi) {
  const Full256<int32_t> d;
  const RebindToFloat<decltype(d)> df;
  constexpr int kShuffle = _MM_SHUFFLE(2, 0, 3, 1);
  return BitCast(d, Vec256<float>{_mm256_shuffle_ps(
                        BitCast(df, lo).raw, BitCast(df, hi).raw, kShuffle)});
}
#endif  // HWY_TARGET <= HWY_AVX2

#if HWY_TARGET <= HWY_AVX3
HWY_API Vec512<uint32_t> Shuffle3120(Vec512<uint32_t> lo, Vec512<uint32_t> hi) {
  const Full512<uint32_t> d;
  const RebindToFloat<decltype(d)> df;
  constexpr int kShuffle = _MM_SHUFFLE(3, 1, 2, 0);
  return BitCast(d, Vec512<float>{_mm512_shuffle_ps(
                        BitCast(df, lo).raw, BitCast(df, hi).raw, kShuffle)});
}
HWY_API Vec512<int32_t> Shuffle3120(Vec512<int32_t> lo, Vec512<int32_t> hi) {
  const Full512<int32_t> d;
  const RebindToFloat<decltype(d)> df;
  constexpr int kShuffle = _MM_SHUFFLE(3, 1, 2, 0);
  return BitCast(d, Vec512<float>{_mm512_shuffle_ps(
                        BitCast(df, lo).raw, BitCast(df, hi).raw, kShuffle)});
}

HWY_API Vec512<uint32_t> Shuffle2031(Vec512<uint32_t> lo, Vec512<uint32_t> hi) {
  const Full512<uint32_t> d;
  const RebindToFloat<decltype(d)> df;
  constexpr int kShuffle = _MM_SHUFFLE(2, 0, 3, 1);
  return BitCast(d, Vec512<float>{_mm512_shuffle_ps(
                        BitCast(df, lo).raw, BitCast(df, hi).raw, kShuffle)});
}
HWY_API Vec512<int32_t> Shuffle2031(Vec512<int32_t> lo, Vec512<int32_t> hi) {
  const Full512<int32_t> d;
  const RebindToFloat<decltype(d)> df;
  constexpr int kShuffle = _MM_SHUFFLE(2, 0, 3, 1);
  return BitCast(d, Vec512<float>{_mm512_shuffle_ps(
                        BitCast(df, lo).raw, BitCast(df, hi).raw, kShuffle)});
}
#endif  // HWY_TARGET <= HWY_AVX3

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
  Vec<D> tmp = v0;
  v0 = Shuffle3120(v0, v1);   // 7520
  v1 = Shuffle2031(tmp, v1);  // 6431
  SortLanesIn2Vectors<kOrder>(v0, v1);

  tmp = v0;
  v0 = InterleaveLower(d, v0, v1);
  v1 = InterleaveUpper(d, tmp, v1);
}

// 4 ops. (Not available for u32 because u64 comparisons are not in the API.)
template <SortOrder kOrder, size_t N, class V = Vec<Simd<int32_t, N>>>
HWY_INLINE void SortAdjacentLanes(Simd<int32_t, N> d, V& v0, V& v1) {
  const RepartitionToWide<decltype(d)> dw;
  const auto wide0 = BitCast(dw, v0);
  const auto wide1 = BitCast(dw, v1);
  const auto swap0 = BitCast(dw, Shuffle2301(v0));
  const auto swap1 = BitCast(dw, Shuffle2301(v1));
  if (kOrder == SortOrder::kAscending) {
    v0 = BitCast(d, Max(wide0, swap0));
    v1 = BitCast(d, Max(wide1, swap1));
  } else {
    v0 = BitCast(d, Min(wide0, swap0));
    v1 = BitCast(d, Min(wide1, swap1));
  }
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

template <class D>
class Runs {
  using T = TFromD<D>;

 public:
  Runs(D d, size_t num_regs) {
    const size_t N = Lanes(d);

    buf_ = AllocateAligned<T>(N);
    consecutive_ = AllocateAligned<T>(num_regs * N);

    num_regs_ = num_regs;
    run_length_ = num_regs * 4;
    num_runs_ = N / 4;
  }

  void ScatterQuartets(D d, const size_t idx_reg, Vec<D> v) {
    HWY_ASSERT(idx_reg < num_regs_);
    const size_t N = Lanes(d);
    for (size_t i = 0; i < N; i += 4) {
      Store(v, d, buf_.get());
      const size_t idx_q = (i / 4) * num_regs_ + idx_reg;
      CopyBytes<16>(buf_.get() + i, consecutive_.get() + idx_q * 4);
    }
  }

  bool IsBitonic() const {
    for (size_t ir = 0; ir < num_runs_; ++ir) {
      const T* p = &consecutive_[ir * run_length_];
      bool is_asc = true;
      bool is_desc = true;
      bool is_zero = true;

      for (size_t i = 0; i < run_length_ / 2 - 1; ++i) {
        is_asc &= (p[i] <= p[i + 1]);
        is_desc &= (p[i] >= p[i + 1]);
      }
      for (size_t i = 0; i < run_length_; ++i) {
        is_zero &= (p[i] == 0);
      }

      bool is_asc2 = true;
      bool is_desc2 = true;
      for (size_t i = run_length_ / 2; i < run_length_ - 1; ++i) {
        is_asc2 &= (p[i] <= p[i + 1]);
        is_desc2 &= (p[i] >= p[i + 1]);
      }

      if (is_zero) continue;
      if (is_asc && is_desc2) continue;
      if (is_desc && is_asc2) continue;
      return false;
    }
    return true;
  }

  void CheckBitonic(int line, int caller) const {
    if (IsBitonic()) return;
    for (size_t ir = 0; ir < num_runs_; ++ir) {
      const T* p = &consecutive_[ir * run_length_];
      printf("run %zu (len %zu)\n", ir, run_length_);
      for (size_t i = 0; i < run_length_; ++i) {
        printf("%.0f\n", static_cast<float>(p[i]));
      }
    }
    printf("caller %d\n", caller);
    hwy::Abort("", line, "not bitonic");
  }

  void CheckSorted(SortOrder kOrder, int line, int caller) const {
    for (size_t ir = 0; ir < num_runs_; ++ir) {
      const T* p = &consecutive_[ir * run_length_];

      for (size_t i = 0; i < run_length_ - 1; ++i) {
        if (!Compare(p[i], p[i + 1], kOrder)) {
          printf("ir%zu run_length=%zu order=%d\n", ir, run_length_,
                 static_cast<int>(kOrder));
          for (size_t i = 0; i < run_length_; ++i) {
            printf(" %.0f\n", static_cast<float>(p[i]));
          }
          printf("caller %d\n", caller);
          hwy::Abort("", line, "not sorted");
        }
      }
    }
  }

 private:
  AlignedFreeUniquePtr<T[]> buf_;
  AlignedFreeUniquePtr<T[]> consecutive_;
  size_t num_regs_;
  size_t run_length_;
  size_t num_runs_;
};

template <class D>
Runs<D> StoreDeinterleavedQuartets(D d, Vec<D> v0) {
  Runs runs(d, 1);
  runs.ScatterQuartets(d, 0, v0);
  return runs;
}

template <class D>
Runs<D> StoreDeinterleavedQuartets(D d, Vec<D> v0, Vec<D> v1) {
  Runs runs(d, 2);
  runs.ScatterQuartets(d, 0, v0);
  runs.ScatterQuartets(d, 1, v1);
  return runs;
}

template <class D>
Runs<D> StoreDeinterleavedQuartets(D d, Vec<D> v0, Vec<D> v1, Vec<D> v2,
                                   Vec<D> v3) {
  Runs runs(d, 4);
  runs.ScatterQuartets(d, 0, v0);
  runs.ScatterQuartets(d, 1, v1);
  runs.ScatterQuartets(d, 2, v2);
  runs.ScatterQuartets(d, 3, v3);
  return runs;
}

template <class D>
Runs<D> StoreDeinterleavedQuartets(D d, Vec<D> v0, Vec<D> v1, Vec<D> v2,
                                   Vec<D> v3, Vec<D> v4, Vec<D> v5, Vec<D> v6,
                                   Vec<D> v7) {
  Runs runs(d, 8);
  runs.ScatterQuartets(d, 0, v0);
  runs.ScatterQuartets(d, 1, v1);
  runs.ScatterQuartets(d, 2, v2);
  runs.ScatterQuartets(d, 3, v3);
  runs.ScatterQuartets(d, 4, v4);
  runs.ScatterQuartets(d, 5, v5);
  runs.ScatterQuartets(d, 6, v6);
  runs.ScatterQuartets(d, 7, v7);
  return runs;
}

template <class D>
Runs<D> StoreDeinterleavedQuartets(D d, Vec<D> v0, Vec<D> v1, Vec<D> v2,
                                   Vec<D> v3, Vec<D> v4, Vec<D> v5, Vec<D> v6,
                                   Vec<D> v7, Vec<D> v8, Vec<D> v9, Vec<D> vA,
                                   Vec<D> vB, Vec<D> vC, Vec<D> vD, Vec<D> vE,
                                   Vec<D> vF) {
  Runs runs(d, 16);
  runs.ScatterQuartets(d, 0x0, v0);
  runs.ScatterQuartets(d, 0x1, v1);
  runs.ScatterQuartets(d, 0x2, v2);
  runs.ScatterQuartets(d, 0x3, v3);
  runs.ScatterQuartets(d, 0x4, v4);
  runs.ScatterQuartets(d, 0x5, v5);
  runs.ScatterQuartets(d, 0x6, v6);
  runs.ScatterQuartets(d, 0x7, v7);
  runs.ScatterQuartets(d, 0x8, v8);
  runs.ScatterQuartets(d, 0x9, v9);
  runs.ScatterQuartets(d, 0xA, vA);
  runs.ScatterQuartets(d, 0xB, vB);
  runs.ScatterQuartets(d, 0xC, vC);
  runs.ScatterQuartets(d, 0xD, vD);
  runs.ScatterQuartets(d, 0xE, vE);
  runs.ScatterQuartets(d, 0xF, vF);
  return runs;
}

template <class D>
Runs<D> StoreDeinterleavedQuartets(
    D d, const Vec<D>& v00, const Vec<D>& v01, const Vec<D>& v02,
    const Vec<D>& v03, const Vec<D>& v04, const Vec<D>& v05, const Vec<D>& v06,
    const Vec<D>& v07, const Vec<D>& v08, const Vec<D>& v09, const Vec<D>& v0A,
    const Vec<D>& v0B, const Vec<D>& v0C, const Vec<D>& v0D, const Vec<D>& v0E,
    const Vec<D>& v0F, const Vec<D>& v10, const Vec<D>& v11, const Vec<D>& v12,
    const Vec<D>& v13, const Vec<D>& v14, const Vec<D>& v15, const Vec<D>& v16,
    const Vec<D>& v17, const Vec<D>& v18, const Vec<D>& v19, const Vec<D>& v1A,
    const Vec<D>& v1B, const Vec<D>& v1C, const Vec<D>& v1D, const Vec<D>& v1E,
    const Vec<D>& v1F) {
  Runs runs(d, 32);
  runs.ScatterQuartets(d, 0x00, v00);
  runs.ScatterQuartets(d, 0x01, v01);
  runs.ScatterQuartets(d, 0x02, v02);
  runs.ScatterQuartets(d, 0x03, v03);
  runs.ScatterQuartets(d, 0x04, v04);
  runs.ScatterQuartets(d, 0x05, v05);
  runs.ScatterQuartets(d, 0x06, v06);
  runs.ScatterQuartets(d, 0x07, v07);
  runs.ScatterQuartets(d, 0x08, v08);
  runs.ScatterQuartets(d, 0x09, v09);
  runs.ScatterQuartets(d, 0x0A, v0A);
  runs.ScatterQuartets(d, 0x0B, v0B);
  runs.ScatterQuartets(d, 0x0C, v0C);
  runs.ScatterQuartets(d, 0x0D, v0D);
  runs.ScatterQuartets(d, 0x0E, v0E);
  runs.ScatterQuartets(d, 0x0F, v0F);
  runs.ScatterQuartets(d, 0x10, v10);
  runs.ScatterQuartets(d, 0x11, v11);
  runs.ScatterQuartets(d, 0x12, v12);
  runs.ScatterQuartets(d, 0x13, v13);
  runs.ScatterQuartets(d, 0x14, v14);
  runs.ScatterQuartets(d, 0x15, v15);
  runs.ScatterQuartets(d, 0x16, v16);
  runs.ScatterQuartets(d, 0x17, v17);
  runs.ScatterQuartets(d, 0x18, v18);
  runs.ScatterQuartets(d, 0x19, v19);
  runs.ScatterQuartets(d, 0x1A, v1A);
  runs.ScatterQuartets(d, 0x1B, v1B);
  runs.ScatterQuartets(d, 0x1C, v1C);
  runs.ScatterQuartets(d, 0x1D, v1D);
  runs.ScatterQuartets(d, 0x1E, v1E);
  runs.ScatterQuartets(d, 0x1F, v1F);
  return runs;
}

#endif

// 12 ops (including 4 swizzle)
// Precondition: v0 and v1 are already sorted according to kOrder.
// Postcondition: concatenate(v0, v1) is sorted and v0 is the lower half.
template <SortOrder kOrder, class D, class V = Vec<D>>
HWY_INLINE void SortedMergePlus4(D d, V& v0, V& v1, int caller) {
#if HWY_SORT_VERIFY
  const Runs<D> input0 = StoreDeinterleavedQuartets(d, v0);
  const Runs<D> input1 = StoreDeinterleavedQuartets(d, v1);
  input0.CheckSorted(kOrder, __LINE__, caller);
  input1.CheckSorted(kOrder, __LINE__, caller);
#endif

  // See figure 5 from https://www.vldb.org/pvldb/vol8/p1274-inoue.pdf.
  // This requires 8 min/max vs 6 for bitonic merge (see Figure 2 in
  // http://www.vldb.org/pvldb/vol1/1454171.pdf), but is faster overall because
  // it needs less shuffling, and does not need a bitonic input.
  SortLanesIn2Vectors<kOrder>(v0, v1);
  v0 = Shuffle0321(v0);
  SortLanesIn2Vectors<kOrder>(v0, v1);
  v0 = Shuffle0321(v0);
  SortLanesIn2Vectors<kOrder>(v0, v1);
  v0 = Shuffle0321(v0);
  SortLanesIn2Vectors<kOrder>(v0, v1);
  v0 = Shuffle0321(v0);

#if HWY_SORT_VERIFY
  auto output = StoreDeinterleavedQuartets(d, v0, v1);
  output.CheckSorted(kOrder, __LINE__, caller);
#endif
}

// 12 ops (including 6 swizzle)
template <SortOrder kOrder, class D, class V = Vec<D>>
HWY_INLINE void BitonicMerge4Plus4(D d, V& v0, V& v1, int caller) {
#if HWY_SORT_VERIFY
  const Runs<D> input = StoreDeinterleavedQuartets(d, v0, v1);
  if (caller == -1) input.CheckBitonic(__LINE__, __LINE__);
#endif

  // Layer 1: lane stride 4 (2 ops)
  SortLanesIn2Vectors<kOrder>(v0, v1);

  // Layer 2: lane stride 2 (6 ops)
  const RepartitionToWide<decltype(d)> dw;
  V tmp = v0;
  // 5410
  v0 = BitCast(d, InterleaveLower(dw, BitCast(dw, v0), BitCast(dw, v1)));
  // 7632
  v1 = BitCast(d, InterleaveUpper(dw, BitCast(dw, tmp), BitCast(dw, v1)));
  SortLanesIn2Vectors<kOrder>(v0, v1);
  // v0, v1: M75 M64 M31 M20, m75 m64 m31 m20
  // Output: M75 M64 m75 m64 M31 M20 m31 m20
  tmp = v0;
  v0 = BitCast(d, InterleaveLower(d, BitCast(dw, v0), BitCast(dw, v1)));
  v1 = BitCast(d, InterleaveUpper(d, BitCast(dw, tmp), BitCast(dw, v1)));

  // Layer 3: lane stride 1 (4 ops)
  SortAdjacentLanes<kOrder>(d, v0, v1);

#if HWY_SORT_VERIFY
  const Runs<D> output = StoreDeinterleavedQuartets(d, v0, v1);
  output.CheckSorted(kOrder, __LINE__, caller);
#endif
}

// 28 ops, more efficient than three 4+4 merges (36 ops).
template <SortOrder kOrder, class D, class V = Vec<D>>
HWY_INLINE void BitonicMergeSorted8Plus8(D d, V& v0, V& v1, V& v2, V& v3,
                                         int caller) {
#if HWY_SORT_VERIFY
  const Runs<D> input = StoreDeinterleavedQuartets(d, v0, v1, v2, v3);
  if (caller == -1) input.CheckBitonic(__LINE__, __LINE__);
#endif

  // Layer 1: lane stride 8
  SortLanesIn2Vectors<kOrder>(v0, v2);
  SortLanesIn2Vectors<kOrder>(v1, v3);

  // Layers 2 to 4
  // Inputs are not fully sorted, so cannot use SortedMergePlus4.
  BitonicMerge4Plus4<kOrder>(d, v0, v1, __LINE__);
  BitonicMerge4Plus4<kOrder>(d, v2, v3, __LINE__);

#if HWY_SORT_VERIFY
  const Runs<D> output = StoreDeinterleavedQuartets(d, v0, v1, v2, v3);
  output.CheckSorted(kOrder, __LINE__, caller);
#endif
}

// 64 ops. concatenate(v0, v1, v2, v3) and concatenate(v4, v5, v6, v7) must be
// sorted in opposite orders.
template <SortOrder kOrder, class D, class V = Vec<D>>
HWY_INLINE void BitonicMergeSorted16Plus16(D d, V& v0, V& v1, V& v2, V& v3,
                                           V& v4, V& v5, V& v6, V& v7,
                                           int caller) {
#if HWY_SORT_VERIFY
  const Runs<D> input =
      StoreDeinterleavedQuartets(d, v0, v1, v2, v3, v4, v5, v6, v7);
  if (caller == -1) input.CheckBitonic(__LINE__, __LINE__);
#endif

  // Layer 1: lane stride 16
  SortLanesIn2Vectors<kOrder>(v0, v4);
  SortLanesIn2Vectors<kOrder>(v1, v5);
  SortLanesIn2Vectors<kOrder>(v2, v6);
  SortLanesIn2Vectors<kOrder>(v3, v7);

  // Layers 2 to 5
  BitonicMergeSorted8Plus8<kOrder>(d, v0, v1, v2, v3, __LINE__);
  BitonicMergeSorted8Plus8<kOrder>(d, v4, v5, v6, v7, __LINE__);

#if HWY_SORT_VERIFY
  const Runs<D> output =
      StoreDeinterleavedQuartets(d, v0, v1, v2, v3, v4, v5, v6, v7);
  output.CheckSorted(kOrder, __LINE__, caller);
#endif
}

// concatenate(v0..7) and concatenate(q8..F) must be sorted in opposite orders.
template <SortOrder kOrder, class D, class V = Vec<D>>
HWY_INLINE void BitonicMergeSorted32Plus32(D d, V& v0, V& v1, V& v2, V& v3,
                                           V& v4, V& v5, V& v6, V& v7, V& q8,
                                           V& q9, V& qA, V& qB, V& qC, V& qD,
                                           V& qE, V& qF, int caller) {
#if HWY_SORT_VERIFY
  const Runs<D> input = StoreDeinterleavedQuartets(
      d, v0, v1, v2, v3, v4, v5, v6, v7, q8, q9, qA, qB, qC, qD, qE, qF);
  if (caller == -1) input.CheckBitonic(__LINE__, __LINE__);
#endif

  // Layer 1: lane stride 32
  SortLanesIn2Vectors<kOrder>(v0, q8);
  SortLanesIn2Vectors<kOrder>(v1, q9);
  SortLanesIn2Vectors<kOrder>(v2, qA);
  SortLanesIn2Vectors<kOrder>(v3, qB);
  SortLanesIn2Vectors<kOrder>(v4, qC);
  SortLanesIn2Vectors<kOrder>(v5, qD);
  SortLanesIn2Vectors<kOrder>(v6, qE);
  SortLanesIn2Vectors<kOrder>(v7, qF);

  // Layers 2 to 6
  BitonicMergeSorted16Plus16<kOrder>(d, v0, v1, v2, v3, v4, v5, v6, v7,
                                     __LINE__);
  BitonicMergeSorted16Plus16<kOrder>(d, q8, q9, qA, qB, qC, qD, qE, qF,
                                     __LINE__);

#if HWY_SORT_VERIFY
  const Runs<D> output = StoreDeinterleavedQuartets(
      d, v0, v1, v2, v3, v4, v5, v6, v7, q8, q9, qA, qB, qC, qD, qE, qF);
  output.CheckSorted(kOrder, __LINE__, caller);
#endif
}

// concatenate(v0..7) and concatenate(v8..F) must be sorted in opposite orders.
template <SortOrder kOrder, class D, class V = Vec<D>>
HWY_INLINE void BitonicMergeSorted64Plus64(
    D d, V& v00, V& v01, V& v02, V& v03, V& v04, V& v05, V& v06, V& v07, V& v08,
    V& v09, V& v0A, V& v0B, V& v0C, V& v0D, V& v0E, V& v0F, V& q10, V& q11,
    V& q12, V& q13, V& q14, V& q15, V& q16, V& q17, V& q18, V& q19, V& q1A,
    V& q1B, V& q1C, V& q1D, V& q1E, V& q1F, int caller) {
#if HWY_SORT_VERIFY
  const Runs<D> input = StoreDeinterleavedQuartets(
      d, v00, v01, v02, v03, v04, v05, v06, v07, v08, v09, v0A, v0B, v0C, v0D,
      v0E, v0F, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19, q1A, q1B, q1C,
      q1D, q1E, q1F);
  if (caller == -1) input.CheckBitonic(__LINE__, __LINE__);
#endif

  // Layer 1: lane stride 64
  SortLanesIn2Vectors<kOrder>(v00, q10);
  SortLanesIn2Vectors<kOrder>(v01, q11);
  SortLanesIn2Vectors<kOrder>(v02, q12);
  SortLanesIn2Vectors<kOrder>(v03, q13);
  SortLanesIn2Vectors<kOrder>(v04, q14);
  SortLanesIn2Vectors<kOrder>(v05, q15);
  SortLanesIn2Vectors<kOrder>(v06, q16);
  SortLanesIn2Vectors<kOrder>(v07, q17);
  SortLanesIn2Vectors<kOrder>(v08, q18);
  SortLanesIn2Vectors<kOrder>(v09, q19);
  SortLanesIn2Vectors<kOrder>(v0A, q1A);
  SortLanesIn2Vectors<kOrder>(v0B, q1B);
  SortLanesIn2Vectors<kOrder>(v0C, q1C);
  SortLanesIn2Vectors<kOrder>(v0D, q1D);
  SortLanesIn2Vectors<kOrder>(v0E, q1E);
  SortLanesIn2Vectors<kOrder>(v0F, q1F);

  // Layers 2 to 7
  BitonicMergeSorted32Plus32<kOrder>(d, v00, v01, v02, v03, v04, v05, v06, v07,
                                     v08, v09, v0A, v0B, v0C, v0D, v0E, v0F,
                                     __LINE__);
  BitonicMergeSorted32Plus32<kOrder>(d, q10, q11, q12, q13, q14, q15, q16, q17,
                                     q18, q19, q1A, q1B, q1C, q1D, q1E, q1F,
                                     __LINE__);

#if HWY_SORT_VERIFY
  const Runs<D> output = StoreDeinterleavedQuartets(
      d, v00, v01, v02, v03, v04, v05, v06, v07, v08, v09, v0A, v0B, v0C, v0D,
      v0E, v0F, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19, q1A, q1B, q1C,
      q1D, q1E, q1F);
  output.CheckSorted(kOrder, __LINE__, caller);
#endif
}

// TODO(janwas): pre-permute arg order at the call site
template <SortOrder kOrder, class D>
HWY_INLINE void CombineTwoHalves(D d, Vec<D>& v0, Vec<D>& v1, Vec<D>& v2,
                                 Vec<D>& v3, Vec<D>& v4, Vec<D>& v5, Vec<D>& v6,
                                 Vec<D>& v7, TFromD<D>* inout) {
  const Half<D> d4;
  using V4 = Vec<decltype(d4)>;
  V4 h0 = LowerHalf(d4, v0);
  V4 h1 = LowerHalf(d4, v1);
  V4 h2 = LowerHalf(d4, v2);
  V4 h3 = LowerHalf(d4, v3);
  V4 h4 = LowerHalf(d4, v4);
  V4 h5 = LowerHalf(d4, v5);
  V4 h6 = LowerHalf(d4, v6);
  V4 h7 = LowerHalf(d4, v7);

  V4 q8 = LowerHalf(d4, Reverse(d, v0));
  V4 q9 = LowerHalf(d4, Reverse(d, v1));
  V4 qA = LowerHalf(d4, Reverse(d, v2));
  V4 qB = LowerHalf(d4, Reverse(d, v3));
  V4 qC = LowerHalf(d4, Reverse(d, v4));
  V4 qD = LowerHalf(d4, Reverse(d, v5));
  V4 qE = LowerHalf(d4, Reverse(d, v6));
  V4 qF = LowerHalf(d4, Reverse(d, v7));

  // Second half: quartets reversed, and passed in reverse order - including
  // the permutation from BitonicMergeSorted16Plus16.
  detail::BitonicMergeSorted32Plus32<kOrder>(
      d4, h0, h1, h4, h5, h2, h3, h6, h7, qF, qE, qB, qA, qD, qC, q9, q8, -1);
  const Vec<D> h54 = Combine(d, h5, h4);
  Store(h0, d4, inout + 0x0 * 4);
  const Vec<D> h32 = Combine(d, h3, h2);
  Store(h1, d4, inout + 0x1 * 4);
  const Vec<D> h76 = Combine(d, h7, h6);
  Store(h54, d, inout + 0x2 * 4);
  const Vec<D> qEF = Combine(d, qE, qF);
  Store(h32, d, inout + 0x4 * 4);
  const Vec<D> qAB = Combine(d, qA, qB);
  Store(h76, d, inout + 0x6 * 4);
  const Vec<D> qCD = Combine(d, qC, qD);
  Store(qEF, d, inout + 0x8 * 4);
  const Vec<D> q89 = Combine(d, q8, q9);
  Store(qAB, d, inout + 0xA * 4);
  Store(qCD, d, inout + 0xC * 4);
  Store(q89, d, inout + 0xE * 4);
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

  detail::BitonicMergeSorted16Plus16<kOrder>(d, v0, v1, v4, v5, v2, v3, v6, v7,
                                             -1);

  if (N == 4) {
    // Only a single quartet per register: store in fully-sorted order.
    Store(v0, d, inout + 0 * N);
    Store(v1, d, inout + 1 * N);
    Store(v4, d, inout + 2 * N);
    Store(v5, d, inout + 3 * N);
    Store(v2, d, inout + 4 * N);
    Store(v3, d, inout + 5 * N);
    Store(v6, d, inout + 6 * N);
    Store(v7, d, inout + 7 * N);
    return;
  }

  // Will not compile for 128-bit SIMD because FixedTag<T, 4> does not match
  // LowerHalf of Vec128.
#if HWY_MAX_BYTES != 16
  if (N == 8) {
    detail::CombineTwoHalves<kOrder>(d, v0, v1, v2, v3, v4, v5, v6, v7, inout);
    return;
  }
#endif

#if HWY_MAX_BYTES == 64
  if (N == 16) {
    const FixedTag<TFromD<D>, 8> d8;
    using V8 = Vec<decltype(d8)>;
    V8 L0 = LowerHalf(d8, v0);
    V8 L1 = LowerHalf(d8, v1);
    V8 L2 = LowerHalf(d8, v4);
    V8 L3 = LowerHalf(d8, v5);
    V8 L4 = LowerHalf(d8, v2);
    V8 L5 = LowerHalf(d8, v3);
    V8 L6 = LowerHalf(d8, v6);
    V8 L7 = LowerHalf(d8, v7);

    // TODO(janwas): for AVX-512, combine into TableLookupLanes
    // Second half: quartets reversed, and passed in reverse order - including
    // the permutation from BitonicMergeSorted16Plus16.
    V8 U0 = Shuffle0123(UpperHalf(d8, v7));
    V8 U1 = Shuffle0123(UpperHalf(d8, v6));
    V8 U2 = Shuffle0123(UpperHalf(d8, v3));
    V8 U3 = Shuffle0123(UpperHalf(d8, v2));
    V8 U4 = Shuffle0123(UpperHalf(d8, v5));
    V8 U5 = Shuffle0123(UpperHalf(d8, v4));
    V8 U6 = Shuffle0123(UpperHalf(d8, v1));
    V8 U7 = Shuffle0123(UpperHalf(d8, v0));

    detail::BitonicMergeSorted32Plus32<kOrder>(
        d8, L0, L1, L2, L3, L4, L5, L6, L7, U0, U1, U2, U3, U4, U5, U6, U7, -1);

    const FixedTag<TFromD<D>, 4> d4;
    using V4 = Vec<decltype(d4)>;
    V4 X0 = LowerHalf(d4, L0);
    V4 X1 = LowerHalf(d4, L1);
    V4 X2 = LowerHalf(d4, L2);
    V4 X3 = LowerHalf(d4, L3);
    V4 X4 = LowerHalf(d4, L4);
    V4 X5 = LowerHalf(d4, L5);
    V4 X6 = LowerHalf(d4, L6);
    V4 X7 = LowerHalf(d4, L7);
    V4 X8 = LowerHalf(d4, U0);
    V4 X9 = LowerHalf(d4, U1);
    V4 XA = LowerHalf(d4, U2);
    V4 XB = LowerHalf(d4, U3);
    V4 XC = LowerHalf(d4, U4);
    V4 XD = LowerHalf(d4, U5);
    V4 XE = LowerHalf(d4, U6);
    V4 XF = LowerHalf(d4, U7);

    // Second half: quartets reversed, and passed in reverse order.
    V4 Y0 = LowerHalf(d4, Reverse(d8, U7));
    V4 Y1 = LowerHalf(d4, Reverse(d8, U6));
    V4 Y2 = LowerHalf(d4, Reverse(d8, U5));
    V4 Y3 = LowerHalf(d4, Reverse(d8, U4));
    V4 Y4 = LowerHalf(d4, Reverse(d8, U3));
    V4 Y5 = LowerHalf(d4, Reverse(d8, U2));
    V4 Y6 = LowerHalf(d4, Reverse(d8, U1));
    V4 Y7 = LowerHalf(d4, Reverse(d8, U0));
    V4 Y8 = LowerHalf(d4, Reverse(d8, L7));
    V4 Y9 = LowerHalf(d4, Reverse(d8, L6));
    V4 YA = LowerHalf(d4, Reverse(d8, L5));
    V4 YB = LowerHalf(d4, Reverse(d8, L4));
    V4 YC = LowerHalf(d4, Reverse(d8, L3));
    V4 YD = LowerHalf(d4, Reverse(d8, L2));
    V4 YE = LowerHalf(d4, Reverse(d8, L1));
    V4 YF = LowerHalf(d4, Reverse(d8, L0));

    detail::BitonicMergeSorted64Plus64<kOrder>(
        d4, X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, XA, XB, XC, XD, XE, XF, Y0,
        Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, YA, YB, YC, YD, YE, YF, -1);
    const Vec<D> sorted0 = Combine(d, Combine(d8, X3, X2), Combine(d8, X1, X0));
    const Vec<D> sorted1 = Combine(d, Combine(d8, X7, X6), Combine(d8, X5, X4));
    const Vec<D> sorted2 = Combine(d, Combine(d8, XB, XA), Combine(d8, X9, X8));
    const Vec<D> sorted3 = Combine(d, Combine(d8, XF, XE), Combine(d8, XD, XC));
    const Vec<D> sorted4 = Combine(d, Combine(d8, Y3, Y2), Combine(d8, Y1, Y0));
    const Vec<D> sorted5 = Combine(d, Combine(d8, Y7, Y6), Combine(d8, Y5, Y4));
    const Vec<D> sorted6 = Combine(d, Combine(d8, YB, YA), Combine(d8, Y9, Y8));
    const Vec<D> sorted7 = Combine(d, Combine(d8, YF, YE), Combine(d8, YD, YC));
    Store(sorted0, d, inout + 0 * 16);
    Store(sorted1, d, inout + 1 * 16);
    Store(sorted2, d, inout + 2 * 16);
    Store(sorted3, d, inout + 3 * 16);
    Store(sorted4, d, inout + 4 * 16);
    Store(sorted5, d, inout + 5 * 16);
    Store(sorted6, d, inout + 6 * 16);
    Store(sorted7, d, inout + 7 * 16);
    return;
  }
#endif
}

#endif  // HWY_TARGET != HWY_SCALAR && HWY_ARCH_X86

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_SORT_SORT_INL_H_
