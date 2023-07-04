// Copyright 2019 Google LLC
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

#include <string.h>  // memset

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/swizzle_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

struct TestGetLane {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v = Iota(d, T(1));
    HWY_ASSERT_EQ(T(1), GetLane(v));
  }
};

HWY_NOINLINE void TestAllGetLane() {
  ForAllTypes(ForPartialVectors<TestGetLane>());
}

struct TestExtractLane {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v = Iota(d, T(1));
    for (size_t i = 0; i < Lanes(d); ++i) {
      const T actual = ExtractLane(v, i);
      HWY_ASSERT_EQ(static_cast<T>(i + 1), actual);
    }
  }
};

HWY_NOINLINE void TestAllExtractLane() {
  ForAllTypes(ForPartialVectors<TestExtractLane>());
}

struct TestInsertLane {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    using V = Vec<D>;
    const V v = Iota(d, T(1));
    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);
    HWY_ASSERT(lanes);
    Store(v, d, lanes.get());

    for (size_t i = 0; i < Lanes(d); ++i) {
      lanes[i] = T{0};
      const V actual = InsertLane(v, i, static_cast<T>(i + 1));
      HWY_ASSERT_VEC_EQ(d, v, actual);
      Store(v, d, lanes.get());  // restore lane i
    }
  }
};

HWY_NOINLINE void TestAllInsertLane() {
  ForAllTypes(ForPartialVectors<TestInsertLane>());
}

struct TestDupEven {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(expected);
    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>((i & ~size_t{1}) + 1);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), DupEven(Iota(d, 1)));
  }
};

HWY_NOINLINE void TestAllDupEven() {
  ForAllTypes(ForShrinkableVectors<TestDupEven>());
}

struct TestDupOdd {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
#if HWY_TARGET != HWY_SCALAR
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(expected);
    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>((i & ~size_t{1}) + 2);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), DupOdd(Iota(d, 1)));
#else
    (void)d;
#endif
  }
};

HWY_NOINLINE void TestAllDupOdd() {
  ForAllTypes(ForShrinkableVectors<TestDupOdd>());
}

struct TestOddEven {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    const auto even = Iota(d, 1);
    const auto odd = Iota(d, static_cast<T>(1 + N));
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(expected);
    for (size_t i = 0; i < N; ++i) {
      expected[i] = static_cast<T>(1 + i + ((i & 1) ? N : 0));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), OddEven(odd, even));
  }
};

HWY_NOINLINE void TestAllOddEven() {
  ForAllTypes(ForShrinkableVectors<TestOddEven>());
}

struct TestTableLookupLanes {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const RebindToSigned<D> di;
    using TI = TFromD<decltype(di)>;
#if HWY_TARGET != HWY_SCALAR
    const size_t N = Lanes(d);
    auto idx = AllocateAligned<TI>(N);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(idx && expected);
    memset(idx.get(), 0, N * sizeof(TI));
    const auto v = Iota(d, 1);

    if (N <= 8) {  // Test all permutations
      for (size_t i0 = 0; i0 < N; ++i0) {
        idx[0] = static_cast<TI>(i0);

        for (size_t i1 = 0; i1 < N; ++i1) {
          if (N >= 2) idx[1] = static_cast<TI>(i1);
          for (size_t i2 = 0; i2 < N; ++i2) {
            if (N >= 4) idx[2] = static_cast<TI>(i2);
            for (size_t i3 = 0; i3 < N; ++i3) {
              if (N >= 4) idx[3] = static_cast<TI>(i3);

              for (size_t i = 0; i < N; ++i) {
                expected[i] = static_cast<T>(idx[i] + 1);  // == v[idx[i]]
              }

              const auto opaque1 = IndicesFromVec(d, Load(di, idx.get()));
              const auto actual1 = TableLookupLanes(v, opaque1);
              HWY_ASSERT_VEC_EQ(d, expected.get(), actual1);

              const auto opaque2 = SetTableIndices(d, idx.get());
              const auto actual2 = TableLookupLanes(v, opaque2);
              HWY_ASSERT_VEC_EQ(d, expected.get(), actual2);
            }
          }
        }
      }
    } else {
      // Too many permutations to test exhaustively; choose one with repeated
      // and cross-block indices and ensure indices do not exceed #lanes.
      // For larger vectors, upper lanes will be zero.
      HWY_ALIGN TI idx_source[16] = {1,  3,  2,  2,  8, 1, 7, 6,
                                     15, 14, 14, 15, 4, 9, 8, 5};
      for (size_t i = 0; i < N; ++i) {
        idx[i] = (i < 16) ? idx_source[i] : 0;
        // Avoid undefined results / asan error for scalar by capping indices.
        if (idx[i] >= static_cast<TI>(N)) {
          idx[i] = static_cast<TI>(N - 1);
        }
        expected[i] = static_cast<T>(idx[i] + 1);  // == v[idx[i]]
      }

      const auto opaque1 = IndicesFromVec(d, Load(di, idx.get()));
      const auto actual1 = TableLookupLanes(v, opaque1);
      HWY_ASSERT_VEC_EQ(d, expected.get(), actual1);

      const auto opaque2 = SetTableIndices(d, idx.get());
      const auto actual2 = TableLookupLanes(v, opaque2);
      HWY_ASSERT_VEC_EQ(d, expected.get(), actual2);
    }
#else
    const TI index = 0;
    const auto v = Set(d, 1);
    const auto opaque1 = SetTableIndices(d, &index);
    HWY_ASSERT_VEC_EQ(d, v, TableLookupLanes(v, opaque1));
    const auto opaque2 = IndicesFromVec(d, Zero(di));
    HWY_ASSERT_VEC_EQ(d, v, TableLookupLanes(v, opaque2));
#endif
  }
};

HWY_NOINLINE void TestAllTableLookupLanes() {
  ForAllTypes(ForPartialVectors<TestTableLookupLanes>());
}

struct TestTwoTablesLookupLanes {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const RebindToUnsigned<D> du;
    using TU = TFromD<decltype(du)>;

    const size_t N = Lanes(d);
    const size_t twiceN = N * 2;
    auto idx = AllocateAligned<TU>(twiceN);
    auto expected = AllocateAligned<T>(twiceN);
    HWY_ASSERT(idx && expected);
    memset(idx.get(), 0, twiceN * sizeof(TU));
    const auto a = Iota(d, 1);
    const auto b = Add(a, Set(d, static_cast<T>(N)));

    if (twiceN <= 8) {  // Test all permutations
      for (size_t i0 = 0; i0 < twiceN; ++i0) {
        idx[0] = static_cast<TU>(i0);

        for (size_t i1 = 0; i1 < twiceN; ++i1) {
          if (twiceN >= 2) idx[1] = static_cast<TU>(i1);
          for (size_t i2 = 0; i2 < twiceN; ++i2) {
            if (twiceN >= 4) idx[2] = static_cast<TU>(i2);
            for (size_t i3 = 0; i3 < twiceN; ++i3) {
              if (twiceN >= 4) idx[3] = static_cast<TU>(i3);

              for (size_t i = 0; i < twiceN; ++i) {
                expected[i] = static_cast<T>(idx[i] + 1);  // == v[idx[i]]
              }

              const auto opaque1_a = IndicesFromVec(d, Load(du, idx.get()));
              const auto opaque1_b = IndicesFromVec(d, Load(du, idx.get() + N));
              const auto actual1_a = TwoTablesLookupLanes(d, a, b, opaque1_a);
              const auto actual1_b = TwoTablesLookupLanes(d, a, b, opaque1_b);
              HWY_ASSERT_VEC_EQ(d, expected.get(), actual1_a);
              HWY_ASSERT_VEC_EQ(d, expected.get() + N, actual1_b);

              const auto opaque2_a = SetTableIndices(d, idx.get());
              const auto opaque2_b = SetTableIndices(d, idx.get() + N);
              const auto actual2_a = TwoTablesLookupLanes(d, a, b, opaque2_a);
              const auto actual2_b = TwoTablesLookupLanes(d, a, b, opaque2_b);
              HWY_ASSERT_VEC_EQ(d, expected.get(), actual2_a);
              HWY_ASSERT_VEC_EQ(d, expected.get() + N, actual2_b);
            }
          }
        }
      }
    } else {
      constexpr size_t kLanesPerBlock = 16 / sizeof(T);
      constexpr size_t kMaxBlockIdx = static_cast<size_t>(LimitsMax<TU>()) >> 1;
      static_assert(kMaxBlockIdx > 0, "kMaxBlockIdx > 0 must be true");

      const size_t num_of_blocks_per_vect = HWY_MAX(N / kLanesPerBlock, 1);
      const size_t num_of_blocks_to_check =
          HWY_MIN(num_of_blocks_per_vect * 2, kMaxBlockIdx);

      for (size_t i = 0; i < num_of_blocks_to_check; i++) {
        // Too many permutations to test exhaustively; choose one with repeated
        // and cross-block indices and ensure indices do not exceed #lanes.
        // For larger vectors, upper lanes will be zero.
        HWY_ALIGN TU idx_source[16] = {1,  3,  2,  2,  8, 1, 7, 6,
                                       15, 14, 14, 15, 4, 9, 8, 5};
        for (size_t j = 0; j < twiceN; ++j) {
          idx[j] = static_cast<TU>((i * kLanesPerBlock + idx_source[j & 15] +
                                    (j & static_cast<size_t>(-16))) &
                                   (twiceN - 1));
          expected[j] = static_cast<T>(idx[j] + 1);  // == v[idx[j]]
        }

        const auto opaque1_a = IndicesFromVec(d, Load(du, idx.get()));
        const auto opaque1_b = IndicesFromVec(d, Load(du, idx.get() + N));
        const auto actual1_a = TwoTablesLookupLanes(d, a, b, opaque1_a);
        const auto actual1_b = TwoTablesLookupLanes(d, a, b, opaque1_b);
        HWY_ASSERT_VEC_EQ(d, expected.get(), actual1_a);
        HWY_ASSERT_VEC_EQ(d, expected.get() + N, actual1_b);

        const auto opaque2_a = SetTableIndices(d, idx.get());
        const auto opaque2_b = SetTableIndices(d, idx.get() + N);
        const auto actual2_a = TwoTablesLookupLanes(d, a, b, opaque2_a);
        const auto actual2_b = TwoTablesLookupLanes(d, a, b, opaque2_b);
        HWY_ASSERT_VEC_EQ(d, expected.get(), actual2_a);
        HWY_ASSERT_VEC_EQ(d, expected.get() + N, actual2_b);
      }
    }
  }
};

HWY_NOINLINE void TestAllTwoTablesLookupLanes() {
  ForAllTypes(ForPartialVectors<TestTwoTablesLookupLanes>());
}

class TestPer4LaneBlockShuffle {
 private:
  template <class D, HWY_IF_LANES_LE_D(D, 1)>
  static HWY_INLINE VFromD<D> InterleaveMaskVectors(D /*d*/, VFromD<D> a,
                                                    VFromD<D> /*b*/) {
    return a;
  }
#if HWY_TARGET != HWY_SCALAR
  template <class D, HWY_IF_LANES_GT_D(D, 1)>
  static HWY_INLINE VFromD<D> InterleaveMaskVectors(D d, VFromD<D> a,
                                                    VFromD<D> b) {
    return InterleaveLower(d, a, b);
  }
#endif
  template <class D>
  static HWY_INLINE Mask<D> Per4LaneBlockShufValidMask(D d, const size_t N,
                                                       const size_t idx1,
                                                       const size_t idx0) {
    if (N < 4) {
      const RebindToSigned<decltype(d)> di;
      using TI = TFromD<decltype(di)>;
      const auto lane_0_valid =
          Set(di, static_cast<TI>(-static_cast<int>(idx0 < N)));
      if (N > 1) {
        const auto lane_1_valid =
            Set(di, static_cast<TI>(-static_cast<int>(idx1 < N)));
        return RebindMask(d, MaskFromVec(InterleaveMaskVectors(di, lane_0_valid,
                                                               lane_1_valid)));
      }
      return RebindMask(d, MaskFromVec(lane_0_valid));
    }

    return FirstN(d, N);
  }

  // TODO(b/287462770): inline to work around incorrect SVE codegen
  template <class D>
  static HWY_INLINE void DoCheckPer4LaneBlkShufResult(
      D d, const size_t N, VFromD<D> actual,
      const TFromD<D>* HWY_RESTRICT src_lanes, TFromD<D>* HWY_RESTRICT expected,
      size_t idx3, size_t idx2, size_t idx1, size_t idx0) {
    for (size_t i = 0; i < N; i += 4) {
      expected[i] = src_lanes[i + idx0];
      expected[i + 1] = src_lanes[i + idx1];
      expected[i + 2] = src_lanes[i + idx2];
      expected[i + 3] = src_lanes[i + idx3];
    }

    if (N < 4) {
      if (idx0 >= N) expected[0] = TFromD<D>{0};
      if (idx1 >= N) expected[1] = TFromD<D>{0};
    }

    const auto valid_lanes_mask = Per4LaneBlockShufValidMask(d, N, idx1, idx0);
    HWY_ASSERT_VEC_EQ(d, expected, IfThenElseZero(valid_lanes_mask, actual));
  }

#if HWY_TARGET != HWY_SCALAR
  template <class D>
  static HWY_NOINLINE void TestTblLookupPer4LaneBlkShuf(
      D d, const size_t N, const TFromD<D>* HWY_RESTRICT src_lanes,
      TFromD<D>* HWY_RESTRICT expected) {
    const auto v = Load(d, src_lanes);
    for (size_t idx3210 = 0; idx3210 <= 0xFF; idx3210++) {
      const size_t idx3 = (idx3210 >> 6) & 3;
      const size_t idx2 = (idx3210 >> 4) & 3;
      const size_t idx1 = (idx3210 >> 2) & 3;
      const size_t idx0 = idx3210 & 3;

      const auto actual = detail::TblLookupPer4LaneBlkShuf(v, idx3210);
      DoCheckPer4LaneBlkShufResult(d, N, actual, src_lanes, expected, idx3,
                                   idx2, idx1, idx0);
    }
  }
#endif

  template <size_t kIdx3, size_t kIdx2, size_t kIdx1, size_t kIdx0, class D>
  static HWY_INLINE void DoTestPer4LaneBlkShuffle(
      D d, const size_t N, const VFromD<D> v,
      const TFromD<D>* HWY_RESTRICT src_lanes,
      TFromD<D>* HWY_RESTRICT expected) {
    const auto actual = Per4LaneBlockShuffle<kIdx3, kIdx2, kIdx1, kIdx0>(v);
    DoCheckPer4LaneBlkShufResult(d, N, actual, src_lanes, expected, kIdx3,
                                 kIdx2, kIdx1, kIdx0);
  }

  template <class D>
  static HWY_NOINLINE void DoTestPer4LaneBlkShuffles(
      D d, const size_t N, const VecArg<VFromD<D>> v,
      TFromD<D>* HWY_RESTRICT src_lanes,
      TFromD<D>* HWY_RESTRICT expected) {
    Store(v, d, src_lanes);
#if HWY_TARGET != HWY_SCALAR
    TestTblLookupPer4LaneBlkShuf(d, N, src_lanes, expected);
#endif
    DoTestPer4LaneBlkShuffle<0, 1, 2, 3>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<0, 1, 3, 2>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<0, 2, 3, 1>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<0, 3, 0, 2>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<1, 0, 1, 0>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<1, 0, 3, 1>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<1, 0, 3, 2>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<1, 2, 0, 3>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<1, 2, 1, 3>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<1, 1, 0, 0>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<2, 0, 1, 3>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<2, 0, 2, 0>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<2, 1, 2, 0>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<2, 2, 0, 0>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<2, 3, 0, 1>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<2, 3, 3, 0>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<3, 0, 2, 1>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<3, 1, 0, 3>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<3, 1, 3, 1>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<3, 2, 1, 0>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<3, 2, 3, 2>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<3, 3, 0, 1>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<3, 3, 1, 1>(d, N, v, src_lanes, expected);
    DoTestPer4LaneBlkShuffle<3, 3, 2, 2>(d, N, v, src_lanes, expected);
  }

  template <class D>
  static HWY_INLINE Vec<D> GenerateTestVect(hwy::NonFloatTag /*tag*/, D d) {
    const RebindToUnsigned<decltype(d)> du;
    using TU = TFromD<decltype(du)>;
    constexpr TU kIotaStart = static_cast<TU>(0x0706050403020101u);
    return BitCast(d, Iota(du, kIotaStart));
  }

  template <class D>
  static HWY_INLINE Vec<D> GenerateTestVect(hwy::FloatTag /*tag*/, D d) {
    const RebindToUnsigned<decltype(d)> du;
    using T = TFromD<decltype(d)>;
    using TU = TFromD<decltype(du)>;

    constexpr size_t kNumOfBitsInT = sizeof(T) * 8;
    constexpr TU kIntBitsMask =
        (kNumOfBitsInT > 16) ? static_cast<TU>(static_cast<TU>(~TU{0}) >> 16)
                             : TU{0};

    const auto flt_iota = Set(d, 1);
    if (kIntBitsMask == 0) return flt_iota;

    const auto int_iota =
        And(GenerateTestVect(hwy::NonFloatTag(), du), Set(du, kIntBitsMask));
    return Or(flt_iota, BitCast(d, int_iota));
  }

 public:
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    const size_t alloc_len = static_cast<size_t>((N + 3) & (~size_t{3}));
    HWY_ASSERT(alloc_len >= 4);

    auto expected = AllocateAligned<T>(alloc_len);
    auto src_lanes = AllocateAligned<T>(alloc_len);
    HWY_ASSERT(expected && src_lanes);

    expected[alloc_len - 4] = T{0};
    expected[alloc_len - 3] = T{0};
    expected[alloc_len - 2] = T{0};
    expected[alloc_len - 1] = T{0};
    src_lanes[alloc_len - 4] = T{0};
    src_lanes[alloc_len - 3] = T{0};
    src_lanes[alloc_len - 2] = T{0};
    src_lanes[alloc_len - 1] = T{0};

    const auto v = GenerateTestVect(hwy::IsFloatTag<T>(), d);
    DoTestPer4LaneBlkShuffles(d, N, v, src_lanes.get(), expected.get());

    const RebindToUnsigned<decltype(d)> du;
    using TU = TFromD<decltype(du)>;
    const auto msb_mask =
        BitCast(d, Set(du, static_cast<TU>(TU{1} << (sizeof(TU) * 8 - 1))));

    DoTestPer4LaneBlkShuffles(d, N, Xor(v, msb_mask), src_lanes.get(),
                              expected.get());
  }
};

HWY_NOINLINE void TestAllPer4LaneBlockShuffle() {
  ForAllTypes(ForPartialFixedOrFullScalableVectors<TestPer4LaneBlockShuffle>());
}

class TestBroadcastLane {
 private:
  template <int kLane, class D,
            HWY_IF_LANES_GT_D(D, static_cast<size_t>(kLane))>
  static HWY_INLINE void DoTestBroadcastLane(D d, const size_t N) {
    using T = TFromD<D>;
    using TU = MakeUnsigned<T>;
    // kLane < HWY_MAX_LANES_D(D) is true
    if (kLane >= N) return;

    constexpr T kExpectedVal = static_cast<T>(static_cast<TU>(kLane) + 1u);
    const auto expected = Set(d, kExpectedVal);

    const BlockDFromD<decltype(d)> d_block;
    static_assert(d_block.MaxLanes() <= d.MaxLanes(),
                  "d_block.MaxLanes() <= d.MaxLanes() must be true");
    constexpr size_t kLanesPer16ByteBlk = 16 / sizeof(T);
    constexpr int kBlockIdx = kLane / static_cast<int>(kLanesPer16ByteBlk);
    constexpr int kLaneInBlkIdx =
        kLane & static_cast<int>(kLanesPer16ByteBlk - 1);

    const Vec<D> v = Iota(d, T{1});
    const Vec<D> actual = BroadcastLane<kLane>(v);
    const Vec<decltype(d_block)> actual_block =
        ExtractBlock<kBlockIdx>(Broadcast<kLaneInBlkIdx>(v));

    HWY_ASSERT_VEC_EQ(d, expected, actual);
    HWY_ASSERT_VEC_EQ(d_block, ResizeBitCast(d_block, expected), actual_block);
  }
  template <int kLane, class D,
            HWY_IF_LANES_LE_D(D, static_cast<size_t>(kLane))>
  static HWY_INLINE void DoTestBroadcastLane(D /*d*/, const size_t /*N*/) {
    // If kLane >= HWY_MAX_LANES_D(D) is true, do nothing
  }

 public:
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto N = Lanes(d);

    DoTestBroadcastLane<0>(d, N);
    DoTestBroadcastLane<1>(d, N);
    DoTestBroadcastLane<2>(d, N);
    DoTestBroadcastLane<3>(d, N);
    DoTestBroadcastLane<6>(d, N);
    DoTestBroadcastLane<14>(d, N);
    DoTestBroadcastLane<29>(d, N);
    DoTestBroadcastLane<53>(d, N);
    DoTestBroadcastLane<115>(d, N);
    DoTestBroadcastLane<251>(d, N);
    DoTestBroadcastLane<257>(d, N);
  }
};

HWY_NOINLINE void TestAllBroadcastLane() {
  ForAllTypes(ForPartialFixedOrFullScalableVectors<TestBroadcastLane>());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
HWY_BEFORE_TEST(HwySwizzleTest);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllGetLane);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllExtractLane);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllInsertLane);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllDupEven);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllDupOdd);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllOddEven);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllTableLookupLanes);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllTwoTablesLookupLanes);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllPer4LaneBlockShuffle);
HWY_EXPORT_AND_TEST_P(HwySwizzleTest, TestAllBroadcastLane);
}  // namespace hwy

#endif
