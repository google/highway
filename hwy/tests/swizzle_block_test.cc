// Copyright 2019 Google LLC
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: BSD-3-Clause
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

#include "hwy/base.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/swizzle_block_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

#ifndef HWY_NATIVE_INTERLEAVE_WHOLE
#error "HWY_NATIVE_INTERLEAVE_WHOLE should be defined by set_macros-inl.h"
#endif

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

struct TestOddEvenBlocks {
  template <class T, class D>
  HWY_NOINLINE HWY_MAYBE_UNUSED void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    const auto even = Iota(d, 1);
    const auto odd = Iota(d, 1 + N);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(expected);
    for (size_t i = 0; i < N; ++i) {
      const size_t idx_block = i / (16 / sizeof(T));
      expected[i] = ConvertScalarTo<T>(1 + i + ((idx_block & 1) ? N : 0));
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), OddEvenBlocks(odd, even));
  }
};

HWY_NOINLINE void TestAllOddEvenBlocks() {
  ForAllTypes(ForGEVectors<128, TestOddEvenBlocks>());
}

struct TestSwapAdjacentBlocks {
  template <class T, class D>
  HWY_NOINLINE HWY_MAYBE_UNUSED void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    constexpr size_t kLanesPerBlock = 16 / sizeof(T);
    if (N < 2 * kLanesPerBlock) return;
    const auto vi = Iota(d, 1);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(expected);
    for (size_t i = 0; i < N; ++i) {
      const size_t idx_block = i / kLanesPerBlock;
      const size_t base = (idx_block ^ 1) * kLanesPerBlock;
      const size_t mod = i % kLanesPerBlock;
      expected[i] = ConvertScalarTo<T>(1 + base + mod);
    }
    HWY_ASSERT_VEC_EQ(d, expected.get(), SwapAdjacentBlocks(vi));
  }
};

HWY_NOINLINE void TestAllSwapAdjacentBlocks() {
  ForAllTypes(ForGEVectors<128, TestSwapAdjacentBlocks>());
}

struct TestInterleaveBlocksEO {
  template <class T, class D>
  HWY_NOINLINE HWY_MAYBE_UNUSED void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    constexpr size_t kLanesPerBlock = 16 / sizeof(T);
    if (N < 2 * kLanesPerBlock) return;
    const VFromD<D> va = Iota(d, 1);
    const VFromD<D> vb = Iota(d, N + 1);
    auto expected_even = AllocateAligned<T>(N);
    auto expected_odd = AllocateAligned<T>(N);
    HWY_ASSERT(expected_even && expected_odd);
    for (size_t i = 0; i < N; ++i) {
      const size_t idx_block = i / kLanesPerBlock;
      const size_t mod = 1 + i % kLanesPerBlock;
      const size_t base_even = RoundDownTo(idx_block, 2) * kLanesPerBlock;
      const size_t base_odd = (idx_block | 1) * kLanesPerBlock;
      if (idx_block & 1) {  // odd blocks come from B, hence + N
        expected_even[i] = ConvertScalarTo<T>(base_even + N + mod);
        expected_odd[i] = ConvertScalarTo<T>(base_odd + N + mod);
      } else {  // even blocks come from A
        expected_even[i] = ConvertScalarTo<T>(base_even + mod);
        expected_odd[i] = ConvertScalarTo<T>(base_odd + mod);
      }
    }
    HWY_ASSERT_VEC_EQ(d, expected_even.get(), InterleaveEvenBlocks(d, va, vb));
    HWY_ASSERT_VEC_EQ(d, expected_odd.get(), InterleaveOddBlocks(d, va, vb));
  }
};

HWY_NOINLINE void TestAllInterleaveBlocksEO() {
  ForAllTypes(ForGEVectors<128, TestInterleaveBlocksEO>());
}

struct TestInterleaveBlocksLU {
  template <class T, class D>
  HWY_NOINLINE HWY_MAYBE_UNUSED void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    constexpr size_t kLanesPerBlock = 16 / sizeof(T);
    if (N < 2 * kLanesPerBlock) return;
    const VFromD<D> va = Iota(d, 1);
    const VFromD<D> vb = Iota(d, N + 1);
    auto expected_lower = AllocateAligned<T>(N);
    auto expected_upper = AllocateAligned<T>(N);
    HWY_ASSERT(expected_lower && expected_upper);

    const size_t num_blocks = N / kLanesPerBlock;
    const size_t num_blocks_div2 = num_blocks / 2;

    for (size_t i = 0; i < N; ++i) {
      const size_t out_block_idx = i / kLanesPerBlock;
      const size_t mod = i % kLanesPerBlock;

      const size_t src_block_lower = out_block_idx / 2;
      const size_t src_block_upper = src_block_lower + num_blocks_div2;

      const size_t val_lower = 1 + (src_block_lower * kLanesPerBlock) + mod;
      const size_t val_upper = 1 + (src_block_upper * kLanesPerBlock) + mod;

      if (out_block_idx % 2 == 0) {
        // Even output blocks come from A
        expected_lower[i] = ConvertScalarTo<T>(val_lower);
        expected_upper[i] = ConvertScalarTo<T>(val_upper);
      } else {
        // Odd output blocks come from B (add N offset for B's values)
        expected_lower[i] = ConvertScalarTo<T>(val_lower + N);
        expected_upper[i] = ConvertScalarTo<T>(val_upper + N);
      }
    }

    HWY_ASSERT_VEC_EQ(d, expected_lower.get(),
                      InterleaveLowerBlocks(d, va, vb));
    HWY_ASSERT_VEC_EQ(d, expected_upper.get(),
                      InterleaveUpperBlocks(d, va, vb));
  }
};

HWY_NOINLINE void TestAllInterleaveBlocksLU() {
  ForAllTypes(ForGEVectors<128, TestInterleaveBlocksLU>());
}

struct TestConcatEvenOddBlocks {
  template <bool kTestInterleaveInverse, class D,
            hwy::EnableIf<!kTestInterleaveInverse || HWY_V_SIZE_D(D) <= 16>* =
                nullptr>
  static HWY_INLINE void VerifyInterleaveInverse(D /*d*/, size_t /*N*/,
                                                 VFromD<D> /*vhi*/,
                                                 VFromD<D> /*vlo*/,
                                                 VFromD<D> /*actual_even*/,
                                                 VFromD<D> /*actual_odd*/) {}

  template <bool kTestInterleaveInverse, class D,
            hwy::EnableIf<kTestInterleaveInverse && (HWY_V_SIZE_D(D) > 16)>* =
                nullptr>
  static HWY_INLINE void VerifyInterleaveInverse(D d, size_t N, VFromD<D> vhi,
                                                 VFromD<D> vlo,
                                                 VFromD<D> actual_even,
                                                 VFromD<D> actual_odd) {
    constexpr size_t kLanesPerBlock = 16 / sizeof(TFromD<D>);
    if (N >= 2 * kLanesPerBlock && (N & (N - 1)) == 0) {
      HWY_ASSERT_VEC_EQ(d, vlo,
                        InterleaveLowerBlocks(d, actual_even, actual_odd));
      HWY_ASSERT_VEC_EQ(d, vhi,
                        InterleaveUpperBlocks(d, actual_even, actual_odd));
      const VFromD<D> inter_lo = InterleaveLowerBlocks(d, vlo, vhi);
      const VFromD<D> inter_hi = InterleaveUpperBlocks(d, vlo, vhi);
      HWY_ASSERT_VEC_EQ(d, vlo, ConcatEvenBlocks(d, inter_hi, inter_lo));
      HWY_ASSERT_VEC_EQ(d, vhi, ConcatOddBlocks(d, inter_hi, inter_lo));
    }
  }

  template <class D, HWY_IF_V_SIZE_LE_D(D, 16)>
  static HWY_INLINE void VerifyTwoBlock(D /*d*/, size_t /*N*/,
                                        VFromD<D> /*vhi*/, VFromD<D> /*vlo*/,
                                        VFromD<D> /*actual_even*/,
                                        VFromD<D> /*actual_odd*/) {}

  template <class D, HWY_IF_V_SIZE_GT_D(D, 16)>
  static HWY_INLINE void VerifyTwoBlock(D d, size_t N, VFromD<D> vhi,
                                        VFromD<D> vlo, VFromD<D> actual_even,
                                        VFromD<D> actual_odd) {
    constexpr size_t kLanesPerBlock = 16 / sizeof(TFromD<D>);
    if (N == 2 * kLanesPerBlock) {
      HWY_ASSERT_VEC_EQ(d, ConcatLowerLower(d, vhi, vlo), actual_even);
      HWY_ASSERT_VEC_EQ(d, ConcatUpperUpper(d, vhi, vlo), actual_odd);
    }
  }
  template <class D, HWY_IF_NOT_FLOAT_NOR_SPECIAL_D(D)>
  static HWY_INLINE void VerifyNegativeZero(D /*d*/, size_t /*N*/) {}

  template <class D, HWY_IF_FLOAT_OR_SPECIAL_D(D)>
  static HWY_INLINE void VerifyNegativeZero(D d, size_t N) {
    using T = TFromD<D>;
    using TU = MakeUnsigned<T>;
    constexpr TU kSignBit = static_cast<TU>(TU{1} << (sizeof(T) * 8 - 1));
    const RebindToUnsigned<D> du;
    const VFromD<D> pos_zero = BitCast(d, Set(du, TU{0}));
    const VFromD<D> neg_zero = BitCast(d, Set(du, kSignBit));
    auto exp_bits = AllocateAligned<TU>(N);
    HWY_ASSERT(exp_bits);
    for (size_t i = 0; i < N / 2; ++i) exp_bits[i] = TU{0};
    for (size_t i = N / 2; i < N; ++i) exp_bits[i] = kSignBit;
    HWY_ASSERT_VEC_EQ(du, exp_bits.get(),
                      BitCast(du, ConcatEvenBlocks(d, neg_zero, pos_zero)));
    HWY_ASSERT_VEC_EQ(du, exp_bits.get(),
                      BitCast(du, ConcatOddBlocks(d, neg_zero, pos_zero)));
  }

  template <bool kTestInterleaveInverse = false, class D>
  static HWY_NOINLINE void VerifyTag(D d) {
    using T = TFromD<D>;
    constexpr size_t kLanesPerBlock = 16 / sizeof(T);
    const size_t N = Lanes(d);
    if (N < 2 * kLanesPerBlock) return;
    // Skip non power of two SVE tags with odd or fractional block counts.
    if ((N % (2 * kLanesPerBlock)) != 0) return;

    auto lo_lanes = AllocateAligned<T>(N);
    auto hi_lanes = AllocateAligned<T>(N);
    auto expected_even = AllocateAligned<T>(N);
    auto expected_odd = AllocateAligned<T>(N);
    HWY_ASSERT(lo_lanes && hi_lanes && expected_even && expected_odd);

    const size_t num_blocks = N / kLanesPerBlock;
    const size_t half_blocks = num_blocks / 2;
    RandomState rng(static_cast<uint64_t>(N) * 0x9E3779B1u + sizeof(T));
    for (size_t rep = 0; rep < 3; ++rep) {
      for (size_t i = 0; i < N; ++i) {
        const int32_t r_lo = static_cast<int32_t>(Random32(&rng) & 127);
        const int32_t r_hi = static_cast<int32_t>(Random32(&rng) & 127);
        lo_lanes[i] = ConvertScalarTo<T>(IsSigned<T>() ? (r_lo - 64) : r_lo);
        hi_lanes[i] = ConvertScalarTo<T>(IsSigned<T>() ? (r_hi - 64) : r_hi);
      }

      for (size_t j = 0; j < half_blocks; ++j) {
        CopyBytes(lo_lanes.get() + (2 * j) * kLanesPerBlock,
                  expected_even.get() + j * kLanesPerBlock, 16);
        CopyBytes(lo_lanes.get() + (2 * j + 1) * kLanesPerBlock,
                  expected_odd.get() + j * kLanesPerBlock, 16);
        CopyBytes(hi_lanes.get() + (2 * j) * kLanesPerBlock,
                  expected_even.get() + (half_blocks + j) * kLanesPerBlock, 16);
        CopyBytes(hi_lanes.get() + (2 * j + 1) * kLanesPerBlock,
                  expected_odd.get() + (half_blocks + j) * kLanesPerBlock, 16);
      }

      const VFromD<D> vlo = Load(d, lo_lanes.get());
      const VFromD<D> vhi = Load(d, hi_lanes.get());
      const VFromD<D> actual_even = ConcatEvenBlocks(d, vhi, vlo);
      const VFromD<D> actual_odd = ConcatOddBlocks(d, vhi, vlo);
      HWY_ASSERT_VEC_EQ(d, expected_even.get(), actual_even);
      HWY_ASSERT_VEC_EQ(d, expected_odd.get(), actual_odd);

      VerifyTwoBlock(d, N, vhi, vlo, actual_even, actual_odd);
      VerifyInterleaveInverse<kTestInterleaveInverse>(d, N, vhi, vlo,
                                                      actual_even, actual_odd);
    }

    // Verify extreme values are preserved without saturation.
    const T lo_extreme = LowestValue<T>();
    const T hi_extreme = HighestValue<T>();
    const VFromD<D> v_min = Set(d, lo_extreme);
    const VFromD<D> v_max = Set(d, hi_extreme);
    for (size_t i = 0; i < N / 2; ++i) expected_even[i] = lo_extreme;
    for (size_t i = N / 2; i < N; ++i) expected_even[i] = hi_extreme;
    HWY_ASSERT_VEC_EQ(d, expected_even.get(),
                      ConcatEvenBlocks(d, v_max, v_min));
    HWY_ASSERT_VEC_EQ(d, expected_even.get(), ConcatOddBlocks(d, v_max, v_min));

    VerifyNegativeZero(d, N);
  }

  template <class T, class D>
  HWY_NOINLINE HWY_MAYBE_UNUSED void operator()(T /*unused*/, D d) {
    VerifyTag</*kTestInterleaveInverse=*/true>(d);
  }
};

struct TestConcatEvenOddBlocksExtraTags {
  template <class T>
  HWY_NOINLINE void operator()(T /*unused*/) const {
#if HWY_TARGET == HWY_RVV
    constexpr size_t kLpb = 16 / sizeof(T);
    // Capped multi register tags exercise PartialVecSetHalf on LMUL 2, 4, 8.
    TestConcatEvenOddBlocks::VerifyTag(CappedTag<T, 2 * kLpb, 1>());
    TestConcatEvenOddBlocks::VerifyTag(CappedTag<T, 2 * kLpb, 2>());
    TestConcatEvenOddBlocks::VerifyTag(CappedTag<T, 2 * kLpb, 3>());
    TestConcatEvenOddBlocks::VerifyTag(CappedTag<T, 4 * kLpb, 3>());
    // Capped fractional tag (including virtual LMUL for 64 bit lanes when
    // VLEN >= 512).
    TestConcatEvenOddBlocks::VerifyTag(CappedTag<T, 2 * kLpb, -1>());
#elif HWY_HAVE_SCALABLE
    constexpr size_t kLpb = 16 / sizeof(T);
    // Capped fractional tag (two blocks when VL >= 512).
    TestConcatEvenOddBlocks::VerifyTag(CappedTag<T, 2 * kLpb, -1>());
#elif HWY_MAX_BYTES >= 32
    // Full width tag with nonzero kPow2 when sizeof(T) > 1.
    TestConcatEvenOddBlocks::VerifyTag(
        Rebind<T, FixedTag<uint8_t, HWY_MAX_BYTES / sizeof(T)>>());
#endif
  }
};

HWY_NOINLINE void TestAllConcatEvenOddBlocks() {
  ForAllTypesAndSpecial(ForPartialVectors<TestConcatEvenOddBlocks>());
  ForAllTypesAndSpecial(TestConcatEvenOddBlocksExtraTags());
}

class TestInsertBlock {
 private:
  template <int kBlock, class D,
            HWY_IF_V_SIZE_GT_D(D, static_cast<size_t>(kBlock) * 16)>
  static HWY_INLINE HWY_MAYBE_UNUSED void DoTestInsertBlock(
      D d, const size_t N, TFromD<D>* HWY_RESTRICT expected) {
    // kBlock * 16 < D.MaxBytes() is true
    using T = TFromD<D>;
    using TI = MakeSigned<T>;
    using TU = MakeUnsigned<T>;

    const RebindToUnsigned<decltype(d)> du;
    const BlockDFromD<decltype(d)> d_block;
    const RebindToUnsigned<decltype(d_block)> du_block;
    using V = Vec<D>;
    using VB = Vec<decltype(d_block)>;
    constexpr TU kPositiveMask = static_cast<TU>(LimitsMax<TI>());
    constexpr TU kSignBit = static_cast<TU>(~kPositiveMask);

    for (size_t i = 0; i < N; i++) {
      const T val = ConvertScalarTo<T>(i);
      TU val_bits;
      CopySameSize(&val, &val_bits);
      val_bits &= kPositiveMask;
      CopySameSize(&val_bits, &expected[i]);
    }

    constexpr size_t kLanesPer16ByteBlk = 16 / sizeof(T);
    constexpr size_t kBlkLaneOffset =
        static_cast<size_t>(kBlock) * kLanesPer16ByteBlk;
    if (kBlkLaneOffset < N) {
      const size_t num_of_lanes_in_blk =
          HWY_MIN(N - kBlkLaneOffset, kLanesPer16ByteBlk);
      for (size_t i = 0; i < num_of_lanes_in_blk; i++) {
        const T val =
            ConvertScalarTo<T>(static_cast<TU>(i) + static_cast<TU>(kBlock));
        TU val_bits;
        CopySameSize(&val, &val_bits);
        val_bits |= kSignBit;
        CopySameSize(&val_bits, &expected[kBlkLaneOffset + i]);
      }
    }

    const V v = And(Iota(d, 0), BitCast(d, Set(du, kPositiveMask)));
    const VB blk_to_insert =
        Or(Iota(d_block, kBlock), BitCast(d_block, Set(du_block, kSignBit)));
    const V actual = InsertBlock<kBlock>(v, blk_to_insert);
    HWY_ASSERT_VEC_EQ(d, expected, actual);
  }
  template <int kBlock, class D,
            HWY_IF_V_SIZE_LE_D(D, static_cast<size_t>(kBlock) * 16)>
  static HWY_INLINE HWY_MAYBE_UNUSED void DoTestInsertBlock(
      D /*d*/, const size_t /*N*/, TFromD<D>* HWY_RESTRICT /*expected*/) {
    // If kBlock * 16 >= D.MaxBytes() is true, do nothing
  }

 public:
  template <class T, class D>
  HWY_NOINLINE HWY_MAYBE_UNUSED void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(expected);

    DoTestInsertBlock<0>(d, N, expected.get());
    DoTestInsertBlock<1>(d, N, expected.get());
    DoTestInsertBlock<2>(d, N, expected.get());
    DoTestInsertBlock<3>(d, N, expected.get());
  }
};

HWY_NOINLINE void TestAllInsertBlock() {
  ForAllTypes(ForPartialFixedOrFullScalableVectors<TestInsertBlock>());
}

class TestExtractBlock {
 private:
  template <int kBlock, class D,
            HWY_IF_V_SIZE_GT_D(D, static_cast<size_t>(kBlock) * 16)>
  static HWY_INLINE HWY_MAYBE_UNUSED void DoTestExtractBlock(
      D d, const size_t N, TFromD<D>* HWY_RESTRICT expected) {
    // kBlock * 16 < D.MaxBytes() is true
    using T = TFromD<D>;

    constexpr size_t kLanesPer16ByteBlk = 16 / sizeof(T);
    constexpr size_t kBlkLaneOffset =
        static_cast<size_t>(kBlock) * kLanesPer16ByteBlk;
    if (kBlkLaneOffset >= N) return;

    const BlockDFromD<decltype(d)> d_block;
    static_assert(d_block.MaxLanes() <= kLanesPer16ByteBlk,
                  "d_block.MaxLanes() <= kLanesPer16ByteBlk must be true");

    for (size_t i = 0; i < kLanesPer16ByteBlk; i++) {
      expected[i] = ConvertScalarTo<T>(kBlkLaneOffset + i);
    }

    const auto v = Iota(d, 0);
    const Vec<BlockDFromD<decltype(d_block)>> actual = ExtractBlock<kBlock>(v);
    HWY_ASSERT_VEC_EQ(d_block, expected, actual);
  }
  template <int kBlock, class D,
            HWY_IF_V_SIZE_LE_D(D, static_cast<size_t>(kBlock) * 16)>
  static HWY_INLINE HWY_MAYBE_UNUSED void DoTestExtractBlock(
      D /*d*/, const size_t /*N*/, TFromD<D>* HWY_RESTRICT /*expected*/) {
    // If kBlock * 16 >= D.MaxBytes() is true, do nothing
  }

 public:
  template <class T, class D>
  HWY_NOINLINE HWY_MAYBE_UNUSED void operator()(T /*unused*/, D d) {
    constexpr size_t kLanesPer16ByteBlk = 16 / sizeof(T);
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(kLanesPer16ByteBlk);
    HWY_ASSERT(expected);

    DoTestExtractBlock<0>(d, N, expected.get());
    DoTestExtractBlock<1>(d, N, expected.get());
    DoTestExtractBlock<2>(d, N, expected.get());
    DoTestExtractBlock<3>(d, N, expected.get());
  }
};

HWY_NOINLINE void TestAllExtractBlock() {
  ForAllTypes(ForPartialFixedOrFullScalableVectors<TestExtractBlock>());
}

class TestBroadcastBlock {
 private:
  template <int kBlock, class D,
            HWY_IF_V_SIZE_GT_D(D, static_cast<size_t>(kBlock) * 16)>
  static HWY_INLINE HWY_MAYBE_UNUSED void DoTestBroadcastBlock(
      D d, const size_t N, TFromD<D>* HWY_RESTRICT expected) {
    // kBlock * 16 < D.MaxBytes() is true
    using T = TFromD<D>;

    constexpr size_t kLanesPer16ByteBlk = 16 / sizeof(T);
    constexpr size_t kBlkLaneOffset =
        static_cast<size_t>(kBlock) * kLanesPer16ByteBlk;
    if (kBlkLaneOffset >= N) return;

    for (size_t i = 0; i < N; i++) {
      const size_t idx_in_blk = i & (kLanesPer16ByteBlk - 1);
      expected[i] =
          ConvertScalarTo<T>(kBlkLaneOffset + kLanesPer16ByteBlk + idx_in_blk);
    }

    const auto v = Iota(d, kLanesPer16ByteBlk);
    const auto actual = BroadcastBlock<kBlock>(v);
    HWY_ASSERT_VEC_EQ(d, expected, actual);
  }
  template <int kBlock, class D,
            HWY_IF_V_SIZE_LE_D(D, static_cast<size_t>(kBlock) * 16)>
  static HWY_INLINE HWY_MAYBE_UNUSED void DoTestBroadcastBlock(
      D /*d*/, const size_t /*N*/, TFromD<D>* HWY_RESTRICT /*expected*/) {
    // If kBlock * 16 >= D.MaxBytes() is true, do nothing
  }

 public:
  template <class T, class D>
  HWY_NOINLINE HWY_MAYBE_UNUSED void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto expected = AllocateAligned<T>(N);
    HWY_ASSERT(expected);

    DoTestBroadcastBlock<0>(d, N, expected.get());
    DoTestBroadcastBlock<1>(d, N, expected.get());
    DoTestBroadcastBlock<2>(d, N, expected.get());
    DoTestBroadcastBlock<3>(d, N, expected.get());
  }
};

HWY_NOINLINE void TestAllBroadcastBlock() {
  ForAllTypes(ForPartialFixedOrFullScalableVectors<TestBroadcastBlock>());
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(HwySwizzleBlockTest);
HWY_EXPORT_AND_TEST_P(HwySwizzleBlockTest, TestAllOddEvenBlocks);
HWY_EXPORT_AND_TEST_P(HwySwizzleBlockTest, TestAllSwapAdjacentBlocks);
HWY_EXPORT_AND_TEST_P(HwySwizzleBlockTest, TestAllInterleaveBlocksEO);
HWY_EXPORT_AND_TEST_P(HwySwizzleBlockTest, TestAllInterleaveBlocksLU);
HWY_EXPORT_AND_TEST_P(HwySwizzleBlockTest, TestAllConcatEvenOddBlocks);
HWY_EXPORT_AND_TEST_P(HwySwizzleBlockTest, TestAllInsertBlock);
HWY_EXPORT_AND_TEST_P(HwySwizzleBlockTest, TestAllExtractBlock);
HWY_EXPORT_AND_TEST_P(HwySwizzleBlockTest, TestAllBroadcastBlock);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
