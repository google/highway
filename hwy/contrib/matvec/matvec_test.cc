// Copyright 2023 Google LLC
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

#include "hwy/aligned_allocator.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/matvec/matvec_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// Must come after foreach_target.h
#include "hwy/contrib/algo/transform-inl.h"
#include "hwy/contrib/matvec/matvec-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

template <typename MatT, typename T>
HWY_NOINLINE void SimpleMatVec(const MatT* mat, const T* vec, size_t rows,
                               size_t cols, T* out, ThreadPool& pool) {
  pool.Run(0, static_cast<uint32_t>(rows), &ThreadPool::NoInit,
           [=](uint32_t r, size_t /*thread*/) {
             T dot = T{0};
             for (size_t c = 0; c < cols; c++) {
               dot += mat[r * cols + c] * vec[c];
             }
             out[r] = dot;
           });
}

HWY_NOINLINE void SimpleMatVec(const hwy::bfloat16_t* mat, const float* vec,
                               size_t rows, size_t cols, float* out,
                               ThreadPool& pool) {
  pool.Run(0, static_cast<uint32_t>(rows), &ThreadPool::NoInit,
           [=](uint32_t r, size_t /*thread*/) {
             float dot = 0.0f;
             for (size_t c = 0; c < cols; c++) {
               dot += F32FromBF16(mat[r * cols + c]) * vec[c];
             }
             out[r] = dot;
           });
}

struct GenerateMod {
  template <class D, HWY_IF_NOT_BF16_D(D), HWY_IF_LANES_GT_D(D, 1)>
  Vec<D> operator()(D d, Vec<RebindToUnsigned<D>> indices) const {
    const RebindToUnsigned<D> du;
    return Reverse2(d, ConvertTo(d, And(indices, Set(du, 0xF))));
  }

  template <class D, HWY_IF_NOT_BF16_D(D), HWY_IF_LANES_LE_D(D, 1)>
  Vec<D> operator()(D d, Vec<RebindToUnsigned<D>> indices) const {
    const RebindToUnsigned<D> du;
    return ConvertTo(d, And(indices, Set(du, 0xF)));
  }

  // Requires >= 4 bf16 lanes for float32 Reverse2.
  template <class D, HWY_IF_BF16_D(D), HWY_IF_LANES_GT_D(D, 2)>
  Vec<D> operator()(D d, Vec<RebindToUnsigned<D>> indices) const {
    const RebindToUnsigned<D> du;
    const RebindToSigned<D> di;
    const RepartitionToWide<decltype(di)> dw;
    const RebindToFloat<decltype(dw)> df;
    indices = And(indices, Set(du, 0xF));
    const Vec<decltype(df)> i0 = ConvertTo(df, PromoteLowerTo(dw, indices));
    const Vec<decltype(df)> i1 = ConvertTo(df, PromoteUpperTo(dw, indices));
    return OrderedDemote2To(d, Reverse2(df, i0), Reverse2(df, i1));
  }

  // For one or two lanes, we don't have OrderedDemote2To nor Reverse2.
  template <class D, HWY_IF_BF16_D(D), HWY_IF_LANES_LE_D(D, 2)>
  Vec<D> operator()(D d, Vec<RebindToUnsigned<D>> indices) const {
    const Rebind<float, D> df;
    return DemoteTo(d, Set(df, GetLane(indices)));
  }
};

// MatT is usually the same as T, but can also be bfloat16_t when T = float.
template <typename MatT>
class TestMatVec {
  template <size_t kRows, size_t kCols, class D, typename T = TFromD<D>>
  void Test(D d, size_t misalign_m, size_t misalign_v, ThreadPool& pool) {
// This target lacks too many ops required in our implementation, use
// HWY_EMU128 instead.
#if HWY_TARGET != HWY_SCALAR
    const Repartition<MatT, D> dm;
    // Fill matrix and vector with small integer values
    const size_t area = kRows * kCols;
    AlignedFreeUniquePtr<MatT[]> storage_m =
        AllocateAligned<MatT>(misalign_m + area);
    AlignedFreeUniquePtr<T[]> storage_v =
        AllocateAligned<T>(misalign_v + kCols);
    HWY_ASSERT(storage_m && storage_v);
    MatT* pm = storage_m.get() + misalign_m;
    T* pv = storage_v.get() + misalign_v;
    Generate(dm, pm, area, GenerateMod());
    Generate(d, pv, kCols, GenerateMod());

    AlignedFreeUniquePtr<T[]> expected = AllocateAligned<T>(kRows);
    SimpleMatVec(pm, pv, kRows, kCols, expected.get(), pool);

    AlignedFreeUniquePtr<T[]> actual = AllocateAligned<T>(kRows);
    MatVec<kRows, kCols>(pm, pv, actual.get(), pool);

    for (size_t i = 0; i < kRows; ++i) {
      const double exp = static_cast<double>(expected[i]);
      const double act = static_cast<double>(actual[i]);
      const double tolerance = exp * Epsilon<T>() * 15;
      if (!(exp - tolerance <= act && act <= exp + tolerance)) {
        fprintf(stderr, "%s %zu x %zu: mismatch at %zu %f %f; tol %f\n",
                TypeName(MatT(), 1).c_str(), kRows, kCols, i, exp, act,
                tolerance);
        HWY_ASSERT(0);
      }
    }
#else
    (void)d;
    (void)misalign_m;
    (void)misalign_v;
    (void)pool;
#endif  // HWY_TARGET != HWY_SCALAR
  }

  // Runs tests with various alignments.
  template <size_t kRows, size_t kCols, class D>
  void ForeachMisalign(D d, ThreadPool& pool) {
    const size_t N = Lanes(d);
    const size_t misalignments[3] = {N / 4, 3 * N / 5};
    for (size_t mm : misalignments) {
      for (size_t mv : misalignments) {
        Test<kRows, kCols>(d, mm, mv, pool);
      }
    }
  }

  // Runs tests with various lengths.
  template <class D>
  void ForeachDim(D d, ThreadPool& pool) {
    ForeachMisalign<192, AdjustedReps(256)>(d, pool);
    ForeachMisalign<40, AdjustedReps(512)>(d, pool);
    ForeachMisalign<AdjustedReps(1024), 50>(d, pool);

    // Too large for low-precision accumulators.
    if (sizeof(TFromD<D>) != 2) {
      ForeachMisalign<AdjustedReps(1536), 1536>(d, pool);
    }
  }

  template <class D>
  void CreatePoolAndTest(D d, size_t num_worker_threads) {
    ThreadPool pool(num_worker_threads);
    ForeachDim(d, pool);
  }

 public:
  template <class T, class D>
  HWY_INLINE void operator()(T /*unused*/, D d) {
    #if HWY_ARCH_WASM
    // Threads might not be work on WASM; run only on main thread.
    CreatePoolAndTest(d, 0);
    #else
    CreatePoolAndTest(d, 13);
    CreatePoolAndTest(d, 16);
    #endif  // HWY_ARCH_WASM
  }
};

void TestAllMatVec() {
#if HWY_HAVE_FLOAT16
  ForPartialVectors<TestMatVec<float16_t>>()(float16_t());
#endif
  ForPartialVectors<TestMatVec<float>>()(float());
#if HWY_HAVE_FLOAT64
  ForPartialVectors<TestMatVec<double>>()(double());
#endif
}

void TestAllMatVecBF16() {
  ForGEVectors<32, TestMatVec<bfloat16_t>>()(float());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
HWY_BEFORE_TEST(MatVecTest);
HWY_EXPORT_AND_TEST_P(MatVecTest, TestAllMatVec);
HWY_EXPORT_AND_TEST_P(MatVecTest, TestAllMatVecBF16);
}  // namespace hwy

#endif
