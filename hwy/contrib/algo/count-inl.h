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

// Per-target include guard
#if defined(HIGHWAY_HWY_CONTRIB_ALGO_COUNT_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_ALGO_COUNT_INL_H_
#undef HIGHWAY_HWY_CONTRIB_ALGO_COUNT_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_ALGO_COUNT_INL_H_
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// Returns the number of elements in `in[0, count)` equal to `value`.
template <class D, typename T = TFromD<D>>
size_t Count(D d, T value, const T* HWY_RESTRICT in, size_t count) {
  const size_t N = Lanes(d);
  const Vec<D> broadcasted = Set(d, value);
  const RebindToSigned<D> di;
  using VI = Vec<decltype(di)>;

  size_t total = 0;
  size_t i = 0;

  if constexpr (sizeof(T) == 1) {
    if constexpr (HWY_MAX_LANES_D(D) >= 4) {
      const RepartitionToWide<decltype(di)> di16;
      const Repartition<int32_t, D> di32;
      auto wide_sum = Zero(di32);

      if (count >= 4 * N) {
        while (i <= count - 4 * N) {
          VI acc0 = Zero(di);
          VI acc1 = Zero(di);
          VI acc2 = Zero(di);
          VI acc3 = Zero(di);
          const size_t cap = HWY_MIN(i + 128 * 4 * N, count);

#ifdef HWY_NATIVE_MASK
          const auto one_i = Set(di, TFromD<decltype(di)>(1));
          for (; i <= cap - 4 * N; i += 4 * N) {
            acc0 = MaskedAddOr(
                acc0, RebindMask(di, Eq(broadcasted, LoadU(d, in + i))), acc0,
                one_i);
            acc1 = MaskedAddOr(
                acc1, RebindMask(di, Eq(broadcasted, LoadU(d, in + i + N))),
                acc1, one_i);
            acc2 = MaskedAddOr(
                acc2, RebindMask(di, Eq(broadcasted, LoadU(d, in + i + 2 * N))),
                acc2, one_i);
            acc3 = MaskedAddOr(
                acc3, RebindMask(di, Eq(broadcasted, LoadU(d, in + i + 3 * N))),
                acc3, one_i);
          }
          acc0 = Add(Add(acc0, acc1), Add(acc2, acc3));
#else
          for (; i <= cap - 4 * N; i += 4 * N) {
            acc0 = Add(
                acc0,
                BitCast(di, VecFromMask(d, Eq(broadcasted, LoadU(d, in + i)))));
            acc1 = Add(acc1,
                       BitCast(di, VecFromMask(d, Eq(broadcasted,
                                                     LoadU(d, in + i + N)))));
            acc2 = Add(
                acc2,
                BitCast(di, VecFromMask(
                                d, Eq(broadcasted, LoadU(d, in + i + 2 * N)))));
            acc3 = Add(
                acc3,
                BitCast(di, VecFromMask(
                                d, Eq(broadcasted, LoadU(d, in + i + 3 * N)))));
          }
          acc0 = Neg(Add(Add(acc0, acc1), Add(acc2, acc3)));
#endif
          const auto acc_u8 = BitCast(RebindToUnsigned<decltype(di)>(), acc0);
          const auto one_i8 = Set(di, TFromD<decltype(di)>(1));
          const auto widened = SatWidenMulPairwiseAdd(di16, acc_u8, one_i8);
          const auto one_i16 = Set(di16, int16_t(1));
          wide_sum =
              SatWidenMulPairwiseAccumulate(di32, widened, one_i16, wide_sum);
        }
      }
      total += static_cast<size_t>(ReduceSum(di32, wide_sum));
    }
  } else if constexpr (sizeof(T) == 2) {
    if constexpr (HWY_MAX_LANES_D(D) >= 2) {
      const Repartition<int32_t, D> di32;
      auto wide_sum = Zero(di32);

      if (count >= 4 * N) {
        while (i <= count - 4 * N) {
          VI acc0 = Zero(di);
          VI acc1 = Zero(di);
          VI acc2 = Zero(di);
          VI acc3 = Zero(di);
          const size_t cap = HWY_MIN(i + 32768 * 4 * N, count);

#ifdef HWY_NATIVE_MASK
          const auto one_i = Set(di, TFromD<decltype(di)>(1));
          for (; i <= cap - 4 * N; i += 4 * N) {
            acc0 = MaskedAddOr(
                acc0, RebindMask(di, Eq(broadcasted, LoadU(d, in + i))), acc0,
                one_i);
            acc1 = MaskedAddOr(
                acc1, RebindMask(di, Eq(broadcasted, LoadU(d, in + i + N))),
                acc1, one_i);
            acc2 = MaskedAddOr(
                acc2, RebindMask(di, Eq(broadcasted, LoadU(d, in + i + 2 * N))),
                acc2, one_i);
            acc3 = MaskedAddOr(
                acc3, RebindMask(di, Eq(broadcasted, LoadU(d, in + i + 3 * N))),
                acc3, one_i);
          }
          acc0 = Add(Add(acc0, acc1), Add(acc2, acc3));
#else
          for (; i <= cap - 4 * N; i += 4 * N) {
            acc0 = Add(
                acc0,
                BitCast(di, VecFromMask(d, Eq(broadcasted, LoadU(d, in + i)))));
            acc1 = Add(acc1,
                       BitCast(di, VecFromMask(d, Eq(broadcasted,
                                                     LoadU(d, in + i + N)))));
            acc2 = Add(
                acc2,
                BitCast(di, VecFromMask(
                                d, Eq(broadcasted, LoadU(d, in + i + 2 * N)))));
            acc3 = Add(
                acc3,
                BitCast(di, VecFromMask(
                                d, Eq(broadcasted, LoadU(d, in + i + 3 * N)))));
          }
          acc0 = Neg(Add(Add(acc0, acc1), Add(acc2, acc3)));
#endif
          const auto one_i16 = Set(di, TFromD<decltype(di)>(1));
          wide_sum =
              SatWidenMulPairwiseAccumulate(di32, acc0, one_i16, wide_sum);
        }
      }
      total += static_cast<size_t>(ReduceSum(di32, wide_sum));
    }
  } else {
    if (count >= 4 * N) {
      VI acc0 = Zero(di);
      VI acc1 = Zero(di);
      VI acc2 = Zero(di);
      VI acc3 = Zero(di);

#ifdef HWY_NATIVE_MASK
      const auto one_i = Set(di, TFromD<decltype(di)>(1));
      for (; i <= count - 4 * N; i += 4 * N) {
        acc0 =
            MaskedAddOr(acc0, RebindMask(di, Eq(broadcasted, LoadU(d, in + i))),
                        acc0, one_i);
        acc1 = MaskedAddOr(
            acc1, RebindMask(di, Eq(broadcasted, LoadU(d, in + i + N))), acc1,
            one_i);
        acc2 = MaskedAddOr(
            acc2, RebindMask(di, Eq(broadcasted, LoadU(d, in + i + 2 * N))),
            acc2, one_i);
        acc3 = MaskedAddOr(
            acc3, RebindMask(di, Eq(broadcasted, LoadU(d, in + i + 3 * N))),
            acc3, one_i);
      }
      acc0 = Add(Add(acc0, acc1), Add(acc2, acc3));
#else
      for (; i <= count - 4 * N; i += 4 * N) {
        acc0 =
            Add(acc0,
                BitCast(di, VecFromMask(d, Eq(broadcasted, LoadU(d, in + i)))));
        acc1 = Add(
            acc1,
            BitCast(di, VecFromMask(d, Eq(broadcasted, LoadU(d, in + i + N)))));
        acc2 = Add(acc2,
                   BitCast(di, VecFromMask(d, Eq(broadcasted,
                                                 LoadU(d, in + i + 2 * N)))));
        acc3 = Add(acc3,
                   BitCast(di, VecFromMask(d, Eq(broadcasted,
                                                 LoadU(d, in + i + 3 * N)))));
      }
      acc0 = Neg(Add(Add(acc0, acc1), Add(acc2, acc3)));
#endif
      total += static_cast<size_t>(ReduceSum(di, acc0));
    }
  }

  if (count >= N) {
    for (; i <= count - N; i += N) {
      total += CountTrue(d, Eq(broadcasted, LoadU(d, in + i)));
    }
  }

  if (i != count) {
    const size_t remaining = count - i;
    HWY_DASSERT(0 != remaining && remaining < N);
    const Vec<D> v = LoadN(d, in + i, remaining);
    total += CountTrue(d, And(Eq(broadcasted, v), FirstN(d, remaining)));
  }

  return total;
}

// Returns the number of elements in `in[0, count)` for which `func(d, vec)`
// returns true.
template <class D, class Func, typename T = TFromD<D>>
size_t CountIf(D d, const T* HWY_RESTRICT in, size_t count, const Func& func) {
  const size_t N = Lanes(d);

  size_t total = 0;
  size_t i = 0;
  if (count >= 4 * N) {
    for (; i <= count - 4 * N; i += 4 * N) {
      total += CountTrue(d, func(d, LoadU(d, in + i)));
      total += CountTrue(d, func(d, LoadU(d, in + i + N)));
      total += CountTrue(d, func(d, LoadU(d, in + i + 2 * N)));
      total += CountTrue(d, func(d, LoadU(d, in + i + 3 * N)));
    }
  }

  if (count >= N) {
    for (; i <= count - N; i += N) {
      total += CountTrue(d, func(d, LoadU(d, in + i)));
    }
  }

  if (i != count) {
#if HWY_MEM_OPS_MIGHT_FAULT
    const CappedTag<T, 1> d1;
    for (; i < count; ++i) {
      if (AllTrue(d1, func(d1, LoadU(d1, in + i)))) {
        total += 1;
      }
    }
#else
    const size_t remaining = count - i;
    HWY_DASSERT(0 != remaining && remaining < N);
    const Mask<D> mask = FirstN(d, remaining);
    const Vec<D> v = MaskedLoad(mask, d, in + i);
    total += CountTrue(d, And(func(d, v), mask));
#endif  // HWY_MEM_OPS_MIGHT_FAULT
  }

  return total;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_ALGO_COUNT_INL_H_
