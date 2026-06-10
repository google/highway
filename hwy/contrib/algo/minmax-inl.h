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
#include <cstddef>
#include <utility>
#if defined(HIGHWAY_HWY_CONTRIB_ALGO_MINMAX_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_ALGO_MINMAX_INL_H_
#undef HIGHWAY_HWY_CONTRIB_ALGO_MINMAX_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_ALGO_MINMAX_INL_H_
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// Returns the minimum value in `in[0, count)` or
// PositiveInfOrHighestValue<T>() if count == 0.
template <class D, typename T = TFromD<D>>
T MinValue(D d, const T* HWY_RESTRICT in, size_t count) {
  const size_t N = Lanes(d);
  const T identity = hwy::PositiveInfOrHighestValue<T>();
  const Vec<D> identity_vec = Set(d, identity);

  Vec<D> acc0 = identity_vec;
  Vec<D> acc1 = identity_vec;
  Vec<D> acc2 = identity_vec;
  Vec<D> acc3 = identity_vec;

  size_t i = 0;
  if (count >= 4 * N) {
    for (; i <= count - 4 * N; i += 4 * N) {
      acc0 = Min(acc0, LoadU(d, in + i));
      acc1 = Min(acc1, LoadU(d, in + i + N));
      acc2 = Min(acc2, LoadU(d, in + i + 2 * N));
      acc3 = Min(acc3, LoadU(d, in + i + 3 * N));
    }
  }

  acc0 = Min(Min(acc0, acc1), Min(acc2, acc3));

  for (; i < count; i += N) {
    const size_t remaining = count - i;
    const size_t n = HWY_MIN(remaining, N);
    acc0 = Min(acc0, LoadNOr(identity_vec, d, in + i, n));
  }

  return ReduceMin(d, acc0);
}

// Returns the maximum value in `in[0, count)` or
// NegativeInfOrLowestValue<T>() if count == 0.
template <class D, typename T = TFromD<D>>
T MaxValue(D d, const T* HWY_RESTRICT in, size_t count) {
  const size_t N = Lanes(d);
  const T identity = hwy::NegativeInfOrLowestValue<T>();
  const Vec<D> identity_vec = Set(d, identity);

  Vec<D> acc0 = identity_vec;
  Vec<D> acc1 = identity_vec;
  Vec<D> acc2 = identity_vec;
  Vec<D> acc3 = identity_vec;

  size_t i = 0;
  if (count >= 4 * N) {
    for (; i <= count - 4 * N; i += 4 * N) {
      acc0 = Max(acc0, LoadU(d, in + i));
      acc1 = Max(acc1, LoadU(d, in + i + N));
      acc2 = Max(acc2, LoadU(d, in + i + 2 * N));
      acc3 = Max(acc3, LoadU(d, in + i + 3 * N));
    }
  }

  acc0 = Max(Max(acc0, acc1), Max(acc2, acc3));

  for (; i < count; i += N) {
    const size_t remaining = count - i;
    const size_t n = HWY_MIN(remaining, N);
    acc0 = Max(acc0, LoadNOr(identity_vec, d, in + i, n));
  }

  return ReduceMax(d, acc0);
}

template <class DF, typename T = TFromD<DF>,
          class DF2 = CappedTag<T, 2>, class VF2 = Vec<DF2>,
          class VF = Vec<DF>, class HRed, class VRed>
static HWY_INLINE std::pair<T, T> Reduce2X(DF df, VF x_0, VF x_1, HRed h_red,
                                           VRed v_red) {
  const DF2 df2;
  if constexpr (MaxLanes(df) < 2) {
    return {h_red(df, x_0), h_red(df, x_1)};
  } else {
    constexpr size_t kMaxLanes = MaxLanes(df);
    HWY_LANES_CONSTEXPR size_t kLanes = Lanes(df);
    HWY_ALIGN T x_transposed[2 * kMaxLanes];
    StoreInterleaved2(x_0, x_1, df, x_transposed);
    VF x01 = v_red(Load(df, x_transposed), Load(df, x_transposed + kLanes));
    Store(x01, df, x_transposed);

    VF2 result = Load(df2, x_transposed);
    for (size_t i = 1; i < kLanes / 2; ++i) {
      result = v_red(result, Load(df2, x_transposed + i * 2));
    }
    HWY_ALIGN T temp[2];
    Store(result, df2, temp);
    return {temp[0], temp[1]};
  }
}

template <class DF, typename T = TFromD<DF>,
          class DF4 = CappedTag<T, 4>, class VF4 = Vec<DF4>,
          class VF = Vec<DF>, class HRed, class VRed>
static HWY_INLINE VF4 Reduce4X(DF df, VF x_0, VF x_1, VF x_2, VF x_3,
                               HRed h_red, VRed v_red) {
  const DF4 df4;
  if constexpr ((HWY_TARGET & (HWY_ALL_NEON | HWY_ALL_SVE | HWY_RVV)) != 0 ||
                MaxLanes(df) < 4) {
    HWY_ALIGN T temp[4] = {h_red(df, x_0), h_red(df, x_1), h_red(df, x_2),
                           h_red(df, x_3)};
    return Load(df4, temp);
  } else {
    constexpr size_t kMaxLanes = MaxLanes(df);
    HWY_LANES_CONSTEXPR size_t kLanes = Lanes(df);
    HWY_ALIGN T x_transposed[4 * kMaxLanes];
    StoreInterleaved4(x_0, x_1, x_2, x_3, df, x_transposed);
    VF x01 = v_red(Load(df, x_transposed), Load(df, x_transposed + kLanes));
    VF x23 = v_red(Load(df, x_transposed + 2 * kLanes),
                   Load(df, x_transposed + 3 * kLanes));
    VF x0123 = v_red(x01, x23);
    Store(x0123, df, x_transposed);

    VF4 result = Load(df4, x_transposed);
    for (size_t i = 1; i < kLanes / 4; ++i) {
      result = v_red(result, Load(df4, x_transposed + i * 4));
    }
    return result;
  }
}

template <class DF, typename T = TFromD<DF>,
          class DF8 = CappedTag<T, 8>, class VF8 = Vec<DF8>,
          class VF = Vec<DF>, class HRed, class VRed>
static HWY_INLINE VF8 Reduce8X(DF df, VF x_0, VF x_1, VF x_2, VF x_3, VF x_4,
                               VF x_5, VF x_6, VF x_7, HRed h_red, VRed v_red) {
  const DF8 df8;
  if constexpr ((HWY_TARGET & (HWY_ALL_NEON | HWY_ALL_SVE | HWY_RVV)) != 0 ||
                MaxLanes(df) < 8) {
    HWY_ALIGN T temp[8] = {h_red(df, x_0), h_red(df, x_1), h_red(df, x_2),
                           h_red(df, x_3), h_red(df, x_4), h_red(df, x_5),
                           h_red(df, x_6), h_red(df, x_7)};
    return Load(df8, temp);
  } else {
    auto res0123 = Reduce4X(df, x_0, x_1, x_2, x_3, h_red, v_red);
    auto res4567 = Reduce4X(df, x_4, x_5, x_6, x_7, h_red, v_red);

    return Combine(df8, res4567, res0123);
  }
}



template <class DF, typename T = TFromD<DF>, class VF = Vec<DF>>
static HWY_INLINE std::pair<T, T> Reduce2Min(DF df, VF x_0, VF x_1) {
  return Reduce2X(
      df, x_0, x_1, [](auto d, auto v) HWY_ATTR { return ReduceMin(d, v); },
      [](auto a, auto b) HWY_ATTR { return Min(a, b); });
}

template <class DF, typename T = TFromD<DF>, class VF = Vec<DF>>
static HWY_INLINE std::pair<T, T> Reduce2Max(DF df, VF x_0, VF x_1) {
  return Reduce2X(
      df, x_0, x_1, [](auto d, auto v) HWY_ATTR { return ReduceMax(d, v); },
      [](auto a, auto b) HWY_ATTR { return Max(a, b); });
}

template <class DF, typename T = TFromD<DF>, class VF = Vec<DF>>
static HWY_INLINE std::pair<T, T> Reduce2Sum(DF df, VF x_0, VF x_1) {
  return Reduce2X(
      df, x_0, x_1, [](auto d, auto v) HWY_ATTR { return ReduceSum(d, v); },
      [](auto a, auto b) HWY_ATTR { return Add(a, b); });
}

template <class DF, typename T = TFromD<DF>,
          class DF4 = CappedTag<T, 4>, class VF4 = Vec<DF4>,
          class VF = Vec<DF>>
static HWY_INLINE VF4 Reduce4Min(DF df, VF x_0, VF x_1, VF x_2, VF x_3) {
  return Reduce4X(
      df, x_0, x_1, x_2, x_3,
      [](auto d, auto v) HWY_ATTR { return ReduceMin(d, v); },
      [](auto a, auto b) HWY_ATTR { return Min(a, b); });
}

template <class DF, typename T = TFromD<DF>,
          class DF4 = CappedTag<T, 4>, class VF4 = Vec<DF4>,
          class VF = Vec<DF>>
static HWY_INLINE VF4 Reduce4Max(DF df, VF x_0, VF x_1, VF x_2, VF x_3) {
  return Reduce4X(
      df, x_0, x_1, x_2, x_3,
      [](auto d, auto v) HWY_ATTR { return ReduceMax(d, v); },
      [](auto a, auto b) HWY_ATTR { return Max(a, b); });
}

template <class DF, typename T = TFromD<DF>,
          class DF4 = CappedTag<T, 4>, class VF4 = Vec<DF4>,
          class VF = Vec<DF>>
static HWY_INLINE VF4 Reduce4Sum(DF df, VF x_0, VF x_1, VF x_2, VF x_3) {
  return Reduce4X(
      df, x_0, x_1, x_2, x_3,
      [](auto d, auto v) HWY_ATTR { return ReduceSum(d, v); },
      [](auto a, auto b) HWY_ATTR { return Add(a, b); });
}

template <class DF, typename T = TFromD<DF>,
          class DF8 = CappedTag<T, 8>, class VF8 = Vec<DF8>,
          class VF = Vec<DF>>
static HWY_INLINE VF8 Reduce8Min(DF df, VF x_0, VF x_1, VF x_2, VF x_3, VF x_4,
                                 VF x_5, VF x_6, VF x_7) {
  return Reduce8X(
      df, x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7,
      [](auto d, auto v) HWY_ATTR { return ReduceMin(d, v); },
      [](auto a, auto b) HWY_ATTR { return Min(a, b); });
}

template <class DF, typename T = TFromD<DF>,
          class DF8 = CappedTag<T, 8>, class VF8 = Vec<DF8>,
          class VF = Vec<DF>>
static HWY_INLINE VF8 Reduce8Max(DF df, VF x_0, VF x_1, VF x_2, VF x_3, VF x_4,
                                 VF x_5, VF x_6, VF x_7) {
  return Reduce8X(
      df, x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7,
      [](auto d, auto v) HWY_ATTR { return ReduceMax(d, v); },
      [](auto a, auto b) HWY_ATTR { return Max(a, b); });
}

template <class DF, typename T = TFromD<DF>,
          class DF8 = CappedTag<T, 8>, class VF8 = Vec<DF8>,
          class VF = Vec<DF>>
static HWY_INLINE VF8 Reduce8Sum(DF df, VF x_0, VF x_1, VF x_2, VF x_3, VF x_4,
                                 VF x_5, VF x_6, VF x_7) {
  return Reduce8X(
      df, x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7,
      [](auto d, auto v) HWY_ATTR { return ReduceSum(d, v); },
      [](auto a, auto b) HWY_ATTR { return Add(a, b); });
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_ALGO_MINMAX_INL_H_
