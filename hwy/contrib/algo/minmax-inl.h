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

// Returns the minimum value in `in[0, count)` or PositiveInfOrHighestValue<T>() if count == 0.
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

// Returns the maximum value in `in[0, count)` or NegativeInfOrLowestValue<T>() if count == 0.
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

// Returns the index of the first occurrence of the minimum value in
// `in[0, count)`, or `count` if `count == 0`. Ties resolve to the lowest index,
// matching `std::min_element`.
template <class D, typename T = TFromD<D>>
size_t IndexOfMin(D d, const T* HWY_RESTRICT in, size_t count) {
  if (HWY_UNLIKELY(count == 0)) return count;

  const RebindToUnsigned<D> du;
  using TU = TFromD<decltype(du)>;
  const size_t N = Lanes(d);

  // Lanes record which block held their best value, so a segment can span at
  // most this many blocks before that counter would wrap. Capped so neither the
  // count nor the multiply below can overflow; for 32-bit and wider lanes this
  // is a single segment in practice.
  const uint64_t block_limit = static_cast<uint64_t>(LimitsMax<TU>());
  const size_t max_blocks = static_cast<size_t>(
      block_limit < (uint64_t{1} << 20) ? block_limit : (uint64_t{1} << 20));

  const T identity = hwy::PositiveInfOrHighestValue<T>();
  const Vec<D> identity_vec = Set(d, identity);

  T best = identity;
  size_t best_idx = 0;

  for (size_t seg = 0; seg < count; seg += max_blocks * N) {
    const size_t seg_len = HWY_MIN(count - seg, max_blocks * N);

    Vec<D> acc = identity_vec;
    VFromD<decltype(du)> blocks = Zero(du);

    TU block = 0;
    for (size_t i = 0; i < seg_len; i += N, ++block) {
      const size_t n = HWY_MIN(seg_len - i, N);
      const Vec<D> v = LoadNOr(identity_vec, d, in + seg + i, n);
      // Strictly less, so an equal value later never displaces an earlier one.
      const Mask<D> lt = Lt(v, acc);
      acc = IfThenElse(lt, v, acc);
      blocks = IfThenElse(RebindMask(du, lt), Set(du, block), blocks);
    }

    // Resolve lanes: smallest value, then earliest block, then lowest lane.
    const T seg_min = ReduceMin(d, acc);
    const Mask<D> is_min = Eq(acc, Set(d, seg_min));
    const VFromD<decltype(du)> cand =
        IfThenElse(RebindMask(du, is_min), blocks, Set(du, LimitsMax<TU>()));
    const TU min_block = ReduceMin(du, cand);
    const Mask<D> winners =
        And(is_min, RebindMask(d, Eq(blocks, Set(du, min_block))));
    const size_t idx =
        seg + static_cast<size_t>(min_block) * N + FindKnownFirstTrue(d, winners);

    if (seg_min < best) {
      best = seg_min;
      best_idx = idx;
    }
  }

  return best_idx;
}

// Returns the index of the first occurrence of the maximum value in
// `in[0, count)`, or `count` if `count == 0`. Ties resolve to the lowest index,
// matching `std::max_element`.
template <class D, typename T = TFromD<D>>
size_t IndexOfMax(D d, const T* HWY_RESTRICT in, size_t count) {
  if (HWY_UNLIKELY(count == 0)) return count;

  const RebindToUnsigned<D> du;
  using TU = TFromD<decltype(du)>;
  const size_t N = Lanes(d);

  // Lanes record which block held their best value, so a segment can span at
  // most this many blocks before that counter would wrap. Capped so neither the
  // count nor the multiply below can overflow; for 32-bit and wider lanes this
  // is a single segment in practice.
  const uint64_t block_limit = static_cast<uint64_t>(LimitsMax<TU>());
  const size_t max_blocks = static_cast<size_t>(
      block_limit < (uint64_t{1} << 20) ? block_limit : (uint64_t{1} << 20));

  const T identity = hwy::NegativeInfOrLowestValue<T>();
  const Vec<D> identity_vec = Set(d, identity);

  T best = identity;
  size_t best_idx = 0;

  for (size_t seg = 0; seg < count; seg += max_blocks * N) {
    const size_t seg_len = HWY_MIN(count - seg, max_blocks * N);

    Vec<D> acc = identity_vec;
    VFromD<decltype(du)> blocks = Zero(du);

    TU block = 0;
    for (size_t i = 0; i < seg_len; i += N, ++block) {
      const size_t n = HWY_MIN(seg_len - i, N);
      const Vec<D> v = LoadNOr(identity_vec, d, in + seg + i, n);
      const Mask<D> gt = Gt(v, acc);
      acc = IfThenElse(gt, v, acc);
      blocks = IfThenElse(RebindMask(du, gt), Set(du, block), blocks);
    }

    const T seg_max = ReduceMax(d, acc);
    const Mask<D> is_max = Eq(acc, Set(d, seg_max));
    const VFromD<decltype(du)> cand =
        IfThenElse(RebindMask(du, is_max), blocks, Set(du, LimitsMax<TU>()));
    const TU min_block = ReduceMin(du, cand);
    const Mask<D> winners =
        And(is_max, RebindMask(d, Eq(blocks, Set(du, min_block))));
    const size_t idx =
        seg + static_cast<size_t>(min_block) * N + FindKnownFirstTrue(d, winners);

    if (seg_max > best) {
      best = seg_max;
      best_idx = idx;
    }
  }

  return best_idx;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_ALGO_MINMAX_INL_H_
