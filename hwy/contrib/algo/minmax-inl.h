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

  // Blocks are counted down rather than up, so the earliest block is the
  // largest counter. That lets the resolution below zero out non-candidate
  // lanes, because zero is the identity for ReduceMax and is never a valid
  // counter: block b in [0, max_blocks) is stored as max_blocks - b, which is
  // at least 1. A lane the loop never touched keeps max_blocks, i.e. block 0.
  const TU inv_base = static_cast<TU>(max_blocks);

  const T identity = hwy::PositiveInfOrHighestValue<T>();
  const Vec<D> identity_vec = Set(d, identity);

  T best = identity;
  size_t best_idx = 0;

  for (size_t seg = 0; seg < count; seg += max_blocks * N) {
    const size_t seg_len = HWY_MIN(count - seg, max_blocks * N);

    Vec<D> acc0 = identity_vec;
    Vec<D> acc1 = identity_vec;
    VFromD<decltype(du)> blocks0 = Set(du, inv_base);
    VFromD<decltype(du)> blocks1 = Set(du, inv_base);

    size_t i = 0;
    TU inv_block = inv_base;

    // Two accumulators, because each select otherwise waits on the previous
    // iteration's acc and the loop becomes a latency chain.
    if (seg_len >= 2 * N) {
      for (; i + 2 * N <= seg_len;
           i += 2 * N, inv_block = static_cast<TU>(inv_block - 2)) {
        const Vec<D> v0 = LoadU(d, in + seg + i);
        const Vec<D> v1 = LoadU(d, in + seg + i + N);
        // Strictly less, so an equal value later never displaces an earlier
        // one.
        const Mask<D> lt0 = Lt(v0, acc0);
        const Mask<D> lt1 = Lt(v1, acc1);
        acc0 = IfThenElse(lt0, v0, acc0);
        acc1 = IfThenElse(lt1, v1, acc1);
        blocks0 = IfThenElse(RebindMask(du, lt0), Set(du, inv_block), blocks0);
        blocks1 = IfThenElse(RebindMask(du, lt1),
                             Set(du, static_cast<TU>(inv_block - 1)), blocks1);
      }

      // Fold the odd blocks into the even ones: the smaller value wins, and on
      // a tie the earlier block does, which is the larger counter. Merging
      // before the tail is what keeps the tail's blocks correctly later.
      const Mask<D> take1 =
          Or(Lt(acc1, acc0),
             And(Eq(acc0, acc1), RebindMask(d, Gt(blocks1, blocks0))));
      acc0 = IfThenElse(take1, acc1, acc0);
      blocks0 = IfThenElse(RebindMask(du, take1), blocks1, blocks0);
    }

    for (; i < seg_len; i += N, --inv_block) {
      const size_t n = HWY_MIN(seg_len - i, N);
      const Vec<D> v = LoadNOr(identity_vec, d, in + seg + i, n);
      const Mask<D> lt = Lt(v, acc0);
      acc0 = IfThenElse(lt, v, acc0);
      blocks0 = IfThenElse(RebindMask(du, lt), Set(du, inv_block), blocks0);
    }

    // Resolve lanes: smallest value, then earliest block, then lowest lane.
    const T seg_min = ReduceMin(d, acc0);
    const Mask<D> is_min = Eq(acc0, Set(d, seg_min));
    const TU best_inv =
        ReduceMax(du, IfThenElseZero(RebindMask(du, is_min), blocks0));
    const Mask<D> winners = RebindMask(
        d, MaskedEq(RebindMask(du, is_min), blocks0, Set(du, best_inv)));
    const size_t min_block =
        static_cast<size_t>(max_blocks) - static_cast<size_t>(best_inv);
    const size_t idx = seg + min_block * N + FindKnownFirstTrue(d, winners);

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

  // Counted down so the earliest block is the largest counter; see IndexOfMin.
  const TU inv_base = static_cast<TU>(max_blocks);

  const T identity = hwy::NegativeInfOrLowestValue<T>();
  const Vec<D> identity_vec = Set(d, identity);

  T best = identity;
  size_t best_idx = 0;

  for (size_t seg = 0; seg < count; seg += max_blocks * N) {
    const size_t seg_len = HWY_MIN(count - seg, max_blocks * N);

    Vec<D> acc0 = identity_vec;
    Vec<D> acc1 = identity_vec;
    VFromD<decltype(du)> blocks0 = Set(du, inv_base);
    VFromD<decltype(du)> blocks1 = Set(du, inv_base);

    size_t i = 0;
    TU inv_block = inv_base;

    if (seg_len >= 2 * N) {
      for (; i + 2 * N <= seg_len;
           i += 2 * N, inv_block = static_cast<TU>(inv_block - 2)) {
        const Vec<D> v0 = LoadU(d, in + seg + i);
        const Vec<D> v1 = LoadU(d, in + seg + i + N);
        const Mask<D> gt0 = Gt(v0, acc0);
        const Mask<D> gt1 = Gt(v1, acc1);
        acc0 = IfThenElse(gt0, v0, acc0);
        acc1 = IfThenElse(gt1, v1, acc1);
        blocks0 = IfThenElse(RebindMask(du, gt0), Set(du, inv_block), blocks0);
        blocks1 = IfThenElse(RebindMask(du, gt1),
                             Set(du, static_cast<TU>(inv_block - 1)), blocks1);
      }

      const Mask<D> take1 =
          Or(Gt(acc1, acc0),
             And(Eq(acc0, acc1), RebindMask(d, Gt(blocks1, blocks0))));
      acc0 = IfThenElse(take1, acc1, acc0);
      blocks0 = IfThenElse(RebindMask(du, take1), blocks1, blocks0);
    }

    for (; i < seg_len; i += N, --inv_block) {
      const size_t n = HWY_MIN(seg_len - i, N);
      const Vec<D> v = LoadNOr(identity_vec, d, in + seg + i, n);
      const Mask<D> gt = Gt(v, acc0);
      acc0 = IfThenElse(gt, v, acc0);
      blocks0 = IfThenElse(RebindMask(du, gt), Set(du, inv_block), blocks0);
    }

    const T seg_max = ReduceMax(d, acc0);
    const Mask<D> is_max = Eq(acc0, Set(d, seg_max));
    const TU best_inv =
        ReduceMax(du, IfThenElseZero(RebindMask(du, is_max), blocks0));
    const Mask<D> winners = RebindMask(
        d, MaskedEq(RebindMask(du, is_max), blocks0, Set(du, best_inv)));
    const size_t min_block =
        static_cast<size_t>(max_blocks) - static_cast<size_t>(best_inv);
    const size_t idx = seg + min_block * N + FindKnownFirstTrue(d, winners);

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
