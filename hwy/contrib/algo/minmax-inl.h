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

#include <utility>

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

// {MinValue(d, in, count), MaxValue(d, in, count)}
template <class D, typename T = TFromD<D>>
std::pair<T, T> MinMaxValue(D d, const T* HWY_RESTRICT in, size_t count) {
  const size_t N = Lanes(d);
  const T min_identity = hwy::PositiveInfOrHighestValue<T>();
  const T max_identity = hwy::NegativeInfOrLowestValue<T>();
  const Vec<D> min_identity_vec = Set(d, min_identity);
  const Vec<D> max_identity_vec = Set(d, max_identity);

  Vec<D> min0 = min_identity_vec;
  Vec<D> min1 = min_identity_vec;
  Vec<D> min2 = min_identity_vec;
  Vec<D> min3 = min_identity_vec;
  Vec<D> max0 = max_identity_vec;
  Vec<D> max1 = max_identity_vec;
  Vec<D> max2 = max_identity_vec;
  Vec<D> max3 = max_identity_vec;

  size_t i = 0;
  if (count >= 4 * N) {
    for (; i <= count - 4 * N; i += 4 * N) {
      const Vec<D> v0 = LoadU(d, in + i);
      const Vec<D> v1 = LoadU(d, in + i + N);
      const Vec<D> v2 = LoadU(d, in + i + 2 * N);
      const Vec<D> v3 = LoadU(d, in + i + 3 * N);
      min0 = Min(min0, v0);
      min1 = Min(min1, v1);
      min2 = Min(min2, v2);
      min3 = Min(min3, v3);
      max0 = Max(max0, v0);
      max1 = Max(max1, v1);
      max2 = Max(max2, v2);
      max3 = Max(max3, v3);
    }
  }

  min0 = Min(Min(min0, min1), Min(min2, min3));
  max0 = Max(Max(max0, max1), Max(max2, max3));

  for (; i < count; i += N) {
    const size_t remaining = count - i;
    const size_t n = HWY_MIN(remaining, N);
    const Vec<D> v = LoadNOr(min_identity_vec, d, in + i, n);
    min0 = Min(min0, v);
    max0 = Max(max0, IfThenElse(FirstN(d, n), v, max_identity_vec));
  }

  return {ReduceMin(d, min0), ReduceMax(d, max0)};
}

// {IndexOfMin(d, in, count), IndexOfMax(d, in, count)}
template <class D, typename T = TFromD<D>>
std::pair<size_t, size_t> IndexOfMinMax(D d, const T* HWY_RESTRICT in,
                                        size_t count) {
  if (HWY_UNLIKELY(count == 0)) {
    return {count, count};
  }

  const RebindToUnsigned<D> du;
  using TU = TFromD<decltype(du)>;
  using VU = VFromD<decltype(du)>;
  using MU = Mask<decltype(du)>;
  const size_t N = Lanes(d);

  constexpr uint64_t kBlockLimit = static_cast<uint64_t>(LimitsMax<TU>());
  constexpr uint64_t kBlockCap = uint64_t{1} << 20;
  constexpr uint64_t kCappedLimit = HWY_MIN(kBlockLimit, kBlockCap);
  constexpr size_t kMaxBlocks = static_cast<size_t>(kCappedLimit);
  constexpr TU kInvBase = static_cast<TU>(kMaxBlocks);

  const T min_identity = hwy::PositiveInfOrHighestValue<T>();
  const T max_identity = hwy::NegativeInfOrLowestValue<T>();
  const Vec<D> min_identity_vec = Set(d, min_identity);
  const Vec<D> max_identity_vec = Set(d, max_identity);

  T best_min = min_identity;
  T best_max = max_identity;
  size_t best_min_idx = 0;
  size_t best_max_idx = 0;

  const size_t max_seg_len = kMaxBlocks * N;
  for (size_t seg = 0; seg < count; seg += max_seg_len) {
    const size_t seg_len = HWY_MIN(count - seg, max_seg_len);
    const T* HWY_RESTRICT seg_in = in + seg;

    Vec<D> min0 = min_identity_vec;
    Vec<D> min1 = min_identity_vec;
    Vec<D> max0 = max_identity_vec;
    Vec<D> max1 = max_identity_vec;
    VU min_blocks0 = Set(du, kInvBase);
    VU min_blocks1 = Set(du, kInvBase);
    VU max_blocks0 = Set(du, kInvBase);
    VU max_blocks1 = Set(du, kInvBase);

    size_t i = 0;
    TU inv_block = kInvBase;

    if (seg_len >= 2 * N) {
      for (; i + 2 * N <= seg_len; i += 2 * N) {
        const Vec<D> v0 = LoadU(d, seg_in + i);
        const Vec<D> v1 = LoadU(d, seg_in + i + N);
        const VU block0 = Set(du, inv_block);
        const VU block1 = Set(du, static_cast<TU>(inv_block - 1));

        const Mask<D> lt0 = Lt(v0, min0);
        const Mask<D> lt1 = Lt(v1, min1);
        const Mask<D> gt0 = Gt(v0, max0);
        const Mask<D> gt1 = Gt(v1, max1);
        min0 = IfThenElse(lt0, v0, min0);
        min1 = IfThenElse(lt1, v1, min1);
        max0 = IfThenElse(gt0, v0, max0);
        max1 = IfThenElse(gt1, v1, max1);
        min_blocks0 = IfThenElse(RebindMask(du, lt0), block0, min_blocks0);
        min_blocks1 = IfThenElse(RebindMask(du, lt1), block1, min_blocks1);
        max_blocks0 = IfThenElse(RebindMask(du, gt0), block0, max_blocks0);
        max_blocks1 = IfThenElse(RebindMask(du, gt1), block1, max_blocks1);
        inv_block = static_cast<TU>(inv_block - 2);
      }

      const Mask<D> min1_smaller = Lt(min1, min0);
      const Mask<D> min_tie = Eq(min0, min1);
      const Mask<D> min1_earlier = RebindMask(d, Gt(min_blocks1, min_blocks0));
      const Mask<D> min1_tie_earlier = And(min_tie, min1_earlier);
      const Mask<D> take_min1 = Or(min1_smaller, min1_tie_earlier);
      min0 = IfThenElse(take_min1, min1, min0);
      const MU take_min1_u = RebindMask(du, take_min1);
      min_blocks0 = IfThenElse(take_min1_u, min_blocks1, min_blocks0);

      const Mask<D> max1_larger = Gt(max1, max0);
      const Mask<D> max_tie = Eq(max0, max1);
      const Mask<D> max1_earlier = RebindMask(d, Gt(max_blocks1, max_blocks0));
      const Mask<D> max1_tie_earlier = And(max_tie, max1_earlier);
      const Mask<D> take_max1 = Or(max1_larger, max1_tie_earlier);
      max0 = IfThenElse(take_max1, max1, max0);
      const MU take_max1_u = RebindMask(du, take_max1);
      max_blocks0 = IfThenElse(take_max1_u, max_blocks1, max_blocks0);
    }

    for (; i < seg_len; i += N, --inv_block) {
      const size_t n = HWY_MIN(seg_len - i, N);
      const Vec<D> v = LoadNOr(min_identity_vec, d, seg_in + i, n);
      const VU block = Set(du, inv_block);
      const Mask<D> lt = Lt(v, min0);
      const Vec<D> v_for_max = IfThenElse(FirstN(d, n), v, max_identity_vec);
      const Mask<D> gt = Gt(v_for_max, max0);
      min0 = IfThenElse(lt, v, min0);
      max0 = IfThenElse(gt, v, max0);
      min_blocks0 = IfThenElse(RebindMask(du, lt), block, min_blocks0);
      max_blocks0 = IfThenElse(RebindMask(du, gt), block, max_blocks0);
    }

    const T seg_min = ReduceMin(d, min0);
    const Mask<D> is_min = Eq(min0, Set(d, seg_min));
    const MU is_min_u = RebindMask(du, is_min);
    const VU min_candidates = IfThenElseZero(is_min_u, min_blocks0);
    const TU min_inv = ReduceMax(du, min_candidates);
    const MU min_winners_u = MaskedEq(is_min_u, min_blocks0, Set(du, min_inv));
    const Mask<D> min_winners = RebindMask(d, min_winners_u);
    const size_t min_block = kMaxBlocks - static_cast<size_t>(min_inv);
    const size_t min_lane = FindKnownFirstTrue(d, min_winners);
    const size_t min_idx = seg + min_block * N + min_lane;

    const T seg_max = ReduceMax(d, max0);
    const Mask<D> is_max = Eq(max0, Set(d, seg_max));
    const MU is_max_u = RebindMask(du, is_max);
    const VU max_candidates = IfThenElseZero(is_max_u, max_blocks0);
    const TU max_inv = ReduceMax(du, max_candidates);
    const MU max_winners_u = MaskedEq(is_max_u, max_blocks0, Set(du, max_inv));
    const Mask<D> max_winners = RebindMask(d, max_winners_u);
    const size_t max_block = kMaxBlocks - static_cast<size_t>(max_inv);
    const size_t max_lane = FindKnownFirstTrue(d, max_winners);
    const size_t max_idx = seg + max_block * N + max_lane;

    if (seg_min < best_min) {
      best_min = seg_min;
      best_min_idx = min_idx;
    }
    if (seg_max > best_max) {
      best_max = seg_max;
      best_max_idx = max_idx;
    }
  }

  return {best_min_idx, best_max_idx};
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_ALGO_MINMAX_INL_H_
