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
#if defined(HIGHWAY_HWY_CONTRIB_ALGO_IS_SORTED_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_ALGO_IS_SORTED_INL_H_
#undef HIGHWAY_HWY_CONTRIB_ALGO_IS_SORTED_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_ALGO_IS_SORTED_INL_H_
#endif

#include <stddef.h>

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

namespace detail {

// Default comparator for IsSorted: strict less-than, which results in a
// check for non-decreasing order.
struct IsSortedLess {
  template <class D, class V>
  Mask<D> operator()(D /*d*/, V a, V b) const {
    return Lt(a, b);
  }
};

}  // namespace detail

// Returns true if `in[0, count)` is sorted with respect to `comp`: there is
// no i in [0, count - 1) for which `comp(d, in[i + 1], in[i])` is true.
// `comp(d, a, b)` returns a mask which is true in lanes where `a` is ordered
// strictly before `b`, matching the comparator semantics of std::is_sorted.
// For example, passing a comparator that returns `Gt(a, b)` checks for
// non-increasing (descending) order. A range of fewer than two elements is
// trivially sorted.
template <class D, class Comp, typename T = TFromD<D>>
bool IsSorted(D d, const T* HWY_RESTRICT in, size_t count, const Comp& comp) {
  if (count < 2) return true;
  HWY_LANES_CONSTEXPR size_t N = Lanes(d);

  // There are count - 1 adjacent pairs. Each step loads a vector and its
  // successors via an overlapping load offset by one element, so pairs that
  // straddle two steps are also covered.
  const size_t num_pairs = count - 1;

  size_t i = 0;
  if (HWY_LIKELY(num_pairs >= 2 * N)) {
    for (; i <= num_pairs - 2 * N; i += 2 * N) {
      const Mask<D> bad0 = comp(d, LoadU(d, in + i + 1), LoadU(d, in + i));
      const Mask<D> bad1 =
          comp(d, LoadU(d, in + i + N + 1), LoadU(d, in + i + N));
      if (HWY_UNLIKELY(!AllFalse(d, Or(bad0, bad1)))) return false;
    }
  }

  size_t remaining = num_pairs - i;
  if (remaining >= N) {
    const Mask<D> bad = comp(d, LoadU(d, in + i + 1), LoadU(d, in + i));
    if (!AllFalse(d, bad)) return false;
    i += N;
    remaining -= N;
  }

  HWY_DASSERT(remaining < N);
  if (remaining != 0) {
    // LoadN zero-pads the upper lanes; mask them out so that the result only
    // depends on the `remaining` valid pairs, whatever `comp` may be.
    const Mask<D> valid = FirstN(d, remaining);
    const Mask<D> bad = comp(d, LoadN(d, in + i + 1, remaining),
                             LoadN(d, in + i, remaining));
    if (!AllFalse(d, And(valid, bad))) return false;
  }

  return true;
}

// Returns true if `in[0, count)` is sorted in non-decreasing order, i.e.
// `in[i] <= in[i + 1]` for all i (equivalent to std::is_sorted).
template <class D, typename T = TFromD<D>>
bool IsSorted(D d, const T* HWY_RESTRICT in, size_t count) {
  return IsSorted(d, in, count, detail::IsSortedLess());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_ALGO_IS_SORTED_INL_H_
