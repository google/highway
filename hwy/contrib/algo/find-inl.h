// Copyright 2022 Google LLC
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
#if defined(HIGHWAY_HWY_CONTRIB_ALGO_FIND_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_ALGO_FIND_INL_H_
#undef HIGHWAY_HWY_CONTRIB_ALGO_FIND_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_ALGO_FIND_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// Returns index of the first element equal to `value` in `in[0, count)`, or
// `count` if not found.
template <class D, typename T = TFromD<D>>
size_t Find(D d, T value, const T* HWY_RESTRICT in, size_t count) {
  HWY_LANES_CONSTEXPR size_t N = Lanes(d);
  const Vec<D> broadcasted = Set(d, value);

  size_t i = 0;
  if (HWY_LIKELY(count >= 2 * N)) {
    for (; i <= count - 2 * N; i += 2 * N) {
      const Mask<D> eq0 = Eq(broadcasted, LoadU(d, in + i + 0 * N));
      const Mask<D> eq1 = Eq(broadcasted, LoadU(d, in + i + 1 * N));
      if (AllFalse(d, Or(eq0, eq1))) continue;
      const intptr_t pos = FindFirstTrue(d, eq0);
      if (pos >= 0) return i + static_cast<size_t>(pos);
      return i + N + FindKnownFirstTrue(d, eq1);
    }
  }

  size_t remaining = count - i;
  if (remaining >= N) {
    const Mask<D> eq = Eq(broadcasted, LoadU(d, in + i));
    const intptr_t pos = FindFirstTrue(d, eq);
    if (pos >= 0) return i + static_cast<size_t>(pos);
    i += N;
    remaining -= N;
  }

  HWY_DASSERT(remaining < N);
  if (remaining != 0) {
    const Mask<D> mask = FirstN(d, remaining);
    const Mask<D> eq = MaskedEq(mask, broadcasted, LoadN(d, in + i, remaining));
    const intptr_t pos = FindFirstTrue(d, eq);
    if (pos >= 0) return i + static_cast<size_t>(pos);
  }

  return count;  // not found
}

// Returns index of the first element in `in[0, count)` for which `func(d, vec)`
// returns true, otherwise `count`.
template <class D, class Func, typename T = TFromD<D>>
size_t FindIf(D d, const T* HWY_RESTRICT in, size_t count, const Func& func) {
  HWY_LANES_CONSTEXPR size_t N = Lanes(d);

  size_t i = 0;
  if (HWY_LIKELY(count >= 2 * N)) {
    for (; i <= count - 2 * N; i += 2 * N) {
      const Mask<D> eq0 = func(d, LoadU(d, in + i + 0 * N));
      const Mask<D> eq1 = func(d, LoadU(d, in + i + 1 * N));
      if (AllFalse(d, Or(eq0, eq1))) continue;
      const intptr_t pos = FindFirstTrue(d, eq0);
      if (pos >= 0) return i + static_cast<size_t>(pos);
      return i + N + FindKnownFirstTrue(d, eq1);
    }
  }

  size_t remaining = count - i;
  if (remaining >= N) {
    const Mask<D> eq = func(d, LoadU(d, in + i));
    const intptr_t pos = FindFirstTrue(d, eq);
    if (pos >= 0) return i + static_cast<size_t>(pos);
    i += N;
    remaining -= N;
  }

  HWY_DASSERT(remaining < N);
  if (remaining != 0) {
    const Mask<D> mask = FirstN(d, remaining);
    const Mask<D> eq = And(mask, func(d, LoadN(d, in + i, remaining)));
    const intptr_t pos = FindFirstTrue(d, eq);
    if (pos >= 0) return i + static_cast<size_t>(pos);
  }

  return count;  // not found
}

// Like std::unique: removes consecutive duplicates in [in, in + count) and
// returns the number of unique elements. Requires sorted/grouped input.
// Operates in-place: the unique elements are packed to the front of `in`.
// Works for integer types. `count` may be zero.
template <class D, typename T = TFromD<D>, HWY_IF_NOT_FLOAT(T)>
size_t Unique(D d, T* HWY_RESTRICT in, size_t count) {
  if (HWY_UNLIKELY(count <= 1)) return count;

  HWY_LANES_CONSTEXPR size_t N = Lanes(d);

  // prev_last tracks the last element of the previous chunk for bridging
  // across vector boundaries. Initialize to bitwise complement of in[0]
  // to guarantee in[0] is always kept.
  Vec<D> prev_last =
      Set(d, static_cast<T>(~static_cast<MakeUnsigned<T>>(in[0])));

  size_t i = 0;    // read position
  size_t num = 0;  // write position = number of unique elements written

  // Main loop: process N elements at a time. CompressBlendedStore is used
  // to avoid overwriting elements beyond the written count, which is
  // necessary because the output overlaps the input (in-place).
  for (; i + N <= count; i += N) {
    const Vec<D> v = LoadU(d, in + i);

    // Shift v up by 1 lane and insert prev_last.
    const Vec<D> prev = SlideUpLanesOr(prev_last, d, v, 1);

    // Lanes where v[lane] != prev[lane] are unique (not a consecutive dup).
    const Mask<D> unique = Ne(v, prev);
    num += CompressBlendedStore(v, unique, d, in + num);

    prev_last = SlideDownLanes(d, v, N - 1);
  }

  // Remainder: fewer than N elements left.
  const size_t remaining = count - i;
  if (remaining != 0) {
    const Mask<D> mask = FirstN(d, remaining);
    const Vec<D> v = LoadN(d, in + i, remaining);
    const Vec<D> prev = SlideUpLanesOr(prev_last, d, v, 1);
    const Mask<D> unique = MaskedNe(mask, v, prev);
    num += CompressBlendedStore(v, unique, d, in + num);
  }

  return num;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_ALGO_FIND_INL_H_
