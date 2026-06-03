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
      const Mask<D> eq0 = func(LoadU(d, in + i + 0 * N));
      const Mask<D> eq1 = func(LoadU(d, in + i + 1 * N));
      if (AllFalse(d, Or(eq0, eq1))) continue;
      const intptr_t pos = FindFirstTrue(d, eq0);
      if (pos >= 0) return i + static_cast<size_t>(pos);
      return i + N + FindKnownFirstTrue(d, eq1);
    }
  }

  size_t remaining = count - i;
  if (remaining >= N) {
    const Mask<D> eq = func(LoadU(d, in + i));
    const intptr_t pos = FindFirstTrue(d, eq);
    if (pos >= 0) return i + static_cast<size_t>(pos);
    i += N;
    remaining -= N;
  }

  HWY_DASSERT(remaining < N);
  if (remaining != 0) {
    const Mask<D> mask = FirstN(d, remaining);
    const Mask<D> eq = And(mask, func(LoadN(d, in + i, remaining)));
    const intptr_t pos = FindFirstTrue(d, eq);
    if (pos >= 0) return i + static_cast<size_t>(pos);
  }

  return count;  // not found
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_ALGO_FIND_INL_H_
