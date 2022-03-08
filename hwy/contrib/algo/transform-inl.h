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
#if defined(HIGHWAY_HWY_CONTRIB_ALGO_TRANSFORM_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_ALGO_TRANSFORM_INL_H_
#undef HIGHWAY_HWY_CONTRIB_ALGO_TRANSFORM_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_ALGO_TRANSFORM_INL_H_
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// These functions avoid having to write a loop plus remainder handling in the
// (unfortunately still common) case where arrays are not aligned/padded. If the
// inputs are known to be aligned/padded, it is more efficient to write a single
// loop using Load(). We do not provide a TransformAlignedPadded because it
// would be more verbose than such a loop.
//
// Func is either a functor with a templated operator()(d, v[, v1[, v2]]), or a
// generic lambda if using C++14. Due to apparent limitations of Clang on
// Windows, it is currently necessary to add HWY_ATTR before the opening { of
// the lambda to avoid errors about "always_inline function .. requires target".
//
// If HWY_MEM_OPS_MIGHT_FAULT, we use scalar code instead of masking. Otherwise,
// we used `MaskedLoad` and `BlendedStore` to read/write the final partial
// vector.

// Replaces `inout[idx]` with `func(d, inout[idx])`. Example usage: multiplying
// array elements by a constant.
template <class D, class Func, typename T = TFromD<D>>
void Transform(D d, T* HWY_RESTRICT inout, size_t count, const Func& func) {
  const size_t N = Lanes(d);

  size_t idx = 0;
  for (; idx + N <= count; idx += N) {
    const Vec<D> v = LoadU(d, inout + idx);
    StoreU(func(d, v), d, inout + idx);
  }

  // `count` was a multiple of the vector length `N`: already done.
  if (HWY_UNLIKELY(idx == count)) return;

#if HWY_MEM_OPS_MIGHT_FAULT
  // Proceed one by one.
  const CappedTag<T, 1> d1;
  for (; idx < count; ++idx) {
    using V1 = Vec<decltype(d1)>;
    const V1 v = LoadU(d1, inout + idx);
    StoreU(func(d1, v), d1, inout + idx);
  }
#else
  const size_t remaining = count - idx;
  HWY_DASSERT(0 != remaining && remaining < N);
  const Mask<D> mask = FirstN(d, remaining);
  const Vec<D> v = MaskedLoad(mask, d, inout + idx);
  BlendedStore(func(d, v), mask, d, inout + idx);
#endif
}

// Replaces `inout[idx]` with `func(d, inout[idx], in1[idx])`. Example usage:
// multiplying array elements by those of another array.
template <class D, class Func, typename T = TFromD<D>>
void Transform1(D d, T* HWY_RESTRICT inout, size_t count,
                const T* HWY_RESTRICT in1, const Func& func) {
  const size_t N = Lanes(d);

  size_t idx = 0;
  for (; idx + N <= count; idx += N) {
    const Vec<D> v = LoadU(d, inout + idx);
    const Vec<D> v1 = LoadU(d, in1 + idx);
    StoreU(func(d, v, v1), d, inout + idx);
  }

  // `count` was a multiple of the vector length `N`: already done.
  if (HWY_UNLIKELY(idx == count)) return;

#if HWY_MEM_OPS_MIGHT_FAULT
  // Proceed one by one.
  const CappedTag<T, 1> d1;
  for (; idx < count; ++idx) {
    using V1 = Vec<decltype(d1)>;
    const V1 v = LoadU(d1, inout + idx);
    const V1 v1 = LoadU(d1, in1 + idx);
    StoreU(func(d1, v, v1), d1, inout + idx);
  }
#else
  const size_t remaining = count - idx;
  HWY_DASSERT(0 != remaining && remaining < N);
  const Mask<D> mask = FirstN(d, remaining);
  const Vec<D> v = MaskedLoad(mask, d, inout + idx);
  const Vec<D> v1 = MaskedLoad(mask, d, in1 + idx);
  BlendedStore(func(d, v, v1), mask, d, inout + idx);
#endif
}

// Replaces `inout[idx]` with `func(d, inout[idx], in1[idx], in2[idx])`. Example
// usage: FMA of elements from three arrays, stored into the first array.
template <class D, class Func, typename T = TFromD<D>>
void Transform2(D d, T* HWY_RESTRICT inout, size_t count,
                const T* HWY_RESTRICT in1, const T* HWY_RESTRICT in2,
                const Func& func) {
  const size_t N = Lanes(d);

  size_t idx = 0;
  for (; idx + N <= count; idx += N) {
    const Vec<D> v = LoadU(d, inout + idx);
    const Vec<D> v1 = LoadU(d, in1 + idx);
    const Vec<D> v2 = LoadU(d, in2 + idx);
    StoreU(func(d, v, v1, v2), d, inout + idx);
  }

  // `count` was a multiple of the vector length `N`: already done.
  if (HWY_UNLIKELY(idx == count)) return;

#if HWY_MEM_OPS_MIGHT_FAULT
  // Proceed one by one.
  const CappedTag<T, 1> d1;
  for (; idx < count; ++idx) {
    using V1 = Vec<decltype(d1)>;
    const V1 v = LoadU(d1, inout + idx);
    const V1 v1 = LoadU(d1, in1 + idx);
    const V1 v2 = LoadU(d1, in2 + idx);
    StoreU(func(d1, v, v1, v2), d1, inout + idx);
  }
#else
  const size_t remaining = count - idx;
  HWY_DASSERT(0 != remaining && remaining < N);
  const Mask<D> mask = FirstN(d, remaining);
  const Vec<D> v = MaskedLoad(mask, d, inout + idx);
  const Vec<D> v1 = MaskedLoad(mask, d, in1 + idx);
  const Vec<D> v2 = MaskedLoad(mask, d, in2 + idx);
  BlendedStore(func(d, v, v1, v2), mask, d, inout + idx);
#endif
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_ALGO_TRANSFORM_INL_H_
