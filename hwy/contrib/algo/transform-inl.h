// Copyright 2022 Google LLC
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
// we 'rewind' to the array end minus the vector length, which overlaps the last
// full vector. Because Func is generally not idempotent (applying it twice
// might be different than doing so once - e.g. accumulating something into an
// array), we do not overwrite the overlapping output elements, but note that
// Func may be called for already updated, non-original elements of `inout`.

// Replaces `inout[idx]` with `func(d, inout[idx])`. Example usage: multiplying
// array elements by a constant.
// NOTE: Func may be called a second time for elements it has already updated,
// but the resulting output will not be written to `inout` again.
template <typename T, class Func>
void Transform(T* HWY_RESTRICT inout, size_t count, const Func& func) {
  const ScalableTag<T> d;
  using V = Vec<decltype(d)>;

  const size_t N = Lanes(d);

  size_t idx = 0;
  for (; idx + N <= count; idx += N) {
    const V v = LoadU(d, inout + idx);
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
  // Start index of the last unaligned whole vector, ending at the array end.
  const size_t last = count - N;
  // Number of elements before `from` or already written.
  const size_t invalid = idx - last;
  HWY_DASSERT(0 != invalid && invalid < N);
  const Mask<decltype(d)> mask = Not(FirstN(d, invalid));
  const V v = MaskedLoad(mask, d, inout + last);
  BlendedStore(func(d, v), mask, d, inout + last);
#endif
}

// Replaces `inout[idx]` with `func(d, inout[idx], in1[idx])`. Example usage:
// multiplying array elements by those of another array. NOTE: Func may be
// called a second time for elements it has already updated, but the resulting
// output will not be written to `inout` again.
template <typename T, class Func>
void Transform1(T* HWY_RESTRICT inout, size_t count, const T* HWY_RESTRICT in1,
                const Func& func) {
  const ScalableTag<T> d;
  using V = Vec<decltype(d)>;

  const size_t N = Lanes(d);

  size_t idx = 0;
  for (; idx + N <= count; idx += N) {
    const V v = LoadU(d, inout + idx);
    const V v1 = LoadU(d, in1 + idx);
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
  // Start index of the last unaligned whole vector, ending at the array end.
  const size_t last = count - N;
  // Number of elements before `from` or already written.
  const size_t invalid = idx - last;
  HWY_DASSERT(0 != invalid && invalid < N);
  const Mask<decltype(d)> mask = Not(FirstN(d, invalid));
  const V v = MaskedLoad(mask, d, inout + last);
  const V v1 = MaskedLoad(mask, d, in1 + last);
  BlendedStore(func(d, v, v1), mask, d, inout + last);
#endif
}

// Replaces `inout[idx]` with `func(d, inout[idx], in1[idx], in2[idx])`. Example
// usage: FMA of elements from three arrays, stored into the first array. NOTE:
// Func may be called a second time for elements it has already updated, but the
// resulting output will not be written to `inout` again.
template <typename T, class Func>
void Transform2(T* HWY_RESTRICT inout, size_t count, const T* HWY_RESTRICT in1,
                const T* HWY_RESTRICT in2, const Func& func) {
  const ScalableTag<T> d;
  using V = Vec<decltype(d)>;

  const size_t N = Lanes(d);

  size_t idx = 0;
  for (; idx + N <= count; idx += N) {
    const V v = LoadU(d, inout + idx);
    const V v1 = LoadU(d, in1 + idx);
    const V v2 = LoadU(d, in2 + idx);
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
  // Start index of the last unaligned whole vector, ending at the array end.
  const size_t last = count - N;
  // Number of elements before `from` or already written.
  const size_t invalid = idx - last;
  HWY_DASSERT(0 != invalid && invalid < N);
  const Mask<decltype(d)> mask = Not(FirstN(d, invalid));
  const V v = MaskedLoad(mask, d, inout + last);
  const V v1 = MaskedLoad(mask, d, in1 + last);
  const V v2 = MaskedLoad(mask, d, in2 + last);
  BlendedStore(func(d, v, v1, v2), mask, d, inout + last);
#endif
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_ALGO_TRANSFORM_INL_H_
