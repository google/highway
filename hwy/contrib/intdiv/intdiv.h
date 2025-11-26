// Copyright 2024 Google LLC
// Copyright 2026 Fujitsu Limited
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: BSD-3-Clause
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

#ifndef HIGHWAY_HWY_CONTRIB_INTDIV_INTDIV_H_
#define HIGHWAY_HWY_CONTRIB_INTDIV_INTDIV_H_

#include "hwy/highway.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/intdiv/intdiv-inl.h"

#if HWY_ONCE

namespace hwy {

template <class D>
using VecD = HWY_NAMESPACE::Vec<D>;

template <class D>
using TFromD_ = HWY_NAMESPACE::TFromD<D>;

template <typename T>
using DivisorParamsU = HWY_NAMESPACE::DivisorParamsU<T>;

template <typename T>
using DivisorParamsS = HWY_NAMESPACE::DivisorParamsS<T>;

template <typename T, HWY_IF_SIGNED(T)>
HWY_INLINE DivisorParamsS<T> ComputeDivisorParams(T d) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "ComputeDivisorParams only supports 8/16/32/64-bit integers");
  return HWY_NAMESPACE::ComputeDivisorParams<T>(d);
}

template <typename T, HWY_IF_UNSIGNED(T)>
HWY_INLINE DivisorParamsU<T> ComputeDivisorParams(T d) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "ComputeDivisorParams only supports 8/16/32/64-bit integers");
  return HWY_NAMESPACE::ComputeDivisorParams<T>(d);
}

template <class D, class V = VecD<D>, typename T = TFromD_<D>>
HWY_INLINE V IntDiv(D d, V a, const DivisorParamsU<T>& p) {
  static_assert(!hwy::IsSigned<T>(), "DivisorParamsU requires unsigned lane type");
  return HWY_NAMESPACE::IntDiv(d, a, p);
}

template <class D, class V = VecD<D>, typename T = TFromD_<D>>
HWY_INLINE V IntDiv(D d, V a, const DivisorParamsS<T>& p) {
  static_assert(hwy::IsSigned<T>(), "DivisorParamsS requires signed lane type");
  return HWY_NAMESPACE::IntDiv(d, a, p);
}

template <class D, class V = VecD<D>, typename T = TFromD_<D>>
HWY_INLINE V IntDivFloor(D d, V a, const DivisorParamsU<T>& p) {
  static_assert(!hwy::IsSigned<T>(), "DivisorParamsU requires unsigned lane type");
  return HWY_NAMESPACE::IntDivFloor(d, a, p);
}
template <class D, class V = VecD<D>, typename T = TFromD_<D>>
HWY_INLINE V IntDivFloor(D d, V a, const DivisorParamsS<T>& p) {
  static_assert(hwy::IsSigned<T>(), "DivisorParamsS requires signed lane type");
  return HWY_NAMESPACE::IntDivFloor(d, a, p);
}

template <class D, class V = VecD<D>, typename T = TFromD_<D>, HWY_IF_UNSIGNED_D(D)>
HWY_INLINE V DivideByScalar(D d, V a, T div) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "DivideByScalar only supports 8/16/32/64-bit integers");
  return HWY_NAMESPACE::DivideByScalar(d, a, div);
}

template <class D, class V = VecD<D>, typename T = TFromD_<D>, HWY_IF_SIGNED_D(D)>
HWY_INLINE V DivideByScalar(D d, V a, T div) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "DivideByScalar only supports 8/16/32/64-bit integers");
  return HWY_NAMESPACE::DivideByScalar(d, a, div);
}

template <class D, class V = VecD<D>, typename T = TFromD_<D>, HWY_IF_UNSIGNED_D(D)>
HWY_INLINE V FloorDivideByScalar(D d, V a, T div) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "FloorDivideByScalar only supports 8/16/32/64-bit integers");
  return HWY_NAMESPACE::FloorDivideByScalar(d, a, div);
}

template <class D, class V = VecD<D>, typename T = TFromD_<D>, HWY_IF_SIGNED_D(D)>
HWY_INLINE V FloorDivideByScalar(D d, V a, T div) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "FloorDivideByScalar only supports 8/16/32/64-bit integers");
  return HWY_NAMESPACE::FloorDivideByScalar(d, a, div);
}

template <typename T, HWY_IF_UNSIGNED(T)>
HWY_INLINE void DivideArrayByScalar(T* HWY_RESTRICT arr, size_t n, T div) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "DivideArrayByScalar only supports 8/16/32/64-bit integers");
  HWY_NAMESPACE::DivideArrayByScalar(arr, n, div);
}

template <typename T, HWY_IF_SIGNED(T)>
HWY_INLINE void DivideArrayByScalar(T* HWY_RESTRICT arr, size_t n, T div) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "DivideArrayByScalar only supports 8/16/32/64-bit integers");
  HWY_NAMESPACE::DivideArrayByScalar(arr, n, div);
}

template <typename T, HWY_IF_SIGNED(T)>
HWY_INLINE void FloorDivideArrayByScalar(T* HWY_RESTRICT arr, size_t n, T div) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "FloorDivideArrayByScalar only supports 8/16/32/64-bit integers");
  HWY_NAMESPACE::FloorDivideArrayByScalar(arr, n, div);
}

template <typename T, HWY_IF_UNSIGNED(T)>
HWY_INLINE void FloorDivideArrayByScalar(T* HWY_RESTRICT arr, size_t n, T div) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "FloorDivideArrayByScalar only supports 8/16/32/64-bit integers");
  HWY_NAMESPACE::FloorDivideArrayByScalar(arr, n, div);
}

}  // namespace hwy

#endif  // HWY_ONCE

#endif  // HIGHWAY_HWY_CONTRIB_INTDIV_INTDIV_H_
