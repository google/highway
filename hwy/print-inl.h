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

// Print() function

#include <inttypes.h>
#include <stdint.h>

#include "hwy/base.h"
#include "hwy/tests/test_util.h"

// Per-target include guard
#if defined(HIGHWAY_HWY_PRINT_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_PRINT_INL_H_
#undef HIGHWAY_HWY_PRINT_INL_H_
#else
#define HIGHWAY_HWY_PRINT_INL_H_
#endif

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

template <typename T, HWY_IF_LANE_SIZE(T, 1)>
HWY_NOINLINE void PrintValue(T value) {
  uint8_t byte;
  CopyBytes<1>(&value, &byte);  // endian-safe: we ensured sizeof(T)=1.
  fprintf(stderr, "0x%02X,", byte);
}

#if HWY_HAVE_FLOAT16
HWY_NOINLINE void PrintValue(float16_t value) {
  uint16_t bits;
  CopyBytes<2>(&value, &bits);
  fprintf(stderr, "0x%02X,", bits);
}
#endif

template <typename T, HWY_IF_NOT_LANE_SIZE(T, 1)>
HWY_NOINLINE void PrintValue(T value) {
  fprintf(stderr, "%g,", double(value));
}

// Prints lanes around `lane`, in memory order.
template <class D, class V = Vec<D>>
void Print(const D d, const char* caption, VecArg<V> v, size_t lane_u = 0,
           size_t max_lanes = 7) {
  using T = TFromD<D>;
  const size_t N = Lanes(d);
  auto lanes = AllocateAligned<T>(N);
  Store(v, d, lanes.get());

  const auto info = hwy::detail::MakeTypeInfo<T>();
  hwy::detail::PrintArray(info, caption, lanes.get(), N, lane_u, max_lanes);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // per-target include guard
