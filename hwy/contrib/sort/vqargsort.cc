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

// Sorting and index extraction shared by the vqargsort_*.cc files, which only
// pack their key type.

#include <stddef.h>
#include <stdint.h>

#include "hwy/contrib/sort/vqsort.h"  // VQSort

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/sort/vqargsort.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep

// After foreach_target
#include "hwy/contrib/sort/vqargsort-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

using detail::ArgSortOp;

struct VQSortLibraryBackend {
  template <ArgSortOp kOp, typename T>
  static void Run(T* HWY_RESTRICT keys, size_t num, size_t k) {
    if constexpr (kOp == ArgSortOp::kSort) {
      (void)k;
      hwy::VQSort(keys, num, SortAscending());
    } else if constexpr (kOp == ArgSortOp::kPartialSort) {
      hwy::VQPartialSort(keys, num, k, SortAscending());
    } else {
      hwy::VQSelect(keys, num, k, SortAscending());
    }
  }
};

// `indices` is empty for uint64_t, whose indices stay in `packed`.
template <ArgSortOp kOp, typename Packed, typename... Indices>
void SortAndGetIndicesOp(bool stable, Packed* HWY_RESTRICT packed, size_t num,
                         size_t k, Indices... indices) {
  if (stable) {
    detail::SortAndGetIndices<kOp, true, VQSortLibraryBackend>(packed, num, k,
                                                               indices...);
  } else {
    detail::SortAndGetIndices<kOp, false, VQSortLibraryBackend>(packed, num, k,
                                                                indices...);
  }
}

template <typename Packed, typename... Indices>
void SortAndGetIndicesAnyOp(ArgSortOp op, bool stable,
                            Packed* HWY_RESTRICT packed, size_t num, size_t k,
                            Indices... indices) {
  switch (op) {
    case ArgSortOp::kSort:
      SortAndGetIndicesOp<ArgSortOp::kSort>(stable, packed, num, k, indices...);
      break;
    case ArgSortOp::kPartialSort:
      SortAndGetIndicesOp<ArgSortOp::kPartialSort>(stable, packed, num, k,
                                                   indices...);
      break;
    case ArgSortOp::kSelect:
      SortAndGetIndicesOp<ArgSortOp::kSelect>(stable, packed, num, k,
                                              indices...);
      break;
  }
}

void SortAndGetIndices64(uint64_t* HWY_RESTRICT packed, size_t num, size_t k,
                         ArgSortOp op, bool stable) {
  SortAndGetIndicesAnyOp(op, stable, packed, num, k);
}

void SortAndGetIndices128(uint128_t* HWY_RESTRICT packed, size_t num, size_t k,
                          uint64_t* HWY_RESTRICT indices, ArgSortOp op,
                          bool stable) {
  SortAndGetIndicesAnyOp(op, stable, packed, num, k, indices);
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_EXPORT(SortAndGetIndices64);
HWY_EXPORT(SortAndGetIndices128);
}  // namespace

namespace detail {

void ArgSortPacked(uint64_t* HWY_RESTRICT packed, size_t num, size_t k,
                   ArgSortOp op, bool stable) {
  HWY_DYNAMIC_DISPATCH(SortAndGetIndices64)(packed, num, k, op, stable);
}

void ArgSortPacked(uint128_t* HWY_RESTRICT packed, size_t num, size_t k,
                   uint64_t* HWY_RESTRICT indices, ArgSortOp op, bool stable) {
  HWY_DYNAMIC_DISPATCH(SortAndGetIndices128)(packed, num, k, indices, op,
                                             stable);
}

}  // namespace detail
}  // namespace hwy
#endif  // HWY_ONCE
