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

// The type-erased functions called by vqargsort.h, which pass each call on to
// the vqargsort_*.cc file for its key type and order.

#include "hwy/contrib/sort/vqargsort.h"

#include <stddef.h>
#include <stdint.h>

#include "hwy/contrib/sort/vqargsort-inl.h"  // ArgSortFunc

namespace hwy {
namespace detail {

namespace {

ArgSortFunc* ChooseArgSort(ArgSortKey key, bool ascending) {
  switch (key) {
    case ArgSortKey::kU16:
      return ascending ? ArgSortU16Asc : ArgSortU16Desc;
    case ArgSortKey::kI16:
      return ascending ? ArgSortI16Asc : ArgSortI16Desc;
    case ArgSortKey::kF16:
      return ascending ? ArgSortF16Asc : ArgSortF16Desc;
    case ArgSortKey::kU32:
      return ascending ? ArgSortU32Asc : ArgSortU32Desc;
    case ArgSortKey::kI32:
      return ascending ? ArgSortI32Asc : ArgSortI32Desc;
    case ArgSortKey::kF32:
      return ascending ? ArgSortF32Asc : ArgSortF32Desc;
    case ArgSortKey::kU64:
      return ascending ? ArgSortU64Asc : ArgSortU64Desc;
    case ArgSortKey::kI64:
      return ascending ? ArgSortI64Asc : ArgSortI64Desc;
    case ArgSortKey::kF64:
      return ascending ? ArgSortF64Asc : ArgSortF64Desc;
  }
  HWY_UNREACHABLE;
}

}  // namespace

void ArgSortErased(ArgSortKey key, const void* HWY_RESTRICT keys, size_t n,
                   uint64_t* HWY_RESTRICT indices,
                   uint128_t* HWY_RESTRICT scratch, bool ascending,
                   bool stable) {
  ChooseArgSort(key, ascending)(keys, n, /*k=*/0, indices, scratch,
                                ArgSortOp::kSort, stable);
}

void ArgPartialSortErased(ArgSortKey key, const void* HWY_RESTRICT keys,
                          size_t n, size_t k, uint64_t* HWY_RESTRICT indices,
                          uint128_t* HWY_RESTRICT scratch, bool ascending,
                          bool stable) {
  ChooseArgSort(key, ascending)(keys, n, k, indices, scratch,
                                ArgSortOp::kPartialSort, stable);
}

void ArgSelectErased(ArgSortKey key, const void* HWY_RESTRICT keys, size_t n,
                     size_t k, uint64_t* HWY_RESTRICT indices,
                     uint128_t* HWY_RESTRICT scratch, bool ascending,
                     bool stable) {
  ChooseArgSort(key, ascending)(keys, n, k, indices, scratch,
                                ArgSortOp::kSelect, stable);
}

}  // namespace detail
}  // namespace hwy
