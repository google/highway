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

#include <stddef.h>
#include <stdint.h>

#include "hwy/contrib/sort/vqargsort.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/sort/vqargsort_f64d.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep

// After foreach_target
#include "hwy/contrib/sort/vqargsort-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

void PackF64Desc(const double* HWY_RESTRICT keys, size_t num,
                 uint128_t* HWY_RESTRICT packed) {
  detail::PackAllKeys<SortDescending>(keys, num, packed);
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_EXPORT(PackF64Desc);

void ArgSortF64Desc(const double* HWY_RESTRICT keys, size_t n, size_t k,
                    uint64_t* HWY_RESTRICT indices,
                    uint128_t* HWY_RESTRICT scratch, detail::ArgSortOp op,
                    bool stable) {
  HWY_DYNAMIC_DISPATCH(PackF64Desc)(keys, n, scratch);
  detail::ArgSortPacked(scratch, n, k, indices, op, stable);
}

}  // namespace

void VQArgSort(const double* HWY_RESTRICT keys, size_t n,
               uint64_t* HWY_RESTRICT indices, uint128_t* HWY_RESTRICT scratch,
               SortDescending) {
  ArgSortF64Desc(keys, n, 0, indices, scratch, detail::ArgSortOp::kSort,
                 /*stable=*/false);
}

void VQStableArgSort(const double* HWY_RESTRICT keys, size_t n,
                     uint64_t* HWY_RESTRICT indices,
                     uint128_t* HWY_RESTRICT scratch, SortDescending) {
  ArgSortF64Desc(keys, n, 0, indices, scratch, detail::ArgSortOp::kSort,
                 /*stable=*/true);
}

void VQArgPartialSort(const double* HWY_RESTRICT keys, size_t n, size_t k,
                      uint64_t* HWY_RESTRICT indices,
                      uint128_t* HWY_RESTRICT scratch, SortDescending) {
  ArgSortF64Desc(keys, n, k, indices, scratch, detail::ArgSortOp::kPartialSort,
                 /*stable=*/false);
}

void VQStableArgPartialSort(const double* HWY_RESTRICT keys, size_t n, size_t k,
                            uint64_t* HWY_RESTRICT indices,
                            uint128_t* HWY_RESTRICT scratch, SortDescending) {
  ArgSortF64Desc(keys, n, k, indices, scratch, detail::ArgSortOp::kPartialSort,
                 /*stable=*/true);
}

void VQArgSelect(const double* HWY_RESTRICT keys, size_t n, size_t k,
                 uint64_t* HWY_RESTRICT indices,
                 uint128_t* HWY_RESTRICT scratch, SortDescending) {
  ArgSortF64Desc(keys, n, k, indices, scratch, detail::ArgSortOp::kSelect,
                 /*stable=*/false);
}

void VQStableArgSelect(const double* HWY_RESTRICT keys, size_t n, size_t k,
                       uint64_t* HWY_RESTRICT indices,
                       uint128_t* HWY_RESTRICT scratch, SortDescending) {
  ArgSortF64Desc(keys, n, k, indices, scratch, detail::ArgSortOp::kSelect,
                 /*stable=*/true);
}

}  // namespace hwy
#endif  // HWY_ONCE
