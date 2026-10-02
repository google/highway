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
#define HWY_TARGET_INCLUDE "hwy/contrib/sort/vqargsort_f32d.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep

// After foreach_target
#include "hwy/contrib/sort/vqargsort-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

void PackF32Desc(const float* HWY_RESTRICT keys, size_t num,
                 uint64_t* HWY_RESTRICT packed) {
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
HWY_EXPORT(PackF32Desc);

void ArgSortF32Desc(const float* HWY_RESTRICT keys, size_t n, size_t k,
                    uint64_t* HWY_RESTRICT indices, detail::ArgSortOp op,
                    bool stable) {
  HWY_DYNAMIC_DISPATCH(PackF32Desc)(keys, n, indices);
  detail::ArgSortPacked(indices, n, k, op, stable);
}

}  // namespace

void VQArgSort(const float* HWY_RESTRICT keys, size_t n,
               uint64_t* HWY_RESTRICT indices, SortDescending) {
  ArgSortF32Desc(keys, n, 0, indices, detail::ArgSortOp::kSort,
                 /*stable=*/false);
}

void VQStableArgSort(const float* HWY_RESTRICT keys, size_t n,
                     uint64_t* HWY_RESTRICT indices, SortDescending) {
  ArgSortF32Desc(keys, n, 0, indices, detail::ArgSortOp::kSort,
                 /*stable=*/true);
}

void VQArgPartialSort(const float* HWY_RESTRICT keys, size_t n, size_t k,
                      uint64_t* HWY_RESTRICT indices, SortDescending) {
  ArgSortF32Desc(keys, n, k, indices, detail::ArgSortOp::kPartialSort,
                 /*stable=*/false);
}

void VQStableArgPartialSort(const float* HWY_RESTRICT keys, size_t n, size_t k,
                            uint64_t* HWY_RESTRICT indices, SortDescending) {
  ArgSortF32Desc(keys, n, k, indices, detail::ArgSortOp::kPartialSort,
                 /*stable=*/true);
}

void VQArgSelect(const float* HWY_RESTRICT keys, size_t n, size_t k,
                 uint64_t* HWY_RESTRICT indices, SortDescending) {
  ArgSortF32Desc(keys, n, k, indices, detail::ArgSortOp::kSelect,
                 /*stable=*/false);
}

void VQStableArgSelect(const float* HWY_RESTRICT keys, size_t n, size_t k,
                       uint64_t* HWY_RESTRICT indices, SortDescending) {
  ArgSortF32Desc(keys, n, k, indices, detail::ArgSortOp::kSelect,
                 /*stable=*/true);
}

}  // namespace hwy
#endif  // HWY_ONCE
