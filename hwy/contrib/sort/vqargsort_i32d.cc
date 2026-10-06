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

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/sort/vqargsort_i32d.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep

// After foreach_target
#include "hwy/contrib/sort/vqargsort-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

void ArgSortI32DescImpl(const void* HWY_RESTRICT keys, size_t n, size_t k,
                        uint64_t* HWY_RESTRICT indices,
                        uint128_t* HWY_RESTRICT scratch, detail::ArgSortOp op,
                        bool stable) {
  detail::ArgSortLibrary<SortDescending>(static_cast<const int32_t*>(keys), n,
                                         k, indices, scratch, op, stable);
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_EXPORT(ArgSortI32DescImpl);
}  // namespace

namespace detail {

void ArgSortI32Desc(const void* keys, size_t n, size_t k, uint64_t* indices,
                    uint128_t* scratch, ArgSortOp op, bool stable) {
  HWY_DYNAMIC_DISPATCH(ArgSortI32DescImpl)(keys, n, k, indices, scratch, op,
                                           stable);
}

}  // namespace detail
}  // namespace hwy
#endif  // HWY_ONCE
