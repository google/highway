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

// Argsort, partial argsort and argselect with dynamic dispatch. For static
// dispatch without any DLLEXPORT, avoid including this header and instead
// define VQSORT_ONLY_STATIC, then call the functions of the same name with a
// Static suffix in vqargsort-inl.h.

#ifndef HIGHWAY_HWY_CONTRIB_SORT_VQARGSORT_H_
#define HIGHWAY_HWY_CONTRIB_SORT_VQARGSORT_H_

// IWYU pragma: begin_exports
#include <stddef.h>
#include <stdint.h>

#include "hwy/base.h"
#include "hwy/contrib/sort/order.h"  // SortAscending
// IWYU pragma: end_exports

namespace hwy {

namespace detail {

// Key types, for the type-erased functions below.
enum class ArgSortKey : uint8_t {
  kU16,
  kI16,
  kF16,
  kU32,
  kI32,
  kF32,
  kU64,
  kI64,
  kF64
};

// Only 64-bit keys take `scratch`.
template <typename Key, bool kHasScratch>
constexpr ArgSortKey ArgSortKeyOf() {
  static_assert(
      sizeof(Key) >= 2 && (IsFloat<Key>() || IsIntegerLaneType<Key>()),
      "Keys must be 16, 32 or 64-bit integers, float16_t, float or double");
  static_assert(kHasScratch == (sizeof(Key) == 8),
                "64-bit keys require scratch, other keys do not take it");
  if constexpr (IsFloat<Key>()) {
    return sizeof(Key) == 2   ? ArgSortKey::kF16
           : sizeof(Key) == 4 ? ArgSortKey::kF32
                              : ArgSortKey::kF64;
  } else if constexpr (IsSigned<Key>()) {
    return sizeof(Key) == 2   ? ArgSortKey::kI16
           : sizeof(Key) == 4 ? ArgSortKey::kI32
                              : ArgSortKey::kI64;
  } else {
    return sizeof(Key) == 2   ? ArgSortKey::kU16
           : sizeof(Key) == 4 ? ArgSortKey::kU32
                              : ArgSortKey::kU64;
  }
}

// `scratch` is null for 16 and 32-bit keys.
HWY_CONTRIB_DLLEXPORT void ArgSortErased(ArgSortKey key,
                                         const void* HWY_RESTRICT keys,
                                         size_t n,
                                         uint64_t* HWY_RESTRICT indices,
                                         uint128_t* HWY_RESTRICT scratch,
                                         bool ascending, bool stable);
HWY_CONTRIB_DLLEXPORT void ArgPartialSortErased(ArgSortKey key,
                                                const void* HWY_RESTRICT keys,
                                                size_t n, size_t k,
                                                uint64_t* HWY_RESTRICT indices,
                                                uint128_t* HWY_RESTRICT scratch,
                                                bool ascending, bool stable);
HWY_CONTRIB_DLLEXPORT void ArgSelectErased(ArgSortKey key,
                                           const void* HWY_RESTRICT keys,
                                           size_t n, size_t k,
                                           uint64_t* HWY_RESTRICT indices,
                                           uint128_t* HWY_RESTRICT scratch,
                                           bool ascending, bool stable);

}  // namespace detail

// All functions below write the order of keys[0, n) to `indices` and do not
// modify `keys`. Every index in [0, n) appears once in indices[0, n).
// Dispatches to the best available instruction set. Does not allocate memory.
//
// VQStable* keep equivalent keys (neither greater nor less than another) in
// their original order. The others may reorder them, which allows vqsort to
// finish runs of equivalent keys earlier.
//
// Floating-point keys: NaN are ordered last in either order, and -0 and +0 are
// equivalent. Unlike VQSort, float16_t and double keys do not require
// VQSortHaveFloat16/64.
//
// 16 and 32-bit keys: `indices` is also used as scratch memory. Requires
// n <= 2^32.
// 64-bit keys: `scratch` must have room for `n` entries.

// Argsort: sets indices[0, n) such that keys[indices[0]], keys[indices[1]], ...
// are in the given order.
template <typename Key, class Order>
void VQArgSort(const Key* HWY_RESTRICT keys, size_t n,
               uint64_t* HWY_RESTRICT indices, Order) {
  detail::ArgSortErased(detail::ArgSortKeyOf<Key, /*kHasScratch=*/false>(),
                        keys, n, indices, nullptr, Order::IsAscending(),
                        /*stable=*/false);
}

template <typename Key, class Order>
void VQArgSort(const Key* HWY_RESTRICT keys, size_t n,
               uint64_t* HWY_RESTRICT indices, uint128_t* HWY_RESTRICT scratch,
               Order) {
  detail::ArgSortErased(detail::ArgSortKeyOf<Key, /*kHasScratch=*/true>(), keys,
                        n, indices, scratch, Order::IsAscending(),
                        /*stable=*/false);
}

template <typename Key, class Order>
void VQStableArgSort(const Key* HWY_RESTRICT keys, size_t n,
                     uint64_t* HWY_RESTRICT indices, Order) {
  detail::ArgSortErased(detail::ArgSortKeyOf<Key, /*kHasScratch=*/false>(),
                        keys, n, indices, nullptr, Order::IsAscending(),
                        /*stable=*/true);
}

template <typename Key, class Order>
void VQStableArgSort(const Key* HWY_RESTRICT keys, size_t n,
                     uint64_t* HWY_RESTRICT indices,
                     uint128_t* HWY_RESTRICT scratch, Order) {
  detail::ArgSortErased(detail::ArgSortKeyOf<Key, /*kHasScratch=*/true>(), keys,
                        n, indices, scratch, Order::IsAscending(),
                        /*stable=*/true);
}

// Partial argsort: sets indices[0, k) to the indices of the first k keys in the
// given order, in that order. indices[k, n) holds the other indices in
// unspecified order. Requires k <= n. The indices[0, k) of
// VQStableArgPartialSort match those of VQStableArgSort.
template <typename Key, class Order>
void VQArgPartialSort(const Key* HWY_RESTRICT keys, size_t n, size_t k,
                      uint64_t* HWY_RESTRICT indices, Order) {
  detail::ArgPartialSortErased(
      detail::ArgSortKeyOf<Key, /*kHasScratch=*/false>(), keys, n, k, indices,
      nullptr, Order::IsAscending(), /*stable=*/false);
}

template <typename Key, class Order>
void VQArgPartialSort(const Key* HWY_RESTRICT keys, size_t n, size_t k,
                      uint64_t* HWY_RESTRICT indices,
                      uint128_t* HWY_RESTRICT scratch, Order) {
  detail::ArgPartialSortErased(
      detail::ArgSortKeyOf<Key, /*kHasScratch=*/true>(), keys, n, k, indices,
      scratch, Order::IsAscending(), /*stable=*/false);
}

template <typename Key, class Order>
void VQStableArgPartialSort(const Key* HWY_RESTRICT keys, size_t n, size_t k,
                            uint64_t* HWY_RESTRICT indices, Order) {
  detail::ArgPartialSortErased(
      detail::ArgSortKeyOf<Key, /*kHasScratch=*/false>(), keys, n, k, indices,
      nullptr, Order::IsAscending(), /*stable=*/true);
}

template <typename Key, class Order>
void VQStableArgPartialSort(const Key* HWY_RESTRICT keys, size_t n, size_t k,
                            uint64_t* HWY_RESTRICT indices,
                            uint128_t* HWY_RESTRICT scratch, Order) {
  detail::ArgPartialSortErased(
      detail::ArgSortKeyOf<Key, /*kHasScratch=*/true>(), keys, n, k, indices,
      scratch, Order::IsAscending(), /*stable=*/true);
}

// Argselect: sets indices[k] to the index of the key at position k of the given
// order. Keys at indices[0, k) are not ordered after it and keys at
// indices[k + 1, n) are not ordered before it. Requires k < n. The indices[k]
// of VQStableArgSelect matches that of VQStableArgSort.
template <typename Key, class Order>
void VQArgSelect(const Key* HWY_RESTRICT keys, size_t n, size_t k,
                 uint64_t* HWY_RESTRICT indices, Order) {
  detail::ArgSelectErased(detail::ArgSortKeyOf<Key, /*kHasScratch=*/false>(),
                          keys, n, k, indices, nullptr, Order::IsAscending(),
                          /*stable=*/false);
}

template <typename Key, class Order>
void VQArgSelect(const Key* HWY_RESTRICT keys, size_t n, size_t k,
                 uint64_t* HWY_RESTRICT indices,
                 uint128_t* HWY_RESTRICT scratch, Order) {
  detail::ArgSelectErased(detail::ArgSortKeyOf<Key, /*kHasScratch=*/true>(),
                          keys, n, k, indices, scratch, Order::IsAscending(),
                          /*stable=*/false);
}

template <typename Key, class Order>
void VQStableArgSelect(const Key* HWY_RESTRICT keys, size_t n, size_t k,
                       uint64_t* HWY_RESTRICT indices, Order) {
  detail::ArgSelectErased(detail::ArgSortKeyOf<Key, /*kHasScratch=*/false>(),
                          keys, n, k, indices, nullptr, Order::IsAscending(),
                          /*stable=*/true);
}

template <typename Key, class Order>
void VQStableArgSelect(const Key* HWY_RESTRICT keys, size_t n, size_t k,
                       uint64_t* HWY_RESTRICT indices,
                       uint128_t* HWY_RESTRICT scratch, Order) {
  detail::ArgSelectErased(detail::ArgSortKeyOf<Key, /*kHasScratch=*/true>(),
                          keys, n, k, indices, scratch, Order::IsAscending(),
                          /*stable=*/true);
}

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_SORT_VQARGSORT_H_
