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
HWY_CONTRIB_DLLEXPORT void VQArgSort(const uint16_t* HWY_RESTRICT keys,
                                     size_t n, uint64_t* HWY_RESTRICT indices,
                                     SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const uint16_t* HWY_RESTRICT keys,
                                     size_t n, uint64_t* HWY_RESTRICT indices,
                                     SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const int16_t* HWY_RESTRICT keys, size_t n,
                                     uint64_t* HWY_RESTRICT indices,
                                     SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const int16_t* HWY_RESTRICT keys, size_t n,
                                     uint64_t* HWY_RESTRICT indices,
                                     SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const float16_t* HWY_RESTRICT keys,
                                     size_t n, uint64_t* HWY_RESTRICT indices,
                                     SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const float16_t* HWY_RESTRICT keys,
                                     size_t n, uint64_t* HWY_RESTRICT indices,
                                     SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const uint32_t* HWY_RESTRICT keys,
                                     size_t n, uint64_t* HWY_RESTRICT indices,
                                     SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const uint32_t* HWY_RESTRICT keys,
                                     size_t n, uint64_t* HWY_RESTRICT indices,
                                     SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const int32_t* HWY_RESTRICT keys, size_t n,
                                     uint64_t* HWY_RESTRICT indices,
                                     SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const int32_t* HWY_RESTRICT keys, size_t n,
                                     uint64_t* HWY_RESTRICT indices,
                                     SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const float* HWY_RESTRICT keys, size_t n,
                                     uint64_t* HWY_RESTRICT indices,
                                     SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const float* HWY_RESTRICT keys, size_t n,
                                     uint64_t* HWY_RESTRICT indices,
                                     SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const uint64_t* HWY_RESTRICT keys,
                                     size_t n, uint64_t* HWY_RESTRICT indices,
                                     uint128_t* HWY_RESTRICT scratch,
                                     SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const uint64_t* HWY_RESTRICT keys,
                                     size_t n, uint64_t* HWY_RESTRICT indices,
                                     uint128_t* HWY_RESTRICT scratch,
                                     SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const int64_t* HWY_RESTRICT keys, size_t n,
                                     uint64_t* HWY_RESTRICT indices,
                                     uint128_t* HWY_RESTRICT scratch,
                                     SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const int64_t* HWY_RESTRICT keys, size_t n,
                                     uint64_t* HWY_RESTRICT indices,
                                     uint128_t* HWY_RESTRICT scratch,
                                     SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const double* HWY_RESTRICT keys, size_t n,
                                     uint64_t* HWY_RESTRICT indices,
                                     uint128_t* HWY_RESTRICT scratch,
                                     SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSort(const double* HWY_RESTRICT keys, size_t n,
                                     uint64_t* HWY_RESTRICT indices,
                                     uint128_t* HWY_RESTRICT scratch,
                                     SortDescending);

HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const uint16_t* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const uint16_t* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const int16_t* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const int16_t* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const float16_t* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const float16_t* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const uint32_t* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const uint32_t* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const int32_t* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const int32_t* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const float* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const float* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const uint64_t* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           uint128_t* HWY_RESTRICT scratch,
                                           SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const uint64_t* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           uint128_t* HWY_RESTRICT scratch,
                                           SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const int64_t* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           uint128_t* HWY_RESTRICT scratch,
                                           SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const int64_t* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           uint128_t* HWY_RESTRICT scratch,
                                           SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const double* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           uint128_t* HWY_RESTRICT scratch,
                                           SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSort(const double* HWY_RESTRICT keys,
                                           size_t n,
                                           uint64_t* HWY_RESTRICT indices,
                                           uint128_t* HWY_RESTRICT scratch,
                                           SortDescending);

// Partial argsort: sets indices[0, k) to the indices of the first k keys in the
// given order, in that order. indices[k, n) holds the other indices in
// unspecified order. Requires k <= n. The indices[0, k) of
// VQStableArgPartialSort match those of VQStableArgSort.
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const uint16_t* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const uint16_t* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const int16_t* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const int16_t* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const float16_t* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const float16_t* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const uint32_t* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const uint32_t* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const int32_t* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const int32_t* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const float* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const float* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const uint64_t* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            uint128_t* HWY_RESTRICT scratch,
                                            SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const uint64_t* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            uint128_t* HWY_RESTRICT scratch,
                                            SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const int64_t* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            uint128_t* HWY_RESTRICT scratch,
                                            SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const int64_t* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            uint128_t* HWY_RESTRICT scratch,
                                            SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const double* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            uint128_t* HWY_RESTRICT scratch,
                                            SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgPartialSort(const double* HWY_RESTRICT keys,
                                            size_t n, size_t k,
                                            uint64_t* HWY_RESTRICT indices,
                                            uint128_t* HWY_RESTRICT scratch,
                                            SortDescending);

HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const uint16_t* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const uint16_t* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const int16_t* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const int16_t* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const float16_t* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const float16_t* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const uint32_t* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const uint32_t* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const int32_t* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const int32_t* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const float* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const float* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const uint64_t* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, uint128_t* HWY_RESTRICT scratch,
    SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const uint64_t* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, uint128_t* HWY_RESTRICT scratch,
    SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const int64_t* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, uint128_t* HWY_RESTRICT scratch,
    SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const int64_t* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, uint128_t* HWY_RESTRICT scratch,
    SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const double* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, uint128_t* HWY_RESTRICT scratch,
    SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgPartialSort(
    const double* HWY_RESTRICT keys, size_t n, size_t k,
    uint64_t* HWY_RESTRICT indices, uint128_t* HWY_RESTRICT scratch,
    SortDescending);

// Argselect: sets indices[k] to the index of the key at position k of the given
// order. Keys at indices[0, k) are not ordered after it and keys at
// indices[k + 1, n) are not ordered before it. Requires k < n. The indices[k]
// of VQStableArgSelect matches that of VQStableArgSort.
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const uint16_t* HWY_RESTRICT keys,
                                       size_t n, size_t k,
                                       uint64_t* HWY_RESTRICT indices,
                                       SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const uint16_t* HWY_RESTRICT keys,
                                       size_t n, size_t k,
                                       uint64_t* HWY_RESTRICT indices,
                                       SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const int16_t* HWY_RESTRICT keys,
                                       size_t n, size_t k,
                                       uint64_t* HWY_RESTRICT indices,
                                       SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const int16_t* HWY_RESTRICT keys,
                                       size_t n, size_t k,
                                       uint64_t* HWY_RESTRICT indices,
                                       SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const float16_t* HWY_RESTRICT keys,
                                       size_t n, size_t k,
                                       uint64_t* HWY_RESTRICT indices,
                                       SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const float16_t* HWY_RESTRICT keys,
                                       size_t n, size_t k,
                                       uint64_t* HWY_RESTRICT indices,
                                       SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const uint32_t* HWY_RESTRICT keys,
                                       size_t n, size_t k,
                                       uint64_t* HWY_RESTRICT indices,
                                       SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const uint32_t* HWY_RESTRICT keys,
                                       size_t n, size_t k,
                                       uint64_t* HWY_RESTRICT indices,
                                       SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const int32_t* HWY_RESTRICT keys,
                                       size_t n, size_t k,
                                       uint64_t* HWY_RESTRICT indices,
                                       SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const int32_t* HWY_RESTRICT keys,
                                       size_t n, size_t k,
                                       uint64_t* HWY_RESTRICT indices,
                                       SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const float* HWY_RESTRICT keys, size_t n,
                                       size_t k, uint64_t* HWY_RESTRICT indices,
                                       SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const float* HWY_RESTRICT keys, size_t n,
                                       size_t k, uint64_t* HWY_RESTRICT indices,
                                       SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const uint64_t* HWY_RESTRICT keys,
                                       size_t n, size_t k,
                                       uint64_t* HWY_RESTRICT indices,
                                       uint128_t* HWY_RESTRICT scratch,
                                       SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const uint64_t* HWY_RESTRICT keys,
                                       size_t n, size_t k,
                                       uint64_t* HWY_RESTRICT indices,
                                       uint128_t* HWY_RESTRICT scratch,
                                       SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const int64_t* HWY_RESTRICT keys,
                                       size_t n, size_t k,
                                       uint64_t* HWY_RESTRICT indices,
                                       uint128_t* HWY_RESTRICT scratch,
                                       SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const int64_t* HWY_RESTRICT keys,
                                       size_t n, size_t k,
                                       uint64_t* HWY_RESTRICT indices,
                                       uint128_t* HWY_RESTRICT scratch,
                                       SortDescending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const double* HWY_RESTRICT keys,
                                       size_t n, size_t k,
                                       uint64_t* HWY_RESTRICT indices,
                                       uint128_t* HWY_RESTRICT scratch,
                                       SortAscending);
HWY_CONTRIB_DLLEXPORT void VQArgSelect(const double* HWY_RESTRICT keys,
                                       size_t n, size_t k,
                                       uint64_t* HWY_RESTRICT indices,
                                       uint128_t* HWY_RESTRICT scratch,
                                       SortDescending);

HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const uint16_t* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const uint16_t* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const int16_t* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const int16_t* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const float16_t* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const float16_t* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const uint32_t* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const uint32_t* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const int32_t* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const int32_t* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const float* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const float* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const uint64_t* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             uint128_t* HWY_RESTRICT scratch,
                                             SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const uint64_t* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             uint128_t* HWY_RESTRICT scratch,
                                             SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const int64_t* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             uint128_t* HWY_RESTRICT scratch,
                                             SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const int64_t* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             uint128_t* HWY_RESTRICT scratch,
                                             SortDescending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const double* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             uint128_t* HWY_RESTRICT scratch,
                                             SortAscending);
HWY_CONTRIB_DLLEXPORT void VQStableArgSelect(const double* HWY_RESTRICT keys,
                                             size_t n, size_t k,
                                             uint64_t* HWY_RESTRICT indices,
                                             uint128_t* HWY_RESTRICT scratch,
                                             SortDescending);

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_SORT_VQARGSORT_H_
