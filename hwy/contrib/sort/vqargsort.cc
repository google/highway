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

#include "hwy/contrib/sort/vqargsort.h"

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

// Calls the precompiled VQSort etc. instead of instantiating vqsort again.
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

template <ArgSortOp kOp, typename Packed>
void SortPackedLibrary(bool stable, Packed* HWY_RESTRICT packed, size_t num,
                       size_t k) {
  if (stable) {
    detail::SortPacked<kOp, true, VQSortLibraryBackend>(packed, num, k);
  } else {
    detail::SortPacked<kOp, false, VQSortLibraryBackend>(packed, num, k);
  }
}

template <class Order, typename Key>
void ArgSortLibrary(const Key* HWY_RESTRICT keys, size_t num, size_t k,
                    uint64_t* HWY_RESTRICT indices,
                    uint128_t* HWY_RESTRICT scratch, ArgSortOp op,
                    bool stable) {
  const auto sort_packed = [op, stable, num, k](auto* packed) HWY_ATTR {
    switch (op) {
      case ArgSortOp::kSort:
        SortPackedLibrary<ArgSortOp::kSort>(stable, packed, num, k);
        break;
      case ArgSortOp::kPartialSort:
        SortPackedLibrary<ArgSortOp::kPartialSort>(stable, packed, num, k);
        break;
      case ArgSortOp::kSelect:
        SortPackedLibrary<ArgSortOp::kSelect>(stable, packed, num, k);
        break;
    }
  };
  if constexpr (sizeof(Key) == 8) {
    detail::ArgSortWith<Order>(keys, num, indices, scratch, sort_packed);
  } else {
    (void)scratch;
    detail::ArgSortWith<Order>(keys, num, indices, sort_packed);
  }
}

// Per-target functions for one key type, exported for dynamic dispatch.
#define HWY_ARGSORT_PER_TARGET(KEY, NAME)                                      \
  void NAME##Asc(const KEY* HWY_RESTRICT keys, size_t num, size_t k,           \
                 uint64_t* HWY_RESTRICT indices,                               \
                 uint128_t* HWY_RESTRICT scratch, ArgSortOp op, bool stable) { \
    ArgSortLibrary<SortAscending>(keys, num, k, indices, scratch, op, stable); \
  }                                                                            \
  void NAME##Desc(const KEY* HWY_RESTRICT keys, size_t num, size_t k,          \
                  uint64_t* HWY_RESTRICT indices,                              \
                  uint128_t* HWY_RESTRICT scratch, ArgSortOp op,               \
                  bool stable) {                                               \
    ArgSortLibrary<SortDescending>(keys, num, k, indices, scratch, op,         \
                                   stable);                                    \
  }

HWY_ARGSORT_PER_TARGET(uint16_t, ArgSortU16)
HWY_ARGSORT_PER_TARGET(int16_t, ArgSortI16)
HWY_ARGSORT_PER_TARGET(float16_t, ArgSortF16)
HWY_ARGSORT_PER_TARGET(uint32_t, ArgSortU32)
HWY_ARGSORT_PER_TARGET(int32_t, ArgSortI32)
HWY_ARGSORT_PER_TARGET(float, ArgSortF32)
HWY_ARGSORT_PER_TARGET(uint64_t, ArgSortU64)
HWY_ARGSORT_PER_TARGET(int64_t, ArgSortI64)
HWY_ARGSORT_PER_TARGET(double, ArgSortF64)

#undef HWY_ARGSORT_PER_TARGET

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {

#define HWY_ARGSORT_EXPORT(NAME) \
  HWY_EXPORT(NAME##Asc);         \
  HWY_EXPORT(NAME##Desc);

HWY_ARGSORT_EXPORT(ArgSortU16)
HWY_ARGSORT_EXPORT(ArgSortI16)
HWY_ARGSORT_EXPORT(ArgSortF16)
HWY_ARGSORT_EXPORT(ArgSortU32)
HWY_ARGSORT_EXPORT(ArgSortI32)
HWY_ARGSORT_EXPORT(ArgSortF32)
HWY_ARGSORT_EXPORT(ArgSortU64)
HWY_ARGSORT_EXPORT(ArgSortI64)
HWY_ARGSORT_EXPORT(ArgSortF64)

#undef HWY_ARGSORT_EXPORT

}  // namespace

// Defines the six functions of vqargsort.h for one 16 or 32-bit key type and
// order. IMPL is the exported per-target function.
#define HWY_ARGSORT_DEFINE(KEY, ORDER, IMPL)                                  \
  void VQArgSort(const KEY* HWY_RESTRICT keys, size_t n,                      \
                 uint64_t* HWY_RESTRICT indices, ORDER) {                     \
    HWY_DYNAMIC_DISPATCH(IMPL)(keys, n, 0, indices, nullptr,                  \
                               detail::ArgSortOp::kSort, /*stable=*/false);   \
  }                                                                           \
  void VQStableArgSort(const KEY* HWY_RESTRICT keys, size_t n,                \
                       uint64_t* HWY_RESTRICT indices, ORDER) {               \
    HWY_DYNAMIC_DISPATCH(IMPL)(keys, n, 0, indices, nullptr,                  \
                               detail::ArgSortOp::kSort, /*stable=*/true);    \
  }                                                                           \
  void VQArgPartialSort(const KEY* HWY_RESTRICT keys, size_t n, size_t k,     \
                        uint64_t* HWY_RESTRICT indices, ORDER) {              \
    HWY_DYNAMIC_DISPATCH(IMPL)(keys, n, k, indices, nullptr,                  \
                               detail::ArgSortOp::kPartialSort,               \
                               /*stable=*/false);                             \
  }                                                                           \
  void VQStableArgPartialSort(const KEY* HWY_RESTRICT keys, size_t n,         \
                              size_t k, uint64_t* HWY_RESTRICT indices,       \
                              ORDER) {                                        \
    HWY_DYNAMIC_DISPATCH(IMPL)(keys, n, k, indices, nullptr,                  \
                               detail::ArgSortOp::kPartialSort,               \
                               /*stable=*/true);                              \
  }                                                                           \
  void VQArgSelect(const KEY* HWY_RESTRICT keys, size_t n, size_t k,          \
                   uint64_t* HWY_RESTRICT indices, ORDER) {                   \
    HWY_DYNAMIC_DISPATCH(IMPL)(keys, n, k, indices, nullptr,                  \
                               detail::ArgSortOp::kSelect, /*stable=*/false); \
  }                                                                           \
  void VQStableArgSelect(const KEY* HWY_RESTRICT keys, size_t n, size_t k,    \
                         uint64_t* HWY_RESTRICT indices, ORDER) {             \
    HWY_DYNAMIC_DISPATCH(IMPL)(keys, n, k, indices, nullptr,                  \
                               detail::ArgSortOp::kSelect, /*stable=*/true);  \
  }

// As above, for 64-bit keys, which also take `scratch`.
#define HWY_ARGSORT_DEFINE_SCRATCH(KEY, ORDER, IMPL)                          \
  void VQArgSort(const KEY* HWY_RESTRICT keys, size_t n,                      \
                 uint64_t* HWY_RESTRICT indices,                              \
                 uint128_t* HWY_RESTRICT scratch, ORDER) {                    \
    HWY_DYNAMIC_DISPATCH(IMPL)(keys, n, 0, indices, scratch,                  \
                               detail::ArgSortOp::kSort, /*stable=*/false);   \
  }                                                                           \
  void VQStableArgSort(const KEY* HWY_RESTRICT keys, size_t n,                \
                       uint64_t* HWY_RESTRICT indices,                        \
                       uint128_t* HWY_RESTRICT scratch, ORDER) {              \
    HWY_DYNAMIC_DISPATCH(IMPL)(keys, n, 0, indices, scratch,                  \
                               detail::ArgSortOp::kSort, /*stable=*/true);    \
  }                                                                           \
  void VQArgPartialSort(const KEY* HWY_RESTRICT keys, size_t n, size_t k,     \
                        uint64_t* HWY_RESTRICT indices,                       \
                        uint128_t* HWY_RESTRICT scratch, ORDER) {             \
    HWY_DYNAMIC_DISPATCH(IMPL)(keys, n, k, indices, scratch,                  \
                               detail::ArgSortOp::kPartialSort,               \
                               /*stable=*/false);                             \
  }                                                                           \
  void VQStableArgPartialSort(const KEY* HWY_RESTRICT keys, size_t n,         \
                              size_t k, uint64_t* HWY_RESTRICT indices,       \
                              uint128_t* HWY_RESTRICT scratch, ORDER) {       \
    HWY_DYNAMIC_DISPATCH(IMPL)(keys, n, k, indices, scratch,                  \
                               detail::ArgSortOp::kPartialSort,               \
                               /*stable=*/true);                              \
  }                                                                           \
  void VQArgSelect(const KEY* HWY_RESTRICT keys, size_t n, size_t k,          \
                   uint64_t* HWY_RESTRICT indices,                            \
                   uint128_t* HWY_RESTRICT scratch, ORDER) {                  \
    HWY_DYNAMIC_DISPATCH(IMPL)(keys, n, k, indices, scratch,                  \
                               detail::ArgSortOp::kSelect, /*stable=*/false); \
  }                                                                           \
  void VQStableArgSelect(const KEY* HWY_RESTRICT keys, size_t n, size_t k,    \
                         uint64_t* HWY_RESTRICT indices,                      \
                         uint128_t* HWY_RESTRICT scratch, ORDER) {            \
    HWY_DYNAMIC_DISPATCH(IMPL)(keys, n, k, indices, scratch,                  \
                               detail::ArgSortOp::kSelect, /*stable=*/true);  \
  }

HWY_ARGSORT_DEFINE(uint16_t, SortAscending, ArgSortU16Asc)
HWY_ARGSORT_DEFINE(uint16_t, SortDescending, ArgSortU16Desc)
HWY_ARGSORT_DEFINE(int16_t, SortAscending, ArgSortI16Asc)
HWY_ARGSORT_DEFINE(int16_t, SortDescending, ArgSortI16Desc)
HWY_ARGSORT_DEFINE(float16_t, SortAscending, ArgSortF16Asc)
HWY_ARGSORT_DEFINE(float16_t, SortDescending, ArgSortF16Desc)
HWY_ARGSORT_DEFINE(uint32_t, SortAscending, ArgSortU32Asc)
HWY_ARGSORT_DEFINE(uint32_t, SortDescending, ArgSortU32Desc)
HWY_ARGSORT_DEFINE(int32_t, SortAscending, ArgSortI32Asc)
HWY_ARGSORT_DEFINE(int32_t, SortDescending, ArgSortI32Desc)
HWY_ARGSORT_DEFINE(float, SortAscending, ArgSortF32Asc)
HWY_ARGSORT_DEFINE(float, SortDescending, ArgSortF32Desc)
HWY_ARGSORT_DEFINE_SCRATCH(uint64_t, SortAscending, ArgSortU64Asc)
HWY_ARGSORT_DEFINE_SCRATCH(uint64_t, SortDescending, ArgSortU64Desc)
HWY_ARGSORT_DEFINE_SCRATCH(int64_t, SortAscending, ArgSortI64Asc)
HWY_ARGSORT_DEFINE_SCRATCH(int64_t, SortDescending, ArgSortI64Desc)
HWY_ARGSORT_DEFINE_SCRATCH(double, SortAscending, ArgSortF64Asc)
HWY_ARGSORT_DEFINE_SCRATCH(double, SortDescending, ArgSortF64Desc)

#undef HWY_ARGSORT_DEFINE
#undef HWY_ARGSORT_DEFINE_SCRATCH

}  // namespace hwy
#endif  // HWY_ONCE
