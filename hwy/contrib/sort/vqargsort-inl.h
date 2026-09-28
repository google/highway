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

// Argsort, partial argsort and argselect without dynamic dispatch. The results
// match the functions of the same name (without Static) in vqargsort.h, which
// documents them. Implemented by packing each key and its index into one
// integer, sorting those with vqsort, then keeping the index.

// Normal include guard for target-independent parts
#ifndef HIGHWAY_HWY_CONTRIB_SORT_VQARGSORT_INL_H_
#define HIGHWAY_HWY_CONTRIB_SORT_VQARGSORT_INL_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>  // std::sort

#include "hwy/base.h"
#include "hwy/contrib/sort/order.h"  // SortAscending

namespace hwy {
namespace detail {

enum class ArgSortOp { kSort, kPartialSort, kSelect };

}  // namespace detail
}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_SORT_VQARGSORT_INL_H_

// Per-target
#if defined(HIGHWAY_HWY_CONTRIB_SORT_VQARGSORT_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_SORT_VQARGSORT_TOGGLE
#undef HIGHWAY_HWY_CONTRIB_SORT_VQARGSORT_TOGGLE
#else
#define HIGHWAY_HWY_CONTRIB_SORT_VQARGSORT_TOGGLE
#endif

#include "hwy/contrib/sort/vqsort-inl.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace detail {

using hwy::detail::ArgSortOp;

template <typename Key>
constexpr bool IsArgSortKey() {
  return sizeof(Key) >= 2 && (IsFloat<Key>() || IsIntegerLaneType<Key>());
}

// Returns bits that compare as unsigned integers in `Order`. NaN becomes the
// largest value, so it sorts last in either order, and -0 becomes +0.
template <typename Key, class Order, class DU>
HWY_INLINE VFromD<DU> OrderedKeyBits(DU du, VFromD<DU> bits) {
  using TU = TFromD<DU>;
  static_assert(sizeof(TU) == sizeof(Key), "Lane and key size must match");
  const VFromD<DU> sign = Set(du, SignMask<TU>());
  if constexpr (IsFloat<Key>()) {
    const RebindToSigned<DU> di;
    const VFromD<DU> abs = AndNot(sign, bits);
    const MFromD<DU> is_nan = Gt(abs, Set(du, ExponentMask<Key>()));
    bits = IfThenZeroElse(Eq(abs, Zero(du)), bits);
    const VFromD<DU> negative =
        BitCast(du, BroadcastSignBit(BitCast(di, bits)));
    bits = Xor(bits, Or(negative, sign));
    if constexpr (!Order::IsAscending()) bits = Not(bits);
    return Or(bits, VecFromMask(du, is_nan));
  } else {
    if constexpr (IsSigned<Key>()) bits = Xor(bits, sign);
    if constexpr (!Order::IsAscending()) bits = Not(bits);
    return bits;
  }
}

// For 16 and 32-bit keys: stores (key bits << 32) | index.
template <typename Key, class Order, class D64>
HWY_INLINE void PackKeys(D64 d64, const MakeUnsigned<Key>* HWY_RESTRICT bits,
                         VFromD<D64> index, uint64_t* HWY_RESTRICT packed) {
  const Rebind<MakeUnsigned<Key>, D64> du;
  const VFromD<decltype(du)> ordered =
      OrderedKeyBits<Key, Order>(du, LoadU(du, bits));
  StoreU(Or(ShiftLeft<32>(PromoteTo(d64, ordered)), index), d64, packed);
}

// For 64-bit keys: stores uint128_t with key bits in `hi` and index in `lo`.
template <typename Key, class Order, class D64>
HWY_INLINE void PackKeys(D64 d64, const uint64_t* HWY_RESTRICT bits,
                         VFromD<D64> index, uint128_t* HWY_RESTRICT packed) {
  const VFromD<D64> ordered = OrderedKeyBits<Key, Order>(d64, LoadU(d64, bits));
  StoreInterleaved2(index, ordered, d64, reinterpret_cast<uint64_t*>(packed));
}

template <class Order, typename Key, typename Packed>
void PackAllKeys(const Key* HWY_RESTRICT keys, size_t num,
                 Packed* HWY_RESTRICT packed) {
  using TU = MakeUnsigned<Key>;
  const TU* HWY_RESTRICT bits = reinterpret_cast<const TU*>(keys);
  const ScalableTag<uint64_t> d64;
  const size_t N = Lanes(d64);
  VFromD<decltype(d64)> index = Iota(d64, 0);
  const VFromD<decltype(d64)> step = Set(d64, static_cast<uint64_t>(N));
  size_t i = 0;
  if (num >= N) {
    for (; i <= num - N; i += N) {
      PackKeys<Key, Order>(d64, bits + i, index, packed + i);
      index = Add(index, step);
    }
  }
  const CappedTag<uint64_t, 1> d1;
  for (; i < num; ++i) {
    PackKeys<Key, Order>(d1, bits + i, Set(d1, static_cast<uint64_t>(i)),
                         packed + i);
  }
}

// Clears the key bits, leaving the index.
HWY_INLINE void KeepIndices(uint64_t* HWY_RESTRICT packed, size_t num) {
  const ScalableTag<uint64_t> d64;
  const size_t N = Lanes(d64);
  const VFromD<decltype(d64)> mask = Set(d64, uint64_t{0xFFFFFFFFu});
  size_t i = 0;
  if (num >= N) {
    for (; i <= num - N; i += N) {
      StoreU(And(LoadU(d64, packed + i), mask), d64, packed + i);
    }
  }
  for (; i < num; ++i) {
    packed[i] &= 0xFFFFFFFFu;
  }
}

HWY_INLINE void CopyIndices(const uint128_t* HWY_RESTRICT packed, size_t num,
                            uint64_t* HWY_RESTRICT indices) {
  const uint64_t* HWY_RESTRICT lanes =
      reinterpret_cast<const uint64_t*>(packed);
  const ScalableTag<uint64_t> d64;
  const size_t N = Lanes(d64);
  size_t i = 0;
  if (num >= N) {
    for (; i <= num - N; i += N) {
      VFromD<decltype(d64)> lo, hi;
      LoadInterleaved2(d64, lanes + 2 * i, lo, hi);
      StoreU(lo, d64, indices + i);
    }
  }
  for (; i < num; ++i) {
    indices[i] = packed[i].lo;
  }
}

// Sorts via VQSortStatic etc. on the current target. vqargsort.cc instead
// calls the precompiled VQSort etc.
struct VQSortStaticBackend {
  template <ArgSortOp kOp, typename T>
  static void Run(T* HWY_RESTRICT keys, size_t num, size_t k) {
    if constexpr (kOp == ArgSortOp::kSort) {
      (void)k;
      VQSortStatic(keys, num, SortAscending());
    } else if constexpr (kOp == ArgSortOp::kPartialSort) {
      VQPartialSortStatic(keys, num, k, SortAscending());
    } else {
      VQSelectStatic(keys, num, k, SortAscending());
    }
  }
};

// Returns false if there is nothing to order. VQPartialSort with k == 0 would
// still recurse with zero keys, which fails a debug assertion.
template <ArgSortOp kOp>
HWY_INLINE bool HaveWork(size_t num, size_t k) {
  if constexpr (kOp == ArgSortOp::kPartialSort) {
    HWY_DASSERT(k <= num);
    return k != 0;
  } else if constexpr (kOp == ArgSortOp::kSelect) {
    HWY_DASSERT(k < num);
  }
  (void)num;
  (void)k;
  return true;
}

// Stable: the whole integer is compared, so the index breaks ties. Unstable:
// K32V32 compares only the key, which lets vqsort finish equal keys early.
template <ArgSortOp kOp, bool kStable, class Backend>
HWY_INLINE void SortPacked(uint64_t* HWY_RESTRICT packed, size_t num,
                           size_t k) {
  if (!HaveWork<kOp>(num, k)) return;
  if constexpr (kStable) {
    Backend::template Run<kOp>(packed, num, k);
  } else {
    Backend::template Run<kOp>(reinterpret_cast<K32V32*>(packed), num, k);
  }
}

template <ArgSortOp kOp, bool kStable, class Backend>
HWY_INLINE void SortPacked(uint128_t* HWY_RESTRICT packed, size_t num,
                           size_t k) {
  if (!HaveWork<kOp>(num, k)) return;
#if HWY_TARGET == HWY_SCALAR
  // vqsort's 128-bit keys require SIMD. Comparing both halves is also a valid
  // unstable order.
  uint128_t* end = packed + num;
  uint128_t* kth = packed + HWY_MIN(k, num);
  if constexpr (kOp == ArgSortOp::kSort) {
    std::sort(packed, end);
  } else if constexpr (kOp == ArgSortOp::kPartialSort) {
    std::partial_sort(packed, kth, end);
  } else {
    std::nth_element(packed, kth, end);
  }
#else
  if constexpr (kStable) {
    Backend::template Run<kOp>(packed, num, k);
  } else {
    Backend::template Run<kOp>(reinterpret_cast<K64V64*>(packed), num, k);
  }
#endif
}

// For 16 and 32-bit keys, `indices` also holds the packed integers.
// `sort_packed(uint64_t*)` sorts, partially sorts or selects them.
template <class Order, typename Key, class SortPackedFunc>
void ArgSortWith(const Key* HWY_RESTRICT keys, size_t num,
                 uint64_t* HWY_RESTRICT indices,
                 const SortPackedFunc& sort_packed) {
  static_assert(IsArgSortKey<Key>(), "Unsupported key type");
  static_assert(sizeof(Key) <= 4, "64-bit keys also require `scratch`");
  if (num == 0) return;
  if constexpr (sizeof(size_t) > 4) {
    HWY_ASSERT(static_cast<uint64_t>(num) <= (uint64_t{1} << 32));
  }
  PackAllKeys<Order>(keys, num, indices);
  sort_packed(indices);
  KeepIndices(indices, num);
}

// For 64-bit keys. `sort_packed(uint128_t*)` is as above.
template <class Order, typename Key, class SortPackedFunc>
void ArgSortWith(const Key* HWY_RESTRICT keys, size_t num,
                 uint64_t* HWY_RESTRICT indices,
                 uint128_t* HWY_RESTRICT scratch,
                 const SortPackedFunc& sort_packed) {
  static_assert(IsArgSortKey<Key>(), "Unsupported key type");
  static_assert(sizeof(Key) == 8, "Only 64-bit keys use `scratch`");
  if (num == 0) return;
  PackAllKeys<Order>(keys, num, scratch);
  sort_packed(scratch);
  CopyIndices(scratch, num, indices);
}

template <ArgSortOp kOp, bool kStable, class Order, typename Key>
void ArgSortStatic(const Key* HWY_RESTRICT keys, size_t num, size_t k,
                   uint64_t* HWY_RESTRICT indices) {
  ArgSortWith<Order>(keys, num, indices, [num, k](uint64_t* packed) HWY_ATTR {
    SortPacked<kOp, kStable, VQSortStaticBackend>(packed, num, k);
  });
}

template <ArgSortOp kOp, bool kStable, class Order, typename Key>
void ArgSortStatic(const Key* HWY_RESTRICT keys, size_t num, size_t k,
                   uint64_t* HWY_RESTRICT indices,
                   uint128_t* HWY_RESTRICT scratch) {
  ArgSortWith<Order>(
      keys, num, indices, scratch, [num, k](uint128_t* packed) HWY_ATTR {
        SortPacked<kOp, kStable, VQSortStaticBackend>(packed, num, k);
      });
}

}  // namespace detail

// 16 and 32-bit keys: u16, i16, float16_t, u32, i32, float. Order is either
// SortAscending or SortDescending.

template <typename Key, class Order>
void VQArgSortStatic(const Key* HWY_RESTRICT keys, size_t n,
                     uint64_t* HWY_RESTRICT indices, Order) {
  detail::ArgSortStatic<detail::ArgSortOp::kSort, false, Order>(keys, n, 0,
                                                                indices);
}

template <typename Key, class Order>
void VQStableArgSortStatic(const Key* HWY_RESTRICT keys, size_t n,
                           uint64_t* HWY_RESTRICT indices, Order) {
  detail::ArgSortStatic<detail::ArgSortOp::kSort, true, Order>(keys, n, 0,
                                                               indices);
}

template <typename Key, class Order>
void VQArgPartialSortStatic(const Key* HWY_RESTRICT keys, size_t n, size_t k,
                            uint64_t* HWY_RESTRICT indices, Order) {
  detail::ArgSortStatic<detail::ArgSortOp::kPartialSort, false, Order>(
      keys, n, k, indices);
}

template <typename Key, class Order>
void VQStableArgPartialSortStatic(const Key* HWY_RESTRICT keys, size_t n,
                                  size_t k, uint64_t* HWY_RESTRICT indices,
                                  Order) {
  detail::ArgSortStatic<detail::ArgSortOp::kPartialSort, true, Order>(
      keys, n, k, indices);
}

template <typename Key, class Order>
void VQArgSelectStatic(const Key* HWY_RESTRICT keys, size_t n, size_t k,
                       uint64_t* HWY_RESTRICT indices, Order) {
  detail::ArgSortStatic<detail::ArgSortOp::kSelect, false, Order>(keys, n, k,
                                                                  indices);
}

template <typename Key, class Order>
void VQStableArgSelectStatic(const Key* HWY_RESTRICT keys, size_t n, size_t k,
                             uint64_t* HWY_RESTRICT indices, Order) {
  detail::ArgSortStatic<detail::ArgSortOp::kSelect, true, Order>(keys, n, k,
                                                                 indices);
}

// 64-bit keys: u64, i64, double. `scratch` has room for `n` entries.

template <typename Key, class Order>
void VQArgSortStatic(const Key* HWY_RESTRICT keys, size_t n,
                     uint64_t* HWY_RESTRICT indices,
                     uint128_t* HWY_RESTRICT scratch, Order) {
  detail::ArgSortStatic<detail::ArgSortOp::kSort, false, Order>(
      keys, n, 0, indices, scratch);
}

template <typename Key, class Order>
void VQStableArgSortStatic(const Key* HWY_RESTRICT keys, size_t n,
                           uint64_t* HWY_RESTRICT indices,
                           uint128_t* HWY_RESTRICT scratch, Order) {
  detail::ArgSortStatic<detail::ArgSortOp::kSort, true, Order>(
      keys, n, 0, indices, scratch);
}

template <typename Key, class Order>
void VQArgPartialSortStatic(const Key* HWY_RESTRICT keys, size_t n, size_t k,
                            uint64_t* HWY_RESTRICT indices,
                            uint128_t* HWY_RESTRICT scratch, Order) {
  detail::ArgSortStatic<detail::ArgSortOp::kPartialSort, false, Order>(
      keys, n, k, indices, scratch);
}

template <typename Key, class Order>
void VQStableArgPartialSortStatic(const Key* HWY_RESTRICT keys, size_t n,
                                  size_t k, uint64_t* HWY_RESTRICT indices,
                                  uint128_t* HWY_RESTRICT scratch, Order) {
  detail::ArgSortStatic<detail::ArgSortOp::kPartialSort, true, Order>(
      keys, n, k, indices, scratch);
}

template <typename Key, class Order>
void VQArgSelectStatic(const Key* HWY_RESTRICT keys, size_t n, size_t k,
                       uint64_t* HWY_RESTRICT indices,
                       uint128_t* HWY_RESTRICT scratch, Order) {
  detail::ArgSortStatic<detail::ArgSortOp::kSelect, false, Order>(
      keys, n, k, indices, scratch);
}

template <typename Key, class Order>
void VQStableArgSelectStatic(const Key* HWY_RESTRICT keys, size_t n, size_t k,
                             uint64_t* HWY_RESTRICT indices,
                             uint128_t* HWY_RESTRICT scratch, Order) {
  detail::ArgSortStatic<detail::ArgSortOp::kSelect, true, Order>(
      keys, n, k, indices, scratch);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_SORT_VQARGSORT_TOGGLE
