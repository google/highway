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
#include <string.h>  // memcmp

#include <algorithm>
#include <numeric>  // std::iota
#include <vector>

#include "hwy/aligned_allocator.h"  // AlignedVector
#include "hwy/base.h"

// After base.h, which defines HWY_IS_DEBUG_BUILD.
#if !defined(HWY_DISABLED_TARGETS) && HWY_IS_DEBUG_BUILD
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3)
#endif

#include "hwy/contrib/sort/vqargsort.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/sort/argsort_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "hwy/contrib/sort/vqargsort-inl.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

using detail::ArgSortOp;

// Reference order from ordinary comparisons, independent of the key conversion
// in vqargsort-inl.h: NaN last in either order, -0 equivalent to +0.
template <class Order, typename Key>
bool KeyLess(Key a, Key b) {
  if constexpr (IsFloat<Key>()) {
    const double da = ConvertScalarTo<double>(a);
    const double db = ConvertScalarTo<double>(b);
    if (ScalarIsNaN(da)) return false;
    if (ScalarIsNaN(db)) return true;
    return Order::IsAscending() ? da < db : db < da;
  } else {
    return Order::IsAscending() ? a < b : b < a;
  }
}

template <class Order, typename Key>
std::vector<uint64_t> ReferenceOrder(const std::vector<Key>& keys) {
  std::vector<uint64_t> order(keys.size());
  std::iota(order.begin(), order.end(), uint64_t{0});
  std::stable_sort(order.begin(), order.end(), [&keys](uint64_t a, uint64_t b) {
    return KeyLess<Order>(keys[static_cast<size_t>(a)],
                          keys[static_cast<size_t>(b)]);
  });
  return order;
}

// Integer limits and their neighbors. For floats: zero, subnormals, inf, NaN
// (several payloads), the largest finite value and 1, each with both signs.
template <typename Key>
std::vector<MakeUnsigned<Key>> SpecialBits() {
  using TU = MakeUnsigned<Key>;
  const TU sign = SignMask<TU>();
  std::vector<TU> bits = {TU{0},
                          TU{1},
                          static_cast<TU>(sign - 1),
                          sign,
                          static_cast<TU>(sign + 1),
                          LimitsMax<TU>()};
  if constexpr (IsFloat<Key>()) {
    const TU inf = ExponentMask<Key>();
    const TU one = BitCastScalar<TU>(ConvertScalarTo<Key>(1.0f));
    for (const TU b :
         {inf, static_cast<TU>(inf - 1), static_cast<TU>(inf + 1), one}) {
      bits.push_back(b);
      bits.push_back(static_cast<TU>(b | sign));
    }
  }
  return bits;
}

enum class Input { kRandom, kThreeValues, kAllEqual, kSorted, kReversed };

template <typename Key>
std::vector<Key> MakeKeys(Input input, size_t num, RandomState& rng) {
  using TU = MakeUnsigned<Key>;
  const std::vector<TU> special = SpecialBits<Key>();
  // One in eight is a special value, so these also mix with random values.
  const auto random_bits = [&special, &rng]() {
    if ((Random32(&rng) & 7) == 0) {
      return special[Random32(&rng) % special.size()];
    }
    return static_cast<TU>(Random64(&rng));
  };

  std::vector<TU> bits(num);
  if (input == Input::kThreeValues) {
    const TU values[3] = {random_bits(), random_bits(), random_bits()};
    for (TU& b : bits) b = values[Random32(&rng) % 3];
  } else if (input == Input::kAllEqual) {
    const TU value = random_bits();
    for (TU& b : bits) b = value;
  } else {
    for (TU& b : bits) b = random_bits();
  }

  std::vector<Key> keys(num);
  for (size_t i = 0; i < num; ++i) keys[i] = BitCastScalar<Key>(bits[i]);
  if (input == Input::kSorted || input == Input::kReversed) {
    std::stable_sort(keys.begin(), keys.end(), KeyLess<SortAscending, Key>);
    if (input == Input::kReversed) std::reverse(keys.begin(), keys.end());
  }
  return keys;
}

// Calls the VQ*Static functions in vqargsort-inl.h.
struct StaticApi {
  template <ArgSortOp kOp, bool kStable, class Order, typename Key>
  static void Call(const Key* keys, size_t n, size_t k, uint64_t* indices,
                   uint128_t* scratch) {
    const Order order{};
    if constexpr (sizeof(Key) == 8) {
      if constexpr (kOp == ArgSortOp::kSort) {
        if constexpr (kStable) {
          VQStableArgSortStatic(keys, n, indices, scratch, order);
        } else {
          VQArgSortStatic(keys, n, indices, scratch, order);
        }
      } else if constexpr (kOp == ArgSortOp::kPartialSort) {
        if constexpr (kStable) {
          VQStableArgPartialSortStatic(keys, n, k, indices, scratch, order);
        } else {
          VQArgPartialSortStatic(keys, n, k, indices, scratch, order);
        }
      } else {
        if constexpr (kStable) {
          VQStableArgSelectStatic(keys, n, k, indices, scratch, order);
        } else {
          VQArgSelectStatic(keys, n, k, indices, scratch, order);
        }
      }
    } else {
      (void)scratch;
      if constexpr (kOp == ArgSortOp::kSort) {
        if constexpr (kStable) {
          VQStableArgSortStatic(keys, n, indices, order);
        } else {
          VQArgSortStatic(keys, n, indices, order);
        }
      } else if constexpr (kOp == ArgSortOp::kPartialSort) {
        if constexpr (kStable) {
          VQStableArgPartialSortStatic(keys, n, k, indices, order);
        } else {
          VQArgPartialSortStatic(keys, n, k, indices, order);
        }
      } else {
        if constexpr (kStable) {
          VQStableArgSelectStatic(keys, n, k, indices, order);
        } else {
          VQArgSelectStatic(keys, n, k, indices, order);
        }
      }
    }
  }
};

// Calls the dynamic-dispatch functions in vqargsort.h.
struct LibraryApi {
  template <ArgSortOp kOp, bool kStable, class Order, typename Key>
  static void Call(const Key* keys, size_t n, size_t k, uint64_t* indices,
                   uint128_t* scratch) {
    const Order order{};
    if constexpr (sizeof(Key) == 8) {
      if constexpr (kOp == ArgSortOp::kSort) {
        if constexpr (kStable) {
          hwy::VQStableArgSort(keys, n, indices, scratch, order);
        } else {
          hwy::VQArgSort(keys, n, indices, scratch, order);
        }
      } else if constexpr (kOp == ArgSortOp::kPartialSort) {
        if constexpr (kStable) {
          hwy::VQStableArgPartialSort(keys, n, k, indices, scratch, order);
        } else {
          hwy::VQArgPartialSort(keys, n, k, indices, scratch, order);
        }
      } else {
        if constexpr (kStable) {
          hwy::VQStableArgSelect(keys, n, k, indices, scratch, order);
        } else {
          hwy::VQArgSelect(keys, n, k, indices, scratch, order);
        }
      }
    } else {
      (void)scratch;
      if constexpr (kOp == ArgSortOp::kSort) {
        if constexpr (kStable) {
          hwy::VQStableArgSort(keys, n, indices, order);
        } else {
          hwy::VQArgSort(keys, n, indices, order);
        }
      } else if constexpr (kOp == ArgSortOp::kPartialSort) {
        if constexpr (kStable) {
          hwy::VQStableArgPartialSort(keys, n, k, indices, order);
        } else {
          hwy::VQArgPartialSort(keys, n, k, indices, order);
        }
      } else {
        if constexpr (kStable) {
          hwy::VQStableArgSelect(keys, n, k, indices, order);
        } else {
          hwy::VQArgSelect(keys, n, k, indices, order);
        }
      }
    }
  }
};

constexpr size_t kRedZone = 4;
// Different, so copying from one red zone into the other is also detected.
constexpr uint64_t kIndicesCanary = 0xDEADBEEFDEADBEEFull;
constexpr uint64_t kScratchCanary = 0x5CA7C45CA7C45CA7ull;

const char* OpName(ArgSortOp op) {
  switch (op) {
    case ArgSortOp::kSort:
      return "ArgSort";
    case ArgSortOp::kPartialSort:
      return "ArgPartialSort";
    case ArgSortOp::kSelect:
      return "ArgSelect";
  }
  return "?";
}

// Runs one function on one input and verifies the result.
template <class Api, ArgSortOp kOp, bool kStable, class Order, typename Key>
void CheckOne(const std::vector<Key>& keys, const std::vector<uint64_t>& ref,
              size_t k) {
  const size_t num = keys.size();
  // Offsetting by `misalign` keys and indices exercises unaligned access.
  const size_t misalign = num & 1;

  AlignedVector<Key> keys_buf(num + 1);
  Key* in = keys_buf.data() + misalign;
  std::copy(keys.begin(), keys.end(), in);
  AlignedVector<uint64_t> indices_buf(num + 2 * kRedZone + 1, kIndicesCanary);
  uint64_t* indices = indices_buf.data() + kRedZone + misalign;
  uint128_t canary128;
  canary128.lo = canary128.hi = kScratchCanary;
  AlignedVector<uint128_t> scratch_buf(num + 2 * kRedZone, canary128);
  uint128_t* scratch = scratch_buf.data() + kRedZone;

  Api::template Call<kOp, kStable, Order>(in, num, k, indices, scratch);

  const auto fail = [&](const char* what, size_t i) {
    HWY_ABORT("%s %s%s %s: %s at %zu (n=%zu k=%zu)\n",
              TypeName(Key(), 1).c_str(), kStable ? "Stable" : "", OpName(kOp),
              Order::IsAscending() ? "asc" : "desc", what, i, num, k);
  };

  if (memcmp(in, keys.data(), num * sizeof(Key)) != 0) fail("keys changed", 0);
  for (size_t i = 0; i < indices_buf.size(); ++i) {
    const bool inside = indices_buf.data() + i >= indices &&
                        indices_buf.data() + i < indices + num;
    if (!inside && indices_buf[i] != kIndicesCanary) {
      fail("wrote outside indices", i);
    }
  }
  for (size_t i = 0; i < scratch_buf.size(); ++i) {
    const bool inside = i >= kRedZone && i < kRedZone + num;
    const bool is_canary = scratch_buf[i].lo == kScratchCanary &&
                           scratch_buf[i].hi == kScratchCanary;
    if (!inside && !is_canary) fail("wrote outside scratch", i);
  }
  std::vector<uint8_t> seen(num, 0);
  for (size_t i = 0; i < num; ++i) {
    const size_t index = static_cast<size_t>(indices[i]);
    if (indices[i] >= num || seen[index]) fail("not a permutation", i);
    seen[index] = 1;
  }

  const auto equivalent = [&keys](uint64_t a, uint64_t b) {
    const Key ka = keys[static_cast<size_t>(a)];
    const Key kb = keys[static_cast<size_t>(b)];
    return !KeyLess<Order>(ka, kb) && !KeyLess<Order>(kb, ka);
  };
  // Stable results are unique; unstable ones only need an equivalent key.
  const auto expect_ref_at = [&](size_t i) {
    const bool ok =
        kStable ? indices[i] == ref[i] : equivalent(indices[i], ref[i]);
    if (!ok) fail("differs from reference", i);
  };

  if constexpr (kOp == ArgSortOp::kSort) {
    for (size_t i = 0; i < num; ++i) expect_ref_at(i);
  } else if constexpr (kOp == ArgSortOp::kPartialSort) {
    for (size_t i = 0; i < k; ++i) expect_ref_at(i);
  } else {
    expect_ref_at(k);
    // Stable: ties are ordered by index. Unstable: only keys are ordered.
    const auto less = [&](uint64_t a, uint64_t b) {
      if (KeyLess<Order>(keys[static_cast<size_t>(a)],
                         keys[static_cast<size_t>(b)])) {
        return true;
      }
      return kStable && equivalent(a, b) && a < b;
    };
    for (size_t i = 0; i < num; ++i) {
      if (i < k && less(indices[k], indices[i])) fail("after kth", i);
      if (i > k && less(indices[i], indices[k])) fail("before kth", i);
    }
  }
}

template <class Api, bool kStable, class Order, typename Key>
void CheckAllOps(const std::vector<Key>& keys, const std::vector<uint64_t>& ref,
                 bool sort_only) {
  const size_t num = keys.size();
  CheckOne<Api, ArgSortOp::kSort, kStable, Order>(keys, ref, 0);
  if (sort_only) return;
  std::vector<size_t> ks = {0, 1, num / 2, num - 1, num};
  std::sort(ks.begin(), ks.end());
  ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
  for (size_t k : ks) {
    if (k > num) continue;  // num - 1 wraps around when num == 0
    CheckOne<Api, ArgSortOp::kPartialSort, kStable, Order>(keys, ref, k);
    if (k < num) {
      CheckOne<Api, ArgSortOp::kSelect, kStable, Order>(keys, ref, k);
    }
  }
}

template <class Api, class Order, typename Key>
void CheckInput(Input input, size_t num, bool sort_only, RandomState& rng) {
  const std::vector<Key> keys = MakeKeys<Key>(input, num, rng);
  const std::vector<uint64_t> ref = ReferenceOrder<Order>(keys);
  CheckAllOps<Api, /*kStable=*/false, Order>(keys, ref, sort_only);
  CheckAllOps<Api, /*kStable=*/true, Order>(keys, ref, sort_only);
}

template <class Api, class Order, typename Key>
void TestKeyAndOrder() {
  RandomState rng;
  // Every size covers vqsort's small-array paths and our remainder loops. The
  // library shares that code, so it only gets the selected sizes.
  constexpr bool kEverySize = IsSame<Api, StaticApi>();
  if constexpr (kEverySize) {
    for (size_t num = 0; num <= AdjustedReps(300); ++num) {
      CheckInput<Api, Order, Key>(Input::kRandom, num, /*sort_only=*/false,
                                  rng);
    }
  }
  for (size_t num : {size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{7},
                     size_t{8}, size_t{9}, size_t{31}, size_t{32}, size_t{33},
                     size_t{255}, size_t{256}, size_t{257}, size_t{1000}}) {
    for (Input input : {Input::kRandom, Input::kThreeValues, Input::kAllEqual,
                        Input::kSorted, Input::kReversed}) {
      CheckInput<Api, Order, Key>(input, num, /*sort_only=*/false, rng);
    }
  }
  if constexpr (kEverySize) {
    CheckInput<Api, Order, Key>(Input::kRandom, AdjustedReps(100000),
                                /*sort_only=*/true, rng);
  }
}

template <class Api>
void TestAllKeys() {
  TestKeyAndOrder<Api, SortAscending, uint16_t>();
  TestKeyAndOrder<Api, SortDescending, uint16_t>();
  TestKeyAndOrder<Api, SortAscending, int16_t>();
  TestKeyAndOrder<Api, SortDescending, int16_t>();
  TestKeyAndOrder<Api, SortAscending, float16_t>();
  TestKeyAndOrder<Api, SortDescending, float16_t>();
  TestKeyAndOrder<Api, SortAscending, uint32_t>();
  TestKeyAndOrder<Api, SortDescending, uint32_t>();
  TestKeyAndOrder<Api, SortAscending, int32_t>();
  TestKeyAndOrder<Api, SortDescending, int32_t>();
  TestKeyAndOrder<Api, SortAscending, float>();
  TestKeyAndOrder<Api, SortDescending, float>();
  TestKeyAndOrder<Api, SortAscending, uint64_t>();
  TestKeyAndOrder<Api, SortDescending, uint64_t>();
  TestKeyAndOrder<Api, SortAscending, int64_t>();
  TestKeyAndOrder<Api, SortDescending, int64_t>();
  TestKeyAndOrder<Api, SortAscending, double>();
  TestKeyAndOrder<Api, SortDescending, double>();
}

void TestAllArgSortStatic() { TestAllKeys<StaticApi>(); }
void TestAllArgSortLibrary() { TestAllKeys<LibraryApi>(); }

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(ArgSortTest);
HWY_EXPORT_AND_TEST_P(ArgSortTest, TestAllArgSortStatic);
HWY_EXPORT_AND_TEST_P(ArgSortTest, TestAllArgSortLibrary);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
