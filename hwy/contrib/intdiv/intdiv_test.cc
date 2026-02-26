// Copyright 2024 Google LLC
// Copyright 2026 Fujitsu Limited
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: BSD-3-Clause
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
#include <string.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/intdiv/intdiv_test.cc"
#include "hwy/foreach_target.h"   // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/intdiv/intdiv-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

template <typename T>
T RandWithin(hwy::RandomState& rng) {
  if constexpr (sizeof(T) <= 4) {
    return static_cast<T>(Random32(&rng));
  } else {
    const uint64_t hi = Random32(&rng);
    const uint64_t lo = Random32(&rng);
    return static_cast<T>((hi << 32) ^ lo);
  }
}

template <typename T>
T SafeFloorDivScalar(T a, T b) {
  if constexpr (!hwy::IsSigned<T>()) {
    return static_cast<T>(a / b);
  } else {
    if (b == T(-1) && a == std::numeric_limits<T>::min()) {
      return T();
    }
    const T q = static_cast<T>(a / b);
    const T r = static_cast<T>(a % b);
    const bool adjust = (r != T(0)) && ((a < T(0)) != (b < T(0)));
    return static_cast<T>(q - (adjust ? T(1) : T(0)));
  }
}

template <typename T>
bool IsPow2(T x) {
  using U = typename hwy::MakeUnsigned<T>;
  const U ux = static_cast<U>(x);
  return ux != U(0) && (ux & (ux - U(1))) == U(0);
}

struct TestBasicDivision {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t /*misalign_b*/,
                  RandomState& /*rng*/) {
    if (count == 0) return;
    
    using T = TFromD<D>;
    const size_t N = Lanes(d);
    
    AlignedFreeUniquePtr<T[]> pa =
        AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    AlignedFreeUniquePtr<T[]> expected = AllocateAligned<T>(HWY_MAX(1, count));
    AlignedFreeUniquePtr<T[]> actual = AllocateAligned<T>(HWY_MAX(1, count));
    HWY_ASSERT(pa && expected && actual);
    
    T* a = pa.get() + misalign_a;
    
    const T divisors[] = {T(1), T(2), T(3), T(5), T(7), T(10), T(11), T(13),
                          T(16), T(17), T(25), T(31), T(32), T(64), T(100),
                          T(127), T(128), T(255), T(256), T(1000)};
    
    for (T divisor : divisors) {
      if (divisor == T(0)) continue;
      
      const auto params = ComputeDivisorParams(divisor);
      
      HWY_ASSERT_EQ(divisor, params.divisor);
      if (IsPow2(divisor) && !hwy::IsSigned<T>()) {
        HWY_ASSERT(params.is_pow2);
      }
      
      for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<T>((static_cast<uint32_t>(i) * 123u) & 255u);
        expected[i] = static_cast<T>(a[i] / divisor);
      }
      
      for (size_t i = 0; i + N <= count; i += N) {
        const auto v = LoadU(d, a + i);
        const auto q = IntDiv(d, v, params);
        StoreU(q, d, actual.get() + i);
      }
      
      if (count % N != 0) {
        const size_t i = count - (count % N);
        const size_t remaining = count - i;
        const auto v = LoadN(d, a + i, remaining);
        const auto q = IntDiv(d, v, params);
        StoreN(q, d, actual.get() + i, remaining);
      }
      
      for (size_t i = 0; i < count; ++i) {
        HWY_ASSERT_EQ(expected[i], actual[i]);
      }
    }
  }
};

struct TestPowerOf2Division {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t /*misalign_b*/,
                  RandomState& /*rng*/) {
    if (count == 0) return;
    
    using T = TFromD<D>;
    const size_t N = Lanes(d);
    
    AlignedFreeUniquePtr<T[]> pa =
        AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    AlignedFreeUniquePtr<T[]> expected = AllocateAligned<T>(HWY_MAX(1, count));
    AlignedFreeUniquePtr<T[]> actual = AllocateAligned<T>(HWY_MAX(1, count));
    HWY_ASSERT(pa && expected && actual);
    
    T* a = pa.get() + misalign_a;
    
    const int max_shift = static_cast<int>(sizeof(T) * 8 - hwy::IsSigned<T>()) - 1;
    
    for (int shift = 0; shift <= max_shift; ++shift) {
      const T divisor = static_cast<T>(T(1) << shift);
      if (divisor <= T(0)) break;
      
      const auto params = ComputeDivisorParams(divisor);
      HWY_ASSERT(params.is_pow2);
      HWY_ASSERT_EQ(shift, params.pow2_shift);
      
      for (size_t i = 0; i < count; ++i) {
        if constexpr (hwy::IsSigned<T>()) {
          a[i] = static_cast<T>(static_cast<int>(i) - static_cast<int>(count / 2));
        } else {
          a[i] = static_cast<T>(i);
        }
        expected[i] = static_cast<T>(a[i] / divisor);
      }
      
      for (size_t i = 0; i + N <= count; i += N) {
        const auto v = LoadU(d, a + i);
        const auto q = IntDiv(d, v, params);
        StoreU(q, d, actual.get() + i);
      }
      
      if (count % N != 0) {
        const size_t i = count - (count % N);
        const size_t remaining = count - i;
        const auto v = LoadN(d, a + i, remaining);
        const auto q = IntDiv(d, v, params);
        StoreN(q, d, actual.get() + i, remaining);
      }
      
      for (size_t i = 0; i < count; ++i) {
        HWY_ASSERT_EQ(expected[i], actual[i]);
      }
    }
  }
};

struct TestSignedDivision {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t /*misalign_b*/,
                  RandomState& /*rng*/) {
    using T = TFromD<D>;
    if (!hwy::IsSigned<T>()) return;
    if (count == 0) return;
    
    const size_t N = Lanes(d);
    
    AlignedFreeUniquePtr<T[]> pa =
        AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    AlignedFreeUniquePtr<T[]> expected = AllocateAligned<T>(HWY_MAX(1, count));
    AlignedFreeUniquePtr<T[]> actual = AllocateAligned<T>(HWY_MAX(1, count));
    HWY_ASSERT(pa && expected && actual);
    
    T* a = pa.get() + misalign_a;
    
    const T divisors[] = {T(3), T(5), T(7), T(-3), T(-5), T(-7),
                          T(17), T(-17), T(100), T(-100)};
    
    for (T divisor : divisors) {
      const auto params = ComputeDivisorParams(divisor);
      
      for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<T>(static_cast<int>(i) - static_cast<int>(count / 2));
        
        if (divisor == T(-1) && a[i] == std::numeric_limits<T>::min()) {
          expected[i] = T();
        } else {
          expected[i] = static_cast<T>(a[i] / divisor);
        }
      }
      
      for (size_t i = 0; i + N <= count; i += N) {
        const auto v = LoadU(d, a + i);
        const auto q = IntDiv(d, v, params);
        StoreU(q, d, actual.get() + i);
      }
      
      if (count % N != 0) {
        const size_t i = count - (count % N);
        const size_t remaining = count - i;
        const auto v = LoadN(d, a + i, remaining);
        const auto q = IntDiv(d, v, params);
        StoreN(q, d, actual.get() + i, remaining);
      }
      
      for (size_t i = 0; i < count; ++i) {
        if (divisor == T(-1) && a[i] == std::numeric_limits<T>::min()) {
          continue;
        }
        HWY_ASSERT_EQ(expected[i], actual[i]);
      }
    }
  }
};

struct TestFloorDivision {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t /*misalign_b*/,
                  RandomState& /*rng*/) {
    if (count == 0) return;
    
    using T = TFromD<D>;
    const size_t N = Lanes(d);
    
    AlignedFreeUniquePtr<T[]> pa =
        AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    AlignedFreeUniquePtr<T[]> expected = AllocateAligned<T>(HWY_MAX(1, count));
    AlignedFreeUniquePtr<T[]> actual = AllocateAligned<T>(HWY_MAX(1, count));
    HWY_ASSERT(pa && expected && actual);
    
    T* a = pa.get() + misalign_a;
    
    std::vector<T> divisors = {T(1), T(2), T(3), T(5), T(7), T(11), T(17), T(100)};
    if constexpr (hwy::IsSigned<T>()) {
      std::vector<T> neg = {T(-1), T(-2), T(-3), T(-5), T(-7), T(-11), T(-17)};
      divisors.insert(divisors.end(), neg.begin(), neg.end());
    }
    
    for (T divisor : divisors) {
      const auto params = ComputeDivisorParams(divisor);
      
      for (size_t i = 0; i < count; ++i) {
        if constexpr (hwy::IsSigned<T>()) {
          a[i] = static_cast<T>(static_cast<int>(i) - 50);
        } else {
          a[i] = static_cast<T>(i);
        }
        expected[i] = SafeFloorDivScalar(a[i], divisor);
      }
      
      for (size_t i = 0; i + N <= count; i += N) {
        const auto v = LoadU(d, a + i);
        const auto q = IntDivFloor(d, v, params);
        StoreU(q, d, actual.get() + i);
      }
      
      if (count % N != 0) {
        const size_t i = count - (count % N);
        const size_t remaining = count - i;
        const auto v = LoadN(d, a + i, remaining);
        const auto q = IntDivFloor(d, v, params);
        StoreN(q, d, actual.get() + i, remaining);
      }
      
      for (size_t i = 0; i < count; ++i) {
        HWY_ASSERT_EQ(expected[i], actual[i]);
      }
    }
  }
};

struct TestEdgeCases {
  template <class D>
  void operator()(D d, size_t /*count*/, size_t /*misalign_a*/,
                  size_t /*misalign_b*/, RandomState& /*rng*/) {
    using T = TFromD<D>;
    
    {
      const auto params = ComputeDivisorParams(T(1));
      for (T val : {T(0), T(1), T(100), T(std::numeric_limits<T>::max())}) {
        HWY_ASSERT_EQ(val, GetLane(IntDiv(d, Set(d, val), params)));
      }
    }
    
    {
      const T divisor = std::numeric_limits<T>::max();
      const auto params = ComputeDivisorParams(divisor);
      HWY_ASSERT_EQ(T(0), GetLane(IntDiv(d, Set(d, T(100)), params)));
      HWY_ASSERT_EQ(T(1), GetLane(IntDiv(d, Set(d, divisor), params)));
    }
    
    if constexpr (hwy::IsSigned<T>()) {
      const T kMin = std::numeric_limits<T>::min();
      const T kMax = std::numeric_limits<T>::max();
      
      {
        const auto params = ComputeDivisorParams(T(-1));
        HWY_ASSERT_EQ(kMin, GetLane(IntDiv(d, Set(d, kMin), params)));
        HWY_ASSERT_EQ(T(0), GetLane(IntDiv(d, Set(d, T(0)), params)));
        HWY_ASSERT_EQ(T(-1), GetLane(IntDiv(d, Set(d, T(1)), params)));
      }
      
      {
        const auto params = ComputeDivisorParams(T(7));
        HWY_ASSERT_EQ(static_cast<T>(kMax / T(7)),
                      GetLane(IntDiv(d, Set(d, kMax), params)));
        HWY_ASSERT_EQ(static_cast<T>((kMin + 1) / T(7)),
                      GetLane(IntDiv(d, Set(d, static_cast<T>(kMin + 1)), params)));
      }
    }
    
    if constexpr (sizeof(T) == 4) {
      for (T divisor : {T(65535), T(65536)}) {
        const auto params = ComputeDivisorParams(divisor);
        for (T dividend : {T(0), T(1), T(100)}) {
          HWY_ASSERT_EQ(static_cast<T>(dividend / divisor),
                        GetLane(IntDiv(d, Set(d, dividend), params)));
        }
      }
    } else if constexpr (sizeof(T) == 8 && !hwy::IsSigned<T>()) {
      for (T divisor : {T(0xFFFFFFFFull), T(0x100000000ull)}) {
        const auto params = ComputeDivisorParams(divisor);
        for (T dividend : {T(0), T(1), T(100)}) {
          HWY_ASSERT_EQ(static_cast<T>(dividend / divisor),
                        GetLane(IntDiv(d, Set(d, dividend), params)));
        }
      }
    }
  }
};


struct TestRandomDivision {
  template <class D>
  void operator()(D d, size_t /*count*/, size_t /*misalign_a*/,
                  size_t /*misalign_b*/, RandomState& rng) {
    using T = TFromD<D>;
    
    std::vector<T> divisors = {T(3), T(7), T(17), T(100), T(1000)};
    if constexpr (hwy::IsSigned<T>()) {
      divisors.insert(divisors.end(), {T(-3), T(-7), T(-17)});
    }
    
    for (T divisor : divisors) {
      const auto params = ComputeDivisorParams(divisor);
      
      for (int iter = 0; iter < 100; ++iter) {
        const T dividend = RandWithin<T>(rng);
        
        if constexpr (hwy::IsSigned<T>()) {
          if (divisor == T(-1) && dividend == std::numeric_limits<T>::min()) {
            continue;
          }
        }
        
        const T expected = static_cast<T>(dividend / divisor);
        HWY_ASSERT_EQ(expected, GetLane(IntDiv(d, Set(d, dividend), params)));
        
        const T expected_floor = SafeFloorDivScalar(dividend, divisor);
        HWY_ASSERT_EQ(expected_floor,
                      GetLane(IntDivFloor(d, Set(d, dividend), params)));
      }
    }
  }
};


struct TestLargeDivisors {
  template <class D>
  void operator()(D d, size_t /*count*/, size_t /*misalign_a*/,
                  size_t /*misalign_b*/, RandomState& rng) {
    using T = TFromD<D>;
    
    std::vector<T> large_divisors;
    
    if constexpr (sizeof(T) == 1) {
      if constexpr (hwy::IsSigned<T>()) {
        large_divisors = {T(125), T(126), T(127), T(-125), T(-126), T(-127), 
                         std::numeric_limits<T>::min()};
      } else {
        large_divisors = {T(250), T(251), T(252), T(253), T(254), T(255)};
      }
    } else if constexpr (sizeof(T) == 2) {
      if constexpr (hwy::IsSigned<T>()) {
        large_divisors = {T(32765), T(32766), T(32767), 
                         T(-32765), T(-32766), T(-32767), 
                         std::numeric_limits<T>::min()};
      } else {
        large_divisors = {T(65530), T(65531), T(65532), T(65533), T(65534), T(65535)};
      }
    } else if constexpr (sizeof(T) == 4) {
      if constexpr (hwy::IsSigned<T>()) {
        const T kMin = std::numeric_limits<T>::min();
        const T kMax = std::numeric_limits<T>::max();
        large_divisors = {static_cast<T>(kMax - 2), static_cast<T>(kMax - 1), kMax,
                         static_cast<T>(kMin + 3), static_cast<T>(kMin + 2), 
                         static_cast<T>(kMin + 1), kMin};
      } else {
        const T kMax = std::numeric_limits<T>::max();
        large_divisors = {static_cast<T>(kMax - 5), static_cast<T>(kMax - 4), 
                         static_cast<T>(kMax - 3), static_cast<T>(kMax - 2),
                         static_cast<T>(kMax - 1), kMax};
      }
    } else if constexpr (sizeof(T) == 8) {
      if constexpr (hwy::IsSigned<T>()) {
        const T kMin = std::numeric_limits<T>::min();
        const T kMax = std::numeric_limits<T>::max();
        large_divisors = {static_cast<T>(kMax - 2), static_cast<T>(kMax - 1), kMax,
                         static_cast<T>(kMin + 3), static_cast<T>(kMin + 2),
                         static_cast<T>(kMin + 1), kMin};
      } else {
        const T kMax = std::numeric_limits<T>::max();
        large_divisors = {static_cast<T>(kMax - 5), static_cast<T>(kMax - 4),
                         static_cast<T>(kMax - 3), static_cast<T>(kMax - 2),
                         static_cast<T>(kMax - 1), kMax};
      }
    }
    
    for (T divisor : large_divisors) {
      const auto params = ComputeDivisorParams(divisor);
      
      for (int iter = 0; iter < 20; ++iter) {
        T dividend = RandWithin<T>(rng);
        
        T expected;
        if constexpr (hwy::IsSigned<T>()) {
          if (divisor == T(-1) && dividend == std::numeric_limits<T>::min()) {
            continue;
          }
          expected = static_cast<T>(dividend / divisor);
        } else {
          expected = static_cast<T>(dividend / divisor);
        }
        
        const T actual = GetLane(IntDiv(d, Set(d, dividend), params));
        HWY_ASSERT_EQ(expected, actual);
        
        const T expected_floor = SafeFloorDivScalar(dividend, divisor);
        const T actual_floor = GetLane(IntDivFloor(d, Set(d, dividend), params));
        HWY_ASSERT_EQ(expected_floor, actual_floor);
      }
      
      std::vector<T> boundary_dividends;
      if constexpr (hwy::IsSigned<T>()) {
        boundary_dividends = {T(0), T(1), T(-1), 
                             std::numeric_limits<T>::max()};

        if (divisor != T(-1)) {
          boundary_dividends.push_back(std::numeric_limits<T>::min());
        }
      } else {
        boundary_dividends = {T(0), T(1), std::numeric_limits<T>::max()};
      }
      
      for (T dividend : boundary_dividends) {

        const T expected = static_cast<T>(dividend / divisor);
        const T actual = GetLane(IntDiv(d, Set(d, dividend), params));
        HWY_ASSERT_EQ(expected, actual);
        
        const T expected_floor = SafeFloorDivScalar(dividend, divisor);
        const T actual_floor = GetLane(IntDivFloor(d, Set(d, dividend), params));
        HWY_ASSERT_EQ(expected_floor, actual_floor);
      }
    }
  }
};


struct TestConvenienceAPI {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t /*misalign_b*/,
                  RandomState& /*rng*/) {
    if (count == 0) return;
    
    using T = TFromD<D>;
    const size_t N = Lanes(d);
    
    AlignedFreeUniquePtr<T[]> pa =
        AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    AlignedFreeUniquePtr<T[]> expected = AllocateAligned<T>(HWY_MAX(1, count));
    AlignedFreeUniquePtr<T[]> actual = AllocateAligned<T>(HWY_MAX(1, count));
    HWY_ASSERT(pa && expected && actual);
    
    T* a = pa.get() + misalign_a;
    
    const T divisor = T(7);
    
    for (size_t i = 0; i < count; ++i) {
      a[i] = static_cast<T>(i * T(10));
      expected[i] = static_cast<T>(a[i] / divisor);
    }
    
    for (size_t i = 0; i + N <= count; i += N) {
      const auto v = LoadU(d, a + i);
      const auto q = DivideByScalar(d, v, divisor);
      StoreU(q, d, actual.get() + i);
    }
    
    if (count % N != 0) {
      const size_t i = count - (count % N);
      const size_t remaining = count - i;
      const auto v = LoadN(d, a + i, remaining);
      const auto q = DivideByScalar(d, v, divisor);
      StoreN(q, d, actual.get() + i, remaining);
    }
    
    for (size_t i = 0; i < count; ++i) {
      HWY_ASSERT_EQ(expected[i], actual[i]);
    }
    
    if constexpr (hwy::IsSigned<T>()) {
      for (size_t i = 0; i < count; ++i) {
        expected[i] = SafeFloorDivScalar(a[i], divisor);
      }
      
      for (size_t i = 0; i + N <= count; i += N) {
        const auto v = LoadU(d, a + i);
        const auto q = FloorDivideByScalar(d, v, divisor);
        StoreU(q, d, actual.get() + i);
      }
      
      if (count % N != 0) {
        const size_t i = count - (count % N);
        const size_t remaining = count - i;
        const auto v = LoadN(d, a + i, remaining);
        const auto q = FloorDivideByScalar(d, v, divisor);
        StoreN(q, d, actual.get() + i, remaining);
      }
      
      for (size_t i = 0; i < count; ++i) {
        HWY_ASSERT_EQ(expected[i], actual[i]);
      }
    }
  }
};

struct TestArrayOperations {
  template <class D>
  void operator()(D /*d*/, size_t /*count*/, size_t /*misalign_a*/,
                  size_t /*misalign_b*/, RandomState& /*rng*/) {
    using T = TFromD<D>;
    
    constexpr size_t kCount = 127;
    auto array = AllocateAligned<T>(kCount);
    auto expected = AllocateAligned<T>(kCount);
    HWY_ASSERT(array && expected);
    
    const T divisor = T(11);
    
    for (size_t i = 0; i < kCount; ++i) {
      array[i] = static_cast<T>((static_cast<uint32_t>(i) * 13u) & 255u);
      expected[i] = static_cast<T>(array[i] / divisor);
    }
    
    DivideArrayByScalar(array.get(), kCount, divisor);
    
    for (size_t i = 0; i < kCount; ++i) {
      HWY_ASSERT_EQ(expected[i], array[i]);
    }
    
    if constexpr (hwy::IsSigned<T>()) {
      constexpr size_t kCount2 = 100;
      auto array2 = AllocateAligned<T>(kCount2);
      auto expected2 = AllocateAligned<T>(kCount2);
      HWY_ASSERT(array2 && expected2);
      
      const T divisor2 = T(-7);
      
      for (size_t i = 0; i < kCount2; ++i) {
        array2[i] = static_cast<T>(static_cast<int>(i) - 50);
        expected2[i] = SafeFloorDivScalar(array2[i], divisor2);
      }
      
      FloorDivideArrayByScalar(array2.get(), kCount2, divisor2);
      
      for (size_t i = 0; i < kCount2; ++i) {
        HWY_ASSERT_EQ(expected2[i], array2[i]);
      }
    }
  }
};

struct TestDivideHighBy {
  HWY_INLINE void operator()() const {
    using detail::DivideHighBy;
    
    {
      const uint64_t out = DivideHighBy(1ull, 3ull);
      HWY_ASSERT_EQ(0x5555555555555555ull, out);
    }
    
    {
      const uint64_t high = 1ull << 63;
      const uint64_t div = 1ull << 63;
      const uint64_t out = DivideHighBy(high, div);
      HWY_ASSERT_EQ(0ull, out);
    }
    
    {
      const uint64_t out = DivideHighBy(1ull, ~0ull);
      HWY_ASSERT_EQ(1ull, out);
    }
  }
};

template <class Test>
struct ForeachCountAndMisalign {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) const {
    RandomState rng;
    const size_t N = Lanes(d);
    const size_t misalignments[3] = {0, N / 4, 3 * N / 5};
    
    for (size_t count = 0; count < 2 * N; ++count) {
      for (size_t ma : misalignments) {
        for (size_t mb : misalignments) {
          Test()(d, count, ma, mb, rng);
        }
      }
    }
    
    for (size_t count : {10 * N, 16 * N, size_t{100}}) {
      for (size_t ma : misalignments) {
        Test()(d, count, ma, 0, rng);
      }
    }
  }
};

void TestAllBasicDivision() {
  ForIntegerTypes(ForPartialVectors<ForeachCountAndMisalign<TestBasicDivision>>());
}

void TestAllPowerOf2Division() {
  ForIntegerTypes(ForPartialVectors<ForeachCountAndMisalign<TestPowerOf2Division>>());
}

void TestAllSignedDivision() {
  ForSignedTypes(ForPartialVectors<ForeachCountAndMisalign<TestSignedDivision>>());
}

void TestAllFloorDivision() {
  ForIntegerTypes(ForPartialVectors<ForeachCountAndMisalign<TestFloorDivision>>());
}

void TestAllEdgeCases() {
  ForIntegerTypes(ForPartialVectors<ForeachCountAndMisalign<TestEdgeCases>>());
}

void TestAllRandomDivision() {
  ForIntegerTypes(ForPartialVectors<ForeachCountAndMisalign<TestRandomDivision>>());
}

void TestAllLargeDivisors() {
  ForIntegerTypes(ForPartialVectors<ForeachCountAndMisalign<TestLargeDivisors>>());
}

void TestAllConvenienceAPI() {
  ForIntegerTypes(ForPartialVectors<ForeachCountAndMisalign<TestConvenienceAPI>>());
}

void TestAllArrayOperations() {
  ForIntegerTypes(ForPartialVectors<ForeachCountAndMisalign<TestArrayOperations>>());
}

void TestAllDivideHighBy() {
  TestDivideHighBy{}();
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {

HWY_BEFORE_TEST(IntDivTest);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllBasicDivision);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllPowerOf2Division);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllSignedDivision);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllFloorDivision);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllEdgeCases);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllRandomDivision);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllLargeDivisors);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllConvenienceAPI);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllArrayOperations);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllDivideHighBy);
HWY_AFTER_TEST();

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
#endif  // HWY_ONCE