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

#include <stdint.h>
#include <stdio.h>

#include <cmath>
#include <limits>
#include <vector>

#include "hwy/base.h"

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_NEON | HWY_SVE)
#endif  // HWY_DISABLED_TARGETS

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/math/float_manip_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/math/float_manip-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

template <typename T>
bool BitEqual(T a, T b) {
  if (std::isnan(a) && std::isnan(b)) return true;
  using TU = MakeUnsigned<T>;
  return BitCastScalar<TU>(a) == BitCastScalar<TU>(b);
}

// A spread of interesting magnitudes for both float32 and float64.
template <typename T>
std::vector<T> SampleValues() {
  const T inf = std::numeric_limits<T>::infinity();
  const T tiny = std::numeric_limits<T>::denorm_min();
  const T min_norm = std::numeric_limits<T>::min();
  const T max_norm = std::numeric_limits<T>::max();
  std::vector<T> v = {T(0),
                      T(-0.0),
                      T(1),
                      T(-1),
                      T(2),
                      T(0.5),
                      T(-0.5),
                      T(3),
                      T(1.5),
                      T(-2.25),
                      T(123.4375),
                      T(-9999.5),
                      T(1e10),
                      T(-1e-10),
                      T(0.1),
                      T(7),
                      min_norm,
                      -min_norm,
                      tiny,
                      -tiny,
                      tiny * 7,
                      min_norm * T(1.5),
                      max_norm,
                      -max_norm,
                      inf,
                      -inf,
                      std::numeric_limits<T>::quiet_NaN()};
  RandomState rng;
  for (int i = 0; i < 64; ++i) {
    const uint64_t bits = Random64(&rng);
    T x;
    if (sizeof(T) == 4) {
      const uint32_t b = static_cast<uint32_t>(bits);
      CopyBytes<4>(&b, &x);
    } else {
      CopyBytes<8>(&bits, &x);
    }
    v.push_back(x);
  }
  return v;
}

struct TestLdexp {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*t*/, D d) {
    const RebindToSigned<D> di;
    for (const T x : SampleValues<T>()) {
      for (int e : {-300, -60, -1, 0, 1, 24, 60, 300}) {
        const T expected = std::ldexp(x, e);
        const T actual = GetLane(Ldexp(d, Set(d, x), Set(di, e)));
        // MulByPow2 clamps very large |e|; only assert where ldexp is exact.
        if (std::isfinite(expected) || std::isnan(expected) ||
            std::abs(e) <= 260) {
          HWY_ASSERT(BitEqual(expected, actual));
        }
      }
    }
  }
};

struct TestIlogb {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*t*/, D d) {
    const RebindToSigned<D> di;
    using TI = TFromD<decltype(di)>;
    for (const T x : SampleValues<T>()) {
      const TI actual = GetLane(Ilogb(d, Set(d, x)));
      if (x == T(0) || std::isnan(x)) {
        HWY_ASSERT_EQ(LimitsMin<TI>(), actual);
      } else if (std::isinf(x)) {
        HWY_ASSERT_EQ(LimitsMax<TI>(), actual);
      } else {
        HWY_ASSERT_EQ(static_cast<TI>(std::ilogb(x)), actual);
      }
    }
  }
};

struct TestLogb {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*t*/, D d) {
    for (const T x : SampleValues<T>()) {
      const T actual = GetLane(Logb(d, Set(d, x)));
      const T expected = std::logb(x);
      HWY_ASSERT(BitEqual(expected, actual));
    }
  }
};

struct TestModf {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*t*/, D d) {
    for (const T x : SampleValues<T>()) {
      T expected_int;
      const T expected_frac = std::modf(x, &expected_int);
      VFromD<D> actual_int;
      const T actual_frac = GetLane(Modf(d, Set(d, x), actual_int));
      if (!BitEqual(expected_frac, actual_frac) ||
          !BitEqual(expected_int, GetLane(actual_int))) {
        fprintf(stderr, "Modf(%g): want frac %g int %g, got frac %g int %g\n",
                static_cast<double>(x), static_cast<double>(expected_frac),
                static_cast<double>(expected_int),
                static_cast<double>(actual_frac),
                static_cast<double>(GetLane(actual_int)));
      }
      HWY_ASSERT(BitEqual(expected_frac, actual_frac));
      HWY_ASSERT(BitEqual(expected_int, GetLane(actual_int)));
    }
  }
};

struct TestNextAfter {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*t*/, D d) {
    const auto vals = SampleValues<T>();
    for (const T a : vals) {
      for (const T b : vals) {
        const T expected = std::nextafter(a, b);
        const T actual = GetLane(NextAfter(d, Set(d, a), Set(d, b)));
        HWY_ASSERT(BitEqual(expected, actual));
      }
    }
  }
};

struct TestFmod {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*t*/, D d) {
    const auto vals = SampleValues<T>();
    for (const T a : vals) {
      for (const T b : vals) {
        const T expected = std::fmod(a, b);
        const T actual = GetLane(Fmod(d, Set(d, a), Set(d, b)));
        if (!BitEqual(expected, actual)) {
          fprintf(stderr, "Fmod(%g, %g): want %g got %g\n",
                  static_cast<double>(a), static_cast<double>(b),
                  static_cast<double>(expected), static_cast<double>(actual));
        }
        HWY_ASSERT(BitEqual(expected, actual));
      }
    }
    // A few explicit large-ratio / subnormal-result cases.
    const T tiny = std::numeric_limits<T>::denorm_min();
    const T maxn = std::numeric_limits<T>::max();
    for (const auto pr : std::vector<std::pair<T, T>>{{maxn, tiny * 3},
                                                      {maxn, T(3)},
                                                      {T(12345.678), tiny * 5},
                                                      {tiny * 7, tiny * 3},
                                                      {T(1), tiny * 11}}) {
      const T expected = std::fmod(pr.first, pr.second);
      const T actual = GetLane(Fmod(d, Set(d, pr.first), Set(d, pr.second)));
      HWY_ASSERT(BitEqual(expected, actual));
    }
  }
};

void TestAllLdexp() { ForFloat3264Types(ForPartialVectors<TestLdexp>()); }
void TestAllIlogb() { ForFloat3264Types(ForPartialVectors<TestIlogb>()); }
void TestAllLogb() { ForFloat3264Types(ForPartialVectors<TestLogb>()); }
void TestAllModf() { ForFloat3264Types(ForPartialVectors<TestModf>()); }
void TestAllNextAfter() {
  ForFloat3264Types(ForPartialVectors<TestNextAfter>());
}
void TestAllFmod() { ForFloat3264Types(ForPartialVectors<TestFmod>()); }

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(FloatManipTest);
HWY_EXPORT_AND_TEST_P(FloatManipTest, TestAllLdexp);
HWY_EXPORT_AND_TEST_P(FloatManipTest, TestAllIlogb);
HWY_EXPORT_AND_TEST_P(FloatManipTest, TestAllLogb);
HWY_EXPORT_AND_TEST_P(FloatManipTest, TestAllModf);
HWY_EXPORT_AND_TEST_P(FloatManipTest, TestAllNextAfter);
HWY_EXPORT_AND_TEST_P(FloatManipTest, TestAllFmod);
HWY_AFTER_TEST();
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
