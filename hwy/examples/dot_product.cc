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
#include <stdio.h>

#include <bit>
#include <random>
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/examples/dot_product.cc"
#include "hwy/aligned_allocator.h"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
//
#include "hwy/contrib/random/random-inl.h"
#include "hwy/nanobenchmark.h"
#include "hwy/timer.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
HWY_INLINE T DotProductSIMD(const T* HWY_RESTRICT array1,
                            const T* HWY_RESTRICT array2, size_t count) {
  // Standard pattern for Highway: define descriptor D, tag d, and vector type
  // V. Alternatively, we could take 'D d' as a parameter and infer V using
  // typename V = VFromD<D>
  using D = hn::ScalableTag<T>;
  const D d;
  using V = hn::Vec<D>;

  V sum = hn::Zero(d);

  size_t i = 0;
  // HWY_LANES_CONSTEXPR allows N to be a constexpr if supported by the
  // architecture, otherwise not.
  HWY_LANES_CONSTEXPR size_t N = hn::Lanes(d);

  if (count >= N) {
    for (; i <= count - N; i += N) {
      // We know memory is aligned, so we can use Load instead of LoadU.
      sum = hn::Add(sum,
                    hn::Mul(hn::Load(d, array1 + i), hn::Load(d, array2 + i)));
    }
  }
  // Handle final remainder with LoadN
  size_t remainder = count - i;
  HWY_DASSERT(remainder < N);
  if (remainder > 0) {
    // LoadN gives us 0s in upper values for elements beyond remainder.
    V loaded_vec1 = hn::LoadN(d, array1 + i, remainder);
    V loaded_vec2 = hn::LoadN(d, array2 + i, remainder);

    sum = hn::Add(sum, hn::Mul(loaded_vec1, loaded_vec2));
  }

  // Use only 1 reduce sum as it is more expensive (uses shuffles).
  return hn::ReduceSum(d, sum);
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

// HWY_ONCE marks code that will be compiled once instead once per target
// architecure
#if HWY_ONCE
namespace hwy {
// Fills array with random values between 0 and max_v
template <typename T>
void FillRandom(const std::uint64_t seed, T* HWY_RESTRICT result,
                const size_t size, T max_v) {
  std::mt19937_64 engine(seed);
  std::uniform_real_distribution<double> dist(0, 1.0);
  for (size_t i = 0; i < size; i++) {
    result[i] = static_cast<T>(dist(engine) * max_v);
  }
}
template <typename T>
T DotProductScalar(const T* array1, const T* array2, size_t count) {
  T sum = T{0};
  for (size_t i = 0; i < count; ++i) {
    sum += array1[i] * array2[i];
  }
  return sum;
}
int RunTests() {
  const size_t count = 10'000;
  const int reps = 100'000;

  auto test = [&](auto type_tag, const char* type_name, auto max_v) HWY_ATTR {
    using T = decltype(type_tag);
    hwy::AlignedVector<T> data1(count);
    hwy::AlignedVector<T> data2(count);

    // Unpredictable1 prevents compiler from optimizing away code as it can't
    // assume value of that 1. Keep in mind it is quite expensive and
    // recommended to use sparingly.

    FillRandom(static_cast<uint64_t>(hwy::Unpredictable1()), data1.data(),
               count, static_cast<T>(max_v));
    FillRandom(static_cast<uint64_t>(hwy::Unpredictable1() * 10), data2.data(),
               count, static_cast<T>(max_v));

    const double t_scalar_0 = hwy::platform::Now();
    T scalar_dot = {0};
    for (int r = 0; r < reps; ++r) {
      scalar_dot = hwy::DotProductScalar(data1.data(), data2.data(), count);
    }
    const double t_scalar_1 = hwy::platform::Now();
    const double dt_scalar = t_scalar_1 - t_scalar_0;

    // HWY_EXPORT_T creates a dispatch table, and dispatch calls calls the best
    // (widest) available implementation.
    HWY_EXPORT_T(DotProductTable, DotProductSIMD<T>);
    const double t_simd_0 = hwy::platform::Now();
    T simd_dot = 0;

    for (int r = 0; r < reps; ++r) {
      simd_dot = HWY_DYNAMIC_DISPATCH_T(DotProductTable)(data1.data(),
                                                         data2.data(), count);
    }
    const double t_simd_1 = hwy::platform::Now();
    const double dt_simd = t_simd_1 - t_simd_0;

    printf("%s - Scalar dot: %f (took %f seconds)\n", type_name,
           static_cast<double>(scalar_dot), dt_scalar);
    printf("%s - SIMD dot:   %f (took %f seconds)\n", type_name,
           static_cast<double>(simd_dot), dt_simd);
    printf("%s - Speedup:    %fx\n", type_name, dt_scalar / dt_simd);

    bool passed = scalar_dot == simd_dot;
    HWY_IF_CONSTEXPR(hwy::IsFloat<T>()) {
      passed = hwy::ScalarAbs(scalar_dot - simd_dot) <= 1e-1;
    }

    if (!passed) {
      printf("%s Validation FAILED\n", type_name);
      return 1;
    }
    return 0;
  };

  if (test(float(), "Float", 1.0) != 0) return 1;
  if (test(double(), "Double", 1.0) != 0) return 1;
  if (test(int32_t(), "Int32", 10) != 0) return 1;

  printf("All Validations PASSED\n");
  return 0;
}

}  // namespace hwy

int main() { return hwy::RunTests(); }
#endif  // HWY_ONCE
