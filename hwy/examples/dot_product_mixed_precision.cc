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
//
#include <stddef.h>
#include <stdint.h>

#include <iostream>
#include <numeric>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/examples/dot_product_mixed_precision.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/timer.h"

/*
Highway SIMD Tutorial: Mixed precision dot product

This example demonstrates how to perform a dot product where the inputs are
uint8_t and are promoted uint32_t to prevent overflow.

*/

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

uint32_t DotProductSIMD(const uint8_t* HWY_RESTRICT a,
                        const uint8_t* HWY_RESTRICT b, size_t count) {
  using DU8 = hn::ScalableTag<uint8_t>;
  const DU8 du8;
  using DU32 = hn::ScalableTag<uint32_t>;
  const DU32 du32;
  using V = hn::Vec<DU32>;
  V sum = hn::Zero(du32);
  const size_t N8 = hn::Lanes(du8);
  size_t i = 0;
  if (count >= N8) {
    for (; i <= count - N8; i += N8) {
      sum = hn::SumOfMulQuadAccumulate(du32, hn::LoadU(du8, a + i),
                                       hn::LoadU(du8, b + i), sum);
    }
  }

  // Use LoadN for remainder, sets 0 for values beyond remainder
  size_t remainder = count - i;
  HWY_DASSERT(remainder < N8);
  if (remainder > 0) {
    sum = hn::SumOfMulQuadAccumulate(du32, hn::LoadN(du8, a + i, remainder),
                                     hn::LoadN(du8, b + i, remainder), sum);
  }

  uint32_t total = hn::ReduceSum(du32, sum);

  return total;
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_EXPORT(DotProductSIMD);

uint32_t DotProductScalar(const uint8_t* HWY_RESTRICT a,
                          const uint8_t* HWY_RESTRICT b, size_t count) {
  size_t i = 0;
  uint32_t total = 0;
  // Scalar dot product with type casting
  for (; i < count; ++i) {
    total += static_cast<uint32_t>(a[i]) * static_cast<uint32_t>(b[i]);
  }
  return total;
}

int Run() {
  const size_t count = 10000025;
  std::vector<uint8_t> a(count);
  std::vector<uint8_t> b(count);
  std::iota(a.begin(), a.end(), 1);
  std::iota(b.begin(), b.end(), 1);
  // Record start time
  const double t_scalar_0 = hwy::platform::Now();
  uint32_t scalar_dot_product = hwy::DotProductScalar(a.data(), b.data(), count);
  // Record end time and print execution time and dot product
  const double t_scalar_1 = hwy::platform::Now();
  const double dt_scalar = 1000.0 * (t_scalar_1 - t_scalar_0);
  std::cout << "Scalar Execution time: " << dt_scalar << " ms" << std::endl;
  std::cout << "Scalar Dot product: " << scalar_dot_product << std::endl;
  // Record start time
  const double t_simd_0 = hwy::platform::Now();
  uint32_t simd_dot_product =
      HWY_DYNAMIC_DISPATCH(DotProductSIMD)(a.data(), b.data(), count);
  // Record stop timer and print time and dot product
  const double t_simd_1 = hwy::platform::Now();
  const double dt_simd = 1000.0 * (t_simd_1 - t_simd_0);
  std::cout << "SIMD Execution time: " << dt_simd << " ms" << std::endl;
  std::cout << "SIMD Dot product: " << simd_dot_product << std::endl;
  bool passed = scalar_dot_product == simd_dot_product;
  if (!passed) {
    std::cout << "Validation failed" << std::endl;
    return 1;
  }
  return 0;
}
}  // namespace hwy

int main() { return hwy::Run(); }
#endif  // HWY_ONCE
