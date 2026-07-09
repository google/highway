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
#include <string.h>

#include <string>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/examples/sum_hex.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/nanobenchmark.h"
#include "hwy/timer.h"

/*
Highway SIMD Tutorial: Hexadecimal String Summation (Table Lookups & Quad
Accumulate)

This example demonstrates how to efficiently decode a hexadecimal ASCII string
and sum the decoded integer values using SIMD table lookups and quad
accumulate.

Key SIMD techniques shown:
1. Table Lookups (Lookup32): Using a 32-byte lookup table to simultaneously map
   ASCII characters ('0'-'9' and 'A'-'F') to their 4-bit nibble values across
   vector lanes without conditional branching.
2. Saturated Subtraction (SaturatedSub): Efficiently computing the index into
   the lookup table by subtracting the ASCII value of '0' from the input
   characters without undederflows.
3. Quad Accumulate (SumOfMulQuadAccumulate): Efficiently accumulating 8-bit
   integers into 32-bit integer sums by multiplying with ones and reducing
   4x8-bit lanes into 32-bit accumulators to avoid 8-bit overflow.
*/

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
namespace hn = hwy::HWY_NAMESPACE;

// Sums hex-encoded string.
// Assumes input contains only valid uppercase hex characters ('0'-'9',
// 'A'-'F'). Other characters may produce undefined results (but will not
// crash).
HWY_INLINE int32_t SumHexSIMD(const char* HWY_RESTRICT hex_chars,
                              size_t count) {
  using DI32 = hn::ScalableTag<int32_t>;
  using DU8 = hn::Repartition<uint8_t, DI32>;
  using DI8 = hn::Repartition<int8_t, DI32>;

  const DI32 di32;
  const DU8 du8;
  const DI8 di8;

  using VI32 = hn::Vec<DI32>;
  using VU8 = hn::Vec<DU8>;
  using VI8 = hn::Vec<DI8>;
  const size_t N8 = hn::Lanes(du8);

  VI32 sum = hn::Zero(di32);
  const VI8 ones = hn::Set(di8, 1);

  // Table for Lookup32. Align to 32 bytes for efficient loading.
  // Maps indices 0-9 (from '0'-'9') to 0-9, and indices 17-22 (from 'A'-'F')
  // to 10-15.
  HWY_ALIGN static constexpr uint8_t kTable[32] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0,
      0, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  const VU8 char_zero = hn::Set(du8, '0');

  auto accumulate = [&](const VU8 v_ascii) HWY_ATTR {
    // Saturated sub is available for UINT8 and UINT16 as dedicated instruction.
    const VU8 v_idx = hn::SaturatedSub(v_ascii, char_zero);
    // Will use either 1 or 2 lookup instructions.
    const VU8 decoded = hn::Lookup32(du8, kTable, v_idx);
    // This is dedicated instruction that performs
    // i32 + i8 * u8 -> i32, allowing for efficient accumulation without
    // overflow.
    sum = hn::SumOfMulQuadAccumulate(di32, decoded, ones, sum);
  };

  size_t i = 0;
  const uint8_t* HWY_RESTRICT hex_chars_as_int8 =
      HWY_RCAST_ALIGNED(const uint8_t*, hex_chars + i);
  // Process in blocks of N8
  for (; i + N8 <= count; i += N8) {
    const VU8 v_ascii =
        hn::LoadU(du8, hex_chars_as_int8);
    accumulate(v_ascii);
  }

  // Handle remainder
  size_t remainder = count - i;
  if (remainder > 0) {
    // LoadN pads with 0.
    // And thanks to saturated sub, padded elements safely decode as 0.
    const VU8 v_ascii = hn::LoadN(
        du8, hex_chars_as_int8 + i, remainder);
    accumulate(v_ascii);
  }

  return hn::ReduceSum(di32, sum);
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {

HWY_EXPORT(SumHexSIMD);

int32_t CallSumHexSIMD(const char* HWY_RESTRICT hex_chars, size_t count) {
  return HWY_DYNAMIC_DISPATCH(SumHexSIMD)(hex_chars, count);
}

int32_t SumHexScalar(const char* hex_chars, size_t count) {
  int32_t sum = 0;
  for (size_t i = 0; i < count; ++i) {
    char c = hex_chars[i];
    if (c >= '0' && c <= '9') {
      sum += c - '0';
    } else if (c >= 'A' && c <= 'F') {
      sum += c - 'A' + 10;
    }
  }
  return sum;
}

std::string GenerateHexString(size_t len, size_t offset = 0) {
  std::string s;
  s.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    const int r = static_cast<int>((i + offset) % 16);
    if (r < 10) {
      s += static_cast<char>('0' + r);
    } else {
      s += static_cast<char>('A' + (r - 10));
    }
  }
  return s;
}

bool TestOne(const std::string& test_str) {
  int32_t expected = SumHexScalar(test_str.data(), test_str.length());
  int32_t actual = CallSumHexSIMD(test_str.data(), test_str.length());
  if (expected != actual) {
    printf("Failed for \"%s\" (len %zu): expected %d, got %d\n",
           test_str.c_str(), test_str.length(), expected, actual);
    return false;
  }
  return true;
}

int RunTests() {
  std::vector<std::string> test_cases;

  // Generate some random hex strings of various lengths, including edge cases
  for (size_t len :
       {0, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1000, 1001}) {
    test_cases.push_back(GenerateHexString(len));
  }

  bool all_passed = true;
  for (const auto& tc : test_cases) {
    if (!TestOne(tc)) {
      all_passed = false;
    }
  }

  if (!all_passed) {
    printf("Validation FAILED!\n");
    return 1;
  }

  printf("Validation PASSED.\n");

  // Benchmark
  const size_t bench_len = 10'000;
  const int reps = 10'000;

  // Use Unpredictable1 to ensure the string content is not known at compile
  // time
  const std::string bench_str =
      GenerateHexString(bench_len, hwy::Unpredictable1());

  printf("Benchmarking with string of length %zu (%d reps)...\n", bench_len,
         reps);

  const double t_scalar_0 = hwy::platform::Now();
  int32_t scalar_sum = 0;
  for (int r = 0; r < reps; ++r) {
    scalar_sum += SumHexScalar(bench_str.data(), bench_len);
  }
  const double t_scalar_1 = hwy::platform::Now();
  const double dt_scalar = t_scalar_1 - t_scalar_0;

  const double t_simd_0 = hwy::platform::Now();
  int32_t simd_sum = 0;
  for (int r = 0; r < reps; ++r) {
    simd_sum += CallSumHexSIMD(bench_str.data(), bench_len);
  }
  const double t_simd_1 = hwy::platform::Now();
  const double dt_simd = t_simd_1 - t_simd_0;

  // Print results to ensure they are not optimized away
  printf("Scalar sum: %d (took %f seconds)\n", scalar_sum, dt_scalar);
  printf("SIMD sum:   %d (took %f seconds, speedup: %fx)\n", simd_sum, dt_simd,
         dt_scalar / dt_simd);

  return 0;
}

}  // namespace hwy

int main() { return hwy::RunTests(); }
#endif  // HWY_ONCE
