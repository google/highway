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
#include <string.h>

#include <vector>

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/iguana/ans_test.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/iguana/ans.h"
#include "hwy/contrib/iguana/ans-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

namespace ans = hwy::iguana_ans::HWY_NAMESPACE;

// Skewed toward small values (compressible), or uniform when skew == 0.
std::vector<uint8_t> MakeData(size_t n, uint64_t seed, int skew) {
  RandomState rng(seed);
  std::vector<uint8_t> v(n);
  for (auto& x : v) {
    if (skew == 0) {
      x = static_cast<uint8_t>(Random32(&rng));
    } else {
      uint32_t a = 0;
      for (int k = 0; k < skew; ++k) a += Random32(&rng) & 0x3F;
      x = static_cast<uint8_t>(a);
    }
  }
  return v;
}

void RoundTrip(const std::vector<uint8_t>& data) {
  std::vector<uint8_t> enc = hwy::iguana::Ans32Encode(data.data(), data.size());
  HWY_ASSERT(!enc.empty());

  std::vector<uint8_t> dec(data.size(), 0xCD);
  HWY_ASSERT(ans::Ans32Decode(enc.data(), enc.size(), dec.data(), data.size()));
  HWY_ASSERT(data.empty() || memcmp(dec.data(), data.data(), data.size()) == 0);

  // The SIMD decoder must match the scalar reference bit-for-bit.
  std::vector<uint8_t> dec2(data.size(), 0xAB);
  HWY_ASSERT(hwy::iguana::Ans32DecodeScalar(enc.data(), enc.size(), dec2.data(),
                                            data.size()));
  HWY_ASSERT(data.empty() ||
             memcmp(dec2.data(), data.data(), data.size()) == 0);
}

void TestRoundTripSizes() {
  const size_t kSmallSizes[] = {0,   1,   2,   31,  32,  33,  63,
                                64,  65,  100, 255, 256, 257, 1024};
  for (size_t n : kSmallSizes) {
    RoundTrip(MakeData(n, n * 3 + 1, 3));
    RoundTrip(MakeData(n, n * 5 + 2, 1));
    RoundTrip(MakeData(n, n * 7 + 3, 0));
  }

  // >1024: AdjustedReps keeps release builds at the original sizes but
  // shrinks them for slow debug/emulated (RVV, ARM, MSVC) runs, so the
  // test doesn't time out there.
  const size_t kLargeSizes[] = {AdjustedReps(4096), AdjustedReps(4097),
                                AdjustedReps(65535), AdjustedReps(65536),
                                AdjustedReps(70000)};
  for (size_t n : kLargeSizes) {
    RoundTrip(MakeData(n, n * 3 + 1, 3));
    RoundTrip(MakeData(n, n * 5 + 2, 1));
    RoundTrip(MakeData(n, n * 7 + 3, 0));
  }
}

void TestRoundTripModels() {
  // Large skewed input (compresses; exercises many vector rounds).
  RoundTrip(MakeData(AdjustedReps(300000), 0xABCDEF, 3));
  // Single distinct byte.
  RoundTrip(std::vector<uint8_t>(AdjustedReps(5000), 0x42));
  // Two symbols.
  std::vector<uint8_t> two(AdjustedReps(4000));
  for (size_t i = 0; i < two.size(); ++i) {
    two[i] = static_cast<uint8_t>((i % 7) ? 0x10 : 0x20);
  }
  RoundTrip(two);
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(IguanaAnsTest);
HWY_EXPORT_AND_TEST_P(IguanaAnsTest, TestRoundTripSizes);
HWY_EXPORT_AND_TEST_P(IguanaAnsTest, TestRoundTripModels);
HWY_AFTER_TEST();
}  // namespace hwy
#endif
