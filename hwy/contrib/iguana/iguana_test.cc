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

#include <string>
#include <vector>

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/iguana/iguana_test.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/iguana/iguana.h"
#include "hwy/contrib/iguana/iguana-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

namespace ig = hwy::HWY_NAMESPACE::iguana_full;

uint64_t NextRandom(uint64_t& state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

std::vector<uint8_t> MakeData(size_t n, uint64_t seed, int mode) {
  uint64_t state = seed | 1;
  std::vector<uint8_t> v(n);
  if (mode == 0) {
    for (auto& x : v) x = static_cast<uint8_t>(NextRandom(state));
  } else if (mode == 1) {  // skewed
    for (auto& x : v) {
      uint32_t a = 0;
      for (int k = 0; k < 3; ++k)
        a += static_cast<uint32_t>(NextRandom(state) & 0x3F);
      x = static_cast<uint8_t>(a);
    }
  } else {  // repetitive words (compresses well)
    static const char* const w[] = {"the ",   "quick ", "brown ", "fox ",
                                    "jumps ", "over ",  "lazy ",  "dog "};
    std::string s;
    while (s.size() < n) s += w[NextRandom(state) & 7];
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(s[i]);
  }
  return v;
}

void RoundTrip(const std::vector<uint8_t>& data) {
  const std::vector<uint8_t> comp =
      hwy::iguana::Compress(data.data(), data.size());
  HWY_ASSERT(!comp.empty());

  std::vector<uint8_t> dec;
  HWY_ASSERT(ig::Decompress(comp.data(), comp.size(), dec));
  HWY_ASSERT(dec.size() == data.size());
  HWY_ASSERT(data.empty() || memcmp(dec.data(), data.data(), data.size()) == 0);

  std::vector<uint8_t> dec2;
  HWY_ASSERT(hwy::iguana::DecompressScalar(comp.data(), comp.size(), dec2));
  HWY_ASSERT(dec2 == dec);
}

void TestRoundTripSizes() {
  const size_t kSizes[] = {0,   1,    10,   31,    36,    37,    64,
                           100, 1000, 5000, 40000, 65536, 200000};
  for (size_t idx = 0; idx < sizeof(kSizes) / sizeof(kSizes[0]); ++idx) {
    const size_t n = kSizes[idx];
    RoundTrip(MakeData(n, n * 7 + 1, 0));
    RoundTrip(MakeData(n, n * 11 + 2, 1));
    RoundTrip(MakeData(n, n * 13 + 3, 2));
  }
}

void TestRoundTripStructure() {
  // A far-back (>64 KiB) repeat forces the 24-bit offset path.
  std::vector<uint8_t> a = MakeData(200000, 4242, 0);
  std::vector<uint8_t> v = a;
  v.insert(v.end(), a.begin(), a.begin() + 6000);
  v.insert(v.end(), a.begin() + 90000, a.begin() + 98000);
  RoundTrip(v);

  // Highly compressible.
  RoundTrip(std::vector<uint8_t>(100000, 0x5A));

  // A short quotable string re-sliced (exercises the raw-copy fallback).
  const std::string s =
      "this is a short string that we will re-slice for small test-cases";
  for (size_t i = 0; i < s.size(); ++i) {
    RoundTrip(
        std::vector<uint8_t>(s.begin() + static_cast<ptrdiff_t>(i), s.end()));
  }
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(IguanaTest);
HWY_EXPORT_AND_TEST_P(IguanaTest, TestRoundTripSizes);
HWY_EXPORT_AND_TEST_P(IguanaTest, TestRoundTripStructure);
HWY_AFTER_TEST();
}  // namespace hwy
#endif
