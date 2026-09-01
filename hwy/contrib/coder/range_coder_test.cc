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
#define HWY_TARGET_INCLUDE "hwy/contrib/coder/range_coder_test.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/coder/range_coder.h"
#include "hwy/contrib/coder/range_coder-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {  // required: unique per target
namespace {

namespace rc = hwy::HWY_NAMESPACE::range_coder;

// Small deterministic xorshift PRNG (test-only).
uint64_t NextRandom(uint64_t& state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

// Skewed toward small byte values (sum of masked randoms), so it compresses.
std::vector<uint8_t> MakeSkewed(size_t n, uint64_t seed) {
  uint64_t state = seed | 1;
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) {
    uint32_t x = 0;
    for (int k = 0; k < 3; ++k) {
      x += static_cast<uint32_t>(NextRandom(state) & 0x3F);
    }
    v[i] = static_cast<uint8_t>(x);
  }
  return v;
}

std::vector<uint8_t> MakeUniform(size_t n, uint64_t seed) {
  uint64_t state = seed | 1;
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = static_cast<uint8_t>(NextRandom(state));
  }
  return v;
}

// Encodes `data`, decodes it with both the SIMD and the scalar decoder, and
// checks both reproduce the input exactly.
void RoundTrip(const std::vector<uint8_t>& data, bool expect_shrink) {
  HWY_ASSERT(!data.empty());

  std::vector<uint32_t> freq(256, 0);
  for (size_t i = 0; i < data.size(); ++i) freq[data[i]]++;

  std::vector<uint32_t> cum_prob;
  HWY_ASSERT(CreateCumulativeProbs(cum_prob, freq));

  std::vector<uint32_t> table;
  BuildDecodeTable(256, cum_prob, table);

  std::vector<uint8_t> enc;
  EncodeInterleaved(data, enc, cum_prob);
  HWY_ASSERT(enc.size() >= kRangeLanes * 3 + 2);

  std::vector<uint8_t> dec(data.size(), 0xCD);
  HWY_ASSERT(rc::DecodeInterleaved(enc.data(), enc.size(), dec.data(),
                                   data.size(), table.data()));
  HWY_ASSERT(memcmp(dec.data(), data.data(), data.size()) == 0);

  std::vector<uint8_t> dec_scalar(data.size(), 0xAB);
  HWY_ASSERT(DecodeInterleavedScalar(enc.data(), enc.size(), dec_scalar.data(),
                                     data.size(), table.data()));
  HWY_ASSERT(memcmp(dec_scalar.data(), data.data(), data.size()) == 0);

  if (expect_shrink) HWY_ASSERT(enc.size() < data.size());
}

// Round-trips a range of sizes: below one 16-lane block, exactly one, several
// whole blocks, and whole blocks + a scalar-tail remainder.
void TestRoundTripSizes() {
  const size_t kSizes[] = {1,  2,   3,   15,   16,   17,   31,    32,   33,
                           64, 127, 256, 1023, 4096, 4097, 65535, 65536};
  for (size_t idx = 0; idx < sizeof(kSizes) / sizeof(kSizes[0]); ++idx) {
    const size_t n = kSizes[idx];
    RoundTrip(MakeSkewed(n, n * 3 + 1), /*expect_shrink=*/false);
    RoundTrip(MakeUniform(n, n * 7 + 5), /*expect_shrink=*/false);
  }
}

// A large skewed input must actually compress, and degenerate models
// (one or two live symbols) must still round-trip.
void TestRoundTripModels() {
  RoundTrip(MakeSkewed(200000, 0xABCDEF), /*expect_shrink=*/true);

  RoundTrip(std::vector<uint8_t>(5000, 0x42), /*expect_shrink=*/true);

  std::vector<uint8_t> two_syms(4000);
  for (size_t i = 0; i < two_syms.size(); ++i) {
    two_syms[i] = static_cast<uint8_t>((i % 5) ? 0x10 : 0x20);
  }
  RoundTrip(two_syms, /*expect_shrink=*/true);
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(RangeCoderTest);
HWY_EXPORT_AND_TEST_P(RangeCoderTest, TestRoundTripSizes);
HWY_EXPORT_AND_TEST_P(RangeCoderTest, TestRoundTripModels);
HWY_AFTER_TEST();
}  // namespace hwy
#endif
