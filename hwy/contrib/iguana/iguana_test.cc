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

std::vector<uint8_t> MakeData(size_t n, uint64_t seed, int mode) {
  RandomState rng(seed | 1);
  std::vector<uint8_t> v(n);
  if (mode == 0) {
    for (auto& x : v) x = static_cast<uint8_t>(Random64(&rng));
  } else if (mode == 1) {  // skewed
    for (auto& x : v) {
      uint32_t a = 0;
      for (int k = 0; k < 3; ++k)
        a += static_cast<uint32_t>(Random64(&rng) & 0x3F);
      x = static_cast<uint8_t>(a);
    }
  } else {  // repetitive words (compresses well)
    static const char* const w[] = {"the ",   "quick ", "brown ", "fox ",
                                    "jumps ", "over ",  "lazy ",  "dog "};
    std::string s;
    while (s.size() < n) s += w[Random64(&rng) & 7];
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


// Malformed or truncated inputs must be rejected: no crash, and no allocation
// driven by an attacker-controlled length.
void TestRejectsMalformed() {
  std::vector<uint8_t> out;

  // No header at all.
  HWY_ASSERT(!hwy::iguana::DecompressScalar(nullptr, 0, out));
  HWY_ASSERT(!ig::Decompress(nullptr, 0, out));

  // A hand-built block whose very first LZ token is a match that reuses the
  // "previous offset" (bit 0x80) when there has not been one yet - the offset
  // would be zero, i.e. read from before the start of the output.
  // Layout: [tokens payload][ulen 5..0][hdr][cmd][uncompressed_len], the
  // control bytes read backwards from the end.
  {
    const uint8_t first_match_reuse[] = {
        0xA0,                                            // token: reuse offs, len 4
        0x80, 0x80, 0x80, 0x80, 0x80, 0x81,              // ulens: 0,0,0,0,0,1 (tokens)
        0x80,                                            // hdr: all streams raw
        0x80 | hwy::iguana::kCmdDecodeIguana,            // command (last)
        0x84,                                            // uncompressed_len = 4
    };
    HWY_ASSERT(!hwy::iguana::DecompressScalar(first_match_reuse,
                                              sizeof(first_match_reuse), out));
    HWY_ASSERT(!ig::Decompress(first_match_reuse,
                               sizeof(first_match_reuse), out));
  }

  // Every truncation of a valid block is rejected or decodes a shorter output.
  const std::vector<uint8_t> data = MakeData(5000, 777, 2);
  const std::vector<uint8_t> comp =
      hwy::iguana::Compress(data.data(), data.size());
  HWY_ASSERT(!comp.empty());
  for (size_t n = 0; n < comp.size(); ++n) {
    std::vector<uint8_t> scalar;
    std::vector<uint8_t> simd;
    const bool ok_scalar = hwy::iguana::DecompressScalar(comp.data(), n, scalar);
    const bool ok_simd = ig::Decompress(comp.data(), n, simd);
    HWY_ASSERT(ok_scalar == ok_simd);
    if (ok_scalar) {
      // A truncated block may decode, but never to more than the original.
      HWY_ASSERT(simd == scalar);
      HWY_ASSERT(scalar.size() <= data.size());
    }
  }

  // Single-byte mutations: accepted output is still bounded by the cap.
  for (size_t i = 0; i < comp.size(); i += 5) {
    std::vector<uint8_t> mutated = comp;
    mutated[i] ^= 0xFF;
    std::vector<uint8_t> dec;
    if (hwy::iguana::DecompressScalar(mutated.data(), mutated.size(), dec)) {
      HWY_ASSERT(dec.size() <= (size_t{1} << 30));
    }
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
HWY_EXPORT_AND_TEST_P(IguanaTest, TestRejectsMalformed);
HWY_AFTER_TEST();
}  // namespace hwy
#endif
