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

#include <vector>

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/crc/crc_test.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/crc/crc-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

namespace crc = hwy::HWY_NAMESPACE::crc;

// Independent bit-at-a-time reference (no tables, no CLMUL).
uint64_t ReferenceCrc64Xz(const uint8_t* data, size_t size) {
  uint64_t crc = ~uint64_t{0};
  for (size_t i = 0; i < size; ++i) {
    crc ^= data[i];
    for (int b = 0; b < 8; ++b) {
      crc = (crc >> 1) ^ (0xC96C5795D7870F42ull & (uint64_t{0} - (crc & 1)));
    }
  }
  return ~crc;
}

uint64_t NextRandom(uint64_t& state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

// The frozen check value from the CRC catalogue.
void TestCheckValue() {
  const uint8_t kMsg[9] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  HWY_ASSERT_EQ(uint64_t{0x995DC9BBDF1939FAull},
                crc::Crc64Xz(kMsg, sizeof(kMsg)));
}

// Every length from 0 through several fold iterations must match the reference.
void TestAgainstReference() {
  std::vector<uint8_t> buf(4096);
  uint64_t state = 0x123456789ABCDEFull;
  for (size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<uint8_t>(NextRandom(state));
  }
  for (size_t n = 0; n <= 600; ++n) {
    HWY_ASSERT_EQ(ReferenceCrc64Xz(buf.data(), n), crc::Crc64Xz(buf.data(), n));
  }
  for (size_t n : {1023u, 1024u, 1025u, 4096u}) {
    HWY_ASSERT_EQ(ReferenceCrc64Xz(buf.data(), n), crc::Crc64Xz(buf.data(), n));
  }
}

// Feeding the data in chunks must equal a single call.
void TestStreaming() {
  std::vector<uint8_t> buf(9000);
  uint64_t state = 0xDEADBEEFCAFEull;
  for (size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<uint8_t>(NextRandom(state));
  }
  const uint64_t whole = crc::Crc64Xz(buf.data(), buf.size());

  for (size_t split : {size_t{0}, size_t{1}, size_t{15}, size_t{16}, size_t{17},
                       size_t{4096}, buf.size()}) {
    const uint64_t a = crc::Crc64Xz(buf.data(), split);
    const uint64_t b = crc::Crc64Xz(buf.data() + split, buf.size() - split, a);
    HWY_ASSERT_EQ(whole, b);
  }
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(CrcTest);
HWY_EXPORT_AND_TEST_P(CrcTest, TestCheckValue);
HWY_EXPORT_AND_TEST_P(CrcTest, TestAgainstReference);
HWY_EXPORT_AND_TEST_P(CrcTest, TestStreaming);
HWY_AFTER_TEST();
}  // namespace hwy
#endif
