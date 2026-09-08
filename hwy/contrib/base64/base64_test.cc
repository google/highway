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

#include <string>
#include <vector>

#include "hwy/base.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/base64/base64_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/base64/base64-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

std::string ScalarEncode(const uint8_t* input, const size_t input_size) {
  static const char kAlphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string encoded(Base64EncodedSize(input_size), '\0');
  size_t in = 0;
  size_t out = 0;
  if (input_size >= 3) {
    for (; in <= input_size - 3; in += 3, out += 4) {
      const uint8_t b0 = input[in + 0];
      const uint8_t b1 = input[in + 1];
      const uint8_t b2 = input[in + 2];
      encoded[out + 0] = kAlphabet[b0 >> 2];
      encoded[out + 1] = kAlphabet[((b0 & 3) << 4) | (b1 >> 4)];
      encoded[out + 2] = kAlphabet[((b1 & 15) << 2) | (b2 >> 6)];
      encoded[out + 3] = kAlphabet[b2 & 63];
    }
  }
  if (in != input_size) {
    const uint8_t b0 = input[in];
    encoded[out + 0] = kAlphabet[b0 >> 2];
    encoded[out + 1] =
        kAlphabet[(b0 & 3) << 4 |
                  (in + 1 < input_size ? input[in + 1] >> 4 : 0)];
    if (in + 1 < input_size) {
      encoded[out + 2] = kAlphabet[(input[in + 1] & 15) << 2];
      encoded[out + 3] = '=';
    } else {
      encoded[out + 2] = '=';
      encoded[out + 3] = '=';
    }
  }
  return encoded;
}

HWY_NOINLINE bool Base64DecodeNoInline(const char* input,
                                       const size_t input_size, uint8_t* output,
                                       size_t* output_size) {
  return Base64Decode(input, input_size, output, output_size);
}

HWY_NOINLINE void TestBase64KnownVectors() {
  struct TestCase {
    const char* decoded;
    const char* encoded;
  };
  static const TestCase kCases[] = {{"", ""},
                                    {"f", "Zg=="},
                                    {"fo", "Zm8="},
                                    {"foo", "Zm9v"},
                                    {"foob", "Zm9vYg=="},
                                    {"fooba", "Zm9vYmE="},
                                    {"foobar", "Zm9vYmFy"}};

  for (const auto& test : kCases) {
    const size_t decoded_size = strlen(test.decoded);
    std::vector<char> encoded(Base64EncodedSize(decoded_size));
    HWY_ASSERT_EQ(encoded.size(),
                  Base64Encode(reinterpret_cast<const uint8_t*>(test.decoded),
                               decoded_size, encoded.data()));
    HWY_ASSERT(std::string(encoded.begin(), encoded.end()) == test.encoded);

    std::vector<uint8_t> decoded((encoded.size() / 4) * 3);
    size_t actual_size = 0;
    HWY_ASSERT(Base64Decode(test.encoded, strlen(test.encoded), decoded.data(),
                            &actual_size));
    HWY_ASSERT_EQ(decoded_size, actual_size);
    HWY_ASSERT_ARRAY_EQ(reinterpret_cast<const uint8_t*>(test.decoded),
                        decoded.data(), decoded_size);
  }
}

HWY_NOINLINE void TestBase64Lengths() {
  RandomState rng;
  for (size_t size = 0; size <= 257; ++size) {
    std::vector<uint8_t> input(size);
    for (uint8_t& byte : input) byte = static_cast<uint8_t>(rng());

    const std::string expected = ScalarEncode(input.data(), input.size());
    std::vector<char> encoded(expected.size());
    HWY_ASSERT_EQ(encoded.size(),
                  Base64Encode(input.data(), input.size(), encoded.data()));
    HWY_ASSERT(std::string(encoded.begin(), encoded.end()) == expected);

    std::vector<uint8_t> decoded((encoded.size() / 4) * 3);
    size_t decoded_size = 0;
    HWY_ASSERT(Base64Decode(encoded.data(), encoded.size(), decoded.data(),
                            &decoded_size));
    HWY_ASSERT_EQ(input.size(), decoded_size);
    HWY_ASSERT_ARRAY_EQ(input.data(), decoded.data(), input.size());
  }
}

HWY_NOINLINE void TestBase64Invalid() {
  static const char* const kInvalid[] = {"A",        "AAA",
                                         "====",     "=AAA",
                                         "A=AA",     "AA=A",
                                         "AAA==",    "AA==AAAA",
                                         "AAA=AAAA", "AA A",
                                         "AA?A",     "AB==",
                                         "AAB=",     "Zm9vYmFyZm9vYmF?"};
  uint8_t decoded[64];
  for (const char* input : kInvalid) {
    size_t decoded_size = 99;
    HWY_ASSERT(
        !Base64DecodeNoInline(input, strlen(input), decoded, &decoded_size));
    HWY_ASSERT_EQ(0, decoded_size);
  }

  std::string invalid_block(64, 'A');
  invalid_block[37] = '?';
  size_t decoded_size = 99;
  HWY_ASSERT(!Base64DecodeNoInline(invalid_block.data(), invalid_block.size(),
                                   decoded, &decoded_size));
  HWY_ASSERT_EQ(0, decoded_size);

  invalid_block[37] = static_cast<char>(0xC1);
  decoded_size = 99;
  HWY_ASSERT(!Base64DecodeNoInline(invalid_block.data(), invalid_block.size(),
                                   decoded, &decoded_size));
  HWY_ASSERT_EQ(0, decoded_size);

  std::string invalid_batch(256, 'A');
  std::vector<uint8_t> batch_decoded(192);
  for (size_t block = 0; block < 4; ++block) {
    invalid_batch[block * 64 + 37] = '?';
    decoded_size = 99;
    HWY_ASSERT(!Base64DecodeNoInline(
        invalid_batch.data(), invalid_batch.size(), batch_decoded.data(),
        &decoded_size));
    HWY_ASSERT_EQ(0, decoded_size);
    invalid_batch[block * 64 + 37] = 'A';
  }
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(HwyBase64Test);
HWY_EXPORT_AND_TEST_P(HwyBase64Test, TestBase64KnownVectors);
HWY_EXPORT_AND_TEST_P(HwyBase64Test, TestBase64Lengths);
HWY_EXPORT_AND_TEST_P(HwyBase64Test, TestBase64Invalid);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
