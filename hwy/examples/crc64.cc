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

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/examples/crc64.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/crc/crc-inl.h"
#include "hwy/tests/test_util-inl.h"  // HWY_ASSERT_EQ

// Highway SIMD Tutorial: CRC-64/XZ
//
// This example demonstrates a portable CRC implementation backed by Highway's
// SIMD carryless-multiply operations. The same source dynamically dispatches
// to the best supported target while preserving identical CRC results.

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

uint64_t Crc64Example(const uint8_t* data, size_t size) {
  return crc::Crc64Xz(data, size);
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {

HWY_EXPORT(Crc64Example);

uint64_t ComputeCrc64(const uint8_t* data, size_t size) {
  return HWY_DYNAMIC_DISPATCH(Crc64Example)(data, size);
}

int RunCrc64Example() {
  // CRC-64/XZ's standard check string. This variant uses the ECMA-182
  // polynomial in reflected form, with all-ones init and xor-out.
  constexpr char kMessage[] = "123456789";
  constexpr uint64_t kExpected = 0x995DC9BBDF1939FAull;

  const uint64_t actual = ComputeCrc64(
      reinterpret_cast<const uint8_t*>(kMessage), sizeof(kMessage) - 1);

  HWY_ASSERT_EQ(kExpected, actual);
  return 0;
}

}  // namespace hwy

int main() { return hwy::RunCrc64Example(); }
#endif  // HWY_ONCE
