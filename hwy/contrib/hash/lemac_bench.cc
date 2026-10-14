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

#ifndef HWY_DISABLED_TARGETS
// VAES requires at least AVX3_DL; no benefit from later AVX-512 targets.
#define HWY_DISABLED_TARGETS                                                   \
  (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4 | HWY_AVX3 | HWY_AVX3_SPR | HWY_AVX3_ZEN4 | \
   HWY_AVX10_2)
#endif  // HWY_DISABLED_TARGETS

#include "hwy/aligned_allocator.h"
#include "hwy/nanobenchmark.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/hash/lemac_bench.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/hash/lemac-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
// LeMac is not available on HWY_SCALAR and HWY_EMU128, see lemac-inl.h.
#if HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128

// Typical sizes for tree hashing: headers, then chunks of 32 KiB to 256 KiB.
constexpr size_t kSizes[] = {1024, 32768, 65536, 262144};
constexpr size_t kMaxSize = 262144;

// Prints the duration of `func`, which hashes `size` bytes.
template <class Func>
HWY_NOINLINE void Report(const char* name, size_t size, const Func& func) {
  FuncInput input = static_cast<FuncInput>(Unpredictable1());
  Result results[1];
  Params params;
  params.seconds_per_eval = 10E-3;
  params.max_evals = HWY_IS_DEBUG_BUILD ? 3 : 6;
  params.verbose = false;
  const size_t num_results = MeasureClosure(func, &input, 1, results, params);
  if (num_results != 1) {
    HWY_WARN("Measurement failed.");
    return;
  }
  const double ns =
      results[0].ticks / platform::InvariantTicksPerSecond() * 1E9;
  printf("%17s %8zu: %10.1f ns = %5.1f GB/s; measurement MAD=%4.2f%%\n", name,
         size, ns, static_cast<double>(size) / ns,
         results[0].variability * 100.0);
}

HWY_NOINLINE void BenchHashes() {
  auto in = AllocateAligned<uint8_t>(kMaxSize);
  HWY_ASSERT(in);
  for (size_t i = 0; i < kMaxSize; ++i) {
    in[i] = static_cast<uint8_t>((i * 0x9E3779B1u) >> 24);
  }
  uint8_t key_bytes[LeMac::kKeyBytes];
  for (size_t i = 0; i < LeMac::kKeyBytes; ++i) {
    key_bytes[i] = static_cast<uint8_t>(i * 17 + 1);
  }
  const uint8_t nonce[LeMac::kNonceBytes] = {1};
  const LeMacKey<1> key1 = LeMacMakeKey<1>(key_bytes);
  const LeMacKey<2> key2 = LeMacMakeKey<2>(key_bytes);
  const LeMacFingerprintKey<1> fp_key1 = LeMacMakeFingerprintKey<1>();
  const LeMacFingerprintKey<2> fp_key2 = LeMacMakeFingerprintKey<2>();

  for (size_t size : kSizes) {
    Report("LeMac-x1", size, [&](FuncInput input) HWY_ATTR {
      in[0] = static_cast<uint8_t>(input);
      uint8_t hash[LeMac::kHashBytes];
      LeMacHash(key1, nonce, in.get(), size, hash);
      return static_cast<FuncOutput>(hash[0]);
    });
    Report("LeMac-x2", size, [&](FuncInput input) HWY_ATTR {
      in[0] = static_cast<uint8_t>(input);
      uint8_t hash[LeMac::kHashBytes];
      LeMacHash(key2, nonce, in.get(), size, hash);
      return static_cast<FuncOutput>(hash[0]);
    });
    Report("Fingerprint-x1", size, [&](FuncInput input) HWY_ATTR {
      in[0] = static_cast<uint8_t>(input);
      uint8_t hash[LeMac::kHashBytes];
      LeMacHash(fp_key1, nonce, in.get(), size, hash);
      return static_cast<FuncOutput>(hash[0]);
    });
    Report("Fingerprint-x2", size, [&](FuncInput input) HWY_ATTR {
      in[0] = static_cast<uint8_t>(input);
      uint8_t hash[LeMac::kHashBytes];
      LeMacHash(fp_key2, nonce, in.get(), size, hash);
      return static_cast<FuncOutput>(hash[0]);
    });
  }
}

#else   // HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128
void BenchHashes() {}
#endif  // HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(LeMacBench);
HWY_EXPORT_AND_TEST_P(LeMacBench, BenchHashes);
HWY_AFTER_TEST();
}  // namespace hwy
#endif  // HWY_ONCE
