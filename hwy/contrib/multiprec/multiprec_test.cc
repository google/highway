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

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/multiprec/multiprec_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h:
#include "hwy/contrib/multiprec/multiprec-inl.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

#if HWY_TARGET != HWY_SCALAR

template <size_t kDigits>
struct TestWideMul {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    constexpr size_t kResultDigits = 2 * kDigits;

    AlignedFreeUniquePtr<T[]> a[kDigits];
    AlignedFreeUniquePtr<T[]> b[kDigits];
    AlignedFreeUniquePtr<T[]> expected[kResultDigits];
    for (size_t i = 0; i < kDigits; ++i) {
      a[i] = AllocateAligned<T>(N);
      b[i] = AllocateAligned<T>(N);
      HWY_ASSERT(a[i] && b[i]);
    }
    for (size_t k = 0; k < kResultDigits; ++k) {
      expected[k] = AllocateAligned<T>(N);
      HWY_ASSERT(expected[k]);
    }

    RandomState rng(12345);
    for (size_t i = 0; i < kDigits; ++i) {
      for (size_t l = 0; l < N; ++l) {
        a[i][l] = static_cast<T>(Random64(&rng) & kWideMulDigitMask);
        b[i][l] = static_cast<T>(Random64(&rng) & kWideMulDigitMask);
      }
    }

    // Scalar reference, one lane at a time; identical structure to the
    // implementation but with scalar 128-bit products.
    for (size_t l = 0; l < N; ++l) {
      uint64_t low[kResultDigits - 1] = {0};
      uint64_t high[kResultDigits - 1] = {0};
      for (size_t i = 0; i < kDigits; ++i) {
        for (size_t j = 0; j < kDigits; ++j) {
          uint64_t hi;
          const uint64_t lo = Mul128(a[i][l], b[j][l], &hi);
          low[i + j] += lo & kWideMulDigitMask;
          high[i + j] += (lo >> kWideMulDigitBits) | (hi << (64 - kWideMulDigitBits));
        }
      }
      uint64_t carry = 0;
      for (size_t k = 0; k + 1 < kResultDigits; ++k) {
        const uint64_t sum = low[k] + carry;
        expected[k][l] = static_cast<T>(sum & kWideMulDigitMask);
        carry = (sum >> kWideMulDigitBits) + high[k];
      }
      expected[kResultDigits - 1][l] =
          static_cast<T>(carry & kWideMulDigitMask);
    }

    Vec<D> va[kDigits];
    Vec<D> vb[kDigits];
    Vec<D> vout[kResultDigits];
    for (size_t i = 0; i < kDigits; ++i) {
      va[i] = Load(d, a[i].get());
      vb[i] = Load(d, b[i].get());
    }
    WideMul<kDigits>::Mul(d, va, vb, vout);
    for (size_t k = 0; k < kResultDigits; ++k) {
      HWY_ASSERT_VEC_EQ(d, expected[k].get(), vout[k]);
    }
  }
};

#endif  // HWY_TARGET != HWY_SCALAR

HWY_NOINLINE void TestAllWideMul() {
#if HWY_TARGET != HWY_SCALAR
  ForPartialVectors<TestWideMul<1>>()(uint64_t());
  ForPartialVectors<TestWideMul<2>>()(uint64_t());
  ForPartialVectors<TestWideMul<3>>()(uint64_t());
  ForPartialVectors<TestWideMul<4>>()(uint64_t());
#endif
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {

HWY_BEFORE_TEST(MultiprecTest);
HWY_EXPORT_AND_TEST_P(MultiprecTest, TestAllWideMul);
HWY_AFTER_TEST();

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
#endif  // HWY_ONCE
