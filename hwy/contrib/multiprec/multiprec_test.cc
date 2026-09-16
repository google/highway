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

#include <type_traits>

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

// The bit-width alias rounds up to whole 52-bit limbs, and the widest supported
// size stays within the accumulator limit.
static_assert(std::is_same<WideMulBits<128>, WideMul<3>>::value,
              "128 / 52 rounds up to 3 limbs");
static_assert(std::is_same<WideMulBits<256>, WideMul<5>>::value,
              "256 / 52 rounds up to 5 limbs");
static_assert(WideMulBits<2048>::kNumResultLimbs == 80,
              "2048 bits is 40 limbs, so 80 limbs of result");

template <size_t kNumLimbs>
struct TestWideMul {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    constexpr size_t kResultLimbs = 2 * kNumLimbs;

    AlignedFreeUniquePtr<T[]> a[kNumLimbs];
    AlignedFreeUniquePtr<T[]> b[kNumLimbs];
    AlignedFreeUniquePtr<T[]> expected[kResultLimbs];
    for (size_t i = 0; i < kNumLimbs; ++i) {
      a[i] = AllocateAligned<T>(N);
      b[i] = AllocateAligned<T>(N);
      HWY_ASSERT(a[i] && b[i]);
    }
    for (size_t k = 0; k < kResultLimbs; ++k) {
      expected[k] = AllocateAligned<T>(N);
      HWY_ASSERT(expected[k]);
    }

    // Two rounds: random limbs, then all-ones limbs, i.e. the largest
    // representable inputs, which exercise every carry (and also squaring).
    for (size_t round = 0; round < 2; ++round) {
      RandomState rng(12345);
      for (size_t i = 0; i < kNumLimbs; ++i) {
        for (size_t l = 0; l < N; ++l) {
          if (round == 0) {
            a[i][l] = static_cast<T>(Random64(&rng) & kWideMulLimbMask);
            b[i][l] = static_cast<T>(Random64(&rng) & kWideMulLimbMask);
          } else {
            a[i][l] = static_cast<T>(kWideMulLimbMask);
            b[i][l] = static_cast<T>(kWideMulLimbMask);
          }
        }
      }

      // Scalar reference, one lane at a time: accumulate the low and high
      // halves of each 128-bit product per output limb, then ripple the
      // carries.
      for (size_t l = 0; l < N; ++l) {
        uint64_t low[kResultLimbs - 1] = {0};
        uint64_t high[kResultLimbs - 1] = {0};
        for (size_t i = 0; i < kNumLimbs; ++i) {
          for (size_t j = 0; j < kNumLimbs; ++j) {
            uint64_t hi;
            const uint64_t lo = Mul128(a[i][l], b[j][l], &hi);
            low[i + j] += lo & kWideMulLimbMask;
            high[i + j] +=
                (lo >> kWideMulLimbBits) | (hi << (64 - kWideMulLimbBits));
          }
        }
        uint64_t carry = 0;
        for (size_t k = 0; k + 1 < kResultLimbs; ++k) {
          const uint64_t sum = low[k] + carry;
          expected[k][l] = static_cast<T>(sum & kWideMulLimbMask);
          carry = (sum >> kWideMulLimbBits) + high[k];
        }
        expected[kResultLimbs - 1][l] =
            static_cast<T>(carry & kWideMulLimbMask);
      }

      // The API works on flat arrays of T: vector types are sizeless on SVE and
      // RVV, so they cannot be stored in arrays.
      AlignedFreeUniquePtr<T[]> va = AllocateAligned<T>(kNumLimbs * N);
      AlignedFreeUniquePtr<T[]> vb = AllocateAligned<T>(kNumLimbs * N);
      AlignedFreeUniquePtr<T[]> vout = AllocateAligned<T>(kResultLimbs * N);
      HWY_ASSERT(va && vb && vout);
      for (size_t i = 0; i < kNumLimbs; ++i) {
        CopyBytes(a[i].get(), va.get() + i * N, N * sizeof(T));
        CopyBytes(b[i].get(), vb.get() + i * N, N * sizeof(T));
      }

      WideMul<kNumLimbs>::Mul(d, va.get(), vb.get(), vout.get());
      for (size_t k = 0; k < kResultLimbs; ++k) {
        for (size_t l = 0; l < N; ++l) {
          HWY_ASSERT_EQ(expected[k][l], vout[k * N + l]);
        }
      }

      // Karatsuba, for the sizes that use it: needs scratch, and must agree
      // with the same reference.
      if constexpr (WideMul<kNumLimbs>::kScratchLimbs != 0) {
        constexpr size_t kScratchLimbs = WideMul<kNumLimbs>::kScratchLimbs;
        AlignedFreeUniquePtr<T[]> scratch =
            AllocateAligned<T>(kScratchLimbs * N);
        HWY_ASSERT(scratch != nullptr);
        for (size_t i = 0; i < kResultLimbs * N; ++i) vout[i] = T{0};
        WideMul<kNumLimbs>::Mul(d, va.get(), vb.get(), vout.get(),
                                scratch.get());
        for (size_t k = 0; k < kResultLimbs; ++k) {
          for (size_t l = 0; l < N; ++l) {
            HWY_ASSERT_EQ(expected[k][l], vout[k * N + l]);
          }
        }
      }
    }
  }
};

// Independent reference: schoolbook with 128-bit partial products added
// limb-by-limb with ripple carry (different structure from the limb-based
// kernel, so agreement is meaningful).
template <size_t kLimbs>
void RefWideMul(const uint64_t* HWY_RESTRICT a, const uint64_t* HWY_RESTRICT b,
                uint64_t* HWY_RESTRICT out) {
  for (size_t k = 0; k < 2 * kLimbs; ++k) out[k] = 0;
  for (size_t i = 0; i < kLimbs; ++i) {
    for (size_t j = 0; j < kLimbs; ++j) {
      uint64_t hi;
      const uint64_t lo = Mul128(a[i], b[j], &hi);
      const size_t k = i + j;
      // Add lo at limb k.
      const uint64_t t0 = out[k] + lo;
      uint64_t carry = (t0 < lo) ? 1 : 0;
      out[k] = t0;
      // Add hi plus that carry at limb k+1.
      const uint64_t t1 = out[k + 1] + hi;
      uint64_t c = (t1 < hi) ? 1 : 0;
      const uint64_t t2 = t1 + carry;
      c += (t2 < carry) ? 1 : 0;
      out[k + 1] = t2;
      carry = c;
      // Ripple the remaining carry.
      for (size_t m = k + 2; carry != 0 && m < 2 * kLimbs; ++m) {
        const uint64_t t = out[m] + carry;
        carry = (t < carry) ? 1 : 0;
        out[m] = t;
      }
    }
  }
}

// ---- Montgomery reference: modular arithmetic on 52-bit limbs by binary
// double-and-add, i.e. no Montgomery and no R. Independent of the CIOS kernel,
// so agreement is meaningful. All values are below n < 2^(52*kNumLimbs), so
// scalar 64-bit arithmetic suffices.

// r = 2r mod n, in place; requires r < n. `carry` is the bit shifted out of the
// top limb (0 or 1), which SubIfGeRef needs to compare correctly.
template <size_t kNumLimbs>
uint64_t DoubleModRef(uint64_t* r) {
  uint64_t carry = 0;
  for (size_t i = 0; i < kNumLimbs; ++i) {
    const uint64_t t = (r[i] << 1) | carry;
    carry = r[i] >> (kWideMulLimbBits - 1);
    r[i] = t & kWideMulLimbMask;
  }
  return carry;
}

// r = (extra, r) - n if that is non-negative; requires (extra, r) < 2n.
template <size_t kNumLimbs>
void SubIfGeRef(uint64_t* r, const uint64_t* n, uint64_t extra) {
  uint64_t diff[kNumLimbs];
  uint64_t borrow = 0;
  for (size_t i = 0; i < kNumLimbs; ++i) {
    const uint64_t sub = n[i] + borrow;
    diff[i] = (r[i] - sub) & kWideMulLimbMask;
    borrow = r[i] < sub ? 1 : 0;
  }
  if (extra >= borrow) {  // no borrow out, so r >= n
    for (size_t i = 0; i < kNumLimbs; ++i) r[i] = diff[i];
  }
}

// r = r + a mod n; requires r, a < n.
template <size_t kNumLimbs>
void AddModRef(uint64_t* r, const uint64_t* a, const uint64_t* n) {
  uint64_t carry = 0;
  for (size_t i = 0; i < kNumLimbs; ++i) {
    const uint64_t t = r[i] + a[i] + carry;
    r[i] = t & kWideMulLimbMask;
    carry = t >> kWideMulLimbBits;
  }
  SubIfGeRef<kNumLimbs>(r, n, carry);
}

// out = a * b mod n.
template <size_t kNumLimbs>
void MulModRef(const uint64_t* a, const uint64_t* b, const uint64_t* n,
               uint64_t* out) {
  uint64_t r[kNumLimbs] = {};
  for (int bit = 52 * static_cast<int>(kNumLimbs) - 1; bit >= 0; --bit) {
    SubIfGeRef<kNumLimbs>(r, n, DoubleModRef<kNumLimbs>(r));
    if ((b[bit / 52] >> (bit % 52)) & 1) AddModRef<kNumLimbs>(r, a, n);
  }
  for (size_t i = 0; i < kNumLimbs; ++i) out[i] = r[i];
}

// out = a mod n for any a below 2^(52*kNumLimbs). The multiplier must be
// below n, so `a` is the multiplier and 1 the multiplicand: each step then
// only adds 1, which is below n.
template <size_t kNumLimbs>
void ModRef(const uint64_t* a, const uint64_t* n, uint64_t* out) {
  uint64_t one[kNumLimbs] = {};
  one[0] = 1;
  MulModRef<kNumLimbs>(one, a, n, out);
}

// out = 2^(52*kNumLimbs) mod n, i.e. R mod n.
template <size_t kNumLimbs>
void RModNRef(const uint64_t* n, uint64_t* out) {
  uint64_t r[kNumLimbs] = {};
  r[0] = 1;
  for (int i = 0; i < 52 * static_cast<int>(kNumLimbs); ++i) {
    SubIfGeRef<kNumLimbs>(r, n, DoubleModRef<kNumLimbs>(r));
  }
  for (size_t i = 0; i < kNumLimbs; ++i) out[i] = r[i];
}

template <size_t kBits>
struct TestWideMulLimbs {
  HWY_NOINLINE void operator()() {
    constexpr size_t kLimbs = kBits / 64;
    uint64_t a[kLimbs];
    uint64_t b[kLimbs];
    uint64_t actual[2 * kLimbs];
    uint64_t expected[2 * kLimbs];

    RandomState rng(6789);
    for (int trial = 0; trial < 20; ++trial) {
      for (size_t i = 0; i < kLimbs; ++i) {
        a[i] = Random64(&rng);
        b[i] = Random64(&rng);
      }
      WideMulLimbs<kBits>(a, b, actual);
      RefWideMul<kLimbs>(a, b, expected);
      for (size_t k = 0; k < 2 * kLimbs; ++k) {
        HWY_ASSERT_EQ(expected[k], actual[k]);
      }
    }

    // Maximum inputs: (2^kBits - 1)^2.
    for (size_t i = 0; i < kLimbs; ++i) a[i] = ~uint64_t{0};
    WideMulLimbs<kBits>(a, a, actual);
    RefWideMul<kLimbs>(a, a, expected);
    for (size_t k = 0; k < 2 * kLimbs; ++k) {
      HWY_ASSERT_EQ(expected[k], actual[k]);
    }
  }
};

HWY_NOINLINE void TestAllWideMulLimbs() {
  TestWideMulLimbs<128>()();
  TestWideMulLimbs<192>()();
  TestWideMulLimbs<256>()();
}

// Montgomery modular multiplication, checked against the binary reference. Each
// case covers four shapes across lanes: random operands, 0, 1, n-1, a modulus
// above R/2 (which needs the second conditional subtraction) and a modulus
// whose low limb is 1 (so n0 is -1).
template <size_t kNumLimbs>
struct TestMontgomery {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    constexpr size_t kScratchLimbs = Montgomery<kNumLimbs>::kScratchLimbs;

    AlignedFreeUniquePtr<T[]> values = AllocateAligned<T>(3 * kNumLimbs * N);
    AlignedFreeUniquePtr<T[]> actual = AllocateAligned<T>(kNumLimbs * N);
    AlignedFreeUniquePtr<T[]> expected = AllocateAligned<T>(kNumLimbs * N);
    AlignedFreeUniquePtr<T[]> scratch = AllocateAligned<T>(kScratchLimbs * N);
    HWY_ASSERT(values && actual && expected && scratch);

    T* const n = values.get();
    T* const mon_a = n + kNumLimbs * N;
    T* const mon_b = mon_a + kNumLimbs * N;

    RandomState rng(31337);
    for (size_t l = 0; l < N; ++l) {
      uint64_t nl[kNumLimbs], al[kNumLimbs], bl[kNumLimbs];
      uint64_t rl[kNumLimbs], tl[kNumLimbs], t0[kNumLimbs];
      for (size_t i = 0; i < kNumLimbs; ++i) {
        nl[i] = Random64(&rng) & kWideMulLimbMask;
        al[i] = Random64(&rng) & kWideMulLimbMask;
        bl[i] = Random64(&rng) & kWideMulLimbMask;
      }
      nl[0] |= 1;  // the modulus must be odd
      // Half the lanes get a modulus above R/2, so the second conditional
      // subtraction is exercised; the others keep the top limb small.
      if ((l & 1) != 0) {
        nl[kNumLimbs - 1] |= uint64_t{1} << (kWideMulLimbBits - 1);
      }
      if (l % 8 == 3) nl[0] = 1;  // n0 becomes -1
      // a and b must be below n, as the API documents: reduce them properly,
      // since a single conditional subtraction is not enough for small moduli.
      uint64_t ar[kNumLimbs], br[kNumLimbs];
      ModRef<kNumLimbs>(al, nl, ar);
      ModRef<kNumLimbs>(bl, nl, br);
      for (size_t i = 0; i < kNumLimbs; ++i) {
        al[i] = ar[i];
        bl[i] = br[i];
      }
      // Extremes, cycling per lane: 0, 1 and n-1.
      if (l % 4 == 1) {
        for (size_t i = 0; i < kNumLimbs; ++i) al[i] = 0;
      } else if (l % 4 == 2) {
        for (size_t i = 0; i < kNumLimbs; ++i) al[i] = i == 0 ? 1 : 0;
      } else if (l % 4 == 3) {
        for (size_t i = 0; i < kNumLimbs; ++i) al[i] = nl[i];
        --al[0];
      }

      // Reference: mon_a = a*R mod n, likewise for b, and the expected result
      // (a*b mod n)*R mod n, all with plain binary modular arithmetic.
      RModNRef<kNumLimbs>(nl, rl);
      MulModRef<kNumLimbs>(al, rl, nl, t0);
      for (size_t i = 0; i < kNumLimbs; ++i)
        mon_a[i * N + l] = static_cast<T>(t0[i]);
      MulModRef<kNumLimbs>(bl, rl, nl, t0);
      for (size_t i = 0; i < kNumLimbs; ++i)
        mon_b[i * N + l] = static_cast<T>(t0[i]);
      MulModRef<kNumLimbs>(al, bl, nl, tl);
      MulModRef<kNumLimbs>(tl, rl, nl, t0);
      for (size_t i = 0; i < kNumLimbs; ++i)
        expected[i * N + l] = static_cast<T>(t0[i]);
      for (size_t i = 0; i < kNumLimbs; ++i)
        n[i * N + l] = static_cast<T>(nl[i]);
    }

    const Vec<D> n0 = MontgomeryN0(d, LoadU(d, n));
    Montgomery<kNumLimbs>::Mul(d, mon_a, mon_b, n, n0, actual.get(),
                               scratch.get());
    for (size_t i = 0; i < kNumLimbs; ++i) {
      for (size_t l = 0; l < N; ++l) {
        HWY_ASSERT_EQ(expected[i * N + l], actual[i * N + l]);
      }
    }
  }
};

HWY_NOINLINE void TestAllMontgomery() {
  ForPartialVectors<TestMontgomery<1>>()(uint64_t());
  ForPartialVectors<TestMontgomery<2>>()(uint64_t());
  ForPartialVectors<TestMontgomery<4>>()(uint64_t());
  ForPartialVectors<TestMontgomery<8>>()(uint64_t());
}

// R^2 mod n and the conversions built on it: entering Montgomery form is
// Mul(x, R2), and leaving it is Mul(y, one) where one = {1, 0, ...}. Both are
// checked against the binary reference, and the round trip must return x.
template <size_t kNumLimbs>
struct TestMontgomeryForms {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    constexpr size_t kScratch = Montgomery<kNumLimbs>::kScratchLimbs;

    AlignedFreeUniquePtr<T[]> values = AllocateAligned<T>(5 * kNumLimbs * N);
    AlignedFreeUniquePtr<T[]> r2 = AllocateAligned<T>(kNumLimbs * N);
    AlignedFreeUniquePtr<T[]> in_form = AllocateAligned<T>(kNumLimbs * N);
    AlignedFreeUniquePtr<T[]> back = AllocateAligned<T>(kNumLimbs * N);
    AlignedFreeUniquePtr<T[]> scratch = AllocateAligned<T>(kScratch * N);
    AlignedFreeUniquePtr<T[]> r2_scratch =
        AllocateAligned<T>(MontgomeryR2ScratchLimbs<kNumLimbs>() * N);
    HWY_ASSERT(values && r2 && in_form && back && scratch && r2_scratch);

    T* const n = values.get();
    T* const x = n + kNumLimbs * N;
    T* const one = x + kNumLimbs * N;

    RandomState rng(90210);
    for (size_t l = 0; l < N; ++l) {
      uint64_t nl[kNumLimbs], xl[kNumLimbs];
      for (size_t i = 0; i < kNumLimbs; ++i) {
        nl[i] = Random64(&rng) & kWideMulLimbMask;
        xl[i] = Random64(&rng) & kWideMulLimbMask;
      }
      nl[0] |= 1;  // odd modulus
      if ((l & 1) != 0) {
        nl[kNumLimbs - 1] |= uint64_t{1} << (kWideMulLimbBits - 1);
      }
      uint64_t xr[kNumLimbs];
      ModRef<kNumLimbs>(xl, nl, xr);  // x < n, as Montgomery<>::Mul requires
      for (size_t i = 0; i < kNumLimbs; ++i) xl[i] = xr[i];
      for (size_t i = 0; i < kNumLimbs; ++i)
        n[i * N + l] = static_cast<T>(nl[i]);
      for (size_t i = 0; i < kNumLimbs; ++i)
        x[i * N + l] = static_cast<T>(xl[i]);
      for (size_t i = 0; i < kNumLimbs; ++i) {
        one[i * N + l] = static_cast<T>(i == 0 ? 1 : 0);
      }
    }
    MontgomeryR2<kNumLimbs>(d, n, r2.get(), r2_scratch.get());
    const Vec<D> n0 = MontgomeryN0(d, LoadU(d, n));
    Montgomery<kNumLimbs>::Mul(d, x, r2.get(), n, n0, in_form.get(),
                               scratch.get());
    Montgomery<kNumLimbs>::Mul(d, in_form.get(), one, n, n0, back.get(),
                               scratch.get());

    for (size_t l = 0; l < N; ++l) {
      uint64_t nl[kNumLimbs], xl[kNumLimbs];
      uint64_t r1[kNumLimbs], r2_ref[kNumLimbs], in_ref[kNumLimbs];
      for (size_t i = 0; i < kNumLimbs; ++i) {
        nl[i] = n[i * N + l];
        xl[i] = x[i * N + l];
      }
      RModNRef<kNumLimbs>(nl, r1);
      MulModRef<kNumLimbs>(r1, r1, nl, r2_ref);
      MulModRef<kNumLimbs>(xl, r1, nl, in_ref);
      for (size_t i = 0; i < kNumLimbs; ++i) {
        HWY_ASSERT_EQ(static_cast<T>(r2_ref[i]), r2[i * N + l]);
        HWY_ASSERT_EQ(static_cast<T>(in_ref[i]), in_form[i * N + l]);
        HWY_ASSERT_EQ(static_cast<T>(xl[i]), back[i * N + l]);
      }
    }
  }
};

HWY_NOINLINE void TestAllMontgomeryForms() {
  ForPartialVectors<TestMontgomeryForms<1>>()(uint64_t());
  ForPartialVectors<TestMontgomeryForms<2>>()(uint64_t());
  ForPartialVectors<TestMontgomeryForms<4>>()(uint64_t());
  ForPartialVectors<TestMontgomeryForms<8>>()(uint64_t());
}

HWY_NOINLINE void TestAllWideMul() {
  ForPartialVectors<TestWideMul<1>>()(uint64_t());
  ForPartialVectors<TestWideMul<2>>()(uint64_t());
  ForPartialVectors<TestWideMul<3>>()(uint64_t());
  ForPartialVectors<TestWideMul<4>>()(uint64_t());
  // Larger sizes, to cover more than one inner iteration.
  ForPartialVectors<TestWideMul<8>>()(uint64_t());
  ForPartialVectors<TestWideMul<16>>()(uint64_t());
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
HWY_EXPORT_AND_TEST_P(MultiprecTest, TestAllMontgomery);
HWY_EXPORT_AND_TEST_P(MultiprecTest, TestAllMontgomeryForms);
HWY_EXPORT_AND_TEST_P(MultiprecTest, TestAllWideMulLimbs);
HWY_AFTER_TEST();

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
#endif  // HWY_ONCE
