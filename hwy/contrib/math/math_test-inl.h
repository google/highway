// Copyright 2020 Google LLC
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

// Shared test harness for math function tests (math_test.cc,
// fast_math_test.cc, math_trig_test.cc, math_hyper_test.cc, math_tan_test.cc).
// Provides TestMath (ULP-based error), TestMathRelative (relative error), and
// the DEFINE_MATH_TEST convenience macro.

// Normal include guard for target-independent parts
#ifndef HIGHWAY_HWY_CONTRIB_MATH_MATH_TEST_INL_H_
#define HIGHWAY_HWY_CONTRIB_MATH_MATH_TEST_INL_H_

#include <stdint.h>
#include <stdio.h>

#include <cmath>  // std::abs

#include "hwy/base.h"

#endif  // HIGHWAY_HWY_CONTRIB_MATH_MATH_TEST_INL_H_

// Per-target
#if defined(HIGHWAY_HWY_CONTRIB_MATH_MATH_TEST_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_MATH_MATH_TEST_TOGGLE
#undef HIGHWAY_HWY_CONTRIB_MATH_MATH_TEST_TOGGLE
#else
#define HIGHWAY_HWY_CONTRIB_MATH_MATH_TEST_TOGGLE
#endif

#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"  // IWYU pragma: export

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// We have had test failures caused by excess precision due to keeping
// intermediate results in 80-bit x87 registers. One such failure mode is that
// Log1p computes a 1.0 which is not exactly equal to 1.0f, causing is_pole to
// incorrectly evaluate to false.
#undef HWY_MATH_TEST_EXCESS_PRECISION
#if HWY_ARCH_X86_32 && HWY_COMPILER_GCC_ACTUAL && \
    (HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128)

// GCC 13+: because CMAKE_CXX_EXTENSIONS is OFF, we build with -std= and hence
// also -fexcess-precision=standard, so there is no problem. See #1708 and
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=323.
#if HWY_COMPILER_GCC_ACTUAL >= 1300
#define HWY_MATH_TEST_EXCESS_PRECISION 0

#else                  // HWY_COMPILER_GCC_ACTUAL < 1300

// The build system must enable SSE2, e.g. via HWY_CMAKE_SSE2 - see
// https://stackoverflow.com/questions/20869904/c-handling-of-excess-precision .
#if defined(__SSE2__)  // correct flag given, no problem
#define HWY_MATH_TEST_EXCESS_PRECISION 0
#else
#define HWY_MATH_TEST_EXCESS_PRECISION 1
#pragma message( \
    "Skipping scalar math_test on 32-bit x86 GCC <13 without HWY_CMAKE_SSE2")
#endif  // defined(__SSE2__)

#endif  // HWY_COMPILER_GCC_ACTUAL
#else   // not (x86-32, GCC, scalar target): running math_test normally
#define HWY_MATH_TEST_EXCESS_PRECISION 0
#endif  // HWY_ARCH_X86_32 etc

// ULP-based test: compares vector math function `fxN` against scalar reference
// `fx1` over the range [min, max], asserting max ULP error <= max_error_ulp.
template <class T, class D>
HWY_NOINLINE void TestMath(const char* name, T (*fx1)(T),
                           Vec<D> (*fxN)(D, VecArg<Vec<D>>), D d, T min, T max,
                           uint64_t max_error_ulp) {
  if (HWY_MATH_TEST_EXCESS_PRECISION) {
    static bool once = true;
    if (once) {
      once = false;
      HWY_WARN("Skipping math_test due to GCC issue with excess precision.\n");
    }
    return;
  }

  using UintT = MakeUnsigned<T>;

  const UintT min_bits = BitCastScalar<UintT>(min);
  const UintT max_bits = BitCastScalar<UintT>(max);

  // If min is negative and max is positive, the range needs to be broken into
  // two pieces, [+0, max] and [-0, min], otherwise [min, max].
  int range_count = 1;
  UintT ranges[2][2] = {{min_bits, max_bits}, {0, 0}};
  if ((min < T{0}) && (max > T{0})) {
    ranges[0][0] = BitCastScalar<UintT>(ConvertScalarTo<T>(+0.0));
    ranges[0][1] = max_bits;
    ranges[1][0] = BitCastScalar<UintT>(ConvertScalarTo<T>(-0.0));
    ranges[1][1] = min_bits;
    range_count = 2;
  } else {
    // If not splitting, ensure we iterate from smaller uint to larger uint.
    // For negative numbers, min (e.g. -1000) has larger uint representation
    // than max (e.g. -1).
    if (ranges[0][0] > ranges[0][1]) {
      auto tmp = ranges[0][0];
      ranges[0][0] = ranges[0][1];
      ranges[0][1] = tmp;
    }
  }

  uint64_t max_ulp = 0;
  // Emulation is slower, so cannot afford as many.
  constexpr UintT kSamplesPerRange =
      static_cast<UintT>(AdjustedReps(static_cast<size_t>(4000)));
  for (int range_index = 0; range_index < range_count; ++range_index) {
    const UintT start = ranges[range_index][0];
    const UintT stop = ranges[range_index][1];
    const UintT step = HWY_MAX(1, ((stop - start) / kSamplesPerRange));
    for (UintT value_bits = start; value_bits <= stop; value_bits += step) {
      // For reasons unknown, the HWY_MAX is necessary on RVV, otherwise
      // value_bits can be less than start, and thus possibly NaN.
      const T value =
          BitCastScalar<T>(HWY_MIN(HWY_MAX(start, value_bits), stop));
      const T actual = GetLane(fxN(d, Set(d, value)));
      const T expected = fx1(value);

      // Skip small inputs and outputs on armv7, it flushes subnormals to zero.
#if HWY_TARGET <= HWY_NEON_WITHOUT_AES && HWY_ARCH_ARM_V7
      if ((std::abs(value) < 1e-37f) || (std::abs(expected) < 1e-37f)) {
        continue;
      }
#endif

      const auto ulp = hwy::detail::ComputeUlpDelta(actual, expected);
      max_ulp = HWY_MAX(max_ulp, ulp);
      if (ulp > max_error_ulp) {
        fprintf(stderr, "%s: %s(%f) expected %E actual %E ulp %g max ulp %u\n",
                hwy::TypeName(T(), Lanes(d)).c_str(), name,
                static_cast<double>(value), static_cast<double>(expected),
                static_cast<double>(actual), static_cast<double>(ulp),
                static_cast<uint32_t>(max_error_ulp));
      }
    }
  }
  fprintf(stderr, "%s: %s max_ulp %g\n", hwy::TypeName(T(), Lanes(d)).c_str(),
          name, static_cast<double>(max_ulp));
  HWY_ASSERT(max_ulp <= max_error_ulp);
}

// Relative-error test: compares vector math function `fxN` against scalar
// reference `fx1` over the range [min, max], asserting max relative error <=
// max_relative_error. Skips expected values with |expected| <= min_abs_expected
// to avoid meaningless large relative errors for near-zero values.
template <class T, class D>
HWY_NOINLINE void TestMathRelative(const char* name, T (*fx1)(T),
                                   Vec<D> (*fxN)(D, VecArg<Vec<D>>), D d, T min,
                                   T max, double max_relative_error,
                                   uint64_t samples = 4000,
                                   double min_abs_expected = 0.0) {
  if (HWY_MATH_TEST_EXCESS_PRECISION) {
    static bool once = true;
    if (once) {
      once = false;
      HWY_WARN("Skipping math_test due to GCC issue with excess precision.\n");
    }
    return;
  }

  using UintT = MakeUnsigned<T>;

  const UintT min_bits = BitCastScalar<UintT>(min);
  const UintT max_bits = BitCastScalar<UintT>(max);

  // If min is negative and max is positive, the range needs to be broken into
  // two pieces, [+0, max] and [-0, min], otherwise [min, max].
  int range_count = 1;
  UintT ranges[2][2] = {{min_bits, max_bits}, {0, 0}};
  if ((min < T{0}) && (max > T{0})) {
    ranges[0][0] = BitCastScalar<UintT>(ConvertScalarTo<T>(+0.0));
    ranges[0][1] = max_bits;
    ranges[1][0] = BitCastScalar<UintT>(ConvertScalarTo<T>(-0.0));
    ranges[1][1] = min_bits;
    range_count = 2;
  } else {
    // If not splitting, ensure we iterate from smaller uint to larger uint.
    if (ranges[0][0] > ranges[0][1]) {
      auto tmp = ranges[0][0];
      ranges[0][0] = ranges[0][1];
      ranges[0][1] = tmp;
    }
  }

  double max_actual_rel_error = 0.0;
  double max_error_value = 0.0;
  double sum_rel_error = 0.0;
  uint64_t count = 0;
  // Emulation is slower, so cannot afford as many.
  const UintT kSamplesPerRange =
      static_cast<UintT>(AdjustedReps(static_cast<size_t>(samples)));
  for (int range_index = 0; range_index < range_count; ++range_index) {
    const UintT start = ranges[range_index][0];
    const UintT stop = ranges[range_index][1];
    const UintT step = HWY_MAX(1, ((stop - start) / kSamplesPerRange));
    for (UintT value_bits = start; value_bits <= stop; value_bits += step) {
      // For reasons unknown, the HWY_MAX is necessary on RVV, otherwise
      // value_bits can be less than start, and thus possibly NaN.
      const T value =
          BitCastScalar<T>(HWY_MIN(HWY_MAX(start, value_bits), stop));
      const T actual = GetLane(fxN(d, Set(d, value)));
      const T expected = fx1(value);

      // Skip small inputs and outputs on armv7, it flushes subnormals to zero.
#if HWY_TARGET <= HWY_NEON_WITHOUT_AES && HWY_ARCH_ARM_V7
      if ((std::abs(value) < 1e-37f) || (std::abs(expected) < 1e-37f)) {
        continue;
      }
#endif

      if (std::abs(expected) > min_abs_expected) {
        double rel = std::abs(static_cast<double>(actual) -
                              static_cast<double>(expected)) /
                     std::abs(static_cast<double>(expected));
        if (ScalarIsNaN(rel) || rel > max_actual_rel_error) {
          max_actual_rel_error = rel;
          max_error_value = static_cast<double>(value);
        }
        sum_rel_error += rel;
        count++;
        if (rel > max_relative_error) {
          static int print_count = 0;
          if (print_count < 10) {
            fprintf(stderr,
                    "%s: %s(%f) expected %E actual %E rel %E max rel %E\n",
                    hwy::TypeName(T(), Lanes(d)).c_str(), name,
                    static_cast<double>(value), static_cast<double>(expected),
                    static_cast<double>(actual), rel, max_relative_error);
            print_count++;
          }
        }
      }
    }
  }
  fprintf(stderr, "%s: %s max_rel_error %E at %E\n",
          hwy::TypeName(T(), Lanes(d)).c_str(), name, max_actual_rel_error,
          max_error_value);
  if (count > 0) {
    fprintf(stderr, "%s: %s avg_rel_error %E\n",
            hwy::TypeName(T(), Lanes(d)).c_str(), name,
            sum_rel_error / static_cast<double>(count));
  }
  HWY_ASSERT(max_actual_rel_error <= max_relative_error);
}

#undef DEFINE_MATH_TEST_FUNC
#define DEFINE_MATH_TEST_FUNC(NAME)                     \
  HWY_NOINLINE void TestAll##NAME() {                   \
    ForFloat3264Types(ForPartialVectors<Test##NAME>()); \
  }

#undef DEFINE_MATH_TEST
#define DEFINE_MATH_TEST(NAME, F32x1, F32xN, F32_MIN, F32_MAX, F32_ERROR, \
                         F64x1, F64xN, F64_MIN, F64_MAX, F64_ERROR)       \
  struct Test##NAME {                                                     \
    template <class T, class D>                                           \
    HWY_NOINLINE void operator()(T, D d) {                                \
      if (sizeof(T) == 4) {                                               \
        TestMath<T, D>(HWY_STR(NAME), F32x1, F32xN, d, F32_MIN, F32_MAX,  \
                       F32_ERROR);                                        \
      } else {                                                            \
        TestMath<T, D>(HWY_STR(NAME), F64x1, F64xN, d,                    \
                       static_cast<T>(F64_MIN), static_cast<T>(F64_MAX),  \
                       F64_ERROR);                                        \
      }                                                                   \
    }                                                                     \
  };                                                                      \
  DEFINE_MATH_TEST_FUNC(NAME)

// ULP distance between two float16 values, in the float16 bit domain.
// hwy::detail::ComputeUlpDelta cannot be used here because std::isnan does
// not support float16_t.
HWY_INLINE uint64_t F16UlpDelta(float16_t actual, float16_t expected) {
  const uint16_t a_bits = BitCastScalar<uint16_t>(actual);
  const uint16_t e_bits = BitCastScalar<uint16_t>(expected);
  if (a_bits == e_bits) return 0;
  const bool a_nan = ScalarIsNaN(F32FromF16(actual));
  const bool e_nan = ScalarIsNaN(F32FromF16(expected));
  if (a_nan && e_nan) return 0;
  if (a_nan != e_nan) return ~uint64_t{0};
  // Infinities must match exactly (equal bits, handled above), so that e.g. a
  // saturating demote cannot pass off 65504 as within 1 ULP of +inf.
  if ((a_bits & 0x7FFF) == 0x7C00 || (e_bits & 0x7FFF) == 0x7C00) {
    return ~uint64_t{0};
  }
  // Map sign-magnitude to a monotonically increasing integer, so that
  // adjacent float16 values differ by 1 and -0 maps to the same value as +0.
  const int32_t a_ord = (a_bits & 0x8000)
                            ? (0x8000 - static_cast<int32_t>(a_bits & 0x7FFF))
                            : (0x8000 + static_cast<int32_t>(a_bits));
  const int32_t e_ord = (e_bits & 0x8000)
                            ? (0x8000 - static_cast<int32_t>(e_bits & 0x7FFF))
                            : (0x8000 + static_cast<int32_t>(e_bits));
  return static_cast<uint64_t>(a_ord >= e_ord ? a_ord - e_ord : e_ord - a_ord);
}

// Rounds the double-precision reference to float16 with a single rounding
// step, via a round-to-odd float intermediate; a plain double->float->f16
// conversion double-rounds, which is off by 1 float16 ULP when the double
// falls within half a float ULP of a float16 rounding boundary. Handling
// overflow up front also avoids the (formally undefined) conversion of
// out-of-float-range doubles such as exp(104) to float.
HWY_INLINE float16_t F16FromF64(double value) {
  if (ScalarIsNaN(value)) return F16FromF32(static_cast<float>(value));
  // Values of at least 65520 (the float16 overflow threshold) round to inf.
  if (value >= 65520.0) return BitCastScalar<float16_t>(uint16_t{0x7C00});
  if (value <= -65520.0) return BitCastScalar<float16_t>(uint16_t{0xFC00});
  const float f = static_cast<float>(value);
  if (static_cast<double>(f) == value) return F16FromF32(f);
  uint32_t bits = BitCastScalar<uint32_t>(f);
  // Round to odd: magnitude-truncate if rounding to float moved away from
  // zero (also correct across exponent boundaries), then set the LSB.
  if (ScalarAbs(static_cast<double>(f)) > ScalarAbs(value)) bits -= 1;
  return F16FromF32(BitCastScalar<float>(bits | 1));
}

// Exhaustive ULP test for float16 math functions: for every float16 bit
// pattern whose value is within [min, max] (given as float), compares the
// vector function `fxN` against the double-precision scalar reference `fx1`
// rounded to float16. All ops used here support float16_t lanes even when
// HWY_HAVE_FLOAT16 is 0, so this runs on every target.
template <class D>
HWY_NOINLINE void TestF16Math(const char* name, double (*fx1)(double),
                              Vec<D> (*fxN)(D, VecArg<Vec<D>>), D d, float min,
                              float max, uint64_t max_error_ulp) {
  if (HWY_MATH_TEST_EXCESS_PRECISION) {
    static bool once = true;
    if (once) {
      once = false;
      HWY_WARN("Skipping math_test due to GCC issue with excess precision.\n");
    }
    return;
  }

  using T = float16_t;
  const size_t N = Lanes(d);
  auto in_lanes = AllocateAligned<T>(N);
  auto actual_lanes = AllocateAligned<T>(N);
  HWY_ASSERT(in_lanes && actual_lanes);
  uint64_t max_ulp = 0;
  // float16 has few enough bit patterns to test them all, in batches of N
  // consecutive patterns so that lanes hold distinct values; this also
  // verifies lane placement, which a splatted input could not. N divides
  // 65536, so base + i never exceeds 0xFFFF. Lanes whose value is NaN or
  // outside [min, max] still go through fxN but are skipped when verifying.
  for (uint32_t base = 0; base <= 0xFFFF; base += static_cast<uint32_t>(N)) {
    for (size_t i = 0; i < N; ++i) {
      const uint32_t bits = base + static_cast<uint32_t>(i);
      in_lanes[i] = BitCastScalar<T>(static_cast<uint16_t>(bits));
    }
    Store(fxN(d, Load(d, in_lanes.get())), d, actual_lanes.get());
    for (size_t i = 0; i < N; ++i) {
      const float value_f32 = F32FromF16(in_lanes[i]);
      if (ScalarIsNaN(value_f32) || value_f32 < min || value_f32 > max) {
        continue;
      }
      const T expected = F16FromF64(fx1(static_cast<double>(value_f32)));
      const uint64_t ulp = F16UlpDelta(actual_lanes[i], expected);
      if (ulp > max_error_ulp) {
        fprintf(stderr, "%s: %s(%f) expected %E actual %E ulp %g max ulp %u\n",
                hwy::TypeName(T(), Lanes(d)).c_str(), name,
                static_cast<double>(value_f32),
                static_cast<double>(F32FromF16(expected)),
                static_cast<double>(F32FromF16(actual_lanes[i])),
                static_cast<double>(ulp), static_cast<uint32_t>(max_error_ulp));
      }
      max_ulp = HWY_MAX(max_ulp, ulp);
    }
  }
  fprintf(stderr, "%s: %s max_ulp %g\n", hwy::TypeName(T(), Lanes(d)).c_str(),
          name, static_cast<double>(max_ulp));
  HWY_ASSERT(max_ulp <= max_error_ulp);
}

// Unlike DEFINE_MATH_TEST, registration does not go through ForFloat16Types,
// which is empty when HWY_HAVE_FLOAT16 is 0: the promote/demote-based f16
// math functions work on all targets.
#undef DEFINE_F16_MATH_TEST
#define DEFINE_F16_MATH_TEST(NAME, Fx1, FxN, F16_MIN, F16_MAX, F16_ERROR)     \
  struct TestF16##NAME {                                                      \
    template <class T, class D>                                               \
    HWY_NOINLINE void operator()(T, D d) {                                    \
      TestF16Math(HWY_STR(NAME), Fx1, FxN, d, F16_MIN, F16_MAX, F16_ERROR);   \
    }                                                                         \
  };                                                                          \
  HWY_NOINLINE void TestAllF16##NAME() {                                      \
    ForPartialVectors<TestF16##NAME>()(float16_t());                          \
  }

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_MATH_MATH_TEST_TOGGLE
