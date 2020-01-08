// Copyright 2019 Google LLC
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

#ifndef HWY_TESTS_TEST_UTIL_H_
#define HWY_TESTS_TEST_UTIL_H_

// SIMD-independent helper functions for use by *_test.cc.

#include <stdio.h>
#include <string.h>

#include <random>

#include "hwy/arch.h"  // kMaxVectorSize
#include "hwy/runtime_dispatch.h"
#include "hwy/type_traits.h"

// Caller wants to use gtest.
#ifdef HWY_USE_GTEST
#include "gtest/gtest.h"
#else
// The tests are gtest-compatible and we only need to define this. Note that
// cmake scans for a TEST() marker in the source code. This macro is followed by
// the body of the test, and we rely on C99/C++ guarantees that reaching the end
// of main is equivalent to return 0. Note that this means we can only have one
// test case per file, which also minimizes runtime-dispatch boilerplate.
#define TEST(test_case, test_name) int main()
#endif  // HWY_USE_GTEST

namespace hwy {
// NOLINTNEXTLINE(google-build-namespaces)
namespace {

// The maximum vector size used in tests when defining test data. This is at
// least the kMaxVectorSize but it can be bigger. If you increased
// kMaxVectorSize, you also need to increase this constant and update all the
// tests that use it to define bigger arrays of test data.
constexpr size_t kTestMaxVectorSize = 64;
static_assert(kTestMaxVectorSize >= kMaxVectorSize,
              "All kTestMaxVectorSize test arrays need to be updated");

// Calls Test()() for each instruction set via TargetBitfield.
template <class Test>
int RunTests() {
  setvbuf(stdin, nullptr, _IONBF, 0);

  const int targets = TargetBitfield().Foreach(Test());
  printf("Successfully tested instruction sets: 0x%x.\n", targets);
  return 0;
}

[[noreturn]] void NotifyFailure(const char* filename, const int line,
                                const size_t bits, const char* type_name,
                                const size_t lane, const char* expected,
                                const char* actual) {
  fprintf(stderr,
          "%s:%d: bits %zu, %s lane %zu mismatch: expected '%s', got '%s'.\n",
          filename, line, bits, type_name, lane, expected, actual);
  HWY_TRAP();
}

// Random numbers
typedef std::mt19937 RandomState;
HWY_INLINE uint32_t Random32(RandomState* rng) {
  return static_cast<uint32_t>((*rng)());
}

// Prevents the compiler from eliding the computations that led to "output".
// Works by indicating to the compiler that "output" is being read and modified.
// The +r constraint avoids unnecessary writes to memory, but only works for
// built-in types.
template <class T>
inline void PreventElision(T&& output) {
#ifndef _MSC_VER
  asm volatile("" : "+r"(output) : : "memory");
#endif
}

// Returns a name for the vector/part/scalar. The type prefix is u/i/f for
// unsigned/signed/floating point, followed by the number of bits per lane;
// then 'x' followed by the number of lanes. Example: u8x16. This is useful for
// understanding which instantiation of a generic test failed.
template <typename T, size_t N>
const char* TypeName() {
  constexpr char prefix = IsFloat<T>() ? 'f' : (IsSigned<T>() ? 'i' : 'u');

  constexpr size_t bits = sizeof(T) * 8;
  constexpr char bits10 = '0' + (bits / 10);
  constexpr char bits1 = '0' + (bits % 10);

  // Scalars: omit the xN suffix.
  if (N == 1) {
    static constexpr char name1[8] = {prefix, bits1};
    static constexpr char name2[8] = {prefix, bits10, bits1};
    return sizeof(T) == 1 ? name1 : name2;
  }

  constexpr char N1 = (N < 10) ? '\0' : '0' + (N % 10);
  constexpr char N10 = (N < 10) ? '0' + (N % 10) : '0' + (N / 10);

  static constexpr char name1[8] = {prefix, bits1, 'x', N10, N1};
  static constexpr char name2[8] = {prefix, bits10, bits1, 'x', N10, N1};
  return sizeof(T) == 1 ? name1 : name2;
}

// Value to string

// Returns end of string (position of '\0').
template <typename T>
inline char* ToString(T value, char* to) {
  char reversed[100];
  char* pos = reversed;
  int64_t before;
  do {
    before = value;
    value /= 10;
    const int64_t mod = before - value * 10;
    *pos++ = "9876543210123456789"[9 + mod];
  } while (value != 0);
  if (before < 0) *pos++ = '-';

  // Reverse the string
  const int num_chars = pos - reversed;
  for (int i = 0; i < num_chars; ++i) {
    to[i] = pos[-1 - i];
  }
  to[num_chars] = '\0';
  return to + num_chars;
}

template <>
inline char* ToString<float>(const float value, char* to) {
  if (!std::isfinite(value)) {
    strncpy(to, "<not-finite>", 13);
    return to + strlen(to);
  }
  const int64_t truncated = static_cast<int64_t>(value);
  char* end = ToString(truncated, to);
  *end++ = '.';
  int64_t frac = static_cast<int64_t>((value - truncated) * 1E8);
  if (frac < 0) frac = -frac;
  return ToString(frac, end);
}

template <>
inline char* ToString<double>(const double value, char* to) {
  if (!std::isfinite(value)) {
    strncpy(to, "<not-finite>", 13);
    return to + strlen(to);
  }
  const int64_t truncated = static_cast<int64_t>(value);
  char* end = ToString(truncated, to);
  *end++ = '.';
  int64_t frac = static_cast<int64_t>((value - truncated) * 1E16);
  if (frac < 0) frac = -frac;
  return ToString(frac, end);
}

// String comparison

template <typename T1, typename T2>
inline bool BytesEqual(const T1* p1, const T2* p2, const size_t size) {
  const uint8_t* bytes1 = reinterpret_cast<const uint8_t*>(p1);
  const uint8_t* bytes2 = reinterpret_cast<const uint8_t*>(p2);
  for (size_t i = 0; i < size; ++i) {
    if (bytes1[i] != bytes2[i]) return false;
  }
  return true;
}

inline bool StringsEqual(const char* s1, const char* s2) {
  while (*s1 == *s2++) {
    if (*s1++ == '\0') return true;
  }
  return false;
}

}  // namespace
}  // namespace hwy

#endif  // HWY_TESTS_TEST_UTIL_H_
