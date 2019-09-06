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

// No include guard - included multiple times by foreach_target.

#ifndef SIMD_TARGET_INCLUDE
#include "third_party/highway/highway/in_target.h"
#include "third_party/highway/highway/test_util.h"
#endif

#ifndef SIMD_NAMESPACE
#error "in_target.h should have set SIMD_NAMESPACE."
#endif

namespace jxl {
namespace SIMD_NAMESPACE {
namespace {

// Prevents the compiler from eliding the computations that led to "output".
// Works by indicating to the compiler that "output" is being read and modified.
// The +r constraint avoids unnecessary writes to memory, but only works for
// built-in types.
template <class T>
inline SIMD_ATTR void PreventElision(T&& output) {
#ifndef _MSC_VER
  asm volatile("" : "+r"(output) : : "memory");
#endif
}

// Compare non-vector, non-string T.
template <typename T>
SIMD_ATTR void AssertEqual(const T expected, const T actual,
                           const char* filename = "", const int line = -1,
                           const size_t lane = 0, const char* name = nullptr) {
  if (name == nullptr) name = type_name<T, 1>();
  char expected_buf[100];
  char actual_buf[100];
  ToString(expected, expected_buf);
  ToString(actual, actual_buf);
  // Rely on string comparison to ensure similar floats are "equal".
  if (!StringsEqual(expected_buf, actual_buf)) {
    NotifyFailure(filename, line, SIMD_BITS, name, lane, expected_buf,
                  actual_buf);
  }
}

#define SIMD_ASSERT_EQ(expected, actual) \
  AssertEqual(expected, actual, __FILE__, __LINE__)

SIMD_ATTR void AssertStringEqual(const char* expected, const char* actual,
                                 const char* filename = "", const int line = -1,
                                 const int lane = 0) {
  if (!StringsEqual(expected, actual)) {
    NotifyFailure(filename, line, SIMD_BITS, "string", lane, expected, actual);
  }
}

#define SIMD_ASSERT_STRING_EQ(expected, actual) \
  AssertStringEqual(expected, actual, __FILE__, __LINE__)

// Compare expected vector to vector.
template <class D, class V>
SIMD_ATTR void AssertVecEqual(D d, const V expected, const V actual,
                              const char* filename, const int line) {
  SIMD_ALIGN typename D::T expected_lanes[d.N];
  SIMD_ALIGN typename D::T actual_lanes[d.N];
  store(expected, d, expected_lanes);
  store(actual, d, actual_lanes);
  for (size_t i = 0; i < d.N; ++i) {
    AssertEqual(expected_lanes[i], actual_lanes[i], filename, line, i,
                type_name<typename D::T, D::N>());
  }
}

// Compare expected lanes to vector.
template <class D, class V>
SIMD_ATTR void AssertVecEqual(D d, const typename D::T (&expected)[D::N],
                              const V actual, const char* filename,
                              const int line) {
  AssertVecEqual(d, load_u(d, expected), actual, filename, line);
}

#define SIMD_ASSERT_VEC_EQ(d, expected, actual) \
  AssertVecEqual(d, expected, actual, __FILE__, __LINE__)

// Type lists

template <class Test, typename T>
SIMD_ATTR void Call() {
  Test().template operator()(T(), SIMD_FULL(T)());
}

// Calls Test::operator()(T, D) for each lane type.
template <class Test>
SIMD_ATTR void ForeachUnsignedLaneType() {
  Call<Test, uint8_t>();
  Call<Test, uint16_t>();
  Call<Test, uint32_t>();
  Call<Test, uint64_t>();
}

template <class Test>
SIMD_ATTR void ForeachSignedLaneType() {
  Call<Test, int8_t>();
  Call<Test, int16_t>();
  Call<Test, int32_t>();
  Call<Test, int64_t>();
}

template <class Test>
SIMD_ATTR void ForeachFloatLaneType() {
  Call<Test, float>();
  Call<Test, double>();
}

template <class Test>
SIMD_ATTR void ForeachLaneType() {
  ForeachUnsignedLaneType<Test>();
  ForeachSignedLaneType<Test>();
  ForeachFloatLaneType<Test>();
}

}  // namespace
}  // namespace SIMD_NAMESPACE  // NOLINT(google-readability-namespace-comments)
}  // namespace jxl
