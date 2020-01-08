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

// Empty include guard to avoid Copybara warning
#ifndef HWY_TESTS_TEST_TARGET_UTIL_H
#define HWY_TESTS_TEST_TARGET_UTIL_H
#endif

#include <stddef.h>

// Tests already include test_util (for TEST()). Including it "again" makes its
// definitions visible to the IDE.
#include "hwy/foreach_target.h"
#include "hwy/tests/test_util.h"

#ifndef HWY_NAMESPACE
#error "foreach_target.h should have set HWY_NAMESPACE."
#endif

namespace hwy {
namespace HWY_NAMESPACE {
// NOLINTNEXTLINE(google-build-namespaces)
namespace {

// Compare non-vector, non-string T.
template <typename T>
void AssertEqual(const T expected, const T actual, const char* filename = "",
                 const int line = -1, const size_t lane = 0,
                 const char* name = nullptr) {
  if (name == nullptr) name = TypeName<T, 1>();
  char expected_buf[100];
  char actual_buf[100];
  ToString(expected, expected_buf);
  ToString(actual, actual_buf);
  // Rely on string comparison to ensure similar floats are "equal".
  if (!StringsEqual(expected_buf, actual_buf)) {
    NotifyFailure(filename, line, HWY_BITS, name, lane, expected_buf,
                  actual_buf);
  }
}

HWY_ATTR void AssertStringEqual(const char* expected, const char* actual,
                                const char* filename = "", const int line = -1,
                                const size_t lane = 0) {
  if (!StringsEqual(expected, actual)) {
    NotifyFailure(filename, line, HWY_BITS, "string", lane, expected, actual);
  }
}

// Compare expected vector to vector.
template <class D, class V>
HWY_ATTR void AssertVecEqual(D d, const V expected, const V actual,
                             const char* filename, const int line) {
  HWY_ALIGN typename D::T expected_lanes[d.N];
  HWY_ALIGN typename D::T actual_lanes[d.N];
  Store(expected, d, expected_lanes);
  Store(actual, d, actual_lanes);
  for (size_t i = 0; i < d.N; ++i) {
    AssertEqual(expected_lanes[i], actual_lanes[i], filename, line, i,
                TypeName<typename D::T, D::N>());
  }
}

// Compare expected lanes to vector.
template <class D, class V>
HWY_ATTR void AssertVecEqual(D d, const typename D::T (&expected)[D::N],
                             V actual, const char* filename, int line) {
  AssertVecEqual(d, LoadU(d, expected), actual, filename, line);
}

// Only define macros once to avoid warnings/errors.
#ifndef HWY_ASSERT_EQ

#define HWY_ASSERT_EQ(expected, actual) \
  AssertEqual(expected, actual, __FILE__, __LINE__)

#define HWY_ASSERT_STRING_EQ(expected, actual) \
  AssertStringEqual(expected, actual, __FILE__, __LINE__)

#define HWY_ASSERT_VEC_EQ(d, expected, actual) \
  AssertVecEqual(d, expected, actual, __FILE__, __LINE__)

// Type lists: call func for Unsigned/Signed lane types.
#if HWY_HAS_INT64

#define HWY_FOREACH_U(func) \
  func(du8);                \
  func(du16);               \
  func(du32);               \
  func(du64);

#define HWY_FOREACH_I(func) \
  func(di8);                \
  func(di16);               \
  func(di32);               \
  func(di64);

#else

#define HWY_FOREACH_U(func) \
  func(du8);                \
  func(du16);               \
  func(du32);

#define HWY_FOREACH_I(func) \
  func(di8);                \
  func(di16);               \
  func(di32);

#endif

#define HWY_FOREACH_UI(func) \
  HWY_FOREACH_U(func);       \
  HWY_FOREACH_I(func);

#endif  // HWY_ASSERT_EQ

// Exception: these change according to HWY_HAS_DOUBLE, so always redefine.
#undef HWY_FOREACH_F
#undef HWY_FOREACH_UIF

#if HWY_HAS_DOUBLE
#define HWY_FOREACH_F(func) \
  func(df);                 \
  func(dd);
#else
#define HWY_FOREACH_F(func) func(df);
#endif

#define HWY_FOREACH_UIF(func) \
  HWY_FOREACH_UI(func);       \
  HWY_FOREACH_F(func);

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
