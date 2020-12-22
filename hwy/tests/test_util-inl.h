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

// Normal include guard for non-SIMD portion of this header.
#ifndef HWY_TESTS_TEST_UTIL_H_
#define HWY_TESTS_TEST_UTIL_H_

// Helper functions for use by *_test.cc.

#include <stdio.h>
#include <string.h>

#include <random>
#include <string>
#include <utility>  // std::forward

#include "gtest/gtest.h"
#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"

namespace hwy {

// The maximum vector size used in tests when defining test data. This is at
// least the kMaxVectorSize but it can be bigger. If you increased
// kMaxVectorSize, you also need to increase this constant and update all the
// tests that use it to define bigger arrays of test data.
constexpr size_t kTestMaxVectorSize = 64;
static_assert(kTestMaxVectorSize >= kMaxVectorSize,
              "All kTestMaxVectorSize test arrays need to be updated");

// googletest before 1.10 didn't define INSTANTIATE_TEST_SUITE_P() but instead
// used INSTANTIATE_TEST_CASE_P which is now deprecated.
#ifdef INSTANTIATE_TEST_SUITE_P
#define HWY_GTEST_INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_SUITE_P
#else
#define HWY_GTEST_INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif

// Helper class to run parametric tests using the hwy target as parameter. To
// use this define the following in your test:
//   class MyTestSuite : public TestWithParamTarget {
//    ...
//   };
//   HWY_TARGET_INSTANTIATE_TEST_SUITE_P(MyTestSuite);
//   TEST_P(MyTestSuite, MyTest) { ... }
class TestWithParamTarget : public testing::TestWithParam<uint32_t> {
 protected:
  void SetUp() override { SetSupportedTargetsForTest(GetParam()); }

  void TearDown() override {
    // Check that the parametric test calls SupportedTargets() when the source
    // was compiled with more than one target. In the single-target case only
    // static dispatch will be used anyway.
#if (HWY_TARGETS & (HWY_TARGETS - 1)) != 0
    EXPECT_TRUE(SupportedTargetsCalledForTest())
        << "This hwy target parametric test doesn't use dynamic-dispatch and "
           "doesn't need to be parametric.";
#endif
    SetSupportedTargetsForTest(0);
  }
};

// Function to convert the test parameter of a TestWithParamTarget for
// displaying it in the gtest test name.
std::string TestParamTargetName(const testing::TestParamInfo<uint32_t>& info) {
  return TargetName(info.param);
}

#define HWY_TARGET_INSTANTIATE_TEST_SUITE_P(suite)              \
  HWY_GTEST_INSTANTIATE_TEST_SUITE_P(                           \
      suite##Group, suite,                                      \
      testing::ValuesIn(::hwy::SupportedAndGeneratedTargets()), \
      ::hwy::TestParamTargetName)

// Helper class similar to TestWithParamTarget to run parametric tests that
// depend on the target and another parametric test. If you need to use multiple
// extra parameters use a std::tuple<> of them and ::testing::Generate(...) as
// the generator. To use this class define the following in your test:
//   class MyTestSuite : public TestWithParamTargetT<int> {
//    ...
//   };
//   HWY_TARGET_INSTANTIATE_TEST_SUITE_P_T(MyTestSuite, ::testing::Range(0, 9));
//   TEST_P(MyTestSuite, MyTest) { ... GetParam() .... }
template <typename T>
class TestWithParamTargetAndT
    : public ::testing::TestWithParam<std::tuple<uint32_t, T>> {
 public:
  // Expose the parametric type here so it can be used by the
  // HWY_TARGET_INSTANTIATE_TEST_SUITE_P_T macro.
  using HwyParamType = T;

 protected:
  void SetUp() override {
    SetSupportedTargetsForTest(std::get<0>(
        ::testing::TestWithParam<std::tuple<uint32_t, T>>::GetParam()));
  }

  void TearDown() override {
    // Check that the parametric test calls SupportedTargets() when the source
    // was compiled with more than one target. In the single-target case only
    // static dispatch will be used anyway.
#if (HWY_TARGETS & (HWY_TARGETS - 1)) != 0
    EXPECT_TRUE(SupportedTargetsCalledForTest())
        << "This hwy target parametric test doesn't use dynamic-dispatch and "
           "doesn't need to be parametric.";
#endif
    SetSupportedTargetsForTest(0);
  }

  T GetParam() {
    return std::get<1>(
        ::testing::TestWithParam<std::tuple<uint32_t, T>>::GetParam());
  }
};

template <typename T>
std::string TestParamTargetNameAndT(
    const testing::TestParamInfo<std::tuple<uint32_t, T>>& info) {
  return std::string(TargetName(std::get<0>(info.param))) + "_" +
         ::testing::PrintToString(std::get<1>(info.param));
}

#define HWY_TARGET_INSTANTIATE_TEST_SUITE_P_T(suite, generator)     \
  HWY_GTEST_INSTANTIATE_TEST_SUITE_P(                               \
      suite##Group, suite,                                          \
      ::testing::Combine(                                           \
          testing::ValuesIn(::hwy::SupportedAndGeneratedTargets()), \
          generator),                                               \
      ::hwy::TestParamTargetNameAndT<suite::HwyParamType>)

// Helper macro to export a function and define a test that tests it. This is
// equivalent to do a HWY_EXPORT of a void(void) function and run it in a test:
//   class MyTestSuite : public TestWithParamTarget {
//    ...
//   };
//   HWY_TARGET_INSTANTIATE_TEST_SUITE_P(MyTestSuite);
//   HWY_EXPORT_AND_TEST_P(MyTestSuite, MyTest);
#define HWY_EXPORT_AND_TEST_P(suite, func_name)                   \
  HWY_EXPORT(func_name);                                          \
  TEST_P(suite, func_name) { HWY_DYNAMIC_DISPATCH(func_name)(); } \
  static_assert(true, "For requiring trailing semicolon")

#define HWY_EXPORT_AND_TEST_P_T(suite, func_name)                           \
  HWY_EXPORT(func_name);                                                    \
  TEST_P(suite, func_name) { HWY_DYNAMIC_DISPATCH(func_name)(GetParam()); } \
  static_assert(true, "For requiring trailing semicolon")

// Calls test for each enabled and available target.
template <class Func, typename... Args>
void RunTest(const Func& func, Args&&... args) {
  SetSupportedTargetsForTest(0);
  auto targets = SupportedAndGeneratedTargets();

  for (uint32_t target : targets) {
    SetSupportedTargetsForTest(target);
    fprintf(stderr, "Testing for target %s.\n",
            TargetName(static_cast<int>(target)));
    func(std::forward<Args>(args)...);
  }
  // Disable the mask after the test.
  SetSupportedTargetsForTest(0);
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
template <typename T>
static inline std::string TypeName(T /*unused*/, size_t N) {
  std::string prefix(IsFloat<T>() ? "f" : (IsSigned<T>() ? "i" : "u"));
  prefix += std::to_string(sizeof(T) * 8);

  // Scalars: omit the xN suffix.
  if (N == 1) return prefix;

  return prefix + 'x' + std::to_string(N);
}

// Value to string

// We specialize for float/double below.
template <typename T>
inline std::string ToString(T value) {
  return std::to_string(value);
}

template <>
inline std::string ToString<float>(const float value) {
  // Ensure -0 and 0 are equivalent (required by some tests).
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7FFFFFFF) == 0) return "0";

  // to_string doesn't return enough digits and sstream is a
  // fairly large dependency (4KLOC).
  char buf[100];
  sprintf(buf, "%.8f", value);
  return buf;
}

template <>
inline std::string ToString<double>(const double value) {
  // Ensure -0 and 0 are equivalent (required by some tests).
  uint64_t bits;
  memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7FFFFFFFFFFFFFFFull) == 0) return "0";

  // to_string doesn't return enough digits and sstream is a
  // fairly large dependency (4KLOC).
  char buf[100];
  sprintf(buf, "%.16f", value);
  return buf;
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

}  // namespace hwy

#endif  // HWY_TESTS_TEST_UTIL_H_

// Per-target include guard
#if defined(HIGHWAY_HWY_TESTS_TEST_UTIL_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_TESTS_TEST_UTIL_INL_H_
#undef HIGHWAY_HWY_TESTS_TEST_UTIL_INL_H_
#else
#define HIGHWAY_HWY_TESTS_TEST_UTIL_INL_H_
#endif

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

HWY_NORETURN void NotifyFailure(const char* filename, const int line,
                                const char* type_name, const size_t lane,
                                const char* expected, const char* actual) {
  hwy::Abort(filename, line,
             "%s, %s lane %zu mismatch: expected '%s', got '%s'.\n",
             hwy::TargetName(HWY_TARGET), type_name, lane, expected, actual);
}

// Compare non-vector, non-string T.
template <typename T>
void AssertEqual(const T expected, const T actual, const std::string& name,
                 const char* filename = "", const int line = -1,
                 const size_t lane = 0) {
  // Rely on string comparison to ensure similar floats are "equal".
  const std::string expected_str = ToString(expected);
  const std::string actual_str = ToString(actual);
  if (expected_str != actual_str) {
    NotifyFailure(filename, line, name.c_str(), lane, expected_str.c_str(),
                  actual_str.c_str());
  }
}

void AssertStringEqual(const char* expected, const char* actual,
                       const char* filename = "", const int line = -1,
                       const size_t lane = 0) {
  if (!hwy::StringsEqual(expected, actual)) {
    NotifyFailure(filename, line, "string", lane, expected, actual);
  }
}

// Compare expected vector to vector.
template <class D, class V>
void AssertVecEqual(D d, const V expected, const V actual, const char* filename,
                    const int line) {
  using T = typename D::T;
  const size_t N = Lanes(d);
  auto expected_lanes = AllocateAligned<T>(N);
  auto actual_lanes = AllocateAligned<T>(N);
  Store(expected, d, expected_lanes.get());
  Store(actual, d, actual_lanes.get());
  for (size_t i = 0; i < N; ++i) {
    AssertEqual(expected_lanes[i], actual_lanes[i],
                hwy::TypeName(expected_lanes[i], Lanes(d)), filename, line, i);
  }
}

// Compare expected lanes to vector.
template <class D, class V>
void AssertVecEqual(D d, const typename D::T* expected, V actual,
                    const char* filename, int line) {
  AssertVecEqual(d, LoadU(d, expected), actual, filename, line);
}

#ifndef HWY_ASSERT_EQ

#define HWY_ASSERT_EQ(expected, actual) \
  AssertEqual(expected, actual, hwy::TypeName(expected, 1), __FILE__, __LINE__)

#define HWY_ASSERT_STRING_EQ(expected, actual) \
  AssertStringEqual(expected, actual, __FILE__, __LINE__)

#define HWY_ASSERT_VEC_EQ(d, expected, actual) \
  AssertVecEqual(d, expected, actual, __FILE__, __LINE__)

#endif  // HWY_ASSERT_EQ

// Helpers for instantiating tests with combinations of lane types / counts.

// For all powers of two in [kMinLanes, N * kMinLanes] (so that recursion stops
// at N == 0)
template <typename T, size_t N, size_t kMinLanes, class Test>
struct ForeachSizeR {
  static void Do() {
    static_assert(N != 0, "End of recursion");
    Test()(T(), Simd<T, N * kMinLanes>());
    ForeachSizeR<T, N / 2, kMinLanes, Test>::Do();
  }
};

// Base case to stop the recursion.
template <typename T, size_t kMinLanes, class Test>
struct ForeachSizeR<T, 0, kMinLanes, Test> {
  static void Do() {}
};

// These adapters may be called directly, or via For*Types:

// Calls Test for all powers of two in [kMinLanes, kMaxLanes / kDivLanes].
// Use a large default for kMaxLanes because we don't have access to T in the
// template argument list.
template <class Test, size_t kDivLanes = 1, size_t kMinLanes = 1,
          size_t kMaxLanes = 1ul << 30>
struct ForPartialVectors {
  template <typename T>
  void operator()(T /*unused*/) const {
    ForeachSizeR<T, HWY_MIN(kMaxLanes, HWY_LANES(T)) / kDivLanes / kMinLanes,
                 kMinLanes, Test>::Do();
  }
};

// Calls Test for all powers of two in [128 bits, max bits].
template <class Test>
struct ForGE128Vectors {
  template <typename T>
  void operator()(T /*unused*/) const {
    ForeachSizeR<T, HWY_LANES(T) / (16 / sizeof(T)), (16 / sizeof(T)),
                 Test>::Do();
  }
};

// Calls Test for all powers of two in [128 bits, max bits/2].
template <class Test>
struct ForExtendableVectors {
  template <typename T>
  void operator()(T /*unused*/) const {
    ForeachSizeR<T, HWY_LANES(T) / 2 / (16 / sizeof(T)), (16 / sizeof(T)),
                 Test>::Do();
  }
};

// Calls Test for full vectors only.
template <class Test>
struct ForFullVectors {
  template <typename T>
  void operator()(T t) const {
    Test()(t, HWY_FULL(T)());
  }
};

// Type lists to shorten call sites:

template <class Func>
void ForSignedTypes(const Func& func) {
  func(int8_t());
  func(int16_t());
  func(int32_t());
#if HWY_CAP_INTEGER64
  func(int64_t());
#endif
}

template <class Func>
void ForUnsignedTypes(const Func& func) {
  func(uint8_t());
  func(uint16_t());
  func(uint32_t());
#if HWY_CAP_INTEGER64
  func(uint64_t());
#endif
}

template <class Func>
void ForIntegerTypes(const Func& func) {
  ForSignedTypes(func);
  ForUnsignedTypes(func);
}

template <class Func>
void ForFloatTypes(const Func& func) {
  func(float());
#if HWY_CAP_FLOAT64
  func(double());
#endif
}

template <class Func>
void ForAllTypes(const Func& func) {
  ForIntegerTypes(func);
  ForFloatTypes(func);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // per-target include guard
