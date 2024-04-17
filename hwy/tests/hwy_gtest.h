// Copyright 2021 Google LLC
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

#ifndef HWY_TESTS_HWY_GTEST_H_
#define HWY_TESTS_HWY_GTEST_H_

// Adapter/replacement for GUnit to run tests for all targets.

#include "hwy/base.h"

// Allow opting out of GUnit.
#ifndef HWY_TEST_STANDALONE
// GUnit and its dependencies no longer support MSVC.
#if HWY_COMPILER_MSVC
#define HWY_TEST_STANDALONE 1
#else
#define HWY_TEST_STANDALONE 0
#endif  // HWY_COMPILER_MSVC
#endif  // HWY_TEST_STANDALONE

#include <stdint.h>

#include <string>
#include <tuple>

#if !HWY_TEST_STANDALONE
#include "gtest/gtest.h"  // IWYU pragma: export
#endif
#include "hwy/detect_targets.h"
#include "hwy/targets.h"

namespace hwy {

#if !HWY_TEST_STANDALONE

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
class TestWithParamTarget : public testing::TestWithParam<int64_t> {
 protected:
  void SetUp() override { SetSupportedTargetsForTest(GetParam()); }

  void TearDown() override {
    // Check that the parametric test calls SupportedTargets() when the source
    // was compiled with more than one target. In the single-target case only
    // static dispatch will be used anyway.
#if (HWY_TARGETS & (HWY_TARGETS - 1)) != 0
    EXPECT_TRUE(GetChosenTarget().IsInitialized())
        << "This hwy target parametric test doesn't use dynamic-dispatch and "
           "doesn't need to be parametric.";
#endif
    SetSupportedTargetsForTest(0);
  }
};

// Function to convert the test parameter of a TestWithParamTarget for
// displaying it in the gtest test name.
static inline std::string TestParamTargetName(
    const testing::TestParamInfo<int64_t>& info) {
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
    : public ::testing::TestWithParam<std::tuple<int64_t, T>> {
 public:
  // Expose the parametric type here so it can be used by the
  // HWY_TARGET_INSTANTIATE_TEST_SUITE_P_T macro.
  using HwyParamType = T;

 protected:
  void SetUp() override {
    SetSupportedTargetsForTest(std::get<0>(
        ::testing::TestWithParam<std::tuple<int64_t, T>>::GetParam()));
  }

  void TearDown() override {
    // Check that the parametric test calls SupportedTargets() when the source
    // was compiled with more than one target. In the single-target case only
    // static dispatch will be used anyway.
#if (HWY_TARGETS & (HWY_TARGETS - 1)) != 0
    EXPECT_TRUE(GetChosenTarget().IsInitialized())
        << "This hwy target parametric test doesn't use dynamic-dispatch and "
           "doesn't need to be parametric.";
#endif
    SetSupportedTargetsForTest(0);
  }

  T GetParam() {
    return std::get<1>(
        ::testing::TestWithParam<std::tuple<int64_t, T>>::GetParam());
  }
};

template <typename T>
std::string TestParamTargetNameAndT(
    const testing::TestParamInfo<std::tuple<int64_t, T>>& info) {
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

#define HWY_BEFORE_TEST(suite)                      \
  class suite : public hwy::TestWithParamTarget {}; \
  HWY_TARGET_INSTANTIATE_TEST_SUITE_P(suite);       \
  static_assert(true, "For requiring trailing semicolon")

#define HWY_AFTER_TEST() static_assert(true, "For requiring trailing semicolon")

#define HWY_TEST_MAIN() static_assert(true, "For requiring trailing semicolon")

#else  // HWY_TEST_STANDALONE

// Cannot be a function, otherwise the HWY_EXPORT table defined here will not
// be visible to HWY_DYNAMIC_DISPATCH.
#define HWY_EXPORT_AND_TEST_P(suite, func_name)                               \
  HWY_EXPORT(func_name);                                                      \
  hwy::SetSupportedTargetsForTest(0);                                         \
  for (int64_t target : hwy::SupportedAndGeneratedTargets()) {                \
    hwy::SetSupportedTargetsForTest(target);                                  \
    fprintf(stderr, "=== %s for %s:\n", #func_name, hwy::TargetName(target)); \
    HWY_DYNAMIC_DISPATCH(func_name)();                                        \
  }                                                                           \
  /* Disable the mask after the test. */                                      \
  hwy::SetSupportedTargetsForTest(0);                                         \
  static_assert(true, "For requiring trailing semicolon")

// HWY_BEFORE_TEST may reside inside a namespace, but HWY_AFTER_TEST will define
// a main() at namespace scope that wants to call into that namespace, so stash
// the function address in a singleton defined in namespace hwy.
using VoidFunc = void (*)(void);

VoidFunc& GetRunAll() {
  static VoidFunc func;
  return func;
}

struct RegisterRunAll {
  RegisterRunAll(VoidFunc func) { hwy::GetRunAll() = func; }
};

#define HWY_BEFORE_TEST(suite)                                 \
  void RunAll();                                               \
  static hwy::RegisterRunAll HWY_CONCAT(reg_, suite)(&RunAll); \
  void RunAll() {                                              \
    static_assert(true, "For requiring trailing semicolon")

// Must be followed by semicolon, then a closing brace for ONE namespace.
#define HWY_AFTER_TEST()                    \
  } /* RunAll*/                             \
  } /* namespace */                         \
  int main(int /*argc*/, char** /*argv*/) { \
    hwy::GetRunAll()();                     \
    fprintf(stderr, "Success.\n");          \
    return 0

// -------------------- Non-SIMD test cases:

struct FuncAndName {
  VoidFunc func;
  const char* name;
};

// Singleton of registered tests to be run by HWY_TEST_MAIN
std::vector<FuncAndName>& GetFuncAndNames() {
  static std::vector<FuncAndName> vec;
  return vec;
}

// For use by TEST; adds to the list.
struct RegisterTest {
  RegisterTest(VoidFunc func, const char* name) {
    hwy::GetFuncAndNames().push_back({func, name});
  }
};

// Registers a function to be called by `HWY_TEST_MAIN`. `suite` is unused.
#define TEST(suite, func)                                          \
  void func();                                                     \
  static hwy::RegisterTest HWY_CONCAT(reg_, func)({&func, #func}); \
  void func()

// Expands to a main() that calls all TEST. Must reside at namespace scope.
#define HWY_TEST_MAIN()                                        \
  int main() {                                                 \
    for (const auto& func_and_name : hwy::GetFuncAndNames()) { \
      fprintf(stderr, "=== %s:\n", func_and_name.name);        \
      func_and_name.func();                                    \
    }                                                          \
    fprintf(stderr, "Success.\n");                             \
    return 0;                                                  \
  }                                                            \
  static_assert(true, "For requiring trailing semicolon")

#endif  // HWY_TEST_STANDALONE

}  // namespace hwy

#endif  // HWY_TESTS_HWY_GTEST_H_
