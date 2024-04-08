// Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: BSD-3-Clause

#include "hwy/abort.h"

#include "hwy/tests/hwy_gtest.h"
#include "hwy/base.h"

#include "hwy/tests/test_util-inl.h"  // HWY_ASSERT_EQ

namespace hwy {

#ifdef GTEST_HAS_DEATH_TEST
namespace {
std::string GetBaseName(std::string const& file_name) {
  auto last_slash = file_name.find_last_of("/\\");
  return file_name.substr(last_slash + 1);
}
}  // namespace

TEST(AbortDeathTest, AbortDefault) {
  std::string expected = std::string("Abort at ") + GetBaseName(__FILE__) +
                         ":" + std::to_string(__LINE__ + 1) + ": Test Abort";
  ASSERT_DEATH(HWY_ABORT("Test %s", "Abort"), expected);
}

TEST(AbortDeathTest, AbortOverride) {
  std::string expected =
      std::string("Test Abort from [0-9]+ of ") + GetBaseName(__FILE__);

  ASSERT_DEATH(
      {
        AbortFunc CustomAbortHandler = [](const char* file, int line,
                                          const char* formatted_err) -> void {
          fprintf(stderr, "%s from %d of %s", formatted_err, line,
                  GetBaseName(file).data());
        };

        SetAbortFunc(CustomAbortHandler);
        HWY_ABORT("Test %s", "Abort");
      },
      expected);
}
#endif

TEST(AbortTest, AbortOverrideChain) {
  AbortFunc FirstHandler = [](const char* file, int line,
                              const char* formatted_err) -> void {
    fprintf(stderr, "%s from %d of %s", formatted_err, line, file);
  };
  AbortFunc SecondHandler = [](const char* file, int line,
                               const char* formatted_err) -> void {
    fprintf(stderr, "%s from %d of %s", formatted_err, line, file);
  };

  HWY_ASSERT(SetAbortFunc(FirstHandler) == nullptr);
  HWY_ASSERT(GetAbortFunc() == FirstHandler);
  HWY_ASSERT(SetAbortFunc(SecondHandler) == FirstHandler);
  HWY_ASSERT(GetAbortFunc() == SecondHandler);
  HWY_ASSERT(SetAbortFunc(nullptr) == SecondHandler);
  HWY_ASSERT(GetAbortFunc() == nullptr);
}

}  // namespace hwy

HWY_TEST_MAIN();
