
#define HWY_HEADER_ONLY

#include <string>

#include "hwy/base.h"
#include "hwy/status.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"  // HWY_ASSERT_EQ

namespace hwy {
namespace {

enum class HeaderOnlyIOCode : Status::Code {
  kCustomIO = Status::FirstError(Status::ModuleBegin(Status::Module::kIO))
};

const char* HeaderOnlyIOCodeName(Status::Code code) {
  return Status::MakeCode(Status::Severity::kError, code) ==
                 static_cast<Status::Code>(HeaderOnlyIOCode::kCustomIO)
             ? "kCustomIO"
             : nullptr;
}

TEST(StatusHeaderOnlyTest, ModuleCodeNameRegistration) {
  const Status::CodeNameFunc prev =
      Status::SetModuleCodeNameFunc(Status::Module::kIO, &HeaderOnlyIOCodeName);
  HWY_ASSERT_STRING_EQ("kCustomIO", Status(HeaderOnlyIOCode::kCustomIO).Name());
  (void)Status::SetModuleCodeNameFunc(Status::Module::kIO, prev);
}

#ifdef GTEST_HAS_DEATH_TEST

TEST(AbortDeathTest, AbortDefault) {
  std::string expected = std::string("Abort at ") + __FILE__ + ":" +
                         std::to_string(__LINE__ + 1) + ": Test Abort";
  ASSERT_DEATH(HWY_ABORT("Test %s", "Abort"), expected);
}
#endif  // GTEST_HAS_DEATH_TEST

}  // namespace
}  // namespace hwy
