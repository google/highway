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

#include "hwy/targets.h"

#include <stdint.h>

#include "hwy/detect_targets.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

namespace fake {

#define DECLARE_FUNCTION(TGT)                                                \
  namespace N_##TGT {                                                        \
    /* Function argument is just to ensure/demonstrate they are possible. */ \
    int64_t FakeFunction(int) { return HWY_##TGT; }                          \
    template <typename T>                                                    \
    int64_t FakeFunctionT(T) {                                               \
      return HWY_##TGT;                                                      \
    }                                                                        \
  }

DECLARE_FUNCTION(AVX3_SPR)
DECLARE_FUNCTION(AVX3_ZEN4)
DECLARE_FUNCTION(AVX3_DL)
DECLARE_FUNCTION(AVX3)
DECLARE_FUNCTION(AVX2)
DECLARE_FUNCTION(SSE4)
DECLARE_FUNCTION(SSSE3)
DECLARE_FUNCTION(SSE2)

DECLARE_FUNCTION(SVE2_128)
DECLARE_FUNCTION(SVE_256)
DECLARE_FUNCTION(SVE2)
DECLARE_FUNCTION(SVE)
DECLARE_FUNCTION(NEON_BF16)
DECLARE_FUNCTION(NEON)
DECLARE_FUNCTION(NEON_WITHOUT_AES)

DECLARE_FUNCTION(PPC10)
DECLARE_FUNCTION(PPC9)
DECLARE_FUNCTION(PPC8)

DECLARE_FUNCTION(Z15)
DECLARE_FUNCTION(Z14)

DECLARE_FUNCTION(WASM)
DECLARE_FUNCTION(WASM_EMU256)

DECLARE_FUNCTION(RVV)

DECLARE_FUNCTION(SCALAR)
DECLARE_FUNCTION(EMU128)

HWY_EXPORT(FakeFunction);

void CallFunctionForTarget(int64_t target, int /*line*/) {
  if ((HWY_TARGETS & target) == 0) return;
  hwy::SetSupportedTargetsForTest(target);

  // Call Update() first to make &HWY_DYNAMIC_DISPATCH() return
  // the pointer to the already cached function.
  hwy::GetChosenTarget().Update(hwy::SupportedTargets());

  HWY_ASSERT_EQ(target, HWY_DYNAMIC_DISPATCH(FakeFunction)(42));

  // Test calling templated functions.
  HWY_EXPORT_T(FakeFunctionT, FakeFunctionT<bool>);
  HWY_ASSERT_EQ(target, HWY_DYNAMIC_DISPATCH_T(FakeFunctionT)(true));

  // Calling DeInit() will test that the initializer function
  // also calls the right function.
  hwy::GetChosenTarget().DeInit();

#if HWY_DISPATCH_WORKAROUND
  const int64_t expected = HWY_STATIC_TARGET;
#else
  const int64_t expected = target;
#endif
  HWY_ASSERT_EQ(expected, HWY_DYNAMIC_DISPATCH(FakeFunction)(42));
  HWY_ASSERT_EQ(expected, HWY_DYNAMIC_DISPATCH_T(FakeFunctionT)(true));

  // Second call uses the cached value from the previous call.
  HWY_ASSERT_EQ(target, HWY_DYNAMIC_DISPATCH(FakeFunction)(42));
  HWY_ASSERT_EQ(target, HWY_DYNAMIC_DISPATCH_T(FakeFunctionT)(true));
}

void CheckFakeFunction() {
  // When adding a target, also add to DECLARE_FUNCTION above.
  CallFunctionForTarget(HWY_AVX3_SPR, __LINE__);
  CallFunctionForTarget(HWY_AVX3_ZEN4, __LINE__);
  CallFunctionForTarget(HWY_AVX3_DL, __LINE__);
  CallFunctionForTarget(HWY_AVX3, __LINE__);
  CallFunctionForTarget(HWY_AVX2, __LINE__);
  CallFunctionForTarget(HWY_SSE4, __LINE__);
  CallFunctionForTarget(HWY_SSSE3, __LINE__);
  CallFunctionForTarget(HWY_SSE2, __LINE__);

  CallFunctionForTarget(HWY_SVE2_128, __LINE__);
  CallFunctionForTarget(HWY_SVE_256, __LINE__);
  CallFunctionForTarget(HWY_SVE2, __LINE__);
  CallFunctionForTarget(HWY_SVE, __LINE__);
  CallFunctionForTarget(HWY_NEON_BF16, __LINE__);
  CallFunctionForTarget(HWY_NEON, __LINE__);
  CallFunctionForTarget(HWY_NEON_WITHOUT_AES, __LINE__);

  CallFunctionForTarget(HWY_PPC10, __LINE__);
  CallFunctionForTarget(HWY_PPC9, __LINE__);
  CallFunctionForTarget(HWY_PPC8, __LINE__);

  CallFunctionForTarget(HWY_WASM, __LINE__);
  CallFunctionForTarget(HWY_WASM_EMU256, __LINE__);

  CallFunctionForTarget(HWY_RVV, __LINE__);
  // The tables only have space for either HWY_SCALAR or HWY_EMU128; the former
  // is opt-in only.
#if defined(HWY_COMPILE_ONLY_SCALAR) || HWY_BROKEN_EMU128
  CallFunctionForTarget(HWY_SCALAR, __LINE__);
#else
  CallFunctionForTarget(HWY_EMU128, __LINE__);
#endif
}

}  // namespace fake

namespace hwy {

#if !HWY_TEST_STANDALONE
class HwyTargetsTest : public testing::Test {};
#endif

// Test that the order in the HWY_EXPORT static array matches the expected
// value of the target bits. This is only checked for the targets that are
// enabled in the current compilation.
TEST(HwyTargetsTest, ChosenTargetOrderTest) { fake::CheckFakeFunction(); }

TEST(HwyTargetsTest, DisabledTargetsTest) {
  SetSupportedTargetsForTest(0);
  DisableTargets(~0LL);
  // Check that disabling everything at least leaves the static target.
  HWY_ASSERT(HWY_STATIC_TARGET == SupportedTargets());

  DisableTargets(0);  // Reset the mask.
  const int64_t current_targets = SupportedTargets();
  const int64_t enabled_baseline = static_cast<int64_t>(HWY_ENABLED_BASELINE);
  // Exclude these two because they are always returned by SupportedTargets.
  const int64_t fallback = HWY_SCALAR | HWY_EMU128;
  if ((current_targets & ~enabled_baseline & ~fallback) == 0) {
    // We can't test anything else if the only compiled target is the baseline.
    return;
  }

  // Get the lowest bit in the mask (the best target) and disable that one.
  const int64_t best_target = current_targets & (~current_targets + 1);
  DisableTargets(best_target);

  // Check that the other targets are still enabled.
  HWY_ASSERT((best_target ^ current_targets) == SupportedTargets());
  DisableTargets(0);  // Reset the mask.
}

}  // namespace hwy

HWY_TEST_MAIN();
