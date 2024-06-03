// This test checks that Highway's selected targets are autodetected correctly
// for google3's major production targets.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "third_party/highway/google_internal/config_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace HWY_NAMESPACE {
HWY_ATTR static void Nop() {}
}  // namespace HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

#if GOOGLE3_SHOULD_HAVE_X86_64
#if !GOOGLE3_SHOULD_HAVE_HASWELL
static_assert(&N_SSE4::Nop != nullptr, "SSE4 must be available");
#endif  // !GOOGLE3_SHOULD_HAVE_HASWELL
static_assert(&N_AVX2::Nop != nullptr, "AVX2 must be available");
static_assert(&N_AVX3::Nop != nullptr, "AVX3 must be available");
static_assert(&N_AVX3_ZEN4::Nop != nullptr, "AVX3_ZEN4 must be available");
static_assert(&N_AVX3_SPR::Nop != nullptr, "AVX3_SPR must be available");
#endif  // GOOGLE3_SHOULD_HAVE_X86_64

#if GOOGLE3_SHOULD_HAVE_ARM
static_assert(&N_NEON::Nop != nullptr, "NEON must be available");
static_assert(&N_SVE::Nop != nullptr, "SVE must be available");
static_assert(&N_SVE2::Nop != nullptr, "SVE2 must be available");
static_assert(&N_SVE_256::Nop != nullptr, "SVE_256 must be available");
static_assert(&N_SVE2_128::Nop != nullptr, "SVE2_128 must be available");
#endif  // GOOGLE3_SHOULD_HAVE_ARM

int main(int, char**) { return 0; }

#endif  // HWY_ONCE
