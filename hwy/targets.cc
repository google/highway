// Copyright 2019 Google LLC
// Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "hwy/detect_targets_impl.h"
#include "hwy/highway.h"

namespace hwy {

#if HWY_ARCH_X86 && HWY_HAVE_RUNTIME_DISPATCH
namespace x86 {
#if HWY_OS_APPLE
class Platform {
 public:
  int Uname(struct utsname* name) const { return uname(name); }
  int SysctlByName(const char* name, void* oldp, size_t* oldlenp, void* newp,
                   size_t newlen) const {
    return sysctlbyname(name, oldp, oldlenp, newp, newlen);
  }
};
#else
class Platform {};
#endif
static int64_t DetectTargets() {
  Platform platform{};
  return DetectTargetsImpl(platform);
}
}  // namespace x86
#endif

// Returns targets supported by the CPU, independently of DisableTargets.
// Factored out of SupportedTargets to make its structure more obvious. Note
// that x86 CPUID may take several hundred cycles.
static int64_t DetectTargets() {
  // Apps will use only one of these (the default is EMU128), but compile flags
  // for this TU may differ from that of the app, so allow both.
  int64_t bits = HWY_SCALAR | HWY_EMU128;

#if HWY_ARCH_X86 && HWY_HAVE_RUNTIME_DISPATCH
  bits |= x86::DetectTargets();
#elif HWY_ARCH_ARM && HWY_HAVE_RUNTIME_DISPATCH
  bits |= arm::DetectTargets();
#elif HWY_ARCH_PPC && HWY_HAVE_RUNTIME_DISPATCH
  bits |= ppc::DetectTargets();
#elif HWY_ARCH_S390X && HWY_HAVE_RUNTIME_DISPATCH
  bits |= s390x::DetectTargets();
#elif HWY_ARCH_RISCV && HWY_HAVE_RUNTIME_DISPATCH
  bits |= rvv::DetectTargets();
#elif HWY_ARCH_LOONGARCH && HWY_HAVE_RUNTIME_DISPATCH
  bits |= loongarch::DetectTargets();

#else
  // TODO(janwas): detect support for WASM.
  // This file is typically compiled without HWY_IS_TEST, but targets_test has
  // it set, and will expect all of its HWY_TARGETS (= all attainable) to be
  // supported.
  bits |= HWY_ENABLED_BASELINE;
#endif  // HWY_ARCH_*

  if ((bits & HWY_ENABLED_BASELINE) != HWY_ENABLED_BASELINE) {
    const uint64_t bits_u = static_cast<uint64_t>(bits);
    const uint64_t enabled = static_cast<uint64_t>(HWY_ENABLED_BASELINE);
    HWY_WARN("CPU supports 0x%08x%08x, software requires 0x%08x%08x\n",
             static_cast<uint32_t>(bits_u >> 32),
             static_cast<uint32_t>(bits_u & 0xFFFFFFFF),
             static_cast<uint32_t>(enabled >> 32),
             static_cast<uint32_t>(enabled & 0xFFFFFFFF));
  }

  return bits;
}

// When running tests, this value can be set to the mocked supported targets
// mask. Only written to from a single thread before the test starts.
static int64_t supported_targets_for_test_ = 0;

// Mask of targets disabled at runtime with DisableTargets.
static int64_t supported_mask_ = LimitsMax<int64_t>();

HWY_DLLEXPORT void DisableTargets(int64_t disabled_targets) {
  supported_mask_ = static_cast<int64_t>(~disabled_targets);
  // This will take effect on the next call to SupportedTargets, which is
  // called right before GetChosenTarget::Update. However, calling Update here
  // would make it appear that HWY_DYNAMIC_DISPATCH was called, which we want
  // to check in tests. We instead de-initialize such that the next
  // HWY_DYNAMIC_DISPATCH calls GetChosenTarget::Update via FunctionCache.
  GetChosenTarget().DeInit();
}

HWY_DLLEXPORT void SetSupportedTargetsForTest(int64_t targets) {
  supported_targets_for_test_ = targets;
  GetChosenTarget().DeInit();  // see comment above
}

HWY_DLLEXPORT int64_t SupportedTargets() {
  int64_t targets = supported_targets_for_test_;
  if (HWY_LIKELY(targets == 0)) {
    // Mock not active. Re-detect instead of caching just in case we're on a
    // heterogeneous ISA (also requires some app support to pin threads). This
    // is only reached on the first HWY_DYNAMIC_DISPATCH or after each call to
    // DisableTargets or SetSupportedTargetsForTest.
    targets = DetectTargets();

    // VectorBytes invokes HWY_DYNAMIC_DISPATCH. To prevent infinite recursion,
    // first set up ChosenTarget. No need to Update() again afterwards with the
    // final targets - that will be done by a caller of this function.
    GetChosenTarget().Update(targets);
  }

  targets &= supported_mask_;
  return targets == 0 ? HWY_STATIC_TARGET : targets;
}

HWY_DLLEXPORT ChosenTarget& GetChosenTarget() {
  static ChosenTarget chosen_target;
  return chosen_target;
}

#if HWY_ARCH_X86_64 && HWY_HAVE_RUNTIME_DISPATCH
// Returns whether the CPU supports all `required_flags` (AMX feature bits) and
// the OS has enabled AMX tile state.
static bool HaveAmx(uint64_t required_flags) {
  const uint64_t flags = x86::FlagsFromCPUID();
  if ((flags & required_flags) != required_flags) {
    return false;
  }

  uint32_t abcd[4];
  x86::Cpuid(1, 0, abcd);
  const bool has_xsave = x86::IsBitSet(abcd[2], 26);
  const bool has_osxsave = x86::IsBitSet(abcd[2], 27);
  if (!has_xsave || !has_osxsave) {
    return false;
  }

#if HWY_OS_LINUX
  // On Linux, request OS permission for dynamic XSAVE tile state
  // (XFEATURE_XTILEDATA) first, before checking XCR0.
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
  const int64_t status =
      syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (status != 0) {
    return false;
  }
#endif  // HWY_OS_LINUX

  const uint32_t xcr0 = x86::ReadXCR0();
  if (!x86::HasYMM(xcr0) || !x86::HasZMM(xcr0) || !x86::HasAMX(xcr0)) {
    return false;
  }

  return true;
}
#endif  // HWY_ARCH_X86_64 && HWY_HAVE_RUNTIME_DISPATCH

HWY_DLLEXPORT bool HaveTile64BMatMulBF16() {
#if HWY_ARCH_X86_64 && HWY_HAVE_RUNTIME_DISPATCH
  // AMX-INT8 is also required because `__tile_loadd` requires it.
  static const bool has_amx_bf16 =
      HaveAmx(x86::Bit(x86::FeatureIndex::kAMX_TILE) |
              x86::Bit(x86::FeatureIndex::kAMX_BF16) |
              x86::Bit(x86::FeatureIndex::kAMX_INT8));
  return has_amx_bf16;
#else
  return false;
#endif
}

HWY_DLLEXPORT bool HaveTile64BMatMulI8() {
#if HWY_ARCH_X86_64 && HWY_HAVE_RUNTIME_DISPATCH
  static const bool has_amx_int8 =
      HaveAmx(x86::Bit(x86::FeatureIndex::kAMX_TILE) |
              x86::Bit(x86::FeatureIndex::kAMX_INT8));
  return has_amx_int8;
#else
  return false;
#endif
}

}  // namespace hwy
