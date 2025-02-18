// Copyright 2025 Google LLC
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

#include "hwy/contrib/thread_pool/spin.h"

#include "hwy/base.h"
// Must be in source file, not header, due to conflict between clang-cl
// intrin.h and gtest, both of which are included in spin_test.cc.
#include "hwy/x86_cpuid.h"

namespace hwy {

HWY_CONTRIB_DLLEXPORT SpinMode DetectSpinMode() {
#if HWY_ARCH_X86
  uint32_t abcd[4];
#if HWY_ENABLE_MONITORX
  if (x86::IsAMD()) {
    x86::Cpuid(0x80000001U, 0, abcd);
    if (x86::IsBitSet(abcd[2], 29)) return SpinMode::kMonitorX;
  }
#endif  // HWY_ENABLE_MONITORX

#if HWY_ENABLE_UMONITOR
  if (x86::MaxLevel() >= 7) {
    x86::Cpuid(7, 0, abcd);
    if (x86::IsBitSet(abcd[2], 5)) return SpinMode::kUMonitor;
  }
#endif  // HWY_ENABLE_UMONITOR

#endif  // HWY_ARCH_X86
  return SpinMode::kPause;
}

}  // namespace hwy
