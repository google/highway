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

#include <atomic>

#include "hwy/base.h"

#ifndef HWY_ENABLE_MONITORX  // allow override
// Clang 3.9 suffices for mwaitx, but the target pragma requires 9.0.
#if HWY_ARCH_X86 && ((HWY_COMPILER_CLANG >= 900) || \
                     (HWY_COMPILER_GCC_ACTUAL >= 502) || defined(__MWAITX__))
#define HWY_ENABLE_MONITORX 1
#else
#define HWY_ENABLE_MONITORX 0
#endif
#endif  // HWY_ENABLE_MONITORX

#ifndef HWY_ENABLE_UMONITOR  // allow override
#if HWY_ARCH_X86 && ((HWY_COMPILER_CLANG >= 900) || \
                     (HWY_COMPILER_GCC_ACTUAL >= 901) || defined(__WAITPKG__))
#define HWY_ENABLE_UMONITOR 1
#else
#define HWY_ENABLE_UMONITOR 0
#endif
#endif  // HWY_ENABLE_UMONITOR

#if HWY_ARCH_X86 && (HWY_ENABLE_MONITORX || HWY_ENABLE_UMONITOR)
// Must be in source file, not header, due to conflict between clang-cl
// intrin.h and gtest, both of which are included in spin_test.cc.
#include <x86intrin.h>

#include "hwy/x86_cpuid.h"
#endif

#include "hwy/cache_control.h"  // Pause

namespace hwy {

#if HWY_ENABLE_MONITORX || HWY_IDE
HWY_PUSH_ATTRIBUTES("mwaitx")

// Implementation for AMD's user-mode monitor/wait (Zen2+).
class MonitorX : public ISpin {
 public:
  const char* String() const override { return "MonitorX_C1"; }
  SpinType Type() const override { return SpinType::kMonitorX; }

  SpinResult UntilDifferent(const uint32_t prev,
                            std::atomic<uint32_t>& watched) override {
    for (uint32_t reps = 0;; ++reps) {
      uint32_t current = watched.load(std::memory_order_acquire);
      if (current != prev) return SpinResult{current, reps};
      // No extensions/hints currently defined.
      _mm_monitorx(&watched, 0, 0);
      // Double-checked 'lock' to avoid missed events:
      current = watched.load(std::memory_order_acquire);
      if (current != prev) return SpinResult{current, reps};
      _mm_mwaitx(kExtensions, kHints, /*cycles=*/0);
    }
  }

  size_t UntilEqual(const uint32_t expected,
                    std::atomic<uint32_t>& watched) override {
    for (size_t reps = 0;; ++reps) {
      uint32_t current = watched.load(std::memory_order_acquire);
      if (current == expected) return reps;
      // No extensions/hints currently defined.
      _mm_monitorx(&watched, 0, 0);
      // Double-checked 'lock' to avoid missed events:
      current = watched.load(std::memory_order_acquire);
      if (current == expected) return reps;
      _mm_mwaitx(kExtensions, kHints, /*cycles=*/0);
    }
  }

 private:
  // 0xF would be C0. Its wakeup latency is less than 0.1 us shorter, and
  // package power is sometimes actually higher than with Pause. The
  // difference in spurious wakeups is minor.
  static constexpr unsigned kHints = 0x0;  // C1: a bit deeper than C0
  // No timeout required, we assume the mwaitx does not miss stores, see
  // https://www.usenix.org/system/files/usenixsecurity23-zhang-ruiyi.pdf.]
  static constexpr unsigned kExtensions = 0;
};

HWY_POP_ATTRIBUTES
#endif  // HWY_ENABLE_MONITORX

#if HWY_ENABLE_UMONITOR || HWY_IDE
HWY_PUSH_ATTRIBUTES("waitpkg")

// Implementation for Intel's user-mode monitor/wait (SPR+).
class UMonitor : public ISpin {
 public:
  const char* String() const override { return "UMonitor_C0.2"; }
  SpinType Type() const override { return SpinType::kUMonitor; }

  SpinResult UntilDifferent(const uint32_t prev,
                            std::atomic<uint32_t>& watched) override {
    for (uint32_t reps = 0;; ++reps) {
      uint32_t current = watched.load(std::memory_order_acquire);
      if (current != prev) return SpinResult{current, reps};
      _umonitor(&watched);
      // Double-checked 'lock' to avoid missed events:
      current = watched.load(std::memory_order_acquire);
      if (current != prev) return SpinResult{current, reps};
      _umwait(kControl, kDeadline);
    }
  }

  size_t UntilEqual(const uint32_t expected,
                    std::atomic<uint32_t>& watched) override {
    for (size_t reps = 0;; ++reps) {
      uint32_t current = watched.load(std::memory_order_acquire);
      if (current == expected) return reps;
      _umonitor(&watched);
      // Double-checked 'lock' to avoid missed events:
      current = watched.load(std::memory_order_acquire);
      if (current == expected) return reps;
      _umwait(kControl, kDeadline);
    }
  }

 private:
  // 1 would be C0.1. C0.2 has 20x fewer spurious wakeups and additional 4%
  // package power savings vs Pause on SPR. It comes at the cost of
  // 0.4-0.6us higher wake latency, but the total is comparable to Zen4.
  static constexpr unsigned kControl = 0;              // C0.2 for deeper sleep
  static constexpr uint64_t kDeadline = ~uint64_t{0};  // no timeout, see above
};

HWY_POP_ATTRIBUTES
#endif  // HWY_ENABLE_UMONITOR

// TODO(janwas): add WFE on Arm. May wake at 10 kHz, but still worthwhile.

// Always supported, but unpredictable: Pause duration varies across CPUs
// (between 5 to 140 cycles), and may be a no-op on some platforms.
struct PauseLoop : public ISpin {
  const char* String() const override { return "Pause"; }
  SpinType Type() const override { return SpinType::kPause; }

  SpinResult UntilDifferent(const uint32_t prev,
                            std::atomic<uint32_t>& watched) override {
    for (uint32_t reps = 0;; ++reps) {
      const uint32_t current = watched.load(std::memory_order_acquire);
      if (current != prev) return SpinResult{current, reps};
      hwy::Pause();
    }
  }

  size_t UntilEqual(const uint32_t expected,
                    std::atomic<uint32_t>& watched) override {
    for (size_t reps = 0;; ++reps) {
      const uint32_t current = watched.load(std::memory_order_acquire);
      if (current == expected) return reps;
      hwy::Pause();
    }
  }
};

HWY_CONTRIB_DLLEXPORT ISpin* ChooseSpin(int disabled) {
  const auto HWY_MAYBE_UNUSED enabled = [disabled](SpinType type) {
    return (disabled & (1 << static_cast<int>(type))) == 0;
  };

#if HWY_ENABLE_MONITORX
  if (enabled(SpinType::kMonitorX) && x86::IsAMD()) {
    uint32_t abcd[4];
    x86::Cpuid(0x80000001U, 0, abcd);
    if (x86::IsBitSet(abcd[2], 29)) {
      static MonitorX monitorx;
      return &monitorx;
    }
  }
#endif  // HWY_ENABLE_MONITORX

#if HWY_ENABLE_UMONITOR
  if (enabled(SpinType::kUMonitor) && x86::MaxLevel() >= 7) {
    uint32_t abcd[4];
    x86::Cpuid(7, 0, abcd);
    if (x86::IsBitSet(abcd[2], 5)) {
      static UMonitor umonitor;
      return &umonitor;
    }
  }
#endif  // HWY_ENABLE_UMONITOR

  if (!enabled(SpinType::kPause)) {
    HWY_WARN("Ignoring attempt to disable Pause, it is the only option left.");
  }
  static PauseLoop pause;
  return &pause;
}

}  // namespace hwy
