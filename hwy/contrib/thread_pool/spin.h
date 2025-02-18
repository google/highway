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

#ifndef HIGHWAY_HWY_CONTRIB_THREAD_POOL_SPIN_H_
#define HIGHWAY_HWY_CONTRIB_THREAD_POOL_SPIN_H_

// Relatively power-efficient spin lock for low-latency synchronization.

#include <stdint.h>

#include <atomic>

#include "hwy/base.h"
#include "hwy/cache_control.h"  // Pause

#ifndef HWY_ENABLE_MONITORX  // allow override
#if HWY_ARCH_X86 && ((HWY_COMPILER_CLANG >= 309) || \
                     (HWY_COMPILER_GCC_ACTUAL >= 502) || defined(__MWAITX__))
#define HWY_ENABLE_MONITORX 1
#else
#define HWY_ENABLE_MONITORX 0
#endif
#endif  // HWY_ENABLE_MONITORX

#ifndef HWY_ENABLE_UMONITOR  // allow override
#if HWY_ARCH_X86 && ((HWY_COMPILER_CLANG >= 700) || \
                     (HWY_COMPILER_GCC_ACTUAL >= 901) || defined(__WAITPKG__))
#define HWY_ENABLE_UMONITOR 1
#else
#define HWY_ENABLE_UMONITOR 0
#endif
#endif  // HWY_ENABLE_UMONITOR

#if HWY_ARCH_X86 && (HWY_ENABLE_MONITORX || HWY_ENABLE_UMONITOR)
#include <x86intrin.h>
#endif
#if HWY_HAS_ATTRIBUTE(target)
#define HWY_ATTR_MONITORX __attribute__((target("mwaitx")))
#define HWY_ATTR_UMONITOR __attribute__((target("waitpkg")))
#else
#define HWY_ATTR_MONITORX
#define HWY_ATTR_UMONITOR
#endif

namespace hwy {

// User-space monitor/wait are supported on Zen2+ AMD and SPR+ Intel. Spin waits
// are rarely called from SIMD code, hence we do not integrate this into the
// HWY_TARGET mechanism. Returned by `DetectSpinMode`.
enum class SpinMode {
#if HWY_ENABLE_MONITORX
  kMonitorX,  // AMD
#endif
#if HWY_ENABLE_UMONITOR
  kUMonitor,  // Intel
#endif
  kPause
};

// For printing which is supported/enabled.
static inline const char* ToString(SpinMode mode) {
  switch (mode) {
#if HWY_ENABLE_MONITORX
    case SpinMode::kMonitorX:
      return "MonitorX";
#endif
#if HWY_ENABLE_UMONITOR
    case SpinMode::kUMonitor:
      return "UMonitor";
#endif
    case SpinMode::kPause:
      return "Pause";
    default:
      return nullptr;
  }
}

HWY_CONTRIB_DLLEXPORT SpinMode DetectSpinMode();

// HWY_NOINLINE to avoid compiler errors about inlining into a function without
// these attributes.
#if HWY_ENABLE_MONITORX || HWY_IDE
HWY_ATTR_MONITORX static HWY_NOINLINE uint32_t Spin_MonitorX(
    const uint32_t prev, std::atomic<uint32_t>& current, size_t& reps) {
  for (reps = 0;; ++reps) {
    uint32_t cmd = current.load(std::memory_order_acquire);
    if (cmd != prev) return cmd;
    // No extensions/hints currently defined.
    _mm_monitorx(&current, 0, 0);
    // Double-checked 'lock':
    cmd = current.load(std::memory_order_acquire);
    if (cmd != prev) return cmd;
    const unsigned hints = 0xF;  // shallowest C0 for fast wakeup
    // Don't want extra timeout because its wake latency is ~1000 cycles
    // [https://www.usenix.org/system/files/usenixsecurity23-zhang-ruiyi.pdf]
    const unsigned extensions = 0;
    _mm_mwaitx(extensions, hints, /*cycles=*/0);
  }
}
#endif  // HWY_ENABLE_MONITORX

#if HWY_ENABLE_UMONITOR || HWY_IDE
HWY_ATTR_UMONITOR static HWY_NOINLINE uint32_t Spin_UMonitor(
    const uint32_t prev, std::atomic<uint32_t>& current, size_t& reps) {
  for (reps = 0;; ++reps) {
    uint32_t cmd = current.load(std::memory_order_acquire);
    if (cmd != prev) return cmd;
    _umonitor(&current);
    // Double-checked 'lock':
    cmd = current.load(std::memory_order_acquire);
    if (cmd != prev) return cmd;
    const unsigned control = 1;              // C0.1 for faster wakeup
    const uint64_t deadline = ~uint64_t{0};  // no timeout
    _umwait(control, deadline);
  }
}
#endif  // HWY_ENABLE_UMONITOR

static HWY_INLINE uint32_t Spin_Pause(const uint32_t prev,
                                      std::atomic<uint32_t>& current,
                                      size_t& reps) {
  for (reps = 0;; ++reps) {
    // Unfortunately, Pause duration varies across CPUs: 5 to 140 cycles.
    hwy::Pause();
    const uint32_t cmd = current.load(std::memory_order_acquire);
    if (cmd != prev) return cmd;
  }
}

// Like futex.h BlockUntilDifferent, but with spinning. `reps` is set to the
// number of loop iterations, useful for ensuring that the monitor/wait is not
// just returning immediately.
static HWY_INLINE uint32_t SpinUntilDifferent(SpinMode mode,
                                              const uint32_t prev,
                                              std::atomic<uint32_t>& current,
                                              size_t& reps) {
  switch (mode) {
#if HWY_ENABLE_MONITORX
    case SpinMode::kMonitorX:
      return Spin_MonitorX(prev, current, reps);
#endif
#if HWY_ENABLE_UMONITOR
    case SpinMode::kUMonitor:
      return Spin_UMonitor(prev, current, reps);
#endif
    case SpinMode::kPause:
      return Spin_Pause(prev, current, reps);
  }
}

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_THREAD_POOL_SPIN_H_
