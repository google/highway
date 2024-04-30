// Copyright 2024 Google LLC
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

#include "hwy/contrib/thread_pool/topology.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "third_party/highway/hwy/contrib/thread_pool/topology.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/highway.h"
#include "hwy/timer-inl.h"

// Non-SIMD includes and types. Note that HWY_ONCE is only true on the last
// compile pass, whereas we want this defined in the first.
#ifndef TOPOLOGY_ONCE
#define TOPOLOGY_ONCE

#include <stddef.h>
#include <stdio.h>

#include <thread>  // NOLINT

#if HWY_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#include <windows.h>
#endif  // HWY_OS_WIN

#if HWY_OS_LINUX || HWY_OS_FREEBSD
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif  // _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <sys/types.h>
#include <unistd.h>
#endif  // HWY_OS_LINUX || HWY_OS_FREEBSD

#if HWY_OS_FREEBSD
// must come after sys/types.h.
#include <sys/cpuset.h>  // CPU_SET
#endif                   // HWY_OS_FREEBSD

#if HWY_ARCH_WASM
#include <emscripten/threading.h>
#endif

namespace hwy {

HWY_DLLEXPORT bool HaveThreadingSupport() {
#if HWY_ARCH_WASM
  return emscripten_has_threading_support() != 0;
#else
  return true;
#endif
}

HWY_DLLEXPORT size_t TotalLogicalProcessors() {
#if HWY_ARCH_WASM
  const int lp = emscripten_num_logical_cores();
  HWY_ASSERT(lp < static_cast<int>(kMaxLogicalProcessors));
  if (lp > 0) return static_cast<size_t>(lp);
#else
  const unsigned lp = std::thread::hardware_concurrency();
  HWY_ASSERT(lp < static_cast<unsigned>(kMaxLogicalProcessors));
  if (lp != 0) return static_cast<size_t>(lp);
#endif

  // WASM or C++ stdlib failed.
  if (HWY_IS_DEBUG_BUILD) {
    fprintf(
        stderr,
        "Unknown TotalLogicalProcessors. HWY_OS_: WIN=%d LINUX=%d APPLE=%d;\n"
        "HWY_ARCH_: WASM=%d X86=%d PPC=%d ARM=%d RISCV=%d S390X=%d\n",
        HWY_OS_WIN, HWY_OS_LINUX, HWY_OS_APPLE, HWY_ARCH_WASM, HWY_ARCH_X86,
        HWY_ARCH_PPC, HWY_ARCH_ARM, HWY_ARCH_RISCV, HWY_ARCH_S390X);
  }
  return 1;
}

HWY_DLLEXPORT bool GetThreadAffinity(LogicalProcessorSet& lps) {
#if HWY_OS_WIN
  // Only support the first 64 because WINE does not support processor groups.
  const HANDLE hThread = GetCurrentThread();
  const DWORD_PTR prev = SetThreadAffinityMask(hThread, ~DWORD_PTR(0));
  if (!prev) return false;
  (void)SetThreadAffinityMask(hThread, prev);
  lps = LogicalProcessorSet();  // clear all
  lps.SetNonzeroBitsFrom64(prev);
  return true;
#elif HWY_OS_LINUX
  cpu_set_t set;
  CPU_ZERO(&set);
  const pid_t pid = 0;  // current thread
  const int err = sched_getaffinity(pid, sizeof(cpu_set_t), &set);
  if (err != 0) return false;
  for (size_t lp = 0; lp < kMaxLogicalProcessors; ++lp) {
    if (CPU_ISSET(static_cast<int>(lp), &set)) {
      lps.Set(lp);
    }
  }
  return true;
#elif HWY_OS_FREEBSD
  cpuset_t set;
  CPU_ZERO(&set);
  const pid_t pid = getpid();  // current thread
  const int err = cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_PID, pid,
                                     sizeof(cpuset_t), &set);
  if (err != 0) return false;
  for (size_t lp = 0; lp < kMaxLogicalProcessors; ++lp) {
    if (CPU_ISSET(static_cast<int>(lp), &set)) {
      lps.Set(lp);
    }
  }
  return true;
#else
  // Do not even set lp=0 to force callers to handle this case.
  (void)lps;
  return false;
#endif
}

HWY_DLLEXPORT bool SetThreadAffinity(const LogicalProcessorSet& lps) {
#if HWY_OS_WIN
  const HANDLE hThread = GetCurrentThread();
  const DWORD_PTR prev = SetThreadAffinityMask(hThread, lps.Get64());
  return prev != 0;
#elif HWY_OS_LINUX
  cpu_set_t set;
  CPU_ZERO(&set);
  lps.Foreach([&set](size_t lp) { CPU_SET(static_cast<int>(lp), &set); });
  const int err = sched_setaffinity(0, sizeof(cpu_set_t), &set);
  if (err != 0) return false;
  return true;
#elif HWY_OS_FREEBSD
  cpuset_t set;
  CPU_ZERO(&set);
  lps.Foreach([&set](size_t lp) { CPU_SET(static_cast<int>(lp), &set); });
  const pid_t pid = getpid();  // current thread
  const int err = cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_PID, pid,
                                     sizeof(cpuset_t), &set);
  if (err != 0) return false;
  return true;
#else
  // Apple THREAD_AFFINITY_POLICY is only an (often ignored) hint.
  (void)lps;
  return false;
#endif
}

}  // namespace hwy
#endif  // TOPOLOGY_ONCE
