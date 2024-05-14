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

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <map>
#include <thread>  // NOLINT
#include <vector>

#include "hwy/detect_compiler_arch.h"  // HWY_OS_WIN

#if HWY_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif  // HWY_OS_WIN

#if HWY_OS_LINUX || HWY_OS_FREEBSD
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
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

#if HWY_OS_LINUX
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#endif  // HWY_OS_LINUX

#include "hwy/base.h"

namespace hwy {

HWY_DLLEXPORT bool HaveThreadingSupport() {
#if HWY_ARCH_WASM
  return emscripten_has_threading_support() != 0;
#else
  return true;
#endif
}

HWY_DLLEXPORT size_t TotalLogicalProcessors() {
  size_t lp = 0;
#if HWY_ARCH_WASM
  const int num_cores = emscripten_num_logical_cores();
  if (num_cores > 0) lp = static_cast<size_t>(num_cores);
#else
  const unsigned concurrency = std::thread::hardware_concurrency();
  if (concurrency != 0) lp = static_cast<size_t>(concurrency);
#endif

  // WASM or C++ stdlib failed to detect #CPUs.
  if (lp == 0) {
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

  // Warn that we are clamping.
  if (lp > kMaxLogicalProcessors) {
    if (HWY_IS_DEBUG_BUILD) {
      fprintf(stderr, "OS reports %zu processors but clamping to %zu\n", lp,
              kMaxLogicalProcessors);
    }
    lp = kMaxLogicalProcessors;
  }

  return lp;
}

#ifdef __ANDROID__
#include <sys/syscall.h>
#endif

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
#ifdef __ANDROID__
  const int err = syscall(__NR_sched_getaffinity, pid, sizeof(cpu_set_t), &set);
#else
  const int err = sched_getaffinity(pid, sizeof(cpu_set_t), &set);
#endif  // __ANDROID__
  if (err != 0) return false;
  for (size_t lp = 0; lp < kMaxLogicalProcessors; ++lp) {
#if HWY_COMPILER_GCC_ACTUAL
    // Workaround for GCC compiler warning with CPU_ISSET macro
    HWY_DIAGNOSTICS(push)
    HWY_DIAGNOSTICS_OFF(disable : 4305 4309, ignored "-Wsign-conversion")
#endif
    if (CPU_ISSET(static_cast<int>(lp), &set)) {
      lps.Set(lp);
    }
#if HWY_COMPILER_GCC_ACTUAL
    HWY_DIAGNOSTICS(pop)
#endif
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
#if HWY_COMPILER_GCC_ACTUAL
    // Workaround for GCC compiler warning with CPU_ISSET macro
    HWY_DIAGNOSTICS(push)
    HWY_DIAGNOSTICS_OFF(disable : 4305 4309, ignored "-Wsign-conversion")
#endif
    if (CPU_ISSET(static_cast<int>(lp), &set)) {
      lps.Set(lp);
    }
#if HWY_COMPILER_GCC_ACTUAL
    HWY_DIAGNOSTICS(pop)
#endif
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
#if HWY_COMPILER_GCC_ACTUAL
  // Workaround for GCC compiler warning with CPU_SET macro
  HWY_DIAGNOSTICS(push)
  HWY_DIAGNOSTICS_OFF(disable : 4305 4309, ignored "-Wsign-conversion")
#endif
  lps.Foreach([&set](size_t lp) { CPU_SET(static_cast<int>(lp), &set); });
#if HWY_COMPILER_GCC_ACTUAL
  HWY_DIAGNOSTICS(pop)
#endif
  const pid_t pid = 0;  // current thread
#ifdef __ANDROID__
  const int err = syscall(__NR_sched_setaffinity, pid, sizeof(cpu_set_t), &set);
#else
  const int err = sched_setaffinity(pid, sizeof(cpu_set_t), &set);
#endif  // __ANDROID__
  if (err != 0) return false;
  return true;
#elif HWY_OS_FREEBSD
  cpuset_t set;
  CPU_ZERO(&set);
#if HWY_COMPILER_GCC_ACTUAL
  // Workaround for GCC compiler warning with CPU_SET macro
  HWY_DIAGNOSTICS(push)
  HWY_DIAGNOSTICS_OFF(disable : 4305 4309, ignored "-Wsign-conversion")
#endif
  lps.Foreach([&set](size_t lp) { CPU_SET(static_cast<int>(lp), &set); });
#if HWY_COMPILER_GCC_ACTUAL
  HWY_DIAGNOSTICS(pop)
#endif
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

#if HWY_OS_LINUX
namespace {

class File {
 public:
  explicit File(const char* path) {
    for (;;) {
      fd_ = open(path, O_RDONLY);
      if (fd_ > 0) return;           // success
      if (errno == EINTR) continue;  // signal: retry
      if (errno == ENOENT) return;   // not found, give up
      if (HWY_IS_DEBUG_BUILD) {
        fprintf(stderr, "Unexpected error opening %s: %d\n", path, errno);
      }
      return;  // unknown error, give up
    }
  }

  ~File() {
    if (fd_ > 0) {
      for (;;) {
        const int ret = close(fd_);
        if (ret == 0) break;           // success
        if (errno == EINTR) continue;  // signal: retry
        if (HWY_IS_DEBUG_BUILD) {
          fprintf(stderr, "Unexpected error closing file: %d\n", errno);
        }
        return;  // unknown error, ignore
      }
    }
  }

  // Returns number of bytes read or 0 on failure.
  size_t Read(char* buf200) const {
    if (fd_ < 0) return 0;
    size_t pos = 0;
    for (;;) {
      // read instead of pread, which might not work for sysfs.
      const auto bytes_read = read(fd_, buf200 + pos, 200 - pos);
      if (bytes_read == 0) {  // EOF: done
        buf200[pos++] = '\0';
        return pos;
      }
      if (bytes_read == -1) {
        if (errno == EINTR) continue;  // signal: retry
        if (HWY_IS_DEBUG_BUILD) {
          fprintf(stderr, "Unexpected error reading file: %d\n", errno);
        }
        return 0;
      }
      pos += static_cast<size_t>(bytes_read);
      HWY_ASSERT(pos <= 200);
    }
  }

 private:
  int fd_;
};

// Interprets as base-10 ASCII, handling an K or M suffix if present.
bool ParseSysfs(const char* str, size_t len, size_t* out) {
  size_t value = 0;
  // 9 digits cannot overflow even 32-bit size_t.
  size_t pos = 0;
  for (; pos < HWY_MIN(len, 9); ++pos) {
    const int c = str[pos];
    if (c < '0' || c > '9') break;
    value *= 10;
    value += static_cast<size_t>(c - '0');
  }
  if (pos == 0) {  // No digits found
    *out = 0;
    return false;
  }
  if (str[pos] == 'K') value <<= 10;
  if (str[pos] == 'M') value <<= 20;
  *out = value;
  return true;
}

bool ReadSysfs(const char* format, size_t lp, size_t* out) {
  char path[200];
  const int bytes_written = snprintf(path, sizeof(path), format, lp);
  HWY_ASSERT(0 < bytes_written &&
             bytes_written < static_cast<int>(sizeof(path) - 1));

  const File file(path);
  char buf200[200];
  const size_t pos = file.Read(buf200);
  if (pos == 0) return false;

  return ParseSysfs(buf200, pos, out);
}

const char* kPackage =
    "/sys/devices/system/cpu/cpu%zu/topology/physical_package_id";
const char* kCluster = "/sys/devices/system/cpu/cpu%zu/cache/index3/id";
const char* kCore = "/sys/devices/system/cpu/cpu%zu/topology/core_id";
const char* kL2Size = "/sys/devices/system/cpu/cpu%zu/cache/index2/size";
const char* kL3Size = "/sys/devices/system/cpu/cpu%zu/cache/index3/size";

// sysfs values can be arbitrarily large, so store in a map and replace with
// indices in order of appearance.
class Remapper {
 public:
  // Returns false on error, or sets `out_index` to the index of the sysfs
  // value selected by `format` and `lp`.
  template <typename T>
  bool operator()(const char* format, size_t lp, T* HWY_RESTRICT out_index) {
    size_t opaque;
    if (!ReadSysfs(format, lp, &opaque)) return false;

    const auto ib = indices_.insert({opaque, num_});
    num_ += ib.second;                      // increment if inserted
    const size_t index = ib.first->second;  // new or existing
    HWY_ASSERT(index < num_);
    HWY_ASSERT(index < hwy::LimitsMax<T>());
    *out_index = static_cast<T>(index);
    return true;
  }

  size_t Num() const { return num_; }

 private:
  std::map<size_t, size_t> indices_;
  size_t num_ = 0;
};

// Stores the global cluster/core values separately for each package so we can
// return per-package arrays.
struct PerPackage {
  Remapper clusters;
  Remapper cores;
  uint8_t smt_per_core[kMaxLogicalProcessors] = {0};
};

// Initializes `lps` and returns a PerPackage vector (empty on failure).
std::vector<PerPackage> DetectPackages(std::vector<Topology::LP>& lps) {
  std::vector<PerPackage> empty;

  Remapper packages;
  for (size_t lp = 0; lp < lps.size(); ++lp) {
    if (!packages(kPackage, lp, &lps[lp].package)) return empty;
  }
  std::vector<PerPackage> per_package(packages.Num());

  for (size_t lp = 0; lp < lps.size(); ++lp) {
    PerPackage& pp = per_package[lps[lp].package];
    if (!pp.clusters(kCluster, lp, &lps[lp].cluster)) return empty;
    if (!pp.cores(kCore, lp, &lps[lp].core)) return empty;

    // SMT ID is how many LP we have already seen assigned to the same core.
    HWY_ASSERT(lps[lp].core < kMaxLogicalProcessors);
    lps[lp].smt = pp.smt_per_core[lps[lp].core]++;
    HWY_ASSERT(lps[lp].smt < 16);
  }

  return per_package;
}

}  // namespace
#endif  // HWY_OS_LINUX

HWY_DLLEXPORT Topology::Topology() {
#if HWY_OS_LINUX
  lps.resize(TotalLogicalProcessors());
  const std::vector<PerPackage>& per_package = DetectPackages(lps);
  if (per_package.empty()) return;

  // Allocate per-package/cluster/core vectors. This indicates to callers that
  // detection succeeded.
  packages.resize(per_package.size());
  for (size_t p = 0; p < packages.size(); ++p) {
    packages[p].clusters.resize(per_package[p].clusters.Num());
    packages[p].cores.resize(per_package[p].cores.Num());
  }

  // Populate the per-cluster/core sets of LP.
  for (size_t lp = 0; lp < lps.size(); ++lp) {
    Package& p = packages[lps[lp].package];
    p.clusters[lps[lp].cluster].lps.Set(lp);
    p.cores[lps[lp].core].lps.Set(lp);
  }

  // Detect cache sizes (only once per cluster)
  for (size_t ip = 0; ip < packages.size(); ++ip) {
    Package& p = packages[ip];
    for (size_t ic = 0; ic < p.clusters.size(); ++ic) {
      Cluster& c = p.clusters[ic];
      const size_t lp = c.lps.First();
      size_t bytes;
      if (ReadSysfs(kL2Size, lp, &bytes)) {
        c.private_kib = bytes >> 10;
      }
      if (ReadSysfs(kL3Size, lp, &bytes)) {
        c.shared_kib = bytes >> 10;
      }
    }
  }
#endif
}

}  // namespace hwy
