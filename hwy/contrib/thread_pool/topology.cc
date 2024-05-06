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

bool SimpleAtoi(const char* str, size_t len, size_t* out) {
  size_t value = 0;
  size_t digits_seen = 0;
  // 9 digits cannot overflow even 32-bit size_t.
  for (size_t i = 0; i < HWY_MIN(len, 9); ++i) {
    if (str[i] < '0' || str[i] > '9') break;
    value *= 10;
    value += static_cast<size_t>(str[i] - '0');
    ++digits_seen;
  }
  if (digits_seen == 0) {
    *out = 0;
    return false;
  }
  *out = value;
  return true;
}

bool ReadSysfs(const char* format, size_t lp, size_t max, size_t* out) {
  char path[200];
  const int bytes_written = snprintf(path, sizeof(path), format, lp);
  HWY_ASSERT(0 < bytes_written &&
             bytes_written < static_cast<int>(sizeof(path) - 1));

  const File file(path);
  char buf200[200];
  const size_t pos = file.Read(buf200);
  if (pos == 0) return false;

  if (!SimpleAtoi(buf200, pos, out)) return false;
  if (*out > max) {
    if (HWY_IS_DEBUG_BUILD) {
      fprintf(stderr, "Value for %s = %zu > %zu\n", path, *out, max);
    }
    return false;
  }
  return true;
}

// Given a set of opaque values, returns a map of each value to the number of
// values that precede it. Used to generate contiguous arrays in the same order.
std::map<size_t, size_t> RanksFromSet(const BitSet4096<>& set) {
  HWY_ASSERT(set.Count() != 0);
  std::map<size_t, size_t> ranks;
  size_t num = 0;
  set.Foreach([&ranks, &num](size_t opaque) {
    const bool inserted = ranks.insert({opaque, num++}).second;
    HWY_ASSERT(inserted);
  });
  HWY_ASSERT(num == set.Count());
  HWY_ASSERT(num == ranks.size());
  return ranks;
}

const char* kPackage =
    "/sys/devices/system/cpu/cpu%zu/topology/physical_package_id";
const char* kCluster = "/sys/devices/system/cpu/cpu%zu/cache/index3/id";
const char* kCore = "/sys/devices/system/cpu/cpu%zu/topology/core_id";

// Returns 0 on error, or number of packages and initializes LP::package.
size_t DetectPackages(std::vector<Topology::LP>& lps) {
  // Packages are typically an index, but could actually be an arbitrary value.
  // We assume they do not exceed kMaxLogicalProcessors and thus fit in a set.
  BitSet4096<> package_set;
  for (size_t lp = 0; lp < lps.size(); ++lp) {
    size_t package;
    if (!ReadSysfs(kPackage, lp, kMaxLogicalProcessors, &package)) return 0;
    package_set.Set(package);
    // Storing a set of lps belonging to each package requires more space/time
    // than storing the package per lp and remapping below.
    lps[lp].package = static_cast<uint16_t>(package);
  }
  HWY_ASSERT(package_set.Count() != 0);

  // Remap the per-lp package to their rank.
  std::map<size_t, size_t> ranks = RanksFromSet(package_set);
  for (size_t lp = 0; lp < lps.size(); ++lp) {
    lps[lp].package = static_cast<uint16_t>(ranks[lps[lp].package]);
  }
  return ranks.size();
}

// Stores the global cluster/core values separately for each package so we can
// return per-package arrays.
struct PerPackage {
  // Use maximum possible set size because Arm values can exceed 1k.
  BitSet4096<> cluster_set;
  BitSet4096<> core_set;

  std::map<size_t, size_t> smt_per_core;

  size_t num_clusters;
  size_t num_cores;
};

// Returns false, or fills per_package and initializes LP::cluster/core/smt.
bool DetectClusterAndCore(std::vector<Topology::LP>& lps,
                          std::vector<PerPackage>& per_package) {
  for (size_t lp = 0; lp < lps.size(); ++lp) {
    PerPackage& pp = per_package[lps[lp].package];
    size_t cluster, core;
    if (!ReadSysfs(kCluster, lp, kMaxLogicalProcessors, &cluster)) return false;
    if (!ReadSysfs(kCore, lp, kMaxLogicalProcessors, &core)) return false;

    // SMT ID is how many LP we have already seen assigned to the same core.
    const size_t smt = pp.smt_per_core[core]++;
    HWY_ASSERT(smt < 16);
    lps[lp].smt = static_cast<uint8_t>(smt);  // already contiguous

    // Certainly core, and likely also cluster, are HW-dependent opaque values.
    // We assume they do not exceed kMaxLogicalProcessors and thus fit in a set.
    pp.cluster_set.Set(cluster);
    pp.core_set.Set(core);
    // Temporary storage for remapping as in DetectPackages.
    lps[lp].cluster = static_cast<uint16_t>(cluster);
    lps[lp].core = static_cast<uint16_t>(core);
  }

  for (size_t p = 0; p < per_package.size(); ++p) {
    PerPackage& pp = per_package[p];
    std::map<size_t, size_t> cluster_ranks = RanksFromSet(pp.cluster_set);
    std::map<size_t, size_t> core_ranks = RanksFromSet(pp.core_set);
    // Remap *this packages'* per-lp cluster/core to their ranks.
    for (size_t lp = 0; lp < lps.size(); ++lp) {
      if (lps[lp].package != p) continue;
      lps[lp].cluster = static_cast<uint16_t>(cluster_ranks[lps[lp].cluster]);
      lps[lp].core = static_cast<uint16_t>(core_ranks[lps[lp].core]);
    }
    pp.num_clusters = cluster_ranks.size();
    pp.num_cores = core_ranks.size();
  }

  return true;
}

#endif  // HWY_OS_LINUX

HWY_DLLEXPORT Topology::Topology() {
#if HWY_OS_LINUX
  lps.resize(TotalLogicalProcessors());

  const size_t num_packages = DetectPackages(lps);
  if (num_packages == 0) return;
  std::vector<PerPackage> per_package(num_packages);
  if (!DetectClusterAndCore(lps, per_package)) return;

  // Allocate per-package/cluster/core vectors. This indicates to callers that
  // detection succeeded.
  packages.resize(num_packages);
  for (size_t p = 0; p < num_packages; ++p) {
    packages[p].clusters.resize(per_package[p].num_clusters);
    packages[p].cores.resize(per_package[p].num_cores);
  }

  // Populate the per-cluster/core sets of LP.
  for (size_t lp = 0; lp < lps.size(); ++lp) {
    packages[lps[lp].package].clusters[lps[lp].cluster].lps.Set(lp);
    packages[lps[lp].package].cores[lps[lp].core].lps.Set(lp);
  }
#endif
}

}  // namespace hwy
