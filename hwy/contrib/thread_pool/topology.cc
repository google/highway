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
#include <string.h>  // strchr

#include <array>
#include <map>
#include <string>
#include <vector>

#include "hwy/base.h"  // HWY_OS_WIN, HWY_WARN

#if HWY_OS_APPLE
#include <sys/sysctl.h>

#include "hwy/aligned_allocator.h"  // HWY_ALIGNMENT
#endif

#if HWY_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601  // Windows 7 / Server 2008
#endif
#include <windows.h>
#endif  // HWY_OS_WIN

#if HWY_OS_LINUX || HWY_OS_FREEBSD
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>  // sysconf
#endif  // HWY_OS_LINUX || HWY_OS_FREEBSD

#if HWY_OS_FREEBSD
// must come after sys/types.h.
#include <sys/cpuset.h>  // CPU_SET
#endif                   // HWY_OS_FREEBSD

#if HWY_ARCH_WASM
#include <emscripten/threading.h>
#endif

namespace hwy {

HWY_CONTRIB_DLLEXPORT bool HaveThreadingSupport() {
#if HWY_ARCH_WASM
  return emscripten_has_threading_support() != 0;
#else
  return true;
#endif
}

// Returns `whole / part`, with a check that `part` evenly divides `whole`,
// which implies the result is exact.
static HWY_MAYBE_UNUSED size_t DivByFactor(size_t whole, size_t part) {
  HWY_ASSERT(part != 0);
  const size_t div = whole / part;
  const size_t mul = div * part;
  if (mul != whole) {
    HWY_ABORT("%zu / %zu = %zu; *%zu = %zu\n", whole, part, div, part, mul);
  }
  return div;
}

#if HWY_OS_APPLE

// Returns whether sysctlbyname() succeeded; if so, writes `val / div` to
// `out`, otherwise sets `err`.
template <typename T>
bool Sysctl(const char* name, size_t div, int& err, T* out) {
  size_t val = 0;
  size_t size = sizeof(val);
  // Last two arguments are for updating the value, which we do not want.
  const int ret = sysctlbyname(name, &val, &size, nullptr, 0);
  if (HWY_UNLIKELY(ret != 0)) {
    // Do not print warnings because some `name` are expected to fail.
    err = ret;
    return false;
  }
  *out = static_cast<T>(DivByFactor(val, div));
  return true;
}

#endif  // HWY_OS_APPLE

HWY_CONTRIB_DLLEXPORT size_t TotalLogicalProcessors() {
  size_t lp = 0;
#if HWY_ARCH_WASM
  const int num_cores = emscripten_num_logical_cores();
  if (num_cores > 0) lp = static_cast<size_t>(num_cores);
#elif HWY_OS_WIN
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);  // always succeeds
  // WARNING: this is only for the current group, hence limited to 64.
  lp = static_cast<size_t>(sysinfo.dwNumberOfProcessors);
#elif HWY_OS_LINUX
  // Use configured, not "online" (_SC_NPROCESSORS_ONLN), because we want an
  // upper bound.
  const long ret = sysconf(_SC_NPROCESSORS_CONF);  // NOLINT(runtime/int)
  if (ret < 0) {
    HWY_WARN("Unexpected _SC_NPROCESSORS_CONF = %d\n", static_cast<int>(ret));
  } else {
    lp = static_cast<size_t>(ret);
  }
#elif HWY_OS_APPLE
  int err;
  if (!Sysctl("hw.logicalcpu", 1, err, &lp)) {
    lp = 0;
  }
#endif

  if (HWY_UNLIKELY(lp == 0)) {  // Failed to detect.
    HWY_IF_CONSTEXPR(HWY_IS_DEBUG_BUILD) {
      HWY_WARN(
          "Unknown TotalLogicalProcessors, assuming 1. "
          "HWY_OS_: WIN=%d LINUX=%d APPLE=%d;\n"
          "HWY_ARCH_: WASM=%d X86=%d PPC=%d ARM=%d RISCV=%d S390X=%d\n",
          HWY_OS_WIN, HWY_OS_LINUX, HWY_OS_APPLE, HWY_ARCH_WASM, HWY_ARCH_X86,
          HWY_ARCH_PPC, HWY_ARCH_ARM, HWY_ARCH_RISCV, HWY_ARCH_S390X);
    }
    return 1;
  }

  // Warn that we are clamping.
  if (HWY_UNLIKELY(lp > kMaxLogicalProcessors)) {
    HWY_IF_CONSTEXPR(HWY_IS_DEBUG_BUILD) {
      HWY_WARN("OS reports %zu processors but clamping to %zu\n", lp,
               kMaxLogicalProcessors);
    }
    lp = kMaxLogicalProcessors;
  }

  return lp;
}

#ifdef __ANDROID__
#include <sys/syscall.h>
#endif

HWY_CONTRIB_DLLEXPORT bool GetThreadAffinity(LogicalProcessorSet& lps) {
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

HWY_CONTRIB_DLLEXPORT bool SetThreadAffinity(const LogicalProcessorSet& lps) {
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
        HWY_WARN("Unexpected error opening %s: %d\n", path, errno);
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
          HWY_WARN("Unexpected error closing file: %d\n", errno);
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
      // read instead of `pread`, which might not work for sysfs.
      const auto bytes_read = read(fd_, buf200 + pos, 200 - pos);
      if (bytes_read == 0) {  // EOF: done
        buf200[pos++] = '\0';
        return pos;
      }
      if (bytes_read == -1) {
        if (errno == EINTR) continue;  // signal: retry
        if (HWY_IS_DEBUG_BUILD) {
          HWY_WARN("Unexpected error reading file: %d\n", errno);
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

// Returns bytes read, or 0 on failure.
size_t ReadSysfs(const char* format, size_t lp, char* buf200) {
  char path[200];
  const int bytes_written = snprintf(path, sizeof(path), format, lp);
  HWY_ASSERT(0 < bytes_written &&
             bytes_written < static_cast<int>(sizeof(path) - 1));

  const File file(path);
  return file.Read(buf200);
}

// Interprets [str + pos, str + end) as base-10 ASCII. Stops when any non-digit
// is found, or at end. Returns false if no digits found.
bool ParseDigits(const char* str, const size_t end, size_t& pos, size_t* out) {
  HWY_ASSERT(pos <= end);
  // 9 digits cannot overflow even 32-bit size_t.
  const size_t stop = pos + 9;
  *out = 0;
  for (; pos < HWY_MIN(end, stop); ++pos) {
    const int c = str[pos];
    if (c < '0' || c > '9') break;
    *out *= 10;
    *out += static_cast<size_t>(c - '0');
  }
  if (pos == 0) {  // No digits found
    *out = 0;
    return false;
  }
  return true;
}

// Number, plus optional K or M suffix, plus terminator.
bool ParseNumberWithOptionalSuffix(const char* str, size_t len, size_t* out) {
  size_t pos = 0;
  if (!ParseDigits(str, len, pos, out)) return false;
  if (str[pos] == 'K') {
    *out <<= 10;
    ++pos;
  }
  if (str[pos] == 'M') {
    *out <<= 20;
    ++pos;
  }
  if (str[pos] != '\0' && str[pos] != '\n') {
    HWY_ABORT("Expected [suffix] terminator at %zu %s\n", pos, str);
  }
  return true;
}

bool ReadNumberWithOptionalSuffix(const char* format, size_t lp, size_t* out) {
  char buf200[200];
  const size_t pos = ReadSysfs(format, lp, buf200);
  if (pos == 0) return false;
  return ParseNumberWithOptionalSuffix(buf200, pos, out);
}

const char* kPackage =
    "/sys/devices/system/cpu/cpu%zu/topology/physical_package_id";
const char* kCluster = "/sys/devices/system/cpu/cpu%zu/cache/index3/id";
const char* kCore = "/sys/devices/system/cpu/cpu%zu/topology/core_id";
const char* kL2Size = "/sys/devices/system/cpu/cpu%zu/cache/index2/size";
const char* kL3Size = "/sys/devices/system/cpu/cpu%zu/cache/index3/size";
const char* kNode = "/sys/devices/system/node/node%zu/cpulist";

// sysfs values can be arbitrarily large, so store in a map and replace with
// indices in order of appearance.
class Remapper {
 public:
  // Returns false on error, or sets `out_index` to the index of the sysfs
  // value selected by `format` and `lp`.
  template <typename T>
  bool operator()(const char* format, size_t lp, T* HWY_RESTRICT out_index) {
    size_t opaque;
    if (!ReadNumberWithOptionalSuffix(format, lp, &opaque)) return false;

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

struct PackageSizes {
  size_t num_clusters;
  size_t num_cores;
};

// For internal use by `DetectPackages`.
struct PerPackage {
  Remapper clusters;
  Remapper cores;
  // We rely on this zero-init and increment it below.
  uint8_t smt_per_core[kMaxLogicalProcessors] = {0};
};

// Initializes `lps` and returns a PackageSizes vector (empty on failure)
// indicating the number of clusters and cores per package.
std::vector<PackageSizes> DetectPackages(std::vector<Topology::LP>& lps) {
  std::vector<PackageSizes> empty;

  Remapper packages;
  for (size_t lp = 0; lp < lps.size(); ++lp) {
    if (!packages(kPackage, lp, &lps[lp].package)) return empty;
  }
  std::vector<PerPackage> per_package(packages.Num());
  HWY_ASSERT(!per_package.empty());

  for (size_t lp = 0; lp < lps.size(); ++lp) {
    PerPackage& pp = per_package[lps[lp].package];
    // Not a failure: some CPUs lack a (shared) L3 cache.
    if (!pp.clusters(kCluster, lp, &lps[lp].cluster)) {
      lps[lp].cluster = 0;
    }

    if (!pp.cores(kCore, lp, &lps[lp].core)) return empty;

    // SMT ID is how many LP we have already seen assigned to the same core.
    HWY_ASSERT(lps[lp].core < kMaxLogicalProcessors);
    lps[lp].smt = pp.smt_per_core[lps[lp].core]++;
    HWY_ASSERT(lps[lp].smt < 16);
  }

  std::vector<PackageSizes> package_sizes(per_package.size());
  for (size_t p = 0; p < package_sizes.size(); ++p) {
    // Was zero if the package has no shared L3, see above.
    package_sizes[p].num_clusters = HWY_MAX(1, per_package[p].clusters.Num());
    package_sizes[p].num_cores = per_package[p].cores.Num();
    HWY_ASSERT(package_sizes[p].num_cores != 0);
  }
  return package_sizes;
}

std::vector<size_t> ExpandList(const char* list, size_t list_end,
                               size_t max_lp) {
  std::vector<size_t> expanded;
  constexpr size_t kNotFound = ~size_t{0};
  size_t pos = 0;

  // Returns first `found_pos >= pos` where `list[found_pos] == c`, or
  // `kNotFound`.
  const auto find = [list, list_end, &pos](char c) -> size_t {
    const char* found_ptr = strchr(list + pos, c);
    if (found_ptr == nullptr) return kNotFound;
    const size_t found_pos = static_cast<size_t>(found_ptr - list);
    HWY_ASSERT(found_pos < list_end && list[found_pos] == c);
    return found_pos;
  };

  // Reads LP number and advances `pos`. `end` is for verifying we did not
  // read past a known terminator, or the end of string.
  const auto parse_lp = [list, list_end, &pos, max_lp](size_t end) -> size_t {
    end = HWY_MIN(end, list_end);
    size_t lp;
    HWY_ASSERT(ParseDigits(list, end, pos, &lp));
    HWY_IF_CONSTEXPR(HWY_ARCH_RISCV) {
      // On RISC-V, both TotalLogicalProcessors and GetThreadAffinity may
      // under-report the count, hence clamp.
      lp = HWY_MIN(lp, max_lp);
    }
    HWY_ASSERT(lp <= max_lp);
    HWY_ASSERT(pos <= end);
    return lp;
  };

  // Parse all [first-]last separated by commas.
  for (;;) {
    // Single number or first of range: ends with dash, comma, or end.
    const size_t lp_range_first = parse_lp(HWY_MIN(find('-'), find(',')));

    if (list[pos] == '-') {  // range
      ++pos;                 // skip dash
      // Last of range ends with comma or end.
      const size_t lp_range_last = parse_lp(find(','));

      expanded.reserve(expanded.size() + lp_range_last - lp_range_first + 1);
      for (size_t lp = lp_range_first; lp <= lp_range_last; ++lp) {
        expanded.push_back(lp);
      }
    } else {  // single number
      expanded.push_back(lp_range_first);
    }

    // Done if reached end of string.
    if (pos == list_end || list[pos] == '\0' || list[pos] == '\n') {
      break;
    }
    // Comma means at least one more term is coming.
    if (list[pos] == ',') {
      ++pos;
      continue;
    }
    HWY_ABORT("Unexpected character at %zu in %s\n", pos, list);
  }  // for pos

  return expanded;
}

// Sets LP.node for all `lps`.
void SetNodes(std::vector<Topology::LP>& lps) {
  // For each NUMA node found via sysfs:
  for (size_t node = 0;; node++) {
    // Read its cpulist so we can scatter `node` to all its `lps`.
    char buf200[200];
    const size_t bytes_read = ReadSysfs(kNode, node, buf200);
    if (bytes_read == 0) break;
    const std::vector<size_t> list =
        ExpandList(buf200, bytes_read, lps.size() - 1);
    for (size_t lp : list) {
      lps[lp].node = static_cast<uint8_t>(node);
    }
  }
}

}  // namespace
#endif  // HWY_OS_LINUX

HWY_CONTRIB_DLLEXPORT Topology::Topology() {
#if HWY_OS_LINUX
  lps.resize(TotalLogicalProcessors());
  const std::vector<PackageSizes>& package_sizes = DetectPackages(lps);
  if (package_sizes.empty()) return;
  SetNodes(lps);

  // Allocate per-package/cluster/core vectors. This indicates to callers that
  // detection succeeded.
  packages.resize(package_sizes.size());
  for (size_t p = 0; p < packages.size(); ++p) {
    packages[p].clusters.resize(package_sizes[p].num_clusters);
    packages[p].cores.resize(package_sizes[p].num_cores);
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
      if (ReadNumberWithOptionalSuffix(kL2Size, lp, &bytes)) {
        c.private_kib = bytes >> 10;
      }
      if (ReadNumberWithOptionalSuffix(kL3Size, lp, &bytes)) {
        c.shared_kib = bytes >> 10;
      }
    }
  }
#endif
}

// ------------------------------ Cache detection

using Caches = std::array<Cache, 4>;

// We assume homogeneous caches across all clusters because some OS APIs return
// a single value for a class of CPUs.

#if HWY_OS_LINUX
std::string ReadString(const char* name, size_t index) {
  // First CPU is usually a P core.
  const std::string path("/sys/devices/system/cpu/cpu0/cache/index%zu/");
  char buf200[200];
  size_t end = ReadSysfs((path + name).c_str(), index, buf200);
  // Remove trailing newline/null to simplify string comparison.
  for (; end != 0; --end) {
    if (buf200[end - 1] != '\0' && buf200[end - 1] != '\n') break;
  }
  return std::string(buf200, buf200 + end);
}

template <typename T>
bool WriteSysfs(const char* name, size_t index, T* out) {
  const std::string str = ReadString(name, index);
  // Do not call `ParseNumberWithOptionalSuffix` because it acts on the
  // K suffix in "size", but we actually want KiB.
  size_t pos = 0;
  size_t val;
  if (!ParseDigits(str.c_str(), str.length(), pos, &val)) return false;
  HWY_ASSERT(pos <= str.length());
  *out = static_cast<T>(val);
  return true;
}

// Reading from sysfs is preferred because sysconf returns L3 associativity = 0
// on some CPUs, and does not indicate sharing across cores.
// https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-devices-system-cpu
static bool InitCachesSysfs(Caches& caches) {
  // For computing shared cache sizes.
  std::vector<hwy::Topology::LP> lps(TotalLogicalProcessors());
  const std::vector<PackageSizes> package_sizes = DetectPackages(lps);
  // `package_sizes` is only used to check that `lps` were filled.
  if (package_sizes.empty()) {
    HWY_WARN("no packages, shared cache sizes may be incorrect\n");
    return false;
  }

  for (size_t i = 0;; ++i) {
    const std::string type = ReadString("type", i);
    if (type.empty()) break;  // done, no more entries
    if (type != "Data" && type != "Unified") continue;
    uint32_t level;
    if (!WriteSysfs("level", i, &level)) continue;
    if (level != 1 && level != 2 && level != 3) continue;
    Cache& c = caches[level];

    // Check before overwriting any fields.
    if (c.size_kib != 0) {
      HWY_WARN("ignoring another L%u, first size %u\n", level, c.size_kib);
      continue;
    }

    const bool ok = WriteSysfs("size", i, &c.size_kib) &&
                    WriteSysfs("ways_of_associativity", i, &c.associativity) &&
                    WriteSysfs("number_of_sets", i, &c.sets);
    if (HWY_UNLIKELY(!ok)) {
      HWY_WARN("skipping partially-detected L%u, error %d\n", level, errno);
      c = Cache();
      continue;
    }

    // Compute line size *before* adjusting the size for sharing. Note that
    // `coherency_line_size` exists, but we are not sure that is the line size.
    const size_t bytes = static_cast<size_t>(c.size_kib) * 1024;
    const size_t lines = c.associativity * c.sets;
    c.bytes_per_line = static_cast<uint16_t>(DivByFactor(bytes, lines));

    // Divide by number of *cores* sharing the cache.
    const std::string shared_str = ReadString("shared_cpu_list", i);
    if (HWY_UNLIKELY(shared_str.empty())) {
      HWY_WARN("no shared_cpu_list for L%u %s\n", level, type.c_str());
      c.cores_sharing = 1;
    } else {
      const std::vector<size_t> shared_lps =
          ExpandList(shared_str.c_str(), shared_str.length(), lps.size() - 1);
      size_t num_cores = 0;
      for (size_t lp : shared_lps) {
        if (HWY_LIKELY(lp < lps.size())) {
          num_cores += lps[lp].smt == 0;
        } else {
          HWY_WARN("out of bounds lp %zu of %zu from %s\n", lp, lps.size(),
                   shared_str.c_str());
        }
      }
      if (num_cores == 0) {
        HWY_WARN("no cores sharing L%u %s, setting to 1\n", level,
                 type.c_str());
        num_cores = 1;
      }
      c.cores_sharing = static_cast<uint16_t>(num_cores);
      // There exist CPUs for which L3 is not evenly divisible by `num_cores`,
      // hence do not use `DivByFactor`. It is safer to round down.
      c.size_kib = static_cast<uint32_t>(c.size_kib / num_cores);
      c.sets = static_cast<uint32_t>(c.sets / num_cores);
    }
  }

  // Require L1 and L2 cache.
  if (HWY_UNLIKELY(caches[1].size_kib == 0 || caches[2].size_kib == 0)) {
    HWY_WARN("sysfs detected L1=%u L2=%u, err %x\n", caches[1].size_kib,
             caches[2].size_kib, errno);
    return false;
  }

  // L3 is optional; if not found, its size is already zero from static init.
  return true;
}

#endif  // HWY_OS_LINUX

#if HWY_OS_WIN
using SLPI = SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;

template <class Func>
bool ForEachSLPI(LOGICAL_PROCESSOR_RELATIONSHIP rel, Func&& func) {
  // Get required buffer size.
  DWORD buf_bytes = 0;
  HWY_ASSERT(!GetLogicalProcessorInformationEx(rel, nullptr, &buf_bytes));
  HWY_ASSERT(GetLastError() == ERROR_INSUFFICIENT_BUFFER);
  // Note: `buf_bytes` may be less than `sizeof(SLPI)`, which has padding.
  uint8_t* buf = static_cast<uint8_t*>(malloc(buf_bytes));
  HWY_ASSERT(buf);

  // Fill the buffer.
  SLPI* info = reinterpret_cast<SLPI*>(buf);
  if (HWY_UNLIKELY(!GetLogicalProcessorInformationEx(rel, info, &buf_bytes))) {
    free(buf);
    return false;
  }

  // Iterate over each SLPI. `sizeof(SLPI)` is unreliable, see above.
  uint8_t* pos = buf;
  while (pos < buf + buf_bytes) {
    info = reinterpret_cast<SLPI*>(pos);
    HWY_ASSERT(info->Relationship == rel);
    func(*info);
    pos += info->Size;
  }
  if (pos != buf + buf_bytes) {
    HWY_WARN("unexpected pos %p, end %p, buf_bytes %lu, sizeof(SLPI) %zu\n",
             pos, buf + buf_bytes, buf_bytes, sizeof(SLPI));
  }

  free(buf);
  return true;
}

static size_t NumBits(size_t num_groups, const GROUP_AFFINITY* affinity) {
  size_t total_bits = 0;
  for (size_t i = 0; i < num_groups; ++i) {
    size_t bits = 0;
    hwy::CopyBytes<sizeof(bits)>(&affinity[i].Mask, &bits);
    total_bits += hwy::PopCount(bits);
  }
  return total_bits;
}

static size_t MaxLogicalPerCore() {
  size_t max_logical = 0;
  ForEachSLPI(RelationProcessorCore, [&max_logical](const SLPI& info) {
    const PROCESSOR_RELATIONSHIP& p = info.Processor;
    max_logical = HWY_MAX(max_logical, NumBits(p.GroupCount, p.GroupMask));
  });
  HWY_ASSERT(max_logical != 0);
  return max_logical;
}

static bool InitCachesWin(Caches& caches) {
  const size_t max_logical_per_core = MaxLogicalPerCore();

  ForEachSLPI(RelationCache, [max_logical_per_core, &caches](const SLPI& info) {
    const CACHE_RELATIONSHIP& cr = info.Cache;
    if (cr.Type != CacheUnified && cr.Type != CacheData) return;
    if (1 <= cr.Level && cr.Level <= 3) {
      Cache& c = caches[cr.Level];
      HWY_ASSERT(c.size_kib == 0);  // not set yet
      c.size_kib = static_cast<uint32_t>(DivByFactor(cr.CacheSize, 1024));
      c.bytes_per_line = static_cast<uint16_t>(cr.LineSize);
      c.associativity = (cr.Associativity == CACHE_FULLY_ASSOCIATIVE)
                            ? Cache::kMaxAssociativity
                            : cr.Associativity;

      // How many cores share this cache?
      size_t shared_with = NumBits(cr.GroupCount, cr.GroupMasks);
      // Divide out hyperthreads. This core may have fewer than
      // `max_logical_per_core`, hence round up.
      shared_with = DivCeil(shared_with, max_logical_per_core);
      if (shared_with == 0) {
        HWY_WARN("no cores sharing L%u, setting to 1\n", cr.Level);
        shared_with = 1;
      }

      // Update `size_kib` to *per-core* portion.
      // There exist CPUs for which L3 is not evenly divisible by `shared_with`,
      // hence do not use `DivByFactor`. It is safer to round down.
      c.size_kib = static_cast<uint32_t>(c.size_kib / shared_with);
      c.cores_sharing = static_cast<uint16_t>(shared_with);
    }
  });

  // Require L1 and L2 cache.
  if (HWY_UNLIKELY(caches[1].size_kib == 0 || caches[2].size_kib == 0)) {
    HWY_WARN("Windows detected L1=%u, L2=%u, err %lx\n", caches[1].size_kib,
             caches[2].size_kib, GetLastError());
    return false;
  }

  // L3 is optional; if not found, its size is already zero from static init.
  return true;
}
#endif  // HWY_OS_WIN

#if HWY_OS_APPLE

static bool InitCachesApple(Caches& caches) {
  int err = 0;
  Cache& L1 = caches[1];
  Cache& L2 = caches[2];
  Cache& L3 = caches[3];

  // Total L1 and L2 size can be reliably queried, but prefer perflevel0
  // (P-cores) because hw.l1dcachesize etc. are documented to describe the
  // "least performant core".
  bool ok = Sysctl("hw.perflevel0.l1dcachesize", 1024, err, &L1.size_kib) ||
            Sysctl("hw.l1dcachesize", 1024, err, &L1.size_kib);
  ok &= Sysctl("hw.perflevel0.l2cachesize", 1024, err, &L2.size_kib) ||
        Sysctl("hw.l2cachesize", 1024, err, &L2.size_kib);
  if (HWY_UNLIKELY(!ok)) {
    HWY_WARN("Apple cache detection failed, error %d\n", err);
    return false;
  }
  L1.cores_sharing = 1;
  if (Sysctl("hw.perflevel0.cpusperl2", 1, err, &L2.cores_sharing)) {
    // There exist CPUs for which L2 is not evenly divisible by `cores_sharing`,
    // hence do not use `DivByFactor`. It is safer to round down.
    L2.size_kib /= L2.cores_sharing;
  } else {
    L2.cores_sharing = 1;
  }

  // Other properties are not always reported. Set `associativity` and
  // `bytes_per_line` based on known models.
  char brand[128] = {0};
  size_t size = sizeof(brand);
  if (!sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0)) {
    if (strncmp(brand, "Apple ", 6) != 0) {
      // Unexpected, but we will continue check the string suffixes.
      HWY_WARN("unexpected Apple brand %s\n", brand);
    }

    if (brand[6] == 'M') {
      // https://dougallj.github.io/applecpu/firestorm.html,
      // https://www.7-cpu.com/cpu/Apple_M1.html:
      L1.bytes_per_line = 64;
      L1.associativity = 8;
      L2.bytes_per_line = 128;
      if (brand[7] == '1') {  // M1
        L2.associativity = 12;
      } else if ('2' <= brand[7] && brand[7] <= '4') {  // M2/M3, maybe also M4
        L2.associativity = 16;
      } else {
        L2.associativity = 0;  // Unknown, set below via sysctl.
      }

      // Although Wikipedia lists SLC sizes per model, we do not know how it is
      // partitioned/allocated, so do not treat it as a reliable L3.
    }  // M*
  }  // brand string

  // This sysctl does not distinguish between L1 and L2 line sizes, so only use
  // it if we have not already set `bytes_per_line` above.
  uint16_t bytes_per_line;
  if (!Sysctl("hw.cachelinesize", 1, err, &bytes_per_line)) {
    bytes_per_line = static_cast<uint16_t>(HWY_ALIGNMENT);  // guess
  }
  for (size_t level = 1; level <= 3; ++level) {
    if (caches[level].bytes_per_line == 0) {
      caches[level].bytes_per_line = bytes_per_line;
    }
  }

  // Fill in associativity if not already set. Unfortunately this is only
  // reported on x86, not on M*.
  if (L1.associativity == 0 && !Sysctl("machdep.cpu.cache.L1_associativity", 1,
                                       err, &L1.associativity)) {
    L1.associativity = 8;  // guess
  }
  if (L2.associativity == 0 && !Sysctl("machdep.cpu.cache.L2_associativity", 1,
                                       err, &L2.associativity)) {
    L2.associativity = 12;  // guess
  }
  // There is no L3_associativity.
  if (L3.associativity == 0) {
    L3.associativity = 12;  // guess
  }

  // Now attempt to query L3. Although this sysctl is documented, M3 does not
  // report an L3 cache.
  if (L3.size_kib == 0 &&
      (Sysctl("hw.perflevel0.l3cachesize", 1024, err, &L3.size_kib) ||
       Sysctl("hw.l3cachesize", 1024, err, &L3.size_kib))) {
    // There exist CPUs for which L3 is not evenly divisible by `cores_sharing`,
    // hence do not use `DivByFactor`. It is safer to round down.
    if (Sysctl("hw.perflevel0.cpusperl3", 1, err, &L3.cores_sharing)) {
      L3.size_kib /= L3.cores_sharing;
    } else {
      L3.cores_sharing = 1;
    }
  }
  // If no L3 cache, reset all fields for consistency.
  if (L3.size_kib == 0) {
    L3 = Cache();
  }

  // Are there other useful sysctls? hw.cacheconfig appears to be how many
  // cores share the memory and caches, though this is not documented, and
  // duplicates information in hw.perflevel0.cpusperl*.

  return true;
}

#endif  // HWY_OS_APPLE

// Most APIs do not set the `sets` field, so compute it from the size and
// associativity, and if a value is already set, ensure it matches.
static HWY_MAYBE_UNUSED void ComputeSets(Cache& c) {
  // If there is no such cache, avoid division by zero.
  if (HWY_UNLIKELY(c.size_kib == 0)) {
    c.sets = 0;
    return;
  }
  const size_t bytes = static_cast<size_t>(c.size_kib) * 1024;
  // `size_kib` may have been rounded down, hence `lines` and `sets` are not
  // necessarily evenly divisible, so round down instead of `DivByFactor`.
  const size_t lines = bytes / c.bytes_per_line;
  const size_t sets = lines / c.associativity;

  if (c.sets == 0) {
    c.sets = static_cast<uint32_t>(sets);
  } else {
    if (c.sets != sets) {
      HWY_ABORT("Inconsistent cache sets %u != %zu\n", c.sets, sets);
    }
  }
}

static const Cache* InitDataCaches() {
  alignas(64) static Caches caches;

  // On failure, return immediately because InitCaches*() already warn.
#if HWY_OS_LINUX
  if (HWY_UNLIKELY(!InitCachesSysfs(caches))) return nullptr;
#elif HWY_OS_WIN
  if (HWY_UNLIKELY(!InitCachesWin(caches))) return nullptr;
#elif HWY_OS_APPLE
  if (HWY_UNLIKELY(!InitCachesApple(caches))) return nullptr;
#else
  HWY_WARN("Cache detection not implemented for this platform.\n");
  (void)caches;
  return nullptr;
#define HWY_NO_CACHE_DETECTION
#endif

  // Prevents "code not reached" warnings on WASM.
#ifndef HWY_NO_CACHE_DETECTION
  for (size_t level = 1; level <= 3; ++level) {
    ComputeSets(caches[level]);
  }

  // Heuristic to ignore SLCs such as on Ampere Altra, which should not be
  // treated as a reliable L3 because of their cache inclusion policy.
  // On Apple M*, these are not even reported as an L3.
  if (caches[3].cores_sharing >= 16 && caches[3].size_kib <= 512) {
    caches[3] = Cache();
  }

  return &caches[0];
#endif  // HWY_NO_CACHE_DETECTION
}

HWY_CONTRIB_DLLEXPORT const Cache* DataCaches() {
  static const Cache* caches = InitDataCaches();
  return caches;
}

}  // namespace hwy
