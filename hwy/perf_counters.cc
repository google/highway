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

#include "hwy/perf_counters.h"

#include "hwy/detect_compiler_arch.h"  // HWY_OS_LINUX

#if (HWY_OS_LINUX && HWY_CXX_LANG >= 201402L) || HWY_IDE
#include <errno.h>
#include <fcntl.h>  // open
#include <linux/perf_event.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>  // strcmp
#include <sys/ioctl.h>
#include <sys/stat.h>  // O_RDONLY
#include <sys/syscall.h>
#include <unistd.h>

#include <vector>

#include "hwy/base.h"  // HWY_ASSERT
#include "hwy/bit_set.h"

#endif  // HWY_OS_LINUX ..

namespace hwy {
namespace platform {

#if HWY_OS_LINUX && HWY_CXX_LANG >= 201402L

struct CounterConfig {  // for perf_event_open
  uint64_t config;
  uint32_t type;
};

CounterConfig FindCounterConfig(const char* name) {
  const auto eq = [name](const char* literal) {
    return !strcmp(name, literal);
  };

  constexpr uint32_t kHW = PERF_TYPE_HARDWARE;
  if (eq("ref_cycle")) return {PERF_COUNT_HW_REF_CPU_CYCLES, kHW};
  if (eq("instruction")) return {PERF_COUNT_HW_INSTRUCTIONS, kHW};
  if (eq("branch")) return {PERF_COUNT_HW_BRANCH_INSTRUCTIONS, kHW};
  if (eq("branch_mispred")) return {PERF_COUNT_HW_BRANCH_MISSES, kHW};
  if (eq("frontend_stall")) return {PERF_COUNT_HW_STALLED_CYCLES_FRONTEND, kHW};
  if (eq("backend_stall")) return {PERF_COUNT_HW_STALLED_CYCLES_BACKEND, kHW};

  constexpr uint64_t kL3 = PERF_COUNT_HW_CACHE_LL;
  constexpr uint64_t kLoad = uint64_t{PERF_COUNT_HW_CACHE_OP_READ} << 8;
  constexpr uint64_t kStore = uint64_t{PERF_COUNT_HW_CACHE_OP_WRITE} << 8;
  constexpr uint64_t kAcc = uint64_t{PERF_COUNT_HW_CACHE_RESULT_ACCESS} << 16;
  constexpr uint64_t kMiss = uint64_t{PERF_COUNT_HW_CACHE_RESULT_MISS} << 16;
  if (eq("l3_load")) return {kL3 | kLoad | kAcc, PERF_TYPE_HW_CACHE};
  if (eq("l3_store")) return {kL3 | kStore | kAcc, PERF_TYPE_HW_CACHE};
  if (eq("l3_load_miss")) return {kL3 | kLoad | kMiss, PERF_TYPE_HW_CACHE};
  if (eq("l3_store_miss")) return {kL3 | kStore | kMiss, PERF_TYPE_HW_CACHE};

  if (eq("page_fault")) return {PERF_COUNT_SW_PAGE_FAULTS, PERF_TYPE_SOFTWARE};

  HWY_ABORT("Bug: name %s does not match any known counter", name);
}

class PMU::Impl {
  static bool PerfCountersSupported() {
    // This is the documented way.
    struct stat s;
    return stat("/proc/sys/kernel/perf_event_paranoid", &s) == 0;
  }

  static perf_event_attr MakeAttr(const CounterConfig& cc) {
    perf_event_attr attr = {};
    attr.type = cc.type;
    attr.size = sizeof(attr);
    attr.config = cc.config;
    // We request more counters than the HW may support. If so, they are
    // multiplexed and only active for a fraction of the runtime. Recording the
    // times lets us extrapolate. Avoid GROUP because we want per-counter times.
    attr.read_format =
        PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    attr.inherit = 1;
    attr.exclude_kernel = 1;  // required if perf_event_paranoid == 1
    attr.exclude_hv = 1;      // = hypervisor
    return attr;
  }

  static int SysPerfEventOpen(const CounterConfig& cc, int group_fd) {
    perf_event_attr attr = MakeAttr(cc);
    // Only disable the group leader; other counters are gated on it.
    if (group_fd == -1) {
      attr.disabled = 1;
    }

    const int pid = 0;   // current process
    const int cpu = -1;  // any CPU
    return syscall(__NR_perf_event_open, &attr, pid, cpu, group_fd,
                   /*flags=*/0);
  }

  struct Buf {
    uint64_t value;
    uint64_t time_enabled;
    uint64_t time_running;
  };

 public:
  Impl() {
    if (!PerfCountersSupported()) {
      fprintf(stderr,
              "This Linux does not support perf counters. The program will"
              "continue, but counters will return zero.\n");
      return;
    }

    // Use groups so that all counters are enabled at the same time.
    int group_fd = -1;

    fds_.reserve(PerfCounters::Num());
    size_t idx_counter = 0;  // for valid_

    PerfCounters counters;  // unused
    counters.ForEach(       // requires C++14 lambda template
        &counters, [&](auto& /*val*/, auto& /*val2*/, const char* name) {
          const CounterConfig config = FindCounterConfig(name);
          const int fd = SysPerfEventOpen(config, group_fd);
          if (fd < 0) {
            fprintf(stderr, "perf_event_open error %d for counter %s\n", errno,
                    name);
          } else {
            // Set event count to zero to make overflow less likely.
            ioctl(fd, PERF_EVENT_IOC_RESET, 0);

            if (group_fd == -1) group_fd = fd;

            valid_.Set(idx_counter);
            fds_.push_back(fd);
          }

          if (idx_counter == 0) {
            // Ensure the first counter is a HW event, because later adding a HW
            // event to a group with only SW events is slow.
            HWY_ASSERT(config.type == PERF_TYPE_HARDWARE);
          }
          ++idx_counter;
        });

    HWY_ASSERT(fds_.size() == valid_.Count());
  }

  ~Impl() {
    for (int fd : fds_) {
      HWY_ASSERT(fd >= 0);
      HWY_ASSERT(close(fd) == 0);
    }
  }

  bool Start() {
    if (fds_.empty()) return false;  // ctor failed
    // Enabling the first fd (group leader) enables all.
    HWY_ASSERT(ioctl(fds_[0], PERF_EVENT_IOC_ENABLE, 0) == 0);
    return true;
  }

  double Stop(PerfCounters& counters) {
    if (fds_.empty()) return 0.0;  // ctor failed

    // First stop all so that we measure over the same time interval.
    ioctl(fds_[0], PERF_EVENT_IOC_DISABLE, 0);

    double min_fraction = 1.0;
    // Visits in the same order they were initialized.
    size_t idx_counter = 0;
    size_t idx_fd = 0;
    counters.ForEach(
        &counters, [&](auto& val, auto& /*val2*/, const char* name) {
          using T = hwy::RemoveRef<decltype(val)>;
          val = T{0};
          Buf buf;
          if (valid_.Get(idx_counter)) {
            const int fd = fds_[idx_fd++];
          AGAIN:
            const ssize_t bytes_read = read(fd, &buf, sizeof(buf));
            if (bytes_read < static_cast<ssize_t>(sizeof(buf))) {
              if (errno == EAGAIN) goto AGAIN;
              fprintf(stderr, "perf_counters read() error %d for %s\n", errno,
                      name);
            } else {
              HWY_ASSERT(buf.time_running <= buf.time_enabled);
              if (buf.time_running != 0) {
                const double fraction =
                    static_cast<double>(buf.time_running) / buf.time_enabled;
                HWY_ASSERT(0.0 < fraction && fraction <= 1.0);
                min_fraction = HWY_MIN(min_fraction, fraction);
                val = static_cast<T>(static_cast<double>(buf.value) / fraction);
              }
            }
          }
          ++idx_counter;
        });
    return min_fraction;
  }

 private:
  BitSet64 valid_;        // which counters are available
  std::vector<int> fds_;  // size == valid_.Count()
};

PMU::PMU() : impl_(new Impl) {}
PMU::~PMU() = default;
bool PMU::Start() { return impl_->Start(); }
double PMU::Stop(PerfCounters& counters) { return impl_->Stop(counters); }
#else
PMU::PMU() {}
PMU::~PMU() = default;
bool PMU::Start() { return false; }
double PMU::Stop(PerfCounters& /*counters*/) { return 0.0; }
#endif  // HWY_OS_LINUX ..

}  // namespace platform
}  // namespace hwy
