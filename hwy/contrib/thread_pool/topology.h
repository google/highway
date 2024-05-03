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

#ifndef HIGHWAY_HWY_CONTRIB_THREAD_POOL_TOPOLOGY_H_
#define HIGHWAY_HWY_CONTRIB_THREAD_POOL_TOPOLOGY_H_

// OS-specific functions for processor topology and thread affinity.

#include <stddef.h>

#include <vector>

#include "hwy/base.h"

namespace hwy {

// 64-bit specialization of std::bitset, which lacks Foreach.
class BitSet64 {
 public:
  // No harm if `i` is already set.
  void Set(size_t i) {
    HWY_DASSERT(i < 64);
    bits_ |= (1ULL << i);
    HWY_DASSERT(Get(i));
  }

  // Equivalent to Set(i) for i in [0, 64) where (bits >> i) & 1. This does
  // not clear any existing bits.
  void SetNonzeroBitsFrom64(uint64_t bits) { bits_ |= bits; }

  void Clear(size_t i) {
    HWY_DASSERT(i < 64);
    bits_ &= ~(1ULL << i);
  }

  bool Get(size_t i) const {
    HWY_DASSERT(i < 64);
    return (bits_ & (1ULL << i)) != 0;
  }

  // Returns true if any Get(i) would return true for i in [0, 64).
  bool Any() const { return bits_ != 0; }

  // Returns uint64_t(Get(i)) << i for i in [0, 64).
  uint64_t Get64() const { return bits_; }

  // Calls func(i) for each i in the set.
  template <class Func>
  void Foreach(const Func& func) const {
    uint64_t remaining_bits = bits_;
    while (remaining_bits != 0) {
      const size_t i = Num0BitsBelowLS1Bit_Nonzero64(remaining_bits);
      remaining_bits &= remaining_bits - 1;  // clear LSB
      func(i);
    }
  }

  size_t Count() const { return PopCount(bits_); }

 private:
  uint64_t bits_ = 0;
};

// Two-level bitset for up to kMaxSize <= 4096 values.
template <size_t kMaxSize = 4096>
class BitSet4096 {
 public:
  // No harm if `i` is already set.
  void Set(size_t i) {
    HWY_DASSERT(i < kMaxSize);
    const size_t idx = i / 64;
    const size_t mod = i % 64;
    bits_[idx].Set(mod);
    nonzero_.Set(idx);
    HWY_DASSERT(Get(i));
  }

  // Equivalent to Set(i) for i in [0, 64) where (bits >> i) & 1. This does
  // not clear any existing bits.
  void SetNonzeroBitsFrom64(uint64_t bits) {
    bits_[0].SetNonzeroBitsFrom64(bits);
    if (bits) nonzero_.Set(0);
  }

  void Clear(size_t i) {
    HWY_DASSERT(i < kMaxSize);
    const size_t idx = i / 64;
    const size_t mod = i % 64;
    bits_[idx].Clear(mod);
    if (!bits_[idx].Any()) {
      nonzero_.Clear(idx);
    }
    HWY_DASSERT(!Get(i));
  }

  bool Get(size_t i) const {
    HWY_DASSERT(i < kMaxSize);
    const size_t idx = i / 64;
    const size_t mod = i % 64;
    return bits_[idx].Get(mod);
  }

  // Returns uint64_t(Get(i)) << i for i in [0, 64).
  uint64_t Get64() const { return bits_[0].Get64(); }

  // Calls func(i) for each i in the set.
  template <class Func>
  void Foreach(const Func& func) const {
    nonzero_.Foreach([&func, this](size_t idx) {
      bits_[idx].Foreach([idx, &func](size_t mod) { func(idx * 64 + mod); });
    });
  }

  size_t Count() const {
    size_t total = 0;
    nonzero_.Foreach(
        [&total, this](size_t idx) { total += bits_[idx].Count(); });
    return total;
  }

 private:
  static_assert(kMaxSize <= 64 * 64, "One BitSet64 insufficient");
  BitSet64 nonzero_;
  BitSet64 bits_[kMaxSize / 64];
};

// Returns false if std::thread should not be used.
HWY_DLLEXPORT bool HaveThreadingSupport();

// Upper bound on logical processors, including hyperthreads.
static constexpr size_t kMaxLogicalProcessors = 1024;  // matches glibc

// Set used by Get/SetThreadAffinity.
using LogicalProcessorSet = BitSet4096<kMaxLogicalProcessors>;

// Returns false, or sets `lps` to all logical processors which are online and
// available to the current thread.
HWY_DLLEXPORT bool GetThreadAffinity(LogicalProcessorSet& lps);

// Ensures the current thread can only run on the logical processors in `lps`.
// Returns false if not supported (in particular on Apple), or if the
// intersection between `lps` and `GetThreadAffinity` is the empty set.
HWY_DLLEXPORT bool SetThreadAffinity(const LogicalProcessorSet& lps);

// Returns false, or ensures the current thread will only run on `lp`, which
// must not exceed `TotalLogicalProcessors`. Note that this merely calls
// `SetThreadAffinity`, see the comment there.
static inline bool PinThreadToLogicalProcessor(size_t lp) {
  LogicalProcessorSet lps;
  lps.Set(lp);
  return SetThreadAffinity(lps);
}

// Returns 1 if unknown, otherwise the total number of logical processors
// provided by the hardware clamped to `kMaxLogicalProcessors`.
// These processors are not necessarily all usable; you can determine which are
// via GetThreadAffinity().
HWY_DLLEXPORT size_t TotalLogicalProcessors();

struct Topology {
  // Caller must check packages.empty(); if so, do not use any fields.
  HWY_DLLEXPORT Topology();

  // Clique of cores with lower latency to each other. On Apple M1 these are
  // four cores sharing an L2. On AMD these 'CCX' are up to eight cores sharing
  // an L3 and a memory controller.
  struct Cluster {
    LogicalProcessorSet lps;
  };

  struct Core {
    LogicalProcessorSet lps;
  };

  struct Package {
    std::vector<Cluster> clusters;
    std::vector<Core> cores;
  };

  std::vector<Package> packages;

  // Several hundred instances, so prefer a compact representation.
#pragma pack(push, 1)
  struct LP {
    uint16_t package = 0;
    uint16_t cluster = 0;  // local to the package, not globally unique
    uint16_t core = 0;     // local to the package, not globally unique
    uint8_t smt = 0;       // local to the package and core
    uint8_t reserved = 0;
  };
#pragma pack(pop)
  std::vector<LP> lps;  // size() == TotalLogicalProcessors().
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_THREAD_POOL_TOPOLOGY_H_
