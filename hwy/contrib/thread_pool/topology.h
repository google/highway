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

// OS-specific functions for thread affinity.

#include <stddef.h>

#include "hwy/base.h"

namespace hwy {

// Returns false if std::thread should not be used.
HWY_DLLEXPORT bool HaveThreadingSupport();

// Upper bound on logical processors, including hyperthreads, to enable
// fixed-size arrays. This limit matches glibc.
static constexpr size_t kMaxLogicalProcessors = 1024;

// Returns 1 if unknown, otherwise the total number of logical processors
// provided by the hardware. We assert this is at most `kMaxLogicalProcessors`.
// These processors are not necessarily all usable; you can determine which are
// via GetThreadAffinity().
HWY_DLLEXPORT size_t TotalLogicalProcessors();

// Custom two-level bitset because std::bitset does not include an iterator.
// Used by Get/SetThreadAffinity.
class LogicalProcessorSet {
 public:
  // No harm if `lp` is already set.
  void Set(size_t lp) {
    HWY_DASSERT(lp < TotalLogicalProcessors());
    const size_t idx = lp / 64;
    const size_t mod = lp % 64;
    bits_[idx] |= (1ULL << mod);
    nonzero_ |= (1ULL << idx);
    HWY_DASSERT(Get(lp));
  }

  // Equivalent to Set(i) for i in [0, 64) where (bits >> i) & 1. This does
  // not clear any existing bits.
  void SetNonzeroBitsFrom64(uint64_t bits) {
    bits_[0] |= bits;
    nonzero_ |= bits_[0] ? 1 : 0;
  }

  void Clear(size_t lp) {
    HWY_DASSERT(lp < TotalLogicalProcessors());
    const size_t idx = lp / 64;
    const size_t mod = lp % 64;
    bits_[idx] &= ~(1ULL << mod);
    if (bits_[idx] == 0) {
      nonzero_ &= ~(1ULL << idx);
    }
  }

  bool Get(size_t lp) const {
    HWY_DASSERT(lp < TotalLogicalProcessors());
    const size_t idx = lp / 64;
    const size_t mod = lp % 64;
    return (bits_[idx] & (1ULL << mod)) != 0;
  }

  // Returns uint64_t(Get(lp)) << lp for lp in [0, 64).
  uint64_t Get64() const { return bits_[0]; }

  // Calls func(lp) for each lp in the set.
  template <class Func>
  void Foreach(const Func& func) const {
    uint64_t remaining_idx = nonzero_;
    while (remaining_idx != 0) {
      const size_t idx = Num0BitsBelowLS1Bit_Nonzero64(remaining_idx);
      remaining_idx &= remaining_idx - 1;  // clear LSB

      uint64_t remaining_bits = bits_[idx];
      while (remaining_bits != 0) {
        const size_t mod = Num0BitsBelowLS1Bit_Nonzero64(remaining_bits);
        remaining_bits &= remaining_bits - 1;  // clear LSB
        func(idx * 64 + mod);
      }
    }
  }

  size_t Count() const {
    size_t total = 0;
    uint64_t remaining_idx = nonzero_;
    while (remaining_idx != 0) {
      const size_t idx = Num0BitsBelowLS1Bit_Nonzero64(remaining_idx);
      remaining_idx &= remaining_idx - 1;  // clear LSB
      total += PopCount(bits_[idx]);
    }
    return total;
  }

 private:
  static_assert(kMaxLogicalProcessors <= 64 * 64, "Single u64 insufficient");
  uint64_t nonzero_ = 0;
  uint64_t bits_[kMaxLogicalProcessors / 64] = {0};
};

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

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_THREAD_POOL_TOPOLOGY_H_
