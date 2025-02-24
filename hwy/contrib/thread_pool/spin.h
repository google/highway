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

namespace hwy {

// User-space monitor/wait are supported on Zen2+ AMD and SPR+ Intel. Spin waits
// are rarely called from SIMD code, hence we do not integrate this into
// `HWY_TARGET` and its runtime dispatch mechanism. Returned by `Type()`, also
// used to disable modes at runtime in `ChooseSpin`.
enum class SpinType {
  kMonitorX,  // AMD
  kUMonitor,  // Intel
  kPause
};

// Returned by `UntilDifferent` in a single register.
struct SpinResult {
  uint32_t current;
  // Number of retries before returning, useful for checking that the
  // monitor/wait did not just return immediately.
  uint32_t reps;
};

// Interface for various spin-wait implementations. We rely on a VTable because
// we may add more spin types and conditions.
struct ISpin {  // abstract base class
  virtual ~ISpin() = default;

  virtual SpinType Type() const = 0;

  // For printing which is in use.
  virtual const char* String() const = 0;

  // Spins until `watched != prev` and returns the new value, similar to
  // `BlockUntilDifferent` in `futex.h`.
  virtual SpinResult UntilDifferent(uint32_t prev,
                                    std::atomic<uint32_t>& watched) = 0;

  // Returns number of retries until `watched == expected`.
  virtual size_t UntilEqual(uint32_t expected,
                            std::atomic<uint32_t>& watched) = 0;
};

// For runtime dispatch. Returns the best-available type whose bit in `disabled`
// is not set. Example: to disable kUMonitor, pass `1 <<
// static_cast<int>(SpinType::kUMonitor)`. Ignores `disabled` for `kPause` if
// that is the only supported and enabled type.
HWY_CONTRIB_DLLEXPORT ISpin* ChooseSpin(int disabled = 0);

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_THREAD_POOL_SPIN_H_
