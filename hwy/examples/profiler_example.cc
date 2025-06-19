// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>

#include "hwy/base.h"  // Abort
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"

namespace hwy {
namespace {

void Spin(const double min_time) {
  const double t0 = hwy::platform::Now();
  for (;;) {
    const double elapsed = hwy::platform::Now() - t0;
    if (elapsed > min_time) {
      break;
    }
  }
}

HWY_NOINLINE void Spin10() {
  PROFILER_FUNC;
  Spin(10E-6);
}

HWY_NOINLINE void Spin20() {
  PROFILER_FUNC;
  Spin(20E-6);
}

HWY_NOINLINE void Spin3060() {
  {
    PROFILER_ZONE("spin30");
    Spin(30E-6);
  }
  {
    PROFILER_ZONE("spin60");
    Spin(60E-6);
  }
}

HWY_NOINLINE void Level3() {
  ThreadPool pool(3);
  pool.Run(0, 5, [](uint64_t /*task*/, HWY_MAYBE_UNUSED size_t thread) {
    PROFILER_ZONE2(static_cast<uint8_t>(thread), "Level3");
    for (int rep = 0; rep < 10; ++rep) {
      double total = 0.0;
      for (int i = 0; i < 100 - rep; ++i) {
        total += std::pow(0.9, i);
      }
      if (std::abs(total - 9.999) > 1E-2) {
        HWY_ABORT("unexpected total %f", total);
      }
    }
  });
}

HWY_NOINLINE void Level2() {
  PROFILER_FUNC;
  Level3();
}

HWY_NOINLINE void Level1() {
  PROFILER_FUNC;
  Level2();
}

void ProfilerExample() {
  {
    PROFILER_FUNC;
    Spin10();
    Spin20();
    Spin3060();
    Level1();
  }
  PROFILER_PRINT_RESULTS();
}

}  // namespace
}  // namespace hwy

int main(int /*argc*/, char* /*argv*/[]) {
  hwy::ProfilerExample();
  return 0;
}
