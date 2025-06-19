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

#ifndef HIGHWAY_HWY_PROFILER_H_
#define HIGHWAY_HWY_PROFILER_H_

// High precision, low overhead time measurements. Returns exact call counts and
// total elapsed time for user-defined 'zones' (code regions, i.e. C++ scopes).
//
// Uses RAII to capture begin/end timestamps, with user-specified zone names:
//   { PROFILER_ZONE("name"); /*code*/ } or
// the name of the current function:
//   void FuncToMeasure() { PROFILER_FUNC; /*code*/ }.
//
// After all threads have exited any zones, invoke PROFILER_PRINT_RESULTS() to
// print call counts and average durations [CPU cycles] to stdout, sorted in
// descending order of total duration.

#include "hwy/base.h"

// If zero, this file has no effect and no measurements will be recorded.
#ifndef PROFILER_ENABLED
#define PROFILER_ENABLED 0
#endif

#if PROFILER_ENABLED || HWY_IDE

#include <stddef.h>
#include <stdint.h>

namespace hwy {

// Copies `name` into the string table and returns its offset, which is used as
// the unique identifier for the zone.
HWY_DLLEXPORT uint32_t ProfilerAddZone(const char* name);
#define PROFILER_ADD_ZONE(name) hwy::ProfilerAddZone(name)

// Records zone enter/exit packets with timestamps.
class Zone {
 public:
  HWY_DLLEXPORT Zone(size_t thread_id, uint32_t zone_id);
  HWY_DLLEXPORT ~Zone();
};

// Registers `zone_id` using static initializers and creates a zone starting
// from here until the end of the current scope.
#define PROFILER_ZONE2(thread_id, name)                                \
  HWY_FENCE;                                                           \
  static const uint32_t HWY_CONCAT(ProfilerZone, __LINE__) =           \
      hwy::ProfilerAddZone(name);                                      \
  const hwy::Zone zone(thread_id, HWY_CONCAT(ProfilerZone, __LINE__)); \
  HWY_FENCE

// Creates a zone for an entire function when placed at its beginning.
#define PROFILER_FUNC2(thread_id) PROFILER_ZONE2(thread_id, __func__)

// Prints results. Must be called exactly once after all threads have exited
// all zones.
HWY_DLLEXPORT void ProfilerPrintResults();
#define PROFILER_PRINT_RESULTS hwy::ProfilerPrintResults

}  // namespace hwy

#else
#define PROFILER_ADD_ZONE(name) 0
#define PROFILER_ZONE2(thread_id, name)
#define PROFILER_FUNC2(thread_id)
#define PROFILER_PRINT_RESULTS()
#endif  // PROFILER_ENABLED || HWY_IDE

// For compatibility with old callers that do not pass thread_id.
#define PROFILER_FUNC PROFILER_FUNC2(0)
#define PROFILER_ZONE(name) PROFILER_ZONE2(0, name)

#endif  // HIGHWAY_HWY_PROFILER_H_
