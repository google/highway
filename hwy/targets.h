// Copyright 2019 Google LLC
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

#ifndef HWY_TARGETS_H_
#define HWY_TARGETS_H_

// Unique bit value for each target. Used in static/runtime_targets.h.
#define HWY_NONE 0
#define HWY_PPC8 1  // v2.07 or 3
#define HWY_AVX2 2
#define HWY_SSE4 4
#define HWY_ARM8 8
#define HWY_AVX512 16

#endif  // HWY_TARGETS_H_
