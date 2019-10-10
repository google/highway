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

#ifndef HIGHWAY_IN_TARGET_H_
#define HIGHWAY_IN_TARGET_H_

// Included from *_target.cc.

// Tolerate static_targets.h having been included (typically via a header) from
// a *_target.cc. We override its SIMD_TARGET for the rest of this translation
// unit. This chooses the specialization to generate (which could use another
// macro), but also governs other macros such as SIMD_ATTR. It is important
// for both static/runtime code to use the same macro, so they can share
// functions or headers.
#undef SIMD_TARGET

// For first #include of target.cc (see example in runtime_dispatch.h),
// which ensures the build system knows about the dependency. We could arrange
// for target.cc to be empty #ifndef SIMD_TARGET_INCLUDE, but then its contents
// would appear grayed out in some IDEs. Hence we let the normal #include
// instantiate for NONE, which is always enabled.
#define SIMD_TARGET NONE

#include "third_party/highway/highway/runtime_targets.h"
#include "third_party/highway/highway/targets.h"

// After runtime_targets.h:
#include "third_party/highway/highway/simd.h"

// Implementations must be wrapped in SIMD_NAMESPACE to prevent ODR violations
// caused by including *_target.cc multiple times via foreach_target.h.
// Note: we cannot wrap the entire *_target.cc in the namespace because they
// include headers (so that IDEs do not warn about unknown identifiers).
#define SIMD_NAMESPACE SIMD_CONCAT(N_, SIMD_TARGET)

#endif  // HIGHWAY_IN_TARGET_H_
