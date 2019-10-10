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

// No include guard (included multiple times), textual inclusion.

#undef SIMD_ALL_TARGETS
#if defined(SIMD_RUNTIME_TARGETS) && defined(SIMD_STATIC_TARGETS)
#define SIMD_ALL_TARGETS (SIMD_RUNTIME_TARGETS | SIMD_STATIC_TARGETS)
#elif defined(SIMD_RUNTIME_TARGETS)
#define SIMD_ALL_TARGETS SIMD_RUNTIME_TARGETS
#elif defined(SIMD_STATIC_TARGETS)
#define SIMD_ALL_TARGETS SIMD_STATIC_TARGETS
#else
#error "Must include static_targets.h or in_target.h"
// Avoid undefined-identifier warnings in IDE:
#define SIMD_ALL_TARGETS 0
#include "third_party/highway/highway/target_bits.h"
#endif

#if SIMD_ALL_TARGETS & SIMD_ARM8
#include "third_party/highway/highway/arm64_neon.h"
#endif

#if SIMD_ALL_TARGETS & SIMD_SSE4
#include "third_party/highway/highway/x86_sse4.h"
#endif

#if SIMD_ALL_TARGETS & SIMD_AVX2
#include "third_party/highway/highway/x86_avx2.h"
#endif

#if SIMD_ALL_TARGETS & SIMD_AVX512
#include "third_party/highway/highway/x86_avx512.h"
#endif

#if defined(SIMD_RUNTIME_TARGETS) || (SIMD_ALL_TARGETS == 0)
#include "third_party/highway/highway/scalar.h"
#endif
