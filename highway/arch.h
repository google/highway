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

#ifndef HIGHWAY_ARCH_H_
#define HIGHWAY_ARCH_H_

// Sets SIMD_ARCH to one of the following based on predefined macros:

#include <stddef.h>

#define SIMD_ARCH_X86 8
#define SIMD_ARCH_PPC 9
#define SIMD_ARCH_ARM 0xA
#define SIMD_ARCH_SCALAR 0xB

#if defined(__x86_64__) || defined(_M_X64)
#define SIMD_ARCH SIMD_ARCH_X86

#elif defined(__powerpc64__) || defined(_M_PPC)
#define SIMD_ARCH SIMD_ARCH_PPC

#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
#define SIMD_ARCH SIMD_ARCH_ARM

#elif defined(__EMSCRIPTEN__)
#define SIMD_ARCH SIMD_ARCH_SCALAR

#else
#error "Unsupported platform"
#endif

#if SIMD_ARCH == SIMD_ARCH_X86
static constexpr size_t kMaxVectorSize = 64;  // AVX512
#else
static constexpr size_t kMaxVectorSize = 16;
#endif

#endif  // HIGHWAY_ARCH_H_
