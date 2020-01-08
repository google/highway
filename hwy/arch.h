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

#ifndef HWY_ARCH_H_
#define HWY_ARCH_H_

// Sets HWY_ARCH to one of the following based on predefined macros:

#include <stddef.h>

#define HWY_ARCH_X86 8
#define HWY_ARCH_PPC 9
#define HWY_ARCH_ARM 0xA
#define HWY_ARCH_SCALAR 0xB
#define HWY_ARCH_WASM 0xC

#if defined(__wasm_simd128__)
#define HWY_ARCH HWY_ARCH_WASM

#elif defined(__i386__) || defined(__x86_64__) || defined(_M_X64)
#define HWY_ARCH HWY_ARCH_X86

#elif defined(__powerpc64__) || defined(_M_PPC)
#define HWY_ARCH HWY_ARCH_PPC

#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
#define HWY_ARCH HWY_ARCH_ARM

#elif defined(__EMSCRIPTEN__)
#define HWY_ARCH HWY_ARCH_SCALAR

#else
#error "Unsupported platform"
#endif

#if HWY_ARCH == HWY_ARCH_X86
static constexpr size_t kMaxVectorSize = 64;  // AVX512
#else
static constexpr size_t kMaxVectorSize = 16;
#endif

#endif  // HWY_ARCH_H_
