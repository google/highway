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

#ifndef HIGHWAY_TARGET_BITS_H_
#define HIGHWAY_TARGET_BITS_H_

// Defines unique bit values for each target for use in bitfields.

#define SIMD_NONE 0
#define SIMD_PPC8 1  // v2.07 or 3
#define SIMD_AVX2 2
#define SIMD_SSE4 4
#define SIMD_ARM8 8
#define SIMD_AVX512 16

// For selecting custom implementations (e.g. 8x8 DCT)
#define SIMD_BITS_NONE 0
#define SIMD_BITS_PPC8 128
#define SIMD_BITS_AVX2 256
#define SIMD_BITS_SSE4 128
#define SIMD_BITS_ARM8 128
#define SIMD_BITS_AVX512 512

#define SIMD_LANES_OR_0_NONE(size) 0
#define SIMD_LANES_OR_0_PPC8(size) (16 / size)
#define SIMD_LANES_OR_0_AVX2(size) (32 / size)
#define SIMD_LANES_OR_0_SSE4(size) (16 / size)
#define SIMD_LANES_OR_0_ARM8(size) (16 / size)
#define SIMD_LANES_OR_0_AVX512(size) (64 / size)

// Whether each target supports gather
#define SIMD_HAS_GATHER_NONE 0
#define SIMD_HAS_GATHER_PPC8 0
#define SIMD_HAS_GATHER_AVX2 1
#define SIMD_HAS_GATHER_SSE4 0
#define SIMD_HAS_GATHER_ARM8 0
#define SIMD_HAS_GATHER_AVX512 1

// Whether each target supports variable shifts (per-lane shift amount)
#define SIMD_HAS_VARIABLE_SHIFT_NONE 1
#define SIMD_HAS_VARIABLE_SHIFT_PPC8 1
#define SIMD_HAS_VARIABLE_SHIFT_AVX2 1
#define SIMD_HAS_VARIABLE_SHIFT_SSE4 0
#define SIMD_HAS_VARIABLE_SHIFT_ARM8 0
#define SIMD_HAS_VARIABLE_SHIFT_AVX512 1

#define SIMD_ATTR_NONE
#define SIMD_ATTR_PPC8
#define SIMD_ATTR_AVX2 SIMD_TARGET_ATTR("avx,avx2,fma")
#define SIMD_ATTR_SSE4 SIMD_TARGET_ATTR("sse4.1")
#define SIMD_ATTR_ARM8 SIMD_TARGET_ATTR("armv8-a+crypto")
#define SIMD_ATTR_AVX512 SIMD_TARGET_ATTR("avx512f,avx512vl,avx512dq,avx512bw")

// Use SIMD_TARGET to derive other macros. NOTE: SIMD_TARGET is only evaluated
// when these macros are expanded.
#define SIMD_CONCAT_IMPL(a, b) a##b
#define SIMD_CONCAT(a, b) SIMD_CONCAT_IMPL(a, b)

// How many bits in a full vector, or 0 for scalars.
#define SIMD_BITS SIMD_CONCAT(SIMD_BITS_, SIMD_TARGET)

// How many lanes of type T in a full vector. NOTE: cannot use in #if because
// this uses sizeof.
#define SIMD_LANES_OR_0(T) SIMD_CONCAT(SIMD_LANES_OR_0_, SIMD_TARGET)(sizeof(T))

// Whether features are available
#define SIMD_HAS_GATHER SIMD_CONCAT(SIMD_HAS_GATHER_, SIMD_TARGET)
#define SIMD_HAS_VARIABLE_SHIFT \
  SIMD_CONCAT(SIMD_HAS_VARIABLE_SHIFT_, SIMD_TARGET)

// Attributes; must precede every function declaration.
#define SIMD_ATTR SIMD_CONCAT(SIMD_ATTR_, SIMD_TARGET)

#endif  // HIGHWAY_TARGET_BITS_H_
