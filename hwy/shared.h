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

#ifndef HWY_SHARED_H_
#define HWY_SHARED_H_

// Definitions shared between target-specific headers and possibly also users.

#include <stddef.h>
#include <stdint.h>

#include "hwy/compiler_specific.h"
#include "hwy/type_traits.h"

// 4 instances of a given literal value, useful as input to LoadDup128.
#define HWY_REP4(literal) literal, literal, literal, literal

#define HWY_MIN(a, b) ((a) < (b) ? (a) : (b))
#define HWY_MAX(a, b) ((a) < (b) ? (b) : (a))

namespace hwy {

// SIMD operations are implemented as overloaded functions selected using a
// "descriptor" D := Desc<T, N>. T is the lane type, N the number of lanes.
// kLanesOr0 is 0 for scalar (some SIMD users need kLanes=1).
template <typename Lane, size_t kLanesOr0>
struct Desc {
  constexpr Desc() = default;

  using T = Lane;

  // How far to advance loop counters (>= 1).
  static constexpr size_t N = HWY_MAX(kLanesOr0, 1);
  static_assert((N & (N - 1)) == 0, "N must be a power of two");

  // For creating a second Desc matching the first, use the original kLanesOr0
  // multiplied/divided by ratio of sizeof(Lane).
  static constexpr size_t LanesOr0() { return kLanesOr0; }
};

// Avoids linker errors in pre-C++17 debug builds.
template <typename Lane, size_t kLanesOr0>
constexpr size_t Desc<Lane, kLanesOr0>::N;

// Alias for the actual vector data, e.g. Vec0<float> for Desc<float, 0>,
// To avoid inadvertent conversions between vectors of different lengths, they
// have distinct types (Vec128<T, N>) on x86. On PPC, we do not use wrapper
// class templates due to poor code generation.
template <class D>
using VT = decltype(Zero(D()));

// Type tags for GetHalf(v).
struct Upper {};
struct Lower {};

// Unfortunately the GCC/Clang intrinsics do not accept int64_t*.
using GatherIndex64 = long long int;  // NOLINT(google-runtime-int)
static_assert(sizeof(GatherIndex64) == 8, "Must be 64-bit type");

// The source/destination must not overlap/alias.
template <size_t kBytes, typename From, typename To>
HWY_INLINE void CopyBytes(const From* from, To* to) {
#if HWY_COMPILER_MSVC
  const uint8_t* HWY_RESTRICT from_bytes =
      reinterpret_cast<const uint8_t*>(from);
  uint8_t* HWY_RESTRICT to_bytes = reinterpret_cast<uint8_t*>(to);
  for (size_t i = 0; i < kBytes; ++i) {
    to_bytes[i] = from_bytes[i];
  }
#else
  // Avoids horrible codegen on Clang (series of PINSRB)
  __builtin_memcpy(to, from, kBytes);
#endif
}

static HWY_INLINE size_t PopCount(const uint64_t x) {
#ifdef _MSC_VER
  return _mm_popcnt_u64(x);
#else
  return static_cast<size_t>(__builtin_popcountll(x));
#endif
}

}  // namespace hwy

#endif  // HWY_SHARED_H_
