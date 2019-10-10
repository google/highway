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

#ifndef HIGHWAY_SHARED_H_
#define HIGHWAY_SHARED_H_

// Shared definitions used by target-specific headers.

#include <stddef.h>
#include <stdint.h>

#include <atomic>

#include "third_party/highway/highway/compiler_specific.h"
#include "third_party/highway/highway/target_bits.h"

// Ensures an array is aligned and suitable for load()/store() functions.
// Example: SIMD_ALIGN T lanes[d.N];
#define SIMD_ALIGN alignas(64)

// 4 instances of a given literal value, useful as input to load_dup128.
#define SIMD_REP4(literal) literal, literal, literal, literal

// Alternative for asm volatile("" : : : "memory"), which has no effect.
#define SIMD_FENCE std::atomic_thread_fence(std::memory_order_acq_rel)

#define SIMD_MIN(a, b) ((a) < (b) ? (a) : (b))
#define SIMD_MAX(a, b) ((a) < (b) ? (b) : (a))

namespace jxl {

// SIMD operations are implemented as overloaded functions selected using a
// "descriptor" D := Desc<T, N>. For example: `VT<D> setzero(D)`. T is the lane
// type, N the number of lanes. The return type `VT<D>` is either a full vector
// of at least 128 bits, an N-lane (=2^j) part, or a scalar.

// Descriptor: properties that uniquely identify a vector/part/scalar. Used to
// select overloaded functions; see Full/Part/Scalar aliases below.
// kLanesOr0 is 0 for scalar (some SIMD users need kLanes=1). This is ensured by
// setting it to SIMD_LANES_OR_0(T) or using SIMD_CAPPED.
template <typename Lane, size_t kLanesOr0>
struct Desc {
  constexpr Desc() = default;

  using T = Lane;

  // For NONE, we still want to advance loop counters by d.N, i.e. >= 1.
  static constexpr size_t N = SIMD_MAX(kLanesOr0, 1);
  static_assert((N & (N - 1)) == 0, "N must be a power of two");
};

// Shorthand for Part/Full. NOTE: uses SIMD_TARGET at the moment of expansion,
// not its current (possibly undefined) value.
#define SIMD_FULL(T) Desc<T, SIMD_LANES_OR_0(T)>

// A vector/part of no more than MAX_N lanes. Useful for 8x8 DCTs.
#define SIMD_CAPPED(T, MAX_N) Desc<T, SIMD_MIN(MAX_N, SIMD_LANES_OR_0(T))>

// Alias for the actual vector data, e.g. scalar<float> for Desc<float, 1>,
// returned by initializers such as setzero(). Parts and full vectors are
// distinct types on x86 to avoid inadvertent conversions. By contrast, PPC
// parts are merely aliases for full vectors to avoid wrapper overhead.
template <class D>
using VT = decltype(setzero(D()));

// Type tags for upper_half(v) etc.
struct Upper {};
struct Lower {};
#define SIMD_HALF Lower()

// Unfortunately the GCC/Clang intrinsics do not accept int64_t*.
using GatherIndex64 = long long int;
static_assert(sizeof(GatherIndex64) == 8, "Must be 64-bit type");

template <bool Condition, class T>
struct enable_if {};
template <class T>
struct enable_if<true, T> {
  using type = T;
};

template <bool Condition, class T = void>
using enable_if_t = typename enable_if<Condition, T>::type;

// Insert into template/function arguments to avoid ambiguity between
// arbitrary T, N (for parts) and NONE/AVX2 etc. overloads.
#define SIMD_IF128(T, N) enable_if_t<N != 0 && N * sizeof(T) <= 16>* = nullptr

// memcpy/memset.

// The source/destination must not overlap/alias.
template <size_t kBytes, typename From, typename To>
SIMD_INLINE void CopyBytes(const From* from, To* to) {
#if SIMD_COMPILER == SIMD_COMPILER_MSVC
  const uint8_t* SIMD_RESTRICT from_bytes =
      reinterpret_cast<const uint8_t*>(from);
  uint8_t* SIMD_RESTRICT to_bytes = reinterpret_cast<uint8_t*>(to);
  for (size_t i = 0; i < kBytes; ++i) {
    to_bytes[i] = from_bytes[i];
  }
#else
  // Avoids horrible codegen on Clang (series of PINSRB)
  __builtin_memcpy(to, from, kBytes);
#endif
}

template <typename T>
SIMD_INLINE void SetBytes(const uint8_t byte, T* t) {
  uint8_t* bytes = reinterpret_cast<uint8_t*>(t);
  for (size_t i = 0; i < sizeof(T); ++i) {
    bytes[i] = byte;
  }
}

// numeric_limits<T>

template <typename T>
constexpr bool IsFloat() {
  return T(1.25) != T(1);
}

template <typename T>
constexpr bool IsSigned() {
  return T(0) > T(-1);
}

// Largest/smallest representable integer values.
template <typename T>
constexpr T LimitsMax() {
  return IsSigned<T>() ? T((1ULL << (sizeof(T) * 8 - 1)) - 1)
                       : static_cast<T>(~0ull);
}
template <typename T>
constexpr T LimitsMin() {
  return IsSigned<T>() ? T(-1) - LimitsMax<T>() : T(0);
}

// Returns a name for the vector/part/scalar. The type prefix is u/i/f for
// unsigned/signed/floating point, followed by the number of bits per lane;
// then 'x' followed by the number of lanes. Example: u8x16. This is useful for
// understanding which instantiation of a generic test failed.
template <typename T, size_t N>
inline const char* type_name() {
  // Avoids depending on <type_traits>.
  constexpr bool is_float = T(2) < T(2.25);
  constexpr bool is_signed = T(-1) < T(0);
  constexpr char prefix = is_float ? 'f' : (is_signed ? 'i' : 'u');

  constexpr size_t bits = sizeof(T) * 8;
  constexpr char bits10 = '0' + (bits / 10);
  constexpr char bits1 = '0' + (bits % 10);

  // Scalars: omit the xN suffix.
  if (N == 1) {
    static constexpr char name1[8] = {prefix, bits1};
    static constexpr char name2[8] = {prefix, bits10, bits1};
    return sizeof(T) == 1 ? name1 : name2;
  }

  constexpr char N1 = (N < 10) ? '\0' : '0' + (N % 10);
  constexpr char N10 = (N < 10) ? '0' + (N % 10) : '0' + (N / 10);

  static constexpr char name1[8] = {prefix, bits1, 'x', N10, N1};
  static constexpr char name2[8] = {prefix, bits10, bits1, 'x', N10, N1};
  return sizeof(T) == 1 ? name1 : name2;
}

// Empty struct used as a size tag type.
template <size_t N>
struct SizeTag {};

}  // namespace jxl

#endif  // HIGHWAY_SHARED_H_
