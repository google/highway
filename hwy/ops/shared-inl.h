// Copyright 2020 Google LLC
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

// Implementation details included from each ops/*.h.

// Separate header because foreach_target.h re-enables its include guard.
#include "hwy/ops/set_macros-inl.h"

// Normal include guard required for macros/symbols in hwy (instead of the
// unique-per-target hwy::NAMESPACE). NOTE: this header also has a per-target
// section after this include guard.
#ifndef HWY_SHARED_INL_H_
#define HWY_SHARED_INL_H_

#include <stddef.h>
#include <stdint.h>

// Clang 3.9 generates VINSERTF128 instead of the desired VBROADCASTF128,
// which would free up port5. However, inline assembly isn't supported on
// MSVC, results in incorrect output on GCC 8.3, and raises "invalid output size
// for constraint" errors on Clang (https://gcc.godbolt.org/z/-Jt_-F), hence we
// disable it.
#ifndef HWY_LOADDUP_ASM
#define HWY_LOADDUP_ASM 0
#endif

namespace hwy {

// Unfortunately the GCC/Clang intrinsics do not accept int64_t*.
using GatherIndex64 = long long int;  // NOLINT(google-runtime-int)
static_assert(sizeof(GatherIndex64) == 8, "Must be 64-bit type");

//------------------------------------------------------------------------------
// Controlling overload resolution

// Insert into template/function arguments to enable this overload only for
// vectors of AT MOST this many bits.
//
// Note that enabling for exactly 128 bits is unnecessary because a function can
// simply be overloaded with Vec128<T> and Full128<T> descriptor. Enabling for
// other sizes (e.g. 64 bit) can be achieved with Simd<T, 8 / sizeof(T)>.
#define HWY_IF_LE128(T, N) hwy::EnableIf<N * sizeof(T) <= 16>* = nullptr
#define HWY_IF_LE64(T, N) hwy::EnableIf<N * sizeof(T) <= 8>* = nullptr
#define HWY_IF_LE32(T, N) hwy::EnableIf<N * sizeof(T) <= 4>* = nullptr

#define HWY_IF_FLOAT(T) hwy::EnableIf<hwy::IsFloat<T>()>* = nullptr
// IsSigned<float>() is true, so cannot use that to differentiate int/float.
#define HWY_IF_NOT_FLOAT(T) hwy::EnableIf<!hwy::IsFloat<T>()>* = nullptr

// Empty struct used as a size tag type.
template <size_t N>
struct SizeTag {};

//------------------------------------------------------------------------------
// Conversion between types of the same size

// Unsigned/signed/floating-point types whose sizes are kSize bytes.
template <size_t kSize>
struct TypesOfSize;
template <>
struct TypesOfSize<1> {
  using Unsigned = uint8_t;
  using Signed = int8_t;
};
template <>
struct TypesOfSize<2> {
  using Unsigned = uint16_t;
  using Signed = int16_t;
};
template <>
struct TypesOfSize<4> {
  using Unsigned = uint32_t;
  using Signed = int32_t;
  using Float = float;
};
template <>
struct TypesOfSize<8> {
  using Unsigned = uint64_t;
  using Signed = int64_t;
  using Float = double;
};

template <typename T>
using MakeUnsigned = typename TypesOfSize<sizeof(T)>::Unsigned;
template <typename T>
using MakeSigned = typename TypesOfSize<sizeof(T)>::Signed;
template <typename T>
using MakeFloat = typename TypesOfSize<sizeof(T)>::Float;

}  // namespace hwy

#endif  // HWY_SHARED_INL_H_

//------------------------------------------------------------------------------
// Per-target definitions (relies on external include guard in highway.h)

// Target-specific types used by ops/*-inl.h.
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// SIMD operations are implemented as overloaded functions selected using a
// "descriptor" D := Simd<T, N>. T is the lane type, N the requested number of
// lanes >= 1 (always a power of two). In the common case, users do not choose N
// directly, but instead use HWY_FULL (the largest available size). N may differ
// from the hardware vector size. If N is less, only that many lanes will be
// loaded/stored.
//
// Only HWY_FULL(T) and N <= 16 / sizeof(T) are guaranteed to be available - the
// latter are useful if >128 bit vectors are unnecessary or undesirable.
//
// Users should not use the N of a Simd<> but instead query the actual number of
// lanes via Lanes().
template <typename Lane, size_t N>
struct Simd {
  constexpr Simd() = default;
  using T = Lane;
  static_assert((N & (N - 1)) == 0 && N != 0, "N must be a power of two");

  // Widening/narrowing ops change the number of lanes and/or their type.
  // To initialize such vectors, we need the corresponding descriptor types:

  // PromoteTo/DemoteTo with another lane type, but same number of lanes.
  template <typename NewLane>
  using Rebind = Simd<NewLane, N>;

  // MulEven with another lane type, but same total size.
  // Round up to correctly handle scalars with N=1.
  template <typename NewLane>
  using Repartition =
      Simd<NewLane, (N * sizeof(Lane) + sizeof(NewLane) - 1) / sizeof(NewLane)>;

  // LowerHalf with the same lane type, but half the lanes.
  // Round up to correctly handle scalars with N=1.
  using Half = Simd<T, (N + 1) / 2>;

  // Combine with the same lane type, but twice the lanes.
  using Twice = Simd<T, 2 * N>;
};

// Descriptor for the same number of lanes as D, but with the LaneType T.
template <class T, class D>
using Rebind = typename D::template Rebind<T>;

// Descriptor for the same total size as D, but with the LaneType T.
template <class T, class D>
using Repartition = typename D::template Repartition<T>;

// Descriptor for the same lane type as D, but half the lanes.
template <class D>
using Half = typename D::Half;

// Descriptor for the same lane type as D, but twice the lanes.
template <class D>
using Twice = typename D::Twice;

// Compile-time-constant, (typically but not guaranteed) an upper bound on the
// number of lanes.
// Prefer instead using Lanes() and dynamic allocation, or Rebind, or
// `#if HWY_CAP_GE*`.
template <typename T, size_t N>
HWY_INLINE HWY_MAYBE_UNUSED constexpr size_t MaxLanes(Simd<T, N>) {
  return N;
}

// (Potentially) non-constant actual size of the vector at runtime, subject to
// the limit imposed by the Simd. Useful for advancing loop counters.
template <typename T, size_t N>
HWY_INLINE HWY_MAYBE_UNUSED size_t Lanes(Simd<T, N>) {
  return N;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
