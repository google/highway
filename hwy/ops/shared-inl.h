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

#include "hwy/base.h"

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

#define HWY_IF_UNSIGNED(T) hwy::EnableIf<!IsSigned<T>()>* = nullptr
#define HWY_IF_SIGNED(T) \
  hwy::EnableIf<IsSigned<T>() && !IsFloat<T>()>* = nullptr
#define HWY_IF_FLOAT(T) hwy::EnableIf<hwy::IsFloat<T>()>* = nullptr
#define HWY_IF_NOT_FLOAT(T) hwy::EnableIf<!hwy::IsFloat<T>()>* = nullptr

#define HWY_IF_LANE_SIZE(T, bytes) \
  hwy::EnableIf<sizeof(T) == (bytes)>* = nullptr
#define HWY_IF_NOT_LANE_SIZE(T, bytes) \
  hwy::EnableIf<sizeof(T) != (bytes)>* = nullptr

// Argument is a Simd<T, N>, defined below.
#define HWY_IF_UNSIGNED_D(D) HWY_IF_UNSIGNED(TFromD<D>)
#define HWY_IF_SIGNED_D(D) HWY_IF_SIGNED(TFromD<D>)
#define HWY_IF_FLOAT_D(D) HWY_IF_FLOAT(TFromD<D>)
#define HWY_IF_NOT_FLOAT_D(D) HWY_IF_NOT_FLOAT(TFromD<D>)

#define HWY_IF_LANE_SIZE_D(D, bytes) HWY_IF_LANE_SIZE(TFromD<D>, bytes)
#define HWY_IF_NOT_LANE_SIZE_D(D, bytes) HWY_IF_NOT_LANE_SIZE(TFromD<D>, bytes)

// Empty struct used as a size tag type.
template <size_t N>
struct SizeTag {};

//------------------------------------------------------------------------------
// Same size types, half-width, double-width

template <typename T>
struct TypeTraits;
template <>
struct TypeTraits<uint8_t> {
  using Unsigned = uint8_t;
  using Signed = int8_t;
  using Wide = uint16_t;
};
template <>
struct TypeTraits<int8_t> {
  using Unsigned = uint8_t;
  using Signed = int8_t;
  using Wide = int16_t;
};
template <>
struct TypeTraits<uint16_t> {
  using Unsigned = uint16_t;
  using Signed = int16_t;
  using Wide = uint32_t;
  using Narrow = uint8_t;
};
template <>
struct TypeTraits<int16_t> {
  using Unsigned = uint16_t;
  using Signed = int16_t;
  using Wide = int32_t;
  using Narrow = int8_t;
};
template <>
struct TypeTraits<uint32_t> {
  using Unsigned = uint32_t;
  using Signed = int32_t;
  using Float = float;
  using Wide = uint64_t;
  using Narrow = uint16_t;
};
template <>
struct TypeTraits<int32_t> {
  using Unsigned = uint32_t;
  using Signed = int32_t;
  using Float = float;
  using Wide = int64_t;
  using Narrow = int16_t;
};
template <>
struct TypeTraits<uint64_t> {
  using Unsigned = uint64_t;
  using Signed = int64_t;
  using Float = double;
  using Narrow = uint32_t;
};
template <>
struct TypeTraits<int64_t> {
  using Unsigned = uint64_t;
  using Signed = int64_t;
  using Float = double;
  using Narrow = int32_t;
};
template <>
struct TypeTraits<float> {
  using Unsigned = uint32_t;
  using Signed = int32_t;
  using Float = float;
  using Wide = double;
};
template <>
struct TypeTraits<double> {
  using Unsigned = uint64_t;
  using Signed = int64_t;
  using Float = double;
  using Narrow = float;
};

template <typename T>
using MakeUnsigned = typename TypeTraits<T>::Unsigned;
template <typename T>
using MakeSigned = typename TypeTraits<T>::Signed;
template <typename T>
using MakeFloat = typename TypeTraits<T>::Float;

template <typename T>
using MakeWide = typename TypeTraits<T>::Wide;
template <typename T>
using MakeNarrow = typename TypeTraits<T>::Narrow;

}  // namespace hwy

#endif  // HWY_SHARED_INL_H_

//------------------------------------------------------------------------------
// Per-target definitions (relies on external include guard in highway.h)

// Target-specific types used by ops/*-inl.h.
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// SIMD operations are implemented as overloaded functions selected using a
// "descriptor" D := Simd<T, N>. T is the lane type, N a number of lanes >= 1
// (always a power of two). Users generally do not choose N directly, but
// instead use HWY_FULL(T[, LMUL]) (the largest available size). N is not
// necessarily the actual number of lanes, which is returned by Lanes(D()).
//
// Only HWY_FULL(T) and N <= 16 / sizeof(T) are guaranteed to be available - the
// latter are useful if >128 bit vectors are unnecessary or undesirable.
template <typename Lane, size_t N>
struct Simd {
  constexpr Simd() = default;
  using T = Lane;
  static_assert((N & (N - 1)) == 0 && N != 0, "N must be a power of two");

  // Widening/narrowing ops change the number of lanes and/or their type.
  // To initialize such vectors, we need the corresponding descriptor types:

  // PromoteTo/DemoteTo() with another lane type, but same number of lanes.
  template <typename NewLane>
  using Rebind = Simd<NewLane, N>;

  // MulEven() with another lane type, but same total size.
  // Round up to correctly handle scalars with N=1.
  template <typename NewLane>
  using Repartition =
      Simd<NewLane, (N * sizeof(Lane) + sizeof(NewLane) - 1) / sizeof(NewLane)>;

  // LowerHalf() with the same lane type, but half the lanes.
  // Round up to correctly handle scalars with N=1.
  using Half = Simd<T, (N + 1) / 2>;

  // Combine() with the same lane type, but twice the lanes.
  using Twice = Simd<T, 2 * N>;
};

template <class D>
using TFromD = typename D::T;

// Descriptor for the same number of lanes as D, but with the LaneType T.
template <class T, class D>
using Rebind = typename D::template Rebind<T>;

template <class D>
using RebindToSigned = Rebind<MakeSigned<typename D::T>, D>;
template <class D>
using RebindToUnsigned = Rebind<MakeUnsigned<typename D::T>, D>;
template <class D>
using RebindToFloat = Rebind<MakeFloat<typename D::T>, D>;

// Descriptor for the same total size as D, but with the LaneType T.
template <class T, class D>
using Repartition = typename D::template Repartition<T>;

template <class D>
using RepartitionToWide = Repartition<MakeWide<typename D::T>, D>;
template <class D>
using RepartitionToNarrow = Repartition<MakeNarrow<typename D::T>, D>;

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

// Targets with non-constexpr Lanes define this themselves.
#if HWY_TARGET != HWY_RVV

// (Potentially) non-constant actual size of the vector at runtime, subject to
// the limit imposed by the Simd. Useful for advancing loop counters.
template <typename T, size_t N>
HWY_INLINE HWY_MAYBE_UNUSED size_t Lanes(Simd<T, N>) {
  return N;
}

#endif

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
