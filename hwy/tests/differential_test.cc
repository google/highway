// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
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

// Differential test harness: each op in the registry below is called with the
// same inputs on every target compiled into this binary, including EMU128, and
// the results are compared bit for bit. This finds both a target disagreeing
// with the other targets and a target disagreeing with an independent plain C++
// scalar reference. Neither is visible to the per-op tests, because those only
// check that their own target matches their own expectations.
//
// To add an op, write a functor that calls it (with constraints matching the
// op's availability) and add a row to the registry. To add a lane type or lane
// count, extend HWY_DIFF_TYPES/HWY_DIFF_LANES. Combinations that a target does
// not implement are reported as skipped rather than failing.
//
// See https://github.com/google/highway/issues/3353.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <limits>
#include <type_traits>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/differential_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

// This file is re-included once per target, but the registry is the same for
// all of them, so define it only once.
#ifndef HWY_TESTS_DIFFERENTIAL_DEFINED_
#define HWY_TESTS_DIFFERENTIAL_DEFINED_

// ------------------------------ Registry -----------------------------------
// Rows are `X(Name, Functor, Domain)`. Functors are defined below (once per
// target); only the driver and the per-target runner below refer to them, so
// both see the same order and indices.
//
// Ternary ops take three vector operands; unary ops are in the second list and
// ignore the operands they do not use, so one runner serves both.
//
// MulAdd52Lo/Hi are not listed yet because they are only in a dependent PR;
// they use kLt52Domain, which is why the input domain is configurable here.
#define HWY_DIFF_TERNARY_OPS(X)                \
  X(Add, OpAdd, kAnyDomain)                    \
  X(Sub, OpSub, kAnyDomain)                    \
  X(Mul, OpMul, kAnyDomain)                    \
  X(Min, OpMin, kAnyDomain)                    \
  X(Max, OpMax, kAnyDomain)                    \
  X(And, OpAnd, kAnyDomain)                    \
  X(Or, OpOr, kAnyDomain)                      \
  X(Xor, OpXor, kAnyDomain)                    \
  X(AndNot, OpAndNot, kAnyDomain)              \
  X(MulHigh, OpMulHigh, kAnyDomain)            \
  X(SaturatedAdd, OpSaturatedAdd, kAnyDomain)  \
  X(SaturatedSub, OpSaturatedSub, kAnyDomain)  \
  X(AverageRound, OpAverageRound, kAnyDomain)

#define HWY_DIFF_UNARY_OPS(X) \
  X(Neg, OpNeg, kAnyDomain)   \
  X(Abs, OpAbs, kAnyDomain)   \
  X(Not, OpNot, kAnyDomain)

#define HWY_DIFF_ALL_OPS(X) \
  HWY_DIFF_TERNARY_OPS(X)   \
  HWY_DIFF_UNARY_OPS(X)

// Generated from the same list, so indices cannot drift.
enum DiffOpIndex {
#define X(Name, Functor, Domain) kOp##Name,
  HWY_DIFF_ALL_OPS(X)
#undef X
  kNumDiffOps
};

// How the inputs for an op may be generated. See MakeInput.
enum DiffDomain {
  kAnyDomain = 0,
  // MulAdd52* require the multiplicands to be < 2^52; the addend is not
  // constrained by the contract, so it is left as is.
  kLt52Domain = 1,
};

// Lane types: every integer size and both 32/64-bit floats.
#define HWY_DIFF_TYPES(X) \
  X(uint8_t, U8)          \
  X(uint16_t, U16)        \
  X(uint32_t, U32)        \
  X(uint64_t, U64)        \
  X(int8_t, I8)           \
  X(int16_t, I16)         \
  X(int32_t, I32)         \
  X(int64_t, I64)         \
  X(float, F32)           \
  X(double, F64)

// Generated from the same list, so indices cannot drift.
enum DiffTypeIndex {
#define X(T, Suffix) kType##Suffix,
  HWY_DIFF_TYPES(X)
#undef X
  kNumDiffTypes
};

// Lane counts. Powers of two sweep full and partial vectors; the odd counts
// (3, 5) cover non-power-of-two tails. Counts above HWY_LANES(T) are capped to
// the full vector (the same lane indices are still compared across targets).
#define HWY_DIFF_LANES(X) \
  X(1) X(2) X(3) X(4) X(5) X(8) X(16) X(32)

// Number of interesting inputs per case; 0 = structured boundary values,
// 1 = a different pairing of those, 2 = pseudo-random values.
constexpr size_t kNumPasses = 3;

#endif  // HWY_TESTS_DIFFERENTIAL_DEFINED_

// ------------------------------ Per-target ---------------------------------
// Everything below is compiled once per target; the driver in the HWY_ONCE
// section at the end of this file calls it via dynamic dispatch.

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

// ---------------------------- Op functors ----------------------------------
// All take three vectors so that a single runner serves unary and ternary ops.
// The constraints must mirror the op's availability: that is what makes
// unsupported combinations SFINAE-friendly in HasRun instead of a compile
// error. See RunOp.
struct OpAdd {
  template <class D>
  static HWY_INLINE Vec<D> Run(D /*d*/, Vec<D> a, Vec<D> b, Vec<D> /*c*/) {
    return Add(a, b);
  }
};

struct OpSub {
  template <class D>
  static HWY_INLINE Vec<D> Run(D /*d*/, Vec<D> a, Vec<D> b, Vec<D> /*c*/) {
    return Sub(a, b);
  }
};

struct OpMul {
  template <class D>
  static HWY_INLINE Vec<D> Run(D /*d*/, Vec<D> a, Vec<D> b, Vec<D> /*c*/) {
    return Mul(a, b);
  }
};

struct OpMin {
  template <class D, HWY_IF_NOT_FLOAT_D(D)>
  static HWY_INLINE Vec<D> Run(D /*d*/, Vec<D> a, Vec<D> b, Vec<D> /*c*/) {
    return Min(a, b);
  }
};

struct OpMax {
  template <class D, HWY_IF_NOT_FLOAT_D(D)>
  static HWY_INLINE Vec<D> Run(D /*d*/, Vec<D> a, Vec<D> b, Vec<D> /*c*/) {
    return Max(a, b);
  }
};

struct OpAnd {
  template <class D, HWY_IF_NOT_FLOAT_D(D)>
  static HWY_INLINE Vec<D> Run(D /*d*/, Vec<D> a, Vec<D> b, Vec<D> /*c*/) {
    return And(a, b);
  }
};

struct OpOr {
  template <class D, HWY_IF_NOT_FLOAT_D(D)>
  static HWY_INLINE Vec<D> Run(D /*d*/, Vec<D> a, Vec<D> b, Vec<D> /*c*/) {
    return Or(a, b);
  }
};

struct OpXor {
  template <class D, HWY_IF_NOT_FLOAT_D(D)>
  static HWY_INLINE Vec<D> Run(D /*d*/, Vec<D> a, Vec<D> b, Vec<D> /*c*/) {
    return Xor(a, b);
  }
};

struct OpAndNot {
  template <class D, HWY_IF_NOT_FLOAT_D(D)>
  static HWY_INLINE Vec<D> Run(D /*d*/, Vec<D> a, Vec<D> b, Vec<D> /*c*/) {
    return AndNot(a, b);
  }
};

struct OpMulHigh {
  template <class D, HWY_IF_NOT_FLOAT_D(D)>
  static HWY_INLINE Vec<D> Run(D /*d*/, Vec<D> a, Vec<D> b, Vec<D> /*c*/) {
    return MulHigh(a, b);
  }
};

// Only 8/16-bit lanes; see quick_reference.
struct OpSaturatedAdd {
  template <class D, HWY_IF_NOT_FLOAT_D(D), HWY_IF_T_SIZE_LE_D(D, 2)>
  static HWY_INLINE Vec<D> Run(D /*d*/, Vec<D> a, Vec<D> b, Vec<D> /*c*/) {
    return SaturatedAdd(a, b);
  }
};

struct OpSaturatedSub {
  template <class D, HWY_IF_NOT_FLOAT_D(D), HWY_IF_T_SIZE_LE_D(D, 2)>
  static HWY_INLINE Vec<D> Run(D /*d*/, Vec<D> a, Vec<D> b, Vec<D> /*c*/) {
    return SaturatedSub(a, b);
  }
};

// Unsigned only: for signed lanes, "a + b + 1" would have to be evaluated with
// a wider intermediate, which not all targets do.
struct OpAverageRound {
  template <class D, HWY_IF_UNSIGNED_D(D)>
  static HWY_INLINE Vec<D> Run(D /*d*/, Vec<D> a, Vec<D> b, Vec<D> /*c*/) {
    return AverageRound(a, b);
  }
};

struct OpNeg {
  template <class D, HWY_IF_NOT_UNSIGNED_D(D)>
  static HWY_INLINE Vec<D> Run(D /*d*/, Vec<D> a, Vec<D> /*b*/, Vec<D> /*c*/) {
    return Neg(a);
  }
};

struct OpAbs {
  template <class D, HWY_IF_NOT_UNSIGNED_D(D)>
  static HWY_INLINE Vec<D> Run(D /*d*/, Vec<D> a, Vec<D> /*b*/, Vec<D> /*c*/) {
    return Abs(a);
  }
};

struct OpNot {
  template <class D, HWY_IF_NOT_FLOAT_D(D)>
  static HWY_INLINE Vec<D> Run(D /*d*/, Vec<D> a, Vec<D> /*b*/, Vec<D> /*c*/) {
    return Not(a);
  }
};

// ------------------------------- Runner ------------------------------------
// Whether Op::Run exists for this (D, V). Ops that a target does not implement
// for a lane type are skipped instead of failing; see RunOpImpl.
template <class Op, class D, class V>
class HasRun {
  template <class O, class D2, class V2>
  static auto Test(int)
      -> decltype(static_cast<void>(O::Run(D2(), V2(), V2(), V2())),
                  std::true_type());
  template <class, class, class>
  static std::false_type Test(...);

 public:
  static constexpr bool value = decltype(Test<Op, D, V>(0))::value;
};

template <class Op, class T, size_t kLanes>
HWY_NOINLINE size_t RunOpImpl(std::true_type /*supported*/, const T* in_a,
                              const T* in_b, const T* in_c, T* out) {
  // Cap to the full vector: a larger count would not compile on this target.
  // The inputs are derived from the lane index, so the remaining lanes are
  // still compared against the same indices of the other targets.
  constexpr size_t kMax = kLanes < HWY_LANES(T) ? kLanes : HWY_LANES(T);
  using D = CappedTag<T, kMax>;
  const D d;
  const size_t num_lanes = Lanes(d);
  const Vec<D> result =
      Op::Run(d, LoadU(d, in_a), LoadU(d, in_b), LoadU(d, in_c));
  StoreU(result, d, out);
  return num_lanes;
}

template <class Op, class T, size_t kLanes>
HWY_NOINLINE size_t RunOpImpl(std::false_type /*unsupported*/, const T*,
                              const T*, const T*, T*) {
  return 0;
}

// Some targets have no vector type for a lane type: 32-bit Arm has no f64
// vectors, so Vec<CappedTag<double, ...>> cannot even be instantiated there.
// Such combinations are reported as not implemented, exactly like ops that a
// target lacks. The macro is per-target, so the bodies below differ per target
// while the case indices in the driver remain the same.
template <class T>
constexpr bool HaveVectorType() {
#if HWY_HAVE_FLOAT64
  return true;
#else
  return !IsSame<T, double>();
#endif
}

template <class Op, class T, size_t kLanes>
HWY_NOINLINE size_t RunOpTag(std::true_type /*have vector type*/,
                             const T* in_a, const T* in_b, const T* in_c,
                             T* out) {
  constexpr size_t kMax = kLanes < HWY_LANES(T) ? kLanes : HWY_LANES(T);
  using D = CappedTag<T, kMax>;
  using V = Vec<D>;
  return RunOpImpl<Op, T, kLanes>(
      std::integral_constant<bool, HasRun<Op, D, V>::value>(), in_a, in_b, in_c,
      out);
}

template <class Op, class T, size_t kLanes>
HWY_NOINLINE size_t RunOpTag(std::false_type /*no vector type*/, const T*,
                             const T*, const T*, T*) {
  return 0;
}

template <class Op, class T, size_t kLanes>
HWY_NOINLINE size_t RunOp(const T* in_a, const T* in_b, const T* in_c, T* out) {
  return RunOpTag<Op, T, kLanes>(
      std::integral_constant<bool, HaveVectorType<T>()>(), in_a, in_b, in_c,
      out);
}

template <class Op, class T>
HWY_NOINLINE size_t RunLanes(size_t lanes, const void* in_a, const void* in_b,
                             const void* in_c, void* out) {
  const T* a = static_cast<const T*>(in_a);
  const T* b = static_cast<const T*>(in_b);
  const T* c = static_cast<const T*>(in_c);
  T* o = static_cast<T*>(out);
  switch (lanes) {
#define X(N) \
  case N:    \
    return RunOp<Op, T, N>(a, b, c, o);
    HWY_DIFF_LANES(X)
#undef X
    default:
      return 0;
  }
}

template <class Op>
HWY_NOINLINE size_t RunType(size_t type_index, size_t lanes, const void* in_a,
                            const void* in_b, const void* in_c, void* out) {
  switch (type_index) {
#define X(T, Suffix) \
  case kType##Suffix: \
    return RunLanes<Op, T>(lanes, in_a, in_b, in_c, out);
    HWY_DIFF_TYPES(X)
#undef X
    default:
      return 0;
  }
}

// Called via dynamic dispatch, once per target. Returns the number of lanes
// written to `out`, or 0 if this target does not implement the case.
HWY_NOINLINE size_t RunCase(size_t op_index, size_t type_index, size_t lanes,
                            const void* in_a, const void* in_b,
                            const void* in_c, void* out) {
  switch (op_index) {
#define X(Name, Functor, Domain)                                       \
  case kOp##Name:                                                      \
    return RunType<Functor>(type_index, lanes, in_a, in_b, in_c, out);
    HWY_DIFF_ALL_OPS(X)
#undef X
    default:
      return 0;
  }
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {

HWY_EXPORT(RunCase);

// ------------------------------- Registry ----------------------------------
// Same order as the X-macros at the top, so indices cannot drift.
struct OpInfo {
  const char* name;
  DiffDomain domain;
};

const OpInfo kOps[] = {
#define X(Name, Functor, Domain) {#Name, Domain},
    HWY_DIFF_ALL_OPS(X)
#undef X
};

const char* const kTypeNames[] = {
#define X(T, Suffix) #T,
    HWY_DIFF_TYPES(X)
#undef X
};
static_assert(kNumDiffTypes == sizeof(kTypeNames) / sizeof(kTypeNames[0]), "");

// Each entry is also the lane count handed to RunCase.
const size_t kLaneCounts[] = {
#define X(N) N,
    HWY_DIFF_LANES(X)
#undef X
};
constexpr size_t kNumLaneCounts = sizeof(kLaneCounts) / sizeof(kLaneCounts[0]);
static_assert(kNumDiffOps == sizeof(kOps) / sizeof(kOps[0]), "");

// Number of lanes we generate inputs for. Must be >= every kLaneCounts entry;
// larger requests are capped by RunCase.
constexpr size_t kMaxLanes = 32;

// Unsigned integer of the same size, used to implement wrap-around arithmetic
// in the reference and bitwise comparison of float lanes.
template <size_t kBytes>
struct UIntOfSize;
template <>
struct UIntOfSize<1> {
  using type = uint8_t;
};
template <>
struct UIntOfSize<2> {
  using type = uint16_t;
};
template <>
struct UIntOfSize<4> {
  using type = uint32_t;
};
template <>
struct UIntOfSize<8> {
  using type = uint64_t;
};

template <class T>
HWY_INLINE typename UIntOfSize<sizeof(T)>::type LaneBits(T value) {
  return BitCastScalar<typename UIntOfSize<sizeof(T)>::type>(value);
}

// ------------------------------ Input corpus -------------------------------
constexpr size_t kMaxBoundaryValues = 24;

// Interesting boundary values for T; infinity/NaN/denormals degenerate to 0 for
// integer types, where duplicates are harmless. Returns the number of values.
template <class T>
size_t BoundaryValues(T* HWY_RESTRICT out) {
  size_t n = 0;
  const T one = static_cast<T>(1);
  const T two = static_cast<T>(2);
  const T max = std::numeric_limits<T>::max();
  const T lowest = std::numeric_limits<T>::lowest();
  out[n++] = static_cast<T>(0);
  out[n++] = one;
  out[n++] = static_cast<T>(0 - one);  // -1 (or max for unsigned)
  out[n++] = two;
  out[n++] = static_cast<T>(0 - two);
  out[n++] = max;
  out[n++] = lowest;
  out[n++] = static_cast<T>(max - one);
  out[n++] = static_cast<T>(lowest + one);
  // Highest bit only; then alternating bits, and values straddling 2^52, where
  // MulAdd52* changes behavior. All are built with uint64_t arithmetic to avoid
  // ULL literals and the conversions they imply for narrow lane types.
  const uint64_t k55 = ~uint64_t{0} / 3;       // 0101...
  const uint64_t kAA = ~uint64_t{0} - k55;     // 1010...
  out[n++] = static_cast<T>(uint64_t{1} << (sizeof(T) * 8 - 1));
  out[n++] = static_cast<T>(k55);
  out[n++] = static_cast<T>(kAA);
  out[n++] = static_cast<T>(uint64_t{1} << 51);
  out[n++] = static_cast<T>((uint64_t{1} << 52) - 1);
  out[n++] = static_cast<T>(uint64_t{1} << 52);
  // Floats: smallest normal, smallest subnormal, infinity, NaN.
  out[n++] = static_cast<T>(std::numeric_limits<T>::min());
  out[n++] = static_cast<T>(std::numeric_limits<T>::denorm_min());
  out[n++] = static_cast<T>(std::numeric_limits<T>::infinity());
  out[n++] = static_cast<T>(-std::numeric_limits<T>::infinity());
  out[n++] = static_cast<T>(std::numeric_limits<T>::quiet_NaN());
  HWY_ASSERT(n <= kMaxBoundaryValues);
  return n;
}

// Pseudo-random value: raw bits for integers, so that we cover extreme
// exponents (and NaN) for floats; a finite value of varied magnitude for floats
// because their extremes are already covered by the boundary values.
template <class T>
HWY_INLINE T RandomValue(hwy::RandomState* rng, std::false_type /*is_float*/) {
  return static_cast<T>(Random64(rng));
}

template <class T>
HWY_INLINE T RandomValue(hwy::RandomState* rng, std::true_type /*is_float*/) {
  return static_cast<T>(static_cast<int64_t>(Random64(rng)));
}

// Fills one operand. Operands start at different offsets so that we also see
// pairs such as (max, max) and (0, max), not only equal lanes.
template <class T>
void MakeInput(size_t num_lanes, size_t pass, size_t operand, DiffDomain domain,
               hwy::RandomState* rng, void* HWY_RESTRICT raw) {
  T* HWY_RESTRICT v = static_cast<T*>(raw);
  if (pass + 1 < kNumPasses) {
    T boundary[kMaxBoundaryValues];
    const size_t num_boundary = BoundaryValues(boundary);
    const size_t offsets[kNumPasses - 1][3] = {{0, 0, 1}, {0, 3, 7}};
    const size_t offset = offsets[pass][operand];
    for (size_t i = 0; i < num_lanes; ++i) {
      v[i] = boundary[(i + offset) % num_boundary];
    }
  } else {
    for (size_t i = 0; i < num_lanes; ++i) {
      v[i] = RandomValue<T>(rng,
                            std::integral_constant<bool, IsFloat<T>()>());
    }
  }
  if (domain == kLt52Domain && operand != 2) {
    // MulAdd52*: the multiplicands must be < 2^52. The addend is not
    // constrained by the contract, so it is left untouched. The mask is typed
    // uint64_t rather than written as an ULL literal, which avoids conversions
    // that -Wsign-conversion/-Wconversion reject on some targets (e.g. s390x).
    const uint64_t mask52 = (uint64_t{1} << 52) - 1;
    for (size_t i = 0; i < num_lanes; ++i) {
      v[i] = static_cast<T>(static_cast<uint64_t>(v[i]) & mask52);
    }
  }
}

// --------------------------- Scalar reference ------------------------------
// The reference is plain C++ over individual lanes, so it shares no code with
// any target. Integer arithmetic wraps; floats only use exact IEEE ops.
template <class T>
HWY_INLINE T RefAdd(T a, T b, std::false_type /*is_float*/) {
  using U = typename UIntOfSize<sizeof(T)>::type;
  return static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
}

template <class T>
HWY_INLINE T RefAdd(T a, T b, std::true_type /*is_float*/) {
  return static_cast<T>(a + b);
}

template <class T>
HWY_INLINE T RefSub(T a, T b, std::false_type /*is_float*/) {
  using U = typename UIntOfSize<sizeof(T)>::type;
  return static_cast<T>(static_cast<U>(a) - static_cast<U>(b));
}

template <class T>
HWY_INLINE T RefSub(T a, T b, std::true_type /*is_float*/) {
  return static_cast<T>(a - b);
}

template <class T>
HWY_INLINE T RefMul(T a, T b, std::false_type /*is_float*/) {
  using U = typename UIntOfSize<sizeof(T)>::type;
  return static_cast<T>(static_cast<U>(a) * static_cast<U>(b));
}

template <class T>
HWY_INLINE T RefMul(T a, T b, std::true_type /*is_float*/) {
  return static_cast<T>(a * b);
}

template <class T>
HWY_INLINE T RefNeg(T a, std::false_type /*is_float*/) {
  using U = typename UIntOfSize<sizeof(T)>::type;
  return static_cast<T>(U{0} - static_cast<U>(a));
}

template <class T>
HWY_INLINE T RefNeg(T a, std::true_type /*is_float*/) {
  return static_cast<T>(-a);
}

// For integers, LimitsMin() maps to LimitsMax() + 1, i.e. wraps around.
template <class T>
HWY_INLINE T RefAbs(T a, std::false_type /*is_float*/) {
  using U = typename UIntOfSize<sizeof(T)>::type;
  return a < T{0} ? static_cast<T>(U{0} - static_cast<U>(a)) : a;
}

template <class T>
HWY_INLINE T RefAbs(T a, std::true_type /*is_float*/) {
  return static_cast<T>(a < T{0} ? -a : a);
}

template <class T>
HWY_INLINE T RefSaturatedAdd(T a, T b, std::true_type /*is_signed*/) {
  const int64_t sum = static_cast<int64_t>(a) + static_cast<int64_t>(b);
  if (sum < static_cast<int64_t>(std::numeric_limits<T>::lowest()))
    return std::numeric_limits<T>::lowest();
  if (sum > static_cast<int64_t>(std::numeric_limits<T>::max()))
    return std::numeric_limits<T>::max();
  return static_cast<T>(sum);
}

template <class T>
HWY_INLINE T RefSaturatedAdd(T a, T b, std::false_type /*is_signed*/) {
  const uint64_t sum = static_cast<uint64_t>(a) + static_cast<uint64_t>(b);
  return sum > static_cast<uint64_t>(std::numeric_limits<T>::max())
             ? std::numeric_limits<T>::max()
             : static_cast<T>(sum);
}

template <class T>
HWY_INLINE T RefSaturatedSub(T a, T b, std::true_type /*is_signed*/) {
  const int64_t diff = static_cast<int64_t>(a) - static_cast<int64_t>(b);
  if (diff < static_cast<int64_t>(std::numeric_limits<T>::lowest()))
    return std::numeric_limits<T>::lowest();
  if (diff > static_cast<int64_t>(std::numeric_limits<T>::max()))
    return std::numeric_limits<T>::max();
  return static_cast<T>(diff);
}

template <class T>
HWY_INLINE T RefSaturatedSub(T a, T b, std::false_type /*is_signed*/) {
  const uint64_t diff = static_cast<uint64_t>(a) - static_cast<uint64_t>(b);
  return diff > static_cast<uint64_t>(std::numeric_limits<T>::max())
             ? std::numeric_limits<T>::lowest()
             : static_cast<T>(diff);
}

// (a + b + 1) >> 1, without overflowing the intermediate.
template <class T>
HWY_INLINE T RefAverageRound(T a, T b) {
  using U = typename UIntOfSize<sizeof(T)>::type;
  const U ua = static_cast<U>(a);
  const U ub = static_cast<U>(b);
  return static_cast<T>((ua >> 1) + (ub >> 1) + ((ua | ub) & U{1}));
}

// Upper half of a * b, written out so that it does not depend on the op being
// tested. Same recipe as the MulHigh test. MulHigh is integer-only, so the
// float overload is only there to keep RefCase's switch compilable; it is never
// called.
template <class T>
HWY_INLINE T RefMulHigh(T /*a*/, T /*b*/, std::true_type /*is_float*/) {
  return T{0};
}

template <class T, hwy::EnableIf<(sizeof(T) != 8)>* = nullptr>
HWY_INLINE T RefMulHigh(T a, T b, std::false_type /*is_float*/) {
  using W = hwy::MakeWide<T>;
  return static_cast<T>(static_cast<W>(a) * static_cast<W>(b) >>
                        (sizeof(T) * 8));
}

template <class T, hwy::EnableIf<(sizeof(T) == 8)>* = nullptr>
HWY_INLINE T RefMulHigh(T a, T b,
                        std::false_type /*is_float*/ /* sizeof(T) == 8 */) {
  T hi;
  Mul128(a, b, &hi);
  return hi;
}

template <class T>
HWY_INLINE T RefMulHigh(T a, T b) {
  return RefMulHigh(a, b, std::integral_constant<bool, IsFloat<T>()>());
}

// Fills `out` with the reference result. Returns false if there is no reference
// for this op (none currently); the caller then only compares targets against
// each other.
template <class T>
bool RefCase(size_t op_index, size_t num_lanes, const void* in_a,
             const void* in_b, const void* in_c, void* out) {
  using U = typename UIntOfSize<sizeof(T)>::type;
  const T* a = static_cast<const T*>(in_a);
  const T* b = static_cast<const T*>(in_b);
  // Only ternary ops use the addend. None of the currently registered ops do,
  // but keeping it in the interface makes adding e.g. MulAdd52* a one-liner.
  HWY_MAYBE_UNUSED const T* c = static_cast<const T*>(in_c);
  T* o = static_cast<T*>(out);
  constexpr bool kFloat = IsFloat<T>();
  constexpr bool kSigned = IsSigned<T>();
  for (size_t i = 0; i < num_lanes; ++i) {
    switch (op_index) {
      case kOpAdd:
        o[i] = RefAdd(a[i], b[i], std::integral_constant<bool, kFloat>());
        break;
      case kOpSub:
        o[i] = RefSub(a[i], b[i], std::integral_constant<bool, kFloat>());
        break;
      case kOpMul:
        o[i] = RefMul(a[i], b[i], std::integral_constant<bool, kFloat>());
        break;
      case kOpMin:
        o[i] = a[i] < b[i] ? a[i] : b[i];
        break;
      case kOpMax:
        o[i] = a[i] > b[i] ? a[i] : b[i];
        break;
      // The bitwise ops are integer-only; going through U keeps the switch
      // compilable (but never executed) for float lanes.
      case kOpAnd:
        o[i] = static_cast<T>(static_cast<U>(a[i]) & static_cast<U>(b[i]));
        break;
      case kOpOr:
        o[i] = static_cast<T>(static_cast<U>(a[i]) | static_cast<U>(b[i]));
        break;
      case kOpXor:
        o[i] = static_cast<T>(static_cast<U>(a[i]) ^ static_cast<U>(b[i]));
        break;
      case kOpAndNot:
        o[i] = static_cast<T>(~static_cast<U>(a[i]) & static_cast<U>(b[i]));
        break;
      case kOpMulHigh:
        o[i] = RefMulHigh(a[i], b[i]);
        break;
      case kOpSaturatedAdd:
        o[i] = RefSaturatedAdd(a[i], b[i],
                               std::integral_constant<bool, kSigned>());
        break;
      case kOpSaturatedSub:
        o[i] = RefSaturatedSub(a[i], b[i],
                               std::integral_constant<bool, kSigned>());
        break;
      case kOpAverageRound:
        o[i] = RefAverageRound(a[i], b[i]);
        break;
      case kOpNeg:
        o[i] = RefNeg(a[i], std::integral_constant<bool, kFloat>());
        break;
      case kOpAbs:
        o[i] = RefAbs(a[i], std::integral_constant<bool, kFloat>());
        break;
      case kOpNot:
        o[i] = static_cast<T>(~static_cast<U>(a[i]));
        break;
      default:
        return false;
    }
  }
  return true;
}

// ------------------------------ Comparison ---------------------------------
// Lanes are equal if their bits are identical, except that any two NaNs are
// considered equal because their payloads are unspecified.
template <class T>
HWY_INLINE bool LaneEqual(T expected, T actual, std::false_type /*is_float*/) {
  return LaneBits(expected) == LaneBits(actual);
}

template <class T>
HWY_INLINE bool LaneEqual(T expected, T actual, std::true_type /*is_float*/) {
  // Any two NaNs compare equal: their payloads are unspecified and targets
  // differ in which NaN they propagate.
  const bool expected_nan = expected != expected;
  const bool actual_nan = actual != actual;
  if (expected_nan || actual_nan) return expected_nan && actual_nan;
  return LaneBits(expected) == LaneBits(actual);
}

// Prints one lane as its bytes, which works for every lane type.
void PrintLane(const void* raw, size_t lane_size, size_t lane) {
  const uint8_t* bytes = static_cast<const uint8_t*>(raw) + lane * lane_size;
  for (size_t i = 0; i < lane_size; ++i) {
    fprintf(stderr, "%02x", static_cast<unsigned>(bytes[i]));
  }
}

// Prints the lanes around `lane`, i.e. the input, output and first differing
// lane index plus a few neighboring lanes.
void PrintWindow(const char* label, const void* raw, size_t lane_size,
                 size_t lane, size_t num_lanes) {
  const size_t first = lane > 3 ? lane - 3 : 0;
  const size_t last = lane + 3 < num_lanes ? lane + 3 : num_lanes - 1;
  fprintf(stderr, "      %s[%zu..%zu]:", label, first, last);
  for (size_t i = first; i <= last; ++i) {
    fprintf(stderr, " ");
    PrintLane(raw, lane_size, i);
  }
  fprintf(stderr, "\n");
}

// Everything needed to check one case: the args are the shared inputs, the
// per-target outputs and the reference output.
struct DiffCase {
  size_t op_index;
  size_t type_index;
  size_t lanes;                // as requested; a target may use fewer
  size_t pass;
  const std::vector<int64_t>* targets;
  const void* in[3];           // operands a, b, c
  const size_t* lanes_done;    // per target; 0 = not implemented there
  const void* const* outputs;  // per target
  void* ref_out;
};

// Compares `expected` with every target that computed all `common` lanes.
template <class T>
size_t CompareAllTargets(const DiffCase& c, const char* expected_name,
                         const void* expected, size_t common) {
  const std::vector<int64_t>& targets = *c.targets;
  const T* e = static_cast<const T*>(expected);
  size_t mismatches = 0;
  for (size_t t = 0; t < targets.size(); ++t) {
    if (c.lanes_done[t] < common) continue;
    const char* actual_name = TargetName(targets[t]);
    const void* actual = c.outputs[t];
    const T* v = static_cast<const T*>(actual);
    for (size_t i = 0; i < common; ++i) {
      if (LaneEqual(e[i], v[i], std::integral_constant<bool, IsFloat<T>()>())) {
        continue;
      }
      ++mismatches;
      fprintf(stderr,
              "Differential mismatch: %s/%s/%zu lanes/pass %zu: %s lane %zu of "
              "%zu\n",
              kOps[c.op_index].name, kTypeNames[c.type_index], c.lanes, c.pass,
              actual_name, i, common);
      PrintWindow("a", c.in[0], sizeof(T), i, common);
      PrintWindow("b", c.in[1], sizeof(T), i, common);
      PrintWindow("c", c.in[2], sizeof(T), i, common);
      PrintWindow(expected_name, expected, sizeof(T), i, common);
      PrintWindow(actual_name, actual, sizeof(T), i, common);
      break;  // One report per target is enough.
    }
  }
  return mismatches;
}

// Computes the scalar reference and compares all targets with it. If there is
// no reference for the op, only the targets are compared with each other
// (EMU128 is the baseline because it is a mandatory fallback target).
template <class T>
size_t CheckCaseTyped(const DiffCase& c) {
  const std::vector<int64_t>& targets = *c.targets;
  size_t common = kMaxLanes;
  for (size_t t = 0; t < targets.size(); ++t) {
    if (c.lanes_done[t] < common) common = c.lanes_done[t];
  }
  if (RefCase<T>(c.op_index, common, c.in[0], c.in[1], c.in[2], c.ref_out)) {
    return CompareAllTargets<T>(c, "reference", c.ref_out, common);
  }
  size_t baseline = 0;
  for (size_t t = 0; t < targets.size(); ++t) {
    if (targets[t] == HWY_EMU128) baseline = t;
  }
  if (c.lanes_done[baseline] < common) return 0;
  return CompareAllTargets<T>(c, TargetName(targets[baseline]),
                              c.outputs[baseline], common);
}

size_t CheckCase(const DiffCase& c) {
  switch (c.type_index) {
#define X(T, Suffix) \
  case kType##Suffix: \
    return CheckCaseTyped<T>(c);
    HWY_DIFF_TYPES(X)
#undef X
    default:
      return 0;
  }
}

// Generates the inputs of one case, identically for all targets.
void MakeCaseInputs(size_t type_index, size_t num_lanes, size_t pass,
                    DiffDomain domain, hwy::RandomState* rng, void* in_a,
                    void* in_b, void* in_c) {
  switch (type_index) {
#define X(T, Suffix)                                     \
  case kType##Suffix:                                    \
    MakeInput<T>(num_lanes, pass, 0, domain, rng, in_a);  \
    MakeInput<T>(num_lanes, pass, 1, domain, rng, in_b);  \
    MakeInput<T>(num_lanes, pass, 2, domain, rng, in_c);  \
    break;
    HWY_DIFF_TYPES(X)
#undef X
    default:
      break;
  }
}

// Runs every registered op for every lane type and lane count on every target
// compiled into this binary (and supported by the CPU), and compares the
// results bit for bit. Unlike the per-op tests, this test is not parametric on
// the target: it sets the supported-targets mask itself so that it can call the
// same inputs on every target within one process.
TEST(HwyDifferentialTest, AllOpsAgreeAcrossTargets) {
  const std::vector<int64_t> targets = SupportedAndGeneratedTargets();
  const size_t num_targets = targets.size();
  fprintf(stderr, "Differential: %zu target(s):", num_targets);
  for (size_t t = 0; t < num_targets; ++t) {
    fprintf(stderr, " %s", TargetName(targets[t]));
  }
  fprintf(stderr, "\n");

  // The inputs are shared; each target writes to its own output buffer.
  std::vector<uint64_t> in_all(3 * kMaxLanes);
  std::vector<uint64_t> out_all(num_targets * kMaxLanes);
  std::vector<uint64_t> ref_all(kMaxLanes);
  std::vector<const void*> outputs(num_targets);
  std::vector<size_t> lanes_done(num_targets);
  for (size_t t = 0; t < num_targets; ++t) {
    outputs[t] = &out_all[t * kMaxLanes];
  }

  size_t total_compared = 0;
  size_t total_skipped = 0;
  size_t total_mismatches = 0;
  for (size_t op_index = 0; op_index < kNumDiffOps; ++op_index) {
    size_t compared = 0;
    size_t skipped = 0;
    size_t op_mismatches = 0;
    for (size_t type_index = 0; type_index < kNumDiffTypes; ++type_index) {
      for (size_t i = 0; i < kNumLaneCounts; ++i) {
        const size_t lanes = kLaneCounts[i];
        for (size_t pass = 0; pass < kNumPasses; ++pass) {
          // The seed is fixed, so a failing case can be reproduced from its
          // (op, lane type, lane count, pass) indices.
          hwy::RandomState rng;
          MakeCaseInputs(type_index, kMaxLanes, pass, kOps[op_index].domain,
                         &rng, &in_all[0], &in_all[kMaxLanes],
                         &in_all[2 * kMaxLanes]);
          for (size_t t = 0; t < num_targets; ++t) {
            memset(&out_all[t * kMaxLanes], 0, kMaxLanes * sizeof(uint64_t));
            // Run this op on exactly one target.
            SetSupportedTargetsForTest(targets[t]);
            lanes_done[t] = HWY_DYNAMIC_DISPATCH(RunCase)(
                op_index, type_index, lanes, &in_all[0], &in_all[kMaxLanes],
                &in_all[2 * kMaxLanes], &out_all[t * kMaxLanes]);
          }
          SetSupportedTargetsForTest(0);
          // Skip combinations that some target does not implement, such as
          // SaturatedAdd for 32-bit lanes or MulAdd52 for non-u64 lanes. These
          // are counted and reported rather than silently ignored.
          bool all_implemented = true;
          for (size_t t = 0; t < num_targets; ++t) {
            if (lanes_done[t] == 0) all_implemented = false;
          }
          if (!all_implemented) {
            ++skipped;
            continue;
          }
          DiffCase c;
          c.op_index = op_index;
          c.type_index = type_index;
          c.lanes = lanes;
          c.pass = pass;
          c.targets = &targets;
          c.in[0] = &in_all[0];
          c.in[1] = &in_all[kMaxLanes];
          c.in[2] = &in_all[2 * kMaxLanes];
          c.lanes_done = lanes_done.data();
          c.outputs = outputs.data();
          c.ref_out = ref_all.data();
          op_mismatches += CheckCase(c);
          ++compared;
        }
      }
    }
    total_compared += compared;
    total_skipped += skipped;
    total_mismatches += op_mismatches;
    fprintf(stderr,
            "Differential: %-13s %5zu compared, %5zu skipped, %zu mismatches\n",
            kOps[op_index].name, compared, skipped, op_mismatches);
  }
  fprintf(stderr,
          "Differential: %zu cases compared, %zu skipped, %zu mismatches\n",
          total_compared, total_skipped, total_mismatches);
  HWY_ASSERT(total_mismatches == 0);
}

}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
