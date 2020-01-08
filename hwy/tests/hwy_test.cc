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

#ifndef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/hwy_test.cc"

#include "hwy/tests/test_util.h"
struct HwyTest {
  HWY_DECLARE(void, ())
};
TEST(HwyTest, Run) { hwy::RunTests<HwyTest>(); }

#endif  // HWY_TARGET_INCLUDE
#include "hwy/tests/test_target_util.h"

namespace hwy {
namespace HWY_NAMESPACE {
namespace {

constexpr HWY_FULL(uint8_t) du8;
constexpr HWY_FULL(uint16_t) du16;
constexpr HWY_FULL(uint32_t) du32;
constexpr HWY_FULL(uint64_t) du64;
constexpr HWY_FULL(int8_t) di8;
constexpr HWY_FULL(int16_t) di16;
constexpr HWY_FULL(int32_t) di32;
constexpr HWY_FULL(int64_t) di64;
constexpr HWY_FULL(float) df;
constexpr HWY_FULL(double) dd;

namespace examples {

namespace {
HWY_NOINLINE HWY_ATTR void FloorLog2(const uint8_t* HWY_RESTRICT values,
                                     uint8_t* HWY_RESTRICT log2) {
  // Descriptors for all required data types:
  const HWY_FULL(int32_t) d32;
  const HWY_FULL(float) df;
  const HWY_CAPPED(uint8_t, d32.N) d8;

  const auto u8 = Load(d8, values);
  const auto bits = BitCast(d32, ConvertTo(df, ConvertTo(d32, u8)));
  const auto exponent = ShiftRight<23>(bits) - Set(d32, 127);
  Store(ConvertTo(d8, exponent), d8, log2);
}
}  // namespace

HWY_NOINLINE HWY_ATTR void TestFloorLog2() {
  const size_t kStep = HWY_FULL(int32_t)::N;
  const size_t kBytes = 32;
  static_assert(kBytes % kStep == 0, "Must be a multiple of kStep");

  uint8_t in[kBytes];
  uint8_t expected[kBytes];
  RandomState rng{1234};
  for (size_t i = 0; i < kBytes; ++i) {
    expected[i] = Random32(&rng) & 7;
    in[i] = 1u << expected[i];
  }
  uint8_t out[32];
  for (size_t i = 0; i < kBytes; i += kStep) {
    FloorLog2(in + i, out + i);
  }
  int sum = 0;
  for (size_t i = 0; i < kBytes; ++i) {
    HWY_ASSERT_EQ(expected[i], out[i]);
    sum += out[i];
  }
  PreventElision(sum);
}

HWY_NOINLINE HWY_ATTR void Copy(const uint8_t* HWY_RESTRICT from,
                                const size_t size, uint8_t* HWY_RESTRICT to) {
  // Width-agnostic (library-specified N)
  const HWY_FULL(uint8_t) d;
  const Scalar<uint8_t> ds;
  size_t i = 0;
  for (; i + d.N <= size; i += d.N) {
    const auto bytes = Load(d, from + i);
    Store(bytes, d, to + i);
  }

  for (; i < size; ++i) {
    // (Same loop body as above, could factor into a shared template)
    const auto bytes = Load(ds, from + i);
    Store(bytes, ds, to + i);
  }
}

HWY_NOINLINE HWY_ATTR void TestCopy() {
  RandomState rng{1234};
  const size_t kSize = 34;
  HWY_ALIGN uint8_t from[kSize];
  for (unsigned char& i : from) {
    i = Random32(&rng) & 0xFF;
  }
  HWY_ALIGN uint8_t to[kSize];
  Copy(from, kSize, to);
  for (size_t i = 0; i < kSize; ++i) {
    HWY_ASSERT_EQ(from[i], to[i]);
  }
}

template <typename T>
HWY_NOINLINE HWY_ATTR void MulAdd(const T* HWY_RESTRICT mul_array,
                                  const T* HWY_RESTRICT add_array,
                                  const size_t size, T* HWY_RESTRICT x_array) {
  // Type-agnostic (caller-specified lane type) and width-agnostic (uses
  // best available instruction set).
  const HWY_FULL(T) d;
  for (size_t i = 0; i < size; i += d.N) {
    const auto mul = Load(d, mul_array + i);
    const auto add = Load(d, add_array + i);
    auto x = Load(d, x_array + i);
    x = MulAdd(mul, x, add);
    Store(x, d, x_array + i);
  }
}

template <typename T>
HWY_NOINLINE HWY_ATTR T SumMulAdd() {
  RandomState rng{1234};
  const size_t kSize = 64;
  HWY_ALIGN T mul[kSize];
  HWY_ALIGN T x[kSize];
  HWY_ALIGN T add[kSize];
  for (size_t i = 0; i < kSize; ++i) {
    mul[i] = Random32(&rng) & 0xF;
    x[i] = Random32(&rng) & 0xFF;
    add[i] = Random32(&rng) & 0xFF;
  }
  MulAdd(mul, add, kSize, x);
  double sum = 0.0;
  for (auto xi : x) {
    sum += xi;
  }
  return sum;
}

HWY_NOINLINE HWY_ATTR void TestExamples() {
  TestFloorLog2();
  TestCopy();

  HWY_ASSERT_EQ(78944.0f, SumMulAdd<float>());
#if HWY_HAS_DOUBLE
  HWY_ASSERT_EQ(78944.0, SumMulAdd<double>());
#endif
}

}  // namespace examples

namespace basic {

// util.h

HWY_NOINLINE HWY_ATTR void TestLimits() {
  HWY_ASSERT_EQ(uint8_t(0), LimitsMin<uint8_t>());
  HWY_ASSERT_EQ(uint16_t(0), LimitsMin<uint16_t>());
  HWY_ASSERT_EQ(uint32_t(0), LimitsMin<uint32_t>());
  HWY_ASSERT_EQ(uint64_t(0), LimitsMin<uint64_t>());

  HWY_ASSERT_EQ(int8_t(-128), LimitsMin<int8_t>());
  HWY_ASSERT_EQ(int16_t(-32768), LimitsMin<int16_t>());
  HWY_ASSERT_EQ(int32_t(0x80000000u), LimitsMin<int32_t>());
  HWY_ASSERT_EQ(int64_t(0x8000000000000000ull), LimitsMin<int64_t>());

  HWY_ASSERT_EQ(uint8_t(0xFF), LimitsMax<uint8_t>());
  HWY_ASSERT_EQ(uint16_t(0xFFFF), LimitsMax<uint16_t>());
  HWY_ASSERT_EQ(uint32_t(0xFFFFFFFFu), LimitsMax<uint32_t>());
  HWY_ASSERT_EQ(uint64_t(0xFFFFFFFFFFFFFFFFull), LimitsMax<uint64_t>());

  HWY_ASSERT_EQ(int8_t(0x7F), LimitsMax<int8_t>());
  HWY_ASSERT_EQ(int16_t(0x7FFF), LimitsMax<int16_t>());
  HWY_ASSERT_EQ(int32_t(0x7FFFFFFFu), LimitsMax<int32_t>());
  HWY_ASSERT_EQ(int64_t(0x7FFFFFFFFFFFFFFFull), LimitsMax<int64_t>());
}

// Test the ToString used to output test failures

HWY_NOINLINE HWY_ATTR void TestToString() {
  char buf[32];
  const char* end;

  end = ToString(int64_t(0), buf);
  HWY_ASSERT_EQ('0', end[-1]);
  HWY_ASSERT_EQ('\0', end[0]);

  end = ToString(int64_t(3), buf);
  HWY_ASSERT_EQ('3', end[-1]);
  HWY_ASSERT_EQ('\0', end[0]);

  end = ToString(int64_t(-1), buf);
  HWY_ASSERT_EQ('-', end[-2]);
  HWY_ASSERT_EQ('1', end[-1]);
  HWY_ASSERT_EQ('\0', end[0]);

  ToString(0x7FFFFFFFFFFFFFFFLL, buf);
  HWY_ASSERT_EQ(true, StringsEqual("9223372036854775807", buf));

  ToString(int64_t(0x8000000000000000ULL), buf);
  HWY_ASSERT_EQ(true, StringsEqual("-9223372036854775808", buf));

  ToString(0.0, buf);
  HWY_ASSERT_EQ(true, StringsEqual("0.0", buf));
  ToString(4.0, buf);
  HWY_ASSERT_EQ(true, StringsEqual("4.0", buf));
  ToString(-1.0, buf);
  HWY_ASSERT_EQ(true, StringsEqual("-1.0", buf));
  ToString(-1.25, buf);
  HWY_ASSERT_STRING_EQ("-1.2500000000000000", const_cast<const char*>(buf));
  ToString(2.125f, buf);
  HWY_ASSERT_STRING_EQ("2.12500000", const_cast<const char*>(buf));
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestIsUnsigned(D /*d*/) {
  using T = typename D::T;
  static_assert(!IsFloat<T>(), "Expected !IsFloat");
  static_assert(!IsSigned<T>(), "Expected !IsSigned");
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestIsSigned(D /*d*/) {
  using T = typename D::T;
  static_assert(!IsFloat<T>(), "Expected !IsFloat");
  static_assert(IsSigned<T>(), "Expected IsSigned");
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestIsFloat(D /*d*/) {
  using T = typename D::T;
  static_assert(IsFloat<T>(), "Expected IsFloat");
  static_assert(IsSigned<T>(), "Floats are also considered signed");
}

HWY_NOINLINE HWY_ATTR void TestType() {
  HWY_FOREACH_U(TestIsUnsigned);
  HWY_FOREACH_I(TestIsSigned);
  HWY_FOREACH_F(TestIsFloat);
}

// Ensures wraparound (mod 2^bits)
template <class D>
HWY_NOINLINE HWY_ATTR void TestOverflowT(D d) {
  using T = typename D::T;
  const auto v1 = Set(d, T(1));
  const auto vmax = Set(d, LimitsMax<T>());
  const auto vmin = Set(d, LimitsMin<T>());
  // Unsigned underflow / negative -> positive
  HWY_ASSERT_VEC_EQ(d, vmax, vmin - v1);
  // Unsigned overflow / positive -> negative
  HWY_ASSERT_VEC_EQ(d, vmin, vmax + v1);
}

HWY_NOINLINE HWY_ATTR void TestOverflow() {
  HWY_FOREACH_U(TestOverflowT);
  HWY_FOREACH_I(TestOverflowT);
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestName(D d) {
  using T = typename D::T;
  char expected[7] = {IsFloat<T>() ? 'f' : (IsSigned<T>() ? 'i' : 'u')};
  char* end = ToString(sizeof(T) * 8, expected + 1);
  if (D::N != 1) {
    *end++ = 'x';
    end = ToString(d.N, end);
  }
  if (!StringsEqual(expected, TypeName<T, D::N>())) {
    NotifyFailure(__FILE__, __LINE__, HWY_BITS, expected, -1, expected,
                  TypeName<T, D::N>());
  }
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestSet(D d) {
  using T = typename D::T;

  // Zero
  const auto v0 = Zero(d);
  HWY_ALIGN T expected[d.N] = {};  // zero-initialized.
  HWY_ASSERT_VEC_EQ(d, expected, v0);

  // Set
  const auto v2 = Set(d, T(2));
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = 2;
  }
  HWY_ASSERT_VEC_EQ(d, expected, v2);

  // iota
  const auto vi = Iota(d, T(5));
  for (size_t i = 0; i < d.N; ++i) {
    expected[i] = 5 + i;
  }
  HWY_ASSERT_VEC_EQ(d, expected, vi);

  // undefined
  const auto vu = Undefined(d);
  Store(vu, d, expected);
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestCopyAndAssign(D d) {
  using V = VT<D>;

  // copy V
  const auto v3 = Iota(d, 3);
  V v3b(v3);
  HWY_ASSERT_VEC_EQ(d, v3, v3b);

  // assign V
  V v3c;
  v3c = v3;
  HWY_ASSERT_VEC_EQ(d, v3, v3c);
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestHalf(D d) {
#if HWY_BITS != 0 || HWY_IDE
  using T = typename D::T;
  size_t i;
  constexpr size_t N2 = (d.N + 1) / 2;
  const Desc<T, N2> d2;

  const auto v = Iota(d, 1);
  HWY_ALIGN T lanes[d.N] = {0};

  Store(LowerHalf(v), d2, lanes);
  i = 0;
  for (; i < N2; ++i) {
    HWY_ASSERT_EQ(T(1 + i), lanes[i]);
  }
  // Other half remains unchanged
  for (; i < d.N; ++i) {
    HWY_ASSERT_EQ(T(0), lanes[i]);
  }
  Store(LowerHalf(v), d2, lanes);  // Also test the wrapper
  i = 0;
  for (; i < N2; ++i) {
    HWY_ASSERT_EQ(T(1 + i), lanes[i]);
  }
  // Other half remains unchanged
  for (; i < d.N; ++i) {
    HWY_ASSERT_EQ(T(0), lanes[i]);
  }

  Store(UpperHalf(v), d2, lanes);
  i = 0;
  for (; i < N2; ++i) {
    HWY_ASSERT_EQ(T(N2 + 1 + i), lanes[i]);
  }
  // Other half remains unchanged
  for (; i < d.N; ++i) {
    HWY_ASSERT_EQ(T(0), lanes[i]);
  }
  Store(UpperHalf(v), d2, lanes);  // Also test the wrapper
  i = 0;
  for (; i < N2; ++i) {
    HWY_ASSERT_EQ(T(N2 + 1 + i), lanes[i]);
  }
  // Other half remains unchanged
  for (; i < d.N; ++i) {
    HWY_ASSERT_EQ(T(0), lanes[i]);
  }

  // Ensure lanes are contiguous
  const auto vi = Iota(d2, 1);
  Store(vi, d2, lanes);
  for (size_t i = 1; i < N2; ++i) {
    HWY_ASSERT_EQ(T(lanes[i - 1] + 1), lanes[i]);
  }
#else
  (void)d;
#endif
}

template <class D>
HWY_NOINLINE HWY_ATTR void TestQuarterT(D d) {
#if HWY_BITS != 0 || HWY_IDE
  using T = typename D::T;
  constexpr size_t N4 = (d.N + 3) / 4;
  const HWY_CAPPED(T, N4) d4;

  const auto v = Iota(d, 1);
  HWY_ALIGN T lanes[d.N];
  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = 123;
  }
  const auto lo = LowerHalf(LowerHalf(v));
  Store(lo, d4, lanes);
  size_t i = 0;
  for (; i < N4; ++i) {
    HWY_ASSERT_EQ(T(i + 1), lanes[i]);
  }
  // Other lanes remain unchanged
  for (; i < d.N; ++i) {
    HWY_ASSERT_EQ(T(123), lanes[i]);
  }
#else
  (void)d;
#endif
}

HWY_NOINLINE HWY_ATTR void TestQuarter() {
  TestQuarterT(du8);
  TestQuarterT(du16);
  TestQuarterT(du32);
  TestQuarterT(di8);
  TestQuarterT(di16);
  TestQuarterT(di32);
  TestQuarterT(df);
}

HWY_NOINLINE HWY_ATTR void TestBasic() {
  (void)dd;
  (void)di64;
  (void)du64;

  TestLimits();
  TestToString();
  TestType();
  HWY_FOREACH_UIF(TestName);
  TestOverflow();
  HWY_FOREACH_UIF(TestSet);
  HWY_FOREACH_UIF(TestCopyAndAssign);
  HWY_FOREACH_UIF(TestHalf);
  TestQuarter();
}

}  // namespace basic

HWY_NOINLINE HWY_ATTR void TestHwy() {
  examples::TestExamples();
  basic::TestBasic();
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy

// Instantiate for the current target.
void HwyTest::HWY_FUNC() { hwy::HWY_NAMESPACE::TestHwy(); }
