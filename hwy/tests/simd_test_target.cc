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

#include "highway/simd_test.h"
#include "highway/test_target_util.h"

namespace jxl {
namespace SIMD_NAMESPACE {
namespace {

namespace examples {

namespace {
SIMD_ATTR void FloorLog2(const uint8_t* SIMD_RESTRICT values,
                         uint8_t* SIMD_RESTRICT log2) {
  // Descriptors for all required data types:
  const SIMD_FULL(int32_t) d32;
  const SIMD_FULL(float) df;
  const SIMD_CAPPED(uint8_t, d32.N) d8;

  const auto u8 = load(d8, values);
  const auto bits = bit_cast(d32, convert_to(df, convert_to(d32, u8)));
  const auto exponent = shift_right<23>(bits) - set1(d32, 127);
  store(convert_to(d8, exponent), d8, log2);
}
}  // namespace

SIMD_ATTR void TestFloorLog2() {
  const size_t kStep = SIMD_FULL(int32_t)::N;
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
    SIMD_ASSERT_EQ(expected[i], out[i]);
    sum += out[i];
  }
  PreventElision(sum);
}

SIMD_ATTR void Copy(const uint8_t* SIMD_RESTRICT from, const size_t size,
                    uint8_t* SIMD_RESTRICT to) {
  // Width-agnostic (library-specified N)
  const SIMD_FULL(uint8_t) d;
  const Scalar<uint8_t> ds;
  size_t i = 0;
  for (; i + d.N <= size; i += d.N) {
    const auto bytes = load(d, from + i);
    store(bytes, d, to + i);
  }

  for (; i < size; ++i) {
    // (Same loop body as above, could factor into a shared template)
    const auto bytes = load(ds, from + i);
    store(bytes, ds, to + i);
  }
}

SIMD_ATTR void TestCopy() {
  RandomState rng{1234};
  const size_t kSize = 34;
  SIMD_ALIGN uint8_t from[kSize];
  for (size_t i = 0; i < kSize; ++i) {
    from[i] = Random32(&rng) & 0xFF;
  }
  SIMD_ALIGN uint8_t to[kSize];
  Copy(from, kSize, to);
  for (size_t i = 0; i < kSize; ++i) {
    SIMD_ASSERT_EQ(from[i], to[i]);
  }
}

template <typename T>
SIMD_ATTR void MulAdd(const T* SIMD_RESTRICT mul_array,
                      const T* SIMD_RESTRICT add_array, const size_t size,
                      T* SIMD_RESTRICT x_array) {
  // Type-agnostic (caller-specified lane type) and width-agnostic (uses
  // best available instruction set).
  const SIMD_FULL(T) d;
  for (size_t i = 0; i < size; i += d.N) {
    const auto mul = load(d, mul_array + i);
    const auto add = load(d, add_array + i);
    auto x = load(d, x_array + i);
    x = mul_add(mul, x, add);
    store(x, d, x_array + i);
  }
}

template <typename T>
SIMD_ATTR T SumMulAdd() {
  RandomState rng{1234};
  const size_t kSize = 64;
  SIMD_ALIGN T mul[kSize];
  SIMD_ALIGN T x[kSize];
  SIMD_ALIGN T add[kSize];
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

SIMD_ATTR void TestExamples() {
  TestFloorLog2();
  TestCopy();

  SIMD_ASSERT_EQ(78944.0f, SumMulAdd<float>());
  SIMD_ASSERT_EQ(78944.0, SumMulAdd<double>());
}

}  // namespace examples

namespace basic {

// util.h

SIMD_ATTR void TestLimits() {
  SIMD_ASSERT_EQ(uint8_t(0), LimitsMin<uint8_t>());
  SIMD_ASSERT_EQ(uint16_t(0), LimitsMin<uint16_t>());
  SIMD_ASSERT_EQ(uint32_t(0), LimitsMin<uint32_t>());
  SIMD_ASSERT_EQ(uint64_t(0), LimitsMin<uint64_t>());

  SIMD_ASSERT_EQ(int8_t(-128), LimitsMin<int8_t>());
  SIMD_ASSERT_EQ(int16_t(-32768), LimitsMin<int16_t>());
  SIMD_ASSERT_EQ(int32_t(0x80000000u), LimitsMin<int32_t>());
  SIMD_ASSERT_EQ(int64_t(0x8000000000000000ull), LimitsMin<int64_t>());

  SIMD_ASSERT_EQ(uint8_t(0xFF), LimitsMax<uint8_t>());
  SIMD_ASSERT_EQ(uint16_t(0xFFFF), LimitsMax<uint16_t>());
  SIMD_ASSERT_EQ(uint32_t(0xFFFFFFFFu), LimitsMax<uint32_t>());
  SIMD_ASSERT_EQ(uint64_t(0xFFFFFFFFFFFFFFFFull), LimitsMax<uint64_t>());

  SIMD_ASSERT_EQ(int8_t(0x7F), LimitsMax<int8_t>());
  SIMD_ASSERT_EQ(int16_t(0x7FFF), LimitsMax<int16_t>());
  SIMD_ASSERT_EQ(int32_t(0x7FFFFFFFu), LimitsMax<int32_t>());
  SIMD_ASSERT_EQ(int64_t(0x7FFFFFFFFFFFFFFFull), LimitsMax<int64_t>());
}

// Test the ToString used to output test failures

SIMD_ATTR void TestToString() {
  char buf[32];
  const char* end;

  end = ToString(int64_t(0), buf);
  SIMD_ASSERT_EQ('0', end[-1]);
  SIMD_ASSERT_EQ('\0', end[0]);

  end = ToString(int64_t(3), buf);
  SIMD_ASSERT_EQ('3', end[-1]);
  SIMD_ASSERT_EQ('\0', end[0]);

  end = ToString(int64_t(-1), buf);
  SIMD_ASSERT_EQ('-', end[-2]);
  SIMD_ASSERT_EQ('1', end[-1]);
  SIMD_ASSERT_EQ('\0', end[0]);

  ToString(0x7FFFFFFFFFFFFFFFLL, buf);
  SIMD_ASSERT_EQ(true, StringsEqual("9223372036854775807", buf));

  ToString(int64_t(0x8000000000000000ULL), buf);
  SIMD_ASSERT_EQ(true, StringsEqual("-9223372036854775808", buf));

  ToString(0.0, buf);
  SIMD_ASSERT_EQ(true, StringsEqual("0.0", buf));
  ToString(4.0, buf);
  SIMD_ASSERT_EQ(true, StringsEqual("4.0", buf));
  ToString(-1.0, buf);
  SIMD_ASSERT_EQ(true, StringsEqual("-1.0", buf));
  ToString(-1.25, buf);
  SIMD_ASSERT_STRING_EQ("-1.2500000000000000", const_cast<const char*>(buf));
  ToString(2.125f, buf);
  SIMD_ASSERT_STRING_EQ("2.12500000", const_cast<const char*>(buf));
}

struct TestIsUnsigned {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    static_assert(!IsFloat<T>(), "Expected !IsFloat");
    static_assert(!IsSigned<T>(), "Expected !IsSigned");
  }
};

struct TestIsSigned {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    static_assert(!IsFloat<T>(), "Expected !IsFloat");
    static_assert(IsSigned<T>(), "Expected IsSigned");
  }
};

struct TestIsFloat {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    static_assert(IsFloat<T>(), "Expected IsFloat");
    static_assert(IsSigned<T>(), "Floats are also considered signed");
  }
};

SIMD_ATTR void TestType() {
  ForeachUnsignedLaneType<TestIsUnsigned>();
  ForeachSignedLaneType<TestIsSigned>();
  ForeachFloatLaneType<TestIsFloat>();
}

// Ensures wraparound (mod 2^bits)
struct TestOverflowT {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v1 = set1(d, T(1));
    const auto vmax = set1(d, LimitsMax<T>());
    const auto vmin = set1(d, LimitsMin<T>());
    // Unsigned underflow / negative -> positive
    SIMD_ASSERT_VEC_EQ(d, vmax, vmin - v1);
    // Unsigned overflow / positive -> negative
    SIMD_ASSERT_VEC_EQ(d, vmin, vmax + v1);
  }
};

SIMD_ATTR void TestOverflow() {
  ForeachUnsignedLaneType<TestOverflowT>();
  ForeachSignedLaneType<TestOverflowT>();
}

struct TestName {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    char expected[7] = {IsFloat<T>() ? 'f' : (IsSigned<T>() ? 'i' : 'u')};
    char* end = ToString(sizeof(T) * 8, expected + 1);
    if (D::N != 1) {
      *end++ = 'x';
      end = ToString(d.N, end);
    }
    if (!StringsEqual(expected, type_name<T, D::N>())) {
      NotifyFailure(__FILE__, __LINE__, SIMD_BITS, expected, -1, expected,
                    type_name<T, D::N>());
    }
  }
};

struct TestSet {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    // setzero
    const auto v0 = setzero(d);
    SIMD_ALIGN T expected[d.N] = {};  // zero-initialized.
    SIMD_ASSERT_VEC_EQ(d, expected, v0);

    // set1
    const auto v2 = set1(d, T(2));
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = 2;
    }
    SIMD_ASSERT_VEC_EQ(d, expected, v2);

    // iota
    const auto vi = iota(d, T(5));
    for (size_t i = 0; i < d.N; ++i) {
      expected[i] = 5 + i;
    }
    SIMD_ASSERT_VEC_EQ(d, expected, vi);

    // undefined
    const auto vu = undefined(d);
    store(vu, d, expected);
  }
};

struct TestCopyAndAssign {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    using V = VT<D>;

    // copy V
    const auto v3 = iota(d, 3);
    V v3b(v3);
    SIMD_ASSERT_VEC_EQ(d, v3, v3b);

    // assign V
    V v3c;
    v3c = v3;
    SIMD_ASSERT_VEC_EQ(d, v3, v3c);
  }
};

struct TestHalf {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
#if SIMD_BITS != 0
    size_t i;
    constexpr size_t N2 = (d.N + 1) / 2;
    const Desc<T, N2> d2;

    const auto v = iota(d, 1);
    SIMD_ALIGN T lanes[d.N] = {0};

    store(lower_half(v), d2, lanes);
    i = 0;
    for (; i < N2; ++i) {
      ASSERT_EQ(T(1 + i), lanes[i]);
    }
    // Other half remains unchanged
    for (; i < d.N; ++i) {
      ASSERT_EQ(T(0), lanes[i]);
    }
    store(lower_half(v), d2, lanes);  // Also test the wrapper
    i = 0;
    for (; i < N2; ++i) {
      ASSERT_EQ(T(1 + i), lanes[i]);
    }
    // Other half remains unchanged
    for (; i < d.N; ++i) {
      ASSERT_EQ(T(0), lanes[i]);
    }

    store(upper_half(v), d2, lanes);
    i = 0;
    for (; i < N2; ++i) {
      ASSERT_EQ(T(N2 + 1 + i), lanes[i]);
    }
    // Other half remains unchanged
    for (; i < d.N; ++i) {
      ASSERT_EQ(T(0), lanes[i]);
    }
    store(upper_half(v), d2, lanes);  // Also test the wrapper
    i = 0;
    for (; i < N2; ++i) {
      ASSERT_EQ(T(N2 + 1 + i), lanes[i]);
    }
    // Other half remains unchanged
    for (; i < d.N; ++i) {
      ASSERT_EQ(T(0), lanes[i]);
    }

    store(any_part(d2, v), d2, lanes);
    i = 0;
    for (; i < N2; ++i) {
      ASSERT_EQ(T(1 + i), lanes[i]);
    }
    // Other half remains unchanged
    for (; i < d.N; ++i) {
      ASSERT_EQ(T(0), lanes[i]);
    }

    // Ensure part lanes are contiguous
    const auto vi = iota(d2, 1);
    store(vi, d2, lanes);
    for (size_t i = 1; i < N2; ++i) {
      ASSERT_EQ(T(lanes[i - 1] + 1), lanes[i]);
    }
#endif
  }
};

struct TestQuarterT {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    constexpr size_t N4 = (d.N + 3) / 4;
    const SIMD_CAPPED(T, N4) d4;

    const auto v = iota(d, 1);
    SIMD_ALIGN T lanes[d.N];
    for (size_t i = 0; i < d.N; ++i) {
      lanes[i] = 123;
    }
    const auto lo = any_part(d4, v);
    store(lo, d4, lanes);
    size_t i = 0;
    for (; i < N4; ++i) {
      SIMD_ASSERT_EQ(T(i + 1), lanes[i]);
    }
    // Other lanes remain unchanged
    for (; i < d.N; ++i) {
      SIMD_ASSERT_EQ(T(123), lanes[i]);
    }
  }
};

SIMD_ATTR void TestQuarter() {
  Call<TestQuarterT, uint8_t>();
  Call<TestQuarterT, uint16_t>();
  Call<TestQuarterT, uint32_t>();
  Call<TestQuarterT, int8_t>();
  Call<TestQuarterT, int16_t>();
  Call<TestQuarterT, int32_t>();
  Call<TestQuarterT, float>();
}

SIMD_ATTR void TestBasic() {
  TestLimits();
  TestToString();
  TestType();
  ForeachLaneType<TestName>();
  TestOverflow();
  ForeachLaneType<TestSet>();
  ForeachLaneType<TestCopyAndAssign>();
  ForeachLaneType<TestHalf>();
  TestQuarter();
}

}  // namespace basic

SIMD_ATTR SIMD_NOINLINE void TestSimd() {
  examples::TestExamples();
  basic::TestBasic();
}

}  // namespace
}  // namespace SIMD_NAMESPACE  // NOLINT(google-readability-namespace-comments)
}  // namespace jxl

// Instantiate for the current target.
template <>
void SimdTest::operator()<jxl::SIMD_TARGET>() {
  jxl::SIMD_NAMESPACE::TestSimd();
}
