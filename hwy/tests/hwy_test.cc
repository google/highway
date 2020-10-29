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

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/hwy_test.cc"
#include "hwy/foreach_target.h"
// ^ must come before highway.h and any *-inl.h.

#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

template <class DF>
HWY_NOINLINE void FloorLog2(const DF df, const uint8_t* HWY_RESTRICT values,
                            uint8_t* HWY_RESTRICT log2) {
  // Descriptors for all required data types:
  const Simd<int32_t, MaxLanes(df)> d32;
  const Simd<uint8_t, MaxLanes(df)> d8;

  const auto u8 = Load(d8, values);
  const auto bits = BitCast(d32, ConvertTo(df, PromoteTo(d32, u8)));
  const auto exponent = ShiftRight<23>(bits) - Set(d32, 127);
  Store(DemoteTo(d8, exponent), d8, log2);
}

struct TestFloorLog2 {
  template <class T, class DF>
  HWY_NOINLINE void operator()(T /*unused*/, DF df) {
    const size_t kBytes = 32;
    static_assert(kBytes % MaxLanes(df) == 0, "Must be divisible");

    HWY_ALIGN uint8_t in[kBytes];
    uint8_t expected[kBytes];
    RandomState rng{1234};
    for (size_t i = 0; i < kBytes; ++i) {
      expected[i] = Random32(&rng) & 7;
      in[i] = static_cast<uint8_t>(1u << expected[i]);
    }
    HWY_ALIGN uint8_t out[32];
    for (size_t i = 0; i < kBytes; i += Lanes(df)) {
      FloorLog2(df, in + i, out + i);
    }
    int sum = 0;
    for (size_t i = 0; i < kBytes; ++i) {
      HWY_ASSERT_EQ(expected[i], out[i]);
      sum += out[i];
    }
    PreventElision(sum);
  }
};

template <class D, typename T>
HWY_NOINLINE void MulAddLoop(const D d, const T* HWY_RESTRICT mul_array,
                             const T* HWY_RESTRICT add_array, const size_t size,
                             T* HWY_RESTRICT x_array) {
  // Type-agnostic (caller-specified lane type) and width-agnostic (uses
  // best available instruction set).
  for (size_t i = 0; i < size; i += Lanes(d)) {
    const auto mul = Load(d, mul_array + i);
    const auto add = Load(d, add_array + i);
    auto x = Load(d, x_array + i);
    x = MulAdd(mul, x, add);
    Store(x, d, x_array + i);
  }
}

struct TestSumMulAdd {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
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
    MulAddLoop(d, mul, add, kSize, x);
    double sum = 0.0;
    for (auto xi : x) {
      sum += static_cast<double>(xi);
    }
    HWY_ASSERT_EQ(78944.0, sum);
  }
};

HWY_NOINLINE void TestExamples() {
  ForPartialVectors<TestFloorLog2>()(float());

  ForPartialVectors<TestSumMulAdd>()(float());
#if HWY_CAP_FLOAT64
  ForPartialVectors<TestSumMulAdd>()(double());
#endif
}

// util.h

HWY_NOINLINE void TestLimits() {
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

HWY_NOINLINE void TestToString() {
  HWY_ASSERT_STRING_EQ("0", std::to_string(int64_t(0)).c_str());
  HWY_ASSERT_STRING_EQ("3", std::to_string(int64_t(3)).c_str());
  HWY_ASSERT_STRING_EQ("-1", std::to_string(int64_t(-1)).c_str());

  HWY_ASSERT_STRING_EQ("9223372036854775807",
                       std::to_string(0x7FFFFFFFFFFFFFFFLL).c_str());
  HWY_ASSERT_STRING_EQ("-9223372036854775808",
                       std::to_string(int64_t(0x8000000000000000ULL)).c_str());

  HWY_ASSERT_STRING_EQ("0.000000", std::to_string(0.0).c_str());
  HWY_ASSERT_STRING_EQ("4.000000", std::to_string(4.0).c_str());
  HWY_ASSERT_STRING_EQ("-1.000000", std::to_string(-1.0).c_str());
  HWY_ASSERT_STRING_EQ("-1.250000", std::to_string(-1.25).c_str());
  HWY_ASSERT_STRING_EQ("2.125000", std::to_string(2.125f).c_str());
}

struct TestIsUnsigned {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D /*unused*/) {
    static_assert(!IsFloat<T>(), "Expected !IsFloat");
    static_assert(!IsSigned<T>(), "Expected !IsSigned");
  }
};

struct TestIsSigned {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D /*unused*/) {
    static_assert(!IsFloat<T>(), "Expected !IsFloat");
    static_assert(IsSigned<T>(), "Expected IsSigned");
  }
};

struct TestIsFloat {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D /*unused*/) {
    static_assert(IsFloat<T>(), "Expected IsFloat");
    static_assert(IsSigned<T>(), "Floats are also considered signed");
  }
};

HWY_NOINLINE void TestType() {
  ForUnsignedTypes(ForPartialVectors<TestIsUnsigned>());
  ForSignedTypes(ForPartialVectors<TestIsSigned>());
  ForFloatTypes(ForPartialVectors<TestIsFloat>());
}

// Ensures wraparound (mod 2^bits)
struct TestOverflowT {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const auto v1 = Set(d, T(1));
    const auto vmax = Set(d, LimitsMax<T>());
    const auto vmin = Set(d, LimitsMin<T>());
    // Unsigned underflow / negative -> positive
    HWY_ASSERT_VEC_EQ(d, vmax, vmin - v1);
    // Unsigned overflow / positive -> negative
    HWY_ASSERT_VEC_EQ(d, vmin, vmax + v1);
  }
};

HWY_NOINLINE void TestOverflow() {
  ForIntegerTypes(ForPartialVectors<TestOverflowT>());
}

struct TestName {
  template <class T, class D>
  HWY_NOINLINE void operator()(T t, D d) {
    std::string expected = IsFloat<T>() ? "f" : (IsSigned<T>() ? "i" : "u");
    expected += std::to_string(sizeof(T) * 8);

    const size_t N = Lanes(d);
    if (N != 1) {
      expected += 'x';
      expected += std::to_string(N);
    }
    const std::string actual = TypeName(t, N);
    if (expected != actual) {
      NotifyFailure(__FILE__, __LINE__, expected.c_str(), 0, expected.c_str(),
                    actual.c_str());
    }
  }
};

struct TestSet {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // Zero
    const auto v0 = Zero(d);
    HWY_ALIGN T expected[MaxLanes(d)] = {};  // zero-initialized.
    HWY_ASSERT_VEC_EQ(d, expected, v0);
    const size_t N = Lanes(d);

    // Set
    const auto v2 = Set(d, T(2));
    for (size_t i = 0; i < N; ++i) {
      expected[i] = 2;
    }
    HWY_ASSERT_VEC_EQ(d, expected, v2);

    // iota
    const auto vi = Iota(d, T(5));
    for (size_t i = 0; i < N; ++i) {
      expected[i] = 5 + i;
    }
    HWY_ASSERT_VEC_EQ(d, expected, vi);

    // undefined
    const auto vu = Undefined(d);
    Store(vu, d, expected);
  }
};

struct TestCopyAndAssign {
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    // copy V
    const auto v3 = Iota(d, 3);
    auto v3b(v3);
    HWY_ASSERT_VEC_EQ(d, v3, v3b);

    // assign V
    auto v3c = Undefined(d);
    v3c = v3;
    HWY_ASSERT_VEC_EQ(d, v3, v3c);
  }
};

HWY_NOINLINE void TestAllSet() { ForAllTypes(ForPartialVectors<TestSet>()); }

HWY_NOINLINE void TestAllNameBasic() {
  ForAllTypes(ForPartialVectors<TestName>());
}

HWY_NOINLINE void TestAllCopyAndAssign() {
  ForAllTypes(ForPartialVectors<TestCopyAndAssign>());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {

class HwyHwyTest : public hwy::TestWithParamTarget {};

HWY_TARGET_INSTANTIATE_TEST_SUITE_P(HwyHwyTest);

HWY_EXPORT_AND_TEST_P(HwyHwyTest, TestLimits);
HWY_EXPORT_AND_TEST_P(HwyHwyTest, TestToString);
HWY_EXPORT_AND_TEST_P(HwyHwyTest, TestType);
HWY_EXPORT_AND_TEST_P(HwyHwyTest, TestOverflow);
HWY_EXPORT_AND_TEST_P(HwyHwyTest, TestAllSet);
HWY_EXPORT_AND_TEST_P(HwyHwyTest, TestAllNameBasic);
HWY_EXPORT_AND_TEST_P(HwyHwyTest, TestAllCopyAndAssign);
HWY_EXPORT_AND_TEST_P(HwyHwyTest, TestExamples);

}  // namespace hwy
#endif
