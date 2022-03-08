// Copyright 2022 Google LLC
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

#include "hwy/aligned_allocator.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/algo/transform_test.cc"
#include "hwy/foreach_target.h"

#include "hwy/contrib/algo/transform-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

// If your project requires C++14 or later, you can ignore this and pass lambdas
// directly to Transform, without requiring an lvalue as we do here for C++11.
#if __cplusplus < 201402L
#define HWY_GENERIC_LAMBDA 0
#else
#define HWY_GENERIC_LAMBDA 1
#endif

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

template <typename T>
T Alpha() {
  return static_cast<T>(1.5);  // arbitrary scalar
}

// Returns random floating-point number in [-8, 8) to ensure computations do
// not exceed float32 precision.
template <typename T>
T Random(RandomState& rng) {
  const int32_t bits = static_cast<int32_t>(Random32(&rng)) & 1023;
  return static_cast<T>(bits - 512) * (T{1} / 64);
}

// SCAL, AXPY names are from BLAS.
template <typename T>
HWY_NOINLINE void SimpleSCAL(const T* x, T* out, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    out[i] = Alpha<T>() * x[i];
  }
}

template <typename T>
HWY_NOINLINE void SimpleAXPY(const T* x, const T* y, T* out, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    out[i] = Alpha<T>() * x[i] + y[i];
  }
}

template <typename T>
HWY_NOINLINE void SimpleFMA4(const T* x, const T* y, const T* z, T* out,
                             size_t count) {
  for (size_t i = 0; i < count; ++i) {
    out[i] = x[i] * y[i] + z[i];
  }
}

// In C++14, we can instead define these as generic lambdas next to where they
// are invoked.
#if !HWY_GENERIC_LAMBDA

struct SCAL {
  template <class D, class V>
  Vec<D> operator()(D d, V v) const {
    using T = TFromD<D>;
    return Mul(Set(d, Alpha<T>()), v);
  }
};

struct AXPY {
  template <class D, class V>
  Vec<D> operator()(D d, V v, V v1) const {
    using T = TFromD<D>;
    return MulAdd(Set(d, Alpha<T>()), v, v1);
  }
};

struct FMA4 {
  template <class D, class V>
  Vec<D> operator()(D /*d*/, V v, V v1, V v2) const {
    return MulAdd(v, v1, v2);
  }
};

#endif  // !HWY_GENERIC_LAMBDA

// Invokes Test (e.g. TestTransform1) with all arg combinations. T comes from
// ForFloatTypes.
template <class Test>
struct ForeachCountAndMisalign {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) const {
    RandomState rng;
    const size_t N = Lanes(d);
    const size_t misalignments[3] = {0, N / 4, 3 * N / 5};

    for (size_t count = 0; count < 2 * N; ++count) {
      for (size_t ma : misalignments) {
        for (size_t mb : misalignments) {
          Test()(d, count, ma, mb, rng);
        }
      }
    }
  }
};

// Zero extra input arrays
struct TestTransform {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t /*misalign_b*/,
                  RandomState& rng) {
    using T = TFromD<D>;
    // Prevents error if size to allocate is zero.
    AlignedFreeUniquePtr<T[]> pa =
        AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    T* a = pa.get() + misalign_a;
    for (size_t i = 0; i < count; ++i) {
      a[i] = Random<T>(rng);
    }

    AlignedFreeUniquePtr<T[]> expected = AllocateAligned<T>(HWY_MAX(1, count));
    SimpleSCAL(a, expected.get(), count);

    // TODO(janwas): can we update the apply_to in HWY_PUSH_ATTRIBUTES so that
    // the attribute also applies to lambdas? If so, remove HWY_ATTR.
#if HWY_GENERIC_LAMBDA
    const auto scal = [](const auto d, const auto v)
                          HWY_ATTR { return Mul(Set(d, Alpha<T>()), v); };
#else
    const SCAL scal;
#endif
    Transform(d, a, count, scal);

    const auto info = hwy::detail::MakeTypeInfo<T>();
    const char* target_name = hwy::TargetName(HWY_TARGET);
    hwy::detail::AssertArrayEqual(info, expected.get(), a, count, target_name,
                                  __FILE__, __LINE__);
  }
};

// One extra input array
struct TestTransform1 {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  RandomState& rng) {
    using T = TFromD<D>;
    // Prevents error if size to allocate is zero.
    AlignedFreeUniquePtr<T[]> pa =
        AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    AlignedFreeUniquePtr<T[]> pb =
        AllocateAligned<T>(HWY_MAX(1, misalign_b + count));
    T* a = pa.get() + misalign_a;
    T* b = pb.get() + misalign_b;
    for (size_t i = 0; i < count; ++i) {
      a[i] = Random<T>(rng);
      b[i] = Random<T>(rng);
    }

    AlignedFreeUniquePtr<T[]> expected = AllocateAligned<T>(HWY_MAX(1, count));
    SimpleAXPY(a, b, expected.get(), count);

#if HWY_GENERIC_LAMBDA
    const auto axpy = [](const auto d, const auto v, const auto v1) HWY_ATTR {
      return MulAdd(Set(d, Alpha<T>()), v, v1);
    };
#else
    const AXPY axpy;
#endif
    Transform1(d, a, count, b, axpy);

    const auto info = hwy::detail::MakeTypeInfo<T>();
    const char* target_name = hwy::TargetName(HWY_TARGET);
    hwy::detail::AssertArrayEqual(info, expected.get(), a, count, target_name,
                                  __FILE__, __LINE__);
  }
};

// Two extra input arrays
struct TestTransform2 {
  template <class D>
  void operator()(D d, size_t count, size_t misalign_a, size_t misalign_b,
                  RandomState& rng) {
    using T = TFromD<D>;
    // Prevents error if size to allocate is zero.
    AlignedFreeUniquePtr<T[]> pa =
        AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    AlignedFreeUniquePtr<T[]> pb =
        AllocateAligned<T>(HWY_MAX(1, misalign_b + count));
    AlignedFreeUniquePtr<T[]> pc =
        AllocateAligned<T>(HWY_MAX(1, misalign_a + count));
    T* a = pa.get() + misalign_a;
    T* b = pb.get() + misalign_b;
    T* c = pc.get() + misalign_a;
    for (size_t i = 0; i < count; ++i) {
      a[i] = Random<T>(rng);
      b[i] = Random<T>(rng);
      c[i] = Random<T>(rng);
    }

    AlignedFreeUniquePtr<T[]> expected = AllocateAligned<T>(HWY_MAX(1, count));
    SimpleFMA4(a, b, c, expected.get(), count);

#if HWY_GENERIC_LAMBDA
    const auto fma4 = [](auto /*d*/, auto v, auto v1, auto v2)
                          HWY_ATTR { return MulAdd(v, v1, v2); };
#else
    const FMA4 fma4;
#endif
    Transform2(d, a, count, b, c, fma4);

    const auto info = hwy::detail::MakeTypeInfo<T>();
    const char* target_name = hwy::TargetName(HWY_TARGET);
    hwy::detail::AssertArrayEqual(info, expected.get(), a, count, target_name,
                                  __FILE__, __LINE__);
  }
};

void TestAllTransform() {
  ForFloatTypes(ForPartialVectors<ForeachCountAndMisalign<TestTransform>>());
}

void TestAllTransform1() {
  ForFloatTypes(ForPartialVectors<ForeachCountAndMisalign<TestTransform1>>());
}

void TestAllTransform2() {
  ForFloatTypes(ForPartialVectors<ForeachCountAndMisalign<TestTransform2>>());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
HWY_BEFORE_TEST(TransformTest);
HWY_EXPORT_AND_TEST_P(TransformTest, TestAllTransform);
HWY_EXPORT_AND_TEST_P(TransformTest, TestAllTransform1);
HWY_EXPORT_AND_TEST_P(TransformTest, TestAllTransform2);
}  // namespace hwy

#endif
