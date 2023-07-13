// Copyright Google LLC 2021
//           Matthew Kolbe 2023
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

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/unroller/unroller_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/unroller/unroller-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

template <typename T>
T SimpleDot(const T* pa, const T* pb, size_t num) {
  T sum = 0;
  for (size_t i = 0; i < num; ++i) {
    sum += pa[i] * pb[i];
  }
  return sum;
}

template <typename T>
T SimpleAcc(const T* pa, size_t num) {
  T sum = 0;
  for (size_t i = 0; i < num; ++i) {
    sum += pa[i];
  }
  return sum;
}

template <typename T>
T SimpleMin(const T* pa, size_t num) {
  T min = HighestValue<T>();
  for (size_t i = 0; i < num; ++i) {
    if (min > pa[i]) min = pa[i];
  }
  return min;
}

template <typename T>
struct MultiplyUnit : UnrollerUnit2D<MultiplyUnit<T>, T, T, T> {
  using TT = hn::ScalableTag<T>;
  hn::Vec<TT> Func(ptrdiff_t idx, const hn::Vec<TT> x0, const hn::Vec<TT> x1,
                   const hn::Vec<TT> y) {
    (void)idx;
    (void)y;
    return hn::Mul(x0, x1);
  }
};

template <typename FROM_T, typename TO_T>
struct ConvertUnit : UnrollerUnit<ConvertUnit<FROM_T, TO_T>, FROM_T, TO_T> {
  static constexpr size_t UnitLanes() {
    return HWY_MIN(HWY_MAX_LANES_D(hn::ScalableTag<FROM_T>),
                   HWY_MAX_LANES_D(hn::ScalableTag<TO_T>));
  }

  using TT_FROM = hn::CappedTag<FROM_T, UnitLanes()>;
  using TT_TO = hn::CappedTag<TO_T, UnitLanes()>;

  template <
      class ToD, class FromV,
      hwy::EnableIf<(sizeof(TFromV<FromV>) > sizeof(TFromD<ToD>))>* = nullptr>
  static HWY_INLINE hn::Vec<ToD> DoConvertVector(ToD d, FromV v) {
    return hn::DemoteTo(d, v);
  }
  template <
      class ToD, class FromV,
      hwy::EnableIf<(sizeof(TFromV<FromV>) == sizeof(TFromD<ToD>))>* = nullptr>
  static HWY_INLINE hn::Vec<ToD> DoConvertVector(ToD d, FromV v) {
    return hn::ConvertTo(d, v);
  }
  template <
      class ToD, class FromV,
      hwy::EnableIf<(sizeof(TFromV<FromV>) < sizeof(TFromD<ToD>))>* = nullptr>
  static HWY_INLINE hn::Vec<ToD> DoConvertVector(ToD d, FromV v) {
    return hn::PromoteTo(d, v);
  }

  hn::Vec<TT_TO> Func(ptrdiff_t idx, const hn::Vec<TT_FROM> x,
                      const hn::Vec<TT_TO> y) {
    (void)idx;
    (void)y;
    TT_TO d;
    return DoConvertVector(d, x);
  }
};

template <typename T>
struct FindUnit : UnrollerUnit<FindUnit<T>, T, ptrdiff_t> {
  static constexpr size_t UnitLanes() {
    return HWY_MIN(HWY_MAX_LANES_D(hn::ScalableTag<T>),
                   HWY_MAX_LANES_D(hn::ScalableTag<ptrdiff_t>));
  }

  using TT = hn::CappedTag<T, UnitLanes()>;
  // using TTSZT = hn::CappedTag<ptrdiff_t, UnitLanes()>;
  using TTSZT = hn::CappedTag<SignedFromSize<sizeof(ptrdiff_t)>, UnitLanes()>;
  hn::Vec<TT> to_find;
  TT d;
  TTSZT szd;

  FindUnit<T>(T find) { to_find = Set(d, find); }

  hn::Vec<TTSZT> Func(ptrdiff_t idx, const hn::Vec<TT> x,
                      const hn::Vec<TTSZT> y) {
    auto msk = x == to_find;
    auto first_idx = hn::FindFirstTrue(d, msk);

    if (first_idx > -1)
      return hn::Set(szd, idx + first_idx);
    else
      return y;
  }

  hn::Vec<TT> X0InitImpl() { return hn::Add(to_find, Set(d, 1)); }

  hn::Vec<TTSZT> YInitImpl() { return hn::Set(szd, -1); }

  hn::Vec<TT> MaskLoadImpl(const ptrdiff_t idx, T* from,
                           const ptrdiff_t places) {
    auto mask = hn::FirstN(d, static_cast<size_t>(places));
    auto maskneg = hn::Not(hn::FirstN(
        d, static_cast<size_t>(places + static_cast<ptrdiff_t>(UnitLanes()))));
    if (places < 0) mask = maskneg;

    return hn::IfThenElse(mask, hn::MaskedLoad(mask, d, from + idx),
                          X0InitImpl());
  }

  bool StoreAndShortCircuitImpl(const ptrdiff_t idx, ptrdiff_t* to,
                                const hn::Vec<TTSZT> x) {
    (void)idx;

    ptrdiff_t a = hn::ExtractLane(x, 0);
    to[0] = a;

    if (a == -1) return true;

    return false;
  }

  ptrdiff_t MaskStoreImpl(const ptrdiff_t idx, ptrdiff_t* to,
                          const hn::Vec<TTSZT> x, const ptrdiff_t places) {
    (void)idx;
    auto mask = hn::FirstN(szd, static_cast<size_t>(places));
    auto maskneg = hn::Not(hn::FirstN(
        szd,
        static_cast<size_t>(places + static_cast<ptrdiff_t>(UnitLanes()))));
    if (places < 0) mask = maskneg;

    ptrdiff_t a = hn::ExtractLane(x, 0);
    to[0] = a;

    return std::abs(places);
  }
};

template <typename T>
struct AccumulateUnit : UnrollerUnit<AccumulateUnit<T>, T, T> {
  using TT = hn::ScalableTag<T>;
  hn::Vec<TT> Func(ptrdiff_t idx, const hn::Vec<TT> x, const hn::Vec<TT> y) {
    (void)idx;
    return hn::Add(x, y);
  }

  bool StoreAndShortCircuitImpl(const ptrdiff_t idx, T* to,
                                const hn::Vec<TT> x) {
    // no stores in a reducer
    (void)idx;
    (void)to;
    (void)x;
    return true;
  }

  ptrdiff_t MaskStoreImpl(const ptrdiff_t idx, T* to, const hn::Vec<TT> x,
                          const ptrdiff_t places) {
    // no stores in a reducer
    (void)idx;
    (void)to;
    (void)x;
    (void)places;
    return 0;
  }

  ptrdiff_t ReduceImpl(const hn::Vec<TT> x, T* to) {
    const hn::ScalableTag<T> d;
    (*to) = hn::ReduceSum(d, x);
    return 1;
  }

  void ReduceImpl(const hn::Vec<TT> x0, const hn::Vec<TT> x1,
                  const hn::Vec<TT> x2, hn::Vec<TT>* y) {
    (*y) = hn::Add(hn::Add(*y, x0), hn::Add(x1, x2));
  }
};

template <typename T>
struct MinUnit : UnrollerUnit<MinUnit<T>, T, T> {
  using TT = hn::ScalableTag<T>;
  TT d;

  hn::Vec<TT> Func(const ptrdiff_t idx, const hn::Vec<TT> x,
                   const hn::Vec<TT> y) {
    (void)idx;
    return hn::Min(y, x);
  }

  hn::Vec<TT> YInitImpl() { return hn::Set(d, HighestValue<T>()); }

  hn::Vec<TT> MaskLoadImpl(const ptrdiff_t idx, T* from,
                           const ptrdiff_t places) {
    auto mask = hn::FirstN(d, static_cast<size_t>(places));
    auto maskneg = hn::Not(hn::FirstN(
        d, static_cast<size_t>(
               places + static_cast<ptrdiff_t>(
                            UnrollerUnit<MinUnit<T>, T, T>::UnitLanes()))));
    if (places < 0) mask = maskneg;

    auto def = YInitImpl();
    return hn::MaskedLoadOr(def, mask, d, from + idx);
  }

  bool StoreAndShortCircuitImpl(const ptrdiff_t idx, T* to,
                                const hn::Vec<TT> x) {
    // no stores in a reducer
    (void)idx;
    (void)to;
    (void)x;
    return true;
  }

  ptrdiff_t MaskStoreImpl(const ptrdiff_t idx, T* to, const hn::Vec<TT> x,
                          const ptrdiff_t places) {
    // no stores in a reducer
    (void)idx;
    (void)to;
    (void)x;
    (void)places;
    return 0;
  }

  ptrdiff_t ReduceImpl(const hn::Vec<TT> x, T* to) {
    const hn::ScalableTag<T> d;
    auto minvect = hn::MinOfLanes(d, x);
    (*to) = hn::ExtractLane(minvect, 0);
    return 1;
  }

  void ReduceImpl(const hn::Vec<TT> x0, const hn::Vec<TT> x1,
                  const hn::Vec<TT> x2, hn::Vec<TT>* y) {
    auto a = hn::Min(x1, x0);
    auto b = hn::Min(*y, x2);
    (*y) = hn::Min(a, b);
  }
};

template <typename T>
struct DotUnit : UnrollerUnit2D<DotUnit<T>, T, T, T> {
  using TT = hn::ScalableTag<T>;

  hn::Vec<TT> Func(const ptrdiff_t idx, const hn::Vec<TT> x0,
                   const hn::Vec<TT> x1, const hn::Vec<TT> y) {
    (void)idx;
    return hn::MulAdd(x0, x1, y);
  }

  bool StoreAndShortCircuitImpl(const ptrdiff_t idx, T* to,
                                const hn::Vec<TT> x) {
    // no stores in a reducer
    (void)idx;
    (void)to;
    (void)x;
    return true;
  }

  ptrdiff_t MaskStoreImpl(const ptrdiff_t idx, T* to, const hn::Vec<TT> x,
                          const ptrdiff_t places) {
    // no stores in a reducer
    (void)idx;
    (void)to;
    (void)x;
    (void)places;
    return 0;
  }

  ptrdiff_t ReduceImpl(const hn::Vec<TT> x, T* to) {
    const hn::ScalableTag<T> d;
    (*to) = hn::ReduceSum(d, x);
    return 1;
  }

  void ReduceImpl(const hn::Vec<TT> x0, const hn::Vec<TT> x1,
                  const hn::Vec<TT> x2, hn::Vec<TT>* y) {
    (*y) = hn::Add(hn::Add(*y, x0), hn::Add(x1, x2));
  }
};

template <typename T>
void SetValue(const float value, T* HWY_RESTRICT ptr) {
  *ptr = static_cast<T>(value);
}

class TestUnroller {
  template <class D>
  void Test(D, size_t num, RandomState& rng) {
    using T = TFromD<D>;

    const auto random_t = [&rng]() {
      const int32_t bits = static_cast<int32_t>(Random32(&rng)) & 1023;
      return static_cast<float>(bits - 512) * (1.0f / 64);
    };

    AlignedFreeUniquePtr<T[]> pa = AllocateAligned<T>(num);
    AlignedFreeUniquePtr<T[]> pb = AllocateAligned<T>(num);
    AlignedFreeUniquePtr<T[]> py = AllocateAligned<T>(num);

    HWY_ASSERT(pa && pb && py);
    T* a = pa.get();
    T* b = pb.get();
    T* y = py.get();

    size_t i = 0;
    for (; i < num; ++i) {
      SetValue(random_t(), a + i);
      SetValue(random_t(), b + i);
    }

    auto expected_dot = SimpleDot(a, b, num);
    MultiplyUnit<T> multfn;
    Unroller(multfn, a, b, y, static_cast<ptrdiff_t>(num));
    AccumulateUnit<T> accfn;
    T dot_via_mul_acc;
    Unroller(accfn, y, &dot_via_mul_acc, static_cast<ptrdiff_t>(num));
    HWY_ASSERT(std::abs(expected_dot - dot_via_mul_acc) < 1e-7);

    DotUnit<T> dotfn;
    T dotr;
    Unroller(dotfn, a, b, &dotr, static_cast<ptrdiff_t>(num));
    // HWY_ASSERT(dotr != 0);
    HWY_ASSERT(std::abs(expected_dot - dotr) < 1e-7);

    auto expected_min = SimpleMin(a, num);
    MinUnit<T> minfn;
    T minr;
    Unroller(minfn, a, &minr, static_cast<ptrdiff_t>(num));

    HWY_ASSERT(std::abs(expected_min - minr) < 1e-7);
  }

  template <typename T>
  void TestConvertToFromInt(size_t num) {
    AlignedFreeUniquePtr<T[]> pa = AllocateAligned<T>(num);
    AlignedFreeUniquePtr<int[]> pto = AllocateAligned<int>(num);
    HWY_ASSERT(pa && pto);
    T* a = pa.get();
    int* to = pto.get();

    for (size_t i = 0; i < num; ++i) a[i] = (T)(((double)i) * 0.25);

    ConvertUnit<T, int> cvtfn;
    Unroller(cvtfn, a, to, static_cast<ptrdiff_t>(num));
    for (size_t i = 0; i < num; ++i) HWY_ASSERT(to[i] == (int)a[i]);

    ConvertUnit<int, T> cvtbackfn;
    Unroller(cvtbackfn, to, a, static_cast<ptrdiff_t>(num));
    for (size_t i = 0; i < num; ++i) HWY_ASSERT(a[i] == (T)to[i]);
  }

  template <typename T>
  void TestFind(size_t num) {
    AlignedFreeUniquePtr<T[]> pa = AllocateAligned<T>(num);
    HWY_ASSERT(pa);
    T* a = pa.get();

    for (size_t i = 0; i < num; ++i) a[i] = (T)i;

    FindUnit<T> cvtfn((T)(num - 1));
    ptrdiff_t idx = 0;
    Unroller(cvtfn, a, &idx, static_cast<ptrdiff_t>(num));
    HWY_ASSERT(a[idx] == (T)(num - 1));

    FindUnit<T> cvtfnzero((T)(0));
    Unroller(cvtfnzero, a, &idx, static_cast<ptrdiff_t>(num));
    HWY_ASSERT(a[idx] == (T)(0));

    FindUnit<T> cvtfnnotin((T)(num));
    Unroller(cvtfnnotin, a, &idx, static_cast<ptrdiff_t>(num));
    HWY_ASSERT(idx == -1);
  }

 public:
  template <class T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    RandomState rng;
    const size_t N = Lanes(d);
    const size_t counts[] = {1,
                             3,
                             7,
                             16,
                             HWY_MAX(N / 2, 1),
                             HWY_MAX(2 * N / 3, 1),
                             N,
                             N + 1,
                             4 * N / 3,
                             3 * N,
                             8 * N,
                             8 * N + 2,
                             256 * N - 1,
                             256 * N};
    for (auto count : counts) {
      Test(d, count, rng);
      TestConvertToFromInt<T>(count);
      TestFind<T>(count);
    }
  }
};

void TestAllUnroller() { ForFloatTypes(ForPartialVectors<TestUnroller>()); }

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
HWY_BEFORE_TEST(UnrollerTest);
HWY_EXPORT_AND_TEST_P(UnrollerTest, TestAllUnroller);
}  // namespace hwy

#endif
