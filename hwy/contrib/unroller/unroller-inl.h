// Copyright 2023 Matthew Kolbe
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

#if defined(HIGHWAY_HWY_CONTRIB_UNROLLER_UNROLLER_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_UNROLLER_UNROLLER_INL_H_
#undef HIGHWAY_HWY_CONTRIB_UNROLLER_UNROLLER_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_UNROLLER_UNROLLER_INL_H_
#endif

#include <cstring>
#include <limits>

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <class DERIVED, typename IN_T, typename OUT_T>
struct UnrollerUnit {
  DERIVED* me() { return static_cast<DERIVED*>(this); }

  static constexpr inline size_t UnitLanes() {
    return HWY_MIN(HWY_MAX_LANES_D(hn::ScalableTag<IN_T>),
                    HWY_MAX_LANES_D(hn::ScalableTag<OUT_T>));
  }

  using IT = hn::CappedTag<IN_T, UnitLanes()>;
  using OT = hn::CappedTag<OUT_T, UnitLanes()>;
  IT d_in;
  OT d_out;

  inline hn::Vec<OT> Func(int idx, hn::Vec<IT> x, hn::Vec<OT> y) {
    return me()->Func(idx, x, y);
  }

  inline hn::Vec<IT> X0Init() { return me()->X0InitImpl(); }

  inline hn::Vec<IT> X0InitImpl() { return hn::Zero(d_in); }

  inline hn::Vec<OT> YInit() { return me()->YInitImpl(); }

  inline hn::Vec<OT> YInitImpl() { return hn::Zero(d_out); }

  inline hn::Vec<IT> Load(int idx, IN_T* from) {
    return me()->LoadImpl(idx, from);
  }

  inline hn::Vec<IT> LoadImpl(int idx, IN_T* from) {
    return hn::LoadU(d_in, from + idx);
  }

  // MaskLoad can take in either a positive or negative number for `places`. if
  // the number is positive, then it loads the top `places` values, and if it's
  // negative, it loads the bottom |places| values. example: places = 3
  //      | o | o | o | x | x | x | x | x |
  // example places = -3
  //      | x | x | x | x | x | o | o | o |
  inline hn::Vec<IT> MaskLoad(int idx, IN_T* from, int places) {
    return me()->MaskLoadImpl(idx, from, places);
  }

  inline hn::Vec<IT> MaskLoadImpl(int idx, IN_T* from, int places) {
    const hn::ScalableTag<int> di;
    using TI = hn::TFromD<decltype(di)>;
    auto mask = hn::RebindMask(
        d_in, hn::detail::Iota0(d_in) < Set(d_in, static_cast<TI>(places)));
    auto maskneg =
        hn::RebindMask(d_in, hn::detail::Iota0(d_in) >=
                                 Set(d_in, static_cast<TI>(places + UnitLanes())));
    if (places < 0) mask = maskneg;

    return hn::MaskedLoad(mask, d_in, from + idx);
  }

  inline int Store(int idx, OUT_T* to, hn::Vec<OT> x) {
    return me()->StoreImpl(idx, to, x);
  }

  inline int StoreImpl(int idx, OUT_T* to, hn::Vec<OT> x) {
    hn::StoreU(x, d_out, to + idx);
    return d_out.MaxLanes();
  }

  inline int MaskStore(int idx, OUT_T* to, hn::Vec<OT> x, int places) {
    return me()->MaskStoreImpl(idx, to, x, places);
  }

  inline int MaskStoreImpl(int idx, OUT_T* to, hn::Vec<OT> x, int places) {
    const hn::ScalableTag<int> di;
    using TI = hn::TFromD<decltype(di)>;
    auto mask = hn::RebindMask(
        d_out, hn::detail::Iota0(d_out) < Set(d_out, static_cast<TI>(places)));
    auto maskneg = hn::RebindMask(
        d_out, hn::detail::Iota0(d_out) >=
                   Set(d_out, static_cast<TI>(places + UnitLanes())));
    if (places < 0) mask = maskneg;

    hn::BlendedStore(x, mask, d_out, to + idx);
    return std::abs(places);
  }

  inline int Reduce(hn::Vec<OT> x, OUT_T* to) {
    return me()->ReduceImpl(x, to);
  }

  inline int ReduceImpl(hn::Vec<OT> x, OUT_T* to) {
    // default does nothing
    return 0;
  }

  inline void Reduce(hn::Vec<OT> x0, hn::Vec<OT> x1, hn::Vec<OT> x2,
                     hn::Vec<OT>& y) {
    me()->ReduceImpl(x0, x1, x2, y);
  }

  inline void ReduceImpl(hn::Vec<OT> x0, hn::Vec<OT> x1, hn::Vec<OT> x2,
                          hn::Vec<OT>& y) {
    // default does nothing
  }
};

template <class DERIVED, typename IN0_T, typename IN1_T, typename OUT_T>
struct UnrollerUnit2 {
  DERIVED* me() { return static_cast<DERIVED*>(this); }

  static constexpr inline size_t UnitLanes() {
    return HWY_MIN(HWY_MAX_LANES_D(hn::ScalableTag<IN1_T>),
                    HWY_MIN(HWY_MAX_LANES_D(hn::ScalableTag<IN0_T>),
                             HWY_MAX_LANES_D(hn::ScalableTag<OUT_T>)));
  }

  using I0T = hn::CappedTag<IN0_T, UnitLanes()>;
  using I1T = hn::CappedTag<IN1_T, UnitLanes()>;
  using OT = hn::CappedTag<OUT_T, UnitLanes()>;
  I0T d_in0;
  I1T d_in1;
  OT d_out;

  inline hn::Vec<OT> Func(int idx, hn::Vec<I0T> x0, hn::Vec<I1T> x1,
                          hn::Vec<OT> y) {
    return me()->Func(idx, x0, x1, y);
  }

  inline hn::Vec<I0T> X0Init() { return me()->X0InitImpl(); }

  inline hn::Vec<I0T> X0InitImpl() { return hn::Zero(d_in0); }

  inline hn::Vec<I1T> X1Init() { return me()->X1InitImpl(); }

  inline hn::Vec<I0T> X1InitImpl() { return hn::Zero(d_in1); }

  inline hn::Vec<OT> YInit() { return me()->YInitImpl(); }

  inline hn::Vec<OT> YInitImpl() { return hn::Zero(d_out); }

  inline hn::Vec<I0T> Load0(int idx, IN0_T* from) {
    return me()->Load0Impl(idx, from);
  }

  inline hn::Vec<I0T> Load0Impl(int idx, IN0_T* from) {
    return hn::LoadU(d_in0, from + idx);
  }

  inline hn::Vec<I1T> Load1(int idx, IN1_T* from) {
    return me()->Load1Impl(idx, from);
  }

  inline hn::Vec<I1T> Load1Impl(int idx, IN1_T* from) {
    return hn::LoadU(d_in1, from + idx);
  }

  // maskload can take in either a positive or negative number for `places`. if
  // the number is positive, then it loads the top `places` values, and if it's
  // negative, it loads the bottom |places| values. example: places = 3
  //      | o | o | o | x | x | x | x | x |
  // example places = -3
  //      | x | x | x | x | x | o | o | o |
  inline hn::Vec<I0T> MaskLoad0(int idx, IN0_T* from, int places) {
    return me()->MaskLoad0Impl(idx, from, places);
  }

  inline hn::Vec<I0T> MaskLoad0Impl(int idx, IN0_T* from, int places) {
    const hn::ScalableTag<int> di;
    using TI = hn::TFromD<decltype(di)>;
    auto mask = hn::RebindMask(
        d_in0, hn::detail::Iota0(d_in0) < Set(d_in0, static_cast<TI>(places)));
    auto maskneg = hn::RebindMask(
        d_in0, hn::detail::Iota0(d_in0) >=
                   Set(d_in0, static_cast<TI>(places + UnitLanes())));
    if (places < 0) mask = maskneg;

    return hn::MaskedLoad(mask, d_in0, from + idx);
  }

  inline hn::Vec<I1T> MaskLoad1(int idx, IN1_T* from, int places) {
    return me()->MaskLoad1Impl(idx, from, places);
  }

  inline hn::Vec<I1T> MaskLoad1Impl(int idx, IN1_T* from, int places) {
    const hn::ScalableTag<int> di;
    using TI = hn::TFromD<decltype(di)>;
    auto mask = hn::RebindMask(
        d_in1, hn::detail::Iota0(d_in1) < Set(d_in1, static_cast<TI>(places)));
    auto maskneg = hn::RebindMask(
        d_in1, hn::detail::Iota0(d_in1) >=
                   Set(d_in1, static_cast<TI>(places + UnitLanes())));
    if (places < 0) mask = maskneg;

    return hn::MaskedLoad(mask, d_in1, from + idx);
  }

  inline int Store(int idx, OUT_T* to, hn::Vec<OT> x) {
    return me()->StoreImpl(idx, to, x);
  }

  inline int StoreImpl(int idx, OUT_T* to, hn::Vec<OT> x) {
    hn::StoreU(x, d_out, to + idx);
    return d_out.MaxLanes();
  }

  inline int MaskStore(int idx, OUT_T* to, hn::Vec<OT> x, int places) {
    return me()->MaskStoreImpl(idx, to, x, places);
  }

  inline int MaskStoreImpl(int idx, OUT_T* to, hn::Vec<OT> x, int places) {
    const hn::ScalableTag<int> di;
    using TI = hn::TFromD<decltype(di)>;
    auto mask = hn::RebindMask(
        d_out, hn::detail::Iota0(d_out) < Set(d_out, static_cast<TI>(places)));
    auto maskneg = hn::RebindMask(
        d_out, hn::detail::Iota0(d_out) >=
                   Set(d_out, static_cast<TI>(places + UnitLanes())));
    if (places < 0) mask = maskneg;

    hn::BlendedStore(x, mask, d_out, to + idx);
    return std::abs(places);
  }

  inline int Reduce(hn::Vec<OT> x, OUT_T* to) {
    return me()->ReduceImpl(x, to);
  }

  inline int ReduceImpl(hn::Vec<OT> x, OUT_T* to) {
    // default does nothing
    return 0;
  }

  inline void Reduce(hn::Vec<OT> x0, hn::Vec<OT> x1, hn::Vec<OT> x2,
                     hn::Vec<OT>& y) {
    me()->ReduceImpl(x0, x1, x2, y);
  }

  inline void ReduceImpl(hn::Vec<OT> x0, hn::Vec<OT> x1, hn::Vec<OT> x2,
                          hn::Vec<OT>& y) {
    // default does nothing
  }
};

template <class FUNC, typename IN_T, typename OUT_T>
inline void unroller(FUNC& f, IN_T* x, OUT_T* y, const unsigned int n) {
  const auto lane_sz = f.UnitLanes();

  auto xx = f.X0Init();
  auto yy = f.YInit();
  int i = 0;

#if HWY_MEM_OPS_MIGHT_FAULT
  if (n < lane_sz) {
    // stack is maybe too small for this in RVV?
    IN_T xtmp[lane_sz];
    OUT_T ytmp[lane_sz];

    memcpy(xtmp, x, n * sizeof(IN_T));
    xx = f.MaskLoad(0, xtmp, n);
    yy = f.Func(0, xx, yy);
    i += f.MaskStore(0, ytmp, yy, n);
    i += f.Reduce(yy, ytmp);
    memcpy(y, ytmp, i * sizeof(OUT_T));
    return;
  }
#endif

  if (n > 4 * lane_sz) {
    auto xx1 = f.X0Init();
    auto yy1 = f.YInit();
    auto xx2 = f.X0Init();
    auto yy2 = f.YInit();
    auto xx3 = f.X0Init();
    auto yy3 = f.YInit();

    while (i + 4 * lane_sz - 1 < n) {
      xx = f.Load(i, x);
      i += lane_sz;
      xx1 = f.Load(i, x);
      i += lane_sz;
      xx2 = f.Load(i, x);
      i += lane_sz;
      xx3 = f.Load(i, x);
      i -= 3 * lane_sz;

      yy = f.Func(i, xx, yy);
      yy1 = f.Func(i + lane_sz, xx1, yy1);
      yy2 = f.Func(i + 2 * lane_sz, xx2, yy2);
      yy3 = f.Func(i + 3 * lane_sz, xx3, yy3);

      f.Store(i, y, yy);
      i += lane_sz;
      f.Store(i, y, yy1);
      i += lane_sz;
      f.Store(i, y, yy2);
      i += lane_sz;
      f.Store(i, y, yy3);
      i += lane_sz;
    }

    f.Reduce(yy3, yy2, yy1, yy);
  }

  while (i + lane_sz - 1 < n) {
    xx = f.Load(i, x);
    yy = f.Func(i, xx, yy);
    f.Store(i, y, yy);
    i += lane_sz;
  }

  if (i != n) {
    xx = f.MaskLoad(n - lane_sz, x, i - n);
    yy = f.Func(n - lane_sz, xx, yy);
    f.MaskStore(n - lane_sz, y, yy, i - n);
  }

  f.Reduce(yy, y);
}

template <class FUNC, typename IN0_T, typename IN1_T, typename OUT_T>
inline void unroller(FUNC& f, IN0_T* x0, IN1_T* x1, OUT_T* y, const int n) {
  const size_t lane_sz = f.UnitLanes();

  auto xx00 = f.X0Init();
  auto xx10 = f.X1Init();
  auto yy = f.YInit();

  int i = 0;

#if HWY_MEM_OPS_MIGHT_FAULT
  if (n < lane_sz) {
    // stack is maybe too small for this in RVV?
    IN0_T xtmp0[lane_sz];
    IN1_T xtmp1[lane_sz];
    OUT_T ytmp[lane_sz];

    memcpy(xtmp0, x0, n * sizeof(IN0_T));
    memcpy(xtmp1, x1, n * sizeof(IN1_T));
    xx00 = f.MaskLoad0(0, xtmp0, n);
    xx10 = f.MaskLoad1(0, xtmp1, n);
    yy = f.Func(0, xx00, xx10, yy);
    i += f.MaskStore(0, ytmp, yy, n);
    i += f.Reduce(yy, ytmp);
    memcpy(y, ytmp, i * sizeof(OUT_T));
    return;
  }
#endif

  if (n > 4 * lane_sz) {
    auto xx01 = f.X0Init();
    auto xx11 = f.X1Init();
    auto yy1 = f.YInit();
    auto xx02 = f.X0Init();
    auto xx12 = f.X1Init();
    auto yy2 = f.YInit();
    auto xx03 = f.X0Init();
    auto xx13 = f.X1Init();
    auto yy3 = f.YInit();

    while (i + 4 * lane_sz - 1 < n) {
      xx00 = f.Load0(i, x0);
      xx10 = f.Load1(i, x1);
      i += lane_sz;
      xx01 = f.Load0(i, x0);
      xx11 = f.Load1(i, x1);
      i += lane_sz;
      xx02 = f.Load0(i, x0);
      xx12 = f.Load1(i, x1);
      i += lane_sz;
      xx03 = f.Load0(i, x0);
      xx13 = f.Load1(i, x1);
      i -= 3 * lane_sz;

      yy = f.Func(i, xx00, xx10, yy);
      yy1 = f.Func(i + lane_sz, xx01, xx11, yy1);
      yy2 = f.Func(i + 2 * lane_sz, xx02, xx12, yy2);
      yy3 = f.Func(i + 3 * lane_sz, xx03, xx13, yy3);

      f.Store(i, y, yy);
      i += lane_sz;
      f.Store(i, y, yy1);
      i += lane_sz;
      f.Store(i, y, yy2);
      i += lane_sz;
      f.Store(i, y, yy3);
      i += lane_sz;
    }

    f.Reduce(yy3, yy2, yy1, yy);
  }

  while (i + lane_sz - 1 < n) {
    xx00 = f.Load0(i, x0);
    xx10 = f.Load1(i, x1);
    yy = f.Func(i, xx00, xx10, yy);
    f.Store(i, y, yy);
    i += lane_sz;
  }

  if (i != n) {
    xx00 = f.MaskLoad0(i, x0, i - n);
    xx10 = f.MaskLoad1(i, x1, i - n);
    yy = f.Func(i, xx00, xx10, yy);
    f.MaskStore(i, y, yy, i - n);
  }

  f.Reduce(yy, y);
}



}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_UNROLLER_UNROLLER_INL_H_