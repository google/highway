

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

// std::min isn't constexpr in c++11, so we use this
template <typename T>
constexpr T constmin(const T a, const T b) {
  return a < b ? a : b;
}

template <class DERIVED, typename IN_T, typename OUT_T>
struct UnrollerUnit {
  DERIVED* me() { return static_cast<DERIVED*>(this); }

  static constexpr inline size_t lanes() {
    return constmin(HWY_MAX_LANES_D(hn::ScalableTag<IN_T>),
                    HWY_MAX_LANES_D(hn::ScalableTag<OUT_T>));
  }

  using IT = hn::CappedTag<IN_T, lanes()>;
  using OT = hn::CappedTag<OUT_T, lanes()>;
  IT d_in;
  OT d_out;

  inline hn::Vec<OT> func(int idx, hn::Vec<IT> x, hn::Vec<OT> y) {
    return me()->func(idx, x, y);
  }

  inline hn::Vec<IT> x0_init() { return me()->x0_init_impl(); }

  inline hn::Vec<IT> x0_init_impl() { return hn::Zero(d_in); }

  inline hn::Vec<OT> y_init() { return me()->y_init_impl(); }

  inline hn::Vec<OT> y_init_impl() { return hn::Zero(d_out); }

  inline hn::Vec<IT> load(int idx, IN_T* from) {
    return me()->load_impl(idx, from);
  }

  inline hn::Vec<IT> load_impl(int idx, IN_T* from) {
    return hn::LoadU(d_in, from + idx);
  }

  // maskload can take in either a positive or negative number for `places`. if
  // the number is positive, then it loads the top `places` values, and if it's
  // negative, it loads the bottom |places| values. example: places = 3
  //      | o | o | o | x | x | x | x | x |
  // example places = -3
  //      | x | x | x | x | x | o | o | o |
  inline hn::Vec<IT> maskload(int idx, IN_T* from, int places) {
    return me()->maskload_impl(idx, from, places);
  }

  inline hn::Vec<IT> maskload_impl(int idx, IN_T* from, int places) {
    const hn::ScalableTag<int> di;
    using TI = hn::TFromD<decltype(di)>;
    auto mask = hn::RebindMask(
        d_in, hn::detail::Iota0(d_in) < Set(d_in, static_cast<TI>(places)));
    auto maskneg =
        hn::RebindMask(d_in, hn::detail::Iota0(d_in) >=
                                 Set(d_in, static_cast<TI>(places + lanes())));
    if (places < 0) mask = maskneg;

    return hn::MaskedLoad(mask, d_in, from + idx);
  }

  inline int store(int idx, OUT_T* to, hn::Vec<OT> x) {
    return me()->store_impl(idx, to, x);
  }

  inline int store_impl(int idx, OUT_T* to, hn::Vec<OT> x) {
    hn::StoreU(x, d_out, to + idx);
    return d_out.MaxLanes();
  }

  inline int maskstore(int idx, OUT_T* to, hn::Vec<OT> x, int places) {
    return me()->maskstore_impl(idx, to, x, places);
  }

  inline int maskstore_impl(int idx, OUT_T* to, hn::Vec<OT> x, int places) {
    const hn::ScalableTag<int> di;
    using TI = hn::TFromD<decltype(di)>;
    auto mask = hn::RebindMask(
        d_out, hn::detail::Iota0(d_out) < Set(d_out, static_cast<TI>(places)));
    auto maskneg = hn::RebindMask(
        d_out, hn::detail::Iota0(d_out) >=
                   Set(d_out, static_cast<TI>(places + lanes())));
    if (places < 0) mask = maskneg;

    hn::BlendedStore(x, mask, d_out, to + idx);
    return std::abs(places);
  }

  inline int reduce(hn::Vec<OT> x, OUT_T* to) {
    return me()->reduce_impl(x, to);
  }

  inline int reduce_impl(hn::Vec<OT> x, OUT_T* to) {
    // default does nothing
    return 0;
  }

  inline void reduce(hn::Vec<OT> x0, hn::Vec<OT> x1, hn::Vec<OT> x2,
                     hn::Vec<OT>& y) {
    me()->reduce_impl(x0, x1, x2, y);
  }

  inline void reduce_impl(hn::Vec<OT> x0, hn::Vec<OT> x1, hn::Vec<OT> x2,
                          hn::Vec<OT>& y) {
    // default does nothing
  }
};

template <class DERIVED, typename IN0_T, typename IN1_T, typename OUT_T>
struct UnrollerUnit2 {
  DERIVED* me() { return static_cast<DERIVED*>(this); }

  static constexpr inline size_t lanes() {
    return constmin(HWY_MAX_LANES_D(hn::ScalableTag<IN1_T>),
                    constmin(HWY_MAX_LANES_D(hn::ScalableTag<IN0_T>),
                             HWY_MAX_LANES_D(hn::ScalableTag<OUT_T>)));
  }

  using I0T = hn::CappedTag<IN0_T, lanes()>;
  using I1T = hn::CappedTag<IN1_T, lanes()>;
  using OT = hn::CappedTag<OUT_T, lanes()>;
  I0T d_in0;
  I1T d_in1;
  OT d_out;

  inline hn::Vec<OT> func(int idx, hn::Vec<I0T> x0, hn::Vec<I1T> x1,
                          hn::Vec<OT> y) {
    return me()->func(idx, x0, x1, y);
  }

  inline hn::Vec<I0T> x0_init() { return me()->x0_init_impl(); }

  inline hn::Vec<I0T> x0_init_impl() { return hn::Zero(d_in0); }

  inline hn::Vec<I1T> x1_init() { return me()->x1_init_impl(); }

  inline hn::Vec<I0T> x1_init_impl() { return hn::Zero(d_in1); }

  inline hn::Vec<OT> y_init() { return me()->y_init_impl(); }

  inline hn::Vec<OT> y_init_impl() { return hn::Zero(d_out); }

  inline hn::Vec<I0T> load0(int idx, IN0_T* from) {
    return me()->load0_impl(idx, from);
  }

  inline hn::Vec<I0T> load0_impl(int idx, IN0_T* from) {
    return hn::LoadU(d_in0, from + idx);
  }

  inline hn::Vec<I1T> load1(int idx, IN1_T* from) {
    return me()->load1_impl(idx, from);
  }

  inline hn::Vec<I1T> load1_impl(int idx, IN1_T* from) {
    return hn::LoadU(d_in1, from + idx);
  }

  // maskload can take in either a positive or negative number for `places`. if
  // the number is positive, then it loads the top `places` values, and if it's
  // negative, it loads the bottom |places| values. example: places = 3
  //      | o | o | o | x | x | x | x | x |
  // example places = -3
  //      | x | x | x | x | x | o | o | o |
  inline hn::Vec<I0T> maskload0(int idx, IN0_T* from, int places) {
    return me()->maskload0_impl(idx, from, places);
  }

  inline hn::Vec<I0T> maskload0_impl(int idx, IN0_T* from, int places) {
    const hn::ScalableTag<int> di;
    using TI = hn::TFromD<decltype(di)>;
    auto mask = hn::RebindMask(
        d_in0, hn::detail::Iota0(d_in0) < Set(d_in0, static_cast<TI>(places)));
    auto maskneg = hn::RebindMask(
        d_in0, hn::detail::Iota0(d_in0) >=
                   Set(d_in0, static_cast<TI>(places + lanes())));
    if (places < 0) mask = maskneg;

    return hn::MaskedLoad(mask, d_in0, from + idx);
  }

  inline hn::Vec<I1T> maskload1(int idx, IN1_T* from, int places) {
    return me()->maskload1_impl(idx, from, places);
  }

  inline hn::Vec<I1T> maskload1_impl(int idx, IN1_T* from, int places) {
    const hn::ScalableTag<int> di;
    using TI = hn::TFromD<decltype(di)>;
    auto mask = hn::RebindMask(
        d_in1, hn::detail::Iota0(d_in1) < Set(d_in1, static_cast<TI>(places)));
    auto maskneg = hn::RebindMask(
        d_in1, hn::detail::Iota0(d_in1) >=
                   Set(d_in1, static_cast<TI>(places + lanes())));
    if (places < 0) mask = maskneg;

    return hn::MaskedLoad(mask, d_in1, from + idx);
  }

  inline int store(int idx, OUT_T* to, hn::Vec<OT> x) {
    return me()->store_impl(idx, to, x);
  }

  inline int store_impl(int idx, OUT_T* to, hn::Vec<OT> x) {
    hn::StoreU(x, d_out, to + idx);
    return d_out.MaxLanes();
  }

  inline int maskstore(int idx, OUT_T* to, hn::Vec<OT> x, int places) {
    return me()->maskstore_impl(idx, to, x, places);
  }

  inline int maskstore_impl(int idx, OUT_T* to, hn::Vec<OT> x, int places) {
    const hn::ScalableTag<int> di;
    using TI = hn::TFromD<decltype(di)>;
    auto mask = hn::RebindMask(
        d_out, hn::detail::Iota0(d_out) < Set(d_out, static_cast<TI>(places)));
    auto maskneg = hn::RebindMask(
        d_out, hn::detail::Iota0(d_out) >=
                   Set(d_out, static_cast<TI>(places + lanes())));
    if (places < 0) mask = maskneg;

    hn::BlendedStore(x, mask, d_out, to + idx);
    return std::abs(places);
  }

  inline int reduce(hn::Vec<OT> x, OUT_T* to) {
    return me()->reduce_impl(x, to);
  }

  inline int reduce_impl(hn::Vec<OT> x, OUT_T* to) {
    // default does nothing
    return 0;
  }

  inline void reduce(hn::Vec<OT> x0, hn::Vec<OT> x1, hn::Vec<OT> x2,
                     hn::Vec<OT>& y) {
    me()->reduce_impl(x0, x1, x2, y);
  }

  inline void reduce_impl(hn::Vec<OT> x0, hn::Vec<OT> x1, hn::Vec<OT> x2,
                          hn::Vec<OT>& y) {
    // default does nothing
  }
};

template <class FUNC, typename IN_T, typename OUT_T>
inline void unroller(FUNC& f, IN_T* x, OUT_T* y, const unsigned int n) {
  const auto lane_sz = FUNC::lanes();

  auto xx = f.x0_init();
  auto yy = f.y_init();
  int i = 0;

#if HWY_MEM_OPS_MIGHT_FAULT
  if (n < lane_sz) {
    // stack is maybe too small for this in RVV?
    IN_T xtmp[lane_sz];
    OUT_T ytmp[lane_sz];

    memcpy(xtmp, x, n * sizeof(IN_T));
    xx = f.maskload(0, xtmp, n);
    yy = f.func(0, xx, yy);
    i += f.maskstore(0, ytmp, yy, n);
    i += f.reduce(yy, ytmp);
    memcpy(y, ytmp, i * sizeof(OUT_T));
    return;
  }
#endif

  if (n > 4 * lane_sz) {
    auto xx1 = f.x0_init();
    auto yy1 = f.y_init();
    auto xx2 = f.x0_init();
    auto yy2 = f.y_init();
    auto xx3 = f.x0_init();
    auto yy3 = f.y_init();

    while (i + 4 * lane_sz - 1 < n) {
      xx = f.load(i, x);
      i += lane_sz;
      xx1 = f.load(i, x);
      i += lane_sz;
      xx2 = f.load(i, x);
      i += lane_sz;
      xx3 = f.load(i, x);
      i -= 3 * lane_sz;

      yy = f.func(i, xx, yy);
      yy1 = f.func(i + lane_sz, xx1, yy1);
      yy2 = f.func(i + 2 * lane_sz, xx2, yy2);
      yy3 = f.func(i + 3 * lane_sz, xx3, yy3);

      f.store(i, y, yy);
      i += lane_sz;
      f.store(i, y, yy1);
      i += lane_sz;
      f.store(i, y, yy2);
      i += lane_sz;
      f.store(i, y, yy3);
      i += lane_sz;
    }

    f.reduce(yy3, yy2, yy1, yy);
  }

  while (i + lane_sz - 1 < n) {
    xx = f.load(i, x);
    yy = f.func(i, xx, yy);
    f.store(i, y, yy);
    i += lane_sz;
  }

  if (i != n) {
    xx = f.maskload(n - lane_sz, x, i - n);
    yy = f.func(n - lane_sz, xx, yy);
    f.maskstore(n - lane_sz, y, yy, i - n);
  }

  f.reduce(yy, y);
}

template <class FUNC, typename IN0_T, typename IN1_T, typename OUT_T>
inline void unroller(FUNC& f, IN0_T* x0, IN1_T* x1, OUT_T* y, const int n) {
  const size_t lane_sz = f.lanes();

  auto xx00 = f.x0_init();
  auto xx10 = f.x1_init();
  auto yy = f.y_init();

  int i = 0;

#if HWY_MEM_OPS_MIGHT_FAULT
  if (n < lane_sz) {
    // stack is maybe too small for this in RVV?
    IN0_T xtmp0[lane_sz];
    IN1_T xtmp1[lane_sz];
    OUT_T ytmp[lane_sz];

    memcpy(xtmp0, x0, n * sizeof(IN0_T));
    memcpy(xtmp1, x1, n * sizeof(IN1_T));
    xx00 = f.maskload0(0, xtmp0, n);
    xx10 = f.maskload1(0, xtmp1, n);
    yy = f.func(0, xx00, xx10, yy);
    i += f.maskstore(0, ytmp, yy, n);
    i += f.reduce(yy, ytmp);
    memcpy(y, ytmp, i * sizeof(OUT_T));
    return;
  }
#endif

  if (n > 4 * lane_sz) {
    auto xx01 = f.x0_init();
    auto xx11 = f.x1_init();
    auto yy1 = f.y_init();
    auto xx02 = f.x0_init();
    auto xx12 = f.x1_init();
    auto yy2 = f.y_init();
    auto xx03 = f.x0_init();
    auto xx13 = f.x1_init();
    auto yy3 = f.y_init();

    while (i + 4 * lane_sz - 1 < n) {
      xx00 = f.load0(i, x0);
      xx10 = f.load1(i, x1);
      i += lane_sz;
      xx01 = f.load0(i, x0);
      xx11 = f.load1(i, x1);
      i += lane_sz;
      xx02 = f.load0(i, x0);
      xx12 = f.load1(i, x1);
      i += lane_sz;
      xx03 = f.load0(i, x0);
      xx13 = f.load1(i, x1);
      i -= 3 * lane_sz;

      yy = f.func(i, xx00, xx10, yy);
      yy1 = f.func(i + lane_sz, xx01, xx11, yy1);
      yy2 = f.func(i + 2 * lane_sz, xx02, xx12, yy2);
      yy3 = f.func(i + 3 * lane_sz, xx03, xx13, yy3);

      f.store(i, y, yy);
      i += lane_sz;
      f.store(i, y, yy1);
      i += lane_sz;
      f.store(i, y, yy2);
      i += lane_sz;
      f.store(i, y, yy3);
      i += lane_sz;
    }

    f.reduce(yy3, yy2, yy1, yy);
  }

  while (i + lane_sz - 1 < n) {
    xx00 = f.load0(i, x0);
    xx10 = f.load1(i, x1);
    yy = f.func(i, xx00, xx10, yy);
    f.store(i, y, yy);
    i += lane_sz;
  }

  if (i != n) {
    xx00 = f.maskload0(i, x0, i - n);
    xx10 = f.maskload1(i, x1, i - n);
    yy = f.func(i, xx00, xx10, yy);
    f.maskstore(i, y, yy, i - n);
  }

  f.reduce(yy, y);
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_UNROLLER_UNROLLER_INL_H_