// Emulated floating-point total order
//
// This implementation sorts floating-point values by reinterpreting them as
// unsigned integer bit patterns instead of using the FPU. It does not depend on
// the floating-point control register, so there is no flush-to-zero handling.
//
// NaNs are already replaced by ±Inf before calling this code, so no special
// handling is needed here.
// Because ordering is emulated, we guarantee a stable rule for zeros: -0.0
// always comes before +0.0.
//
// SPDX-License-Identifier: BSD-3-Clause
#if defined(HIGHWAY_HWY_CONTRIB_SORT_ORDER_EMULATE_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_SORT_ORDER_EMULATE_TOGGLE
#undef HIGHWAY_HWY_CONTRIB_SORT_ORDER_EMULATE_TOGGLE
#else
#define HIGHWAY_HWY_CONTRIB_SORT_ORDER_EMULATE_TOGGLE
#endif

#include <stddef.h>
#include <stdint.h>

#include "hwy/contrib/sort/order.h"       // SortDescending
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace detail {


template <class VU, class D = DFromV<VU>, class DI = RebindToSigned<D>>
HWY_API Vec<DI> LtBinKey(VU a) {
  using TI = TFromD<DI>;
  using VI = Vec<DI>;
  const DI di;
  const VI neg_flip = Set(di, TI(SignMask<TI>() - 1));
  return Xor(BitCast(di, a), IfNegativeThenElseZero(BitCast(di, a), neg_flip));
}

template <class VU, class D = DFromV<VU>, class M = MFromD<D>, HWY_IF_UNSIGNED_D(D)>
HWY_API M LtBin(VU a, VU b) {
  return RebindMask(D{}, Lt(LtBinKey(a), LtBinKey(b)));
}

template <class Base, class Order_>
struct OrderEmulate : public Base {
  using T = typename Base::LaneType;
  using TF = typename Base::KeyType;

  HWY_INLINE bool Equal1(const T* a, const T* b) const {
    return *a == *b;
  }

  template <class D>
  HWY_INLINE Mask<D> EqualKeys(D, Vec<D> a, Vec<D> b) const {
    return Eq(a, b); // Bitwise equality, -0 != +0, +-NaN is equal to itself
  }

  template <class D>
  HWY_INLINE Mask<D> NotEqualKeys(D, Vec<D> a, Vec<D> b) const {
    return Ne(a, b); // bitwise inequality, -0 != +0, +-NaN is equal to itself
  }

  HWY_INLINE bool Compare1(const T* a_, const T* b_) const {
    const T a = *a_;
    const T b = *b_;
    // specialized less than, -0.0 < +0.0, and NaNs are not ordered
    using TI = MakeSigned<T>;
    constexpr int kMSB = 8 * sizeof(T) - 1;
    constexpr T neg_flip = T((T(1) << kMSB) - 1); 
    const T a_neg = 0 - (a >> kMSB);
    const T b_neg = 0 - (b >> kMSB);
    // Signed-domain keys (xor 0x7FFF.. only for negatives)
    const T sa = a ^ (a_neg & neg_flip);
    const T sb = b ^ (b_neg & neg_flip);
    return static_cast<TI>(sa) < static_cast<TI>(sb);
  }
  template <class D>
  HWY_INLINE Mask<D> Compare(D, Vec<D> a, Vec<D> b) const {
    // specialized less than, -0.0 < +0.0, and NaNs are not ordered
    return LtBin(a, b);
  }

  // Two halves of Sort2, used in ScanMinMax.
  template <class D>
  HWY_INLINE Vec<D> First(D /* tag */, const Vec<D> a, const Vec<D> b) const {
    return IfThenElse(LtBin(a, b), a, b);
  }

  template <class D>
  HWY_INLINE Vec<D> Last(D /* tag */, const Vec<D> a, const Vec<D> b) const {
    return IfThenElse(LtBin(a, b), b, a);
  }

  template <class D>
  HWY_INLINE Vec<D> FirstOfLanes(D d, Vec<D> v,
                                 T* HWY_RESTRICT /* buf */) const {
    const RebindToSigned<D> di;
    using VI = Vec<decltype(di)>;
    VI key = LtBinKey(v);
    VI min = MinOfLanes(di, key);
    Mask<D> m = RebindMask(d, Eq(min, key));
    return MaxOfLanes(d, IfThenElseZero(m, v));
  }

  template <class D>
  HWY_INLINE Vec<D> LastOfLanes(D d, Vec<D> v,
                                T* HWY_RESTRICT /* buf */) const {
    const RebindToSigned<D> di;
    using VI = Vec<decltype(di)>;
    VI key = LtBinKey(v);
    VI max = MaxOfLanes(di, key);
    Mask<D> m = RebindMask(d, Eq(max, key));
    return MaxOfLanes(d, IfThenElseZero(m, v));
  }

  template <class D>
  HWY_INLINE Vec<D> FirstValue(D d) const {
    return Set(d, BitCastScalar<T>(NegativeInfOrLowestValue<TF>()));
  }

  template <class D>
  HWY_INLINE Vec<D> LastValue(D d) const {
    return Set(d, BitCastScalar<T>(PositiveInfOrHighestValue<TF>()));
  }

  // Returns the next distinct smaller value unless already -inf.
  template <class D, class V = Vec<D>>
  HWY_INLINE V PrevValue(D, V v) const {
    return NextSortValueBits<true>(v);
  }

  // Next representable value in total order by ±1 ULP, saturating at ±Inf.
  //   IsDown = false → next larger
  //   IsDown = true  → next smaller
  template <bool IsDown, class V>
  HWY_INLINE V NextSortValueBits(V u) const {
    const DFromV<V> d;
    using M = Mask<decltype(d)>;
    constexpr T kSignBit = SignMask<T>();
    constexpr T kBoundaryUp = SignMask<T>() - 1;
    const V sign_bit = Set(d, kSignBit);
    const V all1 = Set(d, T(~T(0)));
    const V one = Set(d, T(1));
    // Detect saturation at ±Inf
    const M is_target_inf = Eq(u, IsDown ? FirstValue(d) : LastValue(d));
    // Transform to monotonic space: flip sign for positives, invert for negatives
    const M is_neg = TestBit(u, sign_bit);
    const V key = Xor(u, IfThenElse(is_neg, all1, sign_bit));
    // Boundary detection: +0/-0 swap needs a step of 2 instead of 1
    const V boundary = Set(d, IsDown ? kSignBit : kBoundaryUp);
    const M at_boundary = Eq(key, boundary);
    // Step size: normally 1, but 2 at zero-boundary
    const V step = Add(one, IfThenElseZero(at_boundary, one));
    // Apply increment/decrement unless already at ±Inf
    const V key2 = IfThenElse(is_target_inf, key,
                              IsDown ? Sub(key, step) : Add(key, step));
    // Transform back from monotonic space
    const M neg_out = Lt(key2, sign_bit);
    return Xor(key2, IfThenElse(neg_out, all1, sign_bit));
  }
};

template <class Base>
struct OrderEmulate<Base, SortDescending> : public OrderEmulate<Base, SortAscending> {
  using T = typename Base::LaneType;

  HWY_INLINE const OrderEmulate<Base, SortAscending>& AscBase() const {
    return *this;
  }

  HWY_INLINE bool Compare1(const T* a, const T* b) const {
    return AscBase().Compare1(b, a);
  }
  template <class D>
  HWY_INLINE Mask<D> Compare(D d, Vec<D> a, Vec<D> b) const {
    return AscBase().Compare(d, b, a);
  }

  template <class D>
  HWY_INLINE Vec<D> First(D d, const Vec<D> a, const Vec<D> b) const {
    return AscBase().Last(d, a, b);
  }

  template <class D>
  HWY_INLINE Vec<D> Last(D d, const Vec<D> a, const Vec<D> b) const {
    return AscBase().First(d, a, b);
  }

  template <class D>
  HWY_INLINE Vec<D> FirstOfLanes(D d, Vec<D> v,
                                 T* HWY_RESTRICT b) const {
    return AscBase().LastOfLanes(d, v, b);
  }

  template <class D>
  HWY_INLINE Vec<D> LastOfLanes(D d, Vec<D> v,
                                T* HWY_RESTRICT b) const {
    return AscBase().FirstOfLanes(d, v, b);
  }

  template <class D>
  HWY_INLINE Vec<D> FirstValue(D d) const {
    return AscBase().LastValue(d);
  }

  template <class D>
  HWY_INLINE Vec<D> LastValue(D d) const {
    return AscBase().FirstValue(d);
  }

  template <class D, class V = Vec<D>>
  HWY_INLINE V PrevValue(D, V v) const {
    return this->template NextSortValueBits<false>(v);
  }
};

} // namespace detail
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_SORT_ORDER_EMULATE_TOGGLE
