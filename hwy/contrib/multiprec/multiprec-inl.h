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

// Per-target include guard
#if defined(HIGHWAY_HWY_CONTRIB_MULTIPREC_MULTIPREC_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_MULTIPREC_MULTIPREC_INL_H_
#undef HIGHWAY_HWY_CONTRIB_MULTIPREC_MULTIPREC_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_MULTIPREC_MULTIPREC_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// Fixed-width unsigned integer multiplication for arbitrary precision.
//
// Limbs are 52 bits: the largest width for which the IFMA MulAdd52Lo/Hi ops are
// exact, since for inputs < 2^52 the two together return the full 104-bit
// product exactly. They exist on every target (native HWY_NATIVE_MULADD52 where
// available, the generic fallback otherwise), so there is no separate code
// path. The 12 bits left over in each 64-bit word are what GMP calls nails;
// here they are simply unused.
//
// Layout is SoA in both directions: limb i (little-endian) of each operand
// lives at a[i * N .. i * N + N), where N = Lanes(d). The lanes hold
// independent multiplications, which is how crypto (several RSA/ECC instances
// in parallel) uses this.
//
// Operands, results and scratch are arrays of T (= TFromD<D>), not arrays of
// vectors: vector types are sizeless on SVE and RVV, so they cannot be stored
// in arrays. The kernel therefore keeps only a few vector locals live and
// Loads/Stores around them.
//
// This header only multiplies. Reduction, in particular Montgomery, is the next
// layer callers need - that is why the directory is multiprec rather than
// wide_mul - but it is not implemented here yet.

constexpr size_t kWideMulLimbBits = 52;
constexpr uint64_t kWideMulLimbMask = (uint64_t{1} << kWideMulLimbBits) - 1;

namespace detail {

// out = a * b, both kNumLimbs limbs, using Comba's method: for each output limb
// k, sum the products a[i] * b[k-i]. Only three vectors are live at a time (the
// carry and the low/high accumulator pair), so this does not spill even for
// large kNumLimbs, unlike accumulating all 2*kNumLimbs-1 columns at once.
template <size_t kNumLimbs, class D>
HWY_INLINE void CombaMul(D d, const TFromD<D>* HWY_RESTRICT a,
                         const TFromD<D>* HWY_RESTRICT b,
                         TFromD<D>* HWY_RESTRICT out) {
  const size_t N = Lanes(d);
  constexpr size_t kNumOutLimbs = 2 * kNumLimbs;
  const Vec<D> mask = Set(d, kWideMulLimbMask);

  Vec<D> carry = Zero(d);
  for (size_t k = 0; k < kNumOutLimbs - 1; ++k) {
    // Sum the low and high halves of each product separately; both accumulators
    // fit in 64 bits because there are at most kNumLimbs terms, each < 2^52.
    Vec<D> lo = carry;
    Vec<D> hi = Zero(d);
    const size_t i_begin = k >= kNumLimbs - 1 ? k - (kNumLimbs - 1) : 0;
    const size_t i_end = HWY_MIN(k, kNumLimbs - 1);
    for (size_t i = i_begin; i <= i_end; ++i) {
      const Vec<D> av = LoadU(d, a + i * N);
      const Vec<D> bv = LoadU(d, b + (k - i) * N);
      lo = MulAdd52Lo(av, bv, lo);
      hi = MulAdd52Hi(av, bv, hi);
    }
    StoreU(And(lo, mask), d, out + k * N);
    carry = Add(ShiftRight<kWideMulLimbBits>(lo), hi);
  }
  // The product has at most 104 * kNumLimbs bits, so the remaining carry fits
  // in the top limb.
  StoreU(And(carry, mask), d, out + (kNumOutLimbs - 1) * N);
}

// out[i] = a[i] + b[i] for i < kCount, returning the carry-out (0 or 1 per
// lane) rather than storing it. Both operands must be < 2^52.
template <class D>
HWY_INLINE Vec<D> AddLimbsCarry(D d, const TFromD<D>* HWY_RESTRICT a,
                                const TFromD<D>* HWY_RESTRICT b, size_t kCount,
                                TFromD<D>* HWY_RESTRICT out) {
  const size_t N = Lanes(d);
  const Vec<D> mask = Set(d, kWideMulLimbMask);
  Vec<D> carry = Zero(d);
  for (size_t i = 0; i < kCount; ++i) {
    const Vec<D> sum =
        Add(Add(LoadU(d, a + i * N), LoadU(d, b + i * N)), carry);
    StoreU(And(sum, mask), d, out + i * N);
    carry = ShiftRight<kWideMulLimbBits>(sum);
  }
  return carry;
}

// dst[kOffset + i] += src[i] for i < kCount, clamped so nothing is written at
// or past kTotal, with carry ripple. kOffset indexes the destination, and the
// same convention is used by AddIntoIf and SubInto below.
template <size_t kOffset, size_t kCount, size_t kTotal, class D>
HWY_INLINE void AddInto(D d, TFromD<D>* HWY_RESTRICT dst,
                        const TFromD<D>* HWY_RESTRICT src) {
  const size_t N = Lanes(d);
  constexpr size_t kUsed =
      kCount <= kTotal - kOffset ? kCount : kTotal - kOffset;
  const Vec<D> mask = Set(d, kWideMulLimbMask);
  Vec<D> carry = Zero(d);
  for (size_t i = 0; i < kUsed; ++i) {
    const Vec<D> sum = Add(
        Add(LoadU(d, dst + (kOffset + i) * N), LoadU(d, src + i * N)), carry);
    StoreU(And(sum, mask), d, dst + (kOffset + i) * N);
    carry = ShiftRight<kWideMulLimbBits>(sum);
  }
  for (size_t i = kOffset + kUsed; i < kTotal; ++i) {
    const Vec<D> sum = Add(LoadU(d, dst + i * N), carry);
    StoreU(And(sum, mask), d, dst + i * N);
    carry = ShiftRight<kWideMulLimbBits>(sum);
  }
}

// dst[kOffset + i] += src[i] where `condition` is 1, for i < kCount, with carry
// ripple. `condition` is 0 or 1 per lane, so no branch is taken and secrets do
// not affect control flow.
template <size_t kOffset, size_t kCount, size_t kTotal, class D>
HWY_INLINE void AddIntoIf(D d, TFromD<D>* HWY_RESTRICT dst,
                          const TFromD<D>* HWY_RESTRICT src, Vec<D> condition) {
  const size_t N = Lanes(d);
  constexpr size_t kUsed =
      kCount <= kTotal - kOffset ? kCount : kTotal - kOffset;
  const Vec<D> mask = Set(d, kWideMulLimbMask);
  // 0 - condition is either 0 or all-ones, which turns the addend on or off.
  const Vec<D> select = Sub(Zero(d), condition);
  Vec<D> carry = Zero(d);
  for (size_t i = 0; i < kUsed; ++i) {
    const Vec<D> addend = And(LoadU(d, src + i * N), select);
    const Vec<D> sum =
        Add(Add(LoadU(d, dst + (kOffset + i) * N), addend), carry);
    StoreU(And(sum, mask), d, dst + (kOffset + i) * N);
    carry = ShiftRight<kWideMulLimbBits>(sum);
  }
  for (size_t i = kOffset + kUsed; i < kTotal; ++i) {
    const Vec<D> sum = Add(LoadU(d, dst + i * N), carry);
    StoreU(And(sum, mask), d, dst + i * N);
    carry = ShiftRight<kWideMulLimbBits>(sum);
  }
}

// dst[kOffset + i] -= src[i] for i < kCount, zero-extending src; the result
// must be non-negative. Same kOffset convention as AddInto.
template <size_t kOffset, size_t kCount, size_t kTotal, class D>
HWY_INLINE void SubInto(D d, TFromD<D>* HWY_RESTRICT dst,
                        const TFromD<D>* HWY_RESTRICT src) {
  const size_t N = Lanes(d);
  const Vec<D> mask = Set(d, kWideMulLimbMask);
  Vec<D> borrow = Zero(d);
  for (size_t i = 0; i < kTotal; ++i) {
    const Vec<D> sv = i < kCount ? LoadU(d, src + i * N) : Zero(d);
    const Vec<D> sub = Add(sv, borrow);
    const Vec<D> diff = Sub(LoadU(d, dst + (kOffset + i) * N), sub);
    // Both operands are below 2^52, so an underflow sets bit 63 of the wrapped
    // difference: no comparison (and hence no mask) is needed.
    borrow = ShiftRight<63>(diff);
    StoreU(And(diff, mask), d, dst + (kOffset + i) * N);
  }
}

// Karatsuba pays off above this many limbs, and only for even counts (see
// MulRec).
constexpr size_t kWideMulKaratsubaMinLimbs = 8;

// Scratch limbs per lane that Karatsuba needs for a given size. The layout
// below is z0 and z2 (kMid = kNumLimbs limbs each), s and t (kHalf each), st
// (kMid + 2) and one limb holding the constant 1, i.e. 4*kNumLimbs + 3 in
// total; the child's requirement is added once because the three recursive
// products run one after another. Zero when the size does not use Karatsuba.
template <size_t kNumLimbs>
constexpr size_t MulScratchLimbs() {
  return (kNumLimbs < kWideMulKaratsubaMinLimbs || (kNumLimbs & 1) != 0)
             ? 0
             : 4 * kNumLimbs + 4 + MulScratchLimbs<kNumLimbs / 2>();
}

template <size_t kNumLimbs, class D>
HWY_INLINE void MulRec(D d, const TFromD<D>* HWY_RESTRICT a,
                       const TFromD<D>* HWY_RESTRICT b,
                       TFromD<D>* HWY_RESTRICT out,
                       TFromD<D>* HWY_RESTRICT scratch);

// out = a * b by Karatsuba: split into two halves of kHalf limbs and recurse at
// kHalf, never at kHalf + 1, so the recursion stays on the same size and the
// base case is Comba. The carries of a_lo + a_hi and b_lo + b_hi are tracked
// separately (as 0/1 vectors) and folded back in with masked adds, instead of
// growing the summed operands to kHalf + 1 limbs.
template <size_t kNumLimbs, class D>
HWY_INLINE void KaratsubaMul(D d, const TFromD<D>* HWY_RESTRICT a,
                             const TFromD<D>* HWY_RESTRICT b,
                             TFromD<D>* HWY_RESTRICT out,
                             TFromD<D>* HWY_RESTRICT scratch) {
  constexpr size_t kHalf = kNumLimbs / 2;
  constexpr size_t kMid = 2 * kHalf;
  constexpr size_t kTotal = 2 * kNumLimbs;
  const size_t N = Lanes(d);

  // All buffers are arrays of T, which is what makes this portable to targets
  // whose vector types are sizeless:
  //   z0, z2: kMid limbs each; s, t: kHalf each; st: kMid + 2; one: 1 limb;
  //   child: the child's own scratch.
  TFromD<D>* const z0 = scratch;
  TFromD<D>* const z2 = z0 + kMid * N;
  TFromD<D>* const s = z2 + kMid * N;
  TFromD<D>* const t = s + kHalf * N;
  TFromD<D>* const st = t + kHalf * N;
  TFromD<D>* const one = st + (kMid + 2) * N;
  TFromD<D>* const child = one + N;

  // z0 = a_lo * b_lo, z2 = a_hi * b_hi.
  MulRec<kHalf>(d, a, b, z0, child);
  MulRec<kHalf>(d, a + kHalf * N, b + kHalf * N, z2, child);

  // s = a_lo + a_hi, t = b_lo + b_hi, kHalf limbs each, with the carry kept
  // aside rather than stored.
  const Vec<D> carry_a = AddLimbsCarry(d, a, a + kHalf * N, kHalf, s);
  const Vec<D> carry_b = AddLimbsCarry(d, b, b + kHalf * N, kHalf, t);

  // st = s * t, and the two limbs above kMid start at zero.
  MulRec<kHalf>(d, s, t, st, child);
  const Vec<D> zero = Zero(d);
  for (size_t i = kMid; i < kMid + 2; ++i) {
    StoreU(zero, d, st + i * N);
  }

  // z1 = s*t - z0 - z2 = a_lo*b_hi + a_hi*b_lo, at limb 0 of st.
  SubInto<0, kMid, kMid + 2>(d, st, z0);
  SubInto<0, kMid, kMid + 2>(d, st, z2);

  // s*t also contains carry_a * t * B^kHalf, carry_b * s * B^kHalf and
  // carry_a * carry_b * B^kMid, which the two subtractions above did not remove
  // because they were not part of z0 or z2.
  AddIntoIf<kHalf, kHalf, kMid + 2>(d, st, t, carry_a);
  AddIntoIf<kHalf, kHalf, kMid + 2>(d, st, s, carry_b);
  StoreU(Set(d, uint64_t{1}), d, one);
  AddIntoIf<kMid, 1, kMid + 2>(d, st, one, And(carry_a, carry_b));

  // out = z0 + z1 * B^kHalf + z2 * B^kMid.
  for (size_t i = 0; i < kTotal * N; ++i) out[i] = TFromD<D>{0};
  AddInto<0, kMid, kTotal>(d, out, z0);
  AddInto<kMid, kMid, kTotal>(d, out, z2);
  AddInto<kHalf, kMid + 2, kTotal>(d, out, st);
}

// Dispatches to Karatsuba when the size is even and large enough and scratch is
// available, else Comba. `if constexpr` keeps Karatsuba out of odd sizes, whose
// split would not be balanced.
template <size_t kNumLimbs, class D>
HWY_INLINE void MulRec(D d, const TFromD<D>* HWY_RESTRICT a,
                       const TFromD<D>* HWY_RESTRICT b,
                       TFromD<D>* HWY_RESTRICT out,
                       TFromD<D>* HWY_RESTRICT scratch) {
  if constexpr (kNumLimbs >= kWideMulKaratsubaMinLimbs &&
                (kNumLimbs & 1) == 0) {
    if (scratch != nullptr) {
      KaratsubaMul<kNumLimbs>(d, a, b, out, scratch);
      return;
    }
  }
  CombaMul<kNumLimbs>(d, a, b, out);
}

}  // namespace detail

// Multiplies two kNumLimbs-limb unsigned integers into 2*kNumLimbs limbs. `d`
// must be a u64 vector descriptor; `a`, `b` and `out` point to arrays of at
// least kNumLimbs*N, kNumLimbs*N and 2*kNumLimbs*N elements respectively, N
// being Lanes(d). All input limbs must be < 2^52, as MulAdd52* requires; only
// the low 52 bits of each are used. `out` must not overlap `a` or `b`.
//
// kNumLimbs is limited so that the per-limb accumulators cannot overflow 64
// bits: each holds at most kNumLimbs products < 2^52, plus the carry from the
// previous limb, which is itself below kNumLimbs * (2^52 + 1). 1024 therefore
// leaves ample headroom below 2^64.
template <size_t kNumLimbs>
struct WideMul {
  static_assert(kNumLimbs >= 1, "need at least one limb");
  static_assert(kNumLimbs <= 1024, "accumulators may overflow");
  static constexpr size_t kNumResultLimbs = 2 * kNumLimbs;
  // Limbs of T per lane required by the Karatsuba overload below; zero when
  // this size does not use Karatsuba.
  static constexpr size_t kScratchLimbs = detail::MulScratchLimbs<kNumLimbs>();

  // Comba: no scratch, always available. This is also what the recursive calls
  // fall back to for small or odd sizes.
  template <class D, HWY_IF_U64_D(D)>
  static HWY_INLINE void Mul(D d, const TFromD<D>* HWY_RESTRICT a,
                             const TFromD<D>* HWY_RESTRICT b,
                             TFromD<D>* HWY_RESTRICT out) {
    detail::CombaMul<kNumLimbs>(d, a, b, out);
  }

  // Karatsuba: fewer multiplies above kWideMulKaratsubaMinLimbs limbs, but the
  // caller must supply scratch of kScratchLimbs * Lanes(d) elements. Nothing is
  // allocated here, and `scratch` must not overlap `a`, `b` or `out`.
  template <class D, HWY_IF_U64_D(D)>
  static HWY_INLINE void Mul(D d, const TFromD<D>* HWY_RESTRICT a,
                             const TFromD<D>* HWY_RESTRICT b,
                             TFromD<D>* HWY_RESTRICT out,
                             TFromD<D>* HWY_RESTRICT scratch) {
    static_assert(kScratchLimbs != 0,
                  "this size does not use Karatsuba; call Mul without scratch");
    detail::KaratsubaMul<kNumLimbs>(d, a, b, out, scratch);
  }
};

// Bit-width form of the API above: kNumLimbs = ceil(kBits / 52), so e.g.
// `WideMulBits<256>` multiplies 256-bit values held as 5 limbs each.
template <size_t kBits>
using WideMulBits = WideMul<(kBits + kWideMulLimbBits - 1) / kWideMulLimbBits>;

namespace detail {}  // namespace detail

// ------------------------------ Montgomery

// Returns -x^{-1} mod 2^52 for every lane, given the low limb of the modulus,
// which must be odd. This is the constant that makes the low limb of T + m*N
// vanish, so callers compute it once per modulus and reuse it for every
// multiplication.
template <class D, HWY_IF_U64_D(D)>
HWY_INLINE Vec<D> MontgomeryN0(D d, Vec<D> n0) {
  const Vec<D> mask = Set(d, kWideMulLimbMask);
  const Vec<D> zero = Zero(d);
  const Vec<D> two = Set(d, 2);
  // Newton iteration for the inverse modulo 2^52. Each step doubles the number
  // of correct bits, starting from 3 (x*x = 1 mod 8 for odd x), so six steps
  // cover 3, 6, 12, 24, 48 and 96 bits. MulAdd52Lo is the low 52 bits of a
  // product, which is exact here because every value is below 2^52.
  Vec<D> x = n0;
  for (int i = 0; i < 6; ++i) {
    const Vec<D> t = MulAdd52Lo(n0, x, zero);  // n0 * x mod 2^52
    // 2 - t mod 2^52: negate modulo 2^64, add, then mask.
    const Vec<D> complement = And(Add(Sub(zero, t), two), mask);
    x = MulAdd52Lo(x, complement, zero);
  }
  return And(Sub(zero, x), mask);  // -x mod 2^52
}

namespace detail {

// out = a * b * R^{-1} mod n by CIOS (Koc): alternate one row of the product
// with one step of the reduction, so the working array is only kNumLimbs + 2
// limbs and no full 2*kNumLimbs product is needed. Scratch is kTotal limbs for
// the accumulator plus kNumLimbs for the tentative difference.
template <size_t kNumLimbs, class D>
HWY_INLINE void CiosMul(D d, const TFromD<D>* HWY_RESTRICT a,
                        const TFromD<D>* HWY_RESTRICT b,
                        const TFromD<D>* HWY_RESTRICT n, Vec<D> n0,
                        TFromD<D>* HWY_RESTRICT out,
                        TFromD<D>* HWY_RESTRICT scratch) {
  constexpr size_t kTotal = kNumLimbs + 2;
  const size_t N = Lanes(d);
  const Vec<D> mask = Set(d, kWideMulLimbMask);
  const Vec<D> zero = Zero(d);

  TFromD<D>* const t = scratch;
  TFromD<D>* const diff = t + kTotal * N;
  for (size_t j = 0; j < kTotal; ++j) {
    StoreU(zero, d, t + j * N);
  }

  for (size_t i = 0; i < kNumLimbs; ++i) {
    const Vec<D> bv = LoadU(d, b + i * N);
    // t += a * b[i], with the low and high halves of each product separate.
    Vec<D> carry = zero;
    for (size_t j = 0; j < kNumLimbs; ++j) {
      const Vec<D> av = LoadU(d, a + j * N);
      const Vec<D> lo = MulAdd52Lo(av, bv, LoadU(d, t + j * N));
      const Vec<D> sum = Add(lo, carry);
      StoreU(And(sum, mask), d, t + j * N);
      carry = Add(ShiftRight<kWideMulLimbBits>(sum), MulAdd52Hi(av, bv, zero));
    }
    {
      const Vec<D> sum = Add(LoadU(d, t + kNumLimbs * N), carry);
      StoreU(And(sum, mask), d, t + kNumLimbs * N);
      StoreU(ShiftRight<kWideMulLimbBits>(sum), d, t + (kNumLimbs + 1) * N);
    }

    // m = t[0] * n0 mod 2^52, then t += m * n and shift down by one limb. The
    // low limb of t cancels exactly by construction of n0, so it is dropped,
    // but its sum can still carry into the next limb; that is why `lo` below
    // adds t[0] rather than starting from zero.
    const Vec<D> m = MulAdd52Lo(LoadU(d, t), n0, zero);
    const Vec<D> n0v = LoadU(d, n);
    {
      const Vec<D> lo = MulAdd52Lo(m, n0v, LoadU(d, t));
      carry = Add(ShiftRight<kWideMulLimbBits>(lo), MulAdd52Hi(m, n0v, zero));
    }
    for (size_t j = 1; j < kNumLimbs; ++j) {
      const Vec<D> nv = LoadU(d, n + j * N);
      const Vec<D> lo = MulAdd52Lo(m, nv, LoadU(d, t + j * N));
      const Vec<D> sum = Add(lo, carry);
      StoreU(And(sum, mask), d, t + (j - 1) * N);
      carry = Add(ShiftRight<kWideMulLimbBits>(sum), MulAdd52Hi(m, nv, zero));
    }
    {
      const Vec<D> sum = Add(LoadU(d, t + kNumLimbs * N), carry);
      StoreU(And(sum, mask), d, t + (kNumLimbs - 1) * N);
      StoreU(Add(LoadU(d, t + (kNumLimbs + 1) * N),
                 ShiftRight<kWideMulLimbBits>(sum)),
             d, t + kNumLimbs * N);
    }
  }

  // r = t[0..kNumLimbs) with an extra top limb is below 2n, so one conditional
  // subtraction suffices when n < R/2; a second pass makes it correct for any
  // odd n below R, where the value is then below 3n. Both take the borrow from
  // bit 63 of the wrapped difference, as SubInto does.
  for (int pass = 0; pass < 2; ++pass) {
    Vec<D> borrow = zero;
    for (size_t j = 0; j < kNumLimbs; ++j) {
      const Vec<D> sub = Add(LoadU(d, n + j * N), borrow);
      const Vec<D> d_j = Sub(LoadU(d, t + j * N), sub);
      borrow = ShiftRight<63>(d_j);
      StoreU(And(d_j, mask), d, diff + j * N);
    }
    const Vec<D> top = Sub(LoadU(d, t + kNumLimbs * N), borrow);
    // The top limb is zero or one, but keep the invariant of limbs < 2^52.
    const Vec<D> top_masked = And(top, mask);
    const Mask<D> keep_diff = Eq(ShiftRight<63>(top), zero);
    for (size_t j = 0; j < kNumLimbs; ++j) {
      StoreU(IfThenElse(keep_diff, LoadU(d, diff + j * N), LoadU(d, t + j * N)),
             d, t + j * N);
    }
    StoreU(IfThenElse(keep_diff, top_masked, LoadU(d, t + kNumLimbs * N)), d,
           t + kNumLimbs * N);
  }

  for (size_t j = 0; j < kNumLimbs; ++j) {
    StoreU(And(LoadU(d, t + j * N), mask), d, out + j * N);
  }
}

}  // namespace detail

// Montgomery modular multiplication, mirroring WideMul: the same limb layout
// (limb i of an operand at [i * Lanes(d)]), and a and b must be below n. The
// result is a * b * R^{-1} mod n with R = 2^(52 * kNumLimbs), which is below n
// when n < R. `n` must be odd, and n0 must come from MontgomeryN0 applied to
// its low limb: compute that once per modulus and reuse it.
//
// To multiply in Montgomery form, represent x as x * R mod n (which the caller
// computes once per value, e.g. from R^2 mod n) and multiply the
// representations.
template <size_t kNumLimbs>
struct Montgomery {
  static_assert(kNumLimbs >= 1, "need at least one limb");
  static_assert(kNumLimbs <= 1024, "accumulators may overflow");
  static constexpr size_t kNumResultLimbs = kNumLimbs;
  // The accumulator (kNumLimbs + 2 limbs) and the tentative difference.
  static constexpr size_t kScratchLimbs = 2 * kNumLimbs + 2;

  template <class D, HWY_IF_U64_D(D)>
  static HWY_INLINE void Mul(D d, const TFromD<D>* HWY_RESTRICT a,
                             const TFromD<D>* HWY_RESTRICT b,
                             const TFromD<D>* HWY_RESTRICT n, Vec<D> n0,
                             TFromD<D>* HWY_RESTRICT out,
                             TFromD<D>* HWY_RESTRICT scratch) {
    detail::CiosMul<kNumLimbs>(d, a, b, n, n0, out, scratch);
  }
};

// Bit-width form of the API above: kNumLimbs = ceil(kBits / 52).
template <size_t kBits>
using MontgomeryBits =
    Montgomery<(kBits + kWideMulLimbBits - 1) / kWideMulLimbBits>;

// out = R^2 mod n, with R = 2^(52 * kNumLimbs). This is the constant needed to
// enter Montgomery form, so the caller can convert without any modular
// arithmetic of its own:
//
//   ToMontgomery(x)   = Montgomery<>::Mul(x, r2, ...)
//   FromMontgomery(y) = Montgomery<>::Mul(y, one, ...)   // one = {1, 0, ...}
//
// n must be odd and below R, as for Montgomery<>::Mul. Both conversions are
// then Montgomery products, and no separate modular multiplication is required.
//
// Computed as 104*kNumLimbs modular doublings starting from 1, which is
// (2^(52*kNumLimbs))^2 mod n. Like the reduction, it is branch-free: the carry
// out of the top limb and the borrow of the conditional subtraction are
// combined into one per-lane select. Scratch is 2*kNumLimbs limbs per lane.
template <size_t kNumLimbs, class D, HWY_IF_U64_D(D)>
HWY_INLINE void MontgomeryR2(D d, const TFromD<D>* HWY_RESTRICT n,
                             TFromD<D>* HWY_RESTRICT out,
                             TFromD<D>* HWY_RESTRICT scratch) {
  static_assert(kNumLimbs >= 1, "need at least one limb");
  const size_t N = Lanes(d);
  const Vec<D> mask = Set(d, kWideMulLimbMask);
  const Vec<D> zero = Zero(d);

  TFromD<D>* const r = scratch;
  TFromD<D>* const diff = r + kNumLimbs * N;
  for (size_t j = 0; j < kNumLimbs; ++j) {
    StoreU(j == 0 ? Set(d, 1) : zero, d, r + j * N);
  }

  for (size_t i = 0; i < 104 * kNumLimbs; ++i) {
    // (carry, r) = 2 * r.
    Vec<D> carry = zero;
    for (size_t j = 0; j < kNumLimbs; ++j) {
      const Vec<D> rv = LoadU(d, r + j * N);
      const Vec<D> sum = Add(Add(rv, rv), carry);
      StoreU(And(sum, mask), d, r + j * N);
      carry = ShiftRight<kWideMulLimbBits>(sum);
    }
    // r -= n when (carry, r) >= n; keep the difference in `diff` meanwhile.
    Vec<D> borrow = zero;
    for (size_t j = 0; j < kNumLimbs; ++j) {
      const Vec<D> sub = Add(LoadU(d, n + j * N), borrow);
      const Vec<D> d_j = Sub(LoadU(d, r + j * N), sub);
      borrow = ShiftRight<63>(d_j);
      StoreU(And(d_j, mask), d, diff + j * N);
    }
    const Vec<D> top = Sub(carry, borrow);
    const Mask<D> keep_diff = Eq(ShiftRight<63>(top), zero);
    for (size_t j = 0; j < kNumLimbs; ++j) {
      StoreU(IfThenElse(keep_diff, LoadU(d, diff + j * N), LoadU(d, r + j * N)),
             d, r + j * N);
    }
  }

  for (size_t j = 0; j < kNumLimbs; ++j) {
    StoreU(LoadU(d, r + j * N), d, out + j * N);
  }
}

// Scratch limbs per lane needed by MontgomeryR2 (the working value and the
// tentative difference).
template <size_t kNumLimbs>
constexpr size_t MontgomeryR2ScratchLimbs() {
  return 2 * kNumLimbs;
}

namespace detail {

// Unpacks kNumLimbs 52-bit limbs from kLimbs little-endian 64-bit limbs.
template <size_t kNumLimbs, size_t kLimbs>
HWY_INLINE void UnpackLimbs52(const uint64_t* HWY_RESTRICT limbs,
                              uint64_t* HWY_RESTRICT out) {
  static_assert((kNumLimbs - 1) * kWideMulLimbBits < kLimbs * 64,
                "the 52-bit limbs do not fit in the 64-bit limbs");
  for (size_t i = 0; i < kNumLimbs; ++i) {
    const size_t bit = i * kWideMulLimbBits;
    const size_t limb = bit / 64;
    const size_t offset = bit % 64;
    uint64_t v = limbs[limb] >> offset;
    if (offset + kWideMulLimbBits > 64) {
      v |= (limb + 1 < kLimbs ? limbs[limb + 1] : uint64_t{0}) << (64 - offset);
    }
    out[i] = v & kWideMulLimbMask;
  }
}

// Packs kNumLimbs 52-bit limbs into kLimbs little-endian 64-bit limbs.
template <size_t kNumLimbs, size_t kLimbs>
HWY_INLINE void PackLimbs52(const uint64_t* HWY_RESTRICT in,
                            uint64_t* HWY_RESTRICT limbs) {
  size_t src = 0;
  size_t src_bit = 0;
  for (size_t j = 0; j < kLimbs; ++j) {
    uint64_t v = 0;
    size_t dst_bit = 0;
    size_t need = 64;
    while (need != 0 && src < kNumLimbs) {
      const size_t avail = kWideMulLimbBits - src_bit;
      const size_t take = HWY_MIN(need, avail);
      const uint64_t part = (in[src] >> src_bit) & ((uint64_t{1} << take) - 1);
      v |= part << dst_bit;
      dst_bit += take;
      need -= take;
      src_bit += take;
      if (src_bit == kWideMulLimbBits) {
        ++src;
        src_bit = 0;
      }
    }
    limbs[j] = v;
  }
}

// Scalar Comba over 52-bit limbs; the same math as the vector kernel, one lane
// at a time.
template <size_t kNumLimbs>
HWY_INLINE void ScalarCombaMul(const uint64_t* HWY_RESTRICT a,
                               const uint64_t* HWY_RESTRICT b,
                               uint64_t* HWY_RESTRICT out) {
  constexpr size_t kNumOutLimbs = 2 * kNumLimbs;
  uint64_t carry = 0;
  for (size_t k = 0; k < kNumOutLimbs - 1; ++k) {
    uint64_t lo = carry;
    uint64_t hi = 0;
    const size_t i_begin = k >= kNumLimbs - 1 ? k - (kNumLimbs - 1) : 0;
    const size_t i_end = HWY_MIN(k, kNumLimbs - 1);
    for (size_t i = i_begin; i <= i_end; ++i) {
      uint64_t p_hi;
      const uint64_t p_lo = Mul128(a[i], b[k - i], &p_hi);
      lo += p_lo & kWideMulLimbMask;
      hi += (p_lo >> kWideMulLimbBits) | (p_hi << (64 - kWideMulLimbBits));
    }
    out[k] = lo & kWideMulLimbMask;
    carry = (lo >> kWideMulLimbBits) + hi;
  }
  out[kNumOutLimbs - 1] = carry & kWideMulLimbMask;
}

}  // namespace detail

// Multiplies two kBits-bit unsigned integers, both given as little-endian
// 64-bit limbs (kBits/64 each), and writes the 2*kBits/64-limb product. This is
// a scalar convenience over the 52-bit limb core, e.g. for u128/u192/u256; it
// does not use the vector unit. kBits must be a multiple of 64.
//
// The top of the result may be a partial 52-bit limb (e.g. a 128-bit product
// has 256 bits, so its 5th limb holds 48 bits); the remaining bits are zero
// because the product fits.
template <size_t kBits>
HWY_INLINE void WideMulLimbs(const uint64_t* HWY_RESTRICT a,
                             const uint64_t* HWY_RESTRICT b,
                             uint64_t* HWY_RESTRICT out) {
  static_assert(kBits % 64 == 0, "kBits must be a multiple of 64");
  constexpr size_t kLimbs = kBits / 64;
  constexpr size_t kNumLimbs =
      (kBits + kWideMulLimbBits - 1) / kWideMulLimbBits;
  uint64_t a52[kNumLimbs];
  uint64_t b52[kNumLimbs];
  uint64_t r52[2 * kNumLimbs];
  detail::UnpackLimbs52<kNumLimbs, kLimbs>(a, a52);
  detail::UnpackLimbs52<kNumLimbs, kLimbs>(b, b52);
  detail::ScalarCombaMul<kNumLimbs>(a52, b52, r52);
  detail::PackLimbs52<2 * kNumLimbs, 2 * kLimbs>(r52, out);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_MULTIPREC_MULTIPREC_INL_H_
