// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause
//
// See the LICENSE file in the project root for the full license text.

#if defined(HIGHWAY_HWY_CONTRIB_DISTRIBUTIONS_UNIFORM_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_DISTRIBUTIONS_UNIFORM_INL_H_
#undef HIGHWAY_HWY_CONTRIB_DISTRIBUTIONS_UNIFORM_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_DISTRIBUTIONS_UNIFORM_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>

#include "hwy/contrib/distributions/uniform.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

#if HWY_HAVE_FLOAT64
namespace distribution_detail {

template <class D>
HWY_INLINE Vec<Rebind<double, D>> UniformFromBits(D /*d*/, Vec<D> bits) {
  const Rebind<double, D> df;
  return Mul(ConvertTo(df, ShiftRight<11>(bits)),
             Set(df, hwy::Uniform::FromBits(uint64_t{1} << 11)));
}

struct UniformSink {
  double* out;

  template <class D>
  HWY_INLINE void operator()(D d, Vec<D> bits, size_t offset,
                             size_t valid_lanes) const {
    const Rebind<double, D> df;
    const auto samples = UniformFromBits(d, bits);
    if (valid_lanes == Lanes(df)) {
      StoreU(samples, df, out + offset);
    } else {
      StoreN(samples, df, out + offset, valid_lanes);
    }
  }
};

}  // namespace distribution_detail

// The same distribution works with scalar u64 and native SIMD backends.
// GenerateBlocks is an optional optimization; scalar backends need only ().
struct Uniform {
  static double FromBits(uint64_t bits) noexcept {
    return hwy::Uniform::FromBits(bits);
  }

  static HWY_INLINE Vec<ScalableTag<double>> FromBits(
      Vec<ScalableTag<uint64_t>> bits) {
    return distribution_detail::UniformFromBits(ScalableTag<uint64_t>(), bits);
  }

  template <class BitGenerator>
  auto operator()(BitGenerator& bits) const -> decltype(FromBits(bits())) {
    return FromBits(bits());
  }

  template <class BitGenerator>
  void Fill(BitGenerator& bits, double* out, size_t count) const {
    FillImpl(bits, out, count, 0);
  }

 private:
  template <class BitGenerator>
  auto FillImpl(BitGenerator& bits, double* out, size_t count, int) const
      -> decltype(bits.GenerateBlocks(count,
                                      distribution_detail::UniformSink{out}),
                  void()) {
    bits.GenerateBlocks(count, distribution_detail::UniformSink{out});
  }

  template <class BitGenerator>
  void FillImpl(BitGenerator& bits, double* out, size_t count, long) const {
    hwy::Uniform().Fill(bits, out, count);
  }
};
#endif  // HWY_HAVE_FLOAT64

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_DISTRIBUTIONS_UNIFORM_INL_H_
