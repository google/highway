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
//
#include <stddef.h>

#include <bitset>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <string>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/examples/mandelbrot.cc"
#include "hwy/aligned_allocator.h"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/timer.h"

/*
Highway SIMD Tutorial: Compute and visualize a Mandelbrot set

This example demonstrates how to compute the Mandelbrot[0] set over a given
domain for a choosen function.  The output is saved in portable pixmap[1] file
format.  Visualization uses a technique inspired by Buddhabrot[2], for each RGB
color, a different maxium iteration is choosen.

Consider the iteration
z_(n+1) = (1-2(z_n)^3)/3(z_n)^2
where z_n is the complex number x_n + i*y_n.
This iteration can also be written as

x_(n+1) = ((x_n^2-y_n^2)*(1-2*(x_n^3-3*x_n*y_n^2))
           -4*x_n*y_n*(3*x_n^2*y_n-y_n^3))
          /(3*(y_n^2+x_n^2)^2)

y_(n+1) = -2*((x_n^2-y_n^2)*(3*x_n^2*y_n-y_n^3)
              +x_n*y_n*(1-2*(x_n^3-3*x_n*y_n^2)))
          /(3*(y_n^2+x_n^2)^2)

A Mandelbrot set is the set of initial starting points which for the value of
x_n^2 + y_n^2 does not tend to infinity.  In a computer implementation, we
choose a maximum iteration number and to avoid overflows, a maxium value after
which we denote the value as tending to infinity.

To visualize the set, three different choices of the maximum iteration
will be choosen for the same predefined maximum value. The resulting arrays
for Mandelbrot sets will be assigned to the red, blue and green colors to create
a color image.  If the point has not escaped, it will be assigned a value of 0,
otherwise it will be assigned a value of

255 * ( maximum number of iterations - iteration of escape) / (maximum number
interations)

rounded up to the nearest integer.

References:
0) https://en.wikipedia.org/wiki/Mandelbrot_set
1) https://en.wikipedia.org/wiki/Netpbm
2) https://en.wikipedia.org/wiki/Buddhabrot
3) https://paulbourke.net/fractals/

*/

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

using DU8 = hn::ScalableTag<uint8_t>;
const DU8 du8;
using MU8 = hn::Mask<decltype(du8)>;
using VU8 = hn::Vec<DU8>;
using DU16 = hn::ScalableTag<uint16_t>;
const DU16 du16;
using MU16 = hn::Mask<decltype(du16)>;
using VU16 = hn::Vec<DU16>;
using DU32 = hn::ScalableTag<uint32_t>;
const DU32 du32;
using MU32 = hn::Mask<decltype(du32)>;
using VU32 = hn::Vec<DU32>;
using DF = hn::ScalableTag<float>;
const DF df;
using MF = hn::Mask<decltype(df)>;
using VF = hn::Vec<DF>;

void MandelbrotScalarSetup(uint8_t* HWY_RESTRICT r, uint8_t* HWY_RESTRICT g,
                           uint8_t* HWY_RESTRICT b, float* HWY_RESTRICT x,
                           float* HWY_RESTRICT y, const size_t x_points,
                           const size_t y_points) {
  float x_max = static_cast<float>(x_points / 2);
  float y_max = static_cast<float>(y_points / 2);
  for (size_t ij = 0; ij < x_points * y_points; ij++) {
    float itemp = static_cast<float>(ij % y_points);
    float i = itemp - x_max;
    float j =
        ((static_cast<float>(ij) - i) / static_cast<float>(y_points)) - y_max;
    // Domain size [-3,3]x[-3,3]
    x[ij] = 3.0f * i / x_max;
    y[ij] = 3.0f * j / y_max;
    r[ij] = 0;
    g[ij] = 0;
    b[ij] = 0;
  }
}

void MandelbrotSimdSetup(uint8_t* HWY_RESTRICT r, uint8_t* HWY_RESTRICT g,
                         uint8_t* HWY_RESTRICT b, float* HWY_RESTRICT x,
                         float* HWY_RESTRICT y, const size_t x_points,
                         const size_t y_points) {
  size_t x_max = x_points / 2;
  size_t y_max = y_points / 2;
  size_t ij = 0;
  float y_points_f = static_cast<float>(y_points);
  float x_max_f = static_cast<float>(x_max);
  float y_max_f = static_cast<float>(y_max);
  size_t remainder;
  const size_t NF = Lanes(df);
  const size_t NU8 = Lanes(du8);
  for (; ij < x_points * y_points; ij += NF) {
    VU32 vij = hn::Iota(du32, ij);
    VU32 itemp_u32 =
        hn::Mod(vij, hn::Set(du32, static_cast<uint32_t>(y_points)));
    VF itemp = hn::ConvertTo(df, itemp_u32);
    VF i = hn::Sub(itemp, hn::Set(df, x_max_f));
    VF j = hn::Sub(
        hn::Div(hn::Sub(hn::ConvertTo(df, vij), i), hn::Set(df, y_points_f)),
        hn::Set(df, y_max_f));
    // Domain size [-3,3]x[-3,3]
    hn::Store(hn::Mul(hn::Set(df, (3.0f / x_max_f)), i), df, x + ij);
    hn::Store(hn::Mul(hn::Set(df, (3.0f / y_max_f)), j), df, y + ij);
  }
  // Use StoreN for remainder, does not access values beyond remainder
  remainder = x_points * y_points - ij;
  if (remainder > 0) {
    VU32 vij = hn::Iota(du32, ij);
    VU32 itemp_u32 =
        hn::Mod(vij, hn::Set(du32, static_cast<uint32_t>(y_points)));
    VF itemp = hn::ConvertTo(df, itemp_u32);
    VF i = hn::Sub(itemp, hn::Set(df, x_max_f));
    VF j = hn::Sub(hn::Div(hn::Sub(hn::ConvertTo(df, vij), itemp),
                           hn::Set(df, y_points_f)),
                   hn::Set(df, y_max_f));
    // Domain size [-3,3]x[-3,3]
    hn::StoreN(hn::Mul(hn::Set(df, (3.0f / x_max_f)), i), df, x + ij,
               remainder);
    hn::StoreN(hn::Mul(hn::Set(df, (3.0f / y_max_f)), j), df, y + ij,
               remainder);
  }
  ij = 0;
  for (; ij < x_points * y_points; ij += NU8) {
    hn::Store(hn::Zero(du8), du8, r + ij);
    hn::Store(hn::Zero(du8), du8, g + ij);
    hn::Store(hn::Zero(du8), du8, b + ij);
  }
  // Use StoreN for remainder, does not access values beyond remainder
  remainder = x_points * y_points - ij;
  if (remainder > 0) {
    hn::StoreN(hn::Zero(du8), du8, r + ij, remainder);
    hn::StoreN(hn::Zero(du8), du8, g + ij, remainder);
    hn::StoreN(hn::Zero(du8), du8, b + ij, remainder);
  }
}

uint8_t CalculateNextScalar(float* x, float* y, const size_t ij) {
  const float sx = x[ij];
  const float sy = y[ij];
  const float sxx = sx * sx;
  const float syy = sy * sy;
  const float sxy = sx * sy;
  const float sxxx = sxx * sx;
  const float syyy = syy * sy;
  const float sxxpyy = sxx + syy;
  const float sxxmyy = sxx - syy;
  /*
  x_(n+1) = ((x_n^2-y_n^2)*(1-2*(x_n^3-3*x_n*y_n^2))
             -4*x_n*y_n*(3*x_n^2*y_n-y_n^3))
           /(3*(y_n^2+x_n^2)^2)

  y_(n+1) = -2*((x_n^2-y_n^2)*(3*x_n^2*y_n-y_n^3)
                +x_n*y_n*(1-2*(x_n^3-3*x_n*y_n^2)))
          /(3*(y_n^2+x_n^2)^2)
  numer1 = (x_n^2-y_n^2)
  numer2 = (1-2*(x_n^3-3*x_n*y_n^2)
  numer3 = (3*x_n^2*y_n-y_n^3)
  denom = (3*(y_n^2+x_n^2)^2)
  */
  const float numer1 = sxxmyy;
  const float numer2 = (1.0f - 2.0f * (sxxx - 3.0f * sx * syy));
  const float numer3 = (3.0f * sxx * sy - syyy);
  const float denom = 3.0f * (sxxpyy * sxxpyy);
  x[ij] = (numer1 * numer2 - 4.0f * sxy * numer3) / denom;
  y[ij] = -2.0f * (numer1 * numer3 + sxy * numer2) / denom;
  // To minimize recomputation, use previous iterates values
  // for determining whether escaped or not
  return static_cast<uint8_t>(std::floor(10.0f * sxxpyy));
}

void MandelbrotScalarCompute(uint8_t* HWY_RESTRICT r, uint8_t* HWY_RESTRICT g,
                             uint8_t* HWY_RESTRICT b, float* HWY_RESTRICT x,
                             float* HWY_RESTRICT y, const size_t x_points,
                             const size_t y_points, const size_t iter_max_r,
                             const size_t iter_max_g, const size_t iter_max_b,
                             const float escape_value) {
  float iter_max_r_f = static_cast<float>(iter_max_r);
  float iter_max_g_f = static_cast<float>(iter_max_g);
  float iter_max_b_f = static_cast<float>(iter_max_b);

  size_t n = 0;
  std::vector<bool> not_escaped(x_points * y_points);
  for (size_t ij = 0; ij < x_points * y_points; ij++) {
    not_escaped[ij] = 1;
  }

  for (; n < iter_max_r; n++) {
    uint8_t r_new = static_cast<uint8_t>(std::ceil(
        255.0f * (iter_max_r_f - static_cast<float>(n)) / iter_max_r_f));
    uint8_t g_new = static_cast<uint8_t>(std::ceil(
        255.0f * (iter_max_g_f - static_cast<float>(n)) / iter_max_g_f));
    uint8_t b_new = static_cast<uint8_t>(std::ceil(
        255.0f * (iter_max_b_f - static_cast<float>(n)) / iter_max_b_f));
    for (size_t ij = 0; ij < x_points * y_points; ij++) {
      if (not_escaped[ij]) {
        uint8_t norm = CalculateNextScalar(x, y, ij);
        if (norm > static_cast<uint8_t>(
                       std::floor(10.0f * escape_value * escape_value))) {
          r[ij] = r_new;
          g[ij] = g_new;
          b[ij] = b_new;
          not_escaped[ij] = 0;
        }
      }
    }
  }

  for (; n < iter_max_g; n++) {
    uint8_t g_new = static_cast<uint8_t>(std::ceil(
        255.0f * (iter_max_g_f - static_cast<float>(n)) / iter_max_g_f));
    uint8_t b_new = static_cast<uint8_t>(std::ceil(
        255.0f * (iter_max_b_f - static_cast<float>(n)) / iter_max_b_f));
    for (size_t ij = 0; ij < x_points * y_points; ij++) {
      if (not_escaped[ij]) {
        uint8_t norm = CalculateNextScalar(x, y, ij);
        if (norm > static_cast<uint8_t>(
                       std::floor(10.0f * escape_value * escape_value))) {
          g[ij] = g_new;
          b[ij] = b_new;
          not_escaped[ij] = 0;
        }
      }
    }
  }

  for (; n < iter_max_b; n++) {
    uint8_t b_new = static_cast<uint8_t>(std::ceil(
        255.0f * (iter_max_b_f - static_cast<float>(n)) / iter_max_b_f));
    for (size_t ij = 0; ij < x_points * y_points; ij++) {
      if (not_escaped[ij]) {
        uint8_t norm = CalculateNextScalar(x, y, ij);
        if (norm > static_cast<uint8_t>(
                       std::floor(10.0f * escape_value * escape_value))) {
          b[ij] = b_new;
          not_escaped[ij] = 0;
        }
      }
    }
  }
}

VU32 CalculateNextSimd(float* HWY_RESTRICT x, float* HWY_RESTRICT y,
                       const size_t ij) {
  const VF vx = hn::Load(df, x + ij);
  const VF vy = hn::Load(df, y + ij);
  const VF v1 = hn::Set(df, 1.0f);
  const VF v2 = hn::Set(df, 2.0f);
  const VF vm2 = hn::Set(df, -2.0f);
  const VF v3 = hn::Set(df, 3.0f);
  const VF v4 = hn::Set(df, 4.0f);
  const VF vxx = hn::Mul(vx, vx);
  const VF vyy = hn::Mul(vy, vy);
  const VF vxy = hn::Mul(vx, vy);
  const VF vxxx = hn::Mul(vxx, vx);
  const VF vyyy = hn::Mul(vyy, vy);
  const VF vxxpyy = hn::Add(vxx, vyy);
  const VF vxxmyy = hn::Sub(vxx, vyy);
  /*
  x_(n+1) = ((x_n^2-y_n^2)*(1-2*(x_n^3-3*x_n*y_n^2))
             -4*x_n*y_n*(3*x_n^2*y_n-y_n^3))
           /(3*(y_n^2+x_n^2)^2)

  y_(n+1) = -2*((x_n^2-y_n^2)*(3*x_n^2*y_n-y_n^3)
                +x_n*y_n*(1-2*(x_n^3-3*x_n*y_n^2)))
          /(3*(y_n^2+x_n^2)^2)
  numer1 = (x_n^2-y_n^2)
  numer2 = (1-2*(x_n^3-3*x_n*y_n^2)
  numer3 = (3*x_n^2*y_n-y_n^3)
  denom = (3*(y_n^2+x_n^2)^2)
  */
  const VF numer1 = vxxmyy;
  const VF numer2 =
      hn::NegMulAdd(v2, hn::NegMulAdd(v3, hn::Mul(vx, vyy), vxxx), v1);
  const VF numer3 = hn::MulSub(v3, hn::Mul(vxx, vy), vyyy);
  const VF denom = hn::Mul(v3, hn::Mul(vxxpyy, vxxpyy));
  // x_(n+1) = (numer1*numer2 - 4*x_n*y_n*numer3) / denom
  const VF x_next = hn::Div(
      hn::MulSub(numer1, numer2, hn::Mul(v4, hn::Mul(vxy, numer3))), denom);
  // y_(n+1) = -2*(numer1*numer3 + x_n*y_n*numer2) / denom
  const VF y_next = hn::Div(
      hn::Mul(vm2, hn::MulAdd(numer1, numer3, hn::Mul(vxy, numer2))), denom);
  hn::Store(x_next, df, x + ij);
  hn::Store(y_next, df, y + ij);
  VF scaling = hn::Set(df, 10.0f);
  // To minimize recomputation, use previous iterates values
  // for determining whether escaped or not
  VU32 scaled_norm = hn::ConvertTo(du32, hn::Floor(hn::Mul(scaling, vxxpyy)));
  return scaled_norm;
}

MU8 GetMask(const size_t ij, const VU32 scaled_norm_0, const VU32 scaled_norm_1,
            const VU32 scaled_norm_2, const VU32 scaled_norm_3,
            const float escape_value, const uint8_t* color,
            std::vector<bool>& HWY_RESTRICT not_escaped) {
  VU16 scaled_norm_u16_01 =
      OrderedTruncate2To(du16, scaled_norm_0, scaled_norm_1);
  VU16 scaled_norm_u16_23 =
      OrderedTruncate2To(du16, scaled_norm_2, scaled_norm_3);
  VU8 scaled_norm_u8 =
      OrderedTruncate2To(du8, scaled_norm_u16_01, scaled_norm_u16_23);
  const VU8 escape_value_u8 = hn::Set(
      du8,
      static_cast<uint8_t>(std::floor(10.0f * escape_value * escape_value)));
  const MU8 mask_escaped = hn::Gt(scaled_norm_u8, escape_value_u8);
  const MU8 mask_previously_not_escaped =
      hn::Eq(hn::Load(du8, color + ij), hn::Zero(du8));
  if (hn::AllTrue(du8, mask_escaped)) {
    not_escaped[ij] = 0;
  }
  return hn::And(mask_escaped, mask_previously_not_escaped);
}

void MandelbrotSimdCompute(uint8_t* HWY_RESTRICT r, uint8_t* HWY_RESTRICT g,
                           uint8_t* HWY_RESTRICT b, float* HWY_RESTRICT x,
                           float* HWY_RESTRICT y, const size_t x_points,
                           const size_t y_points, const size_t iter_max_r,
                           const size_t iter_max_g, const size_t iter_max_b,
                           const float escape_value) {
  size_t n = 0;
  float iter_max_r_f = static_cast<float>(iter_max_r);
  float iter_max_g_f = static_cast<float>(iter_max_g);
  float iter_max_b_f = static_cast<float>(iter_max_b);
  VU32 scaled_norm_0;
  VU32 scaled_norm_1;
  VU32 scaled_norm_2;
  VU32 scaled_norm_3;
  std::vector<bool> not_escaped(x_points * y_points);
  const size_t NF = Lanes(df);
  for (size_t ij = 0; ij < x_points * y_points; ij++) {
    not_escaped[ij] = 1;
  }

  for (; n < iter_max_r; n++) {
    uint8_t r_new = static_cast<uint8_t>(std::ceil(
        255.0f * (iter_max_r_f - static_cast<float>(n)) / iter_max_r_f));
    uint8_t g_new = static_cast<uint8_t>(std::ceil(
        255.0f * (iter_max_g_f - static_cast<float>(n)) / iter_max_g_f));
    uint8_t b_new = static_cast<uint8_t>(std::ceil(
        255.0f * (iter_max_b_f - static_cast<float>(n)) / iter_max_b_f));
    // As x_points and y_points are both even, x_points * y_points
    // is divisible by 4.  This is used to enable conversion from
    // U32 vectors to U8 vectors, which require 1/4 the size
    // registers of U32 vectors
    for (size_t ij = 0; ij < x_points * y_points; ij += 4 * NF) {
      if (not_escaped[ij]) {
        scaled_norm_0 = CalculateNextSimd(x, y, ij);
        scaled_norm_1 = CalculateNextSimd(x, y, ij + NF);
        scaled_norm_2 = CalculateNextSimd(x, y, ij + 2 * NF);
        scaled_norm_3 = CalculateNextSimd(x, y, ij + 3 * NF);
        MU8 mask_newly_escaped =
            GetMask(ij, scaled_norm_0, scaled_norm_1, scaled_norm_2,
                    scaled_norm_3, escape_value, r, not_escaped);
        hn::Store(
            hn::MaskedSetOr(hn::Load(du8, r + ij), mask_newly_escaped, r_new),
            du8, r + ij);
        hn::Store(
            hn::MaskedSetOr(hn::Load(du8, g + ij), mask_newly_escaped, g_new),
            du8, g + ij);
        hn::Store(
            hn::MaskedSetOr(hn::Load(du8, b + ij), mask_newly_escaped, b_new),
            du8, b + ij);
      }
    }
  }

  for (; n < iter_max_g; n++) {
    uint8_t g_new = static_cast<uint8_t>(std::ceil(
        255.0f * (iter_max_g_f - static_cast<float>(n)) / iter_max_g_f));
    uint8_t b_new = static_cast<uint8_t>(std::ceil(
        255.0f * (iter_max_b_f - static_cast<float>(n)) / iter_max_b_f));
    // As x_points and y_points are both even, x_points * y_points
    // is divisible by 4.  This is used to enable conversion from
    // U32 vectors to U8 vectors, which require 1/4 the size
    // registers of U32 vectors
    for (size_t ij = 0; ij < x_points * y_points; ij += 4 * NF) {
      if (not_escaped[ij]) {
        scaled_norm_0 = CalculateNextSimd(x, y, ij);
        scaled_norm_1 = CalculateNextSimd(x, y, ij + NF);
        scaled_norm_2 = CalculateNextSimd(x, y, ij + 2 * NF);
        scaled_norm_3 = CalculateNextSimd(x, y, ij + 3 * NF);
        MU8 mask_newly_escaped =
            GetMask(ij, scaled_norm_0, scaled_norm_1, scaled_norm_2,
                    scaled_norm_3, escape_value, g, not_escaped);
        hn::Store(
            hn::MaskedSetOr(hn::Load(du8, g + ij), mask_newly_escaped, g_new),
            du8, g + ij);
        hn::Store(
            hn::MaskedSetOr(hn::Load(du8, b + ij), mask_newly_escaped, b_new),
            du8, b + ij);
      }
    }
  }

  for (; n < iter_max_b; n++) {
    uint8_t b_new = static_cast<uint8_t>(std::ceil(
        255.0f * (iter_max_b_f - static_cast<float>(n)) / iter_max_b_f));
    // As x_points and y_points are both even, x_points * y_points
    // is divisible by 4.  This is used to enable conversion from
    // U32 vectors to U8 vectors, which require 1/4 the size
    // registers of U32 vectors
    for (size_t ij = 0; ij < x_points * y_points; ij += 4 * NF) {
      if (not_escaped[ij]) {
        scaled_norm_0 = CalculateNextSimd(x, y, ij);
        scaled_norm_1 = CalculateNextSimd(x, y, ij + NF);
        scaled_norm_2 = CalculateNextSimd(x, y, ij + 2 * NF);
        scaled_norm_3 = CalculateNextSimd(x, y, ij + 3 * NF);
        MU8 mask_newly_escaped =
            GetMask(ij, scaled_norm_0, scaled_norm_1, scaled_norm_2,
                    scaled_norm_3, escape_value, b, not_escaped);
        hn::Store(
            hn::MaskedSetOr(hn::Load(du8, b + ij), mask_newly_escaped, b_new),
            du8, b + ij);
      }
    }
  }
}

void CreatePPMScalar(const uint8_t* HWY_RESTRICT r,
                     const uint8_t* HWY_RESTRICT g,
                     const uint8_t* HWY_RESTRICT b, uint8_t* HWY_RESTRICT ppm,
                     const size_t x_points, const size_t y_points) {
  for (size_t n = 0; n < x_points * y_points; ++n) {
    ppm[3 * n] = r[n];
    ppm[3 * n + 1] = g[n];
    ppm[3 * n + 2] = b[n];
  }
  return;
}

void CreatePPMSimd(const uint8_t* HWY_RESTRICT r, const uint8_t* HWY_RESTRICT g,
                   const uint8_t* HWY_RESTRICT b, uint8_t* HWY_RESTRICT ppm,
                   const size_t x_points, const size_t y_points) {
  size_t n = 0;
  const size_t NU8 = Lanes(du8);
  for (; n < x_points * y_points; n += NU8) {
    StoreInterleaved3(hn::Load(du8, r + n), hn::Load(du8, g + n),
                      hn::Load(du8, b + n), du8, ppm + 3 * n);
  }
  // Use SafeCopyN for remainder, does not store values beyond remainder
  const size_t remainder = x_points * y_points - n;
  if (remainder > 0) {
    AlignedVector<uint8_t> ppm_temp(NU8);
    StoreInterleaved3(hn::LoadN(du8, r + n, remainder),
                      hn::LoadN(du8, g + n, remainder),
                      hn::LoadN(du8, b + n, remainder), du8, ppm_temp.data());
    SafeCopyN(remainder, du8, ppm_temp.data(), ppm + 3 * n);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_EXPORT(MandelbrotScalarSetup);
HWY_EXPORT(MandelbrotSimdSetup);
HWY_EXPORT(MandelbrotScalarCompute);
HWY_EXPORT(MandelbrotSimdCompute);
HWY_EXPORT(CreatePPMScalar);
HWY_EXPORT(CreatePPMSimd);

static bool Validate(const uint8_t* ppm_scalar, const uint8_t* ppm_simd,
                     const size_t x_points, const size_t y_points) {
  float diff = 0.0f;
  bool ret = false;
  for (size_t n = 0; n < 3 * x_points * y_points; ++n) {
    uint8_t temp = ppm_scalar[n] - ppm_simd[n];
    diff += static_cast<float>(temp * temp);
  }
  diff = std::sqrt(diff) / static_cast<float>(3 * x_points * y_points);
  std::cout << "Normalized difference : " << diff << std::endl;
  if (diff < 0.01f) {
    ret = true;
  }
  return ret;
}

static void WritePPM(std::string filename, const uint8_t* ppm,
                     const size_t x_points, const size_t y_points) {
  std::ofstream ppmfile;
  ppmfile.open(filename);
  ppmfile << "P3\n";
  ppmfile << "# " << filename << "\n";
  ppmfile << x_points << " " << y_points << "\n";
  ppmfile << "255\n";
  for (size_t n = 0; n < x_points * y_points; ++n) {
    ppmfile << std::to_string(ppm[3 * n]) << " "
            << std::to_string(ppm[3 * n + 1]) << " "
            << std::to_string(ppm[3 * n + 2]) << "\n";
  }
  ppmfile.close();
  return;
}

static void PrintTimeMeasurement(const double t0, const double t1,
                                 const std::string measurement) {
  std::cout << measurement << " time: " << 1000.0 * (t1 - t0) << " ms"
            << std::endl;
  return;
}

static void Run() {
  const size_t x_points =
      500;  // Grid points in x direction, needs to be even, >=6
  const size_t y_points =
      500;  // Grid points in y direction, needs to be even, >=6
  const size_t iter_max_r =
      25;  // Iterations to perform for red, needs to be positive
  const size_t iter_max_g =
      50;  // Iterations to perform for green, needs iter_max_g >= iter_max_r
  const size_t iter_max_b =
      100;  // Iterations to perform for blue, needs iter_max_b >= iter_max_g
  const float escape_value =
      2.0f;  // Value for which point is considered no longer iteratable
  AlignedVector<uint8_t> r_scalar(y_points * x_points);
  AlignedVector<uint8_t> g_scalar(y_points * x_points);
  AlignedVector<uint8_t> b_scalar(y_points * x_points);
  AlignedVector<uint8_t> ppm_scalar(3 * y_points * x_points);
  AlignedVector<float> x_scalar(y_points * x_points);
  AlignedVector<float> y_scalar(y_points * x_points);
  AlignedVector<uint8_t> r_simd(y_points * x_points);
  AlignedVector<uint8_t> g_simd(y_points * x_points);
  AlignedVector<uint8_t> b_simd(y_points * x_points);
  AlignedVector<uint8_t> ppm_simd(3 * y_points * x_points);
  AlignedVector<float> x_simd(y_points * x_points);
  AlignedVector<float> y_simd(y_points * x_points);
  // start timer
  const double t_scalar_setup_0 = hwy::platform::Now();
  HWY_DYNAMIC_DISPATCH(MandelbrotScalarSetup)(
      r_scalar.data(), g_scalar.data(), b_scalar.data(), x_scalar.data(),
      y_scalar.data(), x_points, y_points);
  // stop timing and print execution time
  const double t_scalar_setup_1 = hwy::platform::Now();
  PrintTimeMeasurement(t_scalar_setup_0, t_scalar_setup_1, "Scalar setup");
  // start timer
  const double t_simd_setup_0 = hwy::platform::Now();
  HWY_DYNAMIC_DISPATCH(MandelbrotSimdSetup)(r_simd.data(), g_simd.data(),
                                            b_simd.data(), x_simd.data(),
                                            y_simd.data(), x_points, y_points);
  // stop timing and print execution time
  const double t_simd_setup_1 = hwy::platform::Now();
  PrintTimeMeasurement(t_simd_setup_0, t_simd_setup_1, "SIMD setup");
  // start timer
  const double t_scalar_compute_0 = hwy::platform::Now();
  HWY_DYNAMIC_DISPATCH(MandelbrotScalarCompute)(
      r_scalar.data(), g_scalar.data(), b_scalar.data(), x_scalar.data(),
      y_scalar.data(), x_points, y_points, iter_max_r, iter_max_g, iter_max_b,
      escape_value);
  // stop timing and print execution time
  const double t_scalar_compute_1 = hwy::platform::Now();
  PrintTimeMeasurement(t_scalar_compute_0, t_scalar_compute_1,
                       "Scalar computation");
  // start timer
  const double t_simd_compute_0 = hwy::platform::Now();
  HWY_DYNAMIC_DISPATCH(MandelbrotSimdCompute)(
      r_simd.data(), g_simd.data(), b_simd.data(), x_simd.data(), y_simd.data(),
      x_points, y_points, iter_max_r, iter_max_g, iter_max_b, escape_value);
  // stop timing and print execution time
  const double t_simd_compute_1 = hwy::platform::Now();
  PrintTimeMeasurement(t_simd_compute_0, t_simd_compute_1, "SIMD computation");
  // start timing
  const double t_scalar_convert_0 = hwy::platform::Now();
  HWY_DYNAMIC_DISPATCH(CreatePPMScalar)(r_scalar.data(), g_scalar.data(),
                                        b_scalar.data(), ppm_scalar.data(),
                                        x_points, y_points);
  // stop timing and print execution time
  const double t_scalar_convert_1 = hwy::platform::Now();
  PrintTimeMeasurement(t_scalar_convert_0, t_scalar_convert_1,
                       "Scalar conversion");
  // start timing
  const double t_simd_convert_0 = hwy::platform::Now();
  HWY_DYNAMIC_DISPATCH(CreatePPMSimd)(r_simd.data(), g_simd.data(),
                                      b_simd.data(), ppm_simd.data(), x_points,
                                      y_points);
  // stop timing and print execution time
  const double t_simd_convert_1 = hwy::platform::Now();
  PrintTimeMeasurement(t_simd_convert_0, t_simd_convert_1, "SIMD conversion");
  // write out the image files
  WritePPM("mandelbrot_scalar.ppm", ppm_scalar.data(), x_points, y_points);
  WritePPM("mandelbrot_simd.ppm", ppm_simd.data(), x_points, y_points);
  // Check if results are similar, they may
  // not be exactly the same due to approximations
  // and floating point errors
  if (Validate(ppm_scalar.data(), ppm_simd.data(), x_points, y_points)) {
    std::cout << "Validation passed" << std::endl;
  } else {
    std::cout << "Validation failed" << std::endl;
  }
}

}  // namespace hwy

int main() {
  hwy::Run();
  return 0;
}
#endif  // HWY_ONCE
