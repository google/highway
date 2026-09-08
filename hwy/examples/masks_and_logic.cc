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

/*
Expected output (64x30 grid):
................................................................
................................................................
................................................................
................................................................
................................................................
.........11111111111111111111111................................
.........11111111111111111111111................................
.........11111111111111111111111................................
.........11111111111111111111111................................
.........11111111111111111111111................................
.........11111111111111111111111................................
.........111111111111111XXXXXXXX222222222222222.................
.........111111111111111XXXXXXXX222222222222222.................
.........111111111111111XXXXXXXX222222222222222.................
.........111111111111111XXXXXXXX222222222222222.................
.........111111111111111XXXXXXXX222222222222222.................
.........111111111111111XXXXXXXX222222222222222.................
.........111111111111111XXXXXXXX222222222222222.................
.........111111111111111XXXXXXXX222222222222222.................
.........111111111111111XXXXXXXX222222222222222.................
........................22222222222222222222222.................
........................22222222222222222222222.................
........................22222222222222222222222.................
........................22222222222222222222222.................
........................22222222222222222222222.................
........................22222222222222222222222.................
................................................................
................................................................
................................................................
................................................................
*/

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#ifndef HIGHWAY_ART_SHAPES_H_
#define HIGHWAY_ART_SHAPES_H_
struct Box {
  float cx, cy, w, h;
};

constexpr Box kBox1 = {20.0f, 12.0f, 12.0f, 8.0f};
constexpr Box kBox2 = {35.0f, 18.0f, 12.0f, 8.0f};
#endif

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/examples/masks_and_logic.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
namespace hn = hwy::HWY_NAMESPACE;

// Render a single line using SIMD.
// We use size_t for dimensions and coordinates to avoid warnings.
void RenderLineSIMD(size_t y, size_t width, int32_t* HWY_RESTRICT line_buf) {
  const hn::ScalableTag<float> df;
  const hn::RebindToSigned<decltype(df)> di;  // int32_t tag

  using MF = hn::Mask<decltype(df)>;
  using MI = hn::Mask<decltype(di)>;
  using VF = hn::Vec<decltype(df)>;
  using VI = hn::Vec<decltype(di)>;

  HWY_LANES_CONSTEXPR size_t N = hn::Lanes(df);

  // Lambda for box mask
  auto make_box_mask = [&](VF v_x, float y_val, Box b) HWY_ATTR -> MF {
    // Use AbsDiff to compute absolute difference.
    VF v_bdx = hn::AbsDiff(v_x, hn::Set(df, b.cx));
    VF v_bdy = hn::Set(df, std::abs(y_val - b.cy));
    // Lt (Less Than) returns a mask. We use And to combine masks.
    return hn::And(hn::Lt(v_bdx, hn::Set(df, b.w)),
                   hn::Lt(v_bdy, hn::Set(df, b.h)));
  };

  size_t x = 0;
  // We assume 'width' is a multiple of the vector length N.
  // This allows us to avoid remainder handling and just use a simple loop.
  for (; x < width; x += N) {
    // Use Iota to generate v_x = [x, x+1, x+2, ..., x+N-1]
    // We can pass the start value directly to Iota.
    VF v_x = hn::Iota(df, static_cast<float>(x));

    MF mask1 = make_box_mask(v_x, static_cast<float>(y), kBox1);
    MF mask2 = make_box_mask(v_x, static_cast<float>(y), kBox2);

    // Combine masks using boolean operations.
    // AndNot(a, b) means 'b AND NOT a'.
    MF mask_both = hn::And(mask1, mask2);
    MF mask1_only = hn::AndNot(mask2, mask1);
    MF mask2_only = hn::AndNot(mask1, mask2);

    // Masks generated from floats cannot be directly used to select integers.
    // RebindMask converts the mask representation to match the integer tag di.
    MI imask_both = hn::RebindMask(di, mask_both);
    MI imask1_only = hn::RebindMask(di, mask1_only);
    MI imask2_only = hn::RebindMask(di, mask2_only);

    // Chaining IfThenElse allows simulating nested if-else logic.
    // The execution order is effectively reverse of chain order, or we can
    // think of it as overwriting values.
    VI v_pixels = hn::Set(di, '.');
    v_pixels = hn::IfThenElse(imask2_only, hn::Set(di, '2'), v_pixels);
    v_pixels = hn::IfThenElse(imask1_only, hn::Set(di, '1'), v_pixels);
    v_pixels = hn::IfThenElse(imask_both, hn::Set(di, 'X'), v_pixels);

    hn::StoreU(v_pixels, di, line_buf + x);
  }
  HWY_ASSERT(x == width);
}

// Render the entire art using SIMD.
// We pull the loop over y inside the SIMD function to avoid per-line dispatch
// overhead.
void RenderArtSIMD(size_t width, size_t height, int32_t* HWY_RESTRICT buf) {
  for (size_t y = 0; y < height; ++y) {
    RenderLineSIMD(y, width, buf + y * width);
  }
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_EXPORT(RenderArtSIMD);

static void CallRenderArtSIMD(size_t width, size_t height,
                              int32_t* HWY_RESTRICT buf) {
  HWY_DYNAMIC_DISPATCH(RenderArtSIMD)(width, height, buf);
}

static void RenderArtScalar(size_t width, size_t height,
                            int32_t* HWY_RESTRICT buf) {
  auto check_box = [](float x_val, float y_val, Box b) -> bool {
    return std::abs(x_val - b.cx) < b.w && std::abs(y_val - b.cy) < b.h;
  };

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      bool in_box1 =
          check_box(static_cast<float>(x), static_cast<float>(y), kBox1);
      bool in_box2 =
          check_box(static_cast<float>(x), static_cast<float>(y), kBox2);

      char pixel = '.';
      if (in_box1 && in_box2) {
        pixel = 'X';
      } else if (in_box1) {
        pixel = '1';
      } else if (in_box2) {
        pixel = '2';
      }
      buf[y * width + x] = pixel;
    }
  }
}

static void PrintArt(const std::vector<int32_t>& buf, size_t width,
                     size_t height) {
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      std::cout << static_cast<char>(buf[y * width + x]);
    }
    std::cout << '\n';
  }
}

}  // namespace hwy

int main() {
  // We assume the width is a multiple of the vector length.
  // 64 is a safe choice as it is a power of 2 and all smid vector
  // lengths are powers of 2.
  const size_t width = 64;
  const size_t height = 30;
  std::vector<int32_t> line_buf(width * height);

  std::cout << "Here is image rendered using scalar code:\n";
  hwy::RenderArtScalar(width, height, line_buf.data());
  hwy::PrintArt(line_buf, width, height);

  std::cout << "\nHere is image rendered with SIMD code:\n";
  hwy::CallRenderArtSIMD(width, height, line_buf.data());
  hwy::PrintArt(line_buf, width, height);

  return 0;
}
#endif  // HWY_ONCE
