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
#include <stdint.h>

#include <iostream>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/examples/baker_mix.cc"

#include "hwy/foreach_target.h"  // IWYU pragma: keep
// Put after foreach_target.h to avoid redefinition errors
#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"
#include "hwy/print-inl.h"

/*
Highway SIMD Tutorial: Baker mix

This example demonstrates permiutation of an array using a
transformation similar to the chaotic bakers map[0]. Only
one step is done to enable easy examination of the process.

The program takes in an array of numbers and interleaves
the top half with the bottom half. Followed by a shift.

0) https://en.wikipedia.org/wiki/Baker's_map
*/

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

using DU32 = hn::ScalableTag<uint32_t>;
const DU32 du32;
using VU32 = hn::Vec<DU32>;
using DU32h = hn::Half<DU32>;
const DU32h du32h;
using VU32h = hn::Vec<DU32h>;

void Setup(uint32_t* HWY_RESTRICT data, const size_t points) {
  // Initialize array with linearly increasing sequence of numbers
  uint32_t NU32 = static_cast<uint32_t>(hn::Lanes(du32));
  for (uint32_t i = 0; i + NU32 <= points; i += NU32) {
    hn::StoreU(hn::Iota(du32, i), du32, data + i);
  }
}

void LocalMix(uint32_t* HWY_RESTRICT data, const size_t points) {
  // mix within a vector
  uint32_t NU32 = static_cast<uint32_t>(hn::Lanes(du32));
  for (uint32_t i = 0; i + NU32 <= points; i += NU32) {
    VU32 vec = hn::LoadU(du32, data + i);
    VU32h lower = hn::LowerHalf(du32h, vec);
    VU32h upper = hn::UpperHalf(du32h, vec);
    hn::StoreInterleaved2(upper, lower, du32h, data + i);
  }
  return;
}

void Diffuse(uint32_t* HWY_RESTRICT data, const size_t points,
             const size_t shift) {
  // Shift between different vectors
  uint32_t NU32 = static_cast<uint32_t>(hn::Lanes(du32));
  // Temporary store to put in at end of loop
  VU32 temp = hn::SlideUpLanes(du32, hn::LoadU(du32, data), NU32 - shift);
  for (uint32_t i = 0; i + NU32 < points; i += NU32) {
    VU32 lower = hn::LoadU(du32, data + i);
    VU32 upper =
        hn::SlideUpLanes(du32, hn::LoadU(du32, data + i + NU32), NU32 - shift);
    VU32 vec = hn::SlideDownLanesOr(upper, du32, lower, shift);
    hn::StoreU(vec, du32, data + i);
  }
  VU32 lower = hn::LoadU(du32, data + points - NU32);
  VU32 vec = hn::SlideDownLanesOr(temp, du32, lower, shift);
  hn::StoreU(vec, du32, data + points - NU32);
  // Demonstrate how to print vector contents for debugging
  hn::Print(du32, "\nLast diffused vector\n", vec);
  return;
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_EXPORT(Diffuse);
HWY_EXPORT(LocalMix);
HWY_EXPORT(Setup);

int Run() {
  const size_t points = 512;  // Needs to be less than 2^32 ~ 4*10^9, should
                              // be a multiple of 64 for vector alignment
  const size_t diffuse_length = 3;  // Needs to be positive but less than NU32
                                    // should be odd
  hwy::AlignedVector<uint32_t> numbers(points);
  HWY_DYNAMIC_DISPATCH(Setup)(numbers.data(), points);
  std::cout << "Generated input" << std::endl;
  for (size_t i = 0; i < points - 1; i++) std::cout << numbers[i] << ",";
  std::cout << numbers[points - 1] << std::endl;
  HWY_DYNAMIC_DISPATCH(LocalMix)(numbers.data(), points);
  std::cout << std::endl << "Locally mixed input" << std::endl;
  for (size_t i = 0; i < points - 1; i++) std::cout << numbers[i] << ",";
  std::cout << numbers[points - 1] << std::endl;
  HWY_DYNAMIC_DISPATCH(Diffuse)(numbers.data(), points, diffuse_length);
  std::cout << std::endl << "Diffused input" << std::endl;
  for (size_t i = 0; i < points - 1; i++) std::cout << numbers[i] << ",";
  std::cout << numbers[points - 1] << std::endl;

  return 0;
}

}  // namespace hwy

int main() { return hwy::Run(); }
#endif  // HWY_ONCE
