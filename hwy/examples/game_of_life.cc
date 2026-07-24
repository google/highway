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
#include <vector>

#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/examples/game_of_life.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/timer.h"

/*
Highway SIMD Tutorial: Game of Life

This example demonstrates how to use SIMD vectorization on stencil
computations.  The computation starts with a randomly initialized
boolean grid.  A cell can be alive (True/1) or dead (False/0) The
grid has periodic boundary conditions.  At each iteration, cells
are evolved as follows[0]:
- A live cell with less than two live neighbours dies
- A live cell with more than three live neighbours dies
- A dead cell with exactly three live neighbours is resurrected
- A live cell with two or three live neighbours continues living

0) https://en.wikipedia.org/wiki/Conway's_Game_of_Life

*/

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void InitializeState(uint64_t* HWY_RESTRICT a_1, uint64_t* HWY_RESTRICT a_2, size_t nx, size_t ny) {

  using DU64 = hn::ScalableTag<uint64_t>;
  const DU64 du64;
  const size_t NU64 = hn::Lanes(du64);
  using VU64 = hn::Vec<DU64>;
  VectorXoshiro generator{uint64_t{5}};
  size_t i = 0;
  size_t upp_bound = 1 + (nx*ny/(8*sizeof(uint64_t)));
  for(; i + NU64 <= upp_bound; i += NU64) {
    VU64 temp = generator();
    hn::StoreU(temp, du64, a_1 + i);
    hn::StoreU(temp, du64, a_2 + i);
  }
  // Handle remainder
  size_t remainder = upp_bound - i;
  HWY_DASSERT(remainder < NU64);
  if (remainder > 0) {
    VU64 temp = generator();
    hn::StoreN(temp, du64, a_1 + i, remainder);
    hn::StoreN(temp, du64, a_2 + i, remainder);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_EXPORT(InitializeState);

void NewStateScalar(const uint64_t* HWY_RESTRICT in, uint64_t* HWY_RESTRICT out,
              size_t nx, size_t ny) {

  // Lambda for checking state
  auto check_state = [](const uint64_t* in, const size_t ind) HWY_ATTR -> bool {
    const size_t arr = static_cast<size_t>(ind / (8*sizeof(uint64_t)));
    const size_t bit = static_cast<size_t>(ind % (8*sizeof(uint64_t)));
    const uint8_t one = 1;
    return (in[arr] & static_cast<uint64_t>(one<<bit));
  };

  // Lambda for updating state
  auto update = [](uint64_t* out, const size_t ind, const bool new_state) HWY_ATTR -> void {
    const size_t arr = static_cast<size_t>(ind / (8*sizeof(uint64_t)));
    const size_t bit = static_cast<size_t>(ind % (8*sizeof(uint64_t)));
    const uint64_t one = 1;
    if(new_state) {
      out[arr] |= static_cast<uint64_t>(one<<(bit));
    }else{
      out[arr] &= ~(static_cast<uint64_t>(one<<(bit)));
    }
    return;
  };

  for(size_t i = 0; i < nx; i++) {
    for(size_t j = 0; j < ny; j++) {
       // Count number of neighbours
       size_t neighbours = 0;
       size_t ind;
       // top left
       ind  = ((i-1)%nx) + ((j-1)%ny)*nx;
       neighbours += static_cast<size_t>(check_state(in, ind));
       // top
       ind = (i%nx) + ((j-1)%ny)*nx;
       neighbours += static_cast<size_t>(check_state(in, ind));
       // top right
       ind = ((i+1)%nx) + ((j-1)%ny)*nx;
       neighbours += static_cast<size_t>(check_state(in, ind));
       // right
       ind = ((i+1)%nx) + (j%ny)*nx;
       neighbours += static_cast<uint8_t>(check_state(in, ind));
       // bottom right
       ind = ((i+1)%nx) + ((j+1)%ny)*nx;
       neighbours += static_cast<size_t>(check_state(in, ind));
       // bottom
       ind = (i%nx) + ((j+1)%ny)*nx;
       neighbours += static_cast<size_t>(check_state(in, ind));
       // bottom left
       ind = ((i-1)%nx) + ((j+1)%ny)*nx;
       neighbours += static_cast<size_t>(check_state(in, ind));
       // left
       ind = ((i-1)%nx) + (j%ny)*nx;
       neighbours += static_cast<size_t>(check_state(in, ind));
       // update center
       ind = i + j*nx;
       bool my_state = check_state(in, ind);
       switch (neighbours){
         case 0:
           update(out, ind, false);
           break;
         case 1:
           update(out, ind, false);
           break;
         case 2:
           update(out, ind, my_state);
           break;
         case 3:
           update(out, ind, true);
           break;
         case 4:
           update(out, ind, false);
           break;
       }
    }
  }
  return;
}

void GameOfLifeScalar(uint64_t* HWY_RESTRICT a, uint64_t* HWY_RESTRICT b,
                      size_t nx, size_t ny, size_t iterations) {
  for(size_t iter = 0; iter < iterations; iter += 2) {
    NewStateScalar(a,b,nx,ny);
    NewStateScalar(b,a,nx,ny);
  }
  // Remainder iteration
  if( iterations%2 == 1 ) {
    NewStateScalar(a,b,nx,ny);
  }

  return;
}

int Run() {
  const size_t nx = 200;
  const size_t ny = 200;
  const size_t uint64_size = 1 + (nx * ny )/(8 * sizeof(uint64_t));
  const size_t iterations = 100;
  AlignedFreeUniquePtr<uint64_t[]> a_scalar =
      AllocateAligned<uint64_t>(uint64_size);
  AlignedFreeUniquePtr<uint64_t[]> b_scalar =
      AllocateAligned<uint64_t>(uint64_size);
  AlignedFreeUniquePtr<uint64_t[]> a_simd =
      AllocateAligned<uint64_t>(uint64_size);
  AlignedFreeUniquePtr<uint64_t[]> b_simd =
      AllocateAligned<uint64_t>(uint64_size);

  HWY_DYNAMIC_DISPATCH(InitializeState)(a_scalar.get(), a_simd.get(), nx, ny);
  
  // Record start time
  const double t_scalar_0 = hwy::platform::Now();
  GameOfLifeScalar(a_scalar.get(), b_scalar.get(), nx, ny, iterations);
  // Record end time and print execution time
  const double t_scalar_1 = hwy::platform::Now();
  const double dt_scalar = 1000.0 * (t_scalar_1 - t_scalar_0);
  std::cout << "Scalar Execution time: " << dt_scalar << " ms" << std::endl;

  return 0;
}
}  // namespace hwy

int main() { return hwy::Run(); }
#endif  // HWY_ONCE
