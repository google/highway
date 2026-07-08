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

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {

static void Initialize(bool* HWY_RESTRICT a, size_t nx, size_t ny) {
  for(size_t i = 0; i < nx*ny; ++i) {
    a[i] = (std::rand()%2 == 1);
  }
}

static void NewState(const bool* HWY_RESTRICT in, bool* HWY_RESTRICT out,
              size_t nx, size_t ny) {
    for(size_t i = 0; i < nx; i++) {
      for(size_t j = 0; j < ny; j++) {
         // Get indices to represent periodic boundary conditions
	 size_t ind = i + j*nx;
         size_t top = (i%nx) + ((j-1)%ny)*nx;
         size_t bottom = (i%nx) + ((j+1)%ny)*nx;
         size_t left = ((i-1)%nx) + (j%ny)*nx;
         size_t right = ((i+1)%nx) + (j%ny)*nx;
         size_t neighbours = static_cast<size_t>(in[top])
                            + static_cast<size_t>(in[bottom])
                            + static_cast<size_t>(in[left])
                            + static_cast<size_t>(in[right]);
         switch (neighbours){
           case 0:
             out[ind] = false;
             break;
           case 1:
             out[ind] = false;
             break;
           case 2:
             out[ind] = (in[ind] ? true : false);
             break;
           case 3:
             out[ind] = true;
             break;
           case 4:
             out[ind] = false;
             break;
	 }
      }
    }
    return;
}

static void GameOfLifeScalar(bool* HWY_RESTRICT a, bool* HWY_RESTRICT b,
                      size_t nx, size_t ny, size_t iterations) {
  size_t iter = 0;
  for(; iter < iterations; iter+=2) {
    hwy::NewState(a,b,nx,ny);
    hwy::NewState(b,a,nx,ny);
  }
  // Remainder iteration
  if( iterations%2 == 1 ) {
    hwy::NewState(a,b,nx,ny);
  }

  return;
}

static int Run() {
  const size_t nx = 60;
  const size_t ny = 60;
  const size_t iterations = 100;
  bool* a = (bool*)malloc(nx*ny*sizeof(bool));
  bool* b = (bool*)malloc(nx*ny*sizeof(bool));

  hwy::Initialize(a, nx, ny);
  // Record start time
  const double t_scalar_0 = hwy::platform::Now();
  hwy::GameOfLifeScalar(a, b, nx, ny, iterations);
  // Record end time and print execution time
  const double t_scalar_1 = hwy::platform::Now();
  const double dt_scalar = 1000.0 * (t_scalar_1 - t_scalar_0);
  std::cout << "Scalar Execution time: " << dt_scalar << " ms" << std::endl;
  free(a);
  free(b);
  return 0;
}
}  // namespace hwy

int main() { return hwy::Run(); }
#endif  // HWY_ONCE
