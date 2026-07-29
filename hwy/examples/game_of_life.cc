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

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/examples/game_of_life.cc"

#include "hwy/foreach_target.h"  // IWYU pragma: keep
// Put after foreach_target.h to avoid redefinition errors
#include "hwy/aligned_allocator.h"
#include "hwy/contrib/random/random-inl.h"
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
1)
https://lemire.me/blog/2018/07/18/accelerating-conways-game-of-life-with-simd-instructions
2) https://binary-banter.github.io/game-of-life/
3) https://www.cs.uaf.edu/courses/cs441/notes/simd/
4) https://www.moria.us/old/3/programs/life/
5) https://gist.github.com/CharCoding/52fb584fab2d3632fe2225880890463e
6)
https://colab.research.google.com/github/google-research/blob/master/understanding_convolutions_on_graphs/TheGameOfLifeWithGNNs.ipynb
7) https://tomlam.dev/files/gol_report.pdf
8) https://www.cs.hiroshima-u.ac.jp/cs/_media/life-ijfcs.pdf
*/

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

using DU64 = hn::ScalableTag<uint64_t>;
const DU64 du64;
using VU64 = hn::Vec<DU64>;
using DU8 = hn::ScalableTag<uint8_t>;
const DU8 du8;
using VU8 = hn::Vec<DU8>;
using MU8 = hn::Mask<DU8>;

void InitializeState(uint64_t* HWY_RESTRICT a_bit_1,
                     uint64_t* HWY_RESTRICT a_bit_2,
                     uint8_t* HWY_RESTRICT a_byte_1,
                     uint8_t* HWY_RESTRICT a_byte_2, const size_t nx,
                     const size_t ny) {
  size_t NU64 = hn::Lanes(du64);
  size_t NU8 = hn::Lanes(du8);
  hwy::AlignedVector<uint8_t> temp(NU8);
  VectorXoshiro generator{uint64_t{5}};
  size_t i = 0;
  const size_t upp_bound = 1 + ((nx * ny) / (8 * sizeof(uint64_t)));
  for (; i + NU64 <= upp_bound; i += NU64) {
    VU64 rand = generator();
    hn::StoreU(rand, du64, a_bit_1 + i);
    hn::StoreU(rand, du64, a_bit_2 + i);
  }
  // Handle remainder
  size_t remainder = upp_bound - i;
  HWY_DASSERT(remainder < NU64);
  if (remainder > 0) {
    VU64 rand = generator();
    hn::StoreN(rand, du64, a_bit_1 + i, remainder);
    hn::StoreN(rand, du64, a_bit_2 + i, remainder);
  }
  i = 0;
  for (; i + NU8 <= nx * ny; i += NU8) {
    for (size_t j = 0; j < NU8; j++) {
      const size_t arr = static_cast<size_t>((i + j) / (8 * sizeof(uint64_t)));
      const size_t bit = static_cast<size_t>((i + j) % (8 * sizeof(uint64_t)));
      const uint64_t one = 1;
      temp[j] = (a_bit_1[arr] & static_cast<uint64_t>(one << bit))
                    ? static_cast<uint8_t>(1)
                    : static_cast<uint8_t>(0);
    }
    hn::SafeCopyN(NU8, du8, temp.data(), a_byte_1 + i);
    hn::SafeCopyN(NU8, du8, temp.data(), a_byte_2 + i);
  }
  // Handle remainder
  remainder = nx * ny - i;
  HWY_DASSERT(remainder < NU8);
  if (remainder > 0) {
    for (size_t j = 0; j < remainder; j++) {
      const size_t arr = static_cast<size_t>((i + j) / (8 * sizeof(uint64_t)));
      const size_t bit = static_cast<size_t>((i + j) % (8 * sizeof(uint64_t)));
      const uint64_t one = 1;
      temp[j] = (a_bit_1[arr] & static_cast<uint64_t>(one << bit))
                    ? static_cast<uint8_t>(1)
                    : static_cast<uint8_t>(0);
    }
    hn::SafeCopyN(remainder, du8, temp.data(), a_byte_1 + i);
    hn::SafeCopyN(remainder, du8, temp.data(), a_byte_2 + i);
  }
}

bool ValidateBit(const uint8_t* HWY_RESTRICT ref_byte,
                 const uint64_t* HWY_RESTRICT out, const size_t nx,
                 const size_t ny) {
  size_t NU8 = hn::Lanes(du8);
  hwy::AlignedVector<uint8_t> temp(NU8);
  size_t i = 0;
  bool no_mismatches = true;
  for (; i + NU8 <= nx * ny; i += NU8) {
    for (size_t j = 0; j < NU8; j++) {
      const size_t arr = static_cast<size_t>((i + j) / (8 * sizeof(uint64_t)));
      const size_t bit = static_cast<size_t>((i + j) % (8 * sizeof(uint64_t)));
      const uint64_t one = 1;
      temp[j] = (out[arr] & static_cast<uint64_t>(one << bit))
                    ? static_cast<uint8_t>(1)
                    : static_cast<uint8_t>(0);
    }
    no_mismatches &= hn::AllTrue(
        du8, hn::Eq(hn::LoadU(du8, temp.data()), hn::LoadU(du8, ref_byte + i)));
  }
  // Handle remainder
  size_t remainder = nx * ny - i;
  HWY_DASSERT(remainder < NU8);
  if (remainder > 0) {
    for (size_t j = 0; j < remainder; j++) {
      const size_t arr = static_cast<size_t>((i + j) / (8 * sizeof(uint64_t)));
      const size_t bit = static_cast<size_t>((i + j) % (8 * sizeof(uint64_t)));
      const uint64_t one = 1;
      temp[j] = (out[arr] & static_cast<uint64_t>(one << bit))
                    ? static_cast<uint8_t>(1)
                    : static_cast<uint8_t>(0);
    }
    no_mismatches &=
        hn::AllTrue(du8, hn::Eq(hn::LoadN(du8, temp.data(), remainder),
                                hn::LoadN(du8, ref_byte + i, remainder)));
  }
  return no_mismatches;
}

bool ValidateByte(const uint8_t* HWY_RESTRICT ref_byte,
                  const uint8_t* HWY_RESTRICT out, const size_t nx,
                  const size_t ny) {
  size_t NU8 = hn::Lanes(du8);
  size_t i = 0;
  bool no_mismatches = true;
  for (; i + NU8 <= nx * ny; i += NU8) {
    no_mismatches &= hn::AllTrue(
        du8, hn::Eq(hn::LoadU(du8, out + i), hn::LoadU(du8, ref_byte + i)));
  }
  // Handle remainder
  size_t remainder = nx * ny - i;
  HWY_DASSERT(remainder < NU8);
  if (remainder > 0) {
    no_mismatches &=
        hn::AllTrue(du8, hn::Eq(hn::LoadN(du8, out + i, remainder),
                                hn::LoadN(du8, ref_byte + i, remainder)));
  }
  return no_mismatches;
}

VU8 NeighborCountSimdByte(const VU8 top_values, const VU8 my_values,
                          const VU8 bottom_values,
                          const uint8_t* HWY_RESTRICT in, const size_t NU8,
                          const size_t nx, const size_t ny, const size_t i,
                          const size_t j) {
  size_t ind;
  VU8 neighbor_count = hn::Zero(du8);
  // left points
  ind = j * nx + ((nx - 1 + i) % nx);
  neighbor_count = hn::Slide1UpOr(in[ind], du8, my_values);
  // bottom left points
  ind = ((1 + j) % ny) * nx + ((nx - 1 + i) % nx);
  neighbor_count =
      hn::Add(neighbor_count, hn::Slide1UpOr(in[ind], du8, bottom_values));
  // bottom points
  neighbor_count = hn::Add(neighbor_count, bottom_values);
  // bottom right points
  ind = ((1 + j) % ny) * nx + ((NU8 + nx + i) % nx);
  neighbor_count =
      hn::Add(neighbor_count, hn::Slide1DownOr(in[ind], du8, bottom_values));
  // right points
  ind = j * nx + ((NU8 + nx + i) % nx);
  neighbor_count =
      hn::Add(neighbor_count, hn::Slide1DownOr(in[ind], du8, my_values));
  // top right points
  ind = ((ny - 1 + j) % ny) * nx + ((NU8 + nx + i) % nx);
  neighbor_count =
      hn::Add(neighbor_count, hn::Slide1DownOr(in[ind], du8, top_values));
  // top points
  neighbor_count = hn::Add(neighbor_count, top_values);
  // top left points
  ind = ((ny - 1 + j) % ny) * nx + (nx - 1 + i) % nx;
  neighbor_count =
      hn::Add(neighbor_count, hn::Slide1UpOr(in[ind], du8, top_values));
  return neighbor_count;
}

void NewStateSimdByte(const uint8_t* HWY_RESTRICT in, uint8_t* HWY_RESTRICT out,
                      const size_t nx, const size_t ny) {
  const size_t NU8 = hn::Lanes(du8);
  VU8 neighbor_count;
  VU8 previous_row;
  VU8 current_row;
  VU8 next_row;
  VU8 new_state;
  MU8 three_alive;
  MU8 two_and_me_alive;
  MU8 alive;

  for (size_t i = 0; i + NU8 <= nx; i += NU8) {
    for (size_t j = 0; j < ny; j++) {
      previous_row = hn::LoadU(du8, in + ((ny - 1 + j) % ny) * nx + i);
      current_row = hn::LoadU(du8, in + j * nx + i);
      next_row = hn::LoadU(du8, in + ((1 + j) % ny) * nx + i);
      neighbor_count = NeighborCountSimdByte(previous_row, current_row,
                                             next_row, in, NU8, nx, ny, i, j);
      two_and_me_alive =
          hn::And(hn::Eq(neighbor_count, hn::Set(du8, static_cast<uint8_t>(2))),
                  hn::Eq(current_row, hn::Set(du8, static_cast<uint8_t>(1))));
      three_alive =
          hn::Eq(neighbor_count, hn::Set(du8, static_cast<uint8_t>(3)));
      alive = hn::Or(three_alive, two_and_me_alive);
      new_state = hn::MaskedSet(du8, alive, static_cast<uint8_t>(1));
      hn::StoreU(new_state, du8, out + j * nx + i);
    }
  }
  return;
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_EXPORT(InitializeState);
HWY_EXPORT(ValidateBit);
HWY_EXPORT(NewStateSimdByte);
HWY_EXPORT(ValidateByte);

void NewStateScalarBit(const uint64_t* HWY_RESTRICT in,
                       uint64_t* HWY_RESTRICT out, const size_t nx,
                       const size_t ny) {
  // Lambda for checking state
  auto check_state = [](const uint64_t* in, const size_t ind) HWY_ATTR -> bool {
    const size_t arr =
        static_cast<size_t>(ind / static_cast<size_t>(8 * sizeof(uint64_t)));
    const size_t bit =
        static_cast<size_t>(ind % static_cast<size_t>(8 * sizeof(uint64_t)));
    const uint64_t one = 1;
    return (in[arr] & static_cast<uint64_t>(one << bit));
  };

  // Lambda for updating state
  auto update = [](uint64_t* out, const size_t ind, const bool alive)
                    HWY_ATTR -> void {
    const size_t arr =
        static_cast<size_t>(ind / static_cast<size_t>(8 * sizeof(uint64_t)));
    const size_t bit =
        static_cast<size_t>(ind % static_cast<size_t>(8 * sizeof(uint64_t)));
    uint64_t one = 1;
    alive ? (out[arr] |= static_cast<uint64_t>(one << bit))
          : (out[arr] &= ~(static_cast<uint64_t>(one << bit)));
    return;
  };

  for (size_t j = 0; j < ny; j++) {
    for (size_t i = 0; i < nx; i++) {
      // Count number of neighbours
      size_t neighbours = 0;
      size_t ind;
      // top left
      ind = ((i - 1 + nx) % nx) + ((ny + j - 1) % ny) * nx;
      neighbours += check_state(in, ind) ? static_cast<size_t>(1)
                                         : static_cast<size_t>(0);
      // top
      ind = (i % nx) + ((ny + j - 1) % ny) * nx;
      neighbours += check_state(in, ind) ? static_cast<size_t>(1)
                                         : static_cast<size_t>(0);
      // top right
      ind = ((i + 1) % nx) + ((ny + j - 1) % ny) * nx;
      neighbours += check_state(in, ind) ? static_cast<size_t>(1)
                                         : static_cast<size_t>(0);
      // right
      ind = ((i + 1) % nx) + (j % ny) * nx;
      neighbours += check_state(in, ind) ? static_cast<size_t>(1)
                                         : static_cast<size_t>(0);
      // bottom right
      ind = ((i + 1) % nx) + ((j + 1) % ny) * nx;
      neighbours += check_state(in, ind) ? static_cast<size_t>(1)
                                         : static_cast<size_t>(0);
      // bottom
      ind = (i % nx) + ((j + 1) % ny) * nx;
      neighbours += check_state(in, ind) ? static_cast<size_t>(1)
                                         : static_cast<size_t>(0);
      // bottom left
      ind = ((i + nx - 1) % nx) + ((j + 1) % ny) * nx;
      neighbours += check_state(in, ind) ? static_cast<size_t>(1)
                                         : static_cast<size_t>(0);
      // left
      ind = ((i + nx - 1) % nx) + (j % ny) * nx;
      neighbours += check_state(in, ind) ? static_cast<size_t>(1)
                                         : static_cast<size_t>(0);
      // update center
      ind = i + j * nx;
      bool my_state = check_state(in, ind);
      switch (neighbours) {
        case 2:
          update(out, ind, my_state);
          break;
        case 3:
          update(out, ind, true);
          break;
        default:
          update(out, ind, false);
          break;
      }
    }
  }
  return;
}

void NewStateScalarByte(const uint8_t* HWY_RESTRICT in,
                        uint8_t* HWY_RESTRICT out, const size_t nx,
                        const size_t ny) {
  for (size_t j = 0; j < ny; j++) {
    for (size_t i = 0; i < nx; i++) {
      // Count number of neighbours
      uint8_t neighbours = 0;
      size_t ind;
      // top left
      ind = ((i - 1 + nx) % nx) + ((j - 1 + ny) % ny) * nx;
      neighbours += in[ind];
      // top
      ind = (i % nx) + ((j - 1 + ny) % ny) * nx;
      neighbours += in[ind];
      // top right
      ind = ((i + 1) % nx) + ((j - 1 + ny) % ny) * nx;
      neighbours += in[ind];
      // right
      ind = ((i + 1) % nx) + (j % ny) * nx;
      neighbours += in[ind];
      // bottom right
      ind = ((i + 1) % nx) + ((j + 1) % ny) * nx;
      neighbours += in[ind];
      // bottom
      ind = (i % nx) + ((j + 1) % ny) * nx;
      neighbours += in[ind];
      // bottom left
      ind = ((i - 1 + nx) % nx) + ((j + 1) % ny) * nx;
      neighbours += in[ind];
      // left
      ind = ((i - 1 + nx) % nx) + (j % ny) * nx;
      neighbours += in[ind];
      // update center
      ind = i + j * nx;
      switch (neighbours) {
        case 2:
          out[ind] = in[ind];
          break;
        case 3:
          out[ind] = static_cast<uint8_t>(1);
          break;
        default:
          out[ind] = static_cast<uint8_t>(0);
          break;
      }
    }
  }
  return;
}

void GameOfLifeScalarBit(uint64_t* HWY_RESTRICT a, uint64_t* HWY_RESTRICT b,
                         const size_t nx, const size_t ny,
                         const size_t iterations) {
  size_t iter = 0;
  for (; iter + 1 < iterations; iter += 2) {
    NewStateScalarBit(a, b, nx, ny);
    NewStateScalarBit(b, a, nx, ny);
  }
  // Remainder iteration
  if (iterations - iter > 0) {
    NewStateScalarBit(a, b, nx, ny);
  }

  return;
}

void GameOfLifeScalarByte(uint8_t* HWY_RESTRICT a, uint8_t* HWY_RESTRICT b,
                          const size_t nx, const size_t ny,
                          const size_t iterations) {
  size_t iter = 0;
  for (; iter + 1 < iterations; iter += 2) {
    NewStateScalarByte(a, b, nx, ny);
    NewStateScalarByte(b, a, nx, ny);
  }
  // Remainder iteration
  if (iterations - iter > 0) {
    NewStateScalarByte(a, b, nx, ny);
  }

  return;
}

void GameOfLifeSimdByte(uint8_t* HWY_RESTRICT a, uint8_t* HWY_RESTRICT b,
                        const size_t nx, const size_t ny,
                        const size_t iterations) {
  size_t iter = 0;
  for (; iter + 1 < iterations; iter += 2) {
    HWY_DYNAMIC_DISPATCH(NewStateSimdByte)(a, b, nx, ny);
    HWY_DYNAMIC_DISPATCH(NewStateSimdByte)(b, a, nx, ny);
  }
  // Remainder iteration
  if (iterations - iter > 0) {
    HWY_DYNAMIC_DISPATCH(NewStateSimdByte)(a, b, nx, ny);
  }

  return;
}

int Run() {
  const size_t nx = 512;  // For ease of processing, make divisible by 64
  const size_t ny = 512;  // For ease of processing, make divisible by 64
  // Allocate a little larger than needed
  const size_t uint64_size = 10 + (nx * ny) / (8 * sizeof(uint64_t));
  const size_t iterations = 150;
  bool validated = true;
  AlignedFreeUniquePtr<uint64_t[]> a_scalar_bit =
      AllocateAligned<uint64_t>(uint64_size);
  AlignedFreeUniquePtr<uint64_t[]> b_scalar_bit =
      AllocateAligned<uint64_t>(uint64_size);
  AlignedFreeUniquePtr<uint64_t[]> a_simd_bit =
      AllocateAligned<uint64_t>(uint64_size);
  AlignedFreeUniquePtr<uint64_t[]> b_simd_bit =
      AllocateAligned<uint64_t>(uint64_size);
  hwy::AlignedVector<uint8_t> a_scalar_byte(10 + nx * ny);
  hwy::AlignedVector<uint8_t> b_scalar_byte(10 + nx * ny);
  hwy::AlignedVector<uint8_t> a_simd_byte(10 + nx * ny);
  hwy::AlignedVector<uint8_t> b_simd_byte(10 + nx * ny);

  HWY_DYNAMIC_DISPATCH(InitializeState)(a_scalar_bit.get(), a_simd_bit.get(),
                                        a_scalar_byte.data(),
                                        a_simd_byte.data(), nx, ny);

  // Record start time
  const double t_scalar_byte_0 = hwy::platform::Now();
  GameOfLifeScalarByte(a_scalar_byte.data(), b_scalar_byte.data(), nx, ny,
                       iterations);
  // Record end time and print execution time
  const double t_scalar_byte_1 = hwy::platform::Now();
  const double dt_scalar_byte = 1000.0 * (t_scalar_byte_1 - t_scalar_byte_0);
  std::cout << "Scalar Byte Execution Time: " << dt_scalar_byte << " ms"
            << std::endl;
  // Record start time
  const double t_scalar_bit_0 = hwy::platform::Now();
  GameOfLifeScalarBit(a_scalar_bit.get(), b_scalar_bit.get(), nx, ny,
                      iterations);
  // Record end time and print execution time
  const double t_scalar_bit_1 = hwy::platform::Now();
  const double dt_scalar_bit = 1000.0 * (t_scalar_bit_1 - t_scalar_bit_0);
  bool scalar_bit_validated;
  ((iterations % 2) == 0)
      ? scalar_bit_validated = HWY_DYNAMIC_DISPATCH(ValidateBit)(
            a_scalar_byte.data(), a_scalar_bit.get(), nx, ny)
      : scalar_bit_validated = HWY_DYNAMIC_DISPATCH(ValidateBit)(
            b_scalar_byte.data(), b_scalar_bit.get(), nx, ny);
  std::cout << "Scalar Bit Execution time: " << dt_scalar_bit << " ms"
            << std::endl;
  if (!scalar_bit_validated) {
    std::cout << "Scalar Bit Validation Failed" << std::endl;
    validated &= scalar_bit_validated;
  }
  // Record start time
  const double t_simd_byte_0 = hwy::platform::Now();
  GameOfLifeSimdByte(a_simd_byte.data(), b_simd_byte.data(), nx, ny,
                     iterations);
  // Record end time and print execution time
  const double t_simd_byte_1 = hwy::platform::Now();
  const double dt_simd_byte = 1000.0 * (t_simd_byte_1 - t_simd_byte_0);
  bool simd_byte_validated;
  ((iterations % 2) == 0)
      ? simd_byte_validated = HWY_DYNAMIC_DISPATCH(ValidateByte)(
            a_scalar_byte.data(), a_simd_byte.data(), nx, ny)
      : simd_byte_validated = HWY_DYNAMIC_DISPATCH(ValidateByte)(
            b_scalar_byte.data(), b_simd_byte.data(), nx, ny);
  std::cout << "SIMD Byte Execution Time: " << dt_simd_byte << " ms"
            << std::endl;
  if (!simd_byte_validated) {
    std::cout << "SIMD Byte Validation Failed" << std::endl;
    validated &= simd_byte_validated;
  }

  if (validated) {
    return 0;
  } else {
    return 1;
  }
}
}  // namespace hwy

int main() { return hwy::Run(); }
#endif  // HWY_ONCE
