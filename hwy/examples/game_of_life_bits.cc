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
9) https://web.archive.org/web/20060316100407/http://www.onjava.com/pub/a/onjava/2005/02/02/bitsets.html?page=2
10) Cameron Browne, "BitBoard Methods for Games" https://doi.org/10.3233/ICG-2014-37202 or https://eprints.qut.edu.au/85005/
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
  std::cout << "Calculated" << std::endl;
  for (size_t j = 0; j < 2; j++) {
    for (size_t i = 0; i < nx; i++) {
      const size_t arr = static_cast<size_t>((i + j*nx) / (8 * sizeof(uint64_t)));
      const size_t bit = static_cast<size_t>((i + j*nx) % (8 * sizeof(uint64_t)));
      const uint64_t one = 1;
      uint16_t val = (out[arr] & static_cast<uint64_t>(one << bit))
                    ? static_cast<uint16_t>(1)
                    : static_cast<uint16_t>(0);
      std::cout << val;
    }
    std::cout << std::endl;
  }
  std::cout << "Reference" << std::endl;
  for (size_t j = 0; j < 2; j++) {
    for (size_t i = 0; i < nx; i++) {
      std::cout << static_cast<uint16_t>(ref_byte[i+j*nx]);
    }
    std::cout << std::endl;
  }
  for (i = 0; i + NU8 <= nx * ny; i += NU8) {
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
  MU8 two_and_me_alive_or_three_alive_and_me_dead;
  MU8 alive;

  for (size_t i = 0; i + NU8 <= nx; i += NU8) {
    for (size_t j = 0; j < ny; j++) {
      previous_row = hn::LoadU(du8, in + ((ny - 1 + j) % ny) * nx + i);
      current_row = hn::LoadU(du8, in + j * nx + i);
      next_row = hn::LoadU(du8, in + ((1 + j) % ny) * nx + i);
      neighbor_count = NeighborCountSimdByte(previous_row, current_row,
                                             next_row, in, NU8, nx, ny, i, j);
      two_and_me_alive_or_three_alive_and_me_dead =
          hn::Eq(hn::Add(neighbor_count, current_row),
                 hn::Set(du8, static_cast<uint8_t>(3)));
      three_alive =
          hn::Eq(neighbor_count, hn::Set(du8, static_cast<uint8_t>(3)));
      alive = hn::Or(three_alive, two_and_me_alive_or_three_alive_and_me_dead);
      new_state = hn::MaskedSet(du8, alive, static_cast<uint8_t>(1));
      hn::StoreU(new_state, du8, out + j * nx + i);
    }
  }
  return;
}

VU64 GetNewStateSimdBitPepicelli(
       const VU64 previous_row_left_shift, const VU64 previous_row,
       const VU64 previous_row_right_shift, const VU64 left_shift,
       const VU64 current_row, const VU64 right_shift,
       const VU64 next_row_left_shift, const VU64 next_row,
       const VU64 next_row_right_shift) {

  // Accumulate live neighbor counts
  VU64 s0 = hn::Not(hn::Or(previous_row_left_shift, previous_row));
  VU64 s1 = hn::Xor(previous_row_left_shift, previous_row);
  VU64 s2 = hn::And(previous_row_left_shift, previous_row);

  VU64 s3 = hn::And(s2, left_shift);
  s2 = hn::Or(hn::AndNot(left_shift, s2), hn::And(s1, left_shift));
  s1 = hn::Or(hn::AndNot(left_shift, s1), hn::And(s0, left_shift));
  s0 = hn::AndNot(left_shift, s0);

//  VU64 s4 = hn::And(s3, right_shift);
  s3 = hn::Or(hn::AndNot(right_shift, s3), hn::And(s2, right_shift));
  s2 = hn::Or(hn::AndNot(right_shift, s2), hn::And(s1, right_shift));
  s1 = hn::Or(hn::AndNot(right_shift, s1), hn::And(s0, right_shift));
  s0 = hn::AndNot(right_shift, s0);

//  VU64 s5 = hn::And(s4, previous_row_right_shift);
//  s4 = hn::Or(hn::AndNot(previous_row_right_shift, s4),
//              hn::And(s3, previous_row_right_shift));
  s3 = hn::Or(hn::AndNot(previous_row_right_shift, s3),
              hn::And(s2, previous_row_right_shift));
  s2 = hn::Or(hn::AndNot(previous_row_right_shift, s2),
              hn::And(s1, previous_row_right_shift));
  s1 = hn::Or(hn::AndNot(previous_row_right_shift, s1),
              hn::And(s0, previous_row_right_shift));
  s0 = hn::AndNot(previous_row_right_shift, s0);

//  VU64 s6 = hn::And(s5, next_row_left_shift);
//  s5 = hn::Or(hn::AndNot(next_row_left_shift, s5),
//              hn::And(s4, next_row_left_shift));
//  s4 = hn::Or(hn::AndNot(next_row_left_shift, s4),
//              hn::And(s3, next_row_left_shift));
  s3 = hn::Or(hn::AndNot(next_row_left_shift, s3),
              hn::And(s2, next_row_left_shift));
  s2 = hn::Or(hn::AndNot(next_row_left_shift, s2),
              hn::And(s1, next_row_left_shift));
  s1 = hn::Or(hn::AndNot(next_row_left_shift, s1),
              hn::And(s0, next_row_left_shift));
  s0 = hn::AndNot(next_row_left_shift, s0);

//  VU64 s7 = hn::And(s6, next_row);
//  s6 = hn::Or(hn::AndNot(next_row, s6), hn::And(s5, next_row));
//  s5 = hn::Or(hn::AndNot(next_row, s5), hn::And(s4, next_row));
//  s4 = hn::Or(hn::AndNot(next_row, s4), hn::And(s3, next_row));
  s3 = hn::Or(hn::AndNot(next_row, s3), hn::And(s2, next_row));
  s2 = hn::Or(hn::AndNot(next_row, s2), hn::And(s1, next_row));
  s1 = hn::Or(hn::AndNot(next_row, s1), hn::And(s0, next_row));
//  s0 = hn::AndNot(next_row, s0);

//  VU64 s8 = hn::And(s7, next_row_right_shift);
//  s7 = hn::Or(hn::AndNot(next_row_right_shift, s7),
//              hn::And(s6, next_row_right_shift));
//  s6 = hn::Or(hn::AndNot(next_row_right_shift, s6),
//              hn::And(s5, next_row_right_shift));
//  s5 = hn::Or(hn::AndNot(next_row_right_shift, s5),
//              hn::And(s4, next_row_right_shift));
//  s4 = hn::Or(hn::AndNot(next_row_right_shift, s4),
//              hn::And(s3, next_row_right_shift));
  s3 = hn::Or(hn::AndNot(next_row_right_shift, s3),
              hn::And(s2, next_row_right_shift));
  s2 = hn::Or(hn::AndNot(next_row_right_shift, s2),
              hn::And(s1, next_row_right_shift));
//  s1 = hn::Or(hn::AndNot(next_row_right_shift, s1),
//              hn::And(s0, next_row_right_shift));
//  s0 = hn::AndNot(next_row_right_shift, s0);

  // Return live cases
  return hn::Or(hn::And(current_row, s2), s3);
}

VU64 GetNewStateSimdBitSortingNetwork(
       const VU64 previous_row_left_shift, const VU64 previous_row,
       const VU64 previous_row_right_shift, const VU64 left_shift,
       const VU64 current_row, const VU64 right_shift,
       const VU64 next_row_left_shift, const VU64 next_row,
       const VU64 next_row_right_shift) {

  // Accumulate live neighbor counts
  VU64 One_a0 = hn::And(previous_row_left_shift, previous_row);
  VU64 One_o0 = hn::Or(previous_row_left_shift, previous_row);

  VU64 One_a1 = hn::And(previous_row_right_shift, right_shift);
  VU64 One_o1 = hn::Or(previous_row_right_shift, right_shift);

  VU64 One_a2 = hn::And(next_row_right_shift, next_row);
  VU64 One_o2 = hn::Or(next_row_right_shift, next_row);

  VU64 One_a3 = hn::And(next_row_left_shift, left_shift);
  VU64 One_o3 = hn::Or(next_row_left_shift, left_shift);

  VU64 Two_a0 = hn::And(One_a0, One_a1);
  VU64 Two_o0 = hn::Or(One_a0, One_a1);

  VU64 Two_a1 = hn::And(One_o0, One_o1);
  VU64 Two_o1 = hn::Or(One_o0, One_o1);

  VU64 Two_a2 = hn::And(One_a2, One_a3);
  VU64 Two_o2 = hn::Or(One_a2, One_a3);

  VU64 Two_a3 = hn::And(One_o2, One_o3);
  VU64 Two_o3 = hn::Or(One_o2, One_o3);

  VU64 Three_a0 = hn::And(Two_o0, Two_a1);
  VU64 Three_o0 = hn::Or(Two_o0, Two_a1);

  VU64 Three_a1 = hn::And(Two_o2, Two_a3);
  VU64 Three_o1 = hn::Or(Two_o2, Two_a3);

  VU64 Four_o0 = hn::Or(Two_a0, Two_a2);
  VU64 Four_o1 = hn::Or(Three_a0, Three_a1);
  VU64 Four_a0 = hn::And(Three_o0, Three_o1);
  VU64 Four_o2 = hn::Or(Three_o0, Three_o1);
  VU64 Four_a1 = hn::And(Two_o1, Two_o3);

  VU64 Five_o0 = hn::Or(Four_o0, Four_a0);
  VU64 Five_a0 = hn::And(Four_o1, Four_a1);
  VU64 Five_o1 = hn::Or(Four_o1, Four_a1);

  VU64 Six_o0 = hn::Or(Five_o0, Five_a0);
  VU64 Six_a0 = hn::And(Five_o1, Four_o2);
  VU64 Six_o1 = hn::Or(Five_o1, Four_o2);

  VU64 Seven_a0 = hn::And(Six_o1, current_row);

  VU64 Eight_o0 = hn::Or(Seven_a0, Six_a0);

  
  // Return live cases
  return hn::AndNot(Six_o0, Eight_o0);
}


VU64 GetNewStateSimdBitAdders(
       const VU64 previous_row_left_shift, const VU64 previous_row,
       const VU64 previous_row_right_shift, const VU64 left_shift,
       const VU64 current_row, const VU64 right_shift,
       const VU64 next_row_left_shift, const VU64 next_row,
       const VU64 next_row_right_shift) {

  // Bit plane adders
  VU64 bit1 = hn::Zero(du64);
  VU64 bit2 = hn::Zero(du64);
  VU64 bit3 = hn::Zero(du64);
  // Lambda for checking state
  auto bit_add = [&](const VU64 in) HWY_ATTR -> void {
    VU64 carry1 = hn::And(bit1, in);
    VU64 carry2 = hn::And(bit2, carry1);
    bit1 = hn::Xor(bit1, in);
    bit2 = hn::Xor(bit2, carry1);
    bit3 = hn::Or(bit3, carry2);
    return;
  };
  // Accumulate neighbor counts
  bit_add(previous_row_left_shift);
  bit_add(previous_row);
  bit_add(previous_row_right_shift);
  bit_add(right_shift);
  bit_add(next_row_right_shift);
  bit_add(next_row);
  bit_add(next_row_left_shift);
  bit_add(left_shift);
  return hn::AndNot(bit3, hn::And(bit2, hn::Or(current_row, bit1)));
}

void NewStateSimdBit(const uint64_t* HWY_RESTRICT in,
                     uint64_t* HWY_RESTRICT out,
                     const size_t nx, const size_t ny) {
  const size_t NU64 = hn::Lanes(du64);
  VU64 previous_row_left_shift;
  VU64 previous_row;
  VU64 previous_row_right_shift;
  VU64 right_shift;
  VU64 next_row_right_shift;
  VU64 next_row;
  VU64 next_row_left_shift;
  VU64 left_shift;
  VU64 current_row;
  VU64 new_state;
  size_t ind;
  size_t block_size = 8 * sizeof(uint64_t);
  size_t vec_size = NU64 * block_size;
  for (size_t i = 0; i + vec_size <= nx ; i += vec_size) {
	  std::cout << "Processing " << i << std::endl;
    for (size_t j = 0; j < ny; j++) {
      previous_row = hn::LoadU(du64, in + (((ny - 1 + j) % ny) * nx + i)/
                                          block_size);
      current_row = hn::LoadU(du64, in + (j * nx + i)/block_size);
      next_row = hn::LoadU(du64, in + (((1 + j) % ny) * nx + i) / block_size);
      // previous row left
      ind = (((ny - 1 + j) % ny) * nx + (nx - block_size + i) % nx) /
	    block_size;
      previous_row_left_shift = hn::Or(
        hn::ShiftLeft<1>(previous_row),
        hn::ShiftRight<63>(hn::Slide1UpOr(in[ind], du64, previous_row)));
      // previous row right
      ind = (((ny - 1 + j) % ny) * nx + ((block_size + i) % nx)) / block_size;
      previous_row_right_shift = hn::Or(
        hn::ShiftRight<1>(previous_row),
        hn::ShiftLeft<63>(hn::Slide1DownOr(in[ind], du64, previous_row)));
      // left
      ind = (j * nx + ((nx - block_size + i) % nx)) / block_size;
      left_shift = hn::Or(
        hn::ShiftLeft<1>(current_row),
        hn::ShiftRight<63>(hn::Slide1UpOr(in[ind], du64, current_row)));
      // right
      ind = (j * nx + ((block_size + i) % nx)) / block_size;
      right_shift = hn::Or(
        hn::ShiftRight<1>(current_row),
        hn::ShiftLeft<63>(hn::Slide1DownOr(in[ind], du64, current_row)));
      // next row left
      ind = (((1 + j) % ny) * nx + (nx - block_size + i) % nx) / block_size;
      next_row_left_shift = hn::Or(
        hn::ShiftLeft<1>(next_row),
        hn::ShiftRight<63>(hn::Slide1UpOr(in[ind], du64, next_row)));
      // next row right
      ind = (((1 + j) % ny) * nx + ((block_size + i) % nx)) / block_size;
      next_row_right_shift = hn::Or(
        hn::ShiftRight<1>(next_row),
        hn::ShiftLeft<63>(hn::Slide1DownOr(in[ind], du64, next_row)));
      // Compare GetNewStateSimdBitAdders, GetNewStateSimdBitPepicelli and
      // GetNewStateSimdBitSortingNetwork.  The sorting network has the fewest
      // operations, but Pepicelli may have more favourable memory accesses.
      new_state = GetNewStateSimdBitSortingNetwork(
        previous_row_left_shift, previous_row, previous_row_right_shift,
        left_shift, current_row, right_shift,
	next_row_left_shift, next_row, next_row_right_shift);
      hn::StoreU(new_state, du64, out + (j * nx + i)/block_size);
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
HWY_EXPORT(NewStateSimdBit);
HWY_EXPORT(ValidateByte);

uint64_t UpdateBitSortingNetwork(const uint64_t previous_left,
                                 const uint64_t previous,
                                 const uint64_t previous_right,
                                 const uint64_t current_right,
                                 const uint64_t next_right,
                                 const uint64_t next,
                                 const uint64_t next_left,
                                 const uint64_t current_left,
                                 const uint64_t current) {

// https://www.moria.us/old/3/programs/life/
  // Process live neighbor counts
  uint64_t One_a0 = previous_left & previous;
  uint64_t One_o0 = previous_left | previous;

  uint64_t One_a1 = previous_right & current_right;
  uint64_t One_o1 = previous_right | current_right;

  uint64_t One_a2 = next_right & next;
  uint64_t One_o2 = next_right | next;

  uint64_t One_a3 = next_left & current_left;
  uint64_t One_o3 = next_left | current_left;

  uint64_t Two_a0 = One_a0 & One_a1;
  uint64_t Two_o0 = One_a0 | One_a1;

  uint64_t Two_a1 = One_o0 & One_o1;
  uint64_t Two_o1 = One_o0 | One_o1;

  uint64_t Two_a2 = One_a2 & One_a3;
  uint64_t Two_o2 = One_a2 | One_a3;

  uint64_t Two_a3 = One_o2 & One_o3;
  uint64_t Two_o3 = One_o2 | One_o3;

  uint64_t Three_a0 = Two_o0 & Two_a1;
  uint64_t Three_o0 = Two_o0 | Two_a1;

  uint64_t Three_a1 = Two_o2 & Two_a3;
  uint64_t Three_o1 = Two_o2 | Two_a3;

  uint64_t Four_o0 = Two_a0 | Two_a2;
  uint64_t Four_o1 = Three_a0 | Three_a1;
  uint64_t Four_a0 = Three_o0 & Three_o1;
  uint64_t Four_o2 = Three_o0 | Three_o1;
  uint64_t Four_a1 = Two_o1 & Two_o3;

  uint64_t Five_o0 = Four_o0 | Four_a0;
  uint64_t Five_a0 = Four_o1 & Four_a1;
  uint64_t Five_o1 = Four_o1 | Four_a1;

  uint64_t Six_o0 = Five_o0 | Five_a0;
  uint64_t Six_a0 = Five_o1 & Four_o2;
  uint64_t Six_o1 = Five_o1 | Four_o2;

  uint64_t Seven_a0 = Six_o1 & current;

  uint64_t Eight_o0 = Seven_a0 | Six_a0;

  // Return live cases
  return ((~Six_o0) & Eight_o0);
}


uint64_t UpdateBitPepicelli(const uint64_t previous_left,
                            const uint64_t previous,
                            const uint64_t previous_right,
                            const uint64_t current_right,
                            const uint64_t next_right,
                            const uint64_t next,
                            const uint64_t next_left,
                            const uint64_t current_left,
                            const uint64_t current) {

// https://web.archive.org/web/20060316100407/http://www.onjava.com/pub/a/onjava/2005/02/02/bitsets.html?page=2
  // Accumulate live neighbor counts
  uint64_t s0 = ~(previous_left | previous);
  uint64_t s1 = previous_left ^ previous;
  uint64_t s2 = previous_left & previous;

  uint64_t s3 = s2 & current_left;
  s2 = (s2 & ~current_left) | (s1 & current_left);
  s1 = (s1 & ~current_left) | (s0 & current_left);
  s0 = s0 & ~current_left;

//  uint64_t s4 = s3 & current_right;
  s3 = (s3 & ~current_right) | (s2 & current_right);
  s2 = (s2 & ~current_right) | (s1 & current_right);
  s1 = (s1 & ~current_right) | (s0 & current_right);
  s0 = s0 & ~current_right;

//  uint64_t s5 = s4 & previous_right;
//  s4 = (s4 & ~previous_right) | (s3 & previous_right);
  s3 = (s3 & ~previous_right) | (s2 & previous_right);
  s2 = (s2 & ~previous_right) | (s1 & previous_right);
  s1 = (s1 & ~previous_right) | (s0 & previous_right);
  s0 = s0 & ~previous_right;

//  uint64_t s6 = s5 & next_left;
//  s5 = (s5 & ~next_left) | (s4 & next_left);
//  s4 = (s4 & ~next_left) | (s3 & next_left);
  s3 = (s3 & ~next_left) | (s2 & next_left);
  s2 = (s2 & ~next_left) | (s1 & next_left);
  s1 = (s1 & ~next_left) | (s0 & next_left);
  s0 = s0 & ~next_left;

//  uint64_t s7 = s6 & next;
//  s6 = (s6 & ~next) | (s5 & next);
//  s5 = (s5 & ~next) | (s4 & next);
//  s4 = (s4 & ~next) | (s3 & next);
  s3 = (s3 & ~next) | (s2 & next);
  s2 = (s2 & ~next) | (s1 & next);
  s1 = (s1 & ~next) | (s0 & next);
//  s0 = s0 & ~next;

//  uint64_t s8 = s7 & next_right;
//  s7 = (s7 & ~next_right) | (s6 & next_right);
//  s6 = (s6 & ~next_right) | (s5 & next_right);
//  s5 = (s5 & ~next_right) | (s4 & next_right);
//  s4 = (s4 & ~next_right) | (s3 & next_right);
  s3 = (s3 & ~next_right) | (s2 & next_right);
  s2 = (s2 & ~next_right) | (s1 & next_right);
//  s1 = (s1 & ~next_right) | (s0 & next_right);
//  s0 = s0 & ~next_right;

  // Return live cases
  return ((current & s2) | s3);	
}

uint64_t UpdateBitAdders(const uint64_t previous_left,
                         const uint64_t previous,
                         const uint64_t previous_right,
                         const uint64_t current_right,
                         const uint64_t next_right,
                         const uint64_t next,
                         const uint64_t next_left,
                         const uint64_t current_left,
                         const uint64_t current) {
  // Bit plane adders
  uint64_t bit1 = 0;
  uint64_t bit2 = 0;
  uint64_t bit3 = 0;
  // Lambda function to update state
  auto adder =[&](const uint64_t line) -> void {
    uint64_t carry1 = bit1 & line;
    uint64_t carry2 = bit2 & carry1;
    bit1 ^= line;
    bit2 ^= carry1;
    bit3 |= carry2;
  };
  // Accumulate live neighbor counts
  adder(previous_left);
  adder(previous);
  adder(previous_right);
  adder(current_right);
  adder(next_right);
  adder(next);
  adder(next_left);
  adder(current_left);

  // Return live cases
  return ((current | bit1) & bit2 & ~bit3);
}

void NewStateScalarBit(const uint64_t* HWY_RESTRICT in,
                       uint64_t* HWY_RESTRICT out, const size_t nx,
                       const size_t ny) {

  size_t var_size = 8 * sizeof(uint64_t);
  for (size_t i = 0; i + var_size <= nx; i+=var_size) {
    for (size_t j = 0; j < ny; j++) {
      uint64_t previous = in[(((ny-1+j)%ny)*nx + i)/var_size];
      uint64_t previous_left = (previous << 1) |
                               (in[(((ny-1+j)%ny)*nx + (nx-var_size+i)%nx)/var_size] >> 63);
      uint64_t previous_right = (previous >> 1) |
                               (in[(((ny-1+j)%ny)*nx + (var_size+i)%nx)/var_size] << 63);
      uint64_t current = in[(j*nx + i)/var_size];
      uint64_t current_left = (current << 1) |
                              (in[(j*nx + (nx-var_size+i)%nx)/var_size] >> 63);
      uint64_t current_right = (current >> 1) |
                               (in[(j*nx + (var_size+i)%nx)/var_size] << 63);      
      uint64_t next = in[(((j+1)%ny)*nx + i)/var_size];
      uint64_t next_left = (next << 1) |
                           (in[(((1+j)%ny)*nx + (nx-var_size+i)%nx)/var_size] >> 63);
      uint64_t next_right = (next >> 1) |
                            (in[(((1+j)%ny)*nx + (var_size+i)%nx)/var_size] << 63);
      // Can use either UpdateBitAdders, UpdateBitPepicelli or
      // UpdateSortingNetwork.  BitAdders seems to be the most common method,
      // but the sorting network has a lower operation count and the Pepicelli
      // method may have more favourable data accesses patterns.
      out[(j*nx + i)/var_size] =
        UpdateBitSortingNetwork(previous_left, previous, previous_right,
                        current_right, next_right, next, next_left,
                        current_left, current);
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

void GameOfLifeSimdBit(uint64_t* HWY_RESTRICT a, uint64_t* HWY_RESTRICT b,
                       const size_t nx, const size_t ny,
                       const size_t iterations) {
  size_t iter = 0;
  for (; iter + 1 < iterations; iter += 2) {
    HWY_DYNAMIC_DISPATCH(NewStateSimdBit)(a, b, nx, ny);
    HWY_DYNAMIC_DISPATCH(NewStateSimdBit)(b, a, nx, ny);
  }
  // Remainder iteration
  if (iterations - iter > 0) {
    HWY_DYNAMIC_DISPATCH(NewStateSimdBit)(a, b, nx, ny);
  }

  return;
}


int Run() {
  const size_t nx = 256; // For ease of processing, make divisible by 64
  const size_t ny = 8;  // Want to have at least 3 rows
  // Allocate a little larger than needed
  const size_t uint64_size = 10 + (nx * ny) / (8 * sizeof(uint64_t));
  const size_t iterations = 1;
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
  // Record start time
  const double t_simd_bit_0 = hwy::platform::Now();
  GameOfLifeSimdBit(a_simd_bit.get(), b_simd_bit.get(), nx, ny,
                    iterations);
  // Record end time and print execution time
  const double t_simd_bit_1 = hwy::platform::Now();
  const double dt_simd_bit = 1000.0 * (t_simd_bit_1 - t_simd_bit_0);
  bool simd_bit_validated;
  ((iterations % 2) == 0)
      ? simd_bit_validated = HWY_DYNAMIC_DISPATCH(ValidateBit)(
            a_scalar_byte.data(), a_simd_bit.get(), nx, ny)
      : scalar_bit_validated = HWY_DYNAMIC_DISPATCH(ValidateBit)(
            b_scalar_byte.data(), b_simd_bit.get(), nx, ny);
  std::cout << "Simd Bit Execution time: " << dt_simd_bit << " ms"
            << std::endl;
  if (!simd_bit_validated) {
    std::cout << "Simd Bit Validation Failed" << std::endl;
    validated &= simd_bit_validated;
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
