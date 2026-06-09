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

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <iostream>
#include <numeric>

#include "hwy/aligned_allocator.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "hwy/examples/matrix_transpose_scatter_gather.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

/*
Highway SIMD Tutorial: Parallel Matrix Transpose (Gather & Scatter)

This example demonstrates how to transpose a matrix of size R x C (Rows x
Columns) into a matrix of size C x R using parallel SIMD Gather and Scatter.

1. Transpose via Scatter (Contiguous Load, Strided Scatter Store):
   - Load elements contiguously from a row slice of the input matrix.
   - Scatter them column-wise to the output matrix using strided indices.
2. Transpose via Gather (Strided Gather Load, Contiguous Store):
   - Gather elements column-wise from the input matrix using strided indices.
   - Store them contiguously to a row slice of the output matrix.

Remainder Handling:
If the matrix dimensions are not a multiple of the SIMD vector width, the
implementation cleanly handles remainder tails using LoadN/StoreN and
GatherIndexN/ScatterIndexN for partial vector operations without masking.
*/

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

// Scalar baseline matrix transpose.
void TransposeScalar(const uint32_t* HWY_RESTRICT input, uint32_t R, uint32_t C,
                     uint32_t* HWY_RESTRICT output) {
  for (uint32_t r = 0; r < R; ++r) {
    for (uint32_t c = 0; c < C; ++c) {
      output[c * R + r] = input[r * C + c];
    }
  }
}

// Transpose via Scatter: Contiguous Load -> Strided Scatter Store.
void TransposeScatter(const uint32_t* HWY_RESTRICT input, uint32_t R,
                      uint32_t C, uint32_t* HWY_RESTRICT output) {
  using D = hn::ScalableTag<uint32_t>;
  const D d;
  using V = hn::Vec<D>;
  using DI = hn::RebindToSigned<D>;
  const DI di;
  using VI = hn::Vec<DI>;

  HWY_LANES_CONSTEXPR size_t N = hn::Lanes(d);

  // Precompute constant strided indices vector: [0*R, 1*R, 2*R, ...]
  VI stride_indices =
      hn::Mul(hn::Iota(di, 0), hn::Set(di, static_cast<int32_t>(R)));

  // Loop over input rows
  for (uint32_t r = 0; r < R; ++r) {
    uint32_t c = 0;
    // Process columns in blocks of Lanes (N)
    for (; c + N <= C; c += N) {
      // Load contiguous row slice from input
      V row = hn::Load(d, input + r * C + c);

      // Scatter to output column-wise using constant strided indices and
      // moving base pointer
      hn::ScatterIndex(row, d, output + c * R + r, stride_indices);
    }

    // Handle remainder columns
    uint32_t remainder = C - c;
    if (remainder > 0) {
      // Load remainder using LoadN
      V row = hn::LoadN(d, input + r * C + c, remainder);
      // Scatter remainder using ScatterIndexN
      hn::ScatterIndexN(row, d, output + c * R + r, stride_indices, remainder);
    }
  }
}

// Transpose via Gather: Strided Gather Load -> Contiguous Store.
void TransposeGather(const uint32_t* HWY_RESTRICT input, uint32_t R, uint32_t C,
                     uint32_t* HWY_RESTRICT output) {
  using D = hn::ScalableTag<uint32_t>;
  const D d;
  using V = hn::Vec<D>;
  using DI = hn::RebindToSigned<D>;
  const DI di;
  using VI = hn::Vec<DI>;

  HWY_LANES_CONSTEXPR size_t N = hn::Lanes(d);

  // Precompute constant strided indices vector: [0*C, 1*C, 2*C, ...]
  VI stride_indices =
      hn::Mul(hn::Iota(di, 0), hn::Set(di, static_cast<int32_t>(C)));

  // Loop over output rows (which are input columns)
  for (uint32_t c = 0; c < C; ++c) {
    uint32_t r = 0;
    // Process output columns (input rows) in blocks of Lanes (N)
    for (; r + N <= R; r += N) {
      // Gather column slice from input using constant strided indices and
      // moving base pointer
      V col = hn::GatherIndex(d, input + r * C + c, stride_indices);

      // Store contiguous row slice to output
      hn::StoreU(col, d, output + c * R + r);
    }

    // Handle remainder rows
    uint32_t remainder = R - r;
    if (remainder > 0) {
      // Gather remainder using GatherIndexN
      V col = hn::GatherIndexN(d, input + r * C + c, stride_indices, remainder);
      // Store remainder using StoreN
      hn::StoreN(col, d, output + c * R + r, remainder);
    }
  }
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_EXPORT(TransposeScalar);
HWY_EXPORT(TransposeScatter);
HWY_EXPORT(TransposeGather);

// Visualizes a subgrid of the matrix in console.
static void PrintMatrix(const uint32_t* m, uint32_t R, uint32_t C,
                        uint32_t C_stride, const char* label) {
  std::cout << label << " (" << R << "x" << C << "):\n";
  for (uint32_t r = 0; r < R; ++r) {
    for (uint32_t c = 0; c < C; ++c) {
      printf("%3u ", m[r * C_stride + c]);
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

static void Run() {
  // Dimensions 64x64
  const uint32_t R = 64;
  const uint32_t C = 64;
  const size_t size = R * C;

  AlignedVector<uint32_t> input(size);
  AlignedVector<uint32_t> output(size);

  // Fill input with sequential values 0..size-1
  std::iota(input.begin(), input.end(), 0);

  std::cout << "Matrix Size: " << R << "x" << C << "\n\n";
  std::cout << "Input subgrid:\n";
  PrintMatrix(input.data(), 3, 5, C, "Input");

  // 1. Run Scalar Baseline
  HWY_DYNAMIC_DISPATCH(TransposeScalar)(input.data(), R, C, output.data());
  PrintMatrix(output.data(), 5, 3, R, "Scalar Transposed");

  // 2. Run Transpose via Scatter (SIMD)
  HWY_DYNAMIC_DISPATCH(TransposeScatter)(input.data(), R, C, output.data());
  PrintMatrix(output.data(), 5, 3, R, "SIMD Scatter Transposed");

  // 3. Run Transpose via Gather (SIMD)
  HWY_DYNAMIC_DISPATCH(TransposeGather)(input.data(), R, C, output.data());
  PrintMatrix(output.data(), 5, 3, R, "SIMD Gather Transposed");
}

}  // namespace hwy

int main() {
  hwy::Run();
  return 0;
}
#endif  // HWY_ONCE
