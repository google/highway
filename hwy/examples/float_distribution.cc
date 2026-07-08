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

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "hwy/base.h"

// For dynamic dispatch, specify the name of the current file so that
// foreach_target.h can re-include it.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "hwy/examples/float_distribution.cc"
#include "hwy/aligned_allocator.h"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/nanobenchmark.h"
#include "hwy/per_target.h"
#include "hwy/timer.h"

/*
Highway SIMD Tutorial: Float magnitude based compaction

This example demonstrates how to filter and compact float32 values by
converting them to approximate log2 8-bit integers (via IEEE-754 exponent
extraction and narrowing) and storing only the values exceeding a threshold
using SIMD CompressBlendedStore.

Key SIMD techniques shown:
1. Float manipulation: Abs
2. BitCasts from float to uint32
3. Shift of multiple lanes via ShiftRight.
4. Narrowing Conversions: Demoting four 32-bit vectors to two 16-bit vectors,
   then to one 8-bit vector using ReorderDemote2To.
5. Stream Compression: Using CompressBlendedStore to pack and write only
   unmasked elements to memory without scalar loops or branches.
*/

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// SIMD implementation of float distribution.
// Processes the input in blocks of 4 * Lanes(float) and handles any remainder
// using vector operations (LoadN) without falling back to scalar code.
// Returns the number of bytes written to 'out'.
// 'input_consumed' is updated with the number of input elements processed.
static HWY_INLINE size_t FloatDistributionSIMD(const float* HWY_RESTRICT in,
                                               size_t count,
                                               uint8_t threshold_val,
                                               uint8_t* HWY_RESTRICT out) {
#if HWY_TARGET == HWY_SCALAR
  // FloatDistributionSIMD is not supported on the HWY_SCALAR target

  (void)in;
  (void)count;
  (void)threshold_val;
  (void)out;
  return 0;
#else
  using DF32 = hn::ScalableTag<float>;
  using DU32 = hn::ScalableTag<uint32_t>;
  using DU16 = hn::Repartition<uint16_t, DU32>;
  using DU8 = hn::Repartition<uint8_t, DU32>;

  const DF32 df32;
  const DU32 du32;
  const DU16 du16;
  const DU8 du8;

  using VF32 = hn::Vec<DF32>;
  using VU32 = hn::Vec<DU32>;
  using VU16 = hn::Vec<DU16>;
  using VU8 = hn::Vec<DU8>;
  using MU8 = hn::Mask<DU8>;

  (void)du16;

  HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(df32);
  size_t out_idx = 0;

  const VU8 min_exp = hn::Set(du8, threshold_val);

  size_t i = 0;
  const size_t block_size = 4 * lanes;

  auto to_exponent = [&](VF32 v) HWY_ATTR {
    v = hn::Abs(v);
    return hn::ShiftRight<23>(hn::BitCast(du32, v));
  };

  auto demote_and_store = [&](VU32 exp0, VU32 exp1, VU32 exp2,
                              VU32 exp3) HWY_ATTR {
  // Demote compresses i32 -> i16 -> i8
  // It performs also clamping of values.
  // Reorder version dosen't assume any order of output elements.
  // If you want to preserve order you should use OrderedDemote2To.
    const VU16 exp16_01 = hn::ReorderDemote2To(du16, exp0, exp1);
    const VU16 exp16_23 = hn::ReorderDemote2To(du16, exp2, exp3);
    const VU8 exp8 = hn::ReorderDemote2To(du8, exp16_01, exp16_23);

    const MU8 final_mask = hn::Gt(exp8, min_exp);
    // CompressBlendedStore packs the values in 'exp8' according to 'final_mask'
    // and stores them to 'out'.
    // Blended version will not write outside number of bytes indicated by the
    // mask.
    // There is also CompressStore, which does the same job but faster but can
    // write more than indicated by the mask.
    out_idx += hn::CompressBlendedStore(exp8, final_mask, du8, out + out_idx);
  };

  for (; i + block_size <= count; i += block_size) {
    demote_and_store(to_exponent(hn::LoadU(df32, in + i)),
                     to_exponent(hn::LoadU(df32, in + i + lanes)),
                     to_exponent(hn::LoadU(df32, in + i + 2 * lanes)),
                     to_exponent(hn::LoadU(df32, in + i + 3 * lanes)));
  }

  // Handle remainder using vector operations (LoadN).
  const size_t remainder = count - i;
  if (remainder > 0) {
    auto load_chunk = [&](size_t offset) HWY_ATTR {
      if (remainder <= offset) {
        return hn::Zero(df32);
      }
      const size_t rem = remainder - offset;
      return hn::LoadN(df32, in + i + offset, rem);
    };

    demote_and_store(
        to_exponent(load_chunk(0 * lanes)), to_exponent(load_chunk(1 * lanes)),
        to_exponent(load_chunk(2 * lanes)), to_exponent(load_chunk(3 * lanes)));
    i += remainder;
    (void)i;
  }

  return out_idx;
#endif
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {

// HWY_EXPORT registers the table of pointers to the target-specific
// implementations of FloatDistributionSIMD.
HWY_EXPORT(FloatDistributionSIMD);

// Helper to extract the exponent (approximate log2) of a float.
HWY_INLINE uint32_t GetExponent(float v) {
  float abs_v = std::abs(v);
  uint32_t int_v = hwy::BitCastScalar<uint32_t>(abs_v);
  return int_v >> 23;
}

size_t FloatDistributionScalar(const float* HWY_RESTRICT in, size_t count,
                               float threshold, uint8_t* HWY_RESTRICT out) {
  const uint8_t threshold_val = static_cast<uint8_t>(GetExponent(threshold));
  size_t out_idx = 0;
  for (size_t i = 0; i < count; ++i) {
    const uint32_t exp = GetExponent(in[i]);
    const uint8_t exp8 = static_cast<uint8_t>(exp);

    if (exp8 > threshold_val) {
      out[out_idx++] = exp8;
    }
  }
  return out_idx;
}

// Verifies that SIMD and Scalar implementations produce the same output
// (modulo ordering, since SIMD uses ReorderDemote2To).
bool Verify(const float* HWY_RESTRICT in, size_t count, float threshold) {
  std::vector<uint8_t> expected(count);
  size_t expected_size =
      FloatDistributionScalar(in, count, threshold, expected.data());
  expected.resize(expected_size);

  const uint8_t threshold_val = static_cast<uint8_t>(GetExponent(threshold));
  std::vector<uint8_t> actual(count + hwy::VectorBytes());
  size_t actual_size = HWY_DYNAMIC_DISPATCH(FloatDistributionSIMD)(
      in, count, threshold_val, actual.data());
  actual.resize(actual_size);

  if (expected_size != actual_size) {
    fprintf(stderr, "Size mismatch for threshold %e: expected %zu, got %zu\n",
            threshold, expected_size, actual_size);
    return false;
  }

  // We use XOR to verify since ReorderDemote2To does not preserve order,
  // and XOR sum is order independent.
  uint8_t expected_xor = 0;
  for (uint8_t x : expected) expected_xor ^= x;
  uint8_t actual_xor = 0;
  for (uint8_t x : actual) actual_xor ^= x;

  if (expected_xor ^ actual_xor) {
    fprintf(stderr,
            "XOR sum mismatch for threshold %e (count %zu): expected 0x%02x, "
            "got 0x%02x\n",
            threshold, count, expected_xor, actual_xor);
    fprintf(stderr, "Expected (%zu): ", expected.size());
    for (uint8_t x : expected) fprintf(stderr, "%d ", x);
    fprintf(stderr, "\nActual (%zu):   ", actual.size());
    for (uint8_t x : actual) fprintf(stderr, "%d ", x);
    fprintf(stderr, "\n");
    return false;
  }

  return true;
}

int RunTests() {
  printf("Running tests...\n");

  // 1. Empty input
  if (!Verify(nullptr, 0, 1e-4f)) return 1;

  std::vector<float> thresholds = {0.0f, 1e-5f, 1e-4f, 1e-3f, 0.1f,
                                   0.5f, 0.9f,  1.0f,  2.0f};

  // 2. Random data test across various lengths to test vector remainder
  // handling.
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.5f, 1.5f);
  for (size_t size :
       {size_t{1},   size_t{2},   size_t{3},    size_t{4},    size_t{7},
        size_t{8},   size_t{15},  size_t{16},   size_t{17},   size_t{31},
        size_t{32},  size_t{33},  size_t{63},   size_t{64},   size_t{65},
        size_t{100}, size_t{127}, size_t{128},  size_t{129},  size_t{255},
        size_t{256}, size_t{257}, size_t{1000}, size_t{1024}, size_t{1025}}) {
    AlignedVector<float> in(size);
    for (size_t i = 0; i < size; ++i) {
      in[i] = dis(gen);
    }
    for (float threshold : thresholds) {
      if (!Verify(in.data(), in.size(), threshold)) return 1;
    }
  }

  printf("All tests PASSED.\n");

  // Benchmark
  const size_t bench_size = 100'000;
  const int reps = 1'000;
  AlignedVector<float> bench_in(bench_size);
  AlignedVector<uint8_t> bench_out(bench_size + HWY_MAX_BYTES);
  for (size_t i = 0; i < bench_size; ++i) {
    bench_in[i] = dis(gen);
  }
  const float bench_threshold = 0.1f;
  const uint8_t bench_threshold_val =
      static_cast<uint8_t>(GetExponent(bench_threshold));
  printf("Benchmarking with array of %zu floats (%d reps)...\n", bench_size,
         reps);

  const double t_scalar_0 = hwy::platform::Now();
  size_t scalar_written = 0;
  for (int r = 0; r < reps; ++r) {
    scalar_written = FloatDistributionScalar(bench_in.data(), bench_size,
                                             bench_threshold, bench_out.data());
  }
  const double t_scalar_1 = hwy::platform::Now();
  const double dt_scalar = t_scalar_1 - t_scalar_0;

  const double t_simd_0 = hwy::platform::Now();
  size_t simd_written = 0;
  for (int r = 0; r < reps; ++r) {
    simd_written = HWY_DYNAMIC_DISPATCH(FloatDistributionSIMD)(
        bench_in.data(), bench_size, bench_threshold_val, bench_out.data());
  }
  const double t_simd_1 = hwy::platform::Now();
  const double dt_simd = t_simd_1 - t_simd_0;

  printf("Scalar written: %zu (took %f seconds)\n", scalar_written, dt_scalar);
  printf("SIMD written:   %zu (took %f seconds, speedup: %fx)\n", simd_written,
         dt_simd, dt_scalar / dt_simd);

  return 0;
}

}  // namespace hwy

int main() { return hwy::RunTests(); }
#endif  // HWY_ONCE
