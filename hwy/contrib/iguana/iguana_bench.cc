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
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <utility>

#include "hwy/contrib/iguana/iguana_detail.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_IGUANA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/iguana/iguana.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/nanobenchmark.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/iguana/iguana_bench.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/iguana/iguana-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
#if (HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128) || HWY_IDE

FILE* OpenDataFile(const char* rel_path) {
  FILE* f = fopen(rel_path, "rb");
  if (f != nullptr) return f;
  char candidate[1024];
  const char* test_srcdir = getenv("TEST_SRCDIR");
  if (test_srcdir != nullptr) {
    snprintf(candidate, sizeof(candidate), "%s/google3/%s", test_srcdir,
             rel_path);
    f = fopen(candidate, "rb");
    if (f != nullptr) return f;
  }
  const char* runfiles_dir = getenv("RUNFILES_DIR");
  if (runfiles_dir != nullptr) {
    snprintf(candidate, sizeof(candidate), "%s/google3/%s", runfiles_dir,
             rel_path);
    f = fopen(candidate, "rb");
    if (f != nullptr) return f;
  }
  return nullptr;
}

AlignedVector<uint8_t> ReadFile(const char* rel_path) {
  FILE* f = OpenDataFile(rel_path);
  HWY_ASSERT_M(f != nullptr, rel_path);

  fseek(f, 0, SEEK_END);
  const int64_t file_size = ftell(f);
  HWY_ASSERT(file_size >= 0);
  fseek(f, 0, SEEK_SET);

  AlignedVector<uint8_t> buf(static_cast<size_t>(file_size));
  HWY_ASSERT(fread(buf.data(), 1, buf.size(), f) == buf.size());
  fclose(f);
  return buf;
}

struct MeasureResult {
  double ns;
  double mad_percent;
};

template <class Func>
MeasureResult Measure(const Func& func) {
  FuncInput input = Unpredictable1();
  Params params = DefaultBenchmarkParams();
  params.min_samples_per_eval = 2;
  params.max_evals = 4;
  params.verbose = false;
  Result results[1];

  const size_t num_results = MeasureClosure(func, &input, 1, results, params);
  if (num_results == 1) {
    return {results[0].ticks / platform::InvariantTicksPerSecond() * 1E9,
            results[0].variability * 100.0};
  } else {
    HWY_WARN("Measurement failed.");
    return MeasureResult{};
  }
}

HWY_NOINLINE void TestIguanaThroughput() {
  // Too slow under MSAN/TSAN.
  if constexpr (HWY_IS_MSAN || HWY_IS_TSAN) return;

  AlignedVector<uint8_t> input =
      ReadFile("third_party/highway/hwy/contrib/testdata/silesia_dickens.txt");
  if constexpr (HWY_IS_DEBUG_BUILD) {
    if (input.size() > (size_t{256} << 10)) {
      input.resize(size_t{256} << 10);
    }
  }
  const size_t num_bytes = input.size();
  HWY_ASSERT(num_bytes > 0);

  size_t comp_size = 0;
  AlignedVector<uint8_t> compressed_buf;

  // Compress's throughput depends on where its buffers happen to land. The
  // six stream write cursors, the source and the 2 MiB hash-chain table all
  // contend for the same L1 sets, and two placements of the same code can
  // differ by ~10%. hwy's allocator rotates the alignment offset of each
  // allocation, so measuring several independently allocated buffer sets and
  // taking the median samples that distribution instead of reporting a single
  // draw from it. Without this, changes smaller than ~10% are indistinguishable
  // from placement luck.
  constexpr size_t kPlacements = 5;
  double enc_ns[kPlacements];
  double enc_mad = 0.0;
  // Created once, outside the loop: the ctor starts the threads, and billing
  // that to the first placement would skew the median.
  ThreadPool pool(ThreadPool::NumThreadsFromCores());
  for (size_t p = 0; p < kPlacements; ++p) {
    // All three buffers are reallocated, including the source: it is the most
    // heavily read of them, so holding it at one address for the whole process
    // would leave most of the distribution unsampled.
    AlignedVector<uint8_t> src(num_bytes);
    CopyBytes(input.data(), src.data(), num_bytes);
    const Span<const uint8_t> in_span(src.data(), num_bytes);
    AlignedVector<uint8_t> buf(hwy::iguana::MaxCompressedSize(num_bytes));
    const Span<uint8_t> out_span(buf.data(), buf.size());
    // Reserve up front so the measurement covers only the compression itself,
    // which is how a caller that compresses more than one block would see it.
    hwy::iguana::IguanaWorkspace ws;
    HWY_ASSERT(ws.Reserve(num_bytes, pool.NumWorkers()));
    const MeasureResult r = Measure([&](FuncInput func_input) {
      comp_size = hwy::iguana::Compress(in_span, out_span, ws, pool);
      return buf[func_input];
    });
    enc_ns[p] = r.ns;
    enc_mad = HWY_MAX(enc_mad, r.mad_percent);
    compressed_buf = std::move(buf);  // keep the last for the decode benchmarks
  }
  HWY_ASSERT(comp_size > 0);
  std::sort(enc_ns, enc_ns + kPlacements);
  const double enc_ns_median = enc_ns[kPlacements / 2];

  const Span<const uint8_t> comp_span(compressed_buf.data(), comp_size);
  const double ratio =
      static_cast<double>(num_bytes) / static_cast<double>(comp_size);
  fprintf(stderr,
          "Iguana Compress (%s): %7.2f MB -> %7.2f MB (%4.2fx) in %7.2f ms "
          "[%6.2f..%6.2f over %zu placements] = %6.3f MB/s; MAD=%5.2f%%\n",
          TargetName(HWY_TARGET), static_cast<double>(num_bytes) * 1E-6,
          static_cast<double>(comp_size) * 1E-6, ratio, enc_ns_median * 1E-6,
          enc_ns[0] * 1E-6, enc_ns[kPlacements - 1] * 1E-6, kPlacements,
          static_cast<double>(num_bytes) / enc_ns_median * 1E3, enc_mad);

  hwy::iguana::IguanaWorkspace dec_ws;
  HWY_ASSERT(dec_ws.Reserve(num_bytes, pool.NumWorkers()));

  AlignedVector<uint8_t> dec_simd(num_bytes);
  const Span<uint8_t> dec_simd_span(dec_simd.data(), dec_simd.size());
  const MeasureResult dec_simd_result = Measure([&](FuncInput func_input) {
    const size_t written =
        DecompressStatic(comp_span, dec_simd_span, dec_ws, pool);
    HWY_ASSERT(written == num_bytes);
    return dec_simd[func_input];
  });
  HWY_ASSERT(memcmp(dec_simd.data(), input.data(), num_bytes) == 0);
  fprintf(stderr,
          "Iguana DecompressStatic (%s): %7.2f MB in %7.2f ms = %6.3f GB/s; "
          "MAD=%5.2f%%\n",
          TargetName(HWY_TARGET), static_cast<double>(num_bytes) * 1E-6,
          dec_simd_result.ns * 1E-6,
          static_cast<double>(num_bytes) / dec_simd_result.ns,
          dec_simd_result.mad_percent);

  AlignedVector<uint8_t> dec_scalar(num_bytes);
  const Span<uint8_t> dec_scalar_span(dec_scalar.data(), dec_scalar.size());
  const MeasureResult dec_scalar_result = Measure([&](FuncInput func_input) {
    const size_t written =
        hwy::iguana::DecompressScalar(comp_span, dec_scalar_span, dec_ws, pool);
    HWY_ASSERT(written == num_bytes);
    return dec_scalar[func_input];
  });
  HWY_ASSERT(memcmp(dec_scalar.data(), input.data(), num_bytes) == 0);
  fprintf(stderr,
          "Iguana DecompressScalar: %7.2f MB in %7.2f ms = %6.3f GB/s; "
          "MAD=%5.2f%%\n",
          static_cast<double>(num_bytes) * 1E-6, dec_scalar_result.ns * 1E-6,
          static_cast<double>(num_bytes) / dec_scalar_result.ns,
          dec_scalar_result.mad_percent);
}

#else   // HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128
void TestIguanaThroughput() {}
#endif  // HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(IguanaBench);
HWY_EXPORT_AND_TEST_P(IguanaBench, TestIguanaThroughput);
HWY_AFTER_TEST();
}  // namespace hwy
#endif  // HWY_ONCE
