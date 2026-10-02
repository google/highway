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

// SIMD decode path for Iguana: the container loop routes each entropy-coded
// stream through the vectorized ANS32 decoder (ans-inl.h); the container
// parsing and the LZ77 stage (both inherently serial) are the target-
// independent helpers in iguana.h. Output is identical to
// hwy::iguana::DecompressScalar.

#if defined(HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_INL_H_
#undef HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>

#include "hwy/aligned_allocator.h"
#include "hwy/contrib/iguana/ans-inl.h"
#include "hwy/contrib/iguana/iguana_detail.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// Decompresses a block produced by hi::Compress into pre-allocated `dst`.
// Returns the number of bytes written, or hwy::iguana::kDecompressFailed on
// malformed input. Chunks are decoded on `pool`; see DecompressBlockParallel.
HWY_INLINE size_t DecompressStatic(Span<const uint8_t> src, Span<uint8_t> dst,
                                   hwy::iguana::IguanaWorkspace& ws,
                                   ThreadPool& pool) {
  // HWY_ATTR is required here: the lambda is a separate function that calls
  // SIMD code, so it must be compiled for this target too.
  const auto decode = [](Span<const uint8_t> payload,
                         Span<uint8_t> out) HWY_ATTR {
    return hwy::iguana_ans::HWY_NAMESPACE::Ans32Decode(payload, out);
  };
  return hwy::iguana::DecompressBlockParallel(src, dst, decode, ws, pool);
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // include guard
