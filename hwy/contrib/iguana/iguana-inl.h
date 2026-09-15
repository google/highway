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

#include <vector>

#include "hwy/contrib/iguana/ans-inl.h"
#include "hwy/contrib/iguana/ans.h"
#include "hwy/contrib/iguana/detail.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace iguana_full {

namespace hi = hwy::iguana;
// ans-inl.h nests the SIMD decoder as hwy::iguana_ans::HWY_NAMESPACE.
namespace ha = hwy::iguana_ans::HWY_NAMESPACE;

// Decompresses a block produced by hi::Compress, appending to `out`. The
// container loop is shared with the scalar path (detail.h); only the entropy
// stage differs: here it is the vectorized ANS32 decoder for this target.
// Returns false on malformed input; output matches hi::DecompressScalar.
HWY_INLINE bool Decompress(const uint8_t* HWY_RESTRICT src, size_t src_size,
                           std::vector<uint8_t>& out) {
  const auto decode = [](const uint8_t* payload, size_t payload_size,
                         uint8_t* dst, size_t dst_size) {
    return ha::Ans32Decode(payload, payload_size, dst, dst_size);
  };
  return hi::DecompressBlock(src, src_size, out, decode);
}

}  // namespace iguana_full
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // include guard
