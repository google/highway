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

// Internal bitstream-format constants shared by ans.cc (scalar codec) and
// ans-inl.h (SIMD decoder). Not part of the public interface: callers of
// ans.h's Ans32Encode/Ans32DecodeScalar/AnsStatistics never need these
// directly. kAnsWordMBits/kAnsWordM/kAnsFreqMask stay in ans.h instead,
// since AnsStatistics::Freq/CumFreq (defined inline there) need them.

#ifndef HIGHWAY_HWY_CONTRIB_IGUANA_ANS_DETAIL_H_
#define HIGHWAY_HWY_CONTRIB_IGUANA_ANS_DETAIL_H_

#include <stddef.h>
#include <stdint.h>

namespace hwy {
namespace iguana {

constexpr uint32_t kAnsWordLBits = 16;
constexpr uint32_t kAnsWordL = uint32_t{1} << kAnsWordLBits;  // 65536

// 32 interleaved rANS streams: 16 written/read forwards, 16 backwards.
constexpr int kAnsLanes = 32;

// Longest possible serialized frequency table.
constexpr size_t kAnsCtrlBlockSize = 96;
constexpr size_t kAnsDenseTableMaxLength = kAnsCtrlBlockSize + 384;

}  // namespace iguana
}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_IGUANA_ANS_DETAIL_H_
