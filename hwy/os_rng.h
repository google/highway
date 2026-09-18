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

#ifndef HIGHWAY_HWY_OS_RNG_H_
#define HIGHWAY_HWY_OS_RNG_H_

#include "hwy/base.h"

namespace hwy {

// Returns false or performs the equivalent of `memcpy(bytes, r, 16)`, where r
// is high-quality (unpredictable, uniformly distributed) random bits.
// This used to reside in contrib/ (vqsort.h), hence HWY_CONTRIB_DLLEXPORT.
HWY_CONTRIB_DLLEXPORT bool Fill16BytesSecure(void* bytes);

}  // namespace hwy

#endif  // HIGHWAY_HWY_OS_RNG_H_
