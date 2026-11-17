// Copyright 2021 Google LLC
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: BSD-3-Clause
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

#include "hwy/contrib/sort/vqsort.h"

#include "hwy/base.h"
#include "hwy/contrib/sort/vqsort-inl.h"
namespace hwy {

// Unused, only for ABI compatibility
void Sorter::Fill24Bytes(const void*, size_t, void*) {}
bool Sorter::HaveFloat64() { return VQSortHaveFloat64(); }
Sorter::Sorter() {}
void Sorter::Delete() {}
uint64_t* GetGeneratorState() { return hwy::detail::GetGeneratorStateStatic(); }

}  // namespace hwy
