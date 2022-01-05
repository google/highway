// Copyright 2022 Google LLC
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

#ifndef HIGHWAY_HWY_TESTS_INCLUDE_FARM_SVE_H_
#define HIGHWAY_HWY_TESTS_INCLUDE_FARM_SVE_H_

#if defined(HWY_EMULATE_SVE)
// This must be included before anything that includes highway.h because it
// emulates SVE types/functions which will be used by arm_sve-inl.h included
// via highway.h. This avoids the highway library depending on farm_sve itself.
#include "third_party/farm_sve/farm_sve.h"
#endif

#endif  // HIGHWAY_HWY_TESTS_INCLUDE_FARM_SVE_H_
