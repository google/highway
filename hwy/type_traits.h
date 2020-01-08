// Copyright 2019 Google LLC
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

#ifndef HWY_TYPE_TRAITS_H_
#define HWY_TYPE_TRAITS_H_

// Subset of type_traits / numeric_limits for faster compilation.

#include <stddef.h>
#include <stdint.h>

namespace hwy {

template <bool Condition, class T>
struct EnableIfT {};
template <class T>
struct EnableIfT<true, T> {
  using type = T;
};

template <bool Condition, class T = void>
using EnableIf = typename EnableIfT<Condition, T>::type;

template <typename T>
constexpr bool IsFloat() {
  return T(1.25) != T(1);
}

template <typename T>
constexpr bool IsSigned() {
  return T(0) > T(-1);
}

// Insert into template/function arguments to enable this overload only for
// <= 128-bit vectors (excluding NONE).
#define HWY_IF128(T, N) EnableIf<N != 0 && (N * sizeof(T) <= 16)>* = nullptr

#define HWY_IF_FLOAT(T) EnableIf<IsFloat<T>()>* = nullptr

// Largest/smallest representable integer values.
template <typename T>
constexpr T LimitsMax() {
  return IsSigned<T>() ? T((1ULL << (sizeof(T) * 8 - 1)) - 1)
                       : static_cast<T>(~0ull);
}
template <typename T>
constexpr T LimitsMin() {
  return IsSigned<T>() ? T(-1) - LimitsMax<T>() : T(0);
}

// Empty struct used as a size tag type.
template <size_t N>
struct SizeTag {};

// The unsigned integer type whose size is kSize bytes.
template <size_t kSize>
struct MakeUnsignedT;
template <>
struct MakeUnsignedT<1> {
  using type = uint8_t;
};
template <>
struct MakeUnsignedT<2> {
  using type = uint16_t;
};
template <>
struct MakeUnsignedT<4> {
  using type = uint32_t;
};
template <>
struct MakeUnsignedT<8> {
  using type = uint64_t;
};

template <typename T>
using MakeUnsigned = typename MakeUnsignedT<sizeof(T)>::type;

}  // namespace hwy

#endif  // HWY_TYPE_TRAITS_H_
