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

#include <vector>

#if defined(HIGHWAY_HWY_CONTRIB_DOT_ONE_TO_MANY_KERNELS_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_DOT_ONE_TO_MANY_KERNELS_INL_H_
#undef HIGHWAY_HWY_CONTRIB_DOT_ONE_TO_MANY_KERNELS_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_DOT_ONE_TO_MANY_KERNELS_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>

#include "hwy/base.h"
#include "hwy/contrib/dot/dot-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// =============================================================================
// Reusable Kernels which use Highway's overloaded Dot::Compute.
// =============================================================================

struct OTM_DefaultScorerKernel {
  class FloatToBFloat16 {
   public:
    using AccumT = float;

    static constexpr size_t kBlockDimensions = 256;

    explicit FloatToBFloat16(const float* query_data, size_t query_dims)
        : query_data_(query_data) {
      (void)query_data_;  // Silence unused private field warnings
#if HWY_NATIVE_DOT_BF16 || HWY_IDE
      // If the CPU supports native bfloat16 dot-products (e.g., AVX512_BF16 or
      // ARM Neon BFDOT), it is drastically faster to demote the query to
      // bfloat16 once and execute pure bf16 * bf16 loops instead of promoting
      // every single database element to f32 on the fly!
      query_bf16_data_.resize(query_dims);
      for (size_t d = 0; d < query_dims; ++d) {
        query_bf16_data_[d] = BF16FromF32(query_data[d]);
      }
#endif
    }

    HWY_INLINE void PrepareDimensionBlock(size_t /*dim_offset*/,
                                          size_t /*dim_end*/) {}

    template <typename Policy>
    HWY_INLINE void ScoreBlock(size_t dp_idx, const Policy& policy,
                               size_t dim_offset, size_t /*inner_end*/,
                               size_t dim_end, float& accum) {
      const bfloat16_t* bfloat_ptr =
          static_cast<const bfloat16_t*>(policy.GetBasePtr(dp_idx)) +
          dim_offset;
      size_t num_elements = dim_end - dim_offset;

#if HWY_NATIVE_DOT_BF16 || HWY_IDE
      const hwy::HWY_NAMESPACE::ScalableTag<bfloat16_t> dbf;
      const bfloat16_t* bfloat_query_ptr = query_bf16_data_.data() + dim_offset;
      // Dispatches natively to CPU BFloat16 FMA instructions (AVX512_BF16 /
      // AMX)
      accum += Dot::Compute<0>(dbf, bfloat_query_ptr, bfloat_ptr, num_elements);
#else
      const hwy::HWY_NAMESPACE::ScalableTag<float> df;
      const float* float_ptr = query_data_ + dim_offset;
      // Falls back to highway-abstracted upcast-and-multiply: f32 * bf16 -> f32
      accum += Dot::Compute<0>(df, float_ptr, bfloat_ptr, num_elements);
#endif
    }

   private:
    HWY_MAYBE_UNUSED const float* query_data_;
#if HWY_NATIVE_DOT_BF16 || HWY_IDE
    // std::vector should be included where FloatToBFloat16ScorerKernel is used.
    std::vector<bfloat16_t> query_bf16_data_;
#endif
  };

  // SameType.
  // ---------------------------------------------------------------------------
  // A universal ScorerKernel for computing dot products when the query and
  // database elements are natively stored in the same data type. By leveraging
  // Highway's overloaded Dot::Compute architecture, this struct handles
  // execution and proper accumulator typing automatically (e.g.
  // float*float->float, int16*int16->int32).
  template <typename T, typename AccumType>
  class SameType {
   public:
    using AccumT = AccumType;  // i.e. float for f32/bf16, int32_t for int16

    static constexpr size_t kBlockDimensions = 256;

    explicit SameType(const T* query_data, size_t query_dims)
        : query_data_(query_data) {
      (void)query_dims;  // Silence unused variable warning uniformly
    }

    HWY_INLINE void PrepareDimensionBlock(size_t /*dim_offset*/,
                                          size_t /*dim_end*/) {}

    template <typename Policy>
    HWY_INLINE void ScoreBlock(size_t dp_idx, const Policy& policy,
                               size_t dim_offset, size_t /*inner_end*/,
                               size_t dim_end, AccumT& accum) {
      const T* t_dp_ptr =
          static_cast<const T*>(policy.GetBasePtr(dp_idx)) + dim_offset;
      const T* t_query_ptr = query_data_ + dim_offset;
      size_t num_elements = dim_end - dim_offset;

      const hwy::HWY_NAMESPACE::ScalableTag<T> d;
      accum += Dot::Compute<0>(d, t_query_ptr, t_dp_ptr, num_elements);
    }

   private:
    const T* query_data_;
  };
};  // struct OTM_ReusableKernels

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
#endif  // HIGHWAY_HWY_CONTRIB_DOT_ONE_TO_MANY_KERNELS_INL_H_
