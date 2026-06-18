#include <cstdlib>
#include <string>
#include <vector>

#include "hwy/contrib/pipeline/prefetch_pipeline_2d.h"
#include "hwy/contrib/pipeline/prefetch_pipeline_types.h"
#include "hwy/contrib/pipeline/prefetch_tuner.h"
#include "hwy/contrib/pipeline/prefetch_tuner_registry.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/dot/one_to_many_test.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/dot/one_to_many_kernels-inl.h"
#include "hwy/contrib/dot/one_to_many-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

// =========================================================================
// Example 1: Float32 Query against BFloat16 Dataset
// =========================================================================

// An accessor handles mapping logic from loop iteration `i` to database `idx`,
// and stores the final calibrated scores into output vectors.
class SimpleAccessor {
 public:
  explicit SimpleAccessor(size_t size) : size_(size), scores_(size, 0.0f) {}

  size_t size() const { return size_; }
  size_t Index(size_t i) const { return i; }  // Direct 1:1 mapped indices

  void SetScore(size_t i, float score) { scores_[i] = score; }
  float GetScore(size_t i) const { return scores_[i]; }

 private:
  size_t size_;
  std::vector<float> scores_;
};

// LayoutPolicy dictating how to iterate over our flat BFloat16 array and
// prefetch.
class BFloat16LayoutPolicy {
 public:
  HWY_MAYBE_UNUSED static constexpr size_t kMaxCachelinesPerIter = 1;
  HWY_MAYBE_UNUSED static constexpr size_t kBlockDatapoints = 128;
  HWY_MAYBE_UNUSED static constexpr size_t kPrefetchLookaheadL1 = 4;
  HWY_MAYBE_UNUSED static constexpr size_t kPrefetchLookaheadL3 = 32;

  BFloat16LayoutPolicy(const bfloat16_t* dataset_ptr, size_t dimensions)
      : dataset_ptr_(dataset_ptr), dimensions_(dimensions) {}

  HWY_INLINE void PopulateCachelines(size_t /*dp_idx*/, size_t /*dim_start*/,
                                     size_t /*dim_end*/,
                                     auto& /*collector*/) const {}

  // Gets the exact starting address in memory for a target datapoint
  HWY_INLINE const void* GetBasePtr(size_t dp_idx) const {
    return dataset_ptr_ + dp_idx * dimensions_;
  }

  // Translates intermediate pipeline blocks into final stored outputs.
  // In a real framework, this might apply scaling factors and subtract biases.
  HWY_INLINE void CalibrateScores(const float* raw_accumulators, size_t dp_idx,
                                  size_t dp_end,
                                  SimpleAccessor& accessor) const {
    for (size_t i = dp_idx; i < dp_end; ++i) {
      accessor.SetScore(i, raw_accumulators[i - dp_idx]);
    }
  }

 private:
  const bfloat16_t* dataset_ptr_;
  size_t dimensions_;
};

template <typename T>
class GenericLayoutPolicy {
 public:
  HWY_MAYBE_UNUSED static constexpr size_t kMaxCachelinesPerIter = 1;
  HWY_MAYBE_UNUSED static constexpr size_t kBlockDatapoints = 128;
  HWY_MAYBE_UNUSED static constexpr size_t kPrefetchLookaheadL1 = 4;
  HWY_MAYBE_UNUSED static constexpr size_t kPrefetchLookaheadL3 = 32;

  GenericLayoutPolicy(const T* dataset_ptr, size_t dimensions)
      : dataset_ptr_(dataset_ptr), dimensions_(dimensions) {}

  HWY_INLINE void PopulateCachelines(size_t /*dp_idx*/, size_t /*dim_start*/,
                                     size_t /*dim_end*/,
                                     auto& /*collector*/) const {}

  HWY_INLINE const void* GetBasePtr(size_t dp_idx) const {
    return dataset_ptr_ + dp_idx * dimensions_;
  }

  template <typename AccumT>
  HWY_INLINE void CalibrateScores(const AccumT* raw_accumulators, size_t dp_idx,
                                  size_t dp_end,
                                  SimpleAccessor& accessor) const {
    for (size_t i = dp_idx; i < dp_end; ++i) {
      accessor.SetScore(i, static_cast<float>(raw_accumulators[i - dp_idx]));
    }
  }

 private:
  const T* dataset_ptr_;
  size_t dimensions_;
};

// =========================================================================
// Example 3: LUT16 with PrepareDimensionBlock (Fake Codebook)
// =========================================================================

// LayoutPolicy for 4-bit packed dataset stored as uint8_t (2 dimensions per
// byte)
class LUT16LayoutPolicy {
 public:
  HWY_MAYBE_UNUSED static constexpr size_t kMaxCachelinesPerIter = 1;
  // Since LUT16 stores 2 dimensions per byte (0.5 bytes per dimension),
  // we can heavily scale the datapoint blocking and lookahead arrays
  // while comfortably fitting in L1D cache constraints!
  HWY_MAYBE_UNUSED static constexpr size_t kBlockDatapoints = 512;
  HWY_MAYBE_UNUSED static constexpr size_t kPrefetchLookaheadL1 = 8;
  HWY_MAYBE_UNUSED static constexpr size_t kPrefetchLookaheadL3 = 64;

  LUT16LayoutPolicy(const uint8_t* dataset_ptr, size_t dimensions)
      : dataset_ptr_(dataset_ptr), dimensions_(dimensions) {}

  HWY_INLINE void PopulateCachelines(size_t /*dp_idx*/, size_t /*dim_start*/,
                                     size_t /*dim_end*/,
                                     auto& /*collector*/) const {}

  HWY_INLINE const void* GetBasePtr(size_t dp_idx) const {
    return dataset_ptr_ + dp_idx * (dimensions_ / 2);
  }

  HWY_INLINE void CalibrateScores(const float* raw_accumulators, size_t dp_idx,
                                  size_t dp_end,
                                  SimpleAccessor& accessor) const {
    for (size_t i = dp_idx; i < dp_end; ++i) {
      accessor.SetScore(i, raw_accumulators[i - dp_idx]);
    }
  }

 private:
  const uint8_t* dataset_ptr_;
  size_t dimensions_;
};

class FakeLut16ScorerKernel {
 public:
  using AccumT = float;
  HWY_MAYBE_UNUSED static constexpr size_t kBlockDimensions = 512;

  explicit FakeLut16ScorerKernel(const std::vector<float>& fake_codebooks)
      : fake_codebooks_(fake_codebooks) {}

  HWY_INLINE void PrepareDimensionBlock(size_t dim_offset, size_t dim_end) {
    // Simulates loading a codebook (e.g. centroids) for the current dimensions
    // into hot state/registers.
    active_codebook_.clear();
    for (size_t d = dim_offset; d < dim_end; ++d) {
      // For this fake test, each dimension has 16 float values
      for (size_t c = 0; c < 16; ++c) {
        active_codebook_.push_back(fake_codebooks_[d * 16 + c]);
      }
    }
  }

  template <typename Policy>
  HWY_INLINE void ScoreBlock(size_t dp_idx, const Policy& policy,
                             size_t dim_offset, size_t /*inner_end*/,
                             size_t dim_end, float& accum) {
    const void* dp_ptr = policy.GetBasePtr(dp_idx);
    // Read the 4-bit packed data
    const uint8_t* u8_ptr =
        static_cast<const uint8_t*>(dp_ptr) + (dim_offset / 2);
    size_t num_elements = dim_end - dim_offset;
    size_t num_bytes = num_elements / 2;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    size_t i = 0;

    // Unroll by 4 bytes (8 dimensions) per iteration to maximize FPU pipeline
    // throughput and reduce loop branch conditions drastically.
    //
    // We also use 4 separate local accumulators so the CPU can calculate the
    // addition instructions in parallel without waiting on a single `sum`
    // execution dependency chain!
    for (; i + 3 < num_bytes; i += 4) {
      uint8_t p0 = u8_ptr[i + 0];
      uint8_t p1 = u8_ptr[i + 1];
      uint8_t p2 = u8_ptr[i + 2];
      uint8_t p3 = u8_ptr[i + 3];

      // Evaluate distances dynamically from the active block cache
      sum0 += active_codebook_[((i + 0) * 2) * 16 + (p0 & 0x0F)];
      sum0 += active_codebook_[((i + 0) * 2 + 1) * 16 + (p0 >> 4)];

      sum1 += active_codebook_[((i + 1) * 2) * 16 + (p1 & 0x0F)];
      sum1 += active_codebook_[((i + 1) * 2 + 1) * 16 + (p1 >> 4)];

      sum2 += active_codebook_[((i + 2) * 2) * 16 + (p2 & 0x0F)];
      sum2 += active_codebook_[((i + 2) * 2 + 1) * 16 + (p2 >> 4)];

      sum3 += active_codebook_[((i + 3) * 2) * 16 + (p3 & 0x0F)];
      sum3 += active_codebook_[((i + 3) * 2 + 1) * 16 + (p3 >> 4)];
    }

    // Combine accumulators using an addition tree
    sum0 += sum1;
    sum2 += sum3;
    sum0 += sum2;

    // Scalar cleanup loop
    for (; i < num_bytes; ++i) {
      uint8_t packed = u8_ptr[i];
      sum0 += active_codebook_[(i * 2) * 16 + (packed & 0x0F)];
      sum0 += active_codebook_[(i * 2 + 1) * 16 + (packed >> 4)];
    }
    accum += sum0;
  }

 private:
  const std::vector<float>& fake_codebooks_;
  // Reusable hot state initialized per block by PrepareDimensionBlock
  std::vector<float> active_codebook_;
};

void TestOneToManyFloatFloat() {
  const size_t num_dps = 1000;
  const size_t dims = 512;

  std::vector<float> dataset(num_dps * dims);
  for (size_t i = 0; i < dataset.size(); ++i) {
    dataset[i] = static_cast<float>(i % 5) * 0.1f;
  }

  std::vector<float> query(dims);
  for (size_t i = 0; i < dims; ++i) {
    query[i] = static_cast<float>(i % 3) * 0.2f;
  }

  SimpleAccessor accessor(num_dps);
  GenericLayoutPolicy<float> policy(dataset.data(), dims);
  OTM_DefaultScorerKernel::SameType<float, float> kernel(query.data(), dims);
  dot::low_level::OneToMany2DTiledPipeline::Run(
      policy, accessor, dims, dims, kernel, hwy::pipeline::Tiling2DArgs(),
      hwy::pipeline::PrefetchArgs::DefaultSequential());

  for (size_t i = 0; i < num_dps; ++i) {
    float expected = 0.0f;
    for (size_t d = 0; d < dims; ++d) {
      expected += dataset[i * dims + d] * query[d];
    }

    float actual = accessor.GetScore(i);
    HWY_ASSERT(std::abs(expected - actual) < 1e-4);
  }
}

void TestOneToManyFloatBfloat16() {
  const size_t num_dps = 1000;
  const size_t dims = 512;

  std::vector<bfloat16_t> dataset(num_dps * dims);
  for (size_t i = 0; i < dataset.size(); ++i) {
    dataset[i] = BF16FromF32(static_cast<float>(i % 5) * 0.1f);
  }

  std::vector<float> query(dims);
  for (size_t i = 0; i < dims; ++i) {
    query[i] = static_cast<float>(i % 3) * 0.2f;
  }

  SimpleAccessor accessor(num_dps);
  BFloat16LayoutPolicy policy(dataset.data(), dims);
  OTM_DefaultScorerKernel::FloatToBFloat16 kernel(query.data(), dims);

  dot::low_level::OneToMany2DTiledPipeline::Run(
      policy, accessor, dims, dims, kernel, hwy::pipeline::Tiling2DArgs(),
      hwy::pipeline::PrefetchArgs::DefaultSequential());

  // Validate mathematical computations. Note that BFloat16 only has 8 bits of
  // precision total (7 bits mantissa). Accumulating 512 elements can drift
  // significantly around truncation limits across CPU architectures
  // (e.g. native AVX512_BF16 vs emulated floats), so we must scale tolerance
  // appropriately against the machine epsilon!
  for (size_t i = 0; i < num_dps; ++i) {
    double expected = 0.0;
    for (size_t d = 0; d < dims; ++d) {
      expected +=
          static_cast<double>(F32FromBF16(dataset[i * dims + d])) * query[d];
    }

    float actual = accessor.GetScore(i);
    HWY_ASSERT(std::abs(expected - actual) < 1.0);
  }
}

void TestOneToManyLUT16() {
  const size_t num_dps = 1000;
  const size_t dims = 512;

  // 4-bit LUT16 packs 2 elements per byte.
  std::vector<uint8_t> dataset(num_dps * (dims / 2));
  for (size_t i = 0; i < dataset.size(); ++i) {
    dataset[i] = static_cast<uint8_t>(i % 256);
  }

  // Codebooks contain 16 float centroids per dimension slice.
  std::vector<float> fake_codebooks(dims * 16);
  for (size_t i = 0; i < fake_codebooks.size(); ++i) {
    fake_codebooks[i] = static_cast<float>(i % 7) * 0.1f;
  }

  SimpleAccessor accessor(num_dps);
  LUT16LayoutPolicy policy(dataset.data(), dims);
  FakeLut16ScorerKernel kernel(fake_codebooks);
  dot::low_level::OneToMany2DTiledPipeline::Run(
      policy, accessor, dims, dims, kernel, hwy::pipeline::Tiling2DArgs(),
      hwy::pipeline::PrefetchArgs::DefaultSequential());

  for (size_t i = 0; i < num_dps; ++i) {
    float expected = 0.0f;
    for (size_t d = 0; d < dims / 2; ++d) {
      uint8_t packed = dataset[i * (dims / 2) + d];
      uint8_t c1 = packed & 0x0F;
      uint8_t c2 = packed >> 4;
      expected += fake_codebooks[(d * 2) * 16 + c1];
      expected += fake_codebooks[(d * 2 + 1) * 16 + c2];
    }

    float actual = accessor.GetScore(i);
    HWY_ASSERT(std::abs(expected - actual) < 1e-4);
  }
}

// =========================================================================
// Example 4: Public Wrapper Test Coverage
// =========================================================================

void TestOneToManyPublicWrapperCorrectness() {
  const size_t num_dps = 1000;
  const size_t dims = 512;

  std::vector<float> dataset(num_dps * dims);
  for (size_t i = 0; i < dataset.size(); ++i) {
    dataset[i] = static_cast<float>(i % 5) * 0.1f;
  }

  std::vector<float> query(dims);
  for (size_t i = 0; i < dims; ++i) {
    query[i] = static_cast<float>(i % 3) * 0.2f;
  }

  SimpleAccessor accessor(num_dps);
  GenericLayoutPolicy<float> policy(dataset.data(), dims);
  OTM_DefaultScorerKernel::SameType<float, float> kernel(query.data(), dims);

  // Call the PUBLIC wrapper instead of low_level
  dot::OneToMany2DTiledPipeline::Run<
      hwy::pipeline::PrefetchTuningHint::kSequential>(policy, accessor, dims,
                                                      dims, kernel);

  for (size_t i = 0; i < num_dps; ++i) {
    float expected = 0.0f;
    for (size_t d = 0; d < dims; ++d) {
      expected += dataset[i * dims + d] * query[d];
    }
    float actual = accessor.GetScore(i);
    HWY_ASSERT(std::abs(expected - actual) < 1e-4);
  }
}

struct TestOtmMetricContext {
  bool called = false;
  std::string captured_file;
  int captured_line = 0;
  size_t captured_total = 0;
  bool captured_tiny = false;
  hwy::pipeline::PrefetchTuningHint captured_hint =
      hwy::pipeline::PrefetchTuningHint::kAuto;
};

void FakeOtmMetricCb(void* user_data, float) {
  static_cast<TestOtmMetricContext*>(user_data)->called = true;
}

class MockOtmTuner : public hwy::pipeline::low_level::PrefetchTuner {
 public:
  explicit MockOtmTuner(TestOtmMetricContext* ctx) : ctx_(ctx) {}
  hwy::pipeline::low_level::PrefetchTuningScope CreateScope(
      const hwy::pipeline::low_level::PrefetchTuningContext& context)
      const override {
    if (context.file_loc != nullptr) {
      ctx_->captured_file = context.file_loc;
    }
    ctx_->captured_line = context.line_loc;
    ctx_->captured_total = context.total_elements;
    ctx_->captured_tiny = context.is_ultra_tiny;
    ctx_->captured_hint = context.hint;

    hwy::pipeline::PrefetchArgs args{.deep_lookahead = 12,
                                     .shallow_lookahead = 2};
    return hwy::pipeline::low_level::PrefetchTuningScope(args, FakeOtmMetricCb,
                                                         ctx_, 1.0f);
  }

  hwy::pipeline::CallsiteId RegisterContext(
      const hwy::pipeline::low_level::PrefetchTuningContext& context)
      const override {
    return 0;
  }

  hwy::pipeline::low_level::PrefetchTuningScope CreateScopeByCallsiteId(
      hwy::pipeline::CallsiteId callsite_id,
      const hwy::pipeline::low_level::PrefetchTuningContext& context)
      const override {
    return CreateScope(context);
  }

 private:
  TestOtmMetricContext* ctx_;
};

void TestOneToManyPublicWrapperTunerIntegration() {
  TestOtmMetricContext ctx;
  MockOtmTuner tuner(&ctx);
  hwy::pipeline::low_level::GetGlobalPrefetchTunerRegistry() = &tuner;

  const size_t num_dps = 128000;
  const size_t dims = 128;  // total_elements = 128,000 (Huge bucket!)

  std::vector<float> dataset(num_dps * dims, 0.1f);
  std::vector<float> query(dims, 0.2f);
  SimpleAccessor accessor(num_dps);
  GenericLayoutPolicy<float> policy(dataset.data(), dims);
  OTM_DefaultScorerKernel::SameType<float, float> kernel(query.data(), dims);

  const int expected_base_line = __LINE__ + 1;
  dot::OneToMany2DTiledPipeline::Run<
      hwy::pipeline::PrefetchTuningHint::kRandom>(policy, accessor, dims, dims,
                                                  kernel);

  HWY_ASSERT(ctx.called);
  HWY_ASSERT(ctx.captured_total == 128000);
  HWY_ASSERT(!ctx.captured_tiny);
  HWY_ASSERT(ctx.captured_hint == hwy::pipeline::PrefetchTuningHint::kRandom);
  HWY_ASSERT(ctx.captured_file.find("one_to_many_test.cc") !=
             std::string::npos);  // NOLINT
  // Random Huge (128,000 <= 65536 is false -> Huge Random -> +600000)
  HWY_ASSERT(ctx.captured_line == expected_base_line + 600000);

  hwy::pipeline::low_level::GetGlobalPrefetchTunerRegistry() = nullptr;
}

void TestOneToManyPublicWrapperUltraTinyShortCircuit() {
  TestOtmMetricContext ctx;
  MockOtmTuner tuner(&ctx);
  hwy::pipeline::low_level::GetGlobalPrefetchTunerRegistry() = &tuner;

  const size_t num_dps = 2;
  const size_t dims = 8;  // total_elements = 16 (< 32 UltraTiny!)

  std::vector<float> dataset(num_dps * dims, 0.1f);
  std::vector<float> query(dims, 0.2f);
  SimpleAccessor accessor(num_dps);
  GenericLayoutPolicy<float> policy(dataset.data(), dims);
  OTM_DefaultScorerKernel::SameType<float, float> kernel(query.data(), dims);

  dot::OneToMany2DTiledPipeline::Run<
      hwy::pipeline::PrefetchTuningHint::kSequential>(policy, accessor, dims,
                                                      dims, kernel);

  // Tuner plugin should be bypassed entirely!
  HWY_ASSERT(!ctx.called);

  hwy::pipeline::low_level::GetGlobalPrefetchTunerRegistry() = nullptr;
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(OneToManyTest);
HWY_EXPORT_AND_TEST_P(OneToManyTest, TestOneToManyLUT16);
HWY_EXPORT_AND_TEST_P(OneToManyTest, TestOneToManyFloatFloat);
HWY_EXPORT_AND_TEST_P(OneToManyTest, TestOneToManyFloatBfloat16);
HWY_EXPORT_AND_TEST_P(OneToManyTest, TestOneToManyPublicWrapperCorrectness);
HWY_EXPORT_AND_TEST_P(OneToManyTest,
                      TestOneToManyPublicWrapperTunerIntegration);
HWY_EXPORT_AND_TEST_P(OneToManyTest,
                      TestOneToManyPublicWrapperUltraTinyShortCircuit);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
