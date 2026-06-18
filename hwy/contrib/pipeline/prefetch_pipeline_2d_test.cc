// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include "hwy/contrib/pipeline/prefetch_pipeline_2d.h"

#include <cstddef>
#include <string>
#include <vector>

#include "hwy/contrib/pipeline/prefetch_pipeline_types.h"
#include "hwy/contrib/pipeline/prefetch_tuner.h"
#include "hwy/contrib/pipeline/prefetch_tuner_registry.h"
#include "hwy/tests/hwy_gtest.h"

namespace hwy {
namespace pipeline {
namespace {

template <PrefetchStrategy TargetStrategy>
struct TestPolicy : Default2DTiledPrefetchPolicy {
  static constexpr PrefetchStrategy kStrategy = TargetStrategy;
  static constexpr size_t kMaxCachelinesPerIter = 1;
};

std::vector<std::string> g_trace;

struct Fake2DCallbacks {
  void OnOuterBlockStart(size_t outer_idx, size_t outer_end) {
    g_trace.push_back("OuterStart(" + std::to_string(outer_idx) + "," +
                      std::to_string(outer_end) + ")");
  }

  void PrepareInnerBlock(size_t outer_idx, size_t outer_end, size_t inner_idx,
                         size_t inner_end) {
    g_trace.push_back("PrepareInner(" + std::to_string(inner_idx) + "," +
                      std::to_string(inner_end) + ")");
  }

  template <typename Limits>
  void PopulateCachelines(size_t outer_i, size_t inner_idx, size_t inner_end,
                          CachelineBundle<Limits>& collector) {
    g_trace.push_back("Populate(" + std::to_string(outer_i) + "," +
                      std::to_string(inner_idx) + ")");
    collector.Add(reinterpret_cast<const void*>(outer_i * 1000 + inner_idx));
  }

  void ComputeTask(size_t outer_i, size_t inner_idx, size_t inner_end) {
    g_trace.push_back("Compute(" + std::to_string(outer_i) + "," +
                      std::to_string(inner_idx) + ")");
  }

  void OnOuterBlockFinish(size_t outer_idx, size_t outer_end) {
    g_trace.push_back("OuterFinish(" + std::to_string(outer_idx) + "," +
                      std::to_string(outer_end) + ")");
  }
};

class PrefetchPipeline2DTest : public ::testing::Test {
 protected:
  void SetUp() override { g_trace.clear(); }
};

template <PrefetchStrategy TargetStrategy, size_t Deep = 4, size_t Shallow = 2>
void Call2DPipeline(size_t outer_size, size_t inner_size, Tiling2DArgs tiling) {
  PrefetchArgs args;
  args.deep_lookahead = Deep;
  args.shallow_lookahead = Shallow;
  Fake2DCallbacks cb;
  low_level::PrefetchPipeline2DTiledLoop<TestPolicy<TargetStrategy>,
                                         Fake2DCallbacks>(
      outer_size, inner_size, cb, tiling, args);
}

TEST_F(PrefetchPipeline2DTest, NoPrefetchStrategy) {
  // Outer size = 4, Inner size = 4. Tiling: outer=2, inner=2.
  Call2DPipeline<PrefetchStrategy::kNoPrefetch>(
      4, 4, Tiling2DArgs{.outer_block = 2, .inner_block = 2});

  std::vector<std::string> expected = {
      // Outer Block 1 (0 to 2)
      "OuterStart(0,2)", "PrepareInner(0,2)", "Compute(0,0)", "Compute(1,0)",
      "PrepareInner(2,4)", "Compute(0,2)", "Compute(1,2)", "OuterFinish(0,2)",
      // Outer Block 2 (2 to 4)
      "OuterStart(2,4)", "PrepareInner(0,2)", "Compute(2,0)", "Compute(3,0)",
      "PrepareInner(2,4)", "Compute(2,2)", "Compute(3,2)", "OuterFinish(2,4)"};
  EXPECT_EQ(g_trace, expected);
}

TEST_F(PrefetchPipeline2DTest, EndToEndPublicWrapperTuning) {
  struct TestMetricContext {
    bool called = false;
    std::string captured_file;
    int captured_line = 0;
    size_t captured_total = 0;
    bool captured_tiny = false;
    PrefetchTuningHint captured_hint = PrefetchTuningHint::kAuto;
  } ctx;

  class Mock2DTuner : public low_level::PrefetchTuner {
   public:
    explicit Mock2DTuner(TestMetricContext* ctx) : ctx_(ctx) {}
    low_level::PrefetchTuningScope CreateScope(
        const low_level::PrefetchTuningContext& context) const override {
      if (context.file_loc != nullptr) {
        ctx_->captured_file = context.file_loc;
      }
      ctx_->captured_line = context.line_loc;
      ctx_->captured_total = context.total_elements;
      ctx_->captured_tiny = context.is_ultra_tiny;
      ctx_->captured_hint = context.hint;

      PrefetchArgs args{.deep_lookahead = 0, .shallow_lookahead = 0};
      return low_level::PrefetchTuningScope(
          args,
          [](void* user_data, float) {
            static_cast<TestMetricContext*>(user_data)->called = true;
          },
          ctx_, 1.0f);
    }

    CallsiteId RegisterContext(
        const low_level::PrefetchTuningContext& context) const override {
      return 0;
    }

    low_level::PrefetchTuningScope CreateScopeByCallsiteId(
        CallsiteId callsite_id,
        const low_level::PrefetchTuningContext& context) const override {
      return CreateScope(context);
    }

   private:
    TestMetricContext* ctx_;
  };

  Mock2DTuner tuner(&ctx);
  low_level::GetGlobalPrefetchTunerRegistry() = &tuner;

  g_trace.clear();
  Fake2DCallbacks cb;

  // Execute on 8x8 matrix. total_elements = 32 (>= 32). kAuto hint. kNoPrefetch
  // strategy. Capture the exact line number where the wrapper is invoked!
  const int expected_base_line = __LINE__ + 1;
  PrefetchPipeline2DTiledLoop<PrefetchTuningHint::kAuto,
                              TestPolicy<PrefetchStrategy::kNoPrefetch>,
                              Fake2DCallbacks>(
      8, 8, cb, Tiling2DArgs{.outer_block = 2, .inner_block = 2});

  // 1. Verify RAII Telemetry Firing
  EXPECT_TRUE(ctx.called);

  // 2. Verify Context Propagation
  EXPECT_EQ(ctx.captured_total, 32);
  EXPECT_FALSE(ctx.captured_tiny);
  EXPECT_EQ(ctx.captured_hint, PrefetchTuningHint::kAuto);
  EXPECT_NE(ctx.captured_file.find("prefetch_pipeline_2d_test.cc"),
            std::string::npos);

  // 3. Verify Fission Line Enrichment (64 items <= 256 -> Tiny kAuto ->
  // +700000)
  EXPECT_EQ(ctx.captured_line, expected_base_line + 700000);

  // 4. Verify Low-Level Pipeline Execution Sequence
  EXPECT_FALSE(g_trace.empty());
  EXPECT_EQ(g_trace[0], "OuterStart(0,2)");

  low_level::GetGlobalPrefetchTunerRegistry() = nullptr;
}

TEST_F(PrefetchPipeline2DTest, Tiling2DArgsTotalElements) {
  Tiling2DArgs tiling{.outer_block = 128, .inner_block = 256};

  // Case 1: inner_size <= inner_block (1 tile per row)
  EXPECT_EQ(tiling.TotalElements(10, 100), 10);

  // Case 2: inner_size > inner_block (multiple tiles per row)
  // 300 / 256 -> 2 tiles per row. 10 outer * 2 = 20.
  EXPECT_EQ(tiling.TotalElements(10, 300), 20);

  // Case 3: inner_block == 0 fallback safety (1 tile per row)
  Tiling2DArgs zero_tiling{.outer_block = 128, .inner_block = 0};
  EXPECT_EQ(zero_tiling.TotalElements(5, 500), 5);
}

}  // namespace
}  // namespace pipeline
}  // namespace hwy
