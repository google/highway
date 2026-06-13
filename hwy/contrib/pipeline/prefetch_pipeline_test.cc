#include "hwy/contrib/pipeline/prefetch_pipeline.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "hwy/base.h"
#include "hwy/contrib/pipeline/prefetch_pipeline_types.h"
#include "hwy/contrib/pipeline/prefetch_tuner.h"
#include "hwy/contrib/pipeline/prefetch_tuner_registry.h"
#include "hwy/tests/hwy_gtest.h"

namespace hwy {
namespace pipeline {
namespace {

// A generic policy builder to parameterize the loop strategy for testing.
template <PrefetchStrategy TargetStrategy>
struct TestPolicy : DefaultPrefetchPolicy {
  static constexpr PrefetchStrategy kStrategy = TargetStrategy;
  static constexpr size_t kMaxCachelinesPerIter = 1;
};

// Global trace vector to track the exact chronological sequence of operations.
std::vector<std::string> g_trace;

void FakeDeepPrefetch(const void* ptr) {
  // "PD" --> Deep Prefetch.
  g_trace.push_back("PD(" + std::to_string(reinterpret_cast<size_t>(ptr)) +
                    ")");
}

void FakeShallowPrefetch(const void* ptr) {
  // "PS" --> Shallow Prefetch.
  g_trace.push_back("PS(" + std::to_string(reinterpret_cast<size_t>(ptr)) +
                    ")");
}

// A fake provider that pushes string events into the global trace.
struct FakeCachelinesProvider {
  template <typename Limits>
  void operator()(size_t i, CachelineBundle<Limits>& collector) const {
    // "S" --> Supply cachelines.
    g_trace.push_back("S(" + std::to_string(i) + ")");
    // Add a dummy pointer so the `Prefetch` Assembly compiles cleanly natively.
    collector.Add(reinterpret_cast<const void*>(i));
  }
};

// A fake task to assert chronological execution order.
struct FakeTask {
  void operator()(size_t i) const {
    // "T" --> Task.
    g_trace.push_back("T(" + std::to_string(i) + ")");
  }
};

class PrefetchPipelineTest : public ::testing::Test {
 protected:
  void SetUp() override { g_trace.clear(); }
};

template <PrefetchStrategy TargetStrategy, size_t Deep = 4, size_t Shallow = 2>
void CallPipeline(size_t start, size_t end) {
  PrefetchArgs args;
  args.deep_lookahead = Deep;
  args.shallow_lookahead = Shallow;
  low_level::PrefetchPipelineLoop<TestPolicy<TargetStrategy>,
                                  FakeCachelinesProvider, FakeTask,
                                  FakeDeepPrefetch, FakeShallowPrefetch>(
      start, end, FakeCachelinesProvider(), FakeTask(), args);
}

TEST_F(PrefetchPipelineTest, NoPrefetchStrategy) {
  CallPipeline<PrefetchStrategy::kNoPrefetch>(0, 5);

  std::vector<std::string> expected = {"T(0)", "T(1)", "T(2)", "T(3)", "T(4)"};
  EXPECT_EQ(g_trace, expected);
}

TEST_F(PrefetchPipelineTest, DualTierStrategy) {
  // Dual-Tier relies on a deeply staggered Phase 1, Phase 2, Phase 3 pipeline.
  // We test on an array of length 6. Lookaheads: kShallow = 2, kDeep = 4.
  CallPipeline<PrefetchStrategy::kDualTier>(0, 6);

  std::vector<std::string> expected = {
      // Phase 1: Overlapping limits. L1/L3 horizons are primed (i = 0 to 1).
      // Note: get_cachelines is called once per `i` and then distributed to
      // deep/shallow.
      "S(0)", "PD(0)", "PS(0)", "S(1)", "PD(1)", "PS(1)",
      // Phase 2: Outstanding L3. L1 window is exhausted (i = 2 to 3).
      "S(2)", "PD(2)", "S(3)", "PD(3)",
      // Phase 3a: Main Sequence (i = 0 to 1).
      "S(4)", "PD(4)", "S(2)", "PS(2)", "T(0)", "S(5)", "PD(5)", "S(3)",
      "PS(3)", "T(1)",
      // Phase 3b: Limit Deep (i = 2 to 3). Deep lookahead has reached array
      // bounds.
      "S(4)", "PS(4)", "T(2)", "S(5)", "PS(5)", "T(3)",
      // Phase 3c: Drain (i = 4 to 5). No fetches remaining.
      "T(4)", "T(5)"};
  EXPECT_EQ(g_trace, expected);
}

TEST_F(PrefetchPipelineTest, DualTierStrategy_DifferentLookaheads) {
  // Test with kShallow = 1, kDeep = 3 on an array of length 5.
  CallPipeline<PrefetchStrategy::kDualTier, 3, 1>(0, 5);

  std::vector<std::string> expected = {
      // Phase 1: Overlapping limits. L1/L3 horizons are primed (i = 0 to 0).
      "S(0)", "PD(0)", "PS(0)",
      // Phase 2: Outstanding L3. L1 window is exhausted (i = 1 to 2).
      "S(1)", "PD(1)", "S(2)", "PD(2)",
      // Phase 3a: Main Sequence (i = 0 to 1).
      // i=0 triggers deep+3 (3) and shallow+1 (1)
      "S(3)", "PD(3)", "S(1)", "PS(1)", "T(0)",
      // i=1 triggers deep+3 (4) and shallow+1 (2)
      "S(4)", "PD(4)", "S(2)", "PS(2)", "T(1)",
      // Phase 3b: Limit Deep (i = 2 to 3). Deep lookahead ends.
      // i=2 shallow+1 (3)
      "S(3)", "PS(3)", "T(2)",
      // i=3 shallow+1 (4)
      "S(4)", "PS(4)", "T(3)",
      // Phase 3c: Drain (i = 4 to 4). No fetches remaining.
      "T(4)"};
  EXPECT_EQ(g_trace, expected);
}

TEST_F(PrefetchPipelineTest, ShallowRollingLookaheadStrategy) {
  // Tests the 1D rolling array. Only shallow lookahead is active (kShallow=2).
  CallPipeline<PrefetchStrategy::kShallowLookaheadOnly>(0, 6);

  std::vector<std::string> expected = {// Startup Phase (i = 0 to 1) for L1
                                       "S(0)", "PS(0)", "S(1)", "PS(1)",
                                       // Main Sliding Loop (i = 0 to 3)
                                       "S(2)", "PS(2)", "T(0)", "S(3)", "PS(3)",
                                       "T(1)", "S(4)", "PS(4)", "T(2)", "S(5)",
                                       "PS(5)", "T(3)",
                                       // Drain (i = 4 to 5)
                                       "T(4)", "T(5)"};
  EXPECT_EQ(g_trace, expected);
}

TEST_F(PrefetchPipelineTest, DeepRollingLookaheadStrategy) {
  // Tests the 1D rolling array using the Deep boundary (kDeep=4).
  CallPipeline<PrefetchStrategy::kDeepLookaheadOnly>(0, 6);

  std::vector<std::string> expected = {
      // Startup Phase (i = 0 to 3) for L3 (kDeep = 4)
      "S(0)", "PD(0)", "S(1)", "PD(1)", "S(2)", "PD(2)", "S(3)", "PD(3)",
      // Main Sliding Loop (i = 0 to 1)
      "S(4)", "PD(4)", "T(0)", "S(5)", "PD(5)", "T(1)",
      // Drain (i = 2 to 5)
      "T(2)", "T(3)", "T(4)", "T(5)"};
  EXPECT_EQ(g_trace, expected);
}

TEST_F(PrefetchPipelineTest, MiniBatchShallowStrategy) {
  // Tests the blocked/batch prefetch array. Shallow lookahead = 2 = Batch size.
  CallPipeline<PrefetchStrategy::kMiniBatchShallow>(0, 5);

  std::vector<std::string> expected = {
      // Block 1 (i = 0 to 1)
      "S(0)", "PS(0)", "S(1)", "PS(1)", "T(0)", "T(1)",
      // Block 2 (i = 2 to 3)
      "S(2)", "PS(2)", "S(3)", "PS(3)", "T(2)", "T(3)",
      // Block 3 (remainder, i = 4 to 4)
      "S(4)", "PS(4)", "T(4)"};
  EXPECT_EQ(g_trace, expected);
}

TEST_F(PrefetchPipelineTest, MiniBatchDeepStrategy) {
  // Tests the blocked/batch prefetch array. Deep lookahead = 4 = Batch size.
  CallPipeline<PrefetchStrategy::kMiniBatchDeep>(0, 5);

  std::vector<std::string> expected = {// Block 1 (i = 0 to 3)
                                       "S(0)", "PD(0)", "S(1)", "PD(1)", "S(2)",
                                       "PD(2)", "S(3)", "PD(3)", "T(0)", "T(1)",
                                       "T(2)", "T(3)",
                                       // Block 2 (remainder, i = 4 to 4)
                                       "S(4)", "PD(4)", "T(4)"};
  EXPECT_EQ(g_trace, expected);
}

TEST_F(PrefetchPipelineTest, ZeroShallowZeroDeepFallback) {
  // Disabling both tiers should degrade down to NoPrefetch behavior entirely.
  CallPipeline<PrefetchStrategy::kDualTier, 0, 0>(0, 5);

  std::vector<std::string> expected = {"T(0)", "T(1)", "T(2)", "T(3)", "T(4)"};
  EXPECT_EQ(g_trace, expected);
}

TEST_F(PrefetchPipelineTest, ZeroShallowFallback) {
  // If shallow is 0, we degrade to pure DeepOnly logic (e.g. Rolling L3 Only)
  CallPipeline<PrefetchStrategy::kDualTier, 3, 0>(0, 4);

  std::vector<std::string> expected = {
      // Startup Deep
      "S(0)", "PD(0)", "S(1)", "PD(1)", "S(2)", "PD(2)",
      // Sliding Loop
      "S(3)", "PD(3)", "T(0)",
      // Drain
      "T(1)", "T(2)", "T(3)"};
  EXPECT_EQ(g_trace, expected);
}

TEST_F(PrefetchPipelineTest, ShallowGreaterOrEqualDeepFallback) {
  // If shallow >= deep, the L3 tier is bypassed entirely protecting LFBs.
  // Tests DualTier logic natively degrading to Rolling L1-only behavior.

#if !HWY_IS_DEBUG_BUILD
  // Note: This is only tested in release builds because the check `actual_deep
  // > actual_shallow` is an `assert`.
  CallPipeline<PrefetchStrategy::kDualTier, 2, 4>(0, 5);
#else
  CallPipeline<PrefetchStrategy::kDualTier, 0, 4>(0, 5);
#endif

  std::vector<std::string> expected = {
      // Startup Shallow (for 4 steps)
      "S(0)", "PS(0)", "S(1)", "PS(1)", "S(2)", "PS(2)", "S(3)", "PS(3)",
      // Sliding Loop
      "S(4)", "PS(4)", "T(0)",
      // Drain
      "T(1)", "T(2)", "T(3)", "T(4)"};
  EXPECT_EQ(g_trace, expected);
}

struct TestMetricContext {
  bool called = false;
  float ticks = 0;
  std::string captured_file;
  int captured_line = 0;
  size_t captured_total = 0;
  bool captured_tiny = false;
  PrefetchTuningHint captured_hint = PrefetchTuningHint::kAuto;
  uint32_t captured_dense_id = 9999;
};

void FakeMetricCollectorCb(void* user_data, float elapsed_ticks) {
  auto* ctx = static_cast<TestMetricContext*>(user_data);
  ctx->called = true;
  ctx->ticks = elapsed_ticks;
}

class MockMetricTuner : public low_level::PrefetchTuner {
 public:
  explicit MockMetricTuner(TestMetricContext* ctx) : ctx_(ctx) {}
  low_level::PrefetchTuningScope CreateScope(
      const low_level::PrefetchTuningContext& context) const override {
    if (context.file_loc != nullptr) {
      ctx_->captured_file = context.file_loc;
    }
    ctx_->captured_line = context.line_loc;
    ctx_->captured_total = context.total_elements;
    ctx_->captured_tiny = context.is_ultra_tiny;
    ctx_->captured_hint = context.hint;

    // Return custom lookahead depths: deep=3, shallow=1
    PrefetchArgs args{.deep_lookahead = 3, .shallow_lookahead = 1};
    return low_level::PrefetchTuningScope(args, FakeMetricCollectorCb, ctx_,
                                          1.0f);
  }

  CallsiteId RegisterContext(
      const low_level::PrefetchTuningContext& context) const override {
    CallsiteId id = next_id_++;
    registered_line_locs_[id] = context.line_loc;
    return id;
  }

  low_level::PrefetchTuningScope CreateScopeByCallsiteId(
      CallsiteId callsite_id,
      const low_level::PrefetchTuningContext& context) const override {
    ctx_->captured_dense_id = callsite_id;
    EXPECT_TRUE(registered_line_locs_.find(callsite_id) !=
                registered_line_locs_.end());
    EXPECT_EQ(registered_line_locs_.at(callsite_id), context.line_loc);
    return CreateScope(context);
  }

  mutable CallsiteId next_id_ = 0;
  mutable std::map<CallsiteId, int> registered_line_locs_;

 private:
  TestMetricContext* ctx_;
};

TEST_F(PrefetchPipelineTest, EndToEndPublicWrapperTuning) {
  TestMetricContext ctx;
  MockMetricTuner tuner(&ctx);
  low_level::GetGlobalPrefetchTunerRegistry() = &tuner;

  g_trace.clear();

  // Helper lambda encapsulates a single physical call site in the codebase.
  auto run_callsite_a = [&]() {
    PrefetchPipelineLoop<PrefetchTuningHint::kAuto,
                         TestPolicy<PrefetchStrategy::kDualTier>,
                         FakeCachelinesProvider, FakeTask>(
        0, 35, FakeCachelinesProvider(), FakeTask());
  };

  // 1. First call site execution (Dense ID 0 assigned and cached)
  run_callsite_a();

  EXPECT_TRUE(ctx.called);
  EXPECT_EQ(ctx.captured_total, 35);
  EXPECT_EQ(ctx.captured_dense_id, 0);
  EXPECT_EQ(tuner.next_id_, 1);

  // 2. Execute the FIRST call site again. Reuses cached Dense ID 0!
  ctx.called = false;
  run_callsite_a();

  EXPECT_TRUE(ctx.called);
  EXPECT_EQ(ctx.captured_dense_id, 0);
  EXPECT_EQ(tuner.next_id_, 1);  // No new registration!

  // 3. Execute a SECOND distinct physical call site. Gets Dense ID 1!
  ctx.called = false;
  PrefetchPipelineLoop<PrefetchTuningHint::kSequential,
                       TestPolicy<PrefetchStrategy::kDualTier>,
                       FakeCachelinesProvider, FakeTask>(
      0, 35, FakeCachelinesProvider(), FakeTask());

  EXPECT_TRUE(ctx.called);
  EXPECT_EQ(ctx.captured_dense_id, 1);
  EXPECT_EQ(tuner.next_id_, 2);

  low_level::GetGlobalPrefetchTunerRegistry() = nullptr;
}
}  // namespace
}  // namespace pipeline
}  // namespace hwy
