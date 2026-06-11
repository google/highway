#include "hwy/contrib/pipeline/prefetch_pipeline.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "hwy/contrib/pipeline/prefetch_args.h"
#include "hwy/tests/hwy_gtest.h"

namespace hwy {
namespace HWY_NAMESPACE {
namespace {

// A generic policy builder to parameterize the loop strategy for testing.
template <PrefetchStrategy TargetStrategy>
struct TestLimits : DefaultPrefetchLimits {
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
  template <size_t kMaxCachelinesPerIter>
  void operator()(size_t i,
                  PrefetchCachelines<kMaxCachelinesPerIter>& cachelines) const {
    // "S" --> Supply cachelines.
    g_trace.push_back("S(" + std::to_string(i) + ")");
    // Add a dummy pointer so the `Prefetch` Assembly compiles cleanly natively.
    cachelines.Add(reinterpret_cast<const void*>(i));
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
  PrefetchPipelineLoop<TestLimits<TargetStrategy>, FakeCachelinesProvider,
                       FakeTask, FakeDeepPrefetch, FakeShallowPrefetch>(
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
  CallPipeline<PrefetchStrategy::kDualTier, 2, 4>(0, 5);

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
  uint64_t ticks = 0;
};

void FakeMetricCollectorCb(void* user_data, uint64_t elapsed_ticks) {
  auto* ctx = static_cast<TestMetricContext*>(user_data);
  ctx->called = true;
  ctx->ticks = elapsed_ticks;
}

TEST_F(PrefetchPipelineTest, MetricCollectorCallback) {
  TestMetricContext ctx;
  PrefetchArgs args;
  args.metric_collector_cb = FakeMetricCollectorCb;
  args.user_data = &ctx;

  PrefetchPipelineLoop<TestLimits<PrefetchStrategy::kNoPrefetch>,
                       FakeCachelinesProvider, FakeTask, FakeDeepPrefetch,
                       FakeShallowPrefetch>(0, 5, FakeCachelinesProvider(),
                                            FakeTask(), args);

  EXPECT_TRUE(ctx.called);
  // Elapsed ticks can be small, but it guarantees the callback was fully fired.
}

}  // namespace
}  // namespace HWY_NAMESPACE

}  // namespace hwy
