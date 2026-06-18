// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include "hwy/contrib/pipeline/prefetch_tuner.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "hwy/contrib/pipeline/prefetch_pipeline_types.h"
#include "hwy/tests/hwy_gtest.h"

namespace hwy {
namespace pipeline {
namespace low_level {
namespace {

struct TestMetricContext {
  bool called = false;
  float ticks = 0;
};

void FakeMetricCb(void* user_data, float elapsed_ticks) {
  auto* ctx = static_cast<TestMetricContext*>(user_data);
  ctx->called = true;
  ctx->ticks = elapsed_ticks;
}

struct FakeTimer {
  static uint64_t fake_ticks;
  static uint64_t Start() { return fake_ticks; }
  static uint64_t Stop() { return fake_ticks; }
};
uint64_t FakeTimer::fake_ticks = 0;

TEST(PrefetchTunerTest, ScopeLifecycleAndMoveSemantics) {
  TestMetricContext ctx;
  PrefetchArgs args{.deep_lookahead = 16, .shallow_lookahead = 4};

  {
    PrefetchTuningScope scope1(args, FakeMetricCb, &ctx, 1.0f);
    EXPECT_EQ(scope1.GetArgs().deep_lookahead, 16);
    EXPECT_FALSE(ctx.called);

    // Move construct into scope2
    PrefetchTuningScope scope2(std::move(scope1));
    EXPECT_EQ(scope2.GetArgs().deep_lookahead, 16);
    EXPECT_FALSE(ctx.called);

    // Move assign into scope3
    PrefetchTuningScope scope3;
    scope3 = std::move(scope2);
    EXPECT_EQ(scope3.GetArgs().deep_lookahead, 16);
    EXPECT_FALSE(ctx.called);
  }
  // Destructor of scope3 fires exactly once
  EXPECT_TRUE(ctx.called);
}

TEST(PrefetchTunerTest, CostNormalizationWithFakeTimer) {
  TestMetricContext ctx;
  PrefetchArgs args{.deep_lookahead = 32, .shallow_lookahead = 8};

  FakeTimer::fake_ticks = 1000;
  {
    // Create scope with cost normalization factor = 10.0f
    PrefetchTuningScopeT<FakeTimer> scope(args, FakeMetricCb, &ctx, 10.0f);
    EXPECT_EQ(scope.GetArgs().deep_lookahead, 32);
    EXPECT_FALSE(ctx.called);

    FakeTimer::fake_ticks = 2000;  // elapsed = 1000 ticks
  }

  EXPECT_TRUE(ctx.called);
  // normalized cost = 1000 ticks / 10.0f = 100.0f
  EXPECT_FLOAT_EQ(ctx.ticks, 100.0f);
}

}  // namespace
}  // namespace low_level
}  // namespace pipeline
}  // namespace hwy
