// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include "hwy/contrib/pipeline/prefetch_tuner_registry.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "hwy/contrib/pipeline/prefetch_pipeline_types.h"
#include "hwy/contrib/pipeline/prefetch_tuner.h"
#include "hwy/detect_compiler_arch.h"
#include "hwy/tests/hwy_gtest.h"

namespace hwy {
namespace pipeline {
namespace low_level {
namespace {

struct TestMetricContext {
  bool called = false;
  float ticks = 0;
  uint32_t captured_dense_id = 9999;
  std::string captured_file;
  int captured_line = 0;
  size_t registered_count = 0;
};

void FakeMetricCb(void* user_data, float elapsed_ticks) {
  auto* ctx = static_cast<TestMetricContext*>(user_data);
  ctx->called = true;
  ctx->ticks = elapsed_ticks;
}

class MockTunerPlugin : public PrefetchTuner {
 public:
  explicit MockTunerPlugin(TestMetricContext* ctx) : ctx_(ctx) {}
  PrefetchTuningScope CreateScope(
      const PrefetchTuningContext& context) const override {
    if (context.file_loc != nullptr) ctx_->captured_file = context.file_loc;
    ctx_->captured_line = context.line_loc;
    PrefetchArgs args{.deep_lookahead = 99, .shallow_lookahead = 9};
    return PrefetchTuningScope(args, FakeMetricCb, ctx_, 1.0f);
  }

  CallsiteId RegisterContext(
      const PrefetchTuningContext& context) const override {
    ctx_->registered_count++;
    return 77;
  }

  PrefetchTuningScope CreateScopeByCallsiteId(
      CallsiteId callsite_id,
      const PrefetchTuningContext& context) const override {
    ctx_->captured_dense_id = callsite_id;
    EXPECT_EQ(callsite_id, 77);
    return CreateScope(context);
  }

 private:
  TestMetricContext* ctx_;
};

TEST(PrefetchTunerRegistryTest, SetRegistryAndStandaloneWrappers) {
  TestMetricContext ctx;
  MockTunerPlugin plugin(&ctx);

  // Verify initial state
  EXPECT_EQ(GetGlobalPrefetchTunerRegistry(), nullptr);

  SetGlobalPrefetchTunerRegistry(&plugin);
  EXPECT_EQ(GetGlobalPrefetchTunerRegistry(), &plugin);

  PrefetchTuningContext context;
  context.hint = PrefetchTuningHint::kRandom;
  context.total_elements = 1000;
  context.file_loc = "standalone.cc";
  context.line_loc = 100;

  // 1. Test RegisterPrefetchContext
  CallsiteId dense_id = RegisterPrefetchContext(context);
  EXPECT_EQ(dense_id, 77);
  EXPECT_EQ(ctx.registered_count, 1);

  // 2. Test CreatePrefetchTuningScopeByCallsiteId
  PrefetchTuningScope scope =
      CreatePrefetchTuningScopeByCallsiteId(77, context);
  EXPECT_EQ(scope.GetArgs().deep_lookahead, 99);
  EXPECT_EQ(ctx.captured_dense_id, 77);

  // 3. Test Fallback when registry is nullptr
  GetGlobalPrefetchTunerRegistry() = nullptr;
  EXPECT_EQ(RegisterPrefetchContext(context), 0);

  PrefetchTuningScope fallback_scope =
      CreatePrefetchTuningScopeByCallsiteId(77, context);
#if HWY_ARCH_ARM_A64
  EXPECT_EQ(fallback_scope.GetArgs().deep_lookahead, 64);
#else
  EXPECT_EQ(fallback_scope.GetArgs().deep_lookahead, 32);
#endif
}

TEST(PrefetchTunerRegistryTest, GlobalRegistryRoutingAndShortCircuit) {
  TestMetricContext ctx;
  MockTunerPlugin plugin(&ctx);
  GetGlobalPrefetchTunerRegistry() = &plugin;

  PrefetchTuningContext context;
  context.hint = PrefetchTuningHint::kRandom;
  context.total_elements = 1000;
  context.is_ultra_tiny = false;

  {
    PrefetchTuningScope scope =
        CreatePrefetchTuningScopeByCallsiteId(77, context);
    EXPECT_EQ(scope.GetArgs().deep_lookahead, 99);
  }
  EXPECT_TRUE(ctx.called);

  // Test short-circuiting on ultra-tiny workloads
  ctx.called = false;
  context.is_ultra_tiny = true;
  {
    PrefetchTuningScope scope =
        CreatePrefetchTuningScopeByCallsiteId(77, context);
    // Bypasses plugin, returns DefaultRandom()
#if HWY_ARCH_ARM_A64
    EXPECT_EQ(scope.GetArgs().deep_lookahead, 64);
#else
    EXPECT_EQ(scope.GetArgs().deep_lookahead, 32);
#endif
  }
  EXPECT_FALSE(ctx.called);

  GetGlobalPrefetchTunerRegistry() = nullptr;

  // Test fallback when tuner plugin is not available
  context.is_ultra_tiny = false;
  {
    PrefetchTuningScope scope =
        CreatePrefetchTuningScopeByCallsiteId(77, context);
    // Fallback returns DefaultRandom()
#if HWY_ARCH_ARM_A64
    EXPECT_EQ(scope.GetArgs().deep_lookahead, 64);
#else
    EXPECT_EQ(scope.GetArgs().deep_lookahead, 32);
#endif
  }
}

struct TestFissionPolicy : DefaultPrefetchPolicy {
  [[maybe_unused]] static constexpr size_t kUltraTinyThreshold = 32;
};

struct TestFissionExec {
  int* captured_line;
  bool* captured_tiny;
  int base_line;

  template <size_t FissionOffset>
  void operator()(bool tiny) const {
    *captured_line = base_line + static_cast<int>(FissionOffset);
    *captured_tiny = tiny;
  }
};

TEST(PrefetchTunerRegistryTest, WorkloadFissionLineEnrichment) {
  int captured_line = 0;
  bool captured_tiny = false;
  TestFissionExec exec{&captured_line, &captured_tiny, 42};

  // 1. Ultra-Tiny (< 32)
  DispatchWorkloadFission<PrefetchTuningHint::kSequential, TestFissionPolicy>(
      10, exec);
  EXPECT_EQ(captured_line, 42);
  EXPECT_TRUE(captured_tiny);

  // 2. Sequential Tiny (<= 256)
  DispatchWorkloadFission<PrefetchTuningHint::kSequential, TestFissionPolicy>(
      128, exec);
  EXPECT_EQ(captured_line, 100042);
  EXPECT_FALSE(captured_tiny);

  // 3. Sequential Medium (<= 65536)
  DispatchWorkloadFission<PrefetchTuningHint::kSequential, TestFissionPolicy>(
      1000, exec);
  EXPECT_EQ(captured_line, 200042);
  EXPECT_FALSE(captured_tiny);

  // 4. Sequential Huge (> 65536)
  DispatchWorkloadFission<PrefetchTuningHint::kSequential, TestFissionPolicy>(
      100000, exec);
  EXPECT_EQ(captured_line, 300042);
  EXPECT_FALSE(captured_tiny);

  // 5. Random Tiny (<= 256)
  DispatchWorkloadFission<PrefetchTuningHint::kRandom, TestFissionPolicy>(128,
                                                                          exec);
  EXPECT_EQ(captured_line, 400042);
  EXPECT_FALSE(captured_tiny);

  // 6. Auto Medium (<= 65536)
  DispatchWorkloadFission<PrefetchTuningHint::kAuto, TestFissionPolicy>(1000,
                                                                        exec);
  EXPECT_EQ(captured_line, 800042);
  EXPECT_FALSE(captured_tiny);
}

TEST(PrefetchTunerRegistryTest, UniversalFissionDispatchAndStaticCaching) {
  TestMetricContext ctx;
  MockTunerPlugin plugin(&ctx);
  GetGlobalPrefetchTunerRegistry() = &plugin;

  bool runner_called = false;
  PrefetchArgs runner_args;
  auto loop_runner = [&](const PrefetchArgs& args) {
    runner_called = true;
    runner_args = args;
  };

  // 1. First execution (triggers static init RegisterPrefetchContext)
  DispatchTunedWorkload<PrefetchTuningHint::kSequential, TestFissionPolicy>(
      1000, "fission_test.cc", 50, loop_runner);

  EXPECT_TRUE(runner_called);
  EXPECT_EQ(runner_args.deep_lookahead, 99);
  EXPECT_EQ(ctx.registered_count, 1);
  EXPECT_EQ(ctx.captured_dense_id, 77);
  EXPECT_EQ(ctx.captured_line, 200050);  // 50 + 200000 (Medium Sequential)

  // 2. Re-execute the exact same lambda. Confirms RegisterPrefetchContext is
  // bypassed!
  runner_called = false;
  ctx.captured_dense_id = 9999;
  DispatchTunedWorkload<PrefetchTuningHint::kSequential, TestFissionPolicy>(
      1000, "fission_test.cc", 50, loop_runner);

  EXPECT_TRUE(runner_called);
  EXPECT_EQ(ctx.registered_count, 1);    // Still 1! Zero re-registration!
  EXPECT_EQ(ctx.captured_dense_id, 77);  // Uses statically cached 77!

  GetGlobalPrefetchTunerRegistry() = nullptr;
}

}  // namespace
}  // namespace low_level
}  // namespace pipeline
}  // namespace hwy
