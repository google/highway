// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include "hwy/contrib/pipeline/prefetch_pipeline_types.h"

#include <cstddef>

#include "hwy/base.h"
#include "hwy/tests/hwy_gtest.h"

namespace hwy {
namespace pipeline {
namespace {

TEST(PrefetchPipelineTypesTest, PrefetchArgsFactories) {
  PrefetchArgs r = PrefetchArgs::DefaultRandom();
  PrefetchArgs s = PrefetchArgs::DefaultSequential();

#if HWY_ARCH_ARM_A64
  EXPECT_EQ(r.deep_lookahead, 64);
  EXPECT_EQ(r.shallow_lookahead, 8);
  EXPECT_EQ(s.deep_lookahead, 32);
  EXPECT_EQ(s.shallow_lookahead, 4);
#else
  EXPECT_EQ(r.deep_lookahead, 32);
  EXPECT_EQ(r.shallow_lookahead, 4);
  EXPECT_EQ(s.deep_lookahead, 8);
  EXPECT_EQ(s.shallow_lookahead, 2);
#endif
}

TEST(PrefetchPipelineTypesTest, PrefetchArgsBitFieldLayout) {
  // 1. Ensure sizeof is exactly 2 bytes (16 bits), this is crucial for
  // minimizing the tuner overhead.
  EXPECT_EQ(sizeof(PrefetchArgs), 2);

  // 2. Ensure proper value range (deep: 9 bits -> max 511, shallow: 7 bits ->
  // max 127)
  PrefetchArgs max_args{.deep_lookahead = 511, .shallow_lookahead = 127};
  EXPECT_EQ(max_args.deep_lookahead, 511);
  EXPECT_EQ(max_args.shallow_lookahead, 127);

  // 3. Verify unsigned wrap-around behavior at exact bit boundaries
  PrefetchArgs wrap_args;
  uint16_t deep_val = 512;
  uint16_t shallow_val = 128;
  wrap_args.deep_lookahead = deep_val;
  wrap_args.shallow_lookahead = shallow_val;
  EXPECT_EQ(wrap_args.deep_lookahead, 0);
  EXPECT_EQ(wrap_args.shallow_lookahead, 0);
}

struct TinyPolicy : DefaultPrefetchPolicy {
  static constexpr size_t kMaxCachelinesPerIter = 2;
};

TEST(PrefetchPipelineTypesTest, CachelineBundleAdd) {
  CachelineBundle<TinyPolicy> bundle;
  EXPECT_EQ(bundle.count, 0);

  bundle.Add(reinterpret_cast<const void*>(0x1000));
  EXPECT_EQ(bundle.count, 1);
  EXPECT_EQ(bundle.ptrs[0], reinterpret_cast<const void*>(0x1000));

  bundle.Add(reinterpret_cast<const void*>(0x2000));
  EXPECT_EQ(bundle.count, 2);
  EXPECT_EQ(bundle.ptrs[1], reinterpret_cast<const void*>(0x2000));

  // Adding a 3rd pointer should be safely ignored or asserted in debug builds.
  // In production builds, it short-circuits cleanly without buffer overflow.
#if !HWY_IS_DEBUG_BUILD
  bundle.Add(reinterpret_cast<const void*>(0x3000));
  EXPECT_EQ(bundle.count, 2);
#endif
}

TEST(PrefetchPipelineTypesTest, CachelineBundleContiguousRange) {
  CachelineBundle<TinyPolicy> bundle;
  char buffer[256];

  // Adding 128 bytes (2 cachelines at 64B each) perfectly fills TinyPolicy
  // (capacity 2).
  bundle.AddContiguousRange(buffer, 128);
  EXPECT_EQ(bundle.count, 2);
  EXPECT_EQ(bundle.ptrs[0], buffer);
  EXPECT_EQ(bundle.ptrs[1], buffer + 64);

  // Adding more should short-circuit safely.
#if !HWY_IS_DEBUG_BUILD
  bundle.AddContiguousRange(buffer + 128, 64);
  EXPECT_EQ(bundle.count, 2);
#endif
}

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L

struct ValidCustomPolicy {
  static constexpr size_t kMaxCachelinesPerIter = 4;
  static constexpr size_t kNumMSHRs = 12;
  static constexpr PrefetchStrategy kStrategy = PrefetchStrategy::kDualTier;
  static constexpr size_t kUltraTinyThreshold = 32;
};
static_assert(low_level::IsPrefetchPolicy<ValidCustomPolicy>);
static_assert(low_level::IsPrefetchPolicy<DefaultPrefetchPolicy>);

struct InvalidMissingMSHRs {
  [[maybe_unused]] static constexpr size_t kMaxCachelinesPerIter = 4;
  [[maybe_unused]] static constexpr PrefetchStrategy kStrategy =
      PrefetchStrategy::kDualTier;
  [[maybe_unused]] static constexpr size_t kUltraTinyThreshold = 32;
};
static_assert(!low_level::IsPrefetchPolicy<InvalidMissingMSHRs>);

struct MockCachelineProvider {
  template <typename Policy>
  void operator()(size_t, CachelineBundle<Policy>&) const {}
};
static_assert(low_level::PrefetchPipelineCachelineProvider<
              MockCachelineProvider, DefaultPrefetchPolicy>);

struct MockTask {
  void operator()(size_t) const {}
};
static_assert(low_level::PrefetchPipelineTask<MockTask>);

#endif

}  // namespace
}  // namespace pipeline
}  // namespace hwy
