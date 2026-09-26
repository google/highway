// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause
//
// See the LICENSE file in the project root for the full license text.

#include "hwy/contrib/distributions/uniform.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

#include "hwy/contrib/random/cached.h"
#include "hwy/generator.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/distributions/uniform_test.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/distributions/uniform-inl.h"
#include "hwy/contrib/random/aes_ctr-inl.h"
#include "hwy/contrib/random/xoshiro-inl.h"
#include "hwy/generator-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

// Intentionally has no seed, state API, result_type, min/max, or FillBits.
struct FixedBits {
  uint64_t value;
  size_t calls;

  uint64_t operator()() {
    ++calls;
    return value;
  }
};

class MoveOnlyBits {
 public:
  explicit MoveOnlyBits(uint64_t start) : next_(new uint64_t(start)) {}
  MoveOnlyBits(MoveOnlyBits&&) = default;
  uint64_t operator()() { return (*next_)++; }
  uint64_t Next() const { return *next_; }

 private:
  std::unique_ptr<uint64_t> next_;
};

struct StatefulOffset {
  explicit StatefulOffset(uint64_t value) : offset(value), calls(0) {}
  StatefulOffset(const StatefulOffset&) = delete;

  template <class Bits>
  uint64_t operator()(Bits& bits) {
    ++calls;
    return bits() + offset;
  }

  template <class Bits>
  void Fill(Bits& bits, uint64_t* out, size_t count) {
    for (size_t i = 0; i < count; ++i) out[i] = (*this)(bits);
  }

  uint64_t offset;
  size_t calls;
};

void TestTargetGeneratorComposition() {
  // The target-specific adapter also accepts scalar-only, move-only backends.
  Generator<MoveOnlyBits> generator{MoveOnlyBits(10)};
  StatefulOffset distribution(100);
  HWY_ASSERT_EQ(uint64_t{110}, generator.Sample(distribution));
  uint64_t out[3];
  generator.Fill(distribution, out, 3);
  for (size_t i = 0; i < 3; ++i) {
    HWY_ASSERT_EQ(uint64_t{111} + i, out[i]);
  }
  generator.Fill(distribution, static_cast<uint64_t*>(nullptr), 0);
  HWY_ASSERT_EQ(size_t{4}, distribution.calls);
  const auto& view = generator;
  HWY_ASSERT_EQ(uint64_t{14}, view.GetBitGenerator().Next());

  Generator<MoveOnlyBits> moved(std::move(generator));
  HWY_ASSERT_EQ(uint64_t{34}, moved.Sample(StatefulOffset(20)));
  moved.Fill(StatefulOffset(30), out, 1);
  HWY_ASSERT_EQ(uint64_t{45}, out[0]);
  HWY_ASSERT_EQ(uint64_t{16}, moved());
  HWY_ASSERT_EQ(uint64_t{17}, moved.GetBitGenerator()());
}

void TestUniformEdges() {
  const uint64_t inputs[] = {0, 2047, 2048, uint64_t{1} << 63, UINT64_MAX};
  const double step = 1.0 / 9007199254740992.0;
  const double expected[] = {0.0, 0.0, step, 0.5, 1.0 - step};
  for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
    hwy::Generator<FixedBits> generator(FixedBits{inputs[i], 0});
    HWY_ASSERT_EQ(expected[i], generator.Sample(hwy::Uniform()));
    double out[] = {-2.0, -2.0, -2.0, -2.0};
    generator.Fill(hwy::Uniform(), out + 1, 2);
    generator.Fill(hwy::Uniform(), static_cast<double*>(nullptr), 0);
    HWY_ASSERT_EQ(-2.0, out[0]);
    HWY_ASSERT_EQ(expected[i], out[1]);
    HWY_ASSERT_EQ(expected[i], out[2]);
    HWY_ASSERT_EQ(-2.0, out[3]);
    const auto& const_generator = generator;
    HWY_ASSERT_EQ(size_t{3}, const_generator.GetBitGenerator().calls);

#if HWY_HAVE_FLOAT64
    // The target-specific distribution also accepts a scalar-only backend.
    HWY_ASSERT_EQ(expected[i], generator.Sample(Uniform()));
    generator.Fill(Uniform(), out + 1, 2);
    HWY_ASSERT_EQ(expected[i], out[1]);
    HWY_ASSERT_EQ(expected[i], out[2]);
    HWY_ASSERT_EQ(size_t{6}, generator.GetBitGenerator().calls);
#endif

    generator.GetBitGenerator().value = 0;
    HWY_ASSERT_EQ(0.0, generator.Sample(hwy::Uniform()));
  }
}

void TestNormalizedUniformEdges() {
  const uint64_t inputs[] = {0, uint64_t{1} << 22, uint64_t{1} << 23,
                             uint64_t{1} << 63, UINT64_MAX};
  const float expected[] = {-1.0f, 0.0f, -1.0f, -1.0f,
                            1.0f - 1.0f / 4194304.0f};
  for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
    hwy::Generator<FixedBits> generator(FixedBits{inputs[i], 0});
    HWY_ASSERT_EQ(expected[i], generator.Sample(hwy::NormalizedUniform()));
    float out[] = {-2.0f, -2.0f, -2.0f, -2.0f};
    generator.Fill(hwy::NormalizedUniform(), out + 1, 2);
    generator.Fill(hwy::NormalizedUniform(), static_cast<float*>(nullptr), 0);
    HWY_ASSERT_EQ(-2.0f, out[0]);
    HWY_ASSERT_EQ(expected[i], out[1]);
    HWY_ASSERT_EQ(expected[i], out[2]);
    HWY_ASSERT_EQ(-2.0f, out[3]);
    HWY_ASSERT_EQ(size_t{3}, generator.GetBitGenerator().calls);
  }
}

template <class BitGenerator>
void CheckScalarDistributions(BitGenerator bits, BitGenerator reference) {
  hwy::Generator<BitGenerator> generator(std::move(bits));
  HWY_ASSERT_EQ(hwy::Uniform::FromBits(reference()),
                generator.Sample(hwy::Uniform()));
  double out[7];
  generator.Fill(hwy::Uniform(), static_cast<double*>(nullptr), 0);
  generator.Fill(hwy::Uniform(), out, 7);
  for (double sample : out) {
    HWY_ASSERT_EQ(hwy::Uniform::FromBits(reference()), sample);
  }

#if HWY_HAVE_FLOAT64
  HWY_ASSERT_EQ(hwy::Uniform::FromBits(reference()),
                generator.Sample(Uniform()));
  generator.Fill(Uniform(), out, 7);
  for (double sample : out) {
    HWY_ASSERT_EQ(hwy::Uniform::FromBits(reference()), sample);
  }
#endif

  HWY_ASSERT_EQ(hwy::NormalizedUniform::FromBits(reference()),
                generator.Sample(hwy::NormalizedUniform()));
  float normalized[5];
  generator.Fill(hwy::NormalizedUniform(), normalized, 5);
  for (float sample : normalized) {
    HWY_ASSERT_EQ(hwy::NormalizedUniform::FromBits(reference()), sample);
  }
  HWY_ASSERT_EQ(reference(), generator());
}

void TestScalarBackends() {
  CheckScalarDistributions(random_internal::ScalarXoshiro(123),
                           random_internal::ScalarXoshiro(123));
#if HWY_TARGET != HWY_SCALAR
  const AesCtrEngine engine(/*deterministic=*/true);
  CheckScalarDistributions(RngStream(engine, 19), RngStream(engine, 19));
#endif
}

void TestVectorUniformFill() {
#if HWY_HAVE_FLOAT64
  const ScalableTag<uint64_t> du;
  const ScalableTag<double> df;
  const size_t lanes = Lanes(du);
  const size_t counts[] = {0, 1, lanes - 1, lanes, lanes + 1, 3 * lanes + 1};

  random_internal::ScalarXoshiro first_lane(123);
  first_lane.LongJump();
  first_lane.LongJump();
  std::vector<random_internal::ScalarXoshiro> initial;
  for (size_t lane = 0; lane < lanes; ++lane) {
    initial.push_back(first_lane);
    first_lane.Jump();
  }

  for (size_t count : counts) {
    Generator<XoshiroBitGenerator> generator(XoshiroBitGenerator(123, 2));
    auto reference = initial;
    auto storage = hwy::AllocateAligned<double>(count + 2);
    HWY_ASSERT(storage);
    for (size_t i = 0; i < count + 2; ++i) storage[i] = -2.0;
    // One element after an aligned allocation is unaligned for SIMD targets.
    double* const out = storage.get() + 1;
    generator.Fill(Uniform(), static_cast<double*>(nullptr), 0);
    generator.Fill(Uniform(), out, count);
    HWY_ASSERT_EQ(-2.0, storage[0]);
    HWY_ASSERT_EQ(-2.0, storage[count + 1]);
    for (size_t offset = 0; offset < count; offset += lanes) {
      for (size_t lane = 0; lane < lanes; ++lane) {
        const double expected = hwy::Uniform::FromBits(reference[lane]());
        if (offset + lane < count) {
          HWY_ASSERT_EQ(expected, out[offset + lane]);
        }
      }
    }

    // A partial output still consumes a complete vector, including lanes that
    // were not stored. Zero output must not consume any values.
    Generator<XoshiroBitGenerator> moved(std::move(generator));
    std::vector<double> next(lanes);
    StoreU(moved.Sample(Uniform()), df, next.data());
    for (size_t lane = 0; lane < lanes; ++lane) {
      HWY_ASSERT_EQ(hwy::Uniform::FromBits(reference[lane]()), next[lane]);
    }
    std::vector<uint64_t> raw(lanes);
    StoreU(moved(), du, raw.data());
    for (size_t lane = 0; lane < lanes; ++lane) {
      HWY_ASSERT_EQ(reference[lane](), raw[lane]);
    }
  }
#endif
}

void TestBufferedDistributions() {
  // Small refills also exercise targets with more lanes than cache entries.
  using Buffered = hwy::BufferedBitGenerator<XoshiroBitGenerator, 2>;
  HWY_ASSERT_EQ(uint64_t{0}, Buffered::min());
  HWY_ASSERT_EQ(UINT64_MAX, Buffered::max());
  hwy::Generator<Buffered> generator(Buffered{XoshiroBitGenerator(456)});
  XoshiroBitGenerator reference(456);
  uint64_t expected[14];
  for (size_t offset = 0; offset < 14; offset += 2) {
    reference.FillBits(expected + offset, 2);
  }

  HWY_ASSERT_EQ(hwy::Uniform::FromBits(expected[0]),
                generator.Sample(hwy::Uniform()));
  float normalized[5];
  generator.Fill(hwy::NormalizedUniform(), normalized, 5);
  for (size_t i = 0; i < 5; ++i) {
    HWY_ASSERT_EQ(hwy::NormalizedUniform::FromBits(expected[i + 1]),
                  normalized[i]);
  }
  double uniform[7];
  generator.Fill(hwy::Uniform(), uniform, 7);
  for (size_t i = 0; i < 7; ++i) {
    HWY_ASSERT_EQ(hwy::Uniform::FromBits(expected[i + 6]), uniform[i]);
  }
  HWY_ASSERT_EQ(expected[13], generator());
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(UniformTest);
HWY_EXPORT_AND_TEST_P(UniformTest, TestTargetGeneratorComposition);
HWY_EXPORT_AND_TEST_P(UniformTest, TestUniformEdges);
HWY_EXPORT_AND_TEST_P(UniformTest, TestNormalizedUniformEdges);
HWY_EXPORT_AND_TEST_P(UniformTest, TestScalarBackends);
HWY_EXPORT_AND_TEST_P(UniformTest, TestVectorUniformFill);
HWY_EXPORT_AND_TEST_P(UniformTest, TestBufferedDistributions);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
