#include <cstdint>
#include <cstdio>
#include <ctime>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/random/random_test.cc"

#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/contrib/random/random.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();

namespace hwy {


namespace HWY_NAMESPACE {  // required: unique per target

namespace hn = hwy::HWY_NAMESPACE;

constexpr auto tests = 1UL << 15;

std::uint64_t GetSeed() {
  return static_cast<uint64_t>(std::time(nullptr));
}

void RngLoop(const std::uint64_t seed, std::uint64_t* HWY_RESTRICT result,
             const size_t size) {
  const hn::ScalableTag<std::uint64_t> d;
  using ARRAY_T = decltype(Undefined(hn::ScalableTag<std::uint64_t>()));
  VectorXoshiro<ARRAY_T> generator{seed};
  for (size_t i = 0; i < size; i += Lanes(d)) {
    auto x = generator();
    hn::Store(x, d, result + i);
  }
}

void UniformLoop(const std::uint64_t seed, double* HWY_RESTRICT result,
                 const size_t size) {
  const hn::ScalableTag<double> d;
  using ARRAY_T = decltype(Undefined(hn::ScalableTag<std::uint64_t>()));
  VectorXoshiro<ARRAY_T> gernerator{seed};
  for (size_t i = 0; i < size; i += Lanes(d)) {
    auto x = gernerator.Uniform();
    hn::Store(x, d, result + i);
  }
}

void TestSeeding() {
  const std::uint64_t seed = GetSeed();
  const hn::ScalableTag<std::uint64_t> d;
  using ARRAY_T = decltype(Undefined(d));

  VectorXoshiro<ARRAY_T> gernerator{seed};
  const auto state = gernerator.GetState();
  internal::Xoshiro reference{seed};
  const auto lanes = Lanes(d);
  auto index = 0UL;
  for (auto i = 0UL; i < lanes; ++i) {
    for (auto elem : reference.GetState()) {
      if (state[index++] != elem) {
        fprintf(stderr, "SEED: %lu\n", seed);
        fprintf(stderr, "TEST SEEDING ERROR: state[%lu] -> %lu != %lu\n", index,
                state[index], elem);
        HWY_ASSERT(0);
      }
    }
    reference.Jump();
  }
}

void TestRandomUint64() {
  const std::uint64_t seed = GetSeed();
  const auto result_array = hwy::MakeUniqueAlignedArray<std::uint64_t>(tests);
  RngLoop(seed, result_array.get(), tests);
  internal::Xoshiro reference{seed};
  const hn::ScalableTag<std::uint64_t> d;
  const auto lanes = Lanes(d);

  for (auto i = 0UL; i < tests; i += lanes) {
    const auto result = reference();
    if (result_array[i] != result) {
      fprintf(stderr, "SEED: %lu\n", seed);
      fprintf(stderr,
              "TEST UINT64 GENERATOR ERROR: result_array[%lu] -> %lu != %lu\n",
              i, result_array[i], result);
      HWY_ASSERT(0);
    }
  }
}

void TestUniform() {
  const std::uint64_t seed = GetSeed();
  const auto result_array = hwy::MakeUniqueAlignedArray<double>(tests);
  UniformLoop(seed, result_array.get(), tests);
  internal::Xoshiro reference{seed};
  const hn::ScalableTag<double> d;
  const auto lanes = Lanes(d);
  for (auto i = 0UL; i < tests; i += lanes) {
    const auto result = reference.Uniform();
    if (result_array[i] != result) {
      fprintf(stderr, "SEED: %lu\n", seed);
      fprintf(stderr,
              "TEST UINT64 GENERATOR ERROR: result_array[%lu] -> %f != %f\n", i,
              result_array[i], result);
      HWY_ASSERT(0);
    }
  }
}
}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();  // required if not using HWY_ATTR

#if HWY_ONCE

// This macro declares a static array used for dynamic dispatch.
namespace hwy {
HWY_BEFORE_TEST(HwyRandomTest);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestSeeding);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestRandomUint64);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestUniform);
}  // namespace hwy

#endif
