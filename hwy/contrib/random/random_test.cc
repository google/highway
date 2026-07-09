// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>

#include <cstdio>
#include <ctime>
#include <iostream>  // cerr
#include <random>
#include <vector>

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/random/random_test.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {  // required: unique per target
namespace {

// Power of two because tests do not check for vector remainders.
constexpr uint64_t kNumReps = RoundUpToPow2(AdjustedReps(1UL << 10));

uint64_t GetSeed() { return static_cast<uint64_t>(std::time(nullptr)); }

void RngLoop(const uint64_t seed, uint64_t* HWY_RESTRICT result,
             const size_t size) {
  const ScalableTag<uint64_t> d;
  VectorXoshiro generator{seed};
  for (size_t i = 0; i < size; i += Lanes(d)) {
    Store(generator(), d, result + i);
  }
}

#if HWY_HAVE_FLOAT64
void UniformLoop(const uint64_t seed, double* HWY_RESTRICT result,
                 const size_t size) {
  const ScalableTag<double> d;
  VectorXoshiro generator{seed};
  for (size_t i = 0; i < size; i += Lanes(d)) {
    Store(generator.Uniform(), d, result + i);
  }
}
#endif

void TestSeeding() {
  const uint64_t seed = GetSeed();
  VectorXoshiro generator{seed};
  internal::Xoshiro reference{seed};
  const auto& state = generator.GetState();
  const ScalableTag<uint64_t> d;
  const size_t lanes = Lanes(d);
  for (size_t i = 0UL; i < lanes; ++i) {
    const auto& reference_state = reference.GetState();
    for (size_t j = 0UL; j < reference_state.size(); ++j) {
      if (state[{j}][i] != reference_state[j]) {
        std::cerr << "SEED: " << seed << "\n";
        std::cerr << "TEST SEEDING ERROR: ";
        std::cerr << "state[" << j << "][" << i << "] -> " << state[{j}][i]
                  << " != " << reference_state[j] << "\n";
        HWY_ASSERT(0);
      }
    }
    reference.Jump();
  }
}

void TestMultiThreadSeeding() {
  const uint64_t seed = GetSeed();
  const uint64_t threadId = GetSeed() % 1000;
  VectorXoshiro generator{seed, threadId};
  internal::Xoshiro reference{seed};

  for (size_t i = 0UL; i < threadId; ++i) {
    reference.LongJump();
  }

  const auto& state = generator.GetState();
  const ScalableTag<uint64_t> d;
  const size_t lanes = Lanes(d);
  for (size_t i = 0UL; i < lanes; ++i) {
    const auto& reference_state = reference.GetState();
    for (size_t j = 0UL; j < reference_state.size(); ++j) {
      if (state[{j}][i] != reference_state[j]) {
        std::cerr << "SEED: " << seed << std::endl;
        std::cerr << "TEST SEEDING ERROR: ";
        std::cerr << "state[" << j << "][" << i << "] -> " << state[{j}][i]
                  << " != " << reference_state[j] << "\n";
        HWY_ASSERT(0);
      }
    }
    reference.Jump();
  }
}

void TestRandomUint64() {
  const uint64_t seed = GetSeed();
  const auto result_array = hwy::MakeUniqueAlignedArray<uint64_t>(kNumReps);
  RngLoop(seed, result_array.get(), kNumReps);
  std::vector<internal::Xoshiro> reference;
  reference.emplace_back(seed);
  const ScalableTag<uint64_t> d;
  const size_t lanes = Lanes(d);
  for (size_t i = 1UL; i < lanes; ++i) {
    auto rng = reference.back();
    rng.Jump();
    reference.emplace_back(rng);
  }

  for (size_t i = 0UL; i < kNumReps; i += lanes) {
    for (size_t lane = 0UL; lane < lanes; ++lane) {
      const uint64_t result = reference[lane]();
      if (result_array[i + lane] != result) {
        std::cerr << "SEED: " << seed << std::endl;
        std::cerr << "TEST UINT64 GENERATOR ERROR: result_array[" << i + lane
                  << "] -> " << result_array[i + lane] << " != " << result
                  << std::endl;
        HWY_ASSERT(0);
      }
    }
  }
}
void TestUniformDist() {
#if HWY_HAVE_FLOAT64
  const uint64_t seed = GetSeed();
  const auto result_array = hwy::MakeUniqueAlignedArray<double>(kNumReps);
  UniformLoop(seed, result_array.get(), kNumReps);
  internal::Xoshiro reference{seed};
  const ScalableTag<double> d;
  const size_t lanes = Lanes(d);
  for (size_t i = 0UL; i < kNumReps; i += lanes) {
    const double result = reference.Uniform();
    if (result_array[i] != result) {
      std::cerr << "SEED: " << seed << std::endl;
      std::cerr << "TEST UNIFORM GENERATOR ERROR: result_array[" << i << "] -> "
                << result_array[i] << " != " << result << std::endl;
      HWY_ASSERT(0);
    }
  }
#endif  // HWY_HAVE_FLOAT64
}

void TestNextNRandomUint64() {
  const uint64_t seed = GetSeed();
  VectorXoshiro generator{seed};
  const auto result_array = generator.operator()(kNumReps);
  std::vector<internal::Xoshiro> reference;
  reference.emplace_back(seed);
  const ScalableTag<uint64_t> d;
  const size_t lanes = Lanes(d);
  for (size_t i = 1UL; i < lanes; ++i) {
    auto rng = reference.back();
    rng.Jump();
    reference.emplace_back(rng);
  }

  for (size_t i = 0UL; i < kNumReps; i += lanes) {
    for (size_t lane = 0UL; lane < lanes; ++lane) {
      const uint64_t result = reference[lane]();
      if (result_array[i + lane] != result) {
        std::cerr << "SEED: " << seed << std::endl;
        std::cerr << "TEST UINT64 GENERATOR ERROR: result_array[" << i + lane
                  << "] -> " << result_array[i + lane] << " != " << result
                  << std::endl;
        HWY_ASSERT(0);
      }
    }
  }
}

void TestNextFixedNRandomUint64() {
  const uint64_t seed = GetSeed();
  VectorXoshiro generator{seed};
  const auto result_array = generator.operator()<kNumReps>();
  std::vector<internal::Xoshiro> reference;
  reference.emplace_back(seed);
  const ScalableTag<uint64_t> d;
  const size_t lanes = Lanes(d);
  for (size_t i = 1UL; i < lanes; ++i) {
    auto rng = reference.back();
    rng.Jump();
    reference.emplace_back(rng);
  }

  for (size_t i = 0UL; i < kNumReps; i += lanes) {
    for (size_t lane = 0UL; lane < lanes; ++lane) {
      const uint64_t result = reference[lane]();
      if (result_array[i + lane] != result) {
        std::cerr << "SEED: " << seed << std::endl;
        std::cerr << "TEST UINT64 GENERATOR ERROR: result_array[" << i + lane
                  << "] -> " << result_array[i + lane] << " != " << result
                  << std::endl;

        HWY_ASSERT(0);
      }
    }
  }
}

void TestNextNUniformDist() {
#if HWY_HAVE_FLOAT64
  const uint64_t seed = GetSeed();
  VectorXoshiro generator{seed};
  const auto result_array = generator.Uniform(kNumReps);
  internal::Xoshiro reference{seed};
  const ScalableTag<double> d;
  const size_t lanes = Lanes(d);
  for (size_t i = 0UL; i < kNumReps; i += lanes) {
    const double result = reference.Uniform();
    if (result_array[i] != result) {
      std::cerr << "SEED: " << seed << std::endl;
      std::cerr << "TEST UNIFORM GENERATOR ERROR: result_array[" << i << "] -> "
                << result_array[i] << " != " << result << std::endl;

      HWY_ASSERT(0);
    }
  }
#endif  // HWY_HAVE_FLOAT64
}

void TestNextFixedNUniformDist() {
#if HWY_HAVE_FLOAT64
  const uint64_t seed = GetSeed();
  VectorXoshiro generator{seed};
  const auto result_array = generator.Uniform<kNumReps>();
  internal::Xoshiro reference{seed};
  const ScalableTag<double> d;
  const size_t lanes = Lanes(d);
  for (size_t i = 0UL; i < kNumReps; i += lanes) {
    const double result = reference.Uniform();
    if (result_array[i] != result) {
      std::cerr << "SEED: " << seed << std::endl;
      std::cerr << "TEST UNIFORM GENERATOR ERROR: result_array[" << i << "] -> "
                << result_array[i] << " != " << result << std::endl;
      HWY_ASSERT(0);
    }
  }
#endif  // HWY_HAVE_FLOAT64
}

// Regression test: sizes that are NOT a multiple of Lanes must be generated
// correctly and without writing past the end of the result. The final partial
// vector is handled by a single StoreN; the bulk uses the cheaper Store.
// https://github.com/google/highway/pull/3165
void TestNextNRemainder() {
  const uint64_t seed = GetSeed();
  const ScalableTag<uint64_t> d;
  const size_t lanes = Lanes(d);

  // One reference stream per lane, matching VectorXoshiro's interleaving:
  // result[block * lanes + lane] == stream[lane]'s block-th draw.
  std::vector<internal::Xoshiro> prototype;
  prototype.emplace_back(seed);
  for (size_t i = 1UL; i < lanes; ++i) {
    auto rng = prototype.back();
    rng.Jump();
    prototype.emplace_back(rng);
  }

  // Dynamic-size overloads, with several sizes that are not multiples of Lanes.
  const size_t sizes[] = {size_t{1}, lanes + 1, 3 * lanes + 1};
  for (const size_t n : sizes) {
    {
      VectorXoshiro generator{seed};
      const auto result = generator(n);
      HWY_ASSERT(result.size() == n);
      auto reference = prototype;
      for (size_t i = 0UL; i < n; ++i) {
        HWY_ASSERT(result[i] == reference[i % lanes]());
      }
    }
#if HWY_HAVE_FLOAT64
    {
      VectorXoshiro generator{seed};
      const auto result = generator.Uniform(n);
      HWY_ASSERT(result.size() == n);
      auto reference = prototype;
      for (size_t i = 0UL; i < n; ++i) {
        HWY_ASSERT(result[i] == reference[i % lanes].Uniform());
      }
    }
#endif  // HWY_HAVE_FLOAT64
  }

  // Fixed-size (std::array) overloads with an odd size (never a multiple of
  // Lanes for Lanes > 1).
  constexpr size_t kOdd = 1001;
  {
    VectorXoshiro generator{seed};
    const auto result = generator.operator()<kOdd>();
    auto reference = prototype;
    for (size_t i = 0UL; i < kOdd; ++i) {
      HWY_ASSERT(result[i] == reference[i % lanes]());
    }
  }
#if HWY_HAVE_FLOAT64
  {
    VectorXoshiro generator{seed};
    const auto result = generator.Uniform<kOdd>();
    auto reference = prototype;
    for (size_t i = 0UL; i < kOdd; ++i) {
      HWY_ASSERT(result[i] == reference[i % lanes].Uniform());
    }
  }
#endif  // HWY_HAVE_FLOAT64
}

void TestCachedXorshiro() {
  const uint64_t seed = GetSeed();

  CachedXoshiro<> generator{seed};
  std::vector<internal::Xoshiro> reference;
  reference.emplace_back(seed);
  const ScalableTag<uint64_t> d;
  const size_t lanes = Lanes(d);
  for (size_t i = 1UL; i < lanes; ++i) {
    auto rng = reference.back();
    rng.Jump();
    reference.emplace_back(rng);
  }

  for (size_t i = 0UL; i < kNumReps; i += lanes) {
    for (size_t lane = 0UL; lane < lanes; ++lane) {
      const uint64_t result = reference[lane]();
      const uint64_t got = generator();
      if (got != result) {
        std::cerr << "SEED: " << seed << std::endl;
        std::cerr << "TEST CachedXoshiro GENERATOR ERROR: result_array["
                  << i + lane << "] -> " << got << " != " << result
                  << std::endl;

        HWY_ASSERT(0);
      }
    }
  }
}
void TestUniformCachedXorshiro() {
#if HWY_HAVE_FLOAT64
  const uint64_t seed = GetSeed();

  CachedXoshiro<> generator{seed};
  std::uniform_real_distribution<double> distribution{0., 1.};
  for (size_t i = 0UL; i < kNumReps; ++i) {
    const double result = distribution(generator);

    if (result < 0. || result >= 1.) {
      std::cerr << "SEED: " << seed << std::endl;
      std::cerr << "TEST CachedXoshiro GENERATOR ERROR: result_array[" << i
                << "] -> " << result << " not in interval [0, 1)" << std::endl;
      HWY_ASSERT(0);
    }
  }
#endif  // HWY_HAVE_FLOAT64
}

// ----- AesCtrEngine / RngStream / RandomNormalizedFloat tests -----

#if HWY_TARGET != HWY_SCALAR

void TestAesCtrDeterministic() {
  const AesCtrEngine engine1(/*deterministic=*/true);
  const AesCtrEngine engine2(/*deterministic=*/true);
  RngStream rng1(engine1, 0);
  RngStream rng2(engine2, 0);
  // Remember for later testing after resetting the stream.
  const uint64_t r0 = rng1();
  const uint64_t r1 = rng1();
  // Not consecutive values.
  HWY_ASSERT(r0 != r1);
  // Let rng2 catch up.
  HWY_ASSERT(r0 == rng2());
  HWY_ASSERT(r1 == rng2());

  for (size_t i = 0; i < AdjustedReps(1000); ++i) {
    HWY_ASSERT(rng1() == rng2());
  }

  // Reset counter, ensure it matches the prior sequence.
  rng1 = RngStream(engine1, 0);
  HWY_ASSERT(r0 == rng1());
  HWY_ASSERT(r1 == rng1());
}

void TestAesCtrSeeded() {
  AesCtrEngine engine1(/*deterministic=*/true);
  AesCtrEngine engine2(/*deterministic=*/false);
  RngStream rng1(engine1, 0);
  RngStream rng2(engine2, 0);
  // It would be very unlucky to have even one 64-bit value match, and two are
  // extremely unlikely.
  const uint64_t a0 = rng1();
  const uint64_t a1 = rng1();
  const uint64_t b0 = rng2();
  const uint64_t b1 = rng2();
  HWY_ASSERT(a0 != b0 || a1 != b1);
}

void TestAesCtrStreamsDiffer() {
  AesCtrEngine engine(/*deterministic=*/true);
  // Compare random streams for more coverage than just the first N streams.
  RngStream rng_for_stream(engine, 0);
  for (size_t i = 0; i < AdjustedReps(1000); ++i) {
    RngStream rng1(engine, rng_for_stream());
    RngStream rng2(engine, rng_for_stream());
    // It would be very unlucky to have even one 64-bit value match, and two are
    // extremely unlikely.
    const uint64_t a0 = rng1();
    const uint64_t a1 = rng1();
    const uint64_t b0 = rng2();
    const uint64_t b1 = rng2();
    HWY_ASSERT(a0 != b0 || a1 != b1);
  }
}

// If not close to 50% 1-bits, the RNG is quite broken.
void TestAesCtrBitDistribution() {
  AesCtrEngine engine(/*deterministic=*/true);
  RngStream rng(engine, 0);
  constexpr size_t kCount = AdjustedReps(AdjustedReps(200'000));
  uint64_t one_bits = 0;
  for (size_t i = 0; i < kCount; ++i) {
    one_bits += hwy::PopCount(rng());
  }
  const uint64_t total_bits = kCount * 64;
  const double one_ratio = static_cast<double>(one_bits) / total_bits;
  fprintf(stderr, "AesCtr 1-bit ratio %.5f\n", one_ratio);
  const double kTol = kCount < 10'000 ? 0.01 : 0.001;
  HWY_ASSERT(0.5 - kTol <= one_ratio && one_ratio <= 0.5 + kTol);
}

void TestAesCtrChiSquared() {
  AesCtrEngine engine(/*deterministic=*/true);
  RngStream rng(engine, 0);
  constexpr size_t kCount = AdjustedReps(AdjustedReps(100'000));

  // Test each byte separately.
  for (size_t shift = 0; shift < 64; shift += 8) {
    size_t counts[256] = {};
    for (size_t i = 0; i < kCount; ++i) {
      const size_t byte = (rng() >> shift) & 0xFF;
      counts[byte]++;
    }

    double chi_squared = 0.0;
    const double expected = static_cast<double>(kCount) / 256.0;
    for (size_t i = 0; i < 256; ++i) {
      const double diff = static_cast<double>(counts[i]) - expected;
      chi_squared += diff * diff / expected;
    }
    // Should be within ~0.5% and 99.5% percentiles. See
    // https://www.medcalc.org/manual/chi-square-table.php
    if (chi_squared < 196.0 || chi_squared > 311.0) {
      HWY_ABORT("Chi-squared byte %zu: %.5f \n", shift / 8, chi_squared);
    }
  }
}

void TestRandomNormalizedFloat() {
  AesCtrEngine engine(/*deterministic=*/true);
  RngStream rng(engine, 0);
  constexpr size_t kCount = AdjustedReps(50'000);
  double sum = 0.0;
  for (size_t i = 0; i < kCount; ++i) {
    const float f = RandomNormalizedFloat(rng);
    HWY_ASSERT(-1.0f <= f && f < 1.0f);
    sum += static_cast<double>(f);
  }
  // Mean should be near 0 for uniform [-1, 1).
  const double mean = sum / kCount;
  fprintf(stderr, "RandomNormalizedFloat mean: %.6f\n", mean);
  const double kTol = kCount < 10'000 ? 0.1 : 0.01;
  HWY_ASSERT(-kTol < mean && mean < kTol);
}

#else

void TestAesCtrDeterministic() {}

void TestAesCtrSeeded() {}

void TestAesCtrStreamsDiffer() {}

void TestAesCtrBitDistribution() {}

void TestAesCtrChiSquared() {}

void TestRandomNormalizedFloat() {}

#endif  // HWY_TARGET != HWY_SCALAR

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();  // required if not using HWY_ATTR

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(HwyRandomTest);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestSeeding);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestMultiThreadSeeding);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestRandomUint64);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestNextNRandomUint64);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestNextFixedNRandomUint64);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestCachedXorshiro);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestUniformDist);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestNextNUniformDist);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestNextFixedNUniformDist);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestNextNRemainder);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestUniformCachedXorshiro);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestAesCtrDeterministic);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestAesCtrSeeded);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestAesCtrStreamsDiffer);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestAesCtrBitDistribution);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestAesCtrChiSquared);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestRandomNormalizedFloat);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
