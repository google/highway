// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>

#include <array>
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
#include "hwy/contrib/distributions/uniform-inl.h"
#include "hwy/contrib/distributions/uniform.h"
#include "hwy/contrib/random/aes_ctr-inl.h"
#include "hwy/contrib/random/cached.h"
#include "hwy/contrib/random/xoshiro-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {  // required: unique per target
namespace {

// Power of two because tests do not check for vector remainders.
constexpr uint64_t kNumReps = RoundUpToPow2(AdjustedReps(1UL << 10));

uint64_t GetSeed() { return static_cast<uint64_t>(std::time(nullptr)); }

void AssertXoshiroState(
    const XoshiroBitGenerator& generator,
    const std::vector<random_internal::ScalarXoshiro>& reference) {
  const auto& state = generator.GetState();
  for (size_t lane = 0; lane < reference.size(); ++lane) {
    const auto expected = reference[lane].GetState();
    for (size_t word = 0; word < expected.size(); ++word) {
      HWY_ASSERT_EQ(expected[word], state[{word}][lane]);
    }
  }
}

// A partial vector consumes one draw from every lane, including those whose
// values are discarded. Check both that state and the following full vector.
void AssertXoshiroContinuation(
    const size_t count, XoshiroBitGenerator& generator,
    std::vector<random_internal::ScalarXoshiro>& reference) {
  const ScalableTag<uint64_t> d;
  const size_t lanes = Lanes(d);
  for (size_t i = count; i % lanes != 0; ++i) {
    (void)reference[i % lanes]();
  }
  AssertXoshiroState(generator, reference);
  const auto next = hwy::MakeUniqueAlignedArray<uint64_t>(lanes);
  Store(generator(), d, next.get());
  for (size_t lane = 0; lane < lanes; ++lane) {
    HWY_ASSERT_EQ(reference[lane](), next[lane]);
  }
  AssertXoshiroState(generator, reference);
}

void TestXoshiroKnownAnswers() {
  // Captured from the implementation before the generator/distribution split.
  constexpr uint64_t kExpected[] = {
      0xc1cdea79b61cd477ull, 0x6c9f3f8e08767fd5ull, 0xf42e33f9cec8c13dull,
      0x01b947901fad1954ull, 0xb660face9ceb86bfull, 0x5bc9dd0e61671da7ull,
      0x4256d4a912db46b6ull, 0xea7398fa53d923bfull};
  constexpr uint64_t kJumpExpected[] = {
      0x6bcb673e3a07c56eull, 0xcbf15e5bd08e7c25ull, 0x0b06aaf4982c366bull,
      0xbca57ac3d1430007ull};
  constexpr uint64_t kLongJumpExpected[] = {
      0xb8a630ec647a90d9ull, 0xa4a635fc66df839aull, 0xec12f2743c78598aull,
      0x3011147b0082faf0ull};
  random_internal::ScalarXoshiro scalar{1234};
  XoshiroBitGenerator vector{1234};
  for (const uint64_t expected : kExpected) {
    HWY_ASSERT_EQ(expected, scalar());
    HWY_ASSERT_EQ(expected, GetLane(vector()));
  }

  // Scalar thread IDs use Jump; vector thread IDs use LongJump.
  random_internal::ScalarXoshiro scalar_thread{1234, 1};
  random_internal::ScalarXoshiro long_jumped{1234};
  long_jumped.LongJump();
  XoshiroBitGenerator vector_thread{1234, 1};
  for (size_t i = 0; i < 4; ++i) {
    HWY_ASSERT_EQ(kJumpExpected[i], scalar_thread());
    HWY_ASSERT_EQ(kLongJumpExpected[i], long_jumped());
    HWY_ASSERT_EQ(kLongJumpExpected[i], GetLane(vector_thread()));
  }

#if HWY_HAVE_FLOAT64
  random_internal::ScalarXoshiro scalar_uniform{1234};
  XoshiroBitGenerator vector_uniform{1234};
  for (const uint64_t bits : kExpected) {
    const double expected =
        static_cast<double>(bits >> 11) / 9007199254740992.0;
    HWY_ASSERT_EQ(expected, hwy::Uniform()(scalar_uniform));
    HWY_ASSERT_EQ(expected, GetLane(Uniform()(vector_uniform)));
  }
#endif
}

void TestEmptyXoshiroOutput() {
  XoshiroBitGenerator generator{1234};
  const ScalableTag<uint64_t> d;
  std::vector<random_internal::ScalarXoshiro> reference;
  random_internal::ScalarXoshiro lane_reference{1234};
  for (size_t lane = 0; lane < Lanes(d); ++lane) {
    reference.push_back(lane_reference);
    lane_reference.Jump();
  }

  generator.FillBits(nullptr, 0);
  AssertXoshiroState(generator, reference);
#if HWY_HAVE_FLOAT64
  Uniform().Fill(generator, nullptr, 0);
  AssertXoshiroState(generator, reference);
#endif
  AssertXoshiroContinuation(0, generator, reference);
}

void RngLoop(const uint64_t seed, uint64_t* HWY_RESTRICT result,
             const size_t size) {
  const ScalableTag<uint64_t> d;
  XoshiroBitGenerator generator{seed};
  for (size_t i = 0; i < size; i += Lanes(d)) {
    Store(generator(), d, result + i);
  }
}

#if HWY_HAVE_FLOAT64
void UniformLoop(const uint64_t seed, double* HWY_RESTRICT result,
                 const size_t size) {
  const ScalableTag<double> d;
  XoshiroBitGenerator generator{seed};
  for (size_t i = 0; i < size; i += Lanes(d)) {
    Store(Uniform()(generator), d, result + i);
  }
}
#endif

void TestSeeding() {
  const uint64_t seed = GetSeed();
  XoshiroBitGenerator generator{seed};
  random_internal::ScalarXoshiro reference{seed};
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
  XoshiroBitGenerator generator{seed, threadId};
  random_internal::ScalarXoshiro reference{seed};

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
  std::vector<random_internal::ScalarXoshiro> reference;
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
  random_internal::ScalarXoshiro reference{seed};
  const ScalableTag<double> d;
  const size_t lanes = Lanes(d);
  for (size_t i = 0UL; i < kNumReps; i += lanes) {
    const double result = hwy::Uniform()(reference);
    if (result_array[i] != result) {
      std::cerr << "SEED: " << seed << std::endl;
      std::cerr << "TEST UNIFORM GENERATOR ERROR: result_array[" << i << "] -> "
                << result_array[i] << " != " << result << std::endl;
      HWY_ASSERT(0);
    }
  }
#endif  // HWY_HAVE_FLOAT64
}

void TestFillBits() {
  const uint64_t seed = GetSeed();
  XoshiroBitGenerator generator{seed};
  std::vector<uint64_t> result_array(kNumReps);
  generator.FillBits(result_array.data(), result_array.size());
  std::vector<random_internal::ScalarXoshiro> reference;
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

void TestFillBitsArray() {
  const uint64_t seed = GetSeed();
  XoshiroBitGenerator generator{seed};
  std::array<uint64_t, kNumReps> result_array;
  generator.FillBits(result_array.data(), result_array.size());
  std::vector<random_internal::ScalarXoshiro> reference;
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

void TestFillUniform() {
#if HWY_HAVE_FLOAT64
  const uint64_t seed = GetSeed();
  XoshiroBitGenerator generator{seed};
  std::vector<double> result_array(kNumReps);
  Uniform().Fill(generator, result_array.data(), result_array.size());
  random_internal::ScalarXoshiro reference{seed};
  const ScalableTag<double> d;
  const size_t lanes = Lanes(d);
  for (size_t i = 0UL; i < kNumReps; i += lanes) {
    const double result = hwy::Uniform()(reference);
    if (result_array[i] != result) {
      std::cerr << "SEED: " << seed << std::endl;
      std::cerr << "TEST UNIFORM GENERATOR ERROR: result_array[" << i << "] -> "
                << result_array[i] << " != " << result << std::endl;

      HWY_ASSERT(0);
    }
  }
#endif  // HWY_HAVE_FLOAT64
}

void TestFillUniformArray() {
#if HWY_HAVE_FLOAT64
  const uint64_t seed = GetSeed();
  XoshiroBitGenerator generator{seed};
  std::array<double, kNumReps> result_array;
  Uniform().Fill(generator, result_array.data(), result_array.size());
  random_internal::ScalarXoshiro reference{seed};
  const ScalableTag<double> d;
  const size_t lanes = Lanes(d);
  for (size_t i = 0UL; i < kNumReps; i += lanes) {
    const double result = hwy::Uniform()(reference);
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
// vector is handled by a single StoreN; full vectors use StoreU.
// https://github.com/google/highway/pull/3165
void TestFillRemainder() {
  const uint64_t seed = GetSeed();
  const ScalableTag<uint64_t> d;
  const size_t lanes = Lanes(d);

  // One reference stream per lane, matching XoshiroBitGenerator's interleaving:
  // result[block * lanes + lane] == stream[lane]'s block-th draw.
  std::vector<random_internal::ScalarXoshiro> prototype;
  prototype.emplace_back(seed);
  for (size_t i = 1UL; i < lanes; ++i) {
    auto rng = prototype.back();
    rng.Jump();
    prototype.emplace_back(rng);
  }

  // Dynamic buffers with several sizes that are not multiples of Lanes.
  const size_t sizes[] = {size_t{1}, lanes + 1, 3 * lanes + 1};
  for (const size_t n : sizes) {
    {
      XoshiroBitGenerator generator{seed};
      std::vector<uint64_t> result(n);
      generator.FillBits(result.data(), result.size());
      auto reference = prototype;
      for (size_t i = 0UL; i < n; ++i) {
        HWY_ASSERT(result[i] == reference[i % lanes]());
      }
      AssertXoshiroContinuation(n, generator, reference);
    }
#if HWY_HAVE_FLOAT64
    {
      XoshiroBitGenerator generator{seed};
      std::vector<double> result(n);
      Uniform().Fill(generator, result.data(), result.size());
      auto reference = prototype;
      for (size_t i = 0UL; i < n; ++i) {
        HWY_ASSERT(result[i] == hwy::Uniform()(reference[i % lanes]));
      }
      AssertXoshiroContinuation(n, generator, reference);
    }
#endif  // HWY_HAVE_FLOAT64
  }

  // Stack buffers with an odd size (never a multiple of Lanes for Lanes > 1).
  constexpr size_t kOdd = 1001;
  {
    XoshiroBitGenerator generator{seed};
    std::array<uint64_t, kOdd> result;
    generator.FillBits(result.data(), result.size());
    auto reference = prototype;
    for (size_t i = 0UL; i < kOdd; ++i) {
      HWY_ASSERT(result[i] == reference[i % lanes]());
    }
    AssertXoshiroContinuation(kOdd, generator, reference);
  }
#if HWY_HAVE_FLOAT64
  {
    XoshiroBitGenerator generator{seed};
    std::array<double, kOdd> result;
    Uniform().Fill(generator, result.data(), result.size());
    auto reference = prototype;
    for (size_t i = 0UL; i < kOdd; ++i) {
      HWY_ASSERT(result[i] == hwy::Uniform()(reference[i % lanes]));
    }
    AssertXoshiroContinuation(kOdd, generator, reference);
  }
#endif  // HWY_HAVE_FLOAT64
}

void TestBufferedXoshiro() {
  const uint64_t seed = GetSeed();

  hwy::BufferedBitGenerator<XoshiroBitGenerator> generator{
      XoshiroBitGenerator(seed)};
  std::vector<random_internal::ScalarXoshiro> reference;
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
        std::cerr << "TEST BufferedBitGenerator ERROR: result_array["
                  << i + lane << "] -> " << got << " != " << result
                  << std::endl;

        HWY_ASSERT(0);
      }
    }
  }
}

template <size_t kCacheSize>
void CheckBufferedXoshiroRefills() {
  hwy::BufferedBitGenerator<XoshiroBitGenerator, kCacheSize> generator{
      XoshiroBitGenerator(1234, 1)};
  XoshiroBitGenerator reference{1234, 1};
  // Cross multiple refill boundaries, including a cache smaller than a vector.
  for (size_t refill = 0; refill < 4; ++refill) {
    std::array<uint64_t, kCacheSize> expected;
    reference.FillBits(expected.data(), expected.size());
    for (const uint64_t value : expected) {
      HWY_ASSERT_EQ(value, generator());
    }
  }
}

void TestBufferedXoshiroRefills() {
  CheckBufferedXoshiroRefills<1>();
  CheckBufferedXoshiroRefills<8>();
  CheckBufferedXoshiroRefills<1024>();
}

void TestUniformBufferedXoshiro() {
#if HWY_HAVE_FLOAT64
  const uint64_t seed = GetSeed();

  hwy::BufferedBitGenerator<XoshiroBitGenerator> generator{
      XoshiroBitGenerator(seed)};
  std::uniform_real_distribution<double> distribution{0., 1.};
  for (size_t i = 0UL; i < kNumReps; ++i) {
    const double result = distribution(generator);

    if (result < 0. || result >= 1.) {
      std::cerr << "SEED: " << seed << std::endl;
      std::cerr << "TEST BufferedBitGenerator ERROR: result_array[" << i
                << "] -> " << result << " not in interval [0, 1)" << std::endl;
      HWY_ASSERT(0);
    }
  }
#endif  // HWY_HAVE_FLOAT64
}

// ----- AesCtrEngine / RngStream / NormalizedUniform tests -----

#if HWY_TARGET != HWY_SCALAR

void TestAesCtrKnownAnswers() {
  // Captured from the deterministic engine before the implementation split.
  constexpr uint64_t kExpected[][8] = {
      {0xf42bc87d7aa5332dull, 0x36e9834e41cc6f1bull, 0x8be39c6b9565c0f5ull,
       0xbd7fb00bfda8ebc9ull, 0x6c2da8bb06124b5full, 0x0ef4e885c89e7327ull,
       0xda2ad0887d1bcf94ull, 0x2515d6a4e51f1ac4ull},
      {0xeac52e9ded99840aull, 0xa7bc6249ceb00dadull, 0x93ae974cad561481ull,
       0xa91c77a7fe0e6947ull, 0xf3a95f7cf2a09156ull, 0xc125f2298698b823ull,
       0x2e9534d0af6c5963ull, 0x03180f4b32474041ull}};
  constexpr uint64_t kStreams[] = {0, 42};
  AesCtrEngine engine(/*deterministic=*/true);
  for (size_t i = 0; i < 2; ++i) {
    RngStream rng(engine, kStreams[i]);
    RngStream fill_rng(engine, kStreams[i]);
    uint64_t filled[8];
    fill_rng.FillBits(nullptr, 0);
    fill_rng.FillBits(filled, 8);
    for (size_t counter = 0; counter < 8; ++counter) {
      HWY_ASSERT_EQ(kExpected[i][counter], engine(kStreams[i], counter));
      HWY_ASSERT_EQ(kExpected[i][counter], rng());
      HWY_ASSERT_EQ(kExpected[i][counter], filled[counter]);
    }
    HWY_ASSERT_EQ(rng(), fill_rng());
  }
}

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

void TestNormalizedUniform() {
  AesCtrEngine engine(/*deterministic=*/true);
  RngStream rng(engine, 0);
  constexpr size_t kCount = AdjustedReps(50'000);
  double sum = 0.0;
  for (size_t i = 0; i < kCount; ++i) {
    const float f = hwy::NormalizedUniform()(rng);
    HWY_ASSERT(-1.0f <= f && f < 1.0f);
    sum += static_cast<double>(f);
  }
  // Mean should be near 0 for uniform [-1, 1).
  const double mean = sum / kCount;
  fprintf(stderr, "NormalizedUniform mean: %.6f\n", mean);
  const double kTol = kCount < 10'000 ? 0.1 : 0.01;
  HWY_ASSERT(-kTol < mean && mean < kTol);
}

#else

void TestAesCtrKnownAnswers() {}

void TestAesCtrDeterministic() {}

void TestAesCtrSeeded() {}

void TestAesCtrStreamsDiffer() {}

void TestAesCtrBitDistribution() {}

void TestAesCtrChiSquared() {}

void TestNormalizedUniform() {}

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
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestXoshiroKnownAnswers);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestEmptyXoshiroOutput);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestSeeding);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestMultiThreadSeeding);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestRandomUint64);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestFillBits);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestFillBitsArray);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestBufferedXoshiro);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestBufferedXoshiroRefills);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestUniformDist);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestFillUniform);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestFillUniformArray);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestFillRemainder);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestUniformBufferedXoshiro);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestAesCtrKnownAnswers);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestAesCtrDeterministic);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestAesCtrSeeded);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestAesCtrStreamsDiffer);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestAesCtrBitDistribution);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestAesCtrChiSquared);
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestNormalizedUniform);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
