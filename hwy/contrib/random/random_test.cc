// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <ctime>

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/random/random_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/random/random.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {  // required: unique per target

constexpr auto tests = 1UL << 15;

std::uint64_t GetSeed() { return static_cast<uint64_t>(std::time(nullptr)); }

void RngLoop(const std::uint64_t seed, std::uint64_t* HWY_RESTRICT result,
             const size_t size) {
  const ScalableTag<std::uint64_t> d;
  VectorXoshiro generator{seed};
  for (size_t i = 0; i < size; i += Lanes(d)) {
    auto x = generator();
    Store(x, d, result + i);
  }
}

void UniformLoop(const std::uint64_t seed, double* HWY_RESTRICT result,
                 const size_t size) {
  const ScalableTag<double> d;
  VectorXoshiro generator{seed};
  for (size_t i = 0; i < size; i += Lanes(d)) {
    auto x = generator.Uniform();
    Store(x, d, result + i);
  }
}

void TestSeeding() {
  const std::uint64_t seed = GetSeed();
  VectorXoshiro generator{seed};
  internal::Xoshiro reference{seed};
  const auto& state = generator.GetState();
  const auto lanes = state.size();
  for (auto i = 0UL; i < lanes; ++i) {
    const auto& reference_state = reference.GetState();
    for (auto j = 0UL; j < reference_state.size(); ++j) {
      if (state[j][i] != reference_state[j]) {
        fprintf(stderr, "SEED: %lu\n", seed);
        fprintf(stderr, "TEST SEEDING ERROR: ");
        fprintf(stderr, "state[%lu][%lu] -> %lu != %lu\n", j, i, state[j][i],
                reference_state[j]);
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
  std::vector<internal::Xoshiro> reference;
  reference.emplace_back(seed);
  const ScalableTag<std::uint64_t> d;
  const auto lanes = Lanes(d);
  for (auto i = 1UL; i < lanes; ++i) {
    auto rng = reference.back();
    rng.Jump();
    reference.emplace_back(rng);
  }

  for (auto i = 0UL; i < tests; i += lanes) {
    for (auto lane = 0UL; lane < lanes; ++lane) {
      const auto result = reference[lane]();
      if (result_array[i + lane] != result) {
        fprintf(stderr, "SEED: %lu\n", seed);
        fprintf(
            stderr,
            "TEST UINT64 GENERATOR ERROR: result_array[%lu] -> %lu != %lu\n",
            i + lane, result_array[i + lane], result);
        HWY_ASSERT(0);
      }
    }
  }
}

void TestUniformDist() {
  const std::uint64_t seed = GetSeed();
  const auto result_array = hwy::MakeUniqueAlignedArray<double>(tests);
  UniformLoop(seed, result_array.get(), tests);
  internal::Xoshiro reference{seed};
  const ScalableTag<double> d;
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
HWY_EXPORT_AND_TEST_P(HwyRandomTest, TestUniformDist);
}  // namespace hwy

#endif
