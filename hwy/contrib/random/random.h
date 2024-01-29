
#if defined(HIGHWAY_HWY_CONTRIB_RANDOM_RANDOM_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_RANDOM_RANDOM_H_
#undef HIGHWAY_HWY_CONTRIB_RANDOM_RANDOM_H_
#else
#define HIGHWAY_HWY_CONTRIB_RANDOM_RANDOM_H_
#endif

#include <hwy/aligned_allocator.h>
#include <hwy/highway.h>

#include <array>
#include <cstdint>
#include <limits>

HWY_BEFORE_NAMESPACE();  // required if not using HWY_ATTR

namespace hwy {

namespace HWY_NAMESPACE {  // required: unique per target
namespace internal {

// C++ < 17 does not support hexfloat
#if __cpp_hex_float > 201603L
constexpr double MUL_CONST = 0x1.0p-53;
#else
constexpr double MUL_CONST =
    0.00000000000000011102230246251565404236316680908203125;
#endif

class SplitMix64 {
 public:
  constexpr SplitMix64(const std::uint64_t state) noexcept : state_(state) {}

  HWY_CXX14_CONSTEXPR std::uint64_t operator()() {
    std::uint64_t z = (state_ += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  }

 private:
  std::uint64_t state_;
};

class Xoshiro {
 public:
  HWY_CXX14_CONSTEXPR explicit Xoshiro(const std::uint64_t seed) noexcept : state_{} {
    SplitMix64 splitMix64{seed};
    for (auto& element : state_) {
      element = splitMix64();
    }
  }

  HWY_CXX14_CONSTEXPR explicit Xoshiro(const std::uint64_t seed,
                             const std::uint64_t thread_id) noexcept
      : Xoshiro(seed) {
    for (auto i = UINT64_C(0); i < thread_id; ++i) {
      Jump();
    }
  }

  HWY_CXX14_CONSTEXPR std::uint64_t operator()() noexcept { return Next(); }

  HWY_CXX14_CONSTEXPR double Uniform() noexcept {
    return static_cast<double>(Next() >> 11) * MUL_CONST;
  }

  constexpr std::array<std::uint64_t, 4> GetState() const {
    return {state_[0], state_[1], state_[2], state_[3]};
  }

  HWY_CXX17_CONSTEXPR void SetState(
      std::array<std::uint64_t, 4> state) noexcept {
    state_[0] = state[0];
    state_[1] = state[1];
    state_[2] = state[2];
    state_[3] = state[3];
  }

  static constexpr std::uint64_t StateSize() noexcept { return 4; }

 private:
  std::uint64_t state_[4];

  static constexpr std::uint64_t Rotl(const std::uint64_t x, int k) noexcept {
    return (x << k) | (x >> (64 - k));
  }

  HWY_CXX14_CONSTEXPR std::uint64_t Next() noexcept {
    const std::uint64_t result = Rotl(state_[0] + state_[3], 23) + state_[0];
    const std::uint64_t t = state_[1] << 17;

    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];

    state_[2] ^= t;

    state_[3] = Rotl(state_[3], 45);

    return result;
  }

 public:
  /* This is the jump function for the generator. It is equivalent
 to 2^128 calls to next(); it can be used to generate 2^128
 non-overlapping subsequences for parallel computations. */
  HWY_CXX14_CONSTEXPR void Jump() noexcept {
    constexpr std::uint64_t JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
                                      0xa9582618e03fc9aa, 0x39abdc4529b1661c};
    std::uint64_t s0 = 0;
    std::uint64_t s1 = 0;
    std::uint64_t s2 = 0;
    std::uint64_t s3 = 0;
    for (auto i : JUMP)
      for (auto b = 0; b < 64; b++) {
        if (i & std::uint64_t{1} << b) {
          s0 ^= state_[0];
          s1 ^= state_[1];
          s2 ^= state_[2];
          s3 ^= state_[3];
        }
        Next();
      }

    state_[0] = s0;
    state_[1] = s1;
    state_[2] = s2;
    state_[3] = s3;
  }
};

}  // namespace internal

class VectorXoshiro {
 private:
  using StateType = Vec<decltype(CappedTag<std::uint64_t, 8>{})>;
 public:
  explicit VectorXoshiro(const std::uint64_t seed)
      : state_{}, intTag_({}), floatTag_({}), lanes_(Lanes(intTag_)) {
    internal::Xoshiro xoshiro{seed};
    auto stateArray = MakeUniqueAlignedArray<std::uint64_t>(StateSize());
    for (auto i = 0UL; i < lanes_; ++i) {
      const auto state = xoshiro.GetState();
      for (auto j = 0UL; j < internal::Xoshiro::StateSize(); ++j) {
        const auto index = lanes_ * j + i;
        stateArray[index] = state[j];
      }
      xoshiro.Jump();
    }

    for (auto i = 0UL; i < internal::Xoshiro::StateSize(); ++i) {
      state_[i] = Load(intTag_, &stateArray[i * lanes_]);
    }
  }

  StateType operator()() { return Next(); }

  std::uint64_t StateSize() const noexcept {
    return lanes_ * internal::Xoshiro::StateSize();
  }

  std::vector<std::uint64_t> GetState() const {
    auto stateArray = MakeUniqueAlignedArray<std::uint64_t>(StateSize());
    for (auto i = 0UL; i < internal::Xoshiro::StateSize(); ++i) {
      Store(state_[i], intTag_, stateArray.get() + i * lanes_);
    }
    std::vector<std::uint64_t> state{StateSize()};
    auto index = 0UL;
    for (auto i = 0UL; i < lanes_; ++i) {
      for (auto j = 0UL; j < internal::Xoshiro::StateSize(); ++j) {
        state[index++] = stateArray[lanes_ * j + i];
      }
    }
    return state;
  }

  Vec<decltype(CappedTag<double, 8>{})> Uniform() noexcept {
    const auto bits = ShiftRight<11>(Next());
    const auto real = ConvertTo(floatTag_, bits);
    return Mul(real, MUL_VALUE);
  }

 private:

  static constexpr CappedTag<double, 8> fl{};

  StateType state_[4];
  const ScalableTag<std::uint64_t> intTag_;
  const ScalableTag<double> floatTag_;

  const std::size_t lanes_;
  const Vec<decltype(fl)> MUL_VALUE = Set(fl, internal::MUL_CONST);

  StateType Next() noexcept {
    const auto result =
        Add(RotateRight<41>(Add(state_[0], state_[3])), state_[0]);
    const auto t = ShiftLeft<17>(state_[1]);
    state_[2] = Xor(state_[2], state_[0]);
    state_[3] = Xor(state_[3], state_[1]);
    state_[1] = Xor(state_[1], state_[2]);
    state_[0] = Xor(state_[0], state_[3]);
    state_[2] = Xor(state_[2], t);
    state_[3] = RotateRight<19>(state_[3]);
    return result;
  }
};

}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_MATH_MATH_INL_H_