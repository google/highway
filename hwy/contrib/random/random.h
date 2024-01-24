
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
class SplitMix64 {
 public:
  constexpr SplitMix64(const std::uint64_t state) noexcept : m_state(state) {}

  constexpr std::uint64_t operator()() {
    std::uint64_t z = (m_state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  }

 private:
  std::uint64_t m_state;
};

class Xoshiro {
 public:
  constexpr explicit Xoshiro(const std::uint64_t seed) noexcept : m_state{} {
    SplitMix64 splitMix64{seed};
    for (auto& element : m_state) {
      element = splitMix64();
    }
  }

  constexpr explicit Xoshiro(const std::uint64_t seed,
                             const std::uint64_t thread_id) noexcept
      : Xoshiro(seed) {
    for (auto i = UINT64_C(0); i < thread_id; ++i) {
      Jump();
    }
  }

  constexpr std::uint64_t operator()() noexcept { return Next(); }

  constexpr double Uniform() noexcept {
    return static_cast<double>(Next() >> 11) * 0x1.0p-53;
  }

  constexpr std::array<std::uint64_t, 4> GetState() const {
    return {m_state[0], m_state[1], m_state[2], m_state[3]};
  }

  constexpr void SetState(std::array<std::uint64_t, 4> state) noexcept {
    m_state[0] = state[0];
    m_state[1] = state[1];
    m_state[2] = state[2];
    m_state[3] = state[3];
  }

  static constexpr std::uint64_t StateSize() noexcept { return 4; }

 private:
  std::uint64_t m_state[4];

  static constexpr std::uint64_t Rotl(const std::uint64_t x, int k) noexcept {
    return (x << k) | (x >> (64 - k));
  }

  constexpr std::uint64_t Next() noexcept {
    const std::uint64_t result = Rotl(m_state[0] + m_state[3], 23) + m_state[0];
    const std::uint64_t t = m_state[1] << 17;

    m_state[2] ^= m_state[0];
    m_state[3] ^= m_state[1];
    m_state[1] ^= m_state[2];
    m_state[0] ^= m_state[3];

    m_state[2] ^= t;

    m_state[3] = Rotl(m_state[3], 45);

    return result;
  }

 public:

  /* This is the jump function for the generator. It is equivalent
 to 2^128 calls to next(); it can be used to generate 2^128
 non-overlapping subsequences for parallel computations. */
  constexpr void Jump() noexcept {
    constexpr std::uint64_t JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
                                      0xa9582618e03fc9aa, 0x39abdc4529b1661c};
    std::uint64_t s0 = 0;
    std::uint64_t s1 = 0;
    std::uint64_t s2 = 0;
    std::uint64_t s3 = 0;
    for (auto i : JUMP)
      for (auto b = 0; b < 64; b++) {
        if (i & std::uint64_t{1} << b) {
          s0 ^= m_state[0];
          s1 ^= m_state[1];
          s2 ^= m_state[2];
          s3 ^= m_state[3];
        }
        Next();
      }

    m_state[0] = s0;
    m_state[1] = s1;
    m_state[2] = s2;
    m_state[3] = s3;
  }

};
}  // namespace internal
template <typename T>
class HWY_CONTRIB_DLLEXPORT VectorXoshiro {
 public:
  explicit VectorXoshiro(const std::uint64_t seed) {
    namespace hn = hwy::HWY_NAMESPACE;
    internal::Xoshiro xoshiro{seed};
    const auto lanes = hn::Lanes(hn::DFromV<T>());
    auto stateArray = hwy::MakeUniqueAlignedArray<std::uint64_t>(StateSize());

    for (auto i = 0UL; i < lanes; ++i) {
      const auto state = xoshiro.GetState();
      for (auto j = 0UL; j < internal::Xoshiro::StateSize(); ++j) {
        const auto index = lanes * j + i;
        stateArray[index] = state[j];
      }
      xoshiro.Jump();
    }

    for (auto i = 0UL; i < internal::Xoshiro::StateSize(); ++i) {
      m_state[i] = hn::Load(hn::DFromV<T>(), &stateArray[i * lanes]);
    }
  }

  constexpr auto operator()() { return Next(); }

  static constexpr std::uint64_t StateSize() noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    return hn::Lanes(hn::DFromV<T>()) * internal::Xoshiro::StateSize();
  }

  std::array<std::uint64_t, StateSize()> GetState() const {
    namespace hn = hwy::HWY_NAMESPACE;
    const auto lanes = hn::Lanes(hn::DFromV<T>());
    auto stateArray = hwy::MakeUniqueAlignedArray<std::uint64_t>(StateSize());
    for (auto i = 0UL; i < internal::Xoshiro::StateSize(); ++i) {
      hn::Store(m_state[i], hn::DFromV<T>(), stateArray.get() + i * lanes);
    }
    std::array<std::uint64_t, StateSize()> state;

    auto index = 0UL;
    for (auto i = 0UL; i < lanes; ++i) {
      for (auto j = 0UL; j < internal::Xoshiro::StateSize(); ++j) {
        state[index++] = stateArray[lanes * j + i];
      }
    }
    return state;
  }

  constexpr auto Uniform() noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const auto bits = hn::ShiftRight<11>(Next());
    const auto real = hn::ConvertTo(fl, bits);
    return real * MUL_VALUE;
  }

 private:
  T m_state[4];

  static constexpr hwy::HWY_NAMESPACE::ScalableTag<double> fl{};

  const decltype(hwy::HWY_NAMESPACE::Undefined(fl)) MUL_VALUE =
      hwy::HWY_NAMESPACE::Set(fl, 0x1.0p-53);

  template <int k>
  static constexpr T Rotl(const T x) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    return hn::Or(hn::ShiftLeft<k>(x), hn::ShiftRight<64 - k>(x));
  }

  constexpr T Next() noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const T result =
        hn::Add(Rotl<23>(hn::Add(m_state[0], m_state[3])), m_state[0]);
    const T t = hn::ShiftLeft<17>(m_state[1]);
    //
    m_state[2] = hn::Xor(m_state[2], m_state[0]);
    m_state[3] = hn::Xor(m_state[3], m_state[1]);
    m_state[1] = hn::Xor(m_state[1], m_state[2]);
    m_state[0] = hn::Xor(m_state[0], m_state[3]);
    m_state[2] = hn::Xor(m_state[2], t);
    m_state[3] = Rotl<45>(m_state[3]);
    return result;
  }
};

}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_MATH_MATH_INL_H_