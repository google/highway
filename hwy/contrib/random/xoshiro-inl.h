// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause
//
// See the LICENSE file in the project root for the full license text.

// Vector port by Marco Barbone (m.barbone19@imperial.ac.uk) of the xoshiro256++
// generator. The algorithm was originally designed in 2019 by David Blackman
// and Sebastiano Vigna (vigna@acm.org), with the reference implementation
// available at https://prng.di.unimi.it/. Credit for the algorithm itself
// belongs to the original authors.

#if defined(HIGHWAY_HWY_CONTRIB_RANDOM_XOSHIRO_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_RANDOM_XOSHIRO_H_
#undef HIGHWAY_HWY_CONTRIB_RANDOM_XOSHIRO_H_
#else
#define HIGHWAY_HWY_CONTRIB_RANDOM_XOSHIRO_H_
#endif

#include <stddef.h>

#include <array>
#include <cstdint>

#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace random_internal {

constexpr std::uint64_t kJump[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
                                   0xa9582618e03fc9aa, 0x39abdc4529b1661c};

constexpr std::uint64_t kLongJump[] = {0x76e15d3efefdcbbf, 0xc5004e441c522fb3,
                                       0x77710069854ee241, 0x39109bb02acbe635};

class SplitMix64 {
 public:
  constexpr explicit SplitMix64(const std::uint64_t state) noexcept
      : state_(state) {}

  HWY_CXX14_CONSTEXPR std::uint64_t operator()() {
    std::uint64_t z = (state_ += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  }

 private:
  std::uint64_t state_;
};

class ScalarXoshiro {
 public:
  HWY_CXX14_CONSTEXPR explicit ScalarXoshiro(const std::uint64_t seed) noexcept
      : state_{} {
    SplitMix64 splitMix64{seed};
    for (auto& element : state_) {
      element = splitMix64();
    }
  }

  HWY_CXX14_CONSTEXPR explicit ScalarXoshiro(
      const std::uint64_t seed, const std::uint64_t thread_id) noexcept
      : ScalarXoshiro(seed) {
    for (auto i = UINT64_C(0); i < thread_id; ++i) {
      Jump();
    }
  }

  HWY_CXX14_CONSTEXPR std::uint64_t operator()() noexcept { return Next(); }

  HWY_CXX14_CONSTEXPR std::array<std::uint64_t, 4> GetState() const {
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

  // Equivalent to 2^128 calls to Next(), yielding non-overlapping subsequences.
  HWY_CXX14_CONSTEXPR void Jump() noexcept { Jump(kJump); }

  // Equivalent to 2^192 calls to Next(). Each resulting stream can be split
  // further via Jump() for parallel distributed computations.
  HWY_CXX14_CONSTEXPR void LongJump() noexcept { Jump(kLongJump); }

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

  HWY_CXX14_CONSTEXPR void Jump(const std::uint64_t (&jumpArray)[4]) noexcept {
    std::uint64_t s0 = 0;
    std::uint64_t s1 = 0;
    std::uint64_t s2 = 0;
    std::uint64_t s3 = 0;

    for (const std::uint64_t i : jumpArray) {
      for (std::uint_fast8_t b = 0; b < 64; b++) {
        if (i & std::uint64_t{1UL} << b) {
          s0 ^= state_[0];
          s1 ^= state_[1];
          s2 ^= state_[2];
          s3 ^= state_[3];
        }
        Next();
      }
    }

    state_[0] = s0;
    state_[1] = s1;
    state_[2] = s2;
    state_[3] = s3;
  }
};

}  // namespace random_internal

// Parallel xoshiro256++ streams, one per SIMD lane. Thread numbers select
// LongJump-separated streams; adjacent lanes are separated by Jump().
class XoshiroBitGenerator {
 private:
  using VU64 = Vec<ScalableTag<std::uint64_t>>;
  using StateType = AlignedNDArray<std::uint64_t, 2>;

 public:
  explicit XoshiroBitGenerator(const std::uint64_t seed,
                               const std::uint64_t threadNumber = 0)
      : state_{{random_internal::ScalarXoshiro::StateSize(),
                Lanes(ScalableTag<std::uint64_t>{})}},
        streams_(state_.shape().back()) {
    random_internal::ScalarXoshiro xoshiro{seed};
    for (std::uint64_t i = 0; i < threadNumber; ++i) {
      xoshiro.LongJump();
    }

    for (size_t i = 0; i < streams_; ++i) {
      const auto state = xoshiro.GetState();
      for (size_t j = 0; j < random_internal::ScalarXoshiro::StateSize(); ++j) {
        state_[{j}][i] = state[j];
      }
      xoshiro.Jump();
    }
  }

  // Native vectors must be consumed in the same target's scope.
  VU64 operator()() noexcept {
    const ScalableTag<std::uint64_t> tag;
    auto s0 = Load(tag, state_[{0}].data());
    auto s1 = Load(tag, state_[{1}].data());
    auto s2 = Load(tag, state_[{2}].data());
    auto s3 = Load(tag, state_[{3}].data());
    const auto result = Update(s0, s1, s2, s3);
    Store(s0, tag, state_[{0}].data());
    Store(s1, tag, state_[{1}].data());
    Store(s2, tag, state_[{2}].data());
    Store(s3, tag, state_[{3}].data());
    return result;
  }

  // Calls consume(tag, bits, offset, valid_lanes) for each output vector,
  // keeping state in registers for the entire batch. A partial final vector
  // advances every lane; unused lanes are discarded.
  template <class Consumer>
  HWY_INLINE void GenerateBlocks(size_t count, Consumer&& consume) {
    const ScalableTag<std::uint64_t> tag;
    auto s0 = Load(tag, state_[{0}].data());
    auto s1 = Load(tag, state_[{1}].data());
    auto s2 = Load(tag, state_[{2}].data());
    auto s3 = Load(tag, state_[{3}].data());
    const size_t lanes = Lanes(tag);
    size_t i = 0;
    for (; count - i >= lanes; i += lanes) {
      consume(tag, Update(s0, s1, s2, s3), i, lanes);
    }
    if (i < count) {
      consume(tag, Update(s0, s1, s2, s3), i, count - i);
    }
    Store(s0, tag, state_[{0}].data());
    Store(s1, tag, state_[{1}].data());
    Store(s2, tag, state_[{2}].data());
    Store(s3, tag, state_[{3}].data());
  }

  void FillBits(std::uint64_t* out, size_t count) {
    GenerateBlocks(count, StoreBits{out});
  }

  std::uint64_t StateSize() const noexcept {
    return streams_ * random_internal::ScalarXoshiro::StateSize();
  }

  const StateType& GetState() const { return state_; }

 private:
  struct StoreBits {
    std::uint64_t* out;

    HWY_INLINE void operator()(ScalableTag<std::uint64_t> tag, VU64 bits,
                               size_t offset, size_t valid_lanes) const {
      if (valid_lanes == Lanes(tag)) {
        StoreU(bits, tag, out + offset);
      } else {
        StoreN(bits, tag, out + offset, valid_lanes);
      }
    }
  };

  StateType state_;
  const std::uint64_t streams_;

  HWY_INLINE static VU64 Update(VU64& s0, VU64& s1, VU64& s2,
                                VU64& s3) noexcept {
    const auto result = Add(RotateRight<41>(Add(s0, s3)), s0);
    const auto t = ShiftLeft<17>(s1);
    s2 = Xor(s2, s0);
    s3 = Xor(s3, s1);
    s1 = Xor(s1, s2);
    s0 = Xor(s0, s3);
    s2 = Xor(s2, t);
    s3 = RotateRight<19>(s3);
    return result;
  }
};

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_RANDOM_XOSHIRO_H_
