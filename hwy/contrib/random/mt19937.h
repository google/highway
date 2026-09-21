// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause
//
// See the LICENSE file in the project root for the full license text.

#ifndef HIGHWAY_HWY_CONTRIB_RANDOM_MT19937_H_
#define HIGHWAY_HWY_CONTRIB_RANDOM_MT19937_H_

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <limits>

#include "hwy/base.h"
#include "hwy/contrib/random/mt19937_jump.h"

namespace hwy {

// Scalar MT19937 with the single-integer initialization of std::mt19937.
// Next32() produces the standard 32-bit sequence. operator() combines two
// successive words, first word high, to satisfy the full-range u64 contract
// used by distributions and BufferedBitGenerator. This is not MT19937-64.
// Copying preserves both the state and the position within the current block.
class Mt19937BitGenerator {
 public:
  using result_type = uint64_t;

  static constexpr result_type(min)() { return 0; }
  static constexpr result_type(max)() {
    return (std::numeric_limits<result_type>::max)();
  }

  explicit Mt19937BitGenerator(uint32_t seed = 5489u) noexcept
      : index_(kStateSize) {
    state_[0] = seed;
    for (size_t i = 1; i < kStateSize; ++i) {
      const uint32_t previous = state_[i - 1];
      state_[i] = 1812433253u * (previous ^ (previous >> 30)) +
                  static_cast<uint32_t>(i);
    }
  }

  uint32_t Next32() noexcept {
    if (index_ == kStateSize) Twist();
    uint32_t bits = state_[index_++];
    bits ^= bits >> 11;
    bits ^= (bits << 7) & 0x9d2c5680u;
    bits ^= (bits << 15) & 0xefc60000u;
    return bits ^ (bits >> 18);
  }

  result_type operator()() noexcept {
    const uint64_t high = Next32();
    const uint64_t low = Next32();
    return (high << 32) | low;
  }

  // Shares the cursor with Next32() and operator(); no words are discarded.
  // A zero count does not access out or advance the state.
  void FillBits(uint64_t* out, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) out[i] = (*this)();
  }

  // Matches NumPy 2.5.3 MT19937.jumped() for the same state and cursor, using
  // its 2^128-step polynomial. This includes NumPy's block-boundary behavior;
  // jumping and consuming outputs do not generally commute.
  HWY_NOINLINE void Jump() noexcept {
    // NumPy resets an exhausted cursor without refilling the block.
    const JumpState source{state_, index_ == kStateSize ? 0 : index_};

    // Horner's method over GF(2), with cursor-aligned state addition.
    JumpState result = source;
    int degree = 19936;
    while (!JumpCoefficient(degree)) --degree;
    for (--degree; degree >= 0; --degree) {
      AdvanceJumpState(result);
      if (JumpCoefficient(degree)) AddJumpState(result, source);
    }

    // Keep NumPy's returned layout and cursor for subsequent refills.
    state_ = result.words;
    index_ = result.cursor;
  }

  // Returns an independently owned jumped stream, leaving this one unchanged.
  Mt19937BitGenerator Jumped() const noexcept {
    auto jumped = *this;
    jumped.Jump();
    return jumped;
  }

 private:
  static constexpr size_t kStateSize = 624;
  static constexpr size_t kShiftSize = 397;
  using State = std::array<uint32_t, kStateSize>;

  struct JumpState {
    State words;
    size_t cursor;
  };

  static bool JumpCoefficient(int degree) noexcept {
    return ((random_internal::kMt19937JumpPolynomial[degree / 32] >>
             (degree % 32)) &
            1u) != 0;
  }

  static void AdvanceJumpState(JumpState& state) noexcept {
    const size_t pos = state.cursor;
    const size_t next = pos + 1 == kStateSize ? 0 : pos + 1;
    const size_t tap = pos < kStateSize - kShiftSize
                           ? pos + kShiftSize
                           : pos - (kStateSize - kShiftSize);
    state.words[pos] =
        state.words[tap] ^ TwistWord(state.words[pos], state.words[next]);
    state.cursor = next;
  }

  static void AddJumpState(JumpState& target,
                           const JumpState& source) noexcept {
    size_t target_pos = target.cursor;
    size_t source_pos = source.cursor;
    size_t remaining = kStateSize;
    // Split at either cursor's wrap point to keep each XOR range contiguous.
    while (remaining != 0) {
      size_t count = kStateSize - target_pos;
      if (count > kStateSize - source_pos) count = kStateSize - source_pos;
      if (count > remaining) count = remaining;
      for (size_t i = 0; i < count; ++i) {
        target.words[target_pos + i] ^= source.words[source_pos + i];
      }
      remaining -= count;
      target_pos += count;
      source_pos += count;
      if (target_pos == kStateSize) target_pos = 0;
      if (source_pos == kStateSize) source_pos = 0;
    }
  }

  static uint32_t TwistWord(uint32_t upper, uint32_t lower) noexcept {
    const uint32_t joined = (upper & 0x80000000u) | (lower & 0x7fffffffu);
    return (joined >> 1) ^ ((joined & 1u) ? 0x9908b0dfu : 0u);
  }

  // Keep the infrequent state refill out of Next32 so its small hot path can
  // inline into paired draws and distribution fill loops.
  HWY_NOINLINE void Twist() noexcept {
    // The recurrence first reads the old suffix, then the updated prefix.
    // Separate ranges avoid a modulo operation for every state word.
    size_t i = 0;
    for (; i < kStateSize - kShiftSize; ++i) {
      state_[i] = state_[i + kShiftSize] ^ TwistWord(state_[i], state_[i + 1]);
    }
    for (; i + 1 < kStateSize; ++i) {
      state_[i] = state_[i - (kStateSize - kShiftSize)] ^
                  TwistWord(state_[i], state_[i + 1]);
    }
    state_[kStateSize - 1] =
        state_[kShiftSize - 1] ^ TwistWord(state_[kStateSize - 1], state_[0]);
    index_ = 0;
  }

  State state_;
  size_t index_;
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_RANDOM_MT19937_H_
