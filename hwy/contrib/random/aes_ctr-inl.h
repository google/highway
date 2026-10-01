// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause
//
// See the LICENSE file in the project root for the full license text.

#if defined(HIGHWAY_HWY_CONTRIB_RANDOM_AES_CTR_H_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_RANDOM_AES_CTR_H_
#undef HIGHWAY_HWY_CONTRIB_RANDOM_AES_CTR_H_
#else
#define HIGHWAY_HWY_CONTRIB_RANDOM_AES_CTR_H_
#endif

#include <stddef.h>
#include <stdint.h>

#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"
#include "hwy/os_rng.h"
#include "hwy/timer.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// Non-cryptographic 64-bit pseudo-random number generator. Supports random or
// deterministic seeding.
//
// Based on 5-round AES-CTR. Supports 2^64 streams, each with period 2^64. This
// is useful for parallel sampling. Each thread can generate the stream for a
// particular task, without caring about prior/subsequent generations.
class alignas(16) AesCtrEngine {
  // "Large-scale randomness study of security margins for 100+ cryptographic
  // functions": at least four.
  // "Parallel Random Numbers: As Easy as 1, 2, 3": four not Crush-resistant.
  static constexpr size_t kRounds = 5;

 public:
  // If `deterministic` is true, uses a fixed seed; otherwise, attempts to
  // grab entropy from the OS.
  explicit AesCtrEngine(bool deterministic) {
    // Pi-based nothing up my sleeve numbers from Randen.
    key_[0] = 0x243F6A8885A308D3ull;
    key_[1] = 0x13198A2E03707344ull;

    if (!deterministic) {  // want random seed
      if (!hwy::Fill16BytesSecure(key_)) {
        HWY_WARN("Failed to fill RNG key with secure random bits");
        // Entropy not available. The test requires that we inject some
        // differences relative to the deterministic seeds.
        key_[0] ^= reinterpret_cast<uint64_t>(this);
        key_[1] ^= hwy::timer::Start();
      }
    }

    // Simple key schedule: swap and add constant (also from Randen).
    for (size_t i = 0; i < kRounds; ++i) {
      key_[2 + 2 * i + 0] = key_[2 * i + 1] + 0xA4093822299F31D0ull;
      key_[2 + 2 * i + 1] = key_[2 * i + 0] + 0x082EFA98EC4E6C89ull;
    }
  }

  // Pure and thread safe; typically called via `RngStream`, which increments
  // `counter`. Throughput is about 100M/s on 3 GHz Skylake. It could be
  // increased 4x via unrolling by the AES latency (4-7 cycles), but because
  // users generally call once at a time, this requires buffering, which is not
  // worth the complexity in this application.
  uint64_t operator()(uint64_t stream, uint64_t counter) const {
#if HWY_TARGET != HWY_SCALAR
    using D = Full128<uint8_t>;  // 128 bits for AES
    using V = Vec<D>;
    const Repartition<uint64_t, D> d64;

    auto LoadKey = [](const uint64_t* ptr) HWY_ATTR -> V {
      return Load(D(), reinterpret_cast<const uint8_t*>(ptr));
    };

    V state = BitCast(D(), Dup128VecFromValues(d64, counter, stream));
    state = Xor(state, LoadKey(key_));  // initial whitening

    static_assert(kRounds == 5 && sizeof(key_) == 12 * sizeof(uint64_t), "");
    state = AESRound(state, LoadKey(key_ + 2));
    state = AESRound(state, LoadKey(key_ + 4));
    state = AESRound(state, LoadKey(key_ + 6));
    state = AESRound(state, LoadKey(key_ + 8));
    // Final round: fine to use another AESRound, including MixColumns.
    state = AESRound(state, LoadKey(key_ + 10));

    // Return lower 64 bits of the u8 vector.
    return GetLane(BitCast(d64, state));
#else
    HWY_DASSERT(0);  // Not supported.
    (void)stream;
    (void)counter;
    return 0;
#endif  // HWY_TARGET != HWY_SCALAR
  }

 private:
  uint64_t key_[2 * (1 + kRounds)];
};

// Flyweight per-thread adapter that maintains the counter. Conforms to C++
// `UniformRandomBitGenerator`. Note that this is not a permutation. It is
// possible but very unlikely that two identical u64 values are returned from
// operator(), because we only return the lower half of the AES-CTR output.
class RngStream {
 public:
  RngStream() = default;  // Allow C arrays with subsequent initialization.

  // Binds to an engine, which holds the seed and must outlive this object.
  // Sets the stream; any other `RngStream` with the same `counter_rng` and
  // `stream` will return the same sequence. This is typically the task ID, so
  // that threads can independently generate values for each task.
  RngStream(const AesCtrEngine& counter_rng, uint64_t stream)
      : engine_(&counter_rng), stream_(stream), counter_(0) {}

  using result_type = uint64_t;
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return ~result_type{0}; }
  result_type operator()() { return (*engine_)(stream_, counter_++); }

  void FillBits(uint64_t* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      out[i] = (*this)();
    }
  }

 private:
  const AesCtrEngine* engine_ = nullptr;
  uint64_t stream_ = 0;  // immutable after ctor
  uint64_t counter_ = 0;
  // Prevent false sharing if used by multiple threads.
  HWY_MEMBER_VAR_MAYBE_UNUSED uint8_t
      padding_[HWY_ALIGNMENT - 16 - sizeof(engine_)];
};

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_RANDOM_AES_CTR_H_
