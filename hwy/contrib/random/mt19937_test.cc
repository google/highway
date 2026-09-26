// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause
//
// See the LICENSE file in the project root for the full license text.

#include "hwy/contrib/random/mt19937.h"

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <random>
#include <type_traits>

#include "hwy/contrib/distributions/uniform.h"
#include "hwy/contrib/random/cached.h"
#include "hwy/generator.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util.h"

namespace hwy {
namespace {

static_assert(std::is_same<Mt19937BitGenerator::result_type, uint64_t>::value,
              "The bit generator supplies full-range u64 words");
static_assert((Mt19937BitGenerator::min)() == 0, "");
static_assert((Mt19937BitGenerator::max)() == UINT64_MAX, "");

uint64_t Reference64(std::mt19937& reference) {
  // Sequence the calls explicitly: the first word occupies the high half.
  const uint64_t high = reference();
  const uint64_t low = reference();
  return (high << 32) | low;
}

double ReferenceUniform(std::mt19937& reference) {
  return static_cast<double>(Reference64(reference) >> 11) / 9007199254740992.0;
}

float ReferenceNormalizedUniform(std::mt19937& reference) {
  return static_cast<float>(Reference64(reference) & 0x7fffffu) / 4194304.0f -
         1.0f;
}

struct JumpFixture {
  uint32_t seed;
  size_t consumed;
  uint32_t first[3];
  uint64_t hash;
};

// Native NumPy 2.5.3 MT19937.jumped() outputs. The reference installs
// np.random.RandomState(seed).get_state(), consumes the stated number of u32
// words via random_raw, then calls jumped(). Its returned state and cursor
// are used unchanged; random_raw(2048) supplies the expected words and hash.
const JumpFixture kJumpFixtures[] = {
    {0u, 0, {2212710650u, 596724571u, 758856663u}, 0x272e820b70503178ull},
    {0u, 1, {2063853351u, 2327829647u, 3244087316u}, 0x18cc868cf372d8ffull},
    {0u, 623, {974950615u, 1882781752u, 2427340696u}, 0xde1385274d15cfedull},
    {0u, 624, {1882781752u, 2427340696u, 1345054283u}, 0x994c6d1d31e845b9ull},
    {0u, 625, {1927882788u, 2269862754u, 3143395830u}, 0x039d543153ba2981ull},
    {5489u, 0, {3108938740u, 3962892820u, 1993863073u}, 0x48e17533be3f6ee0ull},
    {5489u, 1, {3401536484u, 1117814353u, 3013880179u}, 0x0a5024ef14c5b50aull},
    {5489u, 623, {658245976u, 409006828u, 2331739336u}, 0x5e4051efdedb5536ull},
    {5489u,
     624,
     {1297186950u, 2930575927u, 3015810866u},
     0xeaab1fe0b4ec0726ull},
    {5489u, 625, {1546350492u, 641712047u, 2783241761u}, 0x99676b05f71d0eedull},
    {1234u, 0, {291412350u, 2578274023u, 284900656u}, 0xa111785572cba824ull},
    {1234u, 1, {3304637831u, 2950013922u, 1564569354u}, 0x845521039d3ea10aull},
    {1234u,
     623,
     {3487782573u, 2420717291u, 4119246277u},
     0x96bb52c29b6a8c6aull},
    {1234u,
     624,
     {2420717291u, 4119246277u, 3052165451u},
     0xc29962aaa316fea0ull},
    {1234u, 625, {903444240u, 960753328u, 766210415u}, 0xdd5381af7096ba7aull},
    {4294967295u,
     0,
     {4161786104u, 1463427248u, 3506687944u},
     0x34a4909abbb485cfull},
    {4294967295u,
     1,
     {1086656750u, 1448293912u, 2067496107u},
     0x433294747eacc289ull},
    {4294967295u,
     623,
     {2395862465u, 1671212794u, 801735890u},
     0xa37ad261c3a81ca2ull},
    {4294967295u,
     624,
     {917363856u, 194972205u, 2142209737u},
     0x42938bfa173739d8ull},
    {4294967295u,
     625,
     {1039543781u, 1145027094u, 838329373u},
     0x311e8a0e39a8f0edull},
};

const JumpFixture kDoubleJumpFixtures[] = {
    {5489u, 0, {4281334838u, 3818379282u, 3230872612u}, 0x932c46d0f309a85aull},
    {1234u,
     625,
     {2209741415u, 2869401764u, 1000457273u},
     0x6a5d918839406eedull},
};

struct JumpCursorFixture {
  JumpFixture sequence;
  size_t after_jump;
};

// Consume up to just before, exactly at, or just after the returned cursor
// reaches 624. The fixtures preserve NumPy's subsequent refill behavior.
const JumpCursorFixture kJumpCursorFixtures[] = {
    {{5489u, 0, {315505831u, 1810632255u, 2136672454u}, 0xc94f236d78cf6dacull},
     34},
    {{5489u, 0, {1810632255u, 2136672454u, 3728764613u}, 0x6b29ca515569a2f7ull},
     35},
    {{5489u, 0, {2136672454u, 3728764613u, 3363741805u}, 0xbe4c45f7a6a666aeull},
     36},
    {{1234u,
      625,
      {2100029774u, 2280538367u, 3922545062u},
      0x1fd6a777f44be7b0ull},
     33},
    {{1234u, 625, {2280538367u, 3922545062u, 9938410u}, 0x8f98d9863b5a6e4eull},
     34},
    {{1234u, 625, {3922545062u, 9938410u, 1642384753u}, 0x1b0062c4d3368ac2ull},
     35},
};

// The same cursor boundaries, followed by a second native jumped() call.
const JumpCursorFixture kInterleavedJumpFixtures[] = {
    {{5489u, 0, {3438550468u, 828842467u, 4064825162u}, 0xb2309e2f219f0f05ull},
     34},
    {{5489u, 0, {1118705204u, 2495583484u, 1382999742u}, 0xb429e0e349270f8cull},
     35},
    {{5489u, 0, {88862220u, 3266829746u, 2371261401u}, 0x4e03a3e42bd55598ull},
     36},
    {{1234u,
      625,
      {2925325576u, 2385252233u, 2854733297u},
      0x27c5a29df8b18729ull},
     33},
    {{1234u,
      625,
      {1478431234u, 3607627962u, 329353143u},
      0xb089e7c41fbe7053ull},
     34},
    {{1234u, 625, {1063540106u, 518098843u, 460237456u}, 0xa3c865f2dfc1e282ull},
     35},
};

void Consume32(Mt19937BitGenerator& bits, size_t count) {
  for (size_t i = 0; i < count; ++i) (void)bits.Next32();
}

void CheckJumpSequence(Mt19937BitGenerator& bits, const JumpFixture& fixture) {
  // FNV-1a over 2048 u32 outputs, each serialized as four little-endian bytes.
  // This checks several complete state blocks without embedding huge tables.
  uint64_t hash = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < 2048; ++i) {
    const uint32_t word = bits.Next32();
    if (i < 3 && word != fixture.first[i]) {
      HWY_ABORT("Jump seed %u consumed %zu word %zu: %u != %u",
                static_cast<unsigned>(fixture.seed), fixture.consumed, i,
                static_cast<unsigned>(word),
                static_cast<unsigned>(fixture.first[i]));
    }
    for (size_t shift = 0; shift < 32; shift += 8) {
      hash ^= (word >> shift) & 0xffu;
      hash *= 0x100000001b3ull;
    }
  }
  if (hash != fixture.hash) {
    HWY_ABORT("Jump seed %u consumed %zu hash: %llx != %llx",
              static_cast<unsigned>(fixture.seed), fixture.consumed,
              static_cast<unsigned long long>(hash),
              static_cast<unsigned long long>(fixture.hash));
  }
}

TEST(Mt19937Test, JumpMatchesNumPy) {
  for (const JumpFixture& fixture : kJumpFixtures) {
    Mt19937BitGenerator bits(fixture.seed);
    Consume32(bits, fixture.consumed);
    bits.Jump();
    CheckJumpSequence(bits, fixture);
  }
}

TEST(Mt19937Test, RepeatedJumpMatchesNumPy) {
  for (const JumpFixture& fixture : kDoubleJumpFixtures) {
    Mt19937BitGenerator bits(fixture.seed);
    Consume32(bits, fixture.consumed);
    bits.Jump();
    bits.Jump();
    CheckJumpSequence(bits, fixture);
  }
}

TEST(Mt19937Test, JumpCursorBoundariesMatchNumPy) {
  for (const JumpCursorFixture& fixture : kJumpCursorFixtures) {
    Mt19937BitGenerator bits(fixture.sequence.seed);
    Consume32(bits, fixture.sequence.consumed);
    bits.Jump();
    Consume32(bits, fixture.after_jump);
    CheckJumpSequence(bits, fixture.sequence);
  }
}

TEST(Mt19937Test, InterleavedJumpsMatchNumPy) {
  for (const JumpCursorFixture& fixture : kInterleavedJumpFixtures) {
    Mt19937BitGenerator bits(fixture.sequence.seed);
    Consume32(bits, fixture.sequence.consumed);
    bits.Jump();
    Consume32(bits, fixture.after_jump);
    bits.Jump();
    CheckJumpSequence(bits, fixture.sequence);
  }
}

TEST(Mt19937Test, JumpedPreservesSourceAndReturnsIndependentCopy) {
  Mt19937BitGenerator source(1234);
  // A mixed path consumes exactly 625 u32 words, crossing a block boundary.
  (void)source.Next32();
  (void)source();
  uint64_t out[311];
  source.FillBits(out, 311);
  source.FillBits(nullptr, 0);

  const Mt19937BitGenerator& view = source;
  auto jumped = view.Jumped();
  std::mt19937 reference(1234);
  reference.discard(625);
  // Advancing the source must neither have been affected by Jumped nor affect
  // the independent result, whose sequence is checked against the oracle.
  for (size_t i = 0; i < 1024; ++i) {
    HWY_ASSERT(source.Next32() == reference());
  }
  const JumpFixture fixture = {
      1234u, 625, {903444240u, 960753328u, 766210415u}, 0xdd5381af7096ba7aull};
  CheckJumpSequence(jumped, fixture);
}

TEST(Mt19937Test, JumpedStreamSharesCursorAcrossConsumptionMethods) {
  Mt19937BitGenerator bits(5489);
  Consume32(bits, 623);
  bits.Jump();
  auto reference = bits;
  for (size_t i = 0; i < 1000; ++i) {
    HWY_ASSERT(bits.Next32() == reference.Next32());
    const uint64_t high = reference.Next32();
    const uint64_t low = reference.Next32();
    HWY_ASSERT(bits() == ((high << 32) | low));
    uint64_t out[3];
    bits.FillBits(out, 3);
    for (uint64_t value : out) {
      const uint64_t expected_high = reference.Next32();
      const uint64_t expected_low = reference.Next32();
      HWY_ASSERT(value == ((expected_high << 32) | expected_low));
    }
  }
  HWY_ASSERT(bits.Next32() == reference.Next32());
}

TEST(Mt19937Test, KnownAnswers) {
  // Default-seed outputs also verified with the standard-library reference.
  const uint32_t expected[] = {
      3499211612u, 581869302u,  3890346734u, 3586334585u, 545404204u,
      4161255391u, 3922919429u, 949333985u,  2715962298u, 1323567403u};
  Mt19937BitGenerator default_seed;
  Mt19937BitGenerator explicit_seed(5489u);
  for (uint32_t word : expected) {
    HWY_ASSERT(default_seed.Next32() == word);
    HWY_ASSERT(explicit_seed.Next32() == word);
  }
  for (size_t i = 10; i < 9999; ++i) {
    (void)default_seed.Next32();
  }
  // The C++ standard specifies this 10000th default-seed output.
  HWY_ASSERT(default_seed.Next32() == 4123659995u);

  Mt19937BitGenerator paired;
  for (size_t i = 0; i < 10; i += 2) {
    const uint64_t bits = (uint64_t{expected[i]} << 32) | expected[i + 1];
    HWY_ASSERT(paired() == bits);
  }
}

TEST(Mt19937Test, MatchesStandardAcrossTwists) {
  const uint32_t seeds[] = {0, 1, 5489, 0x12345678u, UINT32_MAX};
  for (uint32_t seed : seeds) {
    Mt19937BitGenerator bits(seed);
    std::mt19937 reference(seed);
    for (size_t i = 0; i < 10001; ++i) {
      HWY_ASSERT(bits.Next32() == reference());
    }
  }
}

TEST(Mt19937Test, MixedConsumptionSharesCursor) {
  Mt19937BitGenerator bits(123);
  std::mt19937 reference(123);
  // The first 64-bit draw straddles a 624-word twist boundary.
  for (size_t i = 0; i < 623; ++i) {
    HWY_ASSERT(bits.Next32() == reference());
  }
  HWY_ASSERT(bits() == Reference64(reference));

  for (size_t i = 0; i < 1000; ++i) {
    HWY_ASSERT(bits.Next32() == reference());
    HWY_ASSERT(bits() == Reference64(reference));
    uint64_t out[3];
    bits.FillBits(out, 3);
    for (uint64_t value : out) {
      HWY_ASSERT(value == Reference64(reference));
    }
  }
  HWY_ASSERT(bits.Next32() == reference());
}

TEST(Mt19937Test, FillBoundsAndContinuation) {
  const size_t counts[] = {0, 1, 3, 311, 312, 313, 624, 625};
  const uint64_t canary = 0xabcdef1234567890ull;
  for (size_t count : counts) {
    Mt19937BitGenerator bits(456);
    std::mt19937 reference(456);
    alignas(64) std::array<uint64_t, 627> storage;
    storage.fill(canary);
    // Valid u64 alignment, but deliberately not aligned to a SIMD vector.
    uint64_t* const out = storage.data() + 1;
    bits.FillBits(nullptr, 0);
    bits.FillBits(out, count);
    HWY_ASSERT(storage.front() == canary);
    HWY_ASSERT(storage[count + 1] == canary);
    for (size_t i = 0; i < count; ++i) {
      HWY_ASSERT(out[i] == Reference64(reference));
    }
    HWY_ASSERT(bits.Next32() == reference());
    HWY_ASSERT(bits() == Reference64(reference));
  }
}

TEST(Mt19937Test, CopiesCurrentStateAcrossTwists) {
  const size_t offsets[] = {0, 1, 623, 624, 625, 1247};
  for (size_t offset : offsets) {
    Mt19937BitGenerator bits(789);
    std::mt19937 reference(789);
    for (size_t i = 0; i < offset; ++i) {
      HWY_ASSERT(bits.Next32() == reference());
    }
    Mt19937BitGenerator copy(bits);
    Mt19937BitGenerator assigned(0);
    assigned = bits;
    for (size_t i = 0; i < 626; ++i) {
      const uint32_t expected = static_cast<uint32_t>(reference());
      HWY_ASSERT(bits.Next32() == expected);
      HWY_ASSERT(copy.Next32() == expected);
      HWY_ASSERT(assigned.Next32() == expected);
    }
  }
}

TEST(Mt19937Test, ComposesWithSharedDistributions) {
  Generator<Mt19937BitGenerator> generator{Mt19937BitGenerator(1234)};
  std::mt19937 reference(1234);
  HWY_ASSERT(generator.Sample(Uniform()) == ReferenceUniform(reference));
  HWY_ASSERT(generator.Sample(NormalizedUniform()) ==
             ReferenceNormalizedUniform(reference));

  generator.Fill(Uniform(), static_cast<double*>(nullptr), 0);
  generator.Fill(NormalizedUniform(), static_cast<float*>(nullptr), 0);
  std::array<double, 317> uniform;
  generator.Fill(Uniform(), uniform.data(), uniform.size());
  for (double sample : uniform) {
    HWY_ASSERT(sample == ReferenceUniform(reference));
  }
  std::array<float, 319> normalized;
  generator.Fill(NormalizedUniform(), normalized.data(), normalized.size());
  for (float sample : normalized) {
    HWY_ASSERT(sample == ReferenceNormalizedUniform(reference));
  }
  HWY_ASSERT(generator.GetBitGenerator().Next32() == reference());
  HWY_ASSERT(generator() == Reference64(reference));
}

template <size_t kCacheSize>
void CheckBufferedConsumption() {
  BufferedBitGenerator<Mt19937BitGenerator, kCacheSize> buffered{
      Mt19937BitGenerator(987)};
  std::mt19937 reference(987);
  for (size_t i = 0; i < 3; ++i) {
    HWY_ASSERT(buffered() == Reference64(reference));
  }
  // A copy also preserves partially consumed cache entries.
  auto copy = buffered;
  buffered.FillBits(nullptr, 0);
  for (size_t i = 0; i < 1000; ++i) {
    const uint64_t expected = Reference64(reference);
    HWY_ASSERT(buffered() == expected);
    HWY_ASSERT(copy() == expected);
  }
  uint64_t out[7];
  buffered.FillBits(out, 7);
  for (uint64_t value : out) {
    HWY_ASSERT(value == Reference64(reference));
    HWY_ASSERT(copy() == value);
  }
  HWY_ASSERT(buffered() == Reference64(reference));
}

TEST(Mt19937Test, BufferedConsumptionPreservesSequence) {
  CheckBufferedConsumption<1>();
  CheckBufferedConsumption<4>();
  CheckBufferedConsumption<8>();
}

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
