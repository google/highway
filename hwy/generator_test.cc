// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause
//
// See the LICENSE file in the project root for the full license text.

#include "hwy/generator.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <utility>

#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util.h"

namespace hwy {
namespace {

// No SIMD or contrib dependency is needed to supply either side of Generator.
class Counter {
 public:
  explicit Counter(uint64_t start) : next_(new uint64_t(start)) {}
  Counter(Counter&&) = default;
  uint64_t operator()() { return (*next_)++; }
  uint64_t Next() const { return *next_; }

 private:
  std::unique_ptr<uint64_t> next_;
};

struct Offset {
  explicit Offset(uint64_t value) : offset(value), calls(0) {}

  uint64_t offset;
  size_t calls;

  template <class Bits>
  uint64_t operator()(Bits& bits) {
    ++calls;
    return bits() + offset;
  }

  template <class Bits>
  void Fill(Bits& bits, uint64_t* out, size_t count) {
    for (size_t i = 0; i < count; ++i) out[i] = (*this)(bits);
  }
};

TEST(GeneratorTest, ComposesMoveOnlyBackendAndStatefulDistribution) {
  Generator<Counter> generator{Counter(10)};
  Offset distribution{100};
  HWY_ASSERT(generator.Sample(distribution) == 110);
  HWY_ASSERT(distribution.calls == 1);

  uint64_t out[3] = {};
  generator.Fill(distribution, out, 3);
  HWY_ASSERT(out[0] == 111);
  HWY_ASSERT(out[1] == 112);
  HWY_ASSERT(out[2] == 113);
  HWY_ASSERT(distribution.calls == 4);

  generator.Fill(distribution, static_cast<uint64_t*>(nullptr), 0);
  HWY_ASSERT(distribution.calls == 4);
  const Generator<Counter>& view = generator;
  HWY_ASSERT(view.GetBitGenerator().Next() == 14);
  HWY_ASSERT(generator() == 14);
  HWY_ASSERT(generator.GetBitGenerator()() == 15);
}

TEST(GeneratorTest, MovesTheCurrentStream) {
  Generator<Counter> generator{Counter(7)};
  HWY_ASSERT(generator() == 7);
  Generator<Counter> moved(std::move(generator));
  HWY_ASSERT(moved.Sample(Offset{20}) == 28);
  HWY_ASSERT(moved() == 9);
}

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
