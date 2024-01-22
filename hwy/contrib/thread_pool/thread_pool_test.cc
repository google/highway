// Copyright 2023 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Modified from BSD-licensed code
// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// See https://github.com/libjxl/libjxl/blob/main/LICENSE.

#include "hwy/contrib/thread_pool/thread_pool.h"

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <vector>

#include "gtest/gtest.h"
#include "hwy/base.h"                  // PopCount
#include "hwy/detect_compiler_arch.h"  // HWY_ARCH_WASM
#include "hwy/tests/test_util-inl.h"   // AdjustedReps

namespace hwy {
namespace {

using HWY_NAMESPACE::AdjustedReps;

TEST(ThreadPoolTest, TestCoprime) {
  // 1 is coprime with anything
  for (size_t i = 1; i < 500; ++i) {
    HWY_ASSERT(CoprimeNonzero(1, i));
    HWY_ASSERT(CoprimeNonzero(i, 1));
  }

  // Powers of two >= 2 are not coprime
  for (size_t i = 1; i < 20; ++i) {
    for (size_t j = 1; j < 20; ++j) {
      HWY_ASSERT(!CoprimeNonzero(1ULL << i, 1ULL << j));
    }
  }

  // 2^x and 2^x +/- 1 are coprime
  for (size_t i = 1; i < 60; ++i) {
    const uint64_t pow2 = 1ULL << i;
    HWY_ASSERT(CoprimeNonzero(pow2, pow2 + 1));
    HWY_ASSERT(CoprimeNonzero(pow2, pow2 - 1));
    HWY_ASSERT(CoprimeNonzero(pow2 + 1, pow2));
    HWY_ASSERT(CoprimeNonzero(pow2 - 1, pow2));
  }

  // Random number x * random y (both >= 2) is not co-prime with x nor y.
  RandomState rng;
  for (size_t i = 1; i < 5000; ++i) {
    const uint64_t x = static_cast<uint64_t>(Random32(&rng)) + 2;
    const uint64_t y = static_cast<uint64_t>(Random32(&rng)) + 2;
    HWY_ASSERT(!CoprimeNonzero(x * y, x));
    HWY_ASSERT(!CoprimeNonzero(x * y, y));
    HWY_ASSERT(!CoprimeNonzero(x, x * y));
    HWY_ASSERT(!CoprimeNonzero(y, x * y));
  }

  // Primes are all coprime (list from https://oeis.org/A000040)
  static constexpr uint32_t primes[] = {
      2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,  43,  47,
      53,  59,  61,  67,  71,  73,  79,  83,  89,  97,  101, 103, 107, 109, 113,
      127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
      199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271};
  for (size_t i = 0; i < sizeof(primes) / sizeof(primes[0]); ++i) {
    for (size_t j = i + 1; j < sizeof(primes) / sizeof(primes[0]); ++j) {
      HWY_ASSERT(CoprimeNonzero(primes[i], primes[j]));
      HWY_ASSERT(CoprimeNonzero(primes[j], primes[i]));
    }
  }
}

// Ensures [0, size) is visited exactly once.
void VerifyPermutation(uint64_t size, uint64_t prime, uint64_t offset,
                       uint32_t* visited) {
  for (size_t i = 0; i < size; i++) {
    visited[i] = 0;
  }

  for (size_t i = 0; i < size; i++) {
    ++visited[offset];
    offset = PermutationNext(offset, size, prime);
  }

  for (size_t i = 0; i < size; i++) {
    HWY_ASSERT(visited[i] == 1);
  }
}

// Verifies permutations for all sizes and starting offsets.
TEST(ThreadPoolTest, TestRandomPermutation) {
  // Pre-allocated for efficiency.
  constexpr size_t kMaxSize = 40;
  uint32_t visited[kMaxSize];

  for (size_t size = 1; size < kMaxSize; ++size) {
    const uint64_t prime = FindCoprime(size);

    // For every possible starting point:
    for (size_t offset = 0; offset < size; ++offset) {
      VerifyPermutation(size, prime, offset, visited);
    }
  }
}

// Ensures old code with 32-bit tasks and InitClosure still compiles.
TEST(ThreadPoolTest, TestDeprecated) {
  ThreadPool pool(0);
  pool.Run(1, 10, &ThreadPool::NoInit,
           [&](const uint32_t /*task*/, size_t /*thread*/) {});
}

// Ensures task parameter is in bounds, every parameter is reached,
// pool can be reused (multiple consecutive Run calls), pool can be destroyed
// (joining with its threads), num_threads=0 works (runs on current thread).
TEST(ThreadPoolTest, TestPool) {
  if (HWY_ARCH_WASM) return;  // WASM threading is unreliable

  for (size_t num_threads = 0; num_threads <= 6; num_threads += 3) {
    ThreadPool pool(num_threads);
    for (uint64_t num_tasks = 0; num_tasks < 20; ++num_tasks) {
      std::vector<size_t> mementos(num_tasks);
      for (uint64_t begin = 0; begin < AdjustedReps(32); ++begin) {
        ZeroBytes(mementos.data(), mementos.size() * sizeof(size_t));
        pool.Run(begin, begin + num_tasks,
                 [&](const uint64_t task, size_t /*thread*/) {
                   // Parameter is in the given range
                   EXPECT_GE(task, begin);
                   EXPECT_LT(task, begin + num_tasks);

                   // Store mementos to be sure we visited each
                   // task.
                   mementos.at(task - begin) = 1000 + task;
                 });
        for (size_t task = begin; task < begin + num_tasks; ++task) {
          EXPECT_EQ(1000 + task, mementos.at(task - begin));
        }
      }
    }
  }
}

// Verify "thread" parameter when processing few tasks.
TEST(ThreadPoolTest, TestSmallAssignments) {
  if (HWY_ARCH_WASM) return;  // WASM threading is unreliable

  for (size_t num_threads : {1, 2, 3, 5, 8}) {
    ThreadPool pool(num_threads);

    // (Avoid mutex because it may perturb the worker thread scheduling)
    std::atomic<uint64_t> id_bits{0};
    std::atomic<size_t> num_calls{0};

    pool.Run(0, num_threads, [&](uint64_t /*task*/, const size_t thread) {
      num_calls.fetch_add(1, std::memory_order_relaxed);

      EXPECT_LT(thread, num_threads);
      uint64_t bits = id_bits.load(std::memory_order_relaxed);
      while (!id_bits.compare_exchange_weak(bits, bits | (1ULL << thread))) {
      }
    });

    // Correct number of tasks.
    EXPECT_EQ(num_threads, num_calls.load());

    const size_t num_participants = PopCount(id_bits.load());
    // Can't expect equality because other workers may have woken up too late.
    EXPECT_LE(num_participants, num_threads);
  }
}

struct Counter {
  Counter() {
    // Suppress "unused-field" warning.
    (void)padding;
  }
  void Assimilate(const Counter& victim) { counter += victim.counter; }
  std::atomic<uint64_t> counter{0};
  uint64_t padding[15];
};

TEST(ThreadPoolTest, TestCounter) {
  if (HWY_ARCH_WASM) return;  // WASM threading is unreliable

  const size_t kNumThreads = 12;
  ThreadPool pool(kNumThreads);
  alignas(128) Counter counters[kNumThreads];

  const uint64_t kNumTasks = kNumThreads * 19;
  pool.Run(0, kNumTasks, [&counters](const uint64_t task, const size_t thread) {
    counters[thread].counter.fetch_add(task);
  });

  uint64_t expected = 0;
  for (uint64_t i = 0; i < kNumTasks; ++i) {
    expected += i;
  }

  for (size_t i = 1; i < kNumThreads; ++i) {
    counters[0].Assimilate(counters[i]);
  }
  EXPECT_EQ(expected, counters[0].counter);
}

}  // namespace
}  // namespace hwy
