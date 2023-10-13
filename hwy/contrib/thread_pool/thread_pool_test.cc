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

#include <algorithm>
#include <atomic>
#include <vector>

#include "gtest/gtest.h"
#include "hwy/base.h"                  // PopCount
#include "hwy/detect_compiler_arch.h"  // HWY_ARCH_WASM

namespace hwy {
namespace {

// Ensures task parameter is in bounds, every parameter is reached,
// pool can be reused (multiple consecutive Run calls), pool can be destroyed
// (joining with its threads), num_threads=0 works (runs on current thread).
TEST(ThreadPoolTest, TestPool) {
  if (HWY_ARCH_WASM) return;  // WASM threading is unreliable

  for (int num_threads = 0; num_threads <= 15; num_threads += 3) {
    ThreadPool pool(num_threads);
    for (uint32_t num_tasks = 0; num_tasks < 32; ++num_tasks) {
      std::vector<size_t> mementos(num_tasks);
      for (uint32_t begin = 0; begin < 32; ++begin) {
        std::fill(mementos.begin(), mementos.end(), 0);
        EXPECT_TRUE(pool.Run(begin, begin + num_tasks, ThreadPool::NoInit,
                              [&](const uint32_t task, size_t /*thread*/) {
                                // Parameter is in the given range
                                EXPECT_GE(task, begin);
                                EXPECT_LT(task, begin + num_tasks);

                                // Store mementos to be sure we visited each
                                // task.
                                mementos.at(task - begin) = 1000 + task;
                              }));
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

  const size_t kMaxThreads = 8;
  for (size_t num_threads = 1; num_threads <= kMaxThreads; num_threads += 3) {
    ThreadPool pool(num_threads);

    // (Avoid mutex because it may perturb the worker thread scheduling)
    std::atomic<uint64_t> id_bits{0};
    std::atomic<size_t> num_calls{0};

    EXPECT_TRUE(pool.Run(0, static_cast<uint32_t>(num_threads), ThreadPool::NoInit,
                    [&](uint32_t /*task*/, const size_t thread) {
                      num_calls.fetch_add(1, std::memory_order_relaxed);

                      EXPECT_LT(thread, num_threads);
                      uint64_t bits = id_bits.load(std::memory_order_relaxed);
                      while (!id_bits.compare_exchange_weak(
                          bits, bits | (1ULL << thread))) {
                      }
                    }));

    // Correct number of tasks.
    EXPECT_EQ(num_threads, num_calls.load());

    const int num_participants = PopCount(id_bits.load());
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
  int counter = 0;
  int padding[31];
};

TEST(ThreadPoolTest, TestCounter) {
  if (HWY_ARCH_WASM) return;  // WASM threading is unreliable

  const size_t kNumThreads = 12;
  ThreadPool pool(kNumThreads);
  alignas(128) Counter counters[kNumThreads];

  const uint32_t kNumTasks = kNumThreads * 19;
  EXPECT_TRUE(pool.Run(0, kNumTasks, ThreadPool::NoInit,
                        [&counters](const uint32_t task, const size_t thread) {
                          counters[thread].counter += task;
                        }));

  uint32_t expected = 0;
  for (uint32_t i = 0; i < kNumTasks; ++i) {
    expected += i;
  }

  for (size_t i = 1; i < kNumThreads; ++i) {
    counters[0].Assimilate(counters[i]);
  }
  EXPECT_EQ(expected, counters[0].counter);
}

}  // namespace
}  // namespace hwy
