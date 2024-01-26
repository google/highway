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
#include <sys/types.h>

#include <atomic>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT

#include "hwy/base.h"  // HWY_MAX
#include "hwy/cache_control.h"

namespace hwy {

ThreadPool::ThreadPool(const size_t num_worker_threads)
    : num_worker_threads_(num_worker_threads),
      num_threads_(HWY_MAX(num_worker_threads, size_t{1})),
      prime_(static_cast<size_t>(FindCoprime(num_worker_threads))),
      ranges_(num_worker_threads) {
  threads_.reserve(num_worker_threads_);

  // Safely handle spurious worker wakeups.
  worker_start_command_ = kWorkerWait;

  for (size_t i = 0; i < num_worker_threads_; ++i) {
    threads_.emplace_back(ThreadFunc, this, i);
  }

  if (num_worker_threads_ != 0) {
    WorkersReadyBarrier();
  }
}

ThreadPool::~ThreadPool() {
  if (num_worker_threads_ != 0) {
    StartWorkers(kWorkerExit);
  }

  for (std::thread& thread : threads_) {
    HWY_ASSERT(thread.joinable());
    thread.join();
  }
}

void ThreadPool::WorkersReadyBarrier() {
  std::unique_lock<std::mutex> lock(mutex_);
  // Typically only a single iteration.
  workers_ready_cv_.wait(
      lock, [this]() { return workers_ready_ == num_worker_threads_; });

  workers_ready_ = 0;

  // Safely handle spurious worker wakeups.
  worker_start_command_ = kWorkerWait;
}

// Precondition: all workers are ready.
void ThreadPool::StartWorkers(const WorkerCommand worker_command) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    worker_start_command_ = worker_command;
    // Workers will need this lock, so release it before they wake up.
  }
  worker_start_cv_.notify_all();
}

void ThreadPool::DoWork(size_t thread) {
  // Special case for <= 1 task per worker - avoid any shared state.
  const size_t num_tasks = static_cast<size_t>(end_ - begin_);
  const size_t num_workers = num_worker_threads_;
  if (num_tasks <= num_workers) {
    if (thread < num_tasks) {
      const uint64_t task = begin_ + thread;
      run_func_(opaque_, task, thread);
    }
    return;
  }

  // For load-balancing, there are two common approaches:
  // - attempt to assign only as much work as a thread will actually use. For
  //   example, OpenMP's 'guided' strategy assigns remaining/(k*num_threads) in
  //   each iteration, which means there are multiple rounds of work allocation
  //   and the worst-case imbalance is bounded. However, we have no opportunity
  //   to remedy the imbalance, and there is contention on the central counter.
  //
  // - work stealing: idle threads steal work from others. This can deal with
  //   imbalances as they arise, but scalability also requires some care. We
  //   minimize true/false sharing by allocating into separate cache lines and
  //   randomizing the order in which threads choose victims to steal from.
  //   Results: across 10K calls Run(), we observe a mean of 5.1 tasks per
  //   thread, and standard deviation 0.67, indicating good load-balance.

  const uint64_t prime = prime_;  // Encourage keeping in register.

  // Scan once over all workers starting with `thread`, in random order, and
  // attempt to perform all their work.
  for (size_t i = 0; i < num_workers; ++i) {
    const uint64_t my_end =
        ranges_.GetEnd(thread).load(std::memory_order_relaxed);
    std::atomic<uint64_t>& my_task = ranges_.GetTask(thread);

    // Until all of thread's work is done:
    for (;;) {
      // On x86 this generates a LOCK prefix, but that is only expensive if
      // there is actually contention, which is unlikely because we shard the
      // counters, threads do not quite proceed in lockstep due to memory
      // traffic, and stealing happens in semi-random order. An additional
      // relaxed-order check does not help.
      uint64_t task = my_task.fetch_add(1, std::memory_order_relaxed);
      if (HWY_LIKELY(task >= my_end)) {
        hwy::Pause();  // Reduce coherency traffic.
        break;
      }
      run_func_(opaque_, task, thread);
    }

    thread = static_cast<size_t>(PermutationNext(thread, num_workers, prime));
  }
}

// static
void ThreadPool::ThreadFunc(ThreadPool* self, const size_t thread) {
  std::unique_lock<std::mutex> lock(self->mutex_);
  // Until kWorkerExit command received:
  for (;;) {
    // Notify main thread that this thread is ready.
    if (++self->workers_ready_ == self->num_threads_) {
      self->workers_ready_cv_.notify_one();
    }
  RESUME_WAIT:
    // Wait for a command.
    self->worker_start_cv_.wait(lock);
    const WorkerCommand command = self->worker_start_command_;
    switch (command) {
      case kWorkerWait:    // spurious wakeup:
        goto RESUME_WAIT;  // lock still held, avoid incrementing ready.
      case kWorkerExit:
        return;  // exits thread
      default:
        lock.unlock();
        self->DoWork(thread);
        lock.lock();
        break;
    }
  }
}

// Returns true if a and b have no common denominator except 1. Based on
// binary GCD. Assumes a and b are nonzero.
bool CoprimeNonzero(uint64_t a, uint64_t b) {
  const size_t trailing_a = Num0BitsBelowLS1Bit_Nonzero64(a);
  const size_t trailing_b = Num0BitsBelowLS1Bit_Nonzero64(b);
  // If both have at least one trailing zero, they are both divisible by 2.
  if (HWY_MIN(trailing_a, trailing_b) != 0) return false;

  // If one of them has a trailing zero, shift it out.
  a >>= trailing_a;
  b >>= trailing_b;

  while (a != b) {
    // Swap such that a >= b.
    const uint64_t tmp_a = a;
    a = HWY_MAX(tmp_a, b);
    b = HWY_MIN(tmp_a, b);

    a -= b;
    a >>= Num0BitsBelowLS1Bit_Nonzero64(a);
  }

  return a == 1;
}

uint64_t FindCoprime(uint64_t size) {
  if (size <= 2) return 1;
  if (size & 1) {
    for (uint64_t x = size / 2; x < size; ++x) {
      if (CoprimeNonzero(x, size)) return x;
    }
  } else {
    // For even `size`, we know it's going to be odd.
    for (uint64_t x = (size / 2) | 1; x < size; x += 2) {
      if (CoprimeNonzero(x, size)) return x;
    }
  }
  HWY_DASSERT(false);  // none found, but should always exist
  return 0;
}

}  // namespace hwy
