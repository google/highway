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

#include "hwy/base.h"  // HWY_MAX

namespace hwy {

ThreadPool::ThreadPool(const size_t num_worker_threads)
    : num_worker_threads_(num_worker_threads),
      num_threads_(HWY_MAX(num_worker_threads, 1)) {
  threads_.reserve(num_worker_threads_);

  // Suppress "unused-private-field" warning.
  (void)padding1;
  (void)padding2;

  // Safely handle spurious worker wakeups.
  worker_start_command_ = kWorkerWait;

  for (uint32_t i = 0; i < num_worker_threads_; ++i) {
    threads_.emplace_back(ThreadFunc, this, static_cast<size_t>(i));
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
  while (workers_ready_ != num_worker_threads_) {
    workers_ready_cv_.wait(lock);
  }
  workers_ready_ = 0;

  // Safely handle spurious worker wakeups.
  worker_start_command_ = kWorkerWait;
}

// Precondition: all workers are ready.
void ThreadPool::StartWorkers(const WorkerCommand worker_command) {
  mutex_.lock();
  worker_start_command_ = worker_command;
  // Workers will need this lock, so release it before they wake up.
  mutex_.unlock();
  worker_start_cv_.notify_all();
}

// static
void ThreadPool::RunRange(ThreadPool* self, const WorkerCommand command,
                          const size_t thread) {
  const uint32_t begin = command >> 32;
  const uint32_t end = command & 0xFFFFFFFF;
  const uint32_t num_tasks = end - begin;
  const uint32_t num_worker_threads = self->num_worker_threads_;

  // OpenMP introduced several "schedule" strategies:
  // "single" (static assignment of exactly one chunk per thread): slower.
  // "dynamic" (allocates k tasks at a time): competitive for well-chosen k.
  // "guided" (allocates k tasks, decreases k): computing k = remaining/n
  //   is faster than halving k each iteration. We prefer this strategy
  //   because it avoids user-specified parameters.

  for (;;) {
    uint32_t my_size;  // set below
    if (false) {
      // dynamic
      my_size = HWY_MAX(num_tasks / (num_worker_threads * 4), 1);
    } else {
      // guided
      const uint32_t num_reserved =
          self->num_reserved_.load(std::memory_order_relaxed);
      // It is possible that more tasks are reserved than ready to run.
      const uint32_t num_remaining =
          num_tasks - HWY_MIN(num_reserved, num_tasks);
      my_size = HWY_MAX(num_remaining / (num_worker_threads * 4), 1u);
    }
    const uint32_t my_begin = begin + self->num_reserved_.fetch_add(
                                          my_size, std::memory_order_relaxed);
    const uint32_t my_end = HWY_MIN(my_begin + my_size, begin + num_tasks);
    // Another thread already reserved the last task.
    if (my_begin >= my_end) {
      break;
    }
    for (uint32_t task = my_begin; task < my_end; ++task) {
      self->run_func_(self->opaque_, task, thread);
    }
  }
}

// static
void ThreadPool::ThreadFunc(ThreadPool* self, const size_t thread) {
  // Until kWorkerExit command received:
  for (;;) {
    std::unique_lock<std::mutex> lock(self->mutex_);
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
        RunRange(self, command, thread);
        break;
    }
  }
}

}  // namespace hwy
