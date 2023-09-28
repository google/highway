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

#ifndef HIGHWAY_HWY_CONTRIB_THREAD_POOL_THREAD_POOL_H_
#define HIGHWAY_HWY_CONTRIB_THREAD_POOL_THREAD_POOL_H_

// IWYU pragma: begin_exports
#include <stddef.h>
#include <stdint.h>

#include <mutex>   //NOLINT
#include <thread>  //NOLINT
// IWYU pragma: end_exports

#include <atomic>
#include <condition_variable>  //NOLINT
#include <vector>

#include "hwy/base.h"  // HWY_ASSERT

namespace hwy {

class ThreadPool {
 public:
  // Starts the given number of worker threads and blocks until they are ready.
  // If `num_worker_threads` is zero, all tasks will run on the main thread.
  ThreadPool(size_t num_worker_threads =
                 static_cast<size_t>(std::thread::hardware_concurrency()));
  // Waits for all threads to exit.
  ~ThreadPool();

  // Returns maximum number of main or worker threads that may call RunFunc.
  // Useful for allocating per-thread storage.
  size_t NumThreads() const { return num_threads_; }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator&(const ThreadPool&) = delete;

  // Can pass this as init_closure when no initialization is needed.
  static bool NoInit(size_t /*num_threads*/) { return true; }

  // Runs `init_closure(num_threads)` followed by `run_closure(task, thread)` on
  // worker thread(s) for every `task` in `[begin, end)`. This corresponds to a
  // parallel-for loop over `[begin, end)`. Note that the unit of work should be
  // large enough to amortize the function call overhead. Thus `task` usually
  // identifies a chunk of work that involves a loop.
  //
  // Not thread-safe - no two calls to `Run` may overlap. We check for this in
  // debug builds.
  //
  // Returns false on failure.
  template <class InitClosure, class RunClosure>
  bool Run(uint32_t begin, uint32_t end, const InitClosure& init_closure,
              const RunClosure& run_closure) {
    HWY_ASSERT(begin <= end);
    if (begin == end) return true;

    if (!init_closure(NumThreads())) return false;

    // If there are no workers, run sequentially.
    if (num_worker_threads_ == 0) {
      for (uint32_t task = begin; task < end; ++task) {
        run_closure(task, /*thread=*/0);
      }
      return true;
    }

#if HWY_IS_DEBUG_BUILD
    if (depth_.fetch_add(1, std::memory_order_acq_rel) != 0) {
      return false;  // Must not re-enter.
    }
#endif

    const WorkerCommand worker_command =
        (static_cast<WorkerCommand>(begin) << 32) + end;
    // Ensure the inputs do not result in a reserved command.
    HWY_ASSERT(worker_command != kWorkerWait);
    HWY_ASSERT(worker_command != kWorkerExit);

    run_func_ = reinterpret_cast<RunFunc>(&CallClosure<RunClosure>);
    opaque_ = &run_closure;
    num_reserved_.store(0, std::memory_order_relaxed);

    StartWorkers(worker_command);
    WorkersReadyBarrier();

#if HWY_IS_DEBUG_BUILD
    if (depth_.fetch_add(-1, std::memory_order_acq_rel) != 1) {
      return false;
    }
#endif
    return true;
  }

 private:
  // After construction and between calls to Run, workers are "ready", i.e.
  // waiting on worker_start_cv_. They are "started" by sending a "command"
  // and notifying all worker_start_cv_ waiters. (That is why all workers
  // must be ready/waiting - otherwise, the notification will not reach all of
  // them and the main thread waits in vain for them to report readiness.)
  using WorkerCommand = uint64_t;

  // Special values; all others encode the begin/end parameters. Note that all
  // these are no-op ranges (begin >= end) and therefore never used to encode
  // ranges.
  static constexpr WorkerCommand kWorkerWait = ~1ULL;
  static constexpr WorkerCommand kWorkerExit = ~2ULL;

  // Signature of the (internal) function called from thread(s) for each `value`
  // in the [`begin`, `end`) passed to Run(). Closures (lambdas) do not receive
  // the first argument, which points to the lambda object.
  typedef void (*RunFunc)(const void* opaque, uint32_t value, size_t thread_id);

  // Calls run_closure(task, thread). Signature must match RunFunc.
  template <class Closure>
  static void CallClosure(const void* opaque, uint32_t task, size_t thread) {
    (*reinterpret_cast<const Closure*>(opaque))(task, thread);
  }

  void WorkersReadyBarrier();
  void StartWorkers(WorkerCommand worker_command);

  // Attempts to reserve and perform some work from the global range of tasks,
  // which is encoded within `command`. Returns after all tasks are reserved.
  static void RunRange(ThreadPool* self, WorkerCommand command, size_t thread);

  static void ThreadFunc(ThreadPool* self, size_t thread);

  // Unmodified after ctor, but cannot be const because we call thread::join().
  std::vector<std::thread> threads_;

  // Number of *worker* threads created, equal to threads_.size(). Used to see
  // if all are ready.
  const uint32_t num_worker_threads_;
  // Maximum number of threads that will call run_func_, at least 1.
  const uint32_t num_threads_;

  std::atomic<int> depth_{0};  // detects if Run is re-entered (not supported).

  std::mutex mutex_;  // guards both cv and their variables.
  std::condition_variable workers_ready_cv_;
  uint32_t workers_ready_ = 0;
  std::condition_variable worker_start_cv_;
  WorkerCommand worker_start_command_;

  // Written by main thread, read by workers (after mutex lock/unlock).
  RunFunc run_func_;
  const void* opaque_;

  // Updated by workers; padding avoids false sharing.
  uint8_t padding1[64];
  std::atomic<uint32_t> num_reserved_{0};
  uint8_t padding2[64];
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_THREAD_POOL_THREAD_POOL_H_
