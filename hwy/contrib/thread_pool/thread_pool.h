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

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"  // HWY_ASSERT

namespace hwy {

// ThreadPool runs func(i) for i in [begin, end). This holds the sub-ranges
// initially assigned to each thread.
class TaskRanges {
  // To prevent false sharing, each thread's state is in a separate cache line.
  static constexpr size_t kU64PerThread = HWY_ALIGNMENT / sizeof(uint64_t);

 public:
  explicit TaskRanges(size_t num_worker_threads) {
    if (num_worker_threads != 0) {
      // Dynamic allocation because this can be several KiB.
      u64_ = hwy::MakeUniqueAlignedArray<std::atomic<uint64_t>>(
          num_worker_threads * kU64PerThread);
    }
  }

  void Assign(uint64_t begin, uint64_t end, size_t num_worker_threads) {
    const size_t num_tasks = end - begin;
    // Only called if we have tasks and workers.
    HWY_DASSERT(num_tasks != 0 && num_worker_threads != 0);

    // Assigning all remainders to the last thread causes imbalance. We instead
    // give one more to each thread whose index is less.
    const size_t remainder = num_tasks % num_worker_threads;
    const size_t min_tasks = num_tasks / num_worker_threads;

    uint64_t task = begin;
    for (size_t thread = 0; thread < num_worker_threads; ++thread) {
      const size_t my_size = min_tasks + (thread < remainder);
      GetTask(thread).store(task, std::memory_order_relaxed);
      task += my_size;
      GetEnd(thread).store(task, std::memory_order_relaxed);
    }
    HWY_DASSERT(task == end);
  }

  // Returns the atomic variables holding thread's next/end task.
  HWY_INLINE std::atomic<uint64_t>& GetTask(size_t thread) {
    return u64_[thread * kU64PerThread];
  }
  HWY_INLINE std::atomic<uint64_t>& GetEnd(size_t thread) {
    return u64_[thread * kU64PerThread + 1];
  }

 private:
  hwy::AlignedUniquePtr<std::atomic<uint64_t>[]> u64_;
};

class ThreadPool {
 public:
  // Starts the given number of worker threads and blocks until they are ready.
  // If `num_worker_threads` is zero, all tasks will run on the main thread.
  // `num_worker_threads` = 1 will still create a worker thread, which only
  // makes sense for measuring the overhead vs. zero threads.
  HWY_CONTRIB_DLLEXPORT ThreadPool(
      size_t num_worker_threads =
          static_cast<size_t>(std::thread::hardware_concurrency()));
  // Waits for all threads to exit.
  HWY_CONTRIB_DLLEXPORT ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator&(const ThreadPool&) = delete;

  // Returns the number of per-thread storage objects that should be allocated.
  // The largest `thread` index is at most this minus one.
  size_t NumThreads() const { return num_threads_; }

  // Runs `closure(task, thread)` on worker thread(s) for every `task` in
  // `[begin, end)` - this is a parallel-for loop over `[begin, end)`. Note
  // that the unit of work should be large enough to amortize the function call
  // overhead, but small enough that each worker processes a few tasks. Thus
  // `task` is usually a chunk of work that involves a loop.
  //
  // Not thread-safe - no two calls to `Run` may overlap. We check for this in
  // debug builds.
  template <class Closure>
  void Run(uint64_t begin, uint64_t end, const Closure& closure) {
    HWY_DASSERT(begin <= end);
    if (HWY_UNLIKELY(begin == end)) return;

    // If there are no workers, run sequentially.
    if (HWY_UNLIKELY(num_worker_threads_ == 0)) {
      for (uint64_t task = begin; task < end; ++task) {
        closure(task, /*thread=*/0);
      }
      return;
    }

    HWY_DASSERT(depth_.fetch_add(1, std::memory_order_acq_rel) == 0);

    ranges_.Assign(begin, end, num_worker_threads_);

    run_func_ = reinterpret_cast<RunFunc>(&CallClosure<Closure>);
    opaque_ = &closure;
    begin_ = begin;
    end_ = end;
    StartWorkers(kWorkerStart);
    WorkersReadyBarrier();

    HWY_DASSERT(depth_.fetch_add(-1, std::memory_order_acq_rel) == 1);
  }

  // Can pass this as init_closure when no initialization is needed.
  // DEPRECATED, better to call the Run() overload without the init_closure arg.
  static bool NoInit(size_t /*num_threads*/) { return true; }

  // DEPRECATED prior interface with 32-bit tasks and first calling
  // `init_closure(num_threads)`. Instead, perform any init before this, calling
  // NumThreads() for an upper bound on the thread indices, then call the
  // other overload.
  template <class InitClosure, class RunClosure>
  bool Run(uint64_t begin, uint64_t end, const InitClosure& init_closure,
           const RunClosure& run_closure) {
    if (!init_closure(NumThreads())) return false;
    Run(begin, end, run_closure);
    return true;
  }

 private:
  // After construction and between calls to Run, workers are "ready", i.e.
  // waiting on worker_start_cv_. They are "started" by sending a "command"
  // and notifying all worker_start_cv_ waiters. (That is why all workers
  // must be ready/waiting - otherwise, the notification will not reach all of
  // them and the main thread waits in vain for them to report readiness.)
  using WorkerCommand = uint64_t;
  static constexpr WorkerCommand kWorkerWait = 0;
  static constexpr WorkerCommand kWorkerExit = 1;
  static constexpr WorkerCommand kWorkerStart = 2;

  // Signature of the (internal) function called from thread(s) for each `value`
  // in the [`begin`, `end`) passed to Run(). Closures (lambdas) do not receive
  // the first argument, which points to the lambda object.
  typedef void (*RunFunc)(const void* opaque, uint64_t task, size_t thread_id);

  // Calls closure(task, thread). Signature must match RunFunc.
  template <class Closure>
  static void CallClosure(const void* opaque, uint64_t task, size_t thread) {
    (*reinterpret_cast<const Closure*>(opaque))(task, thread);
  }

  HWY_CONTRIB_DLLEXPORT void WorkersReadyBarrier();
  HWY_CONTRIB_DLLEXPORT void StartWorkers(WorkerCommand worker_command);
  HWY_CONTRIB_DLLEXPORT void DoWork(size_t thread);

  static void ThreadFunc(ThreadPool* self, size_t thread);

  // Unmodified after ctor, but cannot be const because we call thread::join().
  std::vector<std::thread> threads_;

  // Number of *worker* threads created, equal to threads_.size(). Used to see
  // if all are ready.
  const size_t num_worker_threads_;
  // Returned by NumThreads(). num_worker_threads_ might be zero; this is that
  // value rounded up to 1 so callers can always allocate this many TLS objects.
  const size_t num_threads_;

  const size_t prime_;  // for PermutationNext
  TaskRanges ranges_;

#if HWY_IS_DEBUG_BUILD
  std::atomic<int> depth_{0};  // detects if Run is re-entered (not supported).
#endif

  std::mutex mutex_;  // guards both cv and their variables.
  std::condition_variable workers_ready_cv_;
  size_t workers_ready_ = 0;
  std::condition_variable worker_start_cv_;

  // Written by main thread, read by workers (after mutex lock/unlock).
  RunFunc run_func_;
  const void* opaque_;
  uint64_t begin_;
  uint64_t end_;
  WorkerCommand worker_start_command_;
};

// For initialization of PermutationNext, also used in tests.
bool CoprimeNonzero(uint64_t a, uint64_t b);
uint64_t FindCoprime(uint64_t size);

// Returns the next item of a permutation of [0, size) starting at `current`,
// using an LCG-like (prime * current + offset) % size.
// See https://lemire.me/blog/2017/09/18/.
HWY_INLINE uint64_t PermutationNext(uint64_t current, uint64_t size,
                                    uint64_t prime) {
  HWY_DASSERT(current < size);
  HWY_DASSERT(size != 0);
  HWY_DASSERT(0 != prime && (prime < size || (prime == 1 && size == 1)));
  const uint64_t next = current + prime;
  // Avoid expensive modulo by noting that prime < size, so simply subtract.
  return (next < size) ? next : next - size;
}

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_THREAD_POOL_THREAD_POOL_H_
