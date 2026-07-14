// Copyright 2026 Google LLC
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

#ifndef HIGHWAY_HWY_CONTRIB_THREAD_POOL_INDEX_RANGE_H_
#define HIGHWAY_HWY_CONTRIB_THREAD_POOL_INDEX_RANGE_H_

// Helpers for splitting many small tasks into ranges.

#include <stddef.h>

#include "hwy/base.h"

namespace hwy {

struct IndexRange {
  IndexRange() = default;
  IndexRange(size_t begin, size_t end) : begin_(begin), end_(end) {
    HWY_DASSERT(begin < end);
  }
  IndexRange(const IndexRange& other) = default;
  IndexRange& operator=(const IndexRange& other) = default;

  size_t Num() const { return end_ - begin_; }
  bool Contains(size_t i) const { return begin_ <= i && i < end_; }

  bool Contains(IndexRange other) const {
    return other.begin_ >= begin_ && other.end_ <= end_;
  }

  // Enable range-based for loops.
  class Iterator {
   public:
    Iterator(size_t i) : i_(i) {}

    Iterator& operator++() {
      ++i_;
      return *this;
    }
    bool operator!=(const Iterator& other) const { return i_ != other.i_; }
    size_t operator*() const { return i_; }
    // Enable using begin() directly as a size_t.
    operator size_t() const { return i_; }

   private:
    size_t i_;
  };
  Iterator begin() const { return Iterator(begin_); }
  Iterator end() const { return Iterator(end_); }

  size_t begin_;
  size_t end_;
};

static inline IndexRange MakeIndexRange(size_t begin, size_t end,
                                        size_t max_size) {
  return IndexRange(begin, HWY_MIN(begin + max_size, end));
}

// Splits `range` into subranges of size `task_size`, except for the last,
// which receives the remainder.
class IndexRangePartition {
 public:
  explicit IndexRangePartition(size_t single_task)
      : range_(0, single_task),
        task_size_(static_cast<uint32_t>(single_task)),
        num_tasks_(1) {}

  IndexRangePartition(const IndexRange& range, const size_t task_size)
      : range_(range), task_size_(static_cast<uint32_t>(task_size)) {
    const uint32_t num = static_cast<uint32_t>(range.Num());
    HWY_DASSERT(task_size_ != 0);
    num_tasks_ = hwy::DivCeil(num, task_size_);
    HWY_DASSERT(num_tasks_ != 0);
    if constexpr (HWY_IS_DEBUG_BUILD) {
      const uint32_t handled = num_tasks_ * task_size_;
      // The last task may extend beyond items, but at most by (task_size_ - 1).
      HWY_DASSERT(num <= handled && handled < num + task_size_);
      (void)handled;
    }
  }

  size_t TaskSize() const { return static_cast<size_t>(task_size_); }
  size_t NumTasks() const { return static_cast<size_t>(num_tasks_); }

  IndexRange Range(size_t task_idx) const {
    HWY_DASSERT(task_idx < NumTasks());
    return MakeIndexRange(range_.begin() + task_idx * TaskSize(), range_.end(),
                          TaskSize());
  }

  template <typename Func>
  void VisitAll(const Func& func) const {
    for (size_t task_idx = 0; task_idx < NumTasks(); ++task_idx) {
      func(Range(task_idx));
    }
  }

  template <typename Func>
  void VisitFirst(const Func& func) const {
    func(Range(0));
  }

  template <typename Func>
  void VisitRemaining(const Func& func) const {
    for (size_t task_idx = 1; task_idx < NumTasks(); ++task_idx) {
      func(Range(task_idx));
    }
  }

 private:
  IndexRange range_;
  uint32_t task_size_;
  uint32_t num_tasks_;
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_THREAD_POOL_INDEX_RANGE_H_
