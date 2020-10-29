// Copyright 2020 Google LLC
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

#include "hwy/aligned_allocator.h"

#include <stddef.h>

#include <new>

#include "gtest/gtest.h"
#include "hwy/base.h"

namespace {

// Sample object that keeps track on an external counter of how many times was
// the explicit constructor and destructor called.
template <size_t N>
class SampleObject {
 public:
  SampleObject() { data_[0] = 'a'; }
  explicit SampleObject(int* counter) : counter_(counter) {
    if (counter) (*counter)++;
    data_[0] = 'b';
  }

  ~SampleObject() {
    if (counter_) (*counter_)--;
  }

  static_assert(N > sizeof(int*), "SampleObject size too small.");
  int* counter_ = nullptr;
  char data_[N - sizeof(int*)];
};

}  // namespace

namespace hwy {

class AlignedAllocatorTest : public testing::Test {};

TEST(AlignedAllocatorTest, FreeNullptr) {
  // Calling free with a nullptr is always ok.
  FreeAlignedBytes(nullptr, nullptr);
}

TEST(AlignedAllocatorTest, AllocDefaultPointers) {
  const size_t kSize = 7777;
  void* ptr = AllocateAlignedBytes(kSize, nullptr);
  ASSERT_NE(nullptr, ptr);
  // Make sure the pointer is actually aligned.
  EXPECT_EQ(0, reinterpret_cast<uintptr_t>(ptr) % kMaxVectorSize);
  char* p = static_cast<char*>(ptr);
  size_t ret = 0;
  for (size_t i = 0; i < kSize; i++) {
    // Performs a computation using p[] to prevent it being optimized away.
    p[i] = i;
    if (i) ret += p[i] * p[i - 1];
  }
  EXPECT_NE(0, ret);
  FreeAlignedBytes(ptr, nullptr);
}

TEST(AlignedAllocatorTest, EmptyAlignedUniquePtr) {
  AlignedUniquePtr<SampleObject<32>> ptr(nullptr, AlignedDeleter(nullptr));
  AlignedUniquePtr<SampleObject<32>[]> arr(nullptr, AlignedDeleter(nullptr));
}

TEST(AlignedAllocatorTest, EmptyAlignedFreeUniquePtr) {
  AlignedFreeUniquePtr<SampleObject<32>> ptr(nullptr, AlignedFreer(nullptr));
  AlignedFreeUniquePtr<SampleObject<32>[]> arr(nullptr, AlignedFreer(nullptr));
}

TEST(AlignedAllocatorTest, MakeUniqueAlignedDefaultConstructor) {
  {
    auto ptr = MakeUniqueAligned<SampleObject<24>>();
    // Default constructor sets the data_[0] to 'a'.
    EXPECT_EQ('a', ptr->data_[0]);
    EXPECT_EQ(nullptr, ptr->counter_);
  }
}

TEST(AlignedAllocatorTest, MakeUniqueAligned) {
  int counter = 0;
  {
    // Creates the object, initializes it with the explicit constructor and
    // returns an unique_ptr to it.
    auto ptr = MakeUniqueAligned<SampleObject<24>>(&counter);
    EXPECT_EQ(1, counter);
    // Custom constructor sets the data_[0] to 'b'.
    EXPECT_EQ('b', ptr->data_[0]);
  }
  EXPECT_EQ(0, counter);
}

TEST(AlignedAllocatorTest, MakeUniqueAlignedArray) {
  int counter = 0;
  {
    // Creates the array of objects and initializes them with the explicit
    // constructor.
    auto arr = MakeUniqueAlignedArray<SampleObject<24>>(7, &counter);
    EXPECT_EQ(7, counter);
    for (size_t i = 0; i < 7; i++) {
      // Custom constructor sets the data_[0] to 'b'.
      EXPECT_EQ('b', arr[i].data_[0]) << "Where i = " << i;
    }
  }
  EXPECT_EQ(0, counter);
}

TEST(AlignedAllocatorTest, AllocSingleInt) {
  auto ptr = AllocateAligned<uint32_t>(1);
  ASSERT_NE(nullptr, ptr.get());
  EXPECT_EQ(0, reinterpret_cast<uintptr_t>(ptr.get()) % kMaxVectorSize);
  // Force delete of the unique_ptr now to check that it doesn't crash.
  ptr.reset(nullptr);
  EXPECT_EQ(nullptr, ptr.get());
}

TEST(AlignedAllocatorTest, AllocMultipleInt) {
  const size_t kSize = 7777;
  auto ptr = AllocateAligned<uint32_t>(kSize);
  ASSERT_NE(nullptr, ptr.get());
  EXPECT_EQ(0, reinterpret_cast<uintptr_t>(ptr.get()) % kMaxVectorSize);
  // ptr[i] is actually (*ptr.get())[i] which will use the operator[] of the
  // underlying type chosen by AllocateAligned() for the std::unique_ptr.
  EXPECT_EQ(&(ptr[0]) + 1, &(ptr[1]));

  size_t ret = 0;
  for (size_t i = 0; i < kSize; i++) {
    // Performs a computation using ptr[] to prevent it being optimized away.
    ptr[i] = i;
    if (i) ret += ptr[i] * ptr[i - 1];
  }
  EXPECT_NE(0, ret);
}

TEST(AlignedAllocatorTest, AllocateAlignedObjectWithoutDestructor) {
  int counter = 0;
  {
    // This doesn't call the constructor.
    auto obj = AllocateAligned<SampleObject<24>>(1);
    obj[0].counter_ = &counter;
  }
  // Destroying the unique_ptr shouldn't have called the destructor of the
  // SampleObject<24>.
  EXPECT_EQ(0, counter);
}

}  // namespace hwy
