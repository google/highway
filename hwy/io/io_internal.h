// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Internal declarations shared by the `hwy_io` sources; not installed.

#ifndef HIGHWAY_HWY_IO_IO_INTERNAL_H_
#define HIGHWAY_HWY_IO_IO_INTERNAL_H_

#include <stdint.h>

#include "hwy/aligned_allocator.h"
#include "hwy/io/file.h"
#include "hwy/status.h"

namespace hwy {

class FileSystem;

namespace detail {

// Update `IOCounters` (no-op unless `PROFILER_ENABLED`). Defined in file.cc.
void RecordRead(uint64_t bytes);
void RecordWrite(uint64_t bytes);
void RecordSyscall();
void RecordEintr();
void RecordShortRead();

// Returns the `PosixFileSystem` (posix_file.cc), `WindowsFileSystem`
// (windows_file.cc), or, if `HWY_IO_HAVE_OS_FILES` is 0, stub (file_system.cc)
// singleton, which claims all paths and is the tail of the `FileSystemList`.
FileSystem& OSFileSystem();

// For `AdviseMap`, which is only supported for OS mappings (and no-op for
// non-owning `MemoryFile`/`FileView` views). Defined next to `OSFileSystem`.

// Returns the `free_fn()` of mappings returned by OS `File::Map*`, or nullptr
// if `HWY_IO_HAVE_OS_FILES` is 0.
FreePtr OSMapFreeFunc();

// Whether a range passed to `AdviseOSMapRange` ends at the end of its mapping,
// in which case `AccessHint::kDontNeed` may also evict the final partial page
// (no other slice of the mapping can share it).
enum class MapEnd { kInterior, kAtEnd };

// Advises on the non-empty `range` within a mapping whose `free_fn()` is
// `OSMapFreeFunc()`.
Status AdviseOSMapRange(Span<const uint8_t> range, AccessHint hint,
                        MapEnd map_end);

}  // namespace detail
}  // namespace hwy

#endif  // HIGHWAY_HWY_IO_IO_INTERNAL_H_
