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

#ifndef HIGHWAY_HWY_IO_FILE_UTIL_H_
#define HIGHWAY_HWY_IO_FILE_UTIL_H_

// Whole-file convenience functions.

#include <stdint.h>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/highway_export.h"
#include "hwy/io/file.h"
#include "hwy/io/path.h"
#include "hwy/status.h"

namespace hwy {

// For files whose `Size()` is 0: returns `Status()` if `file` is actually
// empty, or `Status::kUnimplemented` if it nevertheless has contents, e.g.
// pseudo-files such as /proc/self/status that do not report their size, or the
// error from the 1-byte read at offset 0 used to check this (e.g.
// `IOCode::kWrongMode` for write-only files). For `StreamFile`s, requires
// `Position() == 0` (otherwise returns `Status::kArgument`), does not change
// the position on success, but consumes a byte if there was data.
HWY_IO_DLLEXPORT Status VerifyEmpty(const File& file);

// Reads an entire file into `out` and returns `Status(out.size())` (for an
// empty file, `out` is empty and `Status(0)` is returned), or resets `out` and
// returns an error. If `Size()` is 0, calls `VerifyEmpty` rather than
// silently dropping the contents of pseudo-files such as /proc/self/status.
// Files whose contents are shorter than `Size()` (e.g. sysfs, which reports
// 4096) return `Status::kTruncated`.
HWY_IO_DLLEXPORT Status ReadFileToAligned(const Path& pathname,
                                          AlignedPtr<uint8_t>& out);

// Creates or truncates `pathname` (`FileMode::kWriteTruncate`), writes all of
// `bytes`, and closes the file. Returns `Status(bytes.size())` if both writing
// and `Close()` succeeded, or the first non-ok status from opening, writing, or
// `Close()` (surfacing deferred flush errors on network/distributed filesystems
// that `~File()` would ignore). Not named `WriteFile` to avoid colliding with
// the Win32 function.
HWY_IO_DLLEXPORT Status WriteBytesToFile(const Path& pathname,
                                         Span<const uint8_t> bytes);

}  // namespace hwy

#endif  // HIGHWAY_HWY_IO_FILE_UTIL_H_
