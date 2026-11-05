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

// Helpers shared by the hwy/io tests.

#ifndef HIGHWAY_HWY_IO_IO_TEST_UTIL_H_
#define HIGHWAY_HWY_IO_IO_TEST_UTIL_H_

#include <stdlib.h>

#include <string>

#include "hwy/base.h"
#include "hwy/io/file_system.h"
#include "hwy/io/path.h"
#if !defined(HWY_IO_TEST_NO_CONTRIB)
#include "hwy/contrib/thread_pool/topology.h"
#elif HWY_ARCH_WASM
#include <emscripten/threading.h>
#endif

#if !HWY_OS_WIN && HWY_IO_HAVE_OS_FILES
#include <limits.h>  // IOV_MAX
#include <sys/stat.h>
#include <unistd.h>
#endif

// OS file I/O is unreliable in WASM test environments.
#if HWY_IO_HAVE_OS_FILES && !HWY_ARCH_WASM
#define HWY_IO_TEST_OS_FILES 1
#else
#define HWY_IO_TEST_OS_FILES 0
#endif

namespace hwy {

#if defined(HWY_IO_TEST_NO_CONTRIB)
inline bool HaveThreadingSupport() {
#if HWY_ARCH_WASM
  return emscripten_has_threading_support() != 0;
#else
  return true;
#endif
}
#endif

// Returns the pathname of `filename` within the test's temporary directory.
// Appends the process ID (where available) to avoid collisions between
// concurrent test runs that share a temporary directory.
inline Path TempPathname(const char* filename) {
  std::string unique(filename);
#if !HWY_OS_WIN && HWY_IO_TEST_OS_FILES
  unique += "." + std::to_string(static_cast<long long>(getpid()));
#endif
  const char* test_tmpdir = getenv("TEST_TMPDIR");
  if (test_tmpdir != nullptr && test_tmpdir[0] != '\0') {
    return Path(test_tmpdir).Join(unique);
  }
  return TempDir().Join(unique);
}

// Returns whether a regular file exists at `pathname`.
inline bool IsFile(const Path& pathname) {
  FileStat stat;
  return StatFile(pathname, stat).ok() && stat.kind == EntryKind::kFile;
}

}  // namespace hwy

#endif  // HIGHWAY_HWY_IO_IO_TEST_UTIL_H_
