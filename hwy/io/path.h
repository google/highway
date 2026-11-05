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

#ifndef HIGHWAY_HWY_IO_PATH_H_
#define HIGHWAY_HWY_IO_PATH_H_

// Path names and their lexical decomposition. Freestanding: does not require
// OS file support.

#include <stddef.h>

#include <string>
#include <utility>

#include "hwy/base.h"
#include "hwy/highway_export.h"

namespace hwy {

// Terminology used throughout hwy/io, e.g. for "/a/b/c.tar.gz": the pathname is
// the entire string, i.e. path ("/a/b", `Dirname()`) plus filename ("c.tar.gz",
// `Filename()`). Arguments that name a directory (e.g. of `ListDir`) are paths.
//
// `Path` wraps a UTF-8 pathname (or path). Differentiates vs. arbitrary
// strings, supports shortening for display purposes, and lexical (no file
// system access) decomposition. Separators are '/' and, on Windows, also '\\'.
struct Path {
  Path() = default;
  // `pathname` may be nullptr, which is equivalent to an empty `Path`.
  explicit Path(const char* pathname) : path(pathname ? pathname : "") {}
  explicit Path(std::string pathname) : path(std::move(pathname)) {}

  const char* c_str() const { return path.c_str(); }
  bool empty() const { return path.empty(); }

  // Returns at most about 48 characters (without splitting UTF-8 code points),
  // or "[no path specified]" if empty.
  HWY_IO_DLLEXPORT std::string Shortened() const;

  // Returns this path with the relative pathname `relative` (e.g. a
  // `DirEntry::name` from `ListDir`) appended, inserting a '/' unless this is
  // empty or already ends with a separator: Path("a").Join("b") == "a/b".
  // The `const char*` overload avoids a temporary string for literals; nullptr
  // is equivalent to "".
  HWY_IO_DLLEXPORT Path Join(const std::string& relative) const;
  HWY_IO_DLLEXPORT Path Join(const char* relative) const;

  // Returns the path, i.e. the part before the last separator, without
  // trailing separators unless they are the root: "a/b" -> "a", "a" -> "",
  // "/a" -> "/", "a/b/" -> "a/b" (as in Python's `os.path.dirname`).
  HWY_IO_DLLEXPORT Path Dirname() const;

  // Returns the filename, i.e. the part after the last separator:
  // "a/b.txt" -> "b.txt", "a" -> "a", "a/b/" -> "".
  HWY_IO_DLLEXPORT std::string Filename() const;

  // Returns the extension of `Filename()` including its leading '.', or empty
  // if there is none. Leading dots are part of the name: "a/b.tar.gz" -> ".gz",
  // "a/.bashrc" -> "", "a.b/c" -> "", "a." -> ".".
  HWY_IO_DLLEXPORT std::string Extension() const;

  // Returns `Filename()` without `Extension()`, as in
  // `std::filesystem::path::stem`: "a/b.tar.gz" -> "b.tar",
  // "a/.bashrc" -> ".bashrc".
  HWY_IO_DLLEXPORT std::string Stem() const;

  // Returns this pathname with `Extension()` replaced by `extension`, which
  // should be empty or start with '.': ("a/b.txt", ".bin") -> "a/b.bin",
  // ("a/b", ".bin") -> "a/b.bin", ("a/b.txt", "") -> "a/b".
  // `extension` is typically a literal; nullptr is equivalent to "".
  HWY_IO_DLLEXPORT Path WithExtension(const char* extension) const;

  friend bool operator==(const Path& a, const Path& b) {
    return a.path == b.path;
  }
  friend bool operator!=(const Path& a, const Path& b) {
    return a.path != b.path;
  }

  std::string path;  // UTF-8 pathname.
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_IO_PATH_H_
