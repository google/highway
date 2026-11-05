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

#include "hwy/io/path.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <string>
#include <utility>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

namespace hwy {
namespace {

bool IsSeparator(char c) { return c == '/' || (HWY_OS_WIN != 0 && c == '\\'); }

// Length of the prefix of `pathname` that `Dirname` never strips: leading
// separators, preceded on Windows by an optional drive letter and colon.
size_t RootLength(Span<const char> pathname) {
  size_t pos = 0;
  HWY_IF_CONSTEXPR(HWY_OS_WIN) {
    const char lower =
        static_cast<char>(pathname.empty() ? 0 : (pathname[0] | 0x20));
    if (pathname.size() >= 2 && pathname[1] == ':' && lower >= 'a' &&
        lower <= 'z') {
      pos = 2;
    }
  }
  while (pos < pathname.size() && IsSeparator(pathname[pos])) ++pos;
  return pos;
}

// Returns the offset of the filename within `pathname`.
size_t FilenameBegin(Span<const char> pathname) {
  const size_t root = RootLength(pathname);
  size_t begin = pathname.size();
  while (begin > root && !IsSeparator(pathname[begin - 1])) --begin;
  return begin;
}

// Returns the length of the extension of `pathname`, including its '.'.
size_t ExtensionLength(Span<const char> pathname) {
  const size_t first_of_filename = FilenameBegin(pathname);
  size_t first = first_of_filename;  // Leading dots are part of the name.
  while (first < pathname.size() && pathname[first] == '.') ++first;
  for (size_t pos = pathname.size(); pos > first; --pos) {
    if (pathname[pos - 1] == '.') return pathname.size() - (pos - 1);
  }
  return 0;
}

// Shared by both overloads of `Join`.
Path Joined(Span<const char> base, const char* relative, size_t len) {
  std::string joined;
  joined.reserve(base.size() + 1 + len);
  joined.assign(base.data(), base.size());
  if (!joined.empty() && !IsSeparator(joined.back())) joined += '/';
  joined.append(relative, len);
  return Path(std::move(joined));
}

}  // namespace

std::string Path::Shortened() const {
  constexpr size_t kMaxLen = 48;
  constexpr size_t kCutPoint = kMaxLen / 2 - 5;
  if (path.size() > kMaxLen) {
    // Step to UTF-8 codepoint boundaries (avoid continuation bytes 10xxxxxx).
    size_t prefix_end = kCutPoint;
    while (prefix_end > 0 &&
           (static_cast<uint8_t>(path[prefix_end]) & 0xC0) == 0x80) {
      --prefix_end;
    }
    size_t suffix_begin = path.size() - kCutPoint;
    while (suffix_begin < path.size() &&
           (static_cast<uint8_t>(path[suffix_begin]) & 0xC0) == 0x80) {
      ++suffix_begin;
    }
    return path.substr(0, prefix_end) + " ... " + path.substr(suffix_begin);
  }
  if (path.empty()) return "[no path specified]";
  return path;
}

Path Path::Join(const std::string& relative) const {
  return Joined(path, relative.data(), relative.size());
}

Path Path::Join(const char* relative) const {
  return relative == nullptr ? Joined(path, "", 0)
                             : Joined(path, relative, strlen(relative));
}

Path Path::Dirname() const {
  const size_t root = RootLength(path);
  size_t end = path.size();
  while (end > root && !IsSeparator(path[end - 1])) --end;
  while (end > root && IsSeparator(path[end - 1])) --end;
  return Path(path.substr(0, end));
}

std::string Path::Filename() const { return path.substr(FilenameBegin(path)); }

std::string Path::Extension() const {
  return path.substr(path.size() - ExtensionLength(path));
}

std::string Path::Stem() const {
  const size_t begin = FilenameBegin(path);
  const size_t len = ExtensionLength(path);
  return path.substr(begin, path.size() - begin - len);
}

Path Path::WithExtension(const char* extension) const {
  const size_t len = ExtensionLength(path);
  std::string replaced = path.substr(0, path.size() - len);
  if (extension != nullptr) replaced += extension;
  return Path(std::move(replaced));
}

}  // namespace hwy
