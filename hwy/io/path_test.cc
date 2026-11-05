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

#include "hwy/base.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

namespace hwy {
namespace {

TEST(PathTest, TestPath) {
  const Path empty;
  HWY_ASSERT(empty.empty());
  HWY_ASSERT(empty == Path(nullptr));
  HWY_ASSERT(strcmp("[no path specified]", empty.Shortened().c_str()) == 0);

  const Path short_path("short/file/path.txt");
  HWY_ASSERT(!short_path.empty());
  HWY_ASSERT(short_path != empty);
  const std::string short_str("short/file/path.txt");
  HWY_ASSERT(short_path == Path(short_str));
  // Temporaries are moved into the `Path`.
  HWY_ASSERT(Path(short_str.substr(0, 5)) == Path("short"));
  HWY_ASSERT(Path(std::string("a") + "/" + short_str) ==
             Path("a/short/file/path.txt"));
  HWY_ASSERT(strcmp("short/file/path.txt", short_path.Shortened().c_str()) ==
             0);

  const std::string long_str(120, 'a');
  const Path long_path(long_str);
  const std::string shortened = long_path.Shortened();
  HWY_ASSERT(shortened.size() < long_str.size());

  // Multi-byte UTF-8 characters (3-byte Euro sign \xE2\x82\xAC) around the
  // cut point must not be split in the middle.
  std::string utf8;
  for (size_t i = 0; i < 30; ++i) utf8 += "\xE2\x82\xAC";
  const std::string utf8_short = Path(utf8).Shortened();
  const size_t ellipsis = utf8_short.find(" ... ");
  HWY_ASSERT(ellipsis != std::string::npos);
  HWY_ASSERT_EQ(size_t{0}, ellipsis % 3);
  const size_t suffix_len = utf8_short.size() - (ellipsis + 5);
  HWY_ASSERT_EQ(size_t{0}, suffix_len % 3);

  // Lexical helpers.
  HWY_ASSERT(Path("a").Join("b") == Path("a/b"));
  HWY_ASSERT(Path("a/").Join("b") == Path("a/b"));
  HWY_ASSERT(Path("/").Join("b") == Path("/b"));
  HWY_ASSERT(Path().Join("b") == Path("b"));
  HWY_ASSERT(Path("a").Join("") == Path("a/"));
  HWY_ASSERT(Path("a").Join(nullptr) == Path("a/"));
  const std::string name("c.txt");  // e.g. `DirEntry::name`.
  HWY_ASSERT(Path("a/b").Join(name) == Path("a/b/c.txt"));
  HWY_ASSERT(Path("a/b.txt").WithExtension(nullptr) == Path("a/b"));

  HWY_ASSERT(short_path.Dirname() == Path("short/file"));
  HWY_ASSERT(Path("a").Dirname() == Path());
  HWY_ASSERT(Path("/a").Dirname() == Path("/"));
  HWY_ASSERT(Path("/").Dirname() == Path("/"));
  HWY_ASSERT(Path("a//b").Dirname() == Path("a"));
  HWY_ASSERT(Path("a/b/").Dirname() == Path("a/b"));
  HWY_ASSERT(Path().Dirname() == Path());

  HWY_ASSERT(short_path.Filename() == "path.txt");
  HWY_ASSERT(Path("a").Filename() == "a");
  HWY_ASSERT(Path("/a").Filename() == "a");
  HWY_ASSERT(Path("/").Filename().empty());
  HWY_ASSERT(Path("a/b/").Filename().empty());
  // Dirname and Filename rejoin to the original pathname.
  HWY_ASSERT(short_path.Dirname().Join(short_path.Filename()) == short_path);

  HWY_ASSERT(short_path.Extension() == ".txt");
  HWY_ASSERT(Path("a/b.tar.gz").Extension() == ".gz");
  HWY_ASSERT(Path("a/.bashrc").Extension().empty());
  HWY_ASSERT(Path("a/..").Extension().empty());
  HWY_ASSERT(Path("a.b/c").Extension().empty());
  HWY_ASSERT(Path("a.").Extension() == ".");
  HWY_ASSERT(Path("a/.b.c").Extension() == ".c");

  // Stem is the filename without the extension.
  HWY_ASSERT(short_path.Stem() == "path");
  HWY_ASSERT(Path("a/b.tar.gz").Stem() == "b.tar");
  HWY_ASSERT(Path("a/.bashrc").Stem() == ".bashrc");
  HWY_ASSERT(Path("a.b/c").Stem() == "c");
  HWY_ASSERT(Path("a.").Stem() == "a");
  HWY_ASSERT(Path("a/b/").Stem().empty());
  HWY_ASSERT(short_path.Stem() + short_path.Extension() ==
             short_path.Filename());

  HWY_ASSERT(short_path.WithExtension(".bin") == Path("short/file/path.bin"));
  HWY_ASSERT(Path("a/b").WithExtension(".bin") == Path("a/b.bin"));
  HWY_ASSERT(Path("a/b.txt").WithExtension("") == Path("a/b"));
  HWY_ASSERT(Path("a.b/c").WithExtension(".d") == Path("a.b/c.d"));
#if HWY_OS_WIN
  HWY_ASSERT(Path("C:\\a\\b.txt").Dirname() == Path("C:\\a"));
  HWY_ASSERT(Path("C:\\a").Dirname() == Path("C:\\"));
  HWY_ASSERT(Path("C:a").Filename() == "a");
  HWY_ASSERT(Path("C:\\a\\b.txt").Filename() == "b.txt");
  HWY_ASSERT(Path("C:\\a\\b.txt").Stem() == "b");
  HWY_ASSERT(Path("C:\\a\\").Join("b") == Path("C:\\a\\b"));
#endif
}

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
