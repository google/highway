# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

option(HWY_USE_GTEST "Links tests with GoogleTest")
if(HWY_USE_GTEST)
  find_package(GTest)
endif()

include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
   "int main() {
      #if !defined(__EMSCRIPTEN__)
      static_assert(false, \"__EMSCRIPTEN__ is not defined\");
      #endif
      return 0;
    }"
  HWY_EMSCRIPTEN
)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(HWY_SOURCES
  hwy/arch.h
  hwy/arm64_neon.h
  hwy/cache_control.h
  hwy/compiler_specific.h
  hwy/foreach_target.h
  hwy/include_headers.h
  hwy/runtime_dispatch.cc
  hwy/runtime_dispatch.h
  hwy/runtime_targets.h
  hwy/scalar.h
  hwy/shared.h
  hwy/static_targets.h
  hwy/targets.h
  hwy/tests/test_target_util.h
  hwy/tests/test_util.h
  hwy/type_traits.h
  hwy/x86/avx2.h
  hwy/x86/avx512.h
  hwy/x86/sse4.h
  hwy/wasm.h
)

set(HWY_FLAGS
  # Avoid changing binaries based on the current time and date.
  -Wno-builtin-macro-redefined
  -D__DATE__="redacted"
  -D__TIMESTAMP__="redacted"
  -D__TIME__="redacted"

  # Optimizations
  -fmerge-all-constants

  # Warnings
  -Wall
  -Wextra
  -Wformat-security
  -Wno-unused-function
  -Wnon-virtual-dtor
  -Woverloaded-virtual
  -Wvla
)

if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  list(APPEND HWY_FLAGS
    -Wc++2a-extensions
    -Wfloat-overflow-conversion
    -Wfloat-zero-conversion
    -Wfor-loop-analysis
    -Wgnu-redeclared-enum
    -Winfinite-recursion
    -Wself-assign
    -Wstring-conversion
    -Wtautological-overlap-compare
    -Wthread-safety-analysis
    -Wundefined-func-template

    -fno-cxx-exceptions
    -fno-slp-vectorize
    -fno-vectorize

    # Use color in messages
    -Xclang -fdiagnostics-show-option -fcolor-diagnostics
  )
endif()

if (WIN32)
  list(APPEND HWY_FLAGS
    -Wno-c++98-compat-pedantic
    -Wno-cast-align
    -Wno-double-promotion
    -Wno-float-equal
    -Wno-format-nonliteral
    -Wno-global-constructors
    -Wno-language-extension-token
    -Wno-missing-prototypes
    -Wno-shadow
    -Wno-shadow-field-in-constructor
    -Wno-sign-conversion
    -Wno-unused-member-function
    -Wno-unused-template
    -Wno-used-but-marked-unused
    -Wno-zero-as-null-pointer-constant
  )
else()
  list(APPEND HWY_FLAGS
    -fmath-errno
    -fno-exceptions
  )
endif()

if(HWY_USE_GTEST)
  list(APPEND HWY_FLAGS -DHWY_USE_GTEST)
endif()

if(HWY_EMSCRIPTEN)
  list(APPEND HWY_FLAGS -s SIMD=1)
endif()

add_library(hwy STATIC ${HWY_SOURCES})

target_compile_options(hwy PRIVATE ${HWY_FLAGS})
target_include_directories(hwy PUBLIC ${CMAKE_CURRENT_LIST_DIR})

if(HWY_EMSCRIPTEN)
  target_link_libraries(hwy PUBLIC "-s SIMD=1")
endif()

# -------------------------------------------------------- Tests

enable_testing()

set(HWY_TEST_FILES
  hwy/tests/arithmetic_test.cc
  hwy/tests/compare_test.cc
  hwy/tests/convert_test.cc
  hwy/tests/hwy_test.cc
  hwy/tests/logical_test.cc
  hwy/tests/memory_test.cc
  hwy/tests/swizzle_test.cc
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
foreach (TESTFILE IN LISTS HWY_TEST_FILES)
  # The TESTNAME is the name without the extension or directory.
  get_filename_component(TESTNAME ${TESTFILE} NAME_WE)
  add_executable(${TESTNAME} ${TESTFILE})
  target_compile_options(${TESTNAME} PRIVATE ${HWY_FLAGS})

  target_link_libraries(${TESTNAME} hwy)
  # Output test targets in the test directory.
  set_target_properties(${TESTNAME} PROPERTIES PREFIX "tests/")

  if(HWY_USE_GTEST)
    target_link_libraries(${TESTNAME} gtest gtest_main)
  endif()

  # gtest_discover_tests requires CMake 3.10 and compiling with gtest.
  # gtest_add_tests runs cmake every time a test file changes and requires
  # find_package(GTest) or include(GoogleTest). We know there is only a
  # single test case (due to runtime dispatch), so add manually.
  add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
endforeach ()
