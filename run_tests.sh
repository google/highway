#!/bin/bash

# Switch to directory of this script
MYDIR=$(dirname $(realpath "$0"))
cd "${MYDIR}"

# Exit if anything fails
set -e

echo RELEASE
rm -rf build
mkdir build
cd build
cmake .. -DHWY_WARNINGS_ARE_ERRORS:BOOL=ON
make -j
ctest -j
cd ..
rm -rf build

echo DEBUG GCC
rm -rf build_dbg
mkdir build_dbg
cd build_dbg
CXX=g++ CC=gcc cmake .. -DHWY_WARNINGS_ARE_ERRORS:BOOL=ON -DCMAKE_BUILD_TYPE=Debug
make -j
ctest -j
cd ..
rm -rf build_dbg

echo 32-bit GCC
rm -rf build_32
mkdir build_32
cd build_32
CFLAGS=-m32 CXXFLAGS=-m32 LDFLAGS=-m32 CXX=g++ CC=gcc cmake .. -DHWY_WARNINGS_ARE_ERRORS:BOOL=ON
make -j
ctest -j
cd ..
rm -rf build_32

echo Success
