#!/bin/bash

# Switch to directory of this script
MYDIR=$(dirname $(realpath "$0"))
cd "${MYDIR}"

# Exit if anything fails
set -e

#######################################
echo RELEASE
rm -rf build && mkdir build && cd build
cmake .. -DHWY_WARNINGS_ARE_ERRORS:BOOL=ON -DCMAKE_BUILD_TYPE=Release
make -j && ctest -j && cd .. && rm -rf build

#######################################
echo DEBUG Clang 13
rm -rf build_dbg && mkdir build_dbg && cd build_dbg
CXX=clang++-13 CC=clang-13 cmake .. -DHWY_WARNINGS_ARE_ERRORS:BOOL=ON -DCMAKE_BUILD_TYPE=Debug
make -j && ctest -j && cd .. && rm -rf build_dbg

#######################################
echo 32-bit GCC
rm -rf build_32 && mkdir build_32 && cd build_32
CFLAGS=-m32 CXXFLAGS=-m32 LDFLAGS=-m32 CXX=g++ CC=gcc cmake .. -DHWY_WARNINGS_ARE_ERRORS:BOOL=ON -DHWY_CMAKE_SSE2:BOOL=ON  -DCMAKE_BUILD_TYPE=Release
make -j && ctest -j && cd .. && rm -rf build_32

#######################################
for VER in 11 12 13; do
  echo GCC $VER
  rm -rf build_g$VER && mkdir build_g$VER && cd build_g$VER
  CC=gcc-$VER CXX=g++-$VER cmake .. -DHWY_WARNINGS_ARE_ERRORS:BOOL=ON -DCMAKE_BUILD_TYPE=Release
  make -j && make test && cd .. && rm -rf build_g$VER
done

#######################################
echo Armv7 GCC
export QEMU_LD_PREFIX=/usr/arm-linux-gnueabihf
rm -rf build_arm7 && mkdir build_arm7 && cd build_arm7
CC=arm-linux-gnueabihf-gcc-11 CXX=arm-linux-gnueabihf-g++-11 cmake .. -DHWY_CMAKE_ARM7:BOOL=ON -DHWY_WARNINGS_ARE_ERRORS:BOOL=ON -DCMAKE_BUILD_TYPE=Release
make -j && ctest -j && cd .. && rm -rf build_arm7

#######################################
echo Armv8 GCC
export QEMU_LD_PREFIX=/usr/aarch64-linux-gnu
rm -rf build_arm8 && mkdir build_arm8 && cd build_arm8
CC=aarch64-linux-gnu-gcc-11 CXX=aarch64-linux-gnu-g++-11 cmake .. -DHWY_WARNINGS_ARE_ERRORS:BOOL=ON -DCMAKE_BUILD_TYPE=Release
make -j && ctest -j && cd .. && rm -rf build_arm8

#######################################
echo POWER8 GCC
export QEMU_LD_PREFIX=/usr/powerpc64le-linux-gnu
rm -rf build_ppc8 && mkdir build_ppc8 && cd build_ppc8
CC=powerpc64le-linux-gnu-gcc-12 CXX=powerpc64le-linux-gnu-g++-12 cmake .. -DCMAKE_BUILD_TYPE=Release -DHWY_WARNINGS_ARE_ERRORS:BOOL=ON -DCMAKE_CROSSCOMPILING_EMULATOR=/usr/bin/qemu-ppc64le-static -DCMAKE_C_COMPILER_TARGET="powerpc64le-linux-gnu" -DCMAKE_CXX_COMPILER_TARGET="powerpc64le-linux-gnu" -DCMAKE_CROSSCOMPILING=true -DCMAKE_CXX_FLAGS='-mcpu=power9 -mno-power9-vector -mpower8-vector'
clear && make -j && ctest -j && cd .. && rm -rf build_ppc8

#######################################
echo POWER9 clang
export QEMU_LD_PREFIX=/usr/powerpc64le-linux-gnu
rm -rf build_ppc9 && mkdir build_ppc9 && cd build_ppc9
CC=clang-15 CXX=clang++-15 cmake .. -DCMAKE_BUILD_TYPE=Release -DHWY_WARNINGS_ARE_ERRORS:BOOL=ON -DCMAKE_C_COMPILER_TARGET="powerpc64le-linux-gnu" -DCMAKE_CXX_COMPILER_TARGET="powerpc64le-linux-gnu" -DCMAKE_CROSSCOMPILING=true -DCMAKE_CXX_FLAGS='-mcpu=power9'
clear && make -j && ctest -j && cd .. && rm -rf build_ppc9

#######################################
echo POWER9 big endian GCC
export QEMU_LD_PREFIX=/usr/powerpc64-linux-gnu
rm -rf build_ppc9be && mkdir build_ppc9be && cd build_ppc9be
CC=powerpc64-linux-gnu-gcc-11 CXX=powerpc64-linux-gnu-g++-11 cmake .. -DCMAKE_BUILD_TYPE=Release -DHWY_WARNINGS_ARE_ERRORS:BOOL=ON -DCMAKE_CROSSCOMPILING_EMULATOR=/usr/bin/qemu-ppc64 -DCMAKE_C_COMPILER_TARGET="powerpc64-linux-musl" -DCMAKE_CXX_COMPILER_TARGET="powerpc64-linux-musl" -DCMAKE_CROSSCOMPILING=true  -DCMAKE_CXX_FLAGS='-mcpu=power9'
clear && make -j && ctest -j && cd .. && rm -rf build_ppc9be

#######################################
echo POWER10 requires QEMU 7_2 and gcc because clang 15 crashes
export QEMU_LD_PREFIX=/usr/powerpc64le-linux-gnu
rm -rf build_ppc10 && mkdir build_ppc10 && cd build_ppc10
CC=powerpc64le-linux-gnu-gcc-12 CXX=powerpc64le-linux-gnu-g++-12 cmake .. -DCMAKE_BUILD_TYPE=Release -DHWY_WARNINGS_ARE_ERRORS:BOOL=ON -DCMAKE_C_COMPILER_TARGET="powerpc64le-linux-gnu" -DCMAKE_CXX_COMPILER_TARGET="powerpc64le-linux-gnu" -DCMAKE_CROSSCOMPILING=true -DCMAKE_CXX_FLAGS='-mcpu=power10'
clear && make -j && ctest -j && cd .. && rm -rf build_ppc10

#######################################
echo Z14
export QEMU_LD_PREFIX=/usr/s390x-linux-gnu
rm -rf build_z14 && mkdir build_z14 && cd build_z14
CC=s390x-linux-gnu-gcc-11 CXX=s390x-linux-gnu-g++-11 cmake .. -DCMAKE_C_COMPILER_TARGET="s390x-linux-gnu" -DCMAKE_CXX_COMPILER_TARGET="s390x-linux-gnu" -DCMAKE_C_FLAGS='-march=z14 -mzvector' -DCMAKE_CXX_FLAGS='-march=z14 -mzvector -DHWY_DISABLED_TARGETS="(HWY_SCALAR|HWY_EMU128|HWY_Z15)" -DHWY_COMPILE_ONLY_STATIC=1' -DCMAKE_CROSSCOMPILING=true -DCMAKE_CROSSCOMPILING_EMULATOR="/usr/bin/qemu-s390x;-L;/usr/s390x-linux-gnu"
clear && make -j && ctest -j && cd .. && rm -rf build_z14

#######################################
echo Z15
export QEMU_LD_PREFIX=/usr/s390x-linux-gnu
rm -rf build_z15 && mkdir build_z15 && cd build_z15
CC=s390x-linux-gnu-gcc-11 CXX=s390x-linux-gnu-g++-11 cmake .. -DCMAKE_C_COMPILER_TARGET="s390x-linux-gnu" -DCMAKE_CXX_COMPILER_TARGET="s390x-linux-gnu" -DCMAKE_C_FLAGS='-march=z15 -mzvector' -DCMAKE_CXX_FLAGS='-march=z15 -mzvector -DHWY_DISABLED_TARGETS="(HWY_SCALAR|HWY_EMU128|HWY_Z14)" -DHWY_COMPILE_ONLY_STATIC=1' -DCMAKE_CROSSCOMPILING=true -DCMAKE_CROSSCOMPILING_EMULATOR="/usr/bin/qemu-s390x;-cpu;max,vxeh2=on;-L;/usr/s390x-linux-gnu"
clear && make -j && ctest -j && cd .. && rm -rf build_z15


echo Success
