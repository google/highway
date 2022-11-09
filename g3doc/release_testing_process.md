## Release testing process

We run the following before a release:

### Windows x86

```
run_tests.bat
```

### Linux x86

#### Clang, GCC, ARM cross compile

```
./run_tests.sh
```

Manual test of WASM and WASM_EMU256 targets.

#### JPEG XL clang (debug, asan, msan)

```
for VER in 9 10 11 12 13; do
  rm -rf build_debug$VER && CC=clang-$VER CXX=clang++-$VER BUILD_DIR=build_debug$VER SKIP_TEST=1 ./ci.sh debug && ./ci.sh test -R PassesTest && rm -rf build_debug$VER
  rm -rf build_asan$VER  && CC=clang-$VER CXX=clang++-$VER BUILD_DIR=build_asan$VER  ./ci.sh asan  && rm -rf build_asan$VER
  rm -rf build_msan$VER  && CC=clang-$VER CXX=clang++-$VER BUILD_DIR=build_msan$VER  ./ci.sh msan  && rm -rf build_msan$VER
done
```

#### JPEG XL tests

```
git -C third_party/highway pull -r origin master
git diff
vi deps.sh
git commit -a -m"Highway test"
git push git@github.com:$USER/libjxl.git HEAD:main --force
```
