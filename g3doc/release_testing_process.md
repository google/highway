## Release testing process

We run the following before a release:

### Windows x86 host

```
run_tests.bat
```

### Linux x86 host

Clang, GCC; Arm, PPC cross-compile: `./run_tests.sh`

Manual test of WASM and WASM_EMU256 targets.

Check libjxl build actions at https://github.com/libjxl/libjxl/pull/2269.
