# Integer Division by Multiplication

Fast integer division using the Granlund-Montgomery algorithm for SIMD vectors.

## Overview

This contribution provides optimized integer division for cases where the same divisor is used multiple times. Instead of using expensive hardware division instructions (30-100 cycles), it uses multiplication with precomputed reciprocals (3-5 cycles).

Based on: T. Granlund and P. L. Montgomery, ["Division by invariant integers using multiplication"](https://gmplib.org/~tege/divcnst-pldi94.pdf) (PLDI 1994).

## Features

- **All integer types**: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `uint64_t`, `int64_t`
- **Portable**: Uses only Highway's portable vector API
- **Correct**: Handles all edge cases including signed overflow and division by powers of 2

## Install / Include

Just drop the contrib into your Highway tree and include the dispatcher header:
```cpp
#include "hwy/contrib/intdiv/intdiv.h"  // public API + per-target dispatch
```

- **Note** Include intdiv.h (dispatcher). You generally should not include intdiv-inl.h directly unless you are doing something very custom.

## References

1. Granlund, T., & Montgomery, P. L. (1994). Division by invariant integers using multiplication. *PLDI '94*.
2. [NumPy SIMD implementation](https://github.com/numpy/numpy/blob/main/numpy/_core/src/common/simd/intdiv.h)

## License

Apache 2.0 (same as Highway)

## Contributors

- Implementation Ported to Highway by Abhishek Kumar (Fujitsu Limited)
- Algorithm by Torbjörn Granlund and Peter L. Montgomery