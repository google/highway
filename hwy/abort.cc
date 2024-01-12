// Copyright 2019 Google LLC
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: BSD-3-Clause

#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>

#include "hwy/base.h"

namespace hwy {

HWY_DLLEXPORT HWY_NORETURN void HWY_FORMAT(3, 4)
    Abort(const char* file, int line, const char* format, ...) {
  char buf[800];
  va_list args;
  va_start(args, format);
  vsnprintf(buf, sizeof(buf), format, args);
  va_end(args);

  fprintf(stderr, "Abort at %s:%d: %s\n", file, line, buf);

// If compiled with any sanitizer, they can also print a stack trace.
#if HWY_IS_ASAN || HWY_IS_MSAN || HWY_IS_TSAN
  __sanitizer_print_stack_trace();
#endif  // HWY_IS_*
  fflush(stderr);

// Now terminate the program:
#if HWY_ARCH_RVV
  exit(1);  // trap/abort just freeze Spike.
#elif HWY_IS_DEBUG_BUILD && !HWY_COMPILER_MSVC
  // Facilitates breaking into a debugger, but don't use this in non-debug
  // builds because it looks like "illegal instruction", which is misleading.
  __builtin_trap();
#else
  abort();  // Compile error without this due to HWY_NORETURN.
#endif
}
}
