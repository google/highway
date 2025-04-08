// Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: BSD-3-Clause

#ifndef HIGHWAY_HWY_ABORT_H_
#define HIGHWAY_HWY_ABORT_H_

#if defined(HWY_HEADER_ONLY)
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include "hwy/detect_compiler_arch.h"
#endif

#include "hwy/highway_export.h"

#if defined(HWY_HEADER_ONLY)
#if HWY_IS_ASAN || HWY_IS_MSAN || HWY_IS_TSAN
#include "sanitizer/common_interface_defs.h"  // __sanitizer_print_stack_trace
#endif
#endif

namespace hwy {

#if defined(HWY_HEADER_ONLY)
// Enables error-checking of format strings.
#if HWY_HAS_ATTRIBUTE(__format__)
#define HWY_FORMAT(idx_fmt, idx_arg) \
  __attribute__((__format__(__printf__, idx_fmt, idx_arg)))
#else
#define HWY_FORMAT(idx_fmt, idx_arg)
#endif

#ifndef HWY_NORETURN
#if HWY_COMPILER_MSVC
#define HWY_NORETURN __declspec(noreturn)
#else
#define HWY_NORETURN __attribute__((noreturn))
#endif
#endif

HWY_DLLEXPORT inline void HWY_FORMAT(3, 4)
    Warn(const char* file, int line, const char* format, ...) {
  char buf[800];
  va_list args;
  va_start(args, format);
  vsnprintf(buf, sizeof(buf), format, args);
  va_end(args);

  fprintf(stderr, "Warn at %s:%d: %s\n", file, line, buf);
}

HWY_DLLEXPORT HWY_NORETURN inline void HWY_FORMAT(3, 4)
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
#if HWY_ARCH_RISCV
  exit(1);  // trap/abort just freeze Spike.
#elif HWY_IS_DEBUG_BUILD && !HWY_COMPILER_MSVC && !HWY_ARCH_ARM
  // Facilitates breaking into a debugger, but don't use this in non-debug
  // builds because it looks like "illegal instruction", which is misleading.
  // Also does not work on Arm.
  __builtin_trap();
#else
  abort();  // Compile error without this due to HWY_NORETURN.
#endif
}
#else  // !HWY_HEADER_ONLY
// Interfaces for custom Warn/Abort handlers.
typedef void (*WarnFunc)(const char* file, int line, const char* message);

typedef void (*AbortFunc)(const char* file, int line, const char* message);

// Returns current Warn() handler, or nullptr if no handler was yet registered,
// indicating Highway should print to stderr.
// DEPRECATED because this is thread-hostile and prone to misuse (modifying the
// underlying pointer through the reference).
HWY_DLLEXPORT WarnFunc& GetWarnFunc();

// Returns current Abort() handler, or nullptr if no handler was yet registered,
// indicating Highway should print to stderr and abort.
// DEPRECATED because this is thread-hostile and prone to misuse (modifying the
// underlying pointer through the reference).
HWY_DLLEXPORT AbortFunc& GetAbortFunc();

// Sets a new Warn() handler and returns the previous handler, which is nullptr
// if no previous handler was registered, and should otherwise be called from
// the new handler. Thread-safe.
HWY_DLLEXPORT WarnFunc SetWarnFunc(WarnFunc func);

// Sets a new Abort() handler and returns the previous handler, which is nullptr
// if no previous handler was registered, and should otherwise be called from
// the new handler. If all handlers return, then Highway will terminate the app.
// Thread-safe.
HWY_DLLEXPORT AbortFunc SetAbortFunc(AbortFunc func);

// Abort()/Warn() and HWY_ABORT/HWY_WARN are declared in base.h.
#endif  // HWY_HEADER_ONLY

}  // namespace hwy

#endif  // HIGHWAY_HWY_ABORT_H_
