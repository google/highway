// Copyright 2019 Google LLC
// Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: BSD-3-Clause

#include "hwy/abort.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include <atomic>
#include <string>

#include "hwy/base.h"
#include "hwy/status.h"

#if HWY_IS_ASAN || HWY_IS_MSAN || HWY_IS_TSAN
#include "sanitizer/common_interface_defs.h"  // __sanitizer_print_stack_trace
#endif

namespace hwy {

namespace {

std::atomic<WarnFunc>& AtomicWarnFunc() {
  static std::atomic<WarnFunc> func;
  return func;
}

std::atomic<AbortFunc>& AtomicAbortFunc() {
  static std::atomic<AbortFunc> func;
  return func;
}

// Constant-initialized, hence safe to call SetModuleCodeNameFunc from static
// initializers regardless of TU initialization order.
std::atomic<Status::CodeNameFunc> g_module_code_names[Status::kNumModules] = {};

std::string GetBaseName(std::string const& file_name) {
  auto last_slash = file_name.find_last_of("/\\");
  return file_name.substr(last_slash + 1);
}

}  // namespace

// Returning a reference is unfortunately incompatible with `std::atomic`, which
// is required to safely implement `SetWarnFunc`. As a workaround, we store a
// copy here, update it when called, and return a reference to the copy. This
// has the added benefit of protecting the actual pointer from modification.
HWY_DLLEXPORT WarnFunc& GetWarnFunc() {
  static WarnFunc func;
  func = AtomicWarnFunc().load();
  return func;
}

HWY_DLLEXPORT AbortFunc& GetAbortFunc() {
  static AbortFunc func;
  func = AtomicAbortFunc().load();
  return func;
}

HWY_DLLEXPORT WarnFunc SetWarnFunc(WarnFunc func) {
  return AtomicWarnFunc().exchange(func);
}

HWY_DLLEXPORT AbortFunc SetAbortFunc(AbortFunc func) {
  return AtomicAbortFunc().exchange(func);
}

#if !defined(HWY_HEADER_ONLY)
HWY_DLLEXPORT Status::CodeNameFunc Status::SetModuleCodeNameFunc(
    Module module, CodeNameFunc func) {
  const size_t idx = static_cast<size_t>(module);
  if (idx >= kNumModules) return nullptr;
  return g_module_code_names[idx].exchange(func, std::memory_order_acq_rel);
}

HWY_DLLEXPORT const char* Status::CodeName(Code code) {
  const char* general = GeneralCodeName(code);
  if (general != nullptr) return general;
  const Code id = static_cast<Code>(code & kMaxCode);
  if (id >= kFirstModuleCode && id < kFirstAppCode) {
    const size_t idx =
        static_cast<size_t>(id - kFirstModuleCode) / size_t{kCodesPerModule};
    const CodeNameFunc func =
        g_module_code_names[idx].load(std::memory_order_acquire);
    if (func != nullptr) {
      const char* name = func(code);
      if (name != nullptr) return name;
    }
  }
  return "(module/app code)";
}
#endif  // !HWY_HEADER_ONLY

HWY_DLLEXPORT void HWY_FORMAT(3, 4)
    Warn(const char* file, int line, const char* format, ...) {
  char buf[800];
  va_list args;
  va_start(args, format);
  vsnprintf(buf, sizeof(buf), format, args);
  va_end(args);

  WarnFunc handler = AtomicWarnFunc().load();
  if (handler != nullptr) {
    handler(file, line, buf);
  } else {
    fprintf(stderr, "Warn at %s:%d: %s\n", GetBaseName(file).data(), line, buf);
  }
}

HWY_DLLEXPORT HWY_NORETURN void HWY_FORMAT(3, 4)
    Abort(const char* file, int line, const char* format, ...) {
  char buf[800];
  va_list args;
  va_start(args, format);
  vsnprintf(buf, sizeof(buf), format, args);
  va_end(args);

  AbortFunc handler = AtomicAbortFunc().load();
  if (handler != nullptr) {
    handler(file, line, buf);
  } else {
    fprintf(stderr, "Abort at %s:%d: %s\n", GetBaseName(file).data(), line,
            buf);
  }

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

}  // namespace hwy
