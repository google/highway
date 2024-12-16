// Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: BSD-3-Clause

#ifndef HIGHWAY_HWY_ABORT_H_
#define HIGHWAY_HWY_ABORT_H_

#include "hwy/highway_export.h"

namespace hwy {

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

}  // namespace hwy

#endif  // HIGHWAY_HWY_ABORT_H_
