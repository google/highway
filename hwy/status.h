// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HIGHWAY_HWY_STATUS_H_
#define HIGHWAY_HWY_STATUS_H_

// Compact (8 byte) status: plain success with a 48-bit size/payload, or an
// info, warning, or error code with an optional 48-bit payload. Similar to
// absl::StatusOr<uint64_t>, but without a message and usable in constexpr
// context.
//
// Note that `ok()` is true ONLY for plain success (`Severity::kOk`), NOT for
// `kInfo` or `kWarning`. Three-way check idiom:
// - `status.ok()`: plain success (`status.size()` is valid).
// - `status.IsError()`: failure.
// - `!status.IsError()`: proceed (plain success, info, or warning), e.g. in a
//   streaming loop:
//     Status s = StreamStep(...);
//     if (s.IsError()) return s;
//     if (s.Is(kNeedInput)) { ... }

#include <stdint.h>

#if defined(HWY_HEADER_ONLY)
#include <atomic>
#endif

#include "hwy/base.h"

#if defined(HWY_HEADER_ONLY)
// Header-only inline definitions have no exporting DLL (`abort.cc` is omitted),
// so avoid `__declspec(dllimport)` on Windows; on ELF/Mach-O, keep default
// visibility even under `-fvisibility-inlines-hidden` so the dynamic linker
// coalesces `detail::ModuleCodeNameTable`'s static table across DSOs.
#if !defined(_WIN32) && (HWY_COMPILER_GCC || HWY_COMPILER_CLANG)
#define HWY_STATUS_DLLEXPORT __attribute__((visibility("default")))
#else
#define HWY_STATUS_DLLEXPORT
#endif
#else
#include "hwy/highway_export.h"
#define HWY_STATUS_DLLEXPORT HWY_DLLEXPORT
#endif

namespace hwy {

class HWY_NODISCARD_CLASS Status {
 public:
  // Layout of `rep_`: [62, 64) 2-bit `Severity`, [48, 62) 14-bit code ID,
  // [0, 48) 48-bit payload. `Code` is the upper 16 bits (`rep_ >> 48`), which
  // combines `Severity` in [14, 16) and the 14-bit ID in [0, 14) so that C++
  // enum auto-increment preserves the severity bits. Invariant: the ID is zero
  // iff the severity is `kOk`.
  enum class Severity : uint8_t {
    kOk = 0,       // Plain success with 48-bit size/payload; code ID is 0.
    kInfo = 1,     // Informational (e.g. "need more input") + code + payload.
    kWarning = 2,  // Non-fatal warning + code + payload.
    kError = 3     // Error + code + payload.
  };

  using Code = uint16_t;
  static constexpr uint64_t kPayloadBits = 48;
  static constexpr uint64_t kIdBits = 14;
  static constexpr uint64_t kSeverityShift = kPayloadBits + kIdBits;  // 62
  static constexpr uint64_t kMaxPayload = (uint64_t{1} << kPayloadBits) - 1;
  static constexpr Code kMaxCode = (Code{1} << kIdBits) - 1;  // 0x3FFF

  // Note: any `enum` with underlying type `uint16_t` (`Status::Code`) satisfies
  // this check; using `: Status::Code` only for status enums is a convention,
  // not a guarantee.
  template <typename Enum>
  static constexpr bool IsStatusEnum() {
    return IsEnum<Enum>() && IsSame<UnderlyingType<Enum>, Code>();
  }

  // Code ID ranges within each severity (the same ranges apply to info,
  // warning and error codes):
  // - [1, kFirstModuleCode): general codes that apply to any module; the names
  //   below are modeled on absl::StatusCode and known to `GeneralCodeName`.
  // - [kFirstModuleCode, kFirstAppCode): Highway modules, each owning
  //   `kCodesPerModule` IDs starting at `ModuleBegin(Module)`. Modules can
  //   register a `CodeNameFunc` via `SetModuleCodeNameFunc` so `CodeName` can
  //   resolve their code names.
  // - [kFirstAppCode, kMaxCode]: applications.
  static constexpr Code kFirstModuleCode = 256;
  static constexpr Code kCodesPerModule = 256;
  static constexpr Code kFirstAppCode = 4096;
  static constexpr Code kNumModules =
      (kFirstAppCode - kFirstModuleCode) / kCodesPerModule;  // 15

  // Registry of Highway modules that define status codes. Append new modules;
  // the numeric values are stable.
  enum class Module : uint8_t {
    kIO = 0,         // hwy/io/file.h (`IOCode`)
    kReserved0 = 1,  // Reserved for a future module.
  };

  // Combines `severity` and a 14-bit `id` into a 16-bit `Code`. Note that
  // enum auto-increment past `kMaxCode` would change the severity; this cannot
  // happen for module or application codes that stay within their range.
  static constexpr Code MakeCode(Severity severity, Code id) {
    return static_cast<Code>((static_cast<Code>(severity) << kIdBits) |
                             (id & kMaxCode));
  }

  // First ID of `module`'s range.
  static constexpr Code ModuleBegin(Module module) {
    return static_cast<Code>(kFirstModuleCode +
                             static_cast<Code>(module) * kCodesPerModule);
  }

  // Defining a module or application enum as `enum class MyCode :
  // hwy::Status::Code` whose first enumerator is one of these enables implicit
  // construction of `Status` with that severity, e.g.
  // `kNeedInput = Status::FirstInfo(Status::kFirstAppCode)`, or
  // `kWrongMode = Status::ModuleError(Status::Module::kIO)`.
  static HWY_CXX14_CONSTEXPR Code FirstInfo(Code base_id) {
    HWY_DASSERT(base_id != 0 && base_id <= kMaxCode);
    return MakeCode(Severity::kInfo, base_id);
  }
  static HWY_CXX14_CONSTEXPR Code FirstWarning(Code base_id) {
    HWY_DASSERT(base_id != 0 && base_id <= kMaxCode);
    return MakeCode(Severity::kWarning, base_id);
  }
  static HWY_CXX14_CONSTEXPR Code FirstError(Code base_id) {
    HWY_DASSERT(base_id != 0 && base_id <= kMaxCode);
    return MakeCode(Severity::kError, base_id);
  }
  static HWY_CXX14_CONSTEXPR Code ModuleInfo(Module module) {
    return FirstInfo(ModuleBegin(module));
  }
  static HWY_CXX14_CONSTEXPR Code ModuleWarning(Module module) {
    return FirstWarning(ModuleBegin(module));
  }
  static HWY_CXX14_CONSTEXPR Code ModuleError(Module module) {
    return FirstError(ModuleBegin(module));
  }

  enum GeneralCode : Code {
    kOk = 0,
    kUnknown = (static_cast<Code>(Severity::kError) << kIdBits) | 1,
    kCancelled,
    kArgument,  // invalid argument
    kTimeout,   // deadline exceeded / timed out
    kNotFound,
    kPermission,     // permission denied, e.g. file system or RPC access
    kResource,       // out of memory, disk space, file descriptors
    kPrecondition,   // failed precondition, e.g. not opened, or in use
    kRange,          // out of range
    kUnimplemented,  // e.g. backend lacks the feature
    kInternal,       // bug, please report
    kUnavailable,    // transient, retry may succeed
    kDataLoss,       // corrupt data
    kTruncated,      // input shorter than expected, e.g. read past EOF
    kOSError         // other OS error; payload is errno or GetLastError
  };

  // Default is plain success with size/payload 0.
  constexpr Status() : rep_(0) {}

  // Plain success with `size` bytes. Constrained to non-bool integers so that
  // booleans and enums without `: Status::Code` do not silently convert to a
  // size. Payloads above 2^48 - 1 (`kMaxPayload`) assert in debug builds and
  // saturate in release builds.
  template <typename T, EnableIf<IsInteger<T>() &&
                                 !IsSame<RemoveCvRef<T>, bool>()>* = nullptr>
  explicit HWY_CXX14_CONSTEXPR Status(T size)
      : rep_(ClampPayload(static_cast<uint64_t>(size))) {
    // Also rejects negative signed values, which wrap above kMaxPayload.
    HWY_DASSERT(static_cast<uint64_t>(size) <= kMaxPayload);
  }

  static constexpr Status Ok() { return Status(); }
  template <typename T, EnableIf<IsInteger<T>() &&
                                 !IsSame<RemoveCvRef<T>, bool>()>* = nullptr>
  static HWY_CXX14_CONSTEXPR Status Ok(T size) {
    return Status(size);
  }

  // Implicit from `GeneralCode` and from any module or application enum (scoped
  // or unscoped) whose underlying type is `Status::Code`, with an optional
  // payload (payloads above 2^48 - 1 saturate). `kOk` (0) yields plain success.
  // Debug builds check the invariant that nonzero codes have a non-`kOk`
  // severity.
  template <typename Enum, EnableIf<IsStatusEnum<Enum>()>* = nullptr>
  HWY_CXX14_CONSTEXPR Status(Enum code, uint64_t payload = 0)  // NOLINT
      : rep_((static_cast<uint64_t>(static_cast<Code>(code)) << kPayloadBits) |
             ClampPayload(payload)) {
    HWY_DASSERT(IsValidCode(static_cast<Code>(code)));
  }

  // Explicit factories that set the severity (overriding any severity bits in
  // `code` so raw 14-bit IDs or `GeneralCode`s can be used with any severity).
  // `code` must have a nonzero ID.
  static HWY_CXX14_CONSTEXPR Status Info(Code code, uint64_t payload = 0) {
    HWY_DASSERT((code & kMaxCode) != 0);
    return Status(RawTag(), MakeRep(Severity::kInfo, code, payload));
  }
  template <typename Enum, EnableIf<IsStatusEnum<Enum>()>* = nullptr>
  static HWY_CXX14_CONSTEXPR Status Info(Enum code, uint64_t payload = 0) {
    return Info(static_cast<Code>(code), payload);
  }
  static HWY_CXX14_CONSTEXPR Status Warning(Code code, uint64_t payload = 0) {
    HWY_DASSERT((code & kMaxCode) != 0);
    return Status(RawTag(), MakeRep(Severity::kWarning, code, payload));
  }
  template <typename Enum, EnableIf<IsStatusEnum<Enum>()>* = nullptr>
  static HWY_CXX14_CONSTEXPR Status Warning(Enum code, uint64_t payload = 0) {
    return Warning(static_cast<Code>(code), payload);
  }
  static HWY_CXX14_CONSTEXPR Status Error(Code code, uint64_t payload = 0) {
    HWY_DASSERT((code & kMaxCode) != 0);
    return Status(RawTag(), MakeRep(Severity::kError, code, payload));
  }
  template <typename Enum, EnableIf<IsStatusEnum<Enum>()>* = nullptr>
  static HWY_CXX14_CONSTEXPR Status Error(Enum code, uint64_t payload = 0) {
    return Error(static_cast<Code>(code), payload);
  }

  // True iff plain success (`Severity::kOk` and code 0; no error, warning, or
  // info).
  constexpr bool ok() const { return rep_ <= kMaxPayload; }
  constexpr Severity severity() const {
    return static_cast<Severity>(rep_ >> kSeverityShift);
  }
  constexpr bool IsInfo() const { return severity() == Severity::kInfo; }
  constexpr bool IsWarning() const { return severity() == Severity::kWarning; }
  constexpr bool IsError() const { return severity() == Severity::kError; }

  // Full 16-bit code including the 2-bit severity, for comparing or switching
  // against `GeneralCode` and module/app enums. Returns `kOk` (0) when `ok()`.
  constexpr Code code() const {
    return static_cast<Code>(rep_ >> kPayloadBits);
  }
  // 14-bit code ID without the severity bits.
  constexpr Code id() const {
    return static_cast<Code>((rep_ >> kPayloadBits) & kMaxCode);
  }
  // True iff `code()` equals `code` (severity and ID), ignoring the payload.
  // Accepts `GeneralCode` and module/app enums, including scoped ones.
  template <typename Enum, EnableIf<IsStatusEnum<Enum>()>* = nullptr>
  constexpr bool Is(Enum code) const {
    return this->code() == static_cast<Code>(code);
  }

  // Returns the 48-bit size for plain success (`ok()`). Debug builds assert
  // `ok()`; release builds still mask via `payload()` so non-`ok()` statuses
  // (e.g. `kInfo`/`kWarning` after `!IsError()`) never leak the upper 16
  // severity/code bits into a size calculation.
  HWY_CXX14_CONSTEXPR uint64_t size() const {
    HWY_DASSERT(ok());
    return payload();
  }
  // Valid for any status; typically a size for `ok()`, or OS/context details
  // for info, warnings, and errors (e.g. errno or byte offset).
  constexpr uint64_t payload() const { return rep_ & kMaxPayload; }

  // Returns the name of a general code (regardless of its severity), or
  // nullptr if `code` is not a `GeneralCode`.
  static constexpr const char* GeneralCodeName(Code code) {
    return (code & kMaxCode) == 0                              ? "kOk"
           : MakeCode(Severity::kError, code) == kUnknown      ? "kUnknown"
           : MakeCode(Severity::kError, code) == kCancelled    ? "kCancelled"
           : MakeCode(Severity::kError, code) == kArgument     ? "kArgument"
           : MakeCode(Severity::kError, code) == kTimeout      ? "kTimeout"
           : MakeCode(Severity::kError, code) == kNotFound     ? "kNotFound"
           : MakeCode(Severity::kError, code) == kPermission   ? "kPermission"
           : MakeCode(Severity::kError, code) == kResource     ? "kResource"
           : MakeCode(Severity::kError, code) == kPrecondition ? "kPrecondition"
           : MakeCode(Severity::kError, code) == kRange        ? "kRange"
           : MakeCode(Severity::kError, code) == kUnimplemented
               ? "kUnimplemented"
           : MakeCode(Severity::kError, code) == kInternal    ? "kInternal"
           : MakeCode(Severity::kError, code) == kUnavailable ? "kUnavailable"
           : MakeCode(Severity::kError, code) == kDataLoss    ? "kDataLoss"
           : MakeCode(Severity::kError, code) == kTruncated   ? "kTruncated"
           : MakeCode(Severity::kError, code) == kOSError     ? "kOSError"
                                                              : nullptr;
  }

  // Per-module callback returning the name of `code` (or nullptr if unknown).
  using CodeNameFunc = const char* (*)(Code code);

  // Registers `func` to resolve code names for `module`, returning the previous
  // function (or nullptr). Thread-safe.
  static HWY_STATUS_DLLEXPORT CodeNameFunc
  SetModuleCodeNameFunc(Module module, CodeNameFunc func);

  // Returns the name of a general code or registered module code (regardless of
  // its severity), or "(module/app code)" otherwise.
  static HWY_STATUS_DLLEXPORT const char* CodeName(Code code);
  template <typename Enum, EnableIf<IsStatusEnum<Enum>()>* = nullptr>
  static const char* CodeName(Enum code) {
    return CodeName(static_cast<Code>(code));
  }
  const char* Name() const { return CodeName(code()); }

  // Compares the entire representation including payload.
  friend constexpr bool operator==(Status a, Status b) {
    return a.rep_ == b.rep_;
  }
  friend constexpr bool operator!=(Status a, Status b) {
    return a.rep_ != b.rep_;
  }

  // Prevent `status == kCode`, which would implicitly construct
  // `Status(kCode, 0)` and fail whenever `status` has a nonzero payload (e.g.
  // errno). Use `status.Is(kCode)` to ignore the payload, or
  // `status == Status(kCode, payload)` to compare both.
  template <typename Enum>
  friend If<IsStatusEnum<Enum>(), bool, void> operator==(Status, Enum) = delete;
  template <typename Enum>
  friend If<IsStatusEnum<Enum>(), bool, void> operator!=(Status, Enum) = delete;
  template <typename Enum>
  friend If<IsStatusEnum<Enum>(), bool, void> operator==(Enum, Status) = delete;
  template <typename Enum>
  friend If<IsStatusEnum<Enum>(), bool, void> operator!=(Enum, Status) = delete;

 private:
  struct RawTag {};

  // The ID is zero iff the severity is kOk.
  static constexpr bool IsValidCode(Code code) {
    return ((code & kMaxCode) == 0) == ((code >> kIdBits) == 0);
  }

  static constexpr uint64_t MakeRep(Severity severity, Code code,
                                    uint64_t payload) {
    return (static_cast<uint64_t>(MakeCode(severity, code)) << kPayloadBits) |
           ClampPayload(payload);
  }
  // Payloads above 2^48 - 1 saturate to `kMaxPayload`.
  static constexpr uint64_t ClampPayload(uint64_t payload) {
    return payload <= kMaxPayload ? payload : kMaxPayload;
  }

  constexpr Status(RawTag /*tag*/, uint64_t rep) : rep_(rep) {}

  uint64_t rep_;
};

static_assert(sizeof(Status) == 8, "Status must be 8 bytes");

#if defined(HWY_HEADER_ONLY)
namespace detail {
HWY_STATUS_DLLEXPORT inline std::atomic<Status::CodeNameFunc>*
ModuleCodeNameTable() {
  static std::atomic<Status::CodeNameFunc> table[Status::kNumModules] = {};
  return table;
}
}  // namespace detail

HWY_STATUS_DLLEXPORT inline Status::CodeNameFunc Status::SetModuleCodeNameFunc(
    Module module, CodeNameFunc func) {
  const size_t idx = static_cast<size_t>(module);
  if (idx >= kNumModules) return nullptr;
  return detail::ModuleCodeNameTable()[idx].exchange(func,
                                                     std::memory_order_acq_rel);
}

HWY_STATUS_DLLEXPORT inline const char* Status::CodeName(Code code) {
  const char* general = GeneralCodeName(code);
  if (general != nullptr) return general;
  const Code id = static_cast<Code>(code & kMaxCode);
  if (id >= kFirstModuleCode && id < kFirstAppCode) {
    const size_t idx =
        static_cast<size_t>(id - kFirstModuleCode) / size_t{kCodesPerModule};
    const CodeNameFunc func =
        detail::ModuleCodeNameTable()[idx].load(std::memory_order_acquire);
    if (func != nullptr) {
      const char* name = func(code);
      if (name != nullptr) return name;
    }
  }
  return "(module/app code)";
}
#endif  // HWY_HEADER_ONLY

}  // namespace hwy

#undef HWY_STATUS_DLLEXPORT

#endif  // HIGHWAY_HWY_STATUS_H_
