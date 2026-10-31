/* Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Verifies that base_c.h compiles as C and its macros are usable from C. */

#include "hwy/base_c.h"

#include <stdio.h>

#if !HWY_VERSION_GE(1, 0) || HWY_VERSION_LT(1, 0)
#error "Version macros are inconsistent"
#endif

#if HWY_ARCH_MAX_BYTES < 16
#error "Unexpected HWY_ARCH_MAX_BYTES"
#endif

/* Declaration-only export exercises consumer dllimport under HWY_SHARED_DEFINE.
 */
HWY_DLLEXPORT HWY_MUST_USE_RESULT int BaseCTestExportedDecl(int a, int b);

static HWY_MUST_USE_RESULT HWY_NOINLINE int BaseCTestAdd(int a, int b) {
  return a + b;
}

#if HWY_ARCH_X86_64
HWY_PUSH_ATTRIBUTES("sse2")
static int BaseCTestTargetAttr(int x) { return x + 1; }
HWY_POP_ATTRIBUTES
#else
static int BaseCTestTargetAttr(int x) { return x + 1; }
#endif

static HWY_NORETURN HWY_MAYBE_UNUSED void BaseCTestNoreturn(void) {
  for (;;) {
  }
}

HWY_PRESERVE_MOST static void BaseCTestCheckNonNegative(int* out, int x) {
  if (x < 0) {
    HWY_UNREACHABLE;
  }
  *out = x;
}

HWY_NODISCARD static HWY_INLINE int BaseCTestSum(const int* HWY_RESTRICT p,
                                                 size_t n) {
  const int* HWY_RESTRICT aligned =
      (const int*)HWY_ASSUME_ALIGNED(p, sizeof(int));
  int sum = 0;
  size_t i;
  HWY_DEFAULT_UNROLL
  for (i = 0; i < n; ++i) {
    HWY_ASSUME(i < n);
    sum += aligned[i];
  }
  HWY_UNROLL(2)
  HWY_DISABLE_AUTOVEC
  for (i = 0; i < n; ++i) {
    sum += aligned[i];
  }
  return sum / 2;
}

HWY_API int BaseCTestMax(int a, int b) { return HWY_MAX(a, b); }

static HWY_FORMAT(1, 2) HWY_MAYBE_UNUSED
    void BaseCTestPrint(const char* format, ...) {
  (void)format;
}

int main(void) {
  static const int kValues[4] = {HWY_REP4(2)};
  int HWY_CONCAT(local_, 1) = BaseCTestAdd(1, 2);
  int checked = 0;
  const char* version = HWY_STR(HWY_MAJOR);
  const char* func = HWY_FUNCTION;
  BaseCTestCheckNonNegative(&checked, local_1);
  if (HWY_UNLIKELY(checked != 3)) return 1;
  if (!HWY_LIKELY(BaseCTestSum(kValues, 4) == 8)) return 2;
  if (BaseCTestMax(HWY_MIN(1, 5), 4) != 4) return 3;
  if (version[0] != '1') return 4;
  if (func[0] == '\0') return 5;
  if (BaseCTestTargetAttr(4) != 5) return 6;
  BaseCTestPrint("%s %s", version, func);
  printf("Highway %d.%d.%d debug=%d arch_max_bytes=%d\n", HWY_MAJOR, HWY_MINOR,
         HWY_PATCH, HWY_IS_DEBUG_BUILD, HWY_ARCH_MAX_BYTES);
  return 0;
}
