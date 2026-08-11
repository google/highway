# Instruction matrix

## x86 SSE4, AVX2

| Operations ⬇       Types ⮕                   | i8 | u8 | i16 | u16 | i32 | u32 | i64 | u64 | f32 | f64 |
|----------------------------------------------|----|----|-----|-----|-----|-----|-----|-----|-----|-----|
|**ARITHMATIC**                                |    |    |     |     |     |     |     |     |     |     |
| add                                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| sub                                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| pairwise add                                 |  9 |  9 |  1  |  1  |  1  |  1  |  9  |  9  |  1  |  1  |
| pairwise sub                                 |  1 |  9 |  1  |  1  |  1  |  1  |  9  |  9  |  1  |  1  |
| saturated_add                                |  1 |  1 |  1  |  1  |  9  |  9  |  9  |  9  |     |     |
| saturated_sub                                |  1 |  1 |  1  |  1  |  9  |  9  |  9  |  9  |     |     |
| average                                      |  2 |  1 |  2  |  1  |  2  |  2  |  2  |  2  |     |     |
| shift left/right by constant bits            |  2 |  2 |  1  |  1  |  1  |  1  |  1  |  1  |     |     |
| bit-shift-right-var (independent lanes)      |  9 |  9 |  9  |  9  |  1  |  1  |  9  |  1  |     |     |
| bit-shift-left-var (independent lanes)       |  9 |  9 |  9  |  9  |  1  |  1  |  1  |  1  |     |     |
| abs                                          |  1 |    |  1  |     |  1  |     |  9  |     |  1  |  1  |
| negate                                       |  2 |    |  2  |     |  2  |     |  2  |     |  1  |  1  |
| min/max                                      |  1 |  1 |  1  |  1  |  1  |  1  |  2  |  3  |  1  |  1  |
| mul_truncate (x86 mullo)                     |  9 |  9 |  1  |  1  |  1  |  1  |  9  |  9  |     |     |
| mul_even (PPC mule)                          |  9 |  9 |  9  |  9  |  1  |  1  |     |     |     |     |
| mul_fp                                       |    |    |     |     |     |     |     |     |  1  |  1  |
| muladd_fp (3 variants with +/-)              |    |    |     |     |     |     |     |     |  1  |  1  |
| div_fp                                       |    |    |     |     |     |     |     |     |  1  |  1  |
| reciprocal_approx                            |    |    |     |     |     |     |     |     |  1  |  9  |
| sqrt                                         |    |    |     |     |     |     |     |     |  1  |  1  |
| recip_sqrt_approx                            |    |    |     |     |     |     |     |     |  1  |  9  |
| floor                                        |    |    |     |     |     |     |     |     |  1  |  1  |
| ceil                                         |    |    |     |     |     |     |     |     |  1  |  1  |
| round                                        |    |    |     |     |     |     |     |     |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **COMPARE**                                  |    |    |     |     |     |     |     |     |     |     |
| compare LT/GT                                |  1 |  3 |  1  |  3  |  1  |  3  |  1  |  3  |  1  |  1  |
| compare LE/GE                                |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |  1  |  1  |
| compare ==                                   |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| compare !=                                   |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |  1  |  1  |
| compare entire register to 0                 |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **LOGICAL**                                  |    |    |     |     |     |     |     |     |     |     |
| and                                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| and(not a, b)                                |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| or                                           |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| xor                                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| bitwise NOT                                  |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |  2  |  2  |
| movmskb (concat high bit of each byte)       |  1 |  1 |  2  |  2  |  1  |  1  |  1  |  1  |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **LOAD/STORE**                               |    |    |     |     |     |     |     |     |     |     |
| load_aligned                                 |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| load_unaligned                               |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| load_64_unaligned                            |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| load1_and_broadcast                          |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |  1  |  1  |
| store_aligned                                |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| store_unaligned                              |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| store64_unaligned                            |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| store_32                                     |  3 |  3 |  3  |  3  |  1  |  1  |  1  |  1  |  1  |  1  |
| stream (non-temporal write)                  |  9 |  9 |  9  |  9  |  1  |  1  |  1  |  1  |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **SWIZZLE**                                  |    |    |     |     |     |     |     |     |     |     |
| shift128 left/right by constant bytes        |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |     |     |
| Shift 2x128 bit right in byte increments     |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| broadcast any lane                           |  9 |  9 |  9  |  9  |  1  |  1  |  1  |  1  |  1  |  1  |
| 16-byte shuffle (var indices, >127 to zero)  |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| Shuffle1032, 0321, 2103                      |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| Interleave/zip = unpack                      |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| BlendV with full bit mask, not just MSB      |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **CONVERSION**                               |    |    |     |     |     |     |     |     |     |     |
| Expand to 2x width (u8->u16, f32->f64)       |  1 |  1 |  1  |  1  |  1  |  1  |     |     |  1  |     |
| Reducing to half width (e.g. u16->u8)        |    |    |  1  |  1  |  1  |  1  |  9  |  9  |     |  1  |
| Convert integer -> same size real            |    |    |     |     |  1  |  9  |  9  |  9  |     |     |
| Convert real -> same size integer            |    |    |     |     |     |     |     |     |  1  |  9  |
| Extract lane 0 to reg/aligned mem            |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| Insert reg/aligned mem into lane 0           |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **CRYPTO/HASH**                              |    |    |     |     |     |     |     |     |     |     |
| SHA1                                         |    |    |     |     |     |  1  |     |     |     |     |
| SHA256                                       |    |    |     |     |     |  1  |     |     |     |     |
| AES                                          |    |  1 |     |     |     |     |     |     |     |     |
| CRC32C                                       |    |  1 |     |  1  |     |  1  |     |  1  |     |     |
| CLMUL                                        |    |    |     |     |     |     |     |  1  |     |     |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **EMULATED**                                 |    |    |     |     |     |     |     |     |     |     |
| mulhi16                                      |    |    |  1  |     |     |     |     |     |     |     |
| horz_sum                                     |    |  1 |     |     |     |     |     |     |     |     |

## ppc (POWER 8/9) w/ VSX

| Operations ⬇       Types ⮕                   | i8 | u8 | i16 | u16 | i32 | u32 | i64 | u64 | f32 | f64 |
|----------------------------------------------|----|----|-----|-----|-----|-----|-----|-----|-----|-----|
|**ARITHMATIC**                                |    |    |     |     |     |     |     |     |     |     |
| add                                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| sub                                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| pairwise add                                 |  9 |  9 |  9  |  9  |  9  |  9  |  9  |  9  |  9  |  9  |
| pairwise sub                                 |  9 |  9 |  9  |  9  |  9  |  9  |  9  |  9  |  9  |  9  |
| saturated_add                                |  1 |  1 |  1  |  1  |  1  |  1  |  9  |  9  |     |     |
| saturated_sub                                |  1 |  1 |  1  |  1  |  1  |  1  |  9  |  9  |     |     |
| average                                      |  1 |  1 |  1  |  1  |  1  |  1  |  8  |  8  |     |     |
| shift left/right by constant bits            |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |     |     |
| bit-shift-right-var (independent lanes)      |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |     |     |
| bit-shift-left-var (independent lanes)       |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |     |     |
| abs                                          |  3 |    |  3  |     |  3  |     |  3  |     |  1  |  1  |
| negate                                       |  2 |    |  2  |     |  2  |     |  2  |     |  1  |  1  |
| min/max                                      |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| mul_truncate (x86 mullo)                     |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |     |     |
| mul_even (PPC mule)                          |  1 |  1 |  1  |  1  |  1  |  1  |     |     |     |     |
| mul_fp                                       |    |    |     |     |     |     |     |     |  1  |  1  |
| muladd_fp (3 variants with +/-)              |    |    |     |     |     |     |     |     |  1  |  1  |
| div_fp                                       |    |    |     |     |     |     |     |     |  1  |  1  |
| reciprocal_approx                            |    |    |     |     |     |     |     |     |  1  |  1  |
| sqrt                                         |    |    |     |     |     |     |     |     |  1  |  1  |
| recip_sqrt_approx                            |    |    |     |     |     |     |     |     |  1  |  1  |
| floor                                        |    |    |     |     |     |     |     |     |  1  |  1  |
| ceil                                         |    |    |     |     |     |     |     |     |  1  |  1  |
| round                                        |    |    |     |     |     |     |     |     |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **COMPARE**                                  |    |    |     |     |     |     |     |     |     |     |
| compare LT/GT                                |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| compare LE/GE                                |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| compare ==                                   |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| compare !=                                   |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |  2  |  2  |
| compare entire register to 0                 |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **LOGICAL**                                  |    |    |     |     |     |     |     |     |     |     |
| and                                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| and(not a, b)                                |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| or                                           |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| xor                                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| bitwise NOT                                  |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |  2  |  2  |
| movmskb (concat high bit of each byte)       |  3 |  3 |  3  |  3  |  3  |  3  |  3  |  3  |  3  |  3  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **LOAD/STORE**                               |    |    |     |     |     |     |     |     |     |     |
| load_aligned                                 |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| load_unaligned                               |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| load_64_unaligned                            |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |  2  |  2  |
| load1_and_broadcast                          |  2 |  2 |  2  |  2  |  2  |  2  |  1  |  1  |  1  |  1  |
| store_aligned                                |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| store_unaligned                              |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| store64_unaligned                            |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |  2  |  2  |
| store_32                                     |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| stream (non-temporal write)                  |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **SWIZZLE**                                  |    |    |     |     |     |     |     |     |     |     |
| shift128 left/right by constant bytes        |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |     |     |
| Shift 2x128 bit right in byte increments     |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| broadcast any lane                           |  9 |  9 |  9  |  9  |  1  |  1  |  1  |  1  |  1  |  1  |
| 16-byte shuffle (var indices, >127 to zero)  |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| Shuffle1032, 0321, 2103                      |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| Interleave/zip = unpack                      |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| BlendV with full bit mask, not just MSB      |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **CONVERSION**                               |    |    |     |     |     |     |     |     |     |     |
| Expand to 2x width (u8->u16, f32->f64)       |  1 |  2 |  1  |  2  |  1  |  2  |     |     |  1  |     |
| Reducing to half width (e.g. u16->u8)        |    |    |  1  |  1  |  1  |  1  |  1  |  1  |     |  1  |
| Convert integer -> same size real            |    |    |     |     |  1  |  1  |  1  |  1  |     |     |
| Convert real -> same size integer            |    |    |     |     |     |     |     |     |  1  |  1  |
| Extract lane 0 to reg/aligned mem            |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| Insert reg/aligned mem into lane 0           |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **CRYPTO/HASH**                              |    |    |     |     |     |     |     |     |     |     |
| SHA1                                         |    |    |     |     |     |  1  |     |     |     |     |
| SHA256                                       |    |    |     |     |     |  1  |     |     |     |     |
| AES                                          |    |  1 |     |     |     |     |     |     |     |     |
| CRC32C                                       |    |  3 |     |  3  |     |  3  |     |  3  |     |     |
| CLMUL                                        |    |    |     |     |     |     |     |  1  |     |     |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **EMULATED**                                 |    |    |     |     |     |     |     |     |     |     |
| mulhi16                                      |    |    |  1  |  3  |     |     |     |     |     |     |
| horz_sum                                     |    |  5 |     |     |     |     |     |     |     |     |

Note: Loads/Stores may be big endian on PPC.

## arm64 NEON

| Operations ⬇       Types ⮕                   | i8 | u8 | i16 | u16 | i32 | u32 | i64 | u64 | f32 | f64 |
|----------------------------------------------|----|----|-----|-----|-----|-----|-----|-----|-----|-----|
|**ARITHMATIC**                                |    |    |     |     |     |     |     |     |     |     |
| add                                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| sub                                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| pairwise add                                 |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| pairwise sub                                 |  9 |  9 |  9  |  9  |  9  |  9  |  9  |  9  |  9  |  9  |
| saturated_add                                |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |     |     |
| saturated_sub                                |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |     |     |
| average                                      |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |     |     |
| shift left/right by constant bits            |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |     |     |
| bit-shift-right-var (independent lanes)      |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |     |     |
| bit-shift-left-var (independent lanes)       |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |     |     |
| abs                                          |  1 |    |  1  |     |  1  |     |  1  |     |  1  |  1  |
| negate                                       |  1 |    |  1  |     |  1  |     |  1  |     |  1  |  1  |
| min/max                                      |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| mul_truncate (x86 mullo)                     |  1 |  1 |  1  |  1  |  1  |  1  |  9  |  9  |     |     |
| mul_even (PPC mule)                          |  2 |  2 |  2  |  2  |  2  |  2  |     |     |     |     |
| mul_fp                                       |    |    |     |     |     |     |     |     |  1  |  1  |
| muladd_fp (3 variants with +/-)              |    |    |     |     |     |     |     |     |  1  |  1  |
| div_fp                                       |    |    |     |     |     |     |     |     |  1  |  1  |
| reciprocal_approx                            |    |    |     |     |     |     |     |     |  1  |  1  |
| sqrt                                         |    |    |     |     |     |     |     |     |  1  |  1  |
| recip_sqrt_approx                            |    |    |     |     |     |     |     |     |  1  |  1  |
| floor                                        |    |    |     |     |     |     |     |     |  1  |  1  |
| ceil                                         |    |    |     |     |     |     |     |     |  1  |  1  |
| round                                        |    |    |     |     |     |     |     |     |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **COMPARE**                                  |    |    |     |     |     |     |     |     |     |     |
| compare LT/GT                                |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| compare LE/GE                                |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| compare ==                                   |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| compare !=                                   |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| compare entire register to 0                 |  3 |  3 |  3  |  3  |  3  |  3  |  3  |  3  |  3  |  3  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **LOGICAL**                                  |    |    |     |     |     |     |     |     |     |     |
| and                                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| and(not a, b)                                |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| or                                           |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| xor                                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| bitwise NOT                                  |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| movmskb (concat high bit of each byte)       |  5 |  5 |  5  |  5  |  5  |  5  |  5  |  5  |  5  |  5  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **LOAD/STORE**                               |    |    |     |     |     |     |     |     |     |     |
| load_aligned                                 |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| load_unaligned                               |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| load_64_unaligned                            |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| load1_and_broadcast                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| store_aligned                                |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| store_unaligned                              |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| store64_unaligned                            |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| store_32                                     |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| stream (non-temporal write)                  |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **SWIZZLE**                                  |    |    |     |     |     |     |     |     |     |     |
| shift128 left/right by constant bytes        |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |     |     |
| Shift 2x128 bit right in byte increments     |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| broadcast any lane                           |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| 16-byte shuffle (var indices, >127 to zero)  |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| Shuffle1032, 0321, 2103                      |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| Interleave/zip = unpack                      |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| BlendV with full bit mask, not just MSB      |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **CONVERSION**                               |    |    |     |     |     |     |     |     |     |     |
| Expand to 2x width (u8->u16, f32->f64)       |  1 |  1 |  1  |  1  |  1  |  1  |     |     |  1  |     |
| Reducing to half width (e.g. u16->u8)        |    |    |  1  |  1  |  1  |  1  |  1  |  1  |     |  1  |
| Convert integer -> same size real            |    |    |  1  |  1  |  1  |  1  |  1  |  1  |     |     |
| Convert real -> same size integer            |    |    |     |     |     |     |     |     |  1  |  1  |
| Extract lane 0 to reg/aligned mem            |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| Insert reg/aligned mem into lane 0           |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **CRYPTO/HASH**                              |    |    |     |     |     |     |     |     |     |     |
| SHA1                                         |    |    |     |     |     |  1  |     |     |     |     |
| SHA256                                       |    |    |     |     |     |  1  |     |     |     |     |
| AES                                          |    |  1 |     |     |     |     |     |     |     |     |
| CRC32C                                       |    |  1 |     |  1  |     |  1  |     |  1  |     |     |
| CLMUL                                        |    |    |     |     |     |     |     |     |     |     |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **EMULATED**                                 |    |    |     |     |     |     |     |     |     |     |
| mulhi16                                      |    |    |  2  |     |     |     |     |     |     |     |
| horz_sum                                     |    |  3 |     |     |     |     |     |     |     |     |

## Intersection

Worst case instruction count.


| Operations ⬇       Types ⮕                   | i8 | u8 | i16 | u16 | i32 | u32 | i64 | u64 | f32 | f64 |
|----------------------------------------------|----|----|-----|-----|-----|-----|-----|-----|-----|-----|
|**ARITHMATIC**                                |    |    |     |     |     |     |     |     |     |     |
| add                                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| sub                                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| pairwise add                                 |  9 |  9 |  9  |  9  |  9  |  9  |  9  |  9  |  9  |  9  |
| pairwise sub                                 |  9 |  9 |  9  |  9  |  9  |  9  |  9  |  9  |  9  |  9  |
| saturated_add                                |  1 |  1 |  1  |  1  |  9  |  9  |  9  |  9  |     |     |
| saturated_sub                                |  1 |  1 |  1  |  1  |  9  |  9  |  9  |  9  |     |     |
| average                                      |  2 |  1 |  2  |  1  |  2  |  2  |  8  |  8  |     |     |
| shift left/right by constant bits            |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |     |     |
| bit-shift-right-var (independent lanes)      |  9 |  9 |  9  |  9  |  2  |  2  |  9  |  2  |     |     |
| bit-shift-left-var (independent lanes)       |  9 |  9 |  9  |  9  |  1  |  1  |  1  |  1  |     |     |
| abs                                          |  3 |    |  3  |     |  3  |     |  9  |     |  1  |  1  |
| negate                                       |  2 |    |  2  |     |  2  |     |  2  |     |  1  |  1  |
| min/max                                      |  1 |  1 |  1  |  1  |  1  |  1  |  2  |  3  |  1  |  1  |
| mul_truncate (x86 mullo)                     |  9 |  9 |  1  |  1  |  1  |  1  |  9  |  9  |     |     |
| mul_even (PPC mule)                          |  9 |  9 |  9  |  9  |  2  |  2  |     |     |     |     |
| mul_fp                                       |    |    |     |     |     |     |     |     |  1  |  1  |
| muladd_fp (3 variants with +/-)              |    |    |     |     |     |     |     |     |  1  |  1  |
| div_fp                                       |    |    |     |     |     |     |     |     |  1  |  1  |
| reciprocal_approx                            |    |    |     |     |     |     |     |     |  1  |  9  |
| sqrt                                         |    |    |     |     |     |     |     |     |  1  |  1  |
| recip_sqrt_approx                            |    |    |     |     |     |     |     |     |  1  |  9  |
| floor                                        |    |    |     |     |     |     |     |     |  1  |  1  |
| ceil                                         |    |    |     |     |     |     |     |     |  1  |  1  |
| round                                        |    |    |     |     |     |     |     |     |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **COMPARE**                                  |    |    |     |     |     |     |     |     |     |     |
| compare LT/GT                                |  1 |  3 |  1  |  3  |  1  |  3  |  1  |  3  |  1  |  1  |
| compare LE/GE                                |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |  1  |  1  |
| compare ==                                   |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| compare !=                                   |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |  2  |  2  |
| compare entire register to 0                 |  3 |  3 |  3  |  3  |  3  |  3  |  3  |  3  |  3  |  3  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **LOGICAL**                                  |    |    |     |     |     |     |     |     |     |     |
| and                                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| and(not a, b)                                |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| or                                           |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| xor                                          |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| bitwise NOT                                  |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |  2  |  2  |
| movmskb (concat high bit of each byte)       |  5 |  5 |  5  |  5  |  5  |  5  |  5  |  5  |  5  |  5  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **LOAD/STORE**                               |    |    |     |     |     |     |     |     |     |     |
| load_aligned                                 |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| load_unaligned                               |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| load_64_unaligned                            |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |  2  |  2  |
| load1_and_broadcast                          |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |  1  |  1  |
| store_aligned                                |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| store_unaligned                              |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| store64_unaligned                            |  2 |  2 |  2  |  2  |  2  |  2  |  2  |  2  |  2  |  2  |
| store_32                                     |  3 |  3 |  3  |  3  |  1  |  1  |  1  |  1  |  1  |  1  |
| stream (non-temporal write)                  |  9 |  9 |  9  |  9  |  1  |  1  |  1  |  1  |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **SWIZZLE**                                  |    |    |     |     |     |     |     |     |     |     |
| shift128 left/right by constant bytes        |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |     |     |
| Shift 2x128 bit right in byte increments     |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| broadcast any lane                           |  9 |  9 |  9  |  9  |  1  |  1  |  1  |  1  |  1  |  1  |
| 16-byte shuffle (var indices, >127 to zero)  |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| Shuffle1032, 0321, 2103                      |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| Interleave/zip = unpack                      |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| BlendV with full bit mask, not just MSB      |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **CONVERSION**                               |    |    |     |     |     |     |     |     |     |     |
| Expand to 2x width (u8->u16, f32->f64)       |  1 |  2 |  1  |  2  |  1  |  2  |     |     |  1  |     |
| Reducing to half width (e.g. u16->u8)        |    |    |  1  |  1  |  1  |  1  |  9  |  9  |     |  1  |
| Convert integer -> same size real            |    |    |     |     |  9  |  9  |  9  |  9  |     |     |
| Convert real -> same size integer            |    |    |     |     |     |     |     |     |  1  |  9  |
| Extract lane 0 to reg/aligned mem            |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
| Insert reg/aligned mem into lane 0           |  1 |  1 |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **CRYPTO/HASH**                              |    |    |     |     |     |     |     |     |     |     |
| SHA1                                         |    |    |     |     |     |  1  |     |     |     |     |
| SHA256                                       |    |    |     |     |     |  1  |     |     |     |     |
| AES                                          |    |  1 |     |     |     |     |     |     |     |     |
| CRC32C                                       |    |  3 |     |  3  |     |  3  |     |  3  |     |     |
| CLMUL                                        |    |    |     |     |     |     |     |     |     |     |
|                                              |    |    |     |     |     |     |     |     |     |     |
| **EMULATED**                                 |    |    |     |     |     |     |     |     |     |     |
| mulhi16                                      |    |    |     |     |     |     |     |     |     |     |
| horz_sum                                     |    |    |     |     |     |     |     |     |     |     |
