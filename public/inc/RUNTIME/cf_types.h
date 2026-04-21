/*
 * CF Framework
 * Copyright (C) 2026 Orion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#if !defined(CF_TYPES_H)
#define CF_TYPES_H

/* Standard headers used by the framework type aliases. */
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>

/* Explicitly marks a parameter or local as intentionally unused. */
#define CF_UNUSED(x) ((void) x)

#define CF_ASSERT_TYPE_SIZE(x,y) assert(sizeof(x) == sizeof(y))

/* Framework boolean aliases for readable public APIs. */
#define CF_TRUE true
#define CF_FALSE false
typedef bool cf_bool;

/* Framework null-pointer alias for APIs that prefer CF-prefixed constants. */
#define CF_NULL NULL

/* Signed fixed-width integer types. */
typedef int8_t cf_i8;
typedef int16_t cf_i16;
typedef int32_t cf_i32;
typedef int64_t cf_i64;
typedef __int128_t cf_i128;

/* Unsigned fixed-width integer types. */
typedef uint8_t cf_u8;
typedef uint16_t cf_u16;
typedef uint32_t cf_u32;
typedef uint64_t cf_u64;
typedef __uint128_t cf_u128;

/* Signed and unsigned size types for counts, lengths, and offsets. */
typedef ssize_t cf_isize;
typedef size_t cf_usize;

/* Integer types large enough to hold pointer values for address math. */
typedef uintptr_t cf_uptr;
typedef intptr_t cf_iptr;

/**
 * Group native C values by their byte width.
 *
 * These groupings cover common primitive-sized values and structs whose total
 * size matches one of the listed native widths. Values outside these groups
 * are treated as non-primitive or non-native-sized aggregates.
 */
typedef enum cf_native_group
{
  CF_NATIVE_UNKNOWN = 0,
  CF_NATIVE_1_BYTE  = sizeof(cf_u8),
  CF_NATIVE_2_BYTE  = sizeof(cf_u16),
  CF_NATIVE_4_BYTE  = sizeof(cf_u32),
  CF_NATIVE_8_BYTE  = sizeof(cf_u64),
  CF_NATIVE_16_BYTE = sizeof(cf_u128)
} cf_native_group;

/**
 * Map a raw byte size to the nearest native-size group.
 *
 * Exact matches for 1, 2, 4, 8, and 16 bytes return their corresponding
 * native group. Any other size returns `CF_NATIVE_UNKNOWN`.
 *
 * @param type_size Byte size, typically produced by `sizeof(...)`.
 * @return Matching native-size enum value or `CF_NATIVE_UNKNOWN`.
 */
cf_native_group cf_types_type_size(cf_usize type_size);

/**
 * Convert a byte size to a readable native-size description.
 *
 * The returned text explains what commonly fits in that byte width, including
 * primitive values and structs whose total size matches the same width. Sizes
 * outside the native groups return a message indicating that the value is not
 * a primitive type or a struct with a primitive native size.
 *
 * @param type_size Byte size, typically produced by `sizeof(...)`.
 * @return Stable null-terminated descriptive string.
 */
const char *cf_types_as_char(cf_usize type_size);

#endif /* CF_TYPES_H */
