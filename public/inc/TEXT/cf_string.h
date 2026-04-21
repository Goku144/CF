/*
 * CF Framework
 * Copyright (C) 2026 Orion
 *
 * This program is free software: you can redisibute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is disibuted in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#if !defined(CF_STRING_H)
#define CF_STRING_H

#include "MEMORY/cf_memory.h"

/**
 * @brief Initializes a string with optional character capacity.
 *
 * The string is backed by the buffer implementation, but reserves one extra
 * byte for the trailing null terminator and writes an empty C string after
 * initialization.
 *
 * @param str String to initialize.
 * @param capacity Initial usable character capacity, excluding the terminator.
 * @return `CF_OK` on success, or a status describing null or allocation
 * failure conditions.
 */
cf_status cf_string_init(cf_string *str, cf_usize capacity);

/**
 * @brief Ensures that a string can hold at least the requested characters.
 *
 * The requested capacity is treated as usable string data space. One extra
 * byte is reserved internally for the trailing null terminator.
 *
 * @param str String whose capacity must be ensured.
 * @param capacity Minimum usable character capacity required after the call.
 * @return `CF_OK` on success, or a status describing null, invalid-state, or
 * allocation failure conditions.
 */
cf_status cf_string_reserve(cf_string *str, cf_usize capacity);

/**
 * @brief Releases a string's owned storage and resets it to an empty state.
 *
 * @param str String to destroy.
 * @return void
 */
void cf_string_destroy(cf_string *str);

/**
 * @brief Appends one character to the end of a string.
 *
 * The string remains null-terminated after the append.
 *
 * @param dst Destination string receiving the character.
 * @param c Character to append.
 * @return `CF_OK` on success, or a status describing invalid-state or
 * allocation failure conditions.
 */
cf_status cf_string_append_char(cf_string *dst, char c);

/**
 * @brief Appends a null-terminated C string to the end of a string.
 *
 * The source text is copied into the destination and the destination remains
 * null-terminated after the append.
 *
 * @param dst Destination string receiving the copied text.
 * @param c Null-terminated C string to append.
 * @return `CF_OK` on success, or a status describing invalid-state or
 * allocation failure conditions.
 */
cf_status cf_string_append_cstr(cf_string *dst, char *c);

/**
 * @brief Appends one framework string to another.
 *
 * The source string data is copied into the destination string tail and the
 * destination remains null-terminated after the append.
 *
 * @param dst Destination string receiving the copied text.
 * @param src Source string to append.
 * @return `CF_OK` on success, or a status describing invalid-state or
 * allocation failure conditions.
 */
cf_status cf_string_append_str(cf_string *dst, cf_string *src);

/**
 * @brief Copies a framework string into a newly allocated C string.
 *
 * The returned string is allocated with the source string's allocator and is
 * null-terminated.
 *
 * @param cdst Output pointer receiving the allocated C string.
 * @param src Source string to copy.
 * @return `CF_OK` on success, or a status describing null, invalid-state, or
 * allocation failure conditions.
 */
cf_status cf_string_as_cstr(char **cdst, cf_string *src);

/**
 * @brief Replaces a string's contents with a null-terminated C string.
 *
 * The destination length is reset before the source text is copied, and the
 * destination remains null-terminated afterward.
 *
 * @param dst Destination string to overwrite.
 * @param src Null-terminated C string to copy.
 * @return `CF_OK` on success, or a status describing null, invalid-state, or
 * allocation failure conditions.
 */
cf_status cf_string_from_cstr(cf_string *dst, char *src);

/**
 * @brief Marks a string as empty without releasing its allocated storage.
 *
 * The string length becomes zero and, when backing storage exists, the first
 * byte is set to the null terminator.
 *
 * @param str String to reset.
 * @return `CF_OK` on success, or a status describing null or invalid-state
 * conditions.
 */
cf_status cf_string_reset(cf_string *str);

/**
 * @brief Shrinks the logical length of a string.
 *
 * The backing allocation is not changed. Truncation is only valid when `len`
 * is less than or equal to the current logical length, and the resulting
 * string is null-terminated.
 *
 * @param str String to truncate.
 * @param len New logical string length.
 * @return `CF_OK` on success, or a status describing bounds, null, or
 * invalid-state conditions.
 */
cf_status cf_string_trunc(cf_string *str, cf_usize len);

/**
 * @brief Check whether a string satisfies the framework's structural rules.
 *
 * A valid string is a valid buffer whose logical end contains a trailing null
 * terminator when storage is present.
 *
 * @param str String to validate.
 * @return `CF_TRUE` when the string is structurally valid, otherwise
 * `CF_FALSE`.
 */
cf_bool cf_string_is_valid(cf_string *str);

/**
 * @brief Report whether a string currently contains no logical characters.
 *
 * Invalid strings are treated as not empty.
 *
 * @param str String to inspect.
 * @return `CF_TRUE` when the string is valid and its logical length is zero,
 * otherwise `CF_FALSE`.
 */
cf_bool cf_string_is_empty(cf_string *str);

/**
 * @brief Print a diagnostic summary of a string's current state.
 *
 * The printed information includes the string data, logical length,
 * allocation capacity, and whether each allocator callback field is set.
 *
 * @param str String to inspect and print.
 * @return void
 */
void cf_string_info(cf_string *str);

/**
 * @brief Compare two strings for byte-for-byte equality.
 *
 * @param str1 First string to compare.
 * @param str2 Second string to compare.
 * @return `CF_TRUE` when both strings are valid and contain the same bytes,
 * otherwise `CF_FALSE`.
 */
cf_bool cf_string_eq(cf_string *str1, cf_string *str2);

/**
 * @brief Check whether a string contains a character.
 *
 * @param str String to search.
 * @param c Character to find.
 * @return `CF_TRUE` when the string is valid and contains `c`, otherwise
 * `CF_FALSE`.
 */
cf_bool cf_string_contains_char(cf_string *str, char c);

/**
 * @brief Check whether a string contains a C string.
 *
 * @param str String to search.
 * @param c Null-terminated C string to find.
 * @return `CF_TRUE` when the string is valid and contains `c`, otherwise
 * `CF_FALSE`.
 */
cf_bool cf_string_contains_cstr(cf_string *str, char *c);

/**
 * @brief Check whether one framework string contains another.
 *
 * @param str1 String to search.
 * @param str2 String to find.
 * @return `CF_TRUE` when both strings are valid and `str1` contains `str2`,
 * otherwise `CF_FALSE`.
 */
cf_bool cf_string_contains_str(cf_string *str1, cf_string *str2);

/**
 * @brief Read one character from a string by index.
 *
 * @param str Source string to inspect.
 * @param index Zero-based character index to read.
 * @param c Output pointer receiving the selected character.
 * @return `CF_OK` on success, `CF_ERR_BOUNDS` when `index` is outside the
 * current logical length, or another status for null or invalid-state
 * conditions.
 */
cf_status cf_string_char_at(cf_string *str, cf_usize index, char *c);

/**
 * @brief Copy a string suffix into a newly allocated C string.
 *
 * The copied text starts at `index` and continues through the source string's
 * trailing null terminator.
 *
 * @param str Source string to copy from.
 * @param index Zero-based start index.
 * @param c Output pointer receiving the allocated C string.
 * @return `CF_OK` on success, `CF_ERR_BOUNDS` when `index` is outside the
 * current logical length, or another status for null or invalid-state
 * conditions.
 */
cf_status cf_string_str_at(cf_string *str, cf_usize index, char **c);

/**
 * @brief Removes leading whitespace characters from a string.
 *
 * Whitespace is limited to space, tab, carriage return, and newline.
 *
 * @param str String to trim.
 * @return `CF_OK` on success, or a status describing null or invalid-state
 * conditions.
 */
cf_status cf_string_trim_left(cf_string *str);

/**
 * @brief Removes trailing whitespace characters from a string.
 *
 * Whitespace is limited to space, tab, carriage return, and newline.
 *
 * @param str String to trim.
 * @return `CF_OK` on success, or a status describing null or invalid-state
 * conditions.
 */
cf_status cf_string_trim_right(cf_string *str);

/**
 * @brief Removes leading and trailing whitespace characters from a string.
 *
 * Whitespace is limited to space, tab, carriage return, and newline.
 *
 * @param str String to trim.
 * @return `CF_OK` on success, or a status describing null or invalid-state
 * conditions.
 */
cf_status cf_string_trim(cf_string *str);

/**
 * @brief Removes all whitespace characters from a string.
 *
 * Whitespace is limited to space, tab, carriage return, and newline. The
 * string is compacted in place and remains null-terminated afterward.
 *
 * @param str String to strip.
 * @return `CF_OK` on success, or a status describing null or invalid-state
 * conditions.
 */
cf_status cf_string_strip(cf_string *str);

/**
 * @brief Replaces every matching character in a string.
 *
 * Each occurrence of `targetc` is replaced with `newc` in place. The string
 * length is unchanged and remains null-terminated afterward.
 *
 * @param str String to modify.
 * @param targetc Character to replace.
 * @param newc Replacement character.
 * @return `CF_OK` on success, or a status describing null or invalid-state
 * conditions.
 */
cf_status cf_string_replace(cf_string *str, char targetc, char newc);

/**
 * @brief Copies an inclusive string range into a newly allocated C string.
 *
 * The returned C string is allocated with the source string's allocator and is
 * null-terminated.
 *
 * @param dst Output pointer receiving the allocated C string.
 * @param src Source string to copy from.
 * @param start Inclusive start index.
 * @param end Inclusive end index.
 * @return `CF_OK` on success, `CF_ERR_INVALID` when `start > end`,
 * `CF_ERR_BOUNDS` when the range exceeds the current logical length, or
 * another status for null or invalid-state conditions.
 */
cf_status cf_string_slice(char **dst, cf_string *src, cf_usize start, cf_usize end);

/**
 * @brief Splits a string into non-empty parts separated by a character.
 *
 * Each part is allocated as a null-terminated C string and pushed into `dst` as
 * a `cf_array_element`. Empty parts between adjacent separators are skipped.
 *
 * @param dst Initialized array receiving the split parts.
 * @param src Source string to split.
 * @param c Separator character.
 * @return `CF_OK` on success, or a status describing null, invalid-state, or
 * allocation failure conditions.
 */
cf_status cf_string_split(cf_array *dst, cf_string *src, char c);


#endif /* CF_STRING_H */
