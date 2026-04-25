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

#if !defined(CF_STRING_H)
#define CF_STRING_H

#include "MEMORY/cf_memory.h"
#include "MEMORY/cf_array.h"

/**
 * @brief Check whether a string satisfies the framework's structural rules.
 *
 * A valid string has consistent `data`, `cap`, and `len` fields and stores a
 * trailing null terminator at `data[len]` when storage is present.
 *
 * @param str String to validate.
 * @return `CF_TRUE` when the string is structurally valid, otherwise
 * `CF_FALSE`.
 */
cf_bool cf_string_is_valid(cf_string *str);

/**
 * @brief Initialize a string with optional character capacity.
 *
 * The string is backed by the buffer implementation, but reserves one extra
 * byte for the trailing null terminator and writes an empty C string after
 * initialization.
 *
 * @param str String to initialize.
 * @param capacity Initial usable character capacity, excluding the terminator.
 * @return `CF_OK` on success or `CF_ERR_OOM` when allocation fails.
 */
cf_status cf_string_init(cf_string *str, cf_usize capacity);

/**
 * @brief Ensure that a string can hold at least the requested characters.
 *
 * The requested capacity is treated as usable string data space. One extra
 * byte is reserved internally for the trailing null terminator.
 *
 * @param str String whose capacity must be ensured.
 * @param capacity Minimum usable character capacity required after the call.
 * @return `CF_OK` on success or `CF_ERR_OOM` when reallocation fails.
 */
cf_status cf_string_reserve(cf_string *str, cf_usize capacity);

/**
 * @brief Mark a string as empty without releasing its allocated storage.
 *
 * The string length becomes zero and, when backing storage exists, the first
 * byte is set to the null terminator.
 *
 * @param str String to reset.
 */
void cf_string_reset(cf_string *str);

/**
 * @brief Release a string's owned storage and reset it to an empty state.
 *
 * @param str String to destroy.
 */
void cf_string_destroy(cf_string *str);

/**
 * @brief Append one character to the end of a string.
 *
 * The string remains null-terminated after the append.
 *
 * @param dst Destination string receiving the character.
 * @param c Character to append.
 * @return `CF_OK` on success or `CF_ERR_OOM` when growth fails.
 */
cf_status cf_string_append_char(cf_string *dst, char c);

/**
 * @brief Append a null-terminated C string to the end of a string.
 *
 * The source text is copied into the destination and the destination remains
 * null-terminated after the append.
 *
 * @param dst Destination string receiving the copied text.
 * @param c Null-terminated C string to append.
 * @return `CF_OK` on success or `CF_ERR_OOM` when growth fails.
 */
cf_status cf_string_append_cstr(cf_string *dst, char *c);

/**
 * @brief Append one framework string to another.
 *
 * The source string data is copied into the destination string tail and the
 * destination remains null-terminated after the append.
 *
 * @param dst Destination string receiving the copied text.
 * @param src Source string to append.
 * @return `CF_OK` on success or `CF_ERR_OOM` when growth fails.
 */
cf_status cf_string_append_str(cf_string *dst, cf_string *src);

/**
 * @brief Replace a string's contents with a null-terminated C string.
 *
 * The destination is reset before the source text is copied, and the
 * destination remains null-terminated afterward.
 *
 * @param dst Destination string to overwrite.
 * @param src Null-terminated C string to copy.
 * @return `CF_OK` on success or `CF_ERR_OOM` when growth fails.
 */
cf_status cf_string_from_cstr(cf_string *dst, char *src);

/**
 * @brief Copy a framework string into a newly allocated C string.
 *
 * The returned string is allocated with the source string's allocator and is
 * null-terminated.
 *
 * @param cdst Output pointer receiving the allocated C string.
 * @param src Source string to copy.
 * @return `CF_OK` on success.
 */
cf_status cf_string_as_cstr(char **cdst, cf_string *src);

/**
 * @brief Shrink the logical length of a string.
 *
 * The backing allocation is not changed. Truncation is only valid when `len`
 * is less than or equal to the current logical length, and the resulting
 * string is null-terminated.
 *
 * @param str String to truncate.
 * @param len New logical string length.
 * @return `CF_OK` on success or `CF_ERR_BOUNDS` when `len` exceeds the current
 * logical length.
 */
cf_status cf_string_trunc(cf_string *str, cf_usize len);

/**
 * @brief Report whether a string currently contains no logical characters.
 *
 * @param str String to inspect.
 * @return `CF_TRUE` when the string length is zero, otherwise `CF_FALSE`.
 */
cf_bool cf_string_is_empty(cf_string *str);

/**
 * @brief Print a diagnostic summary of a string's current state.
 *
 * The printed information includes the string data, logical length,
 * allocation capacity, and whether each allocator callback field is set.
 *
 * @param str String to inspect and print.
 */
void cf_string_info(cf_string *str);

/**
 * @brief Compare two strings for byte-for-byte equality.
 *
 * @param str1 First string to compare.
 * @param str2 Second string to compare.
 * @return `CF_TRUE` when both strings contain the same bytes, otherwise
 * `CF_FALSE`.
 */
cf_bool cf_string_eq(cf_string *str1, cf_string *str2);

/**
 * @brief Check whether a string contains a character.
 *
 * @param str String to search.
 * @param c Character to find.
 * @return `CF_TRUE` when the string contains `c`, otherwise `CF_FALSE`.
 */
cf_bool cf_string_contains_char(cf_string *str, char c);

/**
 * @brief Check whether a string contains a C string.
 *
 * @param str String to search.
 * @param c Null-terminated C string to find.
 * @return `CF_TRUE` when the string contains `c`, otherwise `CF_FALSE`.
 */
cf_bool cf_string_contains_cstr(cf_string *str, char *c);

/**
 * @brief Check whether one framework string contains another.
 *
 * @param str1 String to search.
 * @param str2 String to find.
 * @return `CF_TRUE` when `str1` contains `str2`, otherwise `CF_FALSE`.
 */
cf_bool cf_string_contains_str(cf_string *str1, cf_string *str2);

/**
 * @brief Read one character from a string by index.
 *
 * @param str Source string to inspect.
 * @param index Zero-based character index to read.
 * @param c Output pointer receiving the selected character.
 * @return `CF_OK` on success or `CF_ERR_BOUNDS` when `index` is outside the
 * current logical length.
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
 * @return `CF_OK` on success or `CF_ERR_BOUNDS` when `index` is outside the
 * current logical length.
 */
cf_status cf_string_str_at(cf_string *str, cf_usize index, char **c);

/**
 * @brief Remove leading whitespace characters from a string.
 *
 * Whitespace is limited to space, tab, carriage return, and newline.
 *
 * @param str String to trim.
 * @return `CF_OK` on success.
 */
cf_status cf_string_trim_left(cf_string *str);

/**
 * @brief Remove trailing whitespace characters from a string.
 *
 * Whitespace is limited to space, tab, carriage return, and newline.
 *
 * @param str String to trim.
 * @return `CF_OK` on success.
 */
cf_status cf_string_trim_right(cf_string *str);

/**
 * @brief Remove leading and trailing whitespace characters from a string.
 *
 * @param str String to trim.
 * @return `CF_OK` on success.
 */
cf_status cf_string_trim(cf_string *str);

/**
 * @brief Remove all ASCII whitespace characters from a string.
 *
 * @param str String to strip.
 * @return `CF_OK` on success.
 */
cf_status cf_string_strip(cf_string *str);

/**
 * @brief Replace every matching character in a string.
 *
 * @param str String to mutate.
 * @param targetc Character to replace.
 * @param newc Replacement character.
 * @return `CF_OK` on success.
 */
cf_status cf_string_replace(cf_string *str, char targetc, char newc);

/**
 * @brief Copy a substring into a newly allocated C string.
 *
 * The selected range is inclusive on both ends.
 *
 * @param dst Output pointer receiving the allocated C string.
 * @param src Source string to slice.
 * @param start Inclusive start index.
 * @param end Inclusive end index.
 * @return `CF_OK` on success, `CF_ERR_INVALID` when `start > end`, or
 * `CF_ERR_BOUNDS` when the requested range exceeds the current logical length.
 */
cf_status cf_string_slice(char **dst, cf_string *src, cf_usize start, cf_usize end);

/**
 * @brief Split a string on a delimiter and append each token to an array.
 *
 * Empty segments are skipped. Each token is copied into newly allocated
 * null-terminated storage using the source string allocator.
 *
 * @param dst Destination array receiving token elements.
 * @param src Source string to split.
 * @param c Delimiter character.
 * @return `CF_OK` on success, `CF_ERR_NULL` when token allocation fails, or
 * another status propagated from array growth.
 */
cf_status cf_string_split(cf_array *dst, cf_string *src, char c);

#endif /* CF_STRING_H */
