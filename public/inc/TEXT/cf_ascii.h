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

#if !defined(CF_ASCII_H)
#define CF_ASCII_H

#include "RUNTIME/cf_types.h"

/**
 * @brief Check whether a character is an ASCII alphabetic letter.
 *
 * @param c Character to inspect.
 * @return `CF_TRUE` when `c` is in `A-Z` or `a-z`, otherwise `CF_FALSE`.
 */
cf_bool cf_ascii_is_alpha(char c);

/**
 * @brief Check whether a character is an ASCII decimal digit.
 *
 * @param c Character to inspect.
 * @return `CF_TRUE` when `c` is in `0-9`, otherwise `CF_FALSE`.
 */
cf_bool cf_ascii_is_digit(char c);

/**
 * @brief Check whether a character is an ASCII letter or decimal digit.
 *
 * @param c Character to inspect.
 * @return `CF_TRUE` when `c` is alphabetic or numeric, otherwise `CF_FALSE`.
 */
cf_bool cf_ascii_is_alnum(char c);

/**
 * @brief Check whether a character is ASCII whitespace.
 *
 * Whitespace includes space, horizontal tab, newline, vertical tab, form feed,
 * and carriage return.
 *
 * @param c Character to inspect.
 * @return `CF_TRUE` when `c` is ASCII whitespace, otherwise `CF_FALSE`.
 */
cf_bool cf_ascii_is_space(char c);

/**
 * @brief Check whether a character is an uppercase ASCII letter.
 *
 * @param c Character to inspect.
 * @return `CF_TRUE` when `c` is in `A-Z`, otherwise `CF_FALSE`.
 */
cf_bool cf_ascii_is_upper(char c);

/**
 * @brief Check whether a character is a lowercase ASCII letter.
 *
 * @param c Character to inspect.
 * @return `CF_TRUE` when `c` is in `a-z`, otherwise `CF_FALSE`.
 */
cf_bool cf_ascii_is_lower(char c);

/**
 * @brief Convert a lowercase ASCII letter to uppercase.
 *
 * Characters outside `a-z` are returned unchanged.
 *
 * @param c Character to convert.
 * @return Uppercase equivalent for lowercase ASCII letters, otherwise `c`.
 */
char cf_ascii_to_upper(char c);

/**
 * @brief Convert an uppercase ASCII letter to lowercase.
 *
 * Characters outside `A-Z` are returned unchanged.
 *
 * @param c Character to convert.
 * @return Lowercase equivalent for uppercase ASCII letters, otherwise `c`.
 */
char cf_ascii_to_lower(char c);

/**
 * @brief Convert one ASCII hexadecimal digit to its numeric value.
 *
 * Valid input characters are `0-9`, `a-f`, and `A-F`.
 *
 * @param c Character to convert.
 * @return The numeric value `0..15` for valid hex digits, otherwise `-1`.
 */
cf_isize cf_ascii_hex_value(char c);

#endif /* CF_ASCII_H */
