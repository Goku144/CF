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

#include "TEXT/cf_ascii.h"

/*
 * Test for an ASCII alphabetic byte without locale-dependent behavior.
 */
cf_bool cf_ascii_is_alpha(char c)
{
  return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z');
}

/*
 * Test for an ASCII decimal digit.
 */
cf_bool cf_ascii_is_digit(char c)
{
  return '0' <= c && c <= '9';
}

/*
 * Test for an ASCII letter or decimal digit.
 */
cf_bool cf_ascii_is_alnum(char c)
{
  return cf_ascii_is_alpha(c) || cf_ascii_is_digit(c);
}

/*
 * Test for ASCII whitespace used by framework string trimming.
 */
cf_bool cf_ascii_is_space(char c)
{
  return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

/*
 * Test for an uppercase ASCII letter.
 */
cf_bool cf_ascii_is_upper(char c)
{
  return ('A' <= c && c <= 'Z');
}

/*
 * Test for a lowercase ASCII letter.
 */
cf_bool cf_ascii_is_lower(char c)
{
  return ('a' <= c && c <= 'z');
}

/*
 * Convert a lowercase ASCII letter to uppercase; other bytes pass through.
 */
char cf_ascii_to_upper(char c)
{
  return cf_ascii_is_lower(c) ? c - 'a' + 'A' : c;
}

/*
 * Convert an uppercase ASCII letter to lowercase; other bytes pass through.
 */
char cf_ascii_to_lower(char c)
{
  return cf_ascii_is_upper(c) ? c - 'A' + 'a' : c;
}

/*
 * Decode one ASCII hexadecimal digit for security encoders and parsers.
 */
cf_isize cf_ascii_hex_value(char c)
{
  if('0' <= c && c <= '9') return c - '0';
  if('a' <= c && c <= 'f') return 0xA + c - 'a';
  if('A' <= c && c <= 'F') return 0xA + c - 'A';
  return -1;
}
