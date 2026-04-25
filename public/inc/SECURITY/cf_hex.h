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

#if !defined(CF_HEX_H)
#define CF_HEX_H

#include "TEXT/cf_string.h"

static const char CF_HEX_TABLE[] =
{
  '0', '1', '2', '3',
  '4', '5', '6', '7',
  '8', '9', 'A', 'B',
  'C', 'D', 'E', 'F',
};

/**
 * @brief Encode bytes as contiguous uppercase hexadecimal text.
 *
 * Each source byte produces two output characters using `0-9` and `A-F`.
 * The output is appended to `dst`; existing string contents are preserved.
 *
 * @param dst Initialized string receiving encoded text.
 * @param src Byte view to encode.
 * @return `CF_OK` on success or another status propagated from string growth.
 */
cf_status cf_hex_encode(cf_string *dst, cf_bytes src);

/**
 * @brief Decode hexadecimal text into bytes.
 *
 * The source length must be even. Valid input characters are `0-9`, `a-f`,
 * and `A-F`. Decoded bytes are appended to `dst`; existing buffer contents are
 * preserved.
 *
 * @param dst Initialized buffer receiving decoded bytes.
 * @param src Hexadecimal string to decode.
 * @return `CF_OK` on success, `CF_ERR_INVALID` for odd length or invalid hex
 * characters, or another status propagated from buffer growth.
 */
cf_status cf_hex_decode(cf_buffer *dst, cf_string *src);

#endif /* CF_HEX_H */
