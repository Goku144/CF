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

#if !defined(CF_BASE64_H)
#define CF_BASE64_H

#include "RUNTIME/cf_types.h"

#include "TEXT/cf_string.h"

static const char CF_BASE64_TABLE[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static const cf_u8 CF_BASE64_INV_TABLE[] =
{
  0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,
  0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,
  0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,62,0 ,0 ,0 ,63,
  52,53,54,55,56,57,58,59,60,61,0 ,0 ,0 ,0 ,0 ,0 ,
  0 ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10,11,12,13,14,
  15,16,17,18,19,20,21,22,23,24,25,0 ,0 ,0 ,0 ,0 ,
  0 ,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
  41,42,43,44,45,46,47,48,49,50,51,0 ,0 ,0 ,0 ,0 ,
  0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,
  0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,
  0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,
  0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,
  0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,
  0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,
  0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,
  0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,
};

/**
 * @brief Encode bytes as Base64 text.
 *
 * Each group of three source bytes produces four output characters using the
 * standard Base64 alphabet `A-Z`, `a-z`, `0-9`, `+`, and `/`. When the input
 * length is not divisible by three, the output is padded with `=` characters.
 * The output is appended to `dst`; existing string contents are preserved.
 *
 * @param dst Initialized string receiving encoded text.
 * @param src Byte view to encode. `src.elem_size` must be `1`.
 * @return `CF_OK` on success, `CF_ERR_NULL` for null destination or non-empty
 * null source data, `CF_ERR_INVALID` for non-byte input, or another status
 * from string growth.
 */
cf_status cf_base64_encode(cf_string *dst, cf_bytes src);

/**
 * @brief Decode Base64 text into bytes.
 *
 * The source length must be a multiple of four. Valid encoded text uses the
 * standard Base64 alphabet and may end with one or two `=` padding characters.
 * Decoded bytes are appended to `dst`; existing buffer contents are preserved.
 *
 * @param dst Initialized buffer receiving decoded bytes.
 * @param src Valid Base64 string to decode.
 * @return `CF_OK` on success, `CF_ERR_NULL` for null destination,
 * `CF_ERR_STATE` for invalid source string state, `CF_ERR_INVALID` for source
 * lengths that are not a multiple of four, or another status from buffer
 * growth.
 */
cf_status cf_base64_decode(cf_buffer *dst, cf_string *src);

#endif /* CF_BASE64_H */
