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

#include "MEMORY/cf_array.h"

#include "SECURITY/cf_hex.h"

#include "TEXT/cf_ascii.h"

/*
 * Encode raw bytes as lowercase hexadecimal text for security and diagnostic
 * serialization paths.
 */
cf_status cf_hex_encode(cf_string *dst, cf_bytes src)
{
  if(dst == CF_NULL) return CF_ERR_NULL;
  if(cf_string_is_valid(dst) == CF_FALSE) return CF_ERR_STATE;
  if(src.len == 0) return CF_OK;
  if(src.data == CF_NULL) return CF_ERR_NULL;
  if(src.elem_size != sizeof(cf_u8)) return CF_ERR_INVALID;

  for (cf_usize i = 0; i < src.len; i++)
  {
    char c[] = {CF_HEX_TABLE[(((cf_u8 *)src.data)[i] >> 4)& 0x0F], CF_HEX_TABLE[((cf_u8 *)src.data)[i] & 0x0F], '\0'};
    cf_status state = cf_string_append_cstr(dst, c);
    if(state != CF_OK) return state;
  }
  return CF_OK;
}

/*
 * Decode hexadecimal text into raw bytes, rejecting odd lengths and non-hex
 * ASCII digits.
 */
cf_status cf_hex_decode(cf_buffer *dst, cf_string *src)
{
  if(dst == CF_NULL || src == CF_NULL) return CF_ERR_NULL;
  if(cf_buffer_is_valid(dst) == CF_FALSE || cf_string_is_valid(src) == CF_FALSE) return CF_ERR_STATE;

  if(src->len % 2 != 0) return CF_ERR_INVALID;
  cf_usize size = src->len / 2;
  for (size_t i = 0; i < size; i++)
  {
    cf_isize high = cf_ascii_hex_value(src->data[2 * i]);
    cf_isize low = cf_ascii_hex_value(src->data[2 * i + 1]);
    if(high < 0 || low < 0) return CF_ERR_INVALID;
    cf_u8 byte = (cf_u8)(((high << 4) & 0xF0) | (low & 0x0F));
    cf_status state = cf_buffer_append_byte(dst, byte);
    if(state != CF_OK) return state;
  }
  return CF_OK;
}
