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

#include "SECURITY/cf_base64.h"

#include <stdlib.h>
#include <string.h>



static cf_bool cf_base64_is_char(cf_u8 c)
{
  return (c >= 'A' && c <= 'Z') ||
         (c >= 'a' && c <= 'z') ||
         (c >= '0' && c <= '9') ||
         c == '+' ||
         c == '/';
}

cf_status cf_base64_encode(cf_string *dst, cf_bytes src)
{
  if(dst == CF_NULL) return CF_ERR_NULL;
  if(cf_string_is_valid(dst) == CF_FALSE) return CF_ERR_STATE;
  if(src.len == 0) return CF_OK;
  if(src.data == CF_NULL) return CF_ERR_NULL;
  if(src.elem_size != sizeof(cf_u8)) return CF_ERR_INVALID;
  if(src.len > CF_USIZE_MAX - 2) return CF_ERR_OVERFLOW;

  cf_usize rem = src.len % 3;
  cf_usize padd_size = (rem == 0) ? 0 : (3 - rem);

  cf_bytes tmp = {.data = malloc(src.len + padd_size), .elem_size = 1, .len = src.len + padd_size};
  if(tmp.data == CF_NULL) return CF_ERR_OOM;
  memcpy(tmp.data, src.data, src.len);

  for (cf_usize i = 0; i < padd_size; i++) ((cf_u8 *)tmp.data)[src.len + i] = '\0';
  cf_usize loop_size = tmp.len / 3;

  for (cf_usize i = 0; i < loop_size; i++)
  {
    char c[] = 
    {
      CF_BASE64_TABLE[(((cf_u8 *)tmp.data)[3 * i] >> 2) & 0x3F],
      CF_BASE64_TABLE[((((cf_u8 *)tmp.data)[3 * i + 1] >> 4) & 0x0F) | ((((cf_u8 *)tmp.data)[3 * i] << 4    ) & 0x30)],
      CF_BASE64_TABLE[((((cf_u8 *)tmp.data)[3 * i + 2] >> 6) & 0x03) | ((((cf_u8 *)tmp.data)[3 * i + 1] << 2) & 0x3C)],
      CF_BASE64_TABLE[(((cf_u8 *)tmp.data)[3 * i + 2] ) & 0x3F],
      '\0',
    };
    cf_status state = cf_string_append_cstr(dst, c);
    if(state != CF_OK) {free(tmp.data); return state;}
  }
  for (cf_usize i = 0; i < padd_size; i++) dst->data[dst->len -padd_size + i] = '=';
  free(tmp.data);
  return CF_OK;
}

cf_status cf_base64_decode(cf_buffer *dst, cf_string *src)
{
  if(dst == CF_NULL || src == CF_NULL) return CF_ERR_NULL;
  if(cf_buffer_is_valid(dst) == CF_FALSE || cf_string_is_valid(src) == CF_FALSE) return CF_ERR_STATE;

  if(src->len % 4 != 0 ) return CF_ERR_INVALID;
  if(src->len == 0) return CF_OK;

  cf_usize len = src->len / 4;
  for (cf_usize i = 0; i < len; i++)
  {
    cf_bool last_chunk = i == len - 1;
    cf_u8 c0 = src->data[4 * i];
    cf_u8 c1 = src->data[4 * i + 1];
    cf_u8 c2 = src->data[4 * i + 2];
    cf_u8 c3 = src->data[4 * i + 3];

    if(!cf_base64_is_char(c0) || !cf_base64_is_char(c1)) return CF_ERR_INVALID;
    if(c2 == '=' && (!last_chunk || c3 != '=')) return CF_ERR_INVALID;
    if(c3 == '=' && !last_chunk) return CF_ERR_INVALID;
    if(c2 != '=' && !cf_base64_is_char(c2)) return CF_ERR_INVALID;
    if(c3 != '=' && !cf_base64_is_char(c3)) return CF_ERR_INVALID;

    cf_u8 b[] =
    {
      ((CF_BASE64_INV_TABLE[src->data[4 * i]]     << 2) & 0xFC) | ((CF_BASE64_INV_TABLE[src->data[4 * i + 1]] >> 4) & 0x03),
      ((CF_BASE64_INV_TABLE[src->data[4 * i + 2]] >> 2) & 0x0F) | ((CF_BASE64_INV_TABLE[src->data[4 * i + 1]] << 4) & 0xF0),
      ((CF_BASE64_INV_TABLE[src->data[4 * i + 3]]     ) & 0x3F) | ((CF_BASE64_INV_TABLE[src->data[4 * i + 2]] << 6) & 0xC0),
    };
    cf_bytes tmp = {.data = b, .elem_size = 1, .len = 3};
    cf_status state = cf_buffer_append_bytes(dst, tmp);
    if(state != CF_OK) return state;
  }
  dst->len -= (+src->data[src->len - 1] == '=') ? (src->data[src->len - 2] == '=') ? 2 : 1 : 0;
  return CF_OK;
}
