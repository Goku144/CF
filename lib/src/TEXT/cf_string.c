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
#include "MEMORY/cf_memory.h"

#include "TEXT/cf_ascii.h"

#include "TEXT/cf_string.h"

#include <stdio.h>
#include <string.h>


static cf_status cf_string_check(cf_string *str)
{
  if(str == CF_NULL) return CF_ERR_NULL;
  CF_ASSERT_TYPE_SIZE(*str, cf_string);
  if(str->len > str->cap) return CF_ERR_STATE;
  if(str->data != CF_NULL && str->data[str->len] != '\0')
    return CF_ERR_STATE;
  return CF_OK;
}

cf_status cf_string_init(cf_string *str, cf_usize capacity)
{
  cf_status state = cf_buffer_init(str, capacity + 1);
  if(state != CF_OK) return state;
  str->data[str->len] = '\0';
  return CF_OK;
}

cf_status cf_string_reserve(cf_string *str, cf_usize capacity)
{
  cf_status state = cf_buffer_reserve(str, capacity + 1);
  if(state != CF_OK) return state;
  str->data[str->len] = '\0';
  return CF_OK;
}

void cf_string_destroy(cf_string *str)
{
  cf_buffer_destroy(str);
}

cf_status cf_string_append_char(cf_string *dst, char c)
{
  cf_status state = cf_buffer_append_byte(dst, (cf_u8) c);
  if(state != CF_OK) return state;
  dst->data[dst->len] = '\0';
  return CF_OK;
}

cf_status cf_string_append_cstr(cf_string *dst, char *c)
{
  cf_status state = cf_buffer_append_bytes(dst, (cf_bytes) {.data = c, .elem_size = 1, .len = strlen(c)});
  if(state != CF_OK) return state;
  if(dst->cap == dst->len)
  {
    state = cf_buffer_reserve(dst, dst->cap + 1);
    if(state != CF_OK) return state;
  }
  dst->data[dst->len] = '\0';
  return CF_OK;
}

cf_status cf_string_append_str(cf_string *dst, cf_string *src)
{
  cf_status state = cf_buffer_append_bytes(dst, (cf_bytes) {.data = src->data, .elem_size = 1, .len = src->len});
  if(state != CF_OK) return state;
  if(dst->cap == dst->len)
  {
    state = cf_buffer_reserve(dst, dst->cap + 1);
    if(state != CF_OK) return state;
  }
  dst->data[dst->len] = '\0';
  return CF_OK;
}

cf_status cf_string_from_cstr(cf_string *dst, char *src)
{
  if(dst == CF_NULL || src == CF_NULL) return CF_ERR_NULL;
  dst->len = 0;
  cf_status state = cf_buffer_append_bytes(dst, (cf_bytes) {.data = src, .elem_size = 1, .len = strlen(src)});
  if(state != CF_OK) return state;
  if(dst->cap == dst->len)
  {
    state = cf_buffer_reserve(dst, dst->cap + 1);
    if(state != CF_OK) return state;
  }
  dst->data[dst->len] = '\0';
  return CF_OK;
}

cf_status cf_string_as_cstr(char **cdst, cf_string *src)
{
  cf_status state = cf_string_check(src);
  if(state != CF_OK) return state;

  if(cdst == CF_NULL) return CF_ERR_NULL;

  *cdst = src->allocator.alloc(src->allocator.ctx, src->len + 1);
  memcpy(*cdst, src->data, src->len);
  (*cdst)[src->len] = '\0';
  return CF_OK;
}

cf_status cf_string_reset(cf_string *str)
{
  cf_status state = cf_string_check(str);
  if(state != CF_OK) return state;
  str->len = 0;
  if(str->data != CF_NULL)
    str->data[str->len] = '\0';
  return CF_OK;
}

cf_status cf_string_trunc(cf_string *str, cf_usize len)
{
  cf_status state = cf_buffer_trunc(str, len);
  if(state != CF_OK) return state;
  str->data[len] = '\0';
  return CF_OK;
}

cf_bool cf_string_is_valid(cf_string *str)
{
  if(str == CF_NULL) return CF_FALSE;
  if(str->data == CF_NULL)
  {
    if(str->cap != 0) return CF_FALSE;
    if(str->len != 0) return CF_FALSE;
  }
  else
  {
    if(str->cap == 0) return CF_FALSE;
    if(str->cap < str->len) return CF_FALSE;
    if(str->data[str->len] != '\0') return CF_FALSE;
  }
  return CF_TRUE;
}

cf_bool cf_string_is_empty(cf_string *str)
{
  if(!cf_string_is_valid(str)) return CF_FALSE;
  return str->len == 0;
}

void cf_string_info(cf_string *str)
{
  if(str == CF_NULL)
  {
    printf("Null pointer to string\n");
    return;
  }
  if(str->data == CF_NULL)
  {
printf(
"================== String Info ==================\n\
  string (data) -> ( NULL )\n\
");
  }
else
{
printf(
"================== String Info ==================\n\
  string (data): \n\
  {\n\
    %s\n\
  }\n\
", str->data);
}
  printf
  (
"\
  string (len ) -> ( valid data size:  %14zu )\n\
  string (cap ) -> ( full string size: %14zu )\n\
  string(allocator):\n\
    |-> ctx (context) -> (state: %-4s)\n\
    |-> ctx ( alloc ) -> (state: %-4s)\n\
    |-> ctx (realloc) -> (state: %-4s)\n\
    |-> ctx ( free  ) -> (state: %-4s)\n\
=================================================\n", 
    str->len, 
    str->cap, 
    str->allocator.ctx     ? "set" : "NULL",
    str->allocator.alloc   ? "set" : "NULL",
    str->allocator.realloc ? "set" : "NULL",
    str->allocator.free    ? "set" : "NULL"
  );
}

cf_bool cf_string_eq(cf_string *str1, cf_string *str2)
{
  cf_status state = cf_string_check(str1) | cf_string_check(str2);
  if(state != CF_OK) return CF_FALSE;
  if(str1->len != str2->len) return CF_FALSE;
  if(str1->data == CF_NULL) return str2->data == CF_NULL;
  if(str2->data == CF_NULL) return str1->data == CF_NULL;
  return memcmp(str1->data, str2->data, str1->len) == 0;
}

cf_bool cf_string_contains_char(cf_string *str, char c)
{
  cf_status state = cf_string_check(str);
  if(state != CF_OK) return CF_FALSE;
  if(str->data == CF_NULL) return CF_FALSE;
  if(strchr((char *) str->data, c) != CF_NULL) return CF_TRUE;
  return CF_FALSE;
}

cf_bool cf_string_contains_cstr(cf_string *str, char *c)
{
  if(c == CF_NULL) return CF_ERR_NULL;
  cf_status state = cf_string_check(str);
  if(state != CF_OK) return CF_FALSE;
  if(str->data == CF_NULL) return CF_FALSE;
  if(strstr((char *) str->data, c) != CF_NULL) return CF_TRUE;
  return CF_FALSE;
}

cf_bool cf_string_contains_str(cf_string *str1, cf_string *str2)
{
  cf_status state = cf_string_check(str1) | cf_string_check(str2);
  if(state != CF_OK) return CF_FALSE;
  if(str1->len < str2->len) return CF_FALSE;
  if(str1->data == CF_NULL) return str2->data == CF_NULL;
  if(str2->data == CF_NULL) return str1->data == CF_NULL;
  if(strstr((char *) str1->data, (char *) str2->data) != CF_NULL) 
    return CF_TRUE;
  return CF_FALSE;
}

cf_status cf_string_char_at(cf_string *str, cf_usize index, char *c)
{
  if(c == CF_NULL) return CF_ERR_NULL;
  cf_status state = cf_string_check(str);
  if(state != CF_OK) return CF_FALSE;
  if(str->len <= index) return CF_ERR_BOUNDS;
  *c = (char) str->data[index];
  return CF_OK;
}

cf_status cf_string_str_at(cf_string *str, cf_usize index, char **c)
{
  if(c == CF_NULL) return CF_ERR_NULL;
  cf_status state = cf_string_check(str);
  if(state != CF_OK) return CF_FALSE;
  if(str->len <= index) return CF_ERR_BOUNDS;
  cf_usize len = str->len - index + 1;
  *c = str->allocator.alloc(str->allocator.ctx, len);
  memcpy(*c, str->data + index, len);
  return CF_OK;
}

cf_status cf_string_trim_left(cf_string *str)
{
  cf_status state = cf_string_check(str);
  if(state != CF_OK) return CF_FALSE;
  cf_usize index = 0; 
  while(index < str->len && cf_ascii_is_space(str->data[index]))
    index++;
  memmove(str->data, str->data + index, str->len - index + 1);
  str->len -= index;
  str->data[str->len] = '\0';
  return CF_OK;
}

cf_status cf_string_trim_right(cf_string *str)
{
  cf_status state = cf_string_check(str);
  if(state != CF_OK) return CF_FALSE;
  cf_isize index = str->len - 1; 
  while(index >= 0 && cf_ascii_is_space(str->data[index]))
    index--;
  str->len = index + 1;
  str->data[str->len] = '\0';
  return CF_OK;
}

cf_status cf_string_trim(cf_string *str)
{
  cf_status state = cf_string_trim_right(str);
  if(state != CF_OK) return CF_FALSE;
  return cf_string_trim_left(str);
}

cf_status cf_string_strip(cf_string *str)
{
  cf_status state = cf_string_check(str);
  if(state != CF_OK) return CF_FALSE;
  cf_usize index = 0, reel_index = 0; 
  while(index < str->len)
  {
    if(!cf_ascii_is_space(str->data[index]))
    {
      str->data[reel_index] = str->data[index];
      reel_index++;
    }    
    index++;
  }
  str->len = reel_index;
  str->data[str->len] = '\0';
  return CF_OK;
}

cf_status cf_string_replace(cf_string *str, char targetc, char newc)
{
  cf_status state = cf_string_check(str);
  if(state != CF_OK) return CF_FALSE;
  cf_usize index = 0; 
  while(index < str->len)
  {
    if(str->data[index] == targetc)
      str->data[index] = newc;
    index++;
  }
  str->data[str->len] = '\0';
  return CF_OK;
}

cf_status cf_string_slice(char **dst, cf_string *src, cf_usize start, cf_usize end)
{
  if(dst == CF_NULL) return CF_ERR_NULL;
  cf_status state = cf_string_check(src);
  if(state != CF_OK) return CF_FALSE;
  if(start > end) return CF_ERR_INVALID;
  if(end >= src->len) return CF_ERR_BOUNDS;
  cf_usize len = end - start + 1;
  *dst = src->allocator.alloc(src->allocator.ctx, len + 1);
  memcpy(*dst, src->data + start, len);
  (*dst)[len] = '\0';
  return CF_OK;
}

cf_status cf_string_split(cf_array *dst, cf_string *src, char c)
{
  if(dst == CF_NULL) return CF_ERR_NULL;
  if(!cf_string_is_valid(src)) return CF_ERR_STATE;
  cf_usize index = 0, check_point = 0,len = 0;
  while (index <= src->len)
  {
    if(index == src->len || src->data[index] == c)
    {
      len = index - check_point;
      if(len > 0)
      {
        cf_array_element elem = (cf_array_element) 
        {.data = CF_NULL, .elem_size = sizeof (char), .len = len};
        elem.data = src->allocator.alloc(src->allocator.ctx, len + 1);
        if(elem.data == CF_NULL) return CF_ERR_NULL;
        memcpy((char *) elem.data, src->data + check_point, elem.len);
        ((char *) elem.data)[elem.len] = '\0';
        cf_status state = cf_array_push(dst, &elem, CF_NULL);
        if(state != CF_OK) return state;
        check_point = index + 1;
      }
      else check_point = index + 1;
    }
    index++;
  }
  return CF_OK;
}
