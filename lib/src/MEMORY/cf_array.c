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

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

static cf_status cf_array_check(cf_array *array)
{
  if(array == CF_NULL) return CF_ERR_NULL;
  CF_ASSERT_TYPE_SIZE(*array, cf_array);
  if(array->len > array->cap) return CF_ERR_STATE;
  return CF_OK;
}

static cf_usize cf_array_grow_size(cf_array *array, cf_usize required)
{
  required = required < CF_MEMORY_GROWTH_SIZE ? 
              (array->cap ? 
                (array->cap < CF_MEMORY_GROWTH_SIZE ? 
                  array->cap * 2
                  : CF_MEMORY_GROWTH_SIZE
                ): 1
              ) : required;
  if(required > (cf_usize)-1 - array->cap) return CF_ERR_OVERFLOW;
  return array->cap + required;
}

cf_status cf_array_init(cf_array *array, cf_usize capacity)
{
  if(array == CF_NULL) return CF_ERR_NULL;
  CF_ASSERT_TYPE_SIZE(*array, cf_array);

  *array = (cf_array) {0};
  cf_alloc_new(&array->allocator);

  if(capacity == 0) return CF_OK;

  cf_array_element *ptr = array->allocator.alloc(array->allocator.ctx, capacity * sizeof(cf_array_element));
  if(ptr == CF_NULL) return CF_ERR_OOM;
  array->data = ptr; 
  array->cap = capacity;
  return CF_OK;
}

cf_status cf_array_reserve(cf_array *array, cf_usize capacity)
{
  cf_status state = cf_array_check(array);
  if(state != CF_OK) return state;

  if(array->data == CF_NULL)
  {
    state = cf_array_init(array, capacity);
    if(state != CF_OK) return state;
  }
  
  if(capacity > array->cap)
  {
    void *ptr = array->allocator.realloc(array->allocator.ctx, array->data, capacity * sizeof(cf_array_element));
    if(ptr == CF_NULL) return CF_ERR_OOM;
    array->data = ptr; 
    array->cap = capacity;
  }
  return CF_OK;
}

void cf_array_destroy(cf_array *array)
{
  cf_status state = cf_array_check(array);
  if(state != CF_OK) return;
  array->allocator.free(array->allocator.ctx, array->data);
  *array = (cf_array) {0};
}

cf_status cf_array_reset(cf_array *array)
{
  cf_status state = cf_array_check(array);
  if(state != CF_OK) return state;
  array->len = 0;
  return CF_OK;
}

cf_status cf_array_peek(cf_array *array, cf_array_element *element)
{
  cf_status state = cf_array_check(array);
  if(state != CF_OK) return state;

  if(element == CF_NULL) return CF_ERR_NULL;
  CF_ASSERT_TYPE_SIZE(*element, cf_array_element);

  if(array->len == 0)
  {
    *element = (cf_array_element){0};
    return CF_OK;
  }
  element->data = array->data[array->len - 1].data;
  element->len = array->data[array->len - 1].len;
  return CF_OK;
}

cf_status cf_array_push(cf_array *array, cf_array_element *element, ...)
{
  cf_status state = cf_array_check(array);
  if(state != CF_OK) return state;
  if (element == CF_NULL) return CF_OK;

  va_list args;
  va_start(args, element);
  cf_array_element *current = element;

  while (current != CF_NULL)
  {
    CF_ASSERT_TYPE_SIZE(*current, cf_array_element);
    if(array->cap == array->len)
    {
      state = cf_array_reserve(array, cf_array_grow_size(array, 1));
      if (state != CF_OK)
      {
        va_end(args);
        return state;
      }
    }
    memmove(array->data + array->len, current, sizeof(cf_array_element));
    array->len++;
    current = va_arg(args, cf_array_element *);
  }
  va_end(args);
  return CF_OK;
}

cf_status cf_array_pop(cf_array *array, cf_array_element *element)
{
  cf_status state = cf_array_peek(array, element);
  if(state != CF_OK) return state;
  if(array->len != 0) array->len--;
  return CF_OK;
  }

cf_status cf_array_get(cf_array *array, cf_usize index, cf_array_element *element)
{
  cf_status state = cf_array_check(array);
  if(state != CF_OK) return state;
  if (element == CF_NULL) return CF_ERR_NULL;
  if(index >= array->len) return CF_ERR_BOUNDS;
  *element = array->data[index];
  return CF_OK;
}

cf_status cf_array_set(cf_array *array, cf_usize index, cf_array_element *element)
{
  cf_status state = cf_array_check(array);
  if(state != CF_OK) return state;
  if (element == CF_NULL) return CF_ERR_NULL;
  if(index >= array->len) return CF_ERR_BOUNDS;
  array->data[index] = *element;
  return CF_OK;
}

cf_bool cf_array_is_valid(cf_array *array)
{
  if(array == CF_NULL) return CF_FALSE;
  CF_ASSERT_TYPE_SIZE(*array, cf_array);
  if(array->data == CF_NULL)
  {
    if(array->cap != 0) return CF_FALSE;
    if(array->len != 0) return CF_FALSE;
  }
  else
  {
    if(array->cap == 0) return CF_FALSE;
    if(array->cap < array->len) return CF_FALSE;
  }
  return CF_TRUE;
}

cf_bool cf_array_is_empty(cf_array *array)
{
  if(!cf_array_is_valid(array)) return CF_FALSE;
  return array->len == 0;
}

void cf_array_info(cf_array *array)
{
  if(array == CF_NULL)
  {
    printf("Null pointer to array\n");
    return;
  }

if(array->len == 0)
{
printf(
  "========================= Array Info =========================\n\
  array (data) -> ( NULL )\n\
");
}
else
{
printf(
"========================= Array Info =========================\n\
  array (|--(data %8zu) -> ( pointer:          %-14p )\n\
         |  -> (type -> %s)\n\
         | \n\
",(cf_usize)0,(void *)array->data, cf_types_as_char(array->data->elem_size));
for (cf_usize i = 1; i < array->len; i++)
printf(
"\
         |--(data %8zu) -> ( pointer:          %-14p )\n\
         |  -> (type -> %s)\n\
         | \n",
  i, 
  (void *) (array->data + i), cf_types_as_char(array->data[i].elem_size));
printf("          ----------------------------------------------------->)\n");
}
printf(
"\
  array (len ) -> ( valid data size:  %14zu )\n\
  array (cap ) -> ( full array size:  %14zu )\n\
  array(allocator):\n\
    |-> ctx (context) -> (state: %-4s)\n\
    |-> ctx ( alloc ) -> (state: %-4s)\n\
    |-> ctx (realloc) -> (state: %-4s)\n\
    |-> ctx ( free  ) -> (state: %-4s)\n\
==============================================================\n",
    array->len, 
    array->cap, 
    array->allocator.ctx     ? "set" : "NULL",
    array->allocator.alloc   ? "set" : "NULL",
    array->allocator.realloc ? "set" : "NULL",
    array->allocator.free    ? "set" : "NULL"
  );
}
