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

#include "MEMORY/cf_memory.h"

#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/*
 * Compute the next byte-buffer capacity. Small buffers grow geometrically while
 * larger buffers grow in fixed framework chunks to limit realloc churn.
 */
static cf_usize cf_buffer_grow_size(cf_buffer *buffer, cf_usize required)
{
  cf_usize min_cap = buffer->len + required;
  cf_usize new_cap = buffer->cap ? buffer->cap > CF_MEMORY_GROWTH_SIZE ? buffer->cap + CF_MEMORY_GROWTH_SIZE : buffer->cap * 2 : 1;
  if(buffer->cap > CF_USIZE_MAX - CF_MEMORY_GROWTH_SIZE) new_cap = CF_USIZE_MAX;

  if(new_cap < min_cap) new_cap = min_cap;

  return min_cap < buffer->len ? CF_USIZE_MAX : new_cap;
}

/*
 * Validate the core cf_buffer invariant: NULL storage means zero length and
 * capacity; live storage means capacity covers length.
 */
cf_bool cf_buffer_is_valid(cf_buffer *buffer)
{
  if(buffer == CF_NULL) return CF_FALSE;
  CF_ASSERT_TYPE_SIZE(*buffer, cf_buffer);
  if(buffer->data == CF_NULL)
  {
    if(buffer->cap != 0) return CF_FALSE;
    if(buffer->len != 0) return CF_FALSE;
  }
  else
  {
    if(buffer->cap == 0) return CF_FALSE;
    if(buffer->cap < buffer->len) return CF_FALSE;
  }
  return CF_TRUE;
}

/*
 * Initialize a byte buffer with the default allocator and optional starting
 * capacity.
 */
cf_status cf_buffer_init(cf_buffer *buffer, cf_usize capacity)
{
  if(buffer == CF_NULL) return CF_ERR_NULL;

  *buffer = (cf_buffer) {0};
  cf_alloc_new(&buffer->allocator);

  if(capacity == 0) return CF_OK;

  void *ptr = buffer->allocator.alloc(buffer->allocator.ctx, capacity);
  if(ptr == CF_NULL) return CF_ERR_OOM;
  buffer->data = (cf_u8 *)ptr; 
  buffer->cap = capacity;

  return CF_OK;
}

/*
 * Ensure a byte buffer has at least the requested capacity.
 */
cf_status cf_buffer_reserve(cf_buffer *buffer, cf_usize capacity)
{ 
  if(buffer == CF_NULL) return CF_ERR_NULL;
  if(cf_buffer_is_valid(buffer) == CF_FALSE) return CF_ERR_STATE;
  if(buffer->allocator.realloc == CF_NULL) return CF_ERR_STATE;

  if(capacity > buffer->cap)
  {
    void *ptr = buffer->allocator.realloc(buffer->allocator.ctx, buffer->data, capacity);
    if(ptr == CF_NULL) return CF_ERR_OOM;
    buffer->data = (cf_u8 *)ptr; 
    buffer->cap = capacity;
  }
  return CF_OK;
}

/*
 * Release byte-buffer storage through its allocator and reset the object.
 */
void cf_buffer_destroy(cf_buffer *buffer)
{
  if(buffer == CF_NULL) return;
  if(buffer->allocator.free == CF_NULL)
  {
    *buffer = (cf_buffer) {0};
    return;
  }
  buffer->allocator.free(buffer->allocator.ctx,buffer->data);
  *buffer = (cf_buffer) {0};
}

/*
 * Append one byte, growing storage when needed.
 */
cf_status cf_buffer_append_byte(cf_buffer *buffer, cf_u8 byte)
{
  if(buffer == CF_NULL) return CF_ERR_NULL;
  if(cf_buffer_is_valid(buffer) == CF_FALSE) return CF_ERR_STATE;

  cf_status state = CF_OK;
  if(buffer->len == buffer->cap)
  {
    state = cf_buffer_reserve(buffer, cf_buffer_grow_size(buffer, 1));
    if(state != CF_OK) return state;
  }

  buffer->data[buffer->len] = byte;
  buffer->len++;

  return state;
}

/*
 * Append a byte span to the buffer. The source is a view and is not retained.
 */
cf_status cf_buffer_append_bytes(cf_buffer *buffer, cf_bytes bytes)
{
  if(buffer == CF_NULL) return CF_ERR_NULL;
  if(cf_buffer_is_valid(buffer) == CF_FALSE) return CF_ERR_STATE;
  if(bytes.len == 0) return CF_OK;
  if(bytes.data == CF_NULL) return CF_ERR_NULL;

  cf_status state = CF_OK;

  if(bytes.len > buffer->cap - buffer->len)
  {
    state = cf_buffer_reserve(buffer, cf_buffer_grow_size(buffer, bytes.len));
    if(state != CF_OK) return state;
  }

  memmove(buffer->data + buffer->len, (cf_u8 *)bytes.data, bytes.len);
  buffer->len += bytes.len;
  return state;
}

/*
 * Expose a checked slice of the buffer as a cf_bytes view.
 */
cf_status cf_buffer_as_bytes(cf_buffer *buffer, cf_bytes *bytes, cf_usize start, cf_usize end)
{
  if(buffer == CF_NULL || bytes == CF_NULL) return CF_ERR_NULL;
  if(cf_buffer_is_valid(buffer) == CF_FALSE) return CF_ERR_STATE;
  if(start > end) return CF_ERR_INVALID;
  if(end >= buffer->len) return CF_ERR_BOUNDS;

  bytes->data = buffer->data + start;
  bytes->elem_size = sizeof(cf_u8);
  bytes->len = end - start + 1;
  return CF_OK;
}

/*
 * Clear logical contents while keeping allocated capacity for reuse.
 */
void cf_buffer_reset(cf_buffer *buffer)
{
  if(buffer == CF_NULL) return;
  buffer->len = 0;
}

/*
 * Truncate logical length without changing capacity.
 */
cf_status cf_buffer_trunc(cf_buffer *buffer, cf_usize len)
{
  if(buffer == CF_NULL) return CF_ERR_NULL;
  if(cf_buffer_is_valid(buffer) == CF_FALSE) return CF_ERR_STATE;
  if(len > buffer->len) return CF_ERR_BOUNDS;
  buffer->len = len;

  return CF_OK;
}

/*
 * Test whether a buffer currently has no logical bytes.
 */
cf_bool cf_buffer_is_empty(cf_buffer *buffer)
{
  if(buffer == CF_NULL) return CF_FALSE;
  return buffer->len == 0;
}

/*
 * Print byte-buffer internals for debugging allocator and capacity behavior.
 */
void cf_buffer_info(cf_buffer *buffer)
{
  if(buffer == CF_NULL)
  {
    printf("Null pointer to buffer\n");
    return;
  }
  if(buffer->data == CF_NULL)
  {
printf(
"================== Buffer Info ==================\n\
  buffer (data) -> ( NULL )\n\
");
  }
else
{
printf(
"================== Buffer Info ==================\n\
  buffer (data) -> ( pointer:          %-14p )\n\
", buffer->data);
}
  printf
  (
"\
  buffer (len ) -> ( valid data size:  %14zu )\n\
  buffer (cap ) -> ( full buffer size: %14zu )\n\
  buffer(allocator):\n\
    |-> ctx (context) -> (state: %-4s)\n\
    |-> ctx ( alloc ) -> (state: %-4s)\n\
    |-> ctx (realloc) -> (state: %-4s)\n\
    |-> ctx ( free  ) -> (state: %-4s)\n\
=================================================\n", 
    buffer->len, 
    buffer->cap, 
    buffer->allocator.ctx     ? "set" : "NULL",
    buffer->allocator.alloc   ? "set" : "NULL",
    buffer->allocator.realloc ? "set" : "NULL",
    buffer->allocator.free    ? "set" : "NULL"
  );
}
