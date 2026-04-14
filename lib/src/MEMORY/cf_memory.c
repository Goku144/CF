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

/* ------------------------------------------------------------------ */
/* Internal helpers                                                    */
/* ------------------------------------------------------------------ */



/* ------------------------------------------------------------------ */
/* Construction                                                        */
/* ------------------------------------------------------------------ */

cf_bytes cf_bytes_create_empty(void)
{
  return (cf_bytes) {CF_NULL, 0};
}

cf_buffer cf_buffer_create_empty(void)
{
  return (cf_buffer) {CF_NULL, 0, 0, cf_alloc_create_empty()};
}

/* ------------------------------------------------------------------ */
/* Validation                                                          */
/* ------------------------------------------------------------------ */

cf_bool cf_bytes_is_valid(cf_bytes *bytes)
{
  if(bytes == CF_NULL) return CF_FALSE;
  if(bytes->data == CF_NULL && bytes->len != 0) return CF_FALSE;
  return CF_TRUE;
}

cf_bool cf_buffer_is_valid(cf_buffer *buffer)
{
  if(buffer == CF_NULL) return CF_FALSE;
  if (buffer->data == CF_NULL)
  {
    if(buffer->len != 0) return CF_FALSE;
    if(buffer->cap != 0) return CF_FALSE;
  }
  else
  {
    if(buffer->cap == 0) return CF_FALSE;
    if(buffer->cap < buffer->len) return CF_FALSE;
  }
  if(!cf_alloc_is_valid(&buffer->allocator)) return CF_FALSE;
  return CF_TRUE;
}

/* ------------------------------------------------------------------ */
/* Emptiness                                                           */
/* ------------------------------------------------------------------ */

cf_bool cf_bytes_is_empty(cf_bytes *bytes)
{
  if(bytes == CF_NULL) return CF_FALSE;
  if(!cf_bytes_is_valid(bytes)) return CF_FALSE;
  return bytes->len == 0 ? CF_TRUE : CF_FALSE;
}

cf_bool cf_buffer_is_empty(cf_buffer *buffer)
{
  if(buffer == CF_NULL) return CF_FALSE;
  if(!cf_buffer_is_valid(buffer)) return CF_FALSE;
  return buffer->len == 0 ? CF_TRUE : CF_FALSE;
}

/* ------------------------------------------------------------------ */
/* Equality                                                            */
/* ------------------------------------------------------------------ */

cf_bool cf_bytes_is_eq(cf_bytes *b1, cf_bytes *b2)
{
  if(b1 == CF_NULL || b2 == CF_NULL) return CF_FALSE;
  if(!cf_bytes_is_valid(b1) || !cf_bytes_is_valid(b2)) 
    return CF_FALSE;
  if(b1->len != b2->len) return CF_FALSE;
  for (cf_usize i = 0; i < b1->len; i++)
    if(b1->data[i] != b2->data[i]) 
      return CF_FALSE;
  return CF_TRUE;
}

cf_bool cf_buffer_is_eq(cf_buffer *buf1, cf_buffer *buf2)
{
  if(buf1 == CF_NULL || buf2 == CF_NULL) return CF_FALSE;
  if(!cf_buffer_is_valid(buf1) || !cf_buffer_is_valid(buf2)) 
    return CF_FALSE;
  if(buf1->len != buf2->len) return CF_FALSE;
  for (cf_usize i = 0; i < buf1->len; i++)
    if(buf1->data[i] != buf2->data[i]) 
      return CF_FALSE;
  return CF_TRUE;
}

/* ------------------------------------------------------------------ */
/* Slicing                                                             */
/* ------------------------------------------------------------------ */

cf_status cf_bytes_slice(cf_bytes *dst_bytes, cf_bytes *src_bytes, cf_usize index, cf_usize size)
{
  if(src_bytes == CF_NULL || dst_bytes == CF_NULL) return CF_ERR_NULL;
  if(!cf_bytes_is_valid(src_bytes)) return CF_ERR_STATE;
  *dst_bytes = cf_bytes_create_empty();
  if(index > src_bytes->len) return CF_ERR_BOUNDS;
  if(size > src_bytes->len - index) return CF_ERR_BOUNDS;
  dst_bytes->data = src_bytes->data + index;
  dst_bytes->len = size;
  return CF_OK;
}

cf_status cf_buffer_slice(cf_bytes *dst_bytes, cf_buffer *src_buffer, cf_usize index, cf_usize size)
{
  if(src_buffer == CF_NULL || dst_bytes == CF_NULL) return CF_ERR_NULL;
  if(!cf_buffer_is_valid(src_buffer)) return CF_ERR_STATE;
  *dst_bytes = cf_bytes_create_empty();
  if(index > src_buffer->len) return CF_ERR_BOUNDS;
  if(size > src_buffer->len - index) return CF_ERR_BOUNDS;
  dst_bytes->data = src_buffer->data + index;
  dst_bytes->len = size;
  return CF_OK;
}

/* ------------------------------------------------------------------ */
/* Fill / Zero                                                         */
/* ------------------------------------------------------------------ */

cf_status cf_bytes_fill(cf_bytes *bytes, cf_u8 fill, cf_usize size)
{
  if(bytes == CF_NULL) return CF_ERR_NULL;
  if(!cf_bytes_is_valid(bytes)) return CF_ERR_STATE;
  if(size > bytes->len) return CF_ERR_BOUNDS;
  for (cf_usize i = 0; i < size; i++)
    bytes->data[i] = fill;
  return CF_OK;
}

cf_status cf_buffer_fill(cf_buffer *buffer, cf_u8 fill, cf_usize size)
{
  if(buffer == CF_NULL) return CF_ERR_NULL;
  if(!cf_buffer_is_valid(buffer)) return CF_ERR_STATE;
  if(size > buffer->len) return CF_ERR_BOUNDS;
  for (cf_usize i = 0; i < size; i++)
    buffer->data[i] = fill;
  return CF_OK;
}

/* ------------------------------------------------------------------ */
/* Buffer lifecycle                                                    */
/* ------------------------------------------------------------------ */

cf_status cf_buffer_init(cf_buffer *buffer, cf_alloc *allocator, cf_usize capacity)
{
  if(buffer == CF_NULL) return CF_ERR_NULL;
  *buffer = cf_buffer_create_empty();
  if(allocator != CF_NULL) 
  {
    if(!cf_alloc_is_valid(allocator)) return CF_ERR_STATE;
    buffer->allocator = *allocator;
  }
  if(capacity == 0) return CF_OK;
  void *ptr = buffer->allocator.alloc(buffer->allocator.ctx, capacity);
  if(ptr == CF_NULL) return CF_ERR_OOM;
  buffer->data = ptr;
  buffer->cap = capacity;
  return CF_OK;
}

cf_status cf_buffer_reserve(cf_buffer *buffer, cf_usize size)
{
  if(buffer == CF_NULL) return CF_ERR_NULL;
  if(!cf_buffer_is_valid(buffer)) return CF_ERR_STATE;
  if(size > buffer->cap)
  {
    if(buffer->allocator.realloc == CF_NULL) return CF_ERR_OOM;
    void *ptr = buffer->allocator.realloc(buffer->allocator.ctx, buffer->data, size);
    if(ptr == CF_NULL) return CF_ERR_OOM;
    buffer->data = ptr;
    buffer->cap = size;
  }
  return CF_OK;
}

void cf_buffer_clear(cf_buffer *buffer)
{
  if(buffer == CF_NULL) return;
  if(!cf_buffer_is_valid(buffer)) return;
  buffer->len = 0;
}

void cf_buffer_destroy(cf_buffer *buffer)
{
  if(buffer == CF_NULL) return;
  if(!cf_buffer_is_valid(buffer)) return;
  if(buffer->allocator.free == CF_NULL) return;
  buffer->allocator.free(buffer->allocator.ctx, buffer->data);
  *buffer = cf_buffer_create_empty();
}

/* ------------------------------------------------------------------ */
/* Buffer append / set                                                 */
/* ------------------------------------------------------------------ */



/* ------------------------------------------------------------------ */
/* Buffer views                                                        */
/* ------------------------------------------------------------------ */



/* ------------------------------------------------------------------ */
/* Buffer truncate                                                     */
/* ------------------------------------------------------------------ */

