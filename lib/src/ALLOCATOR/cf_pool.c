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

#include "ALLOCATOR/cf_pool.h"

#include <string.h>

#define CF_POOL_EMPTY ((cf_usize)-1)

static cf_bool cf_pool_is_power_of_two(cf_usize value)
{
  return value != 0 && (value & (value - 1U)) == 0;
}

static cf_usize cf_pool_align_forward(cf_usize value, cf_usize alignment)
{
  cf_usize mask = alignment - 1U;
  return (value + mask) & ~mask;
}

static cf_usize cf_pool_stride(cf_usize block_size, cf_usize alignment)
{
  cf_usize stride = block_size < sizeof(cf_usize) ? sizeof(cf_usize) : block_size;
  if(alignment < sizeof(void *)) alignment = sizeof(void *);
  return cf_pool_align_forward(stride, alignment);
}

static void cf_pool_write_next(cf_pool *pool, cf_usize index, cf_usize next)
{
  memcpy((cf_u8 *)pool->data + index * pool->stride, &next, sizeof(next));
}

static cf_usize cf_pool_read_next(cf_pool *pool, cf_usize index)
{
  cf_usize next = CF_POOL_EMPTY;
  memcpy(&next, (cf_u8 *)pool->data + index * pool->stride, sizeof(next));
  return next;
}

static cf_status cf_pool_build_free_list(cf_pool *pool)
{
  if(pool->block_count == 0)
  {
    pool->free_head = CF_POOL_EMPTY;
    pool->free_count = 0;
    return CF_OK;
  }

  if(pool->data == CF_NULL) return CF_ERR_STATE;
  for(cf_usize i = 0; i < pool->block_count; ++i)
    cf_pool_write_next(pool, i, i + 1U < pool->block_count ? i + 1U : CF_POOL_EMPTY);
  pool->free_head = 0;
  pool->free_count = pool->block_count;
  return CF_OK;
}

cf_status cf_pool_init_ex(cf_pool *pool, cf_usize block_size, cf_usize block_count, cf_usize alignment, cf_alloc *allocator)
{
  cf_alloc local_allocator;
  cf_usize stride = 0;
  void *data = CF_NULL;

  if(pool == CF_NULL) return CF_ERR_NULL;
  memset(pool, 0, sizeof(*pool));
  if(block_size == 0 && block_count != 0) return CF_ERR_INVALID;
  if(alignment == 0) alignment = 64U;
  if(alignment < sizeof(void *)) alignment = sizeof(void *);
  if(cf_pool_is_power_of_two(alignment) == CF_FALSE) return CF_ERR_INVALID;

  if(allocator == CF_NULL)
  {
    cf_alloc_new(&local_allocator);
    allocator = &local_allocator;
  }
  if(allocator->alloc == CF_NULL || allocator->free == CF_NULL) return CF_ERR_STATE;

  stride = cf_pool_stride(block_size, alignment);
  if(block_count != 0)
  {
    if(stride > (cf_usize)-1 / block_count) return CF_ERR_OVERFLOW;
    data = cf_alloc_aligned(allocator, alignment, stride * block_count);
    if(data == CF_NULL) return CF_ERR_OOM;
  }

  pool->data = data;
  pool->block_size = block_size;
  pool->block_count = block_count;
  pool->alignment = alignment;
  pool->stride = stride;
  pool->allocator = *allocator;
  pool->owns_data = CF_TRUE;
  return cf_pool_build_free_list(pool);
}

cf_status cf_pool_init(cf_pool *pool, cf_usize block_size, cf_usize block_count, cf_alloc *allocator)
{
  return cf_pool_init_ex(pool, block_size, block_count, 64U, allocator);
}

cf_status cf_pool_init_with_buffer(cf_pool *pool, void *buffer, cf_usize block_size, cf_usize block_count)
{
  if(pool == CF_NULL) return CF_ERR_NULL;
  if(buffer == CF_NULL && block_count != 0) return CF_ERR_NULL;
  if(block_size == 0 && block_count != 0) return CF_ERR_INVALID;

  memset(pool, 0, sizeof(*pool));
  pool->data = buffer;
  pool->block_size = block_size;
  pool->block_count = block_count;
  pool->alignment = sizeof(void *);
  pool->stride = cf_pool_stride(block_size, pool->alignment);
  pool->owns_data = CF_FALSE;
  return cf_pool_build_free_list(pool);
}

cf_status cf_pool_alloc(cf_pool *pool, void **ptr)
{
  cf_usize index = 0;

  if(pool == CF_NULL || ptr == CF_NULL) return CF_ERR_NULL;
  *ptr = CF_NULL;
  if(pool->free_head == CF_POOL_EMPTY) return CF_ERR_OOM;

  index = pool->free_head;
  pool->free_head = cf_pool_read_next(pool, index);
  pool->free_count--;
  *ptr = (void *)((cf_u8 *)pool->data + index * pool->stride);
  return CF_OK;
}

cf_status cf_pool_free(cf_pool *pool, void *ptr)
{
  cf_usize index = 0;
  cf_uptr base = 0;
  cf_uptr addr = 0;

  if(pool == CF_NULL || ptr == CF_NULL) return CF_ERR_NULL;
  if(pool->data == CF_NULL) return CF_ERR_STATE;

  base = (cf_uptr)pool->data;
  addr = (cf_uptr)ptr;
  if(addr < base || addr >= base + pool->stride * pool->block_count) return CF_ERR_BOUNDS;
  if(((addr - base) % pool->stride) != 0) return CF_ERR_INVALID;

  index = (addr - base) / pool->stride;
  cf_pool_write_next(pool, index, pool->free_head);
  pool->free_head = index;
  if(pool->free_count < pool->block_count) pool->free_count++;
  return CF_OK;
}

void cf_pool_reset(cf_pool *pool)
{
  if(pool == CF_NULL) return;
  CF_UNUSED(cf_pool_build_free_list(pool));
}

void cf_pool_destroy(cf_pool *pool)
{
  if(pool == CF_NULL) return;
  if(pool->owns_data == CF_TRUE && pool->data != CF_NULL && pool->allocator.free != CF_NULL)
    cf_alloc_aligned_free(&pool->allocator, pool->data);
  memset(pool, 0, sizeof(*pool));
}
