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

#include "ALLOCATOR/cf_arena.h"

#include <string.h>

static cf_usize cf_arena_align_forward(cf_usize value, cf_usize alignment)
{
  cf_usize mask = alignment - 1U;
  return (value + mask) & ~mask;
}

cf_status cf_arena_init(cf_arena *arena, cf_usize capacity, cf_alloc *allocator)
{
  cf_alloc local_allocator;
  void *data = CF_NULL;

  if(arena == CF_NULL) return CF_ERR_NULL;
  memset(arena, 0, sizeof(*arena));

  if(allocator == CF_NULL)
  {
    cf_alloc_new(&local_allocator);
    allocator = &local_allocator;
  }
  if(allocator->alloc == CF_NULL || allocator->free == CF_NULL) return CF_ERR_STATE;

  if(capacity != 0)
  {
    data = allocator->alloc(allocator->ctx, capacity);
    if(data == CF_NULL) return CF_ERR_OOM;
  }

  arena->data = data;
  arena->capacity = capacity;
  arena->allocator = *allocator;
  arena->owns_data = CF_TRUE;
  return CF_OK;
}

cf_status cf_arena_init_with_buffer(cf_arena *arena, void *buffer, cf_usize capacity)
{
  if(arena == CF_NULL) return CF_ERR_NULL;
  if(buffer == CF_NULL && capacity != 0) return CF_ERR_NULL;

  memset(arena, 0, sizeof(*arena));
  arena->data = buffer;
  arena->capacity = capacity;
  arena->owns_data = CF_FALSE;
  return CF_OK;
}

cf_status cf_arena_alloc(cf_arena *arena, cf_usize size, cf_usize alignment, void **ptr)
{
  cf_usize offset = 0;

  if(arena == CF_NULL || ptr == CF_NULL) return CF_ERR_NULL;
  *ptr = CF_NULL;
  if(size == 0) return CF_OK;
  if(arena->data == CF_NULL) return CF_ERR_STATE;

  if(alignment == 0) alignment = sizeof(void *);
  if((alignment & (alignment - 1U)) != 0) return CF_ERR_INVALID;
  if(arena->offset > (cf_usize)-1 - (alignment - 1U)) return CF_ERR_OVERFLOW;

  offset = cf_arena_align_forward(arena->offset, alignment);
  if(offset > (cf_usize)-1 - size) return CF_ERR_OVERFLOW;
  if(offset + size > arena->capacity) return CF_ERR_BOUNDS;

  *ptr = (void *)((cf_u8 *)arena->data + offset);
  arena->offset = offset + size;
  if(arena->offset > arena->high_water) arena->high_water = arena->offset;

  return CF_OK;
}

void cf_arena_reset(cf_arena *arena)
{
  if(arena == CF_NULL) return;
  arena->offset = 0;
}

void cf_arena_destroy(cf_arena *arena)
{
  if(arena == CF_NULL) return;
  if(arena->owns_data == CF_TRUE && arena->data != CF_NULL && arena->allocator.free != CF_NULL)
    arena->allocator.free(arena->allocator.ctx, arena->data);
  memset(arena, 0, sizeof(*arena));
}
