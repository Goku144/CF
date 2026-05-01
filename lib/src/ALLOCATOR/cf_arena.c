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

static cf_bool cf_arena_is_power_of_two(cf_usize value)
{
  return value != 0 && (value & (value - 1U)) == 0;
}

static cf_usize cf_arena_max_usize(cf_usize a, cf_usize b)
{
  return a >= b ? a : b;
}

static cf_status cf_arena_make_chunk(cf_alloc *allocator, cf_usize capacity, cf_usize alignment, cf_arena_chunk **chunk)
{
  cf_arena_chunk *node = CF_NULL;
  void *data = CF_NULL;

  if(allocator == CF_NULL || chunk == CF_NULL) return CF_ERR_NULL;
  *chunk = CF_NULL;
  if(allocator->alloc == CF_NULL || allocator->free == CF_NULL) return CF_ERR_STATE;

  node = (cf_arena_chunk *)allocator->alloc(allocator->ctx, sizeof(*node));
  if(node == CF_NULL) return CF_ERR_OOM;
  memset(node, 0, sizeof(*node));

  if(capacity != 0)
  {
    data = cf_alloc_aligned(allocator, alignment, capacity);
    if(data == CF_NULL)
    {
      allocator->free(allocator->ctx, node);
      return CF_ERR_OOM;
    }
  }

  node->data = data;
  node->capacity = capacity;
  *chunk = node;
  return CF_OK;
}

static void cf_arena_sync_public_fields(cf_arena *arena)
{
  if(arena == CF_NULL || arena->current == CF_NULL) return;
  arena->data = arena->current->data;
  arena->capacity = arena->current->capacity;
  arena->offset = arena->current->offset;
  if(arena->current->high_water > arena->high_water)
    arena->high_water = arena->current->high_water;
}

static cf_status cf_arena_grow(cf_arena *arena, cf_usize min_size)
{
  cf_usize capacity = 0;
  cf_arena_chunk *chunk = CF_NULL;

  if(arena == CF_NULL) return CF_ERR_NULL;
  if(arena->growable == CF_FALSE || arena->owns_data == CF_FALSE) return CF_ERR_BOUNDS;

  capacity = arena->next_capacity != 0 ? arena->next_capacity : arena->capacity;
  if(capacity == 0) capacity = 4096U;
  while(capacity < min_size)
  {
    if(capacity > (cf_usize)-1 / 2U) return CF_ERR_OVERFLOW;
    capacity *= 2U;
  }

  if(cf_arena_make_chunk(&arena->allocator, capacity, arena->default_alignment, &chunk) != CF_OK)
    return CF_ERR_OOM;

  if(arena->chunks == CF_NULL)
    arena->chunks = chunk;
  else
    arena->current->next = chunk;
  arena->current = chunk;

  if(capacity <= (cf_usize)-1 / 2U) arena->next_capacity = capacity * 2U;
  else arena->next_capacity = capacity;

  cf_arena_sync_public_fields(arena);
  return CF_OK;
}

cf_status cf_arena_init_ex(cf_arena *arena, cf_usize capacity, cf_usize alignment, cf_bool growable, cf_alloc *allocator)
{
  cf_alloc local_allocator;
  cf_arena_chunk *chunk = CF_NULL;
  cf_status status = CF_OK;

  if(arena == CF_NULL) return CF_ERR_NULL;
  memset(arena, 0, sizeof(*arena));

  if(alignment == 0) alignment = 64U;
  if(alignment < sizeof(void *)) alignment = sizeof(void *);
  if(cf_arena_is_power_of_two(alignment) == CF_FALSE) return CF_ERR_INVALID;

  if(allocator == CF_NULL)
  {
    cf_alloc_new(&local_allocator);
    allocator = &local_allocator;
  }
  if(allocator->alloc == CF_NULL || allocator->free == CF_NULL) return CF_ERR_STATE;

  if(capacity != 0)
  {
    status = cf_arena_make_chunk(allocator, capacity, alignment, &chunk);
    if(status != CF_OK) return status;
  }

  arena->allocator = *allocator;
  arena->owns_data = CF_TRUE;
  arena->growable = growable;
  arena->default_alignment = alignment;
  arena->next_capacity = capacity <= (cf_usize)-1 / 2U ? capacity * 2U : capacity;
  arena->chunks = chunk;
  arena->current = chunk;
  cf_arena_sync_public_fields(arena);
  return CF_OK;
}

cf_status cf_arena_init(cf_arena *arena, cf_usize capacity, cf_alloc *allocator)
{
  return cf_arena_init_ex(arena, capacity, 64U, CF_FALSE, allocator);
}

cf_status cf_arena_init_with_buffer(cf_arena *arena, void *buffer, cf_usize capacity)
{
  if(arena == CF_NULL) return CF_ERR_NULL;
  if(buffer == CF_NULL && capacity != 0) return CF_ERR_NULL;

  memset(arena, 0, sizeof(*arena));
  arena->data = buffer;
  arena->capacity = capacity;
  arena->default_alignment = 64U;
  arena->owns_data = CF_FALSE;
  return CF_OK;
}

cf_status cf_arena_alloc(cf_arena *arena, cf_usize size, cf_usize alignment, void **ptr)
{
  cf_usize offset = 0;
  cf_usize base = 0;
  cf_usize aligned = 0;
  cf_status status = CF_OK;

  if(arena == CF_NULL || ptr == CF_NULL) return CF_ERR_NULL;
  *ptr = CF_NULL;
  if(size == 0) return CF_OK;

  if(alignment == 0) alignment = arena->default_alignment != 0 ? arena->default_alignment : sizeof(void *);
  if(alignment < sizeof(void *)) alignment = sizeof(void *);
  if((alignment & (alignment - 1U)) != 0) return CF_ERR_INVALID;

  for(;;)
  {
    if(arena->current != CF_NULL)
    {
      arena->data = arena->current->data;
      arena->capacity = arena->current->capacity;
      arena->offset = arena->current->offset;
    }
    if(arena->data == CF_NULL)
    {
      status = cf_arena_grow(arena, size + alignment);
      if(status != CF_OK) return arena->owns_data == CF_TRUE ? status : CF_ERR_STATE;
      continue;
    }

    base = (cf_usize)(cf_uptr)arena->data;
    if(base > (cf_usize)-1 - arena->offset) return CF_ERR_OVERFLOW;
    if(base + arena->offset > (cf_usize)-1 - (alignment - 1U)) return CF_ERR_OVERFLOW;
    aligned = cf_arena_align_forward(base + arena->offset, alignment);
    if(aligned < base) return CF_ERR_OVERFLOW;
    offset = aligned - base;
    if(offset > (cf_usize)-1 - size) return CF_ERR_OVERFLOW;
    if(offset + size <= arena->capacity) break;

    status = cf_arena_grow(arena, cf_arena_max_usize(size + alignment, arena->capacity + size));
    if(status != CF_OK) return status;
  }

  *ptr = (void *)((cf_u8 *)arena->data + offset);
  offset += size;

  if(arena->current != CF_NULL)
  {
    arena->current->offset = offset;
    if(arena->current->offset > arena->current->high_water) arena->current->high_water = arena->current->offset;
  }
  arena->offset = offset;
  if(arena->offset > arena->high_water) arena->high_water = arena->offset;

  return CF_OK;
}

void cf_arena_reset(cf_arena *arena)
{
  if(arena == CF_NULL) return;
  for(cf_arena_chunk *chunk = arena->chunks; chunk != CF_NULL; chunk = chunk->next)
    chunk->offset = 0;
  arena->current = arena->chunks;
  arena->offset = 0;
  if(arena->current != CF_NULL)
    cf_arena_sync_public_fields(arena);
}

void cf_arena_destroy(cf_arena *arena)
{
  cf_arena_chunk *chunk = CF_NULL;
  cf_arena_chunk *next = CF_NULL;

  if(arena == CF_NULL) return;
  if(arena->owns_data == CF_TRUE)
  {
    chunk = arena->chunks;
    while(chunk != CF_NULL)
    {
      next = chunk->next;
      if(chunk->data != CF_NULL)
        cf_alloc_aligned_free(&arena->allocator, chunk->data);
      if(arena->allocator.free != CF_NULL)
        arena->allocator.free(arena->allocator.ctx, chunk);
      chunk = next;
    }
  }
  memset(arena, 0, sizeof(*arena));
}
