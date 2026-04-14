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

#include<stdlib.h>

/********************************************************************/
/* allocators                                                       */
/********************************************************************/

static void *cf_arena_alloc(void *ctx, cf_usize size)
{
  if(ctx == CF_NULL || size == 0) return CF_NULL;
  cf_arena *arena = (cf_arena *) ctx;
  if(arena->cap - size < arena->offset) return CF_NULL;
  void *ptr = arena->data + arena->offset;
  arena->last_usable = arena->offset;
  arena->offset += size;
  return ptr;
}

static void *cf_arena_realloc(void *ctx, void *ptr, cf_usize size)
{
  if(ctx == CF_NULL || size == 0) return CF_NULL;
  cf_arena *arena = (cf_arena *) ctx;
  if(arena->cap - arena->offset < size) return CF_NULL;
  if((arena->data + arena->last_usable) != ptr) return CF_NULL;
  arena->offset = arena->last_usable + size;
  return ptr;
}

/********************************************************************/
/* construction                                                     */
/********************************************************************/

static cf_arena cf_arena_create_empty(void)
{
  return (cf_arena) {CF_NULL, 0, 0, 0, (cf_alloc) {CF_NULL, cf_arena_alloc, cf_arena_realloc, CF_NULL}};
}

/********************************************************************/
/* validation                                                       */
/********************************************************************/

cf_bool cf_arena_is_valid(cf_arena *arena)
{
  if(arena == CF_NULL) return CF_FALSE;
  if( arena->data == CF_NULL)
  {
    if(arena->cap != 0) return CF_FALSE;
    if(arena->offset != 0) return CF_FALSE;
    if(arena->last_usable != 0) return CF_FALSE;
  }
  else
  {
    if(arena->cap < arena->offset) return CF_FALSE;
    if(arena->offset < arena->last_usable) return CF_FALSE;
  }
  if(arena->allocator.alloc == CF_NULL) return CF_FALSE;
  if(arena->allocator.realloc == CF_NULL) return CF_FALSE;
  return CF_TRUE;
}

/********************************************************************/
/* lifecycle                                                        */
/********************************************************************/

cf_status cf_arena_new(cf_arena *arena, cf_usize size)
{
  if(arena == CF_NULL) return CF_ERR_NULL;
  *arena = cf_arena_create_empty(); 
  if(size == 0) return CF_OK;
  arena->data = malloc(size);
  if(arena->data == CF_NULL) return CF_ERR_OOM;
  arena->cap = size;
  arena->allocator.ctx = arena;
  return CF_OK;
}

void cf_arena_destroy(cf_arena *arena)
{
  if(arena == CF_NULL) return;
  free(arena->data);
  *arena = cf_arena_create_empty();
}

/********************************************************************/
/* operations                                                       */
/********************************************************************/

cf_status cf_arena_reset(cf_arena *arena)
{
  if(arena == CF_NULL) return CF_ERR_NULL;
  if(!cf_arena_is_valid(arena)) return CF_ERR_STATE;
  arena->offset = 0;
  arena->last_usable = 0;
  return CF_OK;
}