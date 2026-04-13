#include "ALLOCATOR/cf_arena.h"

#include<stdlib.h>

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

static void cf_arena_free(void *ctx, void *ptr)
{
  CF_UNUSED(ctx);
  CF_UNUSED(ptr);
}

static cf_arena cf_arena_create_empty()
{
  return (cf_arena) {CF_NULL, 0, 0, 0, (cf_alloc) {CF_NULL, cf_arena_alloc, cf_arena_realloc, cf_arena_free}};
}

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
  if(!cf_alloc_is_valid(&arena->allocator)) return CF_FALSE;
  return CF_TRUE;
}

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

cf_status cf_arena_reset(cf_arena *arena)
{
  if(arena == CF_NULL) return CF_ERR_NULL;
  if(!cf_arena_is_valid(arena)) return CF_ERR_STATE;
  arena->offset = 0;
  arena->last_usable = 0;
  return CF_OK;
}

void cf_arena_destroy(cf_arena *arena)
{
  if(arena == CF_NULL) return;
  free(arena->data);
  *arena = cf_arena_create_empty();
}