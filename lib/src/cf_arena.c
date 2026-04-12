#include "cf_arena.h"

cf_arena cf_arena_empty(void)
{
  return (cf_arena) {CF_NULL, 0, 0, CF_NULL, {0}};
}

cf_bool cf_arena_is_valid(cf_arena arena)
{
  if(arena.data == CF_NULL &&
     arena.cap == 0 &&
     arena.offset == 0 &&
     arena.backing_allocator == CF_NULL &&
     arena.allocator.ctx == CF_NULL &&
     arena.allocator.alloc == CF_NULL &&
     arena.allocator.realloc == CF_NULL &&
     arena.allocator.free == CF_NULL)
    return CF_TRUE;
  if(arena.offset > arena.cap)
    return CF_FALSE;
  if(arena.data == CF_NULL && arena.cap != 0)
    return CF_FALSE;
  if(!cf_allocator_is_valid(arena.backing_allocator))
    return CF_FALSE;
  if(arena.allocator.alloc == CF_NULL ||
     arena.allocator.realloc == CF_NULL ||
     arena.allocator.free == CF_NULL)
    return CF_FALSE;
  return CF_TRUE;
}

cf_bool cf_arena_is_empty(cf_arena arena)
{
  return arena.offset == 0 ? CF_TRUE : CF_FALSE;
}

cf_status cf_arena_init_ex(cf_arena *arena, cf_usize cap, const cf_allocator *allocator)
{
  if(arena == CF_NULL) return CF_ERR_NULL;
  if(!cf_allocator_is_valid(allocator)) return CF_ERR_STATE;
  *arena = cf_arena_empty();
  arena->backing_allocator = allocator;
  arena->allocator.ctx = arena;
  arena->allocator.alloc = cf_arena_alloc;
  arena->allocator.realloc = cf_arena_realloc;
  arena->allocator.free = cf_arena_free;
  if(cap == 0) return CF_OK;
  cf_u8 *tmp = allocator->alloc(allocator->ctx, cap * sizeof(cf_u8));
  if(tmp == CF_NULL)
  {
    *arena = cf_arena_empty();
    return CF_ERR_OOM;
  }
  arena->data = tmp;
  arena->cap = cap;
  arena->offset = 0;
  return CF_OK;
}

cf_status cf_arena_init(cf_arena *arena, cf_usize cap)
{
  return cf_arena_init_ex(arena, cap, cf_default_allocator());
}

void cf_arena_reset(cf_arena *arena)
{
  if(arena == CF_NULL) return;
  if(!cf_arena_is_valid(*arena)) return;
  arena->offset = 0;
}

void cf_arena_destroy(cf_arena *arena)
{
  if(arena == CF_NULL) return;
  if(!cf_arena_is_valid(*arena)) return;
  if(arena->data != CF_NULL)
    arena->backing_allocator->free(arena->backing_allocator->ctx, arena->data);
  *arena = cf_arena_empty();
}

void *cf_arena_alloc(void *ctx, cf_usize size)
{
  if(ctx == CF_NULL) return CF_NULL;
  if(size == 0) return CF_NULL;
  cf_arena *arena = (cf_arena *) ctx;
  if(!cf_arena_is_valid(*arena)) return CF_NULL;
  if((arena->cap - arena->offset) < size) return CF_NULL;
  void *ptr = arena->data + arena->offset;
  arena->offset += size;
  return ptr;
}

void *cf_arena_realloc(void *ctx, void *ptr, cf_usize new_size)
{
  CF_UNUSED(ctx);
  CF_UNUSED(ptr);
  CF_UNUSED(new_size);
  return CF_NULL;
}

void cf_arena_free(void *ctx, void *ptr)
{
  CF_UNUSED(ctx);
  CF_UNUSED(ptr);
}

const cf_allocator *cf_arena_allocator(cf_arena *arena)
{
  if(arena == CF_NULL) return CF_NULL;
  if(!cf_arena_is_valid(*arena)) return CF_NULL;
  return &arena->allocator;
}