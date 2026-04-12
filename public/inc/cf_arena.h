#if !defined(CF_ARENA_H)
#define CF_ARENA_H

#include "cf_types.h"
#include "cf_status.h"
#include "cf_alloc.h"

typedef struct cf_arena
{
  cf_u8 *data;
  cf_usize cap;
  cf_usize offset;
  const cf_allocator *backing_allocator;
  cf_allocator allocator;
} cf_arena;

cf_arena cf_arena_empty(void);

cf_bool cf_arena_is_valid(cf_arena arena);

cf_bool cf_arena_is_empty(cf_arena arena);

cf_status cf_arena_init_ex(cf_arena *arena, cf_usize cap, const cf_allocator *allocator);

cf_status cf_arena_init(cf_arena *arena, cf_usize cap);

void cf_arena_reset(cf_arena *arena);

void cf_arena_destroy(cf_arena *arena);

void *cf_arena_alloc(void *ctx, cf_usize size);

void *cf_arena_realloc(void *ctx, void *ptr, cf_usize new_size);

void cf_arena_free(void *ctx, void *ptr);

const cf_allocator *cf_arena_allocator(cf_arena *arena);

#endif // CF_ARENA_H
