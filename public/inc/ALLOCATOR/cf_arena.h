#if !defined(CF_ARENA_H)
#define CF_ARENA_H

#include "RUNTIME/cf_types.h"
#include "RUNTIME/cf_status.h"
#include "ALLOCATOR/cf_alloc.h"

typedef struct cf_arena
{
  cf_u8 *data;
  cf_usize cap;
  cf_usize offset;
  cf_usize last_usable;
  cf_alloc allocator;
} cf_arena;

cf_bool cf_arena_is_valid(cf_arena *arena);

cf_status cf_arena_new(cf_arena *arena, cf_usize size);

cf_status cf_arena_reset(cf_arena *arena);

void cf_arena_destroy(cf_arena *arena);

#endif /* CF_ARENA_H */
