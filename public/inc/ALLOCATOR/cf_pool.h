#if !defined(CF_POOL_H)
#define CF_POOL_H

#include "ALLOCATOR/cf_alloc.h"
#include "RUNTIME/cf_types.h"
#include "RUNTIME/cf_status.h"

typedef struct cf_pool
{
  cf_u8 *data;
  void *list;
  cf_usize slot_total;
  cf_usize slot_size;
  cf_usize slot_used;
  cf_alloc allocator;
} cf_pool;

cf_pool cf_pool_create_empty();

cf_bool cf_pool_is_valid(cf_pool *pool);

cf_status cf_pool_new(cf_pool *pool, cf_usize slot_total, cf_usize slot_size);

cf_status cf_pool_reset(cf_pool *pool);

void cf_pool_destroy(cf_pool *pool);

#endif /* CF_POOL_H */
