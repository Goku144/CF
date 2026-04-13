#if !defined(CF_SLAB_L)
#define CF_SLAB_L

#include "RUNTIME/cf_types.h"
#include "RUNTIME/cf_status.h"
#include "ALLOCATOR/cf_pool.h"

#define CF_SLAB_MAX_CLASS 8

typedef struct cf_slab_class
{
  cf_pool pool;
  cf_usize class_slot_size;
} cf_slab_class;

typedef struct cf_slab
{
  cf_slab_class class[CF_SLAB_MAX_CLASS];
  cf_usize class_count;
  cf_alloc allocator;
} cf_slab;

cf_bool cf_slab_is_valid(cf_slab *slab);

cf_status cf_slab_new(cf_slab *slab, cf_usize slots_total[], cf_usize slots_size[], cf_usize n);

cf_status cf_slab_reset(cf_slab *slab);

void cf_slab_destroy(cf_slab *slab);

#endif /* CF_SLAB_L */
