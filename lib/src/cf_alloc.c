#include "cf_alloc.h"
#include <stdlib.h>

static void *cf_heap_alloc(void *ctx, cf_usize size)
{
  CF_UNUSED(ctx);
  return malloc(size);
}

static void *cf_heap_realloc(void *ctx, void *ptr, cf_usize new_size)
{
  CF_UNUSED(ctx);
  return realloc(ptr, new_size);
}

static void cf_heap_free(void *ctx, void *ptr)
{
  CF_UNUSED(ctx);
  free(ptr);
}

static const cf_allocator g_cf_default_allocator = {
  .ctx = CF_NULL,
  .alloc = cf_heap_alloc,
  .realloc = cf_heap_realloc,
  .free = cf_heap_free
};

cf_bool cf_allocator_is_valid(const cf_allocator *alloc)
{
  if(alloc == CF_NULL) return CF_FALSE;
  if(alloc->alloc == CF_NULL) return CF_FALSE;
  if(alloc->realloc == CF_NULL) return CF_FALSE;
  if(alloc->free == CF_NULL) return CF_FALSE;
  return CF_TRUE;
}

const cf_allocator *cf_default_allocator(void)
{
  return &g_cf_default_allocator;
}