#include "ALLOCATOR/cf_alloc.h"
#include "RUNTIME/cf_types.h"

#include <stdlib.h>

static void *cf_alloc_alloc(void *ctx, cf_usize size)
{
  CF_UNUSED(ctx);
  return malloc(size);
}

static void *cf_alloc_realloc(void *ctx, void *ptr, cf_usize size)
{
  CF_UNUSED(ctx);
  return realloc(ptr, size);
}

static void cf_alloc_free(void *ctx, void *ptr)
{
  CF_UNUSED(ctx);
  free(ptr);
}

cf_alloc cf_alloc_create_empty()
{
  return  (cf_alloc) {CF_NULL, cf_alloc_alloc, cf_alloc_realloc, cf_alloc_free};
}

cf_bool cf_alloc_is_valid(const cf_alloc *allocator)
{
  if (allocator == CF_NULL) return CF_FALSE;
  if (allocator->alloc == CF_NULL) return CF_FALSE;
  if (allocator->realloc == CF_NULL) return CF_FALSE;
  if (allocator->free == CF_NULL) return CF_FALSE;
  return CF_TRUE;
}

cf_alloc cf_alloc_new()
{
  return cf_alloc_create_empty();
}