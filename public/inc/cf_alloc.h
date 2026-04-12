#if !defined(CF_ALLOC_H)
#define CF_ALLOC_H

#include "cf_types.h"

typedef struct cf_allocator
{
  void *ctx;
  void *(*alloc)(void *ctx, cf_usize size);
  void *(*realloc)(void *ctx, void *ptr, cf_usize new_size);
  void (*free)(void *ctx, void *ptr);
} cf_allocator;

cf_bool cf_allocator_is_valid(const cf_allocator *alloc);

const cf_allocator *cf_default_allocator(void);

#endif // CF_ALLOC_H