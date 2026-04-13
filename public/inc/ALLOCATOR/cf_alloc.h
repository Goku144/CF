#if !defined(CF_ALLOC_H)
#define CF_ALLOC_H

#include "RUNTIME/cf_types.h"

/**
 * cf_alloc structure is 32 Byte
 * pass its argument as poiner
 * for fast handling by functions
 */
typedef struct cf_alloc
{
  void *ctx;
  void *(*alloc) (void *ctx, cf_usize size);
  void *(*realloc) (void *ctx, void *ptr, cf_usize size);
  void  (*free) (void *ctx, void *ptr);
} cf_alloc;

cf_alloc cf_alloc_create_empty();

cf_bool cf_alloc_is_valid(const cf_alloc *allocator);

cf_alloc cf_alloc_new();

#endif /* CF_ALLOC_H */
