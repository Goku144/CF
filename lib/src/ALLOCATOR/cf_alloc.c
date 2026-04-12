#include "ALLOCATOR/cf_alloc.h"
#include "RUNTIME/cf_types.h"

typedef struct cf_allocator
{
  void *(*alloc) (void *ctx, cf_usize size);
  void *(*realloc) (void *ctx, void *ptr, cf_usize new_size);
  void  (*free) (void *ctx, void *ptr);
}cf_allocator;