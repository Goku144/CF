/*
 * CF Framework
 * Copyright (C) 2026 Orion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "ALLOCATOR/cf_alloc.h"

#include "RUNTIME/cf_types.h"

#include <stdlib.h>

#if defined(CF_ALLOC_USE_MIMALLOC)
#include <mimalloc.h>
#endif

static cf_bool cf_alloc_is_power_of_two(cf_usize value)
{
  return value != 0 && (value & (value - 1U)) == 0;
}

static cf_usize cf_alloc_align_forward(cf_usize value, cf_usize alignment)
{
  cf_usize mask = alignment - 1U;
  return (value + mask) & ~mask;
}

static cf_usize cf_alloc_round_up(cf_usize value, cf_usize alignment)
{
  if(alignment == 0) return value;
  return cf_alloc_align_forward(value, alignment);
}

/*
 * Default allocation callback backed by malloc.
 */
static void *cf_alloc_alloc(void *ctx, cf_usize size)
{
  CF_UNUSED(ctx);
#if defined(CF_ALLOC_USE_MIMALLOC)
  return mi_malloc((size_t)size);
#else
  return malloc(size);
#endif
}

/*
 * Default reallocation callback backed by realloc.
 */
static void *cf_alloc_realloc(void *ctx, void *ptr, cf_usize size)
{
  CF_UNUSED(ctx);
#if defined(CF_ALLOC_USE_MIMALLOC)
  return mi_realloc(ptr, (size_t)size);
#else
  return realloc(ptr, size);
#endif
}

/*
 * Default free callback backed by free.
 */
static void cf_alloc_free(void *ctx, void *ptr)
{
  CF_UNUSED(ctx);
#if defined(CF_ALLOC_USE_MIMALLOC)
  mi_free(ptr);
#else
  free(ptr);
#endif
}

static void *cf_alloc_alloc_aligned_default(void *ctx, cf_usize alignment, cf_usize size)
{
  cf_usize rounded = 0;
  CF_UNUSED(ctx);

  if(size == 0) return CF_NULL;
  if(alignment == 0) alignment = sizeof(void *);
  if(alignment < sizeof(void *)) alignment = sizeof(void *);
  if(cf_alloc_is_power_of_two(alignment) == CF_FALSE) return CF_NULL;

#if defined(CF_ALLOC_USE_MIMALLOC)
  return mi_malloc_aligned((size_t)size, (size_t)alignment);
#else
  if(size > (cf_usize)-1 - (alignment - 1U)) return CF_NULL;
  rounded = cf_alloc_round_up(size, alignment);
  if(rounded == 0) return CF_NULL;
  return aligned_alloc((size_t)alignment, (size_t)rounded);
#endif
}

static void cf_alloc_free_aligned_default(void *ctx, void *ptr)
{
  CF_UNUSED(ctx);
#if defined(CF_ALLOC_USE_MIMALLOC)
  mi_free(ptr);
#else
  free(ptr);
#endif
}

/*
 * Build the default allocator vtable used by buffers, arrays, and strings.
 */
static cf_alloc cf_alloc_create(void)
{
  return (cf_alloc) 
  {
    .ctx = CF_NULL,
    .alloc = cf_alloc_alloc,
    .realloc = cf_alloc_realloc,
    .free = cf_alloc_free,
    .alloc_aligned = cf_alloc_alloc_aligned_default,
    .free_aligned = cf_alloc_free_aligned_default,
  };
}

/*
 * Public initializer for the default allocator object.
 */
void cf_alloc_new(cf_alloc *alloc)
{
  if(alloc == CF_NULL) return;
  CF_ASSERT_TYPE_SIZE(*alloc, cf_alloc);
  *alloc = cf_alloc_create();
}

void *cf_alloc_aligned(cf_alloc *alloc, cf_usize alignment, cf_usize size)
{
  cf_alloc local_allocator;
  cf_usize total = 0;
  cf_uptr raw_addr = 0;
  cf_uptr aligned_addr = 0;
  void *raw = CF_NULL;
  void **slot = CF_NULL;

  if(size == 0) return CF_NULL;
  if(alignment == 0) alignment = sizeof(void *);
  if(alignment < sizeof(void *)) alignment = sizeof(void *);
  if(cf_alloc_is_power_of_two(alignment) == CF_FALSE) return CF_NULL;

  if(alloc == CF_NULL)
  {
    cf_alloc_new(&local_allocator);
    alloc = &local_allocator;
  }
  if(alloc->alloc_aligned != CF_NULL)
    return alloc->alloc_aligned(alloc->ctx, alignment, size);
  if(alloc->alloc == CF_NULL) return CF_NULL;

  if(size > (cf_usize)-1 - alignment) return CF_NULL;
  if(size + alignment > (cf_usize)-1 - sizeof(void *)) return CF_NULL;
  total = size + alignment + sizeof(void *);

  raw = alloc->alloc(alloc->ctx, total);
  if(raw == CF_NULL) return CF_NULL;

  raw_addr = (cf_uptr)raw + sizeof(void *);
  aligned_addr = (cf_uptr)cf_alloc_align_forward(raw_addr, alignment);
  slot = (void **)aligned_addr;
  slot[-1] = raw;
  return (void *)aligned_addr;
}

void cf_alloc_aligned_free(cf_alloc *alloc, void *ptr)
{
  cf_alloc local_allocator;
  void *raw = CF_NULL;

  if(ptr == CF_NULL) return;
  if(alloc == CF_NULL)
  {
    cf_alloc_new(&local_allocator);
    alloc = &local_allocator;
  }
  if(alloc->free_aligned != CF_NULL)
  {
    alloc->free_aligned(alloc->ctx, ptr);
    return;
  }
  if(alloc->free == CF_NULL) return;
  raw = ((void **)ptr)[-1];
  alloc->free(alloc->ctx, raw);
}
