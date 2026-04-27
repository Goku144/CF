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
#include <stdarg.h>

/*
 * Default allocation callback backed by malloc.
 */
static void *cf_alloc_alloc(void *ctx, cf_usize size)
{
  CF_UNUSED(ctx);
  return malloc(size);
}

/*
 * Default reallocation callback backed by realloc.
 */
static void *cf_alloc_realloc(void *ctx, void *ptr, cf_usize size)
{
  CF_UNUSED(ctx);
  return realloc(ptr, size);
}

/*
 * Default free callback backed by free.
 */
static void cf_alloc_free(void *ctx, void *ptr)
{
  CF_UNUSED(ctx);
  free(ptr);
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
