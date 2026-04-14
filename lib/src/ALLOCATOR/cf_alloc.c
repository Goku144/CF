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

/********************************************************************/
/* allocators                                                       */
/********************************************************************/

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

/********************************************************************/
/* construction                                                     */
/********************************************************************/

cf_alloc cf_alloc_create_empty(void)
{
  return  (cf_alloc) {CF_NULL, cf_alloc_alloc, cf_alloc_realloc, cf_alloc_free};
}

cf_alloc cf_alloc_new()
{
  return cf_alloc_create_empty();
}

/********************************************************************/
/* validation                                                       */
/********************************************************************/

cf_bool cf_alloc_is_valid(const cf_alloc *allocator)
{
  if (allocator == CF_NULL) return CF_FALSE;
  if (allocator->alloc == CF_NULL) return CF_FALSE;
  if (allocator->realloc == CF_NULL) return CF_FALSE;
  if (allocator->free == CF_NULL) return CF_FALSE;
  return CF_TRUE;
}