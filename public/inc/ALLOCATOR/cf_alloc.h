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

#if !defined(CF_ALLOC_H)
#define CF_ALLOC_H

#include "RUNTIME/cf_types.h"

/**
 * cf_alloc
 *
 * Generic allocator interface backed by three function pointers.
 * Every allocator in the library (arena, pool, slab) exposes itself
 * through this struct so callers can treat them uniformly.
 *
 * Size: 32 bytes on 64-bit platforms.
 * Convention: always pass by pointer to avoid copying the function pointers.
 *
 * @field ctx      Opaque pointer forwarded to every call. May be CF_NULL
 *                 for the default heap allocator.
 * @field alloc    Allocate a fresh block of @p size bytes.
 * @field realloc  Resize an existing block pointed to by @p ptr.
 * @field free     Release a block previously returned by @p alloc.
 */
typedef struct cf_alloc
{
  void *ctx;
  void *(*alloc)   (void *ctx, cf_usize size);
  void *(*realloc) (void *ctx, void *ptr, cf_usize size);
  void  (*free)    (void *ctx, void *ptr);
} cf_alloc;

/********************************************************************/
/* construction                                                     */
/********************************************************************/

/**
 * cf_alloc_create_empty
 *
 * Return a cf_alloc wired to the system heap (malloc / realloc (NULL) / free (NULL)
 * with ctx set to CF_NULL.
 * The returned value is fully usable immediately; no further
 * initialisation is required.
 *
 * @return  A ready-to-fill heap allocator.
 */
cf_alloc cf_alloc_create_empty();

/**
 * cf_alloc_new
 *
 * Return a cf_alloc wired to the system heap (malloc / realloc (NULL) / free (NULL)
 * Provided so callers can follow the same new / destroy naming
 * convention used by the rest of the library.
 *
 * @return  A ready-to-use heap allocator.
 */
cf_alloc cf_alloc_new();

/********************************************************************/
/* validation                                                       */
/********************************************************************/

/**
 * cf_alloc_is_valid
 *
 * Check that @p allocator is non-null and that none of its three
 * function pointers are CF_NULL.
 * Does NOT verify that the functions behave correctly.
 *
 * @param allocator  Pointer to the allocator to inspect. May be CF_NULL.
 * @return  CF_TRUE  if the allocator is safe to use,
 *          CF_FALSE if @p allocator is CF_NULL or any function pointer
 *                   is CF_NULL.
 */
cf_bool cf_alloc_is_valid(const cf_alloc *allocator);

#endif /* CF_ALLOC_H */