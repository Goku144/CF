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

#if !defined(CF_POOL_H)
#define CF_POOL_H

#include "ALLOCATOR/cf_alloc.h"
#include "RUNTIME/cf_types.h"
#include "RUNTIME/cf_status.h"

/**
 * cf_pool
 *
 * Fixed-size block (slab-of-one-class) allocator.
 * The backing buffer is divided into @p slot_total slots, each exactly
 * @p slot_size bytes. Free slots are chained together as an intrusive
 * singly-linked free list stored inside the slots themselves, so
 * @p slot_size must be at least sizeof(void *).
 *
 * Allocation and deallocation are both O(1).
 * Realloc is not supported and always returns CF_NULL.
 *
 * @field data        Pointer to the contiguous backing buffer, or CF_NULL
 *                    for an empty pool.
 * @field list        Head of the intrusive free list; points to the next
 *                    available slot, or CF_NULL when the pool is full.
 * @field slot_total  Total number of slots in the pool.
 * @field slot_size   Size in bytes of each slot (>= sizeof(void *)).
 * @field slot_used   Number of slots currently allocated.
 * @field allocator   cf_alloc interface for this pool; ctx points back
 *                    to the owning cf_pool instance.
 */
typedef struct cf_pool
{
  cf_u8   *data;
  void    *list;
  cf_usize slot_total;
  cf_usize slot_size;
  cf_usize slot_used;
  cf_alloc allocator;
} cf_pool;

/********************************************************************/
/* construction                                                     */
/********************************************************************/

/**
 * cf_pool_create_empty
 *
 * Return a zero-initialised cf_pool with the internal function pointers
 * already set. The returned value satisfies cf_pool_is_valid() but has
 * no backing memory; call cf_pool_new() to allocate it.
 *
 * @return  An empty, valid cf_pool.
 */
cf_pool cf_pool_create_empty(void);

/********************************************************************/
/* validation                                                       */
/********************************************************************/

/**
 * cf_pool_is_valid
 *
 * Perform a structural sanity check on @p pool:
 *   - pointer must be non-null.
 *   - if data is CF_NULL, list / slot_total / slot_size / slot_used
 *     must all be 0 / CF_NULL.
 *   - if data is non-null, slot_size >= sizeof(void *),
 *     slot_total > 0, and slot_used <= slot_total.
 *   - the embedded allocator must pass cf_alloc_is_valid().
 *
 * @param pool  Pool to inspect. May be CF_NULL.
 * @return  CF_TRUE  if the pool is internally consistent,
 *          CF_FALSE otherwise.
 */
cf_bool cf_pool_is_valid(cf_pool *pool);

/********************************************************************/
/* lifecycle                                                        */
/********************************************************************/

/**
 * cf_pool_new
 *
 * Initialise @p pool, allocate a backing buffer of
 * @p slot_total * @p slot_size bytes, and build the initial free list.
 * Passing both @p slot_total and @p slot_size as 0 creates a valid
 * but empty pool (no allocation performed).
 *
 * @param pool        Output parameter. Must not be CF_NULL.
 * @param slot_total  Number of slots to provision. Must be > 0 unless
 *                    @p slot_size is also 0.
 * @param slot_size   Size of each slot in bytes. Must be >= sizeof(void *)
 *                    unless @p slot_total is also 0.
 * @return  CF_OK          on success.
 *          CF_ERR_NULL    if @p pool is CF_NULL.
 *          CF_ERR_INVALID if the size/total combination is illegal.
 *          CF_ERR_OOM     if the heap allocation fails.
 */
cf_status cf_pool_new(cf_pool *pool, cf_usize slot_total, cf_usize slot_size);

/**
 * cf_pool_destroy
 *
 * Free the backing buffer and reset every field to its zero value.
 * The pool may be re-initialised with cf_pool_new() afterwards.
 * Passing CF_NULL is safe and has no effect.
 * If @p pool is already empty (data == CF_NULL) this function returns
 * immediately without touching the pool.
 *
 * @param pool  Pool to destroy. May be CF_NULL.
 */
void cf_pool_destroy(cf_pool *pool);

/********************************************************************/
/* operations                                                       */
/********************************************************************/

/**
 * cf_pool_reset
 *
 * Rebuild the free list so every slot is available again, without
 * freeing the backing buffer. All previously returned pointers become
 * invalid after this call.
 * If the pool is empty (data == CF_NULL) the call succeeds immediately.
 *
 * @param pool  Pool to reset. Must not be CF_NULL.
 * @return  CF_OK        on success.
 *          CF_ERR_NULL  if @p pool is CF_NULL.
 *          CF_ERR_STATE if cf_pool_is_valid() returns CF_FALSE.
 */
cf_status cf_pool_reset(cf_pool *pool);

#endif /* CF_POOL_H */