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

#if !defined(CF_ARENA_H)
#define CF_ARENA_H

#include "RUNTIME/cf_types.h"
#include "RUNTIME/cf_status.h"
#include "ALLOCATOR/cf_alloc.h"

/**
 * cf_arena
 *
 * Bump-pointer / linear allocator.
 * Memory is handed out by advancing an offset into a single contiguous
 * buffer. Individual allocations cannot be freed; the entire arena is
 * reclaimed at once with cf_arena_reset() or cf_arena_destroy().
 *
 * The allocator supports one special realloc: if the pointer being
 * resized is the most recently allocated block (tracked via
 * @p last_usable), the offset is simply adjusted in place at zero cost.
 *
 * @field data          Pointer to the backing buffer, or CF_NULL when
 *                      the arena was created with size 0.
 * @field cap           Total capacity of the backing buffer in bytes.
 * @field offset        Byte offset of the next free position.
 * @field last_usable   Offset at which the most recent allocation began,
 *                      used to detect the last-block realloc fast path.
 * @field allocator     cf_alloc interface for this arena; ctx points back
 *                      to the owning cf_arena instance.
 */
typedef struct cf_arena
{
  cf_u8   *data;
  cf_usize cap;
  cf_usize offset;
  cf_usize last_usable;
  cf_alloc allocator;
} cf_arena;

/********************************************************************/
/* validation                                                       */
/********************************************************************/

/**
 * cf_arena_is_valid
 *
 * Perform a structural sanity check on @p arena:
 *   - pointer must be non-null.
 *   - if data is CF_NULL, cap / offset / last_usable must all be 0.
 *   - if data is non-null, cap >= offset and offset >= last_usable.
 *   - the embedded allocator must pass cf_alloc_is_valid().
 *
 * @param arena  Arena to inspect. May be CF_NULL.
 * @return  CF_TRUE  if the arena is internally consistent,
 *          CF_FALSE otherwise.
 */
cf_bool cf_arena_is_valid(cf_arena *arena);

/********************************************************************/
/* lifecycle                                                        */
/********************************************************************/

/**
 * cf_arena_new
 *
 * Initialise @p arena and allocate a backing buffer of @p size bytes
 * from the system heap.
 * If @p size is 0 the arena is left in a valid but empty state
 * (data == CF_NULL, cap == 0) and no heap allocation is performed.
 *
 * @param arena  Output parameter. Must not be CF_NULL.
 * @param size   Desired capacity in bytes. 0 is allowed.
 * @return  CF_OK       on success.
 *          CF_ERR_NULL if @p arena is CF_NULL.
 *          CF_ERR_OOM  if the heap allocation fails.
 */
cf_status cf_arena_new(cf_arena *arena, cf_usize size);

/**
 * cf_arena_destroy
 *
 * Free the backing buffer and reset every field to its zero value.
 * After this call the arena is in the same state as a default-
 * initialised struct; it may be re-used by calling cf_arena_new() again.
 * Passing CF_NULL is safe and has no effect.
 *
 * @param arena  Arena to destroy. May be CF_NULL.
 */
void cf_arena_destroy(cf_arena *arena);

/********************************************************************/
/* operations                                                       */
/********************************************************************/

/**
 * cf_arena_reset
 *
 * Rewind the bump pointer to the start of the buffer without freeing
 * the backing memory. All previously allocated pointers become invalid
 * after this call.
 * Equivalent to bulk-freeing every allocation made since the last
 * reset (or since cf_arena_new()).
 *
 * @param arena  Arena to reset. Must not be CF_NULL.
 * @return  CF_OK        on success.
 *          CF_ERR_NULL  if @p arena is CF_NULL.
 *          CF_ERR_STATE if cf_arena_is_valid() returns CF_FALSE.
 */
cf_status cf_arena_reset(cf_arena *arena);

#endif /* CF_ARENA_H */