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

#if !defined(CF_ALLOC_DEBUG_H)
#define CF_ALLOC_DEBUG_H

#include "RUNTIME/cf_types.h"
#include "RUNTIME/cf_status.h"
#include "ALLOCATOR/cf_alloc.h"

/**
 * cf_alloc_debug_node
 *
 * One entry in the live-allocation tracking list.
 * Every successful alloc() prepends a node; every free() removes it.
 * The list is kept as an intrusive singly-linked chain allocated
 * directly from the system heap (malloc/free), independent of the
 * backing allocator being observed.
 *
 * @field ptr   The pointer returned to the caller by the backing allocator.
 * @field size  The number of bytes that were requested for this allocation.
 * @field next  Next node in the list, or CF_NULL if this is the tail.
 */
typedef struct cf_alloc_debug_node
{
  void    *ptr;
  cf_usize size;
  struct cf_alloc_debug_node *next;
} cf_alloc_debug_node;

/**
 * cf_alloc_debug
 *
 * Instrumented allocator wrapper.
 * Sits in front of any cf_alloc-compatible backing allocator and
 * intercepts every alloc, realloc, and free call. It maintains a
 * linked list of live allocations and a set of counters so that leaks,
 * double-frees, and invalid frees can be detected at runtime.
 *
 * Usage pattern:
 *   1. Create a backing allocator  (e.g. cf_alloc_new()).
 *   2. Call cf_alloc_debug_new()   to wrap it.
 *   3. Hand out &debug.allocator   instead of the backing allocator.
 *   4. Call cf_alloc_debug_report() at any point to inspect state.
 *   5. Call cf_alloc_debug_destroy() to free tracking nodes and reset.
 *
 * @field backing    The real allocator that performs actual memory
 *                   operations. Every intercepted call is forwarded here.
 * @field allocator  The fake cf_alloc handed to callers. Its ctx points
 *                   back to this cf_alloc_debug instance.
 * @field head       Head of the live-allocation linked list. CF_NULL when
 *                   no allocations are currently live.
 *
 * — call counters —
 * @field alloc_count    Total number of successful alloc() calls.
 * @field realloc_count  Total number of successful realloc() calls.
 * @field free_count     Total number of successful free() calls.
 *
 * — memory counters —
 * @field live_count   Number of allocations currently alive.
 * @field bytes_live   Bytes currently allocated (goes up and down).
 * @field bytes_peak   Highest value bytes_live has ever reached.
 * @field bytes_total  Cumulative bytes ever allocated (never decreases).
 *
 * — error counters —
 * @field invalid_free_count    free() or realloc() called with a pointer
 *                              not present in the live list.
 * @field failed_alloc_count    Number of times the backing alloc() returned
 *                              CF_NULL.
 * @field failed_realloc_count  Number of times the backing realloc() returned
 *                              CF_NULL.
 */
typedef struct cf_alloc_debug
{
  cf_alloc backing;
  cf_alloc allocator;

  cf_alloc_debug_node *head;

  cf_usize alloc_count;
  cf_usize realloc_count;
  cf_usize free_count;

  cf_usize live_count;
  cf_usize bytes_live;
  cf_usize bytes_peak;
  cf_usize bytes_total;

  cf_usize invalid_free_count;
  cf_usize failed_alloc_count;
  cf_usize failed_realloc_count;
} cf_alloc_debug;

/********************************************************************/
/* construction                                                     */
/********************************************************************/

/**
 * cf_alloc_debug_create_empty
 *
 * Return a cf_alloc_debug with every counter zeroed, head set to
 * CF_NULL, and the three internal callback functions already wired up.
 * The backing allocator is set to cf_alloc_create_empty() and
 * allocator.ctx is CF_NULL; call cf_alloc_debug_new() to bind a real
 * backing allocator and make the instance usable.
 *
 * Mirrors the pattern of cf_pool_create_empty() — defines the known
 * zero state that new() and destroy() both rely on.
 *
 * @return  A zero-initialised cf_alloc_debug with function pointers set.
 */
cf_alloc_debug cf_alloc_debug_create_empty(void);

/********************************************************************/
/* validation                                                       */
/********************************************************************/

/**
 * cf_alloc_debug_is_valid
 *
 * Perform a structural sanity check on @p debug:
 *   - pointer must be non-null.
 *   - both backing and allocator must pass cf_alloc_is_valid().
 *   - live_count  must be <= alloc_count.
 *   - free_count  must be <= alloc_count + realloc_count.
 *   - bytes_live  must be <= bytes_total.
 *   - bytes_peak  must be >= bytes_live.
 *
 * @param debug  Debug allocator to inspect. May be CF_NULL.
 * @return  CF_TRUE  if the debug allocator is internally consistent,
 *          CF_FALSE otherwise.
 */
cf_bool cf_alloc_debug_is_valid(cf_alloc_debug *debug);

/********************************************************************/
/* lifecycle                                                        */
/********************************************************************/

/**
 * cf_alloc_debug_new
 *
 * Initialise @p debug and bind @p backing as the allocator that will
 * perform real memory operations. After this call @p debug.allocator
 * is ready to be used in place of @p backing anywhere a cf_alloc *
 * is accepted.
 *
 * @param debug    Output parameter. Must not be CF_NULL.
 * @param backing  The allocator to wrap. Must not be CF_NULL and must
 *                 pass cf_alloc_is_valid().
 * @return  CF_OK          on success.
 *          CF_ERR_NULL    if @p debug or @p backing is CF_NULL.
 *          CF_ERR_INVALID if @p backing does not pass cf_alloc_is_valid().
 */
cf_status cf_alloc_debug_new(cf_alloc_debug *debug, cf_alloc *backing);

/**
 * cf_alloc_debug_destroy
 *
 * Free every live tracking node and its corresponding allocation through
 * the backing allocator, then reset the struct to the zero state defined
 * by cf_alloc_debug_create_empty().
 * Useful as a cleanup that also handles leaks: any allocation the caller
 * forgot to free is freed here.
 * Passing CF_NULL is safe and has no effect.
 *
 * @param debug  Debug allocator to destroy. May be CF_NULL.
 */
void cf_alloc_debug_destroy(cf_alloc_debug *debug);

/********************************************************************/
/* operations                                                       */
/********************************************************************/

/**
 * cf_alloc_debug_report
 *
 * Print a human-readable summary of all counters to stdout.
 * If live_count > 0 the report lists every leaked pointer and its size.
 * If live_count == 0 it prints "[OK] no leaks detected".
 * This function is read-only — it does not modify @p debug in any way
 * and may be called any number of times during the program's lifetime.
 * Passing CF_NULL is safe and has no effect.
 *
 * @param debug  Debug allocator to report on. May be CF_NULL.
 */
void cf_alloc_debug_report(cf_alloc_debug *debug);

#endif /* CF_ALLOC_DEBUG_H */