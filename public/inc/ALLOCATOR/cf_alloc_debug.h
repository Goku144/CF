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

#include "ALLOCATOR/cf_alloc.h"

typedef struct cf_alloc_debug_node
{
  void *ptr;
  struct cf_alloc_debug_node *next;
} cf_alloc_debug_node;

typedef struct cf_alloc_debug
{
  cf_alloc allocator;
  cf_alloc internal_allocator;

  cf_usize ptr_live;
  cf_usize ptr_free;

  cf_usize ptr_max_live;
  cf_usize ptr_max_free;

  cf_usize ptr_all_live;
  cf_usize ptr_all_free;

  cf_usize ptr_invalid_alloc;

  cf_usize ptr_internal_invalid_alloc;
  cf_usize ptr_internal_invalid_realloc;
  cf_usize ptr_internal_invalid_free;

  char *statement;

  void *latest_valid_ptr;
  
  cf_alloc_debug_node *head;
} cf_alloc_debug;

/**
 * @brief Initializes a debug allocator wrapper around an existing allocator.
 *
 * The debug allocator stores the provided backing allocator, resets all debug
 * counters, and configures an instrumented public `allocator` interface for
 * allocation tracking. When `statement` is `CF_NULL`, a default diagnostic
 * string is stored instead.
 *
 * @param alloc_debug Target debug allocator to initialize.
 * @param alloc Backing allocator used for the real memory operations.
 * @param statement Optional human-readable diagnostic label for this allocator.
 * @return void
 */
void cf_alloc_debug_new(cf_alloc_debug *alloc_debug, cf_alloc *alloc, char *statement);

/**
 * @brief Prints the current debug allocator counters and tracked state.
 *
 * The log includes the statement label, allocator addresses, live and free
 * counters, invalid-operation counters, and the latest valid tracked pointer.
 * When `debug` is `CF_NULL`, the function returns without printing anything.
 *
 * @param debug Debug allocator instance to inspect.
 * @param line Caller-provided source line number, typically `__LINE__`.
 * @return void
 */
void cf_alloc_debug_log(cf_alloc_debug *debug, int line);

#endif /* CF_ALLOC_DEBUG_H */
