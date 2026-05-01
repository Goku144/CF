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

/* Allocator dispatch table with optional user context passed to each callback. */
typedef struct cf_alloc
{
  /* User-defined state carried into each allocation callback. */
  void *ctx;
  /* Allocates a new memory block of the requested size. */
  void *(* alloc)   (void *ctx, cf_usize size);
  /* Resizes a previously allocated block, preserving existing contents. */
  void *(* realloc) (void *ctx, void *ptr, cf_usize size);
  /* Releases a block previously returned by alloc or realloc. */
  void  (* free)    (void *ctx, void *ptr);
  /* Optional aligned allocation callback. */
  void *(* alloc_aligned)(void *ctx, cf_usize alignment, cf_usize size);
  /* Optional aligned free callback. */
  void  (* free_aligned)(void *ctx, void *ptr);
} 
cf_alloc;

/**
 * @brief Initializes an allocator with the default heap-backed callbacks.
 *
 * The initialized allocator uses the framework default heap behavior with a
 * `CF_NULL` context and standard alloc, realloc, and free callbacks.
 *
 * @param alloc Allocator to initialize.
 * @return void
 */
void cf_alloc_new(cf_alloc *alloc);

/**
 * @brief Allocate aligned memory through an allocator.
 *
 * If the allocator exposes aligned callbacks, they are used directly. Otherwise
 * the framework over-allocates, aligns the returned address, and stores the
 * original pointer for `cf_alloc_aligned_free`.
 *
 * @param alloc Allocator to use; `CF_NULL` uses the framework default.
 * @param alignment Power-of-two byte alignment; zero uses pointer alignment.
 * @param size Requested bytes.
 * @return Aligned pointer or `CF_NULL` on invalid input/OOM.
 */
void *cf_alloc_aligned(cf_alloc *alloc, cf_usize alignment, cf_usize size);

/**
 * @brief Release a pointer returned by `cf_alloc_aligned`.
 * @param alloc Allocator used for allocation; `CF_NULL` uses the framework default.
 * @param ptr Pointer returned by `cf_alloc_aligned`.
 */
void cf_alloc_aligned_free(cf_alloc *alloc, void *ptr);

#endif /* CF_ALLOC_H */
