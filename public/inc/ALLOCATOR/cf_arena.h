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

#include "ALLOCATOR/cf_alloc.h"
#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_types.h"

typedef struct cf_arena_chunk cf_arena_chunk;

struct cf_arena_chunk
{
  void *data;
  cf_usize capacity;
  cf_usize offset;
  cf_usize high_water;
  cf_arena_chunk *next;
};

typedef struct cf_arena
{
  void *data;

  cf_usize capacity;
  cf_usize offset;
  cf_usize high_water;
  cf_usize default_alignment;
  cf_usize next_capacity;

  cf_arena_chunk *chunks;
  cf_arena_chunk *current;

  cf_alloc allocator;
  cf_bool owns_data;
  cf_bool growable;
} cf_arena;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize an owning monotonic arena.
 * @param arena Arena object to initialize.
 * @param capacity Number of bytes to reserve.
 * @param allocator Optional backing allocator; `CF_NULL` uses `cf_alloc_new`.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, or `CF_ERR_OOM`.
 */
cf_status cf_arena_init(cf_arena *arena, cf_usize capacity, cf_alloc *allocator);

/**
 * @brief Initialize an owning arena with explicit alignment/growth policy.
 * @param arena Arena object to initialize.
 * @param capacity Initial bytes to reserve.
 * @param alignment Default power-of-two alignment; zero uses 64 bytes.
 * @param growable Whether the arena may allocate new chunks when full.
 * @param allocator Optional backing allocator; `CF_NULL` uses `cf_alloc_new`.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, `CF_ERR_INVALID`, or `CF_ERR_OOM`.
 */
cf_status cf_arena_init_ex(cf_arena *arena, cf_usize capacity, cf_usize alignment, cf_bool growable, cf_alloc *allocator);

/**
 * @brief Initialize a non-owning arena over caller-provided storage.
 * @param arena Arena object to initialize.
 * @param buffer Caller-owned byte buffer.
 * @param capacity Number of usable bytes in `buffer`.
 * @return `CF_OK` or `CF_ERR_NULL`.
 */
cf_status cf_arena_init_with_buffer(cf_arena *arena, void *buffer, cf_usize capacity);

/**
 * @brief Allocate a byte slice from the arena.
 * @param arena Arena to allocate from.
 * @param size Requested byte count.
 * @param alignment Power-of-two alignment; zero uses pointer alignment.
 * @param ptr Receives the allocated slice or `CF_NULL` for zero-size requests.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, `CF_ERR_INVALID`, `CF_ERR_OVERFLOW`, or `CF_ERR_BOUNDS`.
 */
cf_status cf_arena_alloc(cf_arena *arena, cf_usize size, cf_usize alignment, void **ptr);

/**
 * @brief Reset the arena offset without releasing backing storage.
 * @param arena Arena to reset.
 */
void cf_arena_reset(cf_arena *arena);

/**
 * @brief Release arena-owned storage and clear the object.
 * @param arena Arena to destroy.
 */
void cf_arena_destroy(cf_arena *arena);

#ifdef __cplusplus
}
#endif

#endif /* CF_ARENA_H */
