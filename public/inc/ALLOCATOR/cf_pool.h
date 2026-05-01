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
#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_types.h"

typedef struct cf_pool
{
  void *data;

  cf_usize block_size;
  cf_usize block_count;
  cf_usize alignment;
  cf_usize stride;

  cf_usize free_head;
  cf_usize free_count;
  cf_alloc allocator;

  cf_bool owns_data;
} cf_pool;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize an owning fixed-size block pool.
 * @param pool Pool object to initialize.
 * @param block_size User-visible bytes per block.
 * @param block_count Number of blocks to reserve.
 * @param allocator Optional backing allocator; `CF_NULL` uses `cf_alloc_new`.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_INVALID`, `CF_ERR_STATE`, `CF_ERR_OVERFLOW`, or `CF_ERR_OOM`.
 */
cf_status cf_pool_init(cf_pool *pool, cf_usize block_size, cf_usize block_count, cf_alloc *allocator);

/**
 * @brief Initialize an owning fixed-size block pool with explicit alignment.
 * @param pool Pool object to initialize.
 * @param block_size User-visible bytes per block.
 * @param block_count Number of blocks to reserve.
 * @param alignment Power-of-two block alignment; zero uses 64 bytes.
 * @param allocator Optional backing allocator; `CF_NULL` uses `cf_alloc_new`.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_INVALID`, `CF_ERR_STATE`, `CF_ERR_OVERFLOW`, or `CF_ERR_OOM`.
 */
cf_status cf_pool_init_ex(cf_pool *pool, cf_usize block_size, cf_usize block_count, cf_usize alignment, cf_alloc *allocator);

/**
 * @brief Initialize a non-owning fixed-size block pool over caller storage.
 * @param pool Pool object to initialize.
 * @param buffer Caller-owned storage.
 * @param block_size User-visible bytes per block.
 * @param block_count Number of blocks in `buffer`.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_INVALID`, or `CF_ERR_STATE`.
 */
cf_status cf_pool_init_with_buffer(cf_pool *pool, void *buffer, cf_usize block_size, cf_usize block_count);

/**
 * @brief Allocate one block from the pool.
 * @param pool Pool to allocate from.
 * @param ptr Receives the block pointer.
 * @return `CF_OK`, `CF_ERR_NULL`, or `CF_ERR_OOM`.
 */
cf_status cf_pool_alloc(cf_pool *pool, void **ptr);

/**
 * @brief Return one block to the pool.
 * @param pool Pool that owns the block.
 * @param ptr Block pointer returned by `cf_pool_alloc`.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, `CF_ERR_BOUNDS`, or `CF_ERR_INVALID`.
 */
cf_status cf_pool_free(cf_pool *pool, void *ptr);

/**
 * @brief Restore every block to the free list.
 * @param pool Pool to reset.
 */
void cf_pool_reset(cf_pool *pool);

/**
 * @brief Release pool-owned storage and clear the object.
 * @param pool Pool to destroy.
 */
void cf_pool_destroy(cf_pool *pool);

#ifdef __cplusplus
}
#endif

#endif /* CF_POOL_H */
