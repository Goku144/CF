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

#if !defined(CF_RANDOM_H)
#define CF_RANDOM_H

#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_types.h"

/**
 * @brief Fill a memory range with cryptographically secure random bytes.
 *
 * The destination is treated as raw byte storage. Passing `len == 0` is a
 * successful no-op.
 *
 * @param dst Destination memory receiving random bytes.
 * @param len Number of bytes to write.
 * @return `CF_OK` on success, `CF_ERR_NULL` when `dst` is null for a non-empty
 * request, or `CF_ERR_RANDOM` when the operating system RNG fails.
 */
cf_status cf_random_bytes(void *dst, cf_usize len);

/**
 * @brief Generate a random 32-bit unsigned integer.
 *
 * @param dst Destination receiving the random value.
 * @return `CF_OK` on success, `CF_ERR_NULL` when `dst` is null, or
 * `CF_ERR_RANDOM` when the operating system RNG fails.
 */
cf_status cf_random_u32(cf_u32 *dst);

/**
 * @brief Generate a random 64-bit unsigned integer.
 *
 * @param dst Destination receiving the random value.
 * @return `CF_OK` on success, `CF_ERR_NULL` when `dst` is null, or
 * `CF_ERR_RANDOM` when the operating system RNG fails.
 */
cf_status cf_random_u64(cf_u64 *dst);

#endif /* CF_RANDOM_H */
