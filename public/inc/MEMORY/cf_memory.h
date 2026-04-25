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

#if !defined(CF_MEMORY_H)
#define CF_MEMORY_H

#include "ALLOCATOR/cf_alloc.h"

#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_types.h"

#define CF_MEMORY_GROWTH_SIZE 5096
#define CF_USIZE_MAX SIZE_MAX

/* Writable non-owning view over a contiguous byte range. */
typedef struct cf_bytes
{
  void *data;
  cf_usize elem_size;
  cf_usize len;
} cf_bytes, cf_array_element;

/* Owned growable byte buffer managed through an allocator. */
typedef struct cf_buffer
{
  cf_u8 *data;
  cf_usize cap;
  cf_usize len;
  cf_alloc allocator;
} cf_buffer, cf_string;

/**
 * @brief Initialize a buffer with the default allocator and optional capacity.
 *
 * The buffer is reset to an empty state, assigned the framework default
 * allocator, and optionally given backing storage for `capacity` bytes.
 *
 * @param buffer Buffer to initialize.
 * @param capacity Initial capacity in bytes to allocate.
 * @return `CF_OK` on success or `CF_ERR_OOM` when allocation fails.
 */
cf_status cf_buffer_init(cf_buffer *buffer, cf_usize capacity);

/**
 * @brief Ensure that a buffer can hold at least the requested number of bytes.
 *
 * When the current capacity is already large enough, the buffer is left
 * unchanged. Otherwise the backing storage is reallocated to the requested
 * capacity.
 *
 * @param buffer Buffer whose capacity must be ensured.
 * @param capacity Minimum capacity in bytes required after the call.
 * @return `CF_OK` on success or `CF_ERR_OOM` when reallocation fails.
 */
cf_status cf_buffer_reserve(cf_buffer *buffer, cf_usize capacity);

/**
 * @brief Release a buffer's owned storage and reset it to an empty state.
 *
 * @param buffer Buffer to destroy.
 */
void cf_buffer_destroy(cf_buffer *buffer);

/**
 * @brief Append one byte to the end of a buffer.
 *
 * When the current capacity is full, the buffer grows before the append is
 * written.
 *
 * @param buffer Destination buffer receiving the new byte.
 * @param byte Byte value to append.
 * @return `CF_OK` on success or `CF_ERR_OOM` when growth fails.
 */
cf_status cf_buffer_append_byte(cf_buffer *buffer, cf_u8 byte);

/**
 * @brief Append a byte view to the end of a buffer.
 *
 * The source bytes are copied into the buffer tail. When additional space is
 * needed, the buffer grows by just enough bytes to fit the appended range.
 * Overlapping source and destination regions are supported.
 *
 * @param buffer Destination buffer receiving the copied bytes.
 * @param bytes Source byte view to append.
 * @return `CF_OK` on success or `CF_ERR_OOM` when growth fails.
 */
cf_status cf_buffer_append_bytes(cf_buffer *buffer, cf_bytes bytes);

/**
 * @brief Expose a contiguous range of a buffer as a non-owning byte view.
 *
 * The returned `cf_bytes` points directly into the buffer storage and does not
 * allocate or copy memory. The requested range is inclusive on both ends.
 *
 * @param buffer Source buffer to view.
 * @param bytes Output byte view receiving the selected range.
 * @param start Inclusive start offset in bytes.
 * @param end Inclusive end offset in bytes.
 * @return `CF_OK` on success, `CF_ERR_INVALID` when `start > end`, or
 * `CF_ERR_BOUNDS` when the requested range exceeds the current buffer length.
 */
cf_status cf_buffer_as_bytes(cf_buffer *buffer, cf_bytes *bytes, cf_usize start, cf_usize end);

/**
 * @brief Mark a buffer as empty without releasing its allocated capacity.
 *
 * @param buffer Buffer to reset.
 */
void cf_buffer_reset(cf_buffer *buffer);

/**
 * @brief Shrink the logical length of a buffer to a smaller size.
 *
 * The backing allocation is not changed. Truncation is only valid when `len`
 * is less than or equal to the current logical length.
 *
 * @param buffer Buffer to truncate.
 * @param len New logical length in bytes.
 * @return `CF_OK` on success or `CF_ERR_BOUNDS` when `len` exceeds the current
 * length.
 */
cf_status cf_buffer_trunc(cf_buffer *buffer, cf_usize len);

/**
 * @brief Check whether a buffer satisfies the framework's structural rules.
 *
 * A valid buffer has internally consistent `data`, `cap`, and `len` fields. A
 * null data pointer is only valid when both logical length and capacity are
 * zero.
 *
 * @param buffer Buffer to validate.
 * @return `CF_TRUE` when the buffer is structurally valid, otherwise
 * `CF_FALSE`.
 */
cf_bool cf_buffer_is_valid(cf_buffer *buffer);

/**
 * @brief Report whether a buffer currently contains no logical data.
 *
 * @param buffer Buffer to inspect.
 * @return `CF_TRUE` when the buffer length is zero, otherwise `CF_FALSE`.
 */
cf_bool cf_buffer_is_empty(cf_buffer *buffer);

/**
 * @brief Print a diagnostic summary of a buffer's current state.
 *
 * The printed information includes the data pointer, logical length,
 * allocation capacity, and whether each allocator callback field is set.
 *
 * @param buffer Buffer to inspect and print.
 */
void cf_buffer_info(cf_buffer *buffer);

#endif /* CF_MEMORY_H */
