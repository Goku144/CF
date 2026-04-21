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

#if !defined(CF_ARRAY_H)
#define CF_ARRAY_H

#include "MEMORY/cf_memory.h"

/* Owned growable array storing fixed-size elements. */
typedef struct cf_array
{
  cf_array_element *data;
  cf_usize len; 
  cf_usize cap; 
  cf_alloc allocator;
} cf_array;

/**
 * @brief Initializes an array with optional capacity for array elements.
 *
 * The array is reset to an empty state, assigned the default allocator, and
 * optionally given backing storage large enough for `capacity`
 * `cf_array_element` values.
 *
 * @param array Array to initialize.
 * @param capacity Initial capacity measured in elements.
 * @return `CF_OK` on success, `CF_ERR_NULL` when `array` is `CF_NULL`, or
 * `CF_ERR_OOM` when the initial allocation fails.
 */
cf_status cf_array_init(cf_array *array, cf_usize capacity);

/**
 * @brief Ensures that an array can hold at least the requested element count.
 *
 * When the current capacity is already large enough, the array is left
 * unchanged. Otherwise the backing storage is reallocated to hold
 * `capacity` `cf_array_element` values.
 *
 * @param array Array whose capacity must be ensured.
 * @param capacity Minimum capacity required after the call, measured in elements.
 * @return `CF_OK` on success, or a status describing null, invalid-state, or
 * allocation failure conditions.
 */
cf_status cf_array_reserve(cf_array *array, cf_usize capacity);

/**
 * @brief Releases an array's owned storage and resets it to an empty state.
 *
 * When the array is valid, its allocator `free` callback is used on the
 * backing storage and all fields are cleared afterward.
 *
 * @param array Array to destroy.
 * @return void
 */
void cf_array_destroy(cf_array *array);

/**
 * @brief Marks an array as empty without releasing its allocated storage.
 *
 * The element count becomes zero while the backing allocation and capacity are
 * preserved for later reuse.
 *
 * @param array Array to reset.
 * @return `CF_OK` on success, or a status describing null or invalid-state
 * conditions.
 */
cf_status cf_array_reset(cf_array *array);

/**
 * @brief Read the last element in an array without removing it.
 *
 * When the array is not empty, the last stored element is copied into
 * `element` and the array length remains unchanged. Empty arrays return
 * `CF_OK` and write a zero-initialized element.
 *
 * @param array Array to inspect.
 * @param element Output pointer receiving the last array element.
 * @return `CF_OK` on success, or a status describing null or invalid-state
 * conditions.
 */
cf_status cf_array_peek(cf_array *array, cf_array_element *element);

/**
 * @brief Appends one or more elements to the end of an array.
 *
 * The first element is provided by `element`, and additional elements may be
 * passed through the variadic argument list. The list must be terminated with
 * `CF_NULL`.
 *
 * @param array Destination array receiving the appended elements.
 * @param element Pointer to the first array element to append, or `CF_NULL` to do
 * nothing.
 * @return `CF_OK` on success, or a status describing null, invalid-state, or
 * allocation failure conditions.
 */
cf_status cf_array_push(cf_array *array, cf_array_element *element, ...);

/**
 * @brief Removes the last element from an array.
 *
 * When the array is not empty, the removed array element is written through
 * `element` and the logical length is decremented by one.
 *
 * @param array Array to pop from.
 * @param element Output pointer receiving the removed array element.
 * @return `CF_OK` on success, or a status describing null or invalid-state
 * conditions.
 */
cf_status cf_array_pop(cf_array *array, cf_array_element *element);

/**
 * @brief Read an element from an array by index.
 *
 * The selected element is copied into `element`. The array contents, length,
 * and capacity are not modified.
 *
 * @param array Source array to read from.
 * @param index Zero-based element index to read.
 * @param element Output pointer receiving the selected element.
 * @return `CF_OK` on success, `CF_ERR_BOUNDS` when `index` is outside the
 * current logical length, or another status for null or invalid-state
 * conditions.
 */
cf_status cf_array_get(cf_array *array, cf_usize index, cf_array_element *element);

/**
 * @brief Replace an existing array element by index.
 *
 * The selected array slot is overwritten with the value pointed to by
 * `element`. The array length and capacity are not changed.
 *
 * @param array Destination array to modify.
 * @param index Zero-based element index to replace.
 * @param element Input pointer providing the replacement element.
 * @return `CF_OK` on success, `CF_ERR_BOUNDS` when `index` is outside the
 * current logical length, or another status for null or invalid-state
 * conditions.
 */
cf_status cf_array_set(cf_array *array, cf_usize index, cf_array_element *element);

/**
 * @brief Check whether an array satisfies the framework's structural rules.
 *
 * A valid array has internally consistent `data`, `cap`, and `len` fields. A
 * null data pointer is only valid when both logical length and capacity are
 * zero.
 *
 * @param array Array to validate.
 * @return `CF_TRUE` when the array is structurally valid, otherwise
 * `CF_FALSE`.
 */
cf_bool cf_array_is_valid(cf_array *array);

/**
 * @brief Report whether an array currently contains no logical elements.
 *
 * Invalid arrays are treated as not empty.
 *
 * @param array Array to inspect.
 * @return `CF_TRUE` when the array is valid and its logical length is zero,
 * otherwise `CF_FALSE`.
 */
cf_bool cf_array_is_empty(cf_array *array);

/**
 * @brief Print a diagnostic summary of an array's current state.
 *
 * The printed information includes the element storage pointer, logical
 * length, allocation capacity, and whether each allocator callback field is
 * set. When elements are present, each stored element slot pointer is also
 * listed.
 *
 * @param array Array to inspect and print.
 * @return void
 */
void cf_array_info(cf_array *array);

#endif /* CF_ARRAY_H */
