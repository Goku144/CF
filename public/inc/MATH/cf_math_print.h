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

#if !defined(CF_MATH_PRINT_H)
#define CF_MATH_PRINT_H

#include "MATH/cf_math.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Print a handle-backed tensor as nested bracketed rows.
 * @param handle Storage handle that owns the tensor memory.
 * @param math Tensor metadata to print.
 * @return `CF_OK` on success, or an error status when memory access fails.
 */
cf_status cf_math_print(cf_math_handle *handle, const cf_math *math);

#ifdef __cplusplus
}
#endif

#endif /* CF_MATH_PRINT_H */
