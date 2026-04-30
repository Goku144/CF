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
 * @brief Print a readable shape summary for a math view.
 * @param x Math view whose metadata should be printed.
 * @return `CF_OK`, `CF_ERR_NULL`, or `CF_ERR_STATE` when the view has no metadata.
 */
cf_status cf_math_print_shape(const cf_math *x);

#ifdef __cplusplus
}
#endif

#endif /* CF_MATH_PRINT_H */
