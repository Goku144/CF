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

#include "MATH/cf_math.h"

static cf_usize cf_math_type_size(cf_math_dtype dtype)
{
  switch (dtype)
  {
    case CF_MATH_DTYPE_BOOL: return sizeof (cf_bool);
    case CF_MATH_DTYPE_I8: return sizeof (cf_i8); 
    case CF_MATH_DTYPE_U8: return sizeof (cf_u8);
    case CF_MATH_DTYPE_I32: return sizeof (cf_i32);
    case CF_MATH_DTYPE_FP8E5M2: return sizeof (cf_u8);
    case CF_MATH_DTYPE_FP8E4M3: return sizeof (cf_u8);
    case CF_MATH_DTYPE_BF16: return sizeof (cf_u16);
    case CF_MATH_DTYPE_F16: return sizeof (cf_u16);
    case CF_MATH_DTYPE_F32: return sizeof (float);
    case CF_MATH_DTYPE_F64: return sizeof (double);
  }
  return (cf_usize) -1;
}