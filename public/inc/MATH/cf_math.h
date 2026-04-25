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

#if !defined(CF_MATH_H)
#define CF_MATH_H

#include "RUNTIME/cf_types.h"

/**
 * @brief Multiply two bytes in AES GF(2^8) using the 0x11B reduction polynomial.
 *
 * This helper performs finite-field multiplication suitable for AES round
 * transformations such as MixColumns and InvMixColumns.
 *
 * @param p Left multiplicand byte.
 * @param q Right multiplicand byte.
 * @return Product reduced in GF(2^8).
 */
cf_u8 cf_math_g8_mul_mod(cf_u8 p, cf_u8 q);

/**
 * @brief Rotate an 8-bit value left.
 *
 * The rotation count is reduced modulo 8, so values greater than 7 wrap
 * naturally.
 *
 * @param x Value to rotate.
 * @param n Bit count to rotate by.
 * @return Rotated 8-bit value.
 */
cf_u8 cf_math_rotl8(cf_u8 x, cf_u8 n);

/**
 * @brief Rotate an 8-bit value right.
 *
 * The rotation count is reduced modulo 8, so values greater than 7 wrap
 * naturally.
 *
 * @param x Value to rotate.
 * @param n Bit count to rotate by.
 * @return Rotated 8-bit value.
 */
cf_u8 cf_math_rotr8(cf_u8 x, cf_u8 n);

/**
 * @brief Rotate a 32-bit value left.
 *
 * The rotation count is reduced modulo 32, so values greater than 31 wrap
 * naturally.
 *
 * @param x Value to rotate.
 * @param n Bit count to rotate by.
 * @return Rotated 32-bit value.
 */
cf_u32 cf_math_rotl32(cf_u32 x, cf_u8 n);

/**
 * @brief Rotate a 32-bit value right.
 *
 * The rotation count is reduced modulo 32, so values greater than 31 wrap
 * naturally.
 *
 * @param x Value to rotate.
 * @param n Bit count to rotate by.
 * @return Rotated 32-bit value.
 */
cf_u32 cf_math_rotr32(cf_u32 x, cf_u8 n);

/**
 * @brief Return the smaller of two `cf_usize` values.
 *
 * @param a First value.
 * @param b Second value.
 * @return `a` when `a <= b`, otherwise `b`.
 */
cf_usize cf_math_min_usize(cf_usize a, cf_usize b);

/**
 * @brief Return the larger of two `cf_usize` values.
 *
 * @param a First value.
 * @param b Second value.
 * @return `a` when `a >= b`, otherwise `b`.
 */
cf_usize cf_math_max_usize(cf_usize a, cf_usize b);

#endif /* CF_MATH_H */
