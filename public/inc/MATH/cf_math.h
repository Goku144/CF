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

#endif /* CF_MATH_H */
