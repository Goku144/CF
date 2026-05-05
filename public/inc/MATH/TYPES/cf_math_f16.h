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

#if !defined(CF_MATH_F16_H)
#define CF_MATH_F16_H

#define CF_GELU_COEFF_A 0.79788456f
#define CF_GELU_COEFF_B 0.03567741f

typedef struct cf_math_handle cf_math_handle;
typedef struct cf_math cf_math;

#ifdef __cplusplus
extern "C" {
#endif

void cf_math_add_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B);

void cf_math_sub_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B);

void cf_math_mul_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B);

void cf_math_div_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B);

void cf_math_neg_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_sqrt_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_exp_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_log_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_tanh_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_relu_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_sigmoid_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_gelu_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_reduce_sum_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_reduce_mean_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

#ifdef __cplusplus
}
#endif

#endif /* CF_MATH_F16_H */
