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

#if !defined(CF_TENSOR_H)
#define CF_TENSOR_H

#include "MEMORY/cf_memory.h"
#include "MEMORY/cf_array.h"

#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_types.h"

#define CF_TENSOR_HIGHEST_RANK 8

/**
 * Device where tensor storage is currently owned.
 *
 * CUDA support is scaffolded here so operation dispatch can be extended without
 * changing the public tensor layout again.
 */
typedef enum cf_tensor_device
{
  CF_TENSOR_DEVICE_CPU = 0,
  CF_TENSOR_DEVICE_CUDA,
} cf_tensor_device;

/**
 * Element type stored by a tensor.
 *
 * The type controls allocation size and how generic tensor operations cast the
 * raw `data` pointer.
 */
typedef enum cf_tensor_type
{
  CF_TENSOR_CHAR = 0,
  CF_TENSOR_SHORT,
  CF_TENSOR_INT,
  CF_TENSOR_LONG,  
  CF_TENSOR_LL,  
  CF_TENSOR_FLOAT,  
  CF_TENSOR_DOUBLE,  
  CF_TENSOR_LD,
  CF_TENSOR_U8,
  CF_TENSOR_U16,
  CF_TENSOR_U32,
  CF_TENSOR_U64,
  CF_TENSOR_U128,
}cf_tensor_type;

/**
 * Runtime tensor layout information.
 *
 * `len` is the number of elements, not bytes. `stride` is expressed in elements
 * and follows row-major layout for tensors created by `cf_tensor_init`.
 */
typedef struct cf_tensor_metadata
{
  cf_usize len;
  cf_usize stride[CF_TENSOR_HIGHEST_RANK];
  cf_usize elem_size;
  cf_tensor_type elem_type;
} cf_tensor_metadata;

/**
 * Dense tensor object with rank up to `CF_TENSOR_HIGHEST_RANK`.
 *
 * `dim[0..rank-1]` stores the active shape. `data` is owned by the tensor after
 * successful initialization and must be released with `cf_tensor_destroy`.
 * `device` identifies where that storage is currently owned.
 */
typedef struct cf_tensor
{
  void *data;
  cf_usize dim[CF_TENSOR_HIGHEST_RANK];
  cf_usize rank;
  cf_tensor_device device;
  cf_tensor_metadata metadata;
}cf_tensor;

/**
 * Check whether a tensor has a consistent shape, data pointer, element type,
 * byte size, length, and row-major stride metadata.
 */
cf_bool cf_tensor_is_valid(cf_tensor *tensor);

/**
 * Initialize a dense tensor.
 *
 * The caller supplies `rank` active dimensions in `dim`. The output tensor owns
 * a zeroed allocation on success and must later be destroyed.
 *
 * @return `CF_OK` on success, `CF_ERR_NULL`, `CF_ERR_INVALID`,
 * `CF_ERR_OVERFLOW`, or `CF_ERR_OOM`.
 */
cf_status cf_tensor_init(cf_tensor *tensor, cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_tensor_type elem_type);

/**
 * Release tensor storage and reset the tensor to zero.
 */
void cf_tensor_destroy(cf_tensor *tensor);

/**
 * Read one element from `tensor` into `out_value`.
 *
 * `indexs` must contain one coordinate for every active rank dimension.
 */
cf_status cf_tensor_get(void *out_value, cf_tensor *tensor, cf_usize indexs[CF_TENSOR_HIGHEST_RANK]);

/**
 * Write one element into `tensor`.
 *
 * `indexs` must contain one coordinate for every active rank dimension.
 */
cf_status cf_tensor_set(cf_tensor *tensor, cf_usize indexs[CF_TENSOR_HIGHEST_RANK], void *value);

/**
 * Print a tensor to stdout in a readable nested matrix form.
 */
void cf_tensor_print(cf_tensor *tensor);

#ifdef CF_CUDA_AVAILABLE

#ifdef __cplusplus
extern "C" {
#endif

/**
 * CUDA elementwise tensor addition.
 *
 * `t_out` must already be initialized with the expected output shape.
 */
cf_status cf_tensor_add_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out);

/**
 * CUDA scalar multiplication entry point.
 *
 * Placeholder API for CUDA builds; implementation can be filled with a kernel
 * later while keeping `cf_tensor_scalar_mul` stable.
 */
cf_status cf_tensor_scalar_mul_gpu(cf_tensor *t1, void *scalar, cf_tensor *t_out);

/**
 * CUDA tensor multiplication entry point.
 *
 * This API is selected by `cf_tensor_mul` when CUDA is available.
 */
cf_status cf_tensor_mul_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out);

/**
 * CUDA matrix multiplication entry point.
 *
 * Placeholder API for CUDA builds; shape validation should stay compatible with
 * the CPU implementation before launching future kernels.
 */
cf_status cf_tensor_matrix_mul_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out);

#ifdef __cplusplus
}
#endif

#define cf_tensor_add(t1, t2, t_out) cf_tensor_add_gpu(t1, t2, t_out)

#define cf_tensor_mul(t1, t2, t_out) cf_tensor_mul_gpu(t1, t2, t_out)

#define cf_tensor_scalar_mul(t1, scalar, t_out) cf_tensor_scalar_mul_gpu(t1, scalar, t_out)

#define cf_tensor_matrice_mul(t1, t2, t_out) cf_tensor_matrix_mul_gpu(t1, t2, t_out)

#else

/**
 * CPU elementwise tensor addition.
 *
 * All tensors must have matching rank and dimensions. `t_out` must already be
 * initialized with the same shape.
 */
cf_status cf_tensor_add_cpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out);

/**
 * CPU scalar multiplication.
 *
 * Multiplies every element in `t1` by `scalar` and writes the result to
 * initialized `t_out`.
 */
cf_status cf_tensor_scalar_mul_cpu(cf_tensor *t1, void *scalar, cf_tensor *t_out);

/**
 * CPU matrix multiplication.
 *
 * Supports scalar fallback for rank 0 inputs, vector/matrix normalization for
 * rank 1 inputs, and batched matrix multiplication with broadcast-compatible
 * leading dimensions. `t_out` must already be initialized with the expected
 * result shape.
 */
cf_status cf_tensor_matrix_mul_cpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out);

#define cf_tensor_add(t1, t2, t_out) cf_tensor_add_cpu(t1, t2, t_out)

#define cf_tensor_scalar_mul(t1, scalar, t_out) cf_tensor_scalar_mul_cpu(t1, scalar, t_out)

#define cf_tensor_matrice_mul(t1, t2, t_out) cf_tensor_matrix_mul_cpu(t1, t2, t_out)

#endif

#endif /* CF_TENSOR_H */
