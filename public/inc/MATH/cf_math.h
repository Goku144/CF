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

#include "MATH/cf_math_storage.h"

#define CF_MATH_MAX_NODE_INPUTS 8
#define CF_MATH_MAX_RANK 8
#define CF_MATH_HIGHEST_RANK CF_MATH_MAX_RANK

typedef struct cf_math_grad_node cf_math_grad_node;
typedef struct cf_math_desc cf_math_desc;
typedef struct cf_math cf_math;

typedef enum cf_math_dtype
{
  CF_MATH_DTYPE_BOOL = 0,
  CF_MATH_DTYPE_I8,
  CF_MATH_DTYPE_U8,
  CF_MATH_DTYPE_I32,
  CF_MATH_DTYPE_FP8E5M2,
  CF_MATH_DTYPE_FP8E4M3,
  CF_MATH_DTYPE_BF16,
  CF_MATH_DTYPE_F16,
  CF_MATH_DTYPE_F32,
  CF_MATH_DTYPE_F64,
} cf_math_dtype;

typedef enum cf_math_grad_state
{
  CF_MATH_GRAD_NONE = 0,
  CF_MATH_GRAD_LEAF,
  CF_MATH_GRAD_INTERIOR,
  CF_MATH_GRAD_DETACHED,
} cf_math_grad_state;

typedef enum cf_math_op_kind
{
  CF_MATH_OP_NONE = 0,

  CF_MATH_OP_ADD,
  CF_MATH_OP_SUB,
  CF_MATH_OP_MUL,
  CF_MATH_OP_DIV,
  CF_MATH_OP_NEG,
  CF_MATH_OP_EXP,
  CF_MATH_OP_LOG,
  CF_MATH_OP_SQRT,

  CF_MATH_OP_MATMUL,
  CF_MATH_OP_LINEAR,

  CF_MATH_OP_RELU,
  CF_MATH_OP_GELU,
  CF_MATH_OP_SIGMOID,
  CF_MATH_OP_TANH,
  CF_MATH_OP_SOFTMAX,
  CF_MATH_OP_CROSS_ENTROPY,
  CF_MATH_OP_SUM,
  CF_MATH_OP_MEAN,

  CF_MATH_OP_LAYER_NORM,
  CF_MATH_OP_ATTENTION
} cf_math_op_kind;

typedef enum cf_math_desc_type
{
  CF_MATH_DESC_NONE = 0,
  CF_MATH_DESC_CUDNN,
  CF_MATH_DESC_LT,
  CF_MATH_DESC_DNNL,
} cf_math_desc_type;

typedef union cf_math_descriptor
{
  cudnnTensorDescriptor_t cudnn_tensor;
  cublasLtMatrixLayout_t  lt_layout;
  dnnl_memory_desc_t dnnl_desc; 
} cf_math_descriptor;

struct cf_math_desc
{
  int rank;
  int dim[CF_MATH_MAX_RANK];
  int strides[CF_MATH_MAX_RANK];

  cf_math_desc_type desc_type;
  cf_math_descriptor desc;

  cf_math_dtype dtype;
};

struct cf_math_grad_node
{
  cf_math_op_kind op;

  cf_math *input[CF_MATH_MAX_NODE_INPUTS];
  cf_usize input_num;

  cf_math *grad;
  cf_math_grad_state grad_state;
};

struct cf_math
{
  cf_usize byte_offset;
  cf_math_desc *desc;
  cf_math_grad_node *grad_fn;
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Multiply two bytes in the AES GF(2^8) finite field.
 * @param p First finite-field byte operand.
 * @param q Second finite-field byte operand.
 * @return Computed 8-bit value.
 */
cf_u8 cf_math_g8_mul_mod(cf_u8 p, cf_u8 q);

/**
 * @brief Rotate an 8-bit value left.
 * @param x Input value or tensor.
 * @param n Bit count, scalar exponent, or batch index.
 * @return Computed 8-bit value.
 */
cf_u8 cf_math_rotl8(cf_u8 x, cf_u8 n);

/**
 * @brief Rotate an 8-bit value right.
 * @param x Input value or tensor.
 * @param n Bit count, scalar exponent, or batch index.
 * @return Computed 8-bit value.
 */
cf_u8 cf_math_rotr8(cf_u8 x, cf_u8 n);

/**
 * @brief Rotate a 32-bit value left.
 * @param x Input value or tensor.
 * @param n Bit count, scalar exponent, or batch index.
 * @return Computed 32-bit value.
 */
cf_u32 cf_math_rotl32(cf_u32 x, cf_u8 n);

/**
 * @brief Rotate a 32-bit value right.
 * @param x Input value or tensor.
 * @param n Bit count, scalar exponent, or batch index.
 * @return Computed 32-bit value.
 */
cf_u32 cf_math_rotr32(cf_u32 x, cf_u8 n);

/**
 * @brief Return the smaller of two cf_usize values.
 * @param a First input value, matrix, or sparse matrix.
 * @param b Second input value, tensor, bias tensor, or dense matrix.
 * @return Computed size value.
 */
cf_usize cf_math_min_usize(cf_usize a, cf_usize b);

/**
 * @brief Return the larger of two cf_usize values.
 * @param a First input value, matrix, or sparse matrix.
 * @param b Second input value, tensor, bias tensor, or dense matrix.
 * @return Computed size value.
 */
cf_usize cf_math_max_usize(cf_usize a, cf_usize b);

/**
 * @brief Initialize a tensor descriptor and optionally create a backend descriptor.
 * @param desc Descriptor object to initialize.
 * @param rank Number of active dimensions in `dim`.
 * @param dim Tensor dimensions.
 * @param dtype Element data type.
 * @param desc_type Backend descriptor type to create, or `CF_MATH_DESC_NONE`.
 * @return `CF_OK` on success, or an error status when backend descriptor creation fails.
 */
cf_status cf_math_desc_create(cf_math_desc *desc, int rank, const int *dim, cf_math_dtype dtype, cf_math_desc_type desc_type);

/**
 * @brief Destroy the active backend descriptor and clear the descriptor object.
 * @param desc Descriptor object to destroy.
 */
void cf_math_desc_destroy(cf_math_desc *desc);

/**
 * @brief Bind a math object to a descriptor and reserve storage from a handle.
 * @param handle Storage handle that owns the backing arena.
 * @param math Math object to initialize and bind.
 * @param desc Descriptor that defines the object's shape, layout, and dtype.
 * @return `CF_OK` on success, or an error status when storage reservation fails.
 */
cf_status cf_math_bind(cf_math_handle *handle, cf_math *math, cf_math_desc *desc);

/**
 * @brief Rebind a math object to a descriptor and reserve fresh storage from a handle.
 * @param handle Storage handle that owns the backing arena.
 * @param math Math object to clear and bind again.
 * @param desc Descriptor that defines the object's new shape, layout, and dtype.
 * @return `CF_OK` on success, or an error status when storage reservation fails.
 */
cf_status cf_math_rebind(cf_math_handle *handle, cf_math *math, cf_math_desc *desc);

/**
 * @brief Clear a math object's descriptor, byte offset, and gradient binding.
 * @param math Math object to clear.
 */
void cf_math_unbind(cf_math *math);


#ifdef __cplusplus
}
#endif

#endif /* CF_MATH_H */
