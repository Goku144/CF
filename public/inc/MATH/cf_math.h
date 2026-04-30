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

typedef struct cf_math_metadata cf_math_metadata;
typedef struct cf_math_node cf_math_node;
typedef struct cf_math cf_math;

typedef enum cf_math_layout
{
  CF_MATH_LAYOUT_ROW_MAJOR = 0,
  CF_MATH_LAYOUT_COL_MAJOR,
  CF_MATH_LAYOUT_NCHW,
  CF_MATH_LAYOUT_NHWC,
  CF_MATH_LAYOUT_STRIDED,
} cf_math_layout;

typedef enum cf_math_shape
{
  CF_MATH_SHAPE_SCALAR = 0,
  CF_MATH_SHAPE_VECTOR,
  CF_MATH_SHAPE_MATRIX,
  CF_MATH_SHAPE_TENSOR,
} cf_math_shape; 

typedef enum cf_math_grad_state
{
  CF_MATH_GRAD_NONE = 0,
  CF_MATH_GRAD_LEAF,
  CF_MATH_GRAD_INTERIOR,
  CF_MATH_GRAD_DETACHED
} cf_math_grad_state;

typedef enum cf_math_op_kind
{
  CF_MATH_OP_NONE = 0,

  CF_MATH_OP_ADD,
  CF_MATH_OP_SUB,
  CF_MATH_OP_MUL,
  CF_MATH_OP_DIV,

  CF_MATH_OP_MATMUL,
  CF_MATH_OP_LINEAR,

  CF_MATH_OP_RELU,
  CF_MATH_OP_GELU,
  CF_MATH_OP_SOFTMAX,
  CF_MATH_OP_CROSS_ENTROPY,

  CF_MATH_OP_LAYER_NORM,
  CF_MATH_OP_ATTENTION
} cf_math_op_kind;

struct cf_math_metadata
{
  cf_usize rank;
  cf_usize dim[CF_MATH_MAX_RANK];
  cf_usize strides[CF_MATH_MAX_RANK];

  cf_usize len;
  cf_math_shape shape;
  cf_math_layout layout;
};

struct cf_math_node
{
  cf_math_op_kind op;
  cf_math *inputs[CF_MATH_MAX_NODE_INPUTS];
  cf_usize input_count;
};

struct cf_math
{
  void *data;
  cf_usize byte_offset;
  cf_usize byte_size;

  cf_math_metadata *metadata;
  cf_math_handle_t *handler;

  cf_math *grad;
  cf_math_node *grad_fn;
  cf_math_grad_state grad_state;
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
 * @brief Initialize reusable shape metadata.
 * @param metadata Metadata object to initialize.
 * @param dim Shape dimensions, required when rank is nonzero.
 * @param rank Number of active dimensions.
 * @param shape Coarse shape kind.
 * @param layout Memory layout interpretation.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_INVALID`, or `CF_ERR_OVERFLOW`.
 */
cf_status cf_math_metadata_init(cf_math_metadata *metadata, cf_usize dim[CF_MATH_MAX_RANK], cf_usize rank, cf_math_shape shape, cf_math_layout layout);

/**
 * @brief Bind a non-owning math view to a handler and metadata.
 * @param x Math view to bind.
 * @param handler Runtime/storage handler.
 * @param metadata Shape/layout metadata.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_INVALID`, `CF_ERR_BOUNDS`, `CF_ERR_OVERFLOW`, or CUDA memory errors.
 */
cf_status cf_math_bind(cf_math *x, cf_math_handle_t *handler, cf_math_metadata *metadata);

/**
 * @brief Unbind a math view and release its slice when no other view uses it.
 * @param x Math view to unbind.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, or `CF_ERR_BOUNDS`.
 */
cf_status cf_math_unbind(cf_math *x);

/**
 * @brief Rebind a math view to a new handler and metadata.
 * @param x Math view to rebind.
 * @param handler New runtime/storage handler.
 * @param metadata New shape/layout metadata.
 * @return Status from unbind or bind.
 */
cf_status cf_math_rebind(cf_math *x, cf_math_handle_t *handler, cf_math_metadata *metadata);

#ifdef __cplusplus
}
#endif

#endif /* CF_MATH_H */
