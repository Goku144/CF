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
 * @brief Apply an in-place elementwise math operation.
 *
 * `op1` is both input and destination; `op2` is read-only. This is a hot-path
 * operation: callers are responsible for ensuring both views are bound,
 * compatible, and have matching dtype/device/length.
 *
 * Supported v1 ops are `CF_MATH_OP_ADD`, `CF_MATH_OP_SUB`,
 * `CF_MATH_OP_MUL`, and `CF_MATH_OP_DIV` for `F32`, `F64`, and `I32`.
 *
 * @param op Operation kind.
 * @param op1 Destination and first input.
 * @param op2 Read-only second input.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, `CF_ERR_UNSUPPORTED`, or a CUDA runtime error.
 */
cf_status cf_math_op(cf_math_op_kind op, cf_math *op1, const cf_math *op2);

/**
 * @brief Validate compatibility for an in-place binary math operation.
 * @param op Operation kind.
 * @param op1 Destination and first input.
 * @param op2 Read-only second input.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, `CF_ERR_INVALID`, or `CF_ERR_UNSUPPORTED`.
 */
cf_status cf_math_op_check(cf_math_op_kind op, const cf_math *op1, const cf_math *op2);

/**
 * @brief Apply an out-of-place elementwise math operation.
 * @param op Operation kind.
 * @param out Bound destination view.
 * @param a Read-only first input.
 * @param b Read-only second input.
 * @return Status from compatibility checks, copy, or `cf_math_op`.
 */
cf_status cf_math_op_out(cf_math_op_kind op, cf_math *out, const cf_math *a, const cf_math *b);

/**
 * @brief Apply an in-place unary math operation.
 * @param op Unary operation kind.
 * @param x Bound input/output view.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, `CF_ERR_UNSUPPORTED`, or a CUDA runtime error.
 */
cf_status cf_math_unary(cf_math_op_kind op, cf_math *x);

/**
 * @brief Apply an out-of-place unary math operation.
 * @param op Unary operation kind.
 * @param out Bound destination view.
 * @param x Read-only input view.
 * @return Status from compatibility checks, copy, or `cf_math_unary`.
 */
cf_status cf_math_unary_out(cf_math_op_kind op, cf_math *out, const cf_math *x);

/**
 * @brief Apply an in-place scalar math operation.
 * @param op Binary operation kind applied with a scalar right operand.
 * @param x Bound input/output view.
 * @param scalar Scalar value; cast to destination dtype where needed.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, `CF_ERR_UNSUPPORTED`, or a CUDA runtime error.
 */
cf_status cf_math_scalar(cf_math_op_kind op, cf_math *x, double scalar);

/**
 * @brief Apply an out-of-place scalar math operation.
 * @param op Binary operation kind applied with a scalar right operand.
 * @param out Bound destination view.
 * @param x Read-only input view.
 * @param scalar Scalar value.
 * @return Status from compatibility checks, copy, or `cf_math_scalar`.
 */
cf_status cf_math_scalar_out(cf_math_op_kind op, cf_math *out, const cf_math *x, double scalar);

/**
 * @brief Reduce all elements into a one-element sum output.
 * @param out Bound one-element destination view.
 * @param x Bound input view.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, `CF_ERR_INVALID`, `CF_ERR_UNSUPPORTED`, or a CUDA runtime error.
 */
cf_status cf_math_reduce_sum(cf_math *out, const cf_math *x);

/**
 * @brief Reduce all elements into a one-element mean output.
 * @param out Bound one-element destination view.
 * @param x Bound input view.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, `CF_ERR_INVALID`, `CF_ERR_UNSUPPORTED`, or a CUDA runtime error.
 */
cf_status cf_math_reduce_mean(cf_math *out, const cf_math *x);

/**
 * @brief Multiply two row-major 2D matrices into a bound output view.
 * @param out Bound output matrix with shape `[M, N]`.
 * @param a Bound left matrix with shape `[M, K]`.
 * @param b Bound right matrix with shape `[K, N]`.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, `CF_ERR_INVALID`, `CF_ERR_UNSUPPORTED`, or a CUDA runtime error.
 */
cf_status cf_math_matmul(cf_math *out, const cf_math *a, const cf_math *b);

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
 * @brief Copy host data into a bound math view.
 * @param dst Bound math view that receives the data.
 * @param host_data CPU-readable source buffer.
 * @param count Number of elements to copy.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, `CF_ERR_BOUNDS`, `CF_ERR_OVERFLOW`, or a copy/sync error.
 */
cf_status cf_math_cpy_h2d(cf_math *dst, const void *host_data, cf_usize count);

/**
 * @brief Copy data from a bound math view into host memory.
 * @param src Bound math view to read.
 * @param host_data CPU-writeable destination buffer.
 * @param count Number of elements to copy.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, `CF_ERR_BOUNDS`, `CF_ERR_OVERFLOW`, or a copy/sync error.
 */
cf_status cf_math_cpy_d2h(const cf_math *src, void *host_data, cf_usize count);

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
