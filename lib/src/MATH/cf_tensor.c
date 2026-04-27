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

#include "MATH/cf_tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CF_TENSOR_TYPE_CASE(tensor_type, type) \
case tensor_type: \
    return sizeof (type) \

#define CF_TENSOR_GET_CASE(tensor_type , to, from, index, type) \
case tensor_type: \
      *((type *) to) = ((type *) from)[index]; \
      break \

#define CF_TENSOR_SET_CASE(tensor_type , to, from, index, type) \
case tensor_type: \
      ((type *) to)[index] = *((type *) from); \
      break \

#define CF_TENSOR_ADD_CASE(tensor_type, a, b, c, len, type) \
case tensor_type: \
  for (cf_usize index = 0; index < len; index++) \
  { \
    ((type *) c)[index] = ((type *) a)[index] + ((type *) b)[index]; \
  } \
  break \

#define CF_TENSOR_MUL_CASE(tensor_type, a, b, c, len, type) \
case tensor_type: \
  for (cf_usize index = 0; index < len; index++) \
  { \
    ((type *) c)[index] = ((type *) a)[index] * ((type *) b)[index]; \
  } \
  break \

#define CF_TENSOR_SCALAR_CASE(tensor_type, a, scalar, c, len, type) \
case tensor_type: \
  for (cf_usize index = 0; index < len; index++) \
  { \
    ((type *) c)[index] = ((type *) a)[index] * *((type *) scalar); \
  } \
  break \

#define CF_TENSOR_MATRIX_MUL_CASE(tensor_type, A, B, C, type) \
case tensor_type :\
{\
  cf_usize M = A->dim[max_rank - 2];\
  cf_usize K = A->dim[max_rank - 1];\
  cf_usize N = B->dim[max_rank - 1];\
  for (size_t batch = 0; batch < count; batch++) \
	  { \
	    cf_usize base_A = 0; \
	    cf_usize base_B = 0; \
    cf_usize base_C = 0; \
    cf_usize rest = batch; \
 \
    for (cf_isize i = (cf_isize) max_rank - 3; i >= 0; i--) \
    { \
      cf_usize coord = rest % C->dim[i]; \
      rest /= C->dim[i]; \
 \
      cf_usize coord_A = A->dim[i] == 1 ? 0 : coord; \
      cf_usize coord_B = B->dim[i] == 1 ? 0 : coord; \
 \
      base_A += coord_A * A->metadata.stride[i]; \
      base_B += coord_B * B->metadata.stride[i]; \
      base_C += coord * C->metadata.stride[i]; \
    } \
     \
	    for (cf_usize m = 0; m < M; m++) \
	    { \
	      for (cf_usize n = 0; n < N; n++) \
	      { \
	        type sum = 0; \
	        for (cf_usize k = 0; k < K; k++) \
	        { \
          sum += ((type *)A->data)[base_A + m * A->metadata.stride[max_rank - 2] + k * A->metadata.stride[max_rank - 1]]*  \
                 ((type *)B->data)[base_B + k * B->metadata.stride[max_rank - 2] + n * B->metadata.stride[max_rank - 1]]; \
        } \
        ((type *)C->data)[base_C + m * C->metadata.stride[max_rank - 2] + n * C->metadata.stride[max_rank - 1]] = sum; \
      } \
	    } \
	  } \
} \
	  break \


/*
 * Map the framework tensor enum to the native storage width used by the CPU
 * backend. This is the root type table for allocation, validation, and typed
 * dispatch in this file.
 */
static cf_usize cf_tensor_type_size(cf_tensor_type elem_type)
{
  switch (elem_type)
  {
    CF_TENSOR_TYPE_CASE(CF_TENSOR_CHAR, char);
    CF_TENSOR_TYPE_CASE(CF_TENSOR_SHORT, short);
    CF_TENSOR_TYPE_CASE(CF_TENSOR_INT, int);
    CF_TENSOR_TYPE_CASE(CF_TENSOR_LONG, long);
    CF_TENSOR_TYPE_CASE(CF_TENSOR_LL, long long);
    CF_TENSOR_TYPE_CASE(CF_TENSOR_FLOAT, float);
    CF_TENSOR_TYPE_CASE(CF_TENSOR_DOUBLE, double);
    CF_TENSOR_TYPE_CASE(CF_TENSOR_LD, long double);
    CF_TENSOR_TYPE_CASE(CF_TENSOR_U8, cf_u8);
    CF_TENSOR_TYPE_CASE(CF_TENSOR_U16, cf_u16);
    CF_TENSOR_TYPE_CASE(CF_TENSOR_U32, cf_u32);
    CF_TENSOR_TYPE_CASE(CF_TENSOR_U64, cf_u64);
    CF_TENSOR_TYPE_CASE(CF_TENSOR_U128, cf_u128);
    default: return 0;
  }
}

/*
 * Decide whether the CPU backend can store and operate on a tensor element
 * type. CPU currently supports every framework tensor type that has a native
 * size entry.
 */
static cf_bool cf_tensor_cpu_type_supported(cf_tensor_type elem_type)
{
  return cf_tensor_type_size(elem_type) != 0;
}

/*
 * Build the common dense tensor metadata used by CPU and mirrored by the CUDA
 * backend: shape, row-major strides, element count, element size, and type.
 * Storage allocation is intentionally left to the caller.
 */
static cf_status cf_tensor_setup_metadata(cf_tensor *tensor, cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_tensor_type elem_type)
{
  if(tensor == CF_NULL || dim == CF_NULL) return CF_ERR_NULL;
  if(rank > CF_TENSOR_HIGHEST_RANK) return CF_ERR_INVALID;
  if(!cf_tensor_cpu_type_supported(elem_type)) return CF_ERR_INVALID;

  *tensor = (cf_tensor) {0};
  tensor->metadata.elem_size = cf_tensor_type_size(elem_type);
  tensor->metadata.elem_type = elem_type;

  for(cf_usize i = 0; i < rank; i++)
  {
    if(dim[i] == 0) return CF_ERR_INVALID;
    tensor->dim[i] = dim[i];
  }
  tensor->rank = rank;

  tensor->metadata.len = 1;
  cf_usize end = rank > 0 ? rank - 1 : 0;
  for (cf_usize i = 0; i < rank; i++)
  {
    tensor->metadata.stride[end - i] = tensor->metadata.len;
    if(tensor->metadata.len > CF_USIZE_MAX / dim[end - i]) return CF_ERR_OVERFLOW;
    tensor->metadata.len *= dim[end - i];
  }

  return CF_OK;
}

/*
 * Fast CPU precondition for operations that directly dereference tensor->data.
 * CUDA-owned tensors are rejected here because CPU loops cannot read
 * device_data without an explicit transfer.
 */
static cf_status cf_tensor_require_data(cf_tensor *tensor)
{
  if(tensor == CF_NULL) return CF_ERR_NULL;
  if(!cf_tensor_is_valid(tensor)) return CF_ERR_INVALID;
  if(tensor->data == CF_NULL || tensor->device != CF_TENSOR_DEVICE_CPU) return CF_ERR_STATE;
  return CF_OK;
}

/*
 * Shared arithmetic precondition: binary tensor operations need all operands
 * and the output buffer to agree on element type and byte width.
 */
static cf_status cf_tensor_require_same_type(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out)
{
  if(t1->metadata.elem_type != t2->metadata.elem_type || t1->metadata.elem_type != t_out->metadata.elem_type)
    return CF_ERR_INVALID;

  if(t1->metadata.elem_size != t2->metadata.elem_size || t1->metadata.elem_size != t_out->metadata.elem_size)
    return CF_ERR_INVALID;

  return CF_OK;
}

/*
 * Validate the structural tensor contract used across the framework: supported
 * device tag, live storage for the active device, consistent element metadata,
 * shape product, and row-major stride table.
 */
cf_bool cf_tensor_is_valid(cf_tensor *tensor)
{
  if(tensor == CF_NULL) return CF_FALSE;
  CF_ASSERT_TYPE_SIZE(*tensor, cf_tensor);
  if(tensor->rank > CF_TENSOR_HIGHEST_RANK) return CF_FALSE;
  if(tensor->device != CF_TENSOR_DEVICE_CPU && tensor->device != CF_TENSOR_DEVICE_CUDA) return CF_FALSE;

  if(tensor->metadata.len == 0) return CF_FALSE;
  if(tensor->metadata.elem_size == 0) return CF_FALSE;
  if(cf_tensor_type_size(tensor->metadata.elem_type) != tensor->metadata.elem_size) return CF_FALSE;

  if(tensor->device == CF_TENSOR_DEVICE_CPU && tensor->data == CF_NULL) return CF_FALSE;
  if(tensor->device == CF_TENSOR_DEVICE_CUDA && tensor->device_data == CF_NULL) return CF_FALSE;

  if(tensor->data == CF_NULL && tensor->device_data == CF_NULL)
  {
    return CF_FALSE;
  }

  if(tensor->rank == 0) return tensor->metadata.len == 1;

  cf_usize elements = 1;
  for(cf_usize i = 0; i < tensor->rank; i++)
  {
    if(tensor->dim[i] == 0) return CF_FALSE;
    if(elements > CF_USIZE_MAX / tensor->dim[i]) return CF_FALSE;
    elements *= tensor->dim[i];
  }

  if(elements > CF_USIZE_MAX / tensor->metadata.elem_size) return CF_FALSE;
  if(elements != tensor->metadata.len) return CF_FALSE;

  cf_usize stride = 1;
  for(cf_usize i = tensor->rank; i > 0; i--)
  {
    cf_usize index = i - 1;
    if(tensor->metadata.stride[index] != stride) return CF_FALSE;
    if(stride > CF_USIZE_MAX / tensor->dim[index]) return CF_FALSE;
    stride *= tensor->dim[index];
  }

  return CF_TRUE;
}

/*
 * Create a CPU-resident dense tensor. The public macro cf_tensor_init maps here
 * in CPU builds; CUDA builds can still call this explicitly when host-side data
 * access is required.
 */
cf_status cf_tensor_init_cpu(cf_tensor *tensor, cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_tensor_type elem_type)
{
  cf_status status = cf_tensor_setup_metadata(tensor, dim, rank, elem_type);
  if(status != CF_OK) return status;
  tensor->device = CF_TENSOR_DEVICE_CPU;

  tensor->data = malloc(tensor->metadata.len * tensor->metadata.elem_size);
  if(tensor->data == CF_NULL) return CF_ERR_OOM;

  memset(tensor->data, 0, tensor->metadata.len * tensor->metadata.elem_size);

  return CF_OK;
}

/*
 * Release CPU-owned storage and reset the object. This function is intentionally
 * small so callers can use it freely in cleanup paths.
 */
void cf_tensor_destroy_cpu(cf_tensor *tensor)
{
  if(tensor == CF_NULL) return;
  if(tensor->data != CF_NULL) free(tensor->data);
  *tensor = (cf_tensor) {0};
}

/*
 * Read one logical tensor element from CPU storage. Multi-dimensional indices
 * are translated through row-major strides into a single flat data offset.
 */
cf_status cf_tensor_get_cpu(void *out_value, cf_tensor *tensor, cf_usize indexs[CF_TENSOR_HIGHEST_RANK])
{
  if(out_value == CF_NULL || indexs == CF_NULL) return CF_ERR_NULL;
  cf_status status = cf_tensor_require_data(tensor);
  if(status != CF_OK) return status;

  cf_usize index = 0;
  for (cf_usize i = 0; i < tensor->rank; i++)
  {
    if(indexs[i] >= tensor->dim[i]) return CF_ERR_BOUNDS;
    index += tensor->metadata.stride[i] * indexs[i];
  }

  switch (tensor->metadata.elem_type)
  {
    CF_TENSOR_GET_CASE(CF_TENSOR_CHAR, out_value, tensor->data, index, char);
    CF_TENSOR_GET_CASE(CF_TENSOR_SHORT, out_value, tensor->data, index, short);
    CF_TENSOR_GET_CASE(CF_TENSOR_INT, out_value, tensor->data, index, int);
    CF_TENSOR_GET_CASE(CF_TENSOR_LONG, out_value, tensor->data, index, long);
    CF_TENSOR_GET_CASE(CF_TENSOR_LL, out_value, tensor->data, index, long long);
    CF_TENSOR_GET_CASE(CF_TENSOR_FLOAT, out_value, tensor->data, index, float);
    CF_TENSOR_GET_CASE(CF_TENSOR_DOUBLE, out_value, tensor->data, index, double);
    CF_TENSOR_GET_CASE(CF_TENSOR_LD, out_value, tensor->data, index, long double);
    CF_TENSOR_GET_CASE(CF_TENSOR_U8, out_value, tensor->data, index, cf_u8);
    CF_TENSOR_GET_CASE(CF_TENSOR_U16, out_value, tensor->data, index, cf_u16);
    CF_TENSOR_GET_CASE(CF_TENSOR_U32, out_value, tensor->data, index, cf_u32);
    CF_TENSOR_GET_CASE(CF_TENSOR_U64, out_value, tensor->data, index, cf_u64);
    CF_TENSOR_GET_CASE(CF_TENSOR_U128, out_value, tensor->data, index, cf_u128);
    default: return CF_ERR_INVALID;
  }
  return CF_OK;
}

/*
 * Write one logical tensor element into CPU storage using the same stride-based
 * index translation as cf_tensor_get_cpu.
 */
cf_status cf_tensor_set_cpu(cf_tensor *tensor, cf_usize indexs[CF_TENSOR_HIGHEST_RANK], void *value)
{
  if(indexs == CF_NULL || value == CF_NULL) return CF_ERR_NULL;
  cf_status status = cf_tensor_require_data(tensor);
  if(status != CF_OK) return status;

  cf_usize index = 0; 
  for (cf_usize i = 0; i < tensor->rank; i++)
  {
    if(indexs[i] >= tensor->dim[i]) return CF_ERR_BOUNDS;
    index += tensor->metadata.stride[i] * indexs[i];
  }

  switch (tensor->metadata.elem_type)
  {
    CF_TENSOR_SET_CASE(CF_TENSOR_CHAR, tensor->data, value, index, char);
    CF_TENSOR_SET_CASE(CF_TENSOR_SHORT, tensor->data, value, index, short);
    CF_TENSOR_SET_CASE(CF_TENSOR_INT, tensor->data, value, index, int);
    CF_TENSOR_SET_CASE(CF_TENSOR_LONG, tensor->data, value, index, long);
    CF_TENSOR_SET_CASE(CF_TENSOR_LL, tensor->data, value, index, long long);
    CF_TENSOR_SET_CASE(CF_TENSOR_FLOAT, tensor->data, value, index, float);
    CF_TENSOR_SET_CASE(CF_TENSOR_DOUBLE, tensor->data, value, index, double);
    CF_TENSOR_SET_CASE(CF_TENSOR_LD, tensor->data, value, index, long double);
    CF_TENSOR_SET_CASE(CF_TENSOR_U8, tensor->data, value, index, cf_u8);
    CF_TENSOR_SET_CASE(CF_TENSOR_U16, tensor->data, value, index, cf_u16);
    CF_TENSOR_SET_CASE(CF_TENSOR_U32, tensor->data, value, index, cf_u32);
    CF_TENSOR_SET_CASE(CF_TENSOR_U64, tensor->data, value, index, cf_u64);
    CF_TENSOR_SET_CASE(CF_TENSOR_U128, tensor->data, value, index, cf_u128);
    default: return CF_ERR_INVALID;
  }
  return CF_OK;
}

/*
 * CPU elementwise addition. This is the scalar host fallback for the framework
 * tensor add API when CUDA is unavailable or when callers choose CPU storage.
 */
cf_status cf_tensor_add_cpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out)
{
  cf_status status = cf_tensor_require_data(t1);
  if(status != CF_OK) return status;
  status = cf_tensor_require_data(t2);
  if(status != CF_OK) return status;
  status = cf_tensor_require_data(t_out);
  if(status != CF_OK) return status;
  status = cf_tensor_require_same_type(t1, t2, t_out);
  if(status != CF_OK) return status;

  if(t1->rank != t2->rank || t1->rank != t_out->rank) return CF_ERR_INVALID;

  for (cf_usize i = 0; i < t_out->rank; i++)
    if(t1->dim[i] != t2->dim[i] || t1->dim[i] != t_out->dim[i]) return CF_ERR_INVALID;

  switch (t_out->metadata.elem_type)
  {
    CF_TENSOR_ADD_CASE(CF_TENSOR_CHAR, t1->data, t2->data, t_out->data, t1->metadata.len, char);
    CF_TENSOR_ADD_CASE(CF_TENSOR_SHORT, t1->data, t2->data, t_out->data, t1->metadata.len, short);
    CF_TENSOR_ADD_CASE(CF_TENSOR_INT, t1->data, t2->data, t_out->data, t1->metadata.len, int);
    CF_TENSOR_ADD_CASE(CF_TENSOR_LONG, t1->data, t2->data, t_out->data, t1->metadata.len, long);
    CF_TENSOR_ADD_CASE(CF_TENSOR_LL, t1->data, t2->data, t_out->data, t1->metadata.len, long long);
    CF_TENSOR_ADD_CASE(CF_TENSOR_FLOAT, t1->data, t2->data, t_out->data, t1->metadata.len, float);
    CF_TENSOR_ADD_CASE(CF_TENSOR_DOUBLE, t1->data, t2->data, t_out->data, t1->metadata.len, double);
    CF_TENSOR_ADD_CASE(CF_TENSOR_LD, t1->data, t2->data, t_out->data, t1->metadata.len, long double);
    CF_TENSOR_ADD_CASE(CF_TENSOR_U8, t1->data, t2->data, t_out->data, t1->metadata.len, cf_u8);
    CF_TENSOR_ADD_CASE(CF_TENSOR_U16, t1->data, t2->data, t_out->data, t1->metadata.len, cf_u16);
    CF_TENSOR_ADD_CASE(CF_TENSOR_U32, t1->data, t2->data, t_out->data, t1->metadata.len, cf_u32);
    CF_TENSOR_ADD_CASE(CF_TENSOR_U64, t1->data, t2->data, t_out->data, t1->metadata.len, cf_u64);
    CF_TENSOR_ADD_CASE(CF_TENSOR_U128, t1->data, t2->data, t_out->data, t1->metadata.len, cf_u128);
    default: return CF_ERR_INVALID;
  }
  return CF_OK;
}

/*
 * CPU elementwise multiplication for tensors with identical shape and type.
 * Matrix-style multiplication is handled separately by cf_tensor_matrix_mul_cpu.
 */
cf_status cf_tensor_mul_cpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out)
{
  cf_status status = cf_tensor_require_data(t1);
  if(status != CF_OK) return status;
  status = cf_tensor_require_data(t2);
  if(status != CF_OK) return status;
  status = cf_tensor_require_data(t_out);
  if(status != CF_OK) return status;
  status = cf_tensor_require_same_type(t1, t2, t_out);
  if(status != CF_OK) return status;

  if(t1->rank != t2->rank || t1->rank != t_out->rank) return CF_ERR_INVALID;

  for (cf_usize i = 0; i < t_out->rank; i++)
    if(t1->dim[i] != t2->dim[i] || t1->dim[i] != t_out->dim[i]) return CF_ERR_INVALID;

  switch (t_out->metadata.elem_type)
  {
    CF_TENSOR_MUL_CASE(CF_TENSOR_CHAR, t1->data, t2->data, t_out->data, t1->metadata.len, char);
    CF_TENSOR_MUL_CASE(CF_TENSOR_SHORT, t1->data, t2->data, t_out->data, t1->metadata.len, short);
    CF_TENSOR_MUL_CASE(CF_TENSOR_INT, t1->data, t2->data, t_out->data, t1->metadata.len, int);
    CF_TENSOR_MUL_CASE(CF_TENSOR_LONG, t1->data, t2->data, t_out->data, t1->metadata.len, long);
    CF_TENSOR_MUL_CASE(CF_TENSOR_LL, t1->data, t2->data, t_out->data, t1->metadata.len, long long);
    CF_TENSOR_MUL_CASE(CF_TENSOR_FLOAT, t1->data, t2->data, t_out->data, t1->metadata.len, float);
    CF_TENSOR_MUL_CASE(CF_TENSOR_DOUBLE, t1->data, t2->data, t_out->data, t1->metadata.len, double);
    CF_TENSOR_MUL_CASE(CF_TENSOR_LD, t1->data, t2->data, t_out->data, t1->metadata.len, long double);
    CF_TENSOR_MUL_CASE(CF_TENSOR_U8, t1->data, t2->data, t_out->data, t1->metadata.len, cf_u8);
    CF_TENSOR_MUL_CASE(CF_TENSOR_U16, t1->data, t2->data, t_out->data, t1->metadata.len, cf_u16);
    CF_TENSOR_MUL_CASE(CF_TENSOR_U32, t1->data, t2->data, t_out->data, t1->metadata.len, cf_u32);
    CF_TENSOR_MUL_CASE(CF_TENSOR_U64, t1->data, t2->data, t_out->data, t1->metadata.len, cf_u64);
    CF_TENSOR_MUL_CASE(CF_TENSOR_U128, t1->data, t2->data, t_out->data, t1->metadata.len, cf_u128);
    default: return CF_ERR_INVALID;
  }
  return CF_OK;
}

/*
 * CPU scalar multiplication. The scalar pointer is interpreted as the same
 * element type as the tensor and applied to every flat element.
 */
cf_status cf_tensor_scalar_mul_cpu(cf_tensor *t1, void *scalar, cf_tensor *t_out)
{
  if(scalar == CF_NULL) return CF_ERR_NULL;
  cf_status status = cf_tensor_require_data(t1);
  if(status != CF_OK) return status;
  status = cf_tensor_require_data(t_out);
  if(status != CF_OK) return status;
  if(t1->metadata.elem_type != t_out->metadata.elem_type || t1->metadata.elem_size != t_out->metadata.elem_size)
    return CF_ERR_INVALID;

  if(t1->rank != t_out->rank) return CF_ERR_INVALID;
  for (cf_usize i = 0; i < t_out->rank; i++)
    if(t1->dim[i] != t_out->dim[i]) return CF_ERR_INVALID;

  switch (t_out->metadata.elem_type)
  {
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_CHAR, t1->data, scalar, t_out->data, t1->metadata.len, char);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_SHORT, t1->data, scalar, t_out->data, t1->metadata.len, short);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_INT, t1->data, scalar, t_out->data, t1->metadata.len, int);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_LONG, t1->data, scalar, t_out->data, t1->metadata.len, long);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_LL, t1->data, scalar, t_out->data, t1->metadata.len, long long);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_FLOAT, t1->data, scalar, t_out->data, t1->metadata.len, float);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_DOUBLE, t1->data, scalar, t_out->data, t1->metadata.len, double);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_LD, t1->data, scalar, t_out->data, t1->metadata.len, long double);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_U8, t1->data, scalar, t_out->data, t1->metadata.len, cf_u8);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_U16, t1->data, scalar, t_out->data, t1->metadata.len, cf_u16);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_U32, t1->data, scalar, t_out->data, t1->metadata.len, cf_u32);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_U64, t1->data, scalar, t_out->data, t1->metadata.len, cf_u64);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_U128, t1->data, scalar, t_out->data, t1->metadata.len, cf_u128);
    default: return CF_ERR_INVALID;
  }
  return CF_OK;
}

/*
 * Check the core matrix multiplication rule before rank normalization:
 * the left inner dimension must match the right inner dimension.
 */
static cf_bool cf_tensor_matrix_mul_inner_dim_match(cf_tensor *t1, cf_tensor *t2)
{
  return t1->dim[t1->rank - 1] == t2->dim[t2->rank == 1 ? 0 : t2->rank - 2];
}

/*
 * Normalize vector and matrix operands into a shared matrix/batch rank. This
 * lets the implementation use one batched matmul loop for rank-1, rank-2, and
 * higher-rank broadcast-style inputs.
 */
static void cf_tensor_matrix_mul_normalize_rank(cf_tensor *t1, cf_tensor *t2, cf_usize *out_max_rank)
{
  cf_usize max_rank = t1->rank > 2 ? t1->rank : 2;
  max_rank = t2->rank > max_rank ? t2->rank : max_rank;
  cf_usize len1 = t1->rank - 1, len2 = t2->rank - 1, len_max = max_rank - 1;
  cf_usize tmp1 = 1, tmp2 = 1;
  for (cf_usize i = 0; i < max_rank; i++)
  {
    t1->metadata.stride[len_max - i] =  len1 >= i ? tmp1 : 0;
    t2->metadata.stride[len_max - i] = len2 >= i ? tmp2: 0;
    t1->dim[len_max - i] = len1 >= i ? t1->dim[len1 - i] : 1;
    t2->dim[len_max - i] = len2 >= i ? t2->dim[len2 - i] : 1;
    tmp1 *= t1->dim[len_max - i];
    tmp2 *= t2->dim[len_max - i];
  }

  if(t2->rank == 1)
  {
    cf_usize tmp = t2->dim[len_max - 1];
    t2->dim[len_max - 1] = t2->dim[len_max];
    t2->dim[len_max] = tmp;

    tmp = t2->metadata.stride[len_max - 1];
    t2->metadata.stride[len_max - 1] = t2->metadata.stride[len_max];
    t2->metadata.stride[len_max] = tmp;
  }

  t1->rank = t2->rank = max_rank;
  *out_max_rank = max_rank;
}

/*
 * Validate leading batch dimensions after normalization and compute how many
 * matrix multiplications must be executed. Dimensions of size 1 are broadcast.
 */
static cf_status cf_tensor_matrix_mul_batch_dims_match(cf_tensor *t1, cf_tensor *t2, cf_usize max_rank, cf_usize *out_count)
{
  cf_usize count = 1;

  for (cf_usize i = 0; i < max_rank - 2; i++)
  {
    if(t1->dim[i] != t2->dim[i] && t1->dim[i] != 1 && t2->dim[i] != 1)
      return CF_ERR_INVALID;

    cf_usize dim = t1->dim[i] > t2->dim[i] ? t1->dim[i] : t2->dim[i];
    if(count > CF_USIZE_MAX / dim) return CF_ERR_OVERFLOW;
    count *= dim;
  }

  *out_count = count;
  return CF_OK;
}

/*
 * Validate that the output tensor shape matches the normalized/broadcasted
 * matrix multiplication result shape.
 */
static cf_status cf_tensor_matrix_mul_output_dims_match(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out, cf_usize max_rank)
{
  if(t_out->rank != max_rank) return CF_ERR_INVALID;

  for (cf_usize i = 0; i < max_rank - 2; i++)
  {
    if(t_out->dim[i] != (t1->dim[i] > t2->dim[i] ? t1->dim[i] : t2->dim[i]))
      return CF_ERR_INVALID;
  }
  
  if(t_out->dim[max_rank - 2] != t1->dim[max_rank - 2] || t_out->dim[max_rank - 1] != t2->dim[max_rank - 1])
    return CF_ERR_INVALID;

  return CF_OK;
}

/*
 * CPU matrix multiplication for scalars, vectors, matrices, and batched matrix
 * operands. The actual typed numeric loop is generated by
 * CF_TENSOR_MATRIX_MUL_CASE after shape normalization.
 */
cf_status cf_tensor_matrix_mul_cpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out)
{
  cf_status status = cf_tensor_require_data(t1);
  if(status != CF_OK) return status;
  status = cf_tensor_require_data(t2);
  if(status != CF_OK) return status;
  status = cf_tensor_require_data(t_out);
  if(status != CF_OK) return status;
  status = cf_tensor_require_same_type(t1, t2, t_out);
  if(status != CF_OK) return status;

  if(t1->rank == 0) return cf_tensor_scalar_mul_cpu(t2, t1->data, t_out);
  if(t2->rank == 0) return cf_tensor_scalar_mul_cpu(t1, t2->data, t_out);

  cf_tensor tmp1, tmp2;
  memcpy(&tmp1, t1, sizeof(cf_tensor));
  memcpy(&tmp2, t2, sizeof(cf_tensor));

  if(!cf_tensor_matrix_mul_inner_dim_match(&tmp1, &tmp2)) return CF_ERR_INVALID;

  cf_usize max_rank;
  cf_tensor_matrix_mul_normalize_rank(&tmp1, &tmp2, &max_rank);

  cf_usize count;
  status = cf_tensor_matrix_mul_batch_dims_match(&tmp1, &tmp2, max_rank, &count);
  if(status != CF_OK) return status;

  status = cf_tensor_matrix_mul_output_dims_match(&tmp1, &tmp2, t_out, max_rank);
  if(status != CF_OK) return status;

  switch (t1->metadata.elem_type)
  {
    CF_TENSOR_MATRIX_MUL_CASE(CF_TENSOR_CHAR, (&tmp1), (&tmp2), t_out, char);
    CF_TENSOR_MATRIX_MUL_CASE(CF_TENSOR_SHORT, (&tmp1), (&tmp2), t_out, short);
    CF_TENSOR_MATRIX_MUL_CASE(CF_TENSOR_INT, (&tmp1), (&tmp2), t_out, int);
    CF_TENSOR_MATRIX_MUL_CASE(CF_TENSOR_LONG, (&tmp1), (&tmp2), t_out, long);
    CF_TENSOR_MATRIX_MUL_CASE(CF_TENSOR_LL, (&tmp1), (&tmp2), t_out, long long);
    CF_TENSOR_MATRIX_MUL_CASE(CF_TENSOR_FLOAT, (&tmp1), (&tmp2), t_out, float);
    CF_TENSOR_MATRIX_MUL_CASE(CF_TENSOR_DOUBLE, (&tmp1), (&tmp2), t_out, double);
    CF_TENSOR_MATRIX_MUL_CASE(CF_TENSOR_LD, (&tmp1), (&tmp2), t_out, long double);
    CF_TENSOR_MATRIX_MUL_CASE(CF_TENSOR_U8, (&tmp1), (&tmp2), t_out, cf_u8);
    CF_TENSOR_MATRIX_MUL_CASE(CF_TENSOR_U16, (&tmp1), (&tmp2), t_out, cf_u16);
    CF_TENSOR_MATRIX_MUL_CASE(CF_TENSOR_U32, (&tmp1), (&tmp2), t_out, cf_u32);
    CF_TENSOR_MATRIX_MUL_CASE(CF_TENSOR_U64, (&tmp1), (&tmp2), t_out, cf_u64);
    CF_TENSOR_MATRIX_MUL_CASE(CF_TENSOR_U128, (&tmp1), (&tmp2), t_out, cf_u128);
    default: return CF_ERR_INVALID;
  }

  return CF_OK;
}

/*
 * Formatting helper for nested tensor printing.
 */
static void cf_tensor_print_indent(cf_usize indent)
{
  for (cf_usize i = 0; i < indent; i++) printf(" ");
}

/*
 * Print the framework 128-bit unsigned integer type without depending on
 * non-portable printf length modifiers.
 */
static void cf_tensor_print_u128(cf_u128 value)
{
  char buffer[40];
  cf_usize i = sizeof(buffer);

  buffer[--i] = '\0';
  if(value == 0)
  {
    printf("%10s", "0");
    return;
  }

  while(value != 0)
  {
    buffer[--i] = (char)('0' + (value % 10));
    value /= 10;
  }

  printf("%10s", &buffer[i]);
}

/*
 * Print one flat tensor element using the format appropriate for its element
 * type. Nested shape traversal is handled by cf_tensor_print_axis.
 */
static void cf_tensor_print_value(cf_tensor *tensor, cf_usize index)
{
  switch (tensor->metadata.elem_type)
  {
    case CF_TENSOR_CHAR:
      printf("%10c", ((char *)tensor->data)[index]);
      break;

    case CF_TENSOR_SHORT:
      printf("%10hd", ((short *)tensor->data)[index]);
      break;

    case CF_TENSOR_INT:
      printf("%10d", ((int *)tensor->data)[index]);
      break;

    case CF_TENSOR_LONG:
      printf("%10ld", ((long *)tensor->data)[index]);
      break;

    case CF_TENSOR_LL:
      printf("%10lld", ((long long *)tensor->data)[index]);
      break;

    case CF_TENSOR_FLOAT:
      printf("%10g", (double)((float *)tensor->data)[index]);
      break;

    case CF_TENSOR_DOUBLE:
      printf("%10g", ((double *)tensor->data)[index]);
      break;

    case CF_TENSOR_LD:
      printf("%10Lg", ((long double *)tensor->data)[index]);
      break;

    case CF_TENSOR_U8:
      printf("%10u", (unsigned int)((cf_u8 *)tensor->data)[index]);
      break;

    case CF_TENSOR_U16:
      printf("%10u", (unsigned int)((cf_u16 *)tensor->data)[index]);
      break;

    case CF_TENSOR_U32:
      printf("%10u", ((cf_u32 *)tensor->data)[index]);
      break;

    case CF_TENSOR_U64:
      printf("%10llu", (unsigned long long)((cf_u64 *)tensor->data)[index]);
      break;

    case CF_TENSOR_U128:
      cf_tensor_print_u128(((cf_u128 *)tensor->data)[index]);
      break;
  }
}

/*
 * Recursively print one tensor axis. The base offset is carried through the
 * row-major stride table so printing follows the logical tensor shape.
 */
static void cf_tensor_print_axis(cf_tensor *tensor, cf_usize axis, cf_usize base, cf_usize indent)
{
  if(axis == tensor->rank - 1)
  {
    printf("[");
    for (cf_usize i = 0; i < tensor->dim[axis]; i++)
    {
      if(i != 0) printf(", ");
      cf_tensor_print_value(tensor, base + i * tensor->metadata.stride[axis]);
    }
    printf("]");
    return;
  }

  printf("[\n");
  for (cf_usize i = 0; i < tensor->dim[axis]; i++)
  {
    cf_tensor_print_indent(indent + 2);
    cf_tensor_print_axis(
      tensor,
      axis + 1,
      base + i * tensor->metadata.stride[axis],
      indent + 2
    );

    if(i + 1 != tensor->dim[axis]) printf(",");
    printf("\n");
  }
  cf_tensor_print_indent(indent);
  printf("]");
}

/*
 * Public debug printer for CPU-visible tensors. CUDA tensors should be copied
 * to CPU before printing so this function can read tensor->data.
 */
void cf_tensor_print(cf_tensor *tensor)
{
  if(tensor == CF_NULL)
  {
    printf("(null)\n");
    return;
  }

  if(tensor->data == CF_NULL)
  {
    printf("(empty tensor)\n");
    return;
  }

  if(!cf_tensor_is_valid(tensor))
  {
    printf("(invalid tensor)\n");
    return;
  }

  if(tensor->rank == 0)
  {
    cf_tensor_print_value(tensor, 0);
    printf("\n");
    return;
  }

  cf_tensor_print_axis(tensor, 0, 0, 0);
  printf("\n");
}
